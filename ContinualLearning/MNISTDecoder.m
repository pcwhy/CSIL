clear;
clc;
close all;
rng default

addpath('../');
addpath('../HypersphereLib/');
set(0,'DefaultTextFontName','Times','DefaultTextFontSize',18,...
   'DefaultAxesFontName','Times','DefaultAxesFontSize',18,...
   'DefaultLineLineWidth',1,'DefaultLineMarkerSize',7.75)

load('./fashion-mnist.mat')
clear XTest
clear YTest
YTrain = categorical(YTrain);
YTrain = grp2idx(YTrain);

% [XTrain,YTrain] = digitTrain4DArrayData;
YTrain = double(YTrain);
cond = YTrain >=8;
revCond = ~cond;
uX = XTrain(:,:,:,cond);
uY = YTrain(cond);
% uX = 2*randn(size(uX));
XTrain = XTrain(:,:,:,revCond);
YTrain = YTrain(revCond);
YTrain = categorical(YTrain);

perm = randperm(numel(YTrain));
XTrain = XTrain(:,:,:,perm);
YTrain = YTrain(perm);
XVal = XTrain(:,:,:,end - round(0.3*size(XTrain,4)):end);
YVal = YTrain(end - round(0.3*size(YTrain,1)):end);
XTrain = XTrain(:,:,:,1:round(0.7*size(XTrain,4)));
YTrain = YTrain(1:round(0.7*size(YTrain,1)));

numClasses = numel(categories(YTrain))
numFeatureDim = numClasses + 1
layers = [
    imageInputLayer([28 28 1],'Name','Input','Mean',0)
    
    convolution2dLayer(3,8,'Padding','same','Name','conv2d_1')
    batchNormalizationLayer('Name','batchNorm_1')
    reluLayer('Name','relu_1')

    maxPooling2dLayer(2,'Stride',2,'Name','maxPooling2d_1')
    
    convolution2dLayer(3,16,'Padding','same','Name','cov2d_2')
    batchNormalizationLayer('Name','batchNorm_2')
    reluLayer('Name','relu_2')
    
    maxPooling2dLayer(2,'Stride',2,'Name','maxPooling2d_2')
    
    convolution2dLayer(3,32,'Padding','same','Name','cov2d_3')
    batchNormalizationLayer('Name','batchNorm_3')
    reluLayer('Name','relu_3')
    tensorVectorLayer('Flatten')
    dropoutLayer(0.3,'Name','someDropouts')
    fullyConnectedLayer(numFeatureDim,'Name','fc_bf_fp')
    yxBatchNorm('yxBatchnorm_1',numFeatureDim)
%     FCLayer(numFeatureDim,numel(categories(YTrain)),'fp',[])
    zeroBiasFCLayer(numFeatureDim,numel(categories(YTrain)),'fp',[])
    yxSoftmaxReverse('softmax')
    
    ];


lgraph = layerGraph(layers);
shallowClassifier = [
        zeroBiasFCLayer(numFeatureDim,numel(categories(YTrain)),'Shallowfp',[])
        yxSoftmax('ShallowSoftmax')];
decoder = [
%     fullyConnectedLayer(28*28*1,'Name','weakDecoderIn')
        tensorVectorLayer('weakDecoderIn')
%         FCLayer(numFeatureDim,28*28*1,'weakDecoderOut',[])
%         dropoutLayer(0.2,'Name','decoderDropouts')
        fullyConnectedLayer(28*28*1,'Name','weakDecoderOut')
    ];
% decoder = fullyConnectedLayer(28*28*1,'Name','weakDecoder');
lgraph = addLayers(lgraph,decoder);
lgraph = connectLayers(lgraph,'relu_1','weakDecoderIn');
YTrain = double(YTrain);
numEpochs = 5;
miniBatchSize = 128;
plots = "training-progress";
executionEnvironment = "auto";
if plots == "training-progress"
    figure(10);
    lineLossTrain = animatedline('Color','#0072BD','lineWidth',1.5);
    lineClassificationLoss = animatedline('Color','#EDB120','lineWidth',1.5);
    ylim([-inf inf])
    xlabel("Iteration")
    ylabel("Loss")
    legend('Loss','classificationLoss');
    grid on;
    
    figure(11);  
    lineCVAccuracy = animatedline('Color','#D95319','lineWidth',1.5);
    ylim([0 1.1])
    xlabel("Iteration")
    ylabel("Loss")    
    legend('CV Acc.');
    grid on;    
    
    figure(12);  
    lineReconstructionLoss = animatedline('Color','#77AC30','lineWidth',1.5);
    ylim([-inf inf])
    xlabel("Iteration")
    ylabel("ReconstructionLoss")
    
    figure(13);
    lineSumDeviation = animatedline('Color','#77AC30','lineWidth',1.5);
    ylim([-inf inf])
    xlabel("Iteration")
    ylabel("Sum of deviations")    
end

L2RegularizationFactor = 0.01;
initialLearnRate = 0.01;
decay = 0.01;
momentumSGD = 0.9;
velocities = [];
learnRates = [];
momentums = [];
gradientMasks = [];
numObservations = numel(YTrain);
numIterationsPerEpoch = floor(numObservations./miniBatchSize);
iteration = 0;
start = tic;
classes = categorical(YTrain);
lgraph2 = lgraph; % No old weights
dlnet = dlnetwork(lgraph2);

% Loop over epochs.
totalIters = 0;
for epoch = 1:numEpochs
    idx = randperm(numel(YTrain));
    XTrain = XTrain(:,:,:,idx);
    YTrain = YTrain(idx); 
    % Loop over mini-batches.
    for i = 1:numIterationsPerEpoch
        iteration = iteration + 1;
        totalIters = totalIters + 1;
        % Read mini-batch of data and convert the labels to dummy
        % variables.
        idx = (i-1)*miniBatchSize+1:i*miniBatchSize;
        Xb = XTrain(:,:,:,idx);
        Yb = zeros(numClasses, miniBatchSize, 'single');
        for c = 1:numClasses
            Yb(c,YTrain(idx)==(c)) = 1;
        end
        % Convert mini-batch of data to dlarray.
        dlX = dlarray(single(Xb),'SSCB');
        % If training on a GPU, then convert data to gpuArray.
        if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
            dlX = gpuArray(dlX);
        end
        % Evaluate the model gradients, state, and loss using dlfeval and the
        % modelGradients function and update the network state.
        [gradients,state,...
            loss,classificationLoss,...
            reconstructionLoss] = dlfeval(@modelGradientsOnWeights,dlnet,dlX,Yb);
%         [gradients,state,loss] = dlfeval(@modelGradientsOnWeights,dlnet,dlX,Yb);        
        dlnet.State = state;
        % Determine learning rate for time-based decay learning rate schedule.
        learnRate = initialLearnRate/(1 + decay*iteration);
        % Update the network parameters using the SGDM optimizer.
        %[dlnet, velocity] = sgdmupdate(dlnet, gradients, velocity, learnRate, momentum);
        % Update the network parameters using the SGD optimizer.
        %dlnet = dlupdate(@sgdFunction,dlnet,gradients);
        if isempty(velocities)
            velocities = packScalar(gradients, 0);
            learnRates = packScalar(gradients, learnRate);
            momentumSGDs = packScalar(gradients, momentumSGD);
            momentums = packScalar(gradients, 0);
            L2Foctors = packScalar(gradients, 0);            
            wd = packScalar(gradients, 0);  
            gradientMasks = packScalar(gradients, 1);   
%             % Let's lock some weights
%             for k = 1:2
%                 gradientMasks.Value{k}=dlarray(zeros(size(gradientMasks.Value{k})));
%             end
        end
%%%%----------- Check Point 2:  
%%%% Here you can specify which optimizer to use, 
%         [dlnet, velocities] = dlupdate(@sgdmFunctionL2, ...
%             dlnet, gradients, velocities, ...
%             learnRates, momentumSGDs, L2Foctors, gradientMasks); % This is
%             % the famous SGD with momentum
        totalIterInPackage = packScalar(gradients, totalIters); % We have to make this...
                                                         % stupid data
                                                         % structure but it
                                                         % only contains
                                                         % the number of
                                                         % iterations
        [dlnet, velocities, momentums] = dlupdate(@adamFunction, ...
                    dlnet, gradients, velocities, ...
                    learnRates, momentums, wd, gradientMasks, ...
                    totalIterInPackage);        
%         [dlnet] = dlupdate(@sgdFunction, ...
%             dlnet, gradients); % the vanilla
%%%%-----------End of Check Point 2 

        % Display the training progress.
        if plots == "training-progress"
            D = duration(0,0,toc(start),'Format','hh:mm:ss');
            XTest = XVal;
            YTest = categorical(YVal);
            if mod(iteration,5) == 0 
                accuracy = cvAccuracy(dlnet, XTest,YTest,miniBatchSize,executionEnvironment,0);
                addpoints(lineCVAccuracy, iteration, accuracy);
            end
            stepSumDeviations = calculateGrossMutualDistance(gpuArray(dlnet.Layers({dlnet.Layers.Name} == "fp").Weights));

            addpoints(lineLossTrain,iteration,double(gather(extractdata(loss))))
            addpoints(lineClassificationLoss,iteration,double(gather(extractdata(classificationLoss))));
            addpoints(lineReconstructionLoss,iteration,double(gather(extractdata(reconstructionLoss))));
            addpoints(lineSumDeviation,iteration, double(stepSumDeviations) );

            title("Epoch: " + epoch + ", Elapsed: " + string(D))
            drawnow
        end
    end
end
accuracy = cvAccuracy(dlnet, XVal, categorical(YVal), miniBatchSize, executionEnvironment, 1)
% imagesc(similarMatrix)

testReconstruction = sigmoid(predict(dlnet, dlarray(XVal(:,:,:,1:10),'SSCB'),'Outputs', 'weakDecoderOut'));
figure
for i = 1:10
    subplot(2,10,i)
    imagesc(squeeze(XVal(:,:,:,i)))
end
for i = 1:10
    subplot(2,10,i+10)
    recsI = reshape(gather(extractdata(testReconstruction(:,i))),[28,28]);
    recsI = rescale(recsI);
    recsI = reshape(recsI,[28,28,1,1]);
    predict(dlnet, dlarray(recsI,'SSCB'))
    imagesc(reshape(gather(extractdata(testReconstruction(:,i))),[28,28]))
end

featuresWithWeakDecoder = squeeze(dlnet.predict(dlarray(XVal,'SSCB'),'Outputs','Flatten'));
[coeff, score, latent, tsquared, explained] = pca(extractdata(featuresWithWeakDecoder)');
featurePCA = sum((cumsum(latent))./sum(latent) <= 0.95)

[sumDevationCosine,~] = calculateGrossMutualDistance(dlnet.Layers({dlnet.Layers.Name} == "fp").Weights)


function [sumDeviationAngles,similarMatrix] = calculateGrossMutualDistance(FPs)
%     weights = FPs;
%     similarMatrix = zeros(size(weights,1), size(weights,1));
%     for i = 1:size(weights,1)
%         curWeight = weights(i,:);  
%         magCurWeight = sqrt(sum(curWeight.^2,2));
%         for j = 1:size(weights,1)
%             nxtWeight = weights(j,:);  
%             magNxtWeight = sqrt(sum(nxtWeight.^2,2));
%             similarMatrix(i,j)=sum(...
%                 (nxtWeight./magNxtWeight)...
%                 .*(curWeight./magCurWeight)...
%                 );
% %             similarMatrix(i,j) = sqrt(sum((nxtWeight - curWeight).^2));
%         end
%     end
    similarMatrix = FPs*FPs'./sqrt(diag(FPs*FPs'))./sqrt(diag(FPs*FPs')');
    sumDeviationAngles = (sum(similarMatrix,'all') - sum(diag(similarMatrix)))./2;
end

function accuracy = cvAccuracy(dlnet, XTest, YTest, miniBatchSize, executionEnvironment, confusionChartFlg)
    dlXTest = dlarray(XTest,'SSCB');
    if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
        dlXTest = gpuArray(dlXTest);
    end
    dlYPred = modelPredictions(dlnet,dlXTest,miniBatchSize);
    [~,idx] = max(extractdata(dlYPred),[],1);
    YPred = categorical(idx);
    accuracy = mean(YPred(:) == YTest(:));
    if confusionChartFlg == 1
        figure
        confusionchart(YPred(:),YTest(:));
    end
end

function dlYPred = modelPredictions(dlnet,dlX,miniBatchSize)
    numObservations = size(dlX,4);
    numIterations = ceil(numObservations / miniBatchSize);
    numClasses = size(dlnet.Layers({dlnet.Layers.Name} == "fp").Weights,1);
    dlYPred = zeros(numClasses,numObservations,'like',dlX);
    for i = 1:numIterations
        idx = (i-1)*miniBatchSize+1:min(i*miniBatchSize,numObservations);
        dlYPred(:,idx) = predict(dlnet,dlX(:,:,:,idx));
    end
end


function [gradients,state,loss,...
    classificationLoss, reconstructionLoss] = modelGradientsOnWeights(dlnet,dlX,Y)
%   %This is only used with softmax of matlab which only applies softmax
%   on 'C' and 'B' channels.
%     [rawPredictions, rawReconstruction, state] = forward(dlnet,dlX,'Outputs', {'fp', 'weakDecoderOut'});
%     dlYPred = softmax(dlarray(squeeze(rawPredictions),'CB'));
    [dlYPred, rawReconstruction, state] = forward(dlnet,dlX,'Outputs', {'softmax', 'weakDecoderOut'});
    reconstruction = (rawReconstruction);
    flatX = squeeze(reshape(dlX,[28*28,1,128]));
%     [dlYPred,state] = forward(dlnet,dlX);
    penalty = 0;
    scalarL2Factor = 0;
    if scalarL2Factor ~= 0
        paramLst = dlnet.Learnables.Value;
        for i = 1:size(paramLst,1)
            penalty = penalty + sum((paramLst{i}(:)).^2);
        end
    end
    
    classificationLoss = crossentropy(squeeze(dlYPred),Y) + scalarL2Factor*penalty;
    reconstructionLoss = sqrt(mse(reconstruction,flatX));
    loss = 0.8.*classificationLoss + 0.1*reconstructionLoss;
%     loss = classificationLoss + 0.2*(max(max(rawPredictions))-min(max(rawPredictions)));
    gradients = dlgradient(loss,dlnet.Learnables);
    %gradients = dlgradient(loss,dlnet.Learnables(4,:));
end

function [params,velocityUpdates,momentumUpdate] = adamFunction(params, rawParamGradients,...
    velocities, learnRates, momentums, wd, gradientMasks, iters)
    % https://arxiv.org/pdf/2010.07468.pdf %%AdaBelief
    % https://arxiv.org/pdf/1711.05101.pdf  %%DeCoupled Weight Decay 
    b1 = 0.9; 
    b2 = 0.999;
    e = 1e-8;
    curIter = iters(:);
    curIter = curIter(1);
    
    gt = rawParamGradients;
    mt = (momentums.*b1 + ((1-b1)).*gt);
    vt = (velocities.*b2 + ((1-b2)).*((gt-mt).^2));

     momentumUpdate = mt;
     velocityUpdates = vt;
    h_mt = mt./(1-b1.^curIter);
    h_vt = (vt+e)./(1-b2.^curIter);
%%%%----------- Check Point 3:  
%%%% Here you can specify whether to use bias correction, 
%%%% or zero-bias dense layer 
%%%% in this test, we can just try to eliminate the effect of varying learning
%%%% rates
%     params = params - 0.001.*(mt./(sqrt(vt)+e)).*gradientMasks...
%         - wd.*params.*gradientMasks; %This works better for zero-bias dense layer
%     params = params - 0.001.*(h_mt./(sqrt(h_vt)+e)).*gradientMasks...
%         -L2Foctors.*params.*gradientMasks;
     params = params - learnRates.*(h_mt./(sqrt(h_vt)+e)).*gradientMasks...
         -2*learnRates .* wd.*params.*gradientMasks;
%%%%
%%%%-----------End of Check Point 3 
end

function param = sgdFunction(param,paramGradient)
    learnRate = 0.01;
    param = param - learnRate.*paramGradient;
end

function [params, velocityUpdates] = sgdmFunction(params, paramGradients,...
    velocities, learnRates, momentums)
% https://towardsdatascience.com/stochastic-gradient-descent-momentum-explanation-8548a1cd264e
%     velocityUpdates = momentums.*velocities+learnRates.*paramGradients;
    velocityUpdates = momentums.*velocities+0.001.*paramGradients;
    params = params - velocityUpdates;
end

function [params, velocityUpdates] = sgdmFunctionL2(params, rawParamGradients,...
    velocities, learnRates, momentums, L2Foctors, gradientMasks)
% https://towardsdatascience.com/stochastic-gradient-descent-momentum-explanation-8548a1cd264e
% https://towardsdatascience.com/intuitions-on-l1-and-l2-regularisation-235f2db4c261
    paramGradients = rawParamGradients + 2*L2Foctors.*params;
    velocityUpdates = momentums.*velocities+learnRates.*paramGradients;
    params = params - (velocityUpdates).*gradientMasks;
end

function tabVars = packScalar(target, scalar)
% The matlabs' silly design results in such a strange function
    tabVars = target;
    for row = 1:size(tabVars(:,3),1)
        tabVars{row,3} = {...
            dlarray(...
            ones(size(tabVars.Value{row})).*scalar...%ones(size(tabVars(row,3).Value{1,1})).*scalar...
            )...
            };
    end
end



