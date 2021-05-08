clc;
close all;
clear;
rng default;

addpath('./matplotlib')  


load('adsb_records_qt.mat');
%load('adsb_bladerf2_10M_qt');
payloadMatrix = reshape(payloadMatrix', ...
    length(payloadMatrix)/length(msgIdLst), length(msgIdLst))';
rawIMatrix = reshape(rawIMatrix', ...
    length(rawIMatrix)/length(msgIdLst), length(msgIdLst))';
rawQMatrix = reshape(rawQMatrix', ...
    length(rawQMatrix)/length(msgIdLst), length(msgIdLst))';
rawCompMatrix = rawIMatrix + rawQMatrix.*1j;
if size(rawCompMatrix,2) < 1024
    appendingBits = (ceil(sqrt(size(rawCompMatrix,2))))^2 - size(rawCompMatrix,2);
    rawCompMatrix = [rawCompMatrix, zeros(size(rawCompMatrix,1), appendingBits)];
else
   rawCompMatrix = rawCompMatrix(:,1:1024); 
end
uIcao = unique(icaoLst);
c = countmember(uIcao,icaoLst);
icaoOccurTb = [uIcao,c];
icaoOccurTb = sortrows(icaoOccurTb,2,'descend');
cond1 = icaoOccurTb(:,2)>=300;
cond2 = icaoOccurTb(:,2)<=5000;
cond3 = icaoOccurTb(:,2)>=250;
cond4 = icaoOccurTb(:,2)<500;

cond = logical(cond1.*cond2);
selectedPlanes = icaoOccurTb(cond,:);
unknowPlanes = icaoOccurTb(logical(cond3.*cond4),:);
allTrainData = [icaoLst, abs(rawCompMatrix)];

minTrainChance = 100;
maxTrainChance = 500;

figure
histogram(icaoOccurTb(icaoOccurTb(:,2)>200,:),'Normalization','probability','NumBins',10)
xlabel('Number of observations');
ylabel('Probability')
set(gca,'FontSize',12)
set(gcf,'position',[405   423   441   186]);

% set(gcf,'position',[405   395   441   214]);
grid on


selectedBasebandData = zeros(size(allTrainData));
selectedRawCompData = zeros(size(rawCompMatrix));
cursor = 1;
for i = 1:size(selectedPlanes,1)
    selection = allTrainData(:,1)==selectedPlanes(i,1);
    localBaseband = allTrainData(selection,:);
    localBaseband(:,1) = ones(size(localBaseband,1),1).*i;
    localComplex = rawCompMatrix(selection,:);
    localAngles = (angle(localComplex(:,:)));

%     figure
%     for k = 1:size(localAngles,1)
%         plot(localAngles(k,:),'.');
%         title(strcat(num2str(k), ' / ', num2str(size(localAngles,1))));
%         pause(30/1000);
%     end    
    
    if size(localBaseband,1) < minTrainChance
        continue;
    elseif size(localBaseband,1) >= maxTrainChance
        rndSeq = randperm(size(localBaseband,1));
        rndSeq = rndSeq(1:maxTrainChance);
        localBaseband = localBaseband(rndSeq,:);
        localComplex = localComplex(rndSeq,:);
    else
        %Nothing to do
    end
    selectedBasebandData(cursor:cursor+size(localBaseband,1)-1,:) = localBaseband;
    selectedRawCompData(cursor:cursor+size(localComplex,1)-1,:) = localComplex;
    cursor = cursor+size(localBaseband,1);    
%     selectedBasebandData = [selectedBasebandData; localBaseband];
%     selectedRawCompData = [selectedRawCompData; localComplex];    
end
selectedBasebandData = selectedBasebandData(1:cursor-1,:);
selectedRawCompData = selectedRawCompData(1:cursor-1,:);

offset = size(selectedPlanes,1);
unknownBasebandData = zeros(size(allTrainData));
unknownRawCompData = zeros(size(rawCompMatrix));
cursor = 1;
for i = 1:size(unknowPlanes,1)
    selection = allTrainData(:,1)==unknowPlanes(i,1);
    localBaseband = allTrainData(selection,:);
    localBaseband(:,1) = ones(size(localBaseband,1),1).*(i+offset);
    localComplex = rawCompMatrix(selection,:);
    localAngles = (angle(localComplex(:,:)));
    unknownBasebandData(cursor:cursor+size(localBaseband,1)-1,:) = localBaseband;
    unknownRawCompData(cursor:cursor+size(localComplex,1)-1,:) = localComplex;
    cursor = cursor+size(localBaseband,1);
%     unknownBasebandData = [unknownBasebandData; localBaseband];
%     unknownRawCompData = [unknownRawCompData; localComplex];    
end
unknownBasebandData = unknownBasebandData(1:cursor-1,:);
unknownRawCompData = unknownRawCompData(1:cursor-1,:);

randSeries = randperm(size(selectedBasebandData,1));
selectedBasebandData = selectedBasebandData(randSeries,:);
selectedRawCompData = selectedRawCompData(randSeries,:);

randSeries = randperm(size(unknownBasebandData,1));
unknownBasebandData = unknownBasebandData(randSeries,:);
unknownRawCompData = unknownRawCompData(randSeries,:);

[X,cX,Y,cY] = makeDataTensor(selectedBasebandData,selectedRawCompData);
[uX,cuX,uY,cuY] = makeDataTensor(unknownBasebandData,unknownRawCompData);

inputSize = [size(X,1) size(X,2) size(X,3)];
numClasses = size(unique(selectedBasebandData(:,1)),1);

layers = [
    imageInputLayer(inputSize, 'Name', 'input')
    convolution2dLayer(5,10, 'Name', 'conv2d_1')
    batchNormalizationLayer('Name', 'batchNorm_1')
    reluLayer('Name', 'relu_1')
    convolution2dLayer(3, 10, 'Padding', 1, 'Name', 'conv2d_2')
    reluLayer('Name', 'relu_2')    
    convolution2dLayer(3, 10, 'Padding', 1, 'Name', 'conv2d_3')
    reluLayer('Name', 'relu_3')    
    %additionLayer(2,'Name', 'add_1')
    depthConcatenationLayer(2,'Name','add_1')    

    fullyConnectedLayer(numClasses, 'Name', 'fc_bf_fp') % 11th
    %tanhLayer('Name','relu_4')
    %reluLayer('Name','relu_4')
    %preluLayer(numClasses,'prelu')
    
    
    zeroBiasFCLayer(numClasses,numClasses,'Fingerprints',[])
    
    %fullyConnectedLayer(numClasses, 'Name', 'Fingerprints') % 11th
    %dropoutLayer(0.3,'Name','dropOut_1')            
    %amplificationLayer(numClasses,'Fingerprints',[])
    %nearestNeighbourLayer(numClasses,numClasses,'Fingerprints',[])
    softmaxLayer('Name', 'softmax_1')
    classificationLayer('Name', 'classify_1')
    ];


lgraph = layerGraph(layers);
lgraph = connectLayers(lgraph, 'relu_1', 'add_1/in2');
%lgraph = connectLayers(lgraph, 'relu_2', 'add_1/in3');
plot(lgraph);
options = trainingOptions('sgdm',...
    'Plots', 'training-progress',...
    'ExecutionEnvironment','auto',...
    'ValidationData',{cX,categorical(cY)},...
    'MaxEpochs', 10, ...
    'MiniBatchSize',128,...
    'L2Regularization',0.01);
[net,info] = trainNetwork(X, categorical(Y), lgraph, options);

%iacc2 = find(~isnan(info.ValidationAccuracy) == 1)
%vacc2 = info.ValidationAccuracy(~isnan(info.ValidationAccuracy) == 1);

YPred = classify(net, cX);
accuracy = sum(categorical(cY) == YPred)/numel(cY)
cm = confusionmat(categorical(cY),YPred);
cm = cm./sum(cm,2);
imagesc(cm);

%%%%%
%Gather fisher information and weights from the old network
lgraph2 = layerGraph(net);
lgraph2 = lgraph2.removeLayers('classify_1');
prevDlnet = dlnetwork(lgraph2);
prevWeights = prevDlnet.Learnables;
prevCX = dlarray(single(cX),'SSCB');
prevCY = zeros(numClasses, size(prevCX,4), 'single');
for c = 1:numClasses
    prevCY(c,cY(:)==(c)) = 1;
end
executionEnvironment = "auto";
% If training on a GPU, then convert data to gpuArray.
if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
    prevCX = gpuArray(prevCX);
end
[prevGradients0,state] = dlfeval(@logModelGradientsOnWeights,prevDlnet,prevCX);
accuracy = cvAccuracy(prevDlnet, prevCX, (cY), 128, executionEnvironment)
prevDlnet.State = state;
prevFisherInfo0 = prevGradients0;
for i = 1:size(prevFisherInfo0,1)
    prevFisherInfo0{i,3}={dlarray(exp(prevGradients0{i,3}{:}.^2))};
%     prevFisherInfo0{i,3}={dlarray(1+(prevGradients0{i,3}{:}.^2))}; %Also works
end
fingerprintLayerIdx = 11;
figure
subplot(2,1,1)
imagesc(extractdata(prevGradients0{fingerprintLayerIdx,3}{:}))
title('Gradients');
subplot(2,1,2)
imagesc(extractdata(prevFisherInfo0{fingerprintLayerIdx,3}{:}))
title('Fisher Info.');
fingerprintLen = size(net.Layers(fingerprintLayerIdx).Weights,2);

figure
cursor = 1;
meanFisherInfo = [];
for i = [2,4,6,8,12]
    subplot(5,1,cursor);
    weights = extractdata(prevFisherInfo0{end-i,3}{:});
    weights = squeeze(weights);
    meanFisherInfo(end+1)=(gather(mean(weights(:))));    
    [~,edges] = histcounts(log10(weights(:)));
    histogram(weights(:),10.^edges,'Normalization','probability');
    set(gca, 'xscale','log');
%     histogram(weights(:),'Normalization','probability');
    cursor = cursor + 1;
%     xlim([0,1]);
end

figure
plot(meanFisherInfo,'LineWidth',1.0)
set(gca, 'yscale','log');

newClassesNum = floor(size(unique(uY),1));
unknownClassLabels = unique(uY);
idx = randperm(size(unknownClassLabels,1));
unknownClassLabels = unknownClassLabels(idx);
unknownClassLabels = unknownClassLabels(1:newClassesNum);

% Generate some initial fingerprints.
%figure
% hold on;
existingFingerprints = net.Layers(fingerprintLayerIdx).Weights;
newFingerprints = [];
continualLearnX = [];
continualLearnY = [];
cursor = 1;
for i = 1:newClassesNum
    selection = uY==unknownClassLabels(i);
    ux_i = uX(:,:,:,selection);
    uy_i = uY(selection);
    samplePerClass = size(uy_i, 1);
    continualLearnX(:,:,:,cursor:cursor+samplePerClass-1) = ux_i;
    continualLearnY(cursor:cursor+samplePerClass-1)=uy_i;
    cursor = cursor+samplePerClass;
    aUx_i = squeeze(activations(net,ux_i,'fc_bf_fp'));
    unitAUx_i=(aUx_i./sqrt(sum(aUx_i.^2)))';
    fp_i = mean(aUx_i,2)';
    magFp_i = sqrt(sum(fp_i.^2));
    unitFp_i = fp_i ./ magFp_i;
    newFingerprints(end+1,:) = unitFp_i;
%     histogram(sum(unitFp_i.*unitFingerprints,2),[-1:0.2:1],'Normalization','probability');
%     hold on;
%     histogram(sum(unitAUx_i.*unitFp_i,2),[-1:0.2:1],'Normalization','probability');
%     legend('Correlation with existing fingerprints','Correlation with own samples');
end
randSeries = randperm(size(continualLearnY,2));
continualLearnX = continualLearnX(:,:,:,randSeries);
continualLearnY = continualLearnY(randSeries);
cvContinualLearnX = continualLearnX(:,:,:,floor(0.6*size(continualLearnX,4)):end);
cvContinualLearnY = continualLearnY(floor(0.6*size(continualLearnY,2)):end);
continualLearnX = continualLearnX(:,:,:,1:floor(0.6*size(continualLearnX,4))-1);
continualLearnY = continualLearnY(1:floor(0.6*size(continualLearnY,2))-1);

concatFingerprints = [existingFingerprints; newFingerprints];
newFingerprintLayer = zeroBiasFCLayer(numClasses,numClasses+newClassesNum,'Fingerprints',concatFingerprints);
% newFingerprintLayer.normMag = prevDlnet.Layers(4).normMag;
% newFingerprintLayer.b1 = prevDlnet.Layers(4).b1;

numClasses = numClasses + newClassesNum;

%Build new network and start continual learning.
lgraph2 = layerGraph(net);
lgraph2 = lgraph2.removeLayers('classify_1');
dlnet = dlnetwork(replaceLayer(lgraph2, 'Fingerprints', newFingerprintLayer));

XTrain = continualLearnX;
YTrain = continualLearnY;
XTest = cvContinualLearnX;
YTest = (cvContinualLearnY);
numEpochs = 10;
miniBatchSize = 20;
plots = "training-progress";
statusFigureIdx = [];
statusFigureAxis = [];
if plots == "training-progress"
    figure;
    statusFigureIdx = gcf;
    statusFigureAxis = gca;
    % Go into the documentation of animatedline for more color codes
    lineNewLossTrain = animatedline('Color', '#0072BD','LineWidth',1,'Marker','.','LineStyle','none');
    lineNewCVAccuracy = animatedline('Color', '#D95319','LineWidth',1);
    lineOldCVAccuracy = animatedline('Color',	'#EDB120','LineWidth',1);
    lineOldLossCV = animatedline('Color',	'#7E2F8E','LineWidth',1,'Marker','.','LineStyle','none');
%    ylim([0 inf])
    ylim([0 2])
    xlabel("Iteration")
    ylabel("Metrics")
    legend('New task loss','New task accuracy','Old task accuracy', 'Old task lost');
    grid on
end

initialLearnRate = 0.01;
decay = 0.01;
momentum = 0.9;
velocities = [];
learnRates = [];
momentums = [];
gradientMasks = [];
numObservations = numel(YTrain);
numIterationsPerEpoch = floor(numObservations./miniBatchSize);
iteration = 0;
start = tic;
classes = categorical(YTrain);

%
prevCY = [prevCY;zeros(newClassesNum,size(prevCY,2))];

% Loop over epochs.
fisherLossLst = [];
for epoch = 1:numEpochs
    idx = randperm(numel(YTrain));
    XTrain = XTrain(:,:,:,idx);
    YTrain = YTrain(idx);    
    % Loop over mini-batches.
    for i = 1:numIterationsPerEpoch
        iteration = iteration + 1;
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
%         [gradients,state,loss] = dlfeval(@modelGradientsOnWeights,dlnet,dlX,Yb);

        [gradients,state,loss,fisherLoss] = dlfeval(@modelGradientsOnWeightsEWC,dlnet,dlX, Yb,...
            prevFisherInfo0, prevDlnet, newClassesNum, 1, fingerprintLayerIdx);
        fisherLossLst(end+1)=extractdata(gather(fisherLoss));

        dlnet.State = state;
        % Determine learning rate for time-based decay learning rate schedule.
        learnRate = initialLearnRate/(1 + decay*iteration);
        if isempty(velocities)
            velocities = packScalar(gradients, 0);
            learnRates = packScalar(gradients, learnRate);
            momentums = packScalar(gradients, momentum);
            L2Foctors = packScalar(gradients, 0.05);            
            gradientMasks = packScalar(gradients, 1);   
             % Let's lock some weights
            %gradientMasks{end-2,3} = {dlarray([zeros(18,18); ones(newClassesNum,fingerprintLen)])};
            for k = 1:fingerprintLayerIdx-1
                gradientMasks.Value{k}=dlarray(zeros(size(gradientMasks.Value{k})));
            end
        end
        %fisherLoss = calcFisherLoss(prevFisherInfo0, dlnet, prevDlnet,newClassesNum)
        [dlnet, velocities] = dlupdate(@sgdmFunctionL2, ...
            dlnet, gradients, velocities, ...
            learnRates, momentums, L2Foctors, gradientMasks);

        % Display the training progress.
        if plots == "training-progress"
            D = duration(0,0,toc(start),'Format','hh:mm:ss');
%             figure(statusFigureIdx);
            if mod(iteration,20) == 0 
                newCvAccuracy = cvAccuracy(dlnet, XTest,YTest,miniBatchSize,executionEnvironment);
                oldCvAccuracy = cvAccuracy(dlnet, cX, (cY), miniBatchSize,executionEnvironment);
                [~,~,oldCVLoss] = dlfeval(@modelGradientsOnWeights,dlnet,prevCX,prevCY);
                addpoints(lineNewCVAccuracy, iteration, newCvAccuracy);
                addpoints(lineOldCVAccuracy, iteration, oldCvAccuracy);
                addpoints(lineOldLossCV, iteration, double(gather(extractdata(oldCVLoss))));
            end
            addpoints(lineNewLossTrain,iteration,double(gather(extractdata(loss))))
            title(statusFigureAxis,"Epoch: " + epoch + ", Elapsed: " + string(D))
            drawnow
        end
    end
end
figure
subplot(2,1,1)
imagesc(prevDlnet.Layers(fingerprintLayerIdx).Weights)
title('Old Finger prints');
subplot(2,1,2)
imagesc(dlnet.Layers(fingerprintLayerIdx).Weights)
title('New Finger prints');

% figure
% subplot(2,1,1)
% imagesc(prevDlnet.Layers(2).Weights)
% title('Old input dense');
% subplot(2,1,2)
% imagesc(dlnet.Layers(2).Weights)
% title('New input dense');

function [gradients,state,loss,fisherLoss] = modelGradientsOnWeightsEWC(dlnet, dlX, Y,...
    prevFisherInfo0, prevDlnet, newClassesNum, ewcLambda, fingerprintLayerIdx)
    [dlYPred,state] = forward(dlnet,dlX);
    penalty = 0;
    scalarL2Factor = 0;
    if scalarL2Factor ~= 0
        paramLst = dlnet.Learnables.Value;
        for i = 1:size(paramLst,1)
            penalty = penalty + sum((paramLst{i}(:)).^2);
        end
    end
    fisherLoss = calcFisherLoss(prevFisherInfo0, dlnet, prevDlnet, newClassesNum, fingerprintLayerIdx)*ewcLambda/2;
    loss = crossentropy(dlYPred,Y) + scalarL2Factor*penalty + fisherLoss;
    gradients = dlgradient(loss, dlnet.Learnables);
end

function fisherLoss = calcFisherLoss(prevFisherInfo0, dlnet, prevDlnet, newClassesNum,fingerprintLayerIdx)
    prevWeights = prevDlnet.Learnables.Value;
    curWeights = dlnet.Learnables.Value;
%     fisherLossMatrix = {};
    sumLoss = 0;
    elementCount = 1;
    for i = 1:size(prevWeights,1)
        if i == fingerprintLayerIdx
            loss = ((prevWeights{i}-curWeights{i}(1:size(curWeights{i},1)-newClassesNum,:)).^2) .* prevFisherInfo0.Value{i};    
%             fisherLossMatrix{end+1} = loss;
            sumLoss = sumLoss + sum(loss(:));
            elementCount = elementCount + numel(prevWeights{i});
        else
            loss = ((prevWeights{i}-curWeights{i}).^2) .* prevFisherInfo0.Value{i};
%             fisherLossMatrix{end+1} = loss;
            sumLoss = sumLoss + sum(loss(:));
            elementCount = elementCount + numel(prevWeights{i});            
        end
    end
    fisherLoss = sumLoss;
end

function accuracy = cvAccuracy(dlnet, XTest, YTest, miniBatchSize, executionEnvironment)
    dlXTest = dlarray(XTest,'SSCB');
    if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
        dlXTest = gpuArray(dlXTest);
    end
    dlYPred = modelPredictions(dlnet,dlXTest,miniBatchSize);
    [~,idx] = max(extractdata(dlYPred),[],1);
    YPred = (idx);
    accuracy = mean(YPred(:) == YTest(:));
end

function dlYPred = modelPredictions(dlnet,dlX,miniBatchSize)
    numObservations = size(dlX,4);
    numIterations = ceil(numObservations / miniBatchSize);
    numClasses = size(dlnet.Layers(end-1).Weights,1);
    dlYPred = zeros(numClasses,numObservations,'like',dlX);
    for i = 1:numIterations
        idx = (i-1)*miniBatchSize+1:min(i*miniBatchSize,numObservations);
        dlYPred(:,idx) = predict(dlnet,dlX(:,:,:,idx));
    end
end

function [gradients,state] = logModelGradientsOnWeights(dlnet,dlX)
    [dlYPred,state] = forward(dlnet,dlX);
    loglikelyhood = log(dlYPred);
    gradients = dlgradient(mean(loglikelyhood(:)),dlnet.Learnables);
end

function [gradients,state,loss] = modelGradientsOnWeights(dlnet,dlX,Y)
    [dlYPred,state] = forward(dlnet,dlX);
    penalty = 0;
    scalarL2Factor = 0;
    if scalarL2Factor ~= 0
        paramLst = dlnet.Learnables.Value;
        for i = 1:size(paramLst,1)
            penalty = penalty + sum((paramLst{i}(:)).^2);
        end
    end
    loss = crossentropy(dlYPred,Y) + scalarL2Factor*penalty;
    gradients = dlgradient(loss,dlnet.Learnables);
    %gradients = dlgradient(loss,dlnet.Learnables(4,:));
end

function [params, velocityUpdates] = sgdmFunction(params, paramGradients,...
    velocities, learnRates, momentums)
% https://towardsdatascience.com/stochastic-gradient-descent-momentum-explanation-8548a1cd264e
    velocityUpdates = momentums.*velocities+learnRates.*paramGradients;
    params = params - velocityUpdates;
end

function [params, velocityUpdates] = sgdmFunctionL2(params, rawParamGradients,...
    velocities, learnRates, momentums, L2Foctors, gradientMasks)
% https://towardsdatascience.com/stochastic-gradient-descent-momentum-explanation-8548a1cd264e
% https://towardsdatascience.com/intuitions-on-l1-and-l2-regularisation-235f2db4c261
    paramGradients = rawParamGradients + 2*L2Foctors.*params;
    %Please be noted that even if rawParamGradients = 0, L2 will still try
    %to reduce the magnitudes of parameters
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



