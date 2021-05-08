% Custom initial training loop 
% Custom continual learning loop 
% Controling the depth of learning

clc;
close all;
clear;
rng default;

addpath('./matplotlib')  
addpath('../');

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

cond1 = latitudeLst >= -200;
cond2 = longitudeLst >= -200;
cond4 = altitudeLst >= 0;
cond3 = cond1.*cond2.*cond4;
DrawPic = [longitudeLst(logical(cond3),:),...
    latitudeLst(logical(cond3),:),...
    altitudeLst(logical(cond3),:),snrLst(logical(cond3),:)];

cond1 = icaoOccurTb(:,2)>=500;
cond2 = icaoOccurTb(:,2)<=5000;
cond3 = icaoOccurTb(:,2)>=250;
cond4 = icaoOccurTb(:,2)<500;

cond = logical(cond1.*cond2);
selectedPlanes = icaoOccurTb(cond,:);
unknowPlanes = icaoOccurTb(logical(cond3.*cond4),:);
%Clip away ICAO IDs.
rawCompMatrix(:,1:32*8) = zeros(size(rawCompMatrix,1),32*8);
allTrainData = [icaoLst, abs(rawCompMatrix)];

minTrainChance = 200;
maxTrainChance = 500;

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
featureDims = numClasses;
layers = [
    imageInputLayer(inputSize, 'Name', 'input', 'Mean', 0)
    convolution2dLayer(5,10, 'Name', 'conv2d_1')
    batchNormalizationLayer('Name', 'batchNorm_1')
%     yxBatchNorm('batchNorm_1',28*28*10);
    reluLayer('Name', 'relu_1')
    convolution2dLayer(3, 10, 'Padding', 1, 'Name', 'conv2d_2')
    reluLayer('Name', 'relu_2')    
    convolution2dLayer(3, 10, 'Padding', 1, 'Name', 'conv2d_3')
    reluLayer('Name', 'relu_3')    
    %additionLayer(2,'Name', 'add_1')
    depthConcatenationLayer(2,'Name','add_1')    
    tensorVectorLayer('Flatten')
%     FCLayerAdapted(15680,featureDims, 'fc_bf_fp',[]) % 11th
    fullyConnectedLayer(featureDims, 'Name', 'fc_bf_fp') % 11th
    zeroBiasFCLayer(featureDims,numClasses,'Fingerprints',[])    
    yxSoftmax('softmax_1')
    classificationLayer('Name', 'classify_1')
    ];

lgraph = layerGraph(layers);
lgraph = connectLayers(lgraph, 'relu_1', 'add_1/in2');

XTrain = X;
YTrain = Y;
numEpochs = 10;
miniBatchSize = 256;
plots = "training-progress";
executionEnvironment = "gpu";
if plots == "training-progress"
    figure(10);
    lineLossTrain = animatedline('Color','#0072BD','lineWidth',1.5);
    lineClassificationLoss = animatedline('Color','#EDB120','lineWidth',1.5);
      
    ylim([-inf inf])
    xlabel("Iteration")
    ylabel("Loss")
    legend("Gross loss“,”Classif. Loss");
    grid on;
    
    figure(11);  
    lineCVAccuracy = animatedline('Color','#D95319','lineWidth',1.5);
    ylim([0 1.1])
    xlabel("Iteration")
    ylabel("CV Acc.")    
    grid on;   
    
    figure(12);  
    lineDepthOfTrain = animatedline('Color','#D95319','lineWidth',1.5);
    ylim([-inf inf])
    xlabel("Iteration")
    ylabel("Depth of Train.")    
    grid on;   
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
% lgraph2 = layerGraph(net); % Also collect old weights
% % OR:
lgraph2 = lgraph; % No old weights
lgraph2 = lgraph2.removeLayers('classify_1');
dlnet = dlnetwork(lgraph2);

% Loop over epochs.
totalIters = 0;
abandonFlg = 0;
for epoch = 1:numEpochs
    if abandonFlg == 1
        break;
    end
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
        [gradients,state,loss,classificationLoss,trainDepth] = dlfeval(@modelGradientsOnWeights,dlnet,dlX,Yb);
%         [gradients,state,loss] = dlfeval(@modelGradientsOnWeights,dlnet,dlX,Yb);        
        dlnet.State = state;
        % Determine learning rate for time-based decay learning rate schedule.
        learnRate = initialLearnRate;    
        
        % Update the network parameters using the SGDM optimizer.
        %[dlnet, velocity] = sgdmupdate(dlnet, gradients, velocity, learnRate, momentum);
        % Update the network parameters using the SGD optimizer.
        %dlnet = dlupdate(@sgdFunction,dlnet,gradients);
        if isempty(velocities)
            velocities = packScalar(gradients, 0);
            learnRates = packScalar(gradients, learnRate);
            momentums = packScalar(gradients, momentumSGD);
            L2Foctors = packScalar(gradients, 0);            
            gradientMasks = packScalar(gradients, 1);   
        end

        [dlnet, velocities] = dlupdate(@sgdmFunctionL2, ...
            dlnet, gradients, velocities, ...
            learnRates, momentums, L2Foctors, gradientMasks);
     
        % Display the training progress.
        if plots == "training-progress"
            D = duration(0,0,toc(start),'Format','hh:mm:ss');
            XTest = cX;
            YTest = categorical(cY);
            if mod(iteration,5) == 0 
                accuracy = cvAccuracy(dlnet, XTest,YTest,miniBatchSize,executionEnvironment,0);
                addpoints(lineCVAccuracy,iteration, accuracy);
%                 if accuracy > 0.9
%                    abandonFlg = 1;
%                    break;
%                 end
            end
            addpoints(lineLossTrain,iteration,double(gather(extractdata(loss))))
            addpoints(lineClassificationLoss,iteration,double(gather(extractdata(classificationLoss))));
%             [prevSumDevationCosine,~] = calculateGrossMutualDistance(dlnet.Layers({dlnet.Layers.Name} == "Fingerprints").Weights);
            addpoints(lineDepthOfTrain,iteration,double(extractdata(trainDepth)));
            title("Epoch: " + epoch + ", Elapsed: " + string(D))
            drawnow
        end
    end
end
[prevSumDevationCosine,simMatrix] = calculateGrossMutualDistance(dlnet.Layers({dlnet.Layers.Name} == "Fingerprints").Weights);
prevSumDevationCosine
accuracy = cvAccuracy(dlnet, cX, categorical(cY), miniBatchSize, executionEnvironment, 1)

%%%%%
%Gather fisher information and weights from the old network
prevDlnet = dlnet;
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
% accuracy = cvAccuracy(prevDlnet, prevCX, categorical(cY), miniBatchSize, executionEnvironment)
prevDlnet.State = state;
prevFisherInfo0 = prevGradients0;
% for i = 1:size(prevGradients0,1)
%     refMax = max(prevGradients0{end,3}{:},[],'all');
%     refMin = min(prevGradients0{end,3}{:},[],'all');
%     prevGradients0{i,3}={rescale(prevGradients0{i,3}{:},refMin,refMax)};
% end

for i = 1:size(prevFisherInfo0,1)
    prevFisherInfo0{i,3}={dlarray(exp(prevGradients0{i,3}{:}.^2))};    
%     prevFisherInfo0{i,3}={dlarray(exp(abs(prevGradients0{i,3}{:})))};
end
% [prevGradients1,state] = dlfeval(@logModelGradientsOnWeightsAnyLayer,...
%     prevDlnet, prevCX, 'fc_bf_fp');
% prevFisherInfo1 = prevGradients1;
% for i = 1:size(prevFisherInfo1,1)
%     prevFisherInfo1{i,3}={dlarray(exp(prevGradients1{i,3}{:}.^2))};    
% %     prevFisherInfo0{i,3}={dlarray(exp(abs(prevGradients0{i,3}{:}.^2)))};
% end



fingerprintLayerIdx = find(dlnet.Learnables{:,1} == "Fingerprints");

newClassesNum = floor(size(unique(uY),1));
unknownClassLabels = unique(uY);
idx = randperm(size(unknownClassLabels,1));
unknownClassLabels = unknownClassLabels(idx);
unknownClassLabels = unknownClassLabels(1:newClassesNum);

existingFingerprints = dlnet.Layers({dlnet.Layers.Name} == "Fingerprints").Weights;
newFingerprints = dlarray([]);
continualLearnX = dlarray([]);
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
    aUx_i = extractdata(squeeze(predict(dlnet,dlarray(ux_i,'SSCB'),'Outputs','fc_bf_fp')));
    unitAUx_i=(aUx_i./sqrt(sum(aUx_i.^2)))';
    fp_i = mean(aUx_i,2)';
    magFp_i = sqrt(sum(fp_i.^2));
    unitFp_i = fp_i ./ magFp_i;
    newFp = (unitFp_i);
%     newFp = zeros(1,numClasses);
    newFingerprints(end+1,:) = newFp;
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

numOldClasses = numClasses;
numClasses = numClasses + newClassesNum;
%Build new network and start continual learning.
lgraph2 = layerGraph(prevDlnet);
dlnet = dlnetwork(replaceLayer(lgraph2, 'Fingerprints', newFingerprintLayer));

XTrain = continualLearnX;
YTrain = continualLearnY;
XTest = cvContinualLearnX;
YTest = (cvContinualLearnY);

numEpochs = 5;
miniBatchSize = 20;
plots = "training-progress";
statusFigureIdx = [];
statusFigureAxis = [];
if plots == "training-progress"
    figure(21);
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
    
    figure(22);  
    lineDepthOfTrain = animatedline('Color','#D95319','lineWidth',1.5);
    ylim([-inf inf])
    xlabel("Iteration")
    ylabel("Depth of Train.")    
    grid on;   
end
L2RegularizationFactor = 0.01;
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

newCvAccuracy = cvAccuracy(dlnet, XTest,categorical(YTest),miniBatchSize,executionEnvironment,0);
oldCvAccuracy = cvAccuracy(dlnet, cX, categorical(cY), miniBatchSize,executionEnvironment,0);
disp('CV accuracy b.f. cont. learning w. random new FPs');
[newCvAccuracy, oldCvAccuracy]

fisherLossLst = [];
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
        [gradients,state,loss,fisherLoss,trainDepth] = dlfeval(@modelGradientsOnWeightsEWC,dlnet,dlX, Yb,...
            prevFisherInfo0, prevDlnet, newClassesNum, 1, fingerprintLayerIdx);                        

        fisherLossLst(end+1)=extractdata(gather(fisherLoss));

        dlnet.State = state;
        % Determine learning rate for time-based decay learning rate schedule.
        learnRate = initialLearnRate/(1 + decay*iteration);
%         learnRate = initialLearnRate;
        if isempty(velocities)
            velocities = packScalar(gradients, 0);
            learnRates = packScalar(gradients, learnRate);
            momentums = packScalar(gradients, momentum);
            L2Foctors = packScalar(gradients, 0.001);            
            gradientMasks = packScalar(gradients, 1);   
            % Let's lock some weights 
            for k = 1:fingerprintLayerIdx-1
                gradientMasks.Value{k}=dlarray(zeros(size(gradientMasks.Value{k})));
            end
            gradientMasks{fingerprintLayerIdx,3} = {dlarray([zeros(numOldClasses,featureDims); ones(newClassesNum,featureDims)])};

        end
        
        [dlnet, velocities] = dlupdate(@sgdmFunctionL2, ...
            dlnet, gradients, velocities, ...
            learnRates, momentums, L2Foctors, gradientMasks);

        % Display the training progress.
        if plots == "training-progress"
            D = duration(0,0,toc(start),'Format','hh:mm:ss');
%             figure(statusFigureIdx);
            if mod(iteration,20) == 0 
                newCvAccuracy = cvAccuracy(dlnet, XTest, categorical(YTest),miniBatchSize,executionEnvironment,0);
                oldCvAccuracy = cvAccuracy(dlnet, cX, categorical(cY), miniBatchSize,executionEnvironment,0);
                [~,~,oldCVLoss] = dlfeval(@modelGradientsOnWeights,dlnet,prevCX,prevCY);
                addpoints(lineNewCVAccuracy, iteration, newCvAccuracy);
                addpoints(lineOldCVAccuracy, iteration, oldCvAccuracy);
                addpoints(lineOldLossCV, iteration, double(gather(extractdata(oldCVLoss))));
            end
            addpoints(lineNewLossTrain,iteration,double(gather(extractdata(loss))))
%             addpoints(lineDepthOfTrain,iteration,double(extractdata(trainDepth)));

            title(statusFigureAxis,"Epoch: " + epoch + ", Elapsed: " + string(D))
            drawnow
        end
    end
end
figure
plot(fisherLossLst);
ylabel('Fisher Info. Loss');

newCvAccuracy = cvAccuracy(dlnet, XTest,categorical(YTest),miniBatchSize,executionEnvironment,0);
oldCvAccuracy = cvAccuracy(dlnet, cX, categorical(cY), miniBatchSize,executionEnvironment,0);
disp('CV accuracy b.f. cont. learning w. random new FPs');
[newCvAccuracy, oldCvAccuracy]

figure
subplot(2,1,1)
imagesc(prevDlnet.Layers({prevDlnet.Layers.Name} == "Fingerprints").Weights)
title('Old Finger prints');
subplot(2,1,2)
imagesc(dlnet.Layers({dlnet.Layers.Name} == "Fingerprints").Weights)
title('New Finger prints');

figure
weights = dlnet.Layers({dlnet.Layers.Name} == "Fingerprints").Weights;
similarMatrix = zeros(size(weights,1), size(weights,1));
for i = 1:size(weights,1)
    curWeight = weights(i,:);  
    magCurWeight = sqrt(sum(curWeight.^2,2));
    for j = 1:size(weights,1)
        nxtWeight = weights(j,:);  
        magNxtWeight = sqrt(sum(nxtWeight.^2,2));
        similarMatrix(i,j)=sum((nxtWeight./magNxtWeight)...
            .*(curWeight./magCurWeight));
    end
end
sumDeviationAngles = (sum(similarMatrix,'all') - sum(diag(similarMatrix)))./2
imagesc(similarMatrix)

[newSumDevationCosine,~] = calculateGrossMutualDistance(dlnet.Layers({dlnet.Layers.Name} == "Fingerprints").Weights)


function [sumDeviationAngles,similarMatrix] = calculateGrossMutualDistance(FPs)
    weights = FPs;
    similarMatrix = zeros(size(weights,1), size(weights,1));
    for i = 1:size(weights,1)
        curWeight = weights(i,:);  
        magCurWeight = sqrt(sum(curWeight.^2,2));
        for j = 1:size(weights,1)
            nxtWeight = weights(j,:);  
            magNxtWeight = sqrt(sum(nxtWeight.^2,2));
            similarMatrix(i,j)=sum(...
                (nxtWeight./magNxtWeight)...
                .*(curWeight./magCurWeight)...
                );
%             similarMatrix(i,j) = sqrt(sum((nxtWeight - curWeight).^2));
        end
    end
    sumDeviationAngles = (sum(similarMatrix,'all') - sum(diag(similarMatrix)))./2;
end

function fisherLoss = calcFisherLoss(prevFisherInfo0, dlnet, prevDlnet, newClassesNum,fingerprintLayerIdx)
    prevWeights = prevDlnet.Learnables.Value;
    curWeights = dlnet.Learnables.Value;
%     fisherLossMatrix = {};
    sumLoss = 0;
    elementCount = 1;
    for i = 1:size(prevWeights,1)
        if i >= fingerprintLayerIdx
            loss = ((prevWeights{i}-curWeights{i}(1:size(curWeights{i},1)-newClassesNum,:)).^2) .* prevFisherInfo0.Value{i};    
%             fisherLossMatrix{end+1} = loss;
            sumLoss = sumLoss + sum(loss(:));
            changedElements = prevWeights{i} ~= curWeights{i}(1:size(curWeights{i},1)-newClassesNum,:);
            elementCount = elementCount + sum(changedElements,'all');
        else
            loss = ((prevWeights{i}-curWeights{i}).^2) .* prevFisherInfo0.Value{i};
            changedElements = prevWeights{i} ~= curWeights{i};
%             fisherLossMatrix{end+1} = loss;
            sumLoss = sumLoss + sum(loss(:));
%             elementCount = elementCount + numel(prevWeights{i});            
            elementCount = elementCount + sum(changedElements,'all');
        end
    end
    fisherLoss = sumLoss;
end

function [gradients,state] = logModelGradientsOnWeights(dlnet,dlX)
    [dlYPred,state] = forward(dlnet,dlX);
    loglikelyhood = log(dlYPred-min(dlYPred(:))+1e-5);
    gradients = dlgradient(mean(loglikelyhood(:)),dlnet.Learnables);
end

function [gradients,state] = logModelGradientsOnWeightsAnyLayer(dlnet,dlX,outputName)
    [dlYPred,state] = (forward(dlnet,dlX,'Outputs',outputName));
    dlYPred = softmax(dlYPred);
    loglikelyhood = log(dlYPred-min(dlYPred(:))+1e-5);
    gradients = dlgradient(mean(loglikelyhood(:)),dlnet.Learnables);
end

function accuracy = cvAccuracy(dlnet, XTest, YTest, miniBatchSize, executionEnvironment, confusionChartFlg)
    dlXTest = dlarray(XTest,'SSCB');
    if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
        dlXTest = gpuArray(dlXTest);
    end
    dlYPred = squeeze(dlnet.predict(dlXTest));
    [~,idx] = max(extractdata(dlYPred),[],1);
    YPred = categorical(gather(idx));
    accuracy = mean(YPred(:) == YTest(:));
    if confusionChartFlg == 1
        figure
        confusionchart(YPred(:),YTest(:));
    end
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

function [gradients,state,loss,fisherLoss,trainDepth] = modelGradientsOnWeightsEWC(dlnet, dlX, Y,...
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
    fisherLoss = calcFisherLoss(prevFisherInfo0, dlnet, prevDlnet, ...
        newClassesNum, fingerprintLayerIdx)*ewcLambda/2;
    depth = 0;
%     FPs = dlnet.Learnables.Value(dlnet.Learnables.Layer == "Fingerprints");    
%     for i = 1:size(FPs{:},1)
%         for j = 1:i-1
%             dpt = sum(FPs{:}(i,:)./sqrt(sum((FPs{:}(i,:)).^2,'all')).*FPs{:}(j,:)./sqrt(sum((FPs{:}(j,:)).^2,'all')),'all');
%             depth = depth + dpt;
%         end
%     end
    trainDepth = depth;    
    
%     loss = crossentropy(squeeze(dlYPred),Y) + scalarL2Factor*penalty + fisherLoss;
    loss = crossentropy(stripdims(squeeze(dlYPred(:,:))),Y,'DataFormat','CB') ... 
        + scalarL2Factor*penalty + fisherLoss + sqrt(depth.^2);
    gradients = dlgradient(loss, dlnet.Learnables);
end

function [gradients,state,...
    loss,classificationLoss,trainDepth] = modelGradientsOnWeights(dlnet,dlX,Y)
%   %This is only used with softmax of matlab which only applies softmax
%   on 'C' and 'B' channels.
%     [rawPredictions,state] = forward(dlnet,dlX,'Outputs', 'Fingerprints');
%     dlYPred = softmax(dlarray(squeeze(rawPredictions),'CB'));
    [dlYPred,state] = forward(dlnet,dlX);
    penalty = 0;
    scalarL2Factor = 0;
    if scalarL2Factor ~= 0
        paramLst = dlnet.Learnables.Value;
        for i = 1:size(paramLst,1)
            penalty = penalty + sum((paramLst{i}(:)).^2);
        end
    end
    classificationLoss = crossentropy(squeeze(dlYPred),Y) + scalarL2Factor*penalty;
%     classificationLoss = crossentropy(stripdims(squeeze(dlYPred(:,:))),Y,'DataFormat','CB');
    FPs = dlnet.Learnables.Value(dlnet.Learnables.Layer == "Fingerprints");
    depth = 0;
    for i = 1:size(FPs{:},1)
        for j = 1:i-1
            dpt = sum(FPs{:}(i,:)./sqrt(sum((FPs{:}(i,:)).^2,'all')).*FPs{:}(j,:)./sqrt(sum((FPs{:}(j,:)).^2,'all')),'all');
            depth = depth + dpt;
        end
    end
    trainDepth = depth;
    loss = classificationLoss + sqrt(depth.^2);
%     loss = classificationLoss + 0.2*(max(max(rawPredictions))-min(max(rawPredictions)));    
    gradients = dlgradient(loss,dlnet.Learnables);
    %gradients = dlgradient(loss,dlnet.Learnables(4,:));
end

function [params,velocityUpdates,momentumUpdate] = adamFunction(params, rawParamGradients,...
    velocities, learnRates, momentums, L2Foctors, gradientMasks, iters)
    % https://arxiv.org/pdf/2010.07468.pdf %%AdaBelief
    % https://arxiv.org/pdf/1711.05101.pdf  %%DeCoupled Weight Decay 
    b1 = 0.5; 
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
    params = params - 0.0001.*(mt./(sqrt(vt)+e)).*gradientMasks...
        -L2Foctors.*params.*gradientMasks; %This works better for zero-bias dense layer
%     params = params - 0.001.*(h_mt./(sqrt(h_vt)+e)).*gradientMasks...
%         -L2Foctors.*params.*gradientMasks;

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


