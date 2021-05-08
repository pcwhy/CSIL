%For experiment with different training depth
%Addition layer and dropout.

clc;
close all;
clear;
rng default;

addpath('./matplotlib')  
addpath('../');

load('adsb_records_qt.mat');
% load('adsb_bladerf2_10M_qt0.mat');
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

cond1 = icaoOccurTb(:,2)>=400;
cond2 = icaoOccurTb(:,2)<=8000;
cond3 = icaoOccurTb(:,2)>=350;
cond4 = icaoOccurTb(:,2)<500;

cond = logical(cond1.*cond2);
selectedPlanes = icaoOccurTb(cond,:);
unknowPlanes = icaoOccurTb(logical(cond3.*cond4),:);
%Clip away ICAO IDs.
rawCompMatrix(:,1:32*8) = zeros(size(rawCompMatrix,1),32*8);
allTrainData = [icaoLst, abs(rawCompMatrix)];

minTrainChance = 300;
maxTrainChance = 400;

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
% numClasses = size(unique(selectedBasebandData(:,1)),1);
numClasses = max(numel(unique(Y)),numel(unique(cY)));

layers = [
    imageInputLayer(inputSize, 'Name', 'input')
    convolution2dLayer(5,10, 'Name', 'conv2d_1')
    batchNormalizationLayer('Name', 'batchNorm_1')
    reluLayer('Name', 'relu_1')
    convolution2dLayer(3, 10, 'Padding', 1, 'Name', 'conv2d_2')
    reluLayer('Name', 'relu_2')    
    convolution2dLayer(3, 10, 'Padding', 1, 'Name', 'conv2d_3')
    reluLayer('Name', 'relu_3')    
    additionLayer(2,'Name', 'add_1')
%     depthConcatenationLayer(2,'Name','add_1')    
    tensorVectorLayer('Flatten')
    dropoutLayer(0.1,'Name','Dropout_1')
    fullyConnectedLayer(numClasses, 'Name', 'fc_bf_fp') % 11th
    batchNormalizationLayer('Name', 'batchNorm_2')

    
    zeroBiasFCLayer(numClasses,numClasses,'Fingerprints',[])
%     fullyConnectedLayer(numClasses, 'Name', 'Fingerprints') 


    %FCLayer(2*numClasses,numClasses,'Fingerprints',[])
    
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
    'MaxEpochs', 20, ...
    'MiniBatchSize',128,...
    'L2Regularization',0.01);
[net,info] = trainNetwork(X, categorical(Y), lgraph, options);
layerOneWeights = zeros(5*size(net.Layers(2).Weights,4)*inputSize(3),5);
cursor = 1;
for i = 1:size(net.Layers(2).Weights,4)
    for j = 1:inputSize(3)
        layerOneWeights(cursor:cursor+5-1,:) ...
            = squeeze(net.Layers(2).Weights(:,:,j,i));
        cursor = cursor + 5;
    end
end
%iacc2 = find(~isnan(info.ValidationAccuracy) == 1)
%vacc2 = info.ValidationAccuracy(~isnan(info.ValidationAccuracy) == 1);
cursor = 1;
layerOneResponse = {};
for i = 1:size(net.Layers(2).Weights,4)*inputSize(3)
    [H,W] = freqz2(layerOneWeights(cursor:cursor+5-1,:),1024);
    cursor = cursor + 5;
    layerOneResponse{end+1} = {H,W};
end
figure
cursor = 1;
for i = 1:5
    for j = 1:6
      subplot(5,6,cursor);
      imagesc(abs(layerOneResponse{cursor}{1}));
      cursor = cursor + 1;
    end
end

YPred = classify(net, cX);
accuracy = sum(categorical(cY) == YPred)/numel(cY)
cm = confusionmat(categorical(cY),YPred);
cm = cm./sum(cm,2);
figure
imagesc(cm);


weights = net.Layers({net.Layers.Name}=="Fingerprints").Weights;
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
rationalSumDevAngles = -numClasses./2
figure;imagesc(similarMatrix)




