
% %%%%
% Gather fisher information and weights from the old network
prevClassNum = 0;
if skipDAGNet == 0
    lgraph2 = layerGraph(net);
    lgraph2 = lgraph2.removeLayers('classify_1');
    weights = net.Layers({net.Layers.Name} == "Fingerprints").Weights;
    numClasses = size(weights, 1);
    prevClassNum = numClasses;
else
    lgraph2 = layerGraph(dlnet);
    numClasses = size(dlnet.Layers(14).Weights, 1);
    prevClassNum = numClasses;
end
prevDlnet = dlnetwork(lgraph2);
prevWeights = prevDlnet.Learnables;
prevCX = dlarray(single(cX), 'SSCB');
prevCY = zeros(numClasses, size(prevCX, 4), 'single');
for c = 1:numClasses
    prevCY(c, cY(:) == (c)) = 1;
end
executionEnvironment = "auto";
% If training on a GPU, then convert data to gpuArray.
if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
    prevCX = gpuArray(prevCX);
end
[prevGradients0, state] = dlfeval(@logModelGradientsOnWeights, prevDlnet, prevCX);
accuracy = cvAccuracy(prevDlnet, prevCX, (cY), 128, executionEnvironment)
prevDlnet.State = state;
prevFisherInfo0 = prevGradients0;
for i = 1:size(prevFisherInfo0, 1)
    prevFisherInfo0{i, 3} = {dlarray(exp(prevGradients0{i, 3}{:} .^ 2))};
    %     prevFisherInfo0{i,3}={dlarray(exp(prevGradients0{i,3}{:}.^2))};
 
end
fingerprintLayerIdx = 14;



newClassesNum = floor(size(unique(uY), 1));
unknownClassLabels = unique(uY);
idx = randperm(size(unknownClassLabels, 1));
unknownClassLabels = unknownClassLabels(idx);
unknownClassLabels = unknownClassLabels(1:newClassesNum);

% Generate some initial fingerprints.
% figure
% hold on;
existingFingerprints = prevDlnet.Layers(fingerprintLayerIdx).Weights;
newFingerprints = dlarray([]);
continualLearnX = dlarray([]);
continualLearnY = [];
cursor = 1;
for i = 1:newClassesNum
    selection = uY == unknownClassLabels(i);
    ux_i = uX(:, :, :, selection);
    uy_i = uY(selection);
    samplePerClass = size(uy_i, 1);
    continualLearnX(:, :, :, cursor:cursor + samplePerClass - 1) = ux_i;
    continualLearnY(cursor:cursor + samplePerClass - 1) = uy_i;
    cursor = cursor + samplePerClass;
    %     aUx_i = squeeze(prevDlnet.predict(dlarray(ux_i,'SSCB'),'Outputs','fc_bf_fp'));
    aUx_i = squeeze(activations(net, ux_i, 'fc_bf_fp'));
    unitAUx_i = (aUx_i ./ sqrt(sum(aUx_i .^ 2)))';
    fp_i = mean(aUx_i, 2)';
    magFp_i = sqrt(sum(fp_i .^ 2));
    unitFp_i = fp_i ./ magFp_i;
    newFp = (unitFp_i);
    %     newFp = findPotentialFingerprint(dlarray(existingFingerprints),dlarray(aUx_i));
    newFingerprints(end + 1, :) = newFp;
    %     histogram(sum(unitFp_i.*unitFingerprints,2),[-1:0.2:1],'Normalization','probability');
    %     hold on;
    %     histogram(sum(unitAUx_i.*unitFp_i,2),[-1:0.2:1],'Normalization','probability');
    %     legend('Correlation with existing fingerprints','Correlation with own samples');
end
randSeries = randperm(size(continualLearnY, 2));
continualLearnX = continualLearnX(:, :, :, randSeries);
continualLearnY = continualLearnY(randSeries);
cvContinualLearnX = continualLearnX(:, :, :, floor(0.6 * size(continualLearnX, 4)):end);
cvContinualLearnY = continualLearnY(floor(0.6 * size(continualLearnY, 2)):end);
continualLearnX = continualLearnX(:, :, :, 1:floor(0.6 * size(continualLearnX, 4)) - 1);
continualLearnY = continualLearnY(1:floor(0.6 * size(continualLearnY, 2)) - 1);

concatFingerprints = [existingFingerprints; newFingerprints];
% newFingerprintLayer = FCLayer(numClasses,numClasses+newClassesNum,'Fingerprints',concatFingerprints);
newFingerprintLayer = zeroBiasFCLayer(numClasses, prevClassNum + newClassesNum, 'Fingerprints', concatFingerprints);

% newFingerprintLayer.normMag = prevDlnet.Layers(4).normMag;
% newFingerprintLayer.b1 = prevDlnet.Layers(4).b1;

numClasses = prevClassNum + newClassesNum;

% Build new network and start continual learning.
lgraph2 = layerGraph(net);
lgraph2 = lgraph2.removeLayers('classify_1');
dlnet = dlnetwork(replaceLayer(lgraph2, 'Fingerprints', newFingerprintLayer));

if AdaptiveEWC == 1


    FPs = dlnet.Learnables.Value{dlnet.Learnables.Layer == "Fingerprints"}./...
        sqrt(sum((dlnet.Learnables.Value{dlnet.Learnables.Layer == "Fingerprints"}).^2,2));
    depth = 0;
    cursor = 1;
    stepSize = 20;
    regionLock = [];
    adaptiveFpMask = dlarray(ones(size(FPs)));
    while cursor + stepSize - 1 <= prevClassNum 
        weights = FPs(cursor:cursor+stepSize-1,:);
        depth = 0;
        for i = 1:size(weights,1)
            for j = 1:i-1
        %             dpt = sum(FPs{:}(i,:)./sqrt(sum((FPs{:}(i,:)).^2,'all')).*FPs{:}(j,:)./sqrt(sum((FPs{:}(j,:)).^2,'all')),'all');
                dpt = sum(weights(i,:).*weights(j,:),'all');
                depth = depth + dpt;
            end
        end       
        newAvgCosine = -1./(numClasses-1);
        prevSumDevLimit = stepSize*(stepSize-1)/2 * newAvgCosine; 
        if depth > prevSumDevLimit
           regionLock(end+1) = 0; 
           adaptiveFpMask(cursor:cursor+stepSize-1,:) = dlarray(zeros(size(weights)));
        else
           regionLock(end+1) = 1; 
           adaptiveFpMask(cursor:cursor+stepSize-1,:) = dlarray(ones(size(weights)));
        end
        cursor = cursor + stepSize;
    end
    regionLock
end



XTrain = continualLearnX;
YTrain = continualLearnY;
XTest = cvContinualLearnX;
YTest = (cvContinualLearnY);
numEpochs = 3;
miniBatchSize = 20;
plots = "training-progress";
statusFigureIdx = [];
statusFigureAxis = [];
if plots == "training-progress"
    figure;
    statusFigureIdx = gcf;
    statusFigureAxis = gca;
    % Go into the documentation of animatedline for more color codes
    lineNewLossTrain = animatedline('Color', '#0072BD', 'LineWidth', 1, 'Marker', '.', 'LineStyle', 'none');
    lineNewCVAccuracy = animatedline('Color', '#D95319', 'LineWidth', 1);
    lineOldCVAccuracy = animatedline('Color', '#EDB120', 'LineWidth', 1);
    lineOldLossCV = animatedline('Color', '#7E2F8E', 'LineWidth', 1, 'Marker', '.', 'LineStyle', 'none');
    %    ylim([0 inf])
    ylim([0 2])
    xlabel("Iteration")
    ylabel("Metrics")
    legend('New task loss', 'New task accuracy', 'Old task accuracy', 'Old task lost');
    grid on

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
numIterationsPerEpoch = floor(numObservations ./ miniBatchSize);
iteration = 0;
start = tic;
classes = categorical(YTrain);

%
prevCY = [prevCY; zeros(newClassesNum, size(prevCY, 2))];

newCvAccuracy = cvAccuracy(dlnet, XTest, YTest, miniBatchSize, executionEnvironment);
oldCvAccuracy = cvAccuracy(dlnet, cX, (cY), miniBatchSize, executionEnvironment);
disp('CV accuracy b.f. cont. learning');
[newCvAccuracy, oldCvAccuracy]
% Loop over epochs.
fisherLossLst = [];
for epoch = 1:numEpochs
    idx = randperm(numel(YTrain));
    XTrain = XTrain(:, :, :, idx);
    YTrain = YTrain(idx);
    % Loop over mini-batches.
    for i = 1:numIterationsPerEpoch
        iteration = iteration + 1;
        % Read mini-batch of data and convert the labels to dummy
        % variables.
        idx = (i - 1) * miniBatchSize + 1:i * miniBatchSize;
        Xb = XTrain(:, :, :, idx);
        Yb = zeros(numClasses, miniBatchSize, 'single');
        for c = 1:numClasses
            Yb(c, YTrain(idx) == (c)) = 1;
        end
        % Convert mini-batch of data to dlarray.
        dlX = dlarray(single(Xb), 'SSCB');
        % If training on a GPU, then convert data to gpuArray.
        if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
            dlX = gpuArray(dlX);
        end
        % Evaluate the model gradients, state, and loss using dlfeval and the
        % modelGradients function and update the network state.
        [gradients, state, loss, fisherLoss] = dlfeval(@modelGradientsOnWeightsEWC, dlnet, dlX, Yb, ...
        prevFisherInfo0, prevDlnet, newClassesNum, 1, fingerprintLayerIdx, prevClassNum);

        fisherLossLst(end + 1) = extractdata(gather(fisherLoss));

        dlnet.State = state;
        % Determine learning rate for time-based decay learning rate schedule.
        learnRate = initialLearnRate / (1 + decay * iteration);
        if isempty(velocities)
            velocities = packScalar(gradients, 0);
            learnRates = packScalar(gradients, learnRate);
            momentums = packScalar(gradients, momentum);
            L2Foctors = packScalar(gradients, 0.05);
            gradientMasks = packScalar(gradients, 1);
            % Let's lock some weights
            % Zero-bias dense layer only.
            %             if epoch <= 5
            %                 gradientMasks{fingerprintLayerIdx,3} = {dlarray([zeros(18,18); ones(newClassesNum,fingerprintLen)])};
            %             else
            %                 gradientMasks{fingerprintLayerIdx,3} = {dlarray([ones(18,18); ones(newClassesNum,fingerprintLen)])};
            %             end
            if lockOldFps == 1
                gradientMasks{fingerprintLayerIdx - 1, 3} = {dlarray([zeros(prevClassNum, 20); ones(newClassesNum, 20)])}; % specify whether lock old fps or not
            %             gradientMasks{9+1,3} = {dlarray([zeros(prevClassNum,1); ones(newClassesNum,1)])};
            end
            
            if AdaptiveEWC == 1
                gradientMasks{fingerprintLayerIdx - 1, 3} = {adaptiveFpMask};
            end            

            for k = 1:fingerprintLayerIdx - 2
                gradientMasks.Value{k} = dlarray(zeros(size(gradientMasks.Value{k})));
            end
        end
        
        FPs = dlnet.Learnables.Value{dlnet.Learnables.Layer == "Fingerprints"}./...
                    sqrt(sum((dlnet.Learnables.Value{dlnet.Learnables.Layer == "Fingerprints"}).^2,2));
        if AdaptiveEWC == 1
            depth = 0;
            cursor = 1;
            stepSize = 20;
            regionLock = [];        
            adaptiveFpMask = dlarray(ones(size(FPs)));
            while cursor + stepSize - 1 <= prevClassNum 
                weights = gpuArray(FPs(cursor:cursor+stepSize-1,:));
                depth = (sum(weights*weights','all')-stepSize)/2;

    %             depth = 0;
    %             for f = 1:size(weights,1)
    %                 for j = 1:f-1
    %             %             dpt = sum(FPs{:}(i,:)./sqrt(sum((FPs{:}(i,:)).^2,'all')).*FPs{:}(j,:)./sqrt(sum((FPs{:}(j,:)).^2,'all')),'all');
    %                     dpt = sum(weights(f,:).*weights(j,:),'all');
    %                     depth = depth + dpt;
    %                 end
    %             end       
                newAvgCosine = -1./(numClasses-1);
                prevSumDevLimit = stepSize*(stepSize-1)/2 * newAvgCosine; 
                if depth > prevSumDevLimit
    %                if cursor == 1
    %                   depth 
    %                end
                   regionLock(end+1) = 0; 
                   adaptiveFpMask(cursor:cursor+stepSize-1,:) = dlarray(zeros(size(weights)));
                else
                   regionLock(end+1) = 1; 
                   adaptiveFpMask(cursor:cursor+stepSize-1,:) = dlarray(ones(size(weights)));
                end
                cursor = cursor + stepSize;
            end   

            gradientMasks{fingerprintLayerIdx - 1, 3} = {adaptiveFpMask};
        end
        
        
        % fisherLoss = calcFisherLoss(prevFisherInfo0, dlnet, prevDlnet,newClassesNum)
        [dlnet, velocities] = dlupdate(@sgdmFunctionL2, ...
        dlnet, gradients, velocities, ...
        learnRates, momentums, L2Foctors, gradientMasks);

        % Display the training progress.
        if plots == "training-progress"
            D = duration(0, 0, toc(start), 'Format', 'hh:mm:ss');
            %             figure(statusFigureIdx);
            if mod(iteration, 20) == 0
                newCvAccuracy = cvAccuracy(dlnet, XTest, YTest, miniBatchSize, executionEnvironment);
                oldCvAccuracy = cvAccuracy(dlnet, cX, (cY), miniBatchSize, executionEnvironment);
                [~, ~, oldCVLoss] = dlfeval(@modelGradientsOnWeights, dlnet, prevCX, prevCY);
                addpoints(lineNewCVAccuracy, iteration, newCvAccuracy);
                addpoints(lineOldCVAccuracy, iteration, oldCvAccuracy);
                addpoints(lineOldLossCV, iteration, double(gather(extractdata(oldCVLoss))));
            end
            addpoints(lineNewLossTrain, iteration, double(gather(extractdata(loss))))
            title(statusFigureAxis, "Epoch: " + epoch + ", Elapsed: " + string(D))
            drawnow
        end
    end

    
end 

figure
weights = dlnet.Layers(14).Weights(:,:);
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
sumDeviationAnglesAll = (sum(similarMatrix,'all') - sum(diag(similarMatrix)))./2
imagesc(similarMatrix)

newCvAccuracy = cvAccuracy(dlnet, XTest,YTest,miniBatchSize,executionEnvironment);
oldCvAccuracy = cvAccuracy(dlnet, cX, (cY), miniBatchSize,executionEnvironment);
disp('CV accuracy a.f. cont. learning');
[newCvAccuracy, oldCvAccuracy]


function [totalLoss,corrOld,corrNew,grad] = evalFp(existingFp, newFeatures, fp)
    magFp = sqrt(sum(fp.^2));
    unitFp = fp ./ magFp;
    corrOld = mean(sum( existingFp./sqrt( sum((existingFp).^2,2)).*unitFp, 2 ));
    corrNew = mean(sum( newFeatures'./sqrt( sum((newFeatures').^2,2)).*unitFp, 2));
    totalLoss = corrOld - 10*corrNew;
    grad = dlgradient(totalLoss, fp);
end

function fp = findPotentialFingerprint(existingFp, newFeatures)
    fp = mean(newFeatures,2)';
%     magFp = sqrt(sum(fp.^2));
%     unitFp = fp ./ magFp;
    record = [];
    for i = 1:20
        [totalLoss,corrOld,corrNew, grad] = dlfeval(@evalFp, (existingFp), (newFeatures), fp);
        record = [record;[totalLoss,corrOld,corrNew]];
        fp = fp - 0.5.*grad;
    end    
%     figure
%     plot(extractdata(record(:,1)),'LineWidth',1.5);
%     hold on;
%     plot(extractdata(record(:,2)),'LineWidth',1.5);
%     plot(extractdata(record(:,3)),'LineWidth',1.5);
end

function [gradients,state,loss,fisherLoss] = modelGradientsOnWeightsEWC(dlnet, dlX, Y,...
    prevFisherInfo0, prevDlnet, newClassesNum, ewcLambda, fingerprintLayerIdx, prevClassNum)
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
        if i >= fingerprintLayerIdx-1
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
    loglikelyhood = log(dlYPred-min(dlYPred(:))+1e-5);
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

