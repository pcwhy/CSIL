function [noiseVector,medoids] = extractNoise(inputSignal)
    mu = mean(inputSignal);
    sigma = std(inputSignal);
    
    % Use probabilistic filtering to restore the rational signal
    maskPostive = inputSignal <= mu + 3*sigma;
    maskNegative = inputSignal >= mu - 3*sigma;
    mask = maskPostive.*maskNegative;
    cleanInput = inputSignal.*mask;
    
    % Estimate the generative model
    lowMean = min(cleanInput);
    highMean = max(cleanInput);
    

    for rounds = 1:3
        highLst = [];
        lowLst = [];        
        for i = 1:length(cleanInput)
            if abs(cleanInput(i) - highMean) <= abs(cleanInput(i)-lowMean)
                highLst(end+1) = i;
            else
                lowLst(end+1) = i;
            end
        end
        highMean = mean(cleanInput(highLst));
        lowMean = mean(cleanInput(lowLst));
    end

     threshold = (highMean - lowMean)/2+lowMean;
     rationalSignal = double(cleanInput >= threshold);
     rationalSignal = rationalSignal.*(highMean - lowMean)+lowMean;
     noiseVector = (rationalSignal - cleanInput);  
     medoids = rationalSignal;

end

