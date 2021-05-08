classdef zeroBiasFCLayerEuc < nnet.layer.Layer
    % Example custom weighted addition layer.

    properties (Learnable)
        % Layer learnable parameters
        Weights;
%         normMag = 1;        
    end
    properties
         normMag;
         b1;%, Biases;
    end
    methods
        function layer = zeroBiasFCLayerEuc(inputDim,outputDim,name,initialWeights) 
            % layer = weightedAdditionLayer(numInputs,name) creates a

            % Set number of inputs.
            %layer.NumInputs = inputDim;
            %layer.NumOutputs = numOutputs;
            % Set layer name.
            layer.Name = name;
            % Set layer description.
            layer.Description = "zero-bias FC layer with " + inputDim +  ... 
                " inputs";
            % Initialize layer weights.
            stdGlorot = sqrt(2/(inputDim + outputDim));
            layer.Weights = dlarray(rand(outputDim,inputDim).*stdGlorot);
            %layer.Weights = rand(outputDim,inputDim); 
            layer.normMag = 5;
            layer.b1 = 1e-9; % a real small number to maintain numerical stability.
            %layer.Biases = rand(outputDim,1);
            if numel(initialWeights) ~= 0
                layer.Weights = initialWeights;
            end
        end
        
        function Z = predict(layer, X)
            if ndims(X) >= 3
                batchSize = size(X,4);
            else
                batchSize = size(X,ndims(X));
            end

            if ndims(X) >= 3
%                 res = ( layer.Weights./sqrt((layer.b1)^2 + sum((layer.Weights).^2,2)) )...
%                     *( layer.normMag*squeeze(X)./sqrt((layer.b1)^2 + sum(squeeze(X).^2,1)));
%                 res = sqrt(-2*layer.Weights*squeeze(X) + sum((layer.Weights).^2,2) ...
%                     + sum(squeeze(X).^2,1));

%                 sx = squeeze(X)./sqrt((layer.b1)^2 + sum(squeeze(X).^2,1));
                sx = squeeze(X);
                res = sqrt(-2*layer.Weights*sx + sum((layer.Weights).^2,2) ...
                     + sum(sx.^2,1));
                Z = reshape(res, [1,1,size(layer.Weights,1),batchSize]); 
                        
            else
%                 res = sqrt(-2*layer.Weights*squeeze(X) + sum((layer.Weights).^2,2) ...
%                     + sum(squeeze(X).^2,1));

%                 sx = squeeze(X)./sqrt((layer.b1)^2 + sum(squeeze(X).^2,1));
                sx = squeeze(X);
                res = sqrt(-2*layer.Weights*sx + sum((layer.Weights).^2,2) ...
                     + sum(sx.^2,1));
                Z = reshape(res, [size(layer.Weights,1),batchSize]); 

            end
        end
    end
end
