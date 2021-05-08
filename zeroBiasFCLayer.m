classdef zeroBiasFCLayer < nnet.layer.Layer
    % Example custom weighted addition layer.

    properties (Learnable)
        % Layer learnable parameters
        Weights;
    end
    properties
         normMag;
         b1;%, Biases;
    end
    methods
        function layer = zeroBiasFCLayer(inputDim,outputDim,name,initialWeights) 
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
            stdGlorot = sqrt(1/(inputDim + outputDim));
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
%             size(X)
%             Z = reshape(layer.Weights*squeeze(X)+(layer.Biases ),...
%                 [1,1,size(layer.Weights,1),batchSize]); 

%             Z = reshape((layer.Weights./sqrt(sum((layer.Weights).^2,2)))*squeeze(X),...
%                 [1,1,size(layer.Weights,1),batchSize]); 
            %size(X)
            if ndims(X) >= 3
                Z = reshape(...
                     ( layer.Weights./sqrt((layer.b1)^2 + sum((layer.Weights).^2,2)) )...
                    *( layer.normMag*squeeze(X)./sqrt((layer.b1)^2 + sum(squeeze(X).^2,1))),...
                                [1,1,size(layer.Weights,1),batchSize])+layer.normMag; 
%                 Z = reshape(...
%                      ( layer.Weights./sqrt((layer.b1)^2 + sum((layer.Weights).^2,2)) )...
%                     *( layer.normMag*squeeze(X)),...
%                                 [1,1,size(layer.Weights,1),batchSize])+layer.normMag;                             
            else
                Z = reshape(...
                     ( layer.Weights./sqrt((layer.b1)^2 + sum((layer.Weights).^2,2)) )...
                    *( layer.normMag*squeeze(X)./sqrt((layer.b1)^2 + sum(squeeze(X).^2,1))),...
                                [size(layer.Weights,1),batchSize])+layer.normMag; 
%                 Z = reshape(...
%                      ( layer.Weights./sqrt((layer.b1)^2 + sum((layer.Weights).^2,2)) )...
%                     *( layer.normMag*squeeze(X)),...
%                                 [size(layer.Weights,1),batchSize])+layer.normMag; 
            end
        end
    end
end
