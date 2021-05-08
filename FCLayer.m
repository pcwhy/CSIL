classdef FCLayer < nnet.layer.Layer

    properties (Learnable)
        % Layer learnable parameters
        Weights;
        Biases;
    end
    
    methods
        function layer = FCLayer(inputDim,outputDim,name,initialWeights) 
            % layer = weightedAdditionLayer(numInputs,name) creates a

            % Set number of inputs.
            %layer.NumInputs = inputDim;
            %layer.NumOutputs = numOutputs;
            % Set layer name.
            layer.Name = name;
            % Set layer description.
            layer.Description = "FC layer without bias neurons with " + inputDim +  ... 
                " inputs";
            % Initialize layer weights.
%             layer.Weights = dlarray(randn(outputDim,inputDim).*0.0001);
%             layer.Biases = dlarray(randn(outputDim,1).*0.0001);
            stdGlorot = sqrt(2/(inputDim + outputDim));
            layer.Weights = dlarray(rand(outputDim,inputDim).*stdGlorot);
            layer.Biases = dlarray(zeros(outputDim,1));
            
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
            Z = (layer.Weights*squeeze(X)+(layer.Biases ));
%             Z = layer.Weights*squeeze(X);

% %             Z = reshape((layer.Weights./sqrt(sum((layer.Weights).^2,2)))*squeeze(X),...
% %                 [1,1,size(layer.Weights,1),batchSize]); 
%             if ndims(X) >= 3
%                 Z = reshape(...
%                      ( layer.Weights )*( squeeze(X) ) + layer.Biases,...
%                                 [1,1,size(layer.Weights,1),batchSize]); 
% %                 Z = reshape(...
% %                      ( layer.Weights )*( squeeze(X) ),...
% %                                 [1,1,size(layer.Weights,1),batchSize]);                             
%             else
%                 Z = reshape(...
%                      ( layer.Weights )*( squeeze(X) ) + layer.Biases,...
%                                 [size(layer.Weights,1),batchSize]); 
% %                 Z = reshape(...
% %                      ( layer.Weights )*( squeeze(X) ) ,...
% %                                 [size(layer.Weights,1),batchSize]);                             
%             end
        end
    end
end
