classdef tensorVectorLayer < nnet.layer.Layer
    % Example custom weighted addition layer.
    methods
        function layer = tensorVectorLayer(name) 
            % layer = weightedAdditionLayer(numInputs,name) creates a

            % Set number of inputs.
            %layer.NumInputs = inputDim;
            %layer.NumOutputs = numOutputs;
            % Set layer name.
            layer.Name = name;
            % Set layer description.
            layer.Description = "flatten any tensor into vector";
        end
        
        function Z = predict(layer, X)
            if ndims(X) >= 3
                batchSize = size(X,4);
            else
                batchSize = size(X,ndims(X));
            end
            sX = squeeze(X);
            flattenX = reshape(sX,[1,1,numel(X)./batchSize,batchSize]);
%             flattenX(1,1,:) = 1e3.*flattenX(1,1,:)./sqrt(sum(flattenX(1,1,:).^2,'all'));
            Z = stripdims(flattenX);
%             Z = reshape(layer.Weights*squeeze(X)+(layer.Biases ),...
%                 [1,1,size(layer.Weights,1),batchSize]); 

%             Z = reshape((layer.Weights./sqrt(sum((layer.Weights).^2,2)))*squeeze(X),...
%                 [1,1,size(layer.Weights,1),batchSize]); 

        end
    end
end
