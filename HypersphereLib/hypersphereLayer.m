classdef hypersphereLayer < nnet.layer.Layer
    % Example custom weighted addition layer.
    properties (Learnable)
        c;
    end
    properties
        c0;
    end

    methods
        function layer = hypersphereLayer(name,inputDim,R,lambda) 
            % layer = weightedAdditionLayer(numInputs,name) creates a

            % Set number of inputs.
            %layer.NumInputs = inputDim;
            %layer.NumOutputs = numOutputs;
            % Set layer name.
            layer.Name = name;
            layer.c = randn(inputDim,1);
            layer.c0 = layer.c+2;
            % Set layer description.
            layer.Description = "calculate the hypersphere loss";
        end
        
        function Z = predict(layer, X)
            if ndims(X) >= 3
                batchSize = size(X,4);
            else
                batchSize = size(X,ndims(X));
            end
            sumDist = sum((squeeze(X)-layer.c).^2,1);
            
            Z = sumDist';
%             Z = reshape(layer.Weights*squeeze(X)+(layer.Biases ),...
%                 [1,1,size(layer.Weights,1),batchSize]); 

        end
    end
end
