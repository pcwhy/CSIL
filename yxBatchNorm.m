classdef yxBatchNorm < nnet.layer.Layer

    properties (Learnable)
        gama;
        beta;
    end
    methods
        function layer = yxBatchNorm(name, indim) 
            % Set layer name.
            layer.Name = name;
            layer.gama = randn(indim,1);
            layer.beta = randn(indim,1);
            % Set layer description.
            layer.Description = "Yongxin's BatchNorm for compatibility";
        end
        function Z = predict(layer, X)
            eps = 1e-9;
            batchMu = mean(X,2);
            batchVar = var(X,0,2);
            Z = layer.gama.*(X-batchMu)./(sqrt(batchVar+eps)) + layer.beta;
        end
    end
end
