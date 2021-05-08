classdef weightedAdditionLayer < nnet.layer.Layer
    % Example custom weighted addition layer.

    properties (Learnable)
        % Layer learnable parameters
            
        % Scaling coefficients
        Weights
    end
    
    methods
        function layer = weightedAdditionLayer(numInputs,name) 
            % layer = weightedAdditionLayer(numInputs,name) creates a
            % weighted addition layer and specifies the number of inputs
            % and the layer name.

            % Set number of inputs.
            layer.NumInputs = numInputs;

            % Set layer name.
            layer.Name = name;

            % Set layer description.
            layer.Description = "Weighted addition of " + numInputs +  ... 
                " inputs";
        
            % Initialize layer weights.
            layer.Weights = rand(1,numInputs); 
        end
        
        function Z = predict(layer, varargin)
            % Z = predict(layer, X1, ..., Xn) forwards the input data X1,
            % ..., Xn through the layer and outputs the result Z.
            
            X = varargin;
            W = layer.Weights;
            
            % Initialize output
            X1 = X{1};
            sz = size(X1);
            Z = zeros(sz,'like',X1);
            
            % Weighted addition
            for i = 1:layer.NumInputs
                Z = Z + W(i)*X{i};
            end
        end
    end
end
