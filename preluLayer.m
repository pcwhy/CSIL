classdef preluLayer < nnet.layer.Layer
    % Example custom PReLU layer.

    properties (Learnable)
        % Layer learnable parameters
        % Scaling coefficient
        Alpha
    end
    
    methods
        function layer = preluLayer(numChannels, name) 
            % layer = preluLayer(numChannels, name) creates a PReLU layer
            % with numChannels channels and specifies the layer name.
            % Set layer name.
            layer.Name = name;
            % Set layer description.
            layer.Description = "PReLU with " + numChannels + " channels";
            % Initialize scaling coefficient.
            layer.Alpha = rand([1 1 numChannels]); 
        end
        
        function Z = predict(layer, X)
            % Z = predict(layer, X) forwards the input data X through the
            % layer and outputs the result Z.
            Z = max(0, X) + layer.Alpha .* min(0, X);
        end
    end
end
