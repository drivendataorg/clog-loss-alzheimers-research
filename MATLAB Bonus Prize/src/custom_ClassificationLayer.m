classdef custom_ClassificationLayer < nnet.layer.ClassificationLayer
    
    properties
        % Vector of weights corresponding to the classes in the training
        % data
        ClassWeights
    end				
				

    methods
        function layer = custom_ClassificationLayer(classWeights,name)

    
            % Set layer name.
            layer.Name = name;
												layer.ClassWeights = classWeights;
            % Set layer description.
            layer.Description = 'custom_ClassificationLayer';
        end
        
								function loss = forwardLoss(layer, Y, T)
									% the predictions Y and the training targets T.
									Y=squeeze(Y);
									T=squeeze(T);
									N = size(Y,2);
									W = layer.ClassWeights;

									%Cross Entropy Loss%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
									lossce = -sum(sum(T.*log(Y),2))/(N);

									%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
									%Weighted Loss%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
									%lossw = -sum(W*(T.*log(Y)))/(N);	
									lossw = -sum(sum(T.*W'.*log(Y),2))/(N);
									loss =2*lossw+lossce;

								end
    end
end