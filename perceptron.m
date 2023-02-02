function [hyperplane, counters] = perceptron(dataSet,learningRate)

% clear
% close all
% clc

% learningRate=0.001;
% dataSet = importdata('Letter2Class.data');

[NSamples,NFields] = size(dataSet.data);
MaxIters = 1000;
tolerance = 1e-3;
targetValues = labelsXAtoTarget1minus1 (dataSet);


% let's use the t vector as in Bishop's book and use A as 1 and X as -1 so
% we run the samples and label as such


%% Evaluate the perceptron function %%
error= zeros(1,NSamples);
weights = zeros(1,NFields + 1); %it includes the bias as the first value

iterating = true;
NIters = 0;
while iterating && (NIters < MaxIters)
    
    counters.TP = 0; % 1 as  1
    counters.TN = 0; %-1 as -1
    counters.FP = 0; % 1 as -1
    counters.FN = 0; %-1 as  1
    
    for kSample=1:NSamples
        % prediction(kSample) = actFun( weights * fixedNonLinTrans(dataSet.data(kSample,:))');
        activation =  weights * [1 dataSet.data(kSample,:)]';
        if activation >=0 
            prediction = 1;
        else
            prediction = -1;
        end
        error(kSample) = targetValues(kSample) - prediction;
        if error(kSample) ~= 0 %% incorrect classification
            if targetValues(kSample) == 1 % 1 as  -1
                counters.FN = counters.FN + 1;
            else % -1 as 1
                counters.FP = counters.FP + 1;
            end
            
        else
            if targetValues(kSample) == 1 % 1 as  1
                counters.TP = counters.TP + 1;
            else % -1 as -1
                counters.TN = counters.TN + 1;
            end
        end
    end
    
    %% calculate new weights
    weightsOld = weights;
    for kSample=1:NSamples
       weights= weights + error(kSample) * learningRate * [1 dataSet.data(kSample,:)];
    end
    weightsNew = weights;
    
    changeWeight = norm(weightsNew - weightsOld);
    
    if changeWeight < tolerance
        iterating = false;
    end
    NIters = NIters + 1;
end
hyperplane = weights;
end