function [projectionplane,counters]= singlenodedecisiontree (dataSet, addConstant)

% clc
% clear
% close all
% % Load the provided data %
% dataSet = importdata('Letter2Class.data');
% addConstant = false;

[NSamples,NFields] = size(dataSet.data);
covMatIni = 1e6;% 1e6; %Young 1984
threshold = 1e-4;
MaxIters = 1000;

counters.TP = 0; % 1 as  1
counters.TN = 0; %-1 as -1
counters.FP = 0; % 1 as -1
counters.FN = 0; %-1 as  1


if addConstant
    errorCov = covMatIni*eye(NFields + 1);
    weight = zeros(1,NFields + 1); % let it have a constant term
else
    errorCov = covMatIni*eye(NFields);
    weight = zeros(1,NFields);
end

projection = zeros(1,NSamples);

targetValues = labelsXAtoTarget1minus1(dataSet);

%% we make the prediction with 0 as threshold%%
prediction = zeros(1,NSamples);
iterating = true;
NIters = 0;
while iterating && (NIters < MaxIters)
    
    counters.TP = 0; % 1 as  1
    counters.TN = 0; %-1 as -1
    counters.FP = 0; % 1 as -1
    counters.FN = 0; %-1 as  1
    
    for kSample=1:NSamples
        if addConstant
            projection(kSample) = weight * [1 dataSet.data(kSample,:)]';
        else
            projection(kSample) = weight * dataSet.data(kSample,:)';
        end
        
        if projection(kSample) >=0
            prediction(kSample) = 1;
        else
            prediction(kSample) = -1;
        end
        
        if prediction(kSample) ~= targetValues(kSample)
            if targetValues(kSample) == 1 % 1 as  -1
                counters.FP = counters.FP + 1;
            else % -1 as 1
                counters.FN = counters.FN + 1;
            end
        else
            if targetValues(kSample) == 1 % 1 as  1
                counters.TP = counters.TP + 1;
            else % -1 as -1
                counters.TN = counters.TN + 1;
            end
        end
    end
    weightOld = weight;
    for kSample=1:NSamples
        
        if addConstant
            Xk = [1 dataSet.data(kSample,:)];
        else
            Xk = dataSet.data(kSample,:);
        end
        
        % The weights are calculated using Recursive Least Squares (RLS)
        % Young 1984
        
        % update errorCovarianceMatrix
        errorCov = errorCov - (errorCov * Xk') * ((1 + Xk * errorCov * Xk' )^-1) * (Xk * errorCov);
        % update weight
        weight = weight - ((errorCov * Xk') * ( Xk * weight' - targetValues(kSample)))';
        
    end
    
    NIters = NIters + 1;
    deltaWeight = norm(weight - weightOld);
    if deltaWeight < threshold
        iterating = false;
    end
    %NIters
    %NErrors
end
projectionplane = weight;
end