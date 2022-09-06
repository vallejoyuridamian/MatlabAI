function [hyperplane counters] = adaboost(dataSet,NClassifiers)

% clc
% clear
% close all
% % Load the provided data %
% dataSet = importdata('Letter2Class.data');
% NClassifiers = 3;

[NSamples,NFields] = size(dataSet.data);

% Initialize the data weighting coefficients {wn} by setting w(1) n = 1/N for
% n = 1, . . . , N.
weight = zeros(NClassifiers,NSamples);
epsilon = zeros(1,NClassifiers);
alpha = zeros(1,NClassifiers);
for kSample=1:NSamples
    weight(1,kSample) = 1/NSamples;
end

targetValues = labelsXAtoTarget1minus1(dataSet);

NErrors = zeros(1,NClassifiers);
indicator = zeros(NClassifiers,NSamples);
weightedSamples = zeros(NSamples,NFields);
addConstant = true;

projection = zeros(1,NSamples);
prediction = zeros(1,NSamples);
models = zeros(NClassifiers,NFields + 1);

for kClassifier=1:NClassifiers
    %% weight the samples
    for kSample=1:NSamples
        weightedSamples(kSample,:) = weight(kClassifier,kSample) * dataSet.data(kSample,:);
    end
    dataTrain.data =weightedSamples;
    dataTrain.textdata =dataSet.textdata;
    dataTrain.rowheaders =dataSet.rowheaders;
    %% fit a classifier to train the weighted data %%
    models(kClassifier,:) = singlenodedecisiontree(dataTrain,addConstant);
    % model = fitcdiscr(weightedSamples, dataSet.rowheaders);
    
    %label = predict(model,dataSet.data(1,:));
    for kSample=1:NSamples
        
        projection(kSample) = models(kClassifier,:) * [1 weightedSamples(kSample,:)]';
        
        if projection(kSample) >=0
            prediction(kSample) = 1;
        else
            prediction(kSample) = -1;
        end
        
        if prediction(kSample) ~= targetValues(kSample)
            indicator(kClassifier,kSample) = 1;
            NErrors(kClassifier) = NErrors(kClassifier) + 1;
        end
    end
    
    %% now we evaluate epsilon and alpha which are:
    %% epsilon = the weighted measures of the error rates of each of the
    %% base classifiers on the data set
    %% and
    %% alpha = weighting coefficients which give greater weight to the more accurate classifiers
    %% when computing the overall output
    epsilonNum = 0;
    epsilonDen = 0;
    for kSample=1:NSamples
        epsilonNum = epsilonNum + weight(kClassifier,kSample) * indicator(kClassifier,kSample);
        epsilonDen = epsilonDen + weight(kClassifier,kSample);
    end
    epsilon(kClassifier) = epsilonNum / epsilonDen;
    alpha(kClassifier) = log((1-epsilon(kClassifier))/epsilon(kClassifier));
    
    %% update the weighting coefficients for the next classifier %%
    if kClassifier < NClassifiers %% we will not do it for the last one
        for kSample=1:NSamples
            weight(kClassifier + 1,kSample) = weight(kClassifier,kSample) *  exp (alpha(kClassifier) * indicator(kClassifier,kSample));
        end
    end
    % hyperplanes(kClassifier,1) = model.Coeffs(1,2).Const;
    % hyperplanes(kClassifier,2:NFields + 1)  = model.Coeffs(1,2).Linear;
    
end

%% The adaboost classifier's decision will be the sum of the alphas(kClassifier)*prediction(kClassifier)
adaBoostClassifier = zeros(1, NFields+1);
for kClassifier=1:NClassifiers
    adaBoostClassifier = adaBoostClassifier + alpha(kClassifier) * models(kClassifier,:);
end

%% let's evaluate the adaBoost with the original data%%
counters.TP = 0; % 1 as  1
counters.TN = 0; %-1 as -1
counters.FP = 0; % 1 as -1
counters.FN = 0; %-1 as  1
for kSample=1:NSamples
    projection(kSample) = adaBoostClassifier * [1 dataSet.data(kSample,:)]';
    
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
hyperplane = adaBoostClassifier;

end