clear all
close all
clc

% set to true to use the dataset, false to use 2D cluster%
NNeuronsPerLayer = 4;
learningRate = 0.01;
ratioRandomSamples = 0.01;

dataSet = importdata('Letter2Class.data');
%[hyperplane counters] = adaboost(dataSet,NClassifiers);
[weights counters] = multilayerperceptron(dataSet,NNeuronsPerLayer, learningRate,ratioRandomSamples)
weights
accuracy = (counters.TP+counters.TN)/(counters.TP+counters.TN+counters.FP+counters.FN)
sensitivity = (counters.TP)/(counters.TP+counters.FN)
specificity = (counters.TN)/(counters.TN+counters.FP)

NRuns=5;
%% cross-validation %%
NSamplesData = length(dataSet.data);
NSamplesCrossValidation = round(NSamplesData*0.7);
for kRun=1:NRuns
    for kSample=1:NSamplesCrossValidation
        iSample = randi([1 NSamplesData]); % select random sample (it can be repeated)
        dataSetCV.data(kSample,:) = dataSet.data(iSample,:);
        dataSetCV.textdata(kSample) = dataSet.textdata(iSample);
        dataSetCV.rowheaders(kSample) = dataSet.rowheaders(iSample)';
    end
    weights = multilayerperceptron(dataSetCV,NNeuronsPerLayer, learningRate,ratioRandomSamples)
end

