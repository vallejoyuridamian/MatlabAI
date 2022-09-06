clear all
close all
clc



% set to true to use the dataset, false to use 2D cluster%
useDataSet = false;
NRuns = 5;
sep = 2; %separation between the clusters
rng default; % uncoment this for reproducibility



if useDataSet
    dataSet = importdata('Letter2Class.data');
    addConstant = true;
    [counters,kFieldBest,thresholdBest,InformationGainBest] = singlenodedecisiontree(dataSet);
    [kFieldBest,thresholdBest,InformationGainBest]
    accuracy = (counters.TP+counters.TN)/(counters.TP+counters.TN+counters.FP+counters.FN)
    sensitivity = (counters.TP)/(counters.TP+counters.FN)
    specificity = (counters.TN)/(counters.TN+counters.FP)
    
    %% cross-validation %%
    NSamplesData = length(dataSet.data);
    NSamplesCrossValidation = round(NSamplesData*0.7);
    for kRun=1:NRuns
        for kSample=1:NSamplesCrossValidation
            iSample = randi([1 NSamplesData]); % select random sample (it can be repeated)
            dataSetCV.data(kSample,:) = dataSet.data(iSample,:);
            dataSetCV.textdata(kSample) = dataSet.textdata(iSample);
            dataSetCV.rowheaders(kSample) = dataSet.rowheaders(iSample);
        end
        [counters,kFieldBest,thresholdBest,InformationGainBest] = singlenodedecisiontree(dataSet);
        [kFieldBest,thresholdBest,InformationGainBest]
    end
    
else
    
    X = [-sep+randn(100,2)*0.75+ones(100,2);-sep+randn(100,2)*0.5-ones(100,2)];
    labelsX = repelem([{'X'}], [length(X)])';
    Y = [sep+randn(100,2)*0.75+ones(100,2);sep+randn(100,2)*0.5-ones(100,2)];
    labelsY = repelem([{'A'}], [length(Y)])';
    
    Z.rowheaders = [labelsX;labelsY];
    Z.textdata = [labelsX;labelsY];
    Z.data = [X ; Y];
    
    NSamples = length(Z.data);
    
    figure;
    hold on;
    for kSample=1:NSamples
        if string(Z.rowheaders(kSample))=='X'
            plot(Z.data(kSample,1),Z.data(kSample,2),'r.','MarkerSize',12)
        end
        if string(Z.rowheaders(kSample))=='A'
            plot(Z.data(kSample,1),Z.data(kSample,2),'b.','MarkerSize',12)
        end
    end
    
    [counters,kFieldBest,thresholdBest,InformationGainBest] = singlenodedecisiontree(Z);
    
    hold on
    if kFieldBest ==1
        f = @(x1,x2)  thresholdBest*x1;
    else
        f = @(x1,x2)  thresholdBest*x2;
    end
    fimplicit(f,'k');
end
