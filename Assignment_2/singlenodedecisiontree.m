function [counters, kFieldBest,thresholdBest,InformationGainBest]= singlenodedecisiontree (dataSet)

% clc
% clear
% close all
% % % Load the provided data %
% dataSet = importdata('Letter2Class.data');
% % addConstant = false;

[NSamples,NFields] = size(dataSet.data);

threshold = 1e-6;
NthresholdsPerVariable=100;

thresholdBest=zeros(1,NFields);
maxVal=zeros(1,NFields);
minVal=zeros(1,NFields);
NErrorsBest=zeros(1,NFields);

counters.TP = 0; % 1 as  1
counters.TN = 0; %-1 as -1
counters.FP = 0; % 1 as -1
counters.FN = 0; %-1 as  1


%% Let's fist calculate the entropy of the whole dataset %%
targetValues = labelsXAtoTarget1minus1(dataSet);
NClass1= 0;
NClass2= 0;
for kSample=1:NSamples
    if targetValues(kSample) == 1
        NClass1 = NClass1 + 1;
    else
        NClass2 = NClass2 + 1;
    end
end

pClass1 = NClass1/(NClass1 + NClass2);
pClass2 = NClass2/(NClass1 + NClass2);

Entropy = -1 * (pClass1*log2(pClass1) + pClass2*log2(pClass2));

% We calculate the best split for each variable and the information gain
% for that split and we keep the variable with the best information gain as
% the
InformationGainBest = 0;
for kField=1:NFields
    minVal(kField) = min(dataSet.data(:,kField));
    maxVal(kField) = max(dataSet.data(:,kField));
    bestThreshold = minVal(kField);
    NErrorsBest(kField) = length(dataSet.data(:,kField));
    for kThreshold=1:NthresholdsPerVariable +1
        threshold = minVal(kField) + (maxVal(kField)-minVal(kField))*(kThreshold - 1)/NthresholdsPerVariable;
        
        % for that treshold we separate the data and count the
        % missclassified samples, we will keep the one with the leas
        % missclassified samples
        NErrors = 0;
        NElementsSet1 = 0;
        NElementsSet2 = 0;
        NClass1InSet1 = 0;
        NClass2InSet1 = 0;
        NClass1InSet2 = 0;
        NClass2InSet2 = 0;
        for kSample=1:NSamples
            if dataSet.data(kSample,kField) >= threshold
                NElementsSet1 = NElementsSet1 + 1;
                if targetValues(kSample) == 1
                    NClass1InSet1 = NClass1InSet1 + 1;
                else
                    NClass2InSet1 = NClass2InSet1 + 1;
                end
            else
                NElementsSet2 = NElementsSet2 + 1;
                if targetValues(kSample) == 1
                    NClass1InSet2 = NClass1InSet2 + 1;
                else
                    NClass2InSet2 = NClass2InSet2 + 1;
                end
            end
        end
        % now we calculate the entropy for that split
        % for set 1
        EntropySet1 = 0;
        if (NClass1InSet1 > 0) && (NClass2InSet1 > 0)
            pClass1InSet1 = NClass1InSet1/(NElementsSet1);
            pClass2InSet1 = NClass2InSet1/(NElementsSet1);
            EntropySet1 = -1 * (pClass1InSet1*log2(pClass1InSet1) + pClass2InSet1*log2(pClass2InSet1));
        end
        % for set 2
        EntropySet2 = 0;
        if (NClass1InSet2 > 0) && (NClass2InSet2 > 0)
            pClass1InSet2 = NClass1InSet2/(NElementsSet2);
            pClass2InSet2 = NClass2InSet2/(NElementsSet2);
            EntropySet2 = -1 * (pClass1InSet2*log2(pClass1InSet2) + pClass2InSet2*log2(pClass2InSet2));
        end
        
        EntropySplit = EntropySet1 + EntropySet2;
        
        InformationGainNew = Entropy - EntropySplit;
        if InformationGainNew > InformationGainBest
            InformationGainBest = InformationGainNew;
            kFieldBest = kField;
            thresholdBest = threshold;
        end
    end
end

% sweep once more to calculate the counters
for kSample=1:NSamples
    if dataSet.data(kSample,kFieldBest) >= thresholdBest
        % here we predict a 1
        if targetValues(kSample) == 1 % 1 as  1
            counters.TP = counters.TP + 1;
        else % -1 as 1
            counters.FP = counters.FP + 1;
        end
    else
        % here we predict a -1
        if targetValues(kSample) == -1 % -1 as  -1
            counters.TN = counters.TN + 1;
        else % 1 as -1
            counters.FN = counters.FN + 1;
        end
    end
end
end