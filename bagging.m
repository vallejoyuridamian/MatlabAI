function [hyperplane, counters] = bagging(dataSet,NBags,ratio)

%clc
%clear
%close all
% 0 < ratio <= 1

withReplacement = true;

% Load the provided data %
% dataSet = importdata('Letter2Class.data');
%dataSet = dataSet1(1:3,2:3);
[NSamples,NFields] = size(dataSet.data);

%NBags = 5;
%ratio = 0.7;
%NBags = 1;

NSamplesBag = round(ratio*NSamples);

%% Bootstraping %%
% so we take round(ratio*NSamples) elements from the data set
% with replacement to make NBags.

bags = zeros(NBags,NSamplesBag,NFields);
labels = strings(NBags,NSamplesBag);
for kBag=1:NBags
    for kSample=1:NSamplesBag
        idxSample = randsample(NSamples,1,withReplacement);
        bags(kBag,kSample,:)  = dataSet.data(idxSample,:);
        labels(kBag,kSample) = dataSet.rowheaders(idxSample);
    end
end

%% Parallel training %%
%% For each bag we calculate the best fitting hyperplane %%
%% we will be using MATLAB's fitted discriminant %%

hyperplanes = zeros(NBags,NFields + 1);
for kBag=1:NBags
    model = fitcdiscr(squeeze(bags(kBag,:,:)),labels(kBag,:));
    % we will put the constant in the beginning
    hyperplanes(kBag,1) = model.Coeffs(1,2).Const;
    hyperplanes(kBag,2:NFields + 1)  = model.Coeffs(1,2).Linear;
end

%% Aggregation %%
%% we average the coefficents of the hyperplanes $$
hyperplane = zeros(1,NFields + 1);
for kField=1:NFields+1
    hyperplane(kField) = mean(hyperplanes(:,kField));
end

%% Evaluation %%
counters.TP = 0; % 1 as  1
counters.TN = 0; %-1 as -1
counters.FP = 0; % 1 as -1
counters.FN = 0; %-1 as  1

for kSample=1:NSamples
    evaluation =  [1 dataSet.data(kSample,:)] * hyperplane';
    if evaluation >=0
        prediction = 'A';
    else
        prediction = 'X';
    end
    if prediction ~= string(dataSet.rowheaders(kSample))
        if string(dataSet.rowheaders(kSample)) == 'A' % X as  A
            counters.FP = counters.FP + 1;
        else % A as x
            counters.FN = counters.FN + 1;
        end
    else
        if string(dataSet.rowheaders(kSample)) == 'A' % 1 as  1
            counters.TP = counters.TP + 1;
        else % -1 as -1
            counters.TN = counters.TN + 1;
        end
    end
end
end