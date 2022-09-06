clc
clear
close all

NClusters = 2;
% Load the provided data %
rawData = importdata('Letter2Class.data');
dataSet = reshape(rawData.data,1576,16);
[NSamples,NFields] = size(dataSet);

%% here we make a csv file to see the labelling of each method %%
v(:,1)=expectationmaximization(dataSet,NClusters);
v(:,2)=kmeans(dataSet,NClusters);

csvwrite([fileparts(mfilename('fullpath')) '/crossValidationEM.csv'],v)



%% statistical results %%
NTrials = 5;
clusters = zeros(NTrials,NFields,NClusters);
for kTrial=1:NTrials
    NTrial = kTrial
    [labels, clusters(kTrial,:,:)] = expectationmaximization(dataSet,NClusters);
end

%% This was used to make the tables in the worksheets %%
%% local results %%
clusters

%% MATLAB's results %%
[v,C] = kmeans(dataSet,NClusters);
C


