clc
clear 
close all

NClusters = 2;
% Load the provided data %
rawData = importdata('Letter2Class.data');
dataSet = reshape(rawData.data,1576,16);
[NSamples,NFields] = size(dataSet);
% dataSet = dataSet1(1:3,2:3);

%% cross validation with kmeans from MATLAB %%
v(:,1)=fuzzycmeans(dataSet,NClusters);
v(:,2)=kmeans(dataSet,NClusters);
csvwrite([fileparts(mfilename('fullpath')) '/crossValidationfuzzycmeans.csv'],v)


%% statistical results %%
NTrials = 5;
clusters = zeros(NTrials,NFields,NClusters);
for kTrial=1:NTrials
    [labels, clusters(kTrial,:,:)] = fuzzycmeans(dataSet,NClusters);
end

%% This was used to make the tables in the worksheets %%
%% local results %%
clusters

%% MATLAB's results %%
[v,C] = kmeans(dataSet,NClusters);
C