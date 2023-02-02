clc
close all
clear 
NClusters = 2;

% Load the provided data %
rawData = importdata('Letter2Class.data');
dataSet = reshape(rawData.data,1576,16);
[NSamples,NFields] = size(dataSet);


%% cross validation with kmeans from MATLAB %%
v(:,1)=kmeanspp(dataSet,NClusters);
v(:,2)=kmeans(dataSet,NClusters);
csvwrite([fileparts(mfilename('fullpath')) '/crossValidationkmeanspp.csv'],v)


%% statistical results %%
NTrials = 5;
clusters = zeros(NTrials,NFields,NClusters);
for kTrial=1:NTrials
    [labels, clusters(kTrial,:,:)] = kmeanspp(dataSet,NClusters);
end

%% This was used to make the tables in the worksheets %%
%% local results %%
clusters

%% MATLAB's results %%
[v,C] = kmeans(dataSet,NClusters);
C