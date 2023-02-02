function [labels, centroids] = expectationmaximization(dataSet,NClusters)
% clc
% % close all
% clear
% NClusters = 2;
% Load the provided data %
% rawData = importdata('Letter2Class.data');
% dataSet = reshape(rawData.data,1576,16);
%dataSet = dataSet1(1:5,2:3);

MaxIters = 1e4;
tolerance = 1e-3;
[NSamples,NFields] = size(dataSet);
maxVal = zeros(1,NFields);
minVal = zeros(1,NFields);
means = zeros(NClusters,NFields);
sigma = zeros(NFields,NFields,NClusters);
sigmaNew = zeros(NFields,NFields,NClusters);
weight = zeros(1,NClusters);
weightNew = zeros(1,NClusters);
sampleWeight = zeros(1,NSamples);
responsibilities = zeros(NSamples,NClusters);
responsibilitiesDen = zeros(1,NSamples);
Nk = zeros(1,NClusters);
labels = zeros(1,NSamples);
probability = zeros(NSamples,NClusters);
meansNew = zeros(NClusters,NFields);
logLikelihood = 0;
%% Algorithm from Bishop's book %%
% Initialize the means ?k, covariances ?k and mixing coefficients ?k, and
% evaluate the initial value of the log likelihood.

%% We first calculate the maximum and minimum of each variable %%
for kField=1:NFields
    maxVal(kField) = max(dataSet(:,kField));
    minVal(kField) = min(dataSet(:,kField));
    %% We randomly select all the means within the range of the variables %%
    for kCluster=1:NClusters
        means(kCluster,kField) = minVal(kField) + rand*(maxVal(kField) - minVal(kField));
    end
end

%% We first calculate the covariance for the whole dataSet %%
for kCluster=1:NClusters
    sigma(:,:,kCluster) = rand*cov(dataSet);
end

%% Now the weigths %%
sumWeigths = 0 ;
for kCluster=1:NClusters;
    weight(kCluster) = rand;
    sumWeigths = sumWeigths  + weight(kCluster);
end
weight = weight/sumWeigths; %% we make them add to 1

iterating = true;
NIters = 0;
while (NIters < MaxIters) && iterating
    
    %% E step, evaluate the responsabilities using the current parameter values %%
    
    for kSample=1:NSamples
        
        %% we calculate the denominator for each sample %%
        responsibilitiesDen(kSample) = 0;
        for jCluster=1:NClusters
            responsibilitiesDen(kSample) = responsibilitiesDen(kSample) + weight(jCluster)*mvnpdf(dataSet(kSample,:),means(jCluster,:),sigma(:,:,jCluster));
        end
        %% calculate the numerator for each cluster and the responsabilities
        for kCluster=1:NClusters
            %% sum ?jN(xn|?j ,?j) in all the clusters
            responsibilities(kSample,kCluster) = weight(kCluster)*mvnpdf(dataSet(kSample,:),means(kCluster,:),sigma(:,:,kCluster)) / responsibilitiesDen(kSample);
        end
    end
    
    
    %% Nk if the number of elements of each cluster and Nk = sum_n=1^N ?(znk). %%
    for kCluster=1:NClusters
        Nk(kCluster) = 0;
        for kSample=1:NSamples
            Nk(kCluster) = Nk(kCluster) + responsibilities(kSample,kCluster);
        end
    end
    
    %% M step. Re-estimate the parameters using the current responsibilities %%
    
    %% meansNew = 1/Nk * sum_1^N(responibilities * Xn)
    for kCluster=1:NClusters
        meansNew(kCluster,:) = zeros(1,NFields);
        for kSample=1:NSamples
            meansNew(kCluster,:) = meansNew(kCluster,:) + dataSet(kSample,:) * responsibilities(kSample,kCluster);
        end
        meansNew(kCluster,:) = meansNew(kCluster,:) / Nk(kCluster);
    end
    
    %% sigmaNew= 1/Nk * sum_1^N(responibilities * (Xn - meansNew) * (Xn - meansNew)T)
    for kCluster=1:NClusters
        sigmaNew(:,:,kCluster) = zeros(NFields,NFields);
        for kSample=1:NSamples
            sigmaNew(:,:,kCluster) = sigmaNew(:,:,kCluster) + responsibilities(kSample,kCluster) * (dataSet(kSample,:) - meansNew(kCluster,:))' *(dataSet(kSample,:) - meansNew(kCluster,:));
        end
        sigmaNew(:,:,kCluster) = sigmaNew(:,:,kCluster) / Nk(kCluster);
    end
    
    %% weightsNew = Nk/N
    for kCluster=1:NClusters
        weightNew(kCluster) = Nk(kCluster)/NSamples;
    end

    
    sigma = sigmaNew;
    means = meansNew;
    weight = weightNew;
   
%     dist = 0;
%     for kCluster=1:NClusters
%         dist = dist + norm(sigmaNew(:,:,kCluster) - sigma(:,:,kCluster));
%     end
%        
%     dist = dist + norm(meansNew - means) + norm(weightNew - weight);
    
   
    
    %% Now we check if any of the clusters is collapsing to one of the points (covariance null, mean equal to a point) %%
    for kCluster=1:NClusters
        sigmaCollapsed = false;
        if (det(sigma(:,:,kCluster)) < 1e-10) || (isnan(det(sigma(:,:,kCluster))))
            sigmaCollapsed = true;
        end
        
        if sigmaCollapsed
            %% we reset sigma and means %%
            for kField=1:NFields
                means(kCluster,kField) = minVal(kField) + rand*(maxVal(kField) - minVal(kField));
            end
            %det(sigma(:,:,kCluster))
            sigma(:,:,kCluster) = 10*rand*cov(dataSet);
        end
    end
    
    % evaluate the initial value of the log likelihood. 
    logLikelihoodNew  = 0;
    for kSample=1:NSamples
        %% in all the samples the log of the sum in all the clusters... so
        %% logLikelihood = logLikelihood  + 1;
        sampleWeight(kSample) = 0;
        for kCluster=1:NClusters
            sampleWeight(kSample) = sampleWeight(kSample) + weight(kCluster)*mvnpdf(dataSet(kSample,:),means(kCluster,:),sigma(:,:,kCluster));
        end
        logLikelihoodNew  = logLikelihoodNew  + log(sampleWeight(kSample));
    end
    
    dist = norm(logLikelihood  - logLikelihoodNew);
    logLikelihood = logLikelihoodNew;

   
    if dist < tolerance
        iterating = false;
    end

    
    
    NIters = NIters + 1;
end

%% let's see for each sample what is the probability of being in each cluster %%
for kSample=1:NSamples
    %% we evaluate the weighted pdf and see which one is greater
    labels(kSample) = 1;
    probability(kSample,1) = weight(1)*mvnpdf(dataSet(kSample,:),means(1,:),sigma(:,:,1));
    probKSampleMax = probability(kSample,1);
    for kCluster=2:NClusters
        probability(kSample,kCluster) = weight(kCluster)*mvnpdf(dataSet(kSample,:),means(kCluster,:),sigma(:,:,kCluster));
        if probability(kSample,kCluster) > probKSampleMax
            probKSampleMax = probability(kSample,kCluster);
            labels(kSample) = kCluster;
        end
    end
end
centroids = means';
labels = labels';
% 
% means
% for kCluster=1:NClusters
%     detSigma= det(sigma(:,:,kCluster))
% end
% weight
% Nk
% responsibilities
end