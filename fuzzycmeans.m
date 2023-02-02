function [labels, clusters] =  fuzzycmeans(dataSet,NClusters)
   
    MaxIters = 1e4;
    tolerance = 1e-6;
    fuzziness  = 3; 
    [NSamples,NFields] = size(dataSet);
    
    maxVal = zeros(1,NFields);
    minVal = zeros(1,NFields);
    clusters = zeros(NClusters,NFields);
    label = zeros(1,NSamples);
    pointDistance = zeros(NSamples,NClusters);
    %% We first calculate the maximum and minimum of each variable %%
    for kField=1:NFields
        maxVal(kField) = max(dataSet(:,kField));
        minVal(kField) = min(dataSet(:,kField));
        %% We randomly select all the clusters within the range of the variables %%
        for kCluster=1:NClusters
            clusters(kCluster,kField) = minVal(kField) + rand*(maxVal(kField)-minVal(kField));
        end
    end
    %% Now we iterate until convergence %%
    NIters = 0;
    membership = zeros(NSamples,NClusters);
    iterating = true;
    newClusters = zeros(NClusters, NFields);
    while (NIters < MaxIters) && iterating

        %% for each sample we calculate the distance to each cluster %%
        for kSample=1:NSamples
            for kCluster=1:NClusters
                pointDistance(kSample,kCluster) = norm(clusters(kCluster,:) - dataSet(kSample,:));
            end
        end

        %% then for each sample we calculate the membership function %%
        for kSample=1:NSamples
            for kCluster=1:NClusters
                sumJ = 0;
                for jCluster=1:NClusters
                    sumJ = sumJ  + (pointDistance(kSample,kCluster) / pointDistance(kSample,jCluster))^(2/(fuzziness - 1));
                end
                membership(kSample,kCluster) = 1 / sumJ;
            end
        end

        %% Then we calculate the new clusters using the membership function %%

        %% The new clusters are calculated as the sum of the points times
        %% the membership function to the power of the fuzziness divided by the sum of the memberships 
        %% to the power of the fuzziness 


        for kCluster=1:NClusters
            sumOfWeightedDistance = zeros(NClusters,NFields);
            weight = zeros(1,NClusters);
            for kSample=1:NSamples
                sumOfWeightedDistance(kCluster,:) = sumOfWeightedDistance(kCluster,:) + dataSet(kSample,:) * membership(kSample,kCluster)^fuzziness;
                weight(kCluster) = weight(kCluster) + membership(kSample,kCluster)^fuzziness;
            end
            newClusters(kCluster,:) = sumOfWeightedDistance(kCluster,:) / weight(kCluster); 
        end

        distance = norm(clusters  - newClusters);
        clusters = newClusters;
        if distance < tolerance
            iterating = false;
        end
        NIters = NIters + 1;
    end
    %% we should output the labels %%
    for kSample=1:NSamples
        [maxMemb,idxMaxMemb] = max(membership(kSample,:)); 
        label(kSample) =   idxMaxMemb; 
    end
    labels = label';
    clusters = clusters';
 end     