function [labels, clusters] = kmeanspp(dataSet,NClusters)
    
    MaxIters=1e4;
    tolerance = 1e-6;
    
    [NSamples,NFields] = size(dataSet);
    clusters = zeros(NClusters,NFields);
    newClusters = zeros(NClusters,NFields);
    distance = zeros(1,NSamples);
    label = zeros(1,NSamples);
    idxS = 1:NSamples;
    %% Randomly select the first cluster %%
    idxRand = randi(NSamples);
    clusters(1,:)= dataSet(idxRand,:);
    %% For each data point compute distance to nearest cluster %%
    for kCluster=2:NClusters
        for kSample=1:NSamples
            distance(kSample) = norm(clusters(1,:) - dataSet(kSample,:));
            for jCluster=kCluster:-1:2
                if norm(clusters(jCluster,:) - dataSet(kSample,:)) < distance(kSample)
                    distance(kSample) = norm(clusters(jCluster,:) - dataSet(kSample,:));
                end
            end
        end
        %% Then select the next cluster with a probability proportional to that distance %%
        idxCluster= randsample(idxS,1,true,distance);
        clusters(kCluster,:) = dataSet(idxCluster,:);
    end

    %% Then we apply kmeans %%
    %% for each point we assign it to the nearest cluster %%
    iterating = true;
    NIters = 0;
    %% we iterate until the labels of the elements don't change or the iterations are too much
    while (NIters < MaxIters) && iterating
        NSamplesKCluster= zeros(1,NClusters);
        SumKCluster= zeros(NClusters,NFields);
        for kSample=1:NSamples
            oldLabel=label(kSample);
            label(kSample) = 1;
            distance(kSample) = norm(clusters(1,:) - dataSet(kSample,:));
            for kCluster=2:NClusters
                if norm(clusters(kCluster,:) - dataSet(kSample,:)) < distance(kSample)  
                    distance(kSample) = norm(clusters(kCluster,:) - dataSet(kSample,:));
                    label(kSample) = kCluster;
                end        
            end
            
            %% we can do the partial calculaion of the next cluster here %%
            NSamplesKCluster(label(kSample))= NSamplesKCluster(label(kSample)) + 1;
            SumKCluster(label(kSample),:)= SumKCluster(label(kSample),:) + dataSet(kSample,:);
        end

        %% Here we can compute the new clusters %%
        for kCluster=1:NClusters
            newClusters(kCluster,:) = SumKCluster(kCluster,:)/NSamplesKCluster(kCluster);
        end
        
        distance = norm(clusters  - newClusters);
        clusters = newClusters;
        if distance < tolerance
            iterating = false;
        end
            
        NIters = NIters + 1;
    end
    labels = label';
    clusters = clusters';
end