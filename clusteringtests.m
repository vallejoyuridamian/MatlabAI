clear all
close all
clc
%rng default; % uncoment this for reproducibility
X = [randn(100,2)*0.75+ones(100,2);randn(100,2)*0.5-ones(100,2)];

%% We call kmeans++ and plot the points and centroids%%
[idx,C] = kmeanspp(X,2);

figure;
plot(X(idx==1,1),X(idx==1,2),'r.','MarkerSize',12)
hold on
plot(X(idx==2,1),X(idx==2,2),'g.','MarkerSize',12)
plot(C(1,:),C(2,:),'kx','MarkerSize',15,'LineWidth',3) 
legend('Cluster 1','Cluster 2','Centroids','Location','NW')
title 'Cluster Assignments and Centroids for k-means++'
hold off

%% We call fuzzy cmeans and plot the points and centroids%%
[idx,C] = fuzzycmeans(X,2);

figure;
plot(X(idx==1,1),X(idx==1,2),'r.','MarkerSize',12)
hold on
plot(X(idx==2,1),X(idx==2,2),'g.','MarkerSize',12)
plot(C(1,:),C(2,:),'kx','MarkerSize',15,'LineWidth',3) 
legend('Cluster 1','Cluster 2','Centroids','Location','NW')
title 'Cluster Assignments and Centroids for fuzzy c-means'
hold off

%% We call EM and plot the points and centroids%%
[idx,C] = expectationmaximization(X,2);

figure;
plot(X(idx==1,1),X(idx==1,2),'r.','MarkerSize',12)
hold on
plot(X(idx==2,1),X(idx==2,2),'g.','MarkerSize',12)
plot(C(1,:),C(2,:),'kx','MarkerSize',15,'LineWidth',3) 
legend('Cluster 1','Cluster 2','Centroids','Location','NW')
title 'Cluster Assignments and Centroids for EM'
hold off