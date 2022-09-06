clc
close all
clear 
input_data = importdata('Letter2Class.data'); % Bringing in the data to matlab and storing it in variable input_data
M = reshape(input_data.data,1576,16);
[m,n] = size(M); % This function is used to know the data dimensions excluding the label
P = 0.7 ; % This is propotion of training data
permuted_data = randperm(m); % Data is shuffled
size(permuted_data) % displays the size of permuted data to ensure that the whole data is used
trainX = M(permuted_data(1:round(P*m)),:); % 70 percent of data is allocated for training
size(trainX) % Displays the size of training data
testX = M(permuted_data(round(P*m)+1:end),:); % All rem
size(testX)
Training_y = input_data.rowheaders(permuted_data(1:round(P*m)),:) ;
Testing_y=input_data.rowheaders(permuted_data(round(P*m)+1:end),:);
trainX=trainX(:,1:2);
testX=testX(:,1:2);

for i=1:length(Training_y)
    if strcmp(Training_y(i,1),{'A'})==1
        training_y(i,1)=1;
    else
        training_y(i,1)=2;
    end
end

for i=1:length(Testing_y)
    if strcmp(Testing_y(i,1),{'A'})==1
        testing_y(i,1)=1;
    else
        testing_y(i,1)=2;
    end
    
end

r = 0.7; % the ratio of sampling in ensemble methods
% K = 3; % the parameter of Rotation Forest;
option=2;
labelT=training_y;
L = 1; % the number of parallel classifiers
preY=[];

for i=1:5
    [trainXsub,trainYsub] = resampling(trainX,labelT,testX,option,r);
     trainXsub=trainXsub(:,1:2);  % Classification is done only with two attributes for the sake of scatter plot
     trained_DT = fitctree(trainXsub,trainYsub);
     XX=trainXsub;
     x1range = min(XX(:,1)):.01:max(XX(:,1));
     x2range = min(XX(:,2)):.01:max(XX(:,2));
     [xx1, xx2] = meshgrid(x1range,x2range);
     XGrid = [xx1(:) xx2(:)]; 
     preY(:,i) = predict(trained_DT, XGrid); % This prediction value 
     clear trained_DT    
end

X=trainXsub;
species=training_y;
y = categorical(species);
figure
gscatter(X(:,1),X(:,2),species);
legend('A','X')
xlabel('A box');
ylabel('B box');
y = categorical(preY);
figure
gscatter(xx1(:), xx2(:),preY(:,1),'rg')
legend('A','X')
title('Decision plane from DT 1')
figure
gscatter(xx1(:), xx2(:),preY(:,2),'rg')
legend('A','X')
title('Decision plane from DT 2')
figure
gscatter(xx1(:), xx2(:),preY(:,3),'rg')
legend('A','X')
title('Decision plane from DT 3')
figure
gscatter(xx1(:), xx2(:),preY(:,4),'rg')
legend('A','X')
title('Decision plane from DT 4')
figure
gscatter(xx1(:), xx2(:),preY(:,5),'rg')
legend('A','X')
title('Decision plane from DT 5')

