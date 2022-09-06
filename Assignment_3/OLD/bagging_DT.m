clc
close all
clear 
raw_data = importdata('Letter2Class.data'); % Bringing in the data to matlab and storing it in variable input_data
M = reshape(raw_data.data,1576,16);
[m,n] = size(M); % This function is used to know the data dimensions excluding the label
P = 0.7 ; % This is propotion of training data
permuted_data = randperm(m); % Data is shuffled
size(permuted_data) % displays the size of permuted data to ensure that the whole data is used
trainX = M(permuted_data(1:round(P*m)),:); % 70 percent of data is allocated for training
size(trainX) % Displays the size of training data
testX = M(permuted_data(round(P*m)+1:end),:); % All rem
size(testX)
Training_y = raw_data.rowheaders(permuted_data(1:round(P*m)),:) ;
Testing_y=raw_data.rowheaders(permuted_data(round(P*m)+1:end),:);
%trainX=trainX(:,1:2);
%testX=testX(:,1:2);

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
%option=2;
trainY=training_y;
L = 1; % the number of parallel classifiers
preY=[];
fm_ = [];
trained = fitctree(trainX,trainY);

Htest=predict(trained, testX);
fm_=[fm_; confusion_matrix(testing_y,Htest)];
