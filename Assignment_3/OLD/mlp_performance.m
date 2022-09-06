clc
close all
clear 
raw_data = importdata('Letter2Class.data');
M = reshape(raw_data.data,1576,16);
M1=raw_data.rowheaders;
[m,n] = size(M) ;
P = 0.7 ;
shuffle = randperm(m);
size(shuffle)
Training_x = M(shuffle(1:round(P*m)),:);
size(Training_x )
Testing_x = M(shuffle(round(P*m)+1:end),:);
size(Testing_x)
Training_y=raw_data.rowheaders(shuffle(1:round(P*m)),:) ;
Testing_y=raw_data.rowheaders(shuffle(round(P*m)+1:end),:);

% The following 'for' loop is to convert the labels 'A' and 'X' into 1 and % 0 respectively for training datasets
for i=1:length(M1)
    if strcmp(M1(i,1),{'A'})==1
        training_y(i,1)=1;
    else
        training_y(i,1)=0;
    end
    
end

% initial network
train_net = network(1, 2, [1; 1], [1;0], [0 0; 1 0], [0 1]);

train_net.adaptFcn = 'adaptwb';
train_net.divideFcn = 'dividerand'; %Set the divide function to dividerand (divide training data randomly).
train_net.performFcn = 'mse';
train_net.trainFcn = 'trainlm'; % set training function to trainlm (Levenberg-Marquardt backpropagation) 
train_net.plotFcns = {'plotperform', 'plottrainstate', 'ploterrhist', 'plotconfusion', 'plotroc'};

%set Layer1
train_net.layers{1}.name = 'Layer 1';
train_net.layers{1}.dimensions = 7;
train_net.layers{1}.initFcn = 'initnw';
train_net.layers{1}.transferFcn = 'tansig';
%set Layer2
train_net.layers{2}.name = 'Layer 2';
train_net.layers{2}.dimensions = 1;
train_net.layers{2}.initFcn = 'initnw';
train_net.layers{2}.transferFcn = 'tansig';
train_net = train(train_net,M',training_y');
% train_net = train(train_net,Training_x',training_y');
y = train_net(M'); %prediction
view(train_net);

