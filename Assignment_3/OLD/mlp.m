% clc
close all
clear 
A = importdata('Letter2Class.data');
% disp("first five instances of data are:")
% disp(A.data(1:5,:));
% disp(A.rowheaders(1:5,:));
M = reshape(A.data,1576,16);
% M=A.data(1:50,:);
[m,n] = size(M) ;
P = 0.65 ;
idx = randperm(m);
size(idx)
Training_x = M(idx(1:round(P*m)),:);
size(Training_x )
Testing_x = M(idx(round(P*m)+1:end),:);
size(Testing_x)
Training_y=A.rowheaders(idx(1:round(P*m)),:) ;
Testing_y=A.rowheaders(idx(round(P*m)+1:end),:);

for i=1:length(Training_y)
    if strcmp(Training_y(i,1),{'A'})==1
        training_y(i,1)=1;
    else
        training_y(i,1)=0;
    end
    
end

for i=1:length(Testing_y)
    if strcmp(Testing_y(i,1),{'A'})==1
        testing_y(i,1)=1;
    else
        testing_y(i,1)=0;
    end
    
end

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

x=Training_x';
t=training_y';

% [x,t] = iris_dataset; %load of the iris data set
train_net = train(train_net,x, t); %training

y = train_net(Testing_x'); %prediction

view(train_net);


X=Training_x(:,1:2);
% species=training_y;
y = training_y;
figure
gscatter(X(:,1),X(:,2),y);
xlabel('x-box');
ylabel('y-box');
x1range = min(X(:,1)):.01:max(X(:,1));
x2range = min(X(:,2)):.01:max(X(:,2));
[xx1, xx2] = meshgrid(x1range,x2range);
XGrid = [xx1(:) xx2(:)];
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
train_net = train(train_net,X', y')
% t2 = fitctree(X,y);
predicted_y=train_net(XGrid');
for i=1:length(predicted_y)
if predicted_y(1,i)>=0.5
    predicted_y(1,i)=1;
else
    predicted_y(1,i)=0;
end
end
figure
y = categorical(predicted_y);
gscatter(xx1(:), xx2(:),predicted_y,'rg')
legend('A','X')