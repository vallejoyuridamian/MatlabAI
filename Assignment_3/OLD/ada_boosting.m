clc
clear 
classifiers=5; % Total number of sequential classifiers
fm_=[];
rawdata = importdata('Letter2Class.data');
M = reshape(rawdata.data,1576,16);
[m,n] = size(M) ;
P = 0.7;
shuffled = randperm(m);
size(shuffled)
Xtrain = M(shuffled(1:round(P*m)),:);
size(Xtrain)
Xtest = M(shuffled(round(P*m)+1:end),:);
size(Xtest)
Training_y=rawdata.rowheaders(shuffled(1:round(P*m)),:) ;
Testing_y=rawdata.rowheaders(shuffled(round(P*m)+1:end),:);

% This following part of the code is used to convert the class {'A'} and {'X'} to class 1 and 2 respectively for train datasets
for i=1:length(Training_y)
    if strcmp(Training_y(i,1),{'A'})==1
        training_y(i,1)=1;
    else
        training_y(i,1)=2;
    end
end
% This following part of the code is used to convert the class {'A'} and {'X'} to class 1 and 2 respectively for test data sets

Ytrain=training_y;
for i=1:length(Testing_y)
    if strcmp(Testing_y(i,1),{'A'})==1
        testing_y(i,1)=1;
    else
        testing_y(i,1)=2;
    end
    
end

for j=1:classifiers
N=size(Xtrain,1);
a=[Xtrain Ytrain];

D=(1/N)*ones(N,1);
Dt=[]; h_=[];

Classifiers=1;
eps=zeros(Classifiers,1);

    p_min=min(D);
    p_max=max(D);
    
    for i=1:length(D)
        p = (p_max-p_min)*rand(1) + p_min;
        
        if D(i)>=p
            d(i,:)=a(i,:);
        end
        
        t=randi(size(d,1));
        Dt=[Dt ;d(t,:)];
    end

    X=Dt(:,1:end-1);
    Y=Dt(:,end);
        
         trained = fitctree(Xtrain,Ytrain);
         preY = predict(trained,X);
         h=preY;
         Dt=Dt(length(Dt)+1:end,:);

    h_=[h_ h];
    T=1;
    % calculation of weighted error
    for i=1:length(Y)
        if (h_(i,T)~=Y(i))
            eps(T)=eps(T)+D(i,:); 
        end  
    end
    
    alpha(T)=0.5*log((1-eps(T))/eps(T));   % Weight calculation
    
    D=D.*exp((-1).*Y.*alpha(T).*h);  % weight adjustment is done here
    D=D./sum(D);

H=predict(trained, Xtrain); % final prediction is done here
ada_train(:,1)=sign(H*alpha');
% for test set
Htest=predict(trained, Xtest);
% ada_test(:,1)=sign((Htest*alpha')./abs(Htest*alpha'));
fm_=[fm_; confusion_matrix(testing_y, Htest)];
j
end
[fm j]=max(fm_);
fm

