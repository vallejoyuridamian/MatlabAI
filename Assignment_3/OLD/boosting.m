clc
close all
clear 

Itr=50;
fm_=[];

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
size(Training_x)
Testing_x = M(idx(round(P*m)+1:end),:);
size(Testing_x)
Training_y=A.rowheaders(idx(1:round(P*m)),:) ;
Testing_y=A.rowheaders(idx(round(P*m)+1:end),:);

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

Xtrain=Training_x;
Ytrain=training_y;
Xtest=Testing_x;

for itr=1:Itr
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
        
    
         t2 = fitctree(Xtrain,Ytrain);
         preY = predict(t2,X);
         h=preY;
         Dt=Dt(length(Dt)+1:end,:);
        cm= confusion_matrix(training_y, preY)

    h_=[h_ h];
    T=1;
    % weighted error
    for i=1:length(Y)
        if (h_(i,T)~=Y(i))
            eps(T)=eps(T)+D(i,:); 
        end  
    end
    
    % Hypothesis weight
    alpha(T)=0.5*log((1-eps(T))/eps(T));
    
    % Update weights
    D=D.*exp((-1).*Y.*alpha(T).*h);
    D=D./sum(D);


% final vote
H=predict(t2, Xtrain);
ada_train(:,1)=sign(H*alpha');

% for test set
Htest=predict(t2, Xtest);
% ada_test(:,1)=sign((Htest*alpha')./abs(Htest*alpha'));
fm_=[fm_; confusion_matrix(testing_y, Htest)];
itr
end
[fm itr]=max(fm_)


X=Training_x(:,1:2);
species=Training_y;
y = categorical(species);
figure
gscatter(X(:,1),X(:,2),species);
xlabel('A box');
ylabel('B box');
x1range = min(X(:,1)):.01:max(X(:,1));
x2range = min(X(:,2)):.01:max(X(:,2));
[xx1, xx2] = meshgrid(x1range,x2range);
XGrid = [xx1(:) xx2(:)];
t2 = fitctree(X,y);
predicted_y=predict(t2, XGrid);
figure
y = categorical(predicted_y);
gscatter(xx1(:), xx2(:),predicted_y,'rg')


% ada_out=ada_test(:,itr);