function  [Asub,labelsub,targetnew,Anew,labelTnew]=resampling(A,labelT,target,option,r,K,selectway,classlabel,numbernot,Tclass)

if (nargin<5 || nargin>9)
    help getsubset
else

    Anew=[];
    labelTnew=[];
    dataset=[];
    datalabel=[];
    
    if (nargin<7)
        selectway=2;
    end
    if (option~=4)
        K=0;
    end

    switch lower(selectway)
        case 1 %%% multi-class
            if (numbernot==1)  % one-one
                dataset=[A(classlabel{Tclass(1),1},:);A(classlabel{Tclass(2),1},:)];
                labelTnew=[labelT(classlabel{Tclass(1)});labelT(classlabel{Tclass(2),1});];
                numberTclassone=length(classlabel(Tclass(1)));
                labelTnew(1:numberTclassone)=1;
                labelTnew(1+numberTclassone:end)=-1;
                datalabel=labelTnew;
               
            else  % one-all
                indexone=[];
                indexone= find(labelT==Tclass);
                indexall=[];
                indexall=find(labelT~=Tclass);
                dataset=[A(indexone,:);A(indexall,:);];
                labelTnew=labelT;
                labelnew(indexone)=1;
                labelnew(indexall)=-1;
                datalabel=labelnew;
            end
        case 2 %%% two class
            dataset=A;
            datalabel=labelT;
            
    end

            [Asub,labelsub,indexselect]=bagging_sub(dataset,datalabel,r);
            targetnew=target;
            Anew=dataset;
    labelTnew=datalabel;
end
%%%% random samples %%%%
function [Asub,labelsub]=rdsample(dataset,datalabel,r)

Asub=[];
labelsub=[];

numberrd=ceil(size(datalabel,1)*r);
[label_new index]=array_hang(datalabel);
indexrd=index(1:numberrd);
labelsub=datalabel(indexrd,:);
if (find(labelsub~=label_new(1:numberrd,:))~=0)
    fprintf('the error of random samples')
    Asub=[];
else
    Asub=dataset(indexrd,:);
end
%%%%% random subspace algorithm %%%%
function [Asub,labelsub,targetnew]=rsm(dataset,datalabel,target,r)

Asub=[];
labelsub=[];

[numberd,numberfeature]=size(dataset);
numberfs=ceil(numberfeature*r);
indexf=[];
indexf=randperm(numberfeature);
Asub=dataset(:,indexf(1:numberfs));
labelsub=datalabel;
targetnew=[];
targetnew=target(:,indexf(1:numberfs));
