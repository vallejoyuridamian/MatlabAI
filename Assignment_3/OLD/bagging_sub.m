function [Xsub,Ysub,selectindex]=bagging_sub(Xini,Yini,Ratio)

if (nargin<2 || nargin>3)
    help bagging
else

    Xsub = [];
    Ysub = [];

    [N,M] = size(Xini);

    A = [];
    A = randperm(N); 

    numberselect = ceil(N*Ratio);
    Aselect = [];
    Aselect = (A(1:numberselect))';

    Xpartini = [];
    Xpartini = Xini(Aselect,:);
    Ypartini = [];
    Ypartini = Yini(Aselect,:);

    numberleaver = N-numberselect;

    index1 = [];
    for i = 1:numberselect
        index1(i,1) = i/numberselect;
    end

    index2 = [];
    index2 = rand(numberleaver,1);
    index4 = [];
    for i = 1:numberleaver
        index3 = index2(i);
        for j=1:numberselect
            if (index1(j)>= index3 && j==1)
                index4(i,1)=j;
            else
                if (index1(j) >= index3 && index1(j-1)<index3)
                    index4(i,1)=j;
                end
            end
        end
    end
    Xsub = [Xpartini;Xpartini(index4,:);];
    Ysub = [Ypartini;Ypartini(index4,:);];
    selectindex = [];
    selectindex = [Aselect;Aselect(index4);];
end
