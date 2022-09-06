function [accuracy,cm,sensitivity,specificity] = confusion_matrix(y_actual,y_predicted )
tp=0;
tn=0;
fp=0;
fn=0;
% These for loops are used to calculate the True positive, true negative, false positive and false negative values 
for i=1:length(y_actual)
if y_actual(i)==1
    if y_actual(i)==y_predicted(i)
        tp=tp+1;
    else
        fn=fn+1;
    end
end
if y_actual(i)==2
    if y_actual(i)==y_predicted(i)
        tn=tn+1;
    else
        fp=fp+1;
    end
end
end
cm=[tp fn; fp tn]
accuracy=((tp+tn)/(tp+fn+fp+tn)) * 100
sensitivity=(tp/(tp+fn)) *100
specificity=(tn/(tn+fp)) *100
end

