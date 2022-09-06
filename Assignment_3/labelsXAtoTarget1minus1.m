%% Auxiliary functions %%
function targetValues = labelsXAtoTarget1minus1(dataSet)
[NSamples,NFields] = size(dataSet.data);
%% we will use 0 as threshold and convert the classes to -1 and 1 %%
for kSample=1:NSamples
    if string(dataSet.rowheaders(kSample))=='X'
        targetValues(kSample) = 1;
    end
    if string(dataSet.rowheaders(kSample))=='A'
        targetValues(kSample) = -1;
    end
end
end