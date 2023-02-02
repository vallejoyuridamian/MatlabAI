function targetValues = labelsXAtoTarget1minus1(dataSet)
[NSamples,NFields] = size(dataSet.data);
for kSample=1:NSamples
    if string(dataSet.rowheaders(kSample))=='X'
        targetValues(kSample) = 1;
    end
    if string(dataSet.rowheaders(kSample))=='A'
        targetValues(kSample) = -1;
    end
end
end