function [weights counters] = multilayerperceptron(dataSet,NNeuronsPerLayer, learningRate,ratioRandomSamples)


% clc
% clear
% close all
% % Load the provided data %
%dataSet = importdata('Letter2Class.data');

%NNeuronsPerLayer = 3;
%learningRate = 0.01;
%ratioRandomSamples = 0.01;

[NSamples,NFields] = size(dataSet.data);
NRandomSamples = round(NSamples*ratioRandomSamples);

NITersMax = 1e3;
tolerance = 1e-6;
% So we will choose the number of hidden layers and the number of neurons
% per layer

% let's say we want a single neuron, so all the inputs go to it, is has all
% the weights and the bias

% so we have the samples
labels = labelsXAtoTarget1minus1 (dataSet);


% let's calculate the output of a network
% we first have the input layer, that has number of inputs equal to
% number of variables
% first layer
% intermediate and final layers
internalOutputs = zeros(NNeuronsPerLayer,NSamples);
%intermediate layers
outputFinalLayer = zeros(1,NSamples);
weightsFirstLayer = rand(NNeuronsPerLayer,NFields + 1) - 0.5;
weightsFinalLayer = rand(1, NNeuronsPerLayer + 1) - 0.5; % inputs
NLayers = 2;
% input layer
% the first layer will have number of inputs equal to number of variables
% the number of weights will be equal to that + 1 to account for the bias

deltaInternal = zeros(NNeuronsPerLayer,NSamples);
derivatives = zeros(NLayers,NNeuronsPerLayer);
prediction=zeros(1,NSamples);

training = true;
NIters = 0;

while training && NIters < NITersMax
    %% calculate forward for the hidden nodes %%
    for kNeuron=1:NNeuronsPerLayer
        for kSample=1:NSamples
            internalOutputs(kNeuron,kSample) = weightsFirstLayer(kNeuron,:) * [1 dataSet.data(kSample,:)]';% we inlcude the bias too
            internalOutputs(kNeuron,kSample) = actFun(internalOutputs(kNeuron,kSample)); % we pass it through the activation function
        end
    end
    
    % output layer will have only one neuron
    for kSample=1:NSamples
        outputFinalLayer(kSample) = weightsFinalLayer * [1 internalOutputs(:,kSample)']';% we inlcude the bias too
    end
    % we calculate the error then using backpropagation and update the weights
    % in this case, the activation function is the identity
    delta = zeros(1,NSamples);
    for kSample=1:NSamples
        % outputFinalLayer(kSample) = actFun(outputFinalLayer(kSample));
        if outputFinalLayer(kSample) >= 0
            outputFinalLayer(kSample) = 1;
        else
            outputFinalLayer(kSample) = -1;
        end
        % label = (labels(kSample) + 1) / 2; %(make it go between 0 and 1)
        % outputFinalLayer(kSample)= prediction(kSample);
        delta(kSample) = outputFinalLayer(kSample) - labels(kSample);
    end
    
    weightsOld = weightsFinalLayer;
    
    % backpropagate the deltas to find the deltaInternal
    % so we sweep the nodes, this is done for each sample, for each layer and
    % for each neuron
    
    % let's do it for some random samples %
    for kRandomSample=1:NRandomSamples
        kSample = randi([1 NSamples]);
        %output layer
        deltaOutputlayer = actFunPrime(outputFinalLayer(kSample)) * delta(kSample);
        %update the weights for the output layer
        NInputs = NNeuronsPerLayer;
        %bias
        weightsFinalLayer(1) = weightsFinalLayer(1) - learningRate * deltaOutputlayer;
        %weights
        for kInput=2:NInputs + 1% output layer's inputs (+1 for bias)
            weightsFinalLayer(kInput) = weightsFinalLayer(kInput) - learningRate * deltaOutputlayer * outputFinalLayer(kSample);
        end
        
        %input layer
        NInputs = NFields;
        for kNeuron=1:NNeuronsPerLayer
            deltaInternal(kNeuron,kSample) = actFunPrime(internalOutputs(kNeuron,kSample)) * weightsFinalLayer(1 + kNeuron) * deltaOutputlayer;
            %bias
            weightsFirstLayer(kNeuron, 1) = weightsFirstLayer(kNeuron, 1) - learningRate * deltaInternal(kNeuron,kSample);
            %weights
            for kInput=2:NInputs + 1 % first layer's inputs (+1 for bias)
                weightsFirstLayer(kNeuron, kInput) =  weightsFirstLayer(kNeuron, kInput) - learningRate * deltaInternal(kNeuron,kSample) * internalOutputs(kNeuron,kSample);
            end
        end
    end
    
    NIters = NIters + 1;
    
    weightsFirstLayer;
    weightsFinalLayer;
    sumAbsDelta = sum(abs(delta));
    weightsNew = weightsFinalLayer;
    
    changeWeight = norm(weightsNew - weightsOld);
    if changeWeight < tolerance
        training = false;
    end
    
    % Evaluate post update, remove %
    for kNeuron=1:NNeuronsPerLayer
        for kSample=1:NSamples
            internalOutputs(kNeuron,kSample) = weightsFirstLayer(kNeuron,:) * [1 dataSet.data(kSample,:)]';% we inlcude the bias too
            internalOutputs(kNeuron,kSample) = actFun(internalOutputs(kNeuron,kSample)); % we pass it through the activation function
        end
    end
    
    % output layer will have only one neuron
    for kSample=1:NSamples
        outputFinalLayer(kSample) = weightsFinalLayer * [1 internalOutputs(:,kSample)']';% we inlcude the bias too
    end
    
    for kSample=1:NSamples
        % outputFinalLayer(kSample) = actFun(outputFinalLayer(kSample));
        if outputFinalLayer(kSample) >= 0
            outputFinalLayer(kSample) = 1;
        else
            outputFinalLayer(kSample) = -1;
        end
    end
    
    %% let's evaluate the adaBoost with the original data%%
    counters.TP = 0; % 1 as  1
    counters.TN = 0; %-1 as -1
    counters.FP = 0; % 1 as -1
    counters.FN = 0; %-1 as  1
    for kSample=1:NSamples
        if outputFinalLayer(kSample) ~= labels(kSample)
            if labels(kSample) == 1 % 1 as  -1
                counters.FP = counters.FP + 1;
            else % -1 as 1
                counters.FN = counters.FN + 1;
            end
        else
            if labels(kSample) == 1 % 1 as  1
                counters.TP = counters.TP + 1;
            else % -1 as -1
                counters.TN = counters.TN + 1;
            end
        end
    end
    
    weights = weightsFinalLayer;
    %TP = counters.TP;
    %TN = counters.TN;

end


function res = actFun(a) % Activation function (f in Bishop's book)
res = 1 /(1 + exp(-a));
end

function res = actFunPrime(a) % Activation function (f in Bishop's book)
res = actFun(a) * (1 - actFun(a));
end

end