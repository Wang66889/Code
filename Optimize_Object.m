function [erro] = Optimize_Object(pop, Train_InPut, Train_OutPut, Train_number, dorp_rate, numFeatures, numResponses, filterSize)
layer = [ ...
    sequenceInputLayer(numFeatures) 
    convolution1dLayer(filterSize,round(pop(2)),'Padding','causal') 
    batchNormalizationLayer 
    reluLayer 
    convolution1dLayer(filterSize/2,round(pop(2)),'Padding','causal') 
    batchNormalizationLayer 
    reluLayer 
    globalAveragePooling1dLayer 
    flattenLayer
    bilstmLayer(round(pop(3))) 
    dropoutLayer(dorp_rate) 
    fullyConnectedLayer(numResponses) 
    softmaxLayer
    classificationLayer]; 

options = trainingOptions('adam', ...   
    'MaxEpochs',Train_number, ...       
    'GradientThreshold',1,...           
    'ExecutionEnvironment','auto',...   
    'InitialLearnRate', pop(1),...      
    'LearnRateSchedule','none',...      
    'Verbose',true);                    
net = trainNetwork(Train_InPut,Train_OutPut, layer, options);
TPred = classify(net,Train_InPut); 
Train_OutPut = double(Train_OutPut);
TPred = double(TPred);
[Train_OutPut, num_Train] = sort(Train_OutPut);
TPred = TPred(num_Train);
Accuracy_Train = sum(TPred == Train_OutPut)/size(Train_OutPut,1); 
erro = 1-Accuracy_Train;
end

