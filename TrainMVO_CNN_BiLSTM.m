clc;
clear;
%% Load training data and validation data

TrainData = table2array(readtable("train.xlsx"));
ValData = table2array(readtable("val.xlsx"));

%% Split the training and validation sets
InPut_num = 2:1:34; 
OutPut_num = 1; 
% Shuffle the training data
randIndicesTrain = randperm(size(TrainData, 1));
TrainData = TrainData(randIndicesTrain, :);

% Shuffle the validation data
randIndicesVal = randperm(size(ValData, 1));
ValData = ValData(randIndicesVal, :);

% Split the training set
Train_InPut = TrainData(:, 2:end);      % Input features
Train_OutPut = TrainData(:, 1);         % Output labels

% Split the validation set
Val_InPut = ValData(:, 2:end); 
Val_OutPut = ValData(:, 1); 

%% Data normalization
% Normalize the input feature data to the range 0-1 for both training and validation sets
[~, Ps] = mapminmax([Train_InPut; Val_InPut]', 0, 1);
Train_InPut = mapminmax('apply', Train_InPut', Ps);
Val_InPut = mapminmax('apply', Val_InPut', Ps);

% Convert the training and validation input data to cell format
Temp_TrI = cell(size(Train_InPut, 2), 1);
Temp_VaI = cell(size(Val_InPut, 2), 1);
for i = 1:size(Train_InPut, 2)
    Temp_TrI{i} = Train_InPut(:, i);
end
Train_InPut = Temp_TrI;

for i = 1:size(Val_InPut, 2)
    Temp_VaI{i} = Val_InPut(:, i);
end
Val_InPut = Temp_VaI;

% Convert the training and validation output data to categorical labels
Train_OutPut = categorical(Train_OutPut);
Val_OutPut = categorical(Val_OutPut);

% Clear temporary variables
clear Temp_TrI Temp_VaI;

%% Set network input parameters
numFeatures = length(InPut_num);                % Number of input features
numResponses = 2;                               % Number of classification categories (adjust based on your data)
Train_number = 100;                             % Number of training epochs
dorp_rate = 0.2;                                % Dropout rate (helps prevent overfitting, 0.2 means 20% dropout)
filterSize = 4;                                 % Convolution kernel size

%% Multiverse Optimization (MVO) settings
Universes_no = 5;                           % Number of universes (individuals)
Max_iteration = 80;                         % Maximum number of iterations
lb = [0.0001, 2, 5];                        % Lower bound for search (learning rate, number of convolution kernels, number of hidden layer neurons)
ub = [0.01, 64, 60];                        % Upper bound for search (learning rate, number of convolution kernels, number of hidden layer neurons)
Optimize_num = 3;                           % Number of variables to optimize (learning rate, number of convolution kernels, number of hidden layer neurons)

%% Start MVO optimization
% Initialize variables to store the best universe's position and inflation rate (fitness)
Best_universe=zeros(1,Optimize_num);
Best_universe_Inflation_rate=inf;

% Initialize multiverse positions
Universes=rand(Universes_no,Optimize_num).*(ub-lb)+lb;  % Individual position variables
% Set the minimum and maximum probability of a wormhole existence
WEP_Max=1;          % Maximum probability
WEP_Min=0.2;        % Minimum probability

Convergence_curve=zeros(1,Max_iteration);   % Create a variable to record the iteration process

t=1;        % Initialize iteration counter
h=waitbar(0,'Iterating optimization, please wait! (Close this window after the progress is completed)');
% Begin the MVO iterations
while t<Max_iteration+1
    WEP=WEP_Min+t*((WEP_Max-WEP_Min)/Max_iteration);            % Update exploration rate
    TDR= 1-((t)^(1/6)/(Max_iteration)^(1/6));                   % Update exploration distance
    Inflation_rates=zeros(1,size(Universes,1));                 % Inflation rates for the universes

     % Update the multiverse individuals
    for i=1:size(Universes,1)
        % Limit individuals that exceed the search space boundaries
        Flag4ub = Universes(i,:)>ub;
        Flag4lb = Universes(i,:)<lb;
        Universes(i,:) = (Universes(i,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;
         % Calculate the fitness value of each individual universe
        Inflation_rates(1,i) = Optimize_Object(Universes(i,:), Train_InPut, Train_OutPut, Train_number, dorp_rate, ...
            numFeatures, numResponses, filterSize); 
        % Select the optimal value
        if Inflation_rates(1,i) < Best_universe_Inflation_rate
            Best_universe_Inflation_rate = Inflation_rates(1,i);
            Best_universe = Universes(i,:);
        end
    end

     % Sort the results
    [sorted_Inflation_rates,sorted_indexes]=sort(Inflation_rates);
    for newindex=1:Universes_no
        Sorted_universes(newindex,:) = Universes(sorted_indexes(newindex),:);
    end

    % Normalize inflation rates
    normalized_sorted_Inflation_rates = normr(sorted_Inflation_rates);
    Universes(1,:)= Sorted_universes(1,:);

    % Update multiverse positions
    for i=2:size(Universes,1)
        Back_hole_index=i;
        for j=1:size(Universes,2)
            r1=rand();
            if r1<normalized_sorted_Inflation_rates(i)
                White_hole_index=RouletteWheelSelection(-sorted_Inflation_rates); % Roulette wheel function
                if White_hole_index==-1
                    White_hole_index=1;
                end
                Universes(Back_hole_index,j)=Sorted_universes(White_hole_index,j);
            end
            r2=rand();
            if r2<WEP
                r3=rand();
                if r3<0.5
                    Universes(i,j)=Best_universe(1,j)+TDR*((ub(j)-lb(j))*rand+lb(j));
                end
                if r3>0.5
                    Universes(i,j)=Best_universe(1,j)-TDR*((ub(j)-lb(j))*rand+lb(j));
                end
            end
        end
    end

    Convergence_curve(t)=Best_universe_Inflation_rate; 
    disp(['Iteration：' num2str(t) ' || Best_Object：' num2str(Convergence_curve(t))]);
    t=t+1;
    waitbar(t/Max_iteration,h);
end

%% Extract the optimal results
BestLearningrate = Best_universe(1);
BestnumFilters = round(Best_universe(2));
BestnumHiddenUnits = round(Best_universe(3));

%% Start building and training the network
layer = [ ...
    sequenceInputLayer(numFeatures) 
    convolution1dLayer(filterSize,BestnumFilters,'Padding','causal') 
    batchNormalizationLayer 
    reluLayer 
    convolution1dLayer(filterSize/2,BestnumFilters,'Padding','causal')
    batchNormalizationLaye
    reluLayer 
    globalAveragePooling1dLayer 
    flattenLayer
    bilstmLayer(BestnumHiddenUnits) 
    dropoutLayer(dorp_rate) 
    fullyConnectedLayer(numResponses) 
    softmaxLayer
    classificationLayer]; 

options = trainingOptions('adam', ...
    'MaxEpochs',Train_number, ... 
    'GradientThreshold',1, ... 
    'ExecutionEnvironment','gpu',... 
    'InitialLearnRate',BestLearningrate, ... 
    'Shuffle', 'every-epoch', ...           
    'ValidationData', {Val_InPut, Val_OutPut}, ...  
    'ValidationFrequency', 50, ...         
    'ValidationPatience', 50, ...           
    'Verbose',0, ...
    'Plots','training-progress');
%% Start training the network
net = trainNetwork(Train_InPut,Train_OutPut,layer,options);

%% Network testing
TPred = classify(net,Train_InPut); 
YPred = classify(net,Val_InPut); 

%% Convert to displayable classification metrics
True_Train = double(Train_OutPut); 
True_Test = double(Val_OutPut); 

% Sort data order
[Train_OutPut, num_Train] = sort(Train_OutPut);
[Val_OutPut , num_Val] = sort(Val_OutPut);
TPred = TPred(num_Train);
YPred = YPred(num_Val);

Accuracy_Train = sum(TPred == Train_OutPut)/size(Train_OutPut,1) * 100; 
Accuracy_Val = sum(YPred == Val_OutPut)/size(Val_OutPut,1) * 100; 
disp(['Training set classification accuracy:', num2str(Accuracy_Train), '%']);
disp(['Validation set classification accuracy:', num2str(Accuracy_Val), '%']);

%% Plotting

% Figure 1: Training set confusion matrix
figure(1);
confusionchart(Train_OutPut, TPred);
title('Train Set Confusion Matrix');

% Figure 2: Validation set confusion matrix
figure(2);
confusionchart(Val_OutPut, YPred);
title('Test Set Confusion Matrix');



