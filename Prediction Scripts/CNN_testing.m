% ===================== DATA LOADING & PREPROCESSING ===================== %
% Load Data from STRIDES Dataset

dataFolder = 'C:\Documents\mdm3task3\dataset\scripts\STRIDES';

% Define subjects and ambulation modes

% availableSubjects = dir(fullfile(dataFolder, 'AB*'));
% subjects = {availableSubjects.name};
% disp('Available subjects:');
% disp(subjects);

%subjects = {'AB06', 'AB07', 'AB08', 'AB30'}; 
subjects = {'AB06', 'AB07', 'AB08', 'AB09', 'AB10', 'AB12', 'AB13', 'AB15', 'AB16', 'AB17', 'AB18','AB19', 'AB20', 'AB21', 'AB23', 'AB24', 'AB25', 'AB27', 'AB28', 'AB30'};

ambulation_modes = {'walk', 'stairascent', 'stairdescent', 'rampascent', 'rampdescent'};

% ===================== VARIABLE SELECTION ===================== %
selectedVariables = {'emg', 'imu', 'gon'}; % Change this to include multiple variables
variableChannels = struct('emg', 5:10, 'imu', 14:19, 'gon', 4); % Modify this to choose which channels to use

% ===================== DATA LOADING ===================== %
X = [];
y = [];

% Load all data
for s = 1:length(subjects);
    for a = 1:length(ambulation_modes)
        filePath = fullfile(dataFolder, subjects{s}, strcat(ambulation_modes{a}, '.mat'));
        if exist(filePath, 'file')
            data = load(filePath);
            for i = 1:length(data.strides)
                concatenated_data = [];
                for v = 1:length(selectedVariables)
                    varName = selectedVariables{v};
                    if isfield(data.strides{i}, varName)
                        selected_data = data.strides{i}.(varName);
                        if istable(selected_data)
                            selected_data = selected_data{:,:};
                        elseif isstruct(selected_data)
                            selected_data = struct2array(selected_data);
                        end
                        if isfield(variableChannels, varName)
                            selected_data = selected_data(:, variableChannels.(varName));
                        end
                        concatenated_data = [concatenated_data, selected_data];
                    else
                        warning('%s not found in subject %s for %s mode.', varName, subjects{s}, ambulation_modes{a});
                    end
                end
                X = cat(3, X, concatenated_data');
                y = [y; a];
            end
        end
    end
end

% Reshape and process data for CNN
X = permute(X, [1, 3, 2]);
X = permute(X, [1, 3, 4, 2]);
y = categorical(y);
numFeatures = size(X, 1);
timeStepsRequired = size(X, 2);

% ===================== TRAINING-TESTING SPLIT ===================== %
trainRatio = 0.8;  % 80% for training data
cv = cvpartition(size(X, 4), 'HoldOut', 1 - trainRatio);

% Create training data
X_train = X(:, :, :, training(cv));
y_train = y(training(cv));

% Create testing data
X_test = X(:, :, :, test(cv));
y_test = y(test(cv));

% ===================== TRUNCATE TESTING DATA ===================== %
% Define the percentage of testing data to use (e.g., 10%)
testDataPercentage = 1;  % Use 10% of the time steps for testing

% Truncate the testing data to the first x% of time steps
timeStepsTruncated = round(timeStepsRequired * testDataPercentage);  % Truncated time steps
X_test_truncated = X_test(:, 1:timeStepsTruncated, :, :);  % Keep only the first x% of time steps

% Resize test data to CNN-compatible size
X_test_resized = zeros(numFeatures, timeStepsRequired, 1, size(X_test_truncated, 4));

% Resize each sample in X_test_truncated to match the required time steps
for i = 1:size(X_test_truncated, 4)
    X_test_resized(:, :, 1, i) = imresize(X_test_truncated(:, :, 1, i), [numFeatures, timeStepsRequired]);
end

% ======================= CNN MODEL DEFINITION ======================= %
layers = [
    imageInputLayer([numFeatures timeStepsRequired 1], 'Name', 'input')  % Dynamic input size
    
    convolution2dLayer([3 3], 16, 'Padding', 'same', 'Name', 'conv1')
    batchNormalizationLayer('Name', 'batchnorm1')
    reluLayer('Name', 'relu1')
    maxPooling2dLayer([2 2], 'Stride', 2, 'Name', 'maxpool1')
    
    convolution2dLayer([3 3], 32, 'Padding', 'same', 'Name', 'conv2')
    batchNormalizationLayer('Name', 'batchnorm2')
    reluLayer('Name', 'relu2')
    maxPooling2dLayer([2 2], 'Stride', 2, 'Name', 'maxpool2')

    fullyConnectedLayer(length(ambulation_modes), 'Name', 'fc')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'classOutput')
];

% ======================== TRAINING THE MODEL ======================== %
options = trainingOptions('adam', ...
    'MaxEpochs',20, ...
    'MiniBatchSize', 32, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', true, ...
    'Plots', 'training-progress');

% Train the network
net = trainNetwork(X_train, y_train, layers, options);

% ========================= MODEL EVALUATION ========================= %
% Classify the testing data
y_pred = classify(net, X_test_resized);

% Evaluate model performance
accuracy = sum(y_pred == y_test) / numel(y_test);

fprintf('Test Accuracy (using first %.0f%% of testing data): %.2f%%\n', testDataPercentage * 100, accuracy * 100);

% Confusion Matrix
figure;
confusionchart(y_test, y_pred);
title(sprintf('Confusion Matrix - CNN Classifier (First %.0f%% of Testing Data)', testDataPercentage * 100));


% Define the directory where you want to save the model
saveDir = 'C:\Documents\mdm3task3\dataset\scripts\STRIDES\preprocessed';

% Check if the directory exists
if ~isfolder(saveDir)
    error('Directory does not exist: %s', saveDir);
end

% Define the model file name
modelFileName = 'trainedCNNModel.mat';

% Save the model to the specified directory
save(fullfile(saveDir, modelFileName), 'net');  % Save the network object
disp(['Model saved to ', fullfile(saveDir, modelFileName)]);