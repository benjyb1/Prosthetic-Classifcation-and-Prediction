    
dataFolder = 'C:\Users\liamj\OneDrive - University of Bristol\Documents\LEGDATA\scripts\STRIDES';

availableSubjects = dir(fullfile(dataFolder, 'AB*'));
availableSubjects = {availableSubjects.name};
% availableSubjects = {'AB06', 'AB07', 'AB08', 'AB09', 'AB10', 'AB12', 'AB13', 'AB15', 'AB16', 'AB17', 'AB18','AB19', 'AB20', 'AB21', 'AB23', 'AB24', 'AB25', 'AB27', 'AB28', 'AB30'};
disp('Available subjects:');
disp(availableSubjects);

% sensors to use
% leave empty if not needed for training.
selectedSensors = struct();
selectedSensors.emg = [5,6,7,8,9,10];    
selectedSensors.ik  = [];     
selectedSensors.imu = [14,15,16,17,18,19];     
selectedSensors.gon = [4];   

batchSize = 4;
numSubjects = length(availableSubjects);
numBatches = ceil(numSubjects / batchSize);

fprintf('Total subjects: %d\n', numSubjects);
fprintf('Batch size: %d\n', batchSize);
fprintf('Number of batches: %d\n', numBatches);

subjectBatches = cell(1, numBatches);
for b = 1:numBatches
    startIdx = (b-1) * batchSize + 1;
    endIdx = min(b * batchSize, numSubjects);
    subjectBatches{b} = availableSubjects(startIdx:endIdx);
    
    fprintf('Batch %d contains subjects: ', b);
    disp(subjectBatches{b});
end

% labels
labelMap = containers.Map();
labelMap('walk') = "Walk";

labelMap('stairascent') = "StairAscent";
labelMap('stairdescent') = "StairDescent";
labelMap('rampascent') = "RampAscent";
labelMap('rampdescent') = "RampDescent";

ambulation_modes = {'walk', 'stairascent', 'stairdescent', 'rampascent', 'rampdescent'};

% ambulation_modes = {'walk'};

outputFolder = fullfile(dataFolder, 'preprocessed');
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end


featureDim = 101;
allData = cell(numBatches, 1);

for batchIdx = 1:numBatches
    fprintf('Processing batch %d of %d...\n', batchIdx, numBatches);
    subjects = subjectBatches{batchIdx};
    
    X = {}; 
    Y = []; 
    
    for s = 1:length(subjects)
        for a = 1:length(ambulation_modes)
            filePath = fullfile(dataFolder, subjects{s}, strcat(ambulation_modes{a}, '.mat'));
            if exist(filePath, 'file')
                data = load(filePath);
                
                % build stack
                numStrides = numel(data.strides);
                combinedData = cell(numStrides, 1);
                for iStride = 1:numStrides
                    sensorStack = buildSensorStack(data.strides{iStride}, selectedSensors);
                    
                    % only 101 
                    if size(sensorStack, 2) == featureDim
                        combinedData{iStride} = sensorStack;
                    else
                        combinedData{iStride} = []; 
                    end
                end
                
                % remove nans 
                validIdx = cellfun(@(x) ~isempty(x), combinedData);
                combinedData = combinedData(validIdx);
                
                % add data and labels
                X = [X; combinedData];
                label = labelMap(ambulation_modes{a});
                numSamples = length(combinedData);
                Y = [Y; repmat(label, numSamples, 1)];
            end
        end
    end
    
    Y = categorical(Y);
    batchData = struct();
    batchData.X = X;
    batchData.Y = Y;
    batchData.subjects = subjects;
    
    allData{batchIdx} = batchData;
    
    savePath = fullfile(outputFolder, sprintf('batch_%d_data.mat', batchIdx));
    save(savePath, 'batchData');
end

X_all = {};
Y_all = [];
for batchIdx = 1:numBatches
    X_all = [X_all; allData{batchIdx}.X];
    Y_all = [Y_all; allData{batchIdx}.Y];
end

% nan values begone
validIdx = cellfun(@(x) ~any(isnan(x(:))) && ~any(isinf(x(:))), X_all);
X_all = X_all(validIdx);
Y_all = Y_all(validIdx);
Y_all = categorical(Y_all);

if ~exist(fullfile(outputFolder, 'split_indices.mat'), 'file')
    
    numSamples = length(X_all);
    randIdx = randperm(numSamples);
    splitIdx = round(0.8 * numSamples);
    
   
    save(fullfile(outputFolder, 'split_indices.mat'), 'randIdx', 'splitIdx', 'numSamples');
else
    
    load(fullfile(outputFolder, 'split_indices.mat'), 'randIdx', 'splitIdx', 'numSamples');
end

% train split
splitIdx = round(0.8 * numSamples);
XTrain = X_all(randIdx(1:splitIdx));
YTrain = Y_all(randIdx(1:splitIdx));
XTest  = X_all(randIdx(splitIdx+1:end));
YTest  = Y_all(randIdx(splitIdx+1:end));

commonCategories = categories(Y_all);
YTrain = categorical(YTrain, commonCategories);
YTest  = categorical(YTest, commonCategories);

% second nan check
hasInvalidData = false;
for i = 1:length(X_all)
    if any(isnan(X_all{i}(:))) || any(isinf(X_all{i}(:)))
        hasInvalidData = true;
        fprintf('Invalid values found in sample %d\n', i);
        break;
    end
end
if hasInvalidData
    fprintf('WARNING: NaN or Inf values found in the data. Fix these before training.\n');
    return;
end

sequenceLength = size(XTrain{1}, 2);
YTrainSeq = cell(size(YTrain));
for i = 1:numel(YTrain)
   
    YTrainSeq{i} = categorical(repmat(string(YTrain(i)), 1, sequenceLength), commonCategories);
end

YTestSeq = cell(size(YTest));
for i = 1:numel(YTest)
    YTestSeq{i} = categorical(repmat(string(YTest(i)), 1, sequenceLength), commonCategories);
end

numHiddenUnits = 200;
numClasses = numel(categories(Y_all));

% number of da features
numFeatures = size(XTrain{1}, 1);

% cnn lstm
layers = [ ...
    sequenceInputLayer(numFeatures, 'Name', 'input', 'MinLength', featureDim) 
    
    convolution1dLayer(3, 16, 'Padding', 'same', 'Name', 'conv1')
    batchNormalizationLayer('Name', 'batchnorm1')
    reluLayer('Name', 'relu1')
    % maxPooling1dLayer(2, 'Stride', 2, 'Name', 'maxpool1')
    
    convolution1dLayer(3, 32, 'Padding', 'same', 'Name', 'conv2')
    batchNormalizationLayer('Name', 'batchnorm2')
    reluLayer('Name', 'relu2')
    % maxPooling1dLayer(2, 'Stride', 2, 'Name', 'maxpool2')
    
    lstmLayer(numHiddenUnits, 'OutputMode', 'sequence', 'Name', 'lstm1')
    dropoutLayer(0.3, 'Name', 'drop1')
    
    lstmLayer(50, 'OutputMode', 'sequence', 'Name', 'lstm2')
    dropoutLayer(0.3, 'Name', 'drop2')
    
    fullyConnectedLayer(numClasses, 'Name', 'fc')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')
];

options = trainingOptions('adam', ...
    'MaxEpochs', 200, ... 
    'GradientThreshold', 2, ...
    'MiniBatchSize', 128, ...
    'Verbose', 0, ...
    'Plots', 'training-progress', ...
    'ValidationData', {XTest, YTestSeq}, ... 
    'ValidationFrequency', 10);



net = trainNetwork(XTrain, YTrainSeq, layers, options);
save(fullfile(outputFolder, 'CNNLSTM200.mat'), 'net');

% evaluation
YPredSeq = classify(net, XTest);
correctTimeSteps = 0;
totalTimeSteps = 0;
correctSequences = 0;
for i = 1:numel(XTest)
    numTimeSteps = size(XTest{i}, 2);
    totalTimeSteps = totalTimeSteps + numTimeSteps;
    
    correct = YPredSeq{i} == YTestSeq{i};
    correctTimeSteps = correctTimeSteps + sum(correct);
    
    if all(correct)
        correctSequences = correctSequences + 1;
    end
end

timeStepAccuracy = correctTimeSteps / totalTimeSteps;
sequenceAccuracy = correctSequences / numel(XTest);

fprintf('Time Step Accuracy: %.2f%%\n', timeStepAccuracy * 100);
fprintf('Complete Sequence Accuracy: %.2f%%\n', sequenceAccuracy * 100);

%% plots
% Confusion Matrix and Per-Class Accuracy
timeStepLabels = [];
timeStepPredictions = [];
for i = 1:numel(XTest)
    timeStepLabels = [timeStepLabels, YTestSeq{i}];
    timeStepPredictions = [timeStepPredictions, YPredSeq{i}];
end

figure;
cm = confusionmat(timeStepLabels, timeStepPredictions);
cmChart = confusionchart(cm, categories(Y_all));
cmChart.Title = 'Confusion Matrix (Time Step Level)';
cmChart.ColumnSummary = 'column-normalized';
cmChart.RowSummary = 'row-normalized';

results = struct();
results.timeStepAccuracy = timeStepAccuracy;
results.sequenceAccuracy = sequenceAccuracy;
results.perClassAccuracy = perClassAccuracy;
results.classNames = classNames;
save(fullfile(outputFolder, 'evaluation_results.mat'), 'results');


function sensorStack = buildSensorStack(stride, selectedSensors)
    % buildSensorStack:
    %   Extracts and concatenates sensor data from the given stride structure
    %   based on the selected columns in the selectedSensors struct.
    %
    %   Input:
    %       stride          - A struct containing sensor data tables (e.g., emg, ik, etc.)
    %       selectedSensors - A struct with fields for each sensor type.
    %                         Each field is an array of column indices to use.
    %   Output:
    %       sensorStack - A matrix with sensor features (rows) and time steps (columns).
    
    sensorStack = [];
    sensorFields = fieldnames(selectedSensors);
    
    for f = 1:numel(sensorFields)
        fieldName = sensorFields{f};
        cols = selectedSensors.(fieldName);
        if ~isempty(cols) && isfield(stride, fieldName)
            % Extract the selected columns from the table and transpose so that
            % rows = features and columns = time steps.
            extracted = table2array(stride.(fieldName)(:, cols))';
        else
            extracted = [];
        end
        sensorStack = [sensorStack; extracted];
    end
end


