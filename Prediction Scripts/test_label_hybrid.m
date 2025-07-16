dataFolder = 'C:\Users\liamj\OneDrive - University of Bristol\Documents\LEGDATA\scripts\STRIDES';

availableSubjects = dir(fullfile(dataFolder, 'AB*'));
availableSubjects = {availableSubjects.name};
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

% availableAmbulation_modes = dir(fullfile(dataFolder, 'AB*'));

ambulation_modes = {'walk', 'stairascent', 'stairdescent', 'rampascent', 'rampdescent', 'rampascent-walk', 'rampdescent-walk', 'stairascent-walk', 'stairdescent-walk' ...
    'stand-walk', 'walk-rampascent', 'walk-rampdescent', 'walk-stairascent', 'walk-stand', 'stand'};
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
                strideLabels = cell(numStrides, 1);
                
                for iStride = 1:numStrides
                    % Get the stride-specific label
                    if isfield(data.strides{iStride}, 'conditions') && isfield(data.strides{iStride}.conditions, 'labels')
                        % Extract the label from the table
                        labelTable = data.strides{iStride}.conditions.labels;
                        if height(labelTable) > 0
                            % Use the value in the second column
                            strideLabels{iStride} = labelTable{1, 2};
                        else
                            % Use a default label if the table is empty
                            strideLabels{iStride} = ambulation_modes{a};
                        end
                    else
                        % Use ambulation mode as default if no labels field
                        strideLabels{iStride} = ambulation_modes{a};
                    end
                    
                    % Build sensor stack as before
                    sensorStack = buildSensorStack(data.strides{iStride}, selectedSensors);
                    
                    % only 101 
                    if size(sensorStack, 2) == featureDim
                        combinedData{iStride} = sensorStack;
                    else
                        combinedData{iStride} = []; 
                        strideLabels{iStride} = []; % Clear label for invalid data
                    end
                end
                
                % remove nans and empty data
                validIdx = cellfun(@(x) ~isempty(x), combinedData);
                combinedData = combinedData(validIdx);
                strideLabels = strideLabels(validIdx);
                
                % add data and labels
                X = [X; combinedData];
                Y = [Y; strideLabels(:)];
            end
        end
    end
    Y = string(Y);
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

% Display unique labels found
uniqueLabels = categories(Y_all);
fprintf('Unique labels found in the dataset:\n');
disp(uniqueLabels);

if ~exist(fullfile(outputFolder, 'split_indices.mat'), 'file')
    % First time running - create and save the split indices
    numSamples = length(X_all);
    randIdx = randperm(numSamples);
    splitIdx = round(0.8 * numSamples);
    
    % Save the indices
    save(fullfile(outputFolder, 'split_indices.mat'), 'randIdx', 'splitIdx', 'numSamples');
else
    % Load existing indices
    load(fullfile(outputFolder, 'split_indices.mat'), 'randIdx', 'splitIdx', 'numSamples');
end

% train split
splitIdx = round(0.8 * numSamples);
XTrain = X_all(randIdx(1:splitIdx));
YTrain = Y_all(randIdx(1:splitIdx));
XTest  = X_all(randIdx(splitIdx+1:end));
YTest  = Y_all(randIdx(splitIdx+1:end));

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

numHiddenUnits = 200;
numClasses = numel(categories(Y_all));

% number of features
numFeatures = size(XTrain{1}, 1);

% Define the hybrid CNN-LSTM architecture.
layers = [ ...
    sequenceInputLayer(numFeatures, 'Name', 'input', 'MinLength', featureDim) 
    convolution1dLayer(3, 64, 'Padding', 'same', 'Name', 'conv1') % 1-D conv layer with filter size 3 and 64 filters
    batchNormalizationLayer('Name','bn1')
    reluLayer('Name','relu1')
    % maxPooling1dLayer(2, 'Stride', 2, 'Name','pool1')  % adjust pool size if needed so that pooled length >= pool size
    lstmLayer(numHiddenUnits, 'OutputMode', 'sequence', 'Name','lstm1')
    dropoutLayer(0.3, 'Name','drop1')
    lstmLayer(50, 'OutputMode', 'sequence', 'Name','lstm2')
    dropoutLayer(0.3, 'Name','drop2')
    fullyConnectedLayer(numClasses, 'Name','fc')
    softmaxLayer('Name','softmax')
    classificationLayer('Name','output')];

options = trainingOptions('adam', ...
    'MaxEpochs', 200, ... 
    'GradientThreshold', 2, ...
    'MiniBatchSize', 32, ...
    'Verbose', 0, ...
    'Plots', 'training-progress');

sequenceLength = size(XTrain{1}, 2);


YTrainSeq = cell(size(YTrain));
for i = 1:numel(YTrain)
    YTrainSeq{i} = repmat(YTrain(i), 1, sequenceLength);
end

YTestSeq = cell(size(YTest));
for i = 1:numel(YTest)
    YTestSeq{i} = repmat(YTest(i), 1, sequenceLength);
end

% Train the LSTM network.
net = trainNetwork(XTrain, YTrainSeq, layers, options);
save(fullfile(outputFolder, 'trained_model_testcnnlstm.mat'), 'net');

%% --- Evaluation ---
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

figure('Position', [100, 100, 1200, 500]);
subplot(1,2,1);
cm = confusionmat(timeStepLabels, timeStepPredictions);
cmChart = confusionchart(cm, categories(Y_all));
cmChart.Title = 'Confusion Matrix (Time Step Level)';
cmChart.ColumnSummary = 'column-normalized';
cmChart.RowSummary = 'row-normalized';

subplot(1,2,2);
classNames = categories(Y_all);
perClassAccuracy = zeros(length(classNames), 1);
for c = 1:length(classNames)
    className = classNames(c);
    classIndices = find(timeStepLabels == className);
    correctPredictions = sum(timeStepPredictions(classIndices) == className);
    perClassAccuracy(c) = correctPredictions / length(classIndices) * 100;
end
bar(perClassAccuracy);
xticklabels(classNames);
xtickangle(45);
ylim([0 100]);
ylabel('Accuracy (%)');
title('Per-Class Accuracy');
grid on;

% test samples
samplesToPlot = min(5, numel(XTest));
figure('Position', [100, 100, 1200, 800]);
for i = 1:samplesToPlot
    subplot(samplesToPlot, 1, i);
    sampleData = XTest{i};
    timeAxis = 1:size(sampleData, 2);
    
    hold on;
    for j = 1:size(sampleData, 1)
        plot(timeAxis, sampleData(j, :), 'Color', [0.7 0.7 0.7 0.3]);
    end
    
    yyaxis right;
    actualNumeric = double(grp2idx(YTestSeq{i}));
    predictedNumeric = double(grp2idx(YPredSeq{i}));
    
    plot(timeAxis, actualNumeric, 'b-', 'LineWidth', 2);
    plot(timeAxis, predictedNumeric, 'r--', 'LineWidth', 2);
    
    yticks(1:numel(classNames));
    yticklabels(classNames);
    
    legend('Signal Data', 'Actual Class', 'Predicted Class');
    title(['Sample #', num2str(i), ' - ', char(YTest(i))]);
    xlabel('Time Steps');
    hold off;
end


figure('Position', [100, 100, 800, 600]);
correctPredictions = zeros(numel(XTest), 1);
for i = 1:numel(XTest)
    correct = YPredSeq{i} == YTestSeq{i};
    correctPredictions(i) = sum(correct) / length(correct) * 100;
end

uniqueClasses = categories(Y_all);
classColors = lines(numel(uniqueClasses));
hold on;
for c = 1:numel(uniqueClasses)
    className = uniqueClasses(c);
    classIndices = find(YTest == className);
    scatter(classIndices, correctPredictions(classIndices), 50, classColors(c, :), 'filled');
end
hold off;
ylabel('Prediction Accuracy (%)');
xlabel('Test Sample Index');
title('Prediction Accuracy for Each Test Sample');
legend(uniqueClasses);
ylim([0 100]);
grid on;

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