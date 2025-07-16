dataFolder = 'C:\Documents\mdm3task3\dataset\scripts\STRIDES';

% availableSubjects = dir(fullfile(dataFolder, 'AB*'));
% availableSubjects = {availableSubjects.name};
% disp('Available subjects:');
% disp(availableSubjects);
availableSubjects = {'AB06', 'AB07', 'AB08', 'AB30','AB11'}; 
% Sensors to use
selectedSensors = struct();
selectedSensors.emg = [7,8,9];    
selectedSensors.ik  = [];     
selectedSensors.imu = [14,15,16,17,18,19];     
selectedSensors.gon = [];     

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

% Labels
labelMap = containers.Map();
labelMap('walk') = "Walk";
labelMap('stairascent') = "StairAscent";
labelMap('stairdescent') = "StairDescent";
labelMap('rampascent') = "RampAscent";
labelMap('rampdescent') = "RampDescent";

ambulation_modes = {'walk', 'stairascent', 'stairdescent', 'rampascent', 'rampdescent'};

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
                
                % Build sensor stack
                numStrides = numel(data.strides);
                combinedData = cell(numStrides, 1);
                for iStride = 1:numStrides
                    sensorStack = buildSensorStack(data.strides{iStride}, selectedSensors);
                    
                    % Ensure correct feature dimension
                    if size(sensorStack, 2) == featureDim
                        combinedData{iStride} = sensorStack;
                    else
                        combinedData{iStride} = []; 
                    end
                end
                
                % Remove NaNs
                validIdx = cellfun(@(x) ~isempty(x), combinedData);
                combinedData = combinedData(validIdx);
                
                % Add data and labels
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

% Remove NaNs
validIdx = cellfun(@(x) ~any(isnan(x(:))) && ~any(isinf(x(:))), X_all);
X_all = X_all(validIdx);
Y_all = Y_all(validIdx);
Y_all = categorical(Y_all);

% Split data
numSamples = length(X_all);
randIdx = randperm(numSamples);
splitIdx = round(0.8 * numSamples);

XTrain = X_all(randIdx(1:splitIdx));
YTrain = Y_all(randIdx(1:splitIdx));
XTest  = X_all(randIdx(splitIdx+1:end));
YTest  = Y_all(randIdx(splitIdx+1:end));

% Simple 1D CNN Model
numChannels = size(XTrain{1}, 1);  % Features per time step
numClasses = numel(categories(Y_all));
filterSize = 3;
numFilters = 64;

layers = [ ...
    sequenceInputLayer(numChannels, 'Name', 'input')
    convolution1dLayer(filterSize, numFilters, 'Padding', 'causal', 'Name', 'conv1')
    reluLayer('Name', 'relu1')
    layerNormalizationLayer('Name', 'ln1')
    convolution1dLayer(filterSize, 2*numFilters, 'Padding', 'causal', 'Name', 'conv2')
    reluLayer('Name', 'relu2')
    layerNormalizationLayer('Name', 'ln2')
    globalAveragePooling1dLayer('Name', 'gap')
    fullyConnectedLayer(numClasses, 'Name', 'fc')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')];


% Training Options
options = trainingOptions('adam', ...
    'MaxEpochs', 10, ... 
    'GradientThreshold', 2, ...
    'MiniBatchSize', 32, ...
    'Verbose', 0, ...
    'Plots', 'training-progress');

% Train the CNN
net = trainNetwork(XTrain, YTrain, layers, options);

saveDir = 'C:\Documents\mdm3task3\dataset\scripts\STRIDES\preprocessed';

% Check if the directory exists
if ~isfolder(saveDir)
    error('Directory does not exist: %s', saveDir);
end

% Define the model file name
modelFileName = 'trained_1D_CNNModel.mat';

% Save the model to the specified directory
save(fullfile(saveDir, modelFileName), 'net');  % Save the network object
disp(['Model saved to ', fullfile(saveDir, modelFileName)]);


% Test the model
YPred = classify(net, XTest);
accuracy = sum(YPred == YTest) / numel(YTest);

fprintf('Test Accuracy: %.2f%%\n', accuracy * 100);

% Confusion Matrix
figure;
cm = confusionchart(YTest, YPred);
cm.Title = 'Confusion Matrix';
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';

% Helper function for sensor stacking
function sensorStack = buildSensorStack(stride, selectedSensors)
    sensorStack = [];
    sensorFields = fieldnames(selectedSensors);
    
    for f = 1:numel(sensorFields)
        fieldName = sensorFields{f};
        cols = selectedSensors.(fieldName);
        if ~isempty(cols) && isfield(stride, fieldName)
            extracted = table2array(stride.(fieldName)(:, cols))';
        else
            extracted = [];
        end
        sensorStack = [sensorStack; extracted];
    end
end

