%% Load and prepare the data
% Load the dataset
data = load('CustomTreadmill.mat');
strides = data.strides;

% Determine total available strides
totalAvailableStrides = length(strides);   % Reduce for faster training
fprintf('Total available strides in dataset: %d\n', totalAvailableStrides);

% Parameters for train/test split
testFraction = 0.1;  % Use 10% of data for testing
numTestStrides = max(round(totalAvailableStrides * testFraction), 1);
numTrainStrides = totalAvailableStrides - numTestStrides;

fprintf('Using %d strides for training and %d strides for testing\n', numTrainStrides, numTestStrides);

% Parameters
inputLength = 40;          % Length of input sequence
outputLength = 60;         % Length of output sequence to predict

% Extract information about gon table
numFeatures = size(strides{1}.gon, 2);
featureNames = strides{1}.gon.Properties.VariableNames;
fprintf('Found %d features in gon table.\n', numFeatures);

% Verify can access speed data
try
    % Check if we can access speed from first stride
    speedTable = strides{1}.conditions.speed;
    if ~istable(speedTable)
        error('Speed data is not in table format as expected');
    end
    fprintf('Successfully accessed speed data: table size is %d x %d\n', size(speedTable, 1), size(speedTable, 2));
    
    % Display speed from first stride to confirm
    speedValue = speedTable{1, 2}; % Assuming second column contains speed value
    fprintf('Speed for first stride: %.2f\n', speedValue);
catch e
    error('Error accessing speed data: %s\nPlease verify the path to speed data in the strides structure.', e.message);
end

%% Check if we have enough data points for the requested input and output lengths
minRequiredPoints = inputLength + outputLength;
for i = 1:totalAvailableStrides
    if size(strides{i}.gon, 1) < minRequiredPoints
        error('Stride %d has only %d points, but %d are required for input+output. Reduce inputLength or outputLength.', ...
            i, size(strides{i}.gon, 1), minRequiredPoints);
    end
end

% Create train/test split indices
rng(42);  % For reproducibility
shuffledIndices = randperm(totalAvailableStrides);
trainStrideIndices = shuffledIndices(1:numTrainStrides);
testStrideIndices = shuffledIndices(numTrainStrides+1:end);

% Save test stride indices for later use
testStrideIdx = testStrideIndices(1);  % use the first test stride for visualization

%% Calculate how many samples we can create
totalSamples = 0;
for i = 1:length(trainStrideIndices)
    strideIdx = trainStrideIndices(i);
    % For each stride, we can create (numPoints - (inputLength + outputLength) + 1) samples
    samplesPerStride = size(strides{strideIdx}.gon, 1) - (inputLength + outputLength) + 1;
    
    % We need at least 1 sample per stride
    if samplesPerStride < 1
        fprintf('Warning: Stride %d cannot generate any samples with current parameters.\n', strideIdx);
        continue;
    end
    
    totalSamples = totalSamples + samplesPerStride;
end

% % Check if there's enough samples
% if totalSamples < 10  % Arbitrary minimum, adjust as needed
%     error('Not enough samples (%d) for training. Try reducing inputLength or outputLength, or use more strides.', totalSamples);
% end

fprintf('Creating %d samples from %d training strides...\n', totalSamples, length(trainStrideIndices));

%% Initialize arrays to store input-output pairs - now for all features including speed
% Format: [sequence_length, features, channels, samples]
X = zeros(inputLength, numFeatures-1, 2, totalSamples);  % Channel 1: gon features, Channel 2: speed
Y = zeros(outputLength, numFeatures-1, totalSamples);    % Format: [sequence_length, features, samples]

% Extract training sequences from strides
sampleIdx = 1;
for i = 1:length(trainStrideIndices)
    strideIdx = trainStrideIndices(i);
    
    % Skip the first column (time)
    gonData = table2array(strides{strideIdx}.gon(:, 2:end));  % Extract all features, excluding time
    
    % Get speed for this stride
    speedData = strides{strideIdx}.conditions.speed{1, 2}; % Extract speed value
    
    numPointsInStride = size(gonData, 1);
    maxStartIdx = numPointsInStride - (inputLength + outputLength) + 1;
    
    % Create sliding windows for input-output pairs
    for j = 1:maxStartIdx
        % Input sequence for all gon features (channel 1)
        X(:, :, 1, sampleIdx) = gonData(j:j+inputLength-1, :);
        
        % Add speed as a repeated value in channel 2
        % We repeat the same speed value across all time points and features
        X(:, :, 2, sampleIdx) = speedData * ones(inputLength, numFeatures-1);
        
        % Output sequence (next values after input) for all features
        Y(:, :, sampleIdx) = gonData(j+inputLength:j+inputLength+outputLength-1, :);
        
        sampleIdx = sampleIdx + 1;
    end
end

% Adjust totalSamples if we didn't collect as many as expected
totalSamples = sampleIdx - 1;
fprintf('Actually created %d samples.\n', totalSamples);

%% Split data into training and validation sets
% Use 80% for training, 20% for validation
rng(42); % For reproducibility
shuffledIndices = randperm(totalSamples);

% Calculate number of training samples, ensuring we have at least 1 validation sample
numTrainSamples = min(round(0.8 * totalSamples), totalSamples - 1);
numValSamples = totalSamples - numTrainSamples;

% Safety check
if numTrainSamples <= 0 || numValSamples <= 0
    error('Not enough samples to split into training and validation sets.');
end

trainIndices = shuffledIndices(1:numTrainSamples);
valIndices = shuffledIndices(numTrainSamples+1:end);

fprintf('Using %d samples for training and %d for validation.\n', numTrainSamples, numValSamples);

XTrain = X(:, :, :, trainIndices);
YTrain = reshape(Y(:, :, trainIndices), [], numTrainSamples)';  % Flatten output for training
XVal = X(:, :, :, valIndices);
YVal = reshape(Y(:, :, valIndices), [], numValSamples)';  % Flatten output for validation

%% Create CNN model architecture
% Define CNN architecture with multiple output features and speed input
inputSize = [inputLength numFeatures-1 2];  % [sequence_length features channels], where channels=2 for gon data + speed
numOutputValues = outputLength * (numFeatures-1);  % Total number of values to predict

layers = [
    % Input layer (now with 2 channels - gon data and speed)
    imageInputLayer(inputSize, 'Name', 'input', 'Normalization', 'none')
    
    % First convolutional block
    convolution2dLayer([5 3], 32, 'Name', 'conv1', 'Padding', 'same')
    batchNormalizationLayer('Name', 'bn1')
    reluLayer('Name', 'relu1')
    
    % Second convolutional block
    convolution2dLayer([5 3], 64, 'Name', 'conv2', 'Padding', 'same')
    batchNormalizationLayer('Name', 'bn2')
    reluLayer('Name', 'relu2')
    
    % Third convolutional block
    convolution2dLayer([5 3], 128, 'Name', 'conv3', 'Padding', 'same')
    batchNormalizationLayer('Name', 'bn3')
    reluLayer('Name', 'relu3')
    
    % Feature aggregation with additional capacity
    globalAveragePooling2dLayer('Name', 'gap')
    
    % Dense layers with improved capacity to account for improve speed
    fullyConnectedLayer(512, 'Name', 'fc1')
    reluLayer('Name', 'relu4')
    dropoutLayer(0.3, 'Name', 'dropout1')
    
    fullyConnectedLayer(256, 'Name', 'fc2')
    reluLayer('Name', 'relu5')
    dropoutLayer(0.2, 'Name', 'dropout2')
    
    % Output layer for all features
    fullyConnectedLayer(numOutputValues, 'Name', 'output')
    regressionLayer('Name', 'regression')
];

%% Training options
options = trainingOptions('adam', ...
    'MaxEpochs', 100, ...
    'MiniBatchSize', min(64, floor(numTrainSamples/10)), ...  % Increased batch size for more strides
    'InitialLearnRate', 0.001, ...
    'GradientThreshold', 1, ...
    'Plots', 'training-progress', ...
    'Verbose', true, ...
    'ValidationData', {XVal, YVal}, ...
    'ValidationFrequency', max(floor(numTrainSamples/20), 30), ...  % Adjusted validation frequency
    'ValidationPatience', 15, ...  % Increased patience
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.5, ...
    'LearnRateDropPeriod', 20, ...
    'Shuffle', 'every-epoch');

%% Train the model
fprintf('Training CNN model...\n');
try
    net = trainNetwork(XTrain, YTrain, layers, options);
    fprintf('Training complete.\n');
catch e
    fprintf('Error during training: %s\n', e.message);
    rethrow(e);
end

%% Evaluate model on validation set
YPred = predict(net, XVal);
% Calculate RMSE for each feature
YValReshaped = reshape(YVal', outputLength, numFeatures-1, []);
YPredReshaped = reshape(YPred', outputLength, numFeatures-1, []);

% Calculate RMSE for each feature
featureRMSE = zeros(numFeatures-1, 1);
for i = 1:(numFeatures-1)
    featureRMSE(i) = sqrt(mean((YPredReshaped(:, i, :) - YValReshaped(:, i, :)).^2, 'all'));
    fprintf('Validation RMSE for feature %s: %.4f\n', featureNames{i+1}, featureRMSE(i));
end

% Replace the test prediction section with this:
% Test prediction on a new stride (from the test set)
if ~isempty(testStrideIndices)
    testStrideIdx = testStrideIndices(1);  % Use the first test stride
    fprintf('Predicting future values for test stride %d...\n', testStrideIdx);
    
    try
        % Predict future values
        predictedValues = predictFutureValuesMultiFeature(net, strides{testStrideIdx}, inputLength, outputLength, numFeatures);
        
        % Get actual values for comparison
        actualValues = table2array(strides{testStrideIdx}.gon(inputLength+1:inputLength+outputLength, 2:end));
        inputValues = table2array(strides{testStrideIdx}.gon(1:inputLength, 2:end));
        
        % Create a single figure with subplots for all features
        figure('Position', [100, 100, 1200, 800]);  % Create a large figure
        
        % Calculate number of rows and columns for subplots
        numCols = min(3, numFeatures-1);  % At most 3 columns
        numRows = ceil((numFeatures-1) / numCols);
        
        % Get the stride speed for the title
        strideSpeed = strides{testStrideIdx}.conditions.speed{1, 2};
        
        % Create a suptitle for the entire figure
        sgtitle(sprintf('Gait Prediction at %.2f m/s (Test Stride %d)', strideSpeed, testStrideIdx), 'FontSize', 16);
        
        % Calculate RMSE for each feature
        testRMSE = zeros(numFeatures-1, 1);
        for i = 1:(numFeatures-1)
            testRMSE(i) = sqrt(mean((predictedValues(:, i) - actualValues(:, i)).^2));
            
            % Create subplot for this feature
            subplot(numRows, numCols, i);
            
            % Plot input data
            plot(1:inputLength, inputValues(:, i), 'b-', 'LineWidth', 2);
            hold on;
            
            % Plot actual future values
            plot(inputLength+1:inputLength+outputLength, actualValues(:, i), 'g-', 'LineWidth', 2);
            
            % Plot predicted future values
            plot(inputLength+1:inputLength+outputLength, predictedValues(:, i), 'r--', 'LineWidth', 2);
            
            % Add labels and title
            title(sprintf('%s (RMSE: %.2f)', featureNames{i+1}, testRMSE(i)));
            xlabel('Gait Cycle (%)');
            ylabel('Angle (deg)');
            grid on;
            
            % Add vertical line to mark transition from input to prediction
            line([inputLength, inputLength], ylim, 'Color', 'k', 'LineStyle', '--');
            
            % Only add legend to the first subplot to avoid cluttering
            if i == 1
                legend('Input Data', 'Actual Future', 'Predicted Future', 'Location', 'best');
            end
        end
        
        % Create a second figure showing all features together in one plot
        figure('Position', [100, 100, 1000, 600]);
        hold on;
        
        % Set up color maps for consistent colors across features
        inputColorMap = winter(numFeatures-1);
        actualColorMap = summer(numFeatures-1);
        predColorMap = autumn(numFeatures-1);
        
        % Plot each feature with distinct colors
        for i = 1:(numFeatures-1)
            % Plot input (blues)
            plot(1:inputLength, inputValues(:, i), 'LineWidth', 2, 'Color', inputColorMap(i,:));
            
            % Plot actual future (greens)
            plot(inputLength+1:inputLength+outputLength, actualValues(:, i), 'LineWidth', 2, 'Color', actualColorMap(i,:));
            
            % Plot predicted future (reds/oranges)
            plot(inputLength+1:inputLength+outputLength, predictedValues(:, i), '--', 'LineWidth', 2, 'Color', predColorMap(i,:));
        end
        
        % Add vertical line to mark transition from input to prediction
        line([inputLength, inputLength], ylim, 'Color', 'k', 'LineStyle', '--');
        
        % Create custom legend entries
        legendEntries = {};
        for i = 1:(numFeatures-1)
            legendEntries = [legendEntries, sprintf('%s (Input)', featureNames{i+1})];
        end
        for i = 1:(numFeatures-1)
            legendEntries = [legendEntries, sprintf('%s (Actual)', featureNames{i+1})];
        end
        for i = 1:(numFeatures-1)
            legendEntries = [legendEntries, sprintf('%s (Pred)', featureNames{i+1})];
        end
        
        % Add labels and legend
        legend(legendEntries, 'Location', 'eastoutside', 'FontSize', 8);
        title(sprintf('All Joint Angles Prediction (Stride %d, Speed %.2f m/s)', testStrideIdx, strideSpeed));
        xlabel('Gait Cycle (%)');
        ylabel('Joint Angle (deg)');
        grid on;
        
        % Print overall performance
        fprintf('Average RMSE across all features: %.4f\n', mean(testRMSE));
        fprintf('Feature-specific RMSE values:\n');
        for i = 1:(numFeatures-1)
            fprintf('  %s: %.4f\n', featureNames{i+1}, testRMSE(i));
        end
    catch e
        fprintf('Error during prediction: %s\n', e.message);
    end
else
    fprintf('No test strides available. Please adjust the testFraction parameter.\n');
end



function predictedValues = predictFutureValuesMultiFeature(net, strideData, inputLength, outputLength, numFeatures)
    % Extract all features from gon table (excluding time column)
    allFeatureData = table2array(strideData.gon(:, 2:end));
    
    % Get speed for this stride
    speedData = strideData.conditions.speed{1, 2}; % Extract speed value
    
    % Take the first 'inputLength' values for all features
    inputData = allFeatureData(1:inputLength, :);
    
    % Create input array with 2 channels
    % Channel 1: gon features
    % Channel 2: speed (repeated to match dimensions)
    inputWithSpeed = zeros(inputLength, numFeatures-1, 2);
    inputWithSpeed(:,:,1) = inputData;
    inputWithSpeed(:,:,2) = speedData * ones(inputLength, numFeatures-1);
    
    % Reshape for CNN input: [inputLength, numFeatures-1, 2, 1]
    inputWithSpeed = reshape(inputWithSpeed, [inputLength, numFeatures-1, 2, 1]);
    
    % Make prediction
    rawPrediction = predict(net, inputWithSpeed);
    
    % Reshape prediction to [outputLength, numFeatures-1]
    predictedValues = reshape(rawPrediction, outputLength, numFeatures-1);
end