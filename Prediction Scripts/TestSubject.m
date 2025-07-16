
clear; close all; clc;

dataFolder = 'C:\Users\liamj\OneDrive - University of Bristol\Documents\LEGDATA\scripts\STRIDES';
controlSubject = 'ControlSubject_AB14';

ambulation_modes = {'walk','stairascent','stairdescent','rampascent','rampdescent'};

selectedSensors = struct();
selectedSensors.emg = [5,6,7,8,9,10];    
selectedSensors.ik  = [];     
selectedSensors.imu = [14,15,16,17,18,19];     
selectedSensors.gon = [4];     

featureDim = 101; % Full sequence length
chunkSizes = [5,10,15,20,25,30,40,50]; % Test different chunk lengths


useAllStrides = true;
numStridesToTest = 80; % Number of strides to sample per ambulation mode (if not using all)

outputFolder = fullfile(dataFolder, 'preprocessed');
modelFile = fullfile(outputFolder, 'CNNLSTM200.mat');
if exist(modelFile, 'file')
    load(modelFile, 'net');
else
    error('Trained model file not found in %s', outputFolder);
end

%% Initialize accuracy tracking
accuracyResults = zeros(length(ambulation_modes), length(chunkSizes));

%% Loop through each ambulation mode
for a = 1:length(ambulation_modes)
    mode = ambulation_modes{a};
    filePath = fullfile(dataFolder, controlSubject, strcat(mode, '.mat'));
    
    if exist(filePath, 'file')
        fprintf('Processing file: %s\n', filePath);
        data = load(filePath);
        strides = data.strides;
        numStrides = numel(strides);
        
        % Determine strides to use based on the flag
        if useAllStrides
            sampledStrides = strides;
            numStridesUsed = numStrides;
        else
            % Randomly sample `numStridesToTest` strides (without replacement)
            numStridesToTestActual = min(numStridesToTest, numStrides); % Ensure we don’t exceed available strides
            sampledIndices = randperm(numStrides, numStridesToTestActual);
            sampledStrides = strides(sampledIndices);
            numStridesUsed = numStridesToTestActual;
        end

        % Loop through chunk sizes
        for c = 1:length(chunkSizes)
            chunkSize = chunkSizes(c);
            correctPredictions = 0;
            totalPredictions = 0;

            for iStride = 1:numStridesUsed
                sensorStack = buildSensorStack(sampledStrides{iStride}, selectedSensors);
                fullPrediction = classify(net, {padSequence(sensorStack, featureDim)}); % Full stride prediction
                
                % Extract the first `chunkSize` time steps from this stride
                chunk = sensorStack(:, 1:min(chunkSize, size(sensorStack, 2)));
                paddedChunk = padSequence(chunk, featureDim);
                chunkPrediction = classify(net, {paddedChunk});

                % Compare chunk prediction with full-stride prediction
                if isequal(chunkPrediction, fullPrediction)
                    correctPredictions = correctPredictions + 1;
                end
                totalPredictions = totalPredictions + 1;
            end

            % Average accuracy over sampled strides
            accuracyResults(a, c) = correctPredictions / totalPredictions; 
            fprintf('Mode: %s | Chunk: %d | Accuracy: %.2f%%\n', mode, chunkSize, accuracyResults(a, c) * 100);
        end
    else
        fprintf('File %s not found for mode %s.\n', filePath, mode);
    end
end

%% Plot results
figure;
hold on;
for a = 1:length(ambulation_modes)
    plot(chunkSizes, accuracyResults(a, :) * 100, '-o', 'LineWidth', 2, 'DisplayName', ambulation_modes{a});
end
hold off;
xlabel('Sequence Length');
ylabel('Classification Accuracy (%)');
legend show;
title('Classification Accuracy vs. Sequence Length');
grid on;

%% Helper Functions

function paddedSensorStack = padSequence(sensorStack, featureDim)
    % Pads or trims sensorStack to be exactly featureDim in length
    currentLength = size(sensorStack, 2);
    if currentLength < featureDim
        paddedSensorStack = [sensorStack, zeros(size(sensorStack,1), featureDim - currentLength)];
    else
        paddedSensorStack = sensorStack(:, 1:featureDim);
    end
end

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
