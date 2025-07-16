classdef LocomotionClassifier < handle
    % LocomotionClassifier - A MATLAB class for classifying locomotion types
    % based on sensor data using convolutional neural networks.
    
    properties
        windowSize = 200;  % Size of the time window for classification
        stride = 100;      % Stride between windows
        model;             % The trained CNN model
        classes;           % Class labels
        scaler;            % Data standardization parameters
        labelEncoder;      % Encoder for class labels
    end
    
    methods
        function obj = LocomotionClassifier(windowSize, stride)
            % Constructor for the LocomotionClassifier class
            %
            % Args:
            %   windowSize (int): Size of the time window for classification
            %   stride (int): Stride between windows
            
            if nargin > 0
                obj.windowSize = windowSize;
            end
            
            if nargin > 1
                obj.stride = stride;
            end
        end
        
        function [X_combined, y_combined] = loadMultipleFiles(obj, folderPath)
            % Load multiple files from a folder, each representing a different locomotion type
            %
            % Args:
            %   folderPath (str): Path to folder containing files
            %
            % Returns:
            %   [X_combined, y_combined]: Combined features and labels
            
            % List all MAT and CSV files
            files = dir(fullfile(folderPath, '*.mat'));
            csvFiles = dir(fullfile(folderPath, '*.csv'));
            files = [files; csvFiles];
            
            if isempty(files)
                fprintf('No MAT or CSV files found in the directory.\n');
                X_combined = [];
                y_combined = [];
                return;
            end
            
            % Display available files
            fprintf('Available files:\n');
            for i = 1:length(files)
                fprintf('%d. %s\n', i, files(i).name);
            end
            
            % Let user select files and assign locomotion types
            selectedFiles = {};
            
            fprintf('\nSelect files to include in the training set.\n');
            fprintf('For each file, you''ll need to specify the locomotion type (e.g., walking, running).\n');
            fprintf('Enter ''done'' when finished selecting files.\n');
            
            while true
                fileNum = input('\nEnter file number to add (or ''done'' to finish): ', 's');
                
                if strcmpi(fileNum, 'done')
                    break;
                end
                
                try
                    fileIdx = str2double(fileNum);
                    if ~isnan(fileIdx) && fileIdx >= 1 && fileIdx <= length(files)
                        fileName = files(fileIdx).name;
                        
                        % Check if file already selected
                        if any(cellfun(@(x) strcmp(x{1}, fileName), selectedFiles))
                            fprintf('File ''%s'' already selected. Choose another file.\n', fileName);
                            continue;
                        end
                        
                        % Get locomotion type
                        locoType = input(sprintf('Enter locomotion type for %s: ', fileName), 's');
                        if isempty(locoType)
                            fprintf('Locomotion type cannot be empty. Try again.\n');
                            continue;
                        end
                        
                        selectedFiles{end+1} = {fileName, locoType}; %#ok<AGROW>
                        fprintf('Added %s as ''%s''\n', fileName, locoType);
                    else
                        fprintf('Invalid file number. Please try again.\n');
                    end
                catch
                    fprintf('Please enter a valid number or ''done''.\n');
                end
            end
            
            if isempty(selectedFiles)
                fprintf('No files selected for training.\n');
                X_combined = [];
                y_combined = [];
                return;
            end
            
            fprintf('\nSelected %d files for training:\n', length(selectedFiles));
            for i = 1:length(selectedFiles)
                fprintf('- %s: %s\n', selectedFiles{i}{1}, selectedFiles{i}{2});
            end
            
            % Process all selected files
            allFeatures = {};
            allLabels = {};
            
            for i = 1:length(selectedFiles)
                fileName = selectedFiles{i}{1};
                locoType = selectedFiles{i}{2};
                filePath = fullfile(folderPath, fileName);
                fprintf('\nProcessing %s (Type: %s)...\n', fileName, locoType);
                
                try
                    % Load the file based on extension
                    [~, ~, ext] = fileparts(fileName);
                    
                    if strcmpi(ext, '.mat')
                        % MAT file handling
                        data = load(filePath);
                        fieldNames = fieldnames(data);
                        
                        if i == 1
                            fprintf('MAT File Fields:\n');
                            disp(fieldNames);
                        end
                        
                        % Ask user which field to use for data
                        if i == 1
                            dataField = input('Enter the field name containing the data: ', 's');
                        end
                        
                        if isfield(data, dataField)
                            X = data.(dataField);
                            
                            % Handle cell arrays like in STRIDES dataset
                            if iscell(X)
                                fprintf('Data is in cell format. Combining cells...\n');
                                combinedX = [];
                                
                                for j = 1:length(X)
                                    if iscell(X{j})
                                        % For nested cell arrays (like STRIDES)
                                        for k = 1:length(X{j})
                                            if isstruct(X{j}{k}) && isfield(X{j}{k}, 'emg')
                                                % If it's structured like STRIDES
                                                emg_data = X{j}{k}.emg;
                                                % Skip the header if present
                                                if istable(emg_data)
                                                    emg_data = table2array(emg_data(:, 2:end));
                                                end
                                                combinedX = cat(1, combinedX, emg_data);
                                            elseif isnumeric(X{j}{k})
                                                combinedX = cat(1, combinedX, X{j}{k});
                                            end
                                        end
                                    elseif isnumeric(X{j})
                                        combinedX = cat(1, combinedX, X{j});
                                    end
                                end
                                
                                X = combinedX;
                            end
                        else
                            fprintf('Field ''%s'' not found in the MAT file.\n', dataField);
                            continue;
                        end
                    else
                        % CSV file handling
                        X = readmatrix(filePath);
                    end
                    
                    % Handle NaN values in features
                    if any(isnan(X(:)))
                        fprintf('Replacing NaN values with column means.\n');
                        colMeans = nanmean(X);
                        [rows, cols] = find(isnan(X));
                        
                        for j = 1:length(rows)
                            X(rows(j), cols(j)) = colMeans(cols(j));
                        end
                    end
                    
                    % Create labels array (all rows labeled with the locomotion type)
                    y = repmat({locoType}, size(X, 1), 1);
                    
                    allFeatures{end+1} = X; %#ok<AGROW>
                    allLabels{end+1} = y; %#ok<AGROW>
                    
                    fprintf('Added %d samples of type ''%s''\n', size(X, 1), locoType);
                    
                catch ME
                    fprintf('Error processing %s: %s\n', fileName, ME.message);
                    continue;
                end
            end
            
            if isempty(allFeatures)
                fprintf('No valid data loaded from any files.\n');
                X_combined = [];
                y_combined = [];
                return;
            end
            
            % Combine all data
            X_combined = vertcat(allFeatures{:});
            y_combined = vertcat(allLabels{:});
            
            fprintf('\nCombined dataset:\n');
            fprintf('- Total samples: %d\n', size(X_combined, 1));
            fprintf('- Feature dimensions: %d\n', size(X_combined, 2));
            fprintf('- Unique locomotion types: ');
            uniqueTypes = unique(y_combined);
            for i = 1:length(uniqueTypes)
                fprintf('%s ', uniqueTypes{i});
            end
            fprintf('\n');
        end
        
        function X = loadSingleFile(obj, filePath)
            % Load a single file for prediction
            %
            % Args:
            %   filePath (str): Path to file
            %
            % Returns:
            %   X: Features matrix
            
            try
                % Load the file based on extension
                [~, ~, ext] = fileparts(filePath);
                
                if strcmpi(ext, '.mat')
                    % MAT file handling
                    data = load(filePath);
                    fieldNames = fieldnames(data);
                    
                    fprintf('MAT File Fields:\n');
                    disp(fieldNames);
                    
                    % Ask user which field to use for data
                    dataField = input('Enter the field name containing the data: ', 's');
                    
                    if isfield(data, dataField)
                        X = data.(dataField);
                        
                        % Handle cell arrays like in STRIDES dataset
                        if iscell(X)
                            fprintf('Data is in cell format. Combining cells...\n');
                            combinedX = [];
                            
                            for j = 1:length(X)
                                if iscell(X{j})
                                    % For nested cell arrays (like STRIDES)
                                    for k = 1:length(X{j})
                                        if isstruct(X{j}{k}) && isfield(X{j}{k}, 'emg')
                                            % If it's structured like STRIDES
                                            emg_data = X{j}{k}.emg;
                                            % Skip the header if present
                                            if istable(emg_data)
                                                emg_data = table2array(emg_data(:, 2:end));
                                            end
                                            combinedX = cat(1, combinedX, emg_data);
                                        elseif isnumeric(X{j}{k})
                                            combinedX = cat(1, combinedX, X{j}{k});
                                        end
                                    end
                                elseif isnumeric(X{j})
                                    combinedX = cat(1, combinedX, X{j});
                                end
                            end
                            
                            X = combinedX;
                        end
                    else
                        fprintf('Field ''%s'' not found in the MAT file.\n', dataField);
                        X = [];
                        return;
                    end
                else
                    % CSV file handling
                    X = readmatrix(filePath);
                end
                
                % Handle NaN values in features
                if any(isnan(X(:)))
                    fprintf('Replacing NaN values with column means.\n');
                    colMeans = nanmean(X);
                    [rows, cols] = find(isnan(X));
                    
                    for j = 1:length(rows)
                        X(rows(j), cols(j)) = colMeans(cols(j));
                    end
                end
                
                fprintf('\nLoaded data from %s\n', filePath);
                fprintf('Data shape: [%d, %d]\n', size(X, 1), size(X, 2));
                
            catch ME
                fprintf('Error loading %s: %s\n', filePath, ME.message);
                X = [];
            end
        end
        
        function [X_processed, y_processed] = preprocessData(obj, X, y, training)
            % Preprocess the data
            %
            % Args:
            %   X: Input features
            %   y: Input labels (optional)
            %   training: Whether this is training data or prediction data
            %
            % Returns:
            %   Processed X (and y if training)
            
            if isempty(X)
                error('No valid data loaded.');
            end
            
            % Convert X to double
            X = double(X);
            
            % Standardize features
            if training
                % Compute mean and std for standardization
                obj.scaler.mean = mean(X);
                obj.scaler.std = std(X) + eps; % Add eps to avoid division by zero
                X_processed = (X - obj.scaler.mean) ./ obj.scaler.std;
            else
                % Use previously computed mean and std
                X_processed = (X - obj.scaler.mean) ./ obj.scaler.std;
            end
            
            % If labels are provided (for training)
            if nargin > 2 && ~isempty(y) && training
                % Create label encoder
                uniqueLabels = unique(y);
                obj.labelEncoder.classes = uniqueLabels;
                
                % Encode labels as integers
                y_processed = zeros(length(y), 1);
                for i = 1:length(uniqueLabels)
                    if iscell(y)
                        y_processed(strcmp(y, uniqueLabels{i})) = i;
                    else
                        y_processed(y == uniqueLabels(i)) = i;
                    end
                end
                
                % Update classes
                obj.classes = uniqueLabels;
                
                return;
            end
            
            y_processed = [];
        end
        
        function history = train(obj, X, y, testSize, valSize, epochs)
            % Train the locomotion classification model
            %
            % Args:
            %   X: Input features
            %   y: Input labels
            %   testSize: Proportion of data to use for testing
            %   valSize: Proportion of training data to use for validation
            %   epochs: Number of training epochs
            %
            % Returns:
            %   History of model training
            
            if nargin < 4
                testSize = 0.2;
            end
            
            if nargin < 5
                valSize = 0.2;
            end
            
            if nargin < 6
                epochs = 50;
            end
            
            % Preprocess data
            [X_processed, y_processed] = obj.preprocessData(X, y, true);
            
            % Convert labels to categorical
            if iscell(obj.classes)
                y_cat = categorical(y_processed, 1:length(obj.classes), obj.classes);
            else
                y_cat = categorical(y_processed);
            end
            
            % Create train/test split
            cv = cvpartition(size(X_processed, 1), 'Holdout', testSize);
            X_train_val = X_processed(training(cv), :);
            y_train_val = y_cat(training(cv));
            X_test = X_processed(test(cv), :);
            y_test = y_cat(test(cv));
            
            % Create train/validation split
            cv_val = cvpartition(size(X_train_val, 1), 'Holdout', valSize);
            X_train = X_train_val(training(cv_val), :);
            y_train = y_train_val(training(cv_val));
            X_val = X_train_val(test(cv_val), :);
            y_val = y_train_val(test(cv_val));
            
            % Reshape for CNN input
            % First, determine if we need to reshape based on dimensionality
            if size(X_train, 3) == 1 || ismatrix(X_train)
                % If 2D, reshape to [samples, height, width, channels]
                X_train_reshaped = reshape(X_train, size(X_train, 1), size(X_train, 2), 1, 1);
                X_val_reshaped = reshape(X_val, size(X_val, 1), size(X_val, 2), 1, 1);
                X_test_reshaped = reshape(X_test, size(X_test, 1), size(X_test, 2), 1, 1);
            else
                % If already 3D+, use as is
                X_train_reshaped = X_train;
                X_val_reshaped = X_val;
                X_test_reshaped = X_test;
            end
            
            % Create model
            numClasses = length(obj.classes);
            fprintf('\nCreating model for %d locomotion types\n', numClasses);
            
            % Define the CNN architecture
            layers = [
                imageInputLayer([size(X_train_reshaped, 2), 1, 1], 'Name', 'input')
                
                convolution2dLayer([3, 1], 64, 'Padding', 'same', 'Name', 'conv1')
                batchNormalizationLayer('Name', 'bn1')
                reluLayer('Name', 'relu1')
                maxPooling2dLayer([2, 1], 'Stride', [2, 1], 'Name', 'pool1')
                dropoutLayer(0.25, 'Name', 'drop1')
                
                convolution2dLayer([3, 1], 128, 'Padding', 'same', 'Name', 'conv2')
                batchNormalizationLayer('Name', 'bn2')
                reluLayer('Name', 'relu2')
                maxPooling2dLayer([2, 1], 'Stride', [2, 1], 'Name', 'pool2')
                dropoutLayer(0.25, 'Name', 'drop2')
                
                convolution2dLayer([3, 1], 256, 'Padding', 'same', 'Name', 'conv3')
                batchNormalizationLayer('Name', 'bn3')
                reluLayer('Name', 'relu3')
                maxPooling2dLayer([2, 1], 'Stride', [2, 1], 'Name', 'pool3')
                dropoutLayer(0.25, 'Name', 'drop3')
                
                flattenLayer('Name', 'flatten')
                fullyConnectedLayer(256, 'Name', 'fc1')
                reluLayer('Name', 'relu4')
                dropoutLayer(0.5, 'Name', 'drop4')
                fullyConnectedLayer(numClasses, 'Name', 'fc2')
                softmaxLayer('Name', 'softmax')
                classificationLayer('Name', 'output')
            ];
            
            % Training options
            options = trainingOptions('adam', ...
                'MaxEpochs', epochs, ...
                'MiniBatchSize', 32, ...
                'Shuffle', 'every-epoch', ...
                'ValidationData', {X_val_reshaped, y_val}, ...
                'ValidationFrequency', 30, ...
                'Verbose', true, ...
                'Plots', 'training-progress', ...
                'ExecutionEnvironment', 'auto', ...
                'InitialLearnRate', 0.001, ...
                'LearnRateSchedule', 'piecewise', ...
                'LearnRateDropFactor', 0.2, ...
                'LearnRateDropPeriod', 5);
            
            % Train the model
            [obj.model, history] = trainNetwork(X_train_reshaped, y_train, layers, options);
            
            % Evaluate model
            y_pred = classify(obj.model, X_test_reshaped);
            accuracy = sum(y_pred == y_test) / numel(y_test);
            fprintf('Test Accuracy: %.2f%%\n', accuracy * 100);
            
            % Display confusion matrix
            figure;
            cm = confusionmat(y_test, y_pred);
            
            % If classes are cell array, convert to string array for display
            if iscell(obj.classes)
                classNames = string(obj.classes);
            else
                classNames = string(obj.classes');
            end
            
            confusionchart(cm, classNames);
            title('Confusion Matrix - Locomotion Classification');
        end
        
        function saveModel(obj, modelDir)
            % Save the trained model and preprocessing components
            %
            % Args:
            %   modelDir: Directory to save model files
            
            if nargin < 2
                modelDir = 'model';
            end
            
            if isempty(obj.model)
                fprintf('No trained model to save.\n');
                return;
            end
            
            % Create directory if it doesn't exist
            if ~exist(modelDir, 'dir')
                mkdir(modelDir);
            end
            
            % Save model
            modelPath = fullfile(modelDir, 'locomotion_model.mat');
            save(modelPath, 'model', '-v7.3', '-struct', 'obj');
            
            % Save scaler and label encoder separately for ease of loading
            scalerPath = fullfile(modelDir, 'scaler.mat');
            save(scalerPath, 'scaler', '-v7.3', '-struct', 'obj');
            
            encoderPath = fullfile(modelDir, 'labelEncoder.mat');
            save(encoderPath, 'labelEncoder', '-v7.3', '-struct', 'obj');
            
            fprintf('Model and preprocessing components saved to %s\n', modelDir);
        end
        
        function success = loadModel(obj, modelDir)
            % Load a trained model and preprocessing components
            %
            % Args:
            %   modelDir: Directory containing model files
            %
            % Returns:
            %   success: Boolean indicating if loading was successful
            
            if nargin < 2
                modelDir = 'model';
            end
            
            modelPath = fullfile(modelDir, 'locomotion_model.mat');
            scalerPath = fullfile(modelDir, 'scaler.mat');
            encoderPath = fullfile(modelDir, 'labelEncoder.mat');
            
            allFilesExist = exist(modelPath, 'file') && ...
                exist(scalerPath, 'file') && ...
                exist(encoderPath, 'file');
            
            if ~allFilesExist
                fprintf('Missing model files. Train a model first.\n');
                success = false;
                return;
            end
            
            try
                % Load model
                modelData = load(modelPath);
                obj.model = modelData.model;
                
                % Load scaler
                scalerData = load(scalerPath);
                obj.scaler = scalerData.scaler;
                
                % Load label encoder
                encoderData = load(encoderPath);
                obj.labelEncoder = encoderData.labelEncoder;
                
                % Get classes
                obj.classes = obj.labelEncoder.classes;
                
                fprintf('Model loaded successfully. Locomotion types: ');
                for i = 1:length(obj.classes)
                    if iscell(obj.classes)
                        fprintf('%s ', obj.classes{i});
                    else
                        fprintf('%s ', obj.classes(i));
                    end
                end
                fprintf('\n');
                
                success = true;
                
            catch ME
                fprintf('Error loading model: %s\n', ME.message);
                success = false;
            end
        end
        
        function results = predict(obj, X)
            % Predict locomotion type for new data
            %
            % Args:
            %   X: Input features
            %
            % Returns:
            %   results: Struct with predicted classes and probabilities
            
            if isempty(obj.model)
                fprintf('No trained model. Train or load a model first.\n');
                results = [];
                return;
            end
            
            % Preprocess data (no labels, not training)
            X_processed = obj.preprocessData(X, [], false);
            
            % Reshape for CNN input
            if size(X_processed, 3) == 1 || ismatrix(X_processed)
                % If 2D, reshape to [samples, height, width, channels]
                X_reshaped = reshape(X_processed, size(X_processed, 1), size(X_processed, 2), 1, 1);
            else
                % If already 3D+, use as is
                X_reshaped = X_processed;
            end
            
            % Get predictions
            [y_pred, scores] = classify(obj.model, X_reshaped);
            
            % Get predicted classes as strings/cells
            predictedLocomotion = y_pred;
            
            % Get class counts
            uniquePredictions = categories(y_pred);
            counts = zeros(length(uniquePredictions), 1);
            
            for i = 1:length(uniquePredictions)
                counts(i) = sum(y_pred == uniquePredictions(i));
            end
            
            percentages = counts / length(y_pred) * 100;
            
            % Create results summary
            results = struct();
            results.predicted_classes = y_pred;
            results.probabilities = scores;
            
            % Create summary structure
            results.summary = struct();
            for i = 1:length(uniquePredictions)
                fieldName = char(uniquePredictions(i));
                fieldName = strrep(fieldName, ' ', '_');
                fieldName = strrep(fieldName, '-', '_');
                
                if ~isvarname(fieldName)
                    fieldName = ['class_', num2str(i)];
                end
                
                results.summary.(fieldName) = percentages(i);
            end
        end
    end
end
