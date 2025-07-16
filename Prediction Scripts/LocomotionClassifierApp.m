%% LocomotionClassifierApp - Main script to run the classifier

function LocomotionClassifierApp()
    % Main script to run the locomotion classifier
    
    % Get folder path
    folderPath = input('Enter the folder path containing data files: ', 's');
    
    if ~exist(folderPath, 'dir')
        fprintf('Error: %s is not a valid directory.\n', folderPath);
        return;
    end
    
    % Ask for operation mode
    fprintf('\nSelect operation mode:\n');
    fprintf('1. Train a new locomotion classifier\n');
    fprintf('2. Classify a new dataset\n');
    
    while true
        try
            mode = str2double(input('Enter mode (1 or 2): ', 's'));
            if ~isnan(mode) && (mode == 1 || mode == 2)
                break;
            else
                fprintf('Invalid choice. Please enter 1 or 2.\n');
            end
        catch
            fprintf('Invalid input. Please enter a number.\n');
        end
    end
    
    if mode == 1
        trainMode(folderPath);
    else
        predictMode(folderPath);
    end
end

function trainMode(folderPath)
    % Run the classifier in training mode
    
    classifier = LocomotionClassifier();
    
    % Load multiple files with different locomotion types
    [X, y] = classifier.loadMultipleFiles(folderPath);
    
    if ~isempty(X) && ~isempty(y)
        % Ask for number of epochs
        while true
            try
                epochsStr = input('Enter number of training epochs (default: 50): ', 's');
                if isempty(epochsStr)
                    epochs = 50;
                    break;
                else
                    epochs = str2double(epochsStr);
                    if ~isnan(epochs) && epochs > 0
                        break;
                    else
                        fprintf('Number of epochs must be positive.\n');
                    end
                end
            catch
                fprintf('Invalid input. Please enter a number.\n');
            end
        end
        
        % Train the model
        classifier.train(X, y, 0.2, 0.2, epochs);
        
        % Save the model
        saveModel = input('Save the trained model? (y/n): ', 's');
        if strcmpi(saveModel, 'y')
            modelDir = input('Enter model directory (default: ''model''): ', 's');
            if isempty(modelDir)
                modelDir = 'model';
            end
            classifier.saveModel(modelDir);
        end
    else
        fprintf('Could not load valid training data.\n');
    end
end

function predictMode(folderPath)
    % Run the classifier in prediction mode
    
    % First load a trained model
    classifier = LocomotionClassifier();
    
    modelDir = input('Enter model directory (default: ''model''): ', 's');
    if isempty(modelDir)
        modelDir = 'model';
    end
    
    if ~classifier.loadModel(modelDir)
        fprintf('Failed to load model. Exiting prediction mode.\n');
        return;
    end
    
    % List all data files in the directory
    files = [dir(fullfile(folderPath, '*.mat')); dir(fullfile(folderPath, '*.csv'))];
    
    if isempty(files)
        fprintf('No MAT or CSV files found in the directory.\n');
        return;
    end
    
    % Display available files
    fprintf('Available files for prediction:\n');
    for i = 1:length(files)
        fprintf('%d. %s\n', i, files(i).name);
    end
    
    % Select a file
    while true
        try
            choice = str2double(input('Enter the number of the file to classify: ', 's'));
            if ~isnan(choice) && choice >= 1 && choice <= length(files)
                break;
            else
                fprintf('Invalid choice. Please enter a number from the list.\n');
            end
        catch
            fprintf('Invalid input. Please enter a number.\n');
        end
    end
    
    % Full path to selected file
    selectedFile = fullfile(folderPath, files(choice).name);
    
    % Load data for prediction
    X = classifier.loadSingleFile(selectedFile);
    
    if ~isempty(X)
        % Make predictions
        fprintf('\nAnalyzing locomotion type...\n');
        results = classifier.predict(X);
        
        if ~isempty(results)
            fprintf('\nPrediction Results:\n');
            fprintf('%s\n', repmat('-', 1, 40));
            fprintf('File: %s\n', files(choice).name);
            fprintf('%s\n', repmat('-', 1, 40));
            fprintf('Locomotion Type Breakdown:\n');
            
            % Get field names and values from summary
            fieldNames = fieldnames(results.summary);
            values = struct2cell(results.summary);
            
            % Sort by percentage (highest first)
            [sortedValues, sortIdx] = sort(cell2mat(values), 'descend');
            sortedNames = fieldNames(sortIdx);
            
            for i = 1:length(sortedNames)
                % Clean up field name for display
                displayName = strrep(sortedNames{i}, '_', ' ');
                fprintf('- %s: %.2f%%\n', displayName, sortedValues(i));
            end
            
            fprintf('%s\n', repmat('-', 1, 40));
            
            % Determine overall classification
            majorityClass = strrep(sortedNames{1}, '_', ' ');
            majorityPercentage = sortedValues(1);
            
            fprintf('Overall Classification: %s (%.2f%%)\n', majorityClass, majorityPercentage);
            fprintf('\nPrediction Results:\n');
            fprintf('%s\n', repmat('-', 1, 40));
            fprintf('File: %s\n', files(choice).name);
            fprintf('%s\n', repmat('-', 1, 40));
            fprintf('Locomotion Type Breakdown:\n');
            
            % Get field names and values from summary
            fieldNames = fieldnames(results.summary);
            values = struct2cell(results.summary);
            
            % Sort by percentage (highest first)
            [sortedValues, sortIdx] = sort(cell2mat(values), 'descend');
            sortedNames = fieldNames(sortIdx);
            
            for i = 1:length(sortedNames)
                % Clean up field name for display
                displayName = strrep(sortedNames{i}, '_', ' ');
                fprintf('- %s: %.2f%%\n', displayName, sortedValues(i));
            end
            
            fprintf('%s\n', repmat('-', 1, 40));
            
            % Determine overall classification
            majorityClass = strrep(sortedNames{1}, '_', ' ');
            majorityPercentage = sortedValues(1);
            
            fprintf('Overall Classification: %s (%.2f%%)\n', majorityClass, majorityPercentage);
            
            % Visualize the predictions
            figure;
            
            % Create a pie chart for locomotion type distribution
            subplot(1, 2, 1);
            pie(sortedValues, cellstr(strrep(sortedNames, '_', ' ')));
            title('Locomotion Type Distribution');
            
            % Create a bar chart for top prediction probabilities
            subplot(1, 2, 2);
            bar(sortedValues(1:min(5, length(sortedValues))));
            set(gca, 'XTickLabel', strrep(sortedNames(1:min(5, length(sortedNames))), '_', ' '));
            title('Top Locomotion Types');
            ylabel('Percentage (%)');
            xtickangle(45);
            
            % Fit figure to screen
            set(gcf, 'Position', get(0, 'Screensize'));
        else
            fprintf('Failed to generate predictions.\n');
        end
    else
        fprintf('Failed to load data from %s.\n', selectedFile);
    end
    
    % Ask if user wants to classify another file
    anotherFile = input('\nClassify another file? (y/n): ', 's');
    if strcmpi(anotherFile, 'y')
        predictMode(folderPath);
    end
end
