% MATLAB code to batch process walk.mat files across multiple subjects
% This script processes walk.mat files from folders AB06-AB30 (excluding AB22, AB26, AB29)

% Define the root directory containing the AB folders
rootDir = pwd;  % Change to be your path to folder containing all subject folders

% Create a main output directory for all processed data
mainOutputDir = fullfile(rootDir, 'all_cnn_ankle_prediction_data');
if ~exist(mainOutputDir, 'dir')
    mkdir(mainOutputDir);
end

% Create log file
logFile = fullfile(mainOutputDir, 'batch_processing_log.txt');
fileID = fopen(logFile, 'w');
fprintf(fileID, 'Batch Processing Log - Started at %s\n\n', datestr(now));

% Define the subject folders to process
startSubject = 6;
endSubject = 30;
missingSubjects = [22, 26, 29];  % These subjects are missing

% Process each subject folder
for subjectNum = startSubject:endSubject
    % Skip missing subjects
    if ismember(subjectNum, missingSubjects)
        fprintf('Skipping AB%02d (known to be missing)\n', subjectNum);
        fprintf(fileID, 'Skipping AB%02d (known to be missing)\n', subjectNum);
        continue;
    end
    
    % Construct folder name and path
    folderName = sprintf('AB%02d', subjectNum);
    folderPath = fullfile(rootDir, folderName);
    
    % Check if folder exists
    if ~exist(folderPath, 'dir')
        fprintf('Folder %s does not exist. Skipping.\n', folderName);
        fprintf(fileID, 'Folder %s does not exist. Skipping.\n', folderName);
        continue;
    end
    
    % Check if walk.mat exists in this folder
    walkMatPath = fullfile(folderPath, 'walk.mat');
    if ~exist(walkMatPath, 'file')
        fprintf('walk.mat not found in %s. Skipping.\n', folderName);
        fprintf(fileID, 'walk.mat not found in %s. Skipping.\n', folderName);
        continue;
    end
    
    % Log that we're processing this folder
    fprintf('\n==== Processing %s ====\n', folderName);
    fprintf(fileID, '\n==== Processing %s ====\n', folderName);
    
    % Start timer
    tic;
    
    try
        % Change to the subject folder
        currentDir = pwd;
        cd(folderPath);
        
        % Load the walk.mat file for this subject
        fprintf('Loading walk.mat for %s...\n', folderName);
        load('walk.mat');
        
        % Create subject-specific output directory
        subjectOutputDir = fullfile(mainOutputDir, folderName);
        if ~exist(subjectOutputDir, 'dir')
            mkdir(subjectOutputDir);
        end
        
        % Define columns to extract from each sensor type
        emg_cols = [5, 6, 7, 8, 9, 10];
        imu_cols = [14, 15, 16, 17, 18, 19];
        gon_cols = 4;
        
        % Target to predict
        target_col = 2;  % Ankle angle in sagittal plane from gon data
        
        % Get dimensions
        numStrides = length(strides);
        numTimePoints = 101;  % All tables have 101 rows (0-100% gait cycle)
        numInputFeatures = length(emg_cols) + length(imu_cols) + length(gon_cols);
        
        fprintf('Found %d strides for subject %s\n', numStrides, folderName);
        fprintf(fileID, 'Found %d strides for subject %s\n', numStrides, folderName);
        
        % Create 3D tensor for input features: [strides × time_points × selected_features]
        inputData = zeros(numStrides, numTimePoints, numInputFeatures);
        targetData = zeros(numStrides, numTimePoints, 1);  % For ankle angle
        
        % Get column names for documentation
        firstStride = strides{1};
        emg_names = firstStride.emg.Properties.VariableNames(emg_cols);
        imu_names = firstStride.imu.Properties.VariableNames(imu_cols);
        gon_names = firstStride.gon.Properties.VariableNames(gon_cols);
        target_name = firstStride.gon.Properties.VariableNames{target_col};
        
        % Combine all feature names
        featureNames = [strcat('emg_', emg_names), strcat('imu_', imu_names), strcat('gon_', gon_names)];
        
        % Fill tensors with data from each stride
        fprintf('Extracting data from each stride...\n');
        for i = 1:numStrides
            currentStride = strides{i};
            
            % Extract selected features
            emg_data = table2array(currentStride.emg(:, emg_cols));
            imu_data = table2array(currentStride.imu(:, imu_cols));
            gon_data = table2array(currentStride.gon(:, gon_cols));
            
            % Combine features into one matrix
            combined_features = [emg_data, imu_data, gon_data];
            
            % Store in 3D tensor
            inputData(i, :, :) = combined_features;
            
            % Extract target (ankle angle)
            targetData(i, :, 1) = table2array(currentStride.gon(:, target_col));
            
            % Print progress every 20 strides
            if mod(i, 20) == 0
                fprintf('  Processed %d of %d strides\n', i, numStrides);
            end
        end
        
        % Save feature names for reference
        fprintf('Saving feature metadata...\n');
        varNamesTable = table(featureNames', 'VariableNames', {'FeatureName'});
        writetable(varNamesTable, fullfile(subjectOutputDir, sprintf('%s_feature_names.csv', folderName)));
        
        % Save target name
        targetNameTable = table({target_name}, 'VariableNames', {'TargetName'});
        writetable(targetNameTable, fullfile(subjectOutputDir, sprintf('%s_target_name.csv', folderName)));
        
        % Save 3D tensors as .mat file
        fprintf('Saving .mat file with 3D tensors...\n');
        save(fullfile(subjectOutputDir, sprintf('%s_ankle_prediction_data.mat', folderName)), 'inputData', 'targetData');
        
        % Create a metadata table with stride information
        metaData = table();
        for i = 1:numStrides
            currentStride = strides{i};
            
            meta = table();
            meta.StrideIndex = i;
            meta.Subject = string(currentStride.info.Subject);
            meta.Trial = string(currentStride.info.Trial);
            meta.Speed = string(currentStride.conditions.speed);
            
            metaData = [metaData; meta];
        end
        
        % Save metadata
        writetable(metaData, fullfile(subjectOutputDir, sprintf('%s_metadata.csv', folderName)));
        
        % Create 2D matrices for CSV output with correct stride ordering
        fprintf('Creating CSV files with proper time series ordering...\n');
        rows_per_stride = numTimePoints;
        total_rows = numStrides * rows_per_stride;
        
        % Initialize matrices with the correct size
        inputMatrix = zeros(total_rows, numInputFeatures);
        targetMatrix = zeros(total_rows, 1);
        
        % Fill matrices stride by stride - keeping all timepoints for each stride together
        for i = 1:numStrides
            % Calculate the row indices for this stride
            start_row = (i-1) * rows_per_stride + 1;
            end_row = i * rows_per_stride;
            
            % Copy the data for this stride into the corresponding rows
            inputMatrix(start_row:end_row, :) = squeeze(inputData(i, :, :));
            targetMatrix(start_row:end_row, :) = targetData(i, :, 1);
        end
        
        % Create stride and time indices that align with the new ordering
        strideIndices = zeros(total_rows, 1);
        timeIndices = zeros(total_rows, 1);
        
        for i = 1:numStrides
            start_row = (i-1) * rows_per_stride + 1;
            end_row = i * rows_per_stride;
            
            strideIndices(start_row:end_row) = i;
            timeIndices(start_row:end_row) = 0:100;
        end
        
        % Create input table with indices
        inputTable = array2table(inputMatrix, 'VariableNames', featureNames);
        inputTable = addvars(inputTable, strideIndices, timeIndices, 'Before', 1, ...
            'NewVariableNames', {'StrideIndex', 'NormalizedTime'});
        
        % Create target table
        targetTable = array2table(targetMatrix, 'VariableNames', {target_name});
        targetTable = addvars(targetTable, strideIndices, timeIndices, 'Before', 1, ...
            'NewVariableNames', {'StrideIndex', 'NormalizedTime'});
        
        % Combine input and target for a single CSV (optional)
        combinedTable = [inputTable, targetTable(:, 3)];  % Only add the target value column
        
        % Save reshaped data
        writetable(inputTable, fullfile(subjectOutputDir, sprintf('%s_input_features.csv', folderName)));
        writetable(targetTable, fullfile(subjectOutputDir, sprintf('%s_target_ankle_angle.csv', folderName)));
        writetable(combinedTable, fullfile(subjectOutputDir, sprintf('%s_combined_data.csv', folderName)));
        
        % Return to the original directory
        cd(currentDir);
        
        % Calculate and log processing time
        processingTime = toc;
        fprintf('Successfully processed %s in %.2f seconds\n', folderName, processingTime);
        fprintf(fileID, 'Successfully processed %s in %.2f seconds\n', folderName, processingTime);
        
    catch ME
        % Handle errors gracefully
        fprintf('ERROR processing %s: %s\n', folderName, ME.message);
        fprintf(fileID, 'ERROR processing %s: %s\n', folderName, ME.message);
        
        % Return to the original directory in case of error
        cd(rootDir);
    end
end

% Close the log file
fclose(fileID);
fprintf('\nBatch processing complete! See log file for details: %s\n', logFile);