import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
import os
import json
import warnings
warnings.filterwarnings('ignore')

# Define constants that match the original training parameters
p = 50  # Past timesteps (input)
f = 10  # Future timesteps (output)

class TimeSeriesCNN(nn.Module):
    def __init__(self, input_channels, output_length=10):
        super(TimeSeriesCNN, self).__init__()

        # First convolutional layer with dilation=1
        # PyTorch Conv1d expects [batch, channels, sequence] format
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=3, padding=1, dilation=1)
        self.relu1 = nn.ReLU()

        # Second convolutional layer with dilation=2 for wider receptive field
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=2, dilation=2)
        self.relu2 = nn.ReLU()

        # Third convolutional layer with dilation=4 for even wider receptive field
        self.conv3 = nn.Conv1d(128, 64, kernel_size=3, padding=4, dilation=4)
        self.relu3 = nn.ReLU()

        # Calculate flattened size after convolutions
        # With proper padding, the sequence length remains p (50)
        self.flat_size = 64 * p

        # Fully connected layers for final prediction
        self.fc1 = nn.Linear(self.flat_size, 100)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(100, output_length)

    def forward(self, x):
        # PyTorch Conv1d expects [batch, channels, sequence] but input is [batch, sequence, channels]
        # So permute the dimensions
        x = x.permute(0, 2, 1)

        # Apply convolutions
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))

        # Flatten the output for the fully connected layers
        x = x.view(x.size(0), -1)

        # Apply fully connected layers
        x = self.relu4(self.fc1(x))
        x = self.fc2(x)

        # Reshape to match expected output dimensions [batch, 10, 1]
        x = x.view(x.size(0), f, 1)

        return x

def load_json_predictions(json_file="predictions_list.json"):
    """Load predictions from JSON file"""
    try:
        with open(json_file, 'r') as file:
            predictions = json.load(file)
        print(f"Successfully loaded {len(predictions)} predictions from {json_file}")
        return predictions
    except Exception as e:
        print(f"Error loading JSON file: {str(e)}")
        return None

def load_pretrained_model(model_path, csv_files):
    """Load the pretrained model with proper error handling"""
    try:
        # Determine device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Get number of features from a sample of the data to initialize model correctly
        sample_df = pd.DataFrame()
        for csv_file in csv_files:
            try:
                sample_df = pd.read_csv(csv_file)
                if not sample_df.empty:
                    print(f"Using {csv_file} to determine model dimensions")
                    break
            except Exception as e:
                print(f"Could not read {csv_file}: {str(e)}")

        if sample_df.empty:
            raise ValueError("All CSV files are empty or couldn't be read, cannot determine input dimensions")
            
        # Assume target is in second column, features start from third column
        num_features = sample_df.shape[1] - 2  # Subtract ID and target columns
        
        # Initialize model
        model = TimeSeriesCNN(input_channels=num_features).to(device)
        
        # Load pretrained weights
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()  # Set model to evaluation mode
        
        print(f"Successfully loaded model from {model_path}")
        print(f"Model initialized with {num_features} input features")
        return model, device
    
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None, None

def preprocess_data(csv_file):
    """Load and preprocess data from CSV file"""
    try:
        print(f"Processing file: {csv_file}")
        data = pd.read_csv(csv_file)
        
        if data.empty:
            print(f"Warning: {csv_file} contains no data")
            return None
            
        print(f"CSV shape: {data.shape}, Columns: {data.columns.tolist()}")
        
        # Extract target (ankle_sagittal should be in second column)
        # Adjust column indices based on your actual data format
        target_col_idx = 1  # Assuming second column contains ankle_sagittal
        input_cols_start = 2  # Assuming features start from third column
        
        target_data = data.iloc[:, target_col_idx:target_col_idx+1]
        feature_data = data.iloc[:, input_cols_start:]
        
        print(f"Target shape: {target_data.shape}, Features shape: {feature_data.shape}")
        
        # Normalize data
        mean_X, std_X = feature_data.mean(), feature_data.std()
        mean_Y, std_Y = target_data.mean().iloc[0], target_data.std().iloc[0]
        
        X_norm = (feature_data - mean_X) / std_X
        Y_norm = (target_data - mean_Y) / std_Y
        
        return {
            'X_norm': X_norm, 
            'Y_norm': Y_norm, 
            'mean_Y': mean_Y, 
            'std_Y': std_Y,
            'feature_names': feature_data.columns.tolist(),
            'original_data': data
        }
        
    except Exception as e:
        print(f"Error preprocessing data: {str(e)}")
        return None

def create_sequences(X_norm, Y_norm, seq_length=p, pred_length=f):
    """Create input sequences for prediction"""
    X_data, Y_data = [], []
    
    for i in range(len(X_norm) - seq_length - pred_length + 1):
        X_data.append(X_norm.iloc[i:i+seq_length, :].values)
        Y_data.append(Y_norm.iloc[i+seq_length:i+seq_length+pred_length, :].values)
    
    # Convert to numpy arrays
    X_data = np.array(X_data)
    Y_data = np.array(Y_data) 
    
    print(f"Created {len(X_data)} sequences, X shape: {X_data.shape}, Y shape: {Y_data.shape}")
    return X_data, Y_data

def predict_with_model(model, X_data, device):
    """Make predictions using the loaded model"""
    model.eval()
    predictions = []
    
    with torch.no_grad():
        # Process in batches to avoid memory issues
        batch_size = 64
        num_batches = int(np.ceil(len(X_data) / batch_size))
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(X_data))
            
            batch_X = torch.FloatTensor(X_data[start_idx:end_idx]).to(device)
            batch_pred = model(batch_X).cpu().numpy()
            predictions.append(batch_pred)
    
    # Concatenate batch predictions
    predictions = np.vstack(predictions)
    return predictions

def evaluate_model(predictions, Y_data, mean_Y, std_Y):
    """Calculate evaluation metrics"""
    # Denormalize predictions and true data
    pred_denorm = predictions * std_Y + mean_Y
    truth_denorm = Y_data * std_Y + mean_Y
    
    # Calculate metrics
    mse = np.mean((pred_denorm - truth_denorm) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(pred_denorm - truth_denorm))
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'pred_denorm': pred_denorm,
        'truth_denorm': truth_denorm
    }

def visualize_results(results, dataset_name, json_predictions=None):
    """Visualize prediction results with JSON predictions overlay"""
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # For visualization clarity, plot a subset of the results
    sample_size = min(200, len(results['pred_denorm']))
    start_idx = 0  # Can be adjusted to show different parts of the data
    
    # 1. Time series plot (first subplot)
    # Extract a subset of the predictions and true data for the first prediction step
    pred_subset = results['pred_denorm'][start_idx:start_idx+sample_size, 0, 0]  # First prediction step
    truth_subset = results['truth_denorm'][start_idx:start_idx+sample_size, 0, 0]  # First target step
    
    time_points = np.arange(len(pred_subset))
    
    # Plot true data
    axes[0].plot(time_points, truth_subset, 'b-', label='True data', linewidth=2)
    
    # Plot model predictions
    axes[0].plot(time_points, pred_subset, 'r--', label='Model Predictions', linewidth=1.5)
    
    # Add JSON predictions if available and if on the first dataset
    if json_predictions and "CNN_vs_LSTM_dataset_test" in dataset_name:
        # Get the subset of JSON predictions to match the same timeframe
        json_subset = json_predictions[:sample_size] if len(json_predictions) >= sample_size else json_predictions
        
        # Plot JSON predictions on the same subplot
        axes[0].plot(time_points[:len(json_subset)], json_subset, 'g-.', 
                     label='JSON Predictions', linewidth=1.5)
        
        # Calculate metrics for JSON predictions
        json_mse = np.mean((np.array(json_subset) - truth_subset[:len(json_subset)]) ** 2)
        json_rmse = np.sqrt(json_mse)
        json_mae = np.mean(np.abs(np.array(json_subset) - truth_subset[:len(json_subset)]))
        
        # Add text for JSON metrics
        json_metrics_text = (
            f"JSON Predictions Metrics:\n"
            f"Mean Squared Error (MSE): {json_mse:.4f}\n"
            f"Root Mean Squared Error (RMSE): {json_rmse:.4f}\n"
            f"Mean Absolute Error (MAE): {json_mae:.4f}"
        )
        
        # Position the JSON metrics text on the figure
        fig.text(0.15, 0.01, json_metrics_text, fontsize=12, 
                 bbox=dict(facecolor='lightgreen', edgecolor='gray', alpha=0.8, boxstyle='round,pad=0.5'))
    
    axes[0].set_title(f'Ankle Flexion Prediction - Time Series View - {dataset_name}', fontsize=14)
    axes[0].set_xlabel('Time Steps', fontsize=12)
    axes[0].set_ylabel('Ankle Flexion Angle (degrees)', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='best')
    
    # 2. Scatter plot (second subplot)
    axes[1].scatter(truth_subset, pred_subset, alpha=0.6, edgecolors='w', s=30, label='Model Predictions')
    
    # Add scatter plot for JSON predictions if available
    if json_predictions and "CNN_vs_LSTM_dataset_test" in dataset_name:
        json_subset = json_predictions[:sample_size] if len(json_predictions) >= sample_size else json_predictions
        axes[1].scatter(truth_subset[:len(json_subset)], json_subset, 
                       alpha=0.6, edgecolors='w', s=30, c='green', label='JSON Predictions')
    
    # Add diagonal line for perfect predictions
    min_val = min(min(truth_subset), min(pred_subset))
    max_val = max(max(truth_subset), max(pred_subset))
    axes[1].plot([min_val, max_val], [min_val, max_val], 'k--', label='Perfect Prediction')
    
    axes[1].set_title(f'Prediction vs. True data - {dataset_name}', fontsize=14)
    axes[1].set_xlabel('True ankle data Angle (degrees)', fontsize=12)
    axes[1].set_ylabel('Predicted Angle (degrees)', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='best')
    
    # Add model metrics to the plot
    model_metrics_text = (
        f"Model Predictions Metrics:\n"
        f"Mean Squared Error (MSE): {results['MSE']:.4f}\n"
        f"Root Mean Squared Error (RMSE): {results['RMSE']:.4f}\n"
        f"Mean Absolute Error (MAE): {results['MAE']:.4f}"
    )
    
    # Position the model metrics text
    x_pos = 0.55 if json_predictions and "CNN_vs_LSTM_dataset_test" in dataset_name else 0.15
    fig.text(x_pos, 0.01, model_metrics_text, fontsize=12, 
             bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Adjust for text at bottom
    
    # Save the figure
    output_file = f"model_evaluation_{dataset_name.replace(' ', '_').replace('.csv', '')}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to {output_file}")
    
    plt.show()

def test_on_dataset(model, device, csv_file, json_predictions=None):
    """Test the model on a specific dataset"""
    file_name = os.path.basename(csv_file)
    print(f"\n{'='*50}")
    print(f"Testing model on {file_name}")
    print(f"{'='*50}\n")
    
    # Preprocess data
    processed_data = preprocess_data(csv_file)
    if processed_data is None:
        print(f"Skipping {file_name} due to preprocessing error")
        return None
    
    # Create sequences
    X_data, Y_data = create_sequences(
        processed_data['X_norm'], 
        processed_data['Y_norm']
    )
    
    # Make predictions
    print("Making predictions...")
    predictions = predict_with_model(model, X_data, device)
    
    # Evaluate model
    results = evaluate_model(predictions, Y_data, processed_data['mean_Y'], processed_data['std_Y'])
    
    # Print evaluation results
    print("\nEvaluation Metrics:")
    print(f"MSE: {results['MSE']:.4f}")
    print(f"RMSE: {results['RMSE']:.4f}")
    print(f"MAE: {results['MAE']:.4f}")
    
    # Generate overall prediction sequence for visualization
    all_predictions = []
    for i in range(predictions.shape[0]):
        # Get prediction for this sequence
        seq_pred = predictions[i, 0, 0]  # Just take the first prediction point
        all_predictions.append(seq_pred)
    
    # Add this to results for potential analysis
    results['prediction_sequence'] = np.array(all_predictions)
    
    # Visualize results
    visualize_results(results, file_name, json_predictions)
    
    return results

def main():
    # Define paths
    model_path = "best_model.pth"
    csv_files = ["C:/Users/awkir/Documents/UniWork/Y3/MDM3/PhaseC/Data/CNN_vs_LSTM_dataset_test.csv", "C:/Users/awkir/Documents/UniWork/Y3/MDM3/PhaseC/Data/CNN_vs_LSTM_subject_14_test.csv"]
    json_file = "C:/Users/awkir/Documents/UniWork/Y3/MDM3/PhaseC/Data/predictions_list.json"
    
    # Check if files exist
    for file in [model_path] + csv_files:
        if not os.path.exists(file):
            print(f"Error: {file} does not exist")
            if file == model_path:
                return
    
    # Load JSON predictions
    json_predictions = load_json_predictions(json_file)
    
    # Load pretrained model
    model, device = load_pretrained_model(model_path, csv_files)
    if model is None:
        return
    
    # Test on each dataset
    results = {}
    for csv_file in csv_files:
        # Only pass JSON predictions for the first dataset
        csv_basename = os.path.basename(csv_file)
        if "CNN_vs_LSTM_dataset_test" in csv_file:
            results[csv_basename] = test_on_dataset(model, device, csv_file, json_predictions)
        else:
            results[csv_basename] = test_on_dataset(model, device, csv_file)
    
    # Compare results between datasets if both were successful
    if all(r is not None for r in results.values()):
        print("\nComparison between datasets:")
        print(f"{'Dataset':<35} {'RMSE':>10} {'MAE':>10}")
        print('-' * 55)
        for file, res in results.items():
            print(f"{file:<35} {res['RMSE']:>10.4f} {res['MAE']:>10.4f}")

if __name__ == "__main__":
    main()