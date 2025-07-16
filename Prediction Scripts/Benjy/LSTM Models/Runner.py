import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import random

# Load the best saved model
best_model = tf.keras.models.load_model("final_best_model.keras")

# Load the dataset
dataset_path = '/Users/benjyb/PycharmProjects/PythonProject1/LSTM Models/Temporary data/combined_dataAB06.csv'
data = pd.read_csv(dataset_path, header=0)  # Ensure first row is treated as column names

# Assume first column is gon_data (output), remaining columns are emg_imu_data (input features)
gon_data = data.iloc[:, [0]]  # First column as DataFrame

# Compute mean and std for normalization
mean_X, std_X = data.mean(), data.std()
mean_Y, std_Y = gon_data.mean().iloc[0], gon_data.std().iloc[0]

# Define parameters
p = 50  # Past timesteps (input)
f = 10  # Future timesteps (output)
predict_window = 200  # Total prediction window in ms
slide_step = 10  # Slide every 10ms

total_slides = predict_window // slide_step

# Select a random starting index that is a multiple of 100
valid_start_indices = [i for i in range(0, len(data) - (p + f + predict_window), 100)]
if not valid_start_indices:
    raise ValueError("Not enough data to select a valid starting index.")
start_idx = random.choice(valid_start_indices)
end_idx = start_idx + p

predictions = []
true_values = []

for _ in range(total_slides):
    if end_idx + f > len(data):
        break  # Avoid exceeding dataset length

    # Normalize input using the training mean and std
    X_input = ((data.iloc[start_idx:end_idx, :] - mean_X) / std_X).values.reshape(1, p, -1)

    # Extract true values (no need to normalize if using raw data for comparison)
    Y_true = gon_data.iloc[end_idx:end_idx + f, :].values

    # Make prediction
    Y_pred = best_model.predict(X_input)

    # Denormalize predictions to match the original scale
    Y_pred = Y_pred * std_Y + mean_Y

    predictions.append(Y_pred.squeeze())  # Store predictions
    true_values.append(Y_true.squeeze())  # Store true values

    start_idx += slide_step
    end_idx += slide_step

# Convert lists to numpy arrays
predictions = np.concatenate(predictions, axis=0)
true_values = np.concatenate(true_values, axis=0)

# Compute test loss (Mean Squared Error)
test_loss = np.mean((predictions - true_values) ** 2)
print(f"Test Loss (MSE): {test_loss:.4f}")

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(np.arange(predict_window), true_values, label='True', linestyle='-', color='b')
plt.plot(np.arange(predict_window), predictions, label='Predicted', linestyle='--', color='r')
plt.title('Prediction of Ankle Flexion 10ms into the future')
plt.xlabel('Time Step (ms)')
plt.ylabel('Ankle Flexion (ms)')
plt.legend()
plt.show()
