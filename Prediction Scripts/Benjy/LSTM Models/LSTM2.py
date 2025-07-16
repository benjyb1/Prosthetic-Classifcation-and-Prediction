import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Reshape, Input
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random

# Load the dataset
dataset_path = '/Users/benjyb/Documents/MATLAB/STRIDES/merged_walk_data.csv'
data = pd.read_csv(dataset_path)

# Separate input (X) and output (Y)
gon_data = data.iloc[:, :1]  # Y (first column)
full_data = data.iloc[:, 1:]  # X (all other columns)

# Compute mean and std for normalization
mean_X, std_X = full_data.mean(), full_data.std()
mean_Y, std_Y = gon_data.mean().iloc[0], gon_data.std().iloc[0]

# Normalize both X and Y globally before creating sequences
X_norm = (full_data - mean_X) / std_X
Y_norm = (gon_data - mean_Y) / std_Y

# Define parameters
p = 50  # Past timesteps (input)
f = 10  # Future timesteps (output)

# Prepare training sequences from the normalized dataset
X_data, Y_data = [], []
for i in range(len(full_data) - p - f):
    X_data.append(X_norm.iloc[i:i+p, :].values)  # Input: past 50ms (already normalized)
    Y_data.append(Y_norm.iloc[i+p:i+p+f, :].values)  # Output: next 10ms (already normalized)

X_data = np.array(X_data)  # Shape: (samples, 50, 40)
Y_data = np.array(Y_data)  # Shape: (samples, 10, 1)

# Split into train/test sets
X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.2, shuffle=False)
print("X_data shape:", X_data.shape)  # Should be (samples, 50, 10)
print("Y_data shape:", Y_data.shape)  # Should be (samples, 10, 1)

# Convert to tf.data.Dataset for better performance
def create_dataset(X, Y, batch_size=16):
    # Convert to TensorFlow Dataset
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    dataset = dataset.batch(batch_size)  # Batch the dataset
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)  # Prefetch for better performance
    return dataset

# Create train and test datasets
train_dataset = create_dataset(X_train, Y_train, batch_size=16)
test_dataset = create_dataset(X_test, Y_test, batch_size=16)

# Define model
model = Sequential([
    Input(shape=(50, X_data.shape[2])),
    LSTM(100, activation='relu', return_sequences=True),
    LSTM(50, activation='relu', return_sequences=False),
    Dense(10),
    Reshape((10, 1))
])

# Compile with lower learning rate and gradient clipping
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0), loss='mean_squared_error')

# Callback to save the best model based on validation loss
checkpoint_callback = ModelCheckpoint(
    filepath="best_model.keras",
    save_best_only=True,
    save_weights_only=False,
    monitor="val_loss",
    mode="min",
    verbose=1
)

# Train the model using the dataset API
history = model.fit(
    train_dataset,
    epochs=100,
    batch_size=32,
    validation_data=test_dataset,
    callbacks=[checkpoint_callback]
)

# Load the best model
best_model = tf.keras.models.load_model("best_model.keras")

# Save it as the final model
best_model.save("final_best_model.keras")

# Initialize X_last with the last test sample
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
    X_input = ((full_data.iloc[start_idx:end_idx, :] - mean_X) / std_X).values.reshape(1, p, -1)
    
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
print("X_input shape:", X_input.shape)  # Should be (1, 50, 10)

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
plt.ylabel('Ankle Flexion')
plt.legend()
plt.show()
