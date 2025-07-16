import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Reshape
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
# Set TensorFlow to use only CPU
tf.config.set_visible_devices([], 'GPU')  # This hides the GPU from TensorFlow

# Load the dataset
dataset_path = '/Users/benjyb/PycharmProjects/PythonProject1/LSTM Models/Temporary data/combined_dataAB06.csv'
data = pd.read_csv(dataset_path)

# Assume first column is gon_data (output), remaining are input features
gon_data = data.iloc[:, :1]

# Normalize input and output separately
full_data = (data - data.mean()) / data.std()
gon_data = (gon_data - gon_data.mean()) / gon_data.std()
# Normalize input and output separately (Apply to X_data and Y_data the same way)
# Compute mean and std for normalization
mean_X, std_X = data.mean(), data.std()
mean_Y, std_Y = gon_data.mean().iloc[0], gon_data.std().iloc[0]

X_norm = (full_data - mean_X) / std_X
Y_norm = (gon_data - mean_Y) / std_Y

# Define parameters
p = 50  # Past timesteps (input)
f = 10  # Future timesteps (output)

# Prepare training sequences from the full dataset
X_data, Y_data = [], []
for i in range(len(full_data) - p - f):
    X_data.append(full_data.iloc[i:i+p, :].values)  # Input: past 50ms
    Y_data.append(gon_data.iloc[i+p:i+p+f, :].values)  # Output: next 10ms

X_data = np.array(X_data)  # Shape: (samples, 50, 40)
Y_data = np.array(Y_data)  # Shape: (samples, 10, 1)

# Split into train/test sets
X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.2, shuffle=False)

# Define model
model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(50, 40), return_sequences=False, stateful=False))
model.add(Dense(10))  # Output: 10 time steps (1 feature each)
model.add(Reshape((10, 1)))

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

# Train the model and save the best one
history = model.fit(
    X_train, Y_train,
    batch_size=16, epochs=1,
    validation_data=(X_test, Y_test),
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
