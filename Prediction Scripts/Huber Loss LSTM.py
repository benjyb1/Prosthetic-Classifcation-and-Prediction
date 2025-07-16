import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import os
import pandas as pd
from glob import glob

csv_path = "/Users/benjyb/Documents/GitHub/Group8_healthcare/TestSubjectAB14/AB14_combined_data.csv"
output_json = "predictions_sim.json"

# Load single CSV file
data = pd.read_csv(csv_path)

# Separate input (X) and output (Y)
full_data = data.iloc[:, 2:-1]  # X (third column to second last column)
gon_data = data.iloc[:, -1:]   # Y (last column)

# Split into train/test sets (no data leakage)
X_train, X_test, Y_train, Y_test = train_test_split(full_data, gon_data, test_size=0.2, shuffle=False)

# Compute mean and std for normalization on the training set ONLY
mean_X, std_X = X_train.mean(), X_train.std()
mean_Y, std_Y = Y_train.mean().iloc[0], Y_train.std().iloc[0]

# Normalize both X and Y using training set statistics
X_train_norm = (X_train - mean_X) / std_X
Y_train_norm = (Y_train - mean_Y) / std_Y
X_test_norm = (X_test - mean_X) / std_X
Y_test_norm = (Y_test - mean_Y) / std_Y

# Define parameters
p = 25  # Past timesteps (input)
f = 13  # Future timesteps (output)

# Custom PyTorch Dataset
class GaitDataset(Dataset):
    def __init__(self, X, Y, p, f):
        self.X, self.Y = [], []
        for i in range(len(X) - p - f):
            self.X.append(X.iloc[i : i + p, :].values)
            self.Y.append(Y.iloc[i + p : i + p + f, :].values)
        self.X = np.array(self.X, dtype=np.float32)
        self.Y = np.array(self.Y, dtype=np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.Y[idx])

# Create datasets and data loaders
train_dataset = GaitDataset(X_train_norm, Y_train_norm, p, f)
test_dataset = GaitDataset(X_test_norm, Y_test_norm, p, f)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
print('Finished data preparation')
# Define the PyTorch model
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=f, num_layers=2, dropout_prob=0.3):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout_prob)
        self.dropout = nn.Dropout(dropout_prob)  # Added missing dropout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out[:, -1, :])  # Take the last output and apply dropout
        return self.fc(lstm_out).unsqueeze(-1)
print('About to start model')
# Initialize model, loss, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMModel(input_dim=X_train.shape[1]).to(device)

# Use Huber loss for training
criterion = nn.HuberLoss(delta=1.0)  # Delta controls the transition between L1 and L2 loss
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

# Training loop with loss tracking
epochs = 1
train_losses = []
val_losses = []

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for X_batch, Y_batch in train_loader:
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
        optimizer.zero_grad()
        Y_pred = model(X_batch)
        loss = criterion(Y_pred, Y_batch)  # Huber loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
        optimizer.step()
        epoch_loss += loss.item()

    avg_train_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    # Validation Loss Calculation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, Y_batch in test_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            Y_pred = model(X_batch)
            loss = criterion(Y_pred, Y_batch)
            val_loss += loss.item()
    
    avg_val_loss = val_loss / len(test_loader)
    val_losses.append(avg_val_loss)
    
    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

# Plot training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs + 1), train_losses, marker='o', linestyle='-', color='b', label='Training Loss')
plt.plot(range(1, epochs + 1), val_losses, marker='o', linestyle='-', color='r', label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

# Save the model
torch.save(model.state_dict(), "/Users/benjyb/best_model_huber.pth")

# Load the best model for inference
model.load_state_dict(torch.load("best_model_huber.pth"))
model.eval()

# Make predictions
predict_window = f * 20  # Total prediction window in ms
slide_step = f  # Slide every 10ms
total_slides = predict_window // slide_step

# Select a random starting index that is a multiple of 100
valid_start_indices = [i for i in range(0, len(data) - (p + f + predict_window), 100)]
if not valid_start_indices:
    raise ValueError("Not enough data to select a valid starting index.")

start_idx = random.choice(valid_start_indices)

end_idx = start_idx + p

predictions, true_values = [], []

with torch.no_grad():
    for _ in range(total_slides):
        if end_idx + f > len(data):
            break

        # Normalize input using training mean and std
        X_input = ((full_data.iloc[start_idx:end_idx, :] - mean_X) / std_X).values.reshape(1, p, -1)
        X_input = torch.tensor(X_input, dtype=torch.float32).to(device)

        # Extract true values
        Y_true = gon_data.iloc[end_idx:end_idx + f, :].values

        # Make prediction
        Y_pred = model(X_input).cpu().numpy()

        # Denormalize predictions
        Y_pred = Y_pred * std_Y + mean_Y

        predictions.append(Y_pred.squeeze())
        true_values.append(Y_true.squeeze())

        start_idx += slide_step
        end_idx += slide_step

# Convert lists to numpy arrays
predictions = np.concatenate(predictions, axis=0)
true_values = np.concatenate(true_values, axis=0)

# Compute test loss (Huber and MSE)
test_mse = np.mean((predictions - true_values) ** 2)
print(f"Test Loss (MSE): {test_mse:.4f}")

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(np.arange(predict_window), true_values, label="True", linestyle="-", color="b")
plt.plot(np.arange(predict_window), predictions, label="Predicted", linestyle="--", color="r")
plt.title("Huber Prediction of Ankle Flexion 10ms into the Future")
plt.xlabel("Time Step (ms)")
plt.ylabel("Ankle Flexion")
plt.legend()
plt.show()