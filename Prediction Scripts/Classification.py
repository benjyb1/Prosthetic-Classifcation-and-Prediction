import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib

class LocomotionClassifier:
    def __init__(self, window_size=200, stride=100):
        """
        Initialize the locomotion classifier
        
        Args:
            window_size (int): Size of the time window for classification
            stride (int): Stride between windows
        """
        self.window_size = window_size
        self.stride = stride
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.model = None
        self.classes = None
        
    def load_multiple_files(self, folder_path):
        """
        Load multiple CSV files from a folder, each representing a different locomotion type
        
        Args:
            folder_path (str): Path to folder containing CSV files
            
        Returns:
            tuple: Combined features (X) and labels (y)
        """
        # List all CSV files
        csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
        
        if not csv_files:
            print("No CSV files found in the directory.")
            return None, None
        
        # Display available files
        print("Available CSV files:")
        for i, file in enumerate(csv_files):
            print(f"{i+1}. {file}")
        
        # Let user select files and assign locomotion types
        selected_files = []
        
        print("\nSelect files to include in the training set.")
        print("For each file, you'll need to specify the locomotion type (e.g., walking, running).")
        print("Enter 'done' when finished selecting files.")
        
        while True:
            file_num = input("\nEnter file number to add (or 'done' to finish): ")
            
            if file_num.lower() == 'done':
                break
                
            try:
                file_idx = int(file_num) - 1
                if 0 <= file_idx < len(csv_files):
                    file_name = csv_files[file_idx]
                    
                    # Check if file already selected
                    if any(file_name == f[0] for f in selected_files):
                        print(f"File '{file_name}' already selected. Choose another file.")
                        continue
                    
                    # Get locomotion type
                    loco_type = input(f"Enter locomotion type for {file_name}: ").strip()
                    if not loco_type:
                        print("Locomotion type cannot be empty. Try again.")
                        continue
                        
                    selected_files.append((file_name, loco_type))
                    print(f"Added {file_name} as '{loco_type}'")
                else:
                    print("Invalid file number. Please try again.")
            except ValueError:
                print("Please enter a valid number or 'done'.")
        
        if not selected_files:
            print("No files selected for training.")
            return None, None
            
        print(f"\nSelected {len(selected_files)} files for training:")
        for file_name, loco_type in selected_files:
            print(f"- {file_name}: {loco_type}")
            
        # Process all selected files
        all_features = []
        all_labels = []
        
        for file_name, loco_type in selected_files:
            file_path = os.path.join(folder_path, file_name)
            print(f"\nProcessing {file_name} (Type: {loco_type})...")
            
            try:
                # Load the CSV file
                df = pd.read_csv(file_path)
                
                # Print column info for first file only
                if file_name == selected_files[0][0]:
                    print("DataFrame Columns:")
                    print(df.columns)
                    print("\nColumn Types:")
                    print(df.dtypes)
                
                # Get features (all columns)
                X = df.values
                
                # Handle NaN values in features
                if np.isnan(X).any():
                    print(f"Replacing {np.isnan(X).sum()} NaN values with column means.")
                    col_means = np.nanmean(X, axis=0)
                    inds = np.where(np.isnan(X))
                    X[inds] = np.take(col_means, inds[1])
                
                # Create labels array (all rows labeled with the locomotion type)
                y = np.array([loco_type] * len(df))
                
                all_features.append(X)
                all_labels.append(y)
                
                print(f"Added {len(df)} samples of type '{loco_type}'")
                
            except Exception as e:
                print(f"Error processing {file_name}: {e}")
                continue
        
        if not all_features:
            print("No valid data loaded from any files.")
            return None, None
            
        # Combine all data
        X_combined = np.vstack(all_features)
        y_combined = np.concatenate(all_labels)
        
        print(f"\nCombined dataset:")
        print(f"- Total samples: {len(X_combined)}")
        print(f"- Feature dimensions: {X_combined.shape[1]}")
        print(f"- Unique locomotion types: {np.unique(y_combined)}")
        
        return X_combined, y_combined

    def load_single_file(self, file_path):
        """
        Load a single CSV file for prediction
        
        Args:
            file_path (str): Path to CSV file
            
        Returns:
            np.array: Features matrix
        """
        try:
            # Load the CSV file
            df = pd.read_csv(file_path)
            
            # Get features (all columns)
            X = df.values
            
            # Handle NaN values in features
            if np.isnan(X).any():
                print(f"Replacing {np.isnan(X).sum()} NaN values with column means.")
                col_means = np.nanmean(X, axis=0)
                inds = np.where(np.isnan(X))
                X[inds] = np.take(col_means, inds[1])
            
            print(f"\nLoaded data from {file_path}")
            print(f"Data shape: {X.shape}")
            
            return X
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None

    def preprocess_data(self, X, y=None, training=True):
        """
        Preprocess the data
        
        Args:
            X (np.array): Input features
            y (np.array, optional): Input labels (required for training)
            training (bool): Whether this is training data or prediction data
            
        Returns:
            tuple or array: Processed X (and y if training)
        """
        if X is None:
            raise ValueError("No valid data loaded.")
        
        # Convert X to float
        X = X.astype(float)
        
        # Standardize features
        if training:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        # If labels are provided (for training)
        if y is not None and training:
            # Encode labels
            y_encoded = self.label_encoder.fit_transform(y)
            
            # Update classes
            self.classes = self.label_encoder.classes_
            
            return X_scaled, y_encoded
        
        return X_scaled

    def create_model(self, input_shape, num_classes):
        """
        Create CNN model for classification
        
        Args:
            input_shape (tuple): Shape of input data
            num_classes (int): Number of classification classes
        
        Returns:
            tf.keras.Model: Compiled CNN model
        """
        model = Sequential([
            Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape, padding='same'),
            MaxPooling1D(pool_size=2),
            Dropout(0.25),
            
            Conv1D(128, kernel_size=3, activation='relu', padding='same'),
            MaxPooling1D(pool_size=2),
            Dropout(0.25),
            
            Conv1D(256, kernel_size=3, activation='relu', padding='same'),
            MaxPooling1D(pool_size=2),
            Dropout(0.25),
            
            Flatten(),
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model

    def train(self, X, y, test_size=0.2, val_size=0.2, epochs=50):
        """
        Train the locomotion classification model
        
        Args:
            X (np.array): Input features
            y (np.array): Input labels
            test_size (float): Proportion of data to use for testing
            val_size (float): Proportion of training data to use for validation
            epochs (int): Number of training epochs
        
        Returns:
            History of model training
        """
        # Suppress specific warnings
        warnings.filterwarnings('ignore', category=UserWarning)
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        
        # Preprocess data
        X_processed, y_processed = self.preprocess_data(X, y, training=True)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y_processed, test_size=test_size, random_state=42, stratify=y_processed
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size, random_state=42, stratify=y_train
        )
        
        # Create model
        num_classes = len(np.unique(y_processed))
        print(f"\nCreating model for {num_classes} locomotion types")
        self.model = self.create_model(
            input_shape=(X_train.shape[1], 1),  # Adjust input shape for 1D convolution 
            num_classes=num_classes
        )
        
        # Callbacks
        early_stop = EarlyStopping(
            monitor='val_loss', 
            patience=10, 
            restore_best_weights=True
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.2, 
            patience=5, 
            min_lr=0.00001
        )
        
        # Train model
        history = self.model.fit(
            X_train.reshape(X_train.shape[0], X_train.shape[1], 1),  # Reshape for 1D convolution
            y_train,
            epochs=epochs,
            batch_size=32,
            validation_data=(
                X_val.reshape(X_val.shape[0], X_val.shape[1], 1), 
                y_val
            ),
            callbacks=[early_stop, reduce_lr]
        )
        
        # Evaluate model
        test_loss, test_accuracy = self.model.evaluate(
            X_test.reshape(X_test.shape[0], X_test.shape[1], 1), 
            y_test
        )
        print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
        
        # Predictions for detailed metrics
        y_pred = self.model.predict(X_test.reshape(X_test.shape[0], X_test.shape[1], 1))
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Classification report
        print(classification_report(
            y_test, 
            y_pred_classes, 
            target_names=self.classes,
            zero_division=0
        ))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred_classes)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', 
                    xticklabels=self.classes, 
                    yticklabels=self.classes)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.show()
        
        return history
    
    def save_model(self, model_dir='model'):
        """
        Save the trained model and preprocessing components
        
        Args:
            model_dir (str): Directory to save model files
        """
        if self.model is None:
            print("No trained model to save.")
            return
            
        # Create directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(model_dir, 'locomotion_model.h5')
        self.model.save(model_path)
        
        # Save scaler
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        joblib.dump(self.scaler, scaler_path)
        
        # Save label encoder
        encoder_path = os.path.join(model_dir, 'label_encoder.pkl')
        joblib.dump(self.label_encoder, encoder_path)
        
        print(f"Model and preprocessing components saved to {model_dir}")
    
    def load_model(self, model_dir='model'):
        """
        Load a trained model and preprocessing components
        
        Args:
            model_dir (str): Directory containing model files
        """
        model_path = os.path.join(model_dir, 'locomotion_model.h5')
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        encoder_path = os.path.join(model_dir, 'label_encoder.pkl')
        
        if not all(os.path.exists(p) for p in [model_path, scaler_path, encoder_path]):
            print("Missing model files. Train a model first.")
            return False
            
        try:
            # Load model
            self.model = load_model(model_path)
            
            # Load scaler
            self.scaler = joblib.load(scaler_path)
            
            # Load label encoder
            self.label_encoder = joblib.load(encoder_path)
            
            # Get classes
            self.classes = self.label_encoder.classes_
            
            print(f"Model loaded successfully. Locomotion types: {self.classes}")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def predict(self, X):
        """
        Predict locomotion type for new data
        
        Args:
            X (np.array): Input features
            
        Returns:
            Predicted locomotion types and probabilities
        """
        if self.model is None:
            print("No trained model. Train or load a model first.")
            return None
            
        # Preprocess data (no labels, not training)
        X_processed = self.preprocess_data(X, training=False)
        
        # Reshape for 1D convolution
        X_reshaped = X_processed.reshape(X_processed.shape[0], X_processed.shape[1], 1)
        
        # Get predictions
        y_pred = self.model.predict(X_reshaped)
        
        # Get predicted classes
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Convert to class names
        predicted_locomotion = self.label_encoder.inverse_transform(y_pred_classes)
        
        # Get class counts
        unique_predictions, counts = np.unique(predicted_locomotion, return_counts=True)
        percentages = counts / len(predicted_locomotion) * 100
        
        # Create results summary
        results = {
            'predicted_classes': predicted_locomotion,
            'probabilities': y_pred,
            'summary': dict(zip(unique_predictions, percentages))
        }
        
        return results

def train_mode(folder_path):
    """Run the classifier in training mode"""
    classifier = LocomotionClassifier()
    
    # Load multiple files with different locomotion types
    X, y = classifier.load_multiple_files(folder_path)
    
    if X is not None and y is not None:
        # Ask for number of epochs
        while True:
            try:
                epochs = int(input("Enter number of training epochs (default: 50): ") or "50")
                if epochs > 0:
                    break
                else:
                    print("Number of epochs must be positive.")
            except ValueError:
                print("Invalid input. Please enter a number.")
        
        # Train the model
        classifier.train(X, y, epochs=epochs)
        
        # Save the model
        save_model = input("Save the trained model? (y/n): ")
        if save_model.lower() == 'y':
            model_dir = input("Enter model directory (default: 'model'): ") or "model"
            classifier.save_model(model_dir)
    else:
        print("Could not load valid training data.")

def predict_mode(folder_path):
    """Run the classifier in prediction mode"""
    # First load a trained model
    classifier = LocomotionClassifier()
    
    model_dir = input("Enter model directory (default: 'model'): ") or "model"
    if not classifier.load_model(model_dir):
        print("Failed to load model. Exiting prediction mode.")
        return
    
    # List all CSV files in the directory
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    
    if not csv_files:
        print("No CSV files found in the directory.")
        return
    
    # Display available files
    print("Available CSV files for prediction:")
    for i, file in enumerate(csv_files):
        print(f"{i + 1}. {file}")
    
    # Select a file
    while True:
        try:
            choice = int(input("Enter the number of the file to classify: "))
            if 1 <= choice <= len(csv_files):
                break
            else:
                print("Invalid choice. Please enter a number from the list.")
        except ValueError:
            print("Invalid input. Please enter a number.")
    
    # Full path to selected file
    selected_file = os.path.join(folder_path, csv_files[choice - 1])
    
    # Load data for prediction
    X = classifier.load_single_file(selected_file)
    
    if X is not None:
        # Make predictions
        print("\nAnalyzing locomotion type...")
        results = classifier.predict(X)
        
        if results:
            print("\nPrediction Results:")
            print("-" * 40)
            print(f"File: {csv_files[choice - 1]}")
            print("-" * 40)
            print("Locomotion Type Breakdown:")
            
            # Sort by percentage (highest first)
            sorted_results = sorted(
                results['summary'].items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            for loco_type, percentage in sorted_results:
                print(f"- {loco_type}: {percentage:.2f}%")
            
            print("-" * 40)
            
            # Determine overall classification
            majority_class = sorted_results[0][0]
            majority_percentage = sorted_results[0][1]
            
            print(f"Overall Classification: {majority_class} ({majority_percentage:.2f}%)")
            
            if majority_percentage < 70:
                print("\nNote: Classification confidence is below 70%. This may indicate:")
                print("- Mixed locomotion types in the file")
                print("- A new locomotion type not in the training set")
                print("- Noisy or unusual data")
    else:
        print("Could not load valid data for prediction.")

def main():
    # Get folder path
    folder_path = input("Enter the folder path containing CSV files: ")
    
    if not os.path.isdir(folder_path):
        print(f"Error: {folder_path} is not a valid directory.")
        return
    
    # Ask for operation mode
    print("\nSelect operation mode:")
    print("1. Train a new locomotion classifier")
    print("2. Classify a new dataset")
    
    while True:
        try:
            mode = int(input("Enter mode (1 or 2): "))
            if mode in [1, 2]:
                break
            else:
                print("Invalid choice. Please enter 1 or 2.")
        except ValueError:
            print("Invalid input. Please enter a number.")
    
    if mode == 1:
        train_mode(folder_path)
    else:
        predict_mode(folder_path)

if __name__ == "__main__":
    main()