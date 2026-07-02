# 🦿 Prosthetic Classification and Prediction

A machine learning system trained on real-world leg sensor data to predict optimal prosthetic movement, improving gait alignment and mobility for users with lower-limb prosthetics.

![Demo of prediction](GIF.gif)
## 🧠 What It Does

This project tackles the problem of intelligent prosthetic control by:
1. **Classifying** the current type of locomotion (e.g., walking, running, stair ascent/descent) using leg sensor data.
2. **Predicting** how a prosthetic leg should flex or move in response to the user’s gait, providing a smooth, natural experience.

## 🎯 Motivation

Machine learning in prosthetic control is an underdeveloped yet critical area in healthcare and robotics. Existing prosthetics often lack adaptive intelligence, leading to discomfort or instability.

This project draws inspiration from the growing interest in prosthetics research — including work done at the **Bristol Robotics Laboratory** and the University of Bristol’s **Intelligent Systems Laboratory** — to explore how real-time prediction can enhance prosthetic responsiveness.

## 🔍 Key Features

- 🔢 **1D CNN classifier** (3 conv blocks, dropout regularisation, stratified split) for detecting locomotion type from a 50ms window of 9-channel sensor data.
- 🔮 **Sequence-to-sequence prediction models** for ankle flexion during gait — three architectures compared: a dilated-convolution CNN, a plain LSTM, and an LSTM trained with Huber loss for outlier robustness.
- 🧹 **Data preprocessing pipeline** — per-feature Z-score normalisation fit on train only, temporal (non-shuffled) train/test split to respect gait sequencing.
- 🧪 **92.4% classification accuracy** across all locomotion types, on a held-out temporal split.

## 📊 Data & Tools

- **Input data:** ~215,000 labelled 50-timestep windows of 9-channel sensor data (IMU/EMG) from a wearable system, drawn from multiple subject recordings.
- **Prediction target:** ankle sagittal-plane flexion angle, forecast 10ms ahead from the preceding 50ms window.
- **Tools used:**
  - 🐍 Python — PyTorch for the CNN prediction models, TensorFlow/Keras for the classifier and LSTM variants, scikit-learn for splits/scaling/metrics.
  - 🦿 [OpenSim](https://opensim.stanford.edu/) referenced for biomechanics context.

## 📉 Results

- 92.4% classification accuracy across walking, running, stairs, and standing.
- Ankle-flexion prediction evaluated with a sliding 10ms-step window over a 200ms test segment, comparing CNN, LSTM, and Huber-loss LSTM on MSE.

## 🚧 Honest limitations

This is exploratory, notebook-driven research code, not a production pipeline — worth being upfront about:

- Trained and evaluated within the same subject(s); cross-subject generalisation is untested.
- Single train/test split, no k-fold cross-validation on the classifier or CNN predictor.
- No systematic comparison across the three prediction architectures beyond the metrics each notebook reports individually.

## 🚧 Future Work

- Cross-subject validation, to see whether the model generalises beyond the people it was trained on.
- Real-time integration with prosthetic hardware.
- Expanded dataset across multiple users and terrains.



---

*This work supports smarter, more human-aligned prosthetics through data-driven control systems.*
