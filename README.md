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

- 🔢 **Hybrid CNN-LSTM classifier** for detecting locomotion type from time-series sensor data.
- 🔮 **LSTM-based prediction model** to forecast ankle flexion during gait cycles.
- 🧹 **Data preprocessing pipeline** for cleaning and standardising raw sensor input.
- 🧪 **92.4% accuracy** across all locomotion types.

## 📊 Data & Tools

- **Input data:** Real-world sensor recordings from a wearable system (e.g., IMUs, force plates)
- **Prediction target:** Joint kinematics (e.g., ankle angles)
- **Tools used:**
  - 🐍 Python + PyTorch for modeling
  - 🦿 [OpenSim](https://opensim.stanford.edu/) for biomechanics simulation
  - 📈 MATLAB for signal processing & visualisation

## 📉 Results

- 92.4% classification accuracy across walking, running, stairs, and standing
- Robust prediction of ankle flexion over a full gait cycle (100 timesteps per step)

## 🚧 Future Work

- Real-time integration with prosthetic hardware
- Expanded dataset across multiple users and terrains
- Reinforcement learning for feedback-based control



---

*This work supports smarter, more human-aligned prosthetics through data-driven control systems.*
