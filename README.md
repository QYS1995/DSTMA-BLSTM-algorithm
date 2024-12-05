Multi-Task Time Series Prediction with Dynamic Attention Mechanism
This repository contains Python code for predicting multiple pollutant concentrations (e.g., NO, NO2, CO2, CO, CH4) using a multi-task deep learning model based on LSTM (Long Short-Term Memory) layers and a custom Dynamic Shared and Task-Specific Attention Mechanism.

The project includes:

Data preprocessing and sequence generation for time series inputs.
Model architecture definition and hyperparameter tuning using .Keras Tuner
Training, validation, and testing of the model with real-world pollutant and meteorological data.
Evaluation, visualization of predictions, and feature importance analysis.

Features
Dynamic Attention Mechanism:
Combines shared and task-specific attention for multi-task learning.
Automatically assigns dynamic weights to each task.
Multi-Task Learning:
Simultaneously predicts multiple pollutant concentrations.
Hyperparameter Tuning:
Uses for Bayesian optimization of hyperparameters.Keras Tuner
Comprehensive Analysis:
Generates error metrics (MSE, RMSE, MAE, R²) for each pollutant.
Visualizes predictions, residuals, and feature importances.


File Overview
first_part.py
This script handles(data_preprocessing_and_model_training):

Data Loading and Preprocessing:
Normalizes input features (e.g., meteorological and traffic-related data) and target pollutant concentrations.
Generates time series sequences for model training and prediction.
Model Architecture and Hyperparameter Tuning:
Defines a multi-task model with LSTM layers and a dynamic attention mechanism.
Optimizes hyperparameters (e.g., LSTM units, learning rate) using Bayesian optimization.
Model Training and Saving:
Trains the model with early stopping and saves the best-performing model.

second_part.py
This script provides(feature_importance_analysis):

Feature Importance Calculation:
Uses permutation importance to evaluate the contribution of meteorological and traffic features to the model’s predictions.
Visualization:
Generates bar plots showing feature importances for each pollutant and overall average importance.

three_part.py
This script performs(model_evaluation_and_analysis):

Model Evaluation:
Loads the pre-trained model and evaluates it on the test dataset.
Outputs predictions for multiple pollutants and calculates error metrics.
Residual and Prediction Visualization:
Generates plots comparing predicted and actual values for each pollutant.
Creates residual plots to analyze prediction errors.
