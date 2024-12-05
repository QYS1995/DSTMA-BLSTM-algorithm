Here’s a detailed `README.md` for your project, summarizing the steps in each Python script and how they interconnect, including information about the pre-trained model. You can modify the parts related to your specific dataset or any other relevant details.

---

# Pollution Prediction Model using Multi-Task Learning and Attention Mechanisms

This project aims to predict the concentration levels of various air pollutants based on meteorological and traffic-related features using deep learning models. The workflow consists of three primary Python scripts: **first_part.py**, **second_part.py**, and **three_part.py**.

## Overview

1. **first_part.py**: Trains a model using the **DynamicSharedTaskSpecificAttention** mechanism and saves the best-performing model to a checkpoint (`best_model_checkpoint.h5`).
2. **second_part.py**: Prepares the data by scaling and creating sequences for training.
3. **three_part.py**: Loads the pre-trained model, makes predictions on new data, and performs error analysis and visualization.

### Project Structure

```plaintext
PollutionPrediction/
│
├── data/
│   └── Data_matrix-1.txt          # Dataset containing pollutant and meteorological features
│
├── scripts/
│   ├── first_part.py              # Model training script
│   ├── second_part.py             # Data preprocessing and sequence generation
│   └── three_part.py              # Prediction, evaluation, and residual analysis
│
├── best_model_checkpoint.h5      # Pre-trained best model saved during training
└── README.md                     # This file
```

### Dependencies

Make sure to install the required libraries before running the scripts:

```bash
pip install -r requirements.txt
```

Where `requirements.txt` contains the following:

```
pandas
numpy
matplotlib
scikit-learn
tensorflow
```

## Workflow Overview

### **1. first_part.py**: Model Training

This script trains a model to predict multiple air pollutants based on meteorological and traffic features. The model uses a custom attention mechanism called **DynamicSharedTaskSpecificAttention**, which combines shared and task-specific attention heads to improve performance across multiple prediction tasks.

#### Key Steps:
- **Data Preprocessing**: The data is loaded, cleaned, and normalized using `MinMaxScaler` to scale the features to the range [0, 1].
- **Sequence Creation**: The data is split into sequences, which are used as input for the model.
- **Model Architecture**: A custom model is built using LSTM layers and a **DynamicSharedTaskSpecificAttention** layer.
- **Training**: The model is trained on the sequences, and the best model is saved to `best_model_checkpoint.h5`.

#### Running the Script:

```bash
python scripts/first_part.py
```

Once the model is trained, the best model will be saved as `best_model_checkpoint.h5` in the project directory.

> **Note:** You can also download the pre-trained model directly from [GitHub Releases](https://github.com/yourusername/PollutionPrediction/releases/tag/v1.0) if you prefer to skip the training step.

### **2. second_part.py**: Data Preprocessing

This script handles the preprocessing of the data by scaling the pollutant and meteorological features and creating input-output sequences. It prepares the data for training or prediction by organizing it into time-series format.

#### Key Steps:
- **Loading the Data**: The data is read from a CSV file (`Data_matrix-1.txt`).
- **Scaling**: The data is scaled using `MinMaxScaler`.
- **Sequence Creation**: The scaled data is split into input-output sequences, which will be used by the model for training and prediction.

#### Running the Script:

```bash
python scripts/second_part.py
```

### **3. three_part.py**: Prediction and Evaluation

After the model has been trained, this script performs predictions on new data, compares the predicted values with actual values, and visualizes the results using **residual analysis**. 

#### Key Steps:
- **Loading the Pre-trained Model**: The best model is loaded from the checkpoint (`best_model_checkpoint.h5`).
- **Making Predictions**: The trained model is used to predict pollutant concentrations.
- **Inverse Transformation**: The scaled predictions and actual values are inverse-transformed to the original units of the pollutants.
- **Residual Analysis**: Residuals (errors) are calculated, and plots are generated to visualize the model’s performance over time.

#### Running the Script:

```bash
python scripts/three_part.py
```

This script will output:
- **Predicted vs Actual Plots**: For each pollutant, a plot comparing the true and predicted values will be displayed.
- **Residual Plots**: Plots showing the residuals over time to analyze prediction errors.

### **Pre-trained Model**:

If you do not wish to train the model yourself, you can directly download the best pre-trained model from the following GitHub Release:

- [Download the best pre-trained model](https://github.com/yourusername/PollutionPrediction/releases/tag/v1.0)

Once downloaded, place the `best_model_checkpoint.h5` file in the same directory as `three_part.py` for easy access.

### Data Format

The input data (`Data_matrix-1.txt`) should have the following columns:

```
NO(ppb), NO2(ppb), CO2(ppm), CO(ppm), CH4(ppm), Pressure, Temperature, Humidity, WindSpeed, WindDirections, TrafficCounts, DieselCounts, GasselCounts
```

Each row represents a time step, and the data columns represent:
- **Pollutants**: NO, NO2, CO2, CO, CH4.
- **Meteorological Features**: Pressure, Temperature, Humidity, WindSpeed, WindDirections.
- **Traffic Features**: TrafficCounts, DieselCounts, GasselCounts.

### Example Output

#### Predicted vs Actual Plot

For each pollutant (e.g., NO, NO2), a plot will show the actual vs predicted concentration values over time. This helps to evaluate the model’s accuracy.

#### Residuals Plot

Residual plots will display the errors (difference between actual and predicted values) for each pollutant. These plots are useful for detecting biases or trends in the model's predictions.

## Additional Notes

- **Model Performance**: If the model doesn’t perform well, consider fine-tuning the architecture or adjusting hyperparameters.
- **Data Quality**: Ensure that the input data is of high quality, with no missing values or outliers, as this can significantly affect the model's performance.
- **GPU Usage**: The model is designed to run on GPUs. If you're using a machine without a GPU, it will run on the CPU, but it may be slower.

---

Feel free to adjust the links, paths, and details related to your specific setup or dataset. This README will guide users through the entire process from data preprocessing, model training, to prediction and evaluation.
