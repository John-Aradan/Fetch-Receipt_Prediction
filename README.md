# Fetch Rewards Take-home Exercise - Machine Learning Engineer

This repository contains a machine learning model to predict the number of scanned receipts in the Fetch Rewards app for each month of 2022 based on historical data from 2021. The solution includes data preprocessing, feature engineering, model training, and visualization of results.

## User Interaction App

### Fetch Rewards Receipt Count Predictor

Welcome to the **Fetch Rewards Receipt Count Predictor**, an interactive app that allows you to forecast receipt counts for a given month based on historical data. This tool predicts receipt submissions and offers different visualization options to help you understand trends in the data.

[**Click here to try the app on Hugging Face**](https://huggingface.co/spaces/John-Aradan/Fetch-Receipt_Count_Predictor)

### Features:
1. **Predict Receipt Counts**: Input a month (from January to December) to forecast the number of receipts.
2. **Flexible Data Visualization**:
   - View the trend from January 2021.
   - Focus on data from the beginning of 2022.
   - Visualize the last N months for a quick overview.

### How It Works:
- The app uses a linear regression model, trained on rolling averages and lag values of receipt counts, to generate predictions.
- Predictions and visualizations are updated dynamically as you interact with the app.

### Usage Instructions:
1. Select a month (1 for January, 12 for December) to predict future receipt counts.
2. Choose how you want the data visualized:
   - From the beginning (January 2021)
   - Beginning of the current year (2022)
   - The last N months for a custom view
3. If you select "Last N months," input the desired number of months for a custom view.
4. Click **Submit** to generate the prediction and the receipt count chart.

### Screenshots

(*Placeholder for screenshots*)

---

## Table of Contents

- [Overview](#overview)
- [Data](#data)
- [File Descriptions](#file-descriptions)
- [Functions Overview](#functions-overview)

## Project Structure

Codes/  
│  
├── Data/  
│   ├── data_daily.csv  
│   ├── data_days.csv  
│   └── data_months.csv  
│  
├── Model-Params/  
│   ├── model_parameters_all.pkl  
│   └── model_parameters_early.pkl  
│  
├── feature_engineering.py  
├── model_building.py  
└── fetch_prediction_iterative.py  


## Overview

The goal of this exercise is to develop an algorithm that predicts the number of scanned receipts based on daily data from 2021. The solution implements a gradient descent algorithm from scratch to train a linear regression model without using high-level libraries like scikit-learn. 

## Data

The dataset consists of daily receipt counts, which is used to create aggregated monthly data for training the model. The following files are included in the `Data` folder:

- `data_daily.csv`: Daily receipts data for the year 2021.
- `data_days.csv`: Processed daily data with additional features.
- `data_months.csv`: Aggregated monthly data with engineered features.

## File Descriptions

### `feature_engineering.py`

This script processes the daily receipts data and creates additional features necessary for the model. Key operations include:

- Loading the daily receipts data.
- Extracting the month and quarter from the date.
- Calculating moving averages (SMA) and lag features.
- Handling missing values by filling them with the mean.
- Saving the processed data to CSV files.

### `model_building.py`

This script is responsible for training the linear regression model using gradient descent. Key functions include:

- **gradient_descent(theta_now, b_now, X, y, L)**: 
  - Performs gradient descent optimization to update model weights (theta) and bias (b).
  - Parameters:
    - `theta_now`: Current weights of the model.
    - `b_now`: Current bias of the model.
    - `X`: Features used for training.
    - `y`: Target variable (receipt count).
    - `L`: Learning rate.

The script initializes model parameters, runs the gradient descent process, and saves the trained model parameters to a file.

### `fetch_prediction_iterative.py`

This script handles user interaction for making predictions and visualizing results. Key features include:

- Taking user input for the desired month and calculating predictions based on previously trained parameters.
- Generating a DataFrame for visualization based on user-selected time frames.
- Visualizing the receipt count using a bar chart.

## Functions Overview

### 1. `feature_engineering.py`

- **Loading Data**: Loads daily receipt data and converts date columns.
- **Feature Engineering**:
  - Extracts `Month`, `Quarter`, and `Day_of_Week`.
  - Computes moving averages (SMA) and lag features.
  - Handles missing values by replacing them with the mean.
- **Saving Processed Data**: Saves the processed DataFrame into CSV files.

### 2. `model_building.py`

- **gradient_descent**:
  - Computes gradients for model weights and bias.
  - Updates weights and bias using the learning rate.
- **Training Process**: 
  - Initializes weights and bias.
  - Runs gradient descent for a specified number of epochs or until a certain error threshold is met.
  - Saves trained parameters for later use.

### 3. `fetch_prediction_iterative.py`

- **User Input**: 
  - Prompts the user to enter a month for prediction.
- **Prediction**: 
  - Uses the last month's data and features to predict receipt counts for the specified months.
- **Visualization**: 
  - Allows users to visualize receipt counts over various time frames.

## Documentations

I have included comments in the python code that would help someone who have not worked with python before understand how the code is being run and results are produced.
