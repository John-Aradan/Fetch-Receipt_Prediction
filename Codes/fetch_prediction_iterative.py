import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os

# Set the folders for saving/loading model parameters and data
param_folder = os.path.join(os.path.dirname(__file__), 'Model-Params')
data_folder = os.path.join(os.path.dirname(__file__), 'Data')

# Load the dataset containing monthly data (receipt counts and other features)
df = pd.read_csv(os.path.join(data_folder, "data_months.csv"))

# Load the trained model's parameters (weights and bias) from the earlier saved file
with open(os.path.join(param_folder, "model_parameters_early.pkl"), "rb") as f:
    theta, b = pickle.load(f)

# Prompt the user to input the month they want a prediction for (1-12)
user_month = int(input("Enter a month (1-12): "))

# List of continuous features to normalize later
continuous_columns = ['SMA_1', 'SMA_3', 'SMA_6', 'Lag_1', 'Lag_3', 'Lag_6']
# Calculate the mean and standard deviation of continuous columns for normalization
df_mean = df[continuous_columns].mean()
df_std = df[continuous_columns].std()

# Initialize the final prediction variable
final_prediction = 0

# Loop to generate predictions up to the month entered by the user
for i in range(1, user_month + 1):
    # Create a new row with features based on past data for prediction
    new_row = {
        'Month': i,  # Current month
        'Quarter': ((i - 1) // 3) + 1,  # Calculate the corresponding quarter
        # Simple Moving Averages (SMA) and Lag features based on past months
        'SMA_1': df['Receipt_Count'].iloc[-1:].mean(),  # Last receipt count (previous month)
        'SMA_3': df['Receipt_Count'].iloc[-3:].mean(),  # Mean of the last 3 months
        'SMA_6': df['Receipt_Count'].iloc[-6:].mean(),  # Mean of the last 6 months
        'Lag_1': df['Receipt_Count'].iloc[-1],  # Lag-1: Receipt count from last month
        'Lag_3': df['Receipt_Count'].iloc[-3],  # Lag-3: Receipt count from 3 months ago
        'Lag_6': df['Receipt_Count'].iloc[-6]   # Lag-6: Receipt count from 6 months ago
    }

    # Convert the new row into a DataFrame for easier manipulation
    df_row = pd.DataFrame([new_row])
    # Convert Month and Quarter into categorical variables
    df_row['Month'] = pd.Categorical(df_row['Month'], categories=range(1, 13))
    df_row['Quarter'] = pd.Categorical(df_row['Quarter'], categories=range(1, 5))
    
    # One-hot encode the Month and Quarter columns
    df_encoded = pd.get_dummies(df_row, columns=['Month', 'Quarter'], prefix=['Month', 'Quarter']).astype(int)

    # Normalize the continuous features (SMA and Lag columns)
    df_encoded[continuous_columns] = (df_encoded[continuous_columns] - df_mean) / df_std

    # Use the model (theta and bias) to predict receipt count for the current month
    prediction = df_encoded.dot(theta) + b
    print(i, " : ", prediction.iloc[0])

    # Add the predicted receipt count back to the dataset for future predictions
    new_row['Receipt_Count'] = prediction.iloc[0]
    df_new = pd.DataFrame([new_row])
    df = pd.concat([df, df_new], ignore_index=True)

    # Store the final prediction for the user's input month
    if i == user_month:
        final_prediction = prediction.iloc[0]
        print("Final Prediction: ", final_prediction)

# Ask the user how they would like to visualize the data
val = int(input("How would you like to visualize data: \n1. From Beginning (Jan 2021) \n2. Beginning of current year (2022) \n3. Last n months \n"))

# Create an empty DataFrame to store the visualization data
df_vis = pd.DataFrame()

# Prepare the visualization dataset with Month labels (2021/2022) and corresponding receipt counts
for i in range(len(df)):
        if i < 12:
            row = {'Month': f"2021-{df['Month'][i]}", 'Receipt_Count': df['Receipt_Count'][i]}
        else:
            row = {'Month': f"2022-{df['Month'][i]}", 'Receipt_Count': df['Receipt_Count'][i]}
        df_vis = pd.concat([df_vis, pd.DataFrame([row])], ignore_index=True)

# Check the user's input and filter the data accordingly
if val not in [1, 2, 3]:
    print("Invalid input")  # Handle invalid input
elif val == 1:
    print("visualizing all values")  # Show all data from Jan 2021
elif val == 2:
    df_vis = df_vis.iloc[12:, :]  # Show data starting from Jan 2022
elif val == 3:
    n = int(input("Enter number of months: "))  # Ask the user for the number of months to show
    df_vis = df_vis.iloc[-n:, :]  # Show the last 'n' months of data

# If there is data to visualize, plot a bar chart
if not df_vis.empty:
    plt.figure(figsize=(10, 6))
    plt.bar(df_vis['Month'].astype(str), df_vis['Receipt_Count'], color='skyblue')
    plt.title('Receipt Count Over Time')
    plt.xlabel('Month')
    plt.ylabel('Receipt Count')
    plt.xticks(rotation=45)  # Rotate the x-axis labels for better readability
    plt.grid(True)  # Add grid lines for clarity
    plt.show()
else:
    print("No data to visualize.")  # Print if there's no data to visualize
