import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os

param_folder = os.path.join(os.path.dirname(__file__), 'Model-Params')
data_folder = os.path.join(os.path.dirname(__file__), 'Data')

df = pd.read_csv(os.path.join(data_folder, "data_months.csv"))

with open(os.path.join(param_folder, "model_parameters_early.pkl"), "rb") as f:
    theta, b = pickle.load(f)

user_month = int(input("Enter a month (1-12): "))

continuous_columns = ['SMA_1', 'SMA_3', 'SMA_6', 'Lag_1', 'Lag_3', 'Lag_6']
df_mean = df[continuous_columns].mean()
df_std = df[continuous_columns].std()

final_prediction = 0

for i in range(1,user_month+1):
  new_row = {
    'Month': i,
    'Quarter': ((i-1)//3)+1,
    'SMA_1': df['Receipt_Count'].iloc[-1:].mean(),  # SMA_1 = the last receipt count value (12th month)
    'SMA_3': df['Receipt_Count'].iloc[-3:].mean(),  # SMA_3 = the mean of the last 3 months (10th, 11th, 12th)
    'SMA_6': df['Receipt_Count'].iloc[-6:].mean(),  # SMA_6 = the mean of the last 6 months (7th-12th)
    'Lag_1': df['Receipt_Count'].iloc[-1],  # Lag_1 = the receipt count from month 12
    'Lag_3': df['Receipt_Count'].iloc[-3],  # Lag_3 = the receipt count from month 10
    'Lag_6': df['Receipt_Count'].iloc[-6]   # Lag_6 = the receipt count from month 7
  }

  df_row = pd.DataFrame([new_row])
  df_row['Month'] = pd.Categorical(df_row['Month'], categories=range(1, 13))
  df_row['Quarter'] = pd.Categorical(df_row['Quarter'], categories=range(1, 5))
  df_encoded = pd.get_dummies(df_row, columns=['Month', 'Quarter'], prefix=['Month', 'Quarter']).astype(int)

  df_encoded[continuous_columns] = (df_encoded[continuous_columns] - df_mean) / df_std

  prediction = df_encoded.dot(theta) + b
  print(i, " : ", prediction.iloc[0])

  new_row['Receipt_Count'] = prediction.iloc[0]
  df_new = pd.DataFrame([new_row])
  df = pd.concat([df, df_new], ignore_index=True)

  if i == user_month:
    final_prediction = prediction.iloc[0]
    print("Final Prediction: ", final_prediction)

val = int(input("How would you like to visualize data: \n1. From Beginning (Jan 2021) \n2. Beginning of current year (2022) \n3. Last n months \n"))

df_vis = pd.DataFrame()

for i in range(len(df)):
        if i < 12:
            row = {'Month': f"2021-{df['Month'][i]}", 'Receipt_Count': df['Receipt_Count'][i]}
        else:
            row = {'Month': f"2022-{df['Month'][i]}", 'Receipt_Count': df['Receipt_Count'][i]}
        df_vis = pd.concat([df_vis, pd.DataFrame([row])], ignore_index=True)

if val not in [1, 2, 3]:
    print("Invalid input")

elif val == 1:
    print("visualizing all values")

elif val == 2:
    df_vis = df_vis.iloc[12:, :]

elif val == 3:
    n = int(input("Enter number of months: "))
    df_vis = df_vis.iloc[-n:, :]

if not df_vis.empty:
    plt.figure(figsize=(10, 6))
    plt.bar(df_vis['Month'].astype(str), df_vis['Receipt_Count'], color='skyblue')
    plt.title('Receipt Count Over Time')
    plt.xlabel('Month')
    plt.ylabel('Receipt Count')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()
else:
    print("No data to visualize.")