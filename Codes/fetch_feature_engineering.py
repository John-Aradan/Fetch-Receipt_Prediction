# Importing necessary libraries
import pandas as pd  # Pandas is used for data manipulation and analysis
import os  # os is used for interacting with the file system

# Defining the folder where the data files are stored
data_folder = os.path.join(os.path.dirname(__file__), 'Data')  # Creates the path to the 'Data' folder

# Reading a CSV file into a pandas DataFrame (a table of data)
df = pd.read_csv(os.path.join(data_folder, "data_daily.csv"))  # Load the 'data_daily.csv' file

# Convert the first column (likely containing dates) into actual date format
df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])

# Ensure the date column is properly formatted
df['# Date'] = pd.to_datetime(df.iloc[:, 0])  # Handles dates if the column name is '# Date'

# Create a new column for the month number extracted from the date
df['Month'] = df['# Date'].dt.month

### Adding additional information based on the date ###
# Add a new column that gives the name of the day (e.g., Monday, Tuesday)
df['Day_of_Week'] = df['# Date'].dt.day_name()

# Add a new column that shows which quarter of the year the date falls into
df['Quarter'] = df['# Date'].dt.quarter

### Calculating moving averages and lag values ###
# Calculate a 1-month (30-day) moving average of the 'Receipt_Count' column
df['SMA_1'] = df['Receipt_Count'].shift(1).rolling(window=30).mean()    # Shift moves the data by 1 day

# Calculate a 3-month (90-day) moving average of 'Receipt_Count'
df['SMA_3'] = df['Receipt_Count'].shift(1).rolling(window=90).mean()

# Calculate a 6-month (180-day) moving average of 'Receipt_Count'
df['SMA_6'] = df['Receipt_Count'].shift(1).rolling(window=180).mean()

# Create lag columns, which show the value of 'Receipt_Count' from previous months
df['Lag_1'] = df['Receipt_Count'].shift(30)  # 1-month lag
df['Lag_3'] = df['Receipt_Count'].shift(90)  # 3-month lag
df['Lag_6'] = df['Receipt_Count'].shift(180)  # 6-month lag

### Handling missing data ###
# If any column has missing values (NaN), fill it with the column's mean
for column in df.columns:
    if df[column].isnull().any():  # Check for missing values
        mean_value = df[column].mean()  # Calculate mean of the column
        df[column] = df[column].fillna(mean_value)  # Replace missing values with the mean

# Remove the '# Date' column as it's no longer needed
df.drop(columns=['# Date'], inplace=True)

### Reordering columns ###
# Rearrange the columns in a specific order for better readability and analysis
new_column_order = [
    'Day_of_Week', 'Month', 'Quarter', 'SMA_1', 'SMA_3',
    'SMA_6', 'Lag_1', 'Lag_3', 'Lag_6', 'Receipt_Count'
]

df = df[new_column_order]  # Reassign the DataFrame with the new column order

# Save the modified DataFrame to a new CSV file in the 'Data' folder
df.to_csv(os.path.join(data_folder, "data_days.csv"), index=False)  # No index column in the output file

### Processing for monthly data ###
# Reading the same data again for monthly aggregation
dfm = pd.read_csv("data_daily.csv")  # Load 'data_daily.csv' again
dfm['# Date'] = pd.to_datetime(dfm['# Date'])  # Convert the date column to proper date format
dfm['Month'] = dfm['# Date'].dt.month  # Extract month number from the date
dfm['Quarter'] = dfm['# Date'].dt.quarter  # Extract the quarter of the year

# Group the data by month and calculate total receipts for each month
dfm = dfm.groupby('Month').agg({'Receipt_Count': 'sum', 'Quarter': 'first'}).reset_index()

### Moving averages and lag values for monthly data ###
# Monthly moving averages and lag values, similar to the previous daily calculations
dfm['SMA_1'] = dfm['Receipt_Count'].shift(1).rolling(window=1).mean()  # 1-month SMA
dfm['SMA_3'] = dfm['Receipt_Count'].shift(1).rolling(window=3).mean()  # 3-month SMA
dfm['SMA_6'] = dfm['Receipt_Count'].shift(1).rolling(window=6).mean()  # 6-month SMA

# Lag values for monthly data
dfm['Lag_1'] = dfm['Receipt_Count'].shift(1)  # 1-month lag
dfm['Lag_3'] = dfm['Receipt_Count'].shift(3)  # 3-month lag
dfm['Lag_6'] = dfm['Receipt_Count'].shift(6)  # 6-month lag

# Handle missing values by filling with the mean, as done earlier
for column in dfm.columns:
    if dfm[column].isnull().any():
        mean_value = dfm[column].mean()
        dfm[column] = dfm[column].fillna(mean_value)

### Reordering columns for monthly data ###
new_column_order = [
    'Month', 'Quarter', 'SMA_1', 'SMA_3',
    'SMA_6', 'Lag_1', 'Lag_3', 'Lag_6', 'Receipt_Count'
]

dfm = dfm[new_column_order]  # Rearrange columns

# Save the modified monthly DataFrame to a new CSV file
dfm.to_csv(os.path.join(data_folder, "data_months.csv"), index=False)
