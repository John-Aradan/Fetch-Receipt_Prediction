import pandas as pd
import os

data_folder = os.path.join(os.path.dirname(__file__), 'Data')

df = pd.read_csv(os.path.join(data_folder, "data_daily.csv"))

df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])

df['# Date'] = pd.to_datetime(df.iloc[:, 0])  # Or df['# Date'] if using column name directly
df['Month'] = df['# Date'].dt.month

### Trends
# Overall positive growth towards end of year (maybe dure to more users being added)
# Each month there are minimas and maximas - is there any relations there?

df['Day_of_Week'] = df['# Date'].dt.day_name()

df['Quarter'] = df['# Date'].dt.quarter

df['SMA_1'] = df['Receipt_Count'].shift(1).rolling(window=30).mean()    # 1-month simple moving average
df['SMA_3'] = df['Receipt_Count'].shift(1).rolling(window=90).mean()    # 3-month simple moving average
df['SMA_6'] = df['Receipt_Count'].shift(1).rolling(window=180).mean()   # 6-month simple moving average

df['Lag_1'] = df['Receipt_Count'].shift(30)
df['Lag_3'] = df['Receipt_Count'].shift(90)
df['Lag_6'] = df['Receipt_Count'].shift(180)

for column in df.columns:
        if df[column].isnull().any():
            mean_value = df[column].mean()
            df[column] = df[column].fillna(mean_value)

df.drop(columns=['# Date'], inplace=True)

new_column_order = [
    'Day_of_Week', 'Month', 'Quarter', 'SMA_1', 'SMA_3',
    'SMA_6', 'Lag_1', 'Lag_3', 'Lag_6', 'Receipt_Count'
]

df = df[new_column_order]

df.to_csv(os.path.join(data_folder, "data_days.csv"), index=False)

dfm = pd.read_csv("data_daily.csv")
dfm['# Date'] = pd.to_datetime(dfm['# Date'])
dfm['Month'] = dfm['# Date'].dt.month
dfm['Quarter'] = dfm['# Date'].dt.quarter

dfm = dfm.groupby('Month').agg({'Receipt_Count': 'sum', 'Quarter': 'first'}).reset_index()

dfm['SMA_1'] = dfm['Receipt_Count'].shift(1).rolling(window=1).mean()    # 1-month SMA excluding the current row
dfm['SMA_3'] = dfm['Receipt_Count'].shift(1).rolling(window=3).mean()    # 3-month SMA excluding the current row
dfm['SMA_6'] = dfm['Receipt_Count'].shift(1).rolling(window=6).mean()

dfm['Lag_1'] = dfm['Receipt_Count'].shift(1)
dfm['Lag_3'] = dfm['Receipt_Count'].shift(3)
dfm['Lag_6'] = dfm['Receipt_Count'].shift(6)


for column in dfm.columns:
        if dfm[column].isnull().any():
            mean_value = dfm[column].mean()
            dfm[column] = dfm[column].fillna(mean_value)

new_column_order = [
    'Month', 'Quarter', 'SMA_1', 'SMA_3',
    'SMA_6', 'Lag_1', 'Lag_3', 'Lag_6', 'Receipt_Count'
]

dfm = dfm[new_column_order]

dfm.to_csv(os.path.join(data_folder, "data_months.csv"), index=False)