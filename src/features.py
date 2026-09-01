import pandas as pd

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')
    df['Date'] = df['Date'].fillna(pd.to_datetime(df['Date'], errors='coerce'))
    df = df.sort_values(['Store', 'Date']).reset_index(drop=True)

    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['week_of_year'] = df['Date'].dt.isocalendar().week.astype(int)

    for lag in [1, 2, 4, 52]:
        df[f'sales_lag_{lag}'] = df.groupby('Store')['Weekly_Sales'].shift(lag)

    df['rolling_mean_4'] = df.groupby('Store')['Weekly_Sales'] \
        .transform(lambda s: s.shift(1).rolling(4).mean())
    df['rolling_std_4'] = df.groupby('Store')['Weekly_Sales'] \
        .transform(lambda s: s.shift(1).rolling(4).std())

    return df

FEATURE_COLS = [
    'Store', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment',
    'year', 'month', 'week_of_year',
    'sales_lag_1', 'sales_lag_2', 'sales_lag_4', 'sales_lag_52',
    'rolling_mean_4', 'rolling_std_4',
]
TARGET_COL = 'Weekly_Sales'


df = build_features(pd.read_csv('data/Walmart_Sales.csv'))
before = len(df)
df = df.dropna().reset_index(drop=True)
print(f"Удалено {before - len(df)} строк без лагов, осталось {len(df)}")