import pandas as pd

df = pd.read_csv('data/Walmart_Sales.csv')
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
df = df.sort_values(['Store', 'Date']).reset_index(drop=True)

print(df.shape)
print(df.isna().sum())
print(df['Store'].nunique())
print(df.groupby('Store').size().min(), 
      df.groupby('Store').size().max())
print(df['Weekly_Sales'].describe())


import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.dates as mdates

weekly_total = df.groupby('Date')['Weekly_Sales'].sum()

plt.figure(figsize=(10, 4))

plt.plot(weekly_total.index, weekly_total.values)

plt.title('Суммарные продажи по неделям')
plt.xlabel('Дата')
plt.ylabel('Продажи')

plt.gca().yaxis.set_major_formatter(
    FuncFormatter(lambda x, pos: f'{x:,.0f}')
)

plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m.%Y'))

plt.xticks(rotation=45)

plt.show()


from matplotlib.ticker import FuncFormatter

df['month'] = df['Date'].dt.month
monthly = df.groupby('month')['Weekly_Sales'].mean()

plt.figure(figsize=(8, 4))

plt.bar(monthly.index, monthly.values)

plt.title('Средние продажи по месяцам')
plt.xlabel('Месяц')
plt.ylabel('Средние продажи')

plt.xticks(range(1, 13))

plt.gca().yaxis.set_major_formatter(
    FuncFormatter(lambda x, pos: f'{x:,.0f}')
)

plt.show()