import pandas as pd
import numpy as np
import joblib
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.features import build_features, FEATURE_COLS, TARGET_COL

df = pd.read_csv('data/Walmart_Sales.csv')
df = build_features(df).dropna().reset_index(drop=True)

cutoff_date = df['Date'].quantile(0.85, interpolation='nearest')
train = df[df['Date'] <= cutoff_date]
test = df[df['Date'] > cutoff_date]

print(f"Train: {train.shape}, период {train['Date'].min().date()} — {train['Date'].max().date()}")
print(f"Test:  {test.shape}, период {test['Date'].min().date()} — {test['Date'].max().date()}")

X_train, y_train = train[FEATURE_COLS], train[TARGET_COL]
X_test, y_test = test[FEATURE_COLS], test[TARGET_COL]

baseline_pred = X_test['sales_lag_1']
baseline_mae = mean_absolute_error(y_test, baseline_pred)
baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))

print(f"Baseline MAE:  {baseline_mae:,.0f}")
print(f"Baseline RMSE: {baseline_rmse:,.0f}")

model = LGBMRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    random_state=42,
    verbosity=-1,
)
model.fit(X_train, y_train)
pred = model.predict(X_test)

mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
mape = np.mean(np.abs((y_test - pred) / y_test)) * 100

print(f"LightGBM MAE:  {mae:,.0f}")
print(f"LightGBM RMSE: {rmse:,.0f}")
print(f"LightGBM MAPE: {mape:.2f}%")
print(f"Улучшение над бейзлайном (MAE): {(1 - mae/baseline_mae)*100:.1f}%")

joblib.dump(model, 'models/model.pkl')