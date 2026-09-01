import pandas as pd
import joblib
import shutil
from datetime import datetime
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
from features import build_features, FEATURE_COLS, TARGET_COL

def retrain():
    df = pd.read_csv('data/Walmart_Sales.csv')
    df = build_features(df).dropna().reset_index(drop=True)

    cutoff_date = df['Date'].quantile(0.85, interpolation='nearest')
    train = df[df['Date'] <= cutoff_date]
    test = df[df['Date'] > cutoff_date]

    model = LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=6,
                           random_state=42, verbosity=-1)
    model.fit(train[FEATURE_COLS], train[TARGET_COL])

    new_mae = mean_absolute_error(test[TARGET_COL], model.predict(test[FEATURE_COLS]))

    try:
        current_model = joblib.load('models/model.pkl')
        current_mae = mean_absolute_error(test[TARGET_COL], current_model.predict(test[FEATURE_COLS]))
    except FileNotFoundError:
        current_mae = float('inf')

    if new_mae < current_mae:
        shutil.copy('models/model.pkl', f'models/model_backup_{datetime.now():%Y%m%d}.pkl')
        joblib.dump(model, 'models/model.pkl')
        print(f"Модель обновлена: MAE {current_mae:,.0f} -> {new_mae:,.0f}")
    else:
        print(f"Новая модель хуже текущей ({new_mae:,.0f} vs {current_mae:,.0f}), не заменяю")

if __name__ == '__main__':
    retrain()