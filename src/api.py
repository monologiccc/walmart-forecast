from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
from datetime import date
import logging

logging.basicConfig(filename='logs/predictions.log', level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Walmart Sales Forecast API")
model = joblib.load('models/model.pkl')

history = pd.read_csv('data/Walmart_Sales.csv')
history['Date'] = pd.to_datetime(history['Date'], format='%d-%m-%Y')


class PredictRequest(BaseModel):
    store: int = Field(..., ge=1, le=45)
    date: date
    temperature: float
    fuel_price: float
    cpi: float
    unemployment: float
    holiday_flag: int = Field(0, ge=0, le=1)


class PredictResponse(BaseModel):
    store: int
    date: date
    predicted_sales: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    store_hist = history[history['Store'] == req.store].sort_values('Date')
    if store_hist.empty:
        raise HTTPException(status_code=404, detail=f"Нет истории по магазину {req.store}")

    last_sales = store_hist['Weekly_Sales'].values
    target_date = pd.Timestamp(req.date)

    def sales_n_weeks_ago(n):
        idx = store_hist[store_hist['Date'] < target_date]
        if len(idx) < n:
            return float(last_sales.mean())
        return float(idx['Weekly_Sales'].values[-n])

    features = pd.DataFrame([{
        'Store': req.store,
        'Holiday_Flag': req.holiday_flag,
        'Temperature': req.temperature,
        'Fuel_Price': req.fuel_price,
        'CPI': req.cpi,
        'Unemployment': req.unemployment,
        'year': target_date.year,
        'month': target_date.month,
        'week_of_year': target_date.isocalendar()[1],
        'sales_lag_1': sales_n_weeks_ago(1),
        'sales_lag_2': sales_n_weeks_ago(2),
        'sales_lag_4': sales_n_weeks_ago(4),
        'sales_lag_52': sales_n_weeks_ago(52),
        'rolling_mean_4': store_hist['Weekly_Sales'].tail(4).mean(),
        'rolling_std_4': store_hist['Weekly_Sales'].tail(4).std(),
    }])

    prediction = float(model.predict(features)[0])
    logger.info(f"store={req.store} date={req.date} prediction={prediction:.2f}")

    return PredictResponse(store=req.store, date=req.date, predicted_sales=round(prediction, 2))