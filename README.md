# Walmart Sales Forecast Service

ML-сервис прогнозирования недельных продаж по 45 магазинам Walmart на основе
исторических данных, макроэкономических показателей и признаков сезонности.

## Архитектура
Данные → Feature pipeline (лаги, скользящие статистики) → LightGBM →
FastAPI → Docker → CI/CD (GitHub Actions) → мониторинг (Streamlit)

## Результаты
- MAE: 39,362 (снижение на 20.1% относительно naive-бейзлайна: 49,266)
- MAPE: 3.88%
- Обучение: LightGBM, time-based train/test split (85/15)

## Запуск
\`\`\`bash
docker build -t walmart-forecast .
docker run -p 8000:8000 walmart-forecast
\`\`\`

## API
POST /predict — прогноз продаж по магазину и дате
GET /health — проверка живости сервиса

## Стек
Python, LightGBM, FastAPI, Docker, GitHub Actions, MLflow, Streamlit
