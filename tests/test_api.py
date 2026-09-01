from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}

def test_predict_valid_store():
    resp = client.post("/predict", json={
        "store": 1, "date": "2012-11-02",
        "temperature": 55.3, "fuel_price": 3.45,
        "cpi": 212.1, "unemployment": 7.8, "holiday_flag": 0
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["store"] == 1
    assert isinstance(body["predicted_sales"], float)
    assert body["predicted_sales"] > 0

def test_predict_invalid_store():
    resp = client.post("/predict", json={
        "store": 999, "date": "2012-11-02",
        "temperature": 55.3, "fuel_price": 3.45,
        "cpi": 212.1, "unemployment": 7.8, "holiday_flag": 0
    })
    assert resp.status_code == 422

def test_predict_store_out_of_range():
    resp = client.post("/predict", json={
        "store": 100, "date": "2012-11-02",
        "temperature": 55.3, "fuel_price": 3.45,
        "cpi": 212.1, "unemployment": 7.8, "holiday_flag": 0
    })
    assert resp.status_code == 422