from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import os
from backend.load_data import load_ohlcv
from backend.features import add_features

app = FastAPI()

# Cho phép CORS để frontend (JS) gọi API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # khi deploy thực tế bạn giới hạn domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

@app.get("/predict")
def predict(ticker: str):
    model_path = os.path.join(MODEL_DIR, f"{ticker}_rf.joblib")
    if not os.path.exists(model_path):
        return {"error": f"Model for {ticker} not found"}

    model = joblib.load(model_path)

    # Tải dữ liệu gần đây
    df = load_ohlcv(ticker, start="2023-01-01")
    df = add_features(df)

    # Features như lúc train
    feature_cols = [
        "return", "log_return", "ma7", "ma21", "ma50",
        "ema12", "ema26", "macd",
        "bb_up", "bb_dn", "rsi14",
        "vol_change", "evm", "evm_ma14",
        "lag_close_1", "lag_close_2", "lag_close_3", "lag_close_5", "lag_close_7",
        "lag_return_1", "lag_return_2", "lag_return_3", "lag_return_5", "lag_return_7",
    ]
    df = df.dropna()
    X = df[feature_cols].values
    preds = model.predict(X)

    # Lấy dự đoán mới nhất
    latest_pred = int(preds[-1])
    signal = "buy" if latest_pred == 1 else "sell"

    # Trả cả dữ liệu giá để frontend vẽ chart
    data = df[['Open','High','Low','Close']].reset_index().to_dict(orient="records")

    return {
        "ticker": ticker,
        "signal": signal,
        "latest_date": str(df.index[-1].date()),
        "data": data,
    }
