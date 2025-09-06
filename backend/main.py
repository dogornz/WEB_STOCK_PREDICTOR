from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import os
import time
import pickle

# Redis client
import redis

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

# === Cache configuration ===
# TTLs in seconds (can override with env vars)
MODEL_CACHE_TTL = int(os.environ.get("MODEL_CACHE_TTL", 300))
DATA_CACHE_TTL = int(os.environ.get("DATA_CACHE_TTL", 300))
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

# Fallback in-memory caches if Redis is unavailable
model_cache = {}  # model_path -> (model, ts)
data_cache = {}  # (ticker, start, end, interval) -> (df, ts)

# Try to create Redis client
redis_client = None
try:
    redis_client = redis.Redis.from_url(REDIS_URL)
    # simple ping to verify connection
    redis_client.ping()
except Exception:
    redis_client = None


def _redis_set_pickle(key: str, value, ttl: int):
    if redis_client is None:
        return False
    try:
        redis_client.setex(key, ttl, pickle.dumps(value))
        return True
    except Exception:
        return False


def _redis_get_pickle(key: str):
    if redis_client is None:
        return None
    try:
        b = redis_client.get(key)
        if b is None:
            return None
        return pickle.loads(b)
    except Exception:
        return None


def get_model_cached(model_path):
    """Load model using Redis cache (pickle) with TTL. Fallback to in-memory cache if Redis unavailable."""
    key = f"model:{os.path.basename(model_path)}"
    now = time.time()

    # Try redis first
    model = _redis_get_pickle(key)
    if model is not None:
        return model

    # Fallback in-memory
    cached = model_cache.get(model_path)
    if cached:
        model, ts = cached
        if now - ts < MODEL_CACHE_TTL:
            return model

    # Load from disk
    model = joblib.load(model_path)

    # Store to caches
    model_cache[model_path] = (model, now)
    try:
        _redis_set_pickle(key, model, MODEL_CACHE_TTL)
    except Exception:
        pass

    return model


def get_ohlcv_cached(ticker, start=None, end=None, interval="1d"):
    """Load OHLCV using Redis cache (pickle). Fallback to in-memory if Redis unavailable."""
    key = f"ohlcv:{ticker}:{start}:{end}:{interval}"
    now = time.time()

    # Try redis
    df = _redis_get_pickle(key)
    if df is not None:
        # return a copy to avoid accidental mutation
        return df.copy()

    # Fallback in-memory
    cached = data_cache.get(key)
    if cached:
        df, ts = cached
        if now - ts < DATA_CACHE_TTL:
            return df.copy()

    # Fetch fresh
    df = load_ohlcv(ticker, start=start, end=end, interval=interval)

    # Store in caches
    data_cache[key] = (df.copy(), now)
    try:
        _redis_set_pickle(key, df, DATA_CACHE_TTL)
    except Exception:
        pass

    return df


@app.get("/predict")
def predict(ticker: str):
    model_path = os.path.join(MODEL_DIR, f"{ticker}_rf.joblib")
    if not os.path.exists(model_path):
        return {"error": f"Model for {ticker} not found"}

    # Load model (cached via Redis or in-memory fallback)
    model = get_model_cached(model_path)

    # Tải dữ liệu gần đây (cached via Redis or in-memory fallback)
    df = get_ohlcv_cached(ticker, start="2023-01-01")
    df = add_features(df)

    # Features như lúc train
    feature_cols = [
        "return",
        "log_return",
        "ma7",
        "ma21",
        "ma50",
        "ema12",
        "ema26",
        "macd",
        "bb_up",
        "bb_dn",
        "rsi14",
        "vol_change",
        "evm",
        "evm_ma14",
        "lag_close_1",
        "lag_close_2",
        "lag_close_3",
        "lag_close_5",
        "lag_close_7",
        "lag_return_1",
        "lag_return_2",
        "lag_return_3",
        "lag_return_5",
        "lag_return_7",
    ]
    df = df.dropna()

    if df.empty:
        return {"error": "Not enough data to compute features/prediction"}

    X = df[feature_cols].values

    # Full-sequence predictions (not strictly necessary but kept for compatibility)
    preds = model.predict(X)

    # Latest feature row for probability / single-row prediction
    X_latest = df[feature_cols].iloc[-1:].values

    # Lấy dự đoán mới nhất
    latest_pred = int(preds[-1])
    signal = "buy" if latest_pred == 1 else "sell"

    # current/latest close price
    latest_close = float(df["Close"].iloc[-1])

    # Estimate next-day return using historical next-day returns
    df_temp = df.copy()
    # future return for horizon=1 (tomorrow)
    df_temp["future_return_1"] = df_temp["Close"].shift(-1) / df_temp["Close"] - 1
    df_future = df_temp["future_return_1"].dropna()

    if not df_future.empty:
        pos_mean = (
            float(df_future[df_future > 0].mean())
            if not df_future[df_future > 0].empty
            else 0.0
        )
        neg_mean = (
            float(df_future[df_future <= 0].mean())
            if not df_future[df_future <= 0].empty
            else 0.0
        )
    else:
        pos_mean = neg_mean = 0.0

    # Try to get probability from model
    probability = None
    try:
        probability = float(model.predict_proba(X_latest)[0][1])
    except Exception:
        probability = None

    # Estimated return for next day
    if not df_future.empty:
        if probability is not None:
            est_return = probability * pos_mean + (1 - probability) * neg_mean
        else:
            # fallback to deterministic class prediction
            est_return = pos_mean if latest_pred == 1 else neg_mean
        predicted_price_tomorrow = latest_close * (1 + est_return)
    else:
        est_return = None
        predicted_price_tomorrow = None

    # Trả cả dữ liệu giá để frontend vẽ chart
    data = df[["Open", "High", "Low", "Close"]].reset_index().to_dict(orient="records")

    return {
        "ticker": ticker,
        "signal": signal,
        "latest_date": str(df.index[-1].date()),
        "current_price": latest_close,
        "predicted_return_tomorrow": est_return,
        "predicted_price_tomorrow": predicted_price_tomorrow,
        "probability": probability,
        "data": data,
    }
