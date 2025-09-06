# backend/predict.py   (hoặc chèn vào backend/app.py)
from fastapi import APIRouter, HTTPException
import os, joblib
import pandas as pd
from .load_data import load_ohlcv
from .features import add_features

router = APIRouter()

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

@router.get("/predict")
def predict(ticker: str, n: int = 250):
    """
    Trả về:
      - ticker
      - latest_date (YYYY-MM-DD)
      - signal ('buy'/'sell')
      - probability (nếu có)
      - data: list các bản ghi {Date (ISO), Open, High, Low, Close, Volume}
    """
    model_path = os.path.join(MODEL_DIR, f"{ticker}_rf.joblib")
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model not found. Run training first.")

    # 1) load raw OHLCV (mặc định load 5 năm trong load_ohlcv nếu không truyền start)
    df = load_ohlcv(ticker)

    # 2) normalize index: loại timezone, set time về 00:00 và groupby ngày (nếu có nhiều bản ghi cùng ngày)
    # chuyển index về datetime nếu chưa
    df.index = pd.to_datetime(df.index)
    try:
        # nếu index có tz -> remove
        if df.index.tz is not None:
            df.index = df.index.tz_convert(None)
    except Exception:
        # ignore nếu không hỗ trợ tz_convert
        pass

    # normalize time -> chỉ giữ ngày (00:00)
    df.index = df.index.normalize()

    # group theo ngày (tránh duplicate nếu có)
    df = df.groupby(df.index).agg(
        Open=('Open', 'first'),
        High=('High', 'max'),
        Low=('Low', 'min'),
        Close=('Close', 'last'),
        Volume=('Volume', 'sum')
    ).sort_index()

    if df.empty:
        raise HTTPException(status_code=404, detail="No OHLCV data after grouping.")

    # 3) Compute features (trên dữ liệu đã được group theo ngày)
    df_feat = add_features(df)

    # 4) load model + predict
    model = joblib.load(model_path)

    feature_cols = [
        "return","log_return","ma7","ma21","ma50",
        "ema12","ema26","macd",
        "bb_up","bb_dn","rsi14",
        "vol_change","evm","evm_ma14",
        "lag_close_1","lag_close_2","lag_close_3","lag_close_5","lag_close_7",
        "lag_return_1","lag_return_2","lag_return_3","lag_return_5","lag_return_7"
    ]

    # dropna để cho chắc
    df_feat = df_feat.dropna()
    if df_feat.empty:
        raise HTTPException(status_code=400, detail="Not enough feature rows to predict (after dropna).")

    # get last N for returning to frontend
    df_out = df.tail(n).reset_index()
    # ensure Date column is ISO (UTC midnight)
    df_out['Date'] = pd.to_datetime(df_out['index']).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    # keep only necessary cols
    records = df_out[['Date','Open','High','Low','Close','Volume']].to_dict(orient='records')

    # prediction: use latest feature row
    X_latest = df_feat[feature_cols].iloc[-1:].values
    try:
        proba = float(model.predict_proba(X_latest)[0][1])
    except Exception:
        # if model doesn't support predict_proba
        proba = None
    pred = int(model.predict(X_latest)[0])
    signal = "buy" if pred == 1 else "sell"

    return {
        "ticker": ticker,
        "latest_date": df_feat.index[-1].strftime("%Y-%m-%d"),
        "signal": signal,
        "probability": proba,
        "data": records
    }
