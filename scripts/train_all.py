# scripts/train_all.py
# Một script để train model cho các ticker định nghĩa trong backend/config.py
# Chạy: python scripts/train_all.py
import os
from backend.config import TICKERS
from backend.load_data import load_ohlcv
from backend.features import add_features
from backend.labeling import add_labels
from backend.train import train_model

# Tạo thư mục models nếu chưa tồn tại (đặt ở root_project/models)
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

feature_cols = [
    "return", "log_return",
    "ma7", "ma21", "ma50",
    "ema12", "ema26", "macd",
    "bb_up", "bb_dn",
    "rsi14",
    "vol_change",
    "evm", "evm_ma14",   # chỉ để tên cột, KHÔNG phải Volume.evm
    "lag_close_1", "lag_close_2", "lag_close_3", "lag_close_5", "lag_close_7",
    "lag_return_1", "lag_return_2", "lag_return_3", "lag_return_5", "lag_return_7",
]

HORIZON = 5
THRESHOLD = 0.0

for ticker in TICKERS:
    print(f"\n=== Training {ticker} ===")
    try:
        # 1) Load OHLCV data
        df = load_ohlcv(ticker)

        # 2) Compute features
        df = add_features(df)

        print("Columns after add_features:", df.columns.tolist())

        # 3) Add labels (future return horizon)
        df = add_labels(df, horizon=HORIZON, threshold=THRESHOLD)

        # 4) Train the RandomForest and save model
        model_path = os.path.join(MODEL_DIR, f"{ticker}_rf.joblib")
        
        train_model(df, feature_cols, model_path)

        print(f"Saved model to {model_path}")
    except Exception as e:
        print(f"Error training {ticker}: {e}")
