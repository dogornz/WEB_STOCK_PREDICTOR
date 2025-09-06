# WEB_STOCK_PREDICTOR

Một micro-project đơn giản để:
- Tải dữ liệu OHLCV (yfinance)
- Tạo feature kỹ thuật (MA, EMA, MACD, RSI, EVM, lag…)
- Gán label nhị phân (tăng/không tăng trong tương lai)
- Huấn luyện RandomForest cho từng ticker và lưu mô hình (.joblib)
- Cung cấp API FastAPI để trả tín hiệu, giá hiện tại và dự đoán giá ngày mai
- Frontend nhẹ sử dụng Chart.js để hiển thị đồ thị và kết quả

---

## Cấu trúc chính
```
├── 📁 .git/ 🚫 (auto-hidden)
├── 📁 backend/
│   ├── 📁 __pycache__/ 🚫 (auto-hidden)
│   ├── 🐍 config.py
│   ├── 🐍 features.py
│   ├── 📄 howtorun.txt
│   ├── 🐍 labeling.py
│   ├── 🐍 load_data.py
│   ├── 🐍 main.py
│   ├── 🐍 predict.py
│   └── 🐍 train.py
├── 📁 frontend/
│   └── 🌐 index.html
├── 📁 models/
│   ├── 📄 VCB.VN_rf.joblib
│   ├── 📄 VHM.VN_rf.joblib
│   └── 📄 VIC.VN_rf.joblib
├── 📁 scripts/
│   ├── 📁 __pycache__/ 🚫 (auto-hidden)
│   └── 🐍 train_all.py
├── 📁 venv/ 🚫 (auto-hidden)
├── 🚫 .gitignore
├── 📖 README.md
└── 📄 requirements.txt
```
- `backend/`
  - `main.py` - FastAPI app (endpoint `/predict`) với caching (Redis hoặc in-memory fallback)
  - `load_data.py` - tải OHLCV bằng yfinance
  - `features.py` - hàm `add_features(df)` tạo các đặc trưng
  - `labeling.py` - hàm `add_labels(df, horizon, threshold)` tạo label
  - `train.py` - hàm `train_model(...)` huấn luyện RandomForest và lưu model
- `models/` - nơi lưu model đã train (bị ignore trong git)
- `frontend/index.html` - UI demo dùng Chart.js
- `scripts/train_all.py` - script huấn luyện nhiều ticker theo `backend/config.py`
- `requirements.txt` - dependencies

---

## Tính năng chính

- Time-series aware training (không shuffle, 80% đầu train / 20% cuối test)
- Lưu model bằng `joblib` (file `models/<TICKER>_rf.joblib`)
- Endpoint `/predict?ticker=XXX&n=200` trả JSON gồm:
  - `ticker`, `latest_date`, `signal` (`buy`/`sell`)
  - `current_price` (giá đóng cửa mới nhất)
  - `predicted_return_tomorrow` (ước tính return kỳ vọng ngày mai)
  - `predicted_price_tomorrow` (ước tính giá ngày mai)
  - `probability` (P(buy) nếu model hỗ trợ `predict_proba`)
  - `data` (mảng bản ghi giá để frontend vẽ biểu đồ)

- Redis-backed caching cho model và OHLCV (cấu hình qua `REDIS_URL`). Nếu Redis không kết nối được, dùng cache in-memory TTL.

---

## Cài đặt nhanh

1. Tạo virtualenv và cài dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. (Tùy chọn) Chạy Redis nếu muốn dùng cache. Mặc định app thử kết nối `redis://localhost:6379/0`.

3. Huấn luyện model cho ticker mẫu:

```bash
python scripts/train_all.py
# hoặc import và gọi train_model từ backend/train.py cho 1 ticker cụ thể
```

4. Chạy backend:

```bash
uvicorn backend.main:app --reload
```

5. Mở `frontend/index.html` trong trình duyệt (hoặc serve file tĩnh) và nhấn "Load & Predict".

---

## Biến môi trường

- `REDIS_URL` - (tùy chọn) đường dẫn Redis, ví dụ `redis://localhost:6379/0`
- `MODEL_CACHE_TTL`, `DATA_CACHE_TTL` - TTL cache (giây)

---
