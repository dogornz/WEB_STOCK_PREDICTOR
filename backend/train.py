import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def train_model(df, feature_cols, model_path, n_estimators=200, random_state=42):
    """
    Train RandomForest trên df đã có cột 'label'.
    - feature_cols: list các cột input
    - model_path: nơi lưu .joblib

    Lưu ý:
    - Đây là time-series nên không shuffle dữ liệu.
    - Sử dụng 80% dữ liệu đầu để train, 20% cuối để test (out-of-time validation).
    """

    df = df.dropna()

    if len(df) < 50:
        raise ValueError("Không đủ dữ liệu để train model.")

    X = df[feature_cols].values
    y = df['label'].values

    # Time-based split (80% train, 20% test)
    split_at = int(0.8 * len(df))
    X_train, X_test = X[:split_at], X[split_at:]
    y_train, y_test = y[:split_at], y[split_at:]

    # Khởi tạo model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1
    )

    # Train
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", acc)
    print(classification_report(y_test, y_pred))

    # Lưu model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)

    return model
