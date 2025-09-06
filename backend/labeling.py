import pandas as pd

def add_labels(df, horizon=5, threshold=0.0):
    """
    Tạo label nhị phân: 1 nếu return trong `horizon` ngày > threshold, ngược lại 0.
    - horizon: số ngày trong tương lai dùng để tính return
    - threshold: ngưỡng return, ví dụ 0.0 cho tăng dương, 0.01 cho tăng >1%
    """
    df = df.copy()

    # Gia dong cua o ngay tuong lai
    df['future_close'] = df['Close'].shift(-horizon)

    # Tinh return tuong lai
    df['future_return'] = df['future_close'] / df['Close'] - 1

    # Label: 1 neu future return > threshold
    df['label'] = (df['future_return'] > threshold).astype(int)

    # Bo cac hang cuoi khong co future close
    df = df.dropna(subset=['future_close', 'future_return'])

    return df
