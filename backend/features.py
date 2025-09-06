import pandas as pd
import numpy as np

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tính toán các đặc trưng (features) từ dữ liệu OHLCV.
    Input: DataFrame có các cột [Open, High, Low, Close, Volume]
    Output: DataFrame có thêm các cột feature.
    """

    df = df.copy()

    # === Returns ===
    df['return'] = df['Close'].pct_change()
    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))

    # === Moving Averages ===
    df['ma7'] = df['Close'].rolling(window=7).mean()
    df['ma21'] = df['Close'].rolling(window=21).mean()
    df['ma50'] = df['Close'].rolling(window=50).mean()

    # === Exponential Moving Averages ===
    df['ema12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['ema26'] = df['Close'].ewm(span=26, adjust=False).mean()

    # === MACD ===
    df['macd'] = df['ema12'] - df['ema26']

    # === Bollinger Bands ===
    df['ma21_std'] = df['Close'].rolling(window=21).std()
    df['bb_up'] = df['ma21'] + 2 * df['ma21_std']
    df['bb_dn'] = df['ma21'] - 2 * df['ma21_std']

    # === RSI (14) ===
    delta = df['Close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain, index=df.index).rolling(window=14).mean()
    avg_loss = pd.Series(loss, index=df.index).rolling(window=14).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df['rsi14'] = 100 - (100 / (1 + rs))

    # === Volume features ===
    df['vol_ma21'] = df['Volume'].rolling(window=21).mean()
    df['vol_change'] = df['Volume'] / (df['vol_ma21'] + 1e-9)

    # === Ease of Movement (EVM) ===
    df['evm'] = (
        ((df['High'] + df['Low'])/2 - (df['High'].shift(1) + df['Low'].shift(1))/2)
        * (df['High'] - df['Low'])
    ) / (df['Volume'] + 1e-9)

    df['evm_ma14'] = df['evm'].rolling(window=14).mean()

    # === Lag features ===
    for lag in [1, 2, 3, 5, 7]:
        df[f'lag_close_{lag}'] = df['Close'].shift(lag)
        df[f'lag_return_{lag}'] = df['return'].shift(lag)

    # Loại bỏ giá trị NaN
    df = df.dropna()

    return df
