import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def load_ohlcv(ticker, start=None, end=None, interval='1d'):
    """
    Tải dữ liệu OHLCV cho ticker bằng yfinance.
    Trả về DataFrame có index Datetime và cột Open/High/Low/Close/Volume.
    """

    if start is None:
        start = (datetime.today() - timedelta(days=365*5)).strftime("%Y-%m-%d")
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")

    # Dùng yfinance
    t = yf.Ticker(ticker)
    df = t.history(start=start, end=end, interval=interval, actions=False)

    if df.empty:
        raise ValueError(f"No data for ticker {ticker} - check symbol and suffix")

    # Chỉ lấy các cột cần thiết
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df.index = pd.to_datetime(df.index)

    # Sắp xếp theo thời gian
    df.sort_index(inplace=True)

    # Điền giá trị thiếu
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    return df
