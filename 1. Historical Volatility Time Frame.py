import pandas as pd
import yfinance as yf
import numpy as np
import datetime


def load_tickers(file_path: str) -> list:
    """
    Load unique tickers from the second column of an Excel file.
    """
    df = pd.read_excel(file_path)
    return df.iloc[:, 1].dropna().unique().tolist()


def calculate_volatility(prices: pd.Series, window: int) -> float:
    """
    Calculate annualized historical volatility for a given time window.
    """
    log_returns = np.log(prices / prices.shift(1)).dropna()
    if len(log_returns) >= window:
        vol = log_returns[-window:].std() * np.sqrt(252)
        return round(vol.item(), 6)
    return None


def extract_volatility_data(file_path: str, end_date: datetime.date) -> pd.DataFrame:
    """
    Download historical prices and compute volatilities for 100 tickers.
    """
    tickers = load_tickers(file_path)
    start_date = end_date - pd.DateOffset(years=4)

    time_windows = {
        "1 Month": 21,
        "3 Months": 63,
        "6 Months": 126,
        "1 Year": 252,
        "2 Years": 504
    }

    results = []

    for ticker in tickers:
        try:
            data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=False
            )

            if 'Adj Close' not in data.columns:
                print(f"[Skipped] {ticker}: 'Adj Close' not found.")
                continue

            row = {"Ticker": ticker}
            for label, days in time_windows.items():
                row[label] = calculate_volatility(data['Adj Close'], days)

            results.append(row)

        except Exception as e:
            print(f"[Error] {ticker}: {e}")

    return pd.DataFrame(results)


def main():
    file_path = "/Users/tiffanyyu/Documents/Dissertation/0. Ticker.xlsx"
    end_date = pd.Timestamp("2025-05-30")

    vol_data = extract_volatility_data(file_path, end_date)
    vol_data.to_excel("1. Historical Volatility Time Frame.xlsx", index=False)

    print("Volatility data saved to '1. Historical Volatility Time Frame.xlsx'")


if __name__ == "__main__":
    main()
