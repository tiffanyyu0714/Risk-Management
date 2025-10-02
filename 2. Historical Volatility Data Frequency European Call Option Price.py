import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import norm
import datetime


def load_volatility_data(file_path: str) -> pd.DataFrame:
    """
    Load ticker and historical volatility data from Excel.
    """
    return pd.read_excel(file_path)


def get_risk_free_rate(ticker: str) -> float:
    """
    Return the 1-month risk-free interest rate based on ticker's country.
    """
    rates = {
        "USA": 0.0433,
        "UK": 0.0437,
        "Ireland": 0.0217,
        "Switzerland": 0.0021
    }

    if ticker == "LIN":
        return rates["UK"]
    elif ticker in ["ACN", "ETN", "MDT"]:
        return rates["Ireland"]
    elif ticker == "CB":
        return rates["Switzerland"]
    return rates["USA"]


def get_spot_price(ticker: str, date: datetime.date) -> float:
    """
    Get the closing stock price on the specified date.
    """
    stock = yf.Ticker(ticker)
    history = stock.history(start=date, end=date + datetime.timedelta(days=1))

    if not history.empty:
        return round(history["Close"].iloc[0], 2)
    return None


def black_scholes_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calculate European call option price using the Black-Scholes model.
    """
    if S <= 0 or sigma <= 0:
        return np.nan

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    return S * norm.cdf(d1) - K * norm.cdf(d2)


def calculate_all_option_prices(file_path: str, target_date: datetime.date) -> pd.DataFrame:
    """
    Calculate option prices using different volatility estimates.
    Save all results into a single DataFrame.
    """
    df = load_volatility_data(file_path)
    results = []
    T = 1 / 12
    frequency_labels = ["Daily", "Weekly", "Bi-Weekly", "Monthly", "Quarterly"]

    for _, row in df.iterrows():
        ticker = row["Ticker"]
        spot_price = get_spot_price(ticker, target_date)

        if spot_price is None:
            continue

        r = get_risk_free_rate(ticker)
        K = spot_price  # ATM strike

        row_result = {
            "Ticker": ticker,
            "Date": target_date,
            "Spot Price": spot_price,
            "Strike Price": K,
            "Risk Free Rate": r,
            "Time to Maturity (1M)": T
        }

        for label in frequency_labels:
            sigma = row.get(label, np.nan)
            option_price = black_scholes_call(spot_price, K, T, r, sigma)
            row_result[f"{label} Volatility"] = sigma
            row_result[f"{label} Option Price"] = option_price

        results.append(row_result)

    return pd.DataFrame(results)


def save_to_excel_single_sheet(df: pd.DataFrame, output_path: str) -> None:
    """
    Save all results to a single Excel sheet using the default sheet name.
    """
    df.to_excel(output_path, index=False)


def main():
    file_path = (
        "/Users/tiffanyyu/Documents/Dissertation/4. Historical Volatility/2. Historical Volatility Data Frequency/1. Historical Volatility Data Frequency.xlsx"
    )
    output_path = "2. Historical Volatility Data Frequency European Call Option Price.xlsx"
    target_date = datetime.date(2025, 5, 30)

    results_df = calculate_all_option_prices(file_path, target_date)
    save_to_excel_single_sheet(results_df, output_path)

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
