import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import norm
import datetime


def load_volatility_data(file_path: str) -> pd.DataFrame:
    """
    Load tickers and historical volatility values from an Excel file.
    """
    return pd.read_excel(file_path)


def get_risk_free_rate(ticker: str) -> float:
    """
    Return the 1-month risk-free rate based on the company’s country.
    """
    country_rates = {
        "USA": 0.0433,
        "UK": 0.0437,
        "Ireland": 0.0217,
        "Switzerland": 0.0021
    }

    uk_tickers = ["LIN"]
    ireland_tickers = ["ACN", "ETN", "MDT"]
    switzerland_tickers = ["CB"]

    if ticker in uk_tickers:
        return country_rates["UK"]
    elif ticker in ireland_tickers:
        return country_rates["Ireland"]
    elif ticker in switzerland_tickers:
        return country_rates["Switzerland"]
    return country_rates["USA"]


def get_spot_price(ticker: str, date: datetime.date) -> float:
    """
    Get the stock closing price on a given date.
    """
    stock = yf.Ticker(ticker)
    history = stock.history(start=date, end=date + datetime.timedelta(days=1))

    if not history.empty:
        return round(history["Close"].iloc[0], 2)
    return None


def black_scholes_call(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float
) -> float:
    """
    Compute the Black-Scholes price for a European call option.
    """
    if S <= 0 or sigma <= 0:
        return np.nan

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    return S * norm.cdf(d1) - K * norm.cdf(d2)


def compute_option_prices(
    df: pd.DataFrame,
    date: datetime.date,
    maturity_days: int = 30
) -> pd.DataFrame:
    """
    Calculate European call option prices for each volatility time frame.
    """
    time_to_maturity = maturity_days / 365
    volatility_labels = {
        "1 Month": "1M",
        "3 Months": "3M",
        "6 Months": "6M",
        "1 Year": "1Y",
        "2 Years": "2Y"
    }

    results = []

    for _, row in df.iterrows():
        ticker = row["Ticker"]
        spot_price = get_spot_price(ticker, date)

        if spot_price is None or np.isnan(spot_price):
            continue

        strike_price = spot_price  # ATM
        risk_free = get_risk_free_rate(ticker)

        entry = {
            "Ticker": ticker,
            "Date": date,
            "Spot Price": spot_price,
            "Strike Price": strike_price,
            "Risk Free Rate": risk_free,
            "Time to Maturity (1M)": time_to_maturity
        }

        for col, label in volatility_labels.items():
            sigma = row.get(col)
            option_price = (
                np.nan if pd.isna(sigma)
                else black_scholes_call(spot_price, strike_price, time_to_maturity, risk_free, sigma)
            )
            entry[f"{label} Volatility"] = sigma
            entry[f"{label} Option Price"] = option_price

        results.append(entry)

    return pd.DataFrame(results)


def main():
    file_path = "/Users/tiffanyyu/Documents/Dissertation/4. Historical Volatility/1. Historical Volatility Time Frame/1. Historical Volatility Time Frame.xlsx"
    output_file = "2. Historical Volatility Time Frame European Call Option Price.xlsx"
    valuation_date = datetime.date(2025, 5, 30)

    df = load_volatility_data(file_path)
    data = compute_option_prices(df, valuation_date)
    data.to_excel(output_file, index=False)

    print(f"Data saved to {output_file}")


if __name__ == "__main__":
    main()
