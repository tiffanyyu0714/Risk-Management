import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


def black_scholes_call_price(s, k, t, r, sigma):
    """Calculate the Black-Scholes price for a European call option."""
    d1 = (np.log(s / k) + (r + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    call_price = s * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d2)
    return call_price


def implied_volatility(market_price, s, k, t, r):
    """Compute the implied volatility using Brent's method."""
    try:
        iv = brentq(
            lambda sigma: black_scholes_call_price(s, k, t, r, sigma) - market_price,
            1e-6, 5.0
        )
    except ValueError:
        iv = np.nan
    return iv


def load_and_process_data(file_path):
    """Load Excel-like data, clean it, and calculate implied volatilities."""
    df = pd.read_excel(file_path)

    # Replace commas with dots and convert to float
    cols_to_clean = ['Spot Price', 'Strike Price (ATM)', 'Risk Free Rate',
                     'Time To Maturity (1M)', 'Option Price']
    for col in cols_to_clean:
        df[col] = df[col].astype(str).str.replace(',', '.').astype(float)

    # Calculate implied volatility
    df['Implied Volatility'] = df.apply(
        lambda row: implied_volatility(
            market_price=row['Option Price'],
            s=row['Spot Price'],
            k=row['Strike Price (ATM)'],
            t=row['Time To Maturity (1M)'],
            r=row['Risk Free Rate']
        ), axis=1
    )

    return df


def main():
    file_path = "/Users/tiffanyyu/Documents/Dissertation/2. European Call Options/2. European Call Options.xlsx"
    data = load_and_process_data(file_path)
    data.to_excel("3. Implied Volatility.xlsx", index=False)
    print("Data saved to: Implied Volatility.xlsx")


if __name__ == "__main__":
    main()
