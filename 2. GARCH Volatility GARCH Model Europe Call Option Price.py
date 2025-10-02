import pandas as pd
import numpy as np
from scipy.stats import norm


def black_scholes_call_price(s, k, t, r, sigma):
    """
    Calculate the European call option price using the Black-Scholes formula.

    Parameters:
        s (float): Spot price of the stock.
        k (float): Strike price of the option.
        t (float): Time to maturity in years.
        r (float): Risk-free interest rate.
        sigma (float): Annualized volatility.

    Returns:
        float: Option price
    """
    if sigma <= 0 or t <= 0:
        return np.nan

    d1 = (np.log(s / k) + (r + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)

    return s * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d2)


def load_and_merge_data(option_path, garch_path):
    """
    Load European option data and GARCH volatility data, then merge them.

    Parameters:
        option_path (str): Path to option data file.
        garch_path (str): Path to GARCH data file.

    Returns:
        DataFrame: Merged dataset
    """
    option_df = pd.read_excel(option_path)
    garch_df = pd.read_excel(garch_path)

    garch_columns = [
        'Ticker', 'GARCH(1,1)', 'GARCH(2,1)', 'GARCH(2,2)'
    ]
    garch_df = garch_df[garch_columns]

    merged_df = pd.merge(option_df, garch_df, on='Ticker')
    return merged_df


def compute_option_prices_all_garch(df):
    """
    Compute option prices using three GARCH model volatilities.

    Parameters:
        df (DataFrame): Merged dataset with all inputs.

    Returns:
        DataFrame: Dataset with calculated prices.
    """
    df['GARCH(1,1) Option Price'] = df.apply(
        lambda row: black_scholes_call_price(
            s=row['Spot Price'],
            k=row['Strike Price (ATM)'],
            t=row['Time To Maturity (1M)'],
            r=row['Risk Free Rate'],
            sigma=row['GARCH(1,1)']/100
        ),
        axis=1
    )

    df['GARCH(2,1) Option Price'] = df.apply(
        lambda row: black_scholes_call_price(
            s=row['Spot Price'],
            k=row['Strike Price (ATM)'],
            t=row['Time To Maturity (1M)'],
            r=row['Risk Free Rate'],
            sigma=row['GARCH(2,1)']/100
        ),
        axis=1
    )

    df['GARCH(2,2) Option Price'] = df.apply(
        lambda row: black_scholes_call_price(
            s=row['Spot Price'],
            k=row['Strike Price (ATM)'],
            t=row['Time To Maturity (1M)'],
            r=row['Risk Free Rate'],
            sigma=row['GARCH(2,2)']/100
        ),
        axis=1
    )

    return df


def main():
    """
    Main execution function.
    """
    option_path = "/Users/tiffanyyu/Documents/Dissertation/2. European Call Options/2. European Call Options.xlsx"
    garch_path = "/Users/tiffanyyu/Documents/Dissertation/5. GARCH Volatility/1. GARCH Volatility GARCH Model/1. GARCH Volatility GARCH Model.xlsx"

    merged_data = load_and_merge_data(option_path, garch_path)
    results = compute_option_prices_all_garch(merged_data)

    results.to_excel("2. GARCH Volatility GARCH Model European Call Option Price.xlsx", index=False)
    print("Prices calculated and saved to '2. GARCH Volatility GARCH Model European Call Option Price.xlsx'")


if __name__ == "__main__":
    main()
