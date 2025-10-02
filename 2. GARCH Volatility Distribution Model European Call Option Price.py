import pandas as pd
import numpy as np
from scipy.stats import norm


def black_scholes_call_price(s, k, t, r, sigma):
    """
    Calculate the European call option price using the Black-Scholes formula.
    """
    if sigma <= 0 or t <= 0:
        return np.nan

    d1 = (np.log(s / k) + (r + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)

    return s * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d2)


def load_and_merge_data(option_path, garch_path):
    """
    Load and merge option data with GARCH volatilities from different distributions.
    """
    option_df = pd.read_excel(option_path)
    garch_df = pd.read_excel(garch_path)

    garch_columns = [
        'Ticker',
        'normal',
        't',
        'skewt'
    ]
    garch_df = garch_df[garch_columns]

    merged_df = pd.merge(option_df, garch_df, on='Ticker')
    return merged_df


def compute_option_prices_with_dists(df):
    """
    Compute BS option prices using GARCH(1,1) volatility under different distributions.
    Divide vol by 100 to convert % to decimal.
    """
    df['normal'] = df.apply(
        lambda row: black_scholes_call_price(
            s=row['Spot Price'],
            k=row['Strike Price (ATM)'],
            t=row['Time To Maturity (1M)'],
            r=row['Risk Free Rate'],
            sigma=row['normal'] / 100
        ), axis=1
    )

    df['t'] = df.apply(
        lambda row: black_scholes_call_price(
            s=row['Spot Price'],
            k=row['Strike Price (ATM)'],
            t=row['Time To Maturity (1M)'],
            r=row['Risk Free Rate'],
            sigma=row['t'] / 100
        ), axis=1
    )

    df['skewt'] = df.apply(
        lambda row: black_scholes_call_price(
            s=row['Spot Price'],
            k=row['Strike Price (ATM)'],
            t=row['Time To Maturity (1M)'],
            r=row['Risk Free Rate'],
            sigma=row['skewt'] / 100
        ), axis=1
    )

    return df


def main():
    option_path = "/Users/tiffanyyu/Documents/Dissertation/2. European Call Options/2. European Call Options.xlsx"
    garch_path = "/Users/tiffanyyu/Documents/Dissertation/5. GARCH Volatility/2. GARCH Volatility Distribution Model/1. GARCH Volatility Distribution Model.xlsx"

    merged_df = load_and_merge_data(option_path, garch_path)
    result_df = compute_option_prices_with_dists(merged_df)

    output_file = "2. GARCH Volatility Distribution Model European Call Option Price.xlsx"
    result_df.to_excel(output_file, index=False)
    print(f"Saved: {output_file}")


if __name__ == "__main__":
    main()
