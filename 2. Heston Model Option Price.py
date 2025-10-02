import numpy as np
import pandas as pd
import scipy.integrate as integrate
from math import pi, log


def read_data(ticker_file, stock_file, option_file):
    tickers = pd.read_excel(ticker_file)["Ticker"].astype(str).tolist()

    df_raw = pd.read_excel(stock_file, index_col=0)
    latest_prices = df_raw.iloc[0]
    stock_prices = latest_prices.to_frame(name="Price")
    stock_prices.index.name = "Ticker"

    options = pd.read_excel(option_file)
    return tickers, stock_prices, options


def heston_char_func(u, S, K, T, r, v0, kappa, theta, sigma, rho, phi):
    a = kappa * theta
    b = kappa + (1 - phi) * rho * sigma
    d = np.sqrt((rho * sigma * u * 1j - b) ** 2 - sigma ** 2 * (2 * phi * u * 1j - u ** 2))
    g = (b - rho * sigma * u * 1j + d) / (b - rho * sigma * u * 1j - d)

    exp_dt = np.exp(d * T)
    C = r * u * 1j * T + (a / sigma**2) * ((b - rho * sigma * u * 1j + d) * T - 2 * np.log((1 - g * exp_dt) / (1 - g)))
    D = ((b - rho * sigma * u * 1j + d) / sigma**2) * ((1 - exp_dt) / (1 - g * exp_dt))

    return np.exp(C + D * v0 + 1j * u * log(S))


def heston_price_call(S, K, T, r, v0, kappa, theta, sigma, rho):
    def integrand(phi):
        u = phi - 1j
        f1 = heston_char_func(u, S, K, T, r, v0, kappa, theta, sigma, rho, 1)
        f2 = heston_char_func(phi, S, K, T, r, v0, kappa, theta, sigma, rho, 2)
        return (np.exp(-1j * phi * np.log(K)) * f2 / (1j * phi)).real

    integral = integrate.quad(lambda phi: integrand(phi), 1e-5, 100, limit=500)[0]
    call_price = S * 0.5 - K * np.exp(-r * T) * (0.5 - 1 / pi * integral)
    return call_price


def calculate_option_prices(tickers, stock_prices, options):
    kappa = 1.7056
    theta = 0.4555
    sigma = 0.0104
    rho = -0.0000
    v0 = 0.3378

    results = []

    for ticker in tickers:
        S = stock_prices.loc[ticker, "Price"]
        df_ticker = options[options["Ticker"] == ticker]

        for _, row in df_ticker.iterrows():
            K = float(row["Strike Price (ATM)"])
            T = float(row["Time To Maturity (1M)"])
            r = float(row["Risk Free Rate"])
            market_price = float(row["Option Price"])

            model_price = heston_price_call(S, K, T, r, v0, kappa, theta, sigma, rho)

            results.append({
                "Ticker": ticker,
                "Strike Price (ATM)": K,
                "Time to Maturity (1M)": T,
                "Risk Free Interest Rate": r,
                "Spot Price": S,
                "Option Price": model_price,
            })

    return pd.DataFrame(results)


if __name__ == "__main__":
    ticker_file = "/Users/tiffanyyu/Documents/Dissertation/0. Ticker.xlsx"
    stock_file = "/Users/tiffanyyu/Documents/Dissertation/1. Daily Stock Prices/1. Daily Stock Prices.xlsx"
    option_file = "/Users/tiffanyyu/Documents/Dissertation/2. European Call Options/2. European Call Options.xlsx"

    tickers, stock_prices, options = read_data(ticker_file, stock_file, option_file)

    df_result = calculate_option_prices(tickers, stock_prices, options)
    df_result.to_excel("2. Heston Model Option Price.xlsx", index=False)
    print("2. Heston model option prices completed and saved to Excel.")
