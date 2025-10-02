import pandas as pd
import yfinance as yf
import datetime


def load_company_data(file_path: str) -> pd.DataFrame:
    """
    Load ticker and country info from an Excel file.
    """
    return pd.read_excel(file_path)


def get_risk_free_rate(country: str) -> float:
    """
    Return the 1-month risk-free interest rate based on the country.
    """
    rates = {
        "USA": 0.0433,
        "UK": 0.04211,
        "Ireland": 0.02161,
        "Switzerland": 0.00207
    }
    return rates.get(country, None)


def get_spot_price(ticker: str, date: datetime.date) -> float:
    """
    Get the spot price of the stock on the given date.
    """
    stock = yf.Ticker(ticker)
    history = stock.history(start=date, end=date + datetime.timedelta(days=1))

    if not history.empty:
        return history["Close"].iloc[0]
    return None


def get_option_price(ticker: str, date: datetime.date, maturity_days: int = 30) -> tuple:
    """
    Find ATM call option with nearest maturity (default 1 month)
    and return strike price and last option price.
    """
    stock = yf.Ticker(ticker)
    try:
        options_dates = stock.options
        target_date = date + datetime.timedelta(days=maturity_days)
        expiry = min(options_dates, key=lambda x: abs(datetime.datetime.strptime(x, "%Y-%m-%d").date() - target_date))

        opt_chain = stock.option_chain(expiry)
        calls = opt_chain.calls

        # Find ATM option
        spot = get_spot_price(ticker, date)
        if spot is None or calls.empty:
            return None, None

        calls["diff"] = (calls["strike"] - spot).abs()
        atm_option = calls.loc[calls["diff"].idxmin()]

        return atm_option["strike"], atm_option["lastPrice"]
    except Exception:
        return None, None


def extract_option_data(file_path: str, target_date: datetime.date) -> pd.DataFrame:
    """
    Process the Excel and return a DataFrame with required option data.
    """
    df = load_company_data(file_path)
    results = []

    for _, row in df.iterrows():
        ticker = row["Ticker"]
        country = row["Country"]
        risk_free_rate = get_risk_free_rate(country)

        spot_price = get_spot_price(ticker, target_date)
        strike_price, option_price = get_option_price(ticker, target_date)

        results.append({
            "Ticker": ticker,
            "Date": target_date,
            "Spot Price": spot_price,
            "Strike Price (ATM)": strike_price,
            "Risk Free Rate": risk_free_rate,
            "Time To Maturity (1M)": 1 / 12,
            "Option Price": option_price
        })

    return pd.DataFrame(results)


def main():
    file_path = "/Users/tiffanyyu/Documents/Dissertation/0. Ticker.xlsx"
    target_date = datetime.date(2025, 5, 30)

    data = extract_option_data(file_path, target_date)
    data.to_excel("2. European Call Options.xlsx", index=False)
    print("Data saved to European Call Options.xlsx")


if __name__ == "__main__":
    main()
