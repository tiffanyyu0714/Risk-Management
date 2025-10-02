import pandas as pd
import yfinance as yf
import numpy as np


def load_tickers(file_path: str) -> list:
    """
    Load tickers from an Excel file (assumed to be in the second column).
    """
    df = pd.read_excel(file_path)
    return df.iloc[:, 1].dropna().unique().tolist()


def set_date_ranges(end_date_str: str = "2025-05-30") -> tuple:
    """
    Return start date (4 years back), cutoff date (2 years back), and end date.
    """
    end_date = pd.Timestamp(end_date_str)
    start_date = end_date - pd.DateOffset(years=4)
    cutoff_date = end_date - pd.DateOffset(years=2)
    return start_date, cutoff_date, end_date


def get_frequency_map() -> dict:
    """
    Define intervals and annualization factors for different frequencies.
    """
    return {
        "Daily": {"interval": "1d", "annualization": 252},
        "Weekly": {"interval": "1wk", "annualization": 52},
        "Bi-Weekly": {"interval": "1d", "annualization": 26},
        "Monthly": {"interval": "1mo", "annualization": 12},
        "Quarterly": {"interval": "1mo", "annualization": 4}
    }


def calculate_volatility(
    tickers: list,
    start_date: pd.Timestamp,
    cutoff_date: pd.Timestamp,
    end_date: pd.Timestamp,
    frequency_map: dict
) -> pd.DataFrame:
    """
    Calculate historical volatility over multiple frequencies for each ticker.
    """
    results = []

    for ticker in tickers:
        row = {"Ticker": ticker}

        for label, freq in frequency_map.items():
            try:
                data = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    interval=freq["interval"],
                    auto_adjust=False,
                    progress=False
                )

                if data.empty or "Adj Close" not in data.columns or data["Adj Close"].dropna().empty:
                    print(f"[Skipped] {ticker} ({label}): No valid data")
                    row[label] = None
                    continue

                data = data.loc[data.index >= cutoff_date]

                if label == "Bi-Weekly":
                    data = data.resample("2W-FRI").last()
                elif label == "Monthly":
                    data = data.resample("ME").last()
                elif label == "Quarterly":
                    data = data.resample("QE").last()

                returns = np.log(data["Adj Close"] / data["Adj Close"].shift(1)).dropna()

                if len(returns) >= 6:
                   vol = returns.std() * np.sqrt(freq["annualization"])
                   if isinstance(vol, pd.Series):
                       row[label] = round(float(vol.iloc[0]), 6)
                   else:
                       row[label] = round(float(vol), 6)

                else:
                    row[label] = None

            except Exception as e:
                print(f"[Error] {ticker} ({label}): {e}")
                row[label] = None

        results.append(row)

    return pd.DataFrame(results)


def export_to_excel(df: pd.DataFrame, output_path: str):
    """
    Save the volatility DataFrame to an Excel file.
    """
    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="Historical Volatility", index=False)
    print(f"Volatility data saved to '{output_path}'")


def main():
    file_path = "/Users/tiffanyyu/Documents/Dissertation/0. Ticker.xlsx"
    output_path = "1. Historical Volatility Data Frequency.xlsx"

    tickers = load_tickers(file_path)
    start_date, cutoff_date, end_date = set_date_ranges()
    frequency_map = get_frequency_map()

    vol_df = calculate_volatility(tickers, start_date, cutoff_date, end_date, frequency_map)

    columns = ["Ticker"] + list(frequency_map.keys())
    vol_df = vol_df[columns]

    export_to_excel(vol_df, output_path)


if __name__ == "__main__":
    main()
