import pandas as pd
import yfinance as yf
from datetime import datetime


def load_tickers_from_excel(file_path: str) -> list[str]:
    """
    Load ticker from Excel file.
    """
    df = pd.read_excel(file_path, header=None)
    tickers = df.iloc[1:, 1].dropna().astype(str).tolist()
    return tickers


def download_stock_data(tickers: list[str],
                        start_date: str,
                        end_date: str) -> pd.DataFrame:
    """
    Download historical adjusted close prices for a list of tickers.
    """
    all_data = []

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
                print(f"Warning: 'Adj Close' not found for {ticker}")
                continue

            adj_close = data[['Adj Close']].copy()
            adj_close.columns = [ticker]
            adj_close.index.name = 'Date'
            all_data.append(adj_close)

        except Exception as e:
            print(f"Error downloading {ticker}: {e}")

    if not all_data:
        return pd.DataFrame()

    return pd.concat(all_data, axis=1)


def save_data_to_excel(data: pd.DataFrame, output_path: str) -> None:
    """
    Clean, sort, and save the DataFrame to Excel file.
    """
    data.columns.name = None
    data = data.reset_index()
    data['Date'] = pd.to_datetime(data['Date']).dt.date
    data = data.sort_values(by='Date', ascending=False)
    data.to_excel(output_path, index=False)
    print(f"Data saved to {output_path}")


def main() -> None:
    """
    Main execution function to load tickers, download data, and save to Excel file.
    """
    file_path = "/Users/tiffanyyu/Documents/Dissertation/0. Ticker.xlsx"
    output_file = "/Users/tiffanyyu/Documents/Dissertation/1. Daily Stock Prices/1. Daily Stock Prices.xlsx"

    start_date = datetime(2021, 5, 30)
    end_date = datetime(2025, 5, 30)

    tickers = load_tickers_from_excel(file_path)
    stock_data = download_stock_data(
        tickers,
        start_date.strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d")
    )

    if stock_data.empty:
        print("No valid stock data collected.")
        return

    save_data_to_excel(stock_data, output_file)


if __name__ == "__main__":
    main()
