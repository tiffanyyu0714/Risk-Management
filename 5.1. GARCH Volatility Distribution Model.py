import pandas as pd
from arch import arch_model


def load_stock_data(file_path: str) -> pd.DataFrame:
    """
    Load stock price data from an Excel file with the date as the index.
    """
    df = pd.read_excel(file_path, index_col=0, parse_dates=True)
    df.sort_index(inplace=True)
    return df


def calculate_annualised_garch_volatility(prices: pd.Series, distribution: str) -> float:
    """
    Calculate annualised GARCH(1,1) volatility using daily returns with given distribution.

    Parameters:
        prices (pd.Series): Daily stock prices
        distribution (str): Distribution type ('normal', 't', or 'skewt')

    Returns:
        float: Annualised volatility
    """
    returns = prices.pct_change().dropna() * 100
    model = arch_model(returns, vol='Garch', p=1, q=1, dist=distribution)
    result = model.fit(disp='off')
    forecast = result.forecast(horizon=1)
    daily_vol = forecast.variance.values[-1, 0] ** 0.5
    return daily_vol * (252 ** 0.5)


def main():
    file_path = '/Users/tiffanyyu/Documents/Dissertation/1. Daily Stock Prices/1. Daily Stock Prices.xlsx'
    stock_data = load_stock_data(file_path)

    end_date = stock_data.index.max()
    start_date = end_date - pd.DateOffset(years=2)

    results = []

    for ticker in stock_data.columns:
        try:
            prices = stock_data[ticker].loc[start_date:end_date].ffill().bfill()

            if len(prices.dropna()) < 252 * 1.5:
                print(f"[Skipped] {ticker}: Not enough data ({len(prices)} records)")
                continue

            row = {'Ticker': ticker}
            for dist in ['normal', 't', 'skewt']:
                try:
                    vol = calculate_annualised_garch_volatility(prices, distribution=dist)
                    row[f'{dist}'] = vol
                except Exception as e:
                    row[f'{dist}'] = None
                    print(f"[Error] {ticker} ({dist}): {e}")

            results.append(row)

        except Exception as error:
            print(f"[Error] {ticker}: {error}")

    results_df = pd.DataFrame(results)
    output_file = '1. GARCH Volatility Distribution Model.xlsx'
    results_df.to_excel(output_file, index=False)

    print(f"Results saved to '{output_file}'")


if __name__ == '__main__':
    main()
