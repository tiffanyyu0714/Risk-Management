import pandas as pd
from arch import arch_model


def load_stock_data(file_path: str) -> pd.DataFrame:
    """
    Load stock price data from an Excel file with the date as the index.
    """
    df = pd.read_excel(file_path, index_col=0, parse_dates=True)
    df.sort_index(inplace=True)
    return df


def calculate_annualised_garch_volatility(prices: pd.Series, p: int, q: int) -> float:
    """
    Calculate annualised GARCH(p,q) volatility using daily returns.

    Parameters:
        prices (pd.Series): Daily stock prices
        p (int): GARCH lag order
        q (int): ARCH lag order

    Returns:
        float: Annualised volatility
    """
    returns = prices.pct_change().dropna() * 100
    model = arch_model(returns, vol='Garch', p=p, q=q)
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
            prices = stock_data[ticker].loc[start_date:end_date].dropna()

            if len(prices) < 252 * 1.5:
                print(f"[Skipped] {ticker}: Not enough data ({len(prices)} records)")
                continue

            garch_1_1 = calculate_annualised_garch_volatility(prices, p=1, q=1)
            garch_2_1 = calculate_annualised_garch_volatility(prices, p=2, q=1)
            garch_2_2 = calculate_annualised_garch_volatility(prices, p=2, q=2)

            results.append({
                'Ticker': ticker,
                'GARCH(1,1)': garch_1_1,
                'GARCH(2,1)': garch_2_1,
                'GARCH(2,2)': garch_2_2
            })

        except Exception as error:
            print(f"[Error] {ticker}: {error}")

    results_df = pd.DataFrame(results)
    output_file = '1. GARCH Volatility GARCH Model.xlsx'
    results_df.to_excel(output_file, index=False)

    print(f"Results saved to '{output_file}'")


if __name__ == '__main__':
    main()
