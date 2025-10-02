import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import minimize
from numpy import log, exp, sqrt, pi, real


def heston_characteristic_function(phi, s, k, t, r, kappa, theta,
                                   sigma, rho, v0, lambda_, p_num):
    x = log(s)
    a = kappa * theta
    u = 0.5 if p_num == 1 else -0.5
    b = kappa + lambda_ - rho * sigma if p_num == 1 else kappa + lambda_
    d = sqrt((rho * sigma * 1j * phi - b) ** 2 -
             sigma ** 2 * (2 * u * 1j * phi - phi ** 2))
    g = (b - rho * sigma * 1j * phi + d) / (b - rho * sigma * 1j * phi - d)

    c = (r * 1j * phi * t +
         (a / sigma ** 2) * ((b - rho * sigma * 1j * phi + d) * t -
         2 * log((1 - g * exp(d * t)) / (1 - g))))
    d_val = ((b - rho * sigma * 1j * phi + d) / sigma ** 2) * (
        (1 - exp(d * t)) / (1 - g * exp(d * t))
    )

    return exp(c + d_val * v0 + 1j * phi * x)


def heston_price_call(s, k, t, r, kappa, theta, sigma, rho, v0, lambda_=0):
    def integrand(phi, p_num):
        numerator = heston_characteristic_function(
            phi, s, k, t, r, kappa, theta, sigma, rho, v0, lambda_, p_num
        )
        return real(exp(-1j * phi * log(k)) * numerator / (1j * phi))

    p1 = 0.5 + (1 / pi) * quad(lambda phi: integrand(phi, 1), 0, 100)[0]
    p2 = 0.5 + (1 / pi) * quad(lambda phi: integrand(phi, 2), 0, 100)[0]

    return s * p1 - k * exp(-r * t) * p2


def calibration_error(params, s, k, t, r, market_prices):
    kappa, theta, sigma, rho, v0 = params
    errors = []

    for i in range(len(s)):
        model_price = heston_price_call(
            s=s[i], k=k[i], t=t[i], r=r[i],
            kappa=kappa, theta=theta, sigma=sigma, rho=rho, v0=v0
        )
        errors.append((model_price - market_prices[i]) ** 2)

    return np.sum(errors)


def calibrate_heston_parameters(s, k, t, r, market_prices):
    initial_params = [2.0, 0.04, 0.3, -0.7, 0.04]
    bounds = [
        (0.01, 10),     # kappa
        (0.0001, 0.5),  # theta
        (0.01, 1.0),    # sigma
        (-0.99, 0.0),   # rho
        (0.0001, 0.5)   # v0
    ]

    result = minimize(calibration_error, initial_params, args=(s, k, t, r, market_prices), bounds=bounds)

    return result


def load_and_prepare_data(file_path):
    df = pd.read_excel(file_path)
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.replace(r'\s+', ' ', regex=True)

    # Convert time to years if given in months
    df['Time to Maturity (Years)'] = df['Time To Maturity (1M)'] / 12

    return df


def main():
    file_path = '/Users/tiffanyyu/Documents/Dissertation/6. Heston Volatility/European Call Options.xlsx'
    df = load_and_prepare_data(file_path)

    # Extract required columns
    s = df['Spot Price'].values
    k = df['Strike Price (ATM)'].values
    t = df['Time to Maturity (Years)'].values
    r = df['Risk Free Rate'].values
    market_prices = df['Option Price'].values

    # Run calibration
    result = calibrate_heston_parameters(s, k, t, r, market_prices)

    if result.success:
        kappa, theta, sigma, rho, v0 = result.x
        print("Calibrated Heston Parameters:")
        print(f"κ (kappa): {kappa:.4f}")
        print(f"θ (theta): {theta:.4f}")
        print(f"σ (vol of vol): {sigma:.4f}")
        print(f"ρ (correlation): {rho:.4f}")
        print(f"ν₀ (initial variance): {v0:.4f}")
    else:
        print("Calibration failed:", result.message)


if __name__ == '__main__':
    main()
