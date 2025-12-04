"""
Core Black-Scholes-Merton pricing model and Greek calculations.

This module provides functions for:
1. Calculating option prices using the Black-Scholes-Merton formula.
2. Calculating the first-order Greeks (Delta, Gamma, Theta, Vega, Rho).
3. Calculating implied volatility from a market price using the Newton-Raphson method.
"""

import numpy as np
from scipy.stats import norm
from numba import jit
from typing import Dict, Literal

OptionType = Literal["call", "put"]

@jit(nopython=True, cache=True)
def _calculate_d1_d2(
    S: float, K: float, T: float, r: float, q: float, sigma: float
) -> tuple[float, float]:
    """
    Calculates the d1 and d2 terms used in the Black-Scholes-Merton model.

    Args:
        S: Spot price of the underlying asset.
        K: Strike price of the option.
        T: Time to expiration in years.
        r: Risk-free interest rate.
        q: Annual dividend yield of the underlying asset.
        sigma: Volatility of the underlying asset.

    Returns:
        A tuple containing d1 and d2.
    """
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2

def black_scholes_price(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    option_type: OptionType,
) -> float:
    """
    Calculates the price of a European option using the Black-Scholes-Merton formula.

    Args:
        S: Spot price of the underlying asset.
        K: Strike price of the option.
        T: Time to expiration in years.
        r: Risk-free interest rate.
        q: Annual dividend yield.
        sigma: Volatility of the underlying asset.
        option_type: Type of the option ('call' or 'put').

    Returns:
        The theoretical price of the option.
    """
    if T <= 0 or sigma <= 0:
        # Handle expired or invalid options
        if option_type == "call":
            return max(0.0, S - K)
        else: # put
            return max(0.0, K - S)
            
    d1, d2 = _calculate_d1_d2(S, K, T, r, q, sigma)

    if option_type == "call":
        price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type. Must be 'call' or 'put'.")

    return price

def calculate_greeks(
    S: float,
    K: float,
    T: float,
    sigma: float,
    option_type: OptionType,
    r: float = 0.04,
    q: float = 0.015,
) -> Dict[str, float]:
    """
    Calculates the Greeks for a European option.

    Args:
        S: Spot price of the underlying asset.
        K: Strike price of the option.
        T: Time to expiration in years.
        sigma: Volatility of the underlying asset.
        option_type: Type of the option ('call' or 'put').
        r: Risk-free interest rate (default: 4.0%).
        q: Annual dividend yield (default: 1.5%).

    Returns:
        A dictionary containing Delta, Gamma, Theta, Vega, and Rho.
    """
    if T <= 0 or sigma <= 0:
        return {
            "Delta": 0.0, "Gamma": 0.0, "Theta": 0.0, "Vega": 0.0, "Rho": 0.0
        }

    d1, d2 = _calculate_d1_d2(S, K, T, r, q, sigma)
    pdf_d1 = norm.pdf(d1)

    # Shared Greeks
    gamma = np.exp(-q * T) * pdf_d1 / (S * sigma * np.sqrt(T))
    vega = S * np.exp(-q * T) * pdf_d1 * np.sqrt(T) / 100 # Per 1% change in vol

    if option_type == "call":
        delta = np.exp(-q * T) * norm.cdf(d1)
        theta = (- (S * sigma * np.exp(-q * T) * pdf_d1) / (2 * np.sqrt(T))
                 - r * K * np.exp(-r * T) * norm.cdf(d2)
                 + q * S * np.exp(-q * T) * norm.cdf(d1)) / 365 # Per day
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100 # Per 1% change in r
    elif option_type == "put":
        delta = np.exp(-q * T) * (norm.cdf(d1) - 1)
        theta = (- (S * sigma * np.exp(-q * T) * pdf_d1) / (2 * np.sqrt(T))
                 + r * K * np.exp(-r * T) * norm.cdf(-d2)
                 - q * S * np.exp(-q * T) * norm.cdf(-d1)) / 365 # Per day
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100 # Per 1% change in r
    else:
        raise ValueError("Invalid option type. Must be 'call' or 'put'.")

    return {
        "Delta": delta,
        "Gamma": gamma,
        "Theta": theta,
        "Vega": vega,
        "Rho": rho,
    }

def get_implied_vol(
    market_price: float,
    S: float,
    K: float,
    T: float,
    option_type: OptionType,
    r: float = 0.04,
    q: float = 0.015,
    tol: float = 1e-5,
    max_iter: int = 100,
) -> float:
    """
    Calculates the implied volatility for a European option using the Newton-Raphson method.

    Args:
        market_price: The observed market price of the option.
        S: Spot price of the underlying asset.
        K: Strike price of the option.
        T: Time to expiration in years.
        option_type: Type of the option ('call' or 'put').
        r: Risk-free interest rate (default: 4.0%).
        q: Annual dividend yield (default: 1.5%).
        tol: Tolerance for the convergence of the solver.
        max_iter: Maximum number of iterations for the solver.

    Returns:
        The implied volatility (sigma), or np.nan if the solver fails to converge.
    """
    # Initial guess for sigma
    sigma = np.sqrt(2 * abs((np.log(S / K) + (r - q) * T) / T))

    for _ in range(max_iter):
        try:
            price = black_scholes_price(S, K, T, r, q, sigma, option_type)
            greeks = calculate_greeks(S, K, T, sigma, option_type, r, q)
            vega = greeks["Vega"] * 100 # Rescale vega back

            if vega < 1e-8: # Vega is too small, solver will be unstable
                return np.nan

            diff = price - market_price
            if abs(diff) < tol:
                return sigma

            sigma = sigma - diff / vega

        except (ValueError, ZeroDivisionError, OverflowError):
            return np.nan # Return NaN on any calculation error

    return np.nan # Failed to converge within max_iter