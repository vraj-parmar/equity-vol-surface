"""
SVI (Stochastic Volatility Inspired) Model Calibration.

This module provides functions for fitting the 'Raw SVI' model to market
implied volatility data for a single expiration. The SVI model provides a
smooth, parametric representation of the volatility smile.
"""

from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import minimize, Bounds


def svi_raw(k: np.ndarray, a: float, b: float, rho: float, m: float, sigma: float) -> np.ndarray:
    """
    Calculates total variance using the Raw SVI formula.

    The SVI formula is defined as:
    w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))

    Args:
        k: Log-moneyness, defined as log(K/S).
        a: Controls the overall level of variance.
        b: Controls the slope of the smile (skew).
        rho: Controls the orientation of the smile.
        m: Controls the horizontal position of the smile.
        sigma: Controls the curvature (ATM convexity) of the smile.

    Returns:
        The total variance (w) for the given log-moneyness array.
    """
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))


def calibrate_svi(
    strikes: np.ndarray,
    ivs: np.ndarray,
    T: float,
    S: float,
) -> Tuple[Dict[str, float], List[float]]:
    """
    Fits the Raw SVI model to market data for a single expiration.

    This function takes market strikes and implied volatilities, converts them
    to log-moneyness and total variance, and then uses an optimization
    routine to find the SVI parameters that best fit the market data.

    Args:
        strikes: Array of option strike prices.
        ivs: Array of corresponding market implied volatilities.
        T: Time to expiration in years.
        S: Spot price of the underlying asset.

    Returns:
        A tuple containing:
        - A dictionary of the fitted SVI parameters (a, b, rho, m, sigma).
        - A list of the fitted implied volatilities corresponding to the input strikes.
    """
    # 1. Convert inputs to the SVI model's natural space
    k = np.log(strikes / S)
    market_total_variance = (ivs ** 2) * T

    # Objective function to be minimized (sum of squared errors)
    def objective_function(params: List[float]) -> float:
        a, b, rho, m, sigma = params
        model_total_variance = svi_raw(k, a, b, rho, m, sigma)
        
        # Calculate the squared error
        error = np.sum((model_total_variance - market_total_variance) ** 2)

        # 2. Add penalties for arbitrage constraints
        # Penalty to ensure b >= 0
        penalty = 0
        if b < 0:
            penalty += 1e6 * (b ** 2)
        # Penalty to ensure |rho| < 1
        if abs(rho) >= 1:
            penalty += 1e6 * (rho ** 2 - 1) ** 2
        # Penalty to ensure sigma > 0 (implicitly handled by bounds, but good practice)
        if sigma <= 0:
            penalty += 1e6 * (sigma ** 2)

        return error + penalty

    # Initial parameter guesses
    # These are empirical starting points and can be refined.
    initial_a = np.min(market_total_variance)
    initial_b = (market_total_variance[-1] - market_total_variance[0]) / (k[-1] - k[0]) if len(k) > 1 else 0.1
    initial_rho = -0.7
    initial_m = np.mean(k)
    initial_sigma = 0.1

    initial_params = [initial_a, initial_b, initial_rho, initial_m, initial_sigma]

    # Define bounds for the parameters to guide the optimizer
    # These help prevent the optimizer from exploring nonsensical parameter spaces.
    # Note: The penalty functions are still useful as a safeguard.
    bounds = Bounds(
        [-np.inf, 0, -0.999, -np.inf, 1e-4],  # Lower bounds [a, b, rho, m, sigma]
        [np.inf, np.inf, 0.999, np.inf, np.inf]    # Upper bounds
    )

    # 3. Use scipy.optimize.minimize to find the optimal parameters
    result = minimize(
        fun=objective_function,
        x0=initial_params,
        method='L-BFGS-B',  # A good quasi-Newton method that handles bounds
        bounds=bounds,
        options={'maxiter': 5000}
    )

    if not result.success:
        print(f"Warning: SVI calibration did not converge for T={T:.2f}. Reason: {result.message}")

    # Extract the fitted parameters
    a_fit, b_fit, rho_fit, m_fit, sigma_fit = result.x
    fitted_params = {
        "a": a_fit,
        "b": b_fit,
        "rho": rho_fit,
        "m": m_fit,
        "sigma": sigma_fit,
    }

    # 4. Return the fitted parameters and a list of 'fitted_ivs'
    fitted_total_variance = svi_raw(k, a_fit, b_fit, rho_fit, m_fit, sigma_fit)
    
    # Ensure variance is non-negative before taking the square root
    fitted_total_variance[fitted_total_variance < 0] = 0
    
    fitted_ivs = np.sqrt(fitted_total_variance / T).tolist()

    return fitted_params, fitted_ivs