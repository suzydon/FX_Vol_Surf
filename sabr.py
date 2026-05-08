"""
SABR stochastic volatility model for FX options.

Implements:
- Hagan et al. (2002) closed-form implied vol approximation
- Calibration via Levenberg-Marquardt with parameter bounds
- Smile interpolation and extrapolation

The SABR dynamics:
    dF = α * F^β * dW_1
    dα = ν * α * dW_2
    dW_1 * dW_2 = ρ * dt
"""

import numpy as np
from scipy.optimize import least_squares
from dataclasses import dataclass
from typing import Optional
from config import SABR_DEFAULTS


@dataclass
class SABRParams:
    """Calibrated SABR parameters for a single tenor."""
    alpha: float    # initial vol level
    beta: float     # CEV exponent (fixed)
    rho: float      # correlation F-vol
    nu: float       # vol-of-vol
    tenor: str = ""
    T: float = 0.0
    fit_rmse: float = 0.0
    fit_max_err: float = 0.0


def sabr_implied_vol(K: float, F: float, T: float,
                     alpha: float, beta: float, rho: float, nu: float) -> float:
    """
    Hagan et al. (2002) SABR implied volatility approximation.

    Parameters
    ----------
    K : float – strike
    F : float – forward price
    T : float – time to expiry
    alpha, beta, rho, nu : SABR parameters

    Returns
    -------
    Implied Black volatility
    """
    if abs(F - K) < 1e-12:
        # ATM limit
        FK_mid = F
        logFK = 0.0
    else:
        FK_mid = (F * K) ** ((1 - beta) / 2)
        logFK = np.log(F / K)

    # Absorb the beta into effective parameters
    one_minus_beta = 1 - beta
    FK_beta = (F * K) ** (one_minus_beta / 2)

    # z and x(z) for the smile correction
    z = (nu / alpha) * FK_beta * logFK
    if abs(z) < 1e-12:
        xz = 1.0
    else:
        sqrt_term = np.sqrt(1 - 2 * rho * z + z**2)
        xz = z / np.log((sqrt_term + z - rho) / (1 - rho))

    # Denominator corrections
    denom1 = FK_beta * (
        1 + one_minus_beta**2 / 24 * logFK**2
        + one_minus_beta**4 / 1920 * logFK**4
    )

    # Numerator correction (time-dependent)
    term1 = one_minus_beta**2 / 24 * alpha**2 / (FK_beta**2)
    term2 = 0.25 * rho * beta * nu * alpha / FK_beta
    term3 = (2 - 3 * rho**2) / 24 * nu**2
    numer_corr = 1 + (term1 + term2 + term3) * T

    sigma = (alpha / denom1) * xz * numer_corr

    return max(sigma, 1e-6)


def sabr_implied_vol_vec(K_vec: np.ndarray, F: float, T: float,
                         alpha: float, beta: float, rho: float, nu: float) -> np.ndarray:
    """Vectorized SABR implied vol computation."""
    return np.array([sabr_implied_vol(K, F, T, alpha, beta, rho, nu) for K in K_vec])


def calibrate_sabr(strikes: np.ndarray, market_vols: np.ndarray,
                   F: float, T: float,
                   beta: float = None,
                   weights: Optional[np.ndarray] = None,
                   tenor_label: str = "") -> SABRParams:
    """
    Calibrate SABR parameters (α, ρ, ν) to market smile.

    Uses Levenberg-Marquardt optimization minimizing weighted
    squared vol differences.

    Parameters
    ----------
    strikes : array – option strikes
    market_vols : array – market implied vols at each strike
    F : float – forward price
    T : float – time to expiry
    beta : float – fixed CEV exponent (default from config)
    weights : array – calibration weights (default: vega-like weighting)
    tenor_label : str – label for reporting

    Returns
    -------
    SABRParams with calibrated parameters and fit quality
    """
    if beta is None:
        beta = SABR_DEFAULTS["beta"]

    n = len(strikes)
    if weights is None:
        # Vega-like weighting: emphasize ATM, less weight on wings
        moneyness = np.abs(np.log(strikes / F))
        weights = np.exp(-2 * moneyness)
        weights /= weights.sum()

    # Initial guess
    x0 = np.array([
        SABR_DEFAULTS["alpha_init"],
        SABR_DEFAULTS["rho_init"],
        SABR_DEFAULTS["nu_init"],
    ])

    # Use ATM vol as better alpha initial guess
    atm_idx = np.argmin(np.abs(strikes - F))
    x0[0] = market_vols[atm_idx] * (F ** (1 - beta))

    def residuals(x):
        alpha, rho, nu = x
        model_vols = sabr_implied_vol_vec(strikes, F, T, alpha, beta, rho, nu)
        return np.sqrt(weights) * (model_vols - market_vols)

    bounds_lower = [SABR_DEFAULTS["alpha_bounds"][0],
                    SABR_DEFAULTS["rho_bounds"][0],
                    SABR_DEFAULTS["nu_bounds"][0]]
    bounds_upper = [SABR_DEFAULTS["alpha_bounds"][1],
                    SABR_DEFAULTS["rho_bounds"][1],
                    SABR_DEFAULTS["nu_bounds"][1]]

    result = least_squares(
        residuals, x0,
        bounds=(bounds_lower, bounds_upper),
        method='trf',
        ftol=1e-12, xtol=1e-12, gtol=1e-12,
        max_nfev=1000,
    )

    alpha_cal, rho_cal, nu_cal = result.x

    # Compute fit quality
    model_vols = sabr_implied_vol_vec(strikes, F, T, alpha_cal, beta, rho_cal, nu_cal)
    errors = model_vols - market_vols
    rmse = np.sqrt(np.mean(errors**2))
    max_err = np.max(np.abs(errors))

    return SABRParams(
        alpha=alpha_cal, beta=beta, rho=rho_cal, nu=nu_cal,
        tenor=tenor_label, T=T,
        fit_rmse=rmse, fit_max_err=max_err,
    )


def calibrate_surface(strike_vol_df, forward_curve: dict) -> list[SABRParams]:
    """
    Calibrate SABR to each tenor slice of the vol surface.

    Parameters
    ----------
    strike_vol_df : DataFrame with columns [tenor, T, strike, implied_vol]
    forward_curve : dict mapping tenor → forward price

    Returns
    -------
    List of SABRParams, one per tenor
    """
    results = []
    for tenor, group in strike_vol_df.groupby("tenor", sort=False):
        T = group["T"].iloc[0]
        F = forward_curve.get(tenor, group["strike"].median())
        strikes = group["strike"].values
        vols = group["implied_vol"].values

        # Sort by strike
        sort_idx = np.argsort(strikes)
        strikes = strikes[sort_idx]
        vols = vols[sort_idx]

        params = calibrate_sabr(strikes, vols, F, T, tenor_label=tenor)
        results.append(params)

    return results


def sabr_smile(params: SABRParams, F: float,
               K_range: Optional[np.ndarray] = None,
               n_points: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a smooth SABR smile curve from calibrated parameters.

    Returns (strikes, implied_vols) arrays for plotting.
    """
    if K_range is None:
        # ±3 standard deviations in log-moneyness
        atm_vol = sabr_implied_vol(F, F, params.T, params.alpha, params.beta, params.rho, params.nu)
        std = atm_vol * np.sqrt(params.T)
        K_range = np.linspace(F * np.exp(-3 * std), F * np.exp(3 * std), n_points)

    vols = sabr_implied_vol_vec(K_range, F, params.T, params.alpha, params.beta, params.rho, params.nu)
    return K_range, vols
