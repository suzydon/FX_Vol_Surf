"""
Smile dynamics analysis and vol-surface risk metrics.

Computes vanna, volga exposures and analyzes sticky-strike vs sticky-delta
regime behavior.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from sabr import SABRParams, sabr_implied_vol, sabr_implied_vol_vec


def compute_greeks_surface(params: SABRParams, F: float, rd: float, rf: float,
                           n_strikes: int = 30) -> pd.DataFrame:
    """
    Compute Black-Scholes Greeks augmented with smile-adjusted vanna and volga.

    Vanna = ∂Δ/∂σ = ∂Vega/∂S  (cross-gamma, key for RR hedging)
    Volga = ∂Vega/∂σ = ∂²V/∂σ² (key for BF hedging)
    """
    T = params.T
    sqrt_T = np.sqrt(T)
    atm_vol = sabr_implied_vol(F, F, T, params.alpha, params.beta, params.rho, params.nu)
    std = atm_vol * sqrt_T

    K_range = np.linspace(F * np.exp(-2.5 * std), F * np.exp(2.5 * std), n_strikes)
    records = []

    for K in K_range:
        sigma = sabr_implied_vol(K, F, T, params.alpha, params.beta, params.rho, params.nu)
        d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T

        # Standard Greeks
        vega = F * np.exp(-rf * T) * norm.pdf(d1) * sqrt_T
        gamma = np.exp(-rf * T) * norm.pdf(d1) / (F * sigma * sqrt_T)

        # Vanna: ∂vega/∂S = -e^{-rf*T} * N'(d1) * d2 / (S * σ * √T) * S ... simplified
        vanna = -np.exp(-rf * T) * norm.pdf(d1) * d2 / sigma

        # Volga: ∂vega/∂σ = vega * d1 * d2 / σ
        volga = vega * d1 * d2 / sigma

        # Charm: ∂Δ/∂T
        charm = -np.exp(-rf * T) * norm.pdf(d1) * (
            rf - d2 * sigma / (2 * sqrt_T * T)
        ) if T > 0.01 else 0.0

        records.append({
            "strike": K,
            "log_moneyness": np.log(K / F),
            "implied_vol": sigma,
            "vega": vega,
            "gamma": gamma,
            "vanna": vanna,
            "volga": volga,
            "charm": charm,
        })

    return pd.DataFrame(records)


def sticky_analysis(params: SABRParams, F: float, spot_shift: float = 0.01,
                    n_points: int = 50) -> dict:
    """
    Analyze sticky-strike vs sticky-delta behavior of SABR smile.

    Sticky-strike: σ(K) unchanged when spot moves → smile stays fixed in K-space
    Sticky-delta: σ(Δ) unchanged when spot moves → smile shifts with spot

    SABR naturally produces sticky-delta behavior for β=1, sticky-strike for β=0.

    Returns metrics comparing the two regimes.
    """
    T = params.T
    atm_vol = sabr_implied_vol(F, F, T, params.alpha, params.beta, params.rho, params.nu)
    std = atm_vol * np.sqrt(T)

    K_range = np.linspace(F * np.exp(-2 * std), F * np.exp(2 * std), n_points)

    # Base smile
    vols_base = sabr_implied_vol_vec(K_range, F, T, params.alpha, params.beta, params.rho, params.nu)

    # Shifted forward
    F_up = F * (1 + spot_shift)
    vols_shifted = sabr_implied_vol_vec(K_range, F_up, T, params.alpha, params.beta, params.rho, params.nu)

    # Sticky-strike: vol at same K should be unchanged
    ss_metric = np.mean(np.abs(vols_shifted - vols_base))

    # Sticky-delta: vol at same moneyness should be unchanged
    K_shifted = K_range * (1 + spot_shift)  # shift strikes with spot
    vols_sd = sabr_implied_vol_vec(K_shifted, F_up, T, params.alpha, params.beta, params.rho, params.nu)
    sd_metric = np.mean(np.abs(vols_sd - vols_base))

    return {
        "tenor": params.tenor,
        "beta": params.beta,
        "sticky_strike_shift": ss_metric,
        "sticky_delta_shift": sd_metric,
        "regime": "sticky-delta" if sd_metric < ss_metric else "sticky-strike",
        "ratio": sd_metric / max(ss_metric, 1e-10),
    }


def term_structure_analysis(params_list: list[SABRParams],
                            forward_curve: dict) -> pd.DataFrame:
    """
    Analyze SABR parameter dynamics across the term structure.

    Tracks how α, ρ, ν evolve with maturity – important for
    understanding skew and curvature term structure.
    """
    records = []
    for p in sorted(params_list, key=lambda x: x.T):
        F = forward_curve.get(p.tenor, 1.0)
        atm_vol = sabr_implied_vol(F, F, p.T, p.alpha, p.beta, p.rho, p.nu)

        # Skew at 25-delta (approximate)
        K_25d = F * np.exp(-0.674 * atm_vol * np.sqrt(p.T))  # ≈ 25Δ put
        K_25c = F * np.exp(0.674 * atm_vol * np.sqrt(p.T))   # ≈ 25Δ call
        skew_25d = (sabr_implied_vol(K_25c, F, p.T, p.alpha, p.beta, p.rho, p.nu)
                    - sabr_implied_vol(K_25d, F, p.T, p.alpha, p.beta, p.rho, p.nu))

        # Curvature (butterfly)
        bf_25d = (0.5 * (sabr_implied_vol(K_25c, F, p.T, p.alpha, p.beta, p.rho, p.nu)
                         + sabr_implied_vol(K_25d, F, p.T, p.alpha, p.beta, p.rho, p.nu))
                  - atm_vol)

        records.append({
            "tenor": p.tenor, "T": p.T,
            "alpha": p.alpha, "rho": p.rho, "nu": p.nu,
            "atm_vol": atm_vol,
            "skew_25d": skew_25d,
            "butterfly_25d": bf_25d,
            "fit_rmse_bps": p.fit_rmse * 10000,
        })

    return pd.DataFrame(records)
