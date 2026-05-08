"""
FX volatility market data processing.

Handles conversion from standard FX vol quoting conventions
(ATM-DNS, Risk Reversals, Butterflies) to strike-space implied vols.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from config import TENORS, G10_PAIRS


def delta_to_strike(delta: float, spot: float, rd: float, rf: float,
                    sigma: float, T: float, is_call: bool = True,
                    premium_adjusted: bool = False) -> float:
    """
    Convert option delta to strike price.

    For vanilla FX options:
        Δ_call = e^{-r_f T} N(d1)   [spot delta, non-premium-adjusted]
        K = S * exp(-d1*σ√T + (rd - rf + 0.5σ²)T)

    Parameters
    ----------
    delta : float – absolute delta value (e.g. 0.25)
    spot : float – current spot rate
    rd : float – domestic risk-free rate
    rf : float – foreign risk-free rate
    sigma : float – implied volatility
    T : float – time to expiry in years
    is_call : bool
    premium_adjusted : bool – True for premium-adjusted delta (AUD, NZD pairs)
    """
    sqrt_T = np.sqrt(T)
    sign = 1 if is_call else -1

    if premium_adjusted:
        # Iterative solve for premium-adjusted delta
        K = _solve_premium_adjusted_strike(delta, spot, rd, rf, sigma, T, is_call)
    else:
        # Spot delta: |Δ_call| = e^{-rf*T} * N(d1), |Δ_put| = e^{-rf*T} * N(-d1)
        # For calls: d1 = norm.ppf(delta * e^{rf*T})
        # For puts:  d1 = -norm.ppf(delta * e^{rf*T})
        d1 = sign * norm.ppf(delta * np.exp(rf * T))
        K = spot * np.exp(-d1 * sigma * sqrt_T + (rd - rf + 0.5 * sigma**2) * T)

    return K


def _solve_premium_adjusted_strike(delta, spot, rd, rf, sigma, T, is_call,
                                   tol=1e-10, max_iter=50):
    """Newton-Raphson for premium-adjusted delta → strike."""
    sign = 1 if is_call else -1
    sqrt_T = np.sqrt(T)

    # Initial guess from non-adjusted
    d1 = sign * norm.ppf(sign * delta * np.exp(rf * T))
    K = spot * np.exp(-d1 * sigma * sqrt_T + (rd - rf + 0.5 * sigma**2) * T)

    for _ in range(max_iter):
        d1 = (np.log(spot / K) + (rd - rf + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T
        # Premium-adjusted: Δ_pa = sign * (K/S) * e^{-rd*T} * N(sign * d2) ... [simplified]
        bs_delta = sign * np.exp(-rf * T) * norm.cdf(sign * d1)
        premium = (spot * np.exp(-rf * T) * norm.cdf(sign * d1)
                   - K * np.exp(-rd * T) * norm.cdf(sign * d2))
        pa_delta = bs_delta - sign * premium / spot if is_call else bs_delta + premium / spot
        err = pa_delta - sign * delta

        # Derivative w.r.t. K
        d_delta_dK = -sign * np.exp(-rd * T) * norm.pdf(d2) / (K * sigma * sqrt_T)
        K -= err / d_delta_dK

        if abs(err) < tol:
            break

    return K


def decode_vol_quotes(atm: float, rr25: float, bf25: float,
                      rr10: float = None, bf10: float = None) -> dict:
    """
    Convert market vol quotes to individual strike volatilities.

    ATM  = σ_ATM (delta-neutral straddle)
    RR25 = σ_25Δ_call - σ_25Δ_put
    BF25 = 0.5*(σ_25Δ_call + σ_25Δ_put) - σ_ATM

    Returns dict: {delta: implied_vol}
    """
    sigma_25c = atm + bf25 + 0.5 * rr25
    sigma_25p = atm + bf25 - 0.5 * rr25

    result = {
        0.50: atm,
        0.25: sigma_25c,   # 25Δ call
        -0.25: sigma_25p,  # 25Δ put (negative = put delta)
    }

    if rr10 is not None and bf10 is not None:
        sigma_10c = atm + bf10 + 0.5 * rr10
        sigma_10p = atm + bf10 - 0.5 * rr10
        result[0.10] = sigma_10c
        result[-0.10] = sigma_10p

    return result


def generate_synthetic_vol_surface(pair: str = "EURUSD",
                                   spot: float = 1.0850,
                                   rd: float = 0.045,
                                   rf: float = 0.035,
                                   seed: int = 42) -> pd.DataFrame:
    """
    Generate realistic synthetic FX vol quotes for a given currency pair.

    Creates market-like smile shapes with:
    - Term structure: slight backwardation or contango
    - Skew: negative for EURUSD (puts > calls), varies by pair
    - Wings: increasing butterfly with tenor (more convexity long-dated)
    """
    rng = np.random.default_rng(seed)

    # Base ATM term structure
    pair_vols = {
        "EURUSD": {"base_atm": 0.078, "skew": -0.008, "convexity": 0.004},
        "USDJPY": {"base_atm": 0.095, "skew": -0.015, "convexity": 0.006},
        "GBPUSD": {"base_atm": 0.085, "skew": -0.005, "convexity": 0.005},
        "AUDUSD": {"base_atm": 0.105, "skew": -0.012, "convexity": 0.007},
        "USDCAD": {"base_atm": 0.068, "skew": 0.003,  "convexity": 0.003},
    }

    params = pair_vols.get(pair, {"base_atm": 0.080, "skew": -0.006, "convexity": 0.004})

    rows = []
    for tenor_label, T in TENORS.items():
        # ATM increases slightly with tenor (typical term structure)
        atm = params["base_atm"] + 0.005 * np.sqrt(T) + rng.normal(0, 0.001)

        # Skew (RR) more pronounced short-dated, dampens with time
        rr25 = params["skew"] * (1 + 0.3 / np.sqrt(T + 0.1)) + rng.normal(0, 0.0005)
        rr10 = rr25 * 1.8 + rng.normal(0, 0.0003)

        # Butterfly increases with tenor (more convexity)
        bf25 = params["convexity"] * np.sqrt(T) + rng.normal(0, 0.0003)
        bf10 = bf25 * 2.5 + rng.normal(0, 0.0005)

        rows.append({
            "pair": pair, "tenor": tenor_label, "T": T,
            "spot": spot, "rd": rd, "rf": rf,
            "ATM": atm, "RR25": rr25, "BF25": bf25,
            "RR10": rr10, "BF10": bf10,
        })

    return pd.DataFrame(rows)


def build_strike_vol_grid(market_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert market quotes DataFrame to a strike/vol grid.

    For each tenor row, decode vol quotes and compute corresponding strikes.
    """
    records = []
    for _, row in market_df.iterrows():
        delta_vols = decode_vol_quotes(
            row["ATM"], row["RR25"], row["BF25"], row["RR10"], row["BF10"]
        )

        for delta, sigma in delta_vols.items():
            is_call = delta > 0
            abs_delta = abs(delta)
            K = delta_to_strike(
                abs_delta, row["spot"], row["rd"], row["rf"],
                sigma, row["T"], is_call=is_call
            )
            records.append({
                "tenor": row["tenor"], "T": row["T"],
                "delta": delta, "strike": K,
                "log_moneyness": np.log(K / row["spot"]),
                "implied_vol": sigma,
            })

    return pd.DataFrame(records)
