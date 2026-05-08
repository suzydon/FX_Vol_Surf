"""
FX Volatility Surface construction, interpolation, and no-arbitrage validation.

Constructs a smooth 2D surface σ(K, T) from calibrated SABR smiles,
with checks for calendar spread and butterfly arbitrage.
"""

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, RectBivariateSpline
from sabr import SABRParams, sabr_implied_vol, sabr_implied_vol_vec


class VolSurface:
    """
    Implied volatility surface for FX options.

    Supports:
    - Query σ(K, T) at arbitrary (strike, tenor) points
    - Total variance surface w(K, T) = σ²T for calendar spread checks
    - Local volatility extraction via Dupire's formula
    """

    def __init__(self, sabr_params_list: list[SABRParams],
                 forward_curve: dict, spot: float):
        """
        Parameters
        ----------
        sabr_params_list : list of SABRParams, one per tenor (sorted by T)
        forward_curve : dict tenor_label → forward price
        spot : current spot rate
        """
        self.params = sorted(sabr_params_list, key=lambda p: p.T)
        self.tenors = [p.T for p in self.params]
        self.tenor_labels = [p.tenor for p in self.params]
        self.forward_curve = forward_curve
        self.spot = spot

        # Build interpolation grid
        self._build_grid()

    def _build_grid(self, n_strikes: int = 50):
        """Pre-compute vol grid for fast 2D interpolation."""
        # Common log-moneyness range
        max_std = max(
            sabr_implied_vol(F, F, p.T, p.alpha, p.beta, p.rho, p.nu) * np.sqrt(p.T)
            for p, F in zip(self.params, [self.forward_curve.get(p.tenor, self.spot) for p in self.params])
        )
        self.log_m_grid = np.linspace(-3 * max_std, 3 * max_std, n_strikes)
        self.T_grid = np.array(self.tenors)

        # Compute vol at each grid point
        self.vol_grid = np.zeros((len(self.T_grid), n_strikes))
        for i, (p, T) in enumerate(zip(self.params, self.tenors)):
            F = self.forward_curve.get(p.tenor, self.spot)
            K_vec = F * np.exp(self.log_m_grid)
            self.vol_grid[i, :] = sabr_implied_vol_vec(K_vec, F, T, p.alpha, p.beta, p.rho, p.nu)

        # Total variance surface: w = σ² * T
        self.total_var_grid = self.vol_grid**2 * self.T_grid[:, np.newaxis]

    def implied_vol(self, K: float, T: float) -> float:
        """Query implied vol at arbitrary (strike, tenor) via interpolation."""
        F = self.spot  # simplified; should use forward for T
        log_m = np.log(K / F)

        # Find bracketing tenors
        if T <= self.tenors[0]:
            idx = 0
            p = self.params[idx]
            F_t = self.forward_curve.get(p.tenor, self.spot)
            return sabr_implied_vol(K, F_t, max(T, 1e-4), p.alpha, p.beta, p.rho, p.nu)
        elif T >= self.tenors[-1]:
            idx = -1
            p = self.params[idx]
            F_t = self.forward_curve.get(p.tenor, self.spot)
            return sabr_implied_vol(K, F_t, T, p.alpha, p.beta, p.rho, p.nu)

        # Linear interpolation in total variance space (flat forward variance)
        i = np.searchsorted(self.tenors, T) - 1
        T1, T2 = self.tenors[i], self.tenors[i + 1]
        w = (T - T1) / (T2 - T1)

        p1, p2 = self.params[i], self.params[i + 1]
        F1 = self.forward_curve.get(p1.tenor, self.spot)
        F2 = self.forward_curve.get(p2.tenor, self.spot)

        v1 = sabr_implied_vol(K, F1, T1, p1.alpha, p1.beta, p1.rho, p1.nu)
        v2 = sabr_implied_vol(K, F2, T2, p2.alpha, p2.beta, p2.rho, p2.nu)

        # Interpolate in total variance: σ²T = (1-w)*σ₁²T₁ + w*σ₂²T₂
        total_var = (1 - w) * v1**2 * T1 + w * v2**2 * T2
        return np.sqrt(total_var / T)

    def check_calendar_arbitrage(self) -> pd.DataFrame:
        """
        Check for calendar spread arbitrage violations.

        No-arbitrage requires total variance w(K,T) to be non-decreasing in T
        for all K. Returns DataFrame of violations.
        """
        violations = []
        for j in range(self.vol_grid.shape[1]):
            for i in range(1, len(self.tenors)):
                if self.total_var_grid[i, j] < self.total_var_grid[i - 1, j] - 1e-8:
                    violations.append({
                        "log_moneyness": self.log_m_grid[j],
                        "T_short": self.tenors[i - 1],
                        "T_long": self.tenors[i],
                        "w_short": self.total_var_grid[i - 1, j],
                        "w_long": self.total_var_grid[i, j],
                        "severity": self.total_var_grid[i - 1, j] - self.total_var_grid[i, j],
                    })

        return pd.DataFrame(violations)

    def check_butterfly_arbitrage(self) -> pd.DataFrame:
        """
        Check for butterfly arbitrage.

        The call price C(K) must be convex in K. Equivalent to checking
        that the probability density g(K) = e^{rT} * ∂²C/∂K² > 0.
        We check d²(total_var)/d(log_m)² + ... > 0 (simplified).
        """
        violations = []
        for i, T in enumerate(self.tenors):
            vols = self.vol_grid[i, :]
            K_vec = self.spot * np.exp(self.log_m_grid)

            # Numerical second derivative of call prices w.r.t. strike
            dk = np.diff(K_vec)
            # Use second differences of vol as proxy
            d2v = np.diff(vols, 2) / (dk[:-1] * dk[1:])

            for j, d2 in enumerate(d2v):
                if d2 < -0.5:  # significant non-convexity
                    violations.append({
                        "tenor": self.tenor_labels[i],
                        "log_moneyness": self.log_m_grid[j + 1],
                        "d2_vol": d2,
                    })

        return pd.DataFrame(violations)

    def local_vol(self, K: float, T: float, dK: float = 0.001,
                  dT: float = 0.001) -> float:
        """
        Compute Dupire local volatility at (K, T).

        σ_local² = (∂w/∂T) / (1 - (y/w)*(∂w/∂y) + 0.25*(-0.25 - 1/w + y²/w²)*(∂w/∂y)² + 0.5*(∂²w/∂y²))

        where w = σ²T is total variance and y = log(K/F).
        Simplified numerical implementation.
        """
        # Numerical derivatives
        v_up_T = self.implied_vol(K, T + dT)
        v_dn_T = self.implied_vol(K, max(T - dT, 1e-4))
        v_mid = self.implied_vol(K, T)

        w_up = v_up_T**2 * (T + dT)
        w_dn = v_dn_T**2 * max(T - dT, 1e-4)
        dw_dT = (w_up - w_dn) / (2 * dT)

        # Strike derivatives
        v_up_K = self.implied_vol(K + dK, T)
        v_dn_K = self.implied_vol(K - dK, T)
        v_up2_K = self.implied_vol(K + 2 * dK, T)
        v_dn2_K = self.implied_vol(K - 2 * dK, T)

        dv_dK = (v_up_K - v_dn_K) / (2 * dK)
        d2v_dK2 = (v_up_K - 2 * v_mid + v_dn_K) / (dK**2)

        # Dupire formula (simplified)
        numerator = dw_dT
        denominator = (1 + K * dv_dK / v_mid)**2 + K**2 * T * (d2v_dK2 - dv_dK**2)
        denominator = max(denominator, 1e-8)

        local_var = numerator / denominator
        return np.sqrt(max(local_var, 1e-8))

    def get_surface_dataframe(self, n_strikes: int = 30) -> pd.DataFrame:
        """Export the vol surface as a tidy DataFrame for visualization."""
        records = []
        for i, (p, T) in enumerate(zip(self.params, self.tenors)):
            F = self.forward_curve.get(p.tenor, self.spot)
            atm_vol = sabr_implied_vol(F, F, T, p.alpha, p.beta, p.rho, p.nu)
            std = atm_vol * np.sqrt(T)
            K_range = np.linspace(F * np.exp(-2.5 * std), F * np.exp(2.5 * std), n_strikes)
            vols = sabr_implied_vol_vec(K_range, F, T, p.alpha, p.beta, p.rho, p.nu)

            for K, v in zip(K_range, vols):
                records.append({
                    "tenor": p.tenor, "T": T,
                    "strike": K, "log_moneyness": np.log(K / F),
                    "implied_vol": v, "total_var": v**2 * T,
                })

        return pd.DataFrame(records)
