"""
FX Volatility Surface Construction & SABR Calibration
=====================================================

Entry point: generates synthetic FX vol data, calibrates SABR,
constructs the vol surface, and produces analytics/visualizations.

Usage:
    python main.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import os

from config import TENORS
from market_data import generate_synthetic_vol_surface, build_strike_vol_grid
from sabr import calibrate_sabr, sabr_smile, sabr_implied_vol, SABRParams
from vol_surface import VolSurface
from smile_analytics import (
    compute_greeks_surface, sticky_analysis, term_structure_analysis
)


def main():
    os.makedirs("outputs", exist_ok=True)

    # === 1. Generate synthetic market data ===
    pair = "EURUSD"
    spot = 1.0850
    rd, rf = 0.045, 0.035  # USD, EUR rates

    print(f"{'='*60}")
    print(f"FX Vol Surface: {pair} | Spot: {spot}")
    print(f"{'='*60}\n")

    market_df = generate_synthetic_vol_surface(pair, spot, rd, rf)
    print("Market vol quotes:")
    print(market_df[["tenor", "ATM", "RR25", "BF25", "RR10", "BF10"]].to_string(index=False))
    print()

    # === 2. Convert to strike/vol grid ===
    strike_vol_df = build_strike_vol_grid(market_df)

    # Forward curve (simplified: F = S * exp((rd - rf) * T))
    forward_curve = {}
    for _, row in market_df.iterrows():
        forward_curve[row["tenor"]] = spot * np.exp((rd - rf) * row["T"])

    # === 3. Calibrate SABR per tenor ===
    print("SABR Calibration Results:")
    print("-" * 75)
    print(f"{'Tenor':>6} {'α':>8} {'ρ':>8} {'ν':>8} {'RMSE(bps)':>10} {'MaxErr(bps)':>12}")
    print("-" * 75)

    sabr_params = []
    for tenor, group in strike_vol_df.groupby("tenor", sort=False):
        T = group["T"].iloc[0]
        F = forward_curve[tenor]
        strikes = group["strike"].values
        vols = group["implied_vol"].values

        idx = np.argsort(strikes)
        params = calibrate_sabr(strikes[idx], vols[idx], F, T, tenor_label=tenor)
        sabr_params.append(params)

        print(f"{tenor:>6} {params.alpha:8.4f} {params.rho:8.4f} {params.nu:8.4f} "
              f"{params.fit_rmse*10000:10.2f} {params.fit_max_err*10000:12.2f}")

    print()

    # === 4. Build vol surface ===
    surface = VolSurface(sabr_params, forward_curve, spot)

    # Calendar arbitrage check
    cal_violations = surface.check_calendar_arbitrage()
    if len(cal_violations) > 0:
        print(f"⚠ {len(cal_violations)} calendar spread violations detected")
    else:
        print("✓ No calendar spread arbitrage violations")

    bf_violations = surface.check_butterfly_arbitrage()
    if len(bf_violations) > 0:
        print(f"⚠ {len(bf_violations)} butterfly arbitrage violations detected")
    else:
        print("✓ No butterfly arbitrage violations")
    print()

    # === 5. Visualizations ===

    # 5a. Smile fits per tenor
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()
    selected_tenors = ["1M", "3M", "6M", "1Y", "18M", "2Y"]

    for ax, tenor in zip(axes, selected_tenors):
        group = strike_vol_df[strike_vol_df["tenor"] == tenor]
        p = next(pp for pp in sabr_params if pp.tenor == tenor)
        F = forward_curve[tenor]

        # Market points
        ax.scatter(group["strike"], group["implied_vol"] * 100,
                   color="red", s=50, zorder=5, label="Market")

        # SABR fit
        K_smooth, vol_smooth = sabr_smile(p, F)
        ax.plot(K_smooth, vol_smooth * 100, "b-", linewidth=1.5, label="SABR fit")

        ax.set_title(f"{tenor} (RMSE: {p.fit_rmse*10000:.1f} bps)")
        ax.set_xlabel("Strike")
        ax.set_ylabel("Implied Vol (%)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"{pair} SABR Smile Calibration", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("outputs/smile_fits.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 5b. 3D Volatility Surface
    surf_df = surface.get_surface_dataframe(n_strikes=40)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    for tenor in surface.tenor_labels:
        subset = surf_df[surf_df["tenor"] == tenor]
        T_val = subset["T"].iloc[0]
        ax.plot(subset["log_moneyness"], [T_val] * len(subset),
                subset["implied_vol"] * 100, alpha=0.7, linewidth=1.5)

    ax.set_xlabel("Log Moneyness")
    ax.set_ylabel("Time to Expiry (years)")
    ax.set_zlabel("Implied Vol (%)")
    ax.set_title(f"{pair} Implied Volatility Surface")
    ax.view_init(elev=25, azim=-60)
    plt.savefig("outputs/vol_surface_3d.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 5c. SABR parameter term structure
    ts_df = term_structure_analysis(sabr_params, forward_curve)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(ts_df["T"], ts_df["atm_vol"] * 100, "bo-")
    axes[0, 0].set_title("ATM Vol Term Structure")
    axes[0, 0].set_ylabel("ATM Vol (%)")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(ts_df["T"], ts_df["rho"], "rs-")
    axes[0, 1].set_title("ρ (Skew Parameter)")
    axes[0, 1].set_ylabel("ρ")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(ts_df["T"], ts_df["nu"], "g^-")
    axes[1, 0].set_title("ν (Vol-of-Vol)")
    axes[1, 0].set_ylabel("ν")
    axes[1, 0].set_xlabel("Tenor (years)")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(ts_df["T"], ts_df["skew_25d"] * 10000, "mo-", label="25Δ RR")
    axes[1, 1].plot(ts_df["T"], ts_df["butterfly_25d"] * 10000, "c^-", label="25Δ BF")
    axes[1, 1].set_title("Skew & Curvature Term Structure")
    axes[1, 1].set_ylabel("bps")
    axes[1, 1].set_xlabel("Tenor (years)")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(f"{pair} SABR Parameter Dynamics", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("outputs/param_term_structure.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 5d. Greeks heatmap (vanna/volga for 6M tenor)
    p_6m = next(pp for pp in sabr_params if pp.tenor == "6M")
    F_6m = forward_curve["6M"]
    greeks_df = compute_greeks_surface(p_6m, F_6m, rd, rf, n_strikes=40)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col, title in zip(axes,
                               ["vanna", "volga", "vega"],
                               ["Vanna (∂Vega/∂S)", "Volga (∂²V/∂σ²)", "Vega"]):
        ax.fill_between(greeks_df["log_moneyness"], greeks_df[col],
                        alpha=0.3, color="steelblue")
        ax.plot(greeks_df["log_moneyness"], greeks_df[col], "b-", linewidth=1.5)
        ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
        ax.set_title(f"6M {title}")
        ax.set_xlabel("Log Moneyness")
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"{pair} 6M Smile Greeks Profile", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("outputs/greeks_profile.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 5e. Sticky analysis
    print("Smile Dynamics (Sticky Analysis):")
    print("-" * 60)
    for p in sabr_params:
        F = forward_curve[p.tenor]
        result = sticky_analysis(p, F)
        print(f"  {result['tenor']:>6}: {result['regime']:<15} "
              f"(SS={result['sticky_strike_shift']*10000:.1f} bps, "
              f"SD={result['sticky_delta_shift']*10000:.1f} bps)")
    print()

    # === 6. Local vol extraction sample ===
    print("Local Vol Extraction (6M tenor):")
    print("-" * 40)
    F_6m = forward_curve["6M"]
    for m in [-0.02, -0.01, 0, 0.01, 0.02]:
        K = F_6m * np.exp(m)
        lv = surface.local_vol(K, 0.5)
        iv = surface.implied_vol(K, 0.5)
        print(f"  K={K:.4f} (m={m:+.2f}): local_vol={lv*100:.2f}% vs implied_vol={iv*100:.2f}%")

    print(f"\n✓ All outputs saved to outputs/")


if __name__ == "__main__":
    main()
