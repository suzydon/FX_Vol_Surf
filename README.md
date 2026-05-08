

# FX Volatility Surface Construction & SABR Calibration

A quantitative framework for constructing FX implied volatility surfaces, calibrating the SABR stochastic volatility model, and analyzing smile dynamics across G10 currency pairs.

## Overview

This project implements production-grade tools for FX volatility surface modeling:

- **Market Data Processing**: Parses standard FX vol quote conventions (ATM, 25Δ/10Δ Risk Reversals & Butterflies) into strike-space implied volatilities
- **SABR Model Calibration**: Calibrates the SABR (α, β, ρ, ν) parameters to market smiles using Levenberg-Marquardt optimization with regularization
- **Surface Interpolation**: Constructs smooth vol surfaces across strike and tenor dimensions using cubic spline and SVI parameterization
- **Smile Dynamics Analysis**: Analyzes sticky-strike vs sticky-delta behavior and tracks smile evolution over time
- **Risk Metrics**: Computes vega, vanna, and volga exposures from the calibrated surface

## FX Vol Quoting Conventions

FX options are quoted in delta-space using market conventions:
- **ATM DNS** (Delta-Neutral Straddle)
- **25Δ RR** (Risk Reversal = σ_25Δ_call - σ_25Δ_put)
- **25Δ BF** (Butterfly = 0.5*(σ_25Δ_call + σ_25Δ_put) - σ_ATM)
- **10Δ RR / 10Δ BF** for wing quotes

This project correctly handles the conversion from (ATM, RR, BF) quotes to individual strike vols.

## Project Structure

```
fx-vol-surface/
├── README.md
├── requirements.txt
├── config.py              # Model & market conventions config
├── market_data.py         # FX vol quote processing & synthetic data
├── sabr.py                # SABR model implementation & calibration
├── vol_surface.py         # Surface construction & interpolation
├── smile_analytics.py     # Smile dynamics & risk analysis
├── main.py                # Entry point: calibrate & visualize
└── outputs/               # Generated plots & reports
```

## Key Technical Details

### SABR Calibration
- Hagan et al. (2002) closed-form approximation for implied vol
- β fixed at 0.5 (log-normal/normal blend common in FX)
- Calibration via `scipy.optimize.least_squares` with parameter bounds
- Residual weighted by vega to prioritize liquid strikes

### Surface Construction
- Tenor interpolation via flat-forward variance
- Strike interpolation via cubic spline on delta-space vols
- No-arbitrage checks: calendar spread and butterfly constraints

## Usage

```bash
pip install -r requirements.txt
python main.py
```

## Sample Output

The framework produces:
1. Calibrated SABR parameters per tenor with fit quality metrics
2. 3D volatility surface visualization
3. Smile dynamics comparison across tenors
4. Greeks heatmaps (vanna/volga)


