"""Configuration for FX vol surface construction."""

# Standard FX tenors (in years)
TENORS = {
    "1W": 1/52, "2W": 2/52, "1M": 1/12, "2M": 2/12, "3M": 3/12,
    "6M": 6/12, "9M": 9/12, "1Y": 1.0, "18M": 1.5, "2Y": 2.0,
}

# Delta pillars for smile construction
DELTA_PILLARS = [0.10, 0.25, 0.50, 0.75, 0.90]  # put deltas mapped to call equiv

# SABR calibration defaults
SABR_DEFAULTS = {
    "beta": 0.5,           # fixed beta (FX convention)
    "alpha_init": 0.15,    # initial ATM vol guess
    "rho_init": -0.10,     # initial correlation guess
    "nu_init": 0.50,       # initial vol-of-vol guess
    "alpha_bounds": (0.001, 2.0),
    "rho_bounds": (-0.999, 0.999),
    "nu_bounds": (0.001, 5.0),
}

# G10 currency pairs for analysis
G10_PAIRS = [
    "EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCAD",
    "USDCHF", "NZDUSD", "EURGBP", "EURJPY", "GBPJPY",
]

# Premium-adjusted delta flag (standard for major pairs)
PREMIUM_ADJUSTED_DELTA = {
    "EURUSD": False, "USDJPY": False, "GBPUSD": False,
    "AUDUSD": True,  "USDCAD": False, "USDCHF": False,
    "NZDUSD": True,  "EURGBP": False, "EURJPY": False,
    "GBPJPY": False,
}
