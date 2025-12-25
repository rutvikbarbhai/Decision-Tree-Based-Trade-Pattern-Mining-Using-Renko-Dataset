"""
Phase 1 tsfresh Feature Extraction
===========================================================
Original problem:
- Window loop created separate rows for w5 and w10 features
- Trades ≥10 appeared twice (once for each window)
- Result: 2x rows, duplicate symbol-dates

 approach:
- Process each trade once
- Combine w5 and w10 features into single row
- No duplicates!

Usage:
   python module2_tsfresh_features_v3.py --type long --ticker BMY --config ./config.json --test basic_ind_osc

"""
from core.settings import Settings
from data.storage import WasabiStorageSystem
import json
import argparse
from datetime import datetime
import gc
from typing import Optional
import numpy as np
import pandas as pd
import warnings
pd.set_option('display.max_columns', None)

warnings.filterwarnings("ignore")
warnings.filterwarnings('ignore')

settings = Settings()

# tsfresh imports
try:
    from tsfresh import extract_features
    from tsfresh.feature_extraction import MinimalFCParameters
    from tsfresh.utilities.dataframe_functions import impute
    TSFRESH_AVAILABLE = True
except ImportError:
    print("ERROR: tsfresh not installed. Install with: pip install tsfresh")
    exit(1)

exclude_cols = [
    'trade_idx', 'name', 'symbol', 'signal', 'strategy',
    'entry_date', 'exit_date', 'entry_cost', 'exit_cost',
    'pnl', 'label', 'max_loss', 'max_profit', 'margin',
    'rrr', 'return', 'minutes', 'spot_return', 'credit',
    'no_bricks', 'brick_size', 'jump_size', 'reversal_bricks',
    'bricks_duration', 'no_bricks_no_delay', 'jump_size_no_delay',
    'reversal_bricks_no_delay', 'bricks_duration_no_delay',
    'desc', 'delay_loss', 'quadrant', 'open_cost', 'close_cost',
    'days', 'weekly_return', 'daily_return', 'weekly_spot_return',
    'daily_spot_return', 'hit_ratio', 'open_date', 'close_date',
    'pos_pnl', 'neg_pnl', 'total_pnl', 'drawdown',
    'theoretical_pnl', 'negative_contribution',
    'pnl_without_negative_contribution', 'category',
    'datetime', 'adjusted_close', 'Unnamed: 0'
]


def apply_dynamic_features(df: pd.DataFrame, formulas: list[tuple[str, str]]) -> pd.DataFrame:
    """
    Applies 'name = expr' formulas with DataFrame.eval(engine='python').
    Supports numpy via @np.
    """
    if not formulas:
        print("ℹ No dynamic base features provided.")
        return df

    print(f"Applying {len(formulas)} base feature formulas...")
    for name, expr in formulas:
        try:
            df[name] = eval(expr, {"np": np}, df.to_dict("series"))
            # print(f"   {name}")
        except Exception as e:
            print(f"   {name} = {expr}  -> {e}")
    return df


def load_pipeline_config(config_path: str):
    with open(config_path, "r") as f:
        cfg = json.load(f)

    base_raw = cfg.get("BASE_FEATURES", {})
    metrics = cfg.get("TSFRESH_METRICS_JSON", {})

    formulas = [(name, expr) for name, expr in base_raw.items()]
    return formulas, metrics

# ============================================================
# INDICATOR REGISTRY
# ============================================================


INDICATOR_REGISTRY = {}


def register_indicator(name):

    """Decorator to register indicator builders."""
    def decorator(func):
        INDICATOR_REGISTRY[name.upper()] = func
        return func
    return decorator


# ============================================================
# HELPER — PAST-ONLY ZSCORE
# ============================================================
def zscore_past(s, roll=20):
    """Pure past-only zscore. Value at t uses info <= t-1."""
    s_shift = s.shift(1)
    mu = s_shift.rolling(roll, min_periods=5).mean()
    sd = s_shift.rolling(roll, min_periods=5).std()
    return (s_shift - mu) / (sd + 1e-9)


# ============================================================
# INDICATOR 1 — CPWO (unchanged, validated, safe)
# ============================================================
@register_indicator("CPWO")
def build_cpwo(df, window=20, smooth=5):
    df = df.copy()
    required = {"max_call_wall_strike", "min_put_wall_strike"}
    if not required.issubset(df.columns):
        print("CPWO skipped — missing columns:", required - set(df.columns))
        return df, {}

    existing = set(df.columns)

    cw = df["max_call_wall_strike"].shift(1)
    pw = df["min_put_wall_strike"].shift(1)

    dcall = cw.diff()
    dput = pw.diff()
    wall_dist_p = cw - pw
    dwall = wall_dist_p.diff()

    bias = (np.sign(dcall) - np.sign(dput)) / 2
    denom = wall_dist_p.shift(1).replace(0, np.nan).fillna(1e-9)
    compression_rate = dwall / (denom + 1e-9)

    compression = -np.sign(dwall) * np.log1p(np.abs(compression_rate + 1e-12))
    raw_osc = (bias * compression).fillna(0)

    # normalizer (past-only)
    min_roll = raw_osc.rolling(window, min_periods=1).min()
    max_roll = raw_osc.rolling(window, min_periods=1).max()
    norm = (max_roll - min_roll).replace(0, np.nan).fillna(1e-9)

    df["CPWO"] = 100 + 100 * (raw_osc - min_roll) / norm
    df["CPWO_slope"] = df["CPWO"] - df["CPWO"].shift(1)
    df["CPWO_direction"] = np.sign(df["CPWO_slope"])
    df["CPWO_signal"] = df["CPWO"].rolling(smooth, min_periods=1).mean()
    df["CPWO_hist"] = df["CPWO"] - df["CPWO_signal"]

    order = 3
    prev_max = df["CPWO"].shift(1).rolling(order).max()
    prev_min = df["CPWO"].shift(1).rolling(order).min()
    df["CPWO_peak"] = df["CPWO"] > prev_max
    df["CPWO_trough"] = df["CPWO"] < prev_min

    # safe crosses
    df["CPWO_cross_up"] = (df["CPWO"].shift(1) < df["CPWO_signal"].shift(1)) & (df["CPWO"] >= df["CPWO_signal"])
    df["CPWO_cross_down"] = (df["CPWO"].shift(1) > df["CPWO_signal"].shift(1)) & (df["CPWO"] <= df["CPWO_signal"])

    df["cross_up_100"] = (df["CPWO"].shift(1) < 100) & (df["CPWO"] >= 100)
    df["cross_down_100"] = (df["CPWO"].shift(1) > 100) & (df["CPWO"] <= 100)
    df["cross_up_150"] = (df["CPWO"].shift(1) < 150) & (df["CPWO"] >= 150)
    df["cross_down_50"] = (df["CPWO"].shift(1) > 50) & (df["CPWO"] <= 50)

    new_cols = [c for c in df.columns if c not in existing]
    return df, {"CPWO": new_cols}


# ============================================================
# INDICATOR 2 — GSM (unchanged but validated)
# ============================================================
@register_indicator("GSM")
def build_gsm(df, roll=20):
    df = df.copy()
    # added below b/c entry_date missing
    if 'entry_cost' not in df.columns:
        df['entry_cost'] = df['adjusted_close']
    required = {"entry_cost", "gamma_flip_strike", "net_gex"}
    if not required.issubset(df.columns):
        print("GSM skipped — missing:", required - set(df.columns))
        return df, {}

    existing = set(df.columns)

    ec = df["entry_cost"].shift(1)
    gf = df["gamma_flip_strike"].shift(1)
    gex = df["net_gex"].shift(1)

    df["gamma_flip_perc"] = 100 * (ec - gf) / (gf + 1e-9)

    df["delta_gex"] = gex.diff()
    delta_p = df["delta_gex"].shift(1)

    mu = delta_p.rolling(roll, min_periods=5).mean()
    sd = delta_p.rolling(roll, min_periods=5).std().replace(0, 1e-9)
    df["delta_gex_z"] = (delta_p - mu) / (sd + 1e-9)

    df["quad_weight"] = np.tanh(np.abs(df["gamma_flip_perc"]) / 2)
    df["regime_sign"] = np.sign(gex)

    df["GSM_raw"] = df["quad_weight"] * df["regime_sign"] * df["delta_gex_z"]
    df["GSM"] = np.tanh(df["GSM_raw"].ewm(span=5).mean())

    def classify(row):
        if row["gamma_flip_perc"] >= 0 and row["regime_sign"] < 0:
            return "Q1_AboveFlip_ShortGamma"
        elif row["gamma_flip_perc"] >= 0 and row["regime_sign"] >= 0:
            return "Q2_AboveFlip_LongGamma"
        elif row["gamma_flip_perc"] < 0 and row["regime_sign"] >= 0:
            return "Q3_BelowFlip_LongGamma"
        else:
            return "Q4_BelowFlip_ShortGamma"

    df["Quadrant"] = df.apply(classify, axis=1)
    df["Market_State"] = np.select(
        [df["GSM"] > 0.3, df["GSM"] < -0.3],
        ["Stable_MeanReverting", "Volatility_Expanding"],
        default="Transition"
    )

    new_cols = [c for c in df.columns if c not in existing]
    return df, {"GSM": new_cols}


# ============================================================
# INDICATOR 3 — STRUCTURAL_ENRICH (validated)
# ============================================================
@register_indicator("STRUCTURAL_ENRICH")
def build_structural_enrichment(df):
    df = df.copy()
    required = {"gamma_flip_strike", "max_call_wall_strike", "min_put_wall_strike"}
    if not required.issubset(df.columns):
        print("STRUCTURAL_ENRICH skipped — missing:", required - set(df.columns))
        return df, {}

    existing = set(df.columns)

    gf = df["gamma_flip_strike"].shift(1)
    cw = df["max_call_wall_strike"].shift(1)
    pw = df["min_put_wall_strike"].shift(1)

    df["wall_distance"] = (cw - pw).fillna(0)
    df["gammaflip_x_walldist"] = (gf * df["wall_distance"]).fillna(0)

    df["gamma_flip_zscore"] = zscore_past(df["gamma_flip_strike"], roll=20)
    df["gamma_flip_momentum"] = gf - gf.shift(5)
    df["gamma_flip_volatility"] = gf.rolling(10, min_periods=5).std()
    df["gamma_flip_acceleration"] = gf.diff() - gf.diff(5)
    df["gamma_flip_regime_strength"] = np.abs(df["gamma_flip_zscore"]) * np.sign(df["gamma_flip_momentum"])

    df["wall_distance_z"] = zscore_past(df["wall_distance"], roll=20)
    df["wall_distance_compression_rate"] = -(df["wall_distance"].diff() / (df["wall_distance"].shift(1) + 1e-9))
    df["wall_distance_volatility"] = df["wall_distance"].rolling(10, min_periods=5).std()
    df["wall_distance_momentum"] = df["wall_distance"] - df["wall_distance"].shift(5)
    df["wall_pressure_score"] = 1 / (1 + np.exp(-df["wall_distance_compression_rate"])) * np.sign(df["wall_distance_momentum"])

    gw = df["gammaflip_x_walldist"].shift(1)
    df["gw_energy_z"] = zscore_past(df["gammaflip_x_walldist"], roll=20)
    df["gw_slope"] = gw - gw.shift(1)
    df["gw_momentum"] = gw - gw.shift(5)
    df["gw_volatility"] = gw.rolling(10, min_periods=5).std()
    df["gw_energy_ratio"] = gw / (df["wall_distance"].shift(1) + 1e-9)

    new_cols = [c for c in df.columns if c not in existing]
    return df, {"STRUCTURAL_ENRICH": new_cols}


# ============================================================
# INDICATOR 4 — AUGMENTED_FEATURES (validated)
# ============================================================
@register_indicator("AUGMENTED_FEATURES")
def build_augmented_features(df, ROLLING_WINDOWS=[10, 20, 50]):
    df = df.copy()
    required = {"symbol", "entry_date", "pnl"}
    if not required.issubset(df.columns):
        print("AUGMENTED_FEATURES skipped — missing:", required - set(df.columns))
        return df, {}

    existing = set(df.columns)

    df["entry_date"] = pd.to_datetime(df["entry_date"], utc=True)
    df = df.sort_values(["entry_date", "symbol"]).reset_index(drop=True)

    df["day_of_week"] = df["entry_date"].dt.dayofweek
    df["month_of_year"] = df["entry_date"].dt.month
    df["hour_of_day"] = df["entry_date"].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    if "net_gex" in df.columns:
        gex_p = df["net_gex"].shift(1)
        df["net_gex_abs"] = gex_p.abs()
        df["gex_zscore"] = gex_p.groupby(df["symbol"]).transform(
            lambda s: (s - s.shift(1).rolling(20, min_periods=5).mean())
            / (s.shift(1).rolling(20, min_periods=5).std() + 1e-6)
        )

    if {"max_call_wall_strike_perc", "min_put_wall_strike_perc"}.issubset(df.columns):
        cw = df["max_call_wall_strike_perc"].shift(1)
        pw = df["min_put_wall_strike_perc"].shift(1)
        df["wall_distance"] = cw - pw
        df["wall_distance_z"] = df.groupby("symbol")["wall_distance"].transform(
            lambda s: (s - s.shift(1).rolling(20).mean())
            / (s.shift(1).rolling(20).std() + 1e-6)
        )
    if "entry_cost" not in df.columns:
        df['entry_cost'] = df['adjusted_close']

    if "entry_cost" in df.columns:
        ec = df["entry_cost"].shift(1)
        df["gamma_flip_dist_norm"] = (df.get("gamma_flip_strike_perc", ec) - ec) / (ec + 1e-6)
        df["call_wall_dist_norm"] = (df.get("max_call_wall_strike_perc", ec) - ec) / (ec + 1e-6)
        df["put_wall_dist_norm"] = (df.get("min_put_wall_strike_perc", ec) - ec) / (ec + 1e-6)

    for L in [1, 5, 15]:
        for col in ["net_gex", "wall_distance"]:
            if col in df.columns:
                p = df[col].shift(1)
                df[f"{col}_mom{L}"] = p - p.shift(L)

    new_cols = [c for c in df.columns if c not in existing]
    return df, {"AUGMENTED_FEATURES": new_cols}


# ============================================================
# INDICATOR 5 — MARKET_VOL_REGIME
# ============================================================
@register_indicator("MARKET_VOL_REGIME")
def build_market_vol_regime(df, roll=50):
    df = df.copy()
    existing = set(df.columns)

    pnl_p = df["pnl"].shift(1) if "pnl" in df.columns else None
    gw_p = df["gammaflip_x_walldist"].shift(1) if "gammaflip_x_walldist" in df.columns else None

    if pnl_p is not None:
        df["mkt_vol_pnl"] = pnl_p.rolling(roll, min_periods=5).std()
        df["mkt_vol_pnl_z"] = zscore_past(df["mkt_vol_pnl"], roll)
    else:
        df["mkt_vol_pnl"] = 0
        df["mkt_vol_pnl_z"] = 0

    if gw_p is not None:
        df["mkt_vol_struct"] = gw_p.rolling(roll, min_periods=5).std()
        df["mkt_vol_struct_z"] = zscore_past(df["mkt_vol_struct"], roll)
    else:
        df["mkt_vol_struct"] = 0
        df["mkt_vol_struct_z"] = 0

    df["market_vol_score"] = (
        df["mkt_vol_pnl_z"].fillna(0)
        + df["mkt_vol_struct_z"].fillna(0)
    )

    df["market_vol_regime"] = np.select(
        [
            df["market_vol_score"] > 1,
            df["market_vol_score"] < -1
        ],
        ["HighVol_Expansion", "LowVol_Contraction"],
        default="Normal"
    )

    new_cols = [c for c in df.columns if c not in existing]
    return df, {"MARKET_VOL_REGIME": new_cols}


# ============================================================
# INDICATOR 6 — LIQUIDITY_REGIME
# ============================================================
@register_indicator("LIQUIDITY_REGIME")
def build_liquidity_regime(df, roll=50):
    df = df.copy()
    existing = set(df.columns)

    wd = df.get("wall_distance", pd.Series(0, index=df.index)).shift(1)

    df["liq_compression"] = -(wd.diff() / (wd.shift(1).replace(0, np.nan) + 1e-9))
    df["liq_compression_z"] = zscore_past(df["liq_compression"], roll)

    gw = df.get("gammaflip_x_walldist", pd.Series(0, index=df.index)).shift(1)
    df["liq_energy_decay"] = gw - gw.shift(5)
    df["liq_energy_decay_z"] = zscore_past(df["liq_energy_decay"], roll)

    df["liquidity_score"] = df["liq_compression_z"] - df["liq_energy_decay_z"]

    df["liquidity_regime"] = np.select(
        [df["liquidity_score"] > 1, df["liquidity_score"] < -1],
        ["Liquidity_Tightening", "Liquidity_Loosening"],
        default="Neutral"
    )

    new_cols = [c for c in df.columns if c not in existing]
    return df, {"LIQUIDITY_REGIME": new_cols}


# ============================================================
# INDICATOR 7 — GAMMA_REGIME_BOUNDARY
# ============================================================
@register_indicator("GAMMA_REGIME_BOUNDARY")
def build_gamma_regime_boundary(df):
    df = df.copy()
    existing = set(df.columns)

    gf = df["gamma_flip_strike"].shift(1)
    if "entry_cost" not in df.columns:
        df['entry_cost'] = df['adjusted_close']

    ec = df["entry_cost"].shift(1)

    if gf.isna().all() or ec.isna().all():
        df["gamma_regime_boundary"] = "Unknown"
        return df, {"GAMMA_REGIME_BOUNDARY": ["gamma_regime_boundary"]}

    dist = ec - gf
    df["gamma_dist"] = dist
    df["gamma_dist_z"] = zscore_past(dist, roll=20)

    df["gamma_flip_cross_up"] = (dist.shift(1) < 0) & (dist >= 0)
    df["gamma_flip_cross_down"] = (dist.shift(1) > 0) & (dist <= 0)

    df["gamma_regime_boundary"] = np.select(
        [
            df["gamma_flip_cross_up"],
            df["gamma_flip_cross_down"],
            dist > 0,
            dist < 0,
        ],
        [
            "Cross_Up_LongGamma",
            "Cross_Down_ShortGamma",
            "AboveFlip_LongGamma",
            "BelowFlip_ShortGamma",
        ],
        default="Gamma_Unknown"
    )

    new_cols = [c for c in df.columns if c not in existing]
    return df, {"GAMMA_REGIME_BOUNDARY": new_cols}


# ============================================================
# INDICATOR 8 — DEALER_FLOW_TILT_REGIME
# ============================================================
@register_indicator("DEALER_FLOW_TILT_REGIME")
def build_dealer_flow_tilt_regime(df, roll=50):
    df = df.copy()
    existing = set(df.columns)

    if "net_gex" not in df.columns:
        df["dealer_flow_tilt_regime"] = "Unknown"
        return df, {"DEALER_FLOW_TILT_REGIME": ["dealer_flow_tilt_regime"]}

    gex_p = df["net_gex"].shift(1)

    df["dealer_flow_delta"] = gex_p.diff()
    df["dealer_flow_accel"] = df["dealer_flow_delta"].diff()
    df["dealer_flow_z"] = zscore_past(gex_p, roll)

    df["dealer_flow_score"] = (
        np.tanh(df["dealer_flow_delta"] / (gex_p.abs() + 1e-6))
        + 0.3 * df["dealer_flow_accel"].fillna(0)
        + df["dealer_flow_z"].fillna(0)
    )

    df["dealer_flow_tilt_regime"] = np.select(
        [
            df["dealer_flow_score"] > 1,
            df["dealer_flow_score"] < -1,
        ],
        ["Dealer_LongFlow_Tilt", "Dealer_ShortFlow_Tilt"],
        default="Dealer_Neutral"
    )

    new_cols = [c for c in df.columns if c not in existing]
    return df, {"DEALER_FLOW_TILT_REGIME": new_cols}


# ============================================================
# INDICATOR 9 — STRUCTURAL_GEX_FEATURES
# ============================================================
@register_indicator("STRUCTURAL_GEX_FEATURES")
def build_structural_gex_features(
    df,
    ridge_threshold=5e5,
    trough_threshold=5e5,
    valley_threshold=1e5,
    cliff_drop_threshold=3e5,
    mom_threshold=1e5
):
    df = df.copy()
    if 'entry_cost' not in df.columns:
        df['entry_cost'] = df['adjusted_close']
    required = {"net_gex", "entry_cost", "gamma_flip_strike"}
    if not required.issubset(df.columns):
        print("STRUCTURAL_GEX_FEATURES skipped — missing:", required - set(df.columns))
        return df, {}

    existing = set(df.columns)

    gex_p = df["net_gex"].shift(1)
    price_p = df["entry_cost"].shift(1)
    gf_p = df["gamma_flip_strike"].shift(1)

    if "gamma_flip_strike_perc" in df.columns:
        gf_perc = df["gamma_flip_strike_perc"].shift(1)
    elif "perc_gamma_flip_strike" in df.columns:
        gf_perc = df["perc_gamma_flip_strike"].shift(1)
    else:
        gf_perc = 100 * (price_p - gf_p) / (gf_p + 1e-9)

    # FLAGS
    df["gamma_ridge_flag"] = (gex_p > ridge_threshold)
    df["gamma_trough_flag"] = (gex_p < -trough_threshold)
    df["gamma_valley_flag"] = (gex_p.abs() < valley_threshold)
    df["gamma_cliff_flag"] = (gex_p.diff() < -cliff_drop_threshold)

    df["gamma_flip_near"] = (gf_perc.abs() < 2)
    df["gamma_flip_far"] = (gf_perc.abs() > 5)

    for c in [
        "gamma_ridge_flag", "gamma_trough_flag", "gamma_valley_flag",
        "gamma_cliff_flag", "gamma_flip_near", "gamma_flip_far"
    ]:
        df[c] = df[c].fillna(False).astype(bool)

    # TREND BIAS
    gf_above = price_p > gf_p
    gf_below = price_p < gf_p

    df["trend_bias_from_ridge"] = gf_above & df["gamma_ridge_flag"]
    df["trend_bias_from_trough"] = gf_below & df["gamma_trough_flag"]

    df["long_bias"] = gf_above & (df["gamma_ridge_flag"] | df["gamma_valley_flag"])
    df["short_bias"] = gf_below & (df["gamma_trough_flag"] | df["gamma_valley_flag"])

    # MOMENTUM
    df["gex_mom1"] = gex_p.diff()
    df["gex_mom3"] = gex_p.diff(3)
    df["gex_accel"] = df["gex_mom1"].diff()

    df["ridge_exit_momentum"] = df["gamma_ridge_flag"].shift(1).fillna(False) & (~df["gamma_ridge_flag"])
    df["trough_entry_momentum"] = (~df["gamma_trough_flag"].shift(1).fillna(False)) & df["gamma_trough_flag"]

    # FLIP DYNAMICS
    df["flip_toward_short"] = gf_above.shift(1).fillna(False) & gf_below
    df["flip_toward_long"] = gf_below.shift(1).fillna(False) & gf_above

    df["gamma_flip_pressure"] = df["gex_mom1"].abs() / (gf_perc.abs() + 1e-6)

    df["gamma_flip_reversal_signal"] = (
        df["gamma_flip_near"]
        & (np.sign(df["gex_mom1"]) != np.sign(df["gex_mom1"].shift(1)))
    )

    # VOLATILITY / TREND INITIATION
    df["structural_volatility_risk"] = gex_p.abs() / (gex_p.abs().rolling(20).mean() + 1e-9)

    df["valley_breakout_pressure"] = (
        df["gamma_valley_flag"]
        & (df["gex_mom3"].abs() > mom_threshold)
    )

    df["cliff_reversal_probability"] = (
        df["gamma_cliff_flag"]
        & (np.sign(df["gex_mom1"]) != np.sign(df["gex_mom1"].shift(1)))
    )

    df["ridge_trend_long"] = df["gamma_ridge_flag"] & gf_above & (df["gex_mom1"] > 0)
    df["ridge_trend_short"] = df["gamma_ridge_flag"] & gf_below & (df["gex_mom1"] < 0)

    # RENKO SIGNALS
    df["long_start_signal"] = (
        (df["gamma_valley_flag"] | df["trough_entry_momentum"])
        & gf_above
        & (df["gex_mom1"] > 0)
    )

    df["short_start_signal"] = (
        (df["gamma_valley_flag"] | df["trough_entry_momentum"])
        & gf_below
        & (df["gex_mom1"] < 0)
    )

    df["cliff_short_signal"] = df["gamma_cliff_flag"] & gf_below
    df["ridge_long_follow"] = df["gamma_ridge_flag"] & gf_above & (df["gex_mom1"] > 0)

    new_cols = [c for c in df.columns if c not in existing]
    return df, {"STRUCTURAL_GEX_FEATURES": new_cols}


# ============================================================
# INDICATOR 10 — VALLEY_BREAKOUT_PREDICTOR
# ============================================================
@register_indicator("VALLEY_BREAKOUT_PREDICTOR")
def build_valley_breakout_predictor(
    df,
    mom_window=50,
    press_window=50,
    vol_excess_floor=1.0,
    mom_z_thresh=0.75
):
    df = df.copy()

    required = {
        "gamma_valley_flag",
        "gex_mom1",
        "gex_mom3",
        "gamma_flip_pressure",
        "structural_volatility_risk"
    }

    if not required.issubset(df.columns):
        print("VALLEY_BREAKOUT_PREDICTOR skipped — missing:", required - set(df.columns))
        return df, {}

    existing = set(df.columns)

    valley = df["gamma_valley_flag"].astype(bool)

    mom_std = df["gex_mom3"].rolling(mom_window).std()
    press_std = df["gamma_flip_pressure"].rolling(press_window).std()

    mom_norm = df["gex_mom3"] / (mom_std + 1e-6)
    press_norm = df["gamma_flip_pressure"] / (press_std + 1e-6)

    vol_excess = (df["structural_volatility_risk"] - vol_excess_floor).clip(lower=0)

    raw_score = mom_norm.abs() + press_norm.abs() + vol_excess
    df["valley_breakout_score"] = valley * np.tanh(raw_score)

    df["valley_breakout_long"] = (
        valley & (mom_norm > mom_z_thresh) & (df["gex_mom1"] > 0)
    )

    df["valley_breakout_short"] = (
        valley & (mom_norm < -mom_z_thresh) & (df["gex_mom1"] < 0)
    )

    df["valley_breakout_direction"] = 0
    df.loc[df["valley_breakout_long"], "valley_breakout_direction"] = 1
    df.loc[df["valley_breakout_short"], "valley_breakout_direction"] = -1

    new_cols = [c for c in df.columns if c not in existing]
    return df, {"VALLEY_BREAKOUT_PREDICTOR": new_cols}

# ============================================================
# INDICATOR 11 — VALLEY_BREAKOUT_OSCILLATOR
# ============================================================


@register_indicator("VALLEY_BREAKOUT_OSCILLATOR")
def build_valley_breakout_oscillator(df, smooth_window=10, pow_compression=1.2):
    df = df.copy()

    required = {"valley_breakout_score"}
    if not required.issubset(df.columns):
        print("VALLEY_BREAKOUT_OSCILLATOR skipped — missing:", required - set(df.columns))
        return df, {}

    existing = set(df.columns)

    v = df["valley_breakout_score"].clip(0, 1)
    v_smooth = v.rolling(smooth_window).mean()
    v_comp = v_smooth ** pow_compression

    df["valley_breakout_osc_100"] = 100 * v_comp
    df["valley_breakout_osc_norm"] = v_comp

    new_cols = [c for c in df.columns if c not in existing]
    return df, {"VALLEY_BREAKOUT_OSCILLATOR": new_cols}

# ============================================================
# APPLY INDICATORS
# ============================================================


def apply_indicators(df, indicators_to_add):
    df_out = df.copy()
    print("Available indicators:", list(INDICATOR_REGISTRY.keys()))
    summary = []

    for ind in indicators_to_add:
        name = ind.upper()
        if name in INDICATOR_REGISTRY:
            func = INDICATOR_REGISTRY[name]
            df_out, new_cols_dict = func(df_out)
            if not new_cols_dict:
                print(f"[{name}] Skipped — no columns added.")
                continue
            new_cols = new_cols_dict.get(name, [])
            print(f"\n[{name}] Added {len(new_cols)} new columns:")
            summary.append({
                "Indicator": name,
                "New_Columns": len(new_cols),
                "Column_Names": new_cols
            })
        else:
            print(f"Indicator '{name}' not found.")

    if summary:
        print(pd.DataFrame(summary))
    return df_out


def build_oscillators(df):
    def zscore(
        time_series: pd.Series,
        window: int,
        mean: Optional[float] = None,
        adjust: bool = False
    ) -> pd.Series:
        '''
        Computes the z-score for the data time series

        Parameters
            data: pd.Series
                the time series
            window: int
                the size of the window
            mean: Optional[float]
                the average value of the column, if None the rolling mean will be used
            adjust: bool
                whether to adjust the standard deviation around the given mean

        Returns
            A pd.Series
        '''
        if mean is None:
            col_mean = time_series.rolling(window=window).mean()
        else:
            col_mean = mean
        if adjust:
            squared_diff = (time_series - col_mean)**2
            col_std = np.sqrt(squared_diff.rolling(window=window).sum() / (window - 1))
        else:
            col_std = time_series.rolling(window=window).std()
        return (time_series - col_mean) / col_std

    df_osc = df.copy()

    # ============================================================
    # 1. AUTO-DETECT SAFE OSCILLATOR CANDIDATES
    # ============================================================

    # Hard exclusion list (boolean, categorical, discrete signals)
    exclude_osc = set([
        # structural flags
        "gamma_ridge_flag", "gamma_trough_flag", "gamma_valley_flag", "gamma_cliff_flag",
        "gamma_flip_near", "gamma_flip_far",

        # directional flags
        "trend_bias_from_ridge", "trend_bias_from_trough",
        "long_bias", "short_bias",
        "ridge_exit_momentum", "trough_entry_momentum",
        "flip_toward_long", "flip_toward_short",
        "cliff_reversal_probability",

        # renko signals
        "long_start_signal", "short_start_signal",
        "cliff_short_signal", "ridge_long_follow",

        # categories
        "Quadrant", "Market_State",
        "market_vol_regime", "liquidity_regime",
        "gamma_regime_boundary",
        "dealer_flow_tilt_regime",
        "valley_breakout_direction",
    ])

    # numeric columns only
    numeric_cols = df.select_dtypes(include='number').columns
    numeric_cols = [col for col in numeric_cols if col not in exclude_cols]

    # auto include continuous / long-range signals
    AUTO_OSC_TARGETS = [
        'net_gex',
        'gex_mom1', 'gex_mom3', 'gex_accel',
        'gamma_flip_pressure',
        'structural_volatility_risk',
        'gamma_flip_momentum', 'gamma_flip_volatility',
        'gamma_flip_acceleration', 'gamma_flip_regime_strength',
        'wall_distance', 'wall_distance_compression_rate',
        'wall_distance_volatility', 'wall_distance_momentum',
        'wall_pressure_score',
        'gw_energy_z', 'gw_momentum', 'gw_volatility',
        'dealer_flow_delta', 'dealer_flow_accel', 'dealer_flow_z', 'dealer_flow_score',
        'gamma_dist', 'gamma_dist_z',
        'gex_zscore', 'net_gex_abs',
        'valley_breakout_score', 'valley_breakout_osc_norm', 'valley_breakout_osc_100',
        'CPWO', 'GSM',
    ]

    # Only keep valid ones
    cols_to_process = [c for c in AUTO_OSC_TARGETS if c in numeric_cols and c not in exclude_osc]

    print("\nDetected continuous oscillator-eligible columns:")
    print(cols_to_process)

    if len(cols_to_process) == 0:
        raise ValueError("No valid oscillator inputs found.")

    # ============================================================
    # 2. FUNCTIONS — RSI and Momentum Oscillators
    # ============================================================

    def compute_rsi(series, n=14):

        """Past-only RSI."""
        delta = series.diff()
        up = np.where(delta > 0, delta, 0)
        down = np.where(delta < 0, -delta, 0)
        roll_up = pd.Series(up, index=series.index).rolling(n, min_periods=5).mean()
        roll_down = pd.Series(down, index=series.index).rolling(n, min_periods=5).mean()
        RS = roll_up / (roll_down + 1e-6)
        return 100 - (100 / (1 + RS))

    # ============================================================
    # 3. BUILD OSCILLATORS
    # ============================================================

    print(f"\nCreating oscillators for ({len(cols_to_process)} columns):")
    print(cols_to_process)

    for col in cols_to_process:
        base = df_osc[col]

        # 1 — z-oscillator (past-only by signal_detectors.zscore)
        df_osc[f"{col}_osc_z"] = zscore(base, window=50)

        # 2 — tanh oscillator
        df_osc[f"{col}_osc_tanh"] = np.tanh(df_osc[f"{col}_osc_z"])

        # 3 — RSI oscillator (classic TA)
        df_osc[f"{col}_osc_rsi"] = compute_rsi(base, n=14)

        # 4 — momentum oscillator (past-only)
        momentum = base - base.shift(10)
        df_osc[f"{col}_osc_mom"] = np.tanh(momentum / (base.rolling(10, min_periods=5).std() + 1e-6))

        # Smooth all oscillators
        for osc_col in [
            f"{col}_osc_z",
            f"{col}_osc_tanh",
            f"{col}_osc_rsi",
            f"{col}_osc_mom"
        ]:
            df_osc[f"{osc_col}_smooth"] = df_osc[osc_col].rolling(5, min_periods=2).mean()

        print(f"  → Created: {col}_osc_z, _tanh, _rsi, _mom (+ smoothed)")

    # ============================================================
    # 4. DONE
    # ============================================================
    return df_osc.copy()


class TSFreshExtractor:
    """ tsfresh extraction without duplicates"""

    def __init__(self, ticker, type, path, suffix, test, config, ind1, ind2, ind3, ind4, spot):
        self.ticker = ticker
        self.type = type
        self.ind1 = ind1
        self.ind2 = ind2
        self.ind3 = ind3
        self.ind4 = ind4
        self.spot = spot
        self.path = path
        self.suffix = suffix
        self.test = test
        self.fc_params = MinimalFCParameters()
        self.windows = [5, 10, 20, 50]
        self.config = config
        self.base_formulas, self.tsfresh_metrics = load_pipeline_config(config)
        self.exclude_cols = exclude_cols
        self.combined_df = None
        self.merged_df = None
        self.trades_df = None
        self.final_merged_df = None

        print(
            f"Loaded {len(self.base_formulas)} base feature formulas from pipeline")
        print(
            f"Loaded {len(self.tsfresh_metrics)} tsfresh metrics from pipeline")

    def load_ind_and_spot_data(self):
        print("\n" + "=" * 70)
        print(f"EXTRACTING {self.type.upper()} INDICATOR AND SPOT DATA FROM WASABI")
        print("=" * 70)

        storage = WasabiStorageSystem(path='renko/indicators', bucket='nufintech-data-analysis')

        all_inds = {}

        for ind in [self.ind1, self.ind2, self.ind3, self.ind4]:
            keys = storage.ls('', f'{self.ticker.upper()}_S:{ind}*.pkl')
            print(f"Found {len(keys)} files for indicator {ind}")

            dfs = [storage.read_pickle(k) for k in keys]
            df = pd.concat(dfs)
            df = df.drop(columns=['t'], errors='ignore')
            all_inds[ind] = df

        cleaned_dict = {}

        for key, df in all_inds.items():
            base_name = key.split('_')[0]
            new_key = f"df_{base_name.lower()}"

            cleaned_dict[new_key] = df

        for key in cleaned_dict:
            df = cleaned_dict[key]
            df = df.sort_index()
            df = df[~df.index.duplicated(keep='first')]
            cleaned_dict[key] = df
            # cleaned_dict[key] = cleaned_dict[key].sort_index()

        spot_keys = storage.ls('', f'{self.ticker.upper()}_{self.spot}*.pkl')
        print(f"Found {len(spot_keys)} spot files")

        spot_dfs = [storage.read_pickle(k) for k in spot_keys]
        spot_df = pd.concat(spot_dfs)
        spot_df = spot_df.sort_index()

        merged_df = pd.merge_asof(
            spot_df,
            cleaned_dict['df_gflipst'],
            left_index=True,
            right_index=True,
            direction='backward'
        )

        merged_df = pd.merge_asof(
            merged_df,
            cleaned_dict['df_mpwstk'],
            left_index=True,
            right_index=True,
            direction='backward'
        )

        merged_df = pd.merge_asof(
            merged_df,
            cleaned_dict['df_mcwstk'],
            left_index=True,
            right_index=True,
            direction='backward'
        )

        merged_df = pd.merge_asof(
            merged_df,
            cleaned_dict['df_ngex'],
            left_index=True,
            right_index=True,
            direction='backward'
        )

        merged_df.insert(1, 'symbol', self.ticker)

        self.merged_df = merged_df
        print("Merged indicator + spot shape:", merged_df.shape)
        return self

    def prepare_base_features(self):
        print("\n" + "=" * 70)
        print("STEP 2: PREPARING BASE FEATURES")
        print("=" * 70)

        df = self.combined_df
        required_cols = [
            'max_call_wall_strike',
            'min_put_wall_strike',
            'gamma_flip_strike',
            'net_gex',
            'gamma_flip_strike',
        ]

        missing = [c for c in required_cols if c not in df.columns]

        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        df = apply_dynamic_features(df, self.base_formulas)
        # After computing base features → fill all NaNs for numeric cols
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].ffill()
        df[num_cols] = df[num_cols].fillna(0)
        self.combined_df = df
        return self

    def _should_skip_metrics(self):
        if not hasattr(self, "tsfresh_metrics"):
            print("TSFRESH metrics attribute missing — skipping extraction.")
            self.tsfresh_features = pd.DataFrame()
            return True
        if isinstance(self.tsfresh_metrics, dict) and len(self.tsfresh_metrics) == 0:
            print("TSFRESH metrics dict is empty — continuing without extraction.")
            self.tsfresh_features = pd.DataFrame()
            return True

        return False

    def _prepare_sorted_symbol_df(self, df):
        df = df.sort_index()
        symbol_df = df[df['symbol'] == self.ticker]
        print(f"Processing symbol: {self.ticker}")

        if len(symbol_df) < max(self.windows):
            print(f"[INFO] Skipping {self.ticker}: only {len(symbol_df)} trades (need ≥ 50).")
            return None

        return symbol_df

    def _extract_window_features(self, symbol_df, idx, window):
        window_df = symbol_df.iloc[idx - window: idx].reset_index(drop=True)

        # Clean only TSFresh input metrics: fill NaNs locally
        for alias, metric in self.tsfresh_metrics.items():
            if metric in window_df.columns:
                window_df[metric] = window_df[metric].ffill()

        ts_data = []

        # convert window into tsfresh long format
        for alias, metric in self.tsfresh_metrics.items():
            if metric not in window_df.columns:
                continue

            for t, v in enumerate(window_df[metric].values):
                ts_data.append({
                    'id': idx,
                    'time': t,
                    'kind': f'{alias}_w{window}',
                    'value': v
                })

        if not ts_data:
            return {}

        ts_df = pd.DataFrame(ts_data)

        # apply tsfresh
        features = extract_features(
            ts_df,
            column_id='id',
            column_sort='time',
            column_kind='kind',
            column_value='value',
            default_fc_parameters=self.fc_params
        )

        # flatten to dict
        result = {}
        for col in features.columns:
            result[col] = features[col].iloc[0]

        gc.collect()
        return result

    def _extract_single_trade(self, symbol_df, idx, windows):
        combined_features = {
            'bar_timestamp': symbol_df.index[idx],
            't': symbol_df.iloc[idx]['t'] if 't' in symbol_df.columns else None,
            'symbol': self.ticker,
            'trade_idx': idx,
        }

        for window in windows:
            print(f'Processing for window {window} and index {idx}')
            if idx >= window:
                fdict = self._extract_window_features(symbol_df, idx, window)
                combined_features.update(fdict)

        return combined_features

    def _finalize_trade_features(self, all_trade_features):
        if not all_trade_features:
            print("WARNING:  No tsfresh features extracted.")
            self.tsfresh_features = pd.DataFrame()
            return

        print(f" Combining {len(all_trade_features)} feature rows...")
        tsfresh_df = pd.DataFrame(all_trade_features)

        if 'bar_timestamp' in tsfresh_df.columns:
            tsfresh_df['bar_timestamp'] = pd.to_datetime(tsfresh_df['bar_timestamp'])
            tsfresh_df = tsfresh_df.set_index('bar_timestamp')
            tsfresh_df = tsfresh_df.sort_index()

        # impute only numeric columns
        numeric_cols = tsfresh_df.select_dtypes(include=['float64', 'int64']).columns

        if not numeric_cols.empty:
            tsfresh_numeric = tsfresh_df[numeric_cols]
            tsfresh_numeric = impute(tsfresh_numeric)
            tsfresh_df[numeric_cols] = tsfresh_numeric

        duplicates = tsfresh_df.index.duplicated().sum()
        if duplicates > 0:
            print(f"WARNING: Removing {duplicates} duplicate rows")
            tsfresh_df = tsfresh_df[~tsfresh_df.index.duplicated(keep='first')]

        self.tsfresh_features = tsfresh_df

    def extract_tsfresh_features_(self, df):
        if self._should_skip_metrics():
            return self

        symbol_df = self._prepare_sorted_symbol_df(df)

        if symbol_df is None:
            return self

        all_trade_features = []

        for idx in range(max(self.windows), len(symbol_df)):
            row = self._extract_single_trade(symbol_df, idx, self.windows)
            if row:
                all_trade_features.append(row)

        self._finalize_trade_features(all_trade_features)
        return self.tsfresh_features

    def load_from_wasabi(self):
        print("\n" + "=" * 70)
        print(f"STEP 7: EXTRACTING {self.type.upper()} DATA FROM WASABI")
        print("=" * 70)
        storage = WasabiStorageSystem(path='renko/trades', bucket='nufintech-data-analysis')

        keys = storage.ls(
            '',
            f'{self.ticker}/{self.path}/'
            f'{self.ticker.upper()}_{self.suffix}_{self.type}.renko_trades.csv'
        )

        print(f"Found keys: {len(keys)}")
        dfs = [storage.read_csv(key) for key in keys]
        df = pd.concat(dfs, ignore_index=True)

        self.trades_df = df.sort_values(
            ['symbol', 'entry_date']).reset_index(drop=True)
        self.trades_df['entry_date'] = (
            pd.to_datetime(self.trades_df['entry_date'])
            .dt.tz_convert(None)
        )

        # final_merged_df = pd.merge_asof(
        #     self.trades_df.sort_values('entry_date'),
        #     self.combined_df.sort_index(),
        #     left_on='entry_date',
        #     right_index=True,
        #     direction='backward'
        # )

        # Reset index to preserve bar_timestamp as column
        combined_with_timestamp = self.combined_df.reset_index().rename(columns={'index': 'bar_timestamp'})

        final_merged_df = pd.merge_asof(
            self.trades_df.sort_values('entry_date'),
            combined_with_timestamp.sort_values('bar_timestamp'),
            left_on='entry_date',
            right_on='bar_timestamp',
            direction='backward'
        )

        # Verify no lookahead bias
        assert (final_merged_df['bar_timestamp'] <= final_merged_df['entry_date']).all(), "Lookahead bias detected!"

        self.final_merged_df = final_merged_df

        return self

    def save_output(self):
        """Save TSFresh features to Wasabi (CSV + PKL only)"""

        print("\n" + "=" * 70)
        print("STEP 8: SAVING OUTPUT TO WASABI")
        print("=" * 70)

        if self.tsfresh_features is None or len(self.tsfresh_features) == 0:
            print("\n No features to save!")
            return self

        if 'bar_timestamp' in self.final_merged_df.columns:
            lookahead = (self.final_merged_df['bar_timestamp'] > self.final_merged_df['entry_date']).sum()
            if lookahead > 0:
                raise ValueError(f"ERROR: LOOKAHEAD BIAS: {lookahead} rows have features from future!")

        # Check for excessive NaNs
        nan_pct = self.final_merged_df.isna().sum().sum() / (
            self.final_merged_df.shape[0] * self.final_merged_df.shape[1])
        if nan_pct > 0.05:
            print(f" Warning: {nan_pct:.2%} of data is NaN")

        # tsfresh features
        ts_df = self.tsfresh_features.copy()

        output_df = ts_df

        # File names based on output path
        base_name_hourly = self.ticker + "_" + self.type + '_tsfresh_hourly_features_' + "test_" + self.test
        base_name_final = self.ticker + "_" + self.type + '_final_trades_df_' + "test_" + self.test
        csv_file_hourly = base_name_hourly + ".csv"
        csv_file_final = base_name_final + ".csv"
        pkl_file_hourly = base_name_hourly + ".pkl"
        pkl_file_final = base_name_final + ".pkl"

        # Upload to Wasabi
        ws_final = WasabiStorageSystem(path=f"ml_nonquad/tsfresh_outputs/{self.ticker}", bucket='nufintech-ai-common')
        ws_hourly = WasabiStorageSystem(path=f"ml_nonquad/hourly_tsfresh_feats/{self.ticker}", bucket='nufintech-ai-common')

        # print(f" Uploading CSV to Wasabi→ ml_nonquad/tsfresh_outputs/{self.ticker}/{csv_file}")
        print("Uploading CSV to Wasabi")
        ws_final.write_csv(self.final_merged_df, symbol="", file=csv_file_final)
        ws_hourly.write_csv(output_df, symbol="", file=csv_file_hourly)
        print(self.final_merged_df.shape)
        print(output_df.shape)

        print("Uploading Pickle to Wasabi")
        ws_final.write_pickle(self.final_merged_df, symbol="", file=pkl_file_final)
        ws_hourly.write_pickle(output_df, symbol="", file=pkl_file_hourly)

        print("\n Upload complete!")
        print(f"   Rows: {len(output_df):,}")
        print(f"   Columns: {len(output_df.columns)}")
        print("=" * 70)

        return self

    def run(self):
        """Execute  tsfresh extraction pipeline"""

        print("\n" + "=" * 70)
        print("  TSFRESH EXTRACTION (No Duplicates)")
        print("=" * 70)
        print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        try:

            # 1. Load raw indicator + spot data
            print("STEP 1")
            self.load_ind_and_spot_data()
            df = self.merged_df.copy()
            print(f"\n \t After Step 1, {df.shape}")

            # 2. Apply indicators
            print("STEP 2")
            df = apply_indicators(
                df,
                [
                    "CPWO",
                    "GSM",
                    "STRUCTURAL_ENRICH",
                    "AUGMENTED_FEATURES",
                    "MARKET_VOL_REGIME",
                    "LIQUIDITY_REGIME",
                    "GAMMA_REGIME_BOUNDARY",
                    "DEALER_FLOW_TILT_REGIME",
                    "STRUCTURAL_GEX_FEATURES",
                    "VALLEY_BREAKOUT_PREDICTOR",
                    "VALLEY_BREAKOUT_OSCILLATOR",
                ]
            )
            print(f"\n \tAfter Step 2, {df.shape}")

            # 3. Apply oscillators
            print("STEP 3")
            df = build_oscillators(df)
            print(f"\n \tAfter Step 3, {df.shape}")

            # 4. Apply base dynamic formulas
            df = apply_dynamic_features(df, self.base_formulas)
            num_cols = df.select_dtypes(include='number').columns
            print(f"\n \tAfter Step 4, {df.shape}")

            print(f"NaNs before: {df[num_cols].isna().sum().sum():,}")
            print(f"Infs before: {np.isinf(df[num_cols]).sum().sum():,}")

            # Replace inf with NaN, then forward fill
            df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
            df[num_cols] = df[num_cols].ffill()
            df[num_cols] = df[num_cols].fillna(0)

            # Drop any remaining unfillable rows
            rows_before = len(df)
            df = df.dropna()
            print(f"Dropped {rows_before - len(df)} rows")

            print(f"NaNs after: {df[num_cols].isna().sum().sum()}")
            print(f"Infs after: {np.isinf(df[num_cols]).sum().sum()}")
            print(f"Final shape: {df.shape}")

            # 5. Now extract tsfresh windows
            tsfresh_df = self.extract_tsfresh_features_(df)
            print(f"\n \tAfter Step 5, {tsfresh_df.shape}")

            # 6. UNIFIED DATAFRAME
            self.combined_df = tsfresh_df.copy()

            # 7. Now merge trades (merge_asof)
            self.load_from_wasabi()
            print(f"\n \tAfter Step 7, {self.final_merged_df.shape}")

            # 8. Save everything
            self.save_output()

            print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            return True

        except Exception as e:
            print(f"\n{'='*70}")
            print("ERROR OCCURRED")
            print(f"{'='*70}")
            print(f"\n{str(e)}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description=' tsfresh Feature Extraction (No Duplicates)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    • Processing each trade only once
    • Combining w5 and w10 features into single row
    • No duplicate symbol-dates!
        """
    )

    parser.add_argument(
        '--ticker',
        type=str,
        required=True,
        help='Name of the ticker'
    )

    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='path to config.json file'
    )

    parser.add_argument(
        '--type',
        type=str,
        required=True,
        choices=['long', 'short'],
        help="Whether to load long or short combined file"
    )
    parser.add_argument(
        "--path",
        type=str,
        default='analysis.noquads.atr.251009.10',
        help="the top level path for the keys on Wasabi"
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default='2023-04-04_2025-09-30_5min',
        help="the suffix for the keys on Wasabi"
    )

    parser.add_argument(
        "--ind1",
        type=str,
        default='GFLIPST_20230401_202510',
        help="the suffix for GFLIPST indicator keys on Wasabi"
    )

    parser.add_argument(
        "--ind2",
        type=str,
        default='MCWSTK_20230401_202510',
        help="the suffix for MCWSTK indicator keys on Wasabi"
    )

    parser.add_argument(
        "--ind3",
        type=str,
        default='MPWSTK_20230401_202510',
        help="the suffix for MPWSTK indicator keys on Wasabi"
    )

    parser.add_argument(
        "--ind4",
        type=str,
        default='NGEX_20230401_202510',
        help="the suffix for NGEX indicator keys on Wasabi"
    )

    parser.add_argument(
        "--spot",
        type=str,
        default='TICKER-A_20230401_202510',
        help="the suffix for TICKER-A keys on Wasabi"
    )

    parser.add_argument(
        "--test",
        type=str,
        default='',
        help="Custom tag added to the output filenames for differentiating runs (e.g '' for base, test 1 etc.)."
    )

    args = parser.parse_args()

    # Run  extraction
    extractor = TSFreshExtractor(args.ticker, args.type, args.path, args.suffix, args.test, args.config, args.ind1, args.ind2, args.ind3, args.ind4, args.spot)
    success = extractor.run()

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
