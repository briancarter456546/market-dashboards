# -*- coding: utf-8 -*-
# =============================================================================
# sma29_entry_backend.py - v1.6
# Last updated: 2026-03-13
# =============================================================================
# v1.6: Health -> Potential (inverted + VIX-boosted)
#   - Scimode validated (n=32,706): low health predicts BETTER forward returns
#   - Q1 health returns +2.17% 21d vs Q5 +1.22% (inverted relationship)
#   - Reframed as "Potential": deeper pullback = more upside potential
#   - VIX conditional boost: +10% at elevated, +20% high, +30% extreme VIX
#   - At VIX 30+, low-health stocks return +13.59% 21d with 84.4% WR
#   - Momentum score VALIDATED: +1.19% Q5-Q1 21d spread (kept as-is)
#
# v1.5: Path quality predictions from scimode_path_quality_v1_0.py
#   - Loads path_quality_lookup.json (3-way: ext_bucket x vix_band x pole_group)
#   - Each ticker gets predicted path grade: SMOOTH / OK / ROUGH / CHOPPY
#   - Based on median MAE + first-green-day from 52,341 historical observations
#   - OOS validated: predicted-bad flag catches 49% bad rate (vs 38% base)
#   - New "Path" column in dashboard with color-coded badge
#
# v1.4: VIX regime banner + gold filtering + validated pole reference
#   - VIX regime banner shows current VIX band and historically strongest sectors
#   - Gold/Precious Metals (pole 5) filtered from OPTIMAL when VIX < 20
#   - Noise poles flagged via validated_poles.json from scimode_pole_validation
#
# v1.3: Per-asset-class extension scoring via pole assignments
#   - Each ticker's primary pole maps to an asset class
#   - Asset-class-specific score adjustments from scimode task #145:
#     Miners: INVERTED behavior (overextension is bullish) -> +30 bonus at 15%+
#     Indexes: danger 3x worse than equities -> -20 penalty at 10%+
#     Bonds: danger threshold at 10% not 25% -> -15 penalty at 10%+
#     Commodities: poor win rate at all extensions -> -10 penalty at 10%+
#     Sector ETFs: slightly better than universal -> +5 bonus at 10%+
#   - Pole-adjusted PF + median_fwd shown per ticker (not universal)
#   - Asset class column added to dashboard
#
# v1.2: Replace Win% with Profit Factor (scimode PF validation)
#   - PF = sum(wins) / sum(|losses|) per extension bucket
#   - PF validated via scimode_pf_validation_v1_0.py (468 tickers, trending filter)
#   - PF captures both win rate AND payoff asymmetry in one number
#   - High-extension PF (30%+) inflated by lottery tails; PF_median tells real story
#   - SMA10 exit thresholds now include PF
#
# v1.1: Added SMA10 exit alert (scimode dual-window finding)
#   - SMA10 at 25%+ extension: median -3.16%, win 43% -> EXIT ALERT
#   - SMA10 at 15-25%: median -0.39%, win 48% -> EXIT WATCH
#   - Renamed dashboard: "Enter & Exit Quality Scanner"
#   - Added column header tooltips defining every metric
#
# v1.0: Initial build - SMA29 Entry Quality dashboard
#   Combines 3 scimode-validated signals into one composite score:
#     1. Momentum quality (from momentum_ranker_data.json)
#     2. Pullback health (from pullback_health_data.json)
#     3. SMA29 extension positioning (from price_cache, scimode-validated)
#
#   SMA29 extension PF + median fwd (scimode_pf_validation, trending filter):
#     0-5%:   PF=1.48  median_fwd=+0.90%  -> OPTIMAL
#     5-10%:  PF=1.50  median_fwd=+1.11%  -> OPTIMAL
#     10-15%: PF=1.52  median_fwd=+1.24%  -> GOOD
#     15-20%: PF=1.67  median_fwd=+1.01%  -> FAIR
#     20-25%: PF=1.58  median_fwd=+0.49%  -> CAUTION
#     25-30%: PF=1.66  median_fwd=+2.33%  -> WARNING (n=321, noisy)
#     30-40%: PF=1.87  median_fwd=+1.46%  -> WARNING (lottery tail inflates PF)
#     40-60%: PF=1.44  median_fwd=-2.21%  -> DANGER (PF_median=1.14, tail-driven)
#     60%+:   PF=1.52  median_fwd=+2.38%  -> EXTREME (n=84, PF_median=0.91)
#
#   SMA10 exit alert (scimode PF validation, 468 tickers):
#     25%+:   PF=1.11  median_fwd=-2.21%  -> EXIT ALERT (PF_median=0.90)
#     15-25%: PF=1.59  median_fwd=+0.49%  -> EXIT WATCH (PF_median=1.07)
#     10-15%: PF=1.51  median_fwd=+0.60%  -> OK
#     <10%:   PF=1.46  normal range, no alert
#
# Run:  python sma29_entry_backend.py
# =============================================================================

import os
import json
import pickle
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from dashboard_writer import DashboardWriter

warnings.filterwarnings('ignore')

# =============================================================================
# PATH SETUP
# =============================================================================

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR   = os.path.normpath(os.path.join(_SCRIPT_DIR, '..', 'perplexity-user-data'))
CACHE_DIR   = os.path.normpath(os.path.join(_DATA_DIR, 'price_cache'))

MOMENTUM_FILE = os.path.join(_SCRIPT_DIR, 'momentum_ranker_data.json')
PULLBACK_FILE = os.path.join(_DATA_DIR, 'pullback_health_data.json')
OUTPUT_JSON   = os.path.join(_DATA_DIR, 'sma29_entry_data.json')
VIX_PKL       = os.path.join(CACHE_DIR, '^VIX.pkl')
VALIDATED_POLES_JSON = os.path.join(_DATA_DIR, 'output', 'scientist', 'validated_poles.json')
PATH_QUALITY_JSON   = os.path.join(_DATA_DIR, 'output', 'scientist', 'path_quality_lookup.json')

# =============================================================================
# VIX REGIME: sector recommendations from scimode_vix_sector_v1_0.py
# Key finding: sector performance varies 4x by VIX band
# Gold worst at ALL VIX levels; Tech consistently best in elevated VIX
# =============================================================================

VIX_BANDS = [
    (0,  15, '<15',   'LOW VOLATILITY'),
    (15, 20, '15-20', 'NORMAL'),
    (20, 25, '20-25', 'ELEVATED'),
    (25, 30, '25-30', 'HIGH'),
    (30, 999, '30+',  'EXTREME'),
]

# Scimode-validated: best-performing poles per VIX band (OPTIMAL extension, 21d fwd)
VIX_SECTOR_RECS = {
    '<15':  ['Semis & Tech', 'Financials', 'Defense', 'LatAm'],
    '15-20': ['Semis & Tech', 'Financials', 'Cybersecurity', 'Consumer Disc'],
    '20-25': ['Telecom', 'Semis & Tech', 'LatAm', 'Financials'],
    '25-30': ['Telecom', 'Semis & Tech', 'LatAm', 'Energy'],
    '30+':   ['Semis & Tech', 'Energy', 'Copper & Metals', 'Defense'],
}

# Pole IDs to EXCLUDE from OPTIMAL recs in low VIX (worst performers)
# Pole 5 = Gold & Precious Metals: worst PF at all VIX levels
GOLD_POLE_IDS = {5}

# =============================================================================
# SCIMODE-VALIDATED EXTENSION BUCKETS
# From scimode_pf_validation_v1_0.py: 468 tickers, trending filter
# (slope>0, rsq>0.10, close>SMA29), 21-day non-overlapping forward returns
# PF = sum(wins) / sum(|losses|) -- captures win rate + payoff asymmetry
# =============================================================================

EXTENSION_BUCKETS = [
    # (lower, upper, pf, median_fwd, label, score)
    (0.0,   5.0,  1.48,  0.90, 'OPTIMAL',  95),
    (5.0,  10.0,  1.50,  1.11, 'OPTIMAL',  90),
    (10.0, 15.0,  1.52,  1.24, 'GOOD',     75),
    (15.0, 20.0,  1.67,  1.01, 'FAIR',     55),
    (20.0, 25.0,  1.58,  0.49, 'CAUTION',  40),
    (25.0, 30.0,  1.66,  2.33, 'WARNING',  25),   # n=321, noisy
    (30.0, 40.0,  1.87,  1.46, 'WARNING',  20),   # lottery tail inflates PF
    (40.0, 60.0,  1.44, -2.21, 'DANGER',   10),   # PF_median=1.14, tail-driven
    (60.0, 999.0, 1.52,  2.38, 'EXTREME',   0),   # n=84, PF_median=0.91
]

# Stocks below SMA29 get a fixed low score (not trending)
BELOW_SMA29_SCORE = 15
BELOW_SMA29_LABEL = 'BELOW SMA'

# =============================================================================
# SMA10 EXIT ALERT BUCKETS
# From scimode_pf_validation_v1_0.py: 468 tickers, trending filter
# SMA10 has 12.7pp win-rate spread (best danger detection of any window)
# =============================================================================

SMA10_EXIT_THRESHOLDS = [
    # (lower, upper, label, pf, median_fwd)
    (25.0, 999.0, 'EXIT ALERT',  1.11, -2.21),   # PF_median=0.90, losing bucket
    (15.0,  25.0, 'EXIT WATCH',  1.59,  0.49),    # PF_median=1.07, barely positive
    (10.0,  15.0, 'ELEVATED',    1.51,  0.60),     # Slightly warm
]
# Below 10%: PF=1.46, no exit alert (normal range)


# =============================================================================
# POLE-TO-ASSET-CLASS MAPPING (from taxonomy_stock_regression_v1_45.py)
# Scimode task #145 proved extension behavior differs by asset class.
# =============================================================================

POLE_DIAGNOSTICS_CSV = os.path.normpath(os.path.join(
    _DATA_DIR, 'output', 'taxonomy', 'stock_polar_diagnostics.csv'))

# Map pole IDs to asset classes for extension scoring
# Based on fmp_pole_metadata.json pole labels
POLE_TO_ASSET_CLASS = {
    # Equity poles
    2: 'equity', 7: 'equity', 8: 'equity', 11: 'equity',
    16: 'equity', 18: 'equity', 19: 'equity', 24: 'equity',
    27: 'equity', 30: 'equity', 37: 'equity', 38: 'equity',
    # Miner poles
    5: 'miner', 22: 'miner', 29: 'miner',
    # Commodity poles
    9: 'commodity', 21: 'commodity', 35: 'commodity', 45: 'commodity',
    # Bond poles
    4: 'bond', 20: 'bond', 26: 'bond',
    # Index/International poles
    1: 'index', 3: 'index', 6: 'index', 10: 'index', 13: 'index',
    15: 'index', 28: 'index', 31: 'index', 33: 'index', 39: 'index',
    42: 'index',
    # Sector ETF poles
    14: 'sector', 32: 'sector', 34: 'sector',
    # Market proxy poles (treat as index)
    17: 'index', 25: 'index', 36: 'index', 41: 'index',
    # Special (treat as equity default)
    12: 'equity', 23: 'equity', 44: 'equity',
}

# Asset-class-specific score adjustments by extension bucket
# From scimode task #145: win_rate and median_fwd per asset class
# Format: {asset_class: {bucket_label: (score_delta, adj_win_rate, adj_median_fwd)}}
#   score_delta: added to universal extension score
#   adj_win_rate: class-specific 21d win rate for this bucket
#   adj_median_fwd: class-specific 21d median forward return
ASSET_CLASS_ADJUSTMENTS = {
    'equity': {
        # Universal buckets are calibrated to equity -- no adjustment
    },
    'miner': {
        # INVERTED: overextension is bullish for miners
        # 15-25%: win 64%, median +3.82% (vs equity 52%, +0.92%)
        # 25-40%: win 61%, median +2.92% (vs equity 50%, +0.22%)
        'OPTIMAL':  (0,  50.7, 0.12),
        'GOOD':     (15, 59.3, 2.55),   # 10-15% = great for miners
        'FAIR':     (30, 64.0, 3.82),   # 15-20% = best zone for miners
        'CAUTION':  (35, 64.0, 3.82),   # 20-25% = still great
        'WARNING':  (40, 61.4, 2.92),   # 25-40% = still bullish
        'DANGER':   (30, 61.4, 2.92),   # 40-60% = treat like warning
        'EXTREME':  (20, 61.4, 2.92),   # 60%+ = treat like caution
    },
    'index': {
        # Danger 3x worse than equities
        # 10-15%: win 40%, median -5.49% (vs equity 53%, +0.96%)
        # 25-40%: win 27%, median -14.83% (vs equity 50%, +0.22%)
        'GOOD':     (-20, 40.0, -5.49),  # 10-15% = already dangerous
        'FAIR':     (-30, 30.8, -10.60), # 15-20% = very dangerous
        'CAUTION':  (-30, 30.8, -10.60), # 20-25%
        'WARNING':  (-20, 27.3, -14.83), # already low score
        'DANGER':   (-10, 27.5, -17.90),
        'EXTREME':  (0,   27.5, -17.90),
    },
    'bond': {
        # Danger threshold at 10% not 25%
        # 10-15%: win 40%, median -0.65% (vs equity 53%, +0.96%)
        # Bonds rarely get to 15%+ (only 52 obs at 10-15%)
        'GOOD':     (-25, 40.4, -0.65),  # 10-15% = danger for bonds
        'FAIR':     (-30, 40.4, -0.65),  # 15%+ extremely rare
        'CAUTION':  (-30, 40.4, -0.65),
        'WARNING':  (-20, 40.4, -0.65),
        'DANGER':   (-10, 40.4, -0.65),
    },
    'commodity': {
        # Poor win rate at extensions, needs SMA10 window
        # 10-15%: win 46%, median -0.90% (vs equity 53%, +0.96%)
        # 15-25%: win 48%, median -1.24%
        'GOOD':     (-15, 46.2, -0.90),  # 10-15%
        'FAIR':     (-20, 47.8, -1.24),  # 15-20%
        'CAUTION':  (-20, 47.8, -1.24),  # 20-25%
        'WARNING':  (-15, 44.4, -0.88),  # 25-40%
        'DANGER':   (-10, 44.4, -0.88),
    },
    'sector': {
        # Slightly better than universal at all levels
        # 10-15%: win 56%, median +1.96% (vs equity 53%, +0.96%)
        # 15-25%: win 62%, median +3.44%
        'GOOD':     (5,  56.2, 1.96),
        'FAIR':     (10, 61.8, 3.44),
        'CAUTION':  (10, 61.8, 3.44),
        'WARNING':  (15, 70.8, 6.84),  # n=24, small but positive
    },
}

ASSET_CLASS_LABELS = {
    'equity': 'Equity',
    'miner': 'Miner',
    'commodity': 'Commodity',
    'bond': 'Bond',
    'index': 'Index',
    'sector': 'Sector',
}


def load_current_vix():
    """Load current VIX level from price_cache/^VIX.pkl."""
    if not os.path.exists(VIX_PKL):
        print("[WARN] VIX pkl not found: {}".format(VIX_PKL))
        return None, None, None
    try:
        vdf = pd.read_pickle(VIX_PKL)
        close_col = 'adjClose' if 'adjClose' in vdf.columns else 'close'
        current_vix = float(vdf[close_col].iloc[-1])
        for vlo, vhi, band_label, band_desc in VIX_BANDS:
            if vlo <= current_vix < vhi:
                return current_vix, band_label, band_desc
        return current_vix, '30+', 'EXTREME'
    except Exception as e:
        print("[WARN] Could not load VIX: {}".format(e))
        return None, None, None


def load_validated_pole_ids():
    """Load validated + marginal pole IDs from scimode output."""
    if not os.path.exists(VALIDATED_POLES_JSON):
        return None, {}
    try:
        with open(VALIDATED_POLES_JSON, 'r', encoding='utf-8') as f:
            vp = json.load(f)
        usable = set(vp.get('usable', []))
        details = vp.get('details', {})
        return usable, details
    except Exception:
        return None, {}


# =============================================================================
# POLE GROUP MAPPING (matches scimode_path_quality_v1_0.py)
# Used for path quality lookup key: ext_bucket|vix_band|pole_group
# =============================================================================

POLE_GROUPS = {
    'US Equity':    [3, 2, 37],
    'Tech':         [16, 27, 8],
    'Financials':   [19, 20],
    'Healthcare':   [7],
    'Energy/Commod': [9, 45, 5],
    'Bonds':        [4],
    'REITs':        [14],
    'Defense':      [11],
    'Telecom':      [18],
    'Materials':    [34],
    'International': [1, 6, 10, 15, 28, 13],
}

POLE_ID_TO_GROUP = {}
for _gname, _pids in POLE_GROUPS.items():
    for _pid in _pids:
        POLE_ID_TO_GROUP[_pid] = _gname


def load_path_quality_lookup():
    """Load path quality lookup from scimode output."""
    if not os.path.exists(PATH_QUALITY_JSON):
        print("[WARN] Path quality lookup not found: {}".format(PATH_QUALITY_JSON))
        return {}, {}
    try:
        with open(PATH_QUALITY_JSON, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get('lookup', {}), data.get('fallback_2way', {})
    except Exception as e:
        print("[WARN] Could not load path quality: {}".format(e))
        return {}, {}


def predict_path_quality(ext_bucket, vix_band, pole_id, pq_lookup, pq_fallback):
    """Look up predicted path quality for an entry.

    Returns dict with: grade, median_mae, median_ttg, prob_good, prob_bad
    Grade: SMOOTH (prob_good >= 0.42), OK (>= 0.30), ROUGH (>= 0.20), CHOPPY (< 0.20)
    """
    pole_group = POLE_ID_TO_GROUP.get(pole_id, 'Other')

    # Map extension labels to lookup keys
    ext_key_map = {
        'OPTIMAL': None,  # need to split 0-5% vs 5-10%
        'GOOD': '10-15%',
        'FAIR': '15-25%',
        'CAUTION': '15-25%',
        'WARNING': '25%+',
        'DANGER': '25%+',
        'EXTREME': '25%+',
        'BELOW SMA': None,
    }

    # Try 3-way lookup first
    if ext_bucket and vix_band and pole_group != 'Other':
        key_3 = '{}|{}|{}'.format(ext_bucket, vix_band, pole_group)
        if key_3 in pq_lookup:
            cell = pq_lookup[key_3]
            return _grade_from_cell(cell)

    # 2-way fallback
    if ext_bucket and vix_band:
        key_2 = '{}|{}'.format(ext_bucket, vix_band)
        if key_2 in pq_fallback:
            cell = pq_fallback[key_2]
            return _grade_from_cell(cell)

    return None


def _grade_from_cell(cell):
    """Convert lookup cell to path quality grade."""
    prob_good = cell.get('prob_good', 0)
    prob_bad = cell.get('prob_bad', 0)
    median_mae = cell.get('median_mae', 0)
    median_ttg = cell.get('median_ttg', 0)

    if prob_good >= 0.42:
        grade = 'SMOOTH'
    elif prob_good >= 0.30:
        grade = 'OK'
    elif prob_good >= 0.20:
        grade = 'ROUGH'
    else:
        grade = 'CHOPPY'

    return {
        'grade': grade,
        'median_mae': median_mae,
        'median_ttg': median_ttg,
        'prob_good': round(prob_good * 100, 1),
        'prob_bad': round(prob_bad * 100, 1),
    }


def load_pole_assignments():
    """Load pole assignments from taxonomy CSV -> {ticker: (asset_class, pole_id)}."""
    if not os.path.exists(POLE_DIAGNOSTICS_CSV):
        print("[WARN] Pole diagnostics not found: {}".format(POLE_DIAGNOSTICS_CSV))
        return {}
    try:
        df = pd.read_csv(POLE_DIAGNOSTICS_CSV, encoding='utf-8')
        result = {}
        for _, row in df.iterrows():
            ticker = row['ticker']
            pole_id = row['primary_pole']
            if pd.isna(pole_id):
                result[ticker] = ('equity', None)
            else:
                pid = int(pole_id)
                result[ticker] = (POLE_TO_ASSET_CLASS.get(pid, 'equity'), pid)
        return result
    except Exception as e:
        print("[WARN] Could not load pole assignments: {}".format(e))
        return {}


# =============================================================================
# HELPERS
# =============================================================================

def clean_nan(obj):
    """Replace NaN/Inf with None for JSON serialization."""
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    elif isinstance(obj, (np.floating, np.integer)):
        val = float(obj)
        if np.isnan(val) or np.isinf(val):
            return None
        return val
    elif isinstance(obj, dict):
        return {k: clean_nan(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_nan(item) for item in obj]
    return obj


def classify_extension(ext_pct, asset_class='equity'):
    """Map extension % to bucket label, score, PF, and median fwd.

    If asset_class is provided and has adjustments, applies:
      - score_delta to the universal score (clamped 0-100)
      - class-specific win_rate and median_fwd override
    """
    if ext_pct < 0:
        return BELOW_SMA29_LABEL, BELOW_SMA29_SCORE, None, None
    for lower, upper, pf, med_fwd, label, score in EXTENSION_BUCKETS:
        if lower <= ext_pct < upper:
            # Apply asset-class adjustment if available
            adj = ASSET_CLASS_ADJUSTMENTS.get(asset_class, {}).get(label)
            if adj:
                score_delta, adj_wr, adj_fwd = adj
                score = max(0, min(100, score + score_delta))
                med_fwd = adj_fwd
            return label, score, pf, med_fwd
    # EXTREME fallback
    adj = ASSET_CLASS_ADJUSTMENTS.get(asset_class, {}).get('EXTREME')
    if adj:
        score_delta, adj_wr, adj_fwd = adj
        return 'EXTREME', max(0, min(100, 0 + score_delta)), 1.52, adj_fwd
    return 'EXTREME', 0, 1.52, 2.38


def classify_sma10_exit(ext10_pct):
    """Map SMA10 extension % to exit alert level."""
    if ext10_pct is None:
        return '', None, None
    for lower, upper, label, pf, med_fwd in SMA10_EXIT_THRESHOLDS:
        if ext10_pct >= lower:
            return label, pf, med_fwd
    return '', None, None


# =============================================================================
# DATA LOADERS
# =============================================================================

def load_momentum_data():
    """Load momentum ranker JSON -> dict keyed by ticker."""
    if not os.path.exists(MOMENTUM_FILE):
        print("[WARN] Momentum ranker data not found: {}".format(MOMENTUM_FILE))
        return {}
    with open(MOMENTUM_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    rows = data.get('data', [])
    return {r['ticker']: r for r in rows}


def load_pullback_data():
    """Load pullback health JSON -> dict keyed by ticker."""
    if not os.path.exists(PULLBACK_FILE):
        print("[WARN] Pullback health data not found: {}".format(PULLBACK_FILE))
        return {}
    with open(PULLBACK_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    results = data.get('results', [])
    return {r['ticker']: r for r in results}


def compute_sma29_extension(ticker):
    """Load price data, compute SMA29, return extension info."""
    pkl_path = os.path.join(CACHE_DIR, '{}.pkl'.format(ticker))
    if not os.path.exists(pkl_path):
        return None

    try:
        df = pd.read_pickle(pkl_path)
        if len(df) < 60:
            return None

        close = df['adjClose'] if 'adjClose' in df.columns else df['close']
        close = close.dropna()
        if len(close) < 29:
            return None

        sma29 = close.rolling(29).mean()
        latest_close = float(close.iloc[-1])
        latest_sma29 = float(sma29.iloc[-1])

        if np.isnan(latest_sma29) or latest_sma29 <= 0:
            return None

        extension_pct = ((latest_close - latest_sma29) / latest_sma29) * 100.0

        # SMA10 for exit alert
        sma10 = close.rolling(10).mean()
        latest_sma10 = float(sma10.iloc[-1])
        if np.isnan(latest_sma10) or latest_sma10 <= 0:
            ext10_pct = None
        else:
            ext10_pct = ((latest_close - latest_sma10) / latest_sma10) * 100.0

        # Also compute slope quality (21-day log-price R-squared)
        if len(close) >= 21:
            log_prices = np.log(close.iloc[-21:].values)
            x = np.arange(21)
            if np.all(np.isfinite(log_prices)):
                slope, intercept = np.polyfit(x, log_prices, 1)
                predicted = slope * x + intercept
                ss_res = np.sum((log_prices - predicted) ** 2)
                ss_tot = np.sum((log_prices - np.mean(log_prices)) ** 2)
                rsq = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                slope_ann = slope * 252 * 100  # annualized %
            else:
                rsq = 0
                slope_ann = 0
        else:
            rsq = 0
            slope_ann = 0

        return {
            'close': round(latest_close, 2),
            'sma29': round(latest_sma29, 2),
            'extension_pct': round(extension_pct, 2),
            'sma10': round(latest_sma10, 2) if ext10_pct is not None else None,
            'ext10_pct': round(ext10_pct, 2) if ext10_pct is not None else None,
            'slope_rsq': round(rsq, 3),
            'slope_ann': round(slope_ann, 1),
        }
    except Exception:
        return None


# =============================================================================
# SCORING
# =============================================================================

# Component weights (must sum to 1.0)
W_MOMENTUM  = 0.35
W_POTENTIAL = 0.30
W_EXTENSION = 0.35


def score_momentum(mr_data):
    """Score 0-100 from momentum ranker score. Already 0-100."""
    if mr_data is None:
        return 0
    return min(100, max(0, float(mr_data.get('score', 0))))


def score_potential(pb_data, current_vix=None):
    """Score 0-100 from pullback health, INVERTED to 'potential'.

    Scimode validated (2026-03-13, n=32,706): low health (deep pullback)
    predicts BETTER forward returns. Q1 health returns +2.17% 21d vs
    Q5 +1.22%. Reframed as 'potential': deeper pullback = more upside.

    VIX boost: at elevated+ VIX, low-health stocks return even more
    (VIX 30+: +13.59% 21d, 84.4% WR). Apply VIX multiplier.
    """
    if pb_data is None:
        return 50  # neutral if no data
    health = min(100, max(0, float(pb_data.get('health', 0) or 0)))
    # Invert: low health = high potential
    potential = 100 - health

    # VIX boost: amplify potential in high-VIX environments
    if current_vix is not None:
        if current_vix >= 30:
            potential = min(100, potential * 1.3)   # 30% boost at extreme VIX
        elif current_vix >= 25:
            potential = min(100, potential * 1.2)   # 20% boost at high VIX
        elif current_vix >= 20:
            potential = min(100, potential * 1.1)   # 10% boost at elevated VIX

    return potential


def score_extension(ext_pct):
    """Score 0-100 from SMA29 extension bucket."""
    _, score, _, _ = classify_extension(ext_pct)
    return score


def combined_score(momentum_score, potential_score, extension_score):
    """Weighted composite. Scimode validated 2026-03-13."""
    return round(
        W_MOMENTUM * momentum_score +
        W_POTENTIAL * potential_score +
        W_EXTENSION * extension_score, 1
    )


# =============================================================================
# MAIN COMPUTATION
# =============================================================================

def build_universe():
    """Build scored universe from all 3 data sources + pole assignments + VIX."""
    print("[1/6] Loading momentum ranker data...")
    mr_data = load_momentum_data()
    print("  {} tickers".format(len(mr_data)))

    print("[2/6] Loading pullback health data...")
    pb_data = load_pullback_data()
    print("  {} tickers".format(len(pb_data)))

    print("[3/6] Loading pole assignments + validated poles...")
    pole_map = load_pole_assignments()
    print("  {} tickers with pole data".format(len(pole_map)))
    usable_poles, pole_details = load_validated_pole_ids()
    if usable_poles:
        print("  {} validated/marginal poles loaded".format(len(usable_poles)))

    print("[4/7] Loading VIX regime...")
    current_vix, vix_band, vix_desc = load_current_vix()
    if current_vix is not None:
        print("  VIX = {:.1f} (band: {} / {})".format(current_vix, vix_band, vix_desc))
    else:
        print("  [WARN] VIX unavailable, skipping regime features")

    print("[5/7] Loading path quality lookup...")
    pq_lookup, pq_fallback = load_path_quality_lookup()
    if pq_lookup:
        print("  {} 3-way cells, {} 2-way fallbacks loaded".format(len(pq_lookup), len(pq_fallback)))
    else:
        print("  [WARN] No path quality data - column will be empty")

    # Universe = union of momentum ranker + pullback health tickers
    all_tickers = sorted(set(list(mr_data.keys()) + list(pb_data.keys())))
    print("[6/7] Computing SMA29 extensions for {} tickers...".format(len(all_tickers)))

    # Parallel SMA29 computation
    ext_data = {}
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(compute_sma29_extension, t): t for t in all_tickers}
        done = 0
        for future in as_completed(futures):
            done += 1
            ticker = futures[future]
            result = future.result()
            if result is not None:
                ext_data[ticker] = result
            if done % 200 == 0:
                print("  ... computed {}/{}".format(done, len(all_tickers)))
    print("  {} tickers with valid SMA29 data".format(len(ext_data)))

    print("[7/7] Scoring with per-asset-class adjustments + path quality...")
    # Count asset class distribution
    ac_counts = {}
    gold_filtered_count = 0
    noise_pole_count = 0
    results = []
    for ticker in all_tickers:
        mr = mr_data.get(ticker)
        pb = pb_data.get(ticker)
        ext = ext_data.get(ticker)

        if ext is None:
            continue  # need at least price data

        # Look up asset class + pole_id from pole assignment
        ac_pid = pole_map.get(ticker, ('equity', None))
        asset_class, pole_id = ac_pid
        ac_counts[asset_class] = ac_counts.get(asset_class, 0) + 1

        # Check if this is a noise pole
        is_noise_pole = False
        if usable_poles is not None and pole_id is not None:
            if pole_id not in usable_poles:
                is_noise_pole = True
                noise_pole_count += 1

        ext_pct = ext['extension_pct']
        ext_label, ext_score_val, ext_pf, ext_fwd = classify_extension(ext_pct, asset_class)

        # Gold filtering: demote Gold/Precious Metals from OPTIMAL when VIX < 20
        gold_filtered = False
        if (pole_id is not None and pole_id in GOLD_POLE_IDS
                and ext_label == 'OPTIMAL'
                and current_vix is not None and current_vix < 20):
            gold_filtered = True
            gold_filtered_count += 1

        # SMA10 exit alert
        ext10_pct = ext.get('ext10_pct')
        exit_alert, exit_pf, exit_fwd = classify_sma10_exit(ext10_pct)

        m_score = score_momentum(mr)
        p_score = score_potential(pb, current_vix)
        e_score = ext_score_val

        combo = combined_score(m_score, p_score, e_score)

        # Path quality prediction
        # Map extension % to lookup bucket key
        if ext_pct < 0:
            pq_ext_bucket = None
        elif ext_pct < 5:
            pq_ext_bucket = '0-5%'
        elif ext_pct < 10:
            pq_ext_bucket = '5-10%'
        elif ext_pct < 15:
            pq_ext_bucket = '10-15%'
        elif ext_pct < 25:
            pq_ext_bucket = '15-25%'
        else:
            pq_ext_bucket = '25%+'

        path_quality = None
        if pq_ext_bucket and vix_band and pq_lookup:
            path_quality = predict_path_quality(
                pq_ext_bucket, vix_band, pole_id, pq_lookup, pq_fallback)

        row = {
            'ticker': ticker,
            'price': ext['close'],
            'sma29': ext['sma29'],
            'extension_pct': ext_pct,
            'extension_label': ext_label,
            'extension_score': e_score,
            'extension_pf': ext_pf,
            'extension_fwd': ext_fwd,
            'sma10': ext.get('sma10'),
            'ext10_pct': ext10_pct,
            'exit_alert': exit_alert,
            'exit_pf': exit_pf,
            'exit_fwd': exit_fwd,
            'slope_rsq': ext['slope_rsq'],
            'slope_ann': ext['slope_ann'],
            'momentum_score': round(m_score, 1),
            'momentum_rank': int(mr.get('rank', 9999)) if mr else None,
            'momentum_flag': mr.get('momentum_flag', '') if mr else '',
            'potential_score': round(p_score, 1),
            'health_verdict': pb.get('verdict', '') if pb else '',
            'dd_pct': round(float(pb.get('dd_pct', 0) or 0), 1) if pb else None,
            'stage': pb.get('stage', None) if pb else None,
            'stage_name': pb.get('stage_name', '') if pb else '',
            'ret_1w': round(float(mr.get('ret_1w', 0) or 0), 2) if mr else None,
            'ret_1m': round(float(mr.get('ret_1m', 0) or 0), 2) if mr else None,
            'ret_3m': round(float(mr.get('ret_3m', 0) or 0), 2) if mr else None,
            'top_pole': mr.get('top_pole', '') if mr else '',
            'asset_class': asset_class,
            'asset_class_label': ASSET_CLASS_LABELS.get(asset_class, 'Equity'),
            'pole_id': pole_id,
            'is_noise_pole': is_noise_pole,
            'gold_filtered': gold_filtered,
            'path_quality': path_quality,
            'combined_score': combo,
        }
        results.append(row)

    # Print asset class distribution
    print("  Asset class distribution:")
    for ac, cnt in sorted(ac_counts.items(), key=lambda x: -x[1]):
        print("    {:12s} {:,}".format(ASSET_CLASS_LABELS.get(ac, ac), cnt))
    if noise_pole_count:
        print("  {} tickers in noise poles (flagged)".format(noise_pole_count))
    if gold_filtered_count:
        print("  {} Gold/PM tickers filtered from OPTIMAL (VIX < 20)".format(gold_filtered_count))

    results.sort(key=lambda r: r['combined_score'], reverse=True)

    vix_context = {
        'current_vix': current_vix,
        'vix_band': vix_band,
        'vix_desc': vix_desc,
    }

    return results, vix_context


# =============================================================================
# HTML DASHBOARD
# =============================================================================

EXTRA_CSS = """
/* Extension zone badges */
.ext-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.78em;
    font-weight: 700;
    letter-spacing: 0.5px;
}
.ext-OPTIMAL  { background: #dcfce7; color: #166534; }
.ext-GOOD     { background: #e0f2fe; color: #075985; }
.ext-FAIR     { background: #fef9c3; color: #854d0e; }
.ext-CAUTION  { background: #fed7aa; color: #9a3412; }
.ext-WARNING  { background: #fecaca; color: #991b1b; }
.ext-DANGER   { background: #f87171; color: #fff; }
.ext-EXTREME  { background: #991b1b; color: #fff; }
.ext-BELOW    { background: #e5e7eb; color: #6b7280; }

/* SMA10 Exit alert badges */
.exit-alert   { background: #dc2626; color: #fff; font-weight: 700; padding: 2px 8px; border-radius: 4px; font-size: 0.78em; animation: pulse-exit 1.5s infinite; }
.exit-watch   { background: #f97316; color: #fff; font-weight: 700; padding: 2px 8px; border-radius: 4px; font-size: 0.78em; }
.exit-elevated { background: #fbbf24; color: #78350f; font-weight: 600; padding: 2px 8px; border-radius: 4px; font-size: 0.78em; }
@keyframes pulse-exit { 0%,100% { opacity: 1; } 50% { opacity: 0.7; } }

/* Path quality badges */
.path-badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.78em; font-weight: 700; letter-spacing: 0.5px; }
.path-SMOOTH { background: #dcfce7; color: #166534; }
.path-OK     { background: #e0f2fe; color: #075985; }
.path-ROUGH  { background: #fed7aa; color: #9a3412; }
.path-CHOPPY { background: #fecaca; color: #991b1b; }

/* Filter bar */
#filter-count { font-weight: 600; }

/* Tooltip on column headers */
th[title] { cursor: help; border-bottom: 1px dashed #999; }

/* Compact table to prevent horizontal overflow */
#mainTable { font-size: 0.82em; }
#mainTable th, #mainTable td { padding: 4px 6px; white-space: nowrap; }

/* Score bars */
.score-bar {
    display: inline-block;
    height: 8px;
    border-radius: 4px;
    vertical-align: middle;
}
.score-bar-m { background: #6366f1; }
.score-bar-h { background: #22c55e; }
.score-bar-e { background: #f59e0b; }

/* Summary cards */
.summary-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
    gap: 12px;
    margin-bottom: 20px;
}
.summary-card {
    background: #fff;
    border: 1px solid #e2e4e8;
    border-radius: 8px;
    padding: 14px 16px;
    text-align: center;
}
.summary-card .sc-value {
    font-size: 1.6em;
    font-weight: 700;
    color: #1a1a2e;
}
.summary-card .sc-label {
    font-size: 0.78em;
    color: #888;
    margin-top: 2px;
}

/* Component mini-bars */
.component-bars {
    display: flex;
    gap: 3px;
    align-items: center;
}
.component-bars .cb {
    height: 6px;
    border-radius: 3px;
    min-width: 2px;
}
"""

EXTRA_JS = """
// Fast sort: pre-extract values, sort in memory, reattach once
function sortTable(colIdx, numeric) {
    var table = document.getElementById('mainTable');
    var tbody = table.tBodies[0];
    var rows = Array.from(tbody.rows);
    var asc = table.getAttribute('data-sort-col') == colIdx
              && table.getAttribute('data-sort-dir') == 'asc';
    // Pre-extract sort keys to avoid repeated DOM access
    var keyed = rows.map(function(r) {
        var v = r.cells[colIdx].getAttribute('data-val');
        if (v === null) v = r.cells[colIdx].textContent;
        if (numeric) { var p = parseFloat(v); v = isNaN(p) ? -9999 : p; }
        return {row: r, val: v};
    });
    keyed.sort(function(a, b) {
        if (a.val < b.val) return asc ? 1 : -1;
        if (a.val > b.val) return asc ? -1 : 1;
        return 0;
    });
    table.setAttribute('data-sort-col', colIdx);
    table.setAttribute('data-sort-dir', asc ? 'desc' : 'asc');
    // Single DOM reflow: append fragment
    var frag = document.createDocumentFragment();
    keyed.forEach(function(k) { frag.appendChild(k.row); });
    tbody.appendChild(frag);
}

// Unified filter system: all dropdowns + ownership
function applyFilters() {
    var table = document.getElementById('mainTable');
    var rows = table.tBodies[0].rows;
    // Read all filter values
    var fZone  = document.getElementById('f-zone').value;
    var fExit  = document.getElementById('f-exit').value;
    var fPath  = document.getElementById('f-path').value;
    var fStage = document.getElementById('f-stage').value;
    var fPole  = document.getElementById('f-pole').value;
    var fClass = document.getElementById('f-class').value;
    var fOwned = document.getElementById('f-owned').value;
    var shown = 0;
    for (var i = 0; i < rows.length; i++) {
        var show = true;
        if (fZone !== 'all' && rows[i].getAttribute('data-zone') !== fZone) show = false;
        if (show && fExit !== 'all') {
            var re = rows[i].getAttribute('data-exit') || 'none';
            if (fExit === 'any-alert' && re === 'none') show = false;
            else if (fExit !== 'any-alert' && re !== fExit) show = false;
        }
        if (show && fPath !== 'all' && rows[i].getAttribute('data-path') !== fPath) show = false;
        if (show && fStage !== 'all' && rows[i].getAttribute('data-stage') !== fStage) show = false;
        if (show && fPole !== 'all' && rows[i].getAttribute('data-pole') !== fPole) show = false;
        if (show && fClass !== 'all' && rows[i].getAttribute('data-class') !== fClass) show = false;
        if (show && fOwned !== 'all') {
            var ticker = rows[i].querySelector('.own-cb');
            if (ticker) ticker = ticker.getAttribute('data-ticker');
            var isOwned = ticker && window._owned && window._owned.has(ticker);
            var isWatched = ticker && window._watched && window._watched.has(ticker);
            if (fOwned === 'owned' && !isOwned) show = false;
            if (fOwned === 'watched' && !isWatched) show = false;
            if (fOwned === 'not-owned' && isOwned) show = false;
        }
        rows[i].style.display = show ? '' : 'none';
        if (show) shown++;
    }
    document.getElementById('filter-count').textContent = shown + ' shown';
}
"""


def _own_cell(ticker):
    """Own checkbox cell."""
    return '<td class="tc"><input type="checkbox" class="own-cb" data-ticker="{t}" {chk} onchange="window._ownToggle(\'{t}\', this)"></td>'.format(
        t=ticker, chk='checked' if False else '')


def _watch_cell(ticker):
    """Watch checkbox cell."""
    return '<td class="tc"><input type="checkbox" class="watch-cb" data-ticker="{t}" {chk} onchange="window._watchToggle(\'{t}\', this)"></td>'.format(
        t=ticker, chk='checked' if False else '')


def _score_bar(val, max_val, css_class):
    """Mini bar for component scores."""
    w = max(0, min(100, val / max_val * 100))
    return '<span class="score-bar {cls}" style="width:{w}px"></span>'.format(
        cls=css_class, w=int(w * 0.6))


def _ext_badge(label):
    """Extension zone badge."""
    css = label.replace(' ', '')
    if css == 'BELOWSMA':
        css = 'BELOW'
    return '<span class="ext-badge ext-{css}">{lbl}</span>'.format(css=css, lbl=label)


def _exit_badge(alert_label):
    """SMA10 exit alert badge."""
    if not alert_label:
        return '<td class="tc" data-val="0">-</td>'
    css_map = {'EXIT ALERT': 'exit-alert', 'EXIT WATCH': 'exit-watch', 'ELEVATED': 'exit-elevated'}
    val_map = {'EXIT ALERT': 3, 'EXIT WATCH': 2, 'ELEVATED': 1}
    css = css_map.get(alert_label, '')
    val = val_map.get(alert_label, 0)
    return '<td class="tc" data-val="{v}"><span class="{css}">{lbl}</span></td>'.format(
        v=val, css=css, lbl=alert_label)


def _color_pct(val):
    """Color a percentage value."""
    if val is None:
        return '<td class="tr">-</td>'
    color = '#22c55e' if val > 0 else '#ef4444' if val < 0 else '#888'
    return '<td class="tr" style="color:{c}" data-val="{v}">{v:+.1f}%</td>'.format(
        c=color, v=val)


def build_html(results, vix_context=None):
    """Build the dashboard HTML."""
    now = datetime.now()
    if vix_context is None:
        vix_context = {}
    current_vix = vix_context.get('current_vix')
    vix_band = vix_context.get('vix_band')
    vix_desc = vix_context.get('vix_desc')

    # Counts by zone
    zone_counts = {}
    for r in results:
        z = r['extension_label']
        zone_counts[z] = zone_counts.get(z, 0) + 1

    # Summary stats
    above_sma = sum(1 for r in results if r['extension_pct'] >= 0)
    optimal = sum(1 for r in results if r['extension_label'] == 'OPTIMAL')
    danger_plus = sum(1 for r in results if r['extension_label'] in ('DANGER', 'EXTREME'))
    exit_alerts = sum(1 for r in results if r.get('exit_alert') == 'EXIT ALERT')
    exit_watches = sum(1 for r in results if r.get('exit_alert') == 'EXIT WATCH')
    avg_combo = np.mean([r['combined_score'] for r in results]) if results else 0
    top50_avg = np.mean([r['combined_score'] for r in results[:50]]) if len(results) >= 50 else avg_combo

    # Banner
    if optimal > 100:
        banner_text = "STRONG FIELD - {} stocks in optimal SMA29 zone".format(optimal)
        banner_class = "banner-bull"
    elif optimal > 50:
        banner_text = "DECENT FIELD - {} stocks in optimal zone, {} total above SMA29".format(optimal, above_sma)
        banner_class = "banner-neutral"
    elif above_sma > len(results) * 0.5:
        banner_text = "MIXED FIELD - Few optimal entries, most overextended or weak"
        banner_class = "banner-neutral"
    else:
        banner_text = "WEAK FIELD - Only {} above SMA29, {} in danger+ zone".format(above_sma, danger_plus)
        banner_class = "banner-bear"

    # Summary cards HTML
    summary_html = """
    <div class="summary-grid">
        <div class="summary-card">
            <div class="sc-value">{}</div>
            <div class="sc-label">Universe</div>
        </div>
        <div class="summary-card">
            <div class="sc-value" style="color:#22c55e">{}</div>
            <div class="sc-label">Above SMA29</div>
        </div>
        <div class="summary-card">
            <div class="sc-value" style="color:#166534">{}</div>
            <div class="sc-label">Optimal Zone (0-10%)</div>
        </div>
        <div class="summary-card">
            <div class="sc-value" style="color:#ef4444">{}</div>
            <div class="sc-label">Danger / Extreme</div>
        </div>
        <div class="summary-card">
            <div class="sc-value" style="color:#dc2626">{}</div>
            <div class="sc-label">Exit Alerts (SMA10)</div>
        </div>
        <div class="summary-card">
            <div class="sc-value" style="color:#f97316">{}</div>
            <div class="sc-label">Exit Watch (SMA10)</div>
        </div>
        <div class="summary-card">
            <div class="sc-value">{:.1f}</div>
            <div class="sc-label">Avg Combined Score</div>
        </div>
        <div class="summary-card">
            <div class="sc-value">{:.1f}</div>
            <div class="sc-label">Top 50 Avg Score</div>
        </div>
    </div>
    """.format(len(results), above_sma, optimal, danger_plus, exit_alerts, exit_watches, avg_combo, top50_avg)

    # Collect unique values for filter dropdowns
    exit_vals = sorted(set(r.get('exit_alert', '') or 'none' for r in results))
    path_vals = sorted(set((r.get('path_quality') or {}).get('grade', 'none') for r in results))
    stage_names_map = {0: 'Decline', 1: 'Basing', 2: 'Uptrend', 3: 'Parabolic'}
    stage_vals = sorted(set(stage_names_map.get(r.get('stage'), r.get('stage_name', '-')) for r in results))
    pole_vals = sorted(set(r.get('top_pole', '') or 'none' for r in results))
    class_vals = sorted(set(r.get('asset_class_label', 'Equity') for r in results))

    def _dropdown(id_attr, label, options):
        """Build a filter dropdown."""
        html = '<label style="font-size:0.82em;margin-right:12px">{}: '.format(label)
        html += '<select id="{}" onchange="applyFilters()" style="padding:3px 6px;border:1px solid #ccc;border-radius:4px;font-size:0.95em">'.format(id_attr)
        html += '<option value="all">All</option>'
        for opt in options:
            if opt == 'none' or opt == '-' or opt == '':
                continue
            html += '<option value="{}">{}</option>'.format(opt, opt)
        html += '</select></label>'
        return html

    filter_bar = '<div style="margin-bottom:14px;display:flex;flex-wrap:wrap;align-items:center;gap:4px 0">'
    # Zone filter
    zone_order = ['OPTIMAL', 'GOOD', 'FAIR', 'CAUTION', 'WARNING', 'DANGER', 'EXTREME', 'BELOW SMA']
    filter_bar += _dropdown('f-zone', 'Zone', zone_order)
    # Exit filter
    exit_options = ['EXIT ALERT', 'EXIT WATCH', 'ELEVATED']
    filter_bar += '<label style="font-size:0.82em;margin-right:12px">Exit: '
    filter_bar += '<select id="f-exit" onchange="applyFilters()" style="padding:3px 6px;border:1px solid #ccc;border-radius:4px;font-size:0.95em">'
    filter_bar += '<option value="all">All</option>'
    filter_bar += '<option value="any-alert">Any Alert</option>'
    for opt in exit_options:
        filter_bar += '<option value="{}">{}</option>'.format(opt, opt)
    filter_bar += '</select></label>'
    # Path filter
    path_order = ['SMOOTH', 'OK', 'ROUGH', 'CHOPPY']
    filter_bar += _dropdown('f-path', 'Path', path_order)
    # Stage filter
    filter_bar += _dropdown('f-stage', 'Stage', ['Basing', 'Uptrend', 'Parabolic', 'Decline'])
    # Pole filter
    pole_vals_clean = [p for p in pole_vals if p and p != 'none']
    filter_bar += _dropdown('f-pole', 'Pole', pole_vals_clean)
    # Class filter
    filter_bar += _dropdown('f-class', 'Class', class_vals)
    # Owned filter
    filter_bar += '<label style="font-size:0.82em;margin-right:12px">Show: '
    filter_bar += '<select id="f-owned" onchange="applyFilters()" style="padding:3px 6px;border:1px solid #ccc;border-radius:4px;font-size:0.95em">'
    filter_bar += '<option value="all">All</option>'
    filter_bar += '<option value="owned">Owned</option>'
    filter_bar += '<option value="watched">Watched</option>'
    filter_bar += '<option value="not-owned">Not Owned</option>'
    filter_bar += '</select></label>'
    # Count
    filter_bar += '<span id="filter-count" style="font-size:0.82em;color:#888;margin-left:8px">{} shown</span>'.format(len(results))
    filter_bar += '</div>'

    # Methodology note
    methodology = """
    <details style="margin-bottom:16px">
    <summary style="cursor:pointer;font-weight:600;color:#4a5568">Methodology (scimode-validated)</summary>
    <div style="padding:8px 12px;font-size:0.82em;color:#555;line-height:1.6">
    <p><b>Combined Score</b> = 35% Momentum + 30% Potential + 35% SMA29 Extension</p>
    <p><b>Profit Factor</b> = sum(winning trades) / sum(|losing trades|). PF &gt; 1.0 = profitable edge, PF &lt; 1.0 = losing. Captures both win rate and payoff asymmetry in one number.</p>
    <p><b>SMA29 Extension</b> = (Close - SMA29) / SMA29. PF from scimode_pf_validation (468 tickers, trending filter):</p>
    <table style="font-size:0.9em;border-collapse:collapse;margin:6px 0">
    <tr><th style="padding:2px 8px;text-align:left">Zone</th><th style="padding:2px 8px">Extension</th><th style="padding:2px 8px">PF</th><th style="padding:2px 8px">Median 21d Fwd</th><th style="padding:2px 8px">Score</th></tr>
    <tr><td style="padding:2px 8px"><span class="ext-badge ext-OPTIMAL">OPTIMAL</span></td><td style="padding:2px 8px;text-align:center">0-10%</td><td style="padding:2px 8px;text-align:center">1.48-1.50</td><td style="padding:2px 8px;text-align:center">+0.90 to +1.11%</td><td style="padding:2px 8px;text-align:center">90-95</td></tr>
    <tr><td style="padding:2px 8px"><span class="ext-badge ext-GOOD">GOOD</span></td><td style="padding:2px 8px;text-align:center">10-15%</td><td style="padding:2px 8px;text-align:center">1.52</td><td style="padding:2px 8px;text-align:center">+1.24%</td><td style="padding:2px 8px;text-align:center">75</td></tr>
    <tr><td style="padding:2px 8px"><span class="ext-badge ext-FAIR">FAIR</span></td><td style="padding:2px 8px;text-align:center">15-20%</td><td style="padding:2px 8px;text-align:center">1.67</td><td style="padding:2px 8px;text-align:center">+1.01%</td><td style="padding:2px 8px;text-align:center">55</td></tr>
    <tr><td style="padding:2px 8px"><span class="ext-badge ext-CAUTION">CAUTION</span></td><td style="padding:2px 8px;text-align:center">20-25%</td><td style="padding:2px 8px;text-align:center">1.58</td><td style="padding:2px 8px;text-align:center">+0.49%</td><td style="padding:2px 8px;text-align:center">40</td></tr>
    <tr><td style="padding:2px 8px"><span class="ext-badge ext-WARNING">WARNING</span></td><td style="padding:2px 8px;text-align:center">25-40%</td><td style="padding:2px 8px;text-align:center">1.66-1.87*</td><td style="padding:2px 8px;text-align:center">+1.46 to +2.33%</td><td style="padding:2px 8px;text-align:center">20-25</td></tr>
    <tr><td style="padding:2px 8px"><span class="ext-badge ext-DANGER">DANGER</span></td><td style="padding:2px 8px;text-align:center">40-60%</td><td style="padding:2px 8px;text-align:center">1.44*</td><td style="padding:2px 8px;text-align:center">-2.21%</td><td style="padding:2px 8px;text-align:center">10</td></tr>
    <tr><td style="padding:2px 8px"><span class="ext-badge ext-EXTREME">EXTREME</span></td><td style="padding:2px 8px;text-align:center">60%+</td><td style="padding:2px 8px;text-align:center">1.52*</td><td style="padding:2px 8px;text-align:center">+2.38%</td><td style="padding:2px 8px;text-align:center">0</td></tr>
    </table>
    <p style="font-size:0.85em;color:#888">* WARNING/DANGER/EXTREME PF inflated by lottery-tail winners. PF_median (typical trader): WARNING 0.64-1.47, DANGER 1.14, EXTREME 0.91. Small samples (n=84-357).</p>
    <p><b>Momentum Score</b>: From momentum_ranker_v1_18 composite (returns + ratios + SPY-relative days).</p>
    <p><b>Potential Score</b>: Inverted pullback health (deep pullback = high potential). Scimode validated: low-health stocks return +2.17% 21d vs high-health +1.22%. VIX-boosted: at VIX 30+, deep pullbacks return +13.6% 21d with 84% WR.</p>
    <p><b>Ideal entry</b>: High momentum + high potential (deep pullback in uptrend) + optimal SMA29 zone (0-10% extension, PF ~1.5). Potential is VIX-boosted: deeper pullbacks at elevated VIX have the strongest forward returns.</p>
    <p style="margin-top:10px"><b>SMA10 Exit Alert</b> (scimode PF validation, 468 tickers):</p>
    <p>SMA10 detects overextension faster than SMA29. PF validates exit signals.</p>
    <table style="font-size:0.9em;border-collapse:collapse;margin:6px 0">
    <tr><th style="padding:2px 8px;text-align:left">Alert</th><th style="padding:2px 8px">SMA10 Ext</th><th style="padding:2px 8px">PF</th><th style="padding:2px 8px">Median 21d Fwd</th></tr>
    <tr><td style="padding:2px 8px"><span class="exit-alert">EXIT ALERT</span></td><td style="padding:2px 8px;text-align:center">25%+</td><td style="padding:2px 8px;text-align:center;color:#dc2626">1.11 (med: 0.90)</td><td style="padding:2px 8px;text-align:center;color:#dc2626">-2.21%</td></tr>
    <tr><td style="padding:2px 8px"><span class="exit-watch">EXIT WATCH</span></td><td style="padding:2px 8px;text-align:center">15-25%</td><td style="padding:2px 8px;text-align:center;color:#f97316">1.59 (med: 1.07)</td><td style="padding:2px 8px;text-align:center;color:#f97316">+0.49%</td></tr>
    <tr><td style="padding:2px 8px"><span class="exit-elevated">ELEVATED</span></td><td style="padding:2px 8px;text-align:center">10-15%</td><td style="padding:2px 8px;text-align:center">1.51</td><td style="padding:2px 8px;text-align:center">+0.60%</td></tr>
    </table>
    <p style="margin-top:10px"><b>Per-Asset-Class Scoring</b> (scimode task #145, 1,376 tickers):</p>
    <p>Extension scores are adjusted by asset class from taxonomy pole assignments. Key differences from universal buckets:</p>
    <table style="font-size:0.9em;border-collapse:collapse;margin:6px 0">
    <tr><th style="padding:2px 8px;text-align:left">Class</th><th style="padding:2px 8px">Key Difference</th><th style="padding:2px 8px">Score Effect</th></tr>
    <tr><td style="padding:2px 8px;color:#a855f7"><b>Miner</b></td><td style="padding:2px 8px">INVERTED: overextension is bullish (64% WR at 15-25%)</td><td style="padding:2px 8px">+30 to +40 at 15%+</td></tr>
    <tr><td style="padding:2px 8px;color:#6366f1"><b>Index</b></td><td style="padding:2px 8px">Danger 3x worse (27% WR, -15% median at 25%+)</td><td style="padding:2px 8px">-20 to -30 at 10%+</td></tr>
    <tr><td style="padding:2px 8px;color:#3b82f6"><b>Bond</b></td><td style="padding:2px 8px">Danger starts at 10% not 25% (40% WR at 10-15%)</td><td style="padding:2px 8px">-25 to -30 at 10%+</td></tr>
    <tr><td style="padding:2px 8px;color:#f59e0b"><b>Commodity</b></td><td style="padding:2px 8px">Poor win rate at all extensions (46-48% at 10%+)</td><td style="padding:2px 8px">-10 to -20 at 10%+</td></tr>
    <tr><td style="padding:2px 8px;color:#14b8a6"><b>Sector</b></td><td style="padding:2px 8px">Slightly better than universal (56-62% at 10%+)</td><td style="padding:2px 8px">+5 to +15 at 10%+</td></tr>
    <tr><td style="padding:2px 8px;color:#6b7280"><b>Equity</b></td><td style="padding:2px 8px">Universal buckets calibrated to equity (no change)</td><td style="padding:2px 8px">None</td></tr>
    </table>
    <p style="margin-top:10px"><b>Path Quality</b> (scimode_path_quality_v1_0, 52K observations, OOS validated):</p>
    <p>Predicts the <i>journey</i>, not just the destination. Based on extension + VIX + sector combination.</p>
    <table style="font-size:0.9em;border-collapse:collapse;margin:6px 0">
    <tr><th style="padding:2px 8px;text-align:left">Grade</th><th style="padding:2px 8px">Meaning</th><th style="padding:2px 8px">Typical MAE</th><th style="padding:2px 8px">P(good path)</th></tr>
    <tr><td style="padding:2px 8px"><span class="path-badge path-SMOOTH">SMOOTH</span></td><td style="padding:2px 8px">Low drawdown, quick green</td><td style="padding:2px 8px;text-align:center">&gt; -2%</td><td style="padding:2px 8px;text-align:center">&gt; 42%</td></tr>
    <tr><td style="padding:2px 8px"><span class="path-badge path-OK">OK</span></td><td style="padding:2px 8px">Moderate drawdown, eventual green</td><td style="padding:2px 8px;text-align:center">-2 to -4%</td><td style="padding:2px 8px;text-align:center">30-42%</td></tr>
    <tr><td style="padding:2px 8px"><span class="path-badge path-ROUGH">ROUGH</span></td><td style="padding:2px 8px">Significant pain before profit</td><td style="padding:2px 8px;text-align:center">-4 to -8%</td><td style="padding:2px 8px;text-align:center">20-30%</td></tr>
    <tr><td style="padding:2px 8px"><span class="path-badge path-CHOPPY">CHOPPY</span></td><td style="padding:2px 8px">Deep drawdown, may never recover</td><td style="padding:2px 8px;text-align:center">&lt; -8%</td><td style="padding:2px 8px;text-align:center">&lt; 20%</td></tr>
    </table>
    <p style="font-size:0.85em;color:#888">Hover over the Path badge for exact MAE, time-to-green, and probabilities. Good path = MAE &gt; -2% AND green within 5 days. Bad path = MAE &lt; -5% OR never durably green.</p>
    <p><b>Key insight</b>: Use SMA29 for entry scoring (stable optimal zone), SMA10 for exit alerts (sharper overextension detection). PF confirms median return signals -- EXIT ALERT at 25%+ SMA10 has PF_median &lt; 1.0 (losing bucket for typical trader).</p>
    </div>
    </details>
    """

    # Table - all headers have title tooltips
    header = """<table id="mainTable" class="dash-table" data-sort-col="0" data-sort-dir="desc">
    <thead><tr>
        <th>Own</th>
        <th>Watch</th>
        <th onclick="sortTable(2,false)" style="cursor:pointer" title="Stock ticker symbol + momentum rank (#1 = highest scored)">Ticker</th>
        <th onclick="sortTable(3,true)" style="cursor:pointer" title="Latest closing price (adjusted for splits)">Price</th>
        <th onclick="sortTable(4,true)" style="cursor:pointer" title="Weighted composite: 35% Momentum + 30% Potential + 35% SMA29 Extension. Higher = better entry opportunity. 0-100 scale. Scimode validated 2026-03-13.">Combined</th>
        <th onclick="sortTable(5,true)" style="cursor:pointer" title="SMA10 exit alert based on scimode PF validation (468 tickers). EXIT ALERT = 25%+ above SMA10 (PF=1.11, PF_median=0.90). EXIT WATCH = 15-25% (PF=1.59, PF_median=1.07).">Exit</th>
        <th onclick="sortTable(6,false)" style="cursor:pointer" title="Predicted path quality (the journey after entry). SMOOTH: Max Adverse Excursion better than -2%, green within 5 days. OK: moderate drawdown, eventually profitable. ROUGH: significant drawdown (-4 to -8%) before profit. CHOPPY: deep drawdown (worse than -8%), may never recover. Based on extension + VIX + sector from 52K historical observations, OOS validated.">Path</th>
        <th onclick="sortTable(7,true)" style="cursor:pointer" title="% distance of close above 29-day SMA. Positive = above SMA29, negative = below.">Ext %</th>
        <th onclick="sortTable(8,false)" style="cursor:pointer" title="SMA29 extension zone from scimode PF validation (468 tickers, trending filter). OPTIMAL = 0-10% above SMA29 (PF ~1.5). Zones scored by profit factor and median forward return.">Zone</th>
        <th onclick="sortTable(9,true)" style="cursor:pointer" title="Momentum ranker composite score (0-100). Blend of: multi-period returns (1d-1y), ratio quality (acceleration checks), and bad-SPY-day resilience. Gated by SMA structure.">Momentum</th>
        <th onclick="sortTable(10,true)" style="cursor:pointer" title="Pullback potential (0-100). Inverted health: deeper pullback = higher potential. VIX-boosted at elevated+ VIX. Scimode validated: low-health Q1 returns +2.17% 21d vs high-health Q5 +1.22%.">Potential</th>
        <th onclick="sortTable(11,true)" style="cursor:pointer" title="Extension bucket score (0-95). From SMA29 zone: OPTIMAL=90-95, GOOD=75, FAIR=55, CAUTION=40, WARNING=20-25, DANGER=10, EXTREME=0.">Ext Score</th>
        <th onclick="sortTable(12,true)" style="cursor:pointer" title="Profit Factor for this SMA29 extension zone = sum(wins)/sum(|losses|). From scimode PF validation (468 tickers, trending filter). PF>1.0 = profitable, PF<1.0 = losing. High-extension PF inflated by lottery tails.">PF</th>
        <th onclick="sortTable(13,true)" style="cursor:pointer" title="Median 21-day forward return for this SMA29 extension zone. From scimode OOS test. Negative in DANGER/EXTREME zones.">Med Fwd</th>
        <th onclick="sortTable(14,true)" style="cursor:pointer" title="R-squared of 21-day log-price regression. Measures trend quality: 1.0 = perfectly linear move, 0.0 = random walk. Above 0.70 = strong trend.">R-sq</th>
        <th onclick="sortTable(15,true)" style="cursor:pointer" title="Annualized slope from 21-day log-price regression. Shows how fast the stock is trending (% per year). Higher = steeper uptrend.">Slope %</th>
        <th onclick="sortTable(16,true)" style="cursor:pointer" title="Current drawdown from 63-day (3-month) rolling high. Shows how far the stock has pulled back from its recent peak. 0% = at high, -10% = pulled back 10%.">DD %</th>
        <th onclick="sortTable(17,false)" style="cursor:pointer" title="Trend stage from slope analysis. Decline = falling, Basing = bottoming, Uptrend = sustained rise, Parabolic = late-stage acceleration.">Stage</th>
        <th onclick="sortTable(18,true)" style="cursor:pointer" title="Total return over the past 5 trading days (1 week).">1W</th>
        <th onclick="sortTable(19,true)" style="cursor:pointer" title="Total return over the past 21 trading days (1 month).">1M</th>
        <th title="Highest-correlated asset from 20-asset correlation universe (63-day window). Shows what this stock moves most like. Only shown if correlation >= 0.30.">Pole</th>
        <th onclick="sortTable(21,false)" style="cursor:pointer" title="Asset class from pole assignment (taxonomy regression). Extension scores are adjusted per class: Miners get bonus (overextension is bullish), Indexes/Bonds get penalty (danger starts earlier).">Class</th>
    </tr></thead>
    <tbody>"""

    rows_html = []
    for r in results:
        zone_attr = r['extension_label']
        ext_pct_str = '{:+.1f}%'.format(r['extension_pct'])
        ext_color = '#22c55e' if r['extension_pct'] >= 0 else '#ef4444'

        pf_str = '{:.2f}'.format(r['extension_pf']) if r['extension_pf'] is not None else '-'
        pf_color = '#22c55e' if (r['extension_pf'] or 0) >= 1.3 else '#f59e0b' if (r['extension_pf'] or 0) >= 1.0 else '#ef4444'
        fwd_str = '{:+.2f}%'.format(r['extension_fwd']) if r['extension_fwd'] is not None else '-'
        fwd_color = '#22c55e' if (r['extension_fwd'] or 0) > 0 else '#ef4444' if (r['extension_fwd'] or 0) < 0 else '#888'

        dd_str = '{:.1f}%'.format(r['dd_pct']) if r['dd_pct'] is not None else '-'

        stage_names = {0: 'Decline', 1: 'Basing', 2: 'Uptrend', 3: 'Parabolic'}
        stage_str = stage_names.get(r['stage'], r.get('stage_name', '-'))

        mr_rank_str = '#{}'.format(r['momentum_rank']) if r['momentum_rank'] and r['momentum_rank'] < 9999 else '-'

        # Gold-filtered or noise-pole indicator
        ticker_suffix = ''
        if r.get('gold_filtered'):
            ticker_suffix = ' <span title="Gold/PM filtered from OPTIMAL (VIX&lt;20)" style="font-size:0.65em;color:#f59e0b">Au</span>'
        elif r.get('is_noise_pole'):
            ticker_suffix = ' <span title="Noise pole (low coherence/stability)" style="font-size:0.65em;color:#d1d5db">?</span>'

        # Path quality for data attributes
        pq = r.get('path_quality')

        # Data attributes for filtering
        exit_attr = r.get('exit_alert', '') or 'none'
        path_attr = pq['grade'] if pq else 'none'
        stage_attr = stage_str
        pole_attr = r.get('top_pole', '') or 'none'
        class_attr = r.get('asset_class_label', 'Equity')

        row = '<tr data-zone="{zone}" data-exit="{exit}" data-path="{path}" data-stage="{stage}" data-pole="{pole}" data-class="{cls}">'.format(
            zone=zone_attr, exit=exit_attr, path=path_attr,
            stage=stage_attr, pole=pole_attr, cls=class_attr)
        row += _own_cell(r['ticker'])
        row += _watch_cell(r['ticker'])
        row += '<td class="tl"><b>{}</b> <span style="font-size:0.72em;color:#999">{}</span>{}</td>'.format(r['ticker'], mr_rank_str, ticker_suffix)
        row += '<td class="tr" data-val="{}">${:,.2f}</td>'.format(r['price'], r['price'])
        row += '<td class="tr" data-val="{v}" style="font-weight:700;color:{c}">{v:.1f}</td>'.format(
            v=r['combined_score'],
            c='#166534' if r['combined_score'] >= 70 else '#854d0e' if r['combined_score'] >= 45 else '#991b1b')
        row += _exit_badge(r.get('exit_alert', ''))
        # Path quality badge
        if pq:
            pq_grade = pq['grade']
            pq_val_map = {'SMOOTH': 4, 'OK': 3, 'ROUGH': 2, 'CHOPPY': 1}
            pq_val = pq_val_map.get(pq_grade, 0)
            pq_tooltip = 'Max Adverse Excursion: {}% | Time to green: {}d | P(good path): {}% | P(bad path): {}%'.format(
                pq['median_mae'], pq['median_ttg'], pq['prob_good'], pq['prob_bad'])
            row += '<td class="tc" data-val="{}"><span class="path-badge path-{}" title="{}">{}</span></td>'.format(
                pq_val, pq_grade, pq_tooltip, pq_grade)
        else:
            row += '<td class="tc" data-val="0">-</td>'
        row += '<td class="tr" data-val="{}" style="color:{}">{}</td>'.format(r['extension_pct'], ext_color, ext_pct_str)
        row += '<td class="tc">{}</td>'.format(_ext_badge(r['extension_label']))
        row += '<td class="tr" data-val="{}">{:.0f}</td>'.format(r['momentum_score'], r['momentum_score'])
        row += '<td class="tr" data-val="{}">{:.0f}</td>'.format(r['potential_score'], r['potential_score'])
        row += '<td class="tr" data-val="{}">{}</td>'.format(r['extension_score'], r['extension_score'])
        row += '<td class="tr" data-val="{}" style="color:{}">{}</td>'.format(r['extension_pf'] or 0, pf_color, pf_str)
        row += '<td class="tr" data-val="{}" style="color:{}">{}</td>'.format(r['extension_fwd'] or 0, fwd_color, fwd_str)
        row += '<td class="tr" data-val="{}">{:.2f}</td>'.format(r['slope_rsq'], r['slope_rsq'])
        row += '<td class="tr" data-val="{}">{:.0f}%</td>'.format(r['slope_ann'], r['slope_ann'])
        row += '<td class="tr" data-val="{}">{}</td>'.format(r['dd_pct'] or 0, dd_str)
        row += '<td class="tc">{}</td>'.format(stage_str)
        row += _color_pct(r['ret_1w'])
        row += _color_pct(r['ret_1m'])
        row += '<td class="tc" style="font-size:0.78em">{}</td>'.format(r['top_pole'] or '-')
        ac_colors = {'Miner': '#a855f7', 'Bond': '#3b82f6', 'Index': '#6366f1',
                     'Commodity': '#f59e0b', 'Sector': '#14b8a6', 'Equity': '#6b7280'}
        ac_label = r.get('asset_class_label', 'Equity')
        ac_color = ac_colors.get(ac_label, '#6b7280')
        row += '<td class="tc" style="font-size:0.72em;color:{}">{}</td>'.format(ac_color, ac_label)
        row += '</tr>'
        rows_html.append(row)

    table_html = header + '\n'.join(rows_html) + '</tbody></table>'

    # Assemble using DashboardWriter API
    dw = DashboardWriter('sma29-entry', 'Enter & Exit Quality Scanner')

    # Stat bar
    stat_bar_data = [
        ('Universe', '{:,}'.format(len(results)), 'neutral'),
        ('Above SMA29', '{:,}'.format(above_sma), 'pos' if above_sma > len(results) * 0.4 else 'neg'),
        ('Optimal Zone', '{:,}'.format(optimal), 'pos' if optimal > 50 else 'warn'),
        ('Exit Alerts', '{:,}'.format(exit_alerts), 'neg' if exit_alerts > 10 else 'neutral'),
        ('Top 50 Avg', '{:.1f}'.format(top50_avg), 'pos' if top50_avg >= 65 else 'warn'),
    ]

    # Banner
    if optimal > 100:
        banner_color = '#22c55e'
    elif optimal > 50:
        banner_color = '#f59e0b'
    else:
        banner_color = '#ef4444'
    banner_score = 'Weights: {}% Momentum + {}% Potential + {}% Extension'.format(
        int(W_MOMENTUM * 100), int(W_POTENTIAL * 100), int(W_EXTENSION * 100))

    # VIX regime banner
    vix_banner_html = ''
    if current_vix is not None:
        vix_color_map = {'<15': '#22c55e', '15-20': '#64748b', '20-25': '#f59e0b',
                         '25-30': '#ef4444', '30+': '#991b1b'}
        vix_color = vix_color_map.get(vix_band, '#64748b')
        recs = VIX_SECTOR_RECS.get(vix_band, [])
        recs_str = ', '.join(recs) if recs else 'N/A'
        gold_note = ''
        if current_vix < 20:
            gold_note = ' | <span style="color:#f87171">Gold/PM filtered from OPTIMAL</span>'
        vix_banner_html = """
        <div style="background:linear-gradient(135deg, {color}22, {color}11);
                    border-left:4px solid {color}; padding:10px 16px; margin-bottom:14px;
                    border-radius:4px; font-size:0.88em">
            <span style="font-weight:700; color:{color}">VIX {vix:.1f} ({desc})</span>
            &nbsp;|&nbsp; Best sectors: <b>{recs}</b>{gold}
            <span style="float:right;font-size:0.82em;color:#888">scimode_vix_sector_v1_0</span>
        </div>""".format(color=vix_color, vix=current_vix, desc=vix_desc,
                         recs=recs_str, gold=gold_note)

    parts = []
    parts.append(dw.build_header(subtitle='SMA29 Entry + SMA10 Exit | Scimode-Validated'))
    parts.append(dw.stat_bar(stat_bar_data))
    parts.append(dw.regime_banner(banner_text, banner_score, color=banner_color))
    if vix_banner_html:
        parts.append(vix_banner_html)
    parts.append(dw.section('Methodology', methodology, hint='Scimode exit_signal_test.py'))
    parts.append(filter_bar)
    parts.append(dw.section('Entry Quality Rankings', table_html, hint='Click headers to sort'))
    parts.append(dw.footer())

    body = '\n'.join(parts)
    dw.write(body, extra_css=EXTRA_CSS, extra_js=EXTRA_JS)

    # CSV snapshot
    csv_path = os.path.join(_SCRIPT_DIR, 'sma29_entry_data_{}.csv'.format(now.strftime('%Y%m%d_%H%M')))
    pd.DataFrame(results).to_csv(csv_path, index=False, encoding='utf-8')
    print("CSV: {}".format(csv_path))

    return dw.index_path


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("ENTER & EXIT QUALITY SCANNER v1.5")
    print("=" * 70)

    results, vix_context = build_universe()

    # Save JSON cache
    output = clean_nan({
        'generated_at': datetime.now().isoformat(),
        'total': len(results),
        'weights': {'momentum': W_MOMENTUM, 'potential': W_POTENTIAL, 'extension': W_EXTENSION},
        'vix': vix_context,
        'results': results,
    })
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)
    print("[CACHE] Saved to {}".format(OUTPUT_JSON))

    # Summary
    above = sum(1 for r in results if r['extension_pct'] >= 0)
    optimal = sum(1 for r in results if r['extension_label'] == 'OPTIMAL')
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("  Universe:    {:,}".format(len(results)))
    print("  Above SMA29: {:,}".format(above))
    print("  Optimal zone: {:,}".format(optimal))
    if vix_context.get('current_vix') is not None:
        print("  VIX regime:  {:.1f} ({})".format(
            vix_context['current_vix'], vix_context['vix_desc']))
    if results:
        print("  Top 5:")
        for r in results[:5]:
            gf = ' [Au filtered]' if r.get('gold_filtered') else ''
            print("    {} {:>7.1f}  ext={:+.1f}% [{}]  mom={:.0f}  pot={:.0f}{}".format(
                r['ticker'].ljust(6), r['combined_score'], r['extension_pct'],
                r['extension_label'], r['momentum_score'], r['potential_score'], gf))

    out_path = build_html(results, vix_context)
    print()
    print("[OK] Dashboard: {}".format(out_path))


if __name__ == '__main__':
    main()
