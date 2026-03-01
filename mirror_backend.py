# -*- coding: utf-8 -*-
# =============================================================================
# mirror_backend.py
# Historical Mirror / Similar Days Dashboard Backend
# =============================================================================
# Computes a 25-year x 25-period bi-weekly similarity grid comparing
# today's market fingerprint against every historical 2-week period.
#
# Three signals (all fully vectorized, no day-by-day loops):
#   1. MOMENTUM  - masked Pearson correlation on 500-ETF z-score features
#   2. SPREADS   - fraction of 26 spread pairs in same posture+trend state
#   3. COMPOSITE - average of momentum and spread similarity scores
#
# Ported from: ../mirror_backend.py
# Changes:
#   - Lives in market-dashboards/ (uses __file__-relative paths)
#   - Outputs HTML dashboard via DashboardWriter instead of JSON
#   - No JSON file output
#
# Run: python mirror_backend.py
# Output: docs/similar-days/index.html (+ dated archive)
#
# Author: Brian + Claude
# Date: 2026-02-26
# Version: 2.0
# =============================================================================

import os
import sys
import pickle
import warnings
import importlib.util
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from dashboard_writer import DashboardWriter

warnings.filterwarnings('ignore')

# =============================================================================
# PATH SETUP - __file__-relative, market-dashboards/ lives one level below root
# =============================================================================

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR   = os.path.normpath(os.path.join(_SCRIPT_DIR, '..', 'perplexity-user-data'))

CACHE_DIR        = os.path.join(_DATA_DIR, 'price_cache')
FEATURE_CACHE    = os.path.join(_DATA_DIR, 'mirror_feature_cache.pkl')
FEATURE_MAX_AGE  = 23  # hours before rebuilding feature matrix

# =============================================================================
# CONFIG
# =============================================================================

SMA_PERIOD       = 50    # matches spread monitor
SLOPE_LOOKBACK   = 10    # matches spread monitor
SLOPE_THRESHOLD  = 0.001

PERIODS_PER_YEAR = 25    # 10-trading-day windows, 25 per year
WINDOW_DAYS      = 10    # trading days per window
START_YEAR       = 2000
MIN_OVERLAP      = 50    # minimum overlapping features for correlation

SLUG  = 'historical-mirror'
TITLE = 'Historical Mirror'

# =============================================================================
# SPREAD DEFINITIONS - 26 spreads across 6 macro dimensions
# =============================================================================

SPREADS = [
    # -- DURATION / RATES (what the bond market says) ----------------------
    {"name": "SPY/TLT",   "num": "SPY",  "den": "TLT"},   # equity vs duration
    {"name": "TIP/IEF",   "num": "TIP",  "den": "IEF"},   # breakeven inflation
    {"name": "GLD/TLT",   "num": "GLD",  "den": "TLT"},   # inflation vs deflation hedge
    {"name": "IEF/SHY",   "num": "IEF",  "den": "SHY"},   # yield curve belly vs front
    {"name": "TLT/HYG",   "num": "TLT",  "den": "HYG"},   # safety vs credit risk

    # -- CREDIT STRUCTURE (what the credit market says) --------------------
    {"name": "HYG/LQD",   "num": "HYG",  "den": "LQD"},   # HY vs IG
    {"name": "JNK/LQD",   "num": "JNK",  "den": "LQD"},   # HY alt vs IG
    {"name": "LQD/IEF",   "num": "LQD",  "den": "IEF"},   # pure IG spread vs duration
    {"name": "KRE/XLF",   "num": "KRE",  "den": "XLF"},   # regional vs large bank stress

    # -- GROWTH vs DEFENSIVE (cycle positioning) ---------------------------
    {"name": "XLY/XLP",   "num": "XLY",  "den": "XLP"},   # consumer cycle vs staples
    {"name": "XLI/XLP",   "num": "XLI",  "den": "XLP"},   # industrials vs staples
    {"name": "XLF/XLU",   "num": "XLF",  "den": "XLU"},   # financials vs utilities
    {"name": "XLE/XLK",   "num": "XLE",  "den": "XLK"},   # energy vs tech
    {"name": "IWM/SPY",   "num": "IWM",  "den": "SPY"},   # small vs large cap
    {"name": "SPHB/SPLV", "num": "SPHB", "den": "SPLV"},  # high beta vs low vol

    # -- FACTOR / STYLE (market structure) ---------------------------------
    {"name": "IWF/IWD",   "num": "IWF",  "den": "IWD"},   # growth vs value
    {"name": "QQQ/RSP",   "num": "QQQ",  "den": "RSP"},   # mega-cap vs equal-weight
    {"name": "GDX/GLD",   "num": "GDX",  "den": "GLD"},   # miners vs gold
    {"name": "SMH/SPY",   "num": "SMH",  "den": "SPY"},   # semis vs market

    # -- INFLATION / REAL ASSETS (commodity regime) ------------------------
    {"name": "CPER/GLD",  "num": "CPER", "den": "GLD"},   # copper vs gold
    {"name": "GLD/SLV",   "num": "GLD",  "den": "SLV"},   # gold vs silver
    {"name": "XLB/XLU",   "num": "XLB",  "den": "XLU"},   # materials vs utilities

    # -- US vs INTERNATIONAL (global capital flows) ------------------------
    {"name": "SPY/EFA",   "num": "SPY",  "den": "EFA"},   # US vs developed ex-US
    {"name": "SPY/EEM",   "num": "SPY",  "den": "EEM"},   # US vs emerging markets
    {"name": "EEM/UUP",   "num": "EEM",  "den": "UUP"},   # EM vs USD

    # -- SENTIMENT / FROTH ------------------------------------------------
    {"name": "XHB/SPY",   "num": "XHB",  "den": "SPY"},   # homebuilders vs market
    {"name": "VNQ/SPY",   "num": "VNQ",  "den": "SPY"},   # REITs vs equity
]

SPREAD_TICKERS = sorted(set(
    t for s in SPREADS for t in [s['num'], s['den']]
))

# Assets to track forward returns for
OUTCOME_ASSETS = {
    'SPY':  'S&P 500',
    'QQQ':  'Nasdaq',
    'IWM':  'Small Cap',
    'TLT':  '20Y Treasury',
    'GLD':  'Gold',
    'XLK':  'Technology',
    'XLF':  'Financials',
    'XLE':  'Energy',
    'XLI':  'Industrials',
    'XLB':  'Materials',
    'XLV':  'Healthcare',
    'XLY':  'Cons Disc',
    'XLP':  'Cons Staples',
    'XLU':  'Utilities',
    'XLRE': 'Real Estate',
    'VTV':  'Value',
}

# Ordered for display: broad first, then sectors
OUTCOME_DISPLAY_ORDER = [
    'SPY', 'QQQ', 'IWM', 'TLT', 'GLD',
    'XLK', 'XLF', 'XLE', 'XLI', 'XLB',
    'XLV', 'XLY', 'XLP', 'XLU', 'XLRE', 'VTV',
]

OUTCOME_HORIZONS = [10, 21, 63]  # trading days: ~2wk, 1mo, 3mo
HORIZON_LABELS   = ['10d (~2wk)', '21d (~1mo)', '63d (~3mo)']

# =============================================================================
# EXTRA CSS
# =============================================================================

EXTRA_CSS = """
/* --- Heatmap table --- */
.heatmap-wrap {
    overflow-x: auto;
    -webkit-overflow-scrolling: touch;
}

.heatmap-table {
    border-collapse: collapse;
    font-size: 0.72em;
    white-space: nowrap;
}

.heatmap-table th {
    background: #f8f9fb;
    color: #555;
    padding: 5px 7px;
    font-size: 0.75em;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    border-bottom: 2px solid #e2e4e8;
    white-space: nowrap;
    cursor: default;
    user-select: none;
}

.heatmap-table td {
    padding: 0;
    border: 1px solid rgba(255,255,255,0.25);
    width: 38px;
    min-width: 38px;
    height: 28px;
    text-align: center;
    vertical-align: middle;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82em;
    font-weight: 600;
    cursor: default;
}

.heatmap-table td.year-cell {
    background: #1e1e2e !important;
    color: #aaa;
    font-size: 0.78em;
    font-weight: 700;
    width: 46px;
    min-width: 46px;
    padding: 0 6px;
    border: none;
    border-bottom: 1px solid #2d2d4a;
    letter-spacing: 0.04em;
}

.heatmap-table td.null-cell {
    background: #f8f9fb !important;
    color: #ddd;
}

/* --- Heatmap layer toggle tabs --- */
.heatmap-tabs {
    display: flex;
    gap: 0;
    margin-bottom: 14px;
    border-bottom: 2px solid #e2e4e8;
}

.heatmap-tab {
    padding: 9px 20px;
    font-size: 0.85em;
    font-weight: 600;
    color: #888;
    cursor: pointer;
    border-bottom: 3px solid transparent;
    margin-bottom: -2px;
    transition: color 0.12s;
    user-select: none;
}

.heatmap-tab:hover { color: #1a1a2e; }
.heatmap-tab.active {
    color: #4f46e5;
    border-bottom-color: #4f46e5;
}

/* --- Forward outcome cards --- */
.outcome-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
    gap: 16px;
}

.outcome-card {
    background: #fff;
    border: 1px solid #e2e4e8;
    border-radius: 8px;
    overflow: hidden;
}

.outcome-card-header {
    background: #f8f9fb;
    border-bottom: 1px solid #e2e4e8;
    padding: 10px 16px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.outcome-card-ticker {
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 700;
    font-size: 0.95em;
    color: #1a1a2e;
}

.outcome-card-name {
    font-size: 0.8em;
    color: #888;
}

.outcome-card table {
    width: 100%;
    font-size: 0.85em;
    border-collapse: collapse;
}

.outcome-card thead th {
    background: transparent;
    padding: 8px 12px;
    font-size: 0.75em;
    color: #999;
    border-bottom: 1px solid #f0f0f0;
    text-align: right;
}

.outcome-card thead th:first-child { text-align: left; }

.outcome-card tbody td {
    padding: 7px 12px;
    border-bottom: 1px solid #f8f8f8;
    font-family: 'IBM Plex Mono', monospace;
    text-align: right;
    font-size: 0.9em;
    vertical-align: middle;
}

.outcome-card tbody td:first-child {
    font-family: 'IBM Plex Sans', sans-serif;
    font-weight: 600;
    font-size: 0.82em;
    color: #555;
    text-align: left;
}

.outcome-card tbody tr:last-child td { border-bottom: none; }

/* --- Analog table --- */
.analog-rank {
    display: inline-block;
    width: 24px;
    height: 24px;
    border-radius: 50%;
    background: #eff6ff;
    color: #1d4ed8;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75em;
    font-weight: 700;
    text-align: center;
    line-height: 24px;
}

.score-bar {
    display: inline-block;
    height: 8px;
    border-radius: 4px;
    background: #e2e4e8;
    width: 80px;
    vertical-align: middle;
    position: relative;
    overflow: hidden;
}

.score-bar-fill {
    position: absolute;
    left: 0; top: 0; bottom: 0;
    border-radius: 4px;
}
"""

# =============================================================================
# EXTRA JS
# =============================================================================

EXTRA_JS = """
// --- Heatmap layer switching ---
(function() {
    var tabs = document.querySelectorAll('.heatmap-tab');
    var tables = {
        'composite': document.getElementById('heatmap-composite'),
        'momentum':  document.getElementById('heatmap-momentum'),
        'spreads':   document.getElementById('heatmap-spreads')
    };

    tabs.forEach(function(tab) {
        tab.addEventListener('click', function() {
            var layer = this.getAttribute('data-layer');
            tabs.forEach(function(t) { t.classList.remove('active'); });
            this.classList.add('active');
            Object.keys(tables).forEach(function(k) {
                if (tables[k]) {
                    tables[k].style.display = (k === layer) ? '' : 'none';
                }
            });
        });
    });

    // Init: show composite only
    Object.keys(tables).forEach(function(k) {
        if (tables[k] && k !== 'composite') {
            tables[k].style.display = 'none';
        }
    });
})();

// --- Score bar rendering (runs after DOM ready) ---
document.querySelectorAll('.score-bar-fill').forEach(function(el) {
    var score = parseFloat(el.getAttribute('data-score'));
    if (!isNaN(score)) {
        el.style.width = (score * 100).toFixed(1) + '%';
        el.style.background = getGradientColor(score);
    }
});
"""

# =============================================================================
# DATA LOADING
# =============================================================================

def resolve_price_col(df):
    priority = ['adjClose', 'adj_close', 'Adj Close', 'close', 'Close']
    for col in priority:
        if col in df.columns:
            return col
    raise ValueError('No price column. Available: {0}'.format(list(df.columns)))


def load_one_pkl(filepath):
    try:
        ticker = os.path.splitext(os.path.basename(filepath))[0]
        with open(filepath, 'rb') as fh:
            df = pickle.load(fh)
        if isinstance(df, pd.Series):
            df = df.to_frame(name='close')
        if not isinstance(df, pd.DataFrame) or len(df) < 60:
            return None
        col = resolve_price_col(df)
        series_df = df[[col]].copy()
        series_df.columns = ['close']
        series_df.index = pd.to_datetime(series_df.index)
        series_df = series_df.sort_index()
        series_df = series_df[~series_df.index.duplicated(keep='last')]
        return ticker, series_df
    except Exception:
        return None


def load_price_cache(tickers=None, max_workers=6):
    """Load pkl files from CACHE_DIR. If tickers=None loads all."""
    if not os.path.isdir(CACHE_DIR):
        print('ERROR: price_cache directory not found at {0}'.format(CACHE_DIR))
        sys.exit(1)

    if tickers:
        files = [os.path.join(CACHE_DIR, '{0}.pkl'.format(t)) for t in tickers
                 if os.path.exists(os.path.join(CACHE_DIR, '{0}.pkl'.format(t)))]
    else:
        files = [os.path.join(CACHE_DIR, f)
                 for f in os.listdir(CACHE_DIR) if f.endswith('.pkl')]

    print('  Loading {0} pkl files...'.format(len(files)))
    results = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(load_one_pkl, fp): fp for fp in files}
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                ticker, df = result
                results[ticker] = df

    return results

# =============================================================================
# BI-WEEKLY DATE GRID
# =============================================================================

def build_date_grid(latest_date):
    """
    Build 25-period x N-year grid using real 10-trading-day windows.

    For each year, steps back from the last available trading day in
    10-trading-day increments to produce exactly 25 non-overlapping windows.
    Period 24 = most recent window (ends at year's last trading day).
    Period 0  = oldest window (ends ~250 trading days before year end).

    Returns: list of (year, period_idx, end_date, start_date) tuples
    """
    all_dates = pd.bdate_range(
        start='{0}-01-01'.format(START_YEAR),
        end=latest_date
    )

    grid = []
    for year in range(START_YEAR, latest_date.year + 1):
        year_dates = all_dates[all_dates.year == year]
        if len(year_dates) == 0:
            continue

        last_idx = all_dates.get_loc(year_dates[-1])

        for p in range(PERIODS_PER_YEAR - 1, -1, -1):
            end_idx   = last_idx - (PERIODS_PER_YEAR - 1 - p) * WINDOW_DAYS
            start_idx = end_idx - WINDOW_DAYS + 1

            if end_idx < 0 or start_idx < 0:
                continue

            end_date   = all_dates[end_idx]
            start_date = all_dates[start_idx]
            grid.append((year, p, end_date, start_date))

    return grid


def period_label(period_idx, end_date=None):
    """Returns display label for a period."""
    if end_date is not None:
        return end_date.strftime('%b %d')
    return str(period_idx)

# =============================================================================
# MOMENTUM FEATURES
# =============================================================================

def calculate_momentum_features(df):
    """
    7 z-scored features per ETF:
      3M, 1M, 10D, 5D momentum z-scores
      slope1 (1M-3M), slope2 (10D-1M), slope3 (5D-10D)
    All vectorized across full history.
    """
    features = pd.DataFrame(index=df.index)

    for period, name in [(63, '3M'), (21, '1M'), (10, '10D'), (5, '5D')]:
        ret = df['close'].pct_change(period) * 100
        rolling_mean = ret.rolling(252, min_periods=60).mean()
        rolling_std  = ret.rolling(252, min_periods=60).std()
        features[name] = ((ret - rolling_mean) / rolling_std.replace(0, np.nan)).clip(-2, 2)

    features['slope1'] = features['1M']  - features['3M']
    features['slope2'] = features['10D'] - features['1M']
    features['slope3'] = features['5D']  - features['10D']

    for col in ['slope1', 'slope2', 'slope3']:
        rolling_mean = features[col].rolling(252, min_periods=60).mean()
        rolling_std  = features[col].rolling(252, min_periods=60).std()
        features[col] = ((features[col] - rolling_mean) /
                         rolling_std.replace(0, np.nan)).clip(-2, 2)

    return features


def build_feature_matrix(all_data):
    """Build full pattern matrix: dates x (ETF x features). Vectorized."""
    print('  Building momentum feature matrix...')
    all_features = {}

    for ticker, df in all_data.items():
        if len(df) < 126:
            continue
        features = calculate_momentum_features(df)
        for col in features.columns:
            all_features['{0}_{1}'.format(ticker, col)] = features[col]

    features_df = pd.DataFrame(all_features)
    print('  Feature matrix: {0} days x {1} features'.format(
        len(features_df), len(features_df.columns)))
    return features_df


def load_or_build_feature_matrix(all_data):
    """Use cached feature matrix if it's newer than the price cache."""
    if os.path.exists(FEATURE_CACHE):
        cache_mtime = os.path.getmtime(FEATURE_CACHE)

        pkl_files = [os.path.join(CACHE_DIR, f)
                     for f in os.listdir(CACHE_DIR) if f.endswith('.pkl')]
        if pkl_files:
            newest_price = max(os.path.getmtime(f) for f in pkl_files)
            if cache_mtime >= newest_price:
                age_hours = (datetime.now().timestamp() - cache_mtime) / 3600
                print('  Loading cached feature matrix ({0:.1f}h old, price data unchanged)...'.format(
                    age_hours))
                with open(FEATURE_CACHE, 'rb') as fh:
                    return pickle.load(fh)
            else:
                print('  Price data is newer than feature cache - rebuilding...')
        else:
            age_hours = (datetime.now().timestamp() - cache_mtime) / 3600
            if age_hours < FEATURE_MAX_AGE:
                print('  Loading cached feature matrix ({0:.1f}h old)...'.format(age_hours))
                with open(FEATURE_CACHE, 'rb') as fh:
                    return pickle.load(fh)

    features_df = build_feature_matrix(all_data)

    with open(FEATURE_CACHE, 'wb') as fh:
        pickle.dump(features_df, fh)
    print('  Feature matrix cached to {0}'.format(FEATURE_CACHE))

    return features_df

# =============================================================================
# MOMENTUM SIMILARITY
# =============================================================================

def masked_correlation(target_row, hist_row, min_overlap=MIN_OVERLAP):
    """
    Pearson correlation on overlapping non-NaN features only.
    Returns float 0-1 (rescaled from -1..1) or None if insufficient overlap.
    """
    target_mask = ~np.isnan(target_row)
    hist_mask   = ~np.isnan(hist_row)
    overlap     = target_mask & hist_mask

    if overlap.sum() < min_overlap:
        return None

    t = target_row[overlap]
    h = hist_row[overlap]

    if t.std() == 0 or h.std() == 0:
        return None

    corr = np.corrcoef(t, h)[0, 1]
    if np.isnan(corr):
        return None

    return float((corr + 1) / 2)


def compute_momentum_grid(features_df, date_grid, today_row):
    """
    For each (year, period, end_date, start_date) in date_grid,
    look up end_date in features_df and compute masked correlation vs today.
    Returns dict: (year, period) -> similarity float 0-1
    """
    print('  Computing momentum similarity grid...')
    today_vals = today_row.values

    results = {}
    for year, period, end_date, start_date in date_grid:
        if end_date not in features_df.index:
            results[(year, period)] = None
            continue

        hist_vals = features_df.loc[end_date].values
        sim = masked_correlation(today_vals, hist_vals)
        results[(year, period)] = sim

    return results

# =============================================================================
# SPREAD SIMILARITY
# =============================================================================

def build_spread_state_series(price_data):
    """
    For each spread, compute daily state across full history:
      state = above_sma (bool) x trending_up (bool)
      encoded as int: 3=above+rising, 2=above+falling,
                      1=below+rising, 0=below+falling

    Returns DataFrame: dates x spread_names (int 0-3)
    """
    print('  Building spread state time series...')
    state_series = {}

    for s in SPREADS:
        num, den, name = s['num'], s['den'], s['name']

        if num not in price_data or den not in price_data:
            print('    Skipping {0} (missing price data)'.format(name))
            continue

        num_series = price_data[num]['close']
        den_series = price_data[den]['close']

        combined = pd.DataFrame({'num': num_series, 'den': den_series}).dropna()
        if len(combined) < SMA_PERIOD + SLOPE_LOOKBACK + 5:
            print('    Skipping {0} (insufficient history)'.format(name))
            continue

        ratio = combined['num'] / combined['den']
        sma   = ratio.rolling(SMA_PERIOD).mean()
        slope = sma.pct_change(SLOPE_LOOKBACK)

        above_sma   = (ratio > sma).astype(float)
        trending_up = (slope > SLOPE_THRESHOLD).astype(float)

        state = (above_sma * 2 + trending_up)
        state[sma.isna() | slope.isna()] = np.nan
        state_series[name] = state

    if not state_series:
        print('  WARNING: No spread states computed')
        return pd.DataFrame()

    spread_states = pd.DataFrame(state_series)
    print('  Spread states: {0} days x {1} spreads'.format(
        len(spread_states), len(spread_states.columns)))
    return spread_states


def compute_spread_grid(spread_states, date_grid, today_states):
    """
    For each (year, period, date), compute fraction of spreads
    in the same state as today.
    Returns dict: (year, period) -> similarity float 0-1
    """
    print('  Computing spread similarity grid...')

    if spread_states.empty:
        return {(y, p): None for y, p, e, s in date_grid}

    results = {}
    for year, period, end_date, start_date in date_grid:
        if end_date not in spread_states.index:
            results[(year, period)] = None
            continue

        hist_row = spread_states.loc[end_date]

        common = today_states.index.intersection(hist_row.index)
        if len(common) == 0:
            results[(year, period)] = None
            continue

        t = today_states[common].dropna()
        h = hist_row[common].dropna()
        common2 = t.index.intersection(h.index)

        if len(common2) < 3:
            results[(year, period)] = None
            continue

        matches = (t[common2] == h[common2]).sum()
        sim = float(matches / len(common2))
        results[(year, period)] = sim

    return results

# =============================================================================
# GRID ASSEMBLY
# =============================================================================

def build_output_grid(date_grid, momentum_grid, spread_grid):
    """
    Assemble results into year x period arrays.
    Returns years, periods, period_labels, comp_grid, mom_grid, spr_grid
    """
    years   = sorted(set(y for y, p, e, s in date_grid))
    periods = list(range(PERIODS_PER_YEAR))

    latest_year = years[-1]
    end_dates = {}
    for year, period, end_date, start_date in date_grid:
        if year == latest_year:
            end_dates[period] = end_date

    period_labels = [period_label(p, end_dates.get(p)) for p in periods]

    comp_grid = []
    mom_grid  = []
    spr_grid  = []

    for year in years:
        comp_row = []
        mom_row  = []
        spr_row  = []

        for period in periods:
            key = (year, period)
            mom = momentum_grid.get(key)
            spr = spread_grid.get(key)

            available = [v for v in [mom, spr] if v is not None]
            comp = float(np.mean(available)) if available else None

            comp_row.append(round(comp, 4) if comp is not None else None)
            mom_row.append(round(mom, 4)   if mom  is not None else None)
            spr_row.append(round(spr, 4)   if spr  is not None else None)

        comp_grid.append(comp_row)
        mom_grid.append(mom_row)
        spr_grid.append(spr_row)

    return years, periods, period_labels, comp_grid, mom_grid, spr_grid


def find_top_analogs(date_grid, comp_grid, mom_grid, spr_grid, years, n=10):
    """Find the N most similar historical periods overall."""
    date_lookup = {(y, p): (e, s) for y, p, e, s in date_grid}

    scored = []
    for yi, year in enumerate(years):
        for period in range(PERIODS_PER_YEAR):
            comp = comp_grid[yi][period]
            if comp is None:
                continue

            end_date, start_date = date_lookup.get((year, period), (None, None))
            end_str   = end_date.strftime('%Y-%m-%d')   if end_date   else None
            start_str = start_date.strftime('%Y-%m-%d') if start_date else None
            label     = period_label(period, end_date)

            scored.append({
                'year':       year,
                'period':     period,
                'date':       end_str,
                'date_start': start_str,
                'label':      label,
                'composite':  comp,
                'momentum':   mom_grid[yi][period],
                'spreads':    spr_grid[yi][period],
            })

    scored.sort(key=lambda x: x['composite'], reverse=True)
    return scored[:n]

# =============================================================================
# FORWARD OUTCOMES
# =============================================================================

def get_forward_return(price_df, date, n_days):
    """Return pct return n_days after date, or None."""
    if date not in price_df.index:
        return None
    idx = price_df.index.get_loc(date)
    if idx + n_days >= len(price_df):
        return None
    p0 = price_df.iloc[idx]['close']
    p1 = price_df.iloc[idx + n_days]['close']
    if pd.isna(p0) or pd.isna(p1) or p0 == 0:
        return None
    return float((p1 / p0 - 1) * 100)


def summarize_returns(rets, weights=None):
    """Compute summary stats from a list of returns."""
    if not rets:
        return None
    arr = np.array(rets)
    if weights is not None:
        w = np.array(weights)
        w = w / w.sum()
        avg = float(np.average(arr, weights=w))
    else:
        avg = float(np.mean(arr))
    return {
        'avg':     round(avg, 2),
        'median':  round(float(np.median(arr)), 2),
        'p20':     round(float(np.percentile(arr, 20)), 2),
        'p80':     round(float(np.percentile(arr, 80)), 2),
        'pct_pos': round(float(100 * np.mean(arr > 0)), 1),
        'n':       len(arr),
    }


def compute_forward_outcomes(all_data, date_grid, comp_grid, mom_grid,
                              spr_grid, years, top_n=50):
    """
    Take top_n analog periods by composite score, look forward 10/21/63d,
    compute return distribution for each OUTCOME_ASSETS ticker.
    """
    print('  Computing forward outcomes for top {0} analogs...'.format(top_n))

    date_lookup = {(y, p): e for y, p, e, s in date_grid}

    all_scored = []
    for yi, year in enumerate(years):
        for period in range(PERIODS_PER_YEAR):
            comp = comp_grid[yi][period]
            if comp is None:
                continue
            date = date_lookup.get((year, period))
            if date is None:
                continue
            all_scored.append({
                'date':      date,
                'composite': comp,
                'momentum':  mom_grid[yi][period],
                'spreads':   spr_grid[yi][period],
            })

    all_scored.sort(key=lambda x: x['composite'], reverse=True)
    top_analogs = all_scored[:top_n]

    if not top_analogs:
        print('  WARNING: No analogs found for outcomes')
        return {}

    print('  Using {0} analogs (composite range: {1:.2f}-{2:.2f})'.format(
        len(top_analogs),
        top_analogs[-1]['composite'],
        top_analogs[0]['composite']))

    raw     = {ticker: {h: [] for h in OUTCOME_HORIZONS} for ticker in OUTCOME_ASSETS}
    weights = {ticker: {h: [] for h in OUTCOME_HORIZONS} for ticker in OUTCOME_ASSETS}

    for analog in top_analogs:
        date   = analog['date']
        weight = analog['composite']

        for ticker in OUTCOME_ASSETS:
            if ticker not in all_data:
                continue
            price_df = all_data[ticker]
            for h in OUTCOME_HORIZONS:
                ret = get_forward_return(price_df, date, h)
                if ret is not None:
                    raw[ticker][h].append(ret)
                    weights[ticker][h].append(weight)

    outcomes = {}
    for ticker, name in OUTCOME_ASSETS.items():
        outcomes[ticker] = {'name': name, 'horizons': {}}
        for h in OUTCOME_HORIZONS:
            rets = raw[ticker][h]
            wts  = weights[ticker][h]
            outcomes[ticker]['horizons'][str(h)] = summarize_returns(rets, wts)

    print('  Outcomes computed for {0} assets x {1} horizons'.format(
        len(outcomes), len(OUTCOME_HORIZONS)))
    return outcomes

# =============================================================================
# HTML RENDERING HELPERS
# =============================================================================

def fmt_score(v, decimals=3):
    """Format a 0-1 score as string, or dash if None."""
    if v is None:
        return '<span class="muted">-</span>'
    return '{0:.{1}f}'.format(v, decimals)


def fmt_pct(v, signed=True):
    """Format a percentage return, or dash if None."""
    if v is None:
        return '<span class="muted">-</span>'
    cls = 'pos' if v > 0 else ('neg' if v < 0 else 'neutral')
    sign = '+' if v > 0 else ''
    return '<span class="num {0}">{1}{2:.2f}%</span>'.format(cls, sign, v)


def pct_class(v):
    """CSS class for a percentage value."""
    if v is None:
        return 'neutral'
    return 'pos' if v > 0 else ('neg' if v < 0 else 'neutral')


def score_color_style(score):
    """
    Inline style for background color using the gradient.
    We compute the color server-side using a Python approximation of getGradientColor,
    then use it as inline style on the heatmap cells.
    """
    if score is None:
        return ''
    # Python implementation of the JS gradient function
    c = max(0.0, min(1.0, score))
    if c < 0.25:
        t = c * 4
        r = int(211 + (255 - 211) * t)
        g = int(47  + (152 - 47)  * t)
        b = int(47  + (0   - 47)  * t)
    elif c < 0.5:
        t = (c - 0.25) * 4
        r = 255
        g = int(152 + (235 - 152) * t)
        b = int(0   + (59  - 0)   * t)
    elif c < 0.75:
        t = (c - 0.5) * 4
        r = int(255 + (139 - 255) * t)
        g = int(235 + (195 - 235) * t)
        b = int(59  + (74  - 59)  * t)
    else:
        t = (c - 0.75) * 4
        r = int(139 + (0   - 139) * t)
        g = int(195 + (200 - 195) * t)
        b = int(74  + (83  - 74)  * t)
    # Darken text for bright mid-range cells
    text_color = '#fff' if (c < 0.35 or c > 0.80) else '#1a1a2e'
    return 'style="background:rgb({0},{1},{2});color:{3};"'.format(r, g, b, text_color)


def build_section1_analogs(top_analogs):
    """Section 1: Top 10 Most Similar Periods table."""
    rows = []
    for i, a in enumerate(top_analogs):
        comp = a['composite']
        mom  = a['momentum']
        spr  = a['spreads']

        comp_pct = int(comp * 100) if comp is not None else 0
        bar_html = (
            '<span class="score-bar">'
            '<span class="score-bar-fill" data-score="{0}"></span>'
            '</span>'
        ).format('{0:.4f}'.format(comp) if comp is not None else '0')

        row = (
            '<tr>'
            '<td><span class="analog-rank">{rank}</span></td>'
            '<td class="num">{date}</td>'
            '<td class="num">{year}</td>'
            '<td>{label}</td>'
            '<td class="num">{bar} {comp_val}</td>'
            '<td class="num">{mom_val}</td>'
            '<td class="num">{spr_val}</td>'
            '</tr>'
        ).format(
            rank=i + 1,
            date=a.get('date', '-'),
            year=a.get('year', '-'),
            label=a.get('label', '-'),
            bar=bar_html,
            comp_val=fmt_score(comp),
            mom_val=fmt_score(mom),
            spr_val=fmt_score(spr),
        )
        rows.append(row)

    table = (
        '<table>'
        '<thead>'
        '<tr>'
        '<th>#</th>'
        '<th>End Date</th>'
        '<th>Year</th>'
        '<th>Period Label</th>'
        '<th>Composite</th>'
        '<th>Momentum</th>'
        '<th>Spreads</th>'
        '</tr>'
        '</thead>'
        '<tbody>'
        '{rows}'
        '</tbody>'
        '</table>'
    ).format(rows='\n'.join(rows))

    return table


def build_section2_outcomes(outcomes):
    """Section 2: Forward Outcomes - cards for each asset."""
    if not outcomes:
        return '<p class="muted">No outcome data available.</p>'

    cards = []
    for ticker in OUTCOME_DISPLAY_ORDER:
        if ticker not in outcomes:
            continue
        asset = outcomes[ticker]
        name  = asset.get('name', ticker)

        rows = []
        for h, hlabel in zip(OUTCOME_HORIZONS, HORIZON_LABELS):
            stats = asset['horizons'].get(str(h))
            if stats is None:
                avg_html    = '<span class="muted">-</span>'
                median_html = '<span class="muted">-</span>'
                p20_html    = '<span class="muted">-</span>'
                p80_html    = '<span class="muted">-</span>'
                pos_html    = '<span class="muted">-</span>'
                n_html      = '<span class="muted">0</span>'
            else:
                avg_html    = fmt_pct(stats.get('avg'))
                median_html = fmt_pct(stats.get('median'))
                p20_html    = fmt_pct(stats.get('p20'))
                p80_html    = fmt_pct(stats.get('p80'))
                pp = stats.get('pct_pos')
                pos_cls = 'pos' if (pp is not None and pp >= 55) else \
                          ('neg' if (pp is not None and pp < 45) else 'neutral')
                pos_html = (
                    '<span class="num {0}">{1:.0f}%</span>'.format(pos_cls, pp)
                    if pp is not None else '<span class="muted">-</span>'
                )
                n_html = '<span class="num muted">{0}</span>'.format(
                    stats.get('n', 0))

            row = (
                '<tr>'
                '<td>{hlabel}</td>'
                '<td>{avg}</td>'
                '<td>{median}</td>'
                '<td>{p20}</td>'
                '<td>{p80}</td>'
                '<td>{pos}</td>'
                '<td>{n}</td>'
                '</tr>'
            ).format(
                hlabel=hlabel,
                avg=avg_html,
                median=median_html,
                p20=p20_html,
                p80=p80_html,
                pos=pos_html,
                n=n_html,
            )
            rows.append(row)

        card = (
            '<div class="outcome-card">'
            '<div class="outcome-card-header">'
            '<span class="outcome-card-ticker">{ticker}</span>'
            '<span class="outcome-card-name">{name}</span>'
            '</div>'
            '<table>'
            '<thead>'
            '<tr>'
            '<th>Horizon</th>'
            '<th>Avg</th>'
            '<th>Median</th>'
            '<th>P20</th>'
            '<th>P80</th>'
            '<th>%Pos</th>'
            '<th>N</th>'
            '</tr>'
            '</thead>'
            '<tbody>{rows}</tbody>'
            '</table>'
            '</div>'
        ).format(ticker=ticker, name=name, rows='\n'.join(rows))
        cards.append(card)

    return '<div class="outcome-grid">{0}</div>'.format('\n'.join(cards))


def build_heatmap_table(years, period_labels, grid, table_id):
    """
    Build one HTML table for the heatmap (one layer: composite/momentum/spreads).
    Rows = years, columns = periods 0-24.
    Each cell is colored by its score using the Python gradient helper.
    """
    # Header row: period labels (abbreviated to 3 chars to fit)
    header_cells = ['<th>Year</th>']
    for lbl in period_labels:
        header_cells.append('<th title="{0}">{1}</th>'.format(lbl, lbl[:6]))

    rows = ['<tr>{0}</tr>'.format(''.join(header_cells))]

    for yi, year in enumerate(years):
        cells = ['<td class="year-cell">{0}</td>'.format(year)]
        row_data = grid[yi] if yi < len(grid) else []
        for period in range(PERIODS_PER_YEAR):
            val = row_data[period] if period < len(row_data) else None
            if val is None:
                cells.append('<td class="null-cell" title="No data">-</td>')
            else:
                style = score_color_style(val)
                display = '{0:.2f}'.format(val)
                cells.append(
                    '<td {style} title="{year} {lbl}: {val:.3f}">{display}</td>'.format(
                        style=style,
                        year=year,
                        lbl=period_labels[period] if period < len(period_labels) else str(period),
                        val=val,
                        display=display,
                    )
                )
        rows.append('<tr>{0}</tr>'.format(''.join(cells)))

    return (
        '<table class="heatmap-table" id="{id}">'
        '<thead>{header}</thead>'
        '<tbody>{body}</tbody>'
        '</table>'
    ).format(
        id=table_id,
        header='<tr>{0}</tr>'.format(''.join(header_cells)),
        body='\n'.join(rows[1:]),  # rows[0] already has header
    )


def build_section3_heatmap(years, period_labels, comp_grid, mom_grid, spr_grid):
    """Section 3: Similarity Heatmap with layer tabs."""
    tabs_html = (
        '<div class="heatmap-tabs">'
        '<div class="heatmap-tab active" data-layer="composite">Composite</div>'
        '<div class="heatmap-tab" data-layer="momentum">Momentum</div>'
        '<div class="heatmap-tab" data-layer="spreads">Spreads</div>'
        '</div>'
    )

    comp_table = build_heatmap_table(years, period_labels, comp_grid, 'heatmap-composite')
    mom_table  = build_heatmap_table(years, period_labels, mom_grid,  'heatmap-momentum')
    spr_table  = build_heatmap_table(years, period_labels, spr_grid,  'heatmap-spreads')

    legend_html = (
        '<div style="display:flex;align-items:center;gap:12px;margin-top:14px;'
        'font-size:0.82em;color:#888;">'
        '<span>Low similarity</span>'
        '<div style="background:linear-gradient(to right,rgb(211,47,47),rgb(255,235,59),'
        'rgb(139,195,74));width:140px;height:10px;border-radius:5px;"></div>'
        '<span>High similarity</span>'
        '</div>'
    )

    return (
        tabs_html +
        '<div class="heatmap-wrap">' +
        comp_table +
        mom_table +
        spr_table +
        '</div>' +
        legend_html
    )

# =============================================================================
# MAIN
# =============================================================================

def main():
    print('=' * 70)
    print('HISTORICAL MIRROR BACKEND v2.0  (HTML output via DashboardWriter)')
    print('=' * 70)
    print()

    # -- Load ETF momentum data --------------------------------------------
    print('[1/6] Loading ETF price cache...')
    all_data = load_price_cache()
    print('  Loaded: {0} ETFs'.format(len(all_data)))

    if not all_data:
        print('ERROR: No data loaded. Check price_cache directory.')
        sys.exit(1)

    # -- Load spread price data --------------------------------------------
    print('\n[2/6] Loading spread ticker data...')
    spread_price_data = {t: df for t, df in all_data.items()
                         if t in SPREAD_TICKERS}
    print('  Found {0} / {1} spread tickers'.format(
        len(spread_price_data), len(SPREAD_TICKERS)))

    # -- Build / load feature matrix ---------------------------------------
    print('\n[3/6] Building momentum feature matrix...')
    features_df = load_or_build_feature_matrix(all_data)

    latest_date = features_df.index.max()
    print('  Latest date: {0}'.format(latest_date.strftime('%Y-%m-%d')))

    today_row = features_df.loc[latest_date]

    # -- Build spread state time series ------------------------------------
    print('\n[4/6] Computing spread states...')
    spread_states = build_spread_state_series(spread_price_data)

    if not spread_states.empty:
        today_states = (spread_states.loc[latest_date]
                        if latest_date in spread_states.index
                        else spread_states.iloc[-1])
    else:
        today_states = pd.Series(dtype=float)

    # -- Compute similarity grids ------------------------------------------
    print('\n[5/6] Computing similarity grids...')
    date_grid = build_date_grid(latest_date)
    print('  Grid: {0} cells ({1}-{2}, {3} periods/yr)'.format(
        len(date_grid), START_YEAR, latest_date.year, PERIODS_PER_YEAR))

    momentum_grid_raw = compute_momentum_grid(features_df, date_grid, today_row)
    spread_grid_raw   = compute_spread_grid(spread_states, date_grid, today_states)

    years, periods, period_labels, comp_grid, mom_grid, spr_grid = \
        build_output_grid(date_grid, momentum_grid_raw, spread_grid_raw)

    top_analogs = find_top_analogs(
        date_grid, comp_grid, mom_grid, spr_grid, years, n=10
    )

    # -- Compute forward outcomes ------------------------------------------
    print('\n[6/6] Computing forward outcomes...')
    outcomes = compute_forward_outcomes(
        all_data, date_grid, comp_grid, mom_grid, spr_grid, years, top_n=50
    )

    # -- Assemble metadata -------------------------------------------------
    etf_count    = len(all_data)
    spread_count = len(spread_states.columns) if not spread_states.empty else 0
    latest_str   = latest_date.strftime('%Y-%m-%d')
    top_score    = top_analogs[0]['composite'] if top_analogs else 0.0
    year_range   = '{0}-{1}'.format(START_YEAR, latest_date.year)

    # -- Build HTML sections -----------------------------------------------
    writer = DashboardWriter(SLUG, TITLE)

    # Stat bar
    stat_score_cls = ('pos' if top_score >= 0.75 else
                      ('warn' if top_score >= 0.60 else 'neutral'))

    stat_bar = writer.stat_bar([
        ('ETFs Analyzed',   str(etf_count),                 'neutral'),
        ('Spreads Tracked', str(spread_count),               'neutral'),
        ('Latest Date',     latest_str,                      'neutral'),
        ('Top Analog Score', '{0:.3f}'.format(top_score),   stat_score_cls),
    ])

    # Header
    header = writer.build_header(
        'Pattern match {0} | {1} periods | {2} analogs'.format(
            year_range, len(date_grid), len(top_analogs))
    )

    # Regime banner: describe the best analog
    if top_analogs:
        best = top_analogs[0]
        banner_label = 'TOP ANALOG: {0} ({1})'.format(
            best.get('date', '?'), best.get('year', '?'))
        score_detail = (
            'Composite: {0:.3f} &nbsp;|&nbsp; '
            'Momentum: {1} &nbsp;|&nbsp; '
            'Spreads: {2} &nbsp;|&nbsp; '
            'Period: {3}'
        ).format(
            best['composite'],
            '{0:.3f}'.format(best['momentum']) if best['momentum'] is not None else '-',
            '{0:.3f}'.format(best['spreads'])  if best['spreads']  is not None else '-',
            best.get('label', '-'),
        )
        banner_color = ('#22c55e' if top_score >= 0.75 else
                        ('#f59e0b' if top_score >= 0.60 else '#888'))
        regime_banner = writer.regime_banner(banner_label, score_detail, color=banner_color)
    else:
        regime_banner = ''

    # Section 1: Top analogs
    sec1_content = build_section1_analogs(top_analogs)
    sec1 = writer.section(
        'Top 10 Most Similar Periods',
        sec1_content,
        hint='Based on composite momentum + spread state match'
    )

    # Section 2: Forward outcomes
    sec2_content = build_section2_outcomes(outcomes)
    sec2 = writer.section(
        'Forward Outcomes (Top 50 Analogs)',
        sec2_content,
        hint='Weighted by composite similarity score'
    )

    # Section 3: Heatmap
    sec3_content = build_section3_heatmap(
        years, period_labels, comp_grid, mom_grid, spr_grid)
    sec3 = writer.section(
        'Similarity Heatmap — {0} x {1} Periods'.format(year_range, PERIODS_PER_YEAR),
        sec3_content,
        hint='Each cell = composite similarity to today | Use tabs to switch layers'
    )

    footer = writer.footer()

    parts = [
        stat_bar,
        header,
        regime_banner,
        sec1,
        sec2,
        sec3,
        footer,
    ]

    writer.write('\n'.join(parts), extra_css=EXTRA_CSS, extra_js=EXTRA_JS)

    # Write CSV - top analogs
    csv_path = os.path.join(_SCRIPT_DIR, 'mirror_data.csv')
    csv_df = pd.DataFrame(top_analogs)
    csv_df.to_csv(csv_path, index=False, encoding='utf-8')
    print('CSV: {0}'.format(csv_path))

    # -- Summary -----------------------------------------------------------
    print()
    print('=' * 70)
    print('COMPLETE')
    print('=' * 70)
    print('  ETFs:          {0}'.format(etf_count))
    print('  Spreads:       {0}'.format(spread_count))
    print('  Grid cells:    {0}'.format(len(date_grid)))
    print('  Years covered: {0}'.format(year_range))
    print()
    print('  TOP 5 ANALOGS:')
    for a in top_analogs[:5]:
        print('    {date}  composite={comp:.3f}  mom={mom}  spreads={spr}'.format(
            date=a['date'],
            comp=a['composite'],
            mom='{0:.3f}'.format(a['momentum']) if a['momentum'] is not None else '-',
            spr='{0:.3f}'.format(a['spreads'])  if a['spreads']  is not None else '-',
        ))
    print()


if __name__ == '__main__':
    main()
