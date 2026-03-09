"""
Sector Rotation Backend v0.8
============================

Pattern Matching predictions with MASKED SIMILARITY (robust to missing ETF data).

Changes from v0.7 -> v0.8:
- Three separate similarity searches (25Y / 5Y / 1Y) — each window gets its own
  top-100 match pool, matching Similar Days Analyzer behavior
- Dominant window detection: whichever window has highest avg correlation scores
- Scoring uses dominant window predictions (win_rate x avg_return x stat_confidence
  x horizon_weight) — formula from v0.7, now window-aware
- Alignment bonus (0-3 pts): cross-window agreement replaces backward returns
- Stat bar: avg correlation per window + dominant window label
- dominant_window saved to JSON for future archive mining
- NaN tail fix: analog days within 20 calendar days of latest_date excluded
  from forward return calculations (prevents recent dates with no fwd data)
- Cell background logic unchanged from v0.5y

Changes from v0.5 -> v0.7 (carried forward):
- New scoring formula: win_rate x avg_return x stat_confidence x horizon_weight
- Added backward returns: 3M, 1M, 10D, 5D, 3D (actual % returns)
- Statistical confidence penalty: sample_count / 20

Changes from v0.4 -> v0.5:
- Replaced find_similar_days() with find_similar_days_masked()
- Added min_overlap parameter (default: 200 features ~= 28 ETFs x 7 features)

Usage:
    python sector_rotation_v0_8.py

Output:
    sector_rotation_data.json

Author: Brian + Claude
Date: 2026-02-24
Version: 0.8
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import warnings
import sys
from pathlib import Path

# Add perplexity-user-data (parent dir) to path so cache_loader and price_cache are found
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR   = os.path.normpath(os.path.join(_SCRIPT_DIR, '..', 'perplexity-user-data'))
if _DATA_DIR not in sys.path:
    sys.path.insert(0, _DATA_DIR)

# Import cache loader - try both naming conventions, then direct file load
load_etf_data = None
list_all_clusters = None
for _loader_name in ['cache_loader_v1_0', 'cache_loader_v1.0']:
    try:
        import importlib
        _mod = importlib.import_module(_loader_name.replace('.', '_') if '.' in _loader_name else _loader_name)
        load_etf_data = _mod.load_etf_data
        list_all_clusters = getattr(_mod, 'list_all_clusters', None)
        break
    except ImportError:
        pass

if load_etf_data is None:
    # Last resort: load directly from file
    import importlib.util
    for _fname in ['cache_loader_v1_0.py', 'cache_loader_v1.0.py']:
        _fpath = os.path.join(_DATA_DIR, _fname)
        if os.path.exists(_fpath):
            _spec = importlib.util.spec_from_file_location('cache_loader', _fpath)
            _mod  = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(_mod)
            load_etf_data     = _mod.load_etf_data
            list_all_clusters = getattr(_mod, 'list_all_clusters', None)
            print('[OK] Loaded cache_loader from: ' + _fpath)
            break

if load_etf_data is None:
    print('ERROR: Could not find cache_loader in: ' + _DATA_DIR)
    print('       Tried: cache_loader_v1_0.py, cache_loader_v1.0.py')
    sys.exit(1)

warnings.filterwarnings('ignore')

# ==============================================================================
# ETF CLUSTERS (for reference + fallback)
# ==============================================================================

ETF_CLUSTERS = {
    'US_BROAD': ['SPY', 'VOO', 'VTI', 'DIA', 'MDY', 'IWM', 'IWO', 'IWN', 'RSP'],
    'US_GROWTH': ['QQQ', 'SOXX', 'FTEC', 'XLC', 'SMH', 'KTEC', 'CQQQ'],
    'US_VALUE': ['VLUE', 'FVAL', 'DJD', 'SPLV', 'QUAL', 'VUG', 'CGDV'],
    'US_FACTORS': ['MTUM', 'MOAT', 'SIZE', 'SPHB', 'XRT'],
    'US_SECTORS': ['XLF', 'XLV', 'XLP', 'XLI', 'XLY', 'XLB', 'XLU', 'XLE', 'XLK', 'XLRE', 'REET'],
    'US_SPECIALTY': ['XBI', 'KRE', 'KBWB', 'XHB'],
    'INTL_DEVELOPED': ['EFA', 'IEFA', 'VEU', 'VGK', 'EWJ', 'EWG', 'EWP', 'EWU', 'EWC', 'EWA', 'EWL', 'FXE', 'FXB', 'FXC', 'FXA', 'FXY', 'FDEV'],
    'EMERGING_MARKETS': ['EEM', 'VWO', 'IEMG', 'EWZ', 'INDA', 'FXI', 'MCHI', 'KWEB', 'CPER', 'TCHI', 'AAXJ', 'AFK', 'ILF', 'EEMA', 'ESGE'],
    'COMMODITIES': ['DBC', 'PDBC', 'RAAX', 'DBA', 'USO', 'XOP', 'TAN', 'INFL', 'IAU', 'GDX', 'SLV', 'SLVP', 'SILJ', 'PALL', 'COPX', 'PLTM', 'STCE', 'CEW', 'ANGL'],
    'BONDS': ['AGG', 'BND', 'TLT', 'IEF', 'SHY', 'TIP', 'LQD', 'HYG', 'JNK', 'EMB', 'SCHZ', 'MUB'],
    'FX_VIX': ['UUP'],
}

ETF_NAMES = {
    'SPY': 'S&P 500', 'VOO': 'S&P 500 (V)', 'VTI': 'Total Market', 'DIA': 'Dow 30', 'MDY': 'Mid Cap', 'IWM': 'Small Cap',
    'QQQ': 'Nasdaq 100', 'SOXX': 'Semiconductors', 'FTEC': 'Tech (F)', 'XLC': 'Comm Svc', 'SMH': 'Semis (VN)', 'KTEC': 'Tech (K)', 'CQQQ': 'China Tech',
    'VLUE': 'Value Factor', 'FVAL': 'Value (F)', 'DJD': 'Dow Dividend', 'SPLV': 'Low Vol', 'QUAL': 'Quality', 'VUG': 'Growth (V)', 'CGDV': 'Dividend Grw',
    'MTUM': 'Momentum', 'MOAT': 'Wide Moat', 'SIZE': 'Size Factor', 'SPHB': 'High Beta', 'XRT': 'Retail',
    'XLF': 'Financials', 'XLV': 'Healthcare', 'XLP': 'Cons Staples', 'XLI': 'Industrials', 'XLY': 'Cons Discret',
    'XLB': 'Materials', 'XLU': 'Utilities', 'XLE': 'Energy', 'XLK': 'Technology', 'XLRE': 'Real Estate', 'REET': 'Global REIT',
    'XBI': 'Biotech', 'KRE': 'Reg Banks', 'KBWB': 'Banks', 'XHB': 'Homebuilders',
    'EFA': 'EAFE', 'VEU': 'World ex-US', 'VGK': 'Europe', 'EWJ': 'Japan', 'EWG': 'Germany', 'EWP': 'Spain',
    'EWU': 'UK', 'EWC': 'Canada', 'EWA': 'Australia', 'EWL': 'Switzerland',
    'FXE': 'Euro FX', 'FXB': 'British Pound', 'FXC': 'Canadian $', 'FXA': 'Aussie $', 'FXY': 'Yen', 'FDEV': 'Dev Mkts',
    'EEM': 'Emerging Mkts', 'VWO': 'EM (V)', 'IEMG': 'EM Core', 'EWZ': 'Brazil', 'INDA': 'India', 'FXI': 'China',
    'MCHI': 'China (MS)', 'KWEB': 'China Internet', 'CPER': 'Copper', 'TCHI': 'China Tech', 'AAXJ': 'Asia ex-Jpn',
    'AFK': 'Africa', 'ILF': 'Latin Am', 'EEMA': 'EM Asia', 'ESGE': 'EM ESG',
    'DBC': 'Commodities', 'PDBC': 'Cmdty (Invsc)', 'RAAX': 'Real Assets', 'DBA': 'Agriculture', 'USO': 'Oil',
    'XOP': 'Oil & Gas Exp', 'TAN': 'Solar', 'INFL': 'Inflation', 'IAU': 'Gold', 'GDX': 'Gold Miners',
    'SLV': 'Silver', 'SLVP': 'Silver Miners', 'PALL': 'Palladium', 'COPX': 'Copper Miners', 'PLTM': 'Platinum',
    'STCE': 'Schwab Crypto', 'CEW': 'EM Currency', 'ANGL': 'Fallen Angels',
    'AGG': 'Agg Bond', 'BND': 'Total Bond', 'TLT': 'Long Treasury', 'IEF': 'Med Treasury', 'SHY': 'Short Treasury',
    'TIP': 'TIPS', 'LQD': 'Inv Grade Corp', 'HYG': 'High Yield', 'JNK': 'Junk Bonds', 'EMB': 'EM Bonds',
    'SCHZ': 'Agg Bond (S)', 'MUB': 'Muni Bonds',
    'UUP': 'US Dollar',
}

TICKERS = sorted(set(sum(ETF_CLUSTERS.values(), [])))

# ==============================================================================
# DATA LOADING FROM CACHE
# ==============================================================================

def load_all_data_from_cache():
    """Load ETF data from price_cache using cache_loader"""
    print('Loading ETF data from cache...')
    all_data = load_etf_data()
    print(f'Loaded: {len(all_data)} ETFs')
    return all_data

# ==============================================================================
# PATTERN MATCHING (Sector Rotation)
# ==============================================================================

def calculate_momentum_features(df):
    """Calculate momentum features for pattern matching"""
    features = pd.DataFrame(index=df.index)
    
    # Returns for different periods
    for period, name in [(63, '3M'), (21, '1M'), (10, '10D'), (5, '5D')]:
        ret = df['close'].pct_change(period) * 100
        
        # Normalize to z-score
        rolling_mean = ret.rolling(252, min_periods=60).mean()
        rolling_std = ret.rolling(252, min_periods=60).std()
        features[name] = ((ret - rolling_mean) / rolling_std.replace(0, np.nan)).clip(-2, 2)
    
    # Slopes (acceleration)
    features['slope1'] = features['1M'] - features['3M']
    features['slope2'] = features['10D'] - features['1M']
    features['slope3'] = features['5D'] - features['10D']
    
    # Normalize slopes
    for col in ['slope1', 'slope2', 'slope3']:
        rolling_mean = features[col].rolling(252, min_periods=60).mean()
        rolling_std = features[col].rolling(252, min_periods=60).std()
        features[col] = ((features[col] - rolling_mean) / rolling_std.replace(0, np.nan)).clip(-2, 2)
    
    return features

def build_pattern_matrix(all_data):
    """Build the pattern feature matrix for all ETFs"""
    print('\nBuilding pattern matrix...')
    all_features = {}
    all_forward = {}
    
    for ticker, df in all_data.items():
        features = calculate_momentum_features(df)
        
        # Add ticker prefix
        for col in features.columns:
            all_features[f'{ticker}_{col}'] = features[col]
        
        # Forward returns
        for period in [3, 5, 10]:
            all_forward[f'{ticker}_fwd_{period}d'] = df['close'].pct_change(period).shift(-period) * 100
    
    features_df = pd.DataFrame(all_features)
    forward_df = pd.DataFrame(all_forward)
    
    print(f'Pattern matrix: {len(features_df)} days x {len(features_df.columns)} features')
    return features_df, forward_df

def find_similar_days_masked(features_df, forward_df, target_date, min_correlation=0.2,
                              top_n=100, min_overlap=200, history_cutoff=None):
    """
    Find historical days most similar to target date using MASKED correlation.
    
    Computes correlation only on non-NaN features that exist in both the target
    and historical dates. Ignores missing ETFs gracefully.
    
    Args:
        features_df: DataFrame with all ETF momentum features (rows=dates, cols=features)
        forward_df: DataFrame with forward returns (for reference, not used here)
        target_date: date to find analogs for
        min_correlation: minimum correlation threshold
        top_n: return top N similar days
        min_overlap: minimum number of overlapping non-NaN features required
                     (default 200 ≈ 28 ETFs × 7 features each)
    
    Returns:
        List of (date, correlation) tuples sorted by correlation descending
    """
    
    if target_date not in features_df.index:
        return []
    
    target_row = features_df.loc[target_date]
    target_available = ~target_row.isna()  # Boolean mask of non-NaN features
    
    if target_available.sum() < min_overlap:
        print(f'WARNING: Target date {target_date.strftime("%Y-%m-%d")} has only '
              f'{target_available.sum()} non-NaN features (threshold: {min_overlap})')
        return []
    
    similarities = []
    cutoff_date = target_date - timedelta(days=15)  # Don't use recent history for analogs
    
    for hist_date in features_df.index:
        if hist_date >= cutoff_date:
            continue
        # Window filter: skip dates outside the requested history window
        if history_cutoff is not None and hist_date < history_cutoff:
            continue
        
        hist_row = features_df.loc[hist_date]
        hist_available = ~hist_row.isna()
        
        # Intersection: features that exist in BOTH target and historical
        overlap_mask = target_available & hist_available
        overlap_count = overlap_mask.sum()
        
        if overlap_count < min_overlap:
            continue
        
        # Extract values for overlapping features only
        target_vals = target_row[overlap_mask].values
        hist_vals = hist_row[overlap_mask].values
        
        # Double-check no NaNs (should be guaranteed by mask, but safeguard)
        if np.isnan(target_vals).any() or np.isnan(hist_vals).any():
            continue
        
        # Compute correlation on the overlapping subset
        corr = np.corrcoef(target_vals, hist_vals)[0, 1]
        
        if not np.isnan(corr) and corr >= min_correlation:
            similarities.append((hist_date, corr))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]

def calculate_predictions(similar_days, forward_df, ticker, history_cutoff=None, latest_date=None):
    """Calculate predicted forward returns based on similar days.
    
    history_cutoff: if set, only use analog days >= this date (window filter)
    latest_date: if set, exclude analog days within 20 calendar days of latest_date
                 (prevents NaN tail — recent dates have no fwd return data yet)
    """
    
    if history_cutoff:
        similar_days = [(d, c) for d, c in similar_days if d >= history_cutoff]
    
    # Exclude analog days too recent to have valid forward return data
    if latest_date is not None:
        min_analog_date = latest_date - timedelta(days=20)
        similar_days = [(d, c) for d, c in similar_days if d <= min_analog_date]
    
    if len(similar_days) < 3:
        return None
    
    results = {}
    
    for period in [3, 5, 10]:
        col = f'{ticker}_fwd_{period}d'
        
        if col not in forward_df.columns:
            results[f'{period}d'] = None
            continue
        
        returns = []
        for date, corr in similar_days:
            if date in forward_df.index:
                ret = forward_df.loc[date, col]
                if not np.isnan(ret):
                    returns.append(ret)
        
        if len(returns) >= 3:
            results[f'{period}d'] = {
                'avg_ret': round(np.mean(returns), 2),
                'pct_pos': round((np.array(returns) > 0).mean() * 100, 1),
                'count': len(returns)
            }
        else:
            results[f'{period}d'] = None
    
    return results

# ==============================================================================
# BUILD SECTOR ROTATION DASHBOARD
# ==============================================================================

def build_sector_rotation_dashboard(all_data):
    """Build the sector rotation dashboard with pattern predictions"""
    
    print('\n' + '='*80)
    print('Building Sector Rotation Dashboard')
    print('='*80)
    
    # Build pattern matrix
    features_df, forward_df = build_pattern_matrix(all_data)
    
    latest_date = features_df.index.max()
    print(f'\nFinding similar historical patterns (as of {latest_date.strftime("%Y-%m-%d")})...')
    print(f'Running separate search per window (25Y / 5Y / 1Y)...')
    
    # -----------------------------------------------------------------------
    # THREE SEPARATE SEARCHES - one per window, matching Similar Days Analyzer
    # -----------------------------------------------------------------------
    window_cutoffs = {
        '25Y': None,
        '5Y':  latest_date - timedelta(days=5*365),
        '1Y':  latest_date - timedelta(days=365),
    }
    
    similar_days_by_window = {}
    window_avg_corr = {}
    
    for window, cutoff in window_cutoffs.items():
        matches = find_similar_days_masked(
            features_df, forward_df, latest_date,
            min_overlap=200,
            history_cutoff=cutoff
        )
        similar_days_by_window[window] = matches
        avg_corr = round(float(np.mean([c for _, c in matches])), 3) if matches else 0.0
        window_avg_corr[window] = avg_corr
        print(f'  {window}: {len(matches)} matches, avg corr = {avg_corr:.3f}')
    
    # Dominant window = highest avg correlation
    dominant_window = max(window_avg_corr, key=window_avg_corr.get)
    print(f'\nDominant window: {dominant_window} (avg corr {window_avg_corr[dominant_window]:.3f})')
    print(f'Top 5 matches ({dominant_window}):')
    for date, corr in similar_days_by_window[dominant_window][:5]:
        print(f'  {date.strftime("%Y-%m-%d")}: {corr:.3f}')
    
    # -----------------------------------------------------------------------
    # BUILD OUTPUT PER ETF
    # -----------------------------------------------------------------------
    print('\nCalculating sector rotation scores for each ETF...')
    dashboard = {}
    
    horizon_weights = {
        '10d': 0.5,
        '5d':  0.3,
        '3d':  0.2,
    }
    
    for i, ticker in enumerate(all_data.keys()):
        if (i + 1) % 20 == 0:
            print(f'  Processed {i+1}/{len(all_data)} ETFs...')
        
        etf_data = {
            'name': ETF_NAMES.get(ticker, ticker),
            'ret_5d': None,
            'sr_score': 0,
            'sr_total': 10,
            'predictions': {},
            'hist_agree': None,
            'ret_3m': None,
            'ret_1m': None,
            'ret_10d': None,
            'ret_5d_actual': None,
            'ret_3d': None,
            'mom_3m': None,
            'mom_1m': None,
            'mom_10d': None,
            'mom_5d': None,
        }
        
        # Backward returns (actual %)
        df = all_data[ticker]
        if len(df) >= 3:
            etf_data['ret_3d'] = round(df['close'].pct_change(3).iloc[-1] * 100, 2)
        if len(df) >= 5:
            etf_data['ret_5d'] = round(df['close'].pct_change(5).iloc[-1] * 100, 2)
            etf_data['ret_5d_actual'] = etf_data['ret_5d']
        if len(df) >= 10:
            etf_data['ret_10d'] = round(df['close'].pct_change(10).iloc[-1] * 100, 2)
        if len(df) >= 21:
            etf_data['ret_1m'] = round(df['close'].pct_change(21).iloc[-1] * 100, 2)
        if len(df) >= 63:
            etf_data['ret_3m'] = round(df['close'].pct_change(63).iloc[-1] * 100, 2)
        
        # Momentum z-scores (from features_df)
        for mom_period, mom_key in [('3M', 'mom_3m'), ('1M', 'mom_1m'), ('10D', 'mom_10d'), ('5D', 'mom_5d')]:
            col_name = f'{ticker}_{mom_period}'
            if col_name in features_df.columns:
                latest_val = features_df[col_name].dropna()
                if len(latest_val) > 0:
                    etf_data[mom_key] = round(latest_val.iloc[-1], 2)
        
        # Predictions for all three windows
        for window, matches in similar_days_by_window.items():
            preds = calculate_predictions(
                matches, forward_df, ticker,
                latest_date=latest_date
            )
            etf_data['predictions'][window] = preds
        
        # ------------------------------------------------------------------
        # SCORING (0-10)
        # Part 1: Pattern score (0-7) using dominant window predictions
        # Part 2: Cross-window alignment bonus (0-3)
        # ------------------------------------------------------------------
        
        pred_dominant = etf_data['predictions'].get(dominant_window) or {}
        
        # Part 1: win_rate x avg_return x stat_confidence x horizon_weight
        pattern_score = 0.0
        pred_direction = 0
        
        for horizon, weight in horizon_weights.items():
            pred_data = pred_dominant.get(horizon) if pred_dominant else None
            if pred_data:
                avg_ret = pred_data.get('avg_ret', 0)
                win_rate = pred_data.get('pct_pos', 0) / 100.0
                sample_count = pred_data.get('count', 0)
                stat_confidence = min(1.0, sample_count / 20.0)
                normalized_return = avg_ret / 5.0
                horizon_score = win_rate * normalized_return * stat_confidence * weight * 7.0
                pattern_score += horizon_score
                if horizon == '10d' and avg_ret != 0:
                    pred_direction = 1 if avg_ret > 0 else -1
        
        pattern_score = max(0, min(7, pattern_score))
        
        # Part 2: Cross-window alignment bonus
        # Count how many other windows agree on 10d direction with dominant
        alignment_score = 0
        if pred_direction != 0:
            other_windows = [w for w in ['25Y', '5Y', '1Y'] if w != dominant_window]
            agreeing = 0
            for w in other_windows:
                other_pred = etf_data['predictions'].get(w)
                if other_pred and other_pred.get('10d'):
                    other_ret = other_pred['10d'].get('avg_ret', 0)
                    if other_ret != 0 and (other_ret > 0) == (pred_direction > 0):
                        agreeing += 1
            if agreeing == 2:
                alignment_score = 3
            elif agreeing == 1:
                alignment_score = 2
        
        sr_score = round(pattern_score + alignment_score, 1)
        etf_data['sr_score'] = sr_score
        etf_data['sr_total'] = 10
        
        # History agreement based on dominant window 10d direction
        if pred_dominant and pred_dominant.get('10d'):
            ret_dom = pred_dominant['10d'].get('avg_ret', 0)
            if ret_dom > 0:
                etf_data['hist_agree'] = 'bullish'
            elif ret_dom < 0:
                etf_data['hist_agree'] = 'bearish'
            else:
                etf_data['hist_agree'] = 'neutral'
        
        dashboard[ticker] = etf_data
    
    # Bundle window metadata for stat bar + JSON archive
    window_meta = {
        'dominant_window': dominant_window,
        'avg_corr_25y': window_avg_corr.get('25Y', 0),
        'avg_corr_5y':  window_avg_corr.get('5Y', 0),
        'avg_corr_1y':  window_avg_corr.get('1Y', 0),
        'match_count_25y': len(similar_days_by_window.get('25Y', [])),
        'match_count_5y':  len(similar_days_by_window.get('5Y', [])),
        'match_count_1y':  len(similar_days_by_window.get('1Y', [])),
    }
    
    return dashboard, latest_date, window_meta

# ==============================================================================
# CLUSTER LOOKUP
# ==============================================================================

TICKER_TO_CLUSTER = {}
for _cluster, _tickers in ETF_CLUSTERS.items():
    for _t in _tickers:
        TICKER_TO_CLUSTER[_t] = _cluster.replace('_', ' ').title()

# ==============================================================================
# HTML BUILDER - JS-rendered table with per-column gradient coloring
# ==============================================================================

EXTRA_CSS = """
/* Sector rotation table - compact for wide data */
#secrot-table-wrap table {
    border-collapse: collapse;
    width: 100%;
    font-size: 0.8em;
}
#secrot-table-wrap thead th {
    padding: 7px 8px;
    font-size: 0.75em;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    cursor: pointer;
    user-select: none;
    border-bottom: 2px solid #e2e4e8;
    white-space: nowrap;
}
#secrot-table-wrap thead th:hover { opacity: 0.8; }
#secrot-table-wrap tbody td {
    padding: 6px 8px;
    border-bottom: 1px solid #f0f0f0;
    vertical-align: middle;
    text-align: center;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.95em;
}
#secrot-table-wrap tbody tr:hover { outline: 2px solid #4f46e5; }

/* Group header row */
.grp-row th { padding: 5px 4px; font-size: 0.78em; font-weight: 700; letter-spacing: 0.05em; }
.grp-blank    { background: #f8f9fb; color: transparent; }
.grp-momentum { background: #1e293b; color: #94a3b8; }
.grp-25y      { background: #14532d; color: #86efac; }
.grp-5y       { background: #1e3a5f; color: #93c5fd; }
.grp-1y       { background: #3b1f5e; color: #c4b5fd; }

/* Ticker / name cells */
.cell-ticker { font-weight: 700; text-align: left !important; color: #1a1a2e; }
.cell-name   { text-align: left !important; color: #555; font-family: 'IBM Plex Sans', sans-serif !important; font-size: 0.88em !important; max-width: 120px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.cell-ret    { font-weight: 600; }
.cell-score  { font-weight: 700; }
.cell-miss   { color: #bbb !important; background: #fafafa !important; }

/* Sort arrows */
th.sort-asc::after  { content: ' ▲'; font-size: 0.7em; }
th.sort-desc::after { content: ' ▼'; font-size: 0.7em; }

/* Hist agree pills */
.ha-bullish { background: #dcfce7; color: #15803d; border-radius: 20px; padding: 2px 8px; font-size: 0.85em; font-weight: 700; font-family: 'IBM Plex Sans', sans-serif; }
.ha-bearish { background: #fee2e2; color: #b91c1c; border-radius: 20px; padding: 2px 8px; font-size: 0.85em; font-weight: 700; font-family: 'IBM Plex Sans', sans-serif; }
.ha-mixed   { background: #fef9c3; color: #854d0e; border-radius: 20px; padding: 2px 8px; font-size: 0.85em; font-weight: 700; font-family: 'IBM Plex Sans', sans-serif; }

/* pred sub-text (win%) */
.pred-pct { display: block; font-size: 0.78em; color: #333; opacity: 0.75; }

/* ── Mobile responsive ── */
@media (max-width: 768px) {
    #secrot-table-wrap { overflow-x: auto; -webkit-overflow-scrolling: touch; }
    #secrot-table-wrap table { font-size: 0.72em; }
    #secrot-table-wrap thead th { padding: 5px 4px; font-size: 0.68em; }
    #secrot-table-wrap tbody td { padding: 4px 4px; font-size: 0.85em; }
    .cell-name { max-width: 80px; }
    .ha-bullish, .ha-bearish, .ha-mixed { font-size: 0.75em; padding: 2px 6px; }
}
"""

SORT_AND_RENDER_JS_TEMPLATE = """
(function() {{
    const DATA = {data_json};
    const CLUSTER = {cluster_json};

    let currentSort = {{ col: 'sr_score', dir: 'desc' }};
    let columnRanges = {{}};

    // ---- Gradient: red(0) -> orange -> yellow -> green(1) ----
    function getGradientColor(norm01) {{
        const c = Math.max(0, Math.min(1, norm01));
        let r, g, b;
        if (c < 0.25) {{
            const t = c * 4;
            r = Math.round(211 + (255 - 211) * t); g = Math.round(47 + (152 - 47) * t); b = Math.round(47 * (1 - t));
        }} else if (c < 0.5) {{
            const t = (c - 0.25) * 4;
            r = 255; g = Math.round(152 + (235 - 152) * t); b = Math.round(59 * t);
        }} else if (c < 0.75) {{
            const t = (c - 0.5) * 4;
            r = Math.round(255 + (139 - 255) * t); g = Math.round(235 + (195 - 235) * t); b = Math.round(59 + (74 - 59) * t);
        }} else {{
            const t = (c - 0.75) * 4;
            r = Math.round(139 * (1 - t)); g = Math.round(195 + (200 - 195) * t); b = Math.round(74 + (83 - 74) * t);
        }}
        return `rgb(${{r}},${{g}},${{b}})`;
    }}

    // ---- Per-column 5th-95th percentile ranges ----
    function calculateColumnRanges() {{
        columnRanges = {{}};
        const cols = {{
            score: [],
            mom_3m: [], mom_1m: [], mom_10d: [], mom_5d: [],
            pred_25y_10d: [], pred_25y_5d: [], pred_25y_3d: [],
            pred_5y_10d:  [], pred_5y_5d:  [], pred_5y_3d:  [],
            pred_1y_10d:  [], pred_1y_5d:  [], pred_1y_3d:  [],
        }};
        for (const [tk, etf] of Object.entries(DATA)) {{
            const st = etf.sr_total || 9;
            if (st > 0) cols.score.push((etf.sr_score || 0) / st);
            for (const k of ['mom_3m','mom_1m','mom_10d','mom_5d']) {{
                if (etf[k] != null) cols[k].push(etf[k]);
            }}
            for (const hist of ['25Y','5Y','1Y']) {{
                for (const p of ['10d','5d','3d']) {{
                    const key = `pred_${{hist.toLowerCase()}}_${{p}}`;
                    const val = etf.predictions?.[hist]?.[p]?.avg_ret;
                    if (val != null) cols[key].push(val);
                }}
            }}
        }}
        for (const [key, vals] of Object.entries(cols)) {{
            if (!vals.length) continue;
            const s = [...vals].sort((a,b) => a - b);
            columnRanges[key] = {{
                min: s[Math.floor(s.length * 0.05)],
                max: s[Math.floor(s.length * 0.95)]
            }};
        }}
    }}

    function normalize(val, key) {{
        if (val == null) return null;
        const r = columnRanges[key];
        if (!r || r.max === r.min) return 0.5;
        return (val - r.min) / (r.max - r.min);
    }}

    function bgColor(val, key) {{
        const n = normalize(val, key);
        return n == null ? '#f5f5f5' : getGradientColor(n);
    }}

    function textColor(bgRgb) {{
        // pick black or white text based on background luminance
        const m = bgRgb.match(/rgb\\((\\d+),(\\d+),(\\d+)\\)/);
        if (!m) return '#000';
        const lum = 0.299*m[1] + 0.587*m[2] + 0.114*m[3];
        return lum > 140 ? '#000' : '#fff';
    }}

    function getSortVal(tk, col) {{
        const etf = DATA[tk];
        if (!etf) return -999;
        switch(col) {{
            case 'owned':        return window._owned.has(tk) ? 1 : 0;
            case 'ticker':       return tk;
            case 'name':         return etf.name || '';
            case 'ret_5d':       return etf.ret_5d ?? -999;
            case 'sr_score':     return etf.sr_score ?? 0;
            case 'hist_agree':   return etf.hist_agree || '';
            case 'mom_3m':       return etf.mom_3m  ?? -999;
            case 'mom_1m':       return etf.mom_1m  ?? -999;
            case 'mom_10d':      return etf.mom_10d ?? -999;
            case 'mom_5d':       return etf.mom_5d  ?? -999;
            default: {{
                // pred_25y_10d etc
                const m = col.match(/^pred_(\\w+)_(\\d+d)$/);
                if (m) {{
                    const hist = m[1].toUpperCase();
                    return etf.predictions?.[hist]?.[m[2]]?.avg_ret ?? -999;
                }}
                return 0;
            }}
        }}
    }}

    function render() {{
        calculateColumnRanges();
        const tickers = Object.keys(DATA).sort((a, b) => {{
            const va = getSortVal(a, currentSort.col);
            const vb = getSortVal(b, currentSort.col);
            if (typeof va === 'string') return currentSort.dir === 'asc' ? va.localeCompare(vb) : vb.localeCompare(va);
            return currentSort.dir === 'asc' ? va - vb : vb - va;
        }});

        const arrow = col => currentSort.col === col ? (currentSort.dir === 'asc' ? 'sort-asc' : 'sort-desc') : '';
        const th = (col, label, title, cls='') =>
            `<th class="${{arrow(col)}} ${{cls}}" onclick="window._secrot_sort('${{col}}')" title="${{title}}">${{label}}</th>`;

        let html = '<table><thead>';
        // Group row
        html += '<tr class="grp-row">';
        html += '<th colspan="6" class="grp-blank"></th>';
        html += '<th colspan="4" class="grp-momentum">Momentum (z-score)</th>';
        html += '<th colspan="3" class="grp-25y">Pred (25Y)</th>';
        html += '<th colspan="3" class="grp-5y">Pred (5Y)</th>';
        html += '<th colspan="3" class="grp-1y">Pred (1Y)</th>';
        html += '</tr>';
        // Column row
        html += '<tr>';
        html += th('owned',      'Own',        'Mark as owned', 'own-th');
        html += th('ticker',     'ETF',        'Ticker symbol');
        html += th('name',       'Name',       'ETF name');
        html += th('sr_score',   'Score ▼',    'SecRot score 0-9');
        html += th('hist_agree', 'Agree',      'All 3 windows agree on direction');
        html += th('ret_5d',     '5D Ret',     '5-day actual return');
        html += th('mom_3m',     '3M',         '3-month momentum z-score',  'grp-momentum');
        html += th('mom_1m',     '1M',         '1-month momentum z-score',  'grp-momentum');
        html += th('mom_10d',    '10D',        '10-day momentum z-score',   'grp-momentum');
        html += th('mom_5d',     '5D',         '5-day momentum z-score',    'grp-momentum');
        html += th('pred_25y_10d','10d',       '10d pred from 25Y history', 'grp-25y');
        html += th('pred_25y_5d', '5d',        '5d pred from 25Y history',  'grp-25y');
        html += th('pred_25y_3d', '3d',        '3d pred from 25Y history',  'grp-25y');
        html += th('pred_5y_10d', '10d',       '10d pred from 5Y history',  'grp-5y');
        html += th('pred_5y_5d',  '5d',        '5d pred from 5Y history',   'grp-5y');
        html += th('pred_5y_3d',  '3d',        '3d pred from 5Y history',   'grp-5y');
        html += th('pred_1y_10d', '10d',       '10d pred from 1Y history',  'grp-1y');
        html += th('pred_1y_5d',  '5d',        '5d pred from 1Y history',   'grp-1y');
        html += th('pred_1y_3d',  '3d',        '3d pred from 1Y history',   'grp-1y');
        html += '</tr></thead><tbody>';

        for (const tk of tickers) {{
            const etf = DATA[tk];
            const score = etf.sr_score ?? 0;
            const total = etf.sr_total ?? 9;
            const scoreBg = bgColor(total > 0 ? score/total : 0, 'score');
            const scoreFg = textColor(scoreBg);

            // 5d return
            const ret5 = etf.ret_5d;
            const retStr = ret5 != null ? (ret5 >= 0 ? '+' : '') + ret5.toFixed(2) + '%' : '—';
            const retColor = ret5 == null ? '#888' : (ret5 >= 0 ? '#16a34a' : '#dc2626');

            // hist agree
            const ha = etf.hist_agree;
            const haHtml = ha === 'bullish' ? `<span class="ha-bullish">BULL</span>`
                         : ha === 'bearish' ? `<span class="ha-bearish">BEAR</span>`
                         : ha === 'mixed'   ? `<span class="ha-mixed">MIX</span>`
                         : '<span style="color:#bbb">--</span>';

            const _isOwned = window._owned.has(tk);
            html += `<tr class="${{_isOwned ? 'row-owned' : ''}}">`;
            html += `<td><input type="checkbox" class="own-cb" ${{_isOwned ? 'checked' : ''}} onclick="window._secrot_toggleOwned('${{tk}}')" title="Mark as owned"></td>`;
            html += `<td class="cell-ticker">${{tk}}</td>`;
            html += `<td class="cell-name" title="${{etf.name}}">${{etf.name}}</td>`;
            html += `<td class="cell-score" style="background:${{scoreBg}};color:${{scoreFg}};">${{score}}/${{total}}</td>`;
            html += `<td style="text-align:center;">${{haHtml}}</td>`;
            html += `<td class="cell-ret" style="color:${{retColor}}">${{retStr}}</td>`;

            // Momentum
            for (const [k, mk] of [['mom_3m','mom_3m'],['mom_1m','mom_1m'],['mom_10d','mom_10d'],['mom_5d','mom_5d']]) {{
                const v = etf[k];
                if (v != null) {{
                    const bg = bgColor(v, mk); const fg = textColor(bg);
                    html += `<td style="background:${{bg}};color:${{fg}};">${{v.toFixed(1)}}</td>`;
                }} else {{
                    html += '<td class="cell-miss">—</td>';
                }}
            }}

            // Predictions
            for (const [hist, hkey] of [['25Y','25y'],['5Y','5y'],['1Y','1y']]) {{
                for (const p of ['10d','5d','3d']) {{
                    const pred = etf.predictions?.[hist]?.[p];
                    const key = `pred_${{hkey}}_${{p}}`;
                    if (pred) {{
                        const bg = bgColor(pred.avg_ret, key); const fg = textColor(bg);
                        const sign = pred.avg_ret >= 0 ? '+' : '';
                        html += `<td style="background:${{bg}};color:${{fg}};" title="n=${{pred.count}}">${{sign}}${{pred.avg_ret.toFixed(1)}}%<span class="pred-pct">${{pred.pct_pos.toFixed(0)}}%</span></td>`;
                    }} else {{
                        html += '<td class="cell-miss">—</td>';
                    }}
                }}
            }}

            html += '</tr>';
        }}

        html += '</tbody></table>';
        document.getElementById('secrot-table-wrap').innerHTML = html;
    }}

    window._secrot_sort = function(col) {{
        currentSort.dir = (currentSort.col === col && currentSort.dir === 'desc') ? 'asc' : 'desc';
        currentSort.col = col;
        render();
    }};

    window._secrot_toggleOwned = function(tk) {{
        window._owned.toggle(tk);
        render();
    }};

    render();
}})();
"""


def build_body_html(dashboard, latest_date, similar_days_count, writer, window_meta=None):
    """Build full body HTML for DashboardWriter.write()."""
    import json as _json

    date_str = latest_date.strftime('%Y-%m-%d')
    etf_count = len(dashboard)
    bullish_count = sum(1 for d in dashboard.values() if d.get('hist_agree') == 'bullish')
    bearish_count = sum(1 for d in dashboard.values() if d.get('hist_agree') == 'bearish')

    # Window metadata for stat bar
    wm = window_meta or {}
    dominant = wm.get('dominant_window', '?')
    corr_25y = f"{wm.get('avg_corr_25y', 0):.3f}" if wm.get('avg_corr_25y') else '--'
    corr_5y  = f"{wm.get('avg_corr_5y',  0):.3f}" if wm.get('avg_corr_5y')  else '--'
    corr_1y  = f"{wm.get('avg_corr_1y',  0):.3f}" if wm.get('avg_corr_1y')  else '--'

    parts = []

    # --- Stat bar ---
    parts.append(writer.stat_bar([
        ('Data Date',      date_str,                   'neutral'),
        ('ETFs',           str(etf_count),              'neutral'),
        ('Dominant',       dominant,                    'neutral'),
        ('25Y Avg Corr',   corr_25y,                   'neutral'),
        ('5Y Avg Corr',    corr_5y,                    'neutral'),
        ('1Y Avg Corr',    corr_1y,                    'neutral'),
        ('All 3 Bullish',  str(bullish_count),          'pos'),
        ('All 3 Bearish',  str(bearish_count),          'neg'),
    ]))

    # --- Page header + nav ---
    parts.append(writer.build_header(
        'Pattern matching across 25Y / 5Y / 1Y history windows'
    ))

    # --- Legend ---
    legend_html = (
        '<div style="font-size:0.88em;color:#555;line-height:1.9;">'
        '<strong>Score 0-9:</strong> Win rate (0-2) + Return magnitude (0-2) + '
        '5Y confirms (0-2) + 25Y confirms (0-2) + Horizon agreement (0-1)<br>'
        '<strong>Agree:</strong> All 3 history windows predict same 10d direction &nbsp;|&nbsp; '
        '<strong>Momentum:</strong> z-score vs history &nbsp;|&nbsp; '
        '<strong>Cell colors:</strong> per-column gradient, 5th-95th percentile range &nbsp;|&nbsp; '
        '<strong>Pred cells:</strong> avg return (top) + win% (bottom) &nbsp;|&nbsp; '
        '<span style="color:#86efac;background:#14532d;padding:1px 6px;border-radius:3px;">25Y</span> '
        '<span style="color:#93c5fd;background:#1e3a5f;padding:1px 6px;border-radius:3px;">5Y</span> '
        '<span style="color:#c4b5fd;background:#3b1f5e;padding:1px 6px;border-radius:3px;">1Y</span>'
        '</div>'
    )
    parts.append(writer.section('How to Read', legend_html))

    # --- Table container (JS fills this) ---
    parts.append(
        '<div class="table-section">'
        '<div class="table-section-header">'
        '<h2>SecRot Deep Dive &mdash; Pattern Matching</h2>'
        '<span style="font-size:0.85em;color:#888;">Click any column header to sort</span>'
        '</div>'
        '<div style="overflow-x:auto;padding:0;">'
        '<div id="secrot-table-wrap" style="padding:0;"></div>'
        '</div>'
        '</div>'
    )

    parts.append(writer.footer())

    # --- Embed data and render JS ---
    data_json = _json.dumps(dashboard, ensure_ascii=False)
    cluster_json = _json.dumps(TICKER_TO_CLUSTER, ensure_ascii=False)
    render_js = SORT_AND_RENDER_JS_TEMPLATE.format(
        data_json=data_json,
        cluster_json=cluster_json,
    )

    # Inject as extra_js (appended inside <script> tag by writer)
    # We store it on the writer temporarily
    writer._secrot_extra_js = render_js

    return '\n'.join(parts)


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == '__main__':
    print('=' * 80)
    print('Sector Rotation Backend v0.8')
    print('=' * 80)
    print()

    try:
        from dashboard_writer import DashboardWriter, write_landing_page
    except ImportError:
        print('ERROR: dashboard_writer.py not found in current directory')
        import sys; sys.exit(1)

    # Load data from cache
    all_data = load_all_data_from_cache()

    if not all_data:
        print('ERROR: No ETF data loaded from cache.')
        print('Run: python create_cache_metadata_v1_1.py')
        sys.exit(1)

    print(f'Successfully loaded: {len(all_data)} ETFs')

    # Build dashboard data
    dashboard, latest_date, window_meta = build_sector_rotation_dashboard(all_data)
    similar_days_count = '?'
    print(f'ETFs processed: {len(dashboard)}')

    # Write HTML dashboard
    writer = DashboardWriter('sector-rotation', 'SecRot Deep Dive - Pattern Matching')
    body = build_body_html(dashboard, latest_date, similar_days_count, writer, window_meta=window_meta)
    extra_js = getattr(writer, '_secrot_extra_js', '')
    writer.write(body, extra_css=EXTRA_CSS, extra_js=extra_js)

    # Write CSV
    csv_path = os.path.join(_SCRIPT_DIR, 'sector_rotation_data_{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M')))
    csv_rows = []
    for ticker, etf in dashboard.items():
        row = {
            'ticker': ticker,
            'name': etf.get('name', ''),
            'sr_score': etf.get('sr_score'),
            'sr_total': etf.get('sr_total'),
            'hist_agree': etf.get('hist_agree'),
            'ret_3d': etf.get('ret_3d'),
            'ret_5d': etf.get('ret_5d'),
            'ret_10d': etf.get('ret_10d'),
            'ret_1m': etf.get('ret_1m'),
            'ret_3m': etf.get('ret_3m'),
            'mom_3m': etf.get('mom_3m'),
            'mom_1m': etf.get('mom_1m'),
            'mom_10d': etf.get('mom_10d'),
            'mom_5d': etf.get('mom_5d'),
        }
        # Flatten dominant window predictions
        dominant = window_meta.get('dominant_window', '25Y') if window_meta else '25Y'
        preds = (etf.get('predictions') or {}).get(dominant) or {}
        for hz in ['3d', '5d', '10d']:
            p = preds.get(hz) or {}
            row['pred_{}_avg'.format(hz)] = p.get('avg_return')
            row['pred_{}_winrate'.format(hz)] = p.get('pct_positive')
            row['pred_{}_n'.format(hz)] = p.get('count')
        csv_rows.append(row)
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False, encoding='utf-8')
    print(f'CSV: {csv_path}')

    print()
    print('Dashboard written to GitHub Pages repo.')
    print(f'Latest date: {latest_date.strftime("%Y-%m-%d")}')
    print(f'ETFs: {len(dashboard)}')
