"""
Similar Days Analyzer v1.12
===========================

Analyzes what happened BEFORE (last 5d) and AFTER (next 5d) pattern-matched similar days.
Shows backward context and forward outcomes across all assets.

Usage:
    python similar_days_analyzer_v1.12.py

Input:
    - price_cache (via cache_loader)
    - Uses same pattern matching logic as sector_rotation_v0_7.py

Output:
    - similar_days_analysis.json (backward + forward patterns)
    - Console output with key findings

Changes in v1.12:
    - Added INDEXES analysis group (SPY, DIA, ^NDX, IAU, VGK, MCHI, TLT)
    - Expanded STYLE factors (added size/IJR, min_vol/USMV, dividend/VIG)
    - Changed value factor from VLUE to VTV (higher AUM, lower expense ratio)
    - Added ETF ticker labels for display in dashboard

Author: Brian + Claude
Date: 2026-01-28
Version: 1.12 (Expanded indexes and style factors)
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import sys
import importlib.util
from collections import defaultdict

# Path fix: scripts run from market-dashboards\ but data is one level up
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR   = os.path.normpath(os.path.join(_SCRIPT_DIR, '..', 'perplexity-user-data'))
if _DATA_DIR not in sys.path:
    sys.path.insert(0, _DATA_DIR)

# Import cache loader - robust multi-name fallback
load_etf_data = None
for _loader_name in ['cache_loader_v1_0', 'cache_loader_v1.0']:
    try:
        _mod = importlib.import_module(_loader_name.replace('.', '_') if '.' in _loader_name else _loader_name)
        load_etf_data = _mod.load_etf_data
        break
    except ImportError:
        pass

if load_etf_data is None:
    for _fname in ['cache_loader_v1_0.py', 'cache_loader_v1.0.py']:
        _fpath = os.path.join(_DATA_DIR, _fname)
        if os.path.exists(_fpath):
            _spec = importlib.util.spec_from_file_location('cache_loader', _fpath)
            _mod  = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(_mod)
            load_etf_data = _mod.load_etf_data
            print('[OK] Loaded cache_loader from: ' + _fpath)
            break

if load_etf_data is None:
    print('ERROR: Could not find cache_loader in: ' + _DATA_DIR)
    sys.exit(1)

# ==============================================================================
# ASSET GROUPINGS
# ==============================================================================

ANALYSIS_GROUPS = {
    'SECTORS': ['XLF', 'XLV', 'XLE', 'XLI', 'XLY', 'XLB', 'XLU', 'XLK', 'XLP', 'XLRE'],
    'MARKET_CAP': {
        'large': 'SPY',
        'mid': 'MDY', 
        'small': 'IWM'
    },
    'STYLE': {
        'growth': 'QQQ',
        'value': 'VTV',      # Changed from VLUE (higher AUM, lower expense ratio)
        'quality': 'QUAL',
        'momentum': 'MTUM',
        'size': 'IJR',       # Small cap factor
        'min_vol': 'USMV',   # Minimum volatility
        'dividend': 'VIG'    # Dividend appreciation
    },
    'INDEXES': {
        'spy': 'SPY',
        'dia': 'DIA',
        'ndx': '^NDX',       # Nasdaq-100 index
        'iau': 'IAU',        # Gold
        'vgk': 'VGK',        # Europe
        'mchi': 'MCHI',      # China
        'tlt': 'TLT'         # 20Y Treasury
    },
    'RISK_ASSETS': {
        'equities': 'SPY',
        'bonds': 'TLT',
        'gold': 'GLD',
        'dollar': 'UUP'
    },
    'INTERNATIONAL': {
        'us': 'SPY',
        'developed': 'EFA',
        'emerging': 'EEM'
    },
    'BREADTH': {
        'cap_weighted': 'SPY',
        'equal_weighted': 'RSP'
    },
    'VOLATILITY': ['VIXY', 'VXX', 'UVXY'],
    'COMMODITIES': ['DBC', 'GLD', 'SLV', 'USO', 'CPER'],
    'BONDS': ['TLT', 'IEF', 'SHY', 'HYG', 'LQD', 'JNK']
}

SECTOR_NAMES = {
    'XLF': 'Financials', 'XLV': 'Healthcare', 'XLE': 'Energy',
    'XLI': 'Industrials', 'XLY': 'Consumer Disc', 'XLB': 'Materials',
    'XLU': 'Utilities', 'XLK': 'Technology', 'XLP': 'Consumer Staples',
    'XLRE': 'Real Estate'
}

STYLE_NAMES = {
    'growth': 'Growth (QQQ)',
    'value': 'Value (VTV)',
    'quality': 'Quality (QUAL)',
    'momentum': 'Momentum (MTUM)',
    'size': 'Size (IJR)',
    'min_vol': 'Min Volatility (USMV)',
    'dividend': 'Dividend (VIG)'
}

INDEX_NAMES = {
    'spy': 'S&P 500 (SPY)',
    'dia': 'Dow Jones (DIA)',
    'ndx': 'Nasdaq-100 (^NDX)',
    'iau': 'Gold (IAU)',
    'vgk': 'Europe (VGK)',
    'mchi': 'China (MCHI)',
    'tlt': '20Y Treasury (TLT)'
}

# ==============================================================================
# PATTERN MATCHING (copied from sector_rotation_v0_7.py)
# ==============================================================================

def calculate_momentum_features(df):
    """Calculate momentum features for pattern matching"""
    features = pd.DataFrame(index=df.index)
    
    for period, name in [(63, '3M'), (21, '1M'), (10, '10D'), (5, '5D')]:
        ret = df['close'].pct_change(period) * 100
        rolling_mean = ret.rolling(252, min_periods=60).mean()
        rolling_std = ret.rolling(252, min_periods=60).std()
        features[name] = ((ret - rolling_mean) / rolling_std.replace(0, np.nan)).clip(-2, 2)
    
    features['slope1'] = features['1M'] - features['3M']
    features['slope2'] = features['10D'] - features['1M']
    features['slope3'] = features['5D'] - features['10D']
    
    for col in ['slope1', 'slope2', 'slope3']:
        rolling_mean = features[col].rolling(252, min_periods=60).mean()
        rolling_std = features[col].rolling(252, min_periods=60).std()
        features[col] = ((features[col] - rolling_mean) / rolling_std.replace(0, np.nan)).clip(-2, 2)
    
    return features

def build_pattern_matrix(all_data):
    """Build pattern feature matrix"""
    print('\nBuilding pattern matrix...')
    all_features = {}
    
    for ticker, df in all_data.items():
        features = calculate_momentum_features(df)
        for col in features.columns:
            all_features[f'{ticker}_{col}'] = features[col]
    
    features_df = pd.DataFrame(all_features)
    print(f'Pattern matrix: {len(features_df)} days x {len(features_df.columns)} features')
    return features_df

def find_similar_days_masked(features_df, target_date, min_correlation=0.2,
                              top_n=100, min_overlap=200, history_cutoff=None):
    """Find similar days using masked correlation"""
    
    if target_date not in features_df.index:
        return []
    
    target_row = features_df.loc[target_date]
    target_available = ~target_row.isna()
    
    if target_available.sum() < min_overlap:
        return []
    
    correlations = []

    for hist_date in features_df.index:
        if hist_date >= target_date:
            continue
        if history_cutoff is not None and hist_date < history_cutoff:
            continue
        
        hist_row = features_df.loc[hist_date]
        hist_available = ~hist_row.isna()
        overlap_mask = target_available & hist_available
        overlap_count = overlap_mask.sum()
        
        if overlap_count < min_overlap:
            continue
        
        target_vals = target_row[overlap_mask].values
        hist_vals = hist_row[overlap_mask].values
        
        if len(target_vals) >= 2:
            corr = np.corrcoef(target_vals, hist_vals)[0, 1]
            if not np.isnan(corr) and corr >= min_correlation:
                correlations.append((hist_date, corr))
    
    correlations.sort(key=lambda x: x[1], reverse=True)
    return correlations[:top_n]

# ==============================================================================
# ANALYSIS FUNCTIONS
# ==============================================================================

def get_period_return(all_data, ticker, start_date, periods=5):
    """
    Get return for ticker over specified periods starting from start_date.
    
    Args:
        all_data: dict of DataFrames
        ticker: ETF ticker
        start_date: starting date
        periods: number of trading days (default 5)
    
    Returns:
        Percentage return over the period, or None if not available
    """
    if ticker not in all_data:
        return None
    
    df = all_data[ticker]
    
    # Find start date
    if start_date not in df.index:
        return None
    
    start_idx = df.index.get_loc(start_date)
    
    # Find end date (periods days later)
    if start_idx + periods >= len(df):
        return None
    
    start_price = df.iloc[start_idx]['close']
    end_price = df.iloc[start_idx + periods]['close']
    
    if pd.isna(start_price) or pd.isna(end_price):
        return None
    
    return (end_price / start_price - 1) * 100

def get_backward_return(all_data, ticker, date, periods=5):
    """Get return for the 5 days BEFORE the date"""
    if ticker not in all_data:
        return None
    
    df = all_data[ticker]
    
    if date not in df.index:
        return None
    
    date_idx = df.index.get_loc(date)
    
    # Need at least 'periods' days before
    if date_idx < periods:
        return None
    
    start_price = df.iloc[date_idx - periods]['close']
    end_price = df.iloc[date_idx]['close']
    
    if pd.isna(start_price) or pd.isna(end_price):
        return None
    
    return (end_price / start_price - 1) * 100

def analyze_sectors_backward_forward(all_data, similar_days, periods=5):
    """Analyze sector performance backward (last 5d) and forward (next 5d)"""
    sector_backward = defaultdict(list)
    sector_forward = defaultdict(list)
    
    for date, corr in similar_days:
        for sector in ANALYSIS_GROUPS['SECTORS']:
            # Backward (5d before similar day)
            back_ret = get_backward_return(all_data, sector, date, periods)
            if back_ret is not None:
                sector_backward[sector].append(back_ret)
            
            # Forward (5d after similar day)
            fwd_ret = get_period_return(all_data, sector, date, periods)
            if fwd_ret is not None:
                sector_forward[sector].append(fwd_ret)
    
    sector_summary = {}
    for sector in ANALYSIS_GROUPS['SECTORS']:
        back_rets = sector_backward[sector]
        fwd_rets = sector_forward[sector]
        
        if back_rets or fwd_rets:
            sector_summary[sector] = {
                'name': SECTOR_NAMES.get(sector, sector),
                'backward': {
                    'avg_return': round(np.mean(back_rets), 2) if back_rets else None,
                    'pct_positive': round(100 * sum(1 for r in back_rets if r > 0) / len(back_rets), 1) if back_rets else None,
                    'sample_count': len(back_rets)
                },
                'forward': {
                    'avg_return': round(np.mean(fwd_rets), 2) if fwd_rets else None,
                    'pct_positive': round(100 * sum(1 for r in fwd_rets if r > 0) / len(fwd_rets), 1) if fwd_rets else None,
                    'sample_count': len(fwd_rets)
                }
            }
    
    # Sort by forward return
    sorted_sectors = sorted(
        [(k, v) for k, v in sector_summary.items() if v['forward']['avg_return'] is not None],
        key=lambda x: x[1]['forward']['avg_return'],
        reverse=True
    )
    
    return dict(sorted_sectors)

def analyze_market_cap_backward_forward(all_data, similar_days, periods=5):
    """Analyze market cap performance backward and forward"""
    cap_backward = defaultdict(list)
    cap_forward = defaultdict(list)
    
    for date, corr in similar_days:
        for cap_type, ticker in ANALYSIS_GROUPS['MARKET_CAP'].items():
            back_ret = get_backward_return(all_data, ticker, date, periods)
            if back_ret is not None:
                cap_backward[cap_type].append(back_ret)
            
            fwd_ret = get_period_return(all_data, ticker, date, periods)
            if fwd_ret is not None:
                cap_forward[cap_type].append(fwd_ret)
    
    cap_summary = {}
    for cap_type in ANALYSIS_GROUPS['MARKET_CAP'].keys():
        back_rets = cap_backward[cap_type]
        fwd_rets = cap_forward[cap_type]
        
        if back_rets or fwd_rets:
            cap_summary[cap_type] = {
                'backward': {
                    'avg_return': round(np.mean(back_rets), 2) if back_rets else None,
                    'pct_positive': round(100 * sum(1 for r in back_rets if r > 0) / len(back_rets), 1) if back_rets else None
                },
                'forward': {
                    'avg_return': round(np.mean(fwd_rets), 2) if fwd_rets else None,
                    'pct_positive': round(100 * sum(1 for r in fwd_rets if r > 0) / len(fwd_rets), 1) if fwd_rets else None
                }
            }
    
    return cap_summary

def analyze_risk_regime_backward_forward(all_data, similar_days, periods=5):
    """Analyze risk regime backward and forward"""
    risk_backward = defaultdict(list)
    risk_forward = defaultdict(list)
    
    for date, corr in similar_days:
        # Backward
        back_returns = {}
        for asset_type, ticker in ANALYSIS_GROUPS['RISK_ASSETS'].items():
            ret = get_backward_return(all_data, ticker, date, periods)
            if ret is not None:
                back_returns[asset_type] = ret
        
        if len(back_returns) >= 3:
            risk_score = back_returns.get('equities', 0) - back_returns.get('bonds', 0) - back_returns.get('gold', 0) - back_returns.get('dollar', 0)
            risk_backward['scores'].append(risk_score)
            risk_backward['equities'].append(back_returns.get('equities', 0))
            risk_backward['bonds'].append(back_returns.get('bonds', 0))
        
        # Forward
        fwd_returns = {}
        for asset_type, ticker in ANALYSIS_GROUPS['RISK_ASSETS'].items():
            ret = get_period_return(all_data, ticker, date, periods)
            if ret is not None:
                fwd_returns[asset_type] = ret
        
        if len(fwd_returns) >= 3:
            risk_score = fwd_returns.get('equities', 0) - fwd_returns.get('bonds', 0) - fwd_returns.get('gold', 0) - fwd_returns.get('dollar', 0)
            risk_forward['scores'].append(risk_score)
            risk_forward['equities'].append(fwd_returns.get('equities', 0))
            risk_forward['bonds'].append(fwd_returns.get('bonds', 0))
    
    if not risk_backward['scores'] or not risk_forward['scores']:
        return None
    
    back_avg_score = np.mean(risk_backward['scores'])
    fwd_avg_score = np.mean(risk_forward['scores'])
    
    back_regime = 'RISK-ON' if back_avg_score > 0.5 else 'RISK-OFF' if back_avg_score < -0.5 else 'MIXED'
    fwd_regime = 'RISK-ON' if fwd_avg_score > 0.5 else 'RISK-OFF' if fwd_avg_score < -0.5 else 'MIXED'
    
    return {
        'backward': {
            'regime': back_regime,
            'avg_risk_score': round(back_avg_score, 2),
            'equities_avg': round(np.mean(risk_backward['equities']), 2),
            'bonds_avg': round(np.mean(risk_backward['bonds']), 2)
        },
        'forward': {
            'regime': fwd_regime,
            'avg_risk_score': round(fwd_avg_score, 2),
            'equities_avg': round(np.mean(risk_forward['equities']), 2),
            'bonds_avg': round(np.mean(risk_forward['bonds']), 2)
        },
        'continuation': back_regime == fwd_regime
    }

def analyze_style_backward_forward(all_data, similar_days, periods=5):
    """Analyze style factors backward and forward"""
    style_backward = defaultdict(list)
    style_forward = defaultdict(list)
    
    for date, corr in similar_days:
        for style_type, ticker in ANALYSIS_GROUPS['STYLE'].items():
            back_ret = get_backward_return(all_data, ticker, date, periods)
            if back_ret is not None:
                style_backward[style_type].append(back_ret)
            
            fwd_ret = get_period_return(all_data, ticker, date, periods)
            if fwd_ret is not None:
                style_forward[style_type].append(fwd_ret)
    
    style_summary = {}
    for style_type in ANALYSIS_GROUPS['STYLE'].keys():
        back_rets = style_backward[style_type]
        fwd_rets = style_forward[style_type]
        
        if back_rets or fwd_rets:
            style_summary[style_type] = {
                'name': STYLE_NAMES.get(style_type, style_type.upper()),
                'backward': {
                    'avg_return': round(np.mean(back_rets), 2) if back_rets else None,
                    'pct_positive': round(100 * sum(1 for r in back_rets if r > 0) / len(back_rets), 1) if back_rets else None
                },
                'forward': {
                    'avg_return': round(np.mean(fwd_rets), 2) if fwd_rets else None,
                    'pct_positive': round(100 * sum(1 for r in fwd_rets if r > 0) / len(fwd_rets), 1) if fwd_rets else None
                }
            }
    
    return style_summary

def analyze_indexes_backward_forward(all_data, similar_days, periods=5):
    """Analyze major indexes backward and forward"""
    index_backward = defaultdict(list)
    index_forward = defaultdict(list)
    
    for date, corr in similar_days:
        for index_type, ticker in ANALYSIS_GROUPS['INDEXES'].items():
            back_ret = get_backward_return(all_data, ticker, date, periods)
            if back_ret is not None:
                index_backward[index_type].append(back_ret)
            
            fwd_ret = get_period_return(all_data, ticker, date, periods)
            if fwd_ret is not None:
                index_forward[index_type].append(fwd_ret)
    
    index_summary = {}
    for index_type in ANALYSIS_GROUPS['INDEXES'].keys():
        back_rets = index_backward[index_type]
        fwd_rets = index_forward[index_type]
        
        if back_rets or fwd_rets:
            index_summary[index_type] = {
                'name': INDEX_NAMES.get(index_type, index_type.upper()),
                'backward': {
                    'avg_return': round(np.mean(back_rets), 2) if back_rets else None,
                    'pct_positive': round(100 * sum(1 for r in back_rets if r > 0) / len(back_rets), 1) if back_rets else None
                },
                'forward': {
                    'avg_return': round(np.mean(fwd_rets), 2) if fwd_rets else None,
                    'pct_positive': round(100 * sum(1 for r in fwd_rets if r > 0) / len(fwd_rets), 1) if fwd_rets else None
                }
            }
    
    return index_summary

def analyze_continuation_patterns(all_data, similar_days, periods=5):
    """Analyze which patterns continue vs reverse"""
    
    patterns = {
        'sector_continuation': [],
        'cap_continuation': [],
        'style_continuation': []
    }
    
    for date, corr in similar_days:
        # Check if sector leaders continue
        sector_back = {}
        sector_fwd = {}
        for sector in ANALYSIS_GROUPS['SECTORS']:
            back_ret = get_backward_return(all_data, sector, date, periods)
            fwd_ret = get_period_return(all_data, sector, date, periods)
            if back_ret is not None:
                sector_back[sector] = back_ret
            if fwd_ret is not None:
                sector_fwd[sector] = fwd_ret
        
        if sector_back and sector_fwd:
            # Find top sector backward
            top_back = max(sector_back.items(), key=lambda x: x[1])
            # Check if it was also top forward
            if top_back[0] in sector_fwd:
                continued = sector_fwd[top_back[0]] > np.mean(list(sector_fwd.values()))
                patterns['sector_continuation'].append(continued)
    
    continuation_summary = {
        'sector_leaders_continued': round(100 * sum(patterns['sector_continuation']) / len(patterns['sector_continuation']), 1) if patterns['sector_continuation'] else None,
        'sample_count': len(patterns['sector_continuation'])
    }
    
    return continuation_summary

# ==============================================================================
# MAIN ANALYSIS
# ==============================================================================

def run_analysis():
    """Run complete similar days analysis across 25Y / 5Y / 1Y windows."""

    print('='*80)
    print('Similar Days Analyzer v1.13')
    print('='*80)
    print()

    # Load data
    print('Loading ETF data from cache...')
    all_data = load_etf_data()
    print(f'Loaded: {len(all_data)} ETFs')

    # Build pattern matrix
    features_df = build_pattern_matrix(all_data)
    latest_date = features_df.index.max()

    # History cutoffs
    cutoffs = {
        '25Y': None,
        '5Y':  latest_date - timedelta(days=5 * 365),
        '1Y':  latest_date - timedelta(days=365),
    }

    # Find similar days separately per window (each gets its own top 100)
    similar_days_by_window = {}
    for window, cutoff in cutoffs.items():
        label = f'since {cutoff.strftime("%Y-%m-%d")}' if cutoff else 'all history'
        print(f'\nFinding similar days ({window} — {label})...')
        matches = find_similar_days_masked(
            features_df, latest_date,
            min_overlap=200, top_n=100,
            history_cutoff=cutoff
        )
        similar_days_by_window[window] = matches
        print(f'  Found {len(matches)} similar days')
        if matches:
            print(f'  Top 5:')
            for d, c in matches[:5]:
                print(f'    {d.strftime("%Y-%m-%d")}: {c:.3f}')

    # Run all analyses per window
    print('\n' + '='*80)
    print('RUNNING ANALYSIS FOR ALL WINDOWS')
    print('='*80)

    results_by_window = {}
    for window, similar_days in similar_days_by_window.items():
        if len(similar_days) < 3:
            print(f'WARNING: {window} has only {len(similar_days)} matches — skipping')
            results_by_window[window] = None
            continue
        print(f'\nAnalyzing {window} ({len(similar_days)} similar days)...')
        results_by_window[window] = {
            'similar_days_count': len(similar_days),
            'similar_days': [
                {'date': d.strftime('%Y-%m-%d'), 'correlation': round(c, 4)}
                for d, c in similar_days
            ],
            'sectors':     analyze_sectors_backward_forward(all_data, similar_days),
            'indexes':     analyze_indexes_backward_forward(all_data, similar_days),
            'styles':      analyze_style_backward_forward(all_data, similar_days),
            'market_cap':  analyze_market_cap_backward_forward(all_data, similar_days),
            'risk_regime': analyze_risk_regime_backward_forward(all_data, similar_days),
            'continuation': analyze_continuation_patterns(all_data, similar_days),
        }

    output = {
        'generated_at':    datetime.now().isoformat(),
        'target_date':     latest_date.strftime('%Y-%m-%d'),
        'windows':         results_by_window,
    }

    return output, latest_date


# ==============================================================================
# HTML BUILDER
# ==============================================================================

EXTRA_CSS = """
/* Similar days table overrides */
.sim-table table {
    border-collapse: collapse;
    width: 100%;
    font-size: 0.8em;
    table-layout: fixed;   /* fixed layout so col widths are respected */
}
/* Col 0: Name */
.sim-table table colgroup col.col-name  { width: 140px; }
/* Cols 1-9: Before / After / Win% per window (3 cols x 3 windows) */
.sim-table table colgroup col.col-before { width: 90px; }
.sim-table table colgroup col.col-after  { width: 80px; }
.sim-table table colgroup col.col-winpct { width: 72px; }
.sim-table thead th {
    padding: 7px 6px;
    font-size: 0.72em;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    cursor: pointer;
    user-select: none;
    border-bottom: 2px solid #e2e4e8;
    white-space: nowrap;
    text-align: center;
    overflow: hidden;
    text-overflow: ellipsis;
}
.sim-table thead th.col-name { text-align: left; }
.sim-table tbody td {
    padding: 6px 6px;
    border-bottom: 1px solid #f0f0f0;
    vertical-align: middle;
    text-align: center;
    font-family: 'IBM Plex Mono', monospace;
    overflow: hidden;
    text-overflow: ellipsis;
}
.sim-table tbody td.col-name {
    text-align: left;
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.9em;
    color: #333;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.sim-table tbody tr:hover { outline: 2px solid #4f46e5; }

/* Group header row */
.grp-row th { padding: 5px 4px; font-size: 0.78em; font-weight: 700; letter-spacing: 0.05em; }
.grp-blank    { background: #f8f9fb; color: transparent; border-bottom: 1px solid #e2e4e8; }
.grp-before   { background: #1e293b; color: #94a3b8; }
.grp-25y      { background: #14532d; color: #86efac; }
.grp-5y       { background: #1e3a5f; color: #93c5fd; }
.grp-1y       { background: #3b1f5e; color: #c4b5fd; }

/* Sort arrows */
th.sort-asc::after  { content: ' ▲'; font-size: 0.7em; }
th.sort-desc::after { content: ' ▼'; font-size: 0.7em; }

/* pred sub-text (win%) */
.pred-pct { display: block; font-size: 0.78em; color: #333; opacity: 0.75; }
.cell-miss { color: #ccc !important; background: #fafafa !important; }

/* Similar dates panels */
.dates-panels {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin-bottom: 22px;
}
.dates-panel {
    background: #fff;
    border: 1px solid #e2e4e8;
    border-radius: 8px;
    overflow: hidden;
}
.dates-panel-header {
    padding: 10px 14px;
    font-size: 0.82em;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.dates-panel-header .count {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.1em;
    font-weight: 700;
}
.dates-panel-body {
    padding: 10px 14px;
    max-height: 280px;
    overflow-y: auto;
}
.date-row {
    display: flex;
    justify-content: space-between;
    font-size: 0.82em;
    padding: 4px 0;
    border-bottom: 1px solid #f5f5f5;
    font-family: 'IBM Plex Mono', monospace;
}
.date-row:last-child { border-bottom: none; }
.date-corr { color: #888; }

/* ── Mobile responsive ── */
@media (max-width: 768px) {
    .sim-table { overflow-x: auto; -webkit-overflow-scrolling: touch; }
    .sim-table table { font-size: 0.72em; }
    .sim-table thead th { padding: 5px 4px; font-size: 0.68em; }
    .sim-table tbody td { padding: 4px 4px; }
    .sim-table tbody td.col-name { font-size: 0.82em; }
    .dates-panels { grid-template-columns: 1fr; gap: 10px; }
    .dates-panel-header { padding: 8px 12px; font-size: 0.78em; }
    .dates-panel-body { padding: 8px 12px; max-height: 200px; }
    .date-row { font-size: 0.78em; }
}
"""

RENDER_JS_TEMPLATE = """
(function() {{
    const DATA = {data_json};

    // ---- Gradient: red(0) -> orange -> yellow -> green(1) ----
    function getGradientColor(norm01) {{
        const c = Math.max(0, Math.min(1, norm01));
        let r, g, b;
        if (c < 0.25) {{
            const t = c * 4;
            r = Math.round(211+(255-211)*t); g = Math.round(47+(152-47)*t); b = Math.round(47*(1-t));
        }} else if (c < 0.5) {{
            const t = (c-0.25)*4;
            r = 255; g = Math.round(152+(235-152)*t); b = Math.round(59*t);
        }} else if (c < 0.75) {{
            const t = (c-0.5)*4;
            r = Math.round(255+(139-255)*t); g = Math.round(235+(195-235)*t); b = Math.round(59+(74-59)*t);
        }} else {{
            const t = (c-0.75)*4;
            r = Math.round(139*(1-t)); g = Math.round(195+(200-195)*t); b = Math.round(74+(83-74)*t);
        }}
        return `rgb(${{r}},${{g}},${{b}})`;
    }}

    function textColor(bgRgb) {{
        const m = bgRgb.match(/rgb[(]([0-9]+),([0-9]+),([0-9]+)[)]/);
        if (!m) return '#000';
        return (0.299*m[1] + 0.587*m[2] + 0.114*m[3]) > 140 ? '#000' : '#fff';
    }}

    // Per-column 5th-95th percentile ranges across all windows for a given tableKey
    // All analysis dicts are {{key: {{name?, backward:{{avg_return,...}}, forward:{{avg_return,...}}}}}}
    function calcRanges(tableKey) {{
        const cols = {{
            before_25Y:[], before_5Y:[], before_1Y:[],
            after_25Y:[], after_5Y:[], after_1Y:[]
        }};
        ['25Y','5Y','1Y'].forEach(win => {{
            const wdata = DATA.windows[win];
            if (!wdata) return;
            const tbl = wdata[tableKey];
            if (!tbl) return;
            Object.values(tbl).forEach(item => {{
                const bv = item?.backward?.avg_return;
                const fv = item?.forward?.avg_return;
                if (bv != null && !isNaN(bv)) cols['before_'+win].push(bv);
                if (fv != null && !isNaN(fv)) cols['after_'+win].push(fv);
            }});
        }});
        const ranges = {{}};
        for (const [k, vals] of Object.entries(cols)) {{
            if (!vals.length) continue;
            const s = [...vals].sort((a,b) => a-b);
            ranges[k] = {{
                min: s[Math.floor(s.length*0.05)],
                max: s[Math.floor(s.length*0.95)]
            }};
        }}
        return ranges;
    }}

    function normalize(val, ranges, key) {{
        if (val == null || isNaN(val)) return null;
        const r = ranges[key];
        if (!r || r.max === r.min) return 0.5;
        return Math.max(0, Math.min(1, (val - r.min) / (r.max - r.min)));
    }}

    function bgCell(val, ranges, key) {{
        const n = normalize(val, ranges, key);
        return n == null ? null : getGradientColor(n);
    }}

    // Get all unique row keys across all windows for a tableKey
    function getRows(tableKey) {{
        const seen = new Set();
        const rows = [];
        // Use 25Y as primary ordering source, then fill from others
        ['25Y','5Y','1Y'].forEach(win => {{
            const wdata = DATA.windows[win];
            if (!wdata) return;
            const tbl = wdata[tableKey];
            if (!tbl) return;
            Object.entries(tbl).forEach(([key, item]) => {{
                if (!seen.has(key)) {{
                    seen.add(key);
                    rows.push({{ key, name: item.name || key }});
                }}
            }});
        }});
        return rows;
    }}

    // Get one item from a table by row key
    function getItem(tableKey, win, rowKey) {{
        return DATA.windows[win]?.[tableKey]?.[rowKey] ?? null;
    }}

    // Sort state per container
    const sortState = {{}};

    function buildTable(containerId, tableKey) {{
        const ss = sortState[containerId] || {{ col: 'after_25Y', dir: 'desc' }};
        sortState[containerId] = ss;

        const ranges = calcRanges(tableKey);
        let rows = getRows(tableKey);

        // Sort
        rows.sort((a, b) => {{
            if (ss.col === 'name') {{
                return ss.dir === 'asc' ? a.name.localeCompare(b.name) : b.name.localeCompare(a.name);
            }}
            // col format: 'before_25Y', 'after_1Y', etc
            const isBack = ss.col.startsWith('before_');
            const win    = ss.col.replace('before_','').replace('after_','');
            const field  = isBack ? 'backward' : 'forward';
            const va = getItem(tableKey, win, a.key)?.[field]?.avg_return ?? -999;
            const vb = getItem(tableKey, win, b.key)?.[field]?.avg_return ?? -999;
            return ss.dir === 'asc' ? va - vb : vb - va;
        }});

        const arrow  = col => ss.col === col ? (ss.dir === 'asc' ? ' ▲' : ' ▼') : '';
        const thSort = (col, label, tip) =>
            `<th style="cursor:pointer;white-space:nowrap;" title="${{tip}}" onclick="window._simSort('${{containerId}}','${{tableKey}}','${{col}}')">${{label}}${{arrow(col)}}</th>`;

        const grpCls = {{ '25Y':'grp-25y', '5Y':'grp-5y', '1Y':'grp-1y' }};
        const wins = ['25Y','5Y','1Y'];

        // Fixed column widths via colgroup - same across all tables
        let html = '<table><colgroup>';
        html += '<col class="col-name">';
        wins.forEach(() => {{
            html += '<col class="col-before">';
            html += '<col class="col-after">';
            html += '<col class="col-winpct">';
        }});
        html += '</colgroup><thead>';
        // Group header row
        html += '<tr class="grp-row">';
        html += '<th class="grp-blank col-name" title="Asset or factor name"></th>';
        wins.forEach(w => html += `<th colspan="3" class="${{grpCls[w]}}" title="Pattern matches from the ${{w}} lookback window">${{w}} Window</th>`);
        html += '</tr>';
        // Column header row
        html += '<tr>';
        html += thSort('name', 'Name', 'Asset or factor name (click to sort)');
        wins.forEach(w => {{
            html += thSort('before_'+w, 'Before 5d', 'Avg return 5 trading days BEFORE match date in '+w+' window (click to sort)');
            html += thSort('after_'+w,  'After \u2192', 'Avg return 5 trading days AFTER match date in '+w+' window (click to sort)');
            html += '<th style="cursor:default;white-space:nowrap;" title="Win rate (% positive) and sample count for this window">Win% / n</th>';
        }});
        html += '</tr></thead><tbody>';

        rows.forEach(row => {{
            html += '<tr>';
            html += `<td class="col-name">${{row.name}}</td>`;
            wins.forEach(w => {{
                const item    = getItem(tableKey, w, row.key);
                const backAvg = item?.backward?.avg_return ?? null;
                const backPct = item?.backward?.pct_positive ?? null;
                const fwdAvg  = item?.forward?.avg_return ?? null;
                const fwdPct  = item?.forward?.pct_positive ?? null;
                const fwdN    = item?.forward?.sample_count ?? item?.backward?.sample_count ?? null;

                // Before cell
                if (backAvg != null) {{
                    const bg = bgCell(backAvg, ranges, 'before_'+w);
                    const fg = textColor(bg);
                    const sign = backAvg >= 0 ? '+' : '';
                    html += `<td style="background:${{bg}};color:${{fg}};">${{sign}}${{backAvg.toFixed(1)}}%`;
                    if (backPct != null) html += `<span class="pred-pct">${{backPct.toFixed(0)}}%</span>`;
                    html += '</td>';
                }} else {{ html += '<td class="cell-miss">—</td>'; }}

                // After cell
                if (fwdAvg != null) {{
                    const bg = bgCell(fwdAvg, ranges, 'after_'+w);
                    const fg = textColor(bg);
                    const sign = fwdAvg >= 0 ? '+' : '';
                    html += `<td style="background:${{bg}};color:${{fg}};">${{sign}}${{fwdAvg.toFixed(1)}}%`;
                    if (fwdPct != null) html += `<span class="pred-pct">${{fwdPct.toFixed(0)}}%</span>`;
                    html += '</td>';
                }} else {{ html += '<td class="cell-miss">—</td>'; }}

                // Win% / n cell
                if (fwdPct != null) {{
                    html += `<td style="color:#555;font-size:0.85em;">${{fwdPct.toFixed(0)}}%`;
                    if (fwdN != null) html += ` <span style="color:#aaa;">n=${{fwdN}}</span>`;
                    html += '</td>';
                }} else {{ html += '<td class="cell-miss">—</td>'; }}
            }});
            html += '</tr>';
        }});

        html += '</tbody></table>';
        document.getElementById(containerId).innerHTML = html;
    }}

    window._simSort = function(containerId, tableKey, col) {{
        const ss = sortState[containerId] || {{ col: 'after_25Y', dir: 'desc' }};
        sortState[containerId] = {{
            col,
            dir: (ss.col === col && ss.dir === 'desc') ? 'asc' : 'desc'
        }};
        buildTable(containerId, tableKey);
    }};

    // Risk regime table — dedicated builder with coloring
    function buildRiskTable() {{
        // Collect numeric values for gradient ranging
        const scoreVals = [], eqVals = [], bondVals = [], contVals = [];
        ['25Y','5Y','1Y'].forEach(w => {{
            const rr = DATA.windows[w]?.risk_regime;
            if (!rr) return;
            if (rr.backward?.avg_risk_score != null) scoreVals.push(rr.backward.avg_risk_score);
            if (rr.forward?.avg_risk_score  != null) scoreVals.push(rr.forward.avg_risk_score);
            if (rr.backward?.equities_avg   != null) eqVals.push(rr.backward.equities_avg);
            if (rr.forward?.equities_avg    != null) eqVals.push(rr.forward.equities_avg);
            if (rr.backward?.bonds_avg      != null) bondVals.push(rr.backward.bonds_avg);
            if (rr.forward?.bonds_avg       != null) bondVals.push(rr.forward.bonds_avg);
            const cont = DATA.windows[w]?.continuation?.sector_leaders_continued;
            if (cont != null) contVals.push(cont);
        }});

        function makeRange(vals) {{
            if (!vals.length) return null;
            const s = [...vals].sort((a,b) => a-b);
            return {{ min: s[0], max: s[s.length-1] }};
        }}
        const rr = {{ score: makeRange(scoreVals), eq: makeRange(eqVals), bond: makeRange(bondVals), cont: makeRange(contVals) }};

        function numTd(val, range, fmt) {{
            if (val == null) return '<td class="cell-miss">—</td>';
            if (!range || range.max === range.min) {{
                return `<td>${{fmt(val)}}</td>`;
            }}
            const n = Math.max(0, Math.min(1, (val - range.min) / (range.max - range.min)));
            const bg = getGradientColor(n);
            const fg = textColor(bg);
            return `<td style="background:${{bg}};color:${{fg}};">${{fmt(val)}}</td>`;
        }}

        function regimeTd(regime) {{
            if (!regime) return '<td class="cell-miss">—</td>';
            const colors = {{ 'RISK-ON':'#16a34a', 'RISK-OFF':'#dc2626', 'MIXED':'#d97706' }};
            const bg = colors[regime] || '#888';
            return `<td style="background:${{bg}};color:#fff;font-weight:700;">${{regime}}</td>`;
        }}

        const grpCls = {{ '25Y':'grp-25y', '5Y':'grp-5y', '1Y':'grp-1y' }};
        let html = '<table><thead><tr class="grp-row">';
        html += '<th class="grp-blank col-name" title="Risk regime metric name">Metric</th>';
        ['25Y','5Y','1Y'].forEach(w => html += `<th colspan="2" class="${{grpCls[w]}}" title="Risk regime data from the ${{w}} lookback window">${{w}} Window</th>`);
        html += '</tr><tr><th class="col-name" title="Risk regime metric name">Metric</th>';
        ['25Y','5Y','1Y'].forEach(() => {{ html += '<th title="Average value 5 trading days BEFORE similar-day matches">Before</th><th title="Average value 5 trading days AFTER similar-day matches">After \u2192</th>'; }});
        html += '</tr></thead><tbody>';

        // Regime
        html += '<tr><td class="col-name">Regime</td>';
        ['25Y','5Y','1Y'].forEach(w => {{
            const r = DATA.windows[w]?.risk_regime;
            html += regimeTd(r?.backward?.regime) + regimeTd(r?.forward?.regime);
        }});
        html += '</tr>';

        // Risk Score
        html += '<tr><td class="col-name">Risk Score</td>';
        ['25Y','5Y','1Y'].forEach(w => {{
            const r = DATA.windows[w]?.risk_regime;
            html += numTd(r?.backward?.avg_risk_score, rr.score, v => (v>=0?'+':'')+v.toFixed(2));
            html += numTd(r?.forward?.avg_risk_score,  rr.score, v => (v>=0?'+':'')+v.toFixed(2));
        }});
        html += '</tr>';

        // Equities Avg
        html += '<tr><td class="col-name">Equities Avg</td>';
        ['25Y','5Y','1Y'].forEach(w => {{
            const r = DATA.windows[w]?.risk_regime;
            html += numTd(r?.backward?.equities_avg, rr.eq, v => (v>=0?'+':'')+v.toFixed(2)+'%');
            html += numTd(r?.forward?.equities_avg,  rr.eq, v => (v>=0?'+':'')+v.toFixed(2)+'%');
        }});
        html += '</tr>';

        // Bonds Avg
        html += '<tr><td class="col-name">Bonds Avg</td>';
        ['25Y','5Y','1Y'].forEach(w => {{
            const r = DATA.windows[w]?.risk_regime;
            html += numTd(r?.backward?.bonds_avg, rr.bond, v => (v>=0?'+':'')+v.toFixed(2)+'%');
            html += numTd(r?.forward?.bonds_avg,  rr.bond, v => (v>=0?'+':'')+v.toFixed(2)+'%');
        }});
        html += '</tr>';

        // Sector Continuation — spans 2 cols per window
        html += '<tr><td class="col-name">Sector Continuation</td>';
        ['25Y','5Y','1Y'].forEach(w => {{
            const cont = DATA.windows[w]?.continuation;
            const pct  = cont?.sector_leaders_continued;
            const n    = cont?.sample_count;
            if (pct != null) {{
                const nr = Math.max(0, Math.min(1, (pct - 40) / 30));  // 40-70% range
                const bg = getGradientColor(nr);
                const fg = textColor(bg);
                html += `<td colspan="2" style="background:${{bg}};color:${{fg}};font-weight:600;">${{pct.toFixed(1)}}% <span style="opacity:0.7;">n=${{n}}</span></td>`;
            }} else {{
                html += '<td colspan="2" class="cell-miss">—</td>';
            }}
        }});
        html += '</tr>';

        html += '</tbody></table>';
        document.getElementById('tbl-risk').innerHTML = html;
    }}

    function renderAll() {{
        buildRiskTable();
        buildTable('tbl-indexes',    'indexes');
        buildTable('tbl-market-cap', 'market_cap');
        buildTable('tbl-sectors',    'sectors');
        buildTable('tbl-styles',     'styles');
    }}

    renderAll();
}})();
"""




def build_body_html(output, latest_date, writer):
    """Build full body HTML for DashboardWriter.write()."""
    import json as _json

    date_str = latest_date.strftime('%Y-%m-%d')
    windows  = output['windows']

    counts = {
        w: (windows[w]['similar_days_count'] if windows[w] else 0)
        for w in ['25Y', '5Y', '1Y']
    }

    parts = []

    # --- Stat bar ---
    parts.append(writer.stat_bar([
        ('Target Date', date_str,          'neutral'),
        ('25Y Matches', str(counts['25Y']), 'neutral'),
        ('5Y Matches',  str(counts['5Y']),  'neutral'),
        ('1Y Matches',  str(counts['1Y']),  'neutral'),
    ]))

    # --- Page header ---
    parts.append(writer.build_header('Pattern-matched similar days: before vs after analysis'))

    # --- Similar dates panels ---
    panel_styles = {
        '25Y': ('background:#14532d;color:#86efac;', '#14532d'),
        '5Y':  ('background:#1e3a5f;color:#93c5fd;', '#1e3a5f'),
        '1Y':  ('background:#3b1f5e;color:#c4b5fd;', '#3b1f5e'),
    }
    panels_html = '<div class="dates-panels">'
    for w in ['25Y', '5Y', '1Y']:
        hdr_style, border_color = panel_styles[w]
        wdata = windows.get(w)
        n = wdata['similar_days_count'] if wdata else 0
        panels_html += (
            f'<div class="dates-panel" style="border-top:4px solid {border_color};">'
            f'<div class="dates-panel-header" style="{hdr_style}">'
            f'<span>{w} Similar Days</span>'
            f'<span class="count">{n}</span>'
            f'</div>'
            f'<div class="dates-panel-body">'
        )
        if wdata and wdata['similar_days']:
            for entry in wdata['similar_days']:
                d = entry['date']
                c = entry['correlation']
                panels_html += (
                    f'<div class="date-row">'
                    f'<span>{d}</span>'
                    f'<span class="date-corr">{c:.3f}</span>'
                    f'</div>'
                )
        else:
            panels_html += '<div style="color:#aaa;font-size:0.85em;">No matches</div>'
        panels_html += '</div></div>'
    panels_html += '</div>'
    parts.append(panels_html)

    # --- Analysis tables (ordered: risk, indexes, market cap, sectors, styles) ---
    def wrap_table(title, div_id):
        return (
            '<div class="table-section" style="margin-bottom:22px;">'
            f'<div class="table-section-header"><h2>{title}</h2>'
            '<span style="font-size:0.85em;color:#888;">Before 5d = what happened leading into similar days'
            ' &nbsp;|&nbsp; After \u2192 = what followed &nbsp;|&nbsp; Click column headers to sort</span>'
            '</div>'
            f'<div class="sim-table" style="overflow-x:auto;padding:0;"><div id="{div_id}"></div></div>'
            '</div>'
        )

    parts.append(wrap_table('Risk Regime &amp; Continuation', 'tbl-risk'))
    parts.append(wrap_table('Major Indexes \u2014 Before vs After', 'tbl-indexes'))
    parts.append(wrap_table('Market Cap \u2014 Before vs After', 'tbl-market-cap'))
    parts.append(wrap_table('Sectors \u2014 Before vs After', 'tbl-sectors'))
    parts.append(wrap_table('Style Factors \u2014 Before vs After', 'tbl-styles'))

    # --- LLM Research Prompt ---
    # Collect top 25 dates per window
    llm_prompt = _build_llm_prompt(windows)
    prompt_html = (
        '<div class="table-section" style="margin-bottom:22px;">'
        '<div class="table-section-header"><h2>LLM Research Prompt</h2>'
        '<span style="font-size:0.85em;color:#888;">Copy into ChatGPT / Claude / Perplexity to research what was happening on these similar days</span>'
        '</div>'
        '<div style="padding:16px;">'
        '<div style="font-size:0.82em;color:#555;margin-bottom:10px;">'
        'Top 25 similar days per window. Paste the prompt below into an LLM to get a structured analysis of market conditions on those dates.'
        '</div>'
        '<button onclick="navigator.clipboard.writeText(document.getElementById(\'llm-prompt-text\').innerText)" '
        'style="margin-bottom:12px;padding:6px 16px;background:#4f46e5;color:#fff;border:none;border-radius:6px;cursor:pointer;font-size:0.85em;">'
        'Copy Prompt</button>'
        f'<pre id="llm-prompt-text" style="background:#f8f9fb;border:1px solid #e2e4e8;border-radius:6px;'
        f'padding:14px;font-size:0.78em;font-family:\'IBM Plex Mono\',monospace;white-space:pre-wrap;'
        f'overflow-x:auto;max-height:400px;overflow-y:auto;">{llm_prompt}</pre>'
        '</div></div>'
    )
    parts.append(prompt_html)

    parts.append(writer.footer())

    # Embed data + render JS
    data_json = _json.dumps(output, ensure_ascii=False, default=str)
    render_js = RENDER_JS_TEMPLATE.format(data_json=data_json)
    writer._sim_extra_js = render_js

    return '\n'.join(parts)


def _build_llm_prompt(windows):
    """Build a research prompt for LLM analysis of top similar days."""
    lines = []
    lines.append("You are a financial market historian and analyst.")
    lines.append("")
    lines.append("I am running a quantitative pattern-matching system that identifies historical market days")
    lines.append("with similar momentum profiles to today. Below are the top 25 most similar days for each")
    lines.append("lookback window (25Y = all history, 5Y = last 5 years, 1Y = last year).")
    lines.append("")
    lines.append("For each window's dates, please analyze what was happening in the market on and around")
    lines.append("those dates across these 10 dimensions:")
    lines.append("")
    lines.append("1. MACRO REGIME — What was the broad economic backdrop? (expansion/contraction, Fed stance)")
    lines.append("2. VOLATILITY — Was the VIX elevated or suppressed? Any fear/complacency signals?")
    lines.append("3. SECTOR LEADERSHIP — Which sectors were leading or lagging?")
    lines.append("4. EARNINGS CYCLE — Where were we in the earnings calendar? Any major beats/misses?")
    lines.append("5. FED & RATES — What was the Fed doing? Direction of yields?")
    lines.append("6. GEOPOLITICAL — Any major geopolitical events or policy shifts?")
    lines.append("7. SENTIMENT — What was investor sentiment? (retail flows, positioning, put/call)")
    lines.append("8. CREDIT CONDITIONS — Was credit tight or loose? HY spreads direction?")
    lines.append("9. DOLLAR & COMMODITIES — Strength/weakness in USD, gold, oil?")
    lines.append("10. WHAT HAPPENED NEXT — What did the market do in the 5-10 days after these dates?")
    lines.append("")
    lines.append("After analyzing each window separately, identify:")
    lines.append("- Common themes across all three windows (what do they share?)")
    lines.append("- Key differences (what makes the 25Y matches different from the 1Y matches?)")
    lines.append("- Your best synthesis: given all three windows, what does the pattern suggest about")
    lines.append("  likely market behavior over the next 5-10 trading days?")
    lines.append("")
    lines.append("=" * 60)

    for w in ['25Y', '5Y', '1Y']:
        wdata = windows.get(w)
        lines.append("")
        lines.append(f"--- {w} WINDOW (top 25 similar days) ---")
        if wdata and wdata.get('similar_days'):
            top25 = wdata['similar_days'][:25]
            for entry in top25:
                lines.append(f"  {entry['date']}  (correlation: {entry['correlation']:.3f})")
        else:
            lines.append("  No data available")

    lines.append("")
    lines.append("=" * 60)
    lines.append("Please structure your response with clear sections for each window,")
    lines.append("followed by the cross-window synthesis.")

    return '\n'.join(lines)


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == '__main__':
    try:
        from dashboard_writer import DashboardWriter
    except ImportError:
        print('ERROR: dashboard_writer.py not found in current directory')
        sys.exit(1)

    output, latest_date = run_analysis()

    writer = DashboardWriter('similar-days', 'Similar Days Analyzer')
    body   = build_body_html(output, latest_date, writer)
    extra_js = getattr(writer, '_sim_extra_js', '')
    writer.write(body, extra_css=EXTRA_CSS, extra_js=extra_js)

    # Write CSV - similar dates per window
    csv_path = os.path.join(_SCRIPT_DIR, 'similar_days_data_{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M')))
    csv_rows = []
    windows = output.get('windows', {})
    for window_key in ['25Y', '5Y', '1Y']:
        wdata = windows.get(window_key)
        if wdata is None:
            continue
        for entry in wdata.get('similar_days', []):
            csv_rows.append({
                'target_date': latest_date.strftime('%Y-%m-%d'),
                'window': window_key,
                'similar_date': entry['date'],
                'correlation': entry['correlation'],
            })
    if csv_rows:
        pd.DataFrame(csv_rows).to_csv(csv_path, index=False, encoding='utf-8')
        print(f'CSV: {csv_path}')

    print()
    print('Dashboard written to GitHub Pages repo.')
    print(f'Target date: {latest_date.strftime("%Y-%m-%d")}')
