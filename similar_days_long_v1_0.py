"""
Similar Days Analyzer (Long) v1.0
=================================

Long-horizon similar days analysis: 5d vs 10d vs 21d forward returns.
Two sections:
  1. 25Y broad universe — cross-asset momentum pattern matching
  2. SPY deep history (~33Y to 1993) — SPY-only momentum matching

Purpose: Historical correlates for rare market patterns (corrections,
recessions, bears) using longer forward windows.

Usage:
    python similar_days_long_v1_0.py

Input:
    - price_cache (via cache_loader)

Output:
    - similar-days-long/index.html (dashboard via DashboardWriter)
    - similar_days_long_data_YYYYMMDD_HHMM.csv

Author: Brian + Claude
Date: 2026-03-27
Version: 1.0
"""

import os
import sys
import json
import importlib
import importlib.util
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict

# ---------------------------------------------------------------------------
# PATH SETUP
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.normpath(os.path.join(_SCRIPT_DIR, '..', 'perplexity-user-data'))
if _DATA_DIR not in sys.path:
    sys.path.insert(0, _DATA_DIR)

# Import cache loader
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
            _mod = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(_mod)
            load_etf_data = _mod.load_etf_data
            print('[OK] Loaded cache_loader from: ' + _fpath)
            break

if load_etf_data is None:
    print('ERROR: Could not find cache_loader in: ' + _DATA_DIR)
    sys.exit(1)

# ---------------------------------------------------------------------------
# PERPLEXITY DEEP RESEARCH
# ---------------------------------------------------------------------------
_PERPLEXITY_CACHE_PATH = os.path.join(_SCRIPT_DIR, 'similar_days_long_perplexity_cache.json')

FORWARD_PERIODS = [5, 10, 21]
PERIOD_LABELS = {5: '5d', 10: '10d', 21: '21d'}


def _call_deep_research(prompt_text):
    """Call Perplexity sonar-deep-research for narrative analysis.

    Returns {'content': str, 'citations': list} or None on failure.
    """
    import requests

    api_key = os.environ.get('PERPLEXITY_API_KEY')
    if not api_key:
        print('[INFO] PERPLEXITY_API_KEY not set -- skipping deep research')
        return None

    headers = {
        'Authorization': 'Bearer ' + api_key,
        'Content-Type': 'application/json',
    }
    payload = {
        'model': 'sonar-deep-research',
        'messages': [
            {'role': 'system', 'content': (
                'You are a financial market historian specializing in bear markets, '
                'corrections, and recessions. Provide detailed analysis with specific '
                'dates, events, drawdown magnitudes, and recovery timelines.'
            )},
            {'role': 'user', 'content': prompt_text},
        ],
    }

    try:
        print('[INFO] Calling Perplexity deep research (this may take 60-120s)...')
        resp = requests.post(
            'https://api.perplexity.ai/chat/completions',
            headers=headers, json=payload, timeout=300
        )
        resp.raise_for_status()
        data = resp.json()
        content = data.get('choices', [{}])[0].get('message', {}).get('content', '')
        citations = data.get('citations', [])
        print('[OK] Deep research returned %d chars, %d citations' % (len(content), len(citations)))
        return {'content': content, 'citations': citations}
    except Exception as e:
        print('[WARN] Deep research call failed: %s' % e)
        return None


def _get_deep_research(prompt_text):
    """Get deep research with same-day caching."""
    today_str = datetime.now().strftime('%Y-%m-%d')

    if os.path.exists(_PERPLEXITY_CACHE_PATH):
        try:
            with open(_PERPLEXITY_CACHE_PATH, 'r', encoding='utf-8') as f:
                cached = json.load(f)
            if cached.get('date') == today_str and cached.get('content'):
                print('[CACHE] Reusing deep research from %s' % today_str)
                return {'content': cached['content'], 'citations': cached.get('citations', [])}
        except Exception:
            pass

    result = _call_deep_research(prompt_text)
    if result is None:
        return None

    try:
        cache_data = {
            'date': today_str,
            'content': result['content'],
            'citations': result.get('citations', []),
        }
        with open(_PERPLEXITY_CACHE_PATH, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
        print('[OK] Cached deep research to %s' % _PERPLEXITY_CACHE_PATH)
    except Exception as e:
        print('[WARN] Failed to write cache: %s' % e)

    return result


# ---------------------------------------------------------------------------
# ASSET GROUPINGS (same as v1.13)
# ---------------------------------------------------------------------------
ANALYSIS_GROUPS = {
    'SECTORS': ['XLF', 'XLV', 'XLE', 'XLI', 'XLY', 'XLB', 'XLU', 'XLK', 'XLP', 'XLRE'],
    'MARKET_CAP': {'large': 'SPY', 'mid': 'MDY', 'small': 'IWM'},
    'STYLE': {
        'growth': 'QQQ', 'value': 'VTV', 'quality': 'QUAL',
        'momentum': 'MTUM', 'size': 'IJR', 'min_vol': 'USMV', 'dividend': 'VIG',
    },
    'INDEXES': {
        'spy': 'SPY', 'dia': 'DIA', 'ndx': '^NDX',
        'iau': 'IAU', 'vgk': 'VGK', 'mchi': 'MCHI', 'tlt': 'TLT',
    },
    'RISK_ASSETS': {'equities': 'SPY', 'bonds': 'TLT', 'gold': 'GLD', 'dollar': 'UUP'},
}

SECTOR_NAMES = {
    'XLF': 'Financials', 'XLV': 'Healthcare', 'XLE': 'Energy',
    'XLI': 'Industrials', 'XLY': 'Consumer Disc', 'XLB': 'Materials',
    'XLU': 'Utilities', 'XLK': 'Technology', 'XLP': 'Consumer Staples',
    'XLRE': 'Real Estate',
}

STYLE_NAMES = {
    'growth': 'Growth (QQQ)', 'value': 'Value (VTV)', 'quality': 'Quality (QUAL)',
    'momentum': 'Momentum (MTUM)', 'size': 'Size (IJR)',
    'min_vol': 'Min Vol (USMV)', 'dividend': 'Dividend (VIG)',
}

INDEX_NAMES = {
    'spy': 'S&P 500 (SPY)', 'dia': 'Dow Jones (DIA)', 'ndx': 'Nasdaq-100 (^NDX)',
    'iau': 'Gold (IAU)', 'vgk': 'Europe (VGK)', 'mchi': 'China (MCHI)',
    'tlt': '20Y Treasury (TLT)',
}

# ---------------------------------------------------------------------------
# MACRO EVENT TAGGER — known market episodes for SPY deep section
# ---------------------------------------------------------------------------
MACRO_EVENTS = [
    ('1993-01-01', '1994-12-31', 'Bond Massacre / Tequila'),
    ('1997-07-01', '1998-10-31', 'Asian Crisis / LTCM'),
    ('2000-03-01', '2002-10-31', 'Dot-com Crash'),
    ('2001-09-01', '2001-12-31', '9/11'),
    ('2007-10-01', '2009-03-31', 'GFC'),
    ('2010-04-01', '2010-07-31', 'Flash Crash / Euro Debt'),
    ('2011-07-01', '2011-10-31', 'US Downgrade / Euro Crisis'),
    ('2015-08-01', '2016-02-29', 'China Deval / Oil Crash'),
    ('2018-01-26', '2018-04-30', 'Volmageddon'),
    ('2018-10-01', '2018-12-31', 'Fed Tightening Selloff'),
    ('2020-02-19', '2020-03-31', 'COVID Crash'),
    ('2020-04-01', '2020-08-31', 'COVID Recovery'),
    ('2022-01-01', '2022-10-31', 'Inflation / Rate Hike Bear'),
    ('2023-03-01', '2023-03-31', 'SVB / Banking Crisis'),
    ('2024-07-15', '2024-08-15', 'Yen Carry Unwind'),
]

_MACRO_RANGES = [
    (pd.Timestamp(s), pd.Timestamp(e), label)
    for s, e, label in MACRO_EVENTS
]


def tag_macro_event(date):
    """Return macro event label for a date, or empty string."""
    ts = pd.Timestamp(date)
    for start, end, label in _MACRO_RANGES:
        if start <= ts <= end:
            return label
    return ''


# ---------------------------------------------------------------------------
# PATTERN MATCHING (cross-asset, same as v1.13)
# ---------------------------------------------------------------------------
def calculate_momentum_features(df):
    """Calculate momentum z-score features for one ticker."""
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
    """Build cross-asset pattern feature matrix."""
    print('\nBuilding pattern matrix...')
    all_features = {}

    for ticker, df in all_data.items():
        features = calculate_momentum_features(df)
        for col in features.columns:
            all_features[ticker + '_' + col] = features[col]

    features_df = pd.DataFrame(all_features)
    print('Pattern matrix: %d days x %d features' % (len(features_df), len(features_df.columns)))
    return features_df


def find_similar_days_masked(features_df, target_date, min_correlation=0.2,
                              top_n=100, min_overlap=200, history_cutoff=None):
    """Find similar days using masked Pearson correlation."""
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


# ---------------------------------------------------------------------------
# SPY-ONLY PATTERN MATCHING (Euclidean distance, full history)
# ---------------------------------------------------------------------------
def find_spy_similar_days(spy_df, target_date, top_n=100):
    """Find similar days using SPY-only momentum features + Euclidean distance.

    Uses 7 features (4 momentum z-scores + 3 slopes) per date.
    Works for full SPY history back to 1993 without cross-asset overlap issues.
    """
    features = calculate_momentum_features(spy_df)

    if target_date not in features.index:
        return []

    target_row = features.loc[target_date].values
    if np.all(np.isnan(target_row)):
        return []

    distances = []
    for hist_date in features.index:
        if hist_date >= target_date:
            continue

        hist_row = features.loc[hist_date].values
        # Both must have all 7 features
        mask = ~np.isnan(target_row) & ~np.isnan(hist_row)
        if mask.sum() < 5:
            continue

        dist = np.sqrt(np.sum((target_row[mask] - hist_row[mask]) ** 2))
        distances.append((hist_date, dist))

    distances.sort(key=lambda x: x[1])
    # Convert distance to similarity score (1 / (1 + dist))
    return [(d, round(1.0 / (1.0 + dist), 4)) for d, dist in distances[:top_n]]


# ---------------------------------------------------------------------------
# MULTI-PERIOD ANALYSIS FUNCTIONS
# ---------------------------------------------------------------------------
def get_forward_return(all_data, ticker, start_date, periods):
    """Get forward return for ticker starting at start_date."""
    if ticker not in all_data:
        return None
    df = all_data[ticker]
    if start_date not in df.index:
        return None
    start_idx = df.index.get_loc(start_date)
    if start_idx + periods >= len(df):
        return None
    start_price = df.iloc[start_idx]['close']
    end_price = df.iloc[start_idx + periods]['close']
    if pd.isna(start_price) or pd.isna(end_price):
        return None
    return (end_price / start_price - 1) * 100


def get_backward_return(all_data, ticker, date, periods=5):
    """Get return for the N days BEFORE the date."""
    if ticker not in all_data:
        return None
    df = all_data[ticker]
    if date not in df.index:
        return None
    date_idx = df.index.get_loc(date)
    if date_idx < periods:
        return None
    start_price = df.iloc[date_idx - periods]['close']
    end_price = df.iloc[date_idx]['close']
    if pd.isna(start_price) or pd.isna(end_price):
        return None
    return (end_price / start_price - 1) * 100


def _summarize_returns(returns_list):
    """Summarize a list of returns into avg, win%, count."""
    if not returns_list:
        return {'avg_return': None, 'pct_positive': None, 'sample_count': 0}
    return {
        'avg_return': round(float(np.mean(returns_list)), 2),
        'pct_positive': round(100 * sum(1 for r in returns_list if r > 0) / len(returns_list), 1),
        'sample_count': len(returns_list),
    }


def analyze_group_multi_period(all_data, similar_days, group_dict, name_map=None):
    """Analyze a group of tickers across all forward periods.

    Returns: {key: {name, backward_5d: {}, fwd_5d: {}, fwd_10d: {}, fwd_21d: {}}}
    """
    # Collect returns
    backward = defaultdict(list)
    forward = {p: defaultdict(list) for p in FORWARD_PERIODS}

    for date, corr in similar_days:
        for key, ticker in group_dict.items():
            back_ret = get_backward_return(all_data, ticker, date, 5)
            if back_ret is not None:
                backward[key].append(back_ret)

            for p in FORWARD_PERIODS:
                fwd_ret = get_forward_return(all_data, ticker, date, p)
                if fwd_ret is not None:
                    forward[p][key].append(fwd_ret)

    summary = {}
    for key in group_dict.keys():
        entry = {
            'name': name_map.get(key, key) if name_map else key,
            'backward_5d': _summarize_returns(backward[key]),
        }
        for p in FORWARD_PERIODS:
            entry['fwd_%dd' % p] = _summarize_returns(forward[p][key])
        summary[key] = entry

    return summary


def analyze_sectors_multi_period(all_data, similar_days):
    """Analyze sectors with multi-period forward returns."""
    backward = defaultdict(list)
    forward = {p: defaultdict(list) for p in FORWARD_PERIODS}

    for date, corr in similar_days:
        for sector in ANALYSIS_GROUPS['SECTORS']:
            back_ret = get_backward_return(all_data, sector, date, 5)
            if back_ret is not None:
                backward[sector].append(back_ret)
            for p in FORWARD_PERIODS:
                fwd_ret = get_forward_return(all_data, sector, date, p)
                if fwd_ret is not None:
                    forward[p][sector].append(fwd_ret)

    summary = {}
    for sector in ANALYSIS_GROUPS['SECTORS']:
        entry = {
            'name': SECTOR_NAMES.get(sector, sector),
            'backward_5d': _summarize_returns(backward[sector]),
        }
        for p in FORWARD_PERIODS:
            entry['fwd_%dd' % p] = _summarize_returns(forward[p][sector])
        summary[sector] = entry

    return summary


def analyze_risk_regime_multi_period(all_data, similar_days):
    """Analyze risk regime with multi-period forward returns."""
    risk_backward = defaultdict(list)
    risk_forward = {p: defaultdict(list) for p in FORWARD_PERIODS}

    for date, corr in similar_days:
        # Backward 5d
        back_returns = {}
        for asset_type, ticker in ANALYSIS_GROUPS['RISK_ASSETS'].items():
            ret = get_backward_return(all_data, ticker, date, 5)
            if ret is not None:
                back_returns[asset_type] = ret
        if len(back_returns) >= 3:
            score = back_returns.get('equities', 0) - back_returns.get('bonds', 0) - back_returns.get('gold', 0) - back_returns.get('dollar', 0)
            risk_backward['scores'].append(score)
            risk_backward['equities'].append(back_returns.get('equities', 0))
            risk_backward['bonds'].append(back_returns.get('bonds', 0))

        # Forward per period
        for p in FORWARD_PERIODS:
            fwd_returns = {}
            for asset_type, ticker in ANALYSIS_GROUPS['RISK_ASSETS'].items():
                ret = get_forward_return(all_data, ticker, date, p)
                if ret is not None:
                    fwd_returns[asset_type] = ret
            if len(fwd_returns) >= 3:
                score = fwd_returns.get('equities', 0) - fwd_returns.get('bonds', 0) - fwd_returns.get('gold', 0) - fwd_returns.get('dollar', 0)
                risk_forward[p]['scores'].append(score)
                risk_forward[p]['equities'].append(fwd_returns.get('equities', 0))
                risk_forward[p]['bonds'].append(fwd_returns.get('bonds', 0))

    if not risk_backward['scores']:
        return None

    def _regime_label(avg_score):
        if avg_score > 0.5:
            return 'RISK-ON'
        if avg_score < -0.5:
            return 'RISK-OFF'
        return 'MIXED'

    back_avg = float(np.mean(risk_backward['scores']))
    result = {
        'backward_5d': {
            'regime': _regime_label(back_avg),
            'avg_risk_score': round(back_avg, 2),
            'equities_avg': round(float(np.mean(risk_backward['equities'])), 2),
            'bonds_avg': round(float(np.mean(risk_backward['bonds'])), 2),
        },
    }

    for p in FORWARD_PERIODS:
        if risk_forward[p]['scores']:
            fwd_avg = float(np.mean(risk_forward[p]['scores']))
            result['fwd_%dd' % p] = {
                'regime': _regime_label(fwd_avg),
                'avg_risk_score': round(fwd_avg, 2),
                'equities_avg': round(float(np.mean(risk_forward[p]['equities'])), 2),
                'bonds_avg': round(float(np.mean(risk_forward[p]['bonds'])), 2),
            }
        else:
            result['fwd_%dd' % p] = None

    return result


def analyze_spy_deep(all_data, spy_similar_days):
    """Build SPY deep history table: date, similarity, event, fwd returns."""
    rows = []
    for date, sim_score in spy_similar_days:
        row = {
            'date': date.strftime('%Y-%m-%d'),
            'similarity': sim_score,
            'event': tag_macro_event(date),
        }
        for p in FORWARD_PERIODS:
            ret = get_forward_return(all_data, 'SPY', date, p)
            row['fwd_%dd' % p] = round(ret, 2) if ret is not None else None
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# MAIN ANALYSIS
# ---------------------------------------------------------------------------
def run_analysis():
    """Run complete long-horizon similar days analysis."""
    print('=' * 80)
    print('Similar Days Analyzer (Long) v1.0')
    print('=' * 80)
    print()

    # Load data
    print('Loading ETF data from cache...')
    all_data = load_etf_data()
    print('Loaded: %d ETFs' % len(all_data))

    # --- SECTION 1: 25Y broad universe ---
    print('\n--- SECTION 1: 25Y Broad Universe ---')
    features_df = build_pattern_matrix(all_data)
    latest_date = features_df.index.max()
    print('Latest date: %s' % latest_date.strftime('%Y-%m-%d'))

    print('\nFinding similar days (25Y, all history)...')
    broad_matches = find_similar_days_masked(
        features_df, latest_date,
        min_overlap=200, top_n=100, history_cutoff=None,
    )
    print('Found %d similar days' % len(broad_matches))
    if broad_matches:
        print('Top 5:')
        for d, c in broad_matches[:5]:
            print('  %s: %.3f' % (d.strftime('%Y-%m-%d'), c))

    # Analyze across all forward periods
    broad_results = None
    if len(broad_matches) >= 3:
        print('\nAnalyzing 25Y matches at 5d/10d/21d...')
        broad_results = {
            'similar_days_count': len(broad_matches),
            'similar_days': [
                {'date': d.strftime('%Y-%m-%d'), 'correlation': round(c, 4)}
                for d, c in broad_matches
            ],
            'sectors': analyze_sectors_multi_period(all_data, broad_matches),
            'indexes': analyze_group_multi_period(
                all_data, broad_matches, ANALYSIS_GROUPS['INDEXES'], INDEX_NAMES
            ),
            'styles': analyze_group_multi_period(
                all_data, broad_matches, ANALYSIS_GROUPS['STYLE'], STYLE_NAMES
            ),
            'market_cap': analyze_group_multi_period(
                all_data, broad_matches, ANALYSIS_GROUPS['MARKET_CAP'],
                {'large': 'Large (SPY)', 'mid': 'Mid (MDY)', 'small': 'Small (IWM)'},
            ),
            'risk_regime': analyze_risk_regime_multi_period(all_data, broad_matches),
        }
    else:
        print('WARNING: Only %d broad matches -- skipping analysis' % len(broad_matches))

    # --- SECTION 2: SPY deep history ---
    print('\n--- SECTION 2: SPY Deep History ---')
    if 'SPY' in all_data:
        spy_df = all_data['SPY']
        spy_start = spy_df.index.min()
        print('SPY history: %s to %s (%d rows)' % (
            spy_start.strftime('%Y-%m-%d'), spy_df.index.max().strftime('%Y-%m-%d'), len(spy_df)
        ))
        spy_matches = find_spy_similar_days(spy_df, latest_date, top_n=100)
        print('Found %d SPY-only similar days' % len(spy_matches))
        if spy_matches:
            print('Top 5 (by similarity):')
            for d, s in spy_matches[:5]:
                event = tag_macro_event(d)
                tag = ' [%s]' % event if event else ''
                print('  %s: %.3f%s' % (d.strftime('%Y-%m-%d'), s, tag))

        spy_deep_rows = analyze_spy_deep(all_data, spy_matches)
        spy_history_years = round((latest_date - spy_start).days / 365.25, 1)
    else:
        spy_deep_rows = []
        spy_history_years = 0
        print('WARNING: SPY not in cache')

    output = {
        'generated_at': datetime.now().isoformat(),
        'target_date': latest_date.strftime('%Y-%m-%d'),
        'broad_25y': broad_results,
        'spy_deep': {
            'history_years': spy_history_years,
            'match_count': len(spy_deep_rows),
            'rows': spy_deep_rows,
        },
    }

    return output, latest_date


# ---------------------------------------------------------------------------
# LLM RESEARCH PROMPT (bear/correction focus)
# ---------------------------------------------------------------------------
def _build_llm_prompt(output):
    """Build deep research prompt focused on bear/correction correlates."""
    lines = []
    lines.append('You are a financial market historian specializing in bear markets, corrections, and recessions.')
    lines.append('')
    lines.append('I am running a quantitative pattern-matching system that identifies historical market days')
    lines.append('with similar momentum profiles to today (March 2026). Below are the top similar days from')
    lines.append('two analyses:')
    lines.append('')
    lines.append('SECTION 1: 25Y cross-asset pattern matching (momentum z-scores across 40+ ETFs)')
    lines.append('SECTION 2: SPY-only deep history matching (back to 1993, ~33 years)')
    lines.append('')
    lines.append('For EACH matched date, analyze:')
    lines.append('1. MACRO REGIME -- expansion/contraction, Fed stance, GDP trajectory')
    lines.append('2. WHAT PRECEDED IT -- what happened in the 1-3 months before this date')
    lines.append('3. WHAT FOLLOWED -- 5d, 10d, 21d, and 63d (3-month) forward returns for SPY')
    lines.append('4. WAS THIS A BEAR/CORRECTION ENTRY POINT? -- did a >10% drawdown follow within 3 months?')
    lines.append('5. RECOVERY TIMELINE -- if a drawdown occurred, how long to recover?')
    lines.append('')
    lines.append('After analyzing individual dates, synthesize:')
    lines.append('- What percentage of these similar days preceded significant drawdowns (>10%)?')
    lines.append('- What is the base rate for the market making new highs vs entering correction?')
    lines.append('- What macro conditions from these historical matches are most similar to NOW (March 2026)?')
    lines.append('- Your probabilistic assessment: what do these patterns suggest about the next 1-3 months?')
    lines.append('')
    lines.append('=' * 60)

    # Section 1: Broad 25Y
    broad = output.get('broad_25y')
    if broad and broad.get('similar_days'):
        lines.append('')
        lines.append('--- 25Y CROSS-ASSET MATCHES (top 25) ---')
        for entry in broad['similar_days'][:25]:
            lines.append('  %s  (correlation: %.3f)' % (entry['date'], entry['correlation']))

    # Section 2: SPY deep
    spy = output.get('spy_deep', {})
    if spy.get('rows'):
        lines.append('')
        lines.append('--- SPY DEEP HISTORY MATCHES (top 50, ~%dY history) ---' % int(spy.get('history_years', 0)))
        for row in spy['rows'][:50]:
            event = ' [%s]' % row['event'] if row['event'] else ''
            fwd5 = ('%.1f%%' % row['fwd_5d']) if row['fwd_5d'] is not None else 'N/A'
            fwd21 = ('%.1f%%' % row['fwd_21d']) if row['fwd_21d'] is not None else 'N/A'
            lines.append('  %s  (sim: %.3f)  fwd5d: %s  fwd21d: %s%s' % (
                row['date'], row['similarity'], fwd5, fwd21, event
            ))

    lines.append('')
    lines.append('=' * 60)
    lines.append('Please structure your response with:')
    lines.append('1. Analysis of clustered date groups (group dates by macro episode)')
    lines.append('2. Bear/correction probability assessment')
    lines.append('3. Most relevant historical analog for current conditions')
    lines.append('4. Actionable synthesis for a portfolio manager')

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# HTML / JS
# ---------------------------------------------------------------------------
EXTRA_CSS = """
/* Similar days long -- multi-period tables */
.sim-long-table table {
    border-collapse: collapse; width: 100%; font-size: 0.8em; table-layout: fixed;
}
.sim-long-table thead th {
    padding: 7px 6px; font-size: 0.72em; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.04em;
    cursor: pointer; user-select: none;
    border-bottom: 2px solid #e2e4e8; white-space: nowrap; text-align: center;
}
.sim-long-table thead th.col-name { text-align: left; }
.sim-long-table tbody td {
    padding: 6px 6px; border-bottom: 1px solid #f0f0f0;
    vertical-align: middle; text-align: center;
    font-family: 'IBM Plex Mono', monospace;
}
.sim-long-table tbody td.col-name {
    text-align: left; font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.9em; color: #333; white-space: nowrap;
}
.sim-long-table tbody tr:hover { outline: 2px solid #4f46e5; }

/* Group header row */
.grp-row th { padding: 5px 4px; font-size: 0.78em; font-weight: 700; letter-spacing: 0.05em; }
.grp-blank  { background: #f8f9fb; color: transparent; border-bottom: 1px solid #e2e4e8; }
.grp-ctx    { background: #1e293b; color: #94a3b8; }
.grp-5d     { background: #14532d; color: #86efac; }
.grp-10d    { background: #1e3a5f; color: #93c5fd; }
.grp-21d    { background: #7c2d12; color: #fed7aa; }

/* Sort arrows */
th.sort-asc::after  { content: ' +'; font-size: 0.7em; }
th.sort-desc::after { content: ' -'; font-size: 0.7em; }

.pred-pct { display: block; font-size: 0.78em; color: #333; opacity: 0.75; }
.cell-miss { color: #ccc !important; background: #fafafa !important; }

/* SPY deep table */
.spy-deep-table table { border-collapse: collapse; width: 100%; font-size: 0.82em; }
.spy-deep-table thead th {
    padding: 7px 8px; font-size: 0.75em; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.04em;
    border-bottom: 2px solid #e2e4e8; text-align: center;
    cursor: pointer; user-select: none;
}
.spy-deep-table tbody td {
    padding: 5px 8px; border-bottom: 1px solid #f0f0f0;
    text-align: center; font-family: 'IBM Plex Mono', monospace;
}
.spy-deep-table tbody tr:hover { outline: 2px solid #4f46e5; }
.event-tag {
    display: inline-block; padding: 2px 8px; border-radius: 4px;
    font-size: 0.78em; font-weight: 600; background: #fef3c7; color: #92400e;
    white-space: nowrap;
}

/* Dates panel (single panel for 25Y) */
.dates-panel-long {
    background: #fff; border: 1px solid #e2e4e8; border-radius: 8px;
    border-top: 4px solid #14532d; margin-bottom: 18px; max-width: 600px;
}
.dates-panel-long .header {
    padding: 10px 14px; background: #14532d; color: #86efac;
    font-size: 0.82em; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.06em; display: flex; justify-content: space-between;
}
.dates-panel-long .body {
    padding: 10px 14px; max-height: 280px; overflow-y: auto;
}
.date-row {
    display: flex; justify-content: space-between;
    font-size: 0.82em; padding: 4px 0; border-bottom: 1px solid #f5f5f5;
    font-family: 'IBM Plex Mono', monospace;
}
.date-row:last-child { border-bottom: none; }
.date-corr { color: #888; }

/* Mobile */
@media (max-width: 768px) {
    .sim-long-table { overflow-x: auto; }
    .sim-long-table table { font-size: 0.72em; }
    .spy-deep-table { overflow-x: auto; }
    .spy-deep-table table { font-size: 0.72em; }
}
"""

RENDER_JS = """
(function() {{
    const DATA = {data_json};
    const PERIODS = [5, 10, 21];
    const P_LABELS = {{5: '5d', 10: '10d', 21: '21d'}};
    const GRP_CLS = {{5: 'grp-5d', 10: 'grp-10d', 21: 'grp-21d'}};

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

    // Collect all numeric values for gradient normalization
    function calcRanges(tableKey) {{
        const ranges = {{}};
        const broad = DATA.broad_25y;
        if (!broad) return ranges;
        const tbl = broad[tableKey];
        if (!tbl) return ranges;

        // backward_5d
        const backVals = [];
        Object.values(tbl).forEach(item => {{
            const v = item?.backward_5d?.avg_return;
            if (v != null && !isNaN(v)) backVals.push(v);
        }});
        if (backVals.length) {{
            const s = [...backVals].sort((a,b)=>a-b);
            ranges['backward_5d'] = {{ min: s[Math.floor(s.length*0.05)], max: s[Math.floor(s.length*0.95)] }};
        }}

        PERIODS.forEach(p => {{
            const key = 'fwd_' + p + 'd';
            const vals = [];
            Object.values(tbl).forEach(item => {{
                const v = item?.[key]?.avg_return;
                if (v != null && !isNaN(v)) vals.push(v);
            }});
            if (vals.length) {{
                const s = [...vals].sort((a,b)=>a-b);
                ranges[key] = {{ min: s[Math.floor(s.length*0.05)], max: s[Math.floor(s.length*0.95)] }};
            }}
        }});

        return ranges;
    }}

    function bgCell(val, ranges, key) {{
        if (val == null || isNaN(val)) return null;
        const r = ranges[key];
        if (!r || r.max === r.min) return getGradientColor(0.5);
        const n = Math.max(0, Math.min(1, (val - r.min) / (r.max - r.min)));
        return getGradientColor(n);
    }}

    const sortState = {{}};

    function buildTable(containerId, tableKey) {{
        const broad = DATA.broad_25y;
        if (!broad || !broad[tableKey]) {{
            document.getElementById(containerId).innerHTML = '<p style="color:#aaa;">No data</p>';
            return;
        }}
        const tbl = broad[tableKey];
        const ss = sortState[containerId] || {{ col: 'fwd_5d', dir: 'desc' }};
        sortState[containerId] = ss;
        const ranges = calcRanges(tableKey);

        let rows = Object.entries(tbl).map(([key, item]) => ({{ key, name: item.name || key, item }}));

        rows.sort((a, b) => {{
            if (ss.col === 'name') return ss.dir === 'asc' ? a.name.localeCompare(b.name) : b.name.localeCompare(a.name);
            const va = a.item?.[ss.col]?.avg_return ?? -999;
            const vb = b.item?.[ss.col]?.avg_return ?? -999;
            return ss.dir === 'asc' ? va - vb : vb - va;
        }});

        const arrow = col => ss.col === col ? (ss.dir === 'asc' ? ' +' : ' -') : '';
        const thSort = (col, label) =>
            `<th style="cursor:pointer;" onclick="window._simLongSort('${{containerId}}','${{tableKey}}','${{col}}')">${{label}}${{arrow(col)}}</th>`;

        let html = '<table><thead>';
        // Group header
        html += '<tr class="grp-row">';
        html += '<th class="grp-blank col-name"></th>';
        html += '<th class="grp-ctx">Context</th>';
        PERIODS.forEach(p => html += `<th colspan="2" class="${{GRP_CLS[p]}}">${{P_LABELS[p]}} Forward</th>`);
        html += '</tr>';
        // Column header
        html += '<tr>';
        html += thSort('name', 'Name');
        html += thSort('backward_5d', 'Before 5d');
        PERIODS.forEach(p => {{
            const key = 'fwd_' + p + 'd';
            html += thSort(key, 'Fwd ' + P_LABELS[p]);
            html += '<th style="cursor:default;">Win%/n</th>';
        }});
        html += '</tr></thead><tbody>';

        rows.forEach(row => {{
            const item = row.item;
            html += '<tr>';
            html += `<td class="col-name">${{row.name}}</td>`;

            // Backward 5d
            const backAvg = item?.backward_5d?.avg_return;
            const backPct = item?.backward_5d?.pct_positive;
            if (backAvg != null) {{
                const bg = bgCell(backAvg, ranges, 'backward_5d');
                const fg = textColor(bg);
                const sign = backAvg >= 0 ? '+' : '';
                html += `<td style="background:${{bg}};color:${{fg}};">${{sign}}${{backAvg.toFixed(1)}}%`;
                if (backPct != null) html += `<span class="pred-pct">${{backPct.toFixed(0)}}%</span>`;
                html += '</td>';
            }} else html += '<td class="cell-miss">--</td>';

            // Forward periods
            PERIODS.forEach(p => {{
                const key = 'fwd_' + p + 'd';
                const fwdAvg = item?.[key]?.avg_return;
                const fwdPct = item?.[key]?.pct_positive;
                const fwdN = item?.[key]?.sample_count;

                if (fwdAvg != null) {{
                    const bg = bgCell(fwdAvg, ranges, key);
                    const fg = textColor(bg);
                    const sign = fwdAvg >= 0 ? '+' : '';
                    html += `<td style="background:${{bg}};color:${{fg}};">${{sign}}${{fwdAvg.toFixed(1)}}%`;
                    if (fwdPct != null) html += `<span class="pred-pct">${{fwdPct.toFixed(0)}}%</span>`;
                    html += '</td>';
                }} else html += '<td class="cell-miss">--</td>';

                if (fwdPct != null) {{
                    html += `<td style="color:#555;font-size:0.85em;">${{fwdPct.toFixed(0)}}%`;
                    if (fwdN != null) html += ` <span style="color:#aaa;">n=${{fwdN}}</span>`;
                    html += '</td>';
                }} else html += '<td class="cell-miss">--</td>';
            }});

            html += '</tr>';
        }});

        html += '</tbody></table>';
        document.getElementById(containerId).innerHTML = html;
    }}

    window._simLongSort = function(containerId, tableKey, col) {{
        const ss = sortState[containerId] || {{ col: 'fwd_5d', dir: 'desc' }};
        sortState[containerId] = {{
            col, dir: (ss.col === col && ss.dir === 'desc') ? 'asc' : 'desc'
        }};
        buildTable(containerId, tableKey);
    }};

    // SPY deep table
    function buildSpyDeepTable() {{
        const spy = DATA.spy_deep;
        if (!spy || !spy.rows || !spy.rows.length) {{
            document.getElementById('tbl-spy-deep').innerHTML = '<p style="color:#aaa;">No SPY deep data</p>';
            return;
        }}

        const ss = sortState['spy-deep'] || {{ col: 'similarity', dir: 'desc' }};
        sortState['spy-deep'] = ss;

        let rows = [...spy.rows];

        rows.sort((a, b) => {{
            if (ss.col === 'date') return ss.dir === 'asc' ? a.date.localeCompare(b.date) : b.date.localeCompare(a.date);
            if (ss.col === 'event') return ss.dir === 'asc' ? (a.event||'').localeCompare(b.event||'') : (b.event||'').localeCompare(a.event||'');
            const va = a[ss.col] ?? -999;
            const vb = b[ss.col] ?? -999;
            return ss.dir === 'asc' ? va - vb : vb - va;
        }});

        // Collect fwd return ranges for gradient
        const fwdRanges = {{}};
        PERIODS.forEach(p => {{
            const key = 'fwd_' + p + 'd';
            const vals = spy.rows.map(r => r[key]).filter(v => v != null);
            if (vals.length) {{
                const s = [...vals].sort((a,b)=>a-b);
                fwdRanges[key] = {{ min: s[Math.floor(s.length*0.05)], max: s[Math.floor(s.length*0.95)] }};
            }}
        }});

        const arrow = col => ss.col === col ? (ss.dir === 'asc' ? ' +' : ' -') : '';
        const thSort = col => `onclick="window._spyDeepSort('${{col}}')" style="cursor:pointer;"`;

        let html = '<table><thead><tr>';
        html += `<th ${{thSort('date')}}>Date${{arrow('date')}}</th>`;
        html += `<th ${{thSort('similarity')}}>Similarity${{arrow('similarity')}}</th>`;
        html += `<th ${{thSort('event')}}>Event${{arrow('event')}}</th>`;
        PERIODS.forEach(p => {{
            const key = 'fwd_' + p + 'd';
            html += `<th ${{thSort(key)}} class="${{GRP_CLS[p]}}" style="cursor:pointer;color:#fff;">Fwd ${{P_LABELS[p]}}${{arrow(key)}}</th>`;
        }});
        html += '</tr></thead><tbody>';

        rows.forEach(row => {{
            html += '<tr>';
            html += `<td style="text-align:left;font-family:'IBM Plex Mono',monospace;">${{row.date}}</td>`;
            html += `<td>${{row.similarity.toFixed(3)}}</td>`;
            html += row.event ? `<td><span class="event-tag">${{row.event}}</span></td>` : '<td></td>';

            PERIODS.forEach(p => {{
                const key = 'fwd_' + p + 'd';
                const val = row[key];
                if (val != null) {{
                    const bg = bgCell(val, fwdRanges, key);
                    const fg = bg ? textColor(bg) : '#000';
                    const sign = val >= 0 ? '+' : '';
                    html += `<td style="background:${{bg||''}};color:${{fg}};">${{sign}}${{val.toFixed(1)}}%</td>`;
                }} else html += '<td class="cell-miss">--</td>';
            }});
            html += '</tr>';
        }});

        html += '</tbody></table>';
        document.getElementById('tbl-spy-deep').innerHTML = html;
    }}

    window._spyDeepSort = function(col) {{
        const ss = sortState['spy-deep'] || {{ col: 'similarity', dir: 'desc' }};
        sortState['spy-deep'] = {{
            col, dir: (ss.col === col && ss.dir === 'desc') ? 'asc' : 'desc'
        }};
        buildSpyDeepTable();
    }};

    // Risk regime table
    function buildRiskTable() {{
        const rr = DATA.broad_25y?.risk_regime;
        if (!rr) {{
            document.getElementById('tbl-risk').innerHTML = '<p style="color:#aaa;">No risk data</p>';
            return;
        }}

        function regimeTd(regime) {{
            if (!regime) return '<td class="cell-miss">--</td>';
            const colors = {{ 'RISK-ON':'#16a34a', 'RISK-OFF':'#dc2626', 'MIXED':'#d97706' }};
            return `<td style="background:${{colors[regime]||'#888'}};color:#fff;font-weight:700;">${{regime}}</td>`;
        }}
        function numTd(val, fmt) {{
            if (val == null) return '<td class="cell-miss">--</td>';
            return `<td>${{fmt(val)}}</td>`;
        }}

        let html = '<table><thead><tr class="grp-row">';
        html += '<th class="grp-blank">Metric</th>';
        html += '<th class="grp-ctx">Context</th>';
        PERIODS.forEach(p => html += `<th class="${{GRP_CLS[p]}}">${{P_LABELS[p]}} Forward</th>`);
        html += '</tr></thead><tbody>';

        // Regime row
        html += '<tr><td class="col-name" style="text-align:left;">Regime</td>';
        html += regimeTd(rr.backward_5d?.regime);
        PERIODS.forEach(p => html += regimeTd(rr['fwd_' + p + 'd']?.regime));
        html += '</tr>';

        // Risk Score
        const fmtScore = v => (v >= 0 ? '+' : '') + v.toFixed(2);
        html += '<tr><td class="col-name" style="text-align:left;">Risk Score</td>';
        html += numTd(rr.backward_5d?.avg_risk_score, fmtScore);
        PERIODS.forEach(p => html += numTd(rr['fwd_' + p + 'd']?.avg_risk_score, fmtScore));
        html += '</tr>';

        // Equities
        const fmtPct = v => (v >= 0 ? '+' : '') + v.toFixed(2) + '%';
        html += '<tr><td class="col-name" style="text-align:left;">Equities Avg</td>';
        html += numTd(rr.backward_5d?.equities_avg, fmtPct);
        PERIODS.forEach(p => html += numTd(rr['fwd_' + p + 'd']?.equities_avg, fmtPct));
        html += '</tr>';

        // Bonds
        html += '<tr><td class="col-name" style="text-align:left;">Bonds Avg</td>';
        html += numTd(rr.backward_5d?.bonds_avg, fmtPct);
        PERIODS.forEach(p => html += numTd(rr['fwd_' + p + 'd']?.bonds_avg, fmtPct));
        html += '</tr>';

        html += '</tbody></table>';
        document.getElementById('tbl-risk').innerHTML = html;
    }}

    // Render all
    buildRiskTable();
    buildTable('tbl-indexes',    'indexes');
    buildTable('tbl-market-cap', 'market_cap');
    buildTable('tbl-sectors',    'sectors');
    buildTable('tbl-styles',     'styles');
    buildSpyDeepTable();
}})();
"""


def build_body_html(output, latest_date, writer):
    """Build full dashboard HTML body."""
    date_str = latest_date.strftime('%Y-%m-%d')
    broad = output.get('broad_25y')
    spy = output.get('spy_deep', {})

    broad_count = broad['similar_days_count'] if broad else 0
    spy_count = spy.get('match_count', 0)
    spy_years = spy.get('history_years', 0)

    parts = []

    # Stat bar
    parts.append(writer.stat_bar([
        ('Target Date', date_str, 'neutral'),
        ('25Y Matches', str(broad_count), 'neutral'),
        ('SPY Matches', str(spy_count), 'neutral'),
        ('SPY History', '%.0fY' % spy_years, 'neutral'),
    ]))

    parts.append(writer.build_header(
        'Long-horizon similar days: 5d / 10d / 21d forward analysis. '
        'Bear market and correction pattern identification.'
    ))

    # --- 25Y similar dates panel ---
    if broad and broad.get('similar_days'):
        panel_html = '<div class="dates-panel-long">'
        panel_html += '<div class="header"><span>25Y Similar Days</span><span>%d</span></div>' % broad_count
        panel_html += '<div class="body">'
        for entry in broad['similar_days']:
            panel_html += (
                '<div class="date-row">'
                '<span>%s</span>'
                '<span class="date-corr">%.3f</span>'
                '</div>' % (entry['date'], entry['correlation'])
            )
        panel_html += '</div></div>'
        parts.append(panel_html)

    # --- Section 1: 25Y Multi-Period Tables ---
    parts.append('<h2 style="margin:28px 0 12px;font-size:1.15em;color:#1e293b;border-bottom:2px solid #e2e4e8;padding-bottom:8px;">'
                 'Section 1: 25Y Cross-Asset Analysis (5d / 10d / 21d)</h2>')

    def wrap_table(title, div_id):
        return (
            '<div class="table-section" style="margin-bottom:22px;">'
            '<div class="table-section-header"><h2>%s</h2>'
            '<span style="font-size:0.85em;color:#888;">'
            'Before 5d = context leading into match | Fwd = forward returns | Click to sort</span>'
            '</div>'
            '<div class="sim-long-table" style="overflow-x:auto;padding:0;">'
            '<div id="%s"></div></div></div>' % (title, div_id)
        )

    parts.append(wrap_table('Risk Regime', 'tbl-risk'))
    parts.append(wrap_table('Major Indexes', 'tbl-indexes'))
    parts.append(wrap_table('Market Cap', 'tbl-market-cap'))
    parts.append(wrap_table('Sectors', 'tbl-sectors'))
    parts.append(wrap_table('Style Factors', 'tbl-styles'))

    # --- Section 2: SPY Deep History ---
    parts.append('<h2 style="margin:28px 0 12px;font-size:1.15em;color:#1e293b;border-bottom:2px solid #e2e4e8;padding-bottom:8px;">'
                 'Section 2: SPY Deep History (~%.0fY, back to 1993)</h2>' % spy_years)

    parts.append(
        '<div class="table-section" style="margin-bottom:22px;">'
        '<div class="table-section-header"><h2>SPY-Only Pattern Matches</h2>'
        '<span style="font-size:0.85em;color:#888;">'
        'Momentum-based matching on SPY alone. Event tags show known macro episodes. Click to sort.</span>'
        '</div>'
        '<div class="spy-deep-table" style="overflow-x:auto;padding:0;">'
        '<div id="tbl-spy-deep"></div></div></div>'
    )

    # --- LLM Research Prompt ---
    llm_prompt = _build_llm_prompt(output)

    prompt_html = (
        '<div class="table-section" style="margin-bottom:22px;">'
        '<div class="table-section-header"><h2>Deep Research Prompt</h2>'
        '<span style="font-size:0.85em;color:#888;">Auto-sent to Perplexity deep research for narrative analysis</span>'
        '</div>'
        '<div style="padding:16px;">'
        '<button onclick="navigator.clipboard.writeText(document.getElementById(\'llm-prompt-text\').innerText)" '
        'style="margin-bottom:12px;padding:6px 16px;background:#4f46e5;color:#fff;border:none;border-radius:6px;cursor:pointer;font-size:0.85em;">'
        'Copy Prompt</button>'
        '<pre id="llm-prompt-text" style="background:#f8f9fb;border:1px solid #e2e4e8;border-radius:6px;'
        'padding:14px;font-size:0.78em;font-family:\'IBM Plex Mono\',monospace;white-space:pre-wrap;'
        'overflow-x:auto;max-height:400px;overflow-y:auto;">%s</pre>'
        '</div></div>' % llm_prompt
    )
    parts.append(prompt_html)

    # --- Perplexity Deep Research Results ---
    research_result = _get_deep_research(llm_prompt)
    if research_result and research_result.get('content'):
        import html as _html
        import re as _re

        raw = research_result['content']
        citations = research_result.get('citations', [])

        rendered_lines = []
        for line in raw.split('\n'):
            line = _html.escape(line)
            if line.startswith('### '):
                line = '<h3 style="margin:18px 0 8px;font-size:1.05em;color:#1e293b;">%s</h3>' % line[4:]
                rendered_lines.append(line)
                continue
            elif line.startswith('## '):
                line = '<h2 style="margin:18px 0 8px;font-size:1.1em;color:#1e293b;">%s</h2>' % line[3:]
                rendered_lines.append(line)
                continue

            line = _re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', line)

            def _cite_link(m):
                num = m.group(1)
                idx = int(num) - 1
                if 0 <= idx < len(citations):
                    url = _html.escape(citations[idx])
                    return '<sup><a href="%s" target="_blank" style="color:#4f46e5;text-decoration:none;">[%s]</a></sup>' % (url, num)
                return '<sup>[%s]</sup>' % num
            line = _re.sub(r'\[(\d+)\]', _cite_link, line)

            if line.strip() == '':
                rendered_lines.append('<br>')
                continue
            rendered_lines.append('<p style="margin:4px 0;line-height:1.6;">%s</p>' % line)

        rendered_content = '\n'.join(rendered_lines)

        citation_html = ''
        if citations:
            citation_items = ''.join(
                '<li style="margin-bottom:4px;"><a href="%s" target="_blank" '
                'style="color:#4f46e5;text-decoration:none;">%s</a></li>' % (_html.escape(c), _html.escape(c))
                for c in citations
            )
            citation_html = (
                '<div style="margin-top:16px;padding-top:12px;border-top:1px solid #e2e4e8;">'
                '<strong style="font-size:0.85em;">Sources:</strong>'
                '<ol style="font-size:0.8em;color:#555;margin-top:6px;padding-left:20px;">%s</ol>'
                '</div>' % citation_items
            )

        research_html = (
            '<div class="table-section" style="margin-bottom:22px;">'
            '<div class="table-section-header"><h2>Perplexity Deep Research Analysis</h2>'
            '<span style="font-size:0.85em;color:#888;">Auto-generated bear/correction narrative via deep research</span>'
            '</div>'
            '<div style="padding:16px;">'
            '<div style="background:#f8f9fb;border:1px solid #e2e4e8;border-radius:6px;'
            'padding:18px;font-size:0.85em;font-family:inherit;">%s</div>'
            '%s</div></div>' % (rendered_content, citation_html)
        )
        parts.append(research_html)

    parts.append(writer.footer())

    # Embed data + render JS
    data_json = json.dumps(output, ensure_ascii=False, default=str)
    render_js = RENDER_JS.format(data_json=data_json)
    writer._sim_extra_js = render_js

    return '\n'.join(parts)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    try:
        from dashboard_writer import DashboardWriter
    except ImportError:
        print('ERROR: dashboard_writer.py not found in current directory')
        sys.exit(1)

    output, latest_date = run_analysis()

    writer = DashboardWriter('similar-days-long', 'Similar Days (Long)')
    body = build_body_html(output, latest_date, writer)
    extra_js = getattr(writer, '_sim_extra_js', '')
    writer.write(body, extra_css=EXTRA_CSS, extra_js=extra_js)

    # Write CSV
    csv_path = os.path.join(
        _SCRIPT_DIR,
        'similar_days_long_data_%s.csv' % datetime.now().strftime('%Y%m%d_%H%M')
    )
    csv_rows = []

    broad = output.get('broad_25y')
    if broad and broad.get('similar_days'):
        for entry in broad['similar_days']:
            csv_rows.append({
                'target_date': latest_date.strftime('%Y-%m-%d'),
                'section': '25Y_broad',
                'similar_date': entry['date'],
                'score': entry['correlation'],
            })

    spy = output.get('spy_deep', {})
    for row in spy.get('rows', []):
        csv_rows.append({
            'target_date': latest_date.strftime('%Y-%m-%d'),
            'section': 'SPY_deep',
            'similar_date': row['date'],
            'score': row['similarity'],
            'event': row.get('event', ''),
            'fwd_5d': row.get('fwd_5d'),
            'fwd_10d': row.get('fwd_10d'),
            'fwd_21d': row.get('fwd_21d'),
        })

    if csv_rows:
        pd.DataFrame(csv_rows).to_csv(csv_path, index=False, encoding='utf-8')
        print('CSV: %s' % csv_path)

    print()
    print('Dashboard written to GitHub Pages repo.')
    print('Target date: %s' % latest_date.strftime('%Y-%m-%d'))
