# -*- coding: utf-8 -*-
# ============================================================================
# meta_dashboard_backend.py - v1.5
# Last updated: 2026-03-06
# ============================================================================
# v1.5: Add pullback health as 8th ticker-level source
#   - Load pullback_health_data.json: health >= 80 (HEALTHY) counts as source
#   - Full mode: 8 sources | Trusted mode: 5 sources (health is trusted)
#   - Health column in both full-mode and trusted-mode tables
#   - Tooltip on Health header
# v1.4: Add slope stage scanner as 7th ticker-level source
#   - Load slope_stage_data.json: Stage 2/3 or 1->2 transition counts as source
#   - Full mode: 7 sources | Trusted mode: 4 sources
#   - Slope Stage column in full-mode agreement table
#   - Stage 3 parabolic risk flags in full mode
# v1.3: Add long ranker + tooltips
#   - Load momentum_ranker_long_data.json as 6th ticker-level source
#   - Rename "Ranker" to "Ranker S/T", add "Ranker L/T" column
#   - Full mode: 6 sources | Trusted mode: 4 sources
#   - Add title= tooltips to all table headers
# v1.2: Fix empty Stock SR + Spread columns
#   - Spread: added POLE_TO_SECTOR map (ETF poles -> sector names for spread lookup)
#   - Stock SR: read full source CSV (995 stocks), score>=5 counts as source,
#     show score for all tickers in the matrix as supplemental info
# v1.1: Added Trusted Only toggle
#   - Toggle between All Sources (5 ticker-level) and Trusted Only (3)
#   - Trusted excludes: Advanced Momentum, Conservative Qualifier, Macro
#   - Both views pre-computed; JS toggles visibility
# v1.0: Initial release
#   - Cross-dashboard agreement matrix from 5 ticker-level sources
#   - Regime context stat bar + routing banner
#   - Intermarket force analysis (4 categories)
#   - Pattern context (similar days, sector rotation, changepoint)
#   - Risk flags table
#   - Reads from 11 other backend outputs -- no API calls, no price_cache
# ============================================================================

import os
import sys
import glob
import json
import datetime
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.normpath(os.path.join(_SCRIPT_DIR, '..', 'perplexity-user-data'))

sys.path.insert(0, _SCRIPT_DIR)
from dashboard_writer import DashboardWriter


# ============================================================================
# HELPERS
# ============================================================================

def _load_latest_csv(pattern):
    """Find most-recently-modified CSV matching glob *pattern*."""
    matches = glob.glob(pattern)
    if not matches:
        return None, None
    latest = max(matches, key=os.path.getmtime)
    try:
        return pd.read_csv(latest, encoding='utf-8'), latest
    except Exception as e:
        print('  WARN: could not read {}: {}'.format(latest, e))
        return None, None


def _load_json(path):
    """Load JSON from *path*, returning None on any failure."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print('  WARN: could not read {}: {}'.format(path, e))
        return None


def _load_latest_json(pattern):
    """Find most-recently-modified JSON matching glob *pattern*."""
    matches = glob.glob(pattern)
    if not matches:
        return None, None
    latest = max(matches, key=os.path.getmtime)
    return _load_json(latest), latest


def _fmt_pct(val, default='--'):
    """Format a number as percentage string."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return default
    return '{:+.2f}%'.format(float(val))


def _fmt_num(val, decimals=2, default='--'):
    """Format a number with given decimal places."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return default
    fmt = '{{:.{}f}}'.format(decimals)
    return fmt.format(float(val))


def _css_class_for_value(val, thresholds=None):
    """Return 'pos' / 'neg' / 'warn' / 'neutral' CSS class."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return 'neutral'
    val = float(val)
    if thresholds:
        if val >= thresholds.get('pos', float('inf')):
            return 'pos'
        if val <= thresholds.get('neg', float('-inf')):
            return 'neg'
        if val <= thresholds.get('warn', float('-inf')):
            return 'warn'
    else:
        if val > 0:
            return 'pos'
        if val < 0:
            return 'neg'
    return 'neutral'


# ============================================================================
# SPREAD -> SECTOR MAPPING
# ============================================================================

SPREAD_SECTOR_MAP = {
    'COPX/GLD': ['Basic Materials'],
    'XLB/XLU':  ['Basic Materials', 'Utilities'],
    'SMH/SPY':  ['Technology'],
    'XLK/XLP':  ['Technology', 'Consumer Defensive'],
    'XLY/XLP':  ['Consumer Cyclical', 'Consumer Defensive'],
    'IWM/SPY':  ['Small Cap'],
    'HYG/LQD':  ['Credit'],
    'EEM/SPY':  ['Emerging Markets'],
    'EFA/SPY':  ['International'],
    'XLF/XLU':  ['Financial Services', 'Utilities'],
    'XLE/XLU':  ['Energy', 'Utilities'],
    'XLI/XLU':  ['Industrials', 'Utilities'],
    'XLV/SPY':  ['Healthcare'],
    'TLT/IEF':  ['Rates'],
    'SHY/TLT':  ['Rates'],
    'GLD/SPY':  ['Commodities'],
    'USO/SPY':  ['Energy'],
    'SPY/TLT':  ['Equities'],
    'GDX/GLD':  ['Basic Materials'],
    'XHB/SPY':  ['Real Estate'],
    'ITB/SPY':  ['Real Estate'],
    'KWEB/SPY': ['Technology', 'Emerging Markets'],
    'XBI/XLV':  ['Healthcare'],
    'ARKK/SPY': ['Technology'],
    'VNQ/SPY':  ['Real Estate'],
    'DBA/SPY':  ['Commodities'],
}

# Reverse map: sector -> spread names
SECTOR_SPREAD_MAP = {}
for spread_name, sectors in SPREAD_SECTOR_MAP.items():
    for sec in sectors:
        SECTOR_SPREAD_MAP.setdefault(sec, []).append(spread_name)

# Ranker poles are ETF tickers -- map them to sector names for spread lookup
POLE_TO_SECTOR = {
    'XLB':  'Basic Materials',
    'XLK':  'Technology',
    'QQQ':  'Technology',
    'XLI':  'Industrials',
    'XLE':  'Energy',
    'XLV':  'Healthcare',
    'XLU':  'Utilities',
    'XLP':  'Consumer Defensive',
    'XLY':  'Consumer Cyclical',
    'XLC':  'Communication Services',
    'XLRE': 'Real Estate',
    'XLF':  'Financial Services',
    'TLT':  'Rates',
    'IAU':  'Commodities',
    'VGK':  'International',
    'VXUS': 'International',
    'EEM':  'Emerging Markets',
    'IWM':  'Small Cap',
    'SMH':  'Technology',
}


# ============================================================================
# DATA LOADERS
# ============================================================================

def load_all_sources():
    """Load all 11 data sources. Returns dict of {name: data_or_None}."""
    sources = {}

    # --- Fixed-path JSON ---
    print('  Loading macro_data.json...')
    sources['macro'] = _load_json(os.path.join(_DATA_DIR, 'macro_data.json'))

    print('  Loading momentum_ranker_data.json...')
    sources['ranker'] = _load_json(os.path.join(_SCRIPT_DIR, 'momentum_ranker_data.json'))

    print('  Loading momentum_ranker_long_data.json...')
    sources['ranker_long'] = _load_json(os.path.join(_SCRIPT_DIR, 'momentum_ranker_long_data.json'))

    print('  Loading advanced_momentum_analysis.json...')
    sources['advanced'] = _load_json(os.path.join(_DATA_DIR, 'advanced_momentum_analysis.json'))

    print('  Loading changepoint_summary.json...')
    sources['changepoint'] = _load_json(os.path.join(_SCRIPT_DIR, 'changepoint_summary.json'))

    # --- Timestamped CSVs ---
    print('  Loading crash detection CSV...')
    sources['crash'], _ = _load_latest_csv(
        os.path.join(_SCRIPT_DIR, 'crash_detection_data_*.csv'))

    print('  Loading momentum qualifier CSV...')
    sources['qualifier'], _ = _load_latest_csv(
        os.path.join(_SCRIPT_DIR, 'conservative_momentum_data_*.csv'))

    print('  Loading sector rotation CSV...')
    sources['secrot'], _ = _load_latest_csv(
        os.path.join(_SCRIPT_DIR, 'sector_rotation_data_*.csv'))

    print('  Loading spread monitor CSV...')
    sources['spreads'], _ = _load_latest_csv(
        os.path.join(_DATA_DIR, 'output', 'spread_monitor', 'spread_monitor_*.csv'))

    print('  Loading HYG/LQD CSV...')
    sources['hyglqd'], _ = _load_latest_csv(
        os.path.join(_SCRIPT_DIR, 'hyglqd_data_*.csv'))

    print('  Loading similar days CSV...')
    sources['similar'], _ = _load_latest_csv(
        os.path.join(_SCRIPT_DIR, 'similar_days_data_*.csv'))

    print('  Loading stock secrot scores CSV...')
    sources['stock_secrot'], _ = _load_latest_csv(
        os.path.join(_DATA_DIR, 'stock_secrot_scores_*.csv'))

    # --- Slope Stage Scanner ---
    print('  Loading slope_stage_data.json...')
    sources['slope_stage'] = _load_json(os.path.join(_DATA_DIR, 'slope_stage_data.json'))

    # --- Pullback Health ---
    print('  Loading pullback_health_data.json...')
    sources['pullback_health'] = _load_json(os.path.join(_DATA_DIR, 'pullback_health_data.json'))

    # --- Routing JSON ---
    print('  Loading secrot routing JSON...')
    sources['routing'], _ = _load_latest_json(
        os.path.join(_DATA_DIR, 'output', 'backtest',
                     'secrot_backtest_v1_5_routing_*.json'))

    loaded = sum(1 for v in sources.values() if v is not None)
    print('  --> {}/{} sources loaded'.format(loaded, len(sources)))
    return sources


# ============================================================================
# SECTION 1: REGIME CONTEXT
# ============================================================================

def build_regime_context(sources):
    """Build stat bar values and routing banner data."""
    macro = sources.get('macro') or {}
    crash = sources.get('crash')
    changepoint = sources.get('changepoint') or {}
    routing_data = sources.get('routing') or {}
    hyglqd = sources.get('hyglqd')

    # Stat bar values
    vix = macro.get('volatility', {}).get('vix')
    breadth_pct = macro.get('breadth', {}).get('pct_above_200ma')
    hyg_lqd_ratio = macro.get('credit', {}).get('hyg_lqd_ratio')
    spread_10y_2y = macro.get('treasury', {}).get('spreads', {}).get('10y_2y')

    # From crash detection CSV
    lambda_max = None
    magnetization = None
    rmt_risk = None
    if crash is not None and len(crash) > 0:
        row = crash.iloc[-1]
        lambda_max = row.get('lambda_max')
        magnetization = row.get('magnetization')
        rmt_risk = row.get('rmt_risk', '')

    # From changepoint
    drift_tier = changepoint.get('drift_tier', 'UNKNOWN')
    norm_drift = changepoint.get('norm_drift')

    # Determine active routing method
    routing = routing_data.get('routing', {})
    active_method = routing.get(drift_tier, 'unknown')

    # Tier performance from routing JSON
    tier_breakdown = routing_data.get('tier_breakdown', {})
    tier_perf = tier_breakdown.get(drift_tier, {}).get(active_method, {})

    stats = [
        {'label': 'VIX',          'value': _fmt_num(vix, 2),
         'css': _css_class_for_value(vix, {'neg': 30, 'warn': 25})},
        {'label': 'Breadth %',    'value': _fmt_num(breadth_pct, 1) + '%' if breadth_pct is not None else '--',
         'css': _css_class_for_value(breadth_pct, {'pos': 50, 'warn': 30, 'neg': 20}) if breadth_pct is not None else 'neutral'},
        {'label': 'HYG/LQD',     'value': _fmt_num(hyg_lqd_ratio, 4),
         'css': _css_class_for_value(hyg_lqd_ratio, {'pos': 0.73, 'neg': 0.70})},
        {'label': '10y-2y',       'value': _fmt_num(spread_10y_2y, 2) + ' bps' if spread_10y_2y is not None else '--',
         'css': 'neutral'},
        {'label': 'Lambda Max',   'value': _fmt_num(lambda_max, 1),
         'css': 'neg' if rmt_risk == 'CRITICAL' else ('warn' if rmt_risk == 'ELEVATED' else 'neutral')},
        {'label': 'Magnetization','value': _fmt_num(magnetization, 3),
         'css': _css_class_for_value(magnetization, {'neg': 0.7, 'warn': 0.6})},
        {'label': 'Drift Tier',   'value': drift_tier,
         'css': {'STABLE': 'pos', 'NORMAL': 'neutral', 'ELEVATED': 'warn', 'UNSTABLE': 'neg'}.get(drift_tier, 'neutral')},
        {'label': 'Active Method','value': active_method.upper() if active_method else '--',
         'css': 'neutral'},
    ]

    # Banner
    banner_color = {
        'STABLE': '#16a34a', 'NORMAL': '#2563eb',
        'ELEVATED': '#ea580c', 'UNSTABLE': '#dc2626',
    }.get(drift_tier, '#6b7280')

    mean_21d = tier_perf.get('mean_21d')
    hit_10d = tier_perf.get('hit_10d')
    n_periods = tier_perf.get('n')

    return {
        'stats': stats,
        'drift_tier': drift_tier,
        'active_method': active_method,
        'banner_color': banner_color,
        'mean_21d': mean_21d,
        'hit_rate': hit_10d,
        'n_periods': n_periods,
        'norm_drift': norm_drift,
    }


# ============================================================================
# SECTION 2: AGREEMENT MATRIX
# ============================================================================

def build_agreement_matrix(sources, exclude=None):
    """Build cross-dashboard agreement matrix from ticker-level sources.

    *exclude* is an optional set of source keys to skip
    (e.g. {'advanced', 'qualifier'} for trusted-only mode).
    """
    exclude = exclude or set()
    tickers = {}  # ticker -> {source_name: data}

    # 1. Momentum Ranker (short-term): rank <= 150
    if 'ranker' not in exclude:
        ranker = sources.get('ranker')
        if ranker and 'data' in ranker:
            for item in ranker['data']:
                if item.get('rank', 9999) <= 150:
                    t = item['ticker']
                    tickers.setdefault(t, {})['ranker'] = item

    # 1b. Momentum Ranker (long-term): rank <= 150
    if 'ranker_long' not in exclude:
        ranker_long = sources.get('ranker_long')
        if ranker_long and 'data' in ranker_long:
            for item in ranker_long['data']:
                if item.get('rank', 9999) <= 150:
                    t = item['ticker']
                    tickers.setdefault(t, {})['ranker_long'] = item

    # 2. Advanced Momentum: signal in (STRONG_BUY, BUY)
    if 'advanced' not in exclude:
        advanced = sources.get('advanced')
        if advanced and 'results' in advanced:
            for item in advanced['results']:
                if item.get('signal') in ('STRONG_BUY', 'BUY'):
                    t = item['symbol']
                    tickers.setdefault(t, {})['advanced'] = item

    # 3. Momentum Qualifier: all present
    if 'qualifier' not in exclude:
        qualifier = sources.get('qualifier')
        if qualifier is not None:
            for _, row in qualifier.iterrows():
                t = row['ticker']
                tickers.setdefault(t, {})['qualifier'] = row.to_dict()

    # 4. Sector Rotation: sr_score > 0 (mostly ETFs)
    secrot = sources.get('secrot')
    if secrot is not None:
        for _, row in secrot.iterrows():
            if row.get('sr_score', 0) > 0:
                t = row['ticker']
                tickers.setdefault(t, {})['secrot'] = row.to_dict()

    # 5. Stock SecRot: total_score >= 5 counts as a source
    stock_sr = sources.get('stock_secrot')
    stock_sr_lookup = {}  # symbol -> row dict (full universe for display)
    if stock_sr is not None:
        for _, row in stock_sr.iterrows():
            t = row['symbol']
            stock_sr_lookup[t] = row.to_dict()
            if row.get('total_score', 0) >= 5:
                tickers.setdefault(t, {})['stock_secrot'] = row.to_dict()

    # 6. Slope Stage: Stage 2/3 or recent 1->2 transition counts as source
    if 'slope_stage' not in exclude:
        slope_data = sources.get('slope_stage')
        if slope_data and 'results' in slope_data:
            for item in slope_data['results']:
                stage = item.get('stage')
                has_transition = bool(item.get('transitions_1_to_2'))
                if stage in (2, 3) or has_transition:
                    t = item['ticker']
                    tickers.setdefault(t, {})['slope_stage'] = item

    # 7. Pullback Health: health >= 80 (HEALTHY verdict) counts as source
    health_data = sources.get('pullback_health')
    health_lookup = {}  # ticker -> item (full universe for display)
    if health_data and 'results' in health_data:
        for item in health_data['results']:
            t = item.get('ticker')
            if t:
                health_lookup[t] = item
                if (item.get('health') or 0) >= 80:
                    tickers.setdefault(t, {})['pullback_health'] = item

    # Filter to 2+ sources
    multi = {}
    for t, src_dict in tickers.items():
        if len(src_dict) >= 2:
            multi[t] = src_dict

    # Build spread support lookup
    spreads_df = sources.get('spreads')
    spread_signals = {}  # spread_name -> signal_score
    if spreads_df is not None:
        for _, row in spreads_df.iterrows():
            name = row.get('ratio_name', '')
            score = row.get('signal_score', 0)
            spread_signals[name] = score

    # Sort: source count desc, then ranker score desc
    def sort_key(item):
        t, src_dict = item
        count = len(src_dict)
        ranker_score = src_dict.get('ranker', {}).get('score', 0) if isinstance(src_dict.get('ranker'), dict) else 0
        return (-count, -ranker_score)

    sorted_tickers = sorted(multi.items(), key=sort_key)

    # Build rows
    rows = []
    for ticker, src_dict in sorted_tickers:
        # Determine sector from stock_secrot (full lookup) or top_pole from ranker
        sector = None
        sr_data = src_dict.get('stock_secrot') or stock_sr_lookup.get(ticker)
        if sr_data:
            sector = sr_data.get('sector')
        pole = None
        if 'ranker' in src_dict:
            pole = src_dict['ranker'].get('top_pole')

        # Check spread support
        spread_support = _check_spread_support(sector, pole, spread_signals)

        # Health data (show for all tickers in matrix, not just source-qualifying)
        h_data = src_dict.get('pullback_health') or health_lookup.get(ticker)

        row = {
            'ticker': ticker,
            'pole': pole or '--',
            'ranker_rank': src_dict['ranker']['rank'] if 'ranker' in src_dict else None,
            'ranker_score': src_dict['ranker'].get('score') if 'ranker' in src_dict else None,
            'ranker_long_rank': src_dict['ranker_long']['rank'] if 'ranker_long' in src_dict else None,
            'ranker_long_score': src_dict['ranker_long'].get('score') if 'ranker_long' in src_dict else None,
            'advanced_signal': src_dict['advanced']['signal'] if 'advanced' in src_dict else None,
            'advanced_confidence': src_dict['advanced'].get('confidence') if 'advanced' in src_dict else None,
            'qualifier_safe': src_dict['qualifier'].get('is_safe') if 'qualifier' in src_dict else None,
            'qualifier_regime': src_dict['qualifier'].get('regime') if 'qualifier' in src_dict else None,
            'secrot_pred5d': src_dict['secrot'].get('pred_5d_avg') if 'secrot' in src_dict else None,
            'secrot_score': src_dict['secrot'].get('sr_score') if 'secrot' in src_dict else None,
            'stock_sr_score': sr_data.get('total_score') if sr_data else None,
            'slope_stage': src_dict['slope_stage'].get('stage') if 'slope_stage' in src_dict else None,
            'slope_tq_score': src_dict['slope_stage'].get('tq_score') if 'slope_stage' in src_dict else None,
            'health_score': h_data.get('health') if h_data else None,
            'health_verdict': h_data.get('verdict') if h_data else None,
            'spread_support': spread_support,
            'source_count': len(src_dict),
            'sources': list(src_dict.keys()),
            'sector': sector,
        }
        rows.append(row)

    return rows


def _check_spread_support(sector, pole, spread_signals):
    """Check if intermarket spreads support the given sector/pole."""
    if not sector and not pole:
        return 'none'

    # Convert pole ETF ticker to sector name
    pole_sector = POLE_TO_SECTOR.get(pole) if pole else None

    # Check sector, pole-mapped-sector
    relevant_spreads = set()
    if sector and sector in SECTOR_SPREAD_MAP:
        relevant_spreads.update(SECTOR_SPREAD_MAP[sector])
    if pole_sector and pole_sector in SECTOR_SPREAD_MAP:
        relevant_spreads.update(SECTOR_SPREAD_MAP[pole_sector])

    if not relevant_spreads:
        return 'none'

    bullish = 0
    bearish = 0
    for sp in relevant_spreads:
        score = spread_signals.get(sp, 0)
        if score > 0:
            bullish += 1
        elif score < 0:
            bearish += 1

    if bullish > bearish:
        return 'bullish'
    elif bearish > bullish:
        return 'bearish'
    elif bullish > 0:
        return 'mixed'
    return 'none'


# ============================================================================
# SECTION 3: INTERMARKET CONTEXT
# ============================================================================

def build_intermarket_context(sources):
    """Group spreads into 4 force categories with net direction."""
    spreads_df = sources.get('spreads')
    if spreads_df is None:
        return {'forces': [], 'sector_conviction': {}}

    forces = {}
    for _, row in spreads_df.iterrows():
        level = str(row.get('market_level', '')).strip()
        if not level:
            # Derive from category column
            cat = str(row.get('category', ''))
            if 'Rate' in cat or 'Duration' in cat:
                level = 'I: Rates / Duration'
            elif 'Earn' in cat or 'Fund' in cat:
                level = 'II: Earnings / Fundamentals'
            elif 'Liq' in cat or 'Struct' in cat:
                level = 'III: Liquidity / Structure'
            else:
                level = 'IV: Sentiment / Positioning'

        forces.setdefault(level, []).append(row.to_dict())

    force_cards = []
    for level_name in sorted(forces.keys()):
        rows = forces[level_name]
        bullish = sum(1 for r in rows if r.get('signal_score', 0) > 0)
        bearish = sum(1 for r in rows if r.get('signal_score', 0) < 0)
        neutral = len(rows) - bullish - bearish

        if bullish > bearish:
            direction = 'bullish'
            arrow = 'up'
        elif bearish > bullish:
            direction = 'bearish'
            arrow = 'down'
        else:
            direction = 'neutral'
            arrow = 'flat'

        # Top signal (highest absolute score)
        top = max(rows, key=lambda r: abs(r.get('signal_score', 0)))

        force_cards.append({
            'level': level_name,
            'bullish': bullish,
            'bearish': bearish,
            'neutral': neutral,
            'direction': direction,
            'arrow': arrow,
            'top_spread': top.get('ratio_name', '--'),
            'top_score': top.get('signal_score', 0),
            'top_playbook': top.get('playbook', ''),
            'top_trend': top.get('trend_qualifier', ''),
        })

    # Sector conviction from spread signals
    sector_conviction = {}
    for _, row in spreads_df.iterrows():
        name = row.get('ratio_name', '')
        score = row.get('signal_score', 0)
        if name in SPREAD_SECTOR_MAP:
            for sec in SPREAD_SECTOR_MAP[name]:
                sector_conviction.setdefault(sec, []).append(score)

    # Net score per sector
    sector_net = {}
    for sec, scores in sector_conviction.items():
        net = sum(scores)
        total = len(scores)
        sector_net[sec] = {
            'net': net,
            'bullish': sum(1 for s in scores if s > 0),
            'bearish': sum(1 for s in scores if s < 0),
            'total': total,
        }

    return {
        'forces': force_cards,
        'sector_conviction': sector_net,
    }


# ============================================================================
# SECTION 4: PATTERN CONTEXT
# ============================================================================

def build_pattern_context(sources):
    """Similar days consensus, sector rotation predictions, changepoint."""
    result = {
        'similar_days': [],
        'similar_consensus': None,
        'secrot_predictions': [],
        'changepoint': {},
    }

    # Similar Days
    similar = sources.get('similar')
    if similar is not None and len(similar) > 0:
        windows = similar['window'].unique() if 'window' in similar.columns else []
        window_data = []
        for w in windows:
            w_df = similar[similar['window'] == w].head(3)
            matches = []
            for _, row in w_df.iterrows():
                matches.append({
                    'date': str(row.get('similar_date', '')),
                    'correlation': row.get('correlation', 0),
                })
            window_data.append({
                'window': w,
                'matches': matches,
            })
        result['similar_days'] = window_data

        # Consensus: check if 2+ windows agree on direction
        # (we don't have forward return in similar_days CSV, so just report top correlations)
        if len(window_data) >= 2:
            result['similar_consensus'] = '{} windows with matches'.format(len(window_data))

    # Sector Rotation predictions: highest pred_5d_winrate
    secrot = sources.get('secrot')
    if secrot is not None and 'pred_5d_winrate' in secrot.columns:
        valid = secrot[secrot['pred_5d_winrate'].notna() & (secrot['pred_5d_winrate'] > 55)]
        top = valid.nlargest(10, 'pred_5d_winrate') if len(valid) > 0 else pd.DataFrame()
        for _, row in top.iterrows():
            result['secrot_predictions'].append({
                'ticker': row['ticker'],
                'name': row.get('name', ''),
                'pred_5d_avg': row.get('pred_5d_avg'),
                'pred_5d_winrate': row.get('pred_5d_winrate'),
                'pred_10d_avg': row.get('pred_10d_avg'),
                'pred_10d_winrate': row.get('pred_10d_winrate'),
                'sr_score': row.get('sr_score'),
            })

    # Changepoint
    changepoint = sources.get('changepoint') or {}
    result['changepoint'] = {
        'top_regime': changepoint.get('top_regime', '--'),
        'top_sim': changepoint.get('top_sim', 0),
        'similarity': changepoint.get('regime_similarity', []),
    }

    return result


# ============================================================================
# SECTION 5: RISK FLAGS
# ============================================================================

def build_risk_flags(sources, agreement_rows, trusted_only=False):
    """Build risk flags table sorted by severity.

    When *trusted_only* is True, skip flags sourced from Advanced Momentum,
    Momentum Qualifier, and Macro Dashboard composites.
    """
    flags = []

    # 1. Parabolic tickers (qualifier: recency_ratio > 3.0)
    if not trusted_only:
        qualifier = sources.get('qualifier')
        if qualifier is not None:
            parabolic = qualifier[qualifier['recency_ratio'] > 3.0] if 'recency_ratio' in qualifier.columns else pd.DataFrame()
            for _, row in parabolic.iterrows():
                flags.append({
                    'flag': 'Parabolic: {}'.format(row['ticker']),
                    'source': 'Momentum Qualifier',
                    'detail': 'Recency ratio: {:.1f}'.format(row['recency_ratio']),
                    'severity': 'high',
                })

    # 2. Overextended tickers (qualifier: extension > 0.15 = 15%)
    if not trusted_only:
        qualifier = sources.get('qualifier')
        if qualifier is not None and 'extension_from_sma29_pct' in qualifier.columns:
            overext = qualifier[qualifier['extension_from_sma29_pct'] > 0.15]
            for _, row in overext.head(10).iterrows():
                flags.append({
                    'flag': 'Overextended: {}'.format(row['ticker']),
                    'source': 'Momentum Qualifier',
                    'detail': 'Extension: {:.1f}%'.format(row['extension_from_sma29_pct'] * 100),
                    'severity': 'medium',
                })

    # 3. OBV divergence (advanced: has_divergence=True)
    if not trusted_only:
        advanced = sources.get('advanced')
        if advanced and 'results' in advanced:
            divs = [r for r in advanced['results']
                    if r.get('has_divergence') and r.get('signal') in ('STRONG_BUY', 'BUY', 'HOLD')]
            for item in divs[:10]:
                flags.append({
                    'flag': 'OBV Divergence: {}'.format(item['symbol']),
                    'source': 'Advanced Momentum',
                    'detail': 'Signal: {} | Type: {}'.format(
                        item.get('signal', ''), item.get('divergence_type', '')),
                    'severity': 'medium',
                })

    # 3b. Slope Stage parabolic (stage 3 with high crash risk)
    if not trusted_only:
        slope_data = sources.get('slope_stage')
        if slope_data and 'results' in slope_data:
            parabolic_items = [r for r in slope_data['results']
                               if r.get('stage') == 3 and r.get('crash_risk', 0) >= 40]
            parabolic_items.sort(key=lambda x: x.get('crash_risk', 0), reverse=True)
            for item in parabolic_items[:10]:
                flags.append({
                    'flag': 'Parabolic (Slope): {}'.format(item['ticker']),
                    'source': 'Slope Stage Scanner',
                    'detail': 'Stage 3 | Slope: {:.0f}% | Crash risk: {:.0f}'.format(
                        item.get('slope_pct', 0), item.get('crash_risk', 0)),
                    'severity': 'high' if item.get('crash_risk', 0) >= 60 else 'medium',
                })

    # 4. Credit/equity divergence -- uses raw HYG/LQD ratio (trusted)
    # but SPY trend from macro; keep in trusted since it's a raw data check
    macro = sources.get('macro') or {}
    spy_trend = macro.get('indices', {}).get('spy_trend', {})
    hyg_lqd = macro.get('credit', {}).get('hyg_lqd_ratio')
    if spy_trend.get('above_200ma') and hyg_lqd is not None and hyg_lqd < 0.71:
        flags.append({
            'flag': 'Credit/Equity Divergence',
            'source': 'Macro + Credit',
            'detail': 'SPY above 200MA but HYG/LQD at {:.4f} (< 0.71)'.format(hyg_lqd),
            'severity': 'high',
        })

    # 5. Breadth deterioration -- raw breadth count, keep in trusted
    breadth_pct = macro.get('breadth', {}).get('pct_above_200ma')
    if breadth_pct is not None and breadth_pct < 30:
        flags.append({
            'flag': 'Breadth Deterioration',
            'source': 'Macro (raw)',
            'detail': '{:.1f}% above 200MA (< 30% threshold)'.format(breadth_pct),
            'severity': 'medium',
        })

    # 6. RMT critical
    crash = sources.get('crash')
    if crash is not None and len(crash) > 0:
        rmt_risk = crash.iloc[-1].get('rmt_risk', '')
        if rmt_risk == 'CRITICAL':
            flags.append({
                'flag': 'RMT Critical',
                'source': 'Crash Detection',
                'detail': 'Lambda max: {:.1f}'.format(crash.iloc[-1].get('lambda_max', 0)),
                'severity': 'high',
            })

    # 7. Conflicting signals in agreement matrix
    if not trusted_only:
        for row in agreement_rows:
            has_buy = row.get('advanced_signal') in ('STRONG_BUY', 'BUY')
            sources_list = row.get('sources', [])
            if len(sources_list) >= 3:
                if ('ranker' in sources_list and 'advanced' in sources_list
                        and 'qualifier' in sources_list
                        and has_buy and row.get('qualifier_safe') is False):
                    flags.append({
                        'flag': 'Conflicting: {}'.format(row['ticker']),
                        'source': 'Cross-Dashboard',
                        'detail': 'BUY signal but qualifier says NOT SAFE (regime: {})'.format(
                            row.get('qualifier_regime', '')),
                        'severity': 'medium',
                    })

    # Sort by severity
    severity_order = {'high': 0, 'medium': 1, 'low': 2}
    flags.sort(key=lambda f: severity_order.get(f['severity'], 3))

    return flags


# ============================================================================
# HTML BUILDERS
# ============================================================================

def _build_stat_bar(regime):
    """Build the dark stat bar at the top."""
    cells = []
    for s in regime['stats']:
        cells.append(
            '<div class="stat">'
            '<div class="stat-label">{label}</div>'
            '<div class="stat-value {css}">{value}</div>'
            '</div>'.format(**s)
        )
    return '<div class="stat-bar">{}</div>'.format('\n'.join(cells))


def _build_routing_banner(regime):
    """Build the regime routing banner below the stat bar."""
    dt = regime['drift_tier']
    method = regime['active_method']
    color = regime['banner_color']

    perf_html = ''
    if regime['mean_21d'] is not None:
        perf_html = (
            '<span class="banner-perf">'
            'Historical 21d return: {:+.2f}%'.format(regime['mean_21d'])
        )
        if regime['hit_rate'] is not None:
            perf_html += ' | Hit rate: {:.1f}%'.format(regime['hit_rate'] * 100)
        if regime['n_periods'] is not None:
            perf_html += ' | n={}'.format(regime['n_periods'])
        perf_html += '</span>'

    drift_str = ''
    if regime['norm_drift'] is not None:
        drift_str = ' ({:.2f}x drift)'.format(regime['norm_drift'])

    return (
        '<div class="routing-banner" style="background:{color};">'
        '<span class="banner-tier">{dt}</span>'
        '<span class="banner-arrow">--></span>'
        '<span class="banner-method">Routing to {method} method{drift}</span>'
        '{perf}'
        '</div>'
    ).format(color=color, dt=dt, method=method.upper(),
             drift=drift_str, perf=perf_html)


def _build_agreement_table(rows, mode='full'):
    """Build the agreement matrix table.

    *mode* = 'full' -> 8 ticker-level columns (Ranker S/T, Ranker L/T, Advanced, Qualifier, SecRot, Stock SR, Slope Stage, Health)
    *mode* = 'trusted' -> 5 ticker-level columns (Ranker S/T, Ranker L/T, SecRot, Stock SR, Health)
    """
    if not rows:
        return '<p>No tickers appear in 2+ sources.</p>'

    trusted = (mode == 'trusted')
    n_sources = 5 if trusted else 8

    if trusted:
        header = (
            '<table class="sortable-table data-table">'
            '<thead><tr>'
            '<th class="own-th" title="Mark tickers you own">Own</th>'
            '<th class="own-th" title="Mark tickers you are watching">Watch</th>'
            '<th title="Ticker symbol">Ticker</th>'
            '<th title="Top correlated sector/ETF pole from ranker">Pole</th>'
            '<th title="Short-term momentum rank (1d-1y returns). Green if top 50">Ranker S/T</th>'
            '<th title="Long-term momentum rank (1m-10y returns). Green if top 50">Ranker L/T</th>'
            '<th title="Sector rotation momentum score from ETF-level analysis">SecRot</th>'
            '<th title="Stock-level sector rotation score (out of 9 patterns)">Stock SR</th>'
            '<th title="Pullback health score 0-100 (NATR drawdown + SMA structure + slope + vol + residual DD + historical recovery). Green if HEALTHY (>=80)">Health</th>'
            '<th title="Intermarket spread support: bullish/bearish/mixed/none">Spread</th>'
            '<th title="Number of independent sources confirming this ticker">Sources</th>'
            '</tr></thead><tbody>'
        )
    else:
        header = (
            '<table class="sortable-table data-table">'
            '<thead><tr>'
            '<th class="own-th" title="Mark tickers you own">Own</th>'
            '<th class="own-th" title="Mark tickers you are watching">Watch</th>'
            '<th title="Ticker symbol">Ticker</th>'
            '<th title="Top correlated sector/ETF pole from ranker">Pole</th>'
            '<th title="Short-term momentum rank (1d-1y returns). Green if top 50">Ranker S/T</th>'
            '<th title="Long-term momentum rank (1m-10y returns). Green if top 50">Ranker L/T</th>'
            '<th title="Advanced momentum signal: OBV divergence + Sortino + trajectory">Advanced</th>'
            '<th title="Conservative momentum qualifier: safe/not-safe with regime context">Qualifier</th>'
            '<th title="Sector rotation momentum score from ETF-level analysis">SecRot</th>'
            '<th title="Stock-level sector rotation score (out of 9 patterns)">Stock SR</th>'
            '<th title="Slope Stage: 90-day trendline stage (2=Uptrend, 3=Parabolic) + TQ score">Slope</th>'
            '<th title="Pullback health score 0-100 (NATR drawdown + SMA structure + slope + vol + residual DD + historical recovery). Green if HEALTHY (>=80)">Health</th>'
            '<th title="Intermarket spread support: bullish/bearish/mixed/none">Spread</th>'
            '<th title="Number of independent sources confirming this ticker">Sources</th>'
            '</tr></thead><tbody>'
        )

    body_rows = []
    for r in rows:
        # Ranker cell
        if r['ranker_rank'] is not None:
            ranker_css = 'agree-bull' if r['ranker_rank'] <= 50 else 'agree-present'
            ranker_val = '#{}'.format(r['ranker_rank'])
        else:
            ranker_css = 'agree-none'
            ranker_val = '--'

        # SecRot cell
        if r['secrot_score'] is not None:
            secrot_css = 'agree-bull' if r['secrot_score'] >= 5 else 'agree-present'
            secrot_val = '{:.1f}'.format(r['secrot_score'])
        else:
            secrot_css = 'agree-none'
            secrot_val = '--'

        # Stock SR cell
        if r['stock_sr_score'] is not None:
            ssr_css = 'agree-bull' if r['stock_sr_score'] >= 8 else 'agree-present'
            ssr_val = '{}/9'.format(int(r['stock_sr_score']))
        else:
            ssr_css = 'agree-none'
            ssr_val = '--'

        # Spread support
        sp = r.get('spread_support', 'none')
        sp_css = {'bullish': 'agree-bull', 'bearish': 'agree-bear',
                  'mixed': 'agree-warn', 'none': 'agree-none'}.get(sp, 'agree-none')
        sp_icon = {'bullish': '[+]', 'bearish': '[-]',
                   'mixed': '[~]', 'none': '--'}.get(sp, '--')

        # Count
        count = r['source_count']
        count_css = 'badge-high' if count >= n_sources - 1 else ('badge-mid' if count >= 2 else 'badge-low')

        # Long ranker cell
        if r['ranker_long_rank'] is not None:
            rl_css = 'agree-bull' if r['ranker_long_rank'] <= 50 else 'agree-present'
            rl_val = '#{}'.format(r['ranker_long_rank'])
        else:
            rl_css = 'agree-none'
            rl_val = '--'

        # Health cell (both modes)
        hs = r.get('health_score')
        hv = r.get('health_verdict', '')
        if hs is not None:
            if hs >= 80:
                health_css = 'agree-bull'
            elif hs >= 50:
                health_css = 'agree-present'
            else:
                health_css = 'agree-warn'
            health_val = '{:.0f}'.format(hs)
            if hv:
                health_val += ' ({})'.format(hv[:4])
            health_sort = hs
        else:
            health_css = 'agree-none'
            health_val = '--'
            health_sort = 0

        if trusted:
            body_rows.append(
                '<tr>'
                '<td><input type="checkbox" class="own-cb" data-ticker="{ticker}"'
                ' onclick="window._ownToggle(\'{ticker}\', this)" title="Mark as owned"></td>'
                '<td><input type="checkbox" class="watch-cb" data-ticker="{ticker}"'
                ' onclick="window._watchToggle(\'{ticker}\', this)" title="Mark as watched"></td>'
                '<td class="ticker-cell" data-sort="{ticker}">{ticker}</td>'
                '<td>{pole}</td>'
                '<td class="{ranker_css}" data-sort="{ranker_sort}">{ranker_val}</td>'
                '<td class="{rl_css}" data-sort="{rl_sort}">{rl_val}</td>'
                '<td class="{secrot_css}" data-sort="{secrot_sort}">{secrot_val}</td>'
                '<td class="{ssr_css}" data-sort="{ssr_sort}">{ssr_val}</td>'
                '<td class="{health_css}" data-sort="{health_sort}">{health_val}</td>'
                '<td class="{sp_css}">{sp_icon}</td>'
                '<td><span class="count-badge {count_css}">{count}/{n_src}</span></td>'
                '</tr>'.format(
                    ticker=r['ticker'],
                    pole=r['pole'],
                    ranker_css=ranker_css,
                    ranker_sort=r['ranker_rank'] if r['ranker_rank'] else 9999,
                    ranker_val=ranker_val,
                    rl_css=rl_css,
                    rl_sort=r['ranker_long_rank'] if r['ranker_long_rank'] else 9999,
                    rl_val=rl_val,
                    secrot_css=secrot_css,
                    secrot_sort=r['secrot_score'] if r['secrot_score'] else 0,
                    secrot_val=secrot_val,
                    ssr_css=ssr_css,
                    ssr_sort=r['stock_sr_score'] if r['stock_sr_score'] else 0,
                    ssr_val=ssr_val,
                    health_css=health_css,
                    health_sort=health_sort,
                    health_val=health_val,
                    sp_css=sp_css,
                    sp_icon=sp_icon,
                    count=count,
                    count_css=count_css,
                    n_src=n_sources,
                )
            )
        else:
            # Advanced cell
            if r['advanced_signal']:
                adv_css = 'agree-bull' if r['advanced_signal'] in ('STRONG_BUY', 'BUY') else 'agree-present'
                adv_val = r['advanced_signal']
                if r['advanced_confidence']:
                    adv_val += ' ({}%)'.format(r['advanced_confidence'])
            else:
                adv_css = 'agree-none'
                adv_val = '--'

            # Qualifier cell
            if r['qualifier_safe'] is not None:
                if r['qualifier_safe'] is True or r['qualifier_safe'] == 'True':
                    qual_css = 'agree-bull'
                    qual_val = 'SAFE'
                else:
                    qual_css = 'agree-warn'
                    qual_val = 'NOT SAFE'
                if r['qualifier_regime']:
                    qual_val += ' ({})'.format(r['qualifier_regime'])
            else:
                qual_css = 'agree-none'
                qual_val = '--'

            # Slope Stage cell
            if r['slope_stage'] is not None:
                ss = r['slope_stage']
                tq = r.get('slope_tq_score')
                if ss == 2:
                    slope_css = 'agree-bull'
                    slope_val = 'S2'
                elif ss == 3:
                    slope_css = 'agree-warn'
                    slope_val = 'S3'
                else:
                    slope_css = 'agree-present'
                    slope_val = 'S{}'.format(ss)
                if tq is not None:
                    slope_val += ' ({:.0f})'.format(tq)
                slope_sort = tq if tq else 0
            else:
                slope_css = 'agree-none'
                slope_val = '--'
                slope_sort = 0

            body_rows.append(
                '<tr>'
                '<td><input type="checkbox" class="own-cb" data-ticker="{ticker}"'
                ' onclick="window._ownToggle(\'{ticker}\', this)" title="Mark as owned"></td>'
                '<td><input type="checkbox" class="watch-cb" data-ticker="{ticker}"'
                ' onclick="window._watchToggle(\'{ticker}\', this)" title="Mark as watched"></td>'
                '<td class="ticker-cell" data-sort="{ticker}">{ticker}</td>'
                '<td>{pole}</td>'
                '<td class="{ranker_css}" data-sort="{ranker_sort}">{ranker_val}</td>'
                '<td class="{rl_css}" data-sort="{rl_sort}">{rl_val}</td>'
                '<td class="{adv_css}">{adv_val}</td>'
                '<td class="{qual_css}">{qual_val}</td>'
                '<td class="{secrot_css}" data-sort="{secrot_sort}">{secrot_val}</td>'
                '<td class="{ssr_css}" data-sort="{ssr_sort}">{ssr_val}</td>'
                '<td class="{slope_css}" data-sort="{slope_sort}">{slope_val}</td>'
                '<td class="{health_css}" data-sort="{health_sort}">{health_val}</td>'
                '<td class="{sp_css}">{sp_icon}</td>'
                '<td><span class="count-badge {count_css}">{count}/{n_src}</span></td>'
                '</tr>'.format(
                    ticker=r['ticker'],
                    pole=r['pole'],
                    ranker_css=ranker_css,
                    ranker_sort=r['ranker_rank'] if r['ranker_rank'] else 9999,
                    ranker_val=ranker_val,
                    rl_css=rl_css,
                    rl_sort=r['ranker_long_rank'] if r['ranker_long_rank'] else 9999,
                    rl_val=rl_val,
                    adv_css=adv_css,
                    adv_val=adv_val,
                    qual_css=qual_css,
                    qual_val=qual_val,
                    secrot_css=secrot_css,
                    secrot_sort=r['secrot_score'] if r['secrot_score'] else 0,
                    secrot_val=secrot_val,
                    ssr_css=ssr_css,
                    ssr_sort=r['stock_sr_score'] if r['stock_sr_score'] else 0,
                    ssr_val=ssr_val,
                    slope_css=slope_css,
                    slope_sort=slope_sort,
                    slope_val=slope_val,
                    health_css=health_css,
                    health_sort=health_sort,
                    health_val=health_val,
                    sp_css=sp_css,
                    sp_icon=sp_icon,
                    count=count,
                    count_css=count_css,
                    n_src=n_sources,
                )
            )

    return (
        '{header}'
        '{rows}'
        '</tbody></table>'
    ).format(header=header, rows='\n'.join(body_rows))


def _build_intermarket_section(intermarket):
    """Build intermarket force cards and sector conviction row."""
    if not intermarket['forces']:
        return '<div class="card"><p>No spread data available.</p></div>'

    cards = []
    for fc in intermarket['forces']:
        arrow_html = {
            'up': '<span class="force-arrow pos">&#9650;</span>',
            'down': '<span class="force-arrow neg">&#9660;</span>',
            'flat': '<span class="force-arrow neutral">&#9644;</span>',
        }.get(fc['arrow'], '')

        cards.append(
            '<div class="force-card force-{dir}">'
            '<div class="force-header">'
            '<span class="force-level">{level}</span>'
            '{arrow}'
            '</div>'
            '<div class="force-counts">'
            '<span class="pos">{bull} bullish</span> / '
            '<span class="neg">{bear} bearish</span> / '
            '<span class="neutral">{neut} neutral</span>'
            '</div>'
            '<div class="force-top">'
            '<strong>{top}</strong> (score: {score})'
            '</div>'
            '<div class="force-playbook">{playbook}</div>'
            '</div>'.format(
                dir=fc['direction'],
                level=fc['level'],
                arrow=arrow_html,
                bull=fc['bullish'],
                bear=fc['bearish'],
                neut=fc['neutral'],
                top=fc['top_spread'],
                score=fc['top_score'],
                playbook=fc['top_playbook'][:200] if fc['top_playbook'] else '',
            )
        )

    # Sector conviction row
    sc = intermarket.get('sector_conviction', {})
    sector_pills = []
    for sec in sorted(sc.keys()):
        data = sc[sec]
        net = data['net']
        css = 'pos' if net > 0 else ('neg' if net < 0 else 'neutral')
        sector_pills.append(
            '<span class="sector-pill {css}">'
            '{sec}: {net:+.0f} ({bull}B/{bear}R)'
            '</span>'.format(
                css=css, sec=sec, net=net,
                bull=data['bullish'], bear=data['bearish'],
            )
        )

    return (
        '<div class="card">'
        '<div class="card-header"><h2>Intermarket Context</h2></div>'
        '<div class="cards-grid">{cards}</div>'
        '<div class="sector-row">'
        '<h3>Sector Conviction (from spreads)</h3>'
        '<div class="pill-row">{pills}</div>'
        '</div>'
        '</div>'
    ).format(cards='\n'.join(cards), pills='\n'.join(sector_pills))


def _build_pattern_section(pattern):
    """Build pattern context section."""
    parts = []

    # Similar Days
    if pattern['similar_days']:
        sim_html = '<div class="pattern-group"><h3>Similar Days</h3>'
        for wd in pattern['similar_days']:
            sim_html += '<div class="pattern-window"><strong>{}</strong>: '.format(wd['window'])
            match_strs = []
            for m in wd['matches']:
                match_strs.append('{} ({:.2f})'.format(m['date'], m['correlation']))
            sim_html += ', '.join(match_strs) + '</div>'
        if pattern['similar_consensus']:
            sim_html += '<div class="pattern-consensus">{}</div>'.format(
                pattern['similar_consensus'])
        sim_html += '</div>'
        parts.append(sim_html)

    # Sector Rotation predictions
    if pattern['secrot_predictions']:
        sr_html = '<div class="pattern-group"><h3>Sector Rotation Predictions (5d WR > 55%)</h3>'
        sr_html += '<table class="data-table compact"><thead><tr>'
        sr_html += ('<th title="ETF ticker symbol">Ticker</th>'
                    '<th title="ETF name">Name</th>'
                    '<th title="Average predicted 5-day forward return">5d Avg</th>'
                    '<th title="Win rate of 5-day forward predictions (% positive)">5d WR</th>'
                    '<th title="Average predicted 10-day forward return">10d Avg</th>'
                    '<th title="Win rate of 10-day forward predictions (% positive)">10d WR</th>'
                    '<th title="Sector rotation composite score">SR Score</th>')
        sr_html += '</tr></thead><tbody>'
        for p in pattern['secrot_predictions']:
            sr_html += '<tr>'
            sr_html += '<td class="ticker-cell">{}</td>'.format(p['ticker'])
            sr_html += '<td>{}</td>'.format(p.get('name', ''))
            sr_html += '<td>{}</td>'.format(_fmt_pct(p.get('pred_5d_avg')))
            sr_html += '<td>{}</td>'.format(_fmt_num(p.get('pred_5d_winrate'), 1) + '%' if p.get('pred_5d_winrate') is not None else '--')
            sr_html += '<td>{}</td>'.format(_fmt_pct(p.get('pred_10d_avg')))
            sr_html += '<td>{}</td>'.format(_fmt_num(p.get('pred_10d_winrate'), 1) + '%' if p.get('pred_10d_winrate') is not None else '--')
            sr_html += '<td>{}</td>'.format(_fmt_num(p.get('sr_score'), 1))
            sr_html += '</tr>'
        sr_html += '</tbody></table></div>'
        parts.append(sr_html)

    # Changepoint
    cp = pattern.get('changepoint', {})
    if cp.get('top_regime', '--') != '--':
        cp_html = '<div class="pattern-group"><h3>Changepoint Regime Similarity</h3>'
        cp_html += '<p>Current regime most similar to: <strong>{}</strong> ({:.1f}%)</p>'.format(
            cp['top_regime'], cp['top_sim'])
        if cp.get('similarity'):
            cp_html += '<table class="data-table compact"><thead><tr>'
            cp_html += ('<th title="Historical regime period name">Period</th>'
                        '<th title="Market context description for this period">Description</th>'
                        '<th title="Cosine similarity to current regime fingerprint (%)">Similarity</th>')
            cp_html += '</tr></thead><tbody>'
            for s in cp['similarity'][:5]:
                cp_html += '<tr><td>{}</td><td>{}</td><td>{:.1f}%</td></tr>'.format(
                    s.get('name', ''), s.get('desc', ''), s.get('sim', 0))
            cp_html += '</tbody></table>'
        cp_html += '</div>'
        parts.append(cp_html)

    if not parts:
        return '<div class="card"><p>No pattern data available.</p></div>'

    return (
        '<div class="card">'
        '<div class="card-header"><h2>Pattern Context</h2></div>'
        '{}'
        '</div>'
    ).format('\n'.join(parts))


def _build_risk_flags(flags):
    """Build risk flags table."""
    if not flags:
        return '<div class="card"><p>No risk flags detected.</p></div>'

    rows = []
    for f in flags:
        sev_css = 'risk-{}'.format(f['severity'])
        sev_label = f['severity'].upper()
        rows.append(
            '<tr class="{sev_css}">'
            '<td><span class="sev-badge sev-{sev}">{sev_label}</span></td>'
            '<td>{flag}</td>'
            '<td>{source}</td>'
            '<td>{detail}</td>'
            '</tr>'.format(
                sev_css=sev_css,
                sev=f['severity'],
                sev_label=sev_label,
                flag=f['flag'],
                source=f['source'],
                detail=f['detail'],
            )
        )

    return (
        '<div class="card">'
        '<div class="card-header"><h2>Risk Flags</h2>'
        '<span class="card-subtitle">{count} active</span></div>'
        '<table class="data-table">'
        '<thead><tr>'
        '<th title="Risk severity level: HIGH / MEDIUM / LOW">Severity</th>'
        '<th title="Description of the risk flag">Flag</th>'
        '<th title="Which dashboard detected this flag">Source</th>'
        '<th title="Specific metric or threshold that triggered the flag">Detail</th>'
        '</tr></thead><tbody>'
        '{rows}'
        '</tbody></table></div>'
    ).format(count=len(flags), rows='\n'.join(rows))


# ============================================================================
# BODY + JS + CSS
# ============================================================================

def build_body(regime, agreement_full, agreement_trusted,
               intermarket, pattern, risk_flags_full, risk_flags_trusted):
    """Assemble the full HTML body with toggle between All / Trusted views."""

    # Toggle bar
    toggle_html = (
        '<div class="mode-toggle">'
        '<button class="toggle-btn active" id="btnTrusted" onclick="setMode(\'trusted\')">'
        'Trusted Only (4 sources)</button>'
        '<button class="toggle-btn" id="btnFull" onclick="setMode(\'full\')">'
        'All Sources (7 sources)</button>'
        '</div>'
    )

    # Agreement section -- both tables, toggled by JS
    agree_section = (
        '<div class="card">'
        '<div class="card-header">'
        '<h2>Cross-Dashboard Agreement Matrix</h2>'
        '<span class="card-subtitle" id="agreeSubFull" style="display:none;">'
        '{n_full} tickers in 2+ of 7 sources</span>'
        '<span class="card-subtitle" id="agreeSubTrusted">'
        '{n_trusted} tickers in 2+ of 4 trusted sources</span>'
        '</div>'
        '<div class="filter-row">'
        '<input type="text" id="agreeFilter" placeholder="Filter by ticker..." '
        'class="text-filter" onkeyup="filterAgreementTable()">'
        '</div>'
        '<div id="agreeTableFull" class="view-full" style="display:none;">{table_full}</div>'
        '<div id="agreeTableTrusted" class="view-trusted">{table_trusted}</div>'
        '</div>'
    ).format(
        n_full=len(agreement_full),
        n_trusted=len(agreement_trusted),
        table_full=_build_agreement_table(agreement_full, mode='full'),
        table_trusted=_build_agreement_table(agreement_trusted, mode='trusted'),
    )

    # Risk flags -- both versions, toggled
    flags_section = (
        '<div id="flagsFull" class="view-full" style="display:none;">'
        '{flags_full}</div>'
        '<div id="flagsTrusted" class="view-trusted">'
        '{flags_trusted}</div>'
    ).format(
        flags_full=_build_risk_flags(risk_flags_full),
        flags_trusted=_build_risk_flags(risk_flags_trusted),
    )

    parts = [
        _build_stat_bar(regime),
        _build_routing_banner(regime),
        # Note: some aggregated sources (HYG/LQD, momentum qualifier composite)
        # were invalidated by scimode. Meta dashboard keeps them for context but
        # users should weight trusted sources (D-Stock, momentum ranker, pullback health).
        '<div style="background:#fef3c7;border:1px solid #f59e0b;border-radius:8px;'
        'padding:14px 24px;margin:0 20px 18px 20px;display:flex;align-items:center;gap:14px;">'
        '<span style="font-weight:700;color:#92400e;font-size:1.1em;">NOTE</span>'
        '<span style="color:#78350f;font-size:0.92em;">'
        'Some aggregated sources (HYG/LQD, Momentum Qualifier composite) were invalidated by scimode. '
        'Weight Trusted-mode sources for trading decisions.</span>'
        '</div>',
        '<div class="container">',
        toggle_html,
        agree_section,
        _build_intermarket_section(intermarket),
        _build_pattern_section(pattern),
        flags_section,
        '</div>',
    ]
    return '\n'.join(parts)


EXTRA_CSS = """
/* --- Routing banner --- */
.routing-banner {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 14px 28px;
    color: #fff;
    font-size: 0.95em;
    flex-wrap: wrap;
}
.banner-tier {
    font-weight: 700;
    font-size: 1.1em;
    letter-spacing: 0.04em;
}
.banner-arrow {
    opacity: 0.7;
    font-family: 'IBM Plex Mono', monospace;
}
.banner-method {
    font-weight: 600;
}
.banner-perf {
    margin-left: auto;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.88em;
    opacity: 0.9;
}

/* --- Agreement matrix cells --- */
.agree-bull  { background: #dcfce7 !important; color: #166534; }
.agree-bear  { background: #fee2e2 !important; color: #991b1b; }
.agree-warn  { background: #fef9c3 !important; color: #854d0e; }
.agree-none  { background: #f3f4f6 !important; color: #9ca3af; }
.agree-present { background: #e0f2fe !important; color: #075985; }

.count-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-weight: 600;
    font-size: 0.85em;
    font-family: 'IBM Plex Mono', monospace;
}
.badge-high { background: #16a34a; color: #fff; }
.badge-mid  { background: #2563eb; color: #fff; }
.badge-low  { background: #6b7280; color: #fff; }

/* --- Filter row --- */
.filter-row {
    padding: 12px 20px 8px;
}
.text-filter {
    padding: 8px 14px;
    border: 1px solid #d1d5db;
    border-radius: 8px;
    font-size: 0.9em;
    width: 260px;
    font-family: 'IBM Plex Sans', sans-serif;
}
.text-filter:focus {
    outline: none;
    border-color: #4f46e5;
    box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.12);
}

/* --- Force cards --- */
.cards-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 16px;
    padding: 16px 20px;
}
.force-card {
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 16px;
    background: #fff;
}
.force-bullish { border-left: 4px solid #16a34a; }
.force-bearish { border-left: 4px solid #dc2626; }
.force-neutral { border-left: 4px solid #6b7280; }

.force-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 8px;
}
.force-level {
    font-weight: 600;
    font-size: 0.9em;
}
.force-arrow { font-size: 1.2em; }
.force-counts {
    font-size: 0.85em;
    margin-bottom: 8px;
}
.force-top {
    font-size: 0.88em;
    margin-bottom: 6px;
    font-family: 'IBM Plex Mono', monospace;
}
.force-playbook {
    font-size: 0.82em;
    color: #6b7280;
    line-height: 1.4;
}

/* --- Sector pills --- */
.sector-row { padding: 12px 20px; }
.sector-row h3 { font-size: 0.95em; margin-bottom: 8px; }
.pill-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}
.sector-pill {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 14px;
    font-size: 0.82em;
    font-weight: 500;
    font-family: 'IBM Plex Mono', monospace;
}
.sector-pill.pos { background: #dcfce7; color: #166534; }
.sector-pill.neg { background: #fee2e2; color: #991b1b; }
.sector-pill.neutral { background: #f3f4f6; color: #6b7280; }

/* --- Pattern section --- */
.pattern-group {
    padding: 14px 20px;
    border-bottom: 1px solid #f0f0f0;
}
.pattern-group:last-child { border-bottom: none; }
.pattern-group h3 {
    font-size: 0.95em;
    margin-bottom: 8px;
    color: #374151;
}
.pattern-window {
    font-size: 0.88em;
    margin-bottom: 4px;
    font-family: 'IBM Plex Mono', monospace;
}
.pattern-consensus {
    margin-top: 8px;
    font-weight: 600;
    color: #4f46e5;
}

/* --- Risk flags --- */
.risk-high   { border-left: 4px solid #dc2626; }
.risk-medium { border-left: 4px solid #ea580c; }
.risk-low    { border-left: 4px solid #eab308; }

.sev-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 10px;
    font-size: 0.78em;
    font-weight: 600;
    letter-spacing: 0.04em;
}
.sev-high   { background: #dc2626; color: #fff; }
.sev-medium { background: #ea580c; color: #fff; }
.sev-low    { background: #eab308; color: #422006; }

/* --- Compact table variant --- */
.data-table.compact td, .data-table.compact th {
    padding: 6px 10px;
    font-size: 0.85em;
}

/* --- Sortable headers --- */
.sortable-table thead th {
    cursor: pointer;
    user-select: none;
    position: relative;
    padding-right: 20px;
}
.sortable-table thead th::after {
    content: '';
    position: absolute;
    right: 6px;
    top: 50%;
    transform: translateY(-50%);
    font-size: 0.7em;
    color: #9ca3af;
}
.sortable-table thead th.sorted-asc::after  { content: '\\25B2'; color: #4f46e5; }
.sortable-table thead th.sorted-desc::after { content: '\\25BC'; color: #4f46e5; }

/* --- Card subtitle --- */
.card-subtitle {
    font-size: 0.85em;
    color: #6b7280;
    margin-left: 12px;
}

/* --- Mode toggle --- */
.mode-toggle {
    display: flex;
    gap: 0;
    margin-bottom: 20px;
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid #d1d5db;
    width: fit-content;
}
.toggle-btn {
    padding: 10px 22px;
    border: none;
    background: #f9fafb;
    color: #6b7280;
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.9em;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.15s;
}
.toggle-btn:hover {
    background: #f3f4f6;
}
.toggle-btn.active {
    background: #1e40af;
    color: #fff;
}
"""


EXTRA_JS = """
// Mode toggle: Trusted Only vs All Sources
function setMode(mode) {
    var fullEls = document.querySelectorAll('.view-full');
    var trustedEls = document.querySelectorAll('.view-trusted');
    var btnFull = document.getElementById('btnFull');
    var btnTrusted = document.getElementById('btnTrusted');
    var subFull = document.getElementById('agreeSubFull');
    var subTrusted = document.getElementById('agreeSubTrusted');

    if (mode === 'full') {
        fullEls.forEach(function(el) { el.style.display = ''; });
        trustedEls.forEach(function(el) { el.style.display = 'none'; });
        btnFull.classList.add('active');
        btnTrusted.classList.remove('active');
        if (subFull) subFull.style.display = '';
        if (subTrusted) subTrusted.style.display = 'none';
    } else {
        fullEls.forEach(function(el) { el.style.display = 'none'; });
        trustedEls.forEach(function(el) { el.style.display = ''; });
        btnTrusted.classList.add('active');
        btnFull.classList.remove('active');
        if (subFull) subFull.style.display = 'none';
        if (subTrusted) subTrusted.style.display = '';
    }
}

// Sortable table headers
document.querySelectorAll('.sortable-table thead th').forEach(function(th, idx) {
    th.addEventListener('click', function() {
        var table = th.closest('table');
        var tbody = table.querySelector('tbody');
        var rows = Array.from(tbody.querySelectorAll('tr'));
        var asc = !th.classList.contains('sorted-asc');

        table.querySelectorAll('th').forEach(function(h) {
            h.classList.remove('sorted-asc', 'sorted-desc');
        });
        th.classList.add(asc ? 'sorted-asc' : 'sorted-desc');

        rows.sort(function(a, b) {
            var av = a.cells[idx].getAttribute('data-sort') || a.cells[idx].textContent;
            var bv = b.cells[idx].getAttribute('data-sort') || b.cells[idx].textContent;
            var an = parseFloat(av), bn = parseFloat(bv);
            if (!isNaN(an) && !isNaN(bn)) return asc ? an - bn : bn - an;
            return asc ? av.localeCompare(bv) : bv.localeCompare(av);
        });

        rows.forEach(function(r) { tbody.appendChild(r); });
    });
});

// Text filter for agreement matrix (filters both tables)
function filterAgreementTable() {
    var input = document.getElementById('agreeFilter');
    if (!input) return;
    var filter = input.value.toUpperCase();
    var tables = input.closest('.card').querySelectorAll('table');
    tables.forEach(function(table) {
        var rows = table.querySelectorAll('tbody tr');
        rows.forEach(function(row) {
            var ticker = row.cells[0].textContent.toUpperCase();
            row.style.display = ticker.indexOf(filter) > -1 ? '' : 'none';
        });
    });
}
"""


# ============================================================================
# MAIN
# ============================================================================

def main():
    print('=' * 68)
    print('META DASHBOARD BACKEND v1.3')
    print('=' * 68)

    # 1. Load all sources
    print('\n[1/7] Loading data sources...')
    sources = load_all_sources()

    # 2. Regime context
    print('\n[2/7] Building regime context...')
    regime = build_regime_context(sources)
    print('  Drift tier: {} | Method: {}'.format(
        regime['drift_tier'], regime['active_method']))

    # 3. Agreement matrices (full + trusted)
    print('\n[3/7] Building agreement matrices...')
    agreement_full = build_agreement_matrix(sources)
    agreement_trusted = build_agreement_matrix(
        sources, exclude={'advanced', 'qualifier', 'slope_stage'})
    print('  Full: {} tickers | Trusted: {} tickers'.format(
        len(agreement_full), len(agreement_trusted)))

    # 4. Intermarket context
    print('\n[4/7] Building intermarket context...')
    intermarket = build_intermarket_context(sources)
    print('  {} force categories'.format(len(intermarket['forces'])))

    # 5. Pattern context
    print('\n[5/7] Building pattern context...')
    pattern = build_pattern_context(sources)

    # 6. Risk flags (full + trusted)
    print('\n[6/7] Building risk flags...')
    risk_flags_full = build_risk_flags(sources, agreement_full, trusted_only=False)
    risk_flags_trusted = build_risk_flags(sources, agreement_trusted, trusted_only=True)
    print('  Full: {} flags | Trusted: {} flags'.format(
        len(risk_flags_full), len(risk_flags_trusted)))

    # 7. Build dashboard
    print('\n[7/7] Writing dashboard...')
    body = build_body(regime, agreement_full, agreement_trusted,
                      intermarket, pattern, risk_flags_full, risk_flags_trusted)
    writer = DashboardWriter('meta-dashboard', 'Meta Dashboard')
    body = writer.build_header('Cross-dashboard agreement + risk flags') + body + writer.footer()
    writer.write(body, extra_css=EXTRA_CSS, extra_js=EXTRA_JS)

    # Save summary CSV (trusted view as default)
    if agreement_trusted:
        csv_path = os.path.join(
            _SCRIPT_DIR,
            'meta_dashboard_data_{}.csv'.format(
                datetime.datetime.now().strftime('%Y%m%d_%H%M')
            )
        )
        df = pd.DataFrame(agreement_trusted)
        df.to_csv(csv_path, index=False, encoding='utf-8')
        print('CSV: {}'.format(csv_path))

    print('\n[DONE] Meta dashboard v1.3 written.')


if __name__ == '__main__':
    main()
