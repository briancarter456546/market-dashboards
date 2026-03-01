# -*- coding: utf-8 -*-
"""
CONSERVATIVE MOMENTUM QUALIFIER - HTML Dashboard Backend
=========================================================
Qualifies 700+ tickers for momentum characteristics (trending, not
mean-reverting), finds dip patterns, assesses current state, and
computes conservative composite scores.

Outputs a static HTML dashboard via DashboardWriter.
No JSON output. No debug mode.

Run:
    python momentum_qualifier_backend.py
from the market-dashboards directory.
"""

import os
import pickle
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

from dashboard_writer import DashboardWriter

# =============================================================================
# PATH SETUP
# =============================================================================

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", "perplexity-user-data"))

PRICE_CACHE = os.path.join(_DATA_DIR, "price_cache")

# =============================================================================
# CONFIGURATION
# =============================================================================

MIN_TREND_PERCENTAGE = 0.60    # 60% of last year above SMA 9
MIN_POSITIVE_MONTHS  = 6       # 6 of last 12 months positive
MAX_SMA21_CROSSES    = 15      # Max whipsaws through SMA 21
SMA_PERIODS          = [9, 21, 29]

# Safe-filter thresholds
SAFE_MAX_RECENCY      = 2.5
SAFE_MAX_EXTENSION    = 0.15
SAFE_MIN_12M          = 0.40
SAFE_MAX_12M          = 3.0
SAFE_MIN_SUST         = 0.75
SAFE_MIN_EXT_SCORE    = 0.80

SLUG  = "conservative-momentum"
TITLE = "Conservative Momentum Qualifier"

# =============================================================================
# EXTRA CSS
# =============================================================================

EXTRA_CSS = """
/* --- Regime badges --- */
.regime-trending  { display:inline-block; padding:4px 12px; border-radius:20px;
                    font-size:0.82em; font-weight:700; letter-spacing:0.04em;
                    background:#dcfce7; color:#15803d; border:1px solid #bbf7d0; }
.regime-pullback  { display:inline-block; padding:4px 12px; border-radius:20px;
                    font-size:0.82em; font-weight:700; letter-spacing:0.04em;
                    background:#fef9c3; color:#a16207; border:1px solid #fde68a; }
.regime-dip       { display:inline-block; padding:4px 12px; border-radius:20px;
                    font-size:0.82em; font-weight:700; letter-spacing:0.04em;
                    background:#ffedd5; color:#c2410c; border:1px solid #fed7aa; }
.regime-breakdown { display:inline-block; padding:4px 12px; border-radius:20px;
                    font-size:0.82em; font-weight:700; letter-spacing:0.04em;
                    background:#fee2e2; color:#b91c1c; border:1px solid #fca5a5; }

/* --- Safe / unsafe badges --- */
.safe-badge   { display:inline-block; padding:4px 12px; border-radius:20px;
                font-size:0.82em; font-weight:700; letter-spacing:0.04em;
                background:#dcfce7; color:#15803d; border:1px solid #bbf7d0; }
.unsafe-badge { display:inline-block; padding:4px 12px; border-radius:20px;
                font-size:0.82em; font-weight:700; letter-spacing:0.04em;
                background:#f1f5f9; color:#64748b; border:1px solid #cbd5e1; }

/* --- Sortable table helpers --- */
thead th.sorted-asc::after  { content: " \\25B2"; }
thead th.sorted-desc::after { content: " \\25BC"; }

/* --- Compact number cells --- */
.n { font-family: 'IBM Plex Mono', monospace; font-size:0.9em; }

/* --- Score bar background shade --- */
.score-cell { font-family:'IBM Plex Mono',monospace; font-size:0.92em; font-weight:600; }

/* ── Mobile responsive ── */
@media (max-width: 768px) {
    .regime-trending, .regime-pullback, .regime-dip, .regime-breakdown { font-size: 0.75em; padding: 3px 10px; }
    .safe-badge, .unsafe-badge { font-size: 0.75em; padding: 3px 10px; }
    .score-cell { font-size: 0.82em; }
    .n { font-size: 0.82em; }
}
"""

# =============================================================================
# EXTRA JS  (column sort, shared across all three tables)
# =============================================================================

SORT_JS = """
(function() {
    function parseCell(td) {
        var raw = td.getAttribute('data-val');
        if (raw !== null) return parseFloat(raw);
        var t = td.textContent.trim().replace('%','').replace(',','');
        var n = parseFloat(t);
        return isNaN(n) ? t.toLowerCase() : n;
    }
    function sortTable(th) {
        var table = th.closest('table');
        var tbody = table.querySelector('tbody');
        var ths   = Array.from(th.parentNode.querySelectorAll('th'));
        var col   = ths.indexOf(th);
        var asc   = th.getAttribute('data-asc') !== 'true';
        th.setAttribute('data-asc', asc);
        ths.forEach(function(h) {
            h.classList.remove('sorted-asc','sorted-desc');
        });
        th.classList.add(asc ? 'sorted-asc' : 'sorted-desc');
        var rows = Array.from(tbody.querySelectorAll('tr'));
        rows.sort(function(a,b) {
            var va = parseCell(a.querySelectorAll('td')[col]);
            var vb = parseCell(b.querySelectorAll('td')[col]);
            if (typeof va === 'string') return asc ? va.localeCompare(vb) : vb.localeCompare(va);
            return asc ? va - vb : vb - va;
        });
        rows.forEach(function(r){ tbody.appendChild(r); });
    }
    document.querySelectorAll('thead th').forEach(function(th) {
        th.addEventListener('click', function(){ sortTable(th); });
    });
})();
"""

# =============================================================================
# STEP 1: DATA LOADING
# =============================================================================

def load_ohlcv(ticker):
    """Load OHLCV data from pickle cache. Returns DataFrame or None."""
    pkl_path = os.path.join(PRICE_CACHE, "{}.pkl".format(ticker))

    if not os.path.exists(pkl_path):
        return None

    try:
        with open(pkl_path, 'rb') as f:
            df = pickle.load(f)

        if not isinstance(df.index, pd.DatetimeIndex):
            if 'date' in df.columns:
                df = df.set_index(pd.to_datetime(df['date']))
            elif 'Date' in df.columns:
                df = df.set_index(pd.to_datetime(df['Date']))

        df.columns = df.columns.str.lower()
        df = df[~df.index.duplicated(keep='first')].sort_index()
        df = df.ffill()

        return df[['open', 'high', 'low', 'close', 'volume']]
    except Exception:
        return None


# =============================================================================
# STEP 2: MOMENTUM QUALIFICATION
# =============================================================================

def qualify_momentum(df):
    """
    Determine if asset qualifies as a momentum play.

    Criteria:
      1. 60%+ of last-year trading days above SMA 9
      2. At least 6 of last 12 months positive
      3. No more than 15 SMA 21 downward crosses (not a whipsaw)

    Returns dict of metrics or None if disqualified.
    """
    if len(df) < 252:
        return None

    last_year = df.tail(252).copy()

    for period in SMA_PERIODS:
        last_year['sma_{}'.format(period)] = last_year['close'].rolling(period).mean()

    # Criterion 1: trending
    above_sma9 = (last_year['close'] > last_year['sma_9']).astype(int)
    pct_above_sma9 = above_sma9.sum() / len(above_sma9)

    runs = []
    current_run = 0
    for val in above_sma9:
        if val == 1:
            current_run += 1
        else:
            if current_run > 0:
                runs.append(current_run)
            current_run = 0
    if current_run > 0:
        runs.append(current_run)
    longest_run = max(runs) if runs else 0

    if pct_above_sma9 < MIN_TREND_PERCENTAGE:
        return None

    # Criterion 2: monthly consistency
    monthly_returns = last_year['close'].resample('M').last().pct_change()
    positive_months = int((monthly_returns > 0).sum())

    if positive_months < MIN_POSITIVE_MONTHS:
        return None

    # Criterion 3: bounce quality (no whipsaw)
    crosses_21 = int((
        (last_year['close'].shift(1) > last_year['sma_21'].shift(1)) &
        (last_year['close'] < last_year['sma_21'])
    ).sum())

    if crosses_21 > MAX_SMA21_CROSSES:
        return None

    # ---- Qualified: compute return metrics from FULL df ----
    last_1wk   = df.tail(5)
    last_1mo   = df.tail(21)
    last_3mo   = df.tail(63)
    last_12mo  = df.tail(252)
    first_11mo = df.tail(252).head(231)

    def pct_gain(subset, min_len):
        if len(subset) < min_len:
            return 0.0
        return float(subset['close'].iloc[-1] / subset['close'].iloc[0] - 1)

    gain_last_1wk   = pct_gain(last_1wk,   5)
    gain_last_1mo   = pct_gain(last_1mo,   21)
    gain_last_3mo   = pct_gain(last_3mo,   63)
    gain_last_12mo  = pct_gain(last_12mo,  252)
    gain_first_11mo = pct_gain(first_11mo, 231)

    rate_last_1mo    = gain_last_1mo
    rate_first_11mo  = gain_first_11mo / 11.0
    recency_ratio    = rate_last_1mo / (rate_first_11mo + 0.01)

    extension_from_sma29 = float(
        (last_year['close'].iloc[-1] - last_year['sma_29'].iloc[-1])
        / last_year['sma_29'].iloc[-1]
    )

    return {
        'pct_above_sma9':           float(pct_above_sma9),
        'longest_run_days':         int(longest_run),
        'positive_months':          positive_months,
        'sma21_crosses':            crosses_21,
        'current_price':            float(last_year['close'].iloc[-1]),
        'sma9':                     float(last_year['sma_9'].iloc[-1]),
        'sma21':                    float(last_year['sma_21'].iloc[-1]),
        'sma29':                    float(last_year['sma_29'].iloc[-1]),
        'above_sma9':               bool(last_year['close'].iloc[-1] > last_year['sma_9'].iloc[-1]),
        'above_sma21':              bool(last_year['close'].iloc[-1] > last_year['sma_21'].iloc[-1]),
        'gain_last_1wk':            float(gain_last_1wk),
        'gain_last_1mo':            float(gain_last_1mo),
        'gain_last_3mo':            float(gain_last_3mo),
        'gain_last_12mo':           float(gain_last_12mo),
        'recency_ratio':            float(recency_ratio),
        'extension_from_sma29_pct': float(extension_from_sma29),
    }


# =============================================================================
# STEP 3: DIP PATTERN EXTRACTION
# =============================================================================

def find_dip_patterns(df, n_patterns=2):
    """
    Find last N completed dip patterns (price crosses below SMA 21 and recovers).
    Returns list of dip characteristic dicts.
    """
    df = df.copy()
    for period in [9, 21, 29]:
        df['sma_{}'.format(period)] = df['close'].rolling(period).mean()

    below_21 = df['close'] < df['sma_21']

    dip_starts = []
    dip_ends   = []
    in_dip     = False
    dip_start  = None

    for i in range(1, len(df)):
        if not below_21.iloc[i - 1] and below_21.iloc[i]:
            in_dip    = True
            dip_start = i
        elif below_21.iloc[i - 1] and not below_21.iloc[i]:
            if in_dip and dip_start is not None:
                dip_starts.append(dip_start)
                dip_ends.append(i)
            in_dip    = False
            dip_start = None

    dip_patterns = []
    for start_idx, end_idx in zip(dip_starts[-n_patterns:], dip_ends[-n_patterns:]):
        segment = df.iloc[start_idx:end_idx]
        if len(segment) < 2:
            continue

        entry_price = float(df.iloc[start_idx]['close'])
        low_price   = float(segment['close'].min())
        exit_price  = float(df.iloc[end_idx]['close'])
        dip_depth   = (low_price - entry_price) / entry_price

        low_idx        = segment['close'].idxmin()
        days_to_bottom = int(df.index.get_loc(low_idx) - start_idx)
        denom          = exit_price - entry_price
        recovery_rate  = float((exit_price - low_price) / denom) if denom != 0 else 0.0

        dip_patterns.append({
            'start_date':    df.index[start_idx].isoformat(),
            'end_date':      df.index[end_idx].isoformat(),
            'duration_days': int(end_idx - start_idx),
            'dip_depth_pct': float(dip_depth * 100),
            'days_to_bottom': days_to_bottom,
            'recovery_rate': recovery_rate,
            'entry_price':   entry_price,
            'low_price':     low_price,
            'exit_price':    exit_price,
        })

    return dip_patterns


# =============================================================================
# STEP 4: CURRENT STATE ASSESSMENT
# =============================================================================

def assess_current_state(df):
    """
    Classify current regime and measure distances from key SMAs.
    Returns dict.
    """
    df = df.copy()
    for period in [9, 21, 29]:
        df['sma_{}'.format(period)] = df['close'].rolling(period).mean()

    current = df.iloc[-1]

    dist_sma9  = float((current['close'] - current['sma_9'])  / current['sma_9']  * 100)
    dist_sma21 = float((current['close'] - current['sma_21']) / current['sma_21'] * 100)
    dist_sma29 = float((current['close'] - current['sma_29']) / current['sma_29'] * 100)

    if current['close'] > current['sma_9']:
        regime = 'TRENDING'
    elif current['close'] > current['sma_21']:
        regime = 'PULLBACK'
    elif current['close'] > current['sma_29']:
        regime = 'DIP'
    else:
        regime = 'BREAKDOWN'

    above_9         = (df['close'] > df['sma_9']).astype(int)
    days_in_regime  = 0
    for val in reversed(above_9.values):
        if (regime == 'TRENDING' and val == 1) or (regime != 'TRENDING' and val == 0):
            days_in_regime += 1
        else:
            break

    return {
        'regime':               regime,
        'days_in_regime':       int(days_in_regime),
        'dist_from_sma9_pct':   dist_sma9,
        'dist_from_sma21_pct':  dist_sma21,
        'dist_from_sma29_pct':  dist_sma29,
        'current_price':        float(current['close']),
        'sma9':                 float(current['sma_9']),
        'sma21':                float(current['sma_21']),
        'sma29':                float(current['sma_29']),
    }


# =============================================================================
# STEP 5: COMPOSITE SCORING
# =============================================================================

def compute_scores(asset):
    """
    Add composite_score, sustainability_score, extension_score, and is_safe
    fields to an asset dict in-place. Returns the asset dict.
    """
    q = asset['qualification']

    ret_12m_capped  = min(q['gain_last_12mo'], 2.0)
    pct_above       = q['pct_above_sma9']
    recency         = q['recency_ratio']
    extension       = q['extension_from_sma29_pct']

    sustainability  = 1.0 / (1.0 + max(0, recency - 1.5))
    extension_score = 1.0 / (1.0 + max(0, extension - 0.10))
    momentum_score  = (pct_above - 0.6) / 0.4     # scale 60-100% -> 0-1

    composite = (
        0.25 * ret_12m_capped +
        0.30 * momentum_score +
        0.35 * sustainability +
        0.10 * extension_score
    )

    is_safe = (
        recency        < SAFE_MAX_RECENCY    and
        extension      < SAFE_MAX_EXTENSION  and
        q['gain_last_12mo'] >= SAFE_MIN_12M  and
        q['gain_last_12mo'] <  SAFE_MAX_12M  and
        sustainability >= SAFE_MIN_SUST      and
        extension_score >= SAFE_MIN_EXT_SCORE
    )

    asset['composite_score']    = round(composite,        4)
    asset['sustainability_score'] = round(sustainability, 4)
    asset['extension_score']    = round(extension_score,  4)
    asset['ret_12m_capped']     = round(ret_12m_capped,   4)
    asset['is_safe']            = is_safe
    return asset


# =============================================================================
# HTML RENDERING HELPERS
# =============================================================================

def _regime_badge(regime):
    cls_map = {
        'TRENDING':  'regime-trending',
        'PULLBACK':  'regime-pullback',
        'DIP':       'regime-dip',
        'BREAKDOWN': 'regime-breakdown',
    }
    cls = cls_map.get(regime, 'regime-trending')
    return '<span class="{}">{}</span>'.format(cls, regime)


def _safe_badge(is_safe):
    if is_safe:
        return '<span class="safe-badge">SAFE</span>'
    return '<span class="unsafe-badge">—</span>'


def _pct(val, decimals=1):
    """Format a float (0.123) as a coloured percentage string."""
    pct_str = '{:.{}f}%'.format(val * 100, decimals)
    if val > 0:
        return '<span class="pos n">{}</span>'.format(pct_str)
    elif val < 0:
        return '<span class="neg n">{}</span>'.format(pct_str)
    return '<span class="n">{}</span>'.format(pct_str)


def _score_cell(val):
    """Format composite score with color hint."""
    if val >= 0.55:
        cls = 'pos'
    elif val >= 0.35:
        cls = 'warn'
    else:
        cls = 'neg'
    return '<span class="score-cell {}">{:.3f}</span>'.format(cls, val)


def _num(val, fmt='{:.2f}'):
    return '<span class="n">{}</span>'.format(fmt.format(val))


def _build_table(assets, table_id, columns):
    """
    Generic table builder.
    columns: list of (header_label, row_fn)
    row_fn receives the asset dict and returns an HTML string for that cell.
    """
    thead_cells = ''.join(
        '<th>{}</th>'.format(h) for h, _ in columns
    )
    rows = []
    for asset in assets:
        cells = ''.join(
            '<td>{}</td>'.format(fn(asset)) for _, fn in columns
        )
        rows.append('<tr>{}</tr>'.format(cells))

    return (
        '<table id="{}">'
        '<thead><tr>{}</tr></thead>'
        '<tbody>{}</tbody>'
        '</table>'
    ).format(table_id, thead_cells, '\n'.join(rows))


def _build_safe_table(assets):
    cols = [
        ('Rank',         lambda a: '<span class="n">{}</span>'.format(a['_rank'])),
        ('Ticker',       lambda a: '<span class="ticker">{}</span>'.format(a['ticker'])),
        ('Score',        lambda a: _score_cell(a['composite_score'])),
        ('12M Ret',      lambda a: _pct(a['qualification']['gain_last_12mo'])),
        ('3M Ret',       lambda a: _pct(a['qualification']['gain_last_3mo'])),
        ('1M Ret',       lambda a: _pct(a['qualification']['gain_last_1mo'])),
        ('1W Ret',       lambda a: _pct(a['qualification']['gain_last_1wk'])),
        ('Recency Ratio',lambda a: _num(a['qualification']['recency_ratio'])),
        ('Sust',         lambda a: _num(a['sustainability_score'])),
        ('Ext',          lambda a: _num(a['extension_score'])),
        ('Regime',       lambda a: _regime_badge(a['current_state']['regime'])),
        ('Days',         lambda a: '<span class="n">{}</span>'.format(
                                       a['current_state']['days_in_regime'])),
    ]
    return _build_table(assets, 'tbl-safe', cols)


def _build_all_table(assets):
    cols = [
        ('Rank',         lambda a: '<span class="n">{}</span>'.format(a['_rank'])),
        ('Ticker',       lambda a: '<span class="ticker">{}</span>'.format(a['ticker'])),
        ('Safe',         lambda a: _safe_badge(a['is_safe'])),
        ('Score',        lambda a: _score_cell(a['composite_score'])),
        ('12M Ret',      lambda a: _pct(a['qualification']['gain_last_12mo'])),
        ('3M Ret',       lambda a: _pct(a['qualification']['gain_last_3mo'])),
        ('1M Ret',       lambda a: _pct(a['qualification']['gain_last_1mo'])),
        ('1W Ret',       lambda a: _pct(a['qualification']['gain_last_1wk'])),
        ('Recency',      lambda a: _num(a['qualification']['recency_ratio'])),
        ('Sust',         lambda a: _num(a['sustainability_score'])),
        ('Ext',          lambda a: _num(a['extension_score'])),
        ('%>SMA9',       lambda a: _pct(a['qualification']['pct_above_sma9'])),
        ('Pos Mo',       lambda a: '<span class="n">{}/12</span>'.format(
                                       a['qualification']['positive_months'])),
        ('Regime',       lambda a: _regime_badge(a['current_state']['regime'])),
        ('Days',         lambda a: '<span class="n">{}</span>'.format(
                                       a['current_state']['days_in_regime'])),
    ]
    return _build_table(assets, 'tbl-all', cols)


def _build_momentum_table(assets):
    cols = [
        ('Ticker',       lambda a: '<span class="ticker">{}</span>'.format(a['ticker'])),
        ('%>SMA9',       lambda a: _pct(a['qualification']['pct_above_sma9'])),
        ('Pos Mo',       lambda a: '<span class="n">{}/12</span>'.format(
                                       a['qualification']['positive_months'])),
        ('Longest Run',  lambda a: '<span class="n">{} d</span>'.format(
                                       a['qualification']['longest_run_days'])),
        ('Regime',       lambda a: _regime_badge(a['current_state']['regime'])),
        ('Dist SMA9',    lambda a: _pct(a['current_state']['dist_from_sma9_pct'] / 100, 2)),
        ('12M Ret',      lambda a: _pct(a['qualification']['gain_last_12mo'])),
        ('Score',        lambda a: _score_cell(a['composite_score'])),
    ]
    return _build_table(assets, 'tbl-momentum', cols)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print(TITLE)
    print("=" * 70)

    # ------------------------------------------------------------------
    # Discover tickers
    # ------------------------------------------------------------------
    pkl_files = sorted(
        f for f in os.listdir(PRICE_CACHE) if f.endswith('.pkl')
    )
    tickers = [os.path.splitext(f)[0] for f in pkl_files]
    total_screened = len(tickers)
    print("Screening {} assets...".format(total_screened))

    # ------------------------------------------------------------------
    # Full scan
    # ------------------------------------------------------------------
    qualified_assets = []

    for i, ticker in enumerate(tickers, 1):
        df = load_ohlcv(ticker)
        if df is None:
            continue

        qual = qualify_momentum(df)
        if qual is None:
            continue

        dip_patterns  = find_dip_patterns(df.tail(252), n_patterns=2)
        current_state = assess_current_state(df)

        asset = {
            'ticker':               ticker,
            'qualification':        qual,
            'recent_dip_patterns':  dip_patterns,
            'current_state':        current_state,
        }
        compute_scores(asset)
        qualified_assets.append(asset)

        if i % 100 == 0:
            print("  Processed {}/{}, qualified: {}".format(
                i, total_screened, len(qualified_assets)))

    total_qualified = len(qualified_assets)
    print()
    print("Results: {}/{} assets qualified ({:.1f}%)".format(
        total_qualified, total_screened,
        total_qualified / total_screened * 100 if total_screened else 0))

    if not qualified_assets:
        print("No qualified assets found - aborting HTML write.")
        return

    # ------------------------------------------------------------------
    # Sort and rank all qualified assets by composite score
    # ------------------------------------------------------------------
    qualified_assets.sort(key=lambda a: a['composite_score'], reverse=True)
    for rank, asset in enumerate(qualified_assets, 1):
        asset['_rank'] = rank

    # ------------------------------------------------------------------
    # Safe filter
    # ------------------------------------------------------------------
    safe_assets  = [a for a in qualified_assets if a['is_safe']]
    safe_assets.sort(key=lambda a: a['composite_score'], reverse=True)
    for rank, asset in enumerate(safe_assets, 1):
        asset['_rank'] = rank      # re-rank within safe subset

    safe_returns = [a['qualification']['gain_last_12mo'] for a in safe_assets]
    avg_safe_ret = sum(safe_returns) / len(safe_returns) if safe_returns else 0.0

    # ------------------------------------------------------------------
    # Pure momentum leaders (top 20 by pct_above_sma9)
    # ------------------------------------------------------------------
    momentum_leaders = sorted(
        qualified_assets,
        key=lambda a: a['qualification']['pct_above_sma9'],
        reverse=True
    )[:20]
    for rank, asset in enumerate(momentum_leaders, 1):
        asset['_rank'] = rank

    # Restore overall ranks on full list (safe list used its own ranks)
    for rank, asset in enumerate(qualified_assets, 1):
        asset['_rank'] = rank

    # For safe list use separate rank counter
    for rank, asset in enumerate(safe_assets, 1):
        asset['_safe_rank'] = rank

    # Swap _rank to _safe_rank for the safe table render
    for asset in safe_assets:
        asset['_rank'] = asset['_safe_rank']

    # ------------------------------------------------------------------
    # Determine banner label from regime distribution of safe assets
    # ------------------------------------------------------------------
    regimes = [a['current_state']['regime'] for a in safe_assets]
    trending_pct = (regimes.count('TRENDING') / len(regimes) * 100) if regimes else 0

    if trending_pct >= 70:
        banner_label = "STRONG MOMENTUM FIELD"
        banner_color = "#22c55e"
    elif trending_pct >= 50:
        banner_label = "MODERATE MOMENTUM FIELD"
        banner_color = "#f59e0b"
    elif trending_pct >= 30:
        banner_label = "MIXED FIELD - PULLBACKS IN PROGRESS"
        banner_color = "#f97316"
    else:
        banner_label = "WEAK FIELD - MOSTLY DIP/BREAKDOWN"
        banner_color = "#ef4444"

    regime_counts = {
        'TRENDING':  regimes.count('TRENDING'),
        'PULLBACK':  regimes.count('PULLBACK'),
        'DIP':       regimes.count('DIP'),
        'BREAKDOWN': regimes.count('BREAKDOWN'),
    }
    banner_score_html = (
        "Safe plays: {safe} &nbsp;|&nbsp; "
        "TRENDING: {tr} &nbsp;|&nbsp; "
        "PULLBACK: {pb} &nbsp;|&nbsp; "
        "DIP: {dp} &nbsp;|&nbsp; "
        "BREAKDOWN: {bd} &nbsp;|&nbsp; "
        "Avg 12M Return: {ret:.1f}%"
    ).format(
        safe=len(safe_assets),
        tr=regime_counts['TRENDING'],
        pb=regime_counts['PULLBACK'],
        dp=regime_counts['DIP'],
        bd=regime_counts['BREAKDOWN'],
        ret=avg_safe_ret * 100,
    )

    # ------------------------------------------------------------------
    # Stat bar values
    # ------------------------------------------------------------------
    safe_ret_class   = 'pos' if avg_safe_ret > 0 else 'neg'
    qual_rate        = total_qualified / total_screened * 100 if total_screened else 0
    qual_rate_class  = 'pos' if qual_rate >= 20 else 'warn'

    stat_bar_data = [
        ("Total Screened",   str(total_screened),                       'neutral'),
        ("Total Qualified",  str(total_qualified),                      qual_rate_class),
        ("Safe Plays",       str(len(safe_assets)),                     'pos' if safe_assets else 'warn'),
        ("Avg Safe 12M",     "{:.1f}%".format(avg_safe_ret * 100),      safe_ret_class),
    ]

    # ------------------------------------------------------------------
    # Section 1: Safe Conservative Plays table
    # ------------------------------------------------------------------
    safe_table_html = _build_safe_table(safe_assets)
    safe_note = (
        '<p style="font-size:0.85em;color:#888;margin-bottom:12px;">'
        'Filters: Recency &lt; {rec} &nbsp;|&nbsp; Extension &lt; {ext:.0f}% &nbsp;|&nbsp; '
        '12M return {min:.0f}%&ndash;{max:.0f}% &nbsp;|&nbsp; '
        'Sustainability &ge; {sust} &nbsp;|&nbsp; '
        'Extension score &ge; {exts}'
        '</p>'
    ).format(
        rec=SAFE_MAX_RECENCY,
        ext=SAFE_MAX_EXTENSION * 100,
        min=SAFE_MIN_12M * 100,
        max=SAFE_MAX_12M * 100,
        sust=SAFE_MIN_SUST,
        exts=SAFE_MIN_EXT_SCORE,
    )

    # ------------------------------------------------------------------
    # Section 2: All Qualified Assets
    # ------------------------------------------------------------------
    all_table_html = _build_all_table(qualified_assets)

    # ------------------------------------------------------------------
    # Section 3: Pure Momentum Leaders
    # ------------------------------------------------------------------
    momentum_table_html = _build_momentum_table(momentum_leaders)
    momentum_note = (
        '<p style="font-size:0.85em;color:#888;margin-bottom:12px;">'
        'Top 20 qualified assets ranked by %% time above SMA 9 over the past year.'
        '</p>'
    )

    # ------------------------------------------------------------------
    # Assemble dashboard
    # ------------------------------------------------------------------
    writer = DashboardWriter(SLUG, TITLE)
    parts  = []

    # Stat bar
    parts.append(writer.stat_bar(stat_bar_data))

    # Page header
    parts.append(writer.build_header(
        "Screened {} &nbsp;|&nbsp; Qualified {} &nbsp;|&nbsp; {}".format(
            total_screened, total_qualified,
            datetime.now().strftime('%Y-%m-%d %H:%M'))
    ))

    # Regime banner
    parts.append(writer.regime_banner(banner_label, banner_score_html, color=banner_color))

    # Section 1: Safe plays
    parts.append(writer.section(
        "Safe Conservative Plays ({})".format(len(safe_assets)),
        safe_note + safe_table_html,
        hint="Click column header to sort"
    ))

    # Section 2: All qualified
    parts.append(writer.section(
        "All Qualified Assets ({})".format(total_qualified),
        all_table_html,
        hint="Click column header to sort"
    ))

    # Section 3: Pure momentum leaders
    parts.append(writer.section(
        "Pure Momentum Leaders (Top 20 by %&gt;SMA9)",
        momentum_note + momentum_table_html,
        hint="Click column header to sort"
    ))

    # Footer
    parts.append(writer.footer())

    writer.write("\n".join(parts), extra_css=EXTRA_CSS, extra_js=SORT_JS)

    # Write CSV
    csv_path = os.path.join(_SCRIPT_DIR, 'conservative_momentum_data.csv')
    csv_rows = []
    for a in qualified_assets:
        q = a['qualification']
        cs = a['current_state']
        csv_rows.append({
            'ticker': a['ticker'],
            'composite_score': a['composite_score'],
            'sustainability_score': a['sustainability_score'],
            'extension_score': a['extension_score'],
            'is_safe': a['is_safe'],
            'pct_above_sma9': q['pct_above_sma9'],
            'longest_run_days': q['longest_run_days'],
            'positive_months': q['positive_months'],
            'sma21_crosses': q['sma21_crosses'],
            'gain_last_1wk': q['gain_last_1wk'],
            'gain_last_1mo': q['gain_last_1mo'],
            'gain_last_3mo': q['gain_last_3mo'],
            'gain_last_12mo': q['gain_last_12mo'],
            'recency_ratio': q['recency_ratio'],
            'extension_from_sma29_pct': q['extension_from_sma29_pct'],
            'regime': cs['regime'],
            'days_in_regime': cs['days_in_regime'],
            'current_price': cs['current_price'],
            'dist_from_sma9_pct': cs['dist_from_sma9_pct'],
            'dist_from_sma21_pct': cs['dist_from_sma21_pct'],
            'dist_from_sma29_pct': cs['dist_from_sma29_pct'],
        })
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False, encoding='utf-8')
    print("CSV: {}".format(csv_path))

    print()
    print("Dashboard written.")
    print("  Safe plays:      {}".format(len(safe_assets)))
    print("  Avg safe 12M:    {:.1f}%".format(avg_safe_ret * 100))
    print("  Banner:          {}".format(banner_label))


if __name__ == "__main__":
    main()
