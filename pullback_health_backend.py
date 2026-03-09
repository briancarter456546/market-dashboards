# -*- coding: utf-8 -*-
# =============================================================================
# pullback_health_backend.py - v1.0
# Last updated: 2026-03-05
# =============================================================================
# v1.0: Initial build - Pullback Health Monitor dashboard
#       6 metrics: NATR drawdown, SMA structure, slope stage, vol expansion,
#       residual drawdown (beta-adjusted vs SPY), historical recovery.
#       Composite health score 0-100 with classification.
#       Scientist-mode validated: ATR normalized by price (not raw),
#       volume dropped from scoring (dark pool problem), Fibonacci skipped.
#
# Run:  python pullback_health_backend.py
# =============================================================================

import os
import json
import pickle
import datetime
import time

import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from dashboard_writer import DashboardWriter

# =============================================================================
# PATH SETUP
# =============================================================================

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR   = os.path.normpath(os.path.join(_SCRIPT_DIR, '..', 'perplexity-user-data'))
CACHE_DIR   = os.path.normpath(os.path.join(_DATA_DIR, 'price_cache'))

# Slope stage data (from slope_stage_backend.py cache)
SLOPE_STAGE_FILE = os.path.join(_DATA_DIR, 'slope_stage_data.json')

# Momentum ranker data (from momentum_ranker_v1_18.py)
MOMENTUM_RANKER_FILE = os.path.join(_SCRIPT_DIR, 'momentum_ranker_data.json')

# Output cache
PULLBACK_CACHE_FILE = os.path.join(_DATA_DIR, 'pullback_health_data.json')

CONFIG = {
    "max_workers":        8,
    "min_history":        126,     # 6 months minimum
    "atr_period":         14,
    "drawdown_lookback":  63,      # 3-month high for drawdown
    "vol_short":          10,      # short realized vol window
    "vol_long":           63,      # long realized vol window
    "beta_lookback":      126,     # 6-month beta estimation
    "recovery_lookback":  252 * 5, # 5 years of history for recovery stats
    "recovery_forward":   21,      # 21-day forward return for recovery
}


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


# =============================================================================
# OVERLAYS: load slope stage + momentum ranker data
# =============================================================================

def load_slope_stage_data():
    """Load slope stage cache -> dict keyed by ticker."""
    if not os.path.exists(SLOPE_STAGE_FILE):
        print("[WARN] Slope stage data not found: {}".format(SLOPE_STAGE_FILE))
        return {}
    with open(SLOPE_STAGE_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    results = data.get('results', [])
    return {r['ticker']: r for r in results}


def load_momentum_ranker_data():
    """Load momentum ranker JSON -> dict keyed by ticker."""
    if not os.path.exists(MOMENTUM_RANKER_FILE):
        print("[WARN] Momentum ranker data not found: {}".format(MOMENTUM_RANKER_FILE))
        return {}
    with open(MOMENTUM_RANKER_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    rows = data.get('data', [])
    return {r['ticker']: r for r in rows}


# =============================================================================
# SPY DATA (for beta + residual drawdown)
# =============================================================================

def load_spy_data():
    """Load SPY price data from cache."""
    spy_file = os.path.join(CACHE_DIR, 'SPY.pkl')
    if not os.path.exists(spy_file):
        print("[ERROR] SPY.pkl not found in price_cache")
        return None
    df = pd.read_pickle(spy_file)
    df = df.sort_index()
    return df


# =============================================================================
# CORE METRICS
# =============================================================================

def compute_atr(high, low, close, period=14):
    """Compute Average True Range."""
    h = high.values
    l = low.values
    c = close.values
    n = len(c)
    tr = np.zeros(n)
    tr[0] = h[0] - l[0]
    for i in range(1, n):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i-1]), abs(l[i] - c[i-1]))
    # Wilder smoothing
    atr = np.zeros(n)
    atr[:period] = np.nan
    if n >= period:
        atr[period - 1] = np.mean(tr[:period])
        for i in range(period, n):
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    return pd.Series(atr, index=close.index)


def compute_natr(high, low, close, period=14):
    """Normalized ATR = ATR / close * 100 (percentage)."""
    atr = compute_atr(high, low, close, period)
    natr = (atr / close) * 100.0
    return natr


def compute_drawdown_natr(close, high, natr, lookback=63):
    """Drawdown from N-day high, normalized by NATR.

    Returns (dd_pct, dd_natr):
      dd_pct  = percentage drawdown from lookback high (negative)
      dd_natr = drawdown magnitude / current NATR (unitless severity)
    """
    rolling_high = high.rolling(window=lookback, min_periods=1).max()
    dd_pct = (close / rolling_high - 1.0) * 100.0

    current_natr = natr.iloc[-1]
    current_dd_pct = dd_pct.iloc[-1]

    if current_natr > 0 and not np.isnan(current_natr):
        dd_natr = abs(current_dd_pct) / current_natr
    else:
        dd_natr = 0.0

    return float(current_dd_pct), float(dd_natr)


def compute_vol_expansion(close, short_window=10, long_window=63):
    """Realized vol ratio: short / long. >1.5 = expanding volatility."""
    log_ret = np.log(close / close.shift(1))
    vol_short = log_ret.rolling(window=short_window).std() * np.sqrt(252)
    vol_long = log_ret.rolling(window=long_window).std() * np.sqrt(252)

    vs = vol_short.iloc[-1]
    vl = vol_long.iloc[-1]

    if vl > 0 and not np.isnan(vl) and not np.isnan(vs):
        ratio = vs / vl
    else:
        ratio = 1.0

    return float(ratio)


def compute_residual_drawdown(ticker_close, spy_close, lookback=126, dd_lookback=63):
    """Beta-adjusted residual drawdown.

    Actual DD minus expected DD (beta * SPY DD) = residual.
    Positive residual = stock fell more than beta predicts = stock-specific.
    """
    # Align dates
    common = ticker_close.index.intersection(spy_close.index)
    if len(common) < lookback:
        return 0.0, 1.0  # (residual, beta)

    tc = ticker_close.loc[common]
    sc = spy_close.loc[common]

    # Compute beta over lookback
    tr = tc.pct_change().dropna()
    sr = sc.pct_change().dropna()
    common_r = tr.index.intersection(sr.index)
    tr = tr.loc[common_r].iloc[-lookback:]
    sr = sr.loc[common_r].iloc[-lookback:]

    if len(tr) < 60:
        return 0.0, 1.0

    cov = np.cov(tr.values, sr.values)
    beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else 1.0
    beta = max(0.0, min(beta, 5.0))  # clamp

    # Actual drawdown
    tc_recent = tc.iloc[-dd_lookback:]
    sc_recent = sc.iloc[-dd_lookback:]

    ticker_dd = (tc_recent.iloc[-1] / tc_recent.max() - 1.0) * 100.0
    spy_dd = (sc_recent.iloc[-1] / sc_recent.max() - 1.0) * 100.0

    # Expected DD = beta * SPY DD
    expected_dd = beta * spy_dd

    # Residual = actual - expected (more negative actual = positive residual = bad)
    residual = abs(ticker_dd) - abs(expected_dd)

    return float(residual), float(round(beta, 2))


def compute_historical_recovery(close, natr, current_dd_natr,
                                lookback=252*5, forward=21, tolerance=0.5):
    """Find past drawdowns of similar NATR severity, measure forward returns.

    Returns (median_fwd_return, n_episodes).
    """
    if len(close) < lookback + forward:
        usable = close
    else:
        usable = close.iloc[-(lookback + forward):]

    if len(usable) < 126 + forward:
        return None, 0

    high_63 = usable.rolling(window=63, min_periods=1).max()
    dd_pct_series = (usable / high_63 - 1.0) * 100.0

    # Compute rolling NATR for the series
    # Simplified: use current NATR level as denominator (stable enough)
    current_natr_val = natr.iloc[-1]
    if current_natr_val <= 0 or np.isnan(current_natr_val):
        return None, 0

    dd_natr_series = dd_pct_series.abs() / current_natr_val

    # Find episodes with similar DD NATR (within tolerance)
    lo = max(0.5, current_dd_natr - tolerance)
    hi = current_dd_natr + tolerance

    mask = (dd_natr_series >= lo) & (dd_natr_series <= hi)

    # Get forward returns for matching episodes
    forward_returns = []
    idx = usable.index
    vals = usable.values

    for i in range(len(usable) - forward):
        if mask.iloc[i]:
            fwd = (vals[i + forward] / vals[i] - 1.0) * 100.0
            forward_returns.append(fwd)

    if len(forward_returns) < 5:
        return None, len(forward_returns)

    return float(np.median(forward_returns)), len(forward_returns)


# =============================================================================
# COMPOSITE SCORE
# =============================================================================

def compute_health_score(dd_natr, sma_count, stage, tq_score,
                         vol_ratio, residual_dd, recovery_median):
    """Compute 0-100 Pullback Health Score.

    Weights (scientist-mode validated):
      NATR Severity      25%  (shallow = high score)
      SMA Structure      25%  (4/4 = 100, 0/4 = 0)
      Slope Stage        20%  (stage 2 = high, stage 0 = zero)
      Vol Expansion      10%  (stable = high, expanding = low)
      Residual DD        10%  (small = high, large = low)
      Historical Recovery 10% (positive median = high)
    """
    # 1. NATR severity (0-100, inverted: shallow=high)
    # dd_natr: 0=no drawdown, 1.5=normal, 3=deep, 5+=severe
    if dd_natr <= 0.5:
        natr_score = 100.0
    elif dd_natr <= 1.5:
        natr_score = 100.0 - (dd_natr - 0.5) * 30.0  # 100 -> 70
    elif dd_natr <= 3.0:
        natr_score = 70.0 - (dd_natr - 1.5) * 33.3    # 70 -> 20
    elif dd_natr <= 5.0:
        natr_score = 20.0 - (dd_natr - 3.0) * 10.0    # 20 -> 0
    else:
        natr_score = 0.0

    # 2. SMA structure (0-100)
    sma_score = (sma_count / 4.0) * 100.0

    # 3. Slope stage (0-100)
    stage_map = {0: 0.0, 1: 25.0, 2: 100.0, 3: 65.0}
    stage_score = stage_map.get(stage, 50.0)
    # Blend with TQ score if available
    if tq_score is not None and not np.isnan(tq_score):
        stage_score = stage_score * 0.6 + tq_score * 0.4

    # 4. Vol expansion (0-100, stable=high)
    # ratio 1.0 = normal, <1.0 = contracting (good), >1.5 = expanding (bad)
    if vol_ratio <= 0.8:
        vol_score = 100.0
    elif vol_ratio <= 1.2:
        vol_score = 100.0 - (vol_ratio - 0.8) * 50.0   # 100 -> 80
    elif vol_ratio <= 1.5:
        vol_score = 80.0 - (vol_ratio - 1.2) * 133.3   # 80 -> 40
    elif vol_ratio <= 2.0:
        vol_score = 40.0 - (vol_ratio - 1.5) * 60.0    # 40 -> 10
    else:
        vol_score = max(0.0, 10.0 - (vol_ratio - 2.0) * 10.0)

    # 5. Residual drawdown (0-100, small=high)
    # residual: 0 = on beta track, >0 = stock fell more than beta predicts
    if residual_dd <= 0.0:
        resid_score = 100.0
    elif residual_dd <= 2.0:
        resid_score = 100.0 - residual_dd * 15.0       # 100 -> 70
    elif residual_dd <= 5.0:
        resid_score = 70.0 - (residual_dd - 2.0) * 16.7  # 70 -> 20
    elif residual_dd <= 10.0:
        resid_score = 20.0 - (residual_dd - 5.0) * 4.0   # 20 -> 0
    else:
        resid_score = 0.0

    # 6. Historical recovery (0-100)
    if recovery_median is None:
        recovery_score = 50.0  # neutral if insufficient data
    elif recovery_median >= 3.0:
        recovery_score = 100.0
    elif recovery_median >= 0.0:
        recovery_score = 50.0 + (recovery_median / 3.0) * 50.0  # 50 -> 100
    elif recovery_median >= -3.0:
        recovery_score = 50.0 + (recovery_median / 3.0) * 50.0  # 50 -> 0
    else:
        recovery_score = 0.0

    # Composite
    composite = (
        natr_score     * 0.25 +
        sma_score      * 0.25 +
        stage_score    * 0.20 +
        vol_score      * 0.10 +
        resid_score    * 0.10 +
        recovery_score * 0.10
    )

    # Risk override (Perplexity feedback): if vol expanding hard or large
    # residual DD, cap the score -- structure alone shouldn't mask instability
    if vol_score < 30 or resid_score < 20:
        composite = min(composite, 60.0)

    return round(max(0.0, min(100.0, composite)), 1)


def classify_health(score):
    """Classify health score into verdict."""
    if score >= 80:
        return "HEALTHY"
    elif score >= 50:
        return "CAUTION"
    elif score >= 20:
        return "WARNING"
    else:
        return "BREAKDOWN"


# =============================================================================
# PROCESS SINGLE ASSET
# =============================================================================

def process_single_asset(ticker, df, spy_close, slope_data, mr_data):
    """Compute all pullback health metrics for one ticker."""
    try:
        df = df.sort_index()
        if len(df) < CONFIG['min_history']:
            return None

        close = df['close']
        high = df['high']
        low = df['low']

        # -- NATR drawdown --
        natr = compute_natr(high, low, close, CONFIG['atr_period'])
        dd_pct, dd_natr = compute_drawdown_natr(
            close, high, natr, CONFIG['drawdown_lookback'])

        # Skip tickers not in any pullback (< -1% from high)
        if dd_pct > -1.0:
            dd_natr = 0.0  # at highs

        # -- Vol expansion --
        vol_ratio = compute_vol_expansion(
            close, CONFIG['vol_short'], CONFIG['vol_long'])

        # -- Residual drawdown --
        residual_dd, beta = compute_residual_drawdown(
            close, spy_close, CONFIG['beta_lookback'], CONFIG['drawdown_lookback'])

        # -- Historical recovery --
        recovery_median, recovery_n = compute_historical_recovery(
            close, natr, dd_natr,
            CONFIG['recovery_lookback'], CONFIG['recovery_forward'])

        # -- SMA structure (from momentum ranker or compute) --
        mr = mr_data.get(ticker, {})
        sma_count = 0
        sma_flags = {}
        for w in [30, 50, 100, 200]:
            key = 'sma{}'.format(w)
            if key in mr:
                val = bool(mr[key])
            else:
                # Compute from price data
                if len(close) >= w:
                    val = float(close.iloc[-1]) > float(close.rolling(w).mean().iloc[-1])
                else:
                    val = False
            sma_flags[key] = val
            if val:
                sma_count += 1

        # -- Slope stage overlay --
        ss = slope_data.get(ticker, {})
        stage = ss.get('stage', -1)
        tq_score = ss.get('tq_score')
        slope_pct = ss.get('slope_pct')
        stage_name = ss.get('stage_name', 'Unknown')

        if stage == -1:
            stage = 1  # default to basing if no data

        # -- Health score --
        health = compute_health_score(
            dd_natr, sma_count, stage, tq_score,
            vol_ratio, residual_dd, recovery_median)
        verdict = classify_health(health)

        # -- Volume context (display only, NOT scored) --
        vol_20 = df['volume'].iloc[-20:].mean() if len(df) >= 20 else 0
        vol_63 = df['volume'].iloc[-63:].mean() if len(df) >= 63 else vol_20
        vol_display = round(vol_20 / vol_63, 2) if vol_63 > 0 else 1.0

        return {
            'ticker':          ticker,
            'price':           round(float(close.iloc[-1]), 2),
            'dd_pct':          round(dd_pct, 2),
            'dd_natr':         round(dd_natr, 2),
            'natr':            round(float(natr.iloc[-1]), 2),
            'sma_count':       sma_count,
            'sma30':           sma_flags.get('sma30', False),
            'sma50':           sma_flags.get('sma50', False),
            'sma100':          sma_flags.get('sma100', False),
            'sma200':          sma_flags.get('sma200', False),
            'stage':           stage,
            'stage_name':      stage_name,
            'tq_score':        tq_score,
            'slope_pct':       slope_pct,
            'vol_ratio':       round(vol_ratio, 2),
            'vol_display':     vol_display,
            'beta':            beta,
            'residual_dd':     round(residual_dd, 2),
            'recovery_median': round(recovery_median, 1) if recovery_median is not None else None,
            'recovery_n':      recovery_n,
            'health':          health,
            'verdict':         verdict,
            # Momentum ranker overlays
            'mr_rank':         mr.get('rank'),
            'mr_score':        mr.get('score'),
            'ret_1w':          mr.get('ret_1w'),
            'ret_1m':          mr.get('ret_1m'),
        }

    except Exception:
        return None


# =============================================================================
# COMPUTATION PIPELINE
# =============================================================================

def _load_and_process(pkl_path, spy_close, slope_data, mr_data):
    """Worker: load pkl and process."""
    ticker = pkl_path.stem
    if ticker in ('aggregated_price_data', 'SPY'):
        return None
    try:
        df = pd.read_pickle(str(pkl_path))
        if 'close' not in df.columns or 'high' not in df.columns or 'low' not in df.columns:
            return None
        return process_single_asset(ticker, df, spy_close, slope_data, mr_data)
    except Exception:
        return None


def run_computation():
    """Run full computation pipeline."""
    pkl_files = [f for f in Path(CACHE_DIR).glob('*.pkl')
                 if f.stem != 'aggregated_price_data']
    total = len(pkl_files)
    print("[LOAD] Found {} price cache files".format(total))

    # Load overlays
    slope_data = load_slope_stage_data()
    print("[LOAD] Slope stage data: {} tickers".format(len(slope_data)))

    mr_data = load_momentum_ranker_data()
    print("[LOAD] Momentum ranker data: {} tickers".format(len(mr_data)))

    # Load SPY
    spy_df = load_spy_data()
    if spy_df is None:
        print("[ERROR] Cannot proceed without SPY data")
        return []
    spy_close = spy_df['close']

    # Also process SPY itself for context
    spy_result = process_single_asset('SPY', spy_df, spy_close, slope_data, mr_data)

    # Process all assets
    print("[COMPUTE] Processing {} assets with {} workers...".format(
        total, CONFIG['max_workers']))
    start_time = time.time()

    results = []
    with ThreadPoolExecutor(max_workers=CONFIG['max_workers']) as executor:
        futures = {
            executor.submit(_load_and_process, f, spy_close, slope_data, mr_data): f
            for f in pkl_files
        }
        done_count = 0
        for future in as_completed(futures):
            done_count += 1
            if done_count % 200 == 0:
                print("  ... processed {}/{}".format(done_count, total))
            result = future.result()
            if result is not None:
                results.append(result)

    # Add SPY
    if spy_result is not None:
        results.append(spy_result)

    elapsed = time.time() - start_time
    print("[OK] Processed {} assets in {:.1f}s ({} valid results)".format(
        total, elapsed, len(results)))

    # Save cache
    cache_data = {
        'total_assets': len(results),
        'generated_at': datetime.datetime.now().isoformat(),
        'spy_dd_pct':   spy_result['dd_pct'] if spy_result else 0.0,
        'results':      clean_nan(results),
    }
    with open(PULLBACK_CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(cache_data, f, indent=2)
    print("[CACHE] Saved to {}".format(PULLBACK_CACHE_FILE))

    return results, spy_result


# =============================================================================
# HTML BUILDING
# =============================================================================

EXTRA_CSS = """
.health-pill {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 0.8em;
    font-weight: 700;
    letter-spacing: 0.03em;
}
.health-HEALTHY   { background: #dcfce7; color: #15803d; border: 1px solid #bbf7d0; }
.health-CAUTION   { background: #fef3c7; color: #b45309; border: 1px solid #fde68a; }
.health-WARNING   { background: #ffedd5; color: #c2410c; border: 1px solid #fed7aa; }
.health-BREAKDOWN { background: #fee2e2; color: #b91c1c; border: 1px solid #fca5a5; }

.sma-dot {
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    margin: 0 1px;
}
.sma-above { background: #16a34a; }
.sma-below { background: #dc2626; }

.metric-bar {
    background: #e5e7eb;
    border-radius: 4px;
    height: 18px;
    width: 80px;
    display: inline-block;
    overflow: hidden;
    vertical-align: middle;
}
.metric-bar-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.3s;
}

.pullback-section { margin-bottom: 24px; }
.filter-bar {
    display: flex; gap: 12px; align-items: center;
    margin-bottom: 12px; flex-wrap: wrap;
}
.filter-bar select, .filter-bar input {
    padding: 4px 8px;
    border: 1px solid #d1d5db;
    border-radius: 4px;
    font-size: 0.85em;
}
.filter-bar label { font-size: 0.85em; color: #555; }
"""

SORT_JS = """
(function() {
    function sortTable(th) {
        var table = th.closest('table');
        var tbody = table.querySelector('tbody');
        var idx   = Array.prototype.indexOf.call(th.parentNode.children, th);
        var asc   = th.classList.contains('sorted-desc') || !th.classList.contains('sorted-asc');

        th.parentNode.querySelectorAll('th').forEach(function(h) {
            h.classList.remove('sorted-asc', 'sorted-desc');
        });
        th.classList.add(asc ? 'sorted-asc' : 'sorted-desc');

        var rows = Array.prototype.slice.call(tbody.querySelectorAll('tr'));
        rows.sort(function(a, b) {
            var va = a.children[idx] ? a.children[idx].getAttribute('data-val') || a.children[idx].innerText : '';
            var vb = b.children[idx] ? b.children[idx].getAttribute('data-val') || b.children[idx].innerText : '';
            var na = parseFloat(va), nb = parseFloat(vb);
            if (!isNaN(na) && !isNaN(nb)) return asc ? na - nb : nb - na;
            return asc ? va.localeCompare(vb) : vb.localeCompare(va);
        });
        rows.forEach(function(r) { tbody.appendChild(r); });
    }

    document.addEventListener('DOMContentLoaded', function() {
        document.querySelectorAll('thead th').forEach(function(th) {
            th.addEventListener('click', function() { sortTable(th); });
        });

        // Filter: show only owned, or only pullbacks
        var filterOwned = document.getElementById('ph-filter-owned');
        var filterVerdict = document.getElementById('ph-filter-verdict');
        if (filterOwned) {
            filterOwned.addEventListener('change', applyFilters);
        }
        if (filterVerdict) {
            filterVerdict.addEventListener('change', applyFilters);
        }
    });

    function applyFilters() {
        var showOwned = document.getElementById('ph-filter-owned').value;
        var showVerdict = document.getElementById('ph-filter-verdict').value;
        var tables = document.querySelectorAll('table.ph-table tbody');

        tables.forEach(function(tbody) {
            var rows = tbody.querySelectorAll('tr');
            rows.forEach(function(row) {
                var isOwned = row.classList.contains('row-owned');
                var verdict = row.getAttribute('data-verdict') || '';
                var show = true;

                if (showOwned === 'owned' && !isOwned) show = false;
                if (showOwned === 'not-owned' && isOwned) show = false;
                if (showVerdict && showVerdict !== 'all' && verdict !== showVerdict) show = false;

                row.style.display = show ? '' : 'none';
            });
        });
    }

    // Expose for ownership toggle
    window._phApplyFilters = applyFilters;
})();
"""


def _own_cell(ticker):
    """Ownership checkbox cell with data-val for sorting."""
    return (
        '<td data-val="0"><input type="checkbox" class="own-cb" data-ticker="{t}"'
        ' onclick="window._ownToggle(\'{t}\', this); window._phApplyFilters();"'
        ' title="Mark as owned"></td>'
    ).format(t=ticker)


def _health_pill(verdict):
    """Inline health verdict pill."""
    return '<span class="health-pill health-{v}">{v}</span>'.format(v=verdict)


def _health_bar(score):
    """Mini bar showing health score 0-100."""
    if score >= 80:
        color = '#16a34a'
    elif score >= 50:
        color = '#d97706'
    elif score >= 20:
        color = '#f97316'
    else:
        color = '#dc2626'
    return (
        '<div class="metric-bar">'
        '<div class="metric-bar-fill" style="width:{w}%;background:{c};"></div>'
        '</div> <strong>{s:.0f}</strong>'
    ).format(w=min(100, score), c=color, s=score)


def _sma_dots(r):
    """4 dots for SMA status."""
    dots = []
    for w in [30, 50, 100, 200]:
        key = 'sma{}'.format(w)
        above = r.get(key, False)
        cls = 'sma-above' if above else 'sma-below'
        dots.append('<span class="sma-dot {c}" title="SMA{w}: {s}"></span>'.format(
            c=cls, w=w, s='above' if above else 'below'))
    return ''.join(dots)


def _num_cell(val, fmt="{:.2f}", suffix="", css=""):
    """Numeric table cell with data-val for sorting."""
    if val is None:
        return '<td class="num muted" data-val="0">n/a</td>'
    cls = css
    if not cls:
        if isinstance(val, (int, float)):
            if val > 0:
                cls = "pos"
            elif val < 0:
                cls = "neg"
            else:
                cls = "neutral"
    text = fmt.format(val) + suffix
    return '<td class="num {c}" data-val="{v}">{t}</td>'.format(
        c=cls, v=val, t=text)


def _stage_cell(stage, stage_name):
    """Stage cell with color."""
    colors = {0: 'neg', 1: 'warn', 2: 'pos', 3: 'warn'}
    css = colors.get(stage, 'neutral')
    return '<td class="num {c}" data-val="{s}">S{s}</td>'.format(c=css, s=stage)


def build_body_html(results, spy_result, writer):
    """Build full HTML body."""
    date_str = datetime.date.today().strftime("%Y-%m-%d")

    # Aggregate stats
    total = len(results)
    in_pullback = [r for r in results if r['dd_pct'] < -3.0]
    verdicts = {}
    for r in results:
        v = r['verdict']
        verdicts[v] = verdicts.get(v, 0) + 1

    spy_dd = spy_result['dd_pct'] if spy_result else 0.0

    parts = []

    # -- Stat bar --
    parts.append(writer.stat_bar([
        ("Date",         date_str,                                     "neutral"),
        ("Universe",     "{:,}".format(total),                         "neutral"),
        ("In Pullback",  "{:,} (>{:.0f}% DD)".format(len(in_pullback), 3), "warn" if in_pullback else "neutral"),
        ("SPY DD",       "{:.1f}%".format(spy_dd),                     "neg" if spy_dd < -3 else "neutral"),
        ("Healthy",      "{}".format(verdicts.get('HEALTHY', 0)),      "pos"),
        ("Warning+",     "{}".format(verdicts.get('WARNING', 0) + verdicts.get('BREAKDOWN', 0)),
                                                                       "neg" if verdicts.get('WARNING', 0) + verdicts.get('BREAKDOWN', 0) > 0 else "neutral"),
    ]))

    # -- Header --
    parts.append(writer.build_header(
        "Pullback Health &nbsp;|&nbsp; Is It Normal or Worse?"
    ))

    # -- LLM description --
    parts.append(writer.llm_block())

    # -- Market context banner --
    if spy_dd < -5:
        banner = "MARKET CORRECTION"
        bcolor = "#dc2626"
        btext = "SPY is {:.1f}% off highs. Most pullbacks are systemic.".format(spy_dd)
    elif spy_dd < -3:
        banner = "MARKET PULLBACK"
        bcolor = "#d97706"
        btext = "SPY is {:.1f}% off highs. Watch for stock-specific weakness.".format(spy_dd)
    elif spy_dd < -1:
        banner = "MILD DIP"
        bcolor = "#0ea5e9"
        btext = "SPY is {:.1f}% off highs. Normal range.".format(spy_dd)
    else:
        banner = "NEAR HIGHS"
        bcolor = "#16a34a"
        btext = "SPY is {:.1f}% from highs. Stock-specific pullbacks are idiosyncratic.".format(spy_dd)

    parts.append(writer.regime_banner(banner, btext, color=bcolor))

    # -- Filter bar --
    filter_html = (
        '<div class="filter-bar">'
        '<label>Show:</label>'
        '<select id="ph-filter-owned">'
        '<option value="all">All tickers</option>'
        '<option value="owned">Owned only</option>'
        '<option value="not-owned">Not owned</option>'
        '</select>'
        '<label>Verdict:</label>'
        '<select id="ph-filter-verdict">'
        '<option value="all">All</option>'
        '<option value="HEALTHY">Healthy</option>'
        '<option value="CAUTION">Caution</option>'
        '<option value="WARNING">Warning</option>'
        '<option value="BREAKDOWN">Breakdown</option>'
        '</select>'
        '</div>'
    )

    # -- Main table --
    # Sort by health score descending (healthiest first)
    sorted_results = sorted(results, key=lambda x: x['health'], reverse=True)

    # Only show tickers in pullback (dd < -1%) or all
    pullback_results = [r for r in sorted_results if r['dd_pct'] < -1.0]

    headers = [
        "Own", "Ticker", "Price", "DD%", "DD NATR", "NATR%",
        "SMAs", "Stg", "VolX", "Beta",
        "Resid", "Recov", "Health", "Verdict",
    ]

    rows = []
    for r in pullback_results:
        # Health bar
        health_html = _health_bar(r['health'])

        # Recovery display
        if r['recovery_median'] is not None:
            recov_text = '{:+.1f}%'.format(r['recovery_median'])
            recov_css = 'pos' if r['recovery_median'] > 0 else 'neg'
            recov_title = 'n={}'.format(r['recovery_n'])
        else:
            recov_text = 'n/a'
            recov_css = 'muted'
            recov_title = 'insufficient data'

        # DD color
        dd_css = 'neg' if r['dd_pct'] < -5 else 'warn' if r['dd_pct'] < -3 else 'neutral'

        # NATR severity color
        natr_css = 'neg' if r['dd_natr'] >= 3.0 else 'warn' if r['dd_natr'] >= 1.5 else 'pos'

        # Vol expansion color
        vol_css = 'neg' if r['vol_ratio'] >= 1.5 else 'warn' if r['vol_ratio'] >= 1.2 else 'pos'

        # Residual color
        resid_css = 'neg' if r['residual_dd'] >= 5.0 else 'warn' if r['residual_dd'] >= 2.0 else 'pos'

        rows.append(
            '<tr data-verdict="{verdict}">'
            '{own}'
            '<td><strong>{ticker}</strong></td>'
            '<td class="num neutral" data-val="{price}">${price:.2f}</td>'
            '<td class="num {dd_css}" data-val="{dd_pct}">{dd_pct:.1f}%</td>'
            '<td class="num {natr_css}" data-val="{dd_natr}">{dd_natr:.1f}</td>'
            '<td class="num neutral" data-val="{natr}">{natr:.2f}%</td>'
            '<td data-val="{sma_count}">{sma_dots} ({sma_count}/4)</td>'
            '{stage_cell}'
            '<td class="num {vol_css}" data-val="{vol_ratio}">{vol_ratio:.2f}x</td>'
            '<td class="num neutral" data-val="{beta}">{beta:.2f}</td>'
            '<td class="num {resid_css}" data-val="{residual_dd}">{residual_dd:+.1f}%</td>'
            '<td class="num {recov_css}" data-val="{recov_val}" title="{recov_title}">{recov_text}</td>'
            '<td data-val="{health}">{health_html}</td>'
            '<td>{verdict_pill}</td>'
            '</tr>'.format(
                verdict=r['verdict'],
                own=_own_cell(r['ticker']),
                ticker=r['ticker'],
                price=r['price'],
                dd_pct=r['dd_pct'],
                dd_css=dd_css,
                dd_natr=r['dd_natr'],
                natr=r['natr'],
                natr_css=natr_css,
                sma_count=r['sma_count'],
                sma_dots=_sma_dots(r),
                stage_cell=_stage_cell(r['stage'], r['stage_name']),
                vol_ratio=r['vol_ratio'],
                vol_css=vol_css,
                beta=r['beta'],
                residual_dd=r['residual_dd'],
                resid_css=resid_css,
                recov_val=r['recovery_median'] if r['recovery_median'] is not None else 0,
                recov_text=recov_text,
                recov_css=recov_css,
                recov_title=recov_title,
                health=r['health'],
                health_html=health_html,
                verdict_pill=_health_pill(r['verdict']),
            )
        )

    header_cells = "".join('<th>{}</th>'.format(h) for h in headers)
    table_html = (
        '<table class="ph-table">'
        '<thead><tr>{hdr}</tr></thead>'
        '<tbody>{rows}</tbody>'
        '</table>'.format(hdr=header_cells, rows="".join(rows))
    )

    parts.append(writer.section(
        "Pullback Monitor ({:,} tickers in pullback)".format(len(pullback_results)),
        '<p style="margin-bottom:8px;color:#555;font-size:0.9em;">'
        'All tickers with >1% drawdown from 63-day high. Sorted worst-first. '
        'Check "Own" to highlight your holdings, then filter to "Owned only."</p>'
        + filter_html + table_html,
        hint="Click column to sort | Use filters above"
    ))

    # -- Metric explainer --
    explainer = (
        '<table style="font-size:0.85em;">'
        '<tr><td><strong>DD%</strong></td><td>Drawdown from 63-day high</td></tr>'
        '<tr><td><strong>DD NATR</strong></td><td>Drawdown severity normalized by ATR/price. '
        '&lt;1.5 shallow, 1.5-3 normal, 3-5 deep, &gt;5 severe</td></tr>'
        '<tr><td><strong>NATR%</strong></td><td>Normalized ATR (ATR/price). Higher = more volatile stock</td></tr>'
        '<tr><td><strong>SMAs</strong></td><td>Price above 30/50/100/200-day SMA. '
        '<span class="sma-dot sma-above"></span>=above '
        '<span class="sma-dot sma-below"></span>=below. 4/4 = strong structure</td></tr>'
        '<tr><td><strong>Stg</strong></td><td>Slope stage (S0=decline, S1=basing, S2=uptrend, S3=parabolic)</td></tr>'
        '<tr><td><strong>VolX</strong></td><td>10-day vol / 63-day vol. &gt;1.5x = volatility expanding (instability)</td></tr>'
        '<tr><td><strong>Beta</strong></td><td>6-month beta vs SPY</td></tr>'
        '<tr><td><strong>Resid</strong></td><td>Residual DD = actual DD minus beta-expected DD. '
        'Positive = stock fell more than beta predicts (stock-specific weakness)</td></tr>'
        '<tr><td><strong>Recov</strong></td><td>Median 21-day forward return from similar past drawdowns for this stock</td></tr>'
        '<tr><td><strong>Health</strong></td><td>Composite score 0-100. '
        'NATR(25%) + SMA(25%) + Stage(20%) + VolX(10%) + Resid(10%) + Recov(10%)</td></tr>'
        '</table>'
    )
    parts.append(writer.section("Metric Guide", explainer))

    # -- Footer --
    parts.append(writer.footer())

    return "\n".join(parts)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("PULLBACK HEALTH MONITOR v1.0")
    print("=" * 70)
    print("Generated: {}".format(datetime.datetime.now().isoformat()))

    results, spy_result = run_computation()

    if not results:
        print("[ERROR] No results - aborting HTML write.")
        return

    # Summary
    verdicts = {}
    for r in results:
        v = r['verdict']
        verdicts[v] = verdicts.get(v, 0) + 1

    in_pullback = sum(1 for r in results if r['dd_pct'] < -3.0)

    print()
    print("=" * 70)
    print("PULLBACK HEALTH SUMMARY")
    print("=" * 70)
    print("  Universe:    {:,} assets".format(len(results)))
    print("  In pullback: {:,} (>3% DD)".format(in_pullback))
    print("  SPY DD:      {:.1f}%".format(spy_result['dd_pct'] if spy_result else 0))
    for v in ['HEALTHY', 'CAUTION', 'WARNING', 'BREAKDOWN']:
        print("  {:10s}   {:,}".format(v, verdicts.get(v, 0)))
    print("=" * 70)

    # Build and write dashboard HTML
    writer = DashboardWriter("pullback-health", "Pullback Health Monitor")
    body = build_body_html(results, spy_result, writer)
    writer.write(body, extra_css=EXTRA_CSS, extra_js=SORT_JS)

    print("[OK] Dashboard written.")


if __name__ == "__main__":
    main()
