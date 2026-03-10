# -*- coding: utf-8 -*-
# =============================================================================
# sma29_entry_backend.py - v1.2
# Last updated: 2026-03-10
# =============================================================================
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


def classify_extension(ext_pct):
    """Map extension % to bucket label, score, PF, and median fwd."""
    if ext_pct < 0:
        return BELOW_SMA29_LABEL, BELOW_SMA29_SCORE, None, None
    for lower, upper, pf, med_fwd, label, score in EXTENSION_BUCKETS:
        if lower <= ext_pct < upper:
            return label, score, pf, med_fwd
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
W_MOMENTUM = 0.35
W_HEALTH   = 0.30
W_EXTENSION = 0.35


def score_momentum(mr_data):
    """Score 0-100 from momentum ranker score. Already 0-100."""
    if mr_data is None:
        return 0
    return min(100, max(0, float(mr_data.get('score', 0))))


def score_health(pb_data):
    """Score 0-100 from pullback health. Already 0-100."""
    if pb_data is None:
        return 0
    return min(100, max(0, float(pb_data.get('health', 0) or 0)))


def score_extension(ext_pct):
    """Score 0-100 from SMA29 extension bucket."""
    _, score, _, _ = classify_extension(ext_pct)
    return score


def combined_score(momentum_score, health_score, extension_score):
    """Weighted composite."""
    return round(
        W_MOMENTUM * momentum_score +
        W_HEALTH * health_score +
        W_EXTENSION * extension_score, 1
    )


# =============================================================================
# MAIN COMPUTATION
# =============================================================================

def build_universe():
    """Build scored universe from all 3 data sources."""
    print("[1/4] Loading momentum ranker data...")
    mr_data = load_momentum_data()
    print("  {} tickers".format(len(mr_data)))

    print("[2/4] Loading pullback health data...")
    pb_data = load_pullback_data()
    print("  {} tickers".format(len(pb_data)))

    # Universe = union of momentum ranker + pullback health tickers
    all_tickers = sorted(set(list(mr_data.keys()) + list(pb_data.keys())))
    print("[3/4] Computing SMA29 extensions for {} tickers...".format(len(all_tickers)))

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

    print("[4/4] Scoring...")
    results = []
    for ticker in all_tickers:
        mr = mr_data.get(ticker)
        pb = pb_data.get(ticker)
        ext = ext_data.get(ticker)

        if ext is None:
            continue  # need at least price data

        ext_pct = ext['extension_pct']
        ext_label, ext_score_val, ext_pf, ext_fwd = classify_extension(ext_pct)

        # SMA10 exit alert
        ext10_pct = ext.get('ext10_pct')
        exit_alert, exit_pf, exit_fwd = classify_sma10_exit(ext10_pct)

        m_score = score_momentum(mr)
        h_score = score_health(pb)
        e_score = ext_score_val

        combo = combined_score(m_score, h_score, e_score)

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
            'health_score': round(h_score, 1),
            'health_verdict': pb.get('verdict', '') if pb else '',
            'dd_pct': round(float(pb.get('dd_pct', 0) or 0), 1) if pb else None,
            'stage': pb.get('stage', None) if pb else None,
            'stage_name': pb.get('stage_name', '') if pb else '',
            'ret_1w': round(float(mr.get('ret_1w', 0) or 0), 2) if mr else None,
            'ret_1m': round(float(mr.get('ret_1m', 0) or 0), 2) if mr else None,
            'ret_3m': round(float(mr.get('ret_3m', 0) or 0), 2) if mr else None,
            'top_pole': mr.get('top_pole', '') if mr else '',
            'combined_score': combo,
        }
        results.append(row)

    results.sort(key=lambda r: r['combined_score'], reverse=True)
    return results


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

// Filter by extension zone
function filterZone(zone) {
    var btn = document.getElementById('btn-' + zone);
    var active = btn && btn.classList.contains('active');

    // Toggle all buttons off
    document.querySelectorAll('.zone-btn').forEach(function(b) { b.classList.remove('active'); });

    if (!active && zone !== 'ALL' && btn) { btn.classList.add('active'); }
    applyFilters();
}

// Filter by ownership/watchlist
var _ownedFilter = 'all';
function filterOwned(mode) {
    _ownedFilter = mode;
    applyFilters();
}
function applyFilters() {
    var table = document.getElementById('mainTable');
    var rows = table.tBodies[0].rows;
    var activeZone = null;
    document.querySelectorAll('.zone-btn.active').forEach(function(b) {
        activeZone = b.id ? b.id.replace('btn-', '') : null;
    });
    for (var i = 0; i < rows.length; i++) {
        var show = true;
        // Zone filter
        if (activeZone) {
            var rowZone = rows[i].getAttribute('data-zone') || '';
            if (rowZone !== activeZone) show = false;
        }
        // Ownership filter
        if (show && _ownedFilter !== 'all') {
            var ticker = rows[i].querySelector('.own-cb');
            if (ticker) ticker = ticker.getAttribute('data-ticker');
            var isOwned = ticker && window._owned && window._owned.has(ticker);
            var isWatched = ticker && window._watched && window._watched.has(ticker);
            if (_ownedFilter === 'owned' && !isOwned) show = false;
            if (_ownedFilter === 'watched' && !isWatched) show = false;
            if (_ownedFilter === 'not-owned' && isOwned) show = false;
        }
        rows[i].style.display = show ? '' : 'none';
    }
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


def build_html(results):
    """Build the dashboard HTML."""
    now = datetime.now()

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

    # Zone filter buttons
    zone_order = ['OPTIMAL', 'GOOD', 'FAIR', 'CAUTION', 'WARNING', 'DANGER', 'EXTREME', 'BELOW SMA']
    zone_btns = '<div style="margin-bottom:14px">'
    zone_btns += '<button class="zone-btn" onclick="filterZone(\'ALL\')" style="margin:2px 4px;padding:4px 12px;border:1px solid #ccc;border-radius:4px;cursor:pointer;background:#f3f4f6">ALL</button>'
    for z in zone_order:
        cnt = zone_counts.get(z, 0)
        if cnt == 0:
            continue
        css = z.replace(' ', '')
        if css == 'BELOWSMA':
            css = 'BELOW'
        zone_btns += '<button id="btn-{z}" class="zone-btn ext-badge ext-{css}" onclick="filterZone(\'{z}\')" style="margin:2px 4px;padding:4px 12px;cursor:pointer;border:1px solid #ccc;border-radius:4px">{z} ({cnt})</button>'.format(
            z=z, css=css, cnt=cnt)
    zone_btns += ' <label style="margin-left:16px;font-size:0.85em">Show: '
    zone_btns += '<select id="filter-owned" onchange="filterOwned(this.value)" style="padding:3px 8px;border:1px solid #ccc;border-radius:4px">'
    zone_btns += '<option value="all">All Stocks</option>'
    zone_btns += '<option value="owned">Owned Only</option>'
    zone_btns += '<option value="watched">Watched Only</option>'
    zone_btns += '<option value="not-owned">Not Owned</option>'
    zone_btns += '</select></label>'
    zone_btns += '</div>'

    # Methodology note
    methodology = """
    <details style="margin-bottom:16px">
    <summary style="cursor:pointer;font-weight:600;color:#4a5568">Methodology (scimode-validated)</summary>
    <div style="padding:8px 12px;font-size:0.82em;color:#555;line-height:1.6">
    <p><b>Combined Score</b> = 35% Momentum + 30% Pullback Health + 35% SMA29 Extension</p>
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
    <p><b>Health Score</b>: From pullback_health (NATR drawdown, SMA structure, slope stage, vol expansion, beta-adjusted DD, historical recovery).</p>
    <p><b>Ideal entry</b>: High momentum + healthy pullback + optimal SMA29 zone (0-10% extension, PF ~1.5). These are stocks trending up with room to run.</p>
    <p style="margin-top:10px"><b>SMA10 Exit Alert</b> (scimode PF validation, 468 tickers):</p>
    <p>SMA10 detects overextension faster than SMA29. PF validates exit signals.</p>
    <table style="font-size:0.9em;border-collapse:collapse;margin:6px 0">
    <tr><th style="padding:2px 8px;text-align:left">Alert</th><th style="padding:2px 8px">SMA10 Ext</th><th style="padding:2px 8px">PF</th><th style="padding:2px 8px">Median 21d Fwd</th></tr>
    <tr><td style="padding:2px 8px"><span class="exit-alert">EXIT ALERT</span></td><td style="padding:2px 8px;text-align:center">25%+</td><td style="padding:2px 8px;text-align:center;color:#dc2626">1.11 (med: 0.90)</td><td style="padding:2px 8px;text-align:center;color:#dc2626">-2.21%</td></tr>
    <tr><td style="padding:2px 8px"><span class="exit-watch">EXIT WATCH</span></td><td style="padding:2px 8px;text-align:center">15-25%</td><td style="padding:2px 8px;text-align:center;color:#f97316">1.59 (med: 1.07)</td><td style="padding:2px 8px;text-align:center;color:#f97316">+0.49%</td></tr>
    <tr><td style="padding:2px 8px"><span class="exit-elevated">ELEVATED</span></td><td style="padding:2px 8px;text-align:center">10-15%</td><td style="padding:2px 8px;text-align:center">1.51</td><td style="padding:2px 8px;text-align:center">+0.60%</td></tr>
    </table>
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
        <th onclick="sortTable(4,true)" style="cursor:pointer" title="Weighted composite: 35% Momentum + 30% Pullback Health + 35% SMA29 Extension. Higher = better entry quality. 0-100 scale.">Combined</th>
        <th onclick="sortTable(5,true)" style="cursor:pointer" title="SMA10 exit alert based on scimode PF validation (468 tickers). EXIT ALERT = 25%+ above SMA10 (PF=1.11, PF_median=0.90). EXIT WATCH = 15-25% (PF=1.59, PF_median=1.07).">Exit</th>
        <th onclick="sortTable(6,true)" style="cursor:pointer" title="% distance of close above 29-day SMA. Positive = above SMA29, negative = below.">Ext %</th>
        <th onclick="sortTable(7,false)" style="cursor:pointer" title="SMA29 extension zone from scimode PF validation (468 tickers, trending filter). OPTIMAL = 0-10% above SMA29 (PF ~1.5). Zones scored by profit factor and median forward return.">Zone</th>
        <th onclick="sortTable(8,true)" style="cursor:pointer" title="Momentum ranker composite score (0-100). Blend of: multi-period returns (1d-1y), ratio quality (acceleration checks), and bad-SPY-day resilience. Gated by SMA structure.">Momentum</th>
        <th onclick="sortTable(9,true)" style="cursor:pointer" title="Pullback health score (0-100). Blend of: drawdown severity (NATR-adjusted), SMA structure (30/50/100/200), slope stage, vol expansion, beta-adjusted DD, and historical recovery rate.">Health</th>
        <th onclick="sortTable(10,true)" style="cursor:pointer" title="Extension bucket score (0-95). From SMA29 zone: OPTIMAL=90-95, GOOD=75, FAIR=55, CAUTION=40, WARNING=20-25, DANGER=10, EXTREME=0.">Ext Score</th>
        <th onclick="sortTable(11,true)" style="cursor:pointer" title="Profit Factor for this SMA29 extension zone = sum(wins)/sum(|losses|). From scimode PF validation (468 tickers, trending filter). PF>1.0 = profitable, PF<1.0 = losing. High-extension PF inflated by lottery tails.">PF</th>
        <th onclick="sortTable(12,true)" style="cursor:pointer" title="Median 21-day forward return for this SMA29 extension zone. From scimode OOS test. Negative in DANGER/EXTREME zones.">Med Fwd</th>
        <th onclick="sortTable(13,true)" style="cursor:pointer" title="R-squared of 21-day log-price regression. Measures trend quality: 1.0 = perfectly linear move, 0.0 = random walk. Above 0.70 = strong trend.">R-sq</th>
        <th onclick="sortTable(14,true)" style="cursor:pointer" title="Annualized slope from 21-day log-price regression. Shows how fast the stock is trending (% per year). Higher = steeper uptrend.">Slope %</th>
        <th onclick="sortTable(15,true)" style="cursor:pointer" title="Current drawdown from 63-day (3-month) rolling high. Shows how far the stock has pulled back from its recent peak. 0% = at high, -10% = pulled back 10%.">DD %</th>
        <th onclick="sortTable(16,false)" style="cursor:pointer" title="Trend stage from slope analysis. Decline = falling, Basing = bottoming, Uptrend = sustained rise, Parabolic = late-stage acceleration.">Stage</th>
        <th onclick="sortTable(17,true)" style="cursor:pointer" title="Total return over the past 5 trading days (1 week).">1W</th>
        <th onclick="sortTable(18,true)" style="cursor:pointer" title="Total return over the past 21 trading days (1 month).">1M</th>
        <th title="Highest-correlated asset from 20-asset correlation universe (63-day window). Shows what this stock moves most like. Only shown if correlation >= 0.30.">Pole</th>
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

        row = '<tr data-zone="{zone}">'.format(zone=zone_attr)
        row += _own_cell(r['ticker'])
        row += _watch_cell(r['ticker'])
        row += '<td class="tl"><b>{}</b> <span style="font-size:0.72em;color:#999">{}</span></td>'.format(r['ticker'], mr_rank_str)
        row += '<td class="tr" data-val="{}">${:,.2f}</td>'.format(r['price'], r['price'])
        row += '<td class="tr" data-val="{v}" style="font-weight:700;color:{c}">{v:.1f}</td>'.format(
            v=r['combined_score'],
            c='#166534' if r['combined_score'] >= 70 else '#854d0e' if r['combined_score'] >= 45 else '#991b1b')
        row += _exit_badge(r.get('exit_alert', ''))
        row += '<td class="tr" data-val="{}" style="color:{}">{}</td>'.format(r['extension_pct'], ext_color, ext_pct_str)
        row += '<td class="tc">{}</td>'.format(_ext_badge(r['extension_label']))
        row += '<td class="tr" data-val="{}">{:.0f}</td>'.format(r['momentum_score'], r['momentum_score'])
        row += '<td class="tr" data-val="{}">{:.0f}</td>'.format(r['health_score'], r['health_score'])
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
    banner_score = 'Weights: {}% Momentum + {}% Health + {}% Extension'.format(
        int(W_MOMENTUM * 100), int(W_HEALTH * 100), int(W_EXTENSION * 100))

    parts = []
    parts.append(dw.build_header(subtitle='SMA29 Entry + SMA10 Exit | Scimode-Validated'))
    parts.append(dw.stat_bar(stat_bar_data))
    parts.append(dw.regime_banner(banner_text, banner_score, color=banner_color))
    parts.append(dw.section('Methodology', methodology, hint='Scimode exit_signal_test.py'))
    parts.append(zone_btns)
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
    print("ENTER & EXIT QUALITY SCANNER v1.2")
    print("=" * 70)

    results = build_universe()

    # Save JSON cache
    output = clean_nan({
        'generated_at': datetime.now().isoformat(),
        'total': len(results),
        'weights': {'momentum': W_MOMENTUM, 'health': W_HEALTH, 'extension': W_EXTENSION},
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
    if results:
        print("  Top 5:")
        for r in results[:5]:
            print("    {} {:>7.1f}  ext={:+.1f}% [{}]  mom={:.0f}  health={:.0f}".format(
                r['ticker'].ljust(6), r['combined_score'], r['extension_pct'],
                r['extension_label'], r['momentum_score'], r['health_score']))

    out_path = build_html(results)
    print()
    print("[OK] Dashboard: {}".format(out_path))


if __name__ == '__main__':
    main()
