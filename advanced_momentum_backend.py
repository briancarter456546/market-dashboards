# -*- coding: utf-8 -*-
# =============================================================================
# ADVANCED MOMENTUM BACKEND - v1.0
# Last updated: 2026-02-26
# =============================================================================
# Converted from advanced_momentum_analyzer_v3.3.py
# - All computation logic preserved (OBV, Sortino, slope accel, trajectory,
#   volume-price divergence, signals, estimates)
# - Output: HTML dashboard via DashboardWriter (NO JSON output)
# - Cache: same-day JSON cache at _DATA_DIR/advanced_momentum_analysis.json
#   (speeds up reruns; not the deliverable)
#
# Run from any directory:
#   python advanced_momentum_backend.py
# =============================================================================

import os
import pickle
import json
import math
import datetime
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm

from dashboard_writer import DashboardWriter

# =============================================================================
# PATHS
# =============================================================================

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR   = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", "perplexity-user-data"))

# =============================================================================
# CONFIG
# =============================================================================

CONFIG = {
    "price_cache_dir":  os.path.join(_DATA_DIR, "price_cache"),
    "cache_file":       os.path.join(_DATA_DIR, "advanced_momentum_analysis.json"),
    "min_data_points":  252,
    "rolling_window":   20,
    "accel_threshold":  0.03,
    "divergence_window": 20,
    "max_workers":      8,
}

# =============================================================================
# NaN HANDLING
# =============================================================================

def clean_nan(obj):
    """Recursively convert NaN/inf values to None for JSON compatibility."""
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    elif isinstance(obj, dict):
        return {k: clean_nan(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_nan(item) for item in obj]
    return obj

# =============================================================================
# CACHING  (same-day invalidation + asset-count guard)
# =============================================================================

def _is_same_trading_day(dt1, dt2):
    """Return True if two datetimes fall on the same market trading day."""
    if dt1.date() == dt2.date():
        return True
    if dt1.weekday() >= 5 and dt2.weekday() >= 5:
        return True
    if dt1.weekday() == 4 and dt2.weekday() in [5, 6]:
        return True
    return False


def load_cached_results(current_symbol_count):
    """Return cached result list if it is fresh and asset-count matches."""
    cache_file = CONFIG["cache_file"]
    if not os.path.exists(cache_file):
        return None

    cache_ts = datetime.datetime.fromtimestamp(os.path.getmtime(cache_file))
    now = datetime.datetime.now()

    if _is_same_trading_day(cache_ts, now):
        with open(cache_file, "r", encoding="utf-8") as fh:
            cached = json.load(fh)
        cached_count = cached.get("total_assets", 0)
        if cached_count == current_symbol_count:
            print("[CACHE] Same trading day - loading {} cached results...".format(cached_count))
            return cached.get("results", None)
        else:
            print("[CACHE] Asset count changed ({} -> {}), rebuilding...".format(
                cached_count, current_symbol_count))
    else:
        print("[CACHE] New trading day detected, rebuilding...")

    return None


def save_cache(results):
    """Persist results list to the cache file."""
    data = {
        "total_assets":   len(results),
        "generated_at":   datetime.datetime.now().isoformat(),
        "results":        clean_nan(results),
    }
    with open(CONFIG["cache_file"], "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    print("[CACHE] Saved {} results to {}".format(len(results), CONFIG["cache_file"]))

# =============================================================================
# DATA LOADING
# =============================================================================

def load_universe():
    """Return sorted list of ticker symbols found in price_cache."""
    cache_dir = CONFIG["price_cache_dir"]
    symbols = [
        os.path.splitext(fname)[0]
        for fname in os.listdir(cache_dir)
        if fname.endswith(".pkl")
    ]
    return sorted(symbols)


def load_price_data(symbol):
    """Load a single symbol's DataFrame from its .pkl file."""
    try:
        filepath = os.path.join(CONFIG["price_cache_dir"], "{}.pkl".format(symbol))
        with open(filepath, "rb") as fh:
            return pickle.load(fh)
    except Exception:
        return None

# =============================================================================
# OBV
# =============================================================================

def calculate_obv(prices, volumes):
    """Vectorised On-Balance Volume."""
    direction = np.sign(prices.diff())
    obv = (volumes * direction).cumsum()
    obv.iloc[0] = 0
    return obv


def detect_volume_price_divergence(prices, obv, window=20):
    """Detect bullish / bearish volume-price divergence over the last `window` bars."""
    if len(prices) < window or len(obv) < window:
        return {"has_divergence": False, "type": None, "strength": 0}

    rp = prices.iloc[-window:]
    ro = obv.iloc[-window:]
    x  = np.arange(window)

    price_slope, _ = np.polyfit(x, rp.values, 1)
    obv_slope,   _ = np.polyfit(x, ro.values, 1)

    p_norm = price_slope / rp.mean() if rp.mean() != 0 else 0
    o_norm = obv_slope   / abs(ro.mean()) if ro.mean() != 0 else 0

    has_div  = False
    div_type = None
    strength = 0.0

    if p_norm > 0.01 and o_norm < -0.01:
        has_div  = True
        div_type = "bearish"
        strength = abs(p_norm - o_norm)
    elif p_norm < -0.01 and o_norm > 0.01:
        has_div  = True
        div_type = "bullish"
        strength = abs(p_norm - o_norm)

    return {
        "has_divergence": has_div,
        "type":           div_type,
        "strength":       float(strength),
    }

# =============================================================================
# SORTINO
# =============================================================================

def calculate_sortino_ratios(prices):
    """Sortino ratios at 1-month and 3-month horizons from daily returns."""
    daily_returns = prices.pct_change().iloc[1:]

    if len(daily_returns) < 21:
        return {"sortino_1m": None, "sortino_3m": None}

    # --- 1-month ---
    r1m = daily_returns.iloc[-21:]
    dn1m = r1m[r1m < 0]
    if len(dn1m) > 0:
        ds1m = dn1m.std()
        tot1m = prices.iloc[-1] / prices.iloc[-21] - 1
        ann1m = tot1m * 12
        sortino_1m = ann1m / (ds1m * np.sqrt(252)) if ds1m > 0 else (999 if tot1m > 0 else 0)
    else:
        sortino_1m = 999

    # --- 3-month ---
    sortino_3m = None
    if len(daily_returns) >= 63:
        r3m = daily_returns.iloc[-63:]
        dn3m = r3m[r3m < 0]
        if len(dn3m) > 0:
            ds3m = dn3m.std()
            tot3m = prices.iloc[-1] / prices.iloc[-63] - 1
            ann3m = tot3m * 4
            sortino_3m = ann3m / (ds3m * np.sqrt(252)) if ds3m > 0 else (999 if tot3m > 0 else 0)
        else:
            sortino_3m = 999

    return {
        "sortino_1m": float(sortino_1m) if sortino_1m is not None else None,
        "sortino_3m": float(sortino_3m) if sortino_3m is not None else None,
    }

# =============================================================================
# SLOPE / ACCELERATION
# =============================================================================

def calculate_weekly_returns(prices):
    """Weekly returns series (NaN-safe via .iloc[1:])."""
    weekly = prices.resample("W").last()
    wr = weekly.pct_change(fill_method=None) * 100
    return wr.iloc[1:]


def calculate_return_slope(weekly_returns, window=20):
    """Rolling linear slope of weekly returns."""
    slopes = []
    for i in range(window, len(weekly_returns)):
        chunk = weekly_returns.iloc[i - window:i].values
        x     = np.arange(len(chunk))
        s, _  = np.polyfit(x, chunk, 1)
        slopes.append(s)
    return pd.Series(slopes, index=weekly_returns.index[window:])


def calculate_slope_acceleration(slopes, window=4):
    """Rate-of-change of the slope series (second derivative)."""
    accels = []
    for i in range(window, len(slopes)):
        chunk = slopes.iloc[i - window:i].values
        x     = np.arange(len(chunk))
        a, _  = np.polyfit(x, chunk, 1)
        accels.append(a)
    return pd.Series(accels, index=slopes.index[window:])

# =============================================================================
# TRAJECTORY DETECTION
# =============================================================================

def detect_trajectory_daily(prices, window=20):
    """Classify recent price action as V-shaped / Fast / Slow / Curving."""
    if len(prices) < window:
        return "Curving"

    daily_returns = prices.pct_change().iloc[1:]
    if len(daily_returns) < window:
        return "Curving"

    recent = daily_returns.iloc[-window:]
    x = np.arange(len(recent))
    slope, _ = np.polyfit(x, recent.values, 1)

    sub_slopes = []
    for i in range(5, len(recent)):
        chunk = recent.iloc[i - 5:i].values
        xs = np.arange(len(chunk))
        ss, _ = np.polyfit(xs, chunk, 1)
        sub_slopes.append(ss)

    if sub_slopes:
        slope_vol = np.std(sub_slopes)
        if slope_vol > 0.002:
            return "V-shaped"
        elif slope > 0.001:
            return "Fast"
        elif abs(slope) < 0.0005:
            return "Slow"

    return "Curving"

# =============================================================================
# PER-ASSET ANALYSIS
# =============================================================================

def analyze_asset(symbol, price_df):
    """Full momentum analysis for one symbol. Returns dict or None."""
    if price_df is None or len(price_df) < CONFIG["min_data_points"]:
        return None

    # Resolve price column
    if "Close" in price_df.columns:
        prices = price_df["Close"].copy()
    elif "close" in price_df.columns:
        prices = price_df["close"].copy()
    else:
        return None

    # OBV / divergence
    has_vol = "Volume" in price_df.columns or "volume" in price_df.columns
    if has_vol:
        volumes   = price_df["Volume"].copy() if "Volume" in price_df.columns \
                    else price_df["volume"].copy()
        obv       = calculate_obv(prices, volumes)
        divergence = detect_volume_price_divergence(
            prices, obv, CONFIG["divergence_window"])
    else:
        divergence = {"has_divergence": False, "type": None, "strength": 0.0}

    # Sortino
    sortino = calculate_sortino_ratios(prices)

    # Weekly returns
    weekly_returns = calculate_weekly_returns(prices)
    if len(weekly_returns) < CONFIG["rolling_window"] + 10:
        return None

    # Slope
    slopes = calculate_return_slope(weekly_returns, CONFIG["rolling_window"])
    if len(slopes) < 10:
        return None

    # Acceleration
    accelerations = calculate_slope_acceleration(slopes, window=4)
    if len(accelerations) < 5:
        return None

    current_slope = slopes.iloc[-1]
    current_accel = accelerations.iloc[-1]
    slope_trend   = slopes.iloc[-4:].mean()
    returns_3m    = weekly_returns.iloc[-12:].mean()

    # Status
    thresh = CONFIG["accel_threshold"]
    if current_accel > thresh:
        status         = "Accelerating"
        is_accelerating = True
        is_decelerating = False
    elif current_accel < -thresh:
        status         = "Decelerating"
        is_accelerating = False
        is_decelerating = True
    else:
        status         = "Stable"
        is_accelerating = False
        is_decelerating = False

    trajectory = detect_trajectory_daily(prices, window=20)

    # Downside risk ratio
    daily_returns = prices.pct_change().iloc[1:]
    if len(daily_returns) >= 20:
        rd20 = daily_returns.iloc[-20:]
        dn20 = rd20[rd20 < 0]
        if len(dn20) > 0:
            ds_recent = dn20.std()
            if len(daily_returns) >= 60:
                d60 = daily_returns.iloc[-60:]
                avg_ds = d60[d60 < 0].std()
            else:
                avg_ds = ds_recent
            downside_ratio = ds_recent / avg_ds if avg_ds > 0 else 1.0
        else:
            downside_ratio = 0.1
    else:
        downside_ratio = 1.0

    # Future estimates
    days_to_peak  = None
    days_to_nadir = None

    if returns_3m > 0:
        if is_accelerating or status == "Stable":
            if divergence["has_divergence"] and divergence["type"] == "bearish":
                days_to_peak = int(30 / (1 + divergence["strength"] * 10))
                days_to_peak = max(5, min(days_to_peak, 90))
            else:
                days_to_peak = int(60 / (1 + abs(current_slope) * 10))
                days_to_peak = max(30, min(days_to_peak, 180))
        elif is_decelerating:
            if abs(current_slope) > 0.01:
                weeks_to_zero = abs(returns_3m / abs(slope_trend)) if slope_trend != 0 else 60
                days_to_peak  = int(weeks_to_zero * 5)
                days_to_peak  = max(5, min(days_to_peak, 180))
            else:
                days_to_peak = 30
    elif returns_3m < 0:
        if is_accelerating:
            if abs(slope_trend) > 0.01:
                weeks_to_pos  = abs(returns_3m / slope_trend)
                days_to_nadir = int(weeks_to_pos * 5)
                days_to_nadir = max(5, min(days_to_nadir, 180))
            else:
                days_to_nadir = 90
        else:
            days_to_nadir = int(120 / (1 + abs(current_slope) * 10))
            days_to_nadir = max(30, min(days_to_nadir, 270))

    # Price returns
    cur  = prices.iloc[-1]
    p1m  = prices.iloc[-21]  if len(prices) >= 21  else cur
    p3m  = prices.iloc[-63]  if len(prices) >= 63  else cur
    p12m = prices.iloc[-252] if len(prices) >= 252 else cur

    ret_1m  = (cur / p1m  - 1) * 100
    ret_3m  = (cur / p3m  - 1) * 100
    ret_12m = (cur / p12m - 1) * 100

    # Signal
    if ret_3m > 5 and is_accelerating:
        signal     = "STRONG_BUY"
        confidence = 90
    elif ret_3m > 0 and is_accelerating:
        signal     = "BUY"
        confidence = 75
    elif is_decelerating and returns_3m > 0 and days_to_peak and days_to_peak <= 60:
        signal     = "SELL_SOON"
        confidence = 70
    elif returns_3m < 0:
        signal     = "SELL"
        confidence = 85
    else:
        signal     = "HOLD"
        confidence = 50

    return {
        "symbol":             symbol,
        "current_price":      float(cur),
        "return_1m":          float(ret_1m),
        "return_3m":          float(ret_3m),
        "return_12m":         float(ret_12m),
        "sortino_1m":         sortino["sortino_1m"],
        "sortino_3m":         sortino["sortino_3m"],
        "current_slope":      float(current_slope),
        "slope_trend":        float(slope_trend),
        "slope_acceleration": float(current_accel),
        "status":             status,
        "is_accelerating":    bool(is_accelerating),
        "is_decelerating":    bool(is_decelerating),
        "trajectory":         trajectory,
        "volatility_ratio":   float(downside_ratio),
        "has_divergence":     bool(divergence["has_divergence"]),
        "divergence_type":    divergence["type"],
        "divergence_strength": float(divergence["strength"]),
        "days_to_peak":       int(days_to_peak)  if days_to_peak  is not None else None,
        "days_to_nadir":      int(days_to_nadir) if days_to_nadir is not None else None,
        "signal":             signal,
        "confidence":         int(confidence),
    }

# =============================================================================
# PARALLEL PROCESSING
# =============================================================================

def process_all_assets(symbols):
    """Analyse all assets in parallel. Returns list of result dicts."""
    n_workers = CONFIG["max_workers"]
    print("[ANALYZE] Processing {} assets (workers={})...".format(len(symbols), n_workers))

    results  = []
    failures = {"no_data": 0, "insufficient_data": 0, "no_close": 0, "other": 0}

    def _worker(sym):
        try:
            df = load_price_data(sym)
            if df is None:
                return None, "no_data"
            if len(df) < CONFIG["min_data_points"]:
                return None, "insufficient_data"
            if "close" not in df.columns and "Close" not in df.columns:
                return None, "no_close"
            res = analyze_asset(sym, df)
            return (res, None) if res else (None, "other")
        except Exception:
            return None, "other"

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        fmap = {pool.submit(_worker, sym): sym for sym in symbols}
        for fut in tqdm(as_completed(fmap), total=len(symbols), desc="Analyzing"):
            res, err = fut.result()
            if res:
                results.append(res)
            elif err:
                failures[err] += 1

    print("[ANALYZE] Succeeded: {}  |  Failures: {}".format(
        len(results),
        "  ".join("{}: {}".format(k, v) for k, v in failures.items() if v > 0)
    ))
    return results

# =============================================================================
# HTML HELPERS
# =============================================================================

# Signal badge CSS class lookup
_SIGNAL_CLASS = {
    "STRONG_BUY": "sig-strong-buy",
    "BUY":        "sig-buy",
    "HOLD":       "sig-hold",
    "SELL_SOON":  "sig-sell-soon",
    "SELL":       "sig-sell",
}

# Status badge CSS class lookup
_STATUS_CLASS = {
    "Accelerating": "stat-accel",
    "Stable":       "stat-stable",
    "Decelerating": "stat-decel",
}

# Trajectory badge CSS class lookup
_TRAJ_CLASS = {
    "V-shaped": "traj-v",
    "Fast":     "traj-fast",
    "Slow":     "traj-slow",
    "Curving":  "traj-curv",
}

# Extra CSS for signal / trajectory / status badges
EXTRA_CSS = """
/* === Signal badges === */
.sig-strong-buy { background:#dcfce7; color:#14532d; border:1px solid #86efac;
                  padding:4px 11px; border-radius:20px; font-size:0.82em;
                  font-weight:700; white-space:nowrap; font-family:'IBM Plex Mono',monospace; }
.sig-buy        { background:#f0fdf4; color:#15803d; border:1px solid #bbf7d0;
                  padding:4px 11px; border-radius:20px; font-size:0.82em;
                  font-weight:700; white-space:nowrap; font-family:'IBM Plex Mono',monospace; }
.sig-hold       { background:#f1f5f9; color:#475569; border:1px solid #cbd5e1;
                  padding:4px 11px; border-radius:20px; font-size:0.82em;
                  font-weight:700; white-space:nowrap; font-family:'IBM Plex Mono',monospace; }
.sig-sell-soon  { background:#fff7ed; color:#c2410c; border:1px solid #fdba74;
                  padding:4px 11px; border-radius:20px; font-size:0.82em;
                  font-weight:700; white-space:nowrap; font-family:'IBM Plex Mono',monospace; }
.sig-sell       { background:#fee2e2; color:#991b1b; border:1px solid #fca5a5;
                  padding:4px 11px; border-radius:20px; font-size:0.82em;
                  font-weight:700; white-space:nowrap; font-family:'IBM Plex Mono',monospace; }

/* === Status badges === */
.stat-accel { background:#dbeafe; color:#1e40af; border:1px solid #93c5fd;
              padding:3px 10px; border-radius:4px; font-size:0.80em; font-weight:700; }
.stat-stable{ background:#f1f5f9; color:#334155; border:1px solid #cbd5e1;
              padding:3px 10px; border-radius:4px; font-size:0.80em; font-weight:700; }
.stat-decel { background:#fef3c7; color:#92400e; border:1px solid #fcd34d;
              padding:3px 10px; border-radius:4px; font-size:0.80em; font-weight:700; }

/* === Trajectory badges === */
.traj-v    { background:#ede9fe; color:#5b21b6; border:1px solid #c4b5fd;
             padding:3px 10px; border-radius:4px; font-size:0.80em; font-weight:700; }
.traj-fast { background:#ecfdf5; color:#065f46; border:1px solid #6ee7b7;
             padding:3px 10px; border-radius:4px; font-size:0.80em; font-weight:700; }
.traj-slow { background:#f0f9ff; color:#0c4a6e; border:1px solid #7dd3fc;
             padding:3px 10px; border-radius:4px; font-size:0.80em; font-weight:700; }
.traj-curv { background:#fdf4ff; color:#701a75; border:1px solid #e879f9;
             padding:3px 10px; border-radius:4px; font-size:0.80em; font-weight:700; }

/* === Sortable-table sort indicators === */
thead th.sorted-asc::after  { content: " \u25b2"; font-size:0.75em; }
thead th.sorted-desc::after { content: " \u25bc"; font-size:0.75em; }

/* === Row striping for large table === */
.all-assets-table tbody tr:nth-child(even) { background:#fafbfc; }

/* ── Mobile responsive ── */
@media (max-width: 768px) {
    .all-assets-table { font-size: 0.78em; }
    .all-assets-table thead th { padding: 6px 6px; font-size: 0.72em; }
    .all-assets-table tbody td { padding: 5px 6px; }
    .sig-strong-buy, .sig-buy, .sig-hold, .sig-sell-soon, .sig-sell { font-size: 0.72em; padding: 3px 8px; }
    .stat-accel, .stat-stable, .stat-decel { font-size: 0.72em; padding: 2px 7px; }
    .traj-v, .traj-fast, .traj-slow, .traj-curv { font-size: 0.72em; padding: 2px 7px; }
}
"""

# Column-sort JavaScript (click any <th> to sort its table ascending/descending)
SORT_JS = """
(function() {
    function sortTable(th) {
        var table = th.closest('table');
        var tbody = table.querySelector('tbody');
        var idx   = Array.prototype.indexOf.call(th.parentNode.children, th);
        var asc   = th.classList.contains('sorted-desc') || !th.classList.contains('sorted-asc');

        // Reset all headers in this table
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
    });
})();
"""


def _fmt_ret(val):
    """Format a return percentage with colour class."""
    if val is None:
        return '<td class="num muted" data-val="0">n/a</td>'
    cls  = "pos" if val >= 0 else "neg"
    sign = "+" if val >= 0 else ""
    return '<td class="num {c}" data-val="{v:.4f}">{s}{v:.1f}%</td>'.format(
        c=cls, v=val, s=sign)


def _fmt_sortino(val):
    """Format a Sortino ratio cell."""
    if val is None:
        return '<td class="num muted" data-val="-9999">n/a</td>'
    if val >= 999:
        return '<td class="num pos" data-val="999">&#8734;</td>'
    cls = "pos" if val >= 1 else ("warn" if val >= 0 else "neg")
    return '<td class="num {c}" data-val="{v:.4f}">{v:.2f}</td>'.format(c=cls, v=val)


def _fmt_price(val):
    return '<td class="num" data-val="{v:.4f}">${v:.2f}</td>'.format(v=val)


def _fmt_conf(val):
    cls = "pos" if val >= 80 else ("warn" if val >= 60 else "neg")
    return '<td class="num {c}" data-val="{v}">{v}%</td>'.format(c=cls, v=val)


TABLE_HEADERS = [
    "Symbol", "Price", "1M%", "3M%", "12M%",
    "Sortino 1M", "Status", "Trajectory", "Signal", "Confidence",
]


def _build_table_row(r):
    """Build a <tr> for one result dict."""
    sig_cls  = _SIGNAL_CLASS.get(r["signal"], "sig-hold")
    stat_cls = _STATUS_CLASS.get(r["status"], "stat-stable")
    traj_cls = _TRAJ_CLASS.get(r["trajectory"], "traj-curv")

    cells = []
    cells.append('<td class="ticker" data-val="{s}">{s}</td>'.format(s=r["symbol"]))
    cells.append(_fmt_price(r["current_price"]))
    cells.append(_fmt_ret(r.get("return_1m")))
    cells.append(_fmt_ret(r.get("return_3m")))
    cells.append(_fmt_ret(r.get("return_12m")))
    cells.append(_fmt_sortino(r.get("sortino_1m")))
    cells.append('<td data-val="{s}"><span class="{c}">{s}</span></td>'.format(
        s=r["status"], c=stat_cls))
    cells.append('<td data-val="{t}"><span class="{c}">{t}</span></td>'.format(
        t=r["trajectory"], c=traj_cls))
    cells.append('<td data-val="{s}"><span class="{c}">{s}</span></td>'.format(
        s=r["signal"], c=sig_cls))
    cells.append(_fmt_conf(r["confidence"]))
    return "<tr>{}</tr>".format("".join(cells))


def _build_table(rows, table_id=""):
    """Wrap rows in a full <table> with sortable headers."""
    id_attr = ' id="{}"'.format(table_id) if table_id else ""
    header_cells = "".join(
        "<th>{}</th>".format(h) for h in TABLE_HEADERS)
    body_rows = "\n".join(_build_table_row(r) for r in rows)
    return (
        '<table{id}>'
        '<thead><tr>{hdr}</tr></thead>'
        '<tbody>{body}</tbody>'
        '</table>'
    ).format(id=id_attr, hdr=header_cells, body=body_rows)


def _empty_section_msg(msg):
    return '<p class="muted" style="padding:10px 0;">{}</p>'.format(msg)

# =============================================================================
# REGIME LOGIC
# =============================================================================

def derive_regime(results):
    """
    Summarise the overall market momentum regime and pick a banner colour.
    Returns (label_str, score_html, color_hex).
    """
    total  = len(results)
    if total == 0:
        return "NO DATA", "No assets analysed.", "#888888"

    n_accel = sum(1 for r in results if r["status"] == "Accelerating")
    n_decel = sum(1 for r in results if r["status"] == "Decelerating")
    n_stable= sum(1 for r in results if r["status"] == "Stable")
    n_sbuy  = sum(1 for r in results if r["signal"] == "STRONG_BUY")
    n_buy   = sum(1 for r in results if r["signal"] == "BUY")
    n_sell  = sum(1 for r in results if r["signal"] in ("SELL", "SELL_SOON"))

    pct_accel = n_accel / total * 100
    pct_decel = n_decel / total * 100

    if pct_accel >= 40:
        label = "BROAD ACCELERATION"
        color = "#22c55e"
    elif pct_accel >= 25:
        label = "MODERATE MOMENTUM"
        color = "#84cc16"
    elif pct_decel >= 40:
        label = "BROAD DECELERATION"
        color = "#ef4444"
    elif pct_decel >= 25:
        label = "SOFTENING MOMENTUM"
        color = "#f59e0b"
    else:
        label = "MIXED / CONSOLIDATING"
        color = "#6366f1"

    score_html = (
        "Accelerating: {na} ({pa:.0f}%) &nbsp;|&nbsp; "
        "Stable: {ns} &nbsp;|&nbsp; "
        "Decelerating: {nd} ({pd:.0f}%) &nbsp;|&nbsp; "
        "Strong Buy: {sb} &nbsp;|&nbsp; "
        "Buy: {b} &nbsp;|&nbsp; "
        "Sell/Sell Soon: {se} &nbsp;|&nbsp; "
        "Total analysed: {tot}"
    ).format(
        na=n_accel, pa=pct_accel,
        ns=n_stable,
        nd=n_decel, pd=pct_decel,
        sb=n_sbuy, b=n_buy, se=n_sell,
        tot=total,
    )
    return label, score_html, color

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("ADVANCED MOMENTUM BACKEND v1.0")
    print("=" * 70)

    # 1. Universe
    symbols = load_universe()
    print("[LOAD] Found {} assets in price_cache".format(len(symbols)))

    # 2. Cache check
    results = load_cached_results(len(symbols))
    if results is None:
        results = process_all_assets(symbols)
        if not results:
            print("[ERROR] No results produced - aborting.")
            return
        save_cache(results)

    print("[OK] {} results ready for rendering".format(len(results)))

    # 3. Aggregate counts
    n_accel  = sum(1 for r in results if r["status"] == "Accelerating")
    n_stable = sum(1 for r in results if r["status"] == "Stable")
    n_decel  = sum(1 for r in results if r["status"] == "Decelerating")
    total    = len(results)

    # Signal buckets
    strong_buy = sorted(
        [r for r in results if r["signal"] == "STRONG_BUY"],
        key=lambda r: r["confidence"], reverse=True)
    buy = sorted(
        [r for r in results if r["signal"] == "BUY"],
        key=lambda r: r["confidence"], reverse=True)
    sell_soon = sorted(
        [r for r in results if r["signal"] == "SELL_SOON"],
        key=lambda r: r["confidence"], reverse=True)
    all_sorted = sorted(results, key=lambda r: r["confidence"], reverse=True)

    # 4. Regime
    regime_label, regime_score_html, regime_color = derive_regime(results)

    # 5. Build HTML
    writer = DashboardWriter("advanced-momentum", "Advanced Momentum Analyzer")
    parts  = []

    # Stat bar
    parts.append(writer.stat_bar([
        ("Total Assets",    str(total),    "neutral"),
        ("Accelerating",    str(n_accel),  "pos"),
        ("Stable",          str(n_stable), "neutral"),
        ("Decelerating",    str(n_decel),  "neg"),
        ("Strong Buy",      str(len(strong_buy)), "pos"),
        ("Sell / Sell Soon",str(sum(1 for r in results if r["signal"] in ("SELL","SELL_SOON"))), "neg"),
    ]))

    # Header
    subtitle = "OBV + Sortino + Slope Acceleration | {} assets | {}".format(
        total, datetime.date.today().strftime("%Y-%m-%d"))
    parts.append(writer.build_header(subtitle))

    # Regime banner
    parts.append(writer.regime_banner(regime_label, regime_score_html, color=regime_color))

    # Section 1: Strong Buy
    if strong_buy:
        tbl = _build_table(strong_buy, table_id="tbl-strong-buy")
        content = tbl
    else:
        content = _empty_section_msg("No assets currently meet STRONG_BUY criteria.")
    parts.append(writer.section(
        "Strong Buy Signals ({})".format(len(strong_buy)),
        content,
        hint="Click column header to sort"))

    # Section 2: Buy
    if buy:
        tbl = _build_table(buy, table_id="tbl-buy")
        content = tbl
    else:
        content = _empty_section_msg("No assets currently meet BUY criteria.")
    parts.append(writer.section(
        "Buy Signals ({})".format(len(buy)),
        content,
        hint="Click column header to sort"))

    # Section 3: Sell Soon Warnings
    if sell_soon:
        tbl = _build_table(sell_soon, table_id="tbl-sell-soon")
        content = tbl
    else:
        content = _empty_section_msg("No assets currently flagged SELL_SOON.")
    parts.append(writer.section(
        "Sell Soon Warnings ({})".format(len(sell_soon)),
        content,
        hint="Click column header to sort"))

    # Section 4: All Assets (full sortable table)
    all_tbl = _build_table(all_sorted, table_id="tbl-all-assets")
    parts.append(writer.section(
        "All Assets ({})".format(total),
        '<div style="overflow-x:auto;">{}</div>'.format(all_tbl),
        hint="Click column header to sort"))

    # Footer
    parts.append(writer.footer())

    # 6. Write HTML
    writer.write("\n".join(parts), extra_css=EXTRA_CSS, extra_js=SORT_JS)

    # 7. Write CSV
    csv_path = os.path.join(_SCRIPT_DIR, 'advanced_momentum_data.csv')
    csv_df = pd.DataFrame(results)
    csv_df.to_csv(csv_path, index=False, encoding='utf-8')
    print("[CSV] {}".format(csv_path))

    print("[DONE] Advanced Momentum dashboard written.")


if __name__ == "__main__":
    main()
