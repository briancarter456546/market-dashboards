# -*- coding: utf-8 -*-
# =============================================================================
# slope_stage_backend.py - v1.1
# Last updated: 2026-03-03
# =============================================================================
# v1.1: Ownership checkboxes, sortable dates, Trend Quality Score, meta integration
# v1.0: Initial dashboard backend ported from slope_trendline_classifier_v1.6.1
#
# Classifies all price_cache tickers into 4 market stages using 90-day linear
# regression slope. Surfaces entry signals (Stage 1->2 transitions), sustained
# uptrends (Stage 2), and parabolic exit watch (Stage 3).
#
# Computation logic ported from perplexity-user-data/slope_trendline_classifier_v1.6.1.py
# Path setup and HTML output for market-dashboards/ layout.
#
# Run:  python slope_stage_backend.py
# =============================================================================

import os
import json
import pickle
import datetime
import time
import threading

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

CONFIG = {
    "cache_file":    os.path.join(_DATA_DIR, "slope_stage_data.json"),
    "max_workers":   8,
    "min_year":      1995,
    "min_history":   252,
    "trendline_window": 90,
    "atr_period":    14,
    "transition_lookback_days": 5,
}

# =============================================================================
# STAGE DEFINITIONS
# =============================================================================

STAGE_THRESHOLDS = np.array([-30.0, 10.0, 80.0])

STAGE_NAMES = {
    0: "Deep Decline",
    1: "Basing",
    2: "Sustained Uptrend",
    3: "Parabolic",
}

STAGE_COLORS = {
    0: "#dc2626",   # red
    1: "#d97706",   # amber
    2: "#16a34a",   # green
    3: "#f97316",   # orange
}

STAGE_CSS_CLASSES = {
    0: "neg",
    1: "warn",
    2: "pos",
    3: "warn",
}

# =============================================================================
# PRE-COMPUTED DESIGN MATRIX (from classifier v1.6.1)
# =============================================================================

TRENDLINE_WINDOW = CONFIG["trendline_window"]
X_REGRESSION = np.arange(TRENDLINE_WINDOW, dtype=np.float64)
X_MEAN = X_REGRESSION.mean()
X_CENTERED = X_REGRESSION - X_MEAN
X_VAR = np.sum(X_CENTERED ** 2)

# =============================================================================
# JSON HELPERS
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
# SAME-DAY CACHE
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
    """Return cached result list if fresh and asset-count matches."""
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
# COMPUTATION (ported from slope_trendline_classifier_v1.6.1)
# =============================================================================

def classify_stages_vectorized(slopes_annual):
    """Vectorized stage classification: 0/1/2/3."""
    stages = np.zeros(len(slopes_annual), dtype=np.int32)
    stages[slopes_annual < -30.0] = 0
    stages[(slopes_annual >= -30.0) & (slopes_annual < 10.0)] = 1
    stages[(slopes_annual >= 10.0) & (slopes_annual < 80.0)] = 2
    stages[slopes_annual >= 80.0] = 3
    return stages


def detect_crash_vectorized(prices, trendlines, window=20):
    """Vectorized crash detection."""
    n = len(prices)
    crash_scores = np.zeros(n)

    for i in range(window, n):
        recent = prices[i-window:i]
        trend_val = trendlines[i]
        below_mask = recent < trend_val
        below_count = np.sum(below_mask)
        pct_below = (below_count / window) * 100.0

        if below_count > 0:
            avg_dist = np.mean((recent[below_mask] - trend_val) / trend_val) * 100.0
        else:
            avg_dist = 0.0

        crash_scores[i] = min(100.0, pct_below + abs(avg_dist) * 2.0)

    return crash_scores


def compute_all_trendlines_vectorized(prices):
    """Compute trendlines for ALL windows at once using sliding window view."""
    windows = np.lib.stride_tricks.sliding_window_view(prices, TRENDLINE_WINDOW)

    y_means = windows.mean(axis=1, keepdims=True)
    slopes = (windows - y_means) @ X_CENTERED / X_VAR
    intercepts = y_means.ravel() - slopes * X_MEAN

    fitted = slopes[:, np.newaxis] * X_REGRESSION + intercepts[:, np.newaxis]
    residuals = windows - fitted

    ss_res = np.sum(residuals ** 2, axis=1)
    ss_tot = np.sum((windows - y_means) ** 2, axis=1)
    r_squared = np.where(ss_tot > 0, 1.0 - (ss_res / ss_tot), 0.0)

    volatility_raw = np.std(residuals, axis=1)

    trendline_values = np.ascontiguousarray(fitted[:, -1])
    current_prices = np.ascontiguousarray(windows[:, -1])

    first_prices = windows[:, 0]
    safe_first = np.where(first_prices != 0, first_prices, np.nan)
    daily_slope_pct = (slopes / safe_first) * 100
    annual_slope_pct = daily_slope_pct * 252

    mean_prices = windows.mean(axis=1)
    safe_mean = np.where(mean_prices != 0, mean_prices, np.nan)
    volatility_pct = (volatility_raw / safe_mean) * 100

    safe_trend = np.where(trendline_values != 0, trendline_values, np.nan)
    distance_from_trend = ((current_prices - trendline_values) / safe_trend) * 100

    return {
        'trendline_values': trendline_values,
        'current_prices': current_prices,
        'annual_slope_pct': annual_slope_pct,
        'volatility_pct': volatility_pct,
        'distance_from_trend_pct': distance_from_trend,
        'r_squared': r_squared,
    }


def compute_trend_quality_score(stage, r_squared, crash_risk, distance_pct):
    """Compute 0-100 Trend Quality Score from four components.

    Components:
      Stage position  30% - Stage 2 best, Stage 3 moderate, Stage 1 low, Stage 0 zero
      R-squared       30% - Higher = cleaner trend = better signal
      Crash risk      25% - Inverted: low crash = high score
      Distance        15% - Penalize extremes: 0-5% above trend is ideal
    """
    # Stage base: 0->0, 1->25, 2->100, 3->65 (parabolic is riskier)
    stage_map = {0: 0.0, 1: 25.0, 2: 100.0, 3: 65.0}
    stage_score = stage_map.get(stage, 0.0)

    # R-squared: 0-1 scaled to 0-100
    r2_score = max(0.0, min(100.0, r_squared * 100.0))

    # Crash risk: invert (0 crash = 100 score, 100 crash = 0 score)
    crash_score = max(0.0, 100.0 - crash_risk)

    # Distance from trend: ideal is 0-5% above, penalize extremes
    if distance_pct is None or np.isnan(distance_pct):
        dist_score = 50.0
    elif 0 <= distance_pct <= 5:
        dist_score = 100.0
    elif 5 < distance_pct <= 20:
        dist_score = max(0.0, 100.0 - (distance_pct - 5) * (100.0 / 15.0))
    elif distance_pct > 20:
        dist_score = 0.0
    elif -5 <= distance_pct < 0:
        dist_score = max(0.0, 60.0 + distance_pct * 12.0)
    else:
        dist_score = 0.0

    composite = (
        stage_score * 0.30 +
        r2_score * 0.30 +
        crash_score * 0.25 +
        dist_score * 0.15
    )
    return round(max(0.0, min(100.0, composite)), 1)


def process_single_asset(ticker, df):
    """Process one ticker, return summary dict or None."""
    try:
        df = df.sort_index()
        df = df[df.index.year >= CONFIG["min_year"]]

        if len(df) < CONFIG["min_history"]:
            return None

        prices = df['close'].values.astype(np.float64)

        if len(prices) < TRENDLINE_WINDOW + 1:
            return None

        trend_data = compute_all_trendlines_vectorized(prices)
        stages = classify_stages_vectorized(trend_data['annual_slope_pct'])

        # Crash risk on the padded segment
        prices_padded = np.ascontiguousarray(prices[TRENDLINE_WINDOW-1:])
        trendlines_padded = trend_data['trendline_values']
        crash_scores = detect_crash_vectorized(prices_padded, trendlines_padded, window=20)

        # Current values (latest point)
        current_stage = int(stages[-1])
        current_slope = float(trend_data['annual_slope_pct'][-1])
        current_distance = float(trend_data['distance_from_trend_pct'][-1])
        current_r2 = float(trend_data['r_squared'][-1])
        current_vol = float(trend_data['volatility_pct'][-1])
        current_crash = float(crash_scores[-1])
        current_price = float(trend_data['current_prices'][-1])

        # Stage transition detection (last N+1 days to find transitions in last N days)
        lookback = CONFIG["transition_lookback_days"]
        transitions = []
        n_stages = len(stages)
        check_range = min(lookback + 1, n_stages)

        dates_padded = df.index[TRENDLINE_WINDOW-1:]

        for i in range(n_stages - check_range, n_stages - 1):
            if i < 0:
                continue
            if stages[i] == 1 and stages[i+1] == 2:
                trans_date = str(dates_padded[i+1].date())
                transitions.append({
                    "date": trans_date,
                    "from_stage": 1,
                    "to_stage": 2,
                })

        # Days in current stage
        days_in_stage = 1
        for i in range(n_stages - 2, -1, -1):
            if stages[i] == current_stage:
                days_in_stage += 1
            else:
                break

        # Trend Quality Score (0-100 composite)
        tq_score = compute_trend_quality_score(
            current_stage, current_r2, current_crash, current_distance)

        return {
            "ticker": ticker,
            "stage": current_stage,
            "stage_name": STAGE_NAMES[current_stage],
            "slope_pct": round(current_slope, 2),
            "distance_pct": round(current_distance, 2),
            "r_squared": round(current_r2, 3),
            "volatility_pct": round(current_vol, 2),
            "crash_risk": round(current_crash, 1),
            "price": round(current_price, 2),
            "days_in_stage": days_in_stage,
            "tq_score": tq_score,
            "transitions_1_to_2": transitions,
        }

    except Exception:
        return None


def _load_and_process(pkl_path):
    """Worker function: load pkl and process."""
    ticker = pkl_path.stem
    if ticker == 'aggregated_price_data':
        return None

    try:
        df = pd.read_pickle(str(pkl_path))
        if 'close' not in df.columns or 'high' not in df.columns or 'low' not in df.columns:
            return None
        return process_single_asset(ticker, df)
    except Exception:
        return None


def run_computation():
    """Run the full computation pipeline. Returns list of result dicts."""
    pkl_files = [f for f in Path(CACHE_DIR).glob('*.pkl')
                 if f.stem != 'aggregated_price_data']
    total = len(pkl_files)
    print("[LOAD] Found {} price cache files".format(total))

    # Check same-day cache
    cached = load_cached_results(total)
    if cached is not None:
        return cached

    # Process all assets in parallel
    print("[COMPUTE] Processing {} assets with {} workers...".format(
        total, CONFIG["max_workers"]))
    start_time = time.time()

    results = []
    with ThreadPoolExecutor(max_workers=CONFIG["max_workers"]) as executor:
        futures = {executor.submit(_load_and_process, f): f for f in pkl_files}
        done_count = 0
        for future in as_completed(futures):
            done_count += 1
            if done_count % 200 == 0:
                print("  ... processed {}/{}".format(done_count, total))
            result = future.result()
            if result is not None:
                results.append(result)

    elapsed = time.time() - start_time
    print("[OK] Processed {} assets in {:.1f}s ({} valid results)".format(
        total, elapsed, len(results)))

    # Save cache
    save_cache(results)

    return results

# =============================================================================
# HTML BUILDING
# =============================================================================

EXTRA_CSS = """
.stage-pill {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 0.8em;
    font-weight: 700;
    letter-spacing: 0.03em;
}
.stage-0 { background: #fee2e2; color: #b91c1c; border: 1px solid #fca5a5; }
.stage-1 { background: #fef3c7; color: #b45309; border: 1px solid #fde68a; }
.stage-2 { background: #dcfce7; color: #15803d; border: 1px solid #bbf7d0; }
.stage-3 { background: #ffedd5; color: #c2410c; border: 1px solid #fed7aa; }

.dist-bar {
    background: #e5e7eb;
    border-radius: 4px;
    height: 22px;
    display: flex;
    overflow: hidden;
    margin: 4px 0;
}
.dist-bar-seg {
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.75em;
    font-weight: 600;
    color: #fff;
    white-space: nowrap;
    padding: 0 4px;
}

.full-universe-toggle {
    cursor: pointer;
    color: #4f46e5;
    font-weight: 600;
    padding: 8px 0;
    user-select: none;
}
.full-universe-toggle:hover { text-decoration: underline; }
.full-universe-body { display: none; }
.full-universe-body.open { display: block; }

/* Mobile */
@media (max-width: 768px) {
    .stage-pill { font-size: 0.7em; padding: 2px 8px; }
    .dist-bar { height: 18px; }
}
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

        // Full universe toggle
        var toggle = document.getElementById('universe-toggle');
        var body = document.getElementById('universe-body');
        if (toggle && body) {
            toggle.addEventListener('click', function() {
                body.classList.toggle('open');
                toggle.textContent = body.classList.contains('open')
                    ? 'Hide Full Universe' : 'Show Full Universe (all tickers)';
            });
        }
    });
})();
"""


def _own_cell(ticker):
    """Ownership checkbox cell with data-val for sorting (0=unowned, 1=owned)."""
    return (
        '<td data-val="0"><input type="checkbox" class="own-cb" data-ticker="{t}"'
        ' onclick="window._ownToggle(\'{t}\', this)" title="Mark as owned"></td>'
    ).format(t=ticker)


def _watch_cell(ticker):
    """Watch checkbox cell with data-val for sorting (0=unwatched, 1=watched)."""
    return (
        '<td data-val="0"><input type="checkbox" class="watch-cb" data-ticker="{t}"'
        ' onclick="window._watchToggle(\'{t}\', this)" title="Mark as watched"></td>'
    ).format(t=ticker)


def _score_cell(val):
    """Trend Quality Score cell with color coding."""
    if val is None:
        return '<td class="num muted" data-val="0">n/a</td>'
    if val >= 70:
        css = "pos"
    elif val >= 40:
        css = "warn"
    else:
        css = "neg"
    return '<td class="num {c}" data-val="{v}">{v:.0f}</td>'.format(c=css, v=val)


def _stage_pill(stage):
    """Inline HTML pill for a stage number."""
    name = STAGE_NAMES.get(stage, "?")
    return '<span class="stage-pill stage-{s}">Stage {s}: {n}</span>'.format(
        s=stage, n=name)


def _num_cell(val, fmt="{:.2f}", suffix="", css=""):
    """Render a numeric table cell with data-val for sorting."""
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


def _crash_cell(val):
    """Crash risk cell with color coding."""
    if val is None:
        return '<td class="num muted" data-val="0">n/a</td>'
    if val >= 60:
        css = "neg"
    elif val >= 30:
        css = "warn"
    else:
        css = "pos"
    return '<td class="num {c}" data-val="{v}">{v:.0f}</td>'.format(
        c=css, v=val)


def _build_distribution_bar(stage_counts, total):
    """Build an inline distribution bar showing stage %."""
    colors = {0: "#dc2626", 1: "#d97706", 2: "#16a34a", 3: "#f97316"}
    parts = []
    for s in range(4):
        count = stage_counts.get(s, 0)
        pct = (count / total * 100) if total > 0 else 0
        if pct < 1:
            continue
        parts.append(
            '<div class="dist-bar-seg" style="width:{pct:.1f}%;background:{col};">'
            '{pct:.0f}%</div>'.format(pct=pct, col=colors[s])
        )
    return '<div class="dist-bar">{}</div>'.format("".join(parts))


def _build_table(headers, rows_html):
    """Build a sortable HTML table. Headers can be strings or (label, tooltip) tuples."""
    parts = []
    for h in headers:
        if isinstance(h, tuple):
            label, tip = h
            if tip:
                parts.append('<th title="{}">{}</th>'.format(tip, label))
            else:
                parts.append('<th>{}</th>'.format(label))
        else:
            parts.append('<th>{}</th>'.format(h))
    header_cells = "".join(parts)
    return (
        '<table>'
        '<thead><tr>{hdr}</tr></thead>'
        '<tbody>{rows}</tbody>'
        '</table>'.format(hdr=header_cells, rows="".join(rows_html))
    )


def build_body_html(results, writer):
    """Build the full body HTML for DashboardWriter.write()."""
    date_str = datetime.date.today().strftime("%Y-%m-%d")

    # Aggregate stats
    total = len(results)
    stage_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for r in results:
        stage_counts[r["stage"]] = stage_counts.get(r["stage"], 0) + 1

    # Collect 1->2 transitions
    entry_signals = []
    for r in results:
        for t in r.get("transitions_1_to_2", []):
            entry_signals.append({
                "ticker": r["ticker"],
                "date": t["date"],
                "slope_pct": r["slope_pct"],
                "distance_pct": r["distance_pct"],
                "r_squared": r["r_squared"],
                "crash_risk": r["crash_risk"],
                "price": r["price"],
                "tq_score": r.get("tq_score"),
            })
    entry_signals.sort(key=lambda x: x["date"], reverse=True)

    # Stage 2 and Stage 3 lists
    stage2 = sorted(
        [r for r in results if r["stage"] == 2],
        key=lambda x: x["slope_pct"], reverse=True)
    stage3 = sorted(
        [r for r in results if r["stage"] == 3],
        key=lambda x: x["slope_pct"], reverse=True)

    parts = []

    # -----------------------------------------------------------------
    # 1. Stat bar
    # -----------------------------------------------------------------
    parts.append(writer.stat_bar([
        ("Date",             date_str,                             "neutral"),
        ("Universe",         "{:,}".format(total),                 "neutral"),
        ("Stage 2 (Uptrend)", "{:,}".format(stage_counts[2]),      "pos"),
        ("1->2 Signals",     "{}".format(len(entry_signals)),      "pos" if entry_signals else "neutral"),
        ("Stage 3 (Watch)",  "{}".format(stage_counts[3]),         "warn" if stage_counts[3] > 0 else "neutral"),
    ]))

    # -----------------------------------------------------------------
    # 2. Header
    # -----------------------------------------------------------------
    parts.append(writer.build_header(
        "90-day Trendline &nbsp;|&nbsp; Stages 0-3"
    ))

    # -----------------------------------------------------------------
    # LLM description block
    # -----------------------------------------------------------------
    parts.append(writer.llm_block())

    # -----------------------------------------------------------------
    # 3. Regime banner - stage distribution
    # -----------------------------------------------------------------
    dist_parts = []
    for s in range(4):
        count = stage_counts[s]
        pct = (count / total * 100) if total > 0 else 0
        dist_parts.append("Stage {}: {} ({:.0f}%)".format(s, count, pct))
    dist_text = " &nbsp;&bull;&nbsp; ".join(dist_parts)

    # Overall market lean
    uptrend_pct = ((stage_counts[2] + stage_counts[3]) / total * 100) if total > 0 else 0
    if uptrend_pct >= 60:
        banner_label = "BROAD UPTREND"
        banner_color = "#16a34a"
    elif uptrend_pct >= 40:
        banner_label = "MIXED"
        banner_color = "#d97706"
    elif uptrend_pct >= 20:
        banner_label = "LEANING DOWN"
        banner_color = "#f97316"
    else:
        banner_label = "BROAD DECLINE"
        banner_color = "#dc2626"

    parts.append(writer.regime_banner(banner_label, dist_text, color=banner_color))

    # Distribution bar visual
    bar_html = _build_distribution_bar(stage_counts, total)
    stage_legend = (
        '<div style="display:flex;gap:16px;flex-wrap:wrap;margin-top:8px;font-size:0.85em;">'
        '<span>{} Stage 0: Deep Decline (&lt;-30%)</span>'
        '<span>{} Stage 1: Basing (-30% to +10%)</span>'
        '<span>{} Stage 2: Sustained Uptrend (+10% to +80%)</span>'
        '<span>{} Stage 3: Parabolic (&gt;+80%)</span>'
        '</div>'
    ).format(
        '<span style="color:#dc2626;">&#9632;</span>',
        '<span style="color:#d97706;">&#9632;</span>',
        '<span style="color:#16a34a;">&#9632;</span>',
        '<span style="color:#f97316;">&#9632;</span>',
    )
    parts.append(writer.section(
        "Stage Distribution",
        bar_html + stage_legend
    ))

    # -----------------------------------------------------------------
    # 4. Entry Signals (Stage 1 -> 2)
    # -----------------------------------------------------------------
    if entry_signals:
        headers = [
            "Own", "Watch", "Ticker",
            ("Transition Date", "Date the stock transitioned from Stage 1 (Basing) to Stage 2 (Sustained Uptrend)"),
            ("Price", "Latest closing price"),
            ("Slope %", "Annualized slope from 21-day log-price regression. Higher = steeper uptrend."),
            ("Distance %", "% distance from SMA29 trendline. Higher = more extended above the trend."),
            ("R-sq", "R-squared of 21-day log-price regression. 1.0 = perfectly linear, 0.0 = random walk."),
            ("Crash Risk", "Composite crash probability from RMT eigenvalue + Ising magnetization model."),
            ("TQ Score", "Trend Quality score 0-100. Blend of slope strength, R-squared, and stage persistence."),
        ]
        rows = []
        for s in entry_signals:
            date_sortval = s["date"].replace("-", "")
            rows.append(
                '<tr>'
                '{}'
                '{}'
                '<td><strong>{}</strong></td>'
                '<td data-val="{}">{}</td>'
                '{}{}{}{}{}{}'
                '</tr>'.format(
                    _own_cell(s["ticker"]),
                    _watch_cell(s["ticker"]),
                    s["ticker"],
                    date_sortval, s["date"],
                    _num_cell(s["price"], fmt="${:.2f}", css="neutral"),
                    _num_cell(s["slope_pct"], fmt="{:.1f}", suffix="%"),
                    _num_cell(s["distance_pct"], fmt="{:.1f}", suffix="%"),
                    _num_cell(s["r_squared"], fmt="{:.3f}", css="neutral"),
                    _crash_cell(s["crash_risk"]),
                    _score_cell(s.get("tq_score")),
                )
            )
        table = _build_table(headers, rows)
        parts.append(writer.section(
            "Entry Signals (Stage 1 -> 2)",
            '<p style="margin-bottom:12px;color:#555;font-size:0.9em;">'
            'Tickers that transitioned from Basing to Sustained Uptrend in the last {} trading days. '
            'These are early trend entries.</p>{}'.format(
                CONFIG["transition_lookback_days"], table),
            hint="Click column to sort"
        ))
    else:
        parts.append(writer.section(
            "Entry Signals (Stage 1 -> 2)",
            '<p style="color:#888;">No Stage 1 -> 2 transitions detected in the last {} trading days.</p>'.format(
                CONFIG["transition_lookback_days"])
        ))

    # -----------------------------------------------------------------
    # 5. Stage 2 - Sustained Uptrends
    # -----------------------------------------------------------------
    headers = [
        "Own", "Watch", "Ticker",
        ("Stage", "Current slope stage: 0=Decline, 1=Basing, 2=Sustained Uptrend, 3=Late Stage/Parabolic"),
        ("Slope %", "Annualized slope from 21-day log-price regression. Higher = steeper uptrend."),
        ("Days in Stage", "Number of trading days the stock has been in its current stage."),
        ("Distance %", "% distance from SMA29 trendline. Higher = more extended above the trend."),
        ("R-sq", "R-squared of 21-day log-price regression. 1.0 = perfectly linear, 0.0 = random walk."),
        ("Vol %", "Annualized realized volatility (21-day). Lower = smoother trend."),
        ("Crash Risk", "Composite crash probability from RMT eigenvalue + Ising magnetization model."),
        ("TQ Score", "Trend Quality score 0-100. Blend of slope strength, R-squared, and stage persistence."),
    ]
    rows = []
    for r in stage2:
        rows.append(
            '<tr>'
            '{}'
            '{}'
            '<td><strong>{}</strong></td>'
            '<td>{}</td>'
            '{}{}{}{}{}{}{}'
            '</tr>'.format(
                _own_cell(r["ticker"]),
                _watch_cell(r["ticker"]),
                r["ticker"],
                _stage_pill(r["stage"]),
                _num_cell(r["slope_pct"], fmt="{:.1f}", suffix="%"),
                _num_cell(r["days_in_stage"], fmt="{:.0f}", css="neutral"),
                _num_cell(r["distance_pct"], fmt="{:.1f}", suffix="%"),
                _num_cell(r["r_squared"], fmt="{:.3f}", css="neutral"),
                _num_cell(r["volatility_pct"], fmt="{:.1f}", suffix="%", css="neutral"),
                _crash_cell(r["crash_risk"]),
                _score_cell(r.get("tq_score")),
            )
        )
    table = _build_table(headers, rows)
    parts.append(writer.section(
        "Stage 2 - Sustained Uptrends ({})".format(len(stage2)),
        '<p style="margin-bottom:12px;color:#555;font-size:0.9em;">'
        'All tickers currently in Stage 2 (slope +10% to +80% annualized). '
        'This is the primary opportunity zone.</p>{}'.format(table),
        hint="Click column to sort"
    ))

    # -----------------------------------------------------------------
    # 6. Stage 3 - Parabolic (Exit Watch)
    # -----------------------------------------------------------------
    headers = [
        "Own", "Watch", "Ticker",
        ("Stage", "Current slope stage. Stage 3 = Late Stage/Parabolic (slope >80% annualized). Exit watch zone."),
        ("Slope %", "Annualized slope from 21-day log-price regression. >80% = parabolic territory."),
        ("Days in Stage", "Number of trading days in current stage. Longer parabolic runs = higher crash risk."),
        ("Distance %", "% distance from SMA29 trendline. Very high = extremely extended."),
        ("R-sq", "R-squared of 21-day log-price regression. High R-sq + high slope = clean parabolic move."),
        ("Vol %", "Annualized realized volatility (21-day)."),
        ("Crash Risk", "Composite crash probability from RMT eigenvalue + Ising magnetization model."),
        ("TQ Score", "Trend Quality score 0-100."),
    ]
    rows = []
    for r in stage3:
        rows.append(
            '<tr>'
            '{}'
            '{}'
            '<td><strong>{}</strong></td>'
            '<td>{}</td>'
            '{}{}{}{}{}{}{}'
            '</tr>'.format(
                _own_cell(r["ticker"]),
                _watch_cell(r["ticker"]),
                r["ticker"],
                _stage_pill(r["stage"]),
                _num_cell(r["slope_pct"], fmt="{:.1f}", suffix="%"),
                _num_cell(r["days_in_stage"], fmt="{:.0f}", css="neutral"),
                _num_cell(r["distance_pct"], fmt="{:.1f}", suffix="%"),
                _num_cell(r["r_squared"], fmt="{:.3f}", css="neutral"),
                _num_cell(r["volatility_pct"], fmt="{:.1f}", suffix="%", css="neutral"),
                _crash_cell(r["crash_risk"]),
                _score_cell(r.get("tq_score")),
            )
        )
    table = _build_table(headers, rows)
    parts.append(writer.section(
        "Stage 3 - Parabolic / Exit Watch ({})".format(len(stage3)),
        '<p style="margin-bottom:12px;color:#555;font-size:0.9em;">'
        'Tickers with slope &gt;+80% annualized. These are running hot -- '
        'consider tightening stops or taking profits.</p>{}'.format(table),
        hint="Click column to sort"
    ))

    # -----------------------------------------------------------------
    # 7. Full Universe (collapsible)
    # -----------------------------------------------------------------
    all_sorted = sorted(results, key=lambda x: (-x["stage"], -x["slope_pct"]))

    headers = [
        "Own", "Watch", "Ticker",
        ("Stage", "Slope stage: 0=Decline, 1=Basing, 2=Sustained Uptrend, 3=Late Stage/Parabolic"),
        ("Slope %", "Annualized slope from 21-day log-price regression."),
        ("Days in Stage", "Trading days in current stage."),
        ("Distance %", "% distance from SMA29 trendline."),
        ("R-sq", "R-squared of 21-day log-price regression. 1.0 = linear, 0.0 = random."),
        ("Vol %", "Annualized realized volatility (21-day)."),
        ("Crash Risk", "Composite crash probability (RMT + Ising)."),
        ("TQ Score", "Trend Quality score 0-100."),
    ]
    rows = []
    for r in all_sorted:
        rows.append(
            '<tr>'
            '{}'
            '{}'
            '<td><strong>{}</strong></td>'
            '<td>{}</td>'
            '{}{}{}{}{}{}{}'
            '</tr>'.format(
                _own_cell(r["ticker"]),
                _watch_cell(r["ticker"]),
                r["ticker"],
                _stage_pill(r["stage"]),
                _num_cell(r["slope_pct"], fmt="{:.1f}", suffix="%"),
                _num_cell(r["days_in_stage"], fmt="{:.0f}", css="neutral"),
                _num_cell(r["distance_pct"], fmt="{:.1f}", suffix="%"),
                _num_cell(r["r_squared"], fmt="{:.3f}", css="neutral"),
                _num_cell(r["volatility_pct"], fmt="{:.1f}", suffix="%", css="neutral"),
                _crash_cell(r["crash_risk"]),
                _score_cell(r.get("tq_score")),
            )
        )
    table = _build_table(headers, rows)

    universe_html = (
        '<div id="universe-toggle" class="full-universe-toggle">'
        'Show Full Universe (all tickers)</div>'
        '<div id="universe-body" class="full-universe-body">{}</div>'.format(table)
    )
    parts.append(writer.section(
        "Full Universe ({:,} tickers)".format(total),
        universe_html,
        hint="Click column to sort"
    ))

    # -----------------------------------------------------------------
    # 8. Footer
    # -----------------------------------------------------------------
    parts.append(writer.footer())

    return "\n".join(parts)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("SLOPE STAGE SCANNER BACKEND v1.0")
    print("=" * 70)
    print("Generated: {}".format(datetime.datetime.now().isoformat()))

    results = run_computation()

    if not results:
        print("[ERROR] No results - aborting HTML write.")
        return

    # Summary
    stage_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for r in results:
        stage_counts[r["stage"]] = stage_counts.get(r["stage"], 0) + 1

    entry_signals = sum(1 for r in results if r.get("transitions_1_to_2"))

    print()
    print("=" * 70)
    print("SLOPE STAGE SUMMARY")
    print("=" * 70)
    print("  Universe:    {:,} assets".format(len(results)))
    for s in range(4):
        pct = stage_counts[s] / len(results) * 100
        print("  Stage {}:     {:,} ({:.1f}%) - {}".format(
            s, stage_counts[s], pct, STAGE_NAMES[s]))
    print("  1->2 Signals: {}".format(entry_signals))
    print("=" * 70)

    # Build and write dashboard HTML
    writer = DashboardWriter("slope-stage", "Slope Stage Scanner")
    body = build_body_html(results, writer)
    writer.write(body, extra_css=EXTRA_CSS, extra_js=SORT_JS)

    print("[OK] Dashboard written.")


if __name__ == "__main__":
    main()
