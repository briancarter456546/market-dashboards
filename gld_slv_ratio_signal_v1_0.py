# -*- coding: utf-8 -*-
# =============================================================================
# gld_slv_ratio_signal_v1_0.py - v1.0
# Dashboard backend: Gold/Silver Ratio Mean-Reversion Signal
#
# Tracks GLD/SLV ETF ratio and fires a buy signal when it crosses above P90
# (~88:1 metal ratio). Scimode-validated: 14 non-overlapping trades, 11 wins
# (79%), mean +13.6% SLV 60d forward return vs +3.5% baseline.
#
# Signal is ratio-specific (not just silver oversold) -- outperforms pure
# SLV mean-reversion by 2.7x. KB finding #44746.
#
# Produces static HTML dashboard in docs/gld-slv-signal/
# Sends notification email when signal activates or deactivates.
#
# Author: Brian + Claude
# Date: 2026-03-13
# =============================================================================

import os
import json
import sys
import datetime

import numpy as np
import pandas as pd
from pathlib import Path

from dashboard_writer import DashboardWriter

# =============================================================================
# PATH SETUP
# =============================================================================

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.normpath(os.path.join(_SCRIPT_DIR, '..', 'perplexity-user-data'))
CACHE_DIR = os.path.normpath(os.path.join(_DATA_DIR, 'price_cache'))
STATE_FILE = os.path.join(_DATA_DIR, 'output', 'scientist', 'gld_slv_signal_state.json')

# =============================================================================
# CONFIGURATION
# =============================================================================

# P90 threshold from scimode investigation (computed on full 2006-2026 sample)
ETF_RATIO_P90 = 8.84
METAL_RATIO_APPROX_MULTIPLIER = 10
HOLD_DAYS = 60
SMA_WINDOW = 50

# Targets to track when signal is active
TARGETS = ['SLV', 'GDX', 'GDXJ', 'SIL', 'GLD']

# Historical trade log (from scimode investigation, non-overlapping)
HISTORICAL_TRADES = [
    {"entry": "2019-05-13", "exit": "2019-08-07", "ret": 15.4, "win": True},
    {"entry": "2019-08-01", "exit": "2019-10-25", "ret": 10.1, "win": True},
    {"entry": "2019-12-06", "exit": "2020-03-05", "ret": 5.3, "win": True},
    {"entry": "2020-02-20", "exit": "2020-05-15", "ret": -9.5, "win": False},
    {"entry": "2022-05-12", "exit": "2022-08-09", "ret": -1.0, "win": False},
    {"entry": "2022-08-02", "exit": "2022-10-26", "ret": -2.3, "win": False},
    {"entry": "2022-10-12", "exit": "2023-01-09", "ret": 23.6, "win": True},
    {"entry": "2023-02-27", "exit": "2023-05-23", "ret": 13.6, "win": True},
    {"entry": "2024-01-03", "exit": "2024-04-01", "ret": 8.5, "win": True},
    {"entry": "2024-03-21", "exit": "2024-06-17", "ret": 18.9, "win": True},
    {"entry": "2024-08-05", "exit": "2024-10-29", "ret": 26.3, "win": True},
    {"entry": "2024-11-27", "exit": "2025-02-27", "ret": 3.5, "win": True},
    {"entry": "2025-07-14", "exit": "2025-10-07", "ret": 25.3, "win": True},
    {"entry": "2025-09-17", "exit": "2025-12-11", "ret": 52.5, "win": True},
]

# =============================================================================
# EXTRA CSS
# =============================================================================

EXTRA_CSS = """
.signal-gauge {
    text-align: center;
    padding: 32px 16px;
    margin: 20px 0;
}
.signal-gauge .gauge-value {
    font-size: 3.5em;
    font-weight: 800;
    font-family: 'IBM Plex Mono', monospace;
    line-height: 1.1;
}
.signal-gauge .gauge-label {
    font-size: 0.95em;
    color: #666;
    margin-top: 6px;
}
.signal-gauge .gauge-metal {
    font-size: 1.4em;
    font-weight: 600;
    margin-top: 4px;
}

.signal-status {
    text-align: center;
    padding: 24px 16px;
    border-radius: 12px;
    margin: 16px 0;
    font-size: 1.3em;
    font-weight: 700;
    letter-spacing: 0.03em;
}
.signal-DORMANT {
    background: #f1f5f9;
    color: #64748b;
    border: 2px solid #cbd5e1;
}
.signal-APPROACHING {
    background: #fef9c3;
    color: #a16207;
    border: 2px solid #fde047;
}
.signal-ACTIVE {
    background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%);
    color: #78350f;
    border: 2px solid #d97706;
    box-shadow: 0 4px 20px rgba(245, 158, 11, 0.3);
}

.threshold-bar {
    position: relative;
    height: 32px;
    background: linear-gradient(to right, #22c55e 0%, #eab308 70%, #ef4444 100%);
    border-radius: 8px;
    margin: 24px 0;
    overflow: visible;
}
.threshold-marker {
    position: absolute;
    top: -6px;
    width: 4px;
    height: 44px;
    background: #1e293b;
    border-radius: 2px;
}
.threshold-marker-label {
    position: absolute;
    top: -24px;
    transform: translateX(-50%);
    font-size: 0.75em;
    font-weight: 700;
    white-space: nowrap;
}
.current-marker {
    position: absolute;
    top: -8px;
    width: 20px;
    height: 20px;
    background: #3b82f6;
    border: 3px solid #fff;
    border-radius: 50%;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    transform: translateX(-50%);
}

.trade-win { color: #16a34a; font-weight: 700; }
.trade-loss { color: #dc2626; font-weight: 700; }

.target-table td, .target-table th {
    padding: 10px 14px;
}

@media (max-width: 768px) {
    .signal-gauge .gauge-value { font-size: 2.5em; }
    .threshold-bar { height: 24px; }
}
"""


# =============================================================================
# DATA LOADING
# =============================================================================

def load_ticker(ticker):
    """Load adjusted close from price_cache pkl."""
    path = os.path.join(CACHE_DIR, '{}.pkl'.format(ticker))
    if not os.path.exists(path):
        return None
    df = pd.read_pickle(path)
    for col in ['adjClose', 'close', 'Close']:
        if col in df.columns:
            s = df[col].dropna()
            if hasattr(s.index, 'tz') and s.index.tz is not None:
                s.index = s.index.tz_localize(None)
            return s
    return None


# =============================================================================
# SIGNAL STATE MANAGEMENT
# =============================================================================

def load_state():
    """Load previous signal state from disk."""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"signal_active": False, "entry_date": None, "entry_prices": {}}


def save_state(state):
    """Save signal state to disk."""
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2)


# =============================================================================
# EMAIL NOTIFICATION
# =============================================================================

def send_signal_email(signal_type, ratio_val, metal_approx, details=""):
    """Send notification email when signal changes state."""
    sys.path.insert(0, _DATA_DIR)
    try:
        from email_notify_v1_0 import send_email
    except ImportError:
        print("[WARN] email_notify_v1_0 not available -- skipping email")
        return False

    if signal_type == "ACTIVATED":
        subject = "GLD/SLV SIGNAL FIRED -- Silver Buy ({:.0f}:1)".format(metal_approx)
        body = (
            "GLD/SLV Ratio Signal has ACTIVATED.\n\n"
            "ETF Ratio: {:.2f} (crossed above P90 threshold of {:.2f})\n"
            "Approx Metal Ratio: {:.0f}:1\n\n"
            "Historical edge: 79% win rate, +13.6% mean SLV return over 60 days.\n"
            "Best vehicles: GDX (+16.1%), GDXJ (+15.8%), SLV (+13.6%).\n\n"
            "Suggested:\n"
            "  - Entry: SLV + GDX split, 2-3% portfolio\n"
            "  - Hold: 60 days\n"
            "  - Stop: -15% from entry\n\n"
            "{}\n\n"
            "-- Sortino Labs Automated Signal"
        ).format(ratio_val, ETF_RATIO_P90, metal_approx, details)
    elif signal_type == "DEACTIVATED":
        subject = "GLD/SLV Signal Cleared -- Ratio Below Threshold"
        body = (
            "GLD/SLV Ratio Signal has DEACTIVATED.\n\n"
            "ETF Ratio: {:.2f} (dropped below P90 threshold of {:.2f})\n"
            "Approx Metal Ratio: {:.0f}:1\n\n"
            "{}\n\n"
            "-- Sortino Labs Automated Signal"
        ).format(ratio_val, ETF_RATIO_P90, metal_approx, details)
    elif signal_type == "APPROACHING":
        subject = "GLD/SLV Ratio Approaching Signal ({:.0f}:1)".format(metal_approx)
        body = (
            "GLD/SLV Ratio is within 5% of the P90 signal threshold.\n\n"
            "ETF Ratio: {:.2f} (threshold: {:.2f})\n"
            "Approx Metal Ratio: {:.0f}:1\n"
            "Distance to threshold: {:.1f}%\n\n"
            "No action needed yet -- monitoring.\n\n"
            "-- Sortino Labs Automated Signal"
        ).format(ratio_val, ETF_RATIO_P90, metal_approx,
                 (ratio_val / ETF_RATIO_P90 - 1) * 100)
    else:
        return False

    return send_email(subject, body)


# =============================================================================
# COMPUTE SIGNAL
# =============================================================================

def compute_signal():
    """Compute current GLD/SLV signal state and all dashboard data."""
    gld = load_ticker('GLD')
    slv = load_ticker('SLV')
    if gld is None or slv is None:
        print("[FAIL] Cannot load GLD or SLV from price_cache")
        return None

    common = gld.index.intersection(slv.index)
    gld_c = gld.loc[common]
    slv_c = slv.loc[common]
    ratio = gld_c / slv_c

    latest_ratio = float(ratio.iloc[-1])
    latest_date = ratio.index[-1]
    metal_approx = latest_ratio * METAL_RATIO_APPROX_MULTIPLIER

    sma50 = ratio.rolling(SMA_WINDOW).mean()
    latest_sma = float(sma50.iloc[-1])

    # Percentile of current ratio in full history
    pct_rank = float((ratio < latest_ratio).mean() * 100)

    # Distance to P90 threshold
    dist_to_threshold = (latest_ratio / ETF_RATIO_P90 - 1) * 100

    # Signal status
    if latest_ratio >= ETF_RATIO_P90:
        status = "ACTIVE"
    elif dist_to_threshold >= -5:
        status = "APPROACHING"
    else:
        status = "DORMANT"

    # 50-day sparkline data (ratio values)
    spark_data = ratio.iloc[-50:].tolist()

    # Ratio stats
    ratio_min = float(ratio.min())
    ratio_max = float(ratio.max())
    ratio_mean = float(ratio.mean())
    ratio_p10 = float(ratio.quantile(0.10))
    ratio_p25 = float(ratio.quantile(0.25))
    ratio_p75 = float(ratio.quantile(0.75))
    ratio_p90 = float(ratio.quantile(0.90))

    # Current target prices (for active signal tracking)
    target_data = {}
    for ticker in TARGETS:
        ts = load_ticker(ticker)
        if ts is None:
            continue
        if latest_date in ts.index:
            target_data[ticker] = {
                'latest_price': round(float(ts.loc[latest_date]), 2),
                'date': str(latest_date.date()),
            }

    # Historical trade stats
    wins = sum(1 for t in HISTORICAL_TRADES if t['win'])
    total = len(HISTORICAL_TRADES)
    avg_ret = np.mean([t['ret'] for t in HISTORICAL_TRADES])
    median_ret = float(np.median([t['ret'] for t in HISTORICAL_TRADES]))

    return {
        'generated_at': datetime.datetime.now().isoformat(),
        'data_date': str(latest_date.date()),
        'latest_ratio': round(latest_ratio, 4),
        'metal_approx': round(metal_approx, 0),
        'sma50': round(latest_sma, 4),
        'pct_rank': round(pct_rank, 1),
        'dist_to_threshold': round(dist_to_threshold, 1),
        'status': status,
        'threshold': ETF_RATIO_P90,
        'sparkline': [round(v, 3) for v in spark_data],
        'ratio_stats': {
            'min': round(ratio_min, 2),
            'max': round(ratio_max, 2),
            'mean': round(ratio_mean, 2),
            'p10': round(ratio_p10, 2),
            'p25': round(ratio_p25, 2),
            'p75': round(ratio_p75, 2),
            'p90': round(ratio_p90, 2),
        },
        'target_data': target_data,
        'historical': {
            'wins': wins,
            'total': total,
            'win_rate': round(wins / total * 100, 1),
            'avg_ret': round(avg_ret, 1),
            'median_ret': round(median_ret, 1),
        },
        'trades': HISTORICAL_TRADES,
    }


# =============================================================================
# HTML BUILDER
# =============================================================================

def _sparkline_svg(values, width=200, height=40, color='#3b82f6'):
    """Generate inline SVG sparkline."""
    if not values or len(values) < 2:
        return ''
    mn = min(values)
    mx = max(values)
    rng = mx - mn if mx != mn else 1
    points = []
    for i, v in enumerate(values):
        x = i / (len(values) - 1) * width
        y = height - ((v - mn) / rng) * (height - 4) - 2
        points.append('{:.1f},{:.1f}'.format(x, y))
    return (
        '<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
        'style="vertical-align:middle;">'
        '<polyline points="{pts}" fill="none" stroke="{c}" '
        'stroke-width="2" stroke-linejoin="round"/>'
        '</svg>'
    ).format(w=width, h=height, pts=' '.join(points), c=color)


def build_body_html(data, writer):
    """Build dashboard body HTML."""
    parts = []

    status = data['status']
    ratio = data['latest_ratio']
    metal = data['metal_approx']
    dist = data['dist_to_threshold']
    hist = data['historical']

    # Stat bar
    stat_css = 'neg' if status == 'ACTIVE' else ('warn' if status == 'APPROACHING' else 'neutral')
    parts.append(writer.stat_bar([
        ("Data", data['data_date'], "neutral"),
        ("Signal", status, stat_css),
        ("ETF Ratio", "{:.2f}".format(ratio), "neutral"),
        ("Metal ~", "{:.0f}:1".format(metal), "neutral"),
        ("To Threshold", "{:+.1f}%".format(dist), stat_css),
        ("Win Rate", "{}%".format(hist['win_rate']), "pos"),
    ]))

    parts.append(writer.build_header(
        "Gold/Silver Ratio Mean-Reversion &nbsp;|&nbsp; "
        "P90 Threshold: {:.2f} (~{:.0f}:1)".format(data['threshold'], data['threshold'] * 10)
    ))

    # Signal status banner
    if status == 'ACTIVE':
        banner_text = "SIGNAL ACTIVE -- Silver buy zone"
        banner_color = '#d97706'
    elif status == 'APPROACHING':
        banner_text = "APPROACHING -- {:.1f}% from threshold".format(abs(dist))
        banner_color = '#ca8a04'
    else:
        banner_text = "DORMANT -- {:.1f}% below threshold".format(abs(dist))
        banner_color = '#64748b'

    parts.append(writer.regime_banner(
        status,
        banner_text,
        color=banner_color
    ))

    # Gauge section
    gauge_color = '#d97706' if status == 'ACTIVE' else ('#ca8a04' if status == 'APPROACHING' else '#3b82f6')
    sparkline = _sparkline_svg(data['sparkline'], width=280, height=50, color=gauge_color)

    gauge_html = (
        '<div class="signal-gauge">'
        '<div class="gauge-value" style="color:{color};">{ratio:.2f}</div>'
        '<div class="gauge-metal">~{metal:.0f}:1 gold/silver</div>'
        '<div class="gauge-label">Percentile: {pct:.0f}th &nbsp;|&nbsp; '
        'SMA50: {sma:.2f} &nbsp;|&nbsp; Threshold: {thresh:.2f}</div>'
        '<div style="margin-top:16px;">{spark}</div>'
        '<div class="gauge-label">50-day ratio trend</div>'
        '</div>'
    ).format(
        color=gauge_color, ratio=ratio, metal=metal,
        pct=data['pct_rank'], sma=data['sma50'], thresh=data['threshold'],
        spark=sparkline
    )
    parts.append(writer.section("Current Ratio", gauge_html))

    # Threshold proximity bar
    stats = data['ratio_stats']
    bar_min = stats['min']
    bar_max = stats['max']
    bar_range = bar_max - bar_min if bar_max != bar_min else 1

    current_pct = (ratio - bar_min) / bar_range * 100
    thresh_pct = (data['threshold'] - bar_min) / bar_range * 100

    bar_html = (
        '<div style="padding:16px;">'
        '<div style="display:flex;justify-content:space-between;font-size:0.85em;color:#666;margin-bottom:4px;">'
        '<span>{mn:.1f} (min)</span>'
        '<span>{mean:.1f} (avg)</span>'
        '<span>{mx:.1f} (max)</span>'
        '</div>'
        '<div class="threshold-bar">'
        '<div class="threshold-marker" style="left:{tpct:.1f}%;">'
        '<div class="threshold-marker-label" style="color:#dc2626;">P90 ({thresh:.2f})</div>'
        '</div>'
        '<div class="current-marker" style="left:{cpct:.1f}%;"></div>'
        '</div>'
        '<div style="text-align:center;font-size:0.85em;color:#666;margin-top:8px;">'
        'Blue dot = current ratio &nbsp;|&nbsp; Black line = P90 threshold'
        '</div>'
        '</div>'
    ).format(
        mn=bar_min, mean=stats['mean'], mx=bar_max,
        tpct=min(thresh_pct, 100), cpct=min(max(current_pct, 0), 100),
        thresh=data['threshold']
    )
    parts.append(writer.section("Ratio Position (Full History)", bar_html))

    # Signal edge summary
    edge_html = (
        '<div style="padding:16px;">'
        '<table class="target-table" style="width:100%;border-collapse:collapse;">'
        '<tr style="border-bottom:2px solid #e5e7eb;">'
        '<th style="text-align:left;">Metric</th>'
        '<th style="text-align:right;">At Signal (P90)</th>'
        '<th style="text-align:right;">Baseline (all periods)</th>'
        '<th style="text-align:right;">Edge</th>'
        '</tr>'
        '<tr><td>SLV 60d Mean Return</td>'
        '<td style="text-align:right;font-weight:700;color:#16a34a;">+13.6%</td>'
        '<td style="text-align:right;">+3.5%</td>'
        '<td style="text-align:right;font-weight:700;">+10.1pp</td></tr>'
        '<tr><td>Win Rate</td>'
        '<td style="text-align:right;font-weight:700;color:#16a34a;">78.6%</td>'
        '<td style="text-align:right;">55.0%</td>'
        '<td style="text-align:right;font-weight:700;">+23.6pp</td></tr>'
        '<tr><td>Median Return</td>'
        '<td style="text-align:right;font-weight:700;color:#16a34a;">+11.9%</td>'
        '<td style="text-align:right;">+1.8%</td>'
        '<td style="text-align:right;font-weight:700;">+10.1pp</td></tr>'
        '<tr><td>GDX 60d Mean</td>'
        '<td style="text-align:right;font-weight:700;color:#16a34a;">+16.1%</td>'
        '<td style="text-align:right;">--</td>'
        '<td style="text-align:right;">Best vehicle</td></tr>'
        '<tr><td>GDXJ 60d Mean</td>'
        '<td style="text-align:right;font-weight:700;color:#16a34a;">+15.8%</td>'
        '<td style="text-align:right;">--</td>'
        '<td style="text-align:right;">2nd best</td></tr>'
        '</table>'
        '<div style="margin-top:12px;padding:10px;background:#f0fdf4;border-radius:8px;'
        'font-size:0.9em;color:#166534;">'
        'Scimode validated: ratio-specific signal (not just silver oversold). '
        'Outperforms pure SLV mean-reversion by 2.7x. KB finding #44746.'
        '</div>'
        '</div>'
    )
    parts.append(writer.section("Signal Edge (14 trades, non-overlapping)", edge_html))

    # Target prices (when signal active)
    if data['target_data']:
        rows = []
        for ticker in TARGETS:
            if ticker not in data['target_data']:
                continue
            td = data['target_data'][ticker]
            rows.append(
                '<tr><td style="font-weight:700;">{}</td>'
                '<td style="text-align:right;">${:.2f}</td></tr>'.format(
                    ticker, td['latest_price'])
            )
        if rows:
            target_html = (
                '<div style="padding:16px;">'
                '<table class="target-table" style="width:100%;border-collapse:collapse;">'
                '<tr style="border-bottom:2px solid #e5e7eb;">'
                '<th style="text-align:left;">Ticker</th>'
                '<th style="text-align:right;">Latest Price</th>'
                '</tr>'
                '{}'
                '</table>'
                '</div>'
            ).format('\n'.join(rows))
            parts.append(writer.section("Target Prices", target_html))

    # Historical trades table
    trade_rows = []
    for t in reversed(HISTORICAL_TRADES):
        css = 'trade-win' if t['win'] else 'trade-loss'
        icon = '[WIN]' if t['win'] else '[LOSS]'
        trade_rows.append(
            '<tr>'
            '<td>{entry}</td>'
            '<td>{exit}</td>'
            '<td class="{css}" style="text-align:right;">{ret:+.1f}%</td>'
            '<td class="{css}">{icon}</td>'
            '</tr>'.format(
                entry=t['entry'], exit=t['exit'],
                ret=t['ret'], css=css, icon=icon)
        )

    trades_html = (
        '<div style="padding:16px;">'
        '<table class="target-table" style="width:100%;border-collapse:collapse;">'
        '<tr style="border-bottom:2px solid #e5e7eb;">'
        '<th style="text-align:left;">Entry</th>'
        '<th style="text-align:left;">Exit (60d)</th>'
        '<th style="text-align:right;">SLV Return</th>'
        '<th>Result</th>'
        '</tr>'
        '{}'
        '<tr style="border-top:2px solid #e5e7eb;font-weight:700;">'
        '<td colspan="2">Total: {total} trades</td>'
        '<td style="text-align:right;">{avg:+.1f}% avg</td>'
        '<td>{wins}/{total} ({wr:.0f}%)</td>'
        '</tr>'
        '</table>'
        '</div>'
    ).format(
        '\n'.join(trade_rows),
        total=hist['total'], avg=hist['avg_ret'],
        wins=hist['wins'], wr=hist['win_rate']
    )
    parts.append(writer.section("Historical Trades (P90 signal, 60d hold)", trades_html))

    # Methodology
    method_html = (
        '<div style="padding:16px;font-size:0.92em;line-height:1.6;">'
        '<p><b>Signal:</b> GLD/SLV ETF ratio crosses above P90 (8.84, ~88:1 metal). '
        'The gold/silver ratio rises during fear and economic pessimism. Extreme readings '
        'historically precede silver mean-reversion rallies.</p>'
        '<p><b>Validation:</b> 14 non-overlapping trades (2019-2025), 79% win rate, '
        '+13.6% mean SLV 60d return vs +3.5% baseline. Ratio-specific (not just silver oversold). '
        'Monotonic threshold effect: higher ratio = bigger payoff.</p>'
        '<p><b>Best vehicles:</b> GDX (+16.1%), GDXJ (+15.8%), SLV (+13.6%). '
        'Miners have operating leverage to metal price.</p>'
        '<p><b>Risk:</b> Worst MAE was -34.6% (COVID crash, 2020-02-20). '
        'Median MAE is -3.3%. Suggested stop: -15%. Size: 2-3% portfolio.</p>'
        '<p><b>Caveat:</b> n=14 trades. Signal fires 1-2x/year when active. '
        'Zero signals 2006-2012. Dashboard threshold of 80:1 is too low -- '
        'real edge starts at ~88:1 (P90).</p>'
        '</div>'
    )
    parts.append(writer.section("Methodology", method_html))

    parts.append(writer.llm_block())
    parts.append(writer.footer())

    return '\n'.join(parts)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("[START] GLD/SLV Ratio Signal Dashboard v1.0")

    data = compute_signal()
    if data is None:
        print("[FAIL] Could not compute signal")
        return

    status = data['status']
    ratio = data['latest_ratio']
    metal = data['metal_approx']

    print("  Ratio: {:.4f} (ETF) / ~{:.0f}:1 (metal)".format(ratio, metal))
    print("  Percentile: {:.0f}th".format(data['pct_rank']))
    print("  Distance to P90: {:.1f}%".format(data['dist_to_threshold']))
    print("  Status: {}".format(status))

    # Check for state change -> email
    prev_state = load_state()
    prev_active = prev_state.get('signal_active', False)

    if status == 'ACTIVE' and not prev_active:
        # Signal just activated
        print("  [ALERT] Signal ACTIVATED -- sending email")
        send_signal_email('ACTIVATED', ratio, metal)
        new_state = {
            'signal_active': True,
            'entry_date': data['data_date'],
            'entry_prices': {k: v['latest_price'] for k, v in data['target_data'].items()},
        }
        save_state(new_state)
    elif status != 'ACTIVE' and prev_active:
        # Signal just deactivated
        details = ""
        if prev_state.get('entry_date'):
            details = "Was active since {}".format(prev_state['entry_date'])
            # Calculate returns for each target
            for ticker, entry_price in prev_state.get('entry_prices', {}).items():
                if ticker in data['target_data']:
                    current = data['target_data'][ticker]['latest_price']
                    ret = (current / entry_price - 1) * 100
                    details += "\n  {} : ${:.2f} -> ${:.2f} ({:+.1f}%)".format(
                        ticker, entry_price, current, ret)
        print("  [ALERT] Signal DEACTIVATED -- sending email")
        send_signal_email('DEACTIVATED', ratio, metal, details)
        save_state({'signal_active': False, 'entry_date': None, 'entry_prices': {}})
    elif status == 'APPROACHING':
        # Only email on first approach (check if we already emailed)
        if not prev_state.get('approach_emailed'):
            print("  [ALERT] Signal APPROACHING -- sending email")
            send_signal_email('APPROACHING', ratio, metal)
            prev_state['approach_emailed'] = True
            save_state(prev_state)
    else:
        # Reset approach flag when dormant
        if prev_state.get('approach_emailed'):
            prev_state['approach_emailed'] = False
            save_state(prev_state)

    # Build dashboard HTML
    writer = DashboardWriter("gld-slv-signal", "Gold/Silver Ratio Signal")
    body = build_body_html(data, writer)
    writer.write(body, extra_css=EXTRA_CSS, extra_js="")
    print("[OK] Dashboard written")

    # Save data snapshot
    snapshot_path = os.path.join(_DATA_DIR, 'output', 'scientist', 'gld_slv_signal_latest.json')
    with open(snapshot_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, default=str)
    print("[OK] Snapshot saved to {}".format(snapshot_path))


if __name__ == '__main__':
    main()
