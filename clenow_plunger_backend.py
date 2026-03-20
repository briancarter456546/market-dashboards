# -*- coding: utf-8 -*-
# =============================================================================
# clenow_plunger_backend.py - v1.0
# Clenow Plunger Dashboard Backend
# =============================================================================
# Calculates the Clenow Plunger (ATR-normalized pullback from 20-day extreme)
# for all tickers in price_cache and writes a static HTML dashboard.
#
# Formula (Clenow original, ATR(50)):
#   Uptrend  (EMA50 > EMA100): (20d High - Close) / ATR(50)
#   Downtrend (EMA50 < EMA100): (Close - 20d Low) / ATR(50)
#
# Run: python clenow_plunger_backend.py
# Output: docs/clenow-plunger/index.html (+ dated archive)
# =============================================================================

import os
import gc
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

from dashboard_writer import DashboardWriter, price_cache_freshness

warnings.filterwarnings('ignore')

# =============================================================================
# PATH SETUP
# =============================================================================

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.normpath(os.path.join(_SCRIPT_DIR, '..', 'perplexity-user-data'))
CACHE_DIR = os.path.normpath(os.path.join(_DATA_DIR, 'price_cache'))

# Parameters (Clenow original)
EMA_SHORT = 50
EMA_LONG = 100
ATR_PERIOD = 50
HL_PERIOD = 20

# Key tickers to always show
KEY_TICKERS = [
    'SPY', 'QQQ', 'IWM', 'DIA', 'TQQQ', 'UPRO',
    'XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLP', 'XLU', 'XLB', 'XLRE', 'XLC',
    'GLD', 'SLV', 'TLT', 'HYG', 'SMH', 'XBI', 'ARKK', 'IYR',
    'EEM', 'EFA', 'FXI', 'VWO',
    'IAU', 'JNJ', 'SH', 'PSQ',
]

# =============================================================================
# EXTRA CSS
# =============================================================================

EXTRA_CSS = """
.plunger-bar {
    display: inline-block;
    height: 14px;
    border-radius: 3px;
    min-width: 2px;
    vertical-align: middle;
}
.plunger-bar.pain { background: #ef4444; }
.plunger-bar.moderate { background: #f59e0b; }
.plunger-bar.intact { background: #22c55e; }
.plunger-bar.low { background: #94a3b8; }

.trend-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.8em;
    font-weight: 600;
}
.trend-badge.up { background: #dcfce7; color: #166534; }
.trend-badge.down { background: #fee2e2; color: #991b1b; }

.breadth-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin: 16px 0;
}
.breadth-card {
    background: #f8f9fb;
    border-radius: 8px;
    padding: 16px;
    text-align: center;
}
.breadth-card .label {
    font-size: 0.8em;
    color: #666;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 4px;
}
.breadth-card .value {
    font-size: 1.8em;
    font-weight: 700;
    font-family: 'IBM Plex Mono', monospace;
}
.breadth-card .pct {
    font-size: 0.85em;
    color: #888;
    margin-top: 2px;
}
"""

EXTRA_JS = """
/* Sortable table */
document.querySelectorAll('th.sortable').forEach(function(th) {
    th.style.cursor = 'pointer';
    th.addEventListener('click', function() {
        var table = th.closest('table');
        var tbody = table.querySelector('tbody');
        var rows = Array.from(tbody.querySelectorAll('tr'));
        var idx = Array.from(th.parentNode.children).indexOf(th);
        var asc = th.dataset.sort !== 'asc';
        th.dataset.sort = asc ? 'asc' : 'desc';
        // Reset other headers
        th.parentNode.querySelectorAll('th').forEach(function(h) {
            if (h !== th) h.dataset.sort = '';
        });
        rows.sort(function(a, b) {
            var av = parseFloat(a.children[idx].dataset.val || a.children[idx].textContent) || 0;
            var bv = parseFloat(b.children[idx].dataset.val || b.children[idx].textContent) || 0;
            return asc ? av - bv : bv - av;
        });
        rows.forEach(function(r) { tbody.appendChild(r); });
    });
});

/* Filter input */
var filterInput = document.getElementById('ticker-filter');
if (filterInput) {
    filterInput.addEventListener('input', function() {
        var val = this.value.toUpperCase();
        document.querySelectorAll('.plunger-table tbody tr').forEach(function(row) {
            var ticker = row.children[0].textContent.toUpperCase();
            row.style.display = ticker.indexOf(val) !== -1 ? '' : 'none';
        });
    });
}
"""


def load_pkl(ticker):
    path = os.path.join(CACHE_DIR, f'{ticker}.pkl')
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_pickle(path)
        if 'adjClose' in df.columns:
            df['close'] = df['adjClose'].fillna(df['close'])
        return df
    except Exception:
        return None


def compute_plunger(df):
    """Compute Clenow Plunger using ATR(50)."""
    c = df['close'].values.astype(np.float32)
    h = df['high'].values.astype(np.float32)
    lo = df['low'].values.astype(np.float32)

    ema_s = pd.Series(c).ewm(span=EMA_SHORT, adjust=False).mean().values
    ema_l = pd.Series(c).ewm(span=EMA_LONG, adjust=False).mean().values
    uptrend = ema_s > ema_l

    prev_c = np.roll(c, 1)
    prev_c[0] = c[0]
    tr = np.maximum(h - lo, np.maximum(np.abs(h - prev_c), np.abs(lo - prev_c)))
    atr = pd.Series(tr).rolling(ATR_PERIOD, min_periods=ATR_PERIOD).mean().values

    hh20 = pd.Series(h).rolling(HL_PERIOD, min_periods=HL_PERIOD).max().values
    ll20 = pd.Series(lo).rolling(HL_PERIOD, min_periods=HL_PERIOD).min().values

    with np.errstate(divide='ignore', invalid='ignore'):
        plunger = np.where(
            uptrend,
            (hh20 - c) / np.where(atr > 0, atr, np.nan),
            (c - ll20) / np.where(atr > 0, atr, np.nan)
        )

    return plunger, uptrend, atr


def scan_all():
    """Scan all tickers and return results list."""
    pkl_files = [f for f in os.listdir(CACHE_DIR) if f.endswith('.pkl')]
    results = []
    for i, fname in enumerate(pkl_files):
        ticker = fname.replace('.pkl', '')
        df = load_pkl(ticker)
        if df is None or len(df) < 200:
            continue
        try:
            plunger, uptrend, atr = compute_plunger(df)
            idx = len(df) - 1
            if np.isnan(plunger[idx]):
                continue

            close = float(df['close'].iloc[idx])
            p = float(plunger[idx])
            p5 = float(plunger[max(0, idx - 5)]) if not np.isnan(plunger[max(0, idx - 5)]) else None
            p10 = float(plunger[max(0, idx - 10)]) if not np.isnan(plunger[max(0, idx - 10)]) else None

            # 20d plunger history
            recent = plunger[max(0, idx - 19):idx + 1]
            recent = recent[~np.isnan(recent)]
            p_max20 = float(np.nanmax(recent)) if len(recent) > 0 else None

            results.append({
                'ticker': ticker,
                'close': round(close, 2),
                'plunger': round(p, 3),
                'uptrend': bool(uptrend[idx]),
                'atr50': round(float(atr[idx]), 2) if not np.isnan(atr[idx]) else 0,
                'plunger_5d': round(p5, 3) if p5 is not None else None,
                'plunger_10d': round(p10, 3) if p10 is not None else None,
                'plunger_max20': round(p_max20, 3) if p_max20 is not None else None,
                'delta_5d': round(p - p5, 3) if p5 is not None else None,
            })
        except Exception:
            pass

        if (i + 1) % 300 == 0:
            gc.collect()

    return results


def classify_plunger(p):
    if p >= 3.0:
        return 'pain', 'Pain Zone'
    elif p >= 1.5:
        return 'moderate', 'Moderate Pullback'
    elif p >= 0.5:
        return 'intact', 'Trend Intact'
    else:
        return 'low', 'At Extreme'


def plunger_bar_html(p):
    cls, _ = classify_plunger(p)
    width = min(max(int(p * 25), 2), 200)
    return '<span class="plunger-bar {}" style="width:{}px;"></span>'.format(cls, width)


def build_breadth_section(results):
    """Build market breadth overview cards."""
    total = len(results)
    if total == 0:
        return '<p>No data</p>'

    plungers = [r['plunger'] for r in results]
    uptrend_n = sum(1 for r in results if r['uptrend'])
    pain_n = sum(1 for r in results if r['plunger'] >= 3.0)
    moderate_n = sum(1 for r in results if 1.5 <= r['plunger'] < 3.0)
    intact_n = sum(1 for r in results if r['plunger'] < 0.5)

    mean_p = np.mean(plungers)
    median_p = np.median(plungers)

    cards = [
        ('Mean Plunger', '{:.2f}'.format(mean_p), 'of {}'.format(total)),
        ('Median', '{:.2f}'.format(median_p), ''),
        ('% Uptrend', '{:.0f}%'.format(uptrend_n / total * 100), '{} tickers'.format(uptrend_n)),
        ('Pain Zone', str(pain_n), '>= 3.0 ATR ({:.1f}%)'.format(pain_n / total * 100)),
    ]

    html = '<div class="breadth-grid">'
    for label, value, pct in cards:
        html += (
            '<div class="breadth-card">'
            '<div class="label">{}</div>'
            '<div class="value">{}</div>'
            '<div class="pct">{}</div>'
            '</div>'.format(label, value, pct)
        )
    html += '</div>'

    # Distribution bar
    pain_pct = pain_n / total * 100
    mod_pct = moderate_n / total * 100
    intact_pct = intact_n / total * 100
    other_pct = 100 - pain_pct - mod_pct - intact_pct

    html += '<div style="margin:12px 0;">'
    html += '<div style="display:flex;height:24px;border-radius:6px;overflow:hidden;">'
    if pain_pct > 0:
        html += '<div style="width:{:.1f}%;background:#ef4444;" title="Pain Zone: {:.1f}%"></div>'.format(pain_pct, pain_pct)
    if mod_pct > 0:
        html += '<div style="width:{:.1f}%;background:#f59e0b;" title="Moderate: {:.1f}%"></div>'.format(mod_pct, mod_pct)
    if other_pct > 0:
        html += '<div style="width:{:.1f}%;background:#60a5fa;" title="Mild: {:.1f}%"></div>'.format(other_pct, other_pct)
    if intact_pct > 0:
        html += '<div style="width:{:.1f}%;background:#22c55e;" title="At Extreme: {:.1f}%"></div>'.format(intact_pct, intact_pct)
    html += '</div>'
    html += '<div style="display:flex;justify-content:space-between;font-size:0.75em;color:#888;margin-top:4px;">'
    html += '<span>Pain ({:.0f}%)</span>'.format(pain_pct)
    html += '<span>Moderate ({:.0f}%)</span>'.format(mod_pct)
    html += '<span>Mild ({:.0f}%)</span>'.format(other_pct)
    html += '<span>At Extreme ({:.0f}%)</span>'.format(intact_pct)
    html += '</div></div>'

    return html


def build_table(rows, table_id, show_filter=True):
    """Build a sortable HTML table of plunger data."""
    html = ''
    if show_filter:
        html += '<div style="margin-bottom:12px;"><input type="text" id="ticker-filter" placeholder="Filter ticker..." style="padding:6px 12px;border:1px solid #ddd;border-radius:4px;font-size:0.9em;width:200px;"></div>'

    html += '<table class="plunger-table" id="{}" style="width:100%;border-collapse:collapse;">'.format(table_id)
    html += '<thead><tr>'
    headers = [
        ('Ticker', False),
        ('Close', True),
        ('Plunger', True),
        ('', False),  # bar
        ('Zone', False),
        ('Trend', False),
        ('ATR(50)', True),
        ('5d Ago', True),
        ('10d Ago', True),
        ('20d Max', True),
        ('5d Chg', True),
    ]
    for label, sortable in headers:
        cls = ' class="sortable"' if sortable else ''
        html += '<th{}>{}</th>'.format(cls, label)
    html += '</tr></thead><tbody>'

    for r in rows:
        cls_name, zone_label = classify_plunger(r['plunger'])
        trend_cls = 'up' if r['uptrend'] else 'down'
        trend_label = 'UP' if r['uptrend'] else 'DN'

        delta_html = ''
        if r.get('delta_5d') is not None:
            d = r['delta_5d']
            color = '#ef4444' if d > 0.3 else ('#22c55e' if d < -0.3 else '#666')
            delta_html = '<span style="color:{};">{:+.2f}</span>'.format(color, d)

        html += '<tr>'
        html += '<td style="font-weight:600;">{}</td>'.format(r['ticker'])
        html += '<td data-val="{}">${:.2f}</td>'.format(r['close'], r['close'])
        html += '<td data-val="{}" style="font-weight:700;">{:.2f}</td>'.format(r['plunger'], r['plunger'])
        html += '<td>{}</td>'.format(plunger_bar_html(r['plunger']))
        html += '<td><span style="color:{};">{}</span></td>'.format(
            '#ef4444' if cls_name == 'pain' else '#f59e0b' if cls_name == 'moderate' else '#22c55e' if cls_name == 'intact' else '#888',
            zone_label
        )
        html += '<td><span class="trend-badge {}">{}</span></td>'.format(trend_cls, trend_label)
        html += '<td data-val="{}">{:.2f}</td>'.format(r['atr50'], r['atr50'])
        html += '<td data-val="{}">{}</td>'.format(
            r.get('plunger_5d', 0) or 0,
            '{:.2f}'.format(r['plunger_5d']) if r.get('plunger_5d') is not None else '--'
        )
        html += '<td data-val="{}">{}</td>'.format(
            r.get('plunger_10d', 0) or 0,
            '{:.2f}'.format(r['plunger_10d']) if r.get('plunger_10d') is not None else '--'
        )
        html += '<td data-val="{}">{}</td>'.format(
            r.get('plunger_max20', 0) or 0,
            '{:.2f}'.format(r['plunger_max20']) if r.get('plunger_max20') is not None else '--'
        )
        html += '<td>{}</td>'.format(delta_html)
        html += '</tr>'

    html += '</tbody></table>'
    return html


def main():
    print("=" * 60)
    print("  Clenow Plunger Dashboard Backend v1.0")
    print("=" * 60)

    if not os.path.isdir(CACHE_DIR):
        print("  [FAIL] price_cache not found: {}".format(CACHE_DIR))
        return

    results = scan_all()
    if not results:
        print("  [FAIL] No results")
        return

    print("  [OK] {} tickers scanned".format(len(results)))

    # Sort by plunger descending
    results.sort(key=lambda x: x['plunger'], reverse=True)

    # Build dashboard
    writer = DashboardWriter("clenow-plunger", "Clenow Plunger Scanner")
    body = writer.build_header(subtitle="ATR(50) | EMA 50/100 | 20-day extremes")

    # Stat bar
    total = len(results)
    pain_n = sum(1 for r in results if r['plunger'] >= 3.0)
    mod_n = sum(1 for r in results if 1.5 <= r['plunger'] < 3.0)
    up_n = sum(1 for r in results if r['uptrend'])
    mean_p = np.mean([r['plunger'] for r in results])

    body += writer.stat_bar([
        ("Mean Plunger", "{:.2f}".format(mean_p), "warn" if mean_p > 1.5 else "neutral"),
        ("Pain Zone (>=3)", str(pain_n), "neg" if pain_n > 20 else "warn" if pain_n > 5 else "neutral"),
        ("Moderate (1.5-3)", str(mod_n), "warn"),
        ("Uptrending", "{}%".format(int(up_n / total * 100)), "pos" if up_n / total > 0.6 else "neg"),
        ("Total Tickers", str(total), "neutral"),
    ])

    # Regime banner
    if mean_p >= 2.0:
        banner_label = "HIGH PAIN -- TREND FOLLOWER FLUSH"
        banner_color = "#ef4444"
    elif mean_p >= 1.0:
        banner_label = "MODERATE PULLBACK REGIME"
        banner_color = "#f59e0b"
    else:
        banner_label = "TRENDS INTACT"
        banner_color = "#22c55e"

    body += writer.regime_banner(
        banner_label,
        "Mean Plunger: {:.2f} | {} pain zone tickers | {} uptrending".format(mean_p, pain_n, up_n),
        banner_color
    )

    # Market Breadth section
    body += writer.section(
        "Market Breadth",
        build_breadth_section(results),
        hint="Distribution of Plunger values across all tickers"
    )

    # Key Tickers section
    key_rows = [r for r in results if r['ticker'] in KEY_TICKERS]
    key_rows.sort(key=lambda x: x['plunger'], reverse=True)
    if key_rows:
        body += writer.section(
            "Key Tickers",
            build_table(key_rows, "key-table", show_filter=False),
            hint="Major indexes, sectors, and portfolio tickers"
        )

    # Pain Zone tickers
    pain_rows = [r for r in results if r['plunger'] >= 3.0]
    if pain_rows:
        body += writer.section(
            "Pain Zone (Plunger >= 3.0)",
            build_table(pain_rows, "pain-table", show_filter=False),
            hint="{} tickers at trend-follower stop-out levels".format(len(pain_rows))
        )

    # Moderate Pullback tickers
    mod_rows = [r for r in results if 1.5 <= r['plunger'] < 3.0]
    if mod_rows:
        body += writer.section(
            "Moderate Pullback (1.5 - 3.0)",
            build_table(mod_rows[:100], "mod-table", show_filter=False),
            hint="Top {} of {} tickers".format(min(100, len(mod_rows)), len(mod_rows))
        )

    # Full table (all tickers)
    body += writer.section(
        "All Tickers",
        build_table(results, "all-table", show_filter=True),
        hint="Click column headers to sort | {} tickers".format(len(results))
    )

    # About section
    body += writer.section(
        "About the Clenow Plunger",
        '<div style="max-width:700px;line-height:1.7;">'
        '<p>The <b>Clenow Plunger</b> (Andreas Clenow, ACIES Asset Management) measures how many ATR units '
        'the price has pulled back from its recent 20-day extreme. Trend-following CTAs typically use a '
        '3-ATR trailing stop on 50-day breakout systems, so when the Plunger hits ~3.0, it signals that '
        'many trend followers are being forced out.</p>'
        '<p><b>Formula:</b></p>'
        '<ul>'
        '<li>Uptrend (EMA50 > EMA100): Plunger = (20d High - Close) / ATR(50)</li>'
        '<li>Downtrend (EMA50 < EMA100): Plunger = (Close - 20d Low) / ATR(50)</li>'
        '</ul>'
        '<p><b>Interpretation:</b></p>'
        '<ul>'
        '<li><span style="color:#22c55e;">~0:</span> Price at or near its 20-day extreme -- trend is intact</li>'
        '<li><span style="color:#f59e0b;">1-2:</span> Moderate pullback, manageable for trend followers</li>'
        '<li><span style="color:#ef4444;">3+:</span> Pain zone -- trend followers hitting stops. '
        'Potential counter-trend entry or re-entry point.</li>'
        '</ul>'
        '<p><b>Scimode validation:</b> Plunger >= 2.0 as RSI2 confirmation gate improves PF from 2.48 to 4.07 on SPY (n=28). '
        'Standalone Plunger >= 2.5 on QQQ: PF 2.61, WR 75% (n=72).</p>'
        '</div>'
    )

    body += writer.llm_block()
    body += writer.footer()

    writer.write(body, extra_css=EXTRA_CSS, extra_js=EXTRA_JS)
    print("  [OK] Dashboard written")


if __name__ == '__main__':
    main()
