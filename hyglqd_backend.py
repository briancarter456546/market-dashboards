# -*- coding: utf-8 -*-
# =============================================================================
# hyglqd_backend.py
# HYG/LQD Credit Spread Dashboard Backend
# =============================================================================
# Computes HYG/LQD credit spread analysis and writes a static HTML dashboard
# via DashboardWriter.
#
# Ported from: ../hyglqd_backend_v1.0.py
# Changes:
#   - Lives in market-dashboards/ (uses __file__-relative paths)
#   - Outputs HTML dashboard instead of JSON
#   - No JSON file output
#
# Run: python hyglqd_backend.py
# Output: docs/hyglqd-credit/index.html (+ dated archive)
#
# Author: Brian + Claude
# =============================================================================

import os
import pickle
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

from dashboard_writer import DashboardWriter

warnings.filterwarnings('ignore')

# =============================================================================
# PATH SETUP - __file__-relative, market-dashboards/ lives one level below root
# =============================================================================

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR   = os.path.normpath(os.path.join(_SCRIPT_DIR, '..', 'perplexity-user-data'))
CACHE_DIR   = os.path.normpath(os.path.join(_DATA_DIR, 'price_cache'))

# =============================================================================
# CONFIGURATION
# =============================================================================

RISK_ON_THRESHOLD  = 0.74
RISK_OFF_THRESHOLD = 0.71
FORWARD_WINDOWS    = [5, 10, 20]

# =============================================================================
# EXTRA CSS (dashboard-specific additions on top of shared theme)
# =============================================================================

EXTRA_CSS = """
/* Forward-return cards: tighter sub-label spacing */
.fwd-card .value {
    font-size: 1.7em;
}
.fwd-card .sub-row {
    display: flex;
    justify-content: space-between;
    font-size: 0.82em;
    color: #666;
    margin-top: 6px;
}
.fwd-card .sub-row span {
    font-family: 'IBM Plex Mono', monospace;
}

/* Threshold table */
.thresh-table td, .thresh-table th {
    padding: 10px 16px;
    border-bottom: 1px solid #f0f0f0;
    font-size: 0.93em;
}
.thresh-table th {
    background: #f8f9fb;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-size: 0.8em;
    color: #555;
    border-bottom: 2px solid #e2e4e8;
}
.thresh-table td:not(:first-child) {
    font-family: 'IBM Plex Mono', monospace;
}
.thresh-row-current td {
    font-weight: 700;
    background: #fafbfd;
}

/* Price table */
.price-table td, .price-table th {
    padding: 11px 16px;
    border-bottom: 1px solid #f0f0f0;
}
.price-table th {
    background: #f8f9fb;
    font-size: 0.8em;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #555;
    border-bottom: 2px solid #e2e4e8;
}
.price-table td:not(:first-child) {
    font-family: 'IBM Plex Mono', monospace;
}

/* ── Mobile responsive ── */
@media (max-width: 768px) {
    .fwd-card .value { font-size: 1.3em; }
    .fwd-card .sub-row { font-size: 0.75em; }
    .thresh-table td, .thresh-table th { padding: 7px 10px; font-size: 0.85em; }
    .price-table td, .price-table th { padding: 8px 10px; }
}
"""

# =============================================================================
# DATA LOADING
# =============================================================================

def load_from_cache(symbol):
    """Load a single price series from the pickle cache."""
    filepath = os.path.join(CACHE_DIR, '{}.pkl'.format(symbol))
    with open(filepath, 'rb') as f:
        df = pickle.load(f)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df.sort_index()


def load_credit_data():
    """Load HYG, LQD, and SPY; return aligned DataFrame."""
    hyg = load_from_cache('HYG')
    lqd = load_from_cache('LQD')
    spy = load_from_cache('SPY')

    hyg_close = hyg['close'] if 'close' in hyg.columns else hyg['adjClose']
    lqd_close = lqd['close'] if 'close' in lqd.columns else lqd['adjClose']
    spy_close = spy['close'] if 'close' in spy.columns else spy['adjClose']

    data = pd.DataFrame({
        'HYG': hyg_close,
        'LQD': lqd_close,
        'SPY': spy_close,
    }).dropna()

    return data

# =============================================================================
# COMPUTATION
# =============================================================================

def compute_hyglqd_ratio(data):
    """Add hyglqd_ratio column to data."""
    data['hyglqd_ratio'] = data['HYG'] / data['LQD']
    return data


def compute_forward_returns(data):
    """Add spy_fwd_Nd columns for each forward window."""
    for window in FORWARD_WINDOWS:
        data['spy_fwd_{}d'.format(window)] = (
            data['SPY'].pct_change(window).shift(-window) * 100
        )
    return data


def compute_percentile_rank(data):
    """Return percentile rank (0-100) of the current ratio vs all history."""
    current_ratio = data['hyglqd_ratio'].iloc[-1]
    percentile = (data['hyglqd_ratio'] < current_ratio).sum() / len(data) * 100
    return percentile


def compute_expected_returns(data):
    """
    For each forward window compute avg return, median, win rate, and sample
    size among days with ratio within ±0.02 of the current reading.
    """
    current_ratio = data['hyglqd_ratio'].iloc[-1]
    bucket_size = 0.02
    similar_days = data[
        (data['hyglqd_ratio'] >= current_ratio - bucket_size) &
        (data['hyglqd_ratio'] <= current_ratio + bucket_size)
    ]

    results = {}
    for window in FORWARD_WINDOWS:
        fwd_col = 'spy_fwd_{}d'.format(window)
        fwd_returns = similar_days[fwd_col].dropna()
        if len(fwd_returns) > 0:
            results['{}d'.format(window)] = {
                'avg_return':    round(fwd_returns.mean(), 2),
                'median_return': round(fwd_returns.median(), 2),
                'win_rate':      round((fwd_returns > 0).sum() / len(fwd_returns) * 100, 1),
                'sample_size':   len(fwd_returns),
            }
        else:
            results['{}d'.format(window)] = {
                'avg_return':    None,
                'median_return': None,
                'win_rate':      None,
                'sample_size':   0,
            }
    return results


def classify_regime(ratio):
    """Return (regime_key, regime_label, description) based on ratio level."""
    if ratio >= RISK_ON_THRESHOLD:
        return ('risk_on',  'Risk-On',  'High yield outperforming - risk appetite strong')
    elif ratio >= RISK_OFF_THRESHOLD:
        return ('neutral',  'Neutral',  'Balanced risk appetite')
    else:
        return ('risk_off', 'Risk-Off', 'Flight to quality - risk aversion elevated')


def compute_5d_change(data):
    """Return the 5-day percentage change in the HYG/LQD ratio."""
    current  = data['hyglqd_ratio'].iloc[-1]
    prev_5d  = data['hyglqd_ratio'].iloc[-6] if len(data) >= 6 else current
    return round(((current - prev_5d) / prev_5d) * 100, 2)

# =============================================================================
# HTML BUILDING HELPERS
# =============================================================================

def _fmt_return(value):
    """Format a return value as +X.XX% or '--' if None."""
    if value is None:
        return '--'
    return '{:+.2f}%'.format(value)


def _fmt_winrate(value):
    """Format a win-rate as X.X% or '--' if None."""
    if value is None:
        return '--'
    return '{:.1f}%'.format(value)


def _return_color(value):
    """CSS class based on sign of a return (or neutral if None)."""
    if value is None:
        return 'muted'
    return 'pos' if value > 0 else 'neg'


def build_forward_cards(expected_returns):
    """
    Build a 3-card grid — one per forward window — showing expected SPY returns.
    Card border-top is green when avg_return > 0, red otherwise.
    """
    card_parts = []
    for window in FORWARD_WINDOWS:
        key     = '{}d'.format(window)
        metrics = expected_returns.get(key, {})
        avg     = metrics.get('avg_return')
        med     = metrics.get('median_return')
        wr      = metrics.get('win_rate')
        n       = metrics.get('sample_size', 0)

        border_color = '#22c55e' if (avg is not None and avg > 0) else '#ef4444'
        val_class    = _return_color(avg)

        card_parts.append(
            '<div class="card fwd-card" style="border-top-color:{border};">'
            '<div class="label">SPY {window}-Day Forward</div>'
            '<div class="value {val_class}">{avg}</div>'
            '<div class="sub-row">'
            '  <span>Median: {med}</span>'
            '  <span>Win Rate: {wr}</span>'
            '  <span>n={n}</span>'
            '</div>'
            '</div>'.format(
                border=border_color,
                window=window,
                val_class=val_class,
                avg=_fmt_return(avg),
                med=_fmt_return(med),
                wr=_fmt_winrate(wr),
                n=n,
            )
        )

    return '<div class="cards">\n{}\n</div>'.format('\n'.join(card_parts))


def build_prices_table(current_hyg, current_lqd, current_spy, latest_date):
    """Small table showing the current prices for HYG, LQD, SPY."""
    rows_html = (
        '<tr>'
        '<td class="ticker">HYG</td>'
        '<td class="num">${hyg:.2f}</td>'
        '<td>iShares iBoxx High Yield Corporate Bond ETF</td>'
        '</tr>'
        '<tr>'
        '<td class="ticker">LQD</td>'
        '<td class="num">${lqd:.2f}</td>'
        '<td>iShares iBoxx Investment Grade Corporate Bond ETF</td>'
        '</tr>'
        '<tr>'
        '<td class="ticker">SPY</td>'
        '<td class="num">${spy:.2f}</td>'
        '<td>SPDR S&amp;P 500 ETF Trust</td>'
        '</tr>'
    ).format(hyg=current_hyg, lqd=current_lqd, spy=current_spy)

    table_html = (
        '<table class="price-table">'
        '<thead><tr>'
        '<th title="ETF ticker symbol">Ticker</th>'
        '<th title="Most recent closing price in USD">Close</th>'
        '<th title="Full name of the ETF fund">Description</th>'
        '</tr></thead>'
        '<tbody>{rows}</tbody>'
        '</table>'
        '<p style="margin-top:10px;font-size:0.82em;color:#aaa;">'
        'As of {date}'
        '</p>'
    ).format(rows=rows_html, date=latest_date)

    return table_html


def build_thresholds_table(current_ratio, regime_key):
    """
    Table showing the two threshold levels and where the current ratio sits.
    The current ratio row is highlighted.
    """
    # Determine position label
    if current_ratio >= RISK_ON_THRESHOLD:
        position = 'Above risk-on threshold'
    elif current_ratio >= RISK_OFF_THRESHOLD:
        position = 'Between thresholds (neutral zone)'
    else:
        position = 'Below risk-off threshold'

    def _row(label, value, description, highlight=False):
        css = ' class="thresh-row-current"' if highlight else ''
        return (
            '<tr{css}>'
            '<td>{label}</td>'
            '<td>{value:.4f}</td>'
            '<td>{description}</td>'
            '</tr>'
        ).format(css=css, label=label, value=value, description=description)

    rows_html = (
        _row('Risk-On Threshold',  RISK_ON_THRESHOLD,  'Ratio at or above this level = risk appetite strong')
        + _row('Current Ratio',    current_ratio,       position, highlight=True)
        + _row('Risk-Off Threshold', RISK_OFF_THRESHOLD, 'Ratio below this level = flight to quality')
    )

    table_html = (
        '<table class="thresh-table">'
        '<thead><tr>'
        '<th title="Threshold level or current ratio reading">Level</th>'
        '<th title="HYG/LQD price ratio value at this level">Ratio Value</th>'
        '<th title="What this ratio level means for risk appetite">Interpretation</th>'
        '</tr></thead>'
        '<tbody>{rows}</tbody>'
        '</table>'
    ).format(rows=rows_html)

    return table_html

# =============================================================================
# MAIN
# =============================================================================

def main():
    print('=' * 70)
    print('HYG/LQD CREDIT SPREAD DASHBOARD BACKEND')
    print('=' * 70)
    print('Time: {}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    print('Cache: {}'.format(CACHE_DIR))
    print()

    try:
        # ------------------------------------------------------------------
        # Load and compute
        # ------------------------------------------------------------------
        print('Loading credit spread data...')
        data = load_credit_data()
        print('  Loaded {} days  ({} to {})'.format(
            len(data),
            data.index.min().date(),
            data.index.max().date(),
        ))

        print('Computing HYG/LQD ratio...')
        data = compute_hyglqd_ratio(data)

        print('Computing forward returns...')
        data = compute_forward_returns(data)

        # Current snapshot
        latest_date   = data.index[-1].strftime('%Y-%m-%d')
        current_ratio = data['hyglqd_ratio'].iloc[-1]
        current_hyg   = data['HYG'].iloc[-1]
        current_lqd   = data['LQD'].iloc[-1]
        current_spy   = data['SPY'].iloc[-1]

        print('Analyzing historical context...')
        percentile       = compute_percentile_rank(data)
        expected_returns = compute_expected_returns(data)
        regime_key, regime_label, regime_desc = classify_regime(current_ratio)
        change_5d        = compute_5d_change(data)

        # ------------------------------------------------------------------
        # Console summary
        # ------------------------------------------------------------------
        print()
        print('  Date:          {}'.format(latest_date))
        print('  Ratio:         {:.4f}'.format(current_ratio))
        print('  5-Day Change:  {:+.2f}%'.format(change_5d))
        print('  Percentile:    {:.1f}%'.format(percentile))
        print('  Regime:        {}'.format(regime_label))
        for w in FORWARD_WINDOWS:
            m = expected_returns.get('{}d'.format(w), {})
            if m.get('avg_return') is not None:
                print('  {}d Expected:  {:+.2f}% avg (win rate: {:.1f}%, n={})'.format(
                    w, m['avg_return'], m['win_rate'], m['sample_size']))
            else:
                print('  {}d Expected:  No data'.format(w))
        print()

        # ------------------------------------------------------------------
        # Build HTML body
        # ------------------------------------------------------------------
        writer = DashboardWriter('hyglqd-credit', 'HYG/LQD Credit Spread Analysis')

        # 1. Stat bar
        change_sign  = '+' if change_5d >= 0 else ''
        change_class = 'pos' if change_5d >= 0 else 'neg'
        regime_stat_class = (
            'pos'  if regime_key == 'risk_on'  else
            'warn' if regime_key == 'neutral'  else
            'neg'
        )

        stat_items = [
            ('Date',          latest_date,                                    'neutral'),
            ('HYG/LQD Ratio', '{:.4f}'.format(current_ratio),               'neutral'),
            ('5D Change',     '{}{:.2f}%'.format(change_sign, change_5d),    change_class),
            ('Percentile',    '{:.1f}th'.format(percentile),                 'neutral'),
            ('Regime',        regime_label,                                   regime_stat_class),
        ]

        # 2. Regime banner color
        regime_color = (
            '#22c55e' if regime_key == 'risk_on'  else
            '#f59e0b' if regime_key == 'neutral'  else
            '#ef4444'
        )

        # Score HTML shown in the right panel of the banner
        score_html = (
            'Ratio: {ratio:.4f} &nbsp;|&nbsp; '
            'Percentile: {pct:.1f}th &nbsp;|&nbsp; '
            '5D Change: {chg:+.2f}%<br>'
            '{desc}'
        ).format(
            ratio=current_ratio,
            pct=percentile,
            chg=change_5d,
            desc=regime_desc,
        )

        # 3. Assemble body
        parts = []
        parts.append(writer.stat_bar(stat_items))
        parts.append(writer.build_header(
            'HYG/LQD Ratio Analysis &nbsp;|&nbsp; SPY Forward Returns'
        ))
        parts.append(writer.regime_banner(
            regime_label.upper(), score_html, color=regime_color
        ))

        # 4. Forward-return cards
        parts.append(build_forward_cards(expected_returns))

        # 5. Current prices section
        parts.append(writer.section(
            'Current Prices',
            build_prices_table(current_hyg, current_lqd, current_spy, latest_date),
        ))

        # 6. Thresholds section
        parts.append(writer.section(
            'Regime Thresholds',
            build_thresholds_table(current_ratio, regime_key),
        ))

        # 7. Footer (also closes .content div)
        parts.append(writer.footer())

        body = '\n'.join(parts)

        # ------------------------------------------------------------------
        # Write dashboard
        # ------------------------------------------------------------------
        writer.write(body, extra_css=EXTRA_CSS, extra_js='')

        # Write CSV
        csv_path = os.path.join(_SCRIPT_DIR, 'hyglqd_data_{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M')))
        csv_row = {
            'date': latest_date,
            'hyg_price': current_hyg,
            'lqd_price': current_lqd,
            'spy_price': current_spy,
            'hyglqd_ratio': current_ratio,
            'percentile': percentile,
            'change_5d': change_5d,
            'regime': regime_label,
        }
        for w in FORWARD_WINDOWS:
            m = expected_returns.get('{}d'.format(w), {})
            csv_row['fwd_{}d_avg'.format(w)] = m.get('avg_return')
            csv_row['fwd_{}d_median'.format(w)] = m.get('median_return')
            csv_row['fwd_{}d_winrate'.format(w)] = m.get('win_rate')
            csv_row['fwd_{}d_n'.format(w)] = m.get('sample_size')
        pd.DataFrame([csv_row]).to_csv(csv_path, index=False, encoding='utf-8')
        print('CSV: {}'.format(csv_path))

        print()
        print('Done.')

    except Exception as e:
        print('\nERROR: {}'.format(e))
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
