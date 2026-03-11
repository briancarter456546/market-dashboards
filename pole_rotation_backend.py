# -*- coding: utf-8 -*-
# =============================================================================
# pole_rotation_backend.py - v1.0
# Last updated: 2026-03-08
# =============================================================================
# v1.0: Initial release - Proven Pole Rotation dashboard
#   - Shows ~20 ETF poles that passed RSI(2) reliability testing (PF >= 1.5)
#   - Heatmap cards with 1W/1M/3M color coding
#   - Sortable performance table with backtest trust scores
#   - Reads fmp_pole_returns_raw.csv (daily log returns per pole)
# =============================================================================

import os
import json
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

from dashboard_writer import DashboardWriter

warnings.filterwarnings('ignore')

# =============================================================================
# PATH SETUP
# =============================================================================

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR   = os.path.normpath(os.path.join(_SCRIPT_DIR, '..', 'perplexity-user-data'))

FMP_RETURNS_PATH = os.path.normpath(os.path.join(
    _DATA_DIR, 'output', 'taxonomy', 'fmp_pole_returns_raw.csv'))
POLE_META_PATH   = os.path.normpath(os.path.join(
    _DATA_DIR, 'output', 'taxonomy', 'fmp_pole_metadata.json'))
_bt_dir = os.path.join(_DATA_DIR, 'output', 'backtest')
_bt_candidates = sorted(
    [f for f in os.listdir(_bt_dir) if f.startswith('rsi2_43pole_comparison_')],
    reverse=True,
) if os.path.isdir(_bt_dir) else []
BACKTEST_PATH = os.path.normpath(os.path.join(_bt_dir, _bt_candidates[0])) if _bt_candidates else ''

# =============================================================================
# PROVEN POLES - PF >= 1.5 from RSI(2) backtest, excluding regression-excluded
# poles (17, 23, 26, 36, 41) and the index (^GSPC / pole 0).
# =============================================================================

EXCLUDED_POLE_IDS = {0, 17, 23, 26, 36, 41}
MIN_PROFIT_FACTOR = 1.5

# =============================================================================
# EXTRA CSS
# =============================================================================

EXTRA_CSS = """
/* Heatmap card grid */
.pole-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
    gap: 14px;
    margin-bottom: 22px;
}

.pole-card {
    background: #fff;
    border: 1px solid #e2e4e8;
    border-radius: 8px;
    padding: 16px 18px;
    border-top: 5px solid #ccc;
    position: relative;
}

.pole-card .pole-name {
    font-size: 0.82em;
    font-weight: 700;
    color: #1a1a2e;
    margin-bottom: 4px;
    line-height: 1.3;
}

.pole-card .pole-members {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72em;
    color: #888;
    margin-bottom: 10px;
}

.pole-card .pole-returns {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 6px;
}

.pole-card .ret-cell {
    text-align: center;
    padding: 6px 4px;
    border-radius: 4px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78em;
    font-weight: 600;
}

.pole-card .ret-label {
    font-size: 0.65em;
    font-weight: 600;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 2px;
}

.pole-card .pf-badge {
    position: absolute;
    top: 10px;
    right: 12px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72em;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 10px;
    background: #f0fdf4;
    color: #16a34a;
    border: 1px solid #bbf7d0;
}

.pole-card .pf-badge.elite {
    background: #fef3c7;
    color: #d97706;
    border-color: #fde68a;
}

/* Beta badge */
.pole-card .beta-badge {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68em;
    color: #888;
    margin-top: 8px;
}

/* Return color cells */
.ret-hot-3  { background: #14532d; color: #fff; }
.ret-hot-2  { background: #166534; color: #fff; }
.ret-hot-1  { background: #22c55e; color: #fff; }
.ret-warm   { background: #dcfce7; color: #166534; }
.ret-flat   { background: #f5f5f5; color: #888; }
.ret-cool   { background: #fee2e2; color: #991b1b; }
.ret-cold-1 { background: #ef4444; color: #fff; }
.ret-cold-2 { background: #991b1b; color: #fff; }
.ret-cold-3 { background: #450a0a; color: #fff; }

/* Performance table */
.perf-table td, .perf-table th {
    padding: 11px 14px;
    border-bottom: 1px solid #f0f0f0;
    font-size: 0.90em;
}
.perf-table th {
    background: #f8f9fb;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-size: 0.78em;
    color: #555;
    border-bottom: 2px solid #e2e4e8;
}
.perf-table td:not(:first-child):not(:nth-child(2)) {
    font-family: 'IBM Plex Mono', monospace;
    text-align: right;
}
.perf-table td:first-child {
    font-weight: 600;
    color: #1a1a2e;
}
.perf-table td:nth-child(2) {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78em;
    color: #888;
}
"""

# =============================================================================
# EXTRA JS - table sorting
# =============================================================================

EXTRA_JS = """
document.querySelectorAll('.sortable-table thead th').forEach(function(th, colIdx) {
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
            var aVal = a.cells[colIdx].getAttribute('data-sort') || a.cells[colIdx].textContent.trim();
            var bVal = b.cells[colIdx].getAttribute('data-sort') || b.cells[colIdx].textContent.trim();
            var aNum = parseFloat(aVal);
            var bNum = parseFloat(bVal);
            if (!isNaN(aNum) && !isNaN(bNum)) {
                return asc ? aNum - bNum : bNum - aNum;
            }
            return asc ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
        });

        rows.forEach(function(row) { tbody.appendChild(row); });
    });
});
"""


# =============================================================================
# HELPERS
# =============================================================================

def load_proven_poles(backtest_path):
    """Load backtest results and filter to proven poles."""
    bt = pd.read_csv(backtest_path, encoding='utf-8')
    # Exclude regression-excluded poles and index
    bt = bt[~bt['pole_id'].isin(EXCLUDED_POLE_IDS)]
    # Filter by profit factor
    proven = bt[bt['profit_factor'] >= MIN_PROFIT_FACTOR].copy()
    proven = proven.sort_values('profit_factor', ascending=False)
    return proven


def load_pole_metadata(meta_path):
    """Load pole metadata JSON."""
    with open(meta_path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    # Convert string keys to int
    return {int(k): v for k, v in raw.items() if k.isdigit()}


def compute_pole_returns(returns_path, proven_pole_ids):
    """Compute cumulative returns over various windows for proven poles."""
    df = pd.read_csv(returns_path, encoding='utf-8', parse_dates=['date'])
    df = df.set_index('date').sort_index()

    # Build pole_id -> column name mapping
    col_map = {}
    for col in df.columns:
        parts = col.split('_', 1)
        if parts[0].isdigit():
            pid = int(parts[0])
            if pid in proven_pole_ids:
                col_map[pid] = col

    # Compute cumulative returns for each window
    today = df.index[-1]
    windows = {
        '1W': 5,
        '1M': 21,
        '3M': 63,
        '6M': 126,
        'YTD': None,  # handled separately
    }

    results = {}
    for pid in proven_pole_ids:
        if pid not in col_map:
            continue
        col = col_map[pid]
        series = df[col].dropna()
        if len(series) < 5:
            continue

        r = {}
        for label, days in windows.items():
            if label == 'YTD':
                # From first trading day of current year
                year_start = pd.Timestamp(today.year, 1, 1)
                ytd_data = series[series.index >= year_start]
                if len(ytd_data) > 0:
                    # Log returns sum to get cumulative
                    r['YTD'] = (np.exp(ytd_data.sum()) - 1) * 100
                else:
                    r['YTD'] = 0.0
            else:
                tail = series.tail(days)
                if len(tail) > 0:
                    r[label] = (np.exp(tail.sum()) - 1) * 100
                else:
                    r[label] = 0.0

        r['last_date'] = series.index[-1].strftime('%Y-%m-%d')
        results[pid] = r

    return results


def return_color_class(val):
    """Map a return percentage to a CSS color class."""
    if val > 15:
        return 'ret-hot-3'
    elif val > 8:
        return 'ret-hot-2'
    elif val > 3:
        return 'ret-hot-1'
    elif val > 0:
        return 'ret-warm'
    elif val > -3:
        return 'ret-cool'
    elif val > -8:
        return 'ret-cold-1'
    elif val > -15:
        return 'ret-cold-2'
    else:
        return 'ret-cold-3'


def border_color_from_1m(val):
    """Get card top-border color based on 1M return."""
    if val > 5:
        return '#16a34a'
    elif val > 0:
        return '#86efac'
    elif val > -5:
        return '#fca5a5'
    else:
        return '#dc2626'


def format_pct(val):
    """Format a percentage value."""
    if abs(val) < 0.05:
        return '0.0%'
    return '{:+.1f}%'.format(val)


def build_heatmap_cards(proven_df, pole_meta, pole_returns):
    """Build HTML grid of pole cards with return heatmaps."""
    cards = []
    for _, row in proven_df.iterrows():
        pid = int(row['pole_id'])
        if pid not in pole_returns:
            continue
        ret = pole_returns[pid]
        meta = pole_meta.get(pid, {})
        members = meta.get('members', [])
        beta = meta.get('avg_spy_beta', 0)
        pf = row['profit_factor']

        # Card border color from 1M return
        border_col = border_color_from_1m(ret.get('1M', 0))

        # PF badge class
        pf_class = 'pf-badge elite' if pf >= 2.0 else 'pf-badge'

        card = []
        card.append('<div class="pole-card" style="border-top-color:{};">'.format(border_col))
        card.append('  <div class="pf-badge {}">PF {:.2f}</div>'.format(
            'elite' if pf >= 2.0 else '', pf))
        card.append('  <div class="pole-name">{}</div>'.format(
            meta.get('pole_label', row.get('pole_name', 'Pole {}'.format(pid)))))
        card.append('  <div class="pole-members">{}</div>'.format(
            ' '.join(members[:5])))

        # 3 return cells: 1W, 1M, 3M
        card.append('  <div class="pole-returns">')
        for label in ['1W', '1M', '3M']:
            val = ret.get(label, 0)
            css = return_color_class(val)
            card.append('    <div class="ret-cell {}">'.format(css))
            card.append('      <div class="ret-label">{}</div>'.format(label))
            card.append('      {}'.format(format_pct(val)))
            card.append('    </div>')
        card.append('  </div>')

        # Beta
        card.append('  <div class="beta-badge">Beta {:.2f} | WR {:.0f}%</div>'.format(
            beta, row['win_rate']))
        card.append('</div>')
        cards.append('\n'.join(card))

    return '<div class="pole-grid">{}</div>'.format('\n'.join(cards))


def build_performance_table(proven_df, pole_meta, pole_returns):
    """Build sortable HTML table with full performance data."""
    rows = []
    for _, row in proven_df.iterrows():
        pid = int(row['pole_id'])
        if pid not in pole_returns:
            continue
        ret = pole_returns[pid]
        meta = pole_meta.get(pid, {})
        members = meta.get('members', [])
        pf = row['profit_factor']
        wr = row['win_rate']

        def ret_td(label):
            val = ret.get(label, 0)
            css = 'pos' if val > 0 else ('neg' if val < 0 else 'muted')
            return '<td class="num {}" data-sort="{:.4f}">{}</td>'.format(
                css, val, format_pct(val))

        tr = []
        tr.append('<tr>')
        tr.append('  <td>{}</td>'.format(
            meta.get('pole_label', row.get('pole_name', 'Pole {}'.format(pid)))))
        tr.append('  <td>{}</td>'.format(' '.join(members[:3])))
        tr.append(ret_td('1W'))
        tr.append(ret_td('1M'))
        tr.append(ret_td('3M'))
        tr.append(ret_td('6M'))
        tr.append(ret_td('YTD'))
        tr.append('  <td class="num" data-sort="{:.2f}" style="font-weight:700;'
                   'color:{};">{:.2f}</td>'.format(
                       pf, '#d97706' if pf >= 2.0 else '#16a34a', pf))
        tr.append('  <td class="num" data-sort="{:.1f}">{:.1f}%</td>'.format(wr, wr))
        tr.append('  <td class="num" data-sort="{:.2f}">{:.2f}</td>'.format(
            meta.get('avg_spy_beta', 0), meta.get('avg_spy_beta', 0)))
        tr.append('</tr>')
        rows.append('\n'.join(tr))

    header = (
        '<thead><tr>'
        '<th title="Factor-Mimicking Portfolio pole name (equal-weighted top 5 ETF constituents)">Pole</th>'
        '<th title="Top ETF constituents in this pole by correlation strength">Top ETFs</th>'
        '<th title="1-week return of the pole portfolio">1W</th>'
        '<th title="1-month return of the pole portfolio">1M</th>'
        '<th title="3-month return of the pole portfolio">3M</th>'
        '<th title="6-month return of the pole portfolio">6M</th>'
        '<th title="Year-to-date return of the pole portfolio">YTD</th>'
        '<th title="RSI(2) mean-reversion backtest Profit Factor. PF >= 1.5 = reliable pole. Higher = stronger mean-reversion edge.">PF</th>'
        '<th title="RSI(2) backtest win rate. % of trades that were profitable.">Win Rate</th>'
        '<th title="6-month regression beta vs SPY. Shows how much this pole moves with the market.">Beta</th>'
        '</tr></thead>'
    )

    return (
        '<table class="perf-table sortable-table">'
        '{}'
        '<tbody>{}</tbody>'
        '</table>'.format(header, '\n'.join(rows))
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    try:
        print('[INFO] Loading backtest data...')
        proven_df = load_proven_poles(BACKTEST_PATH)
        proven_ids = set(proven_df['pole_id'].astype(int).tolist())
        print('[OK] {} proven poles (PF >= {})'.format(len(proven_ids), MIN_PROFIT_FACTOR))

        print('[INFO] Loading pole metadata...')
        pole_meta = load_pole_metadata(POLE_META_PATH)

        print('[INFO] Computing pole returns from FMP data...')
        pole_returns = compute_pole_returns(FMP_RETURNS_PATH, proven_ids)
        print('[OK] Returns computed for {} poles'.format(len(pole_returns)))

        # Count hot/cold
        hot = sum(1 for r in pole_returns.values() if r.get('1M', 0) > 0)
        cold = len(pole_returns) - hot

        # Find latest date
        dates = [r.get('last_date', '') for r in pole_returns.values()]
        latest_date = max(dates) if dates else datetime.now().strftime('%Y-%m-%d')

        # === BUILD HTML ===
        writer = DashboardWriter('pole-rotation', 'Proven Pole Rotation')

        parts = []

        # Stat bar
        parts.append(writer.stat_bar([
            ('Date', latest_date, 'neutral'),
            ('Proven Poles', str(len(proven_ids)), 'neutral'),
            ('Hot (1M+)', str(hot), 'pos'),
            ('Cold (1M-)', str(cold), 'neg'),
            ('Min PF', '{:.1f}'.format(MIN_PROFIT_FACTOR), 'warn'),
        ]))

        # Header
        parts.append(writer.build_header(
            'RSI(2) Backtest Validated | PF >= {:.1f} | {} of 43 poles'.format(
                MIN_PROFIT_FACTOR, len(proven_ids))))

        # Regime banner - summary
        top_pole = proven_df.iloc[0]
        top_meta = pole_meta.get(int(top_pole['pole_id']), {})
        hottest_pid = max(pole_returns, key=lambda p: pole_returns[p].get('1M', -999))
        hottest_meta = pole_meta.get(hottest_pid, {})
        hottest_1m = pole_returns[hottest_pid].get('1M', 0)

        coldest_pid = min(pole_returns, key=lambda p: pole_returns[p].get('1M', 999))
        coldest_meta = pole_meta.get(coldest_pid, {})
        coldest_1m = pole_returns[coldest_pid].get('1M', 0)

        score_html = (
            'Hottest 1M: <b>{}</b> ({:+.1f}%)<br>'
            'Coldest 1M: <b>{}</b> ({:+.1f}%)<br>'
            'Top PF: <b>{}</b> ({:.2f})'.format(
                hottest_meta.get('pole_label', '?'), hottest_1m,
                coldest_meta.get('pole_label', '?'), coldest_1m,
                top_meta.get('pole_label', '?'), top_pole['profit_factor']))

        banner_color = '#22c55e' if hot > cold else '#ef4444'
        parts.append(writer.regime_banner(
            '{} OF {} HOT'.format(hot, len(pole_returns)),
            score_html, color=banner_color))

        # Heatmap cards section
        cards_html = build_heatmap_cards(proven_df, pole_meta, pole_returns)
        parts.append(writer.section(
            'Pole Rotation Heatmap',
            cards_html,
            hint='Sorted by Profit Factor | Cards colored by 1M return'))

        # Performance table
        table_html = build_performance_table(proven_df, pole_meta, pole_returns)
        parts.append(writer.section(
            'Performance Table',
            table_html,
            hint='Click headers to sort'))

        # Methodology note
        method_html = (
            '<div style="font-size:0.88em;color:#666;line-height:1.7;">'
            '<b>Methodology:</b> 43 ETF poles were tested using an RSI(2) mean-reversion '
            'pullback strategy across full history. Poles with Profit Factor >= 1.5 after '
            'excluding regression-excluded poles (US Large Cap Core, US Dollar, Cash, Dow, '
            'S&P Variants) are shown here as "proven." Returns are computed from equal-weighted '
            'FMP (Factor-Mimicking Portfolio) log returns of each pole\'s top 5 ETF constituents.'
            '<br><br>'
            '<b>PF Tiers:</b> '
            '<span style="color:#d97706;font-weight:700;">Gold (PF >= 2.0)</span> = Elite reliability. '
            '<span style="color:#16a34a;font-weight:700;">Green (PF 1.5-2.0)</span> = Proven reliability.'
            '</div>'
        )
        parts.append(writer.section('Methodology', method_html))

        # LLM block
        llm_html = writer.llm_block()
        if llm_html:
            parts.append(llm_html)

        # Footer
        parts.append(writer.footer())

        body = '\n'.join(parts)
        writer.write(body, extra_css=EXTRA_CSS, extra_js=EXTRA_JS)

        print('[OK] Proven Pole Rotation dashboard written successfully')
        return 0

    except Exception as e:
        print('[FAIL] {}'.format(e))
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
