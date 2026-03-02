# -*- coding: utf-8 -*-
# =============================================================================
# stock_secrot_backend.py - v1.0
# Last updated: 2026-02-26
# =============================================================================
# Reads the latest stock_secrot_scores_*.csv from the parent data dir,
# filters to high scorers (total_score >= 7/9), and writes a static HTML
# dashboard via DashboardWriter.
#
# Output: market-dashboards/docs/stock-secrot/index.html
#         market-dashboards/docs/stock-secrot/archive/dashboard_YYYYMMDD.html
#
# Run from any location - uses __file__-relative paths throughout.
# =============================================================================

import os
import glob
import datetime

import pandas as pd

from dashboard_writer import DashboardWriter

# =============================================================================
# PATH SETUP
# =============================================================================

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR   = os.path.normpath(os.path.join(_SCRIPT_DIR, '..', 'perplexity-user-data'))

# =============================================================================
# EXTRA CSS (stock-secrot specific)
# =============================================================================

EXTRA_CSS = """
.score-high {
    background: #dcfce7;
    color: #15803d;
    border: 1px solid #bbf7d0;
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.9em;
    font-weight: 700;
    min-width: 52px;
    text-align: center;
}

.score-mid {
    background: #fffbeb;
    color: #d97706;
    border: 1px solid #fde68a;
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.9em;
    font-weight: 700;
    min-width: 52px;
    text-align: center;
}

.sector-pill {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 4px;
    font-size: 0.82em;
    font-weight: 600;
    background: #f1f5f9;
    color: #475569;
    white-space: nowrap;
}

.metals-tag {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.75em;
    font-weight: 700;
    background: #fef3c7;
    color: #92400e;
    margin-left: 6px;
    vertical-align: middle;
    letter-spacing: 0.04em;
}

thead th.sorted-asc::after  { content: ' \\25B2'; font-size: 0.75em; }
thead th.sorted-desc::after { content: ' \\25BC'; font-size: 0.75em; }

/* ── Mobile responsive ── */
@media (max-width: 768px) {
    .score-high, .score-mid { font-size: 0.8em; padding: 3px 10px; min-width: 40px; }
    .sector-pill { font-size: 0.75em; padding: 2px 8px; }
    .metals-tag { font-size: 0.68em; padding: 2px 6px; }
}
"""

# =============================================================================
# SORT JS (follow exact pattern from spread monitor)
# =============================================================================

SORT_JS = """
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
"""


# =============================================================================
# HELPERS
# =============================================================================

def _fmt_pct(val, decimals=1):
    """Format a float as a percentage string, or '--' if missing."""
    try:
        f = float(val)
        return "{:+.{d}f}%".format(f, d=decimals)
    except (TypeError, ValueError):
        return "--"


def _pct_class(val):
    """Return 'pos' or 'neg' CSS class based on sign of val."""
    try:
        return "pos" if float(val) >= 0 else "neg"
    except (TypeError, ValueError):
        return "muted"


def _score_css(score, max_score=9):
    """Return CSS class for score pill: green for 8-9, amber for 7."""
    try:
        s = int(score)
        if s >= 8:
            return "score-high"
        return "score-mid"
    except (TypeError, ValueError):
        return "score-mid"


def _safe(val, fmt=None):
    """Return formatted value or '--' if NaN/None."""
    try:
        if pd.isna(val):
            return "--"
    except (TypeError, ValueError):
        pass
    if fmt is not None:
        try:
            return fmt.format(float(val))
        except (TypeError, ValueError):
            return "--"
    return str(val)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_latest_csv():
    """
    Find and load the most recently modified stock_secrot_scores_*.csv
    from _DATA_DIR.  Returns (df, csv_path) or raises FileNotFoundError.
    """
    pattern = os.path.join(_DATA_DIR, 'stock_secrot_scores_*.csv')
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(
            "No stock_secrot_scores_*.csv files found in: {}".format(_DATA_DIR)
        )
    latest = max(matches, key=os.path.getmtime)
    df = pd.read_csv(latest)
    return df, latest


# =============================================================================
# HTML BUILDING
# =============================================================================

def build_body_html(df, high_scorers, scan_date, csv_path, writer):
    """
    Build and return the full body HTML string for DashboardWriter.write().

    Parameters
    ----------
    df           : full DataFrame (all stocks from CSV)
    high_scorers : filtered DataFrame (total_score >= 7)
    scan_date    : str date from CSV scan_date column
    csv_path     : str path to the CSV file (for display)
    writer       : DashboardWriter instance
    """
    date_str    = datetime.date.today().strftime("%Y-%m-%d")
    n_total     = len(df)
    n_high      = len(high_scorers)
    n_perfect   = int((high_scorers['total_score'] == 9).sum())
    n_strong    = int((high_scorers['total_score'] == 8).sum())

    # Sort by score desc, then 20d return desc
    sorted_df = high_scorers.sort_values(
        ['total_score', 'ret_20d'], ascending=[False, False]
    ).reset_index(drop=True)

    parts = []

    # --- Stat bar ---
    parts.append(writer.stat_bar([
        ("Generated",       date_str,       "neutral"),
        ("Stocks Scanned",  str(n_total),   "neutral"),
        ("High Scorers",    str(n_high),    "pos"),
        ("9/9 Perfect",     str(n_perfect), "pos"),
        ("Scan Date",       scan_date,      "neutral"),
    ]))

    # --- Page header ---
    parts.append(writer.build_header(
        "Score threshold: 7/9 &nbsp;|&nbsp; Source: {}".format(
            os.path.basename(csv_path)
        )
    ))

    # --- Stocks table ---
    table_rows = []
    for rank, (_, row) in enumerate(sorted_df.iterrows(), start=1):
        score     = int(row['total_score'])
        max_score = int(row.get('max_score', 9))
        score_lbl = "{}/{}".format(score, max_score)
        score_cls = _score_css(score, max_score)

        # Return values
        ret_5d  = row.get('ret_5d')
        ret_20d = row.get('ret_20d')
        ret_60d = row.get('ret_60d')

        ret_5d_str  = _fmt_pct(ret_5d)
        ret_20d_str = _fmt_pct(ret_20d)
        ret_60d_str = _fmt_pct(ret_60d)

        ret_5d_cls  = _pct_class(ret_5d)
        ret_20d_cls = _pct_class(ret_20d)
        ret_60d_cls = _pct_class(ret_60d)

        # RSI / ADX
        rsi_val = _safe(row.get('rsi'), "{:.1f}")
        adx_val = _safe(row.get('adx'), "{:.1f}")

        # RS vs ETF
        rs_val  = _safe(row.get('rs_vs_etf'), "{:+.1f}%")
        rs_cls  = _pct_class(row.get('rs_vs_etf'))

        # Volume increase
        vol_val = _safe(row.get('vol_increase_pct'), "{:+.1f}%")
        vol_cls = _pct_class(row.get('vol_increase_pct'))

        # Metals tag
        is_metals = str(row.get('is_metals', '0')).strip() in ('1', '1.0', 'True', 'true')
        metals_html = '<span class="metals-tag">METALS</span>' if is_metals else ''

        # Symbol and company
        symbol   = str(row.get('symbol', '')).strip()
        company  = str(row.get('company_name', '')).strip()
        sector   = str(row.get('sector', '')).strip()

        table_rows.append(
            "<tr>"
            '<td><input type="checkbox" class="own-cb" data-ticker="{sym}"'
            ' onclick="window._ownToggle(\'{sym}\', this)" title="Mark as owned"></td>'
            '<td data-sort="{rank}" class="num">{rank}</td>'
            '<td data-sort="{sym}"><span class="ticker">{sym}</span>{metals}</td>'
            '<td data-sort="{co}" style="font-size:0.9em;color:#444;">{co}</td>'
            '<td data-sort="{sec}"><span class="sector-pill">{sec}</span></td>'
            '<td data-sort="{score_raw}"><span class="{score_cls}">{score_lbl}</span></td>'
            '<td data-sort="{ret5raw}" class="{ret5cls} num">{ret5s}</td>'
            '<td data-sort="{ret20raw}" class="{ret20cls} num">{ret20s}</td>'
            '<td data-sort="{ret60raw}" class="{ret60cls} num">{ret60s}</td>'
            '<td data-sort="{rsi}" class="num">{rsi}</td>'
            '<td data-sort="{adx}" class="num">{adx}</td>'
            '<td data-sort="{rs_raw}" class="{rs_cls} num">{rs_val}</td>'
            '<td data-sort="{vol_raw}" class="{vol_cls} num">{vol_val}</td>'
            "</tr>".format(
                rank=rank,
                sym=symbol,
                metals=metals_html,
                co=company,
                sec=sector,
                score_raw=score,
                score_cls=score_cls,
                score_lbl=score_lbl,
                ret5raw=ret_5d  if ret_5d  is not None else -9999,
                ret5cls=ret_5d_cls,
                ret5s=ret_5d_str,
                ret20raw=ret_20d if ret_20d is not None else -9999,
                ret20cls=ret_20d_cls,
                ret20s=ret_20d_str,
                ret60raw=ret_60d if ret_60d is not None else -9999,
                ret60cls=ret_60d_cls,
                ret60s=ret_60d_str,
                rsi=rsi_val,
                adx=adx_val,
                rs_raw=row.get('rs_vs_etf', -9999),
                rs_cls=rs_cls,
                rs_val=rs_val,
                vol_raw=row.get('vol_increase_pct', -9999),
                vol_cls=vol_cls,
                vol_val=vol_val,
            )
        )

    table_html = (
        '<table class="sortable-table">'
        "<thead><tr>"
        '<th class="own-th" title="Mark tickers you own">Own</th>'
        '<th title="Overall rank by composite score">Rank</th>'
        '<th title="Ticker symbol">Symbol</th>'
        '<th title="Company name">Company</th>'
        '<th title="GICS sector classification">Sector</th>'
        '<th title="Composite sector rotation score out of 9 pattern tests">Score &#9660;</th>'
        '<th title="5-trading-day return percentage">5D Ret%</th>'
        '<th title="20-trading-day (1 month) return percentage">20D Ret%</th>'
        '<th title="60-trading-day (3 month) return percentage">60D Ret%</th>'
        '<th title="Relative Strength Index, 14-day. Overbought >70, oversold <30">RSI</th>'
        '<th title="Average Directional Index. Strong trend >25, weak <20">ADX</th>'
        '<th title="Relative strength vs sector ETF (positive = outperforming)">RS vs ETF</th>'
        '<th title="Volume increase vs 20-day average (higher = more interest)">Vol Increase%</th>'
        "</tr></thead>"
        "<tbody>" + "\n".join(table_rows) + "</tbody>"
        "</table>"
    )

    section_title = "High-Scoring Stocks ({} of {} &mdash; score >= 7/9)".format(
        n_high, n_total
    )
    parts.append(writer.section(section_title, table_html, hint="Click any column to sort"))

    parts.append(writer.footer())

    return "\n".join(parts)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("STOCK SECTOR ROTATION BACKEND - v1.0")
    print("=" * 60)

    # --- Load data ---
    print("Looking for latest CSV in: {}".format(_DATA_DIR))
    try:
        df, csv_path = load_latest_csv()
    except FileNotFoundError as exc:
        print("[ERROR] {}".format(exc))
        return 1

    print("Loaded: {} (rows: {})".format(os.path.basename(csv_path), len(df)))

    # --- Filter ---
    high_scorers = df[df['total_score'] >= 7].copy()
    print("High scorers (>= 7/9): {}".format(len(high_scorers)))

    # --- Scan date ---
    if 'scan_date' in df.columns and len(df) > 0:
        scan_date = str(df['scan_date'].iloc[0]).strip()
    else:
        scan_date = datetime.date.today().strftime("%Y-%m-%d")

    # --- Build dashboard ---
    writer = DashboardWriter("stock-secrot", "Stock Sector Rotation")
    body = build_body_html(df, high_scorers, scan_date, csv_path, writer)
    writer.write(body, extra_css=EXTRA_CSS, extra_js=SORT_JS)

    # Write CSV (high scorers)
    csv_path = os.path.join(_SCRIPT_DIR, 'stock_secrot_data_{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M')))
    high_scorers.to_csv(csv_path, index=False, encoding='utf-8')
    print("CSV: {}".format(csv_path))

    print("=" * 60)
    print("STATUS: Complete")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
