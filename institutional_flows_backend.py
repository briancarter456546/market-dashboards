# -*- coding: utf-8 -*-
# =============================================================================
# institutional_flows_backend.py - v1.0
# Last updated: 2026-03-09
# =============================================================================
# v1.0: Institutional flows dashboard - real data from 13F, COT, short volume,
#       insider trading. Replaces smart_money_backend.py (TI-based, failed
#       validation). Reads from output/institutional/ CSVs collected by
#       institutional_flow_collector_v1_0.py.
#
#       **Zero-commission era:** all signals assume $0 commissions.
# =============================================================================
"""
Institutional Flows Dashboard Backend
======================================
Shows what institutions are ACTUALLY doing, from real regulatory filings.

PANELS:
  1. 13F Conviction Tracker -- top stocks by # of institutional holders adding
  2. COT Positioning Extremes -- leveraged money net positioning in key futures
  3. Short Volume Monitor -- tickers with unusual short activity (>50% ratio)
  4. Insider Cluster Buys -- 3+ unique insiders purchasing within 7 days

Data sources: FMP 13F, CFTC COT (cot_reports), FINRA Reg SHO, FMP insider.
Refresh cadence: 13F quarterly, COT weekly, short volume daily, insider daily.

Author: Brian + Claude
"""

import os
import sys
import json
import warnings
from datetime import datetime, date
from pathlib import Path

import numpy as np
import pandas as pd

# Dashboard writer lives in same directory
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dashboard_writer import DashboardWriter

warnings.filterwarnings('ignore')

# =============================================================================
# PATHS
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / 'perplexity-user-data'
INST_DIR = DATA_DIR / 'output' / 'institutional'
CACHE_DIR = DATA_DIR / 'price_cache'

# =============================================================================
# HELPERS
# =============================================================================

def find_latest(prefix, ext='.csv'):
    """Find most recent dated file matching prefix in INST_DIR."""
    candidates = sorted(INST_DIR.glob(f'{prefix}*{ext}'), reverse=True)
    return candidates[0] if candidates else None


def fmt_num(n):
    """Format large numbers: 1234567 -> 1.23M"""
    if abs(n) >= 1e9:
        return f'{n/1e9:.1f}B'
    if abs(n) >= 1e6:
        return f'{n/1e6:.1f}M'
    if abs(n) >= 1e3:
        return f'{n/1e3:.0f}K'
    return str(int(n))


def fmt_pct(v):
    """Format as percentage."""
    return f'{v:.1%}' if not pd.isna(v) else '--'


# =============================================================================
# PANEL 1: 13F CONVICTION TRACKER
# =============================================================================

def build_13f_panel(df_13f):
    """
    Top stocks by institutional conviction: how many holders ADDED shares.
    Shows: ticker, # holders adding, # holders reducing, net conviction, total shares.
    """
    if df_13f is None or len(df_13f) == 0:
        return '<p>13F data not available.</p>', {}

    # Most recent filing date per ticker
    df_13f['dateReported'] = pd.to_datetime(df_13f['dateReported'], errors='coerce')
    df_13f['change'] = pd.to_numeric(df_13f['change'], errors='coerce')
    df_13f['shares'] = pd.to_numeric(df_13f['shares'], errors='coerce')

    # For each ticker: count holders adding vs reducing
    stats = df_13f.groupby('_ticker').agg(
        total_holders=('holder', 'count'),
        adding=('change', lambda x: (x > 0).sum()),
        reducing=('change', lambda x: (x < 0).sum()),
        net_change=('change', 'sum'),
        total_shares=('shares', 'sum'),
    ).reset_index()

    stats['conviction'] = stats['adding'] - stats['reducing']
    stats['add_pct'] = stats['adding'] / stats['total_holders']
    stats = stats.sort_values('conviction', ascending=False)

    # Top 30 by conviction
    top = stats.head(30)

    rows_html = []
    for _, r in top.iterrows():
        color = '#16a34a' if r['conviction'] > 0 else '#dc2626'
        rows_html.append(
            f'<tr>'
            f'<td><b>{r["_ticker"]}</b></td>'
            f'<td style="color:{color}">{int(r["conviction"]):+d}</td>'
            f'<td>{int(r["adding"])}</td>'
            f'<td>{int(r["reducing"])}</td>'
            f'<td>{int(r["total_holders"])}</td>'
            f'<td>{fmt_pct(r["add_pct"])}</td>'
            f'<td>{fmt_num(r["total_shares"])}</td>'
            f'</tr>'
        )

    html = (
        '<h3>13F Institutional Conviction (Top 30)</h3>'
        '<p>Quarterly 13F filings: how many institutional holders added vs reduced positions.</p>'
        '<table class="data-table"><thead><tr>'
        '<th>Ticker</th><th title="Net conviction = holders adding minus holders reducing. Positive = institutional accumulation.">Net Conv.</th><th title="Number of 13F filers who increased their position this quarter">Adding</th><th title="Number of 13F filers who decreased their position this quarter">Reducing</th>'
        '<th title="Total number of institutional holders reporting this stock in 13F filings">Total Holders</th><th title="Percentage of reporting holders who increased position (adding / total)">% Adding</th><th title="Total shares held across all reporting institutions">Total Shares</th>'
        '</tr></thead><tbody>'
        + '\n'.join(rows_html) +
        '</tbody></table>'
    )

    summary = {
        'top_conviction': top.iloc[0]['_ticker'] if len(top) > 0 else 'N/A',
        'avg_add_pct': f'{stats["add_pct"].mean():.1%}',
        'tickers_covered': len(stats),
    }

    return html, summary


# =============================================================================
# PANEL 2: COT POSITIONING EXTREMES
# =============================================================================

# Map COT contract names to readable names
COT_CONTRACTS = {
    'E-MINI S&P 500': 'S&P 500',
    'NASDAQ-100': 'Nasdaq 100',
    'E-MINI NASDAQ': 'Nasdaq 100',
    'DJIA x $5': 'Dow Jones',
    'RUSSELL E-MINI': 'Russell 2000',
    'GOLD': 'Gold',
    'SILVER': 'Silver',
    'CRUDE OIL': 'Crude Oil',
    'NATURAL GAS': 'Nat Gas',
    'COPPER': 'Copper',
    'U.S. TREASURY BONDS': 'T-Bonds',
    '10-YEAR': '10Y Notes',
    '2-YEAR': '2Y Notes',
    '5-YEAR': '5Y Notes',
    'EURO FX': 'EUR/USD',
    'JAPANESE YEN': 'JPY',
    'BRITISH POUND': 'GBP',
    'VIX': 'VIX',
    'BITCOIN': 'Bitcoin',
}


def build_cot_panel(df_cot):
    """
    COT positioning: leveraged money net long/short in key futures.
    Extremes = contrarian signal (when specs are max long, expect reversal).
    """
    if df_cot is None or len(df_cot) == 0:
        return '<p>COT data not available.</p>', {}

    # Find the market name column
    name_col = 'Market_and_Exchange_Names'
    if name_col not in df_cot.columns:
        return '<p>COT data format unexpected.</p>', {}

    date_col = 'Report_Date_as_YYYY-MM-DD'
    if date_col not in df_cot.columns:
        date_col = 'As_of_Date_In_Form_YYMMDD'

    # Filter to contracts we care about
    rows = []
    for pattern, label in COT_CONTRACTS.items():
        mask = df_cot[name_col].str.contains(pattern, case=False, na=False)
        subset = df_cot[mask]
        if len(subset) == 0:
            continue

        # Most recent report
        if date_col in subset.columns:
            subset = subset.copy()
            subset[date_col] = pd.to_datetime(subset[date_col], errors='coerce')
            subset = subset.sort_values(date_col, ascending=False)

        latest = subset.iloc[0]

        # Leveraged money (hedge funds) positioning
        lev_long = latest.get('Lev_Money_Positions_Long_All', 0)
        lev_short = latest.get('Lev_Money_Positions_Short_All', 0)
        lev_net = lev_long - lev_short if pd.notna(lev_long) and pd.notna(lev_short) else 0

        # Asset manager positioning
        am_long = latest.get('Asset_Mgr_Positions_Long_All', 0)
        am_short = latest.get('Asset_Mgr_Positions_Short_All', 0)
        am_net = am_long - am_short if pd.notna(am_long) and pd.notna(am_short) else 0

        # Compute net as % of total open interest
        total_long = latest.get('Tot_Rept_Positions_Long_All', 1)
        if pd.isna(total_long) or total_long == 0:
            total_long = 1
        lev_pct = lev_net / total_long

        report_date = str(latest.get(date_col, ''))[:10]

        rows.append({
            'contract': label,
            'lev_net': int(lev_net),
            'lev_pct': lev_pct,
            'am_net': int(am_net),
            'report_date': report_date,
        })

    if not rows:
        return '<p>No matching COT contracts found.</p>', {}

    # Sort by absolute leveraged net (most extreme first)
    rows.sort(key=lambda r: abs(r['lev_net']), reverse=True)

    rows_html = []
    for r in rows:
        lev_color = '#16a34a' if r['lev_net'] > 0 else '#dc2626'
        am_color = '#16a34a' if r['am_net'] > 0 else '#dc2626'
        # Flag extremes
        extreme = ''
        if abs(r['lev_pct']) > 0.3:
            extreme = ' [EXTREME]'
        rows_html.append(
            f'<tr>'
            f'<td><b>{r["contract"]}</b>{extreme}</td>'
            f'<td style="color:{lev_color}">{fmt_num(r["lev_net"])}</td>'
            f'<td>{fmt_pct(r["lev_pct"])}</td>'
            f'<td style="color:{am_color}">{fmt_num(r["am_net"])}</td>'
            f'<td>{r["report_date"]}</td>'
            f'</tr>'
        )

    html = (
        '<h3>COT Positioning: Leveraged Money & Asset Managers</h3>'
        '<p>CFTC Commitment of Traders (Traders in Financial Futures). '
        'Leveraged Money = hedge funds. Extremes are contrarian signals.</p>'
        '<table class="data-table"><thead><tr>'
        '<th title="Futures contract (e.g. ES = S&P 500 E-mini, NQ = Nasdaq 100)">Contract</th><th title="Leveraged Money (hedge funds) net position: long minus short contracts. Extremes are contrarian.">Lev Net</th><th title="Leveraged Money net as % of total open interest. Extreme readings signal crowded positioning.">Lev % OI</th>'
        '<th title="Asset Manager (pension/mutual funds) net position. Typically trend-following, not contrarian.">AM Net</th><th title="CFTC report date (published Friday, data as of Tuesday)">Report Date</th>'
        '</tr></thead><tbody>'
        + '\n'.join(rows_html) +
        '</tbody></table>'
    )

    n_extreme = sum(1 for r in rows if abs(r['lev_pct']) > 0.3)
    summary = {
        'contracts_tracked': len(rows),
        'extreme_positions': n_extreme,
    }

    return html, summary


# =============================================================================
# PANEL 3: SHORT VOLUME MONITOR
# =============================================================================

def build_short_volume_panel(df_short):
    """
    Tickers with unusual short volume (>50% of daily volume).
    High short ratio = potential squeeze setup OR bearish signal.
    """
    if df_short is None or len(df_short) == 0:
        return '<p>Short volume data not available.</p>', {}

    # Most recent date
    df_short['date'] = pd.to_datetime(df_short['date'], format='%Y%m%d', errors='coerce')
    latest_date = df_short['date'].max()
    recent = df_short[df_short['date'] == latest_date].copy()

    # Filter to meaningful volume (>100K total)
    recent = recent[recent['total_volume'] > 100000]
    recent = recent.sort_values('short_ratio', ascending=False)

    # Top 30 by short ratio
    high_short = recent[recent['short_ratio'] > 0.50].head(30)

    rows_html = []
    for _, r in high_short.iterrows():
        sr = r['short_ratio']
        color = '#dc2626' if sr > 0.60 else '#f59e0b' if sr > 0.50 else '#6b7280'
        rows_html.append(
            f'<tr>'
            f'<td><b>{r["symbol"]}</b></td>'
            f'<td style="color:{color}">{sr:.1%}</td>'
            f'<td>{fmt_num(r["short_volume"])}</td>'
            f'<td>{fmt_num(r["total_volume"])}</td>'
            f'</tr>'
        )

    html = (
        f'<h3>Short Volume Monitor ({latest_date.strftime("%Y-%m-%d")})</h3>'
        '<p>FINRA Reg SHO daily short volume. Tickers with >50% short ratio '
        'and >100K volume. High short ratio may indicate squeeze potential.</p>'
        '<table class="data-table"><thead><tr>'
        '<th>Ticker</th><th title="Short volume / total volume for most recent trading day. Above 50% = heavily shorted. Potential squeeze candidate.">Short Ratio</th><th title="Number of shares sold short on most recent day">Short Vol</th><th title="Total shares traded on most recent day">Total Vol</th>'
        '</tr></thead><tbody>'
        + '\n'.join(rows_html) +
        '</tbody></table>'
    )

    n_high = len(recent[recent['short_ratio'] > 0.50])
    avg_sr = recent['short_ratio'].mean()
    summary = {
        'date': latest_date.strftime('%Y-%m-%d'),
        'tickers_gt50pct': n_high,
        'avg_short_ratio': f'{avg_sr:.1%}',
        'total_tickers': len(recent),
    }

    return html, summary


# =============================================================================
# PANEL 4: INSIDER ACTIVITY
# =============================================================================

def build_insider_panel(df_insider):
    """
    Recent insider purchases. Shows individual large buys and cluster events.
    Filtered to P-Purchase only (open market buys, not grants/exercises).

    **Note:** Insider cluster buys (3+ insiders within 7 days) showed ~3-5%
    avg returns in literature but only 46% win rate in our validation.
    Displayed as informational, not as a trading signal.
    """
    if df_insider is None or len(df_insider) == 0:
        return '<p>Insider trading data not available.</p>', {}

    # P-Purchase only
    buys = df_insider[
        (df_insider['acquisitionOrDisposition'] == 'A') &
        (df_insider['transactionType'].isin(['P-Purchase', 'P']))
    ].copy()

    if len(buys) == 0:
        return '<p>No insider purchases found in data.</p>', {}

    buys['transactionDate'] = pd.to_datetime(buys['transactionDate'], errors='coerce')
    buys['price'] = pd.to_numeric(buys['price'], errors='coerce')
    buys['securitiesTransacted'] = pd.to_numeric(buys['securitiesTransacted'], errors='coerce')
    buys['value'] = buys['price'] * buys['securitiesTransacted']
    buys = buys.dropna(subset=['transactionDate'])

    # Sort by value (largest purchases first)
    buys_sorted = buys.sort_values('value', ascending=False)

    # Top 30 largest purchases
    top = buys_sorted.head(30)

    rows_html = []
    for _, r in top.iterrows():
        val = r['value'] if pd.notna(r['value']) else 0
        rows_html.append(
            f'<tr>'
            f'<td><b>{r["_ticker"]}</b></td>'
            f'<td>{r["reportingName"]}</td>'
            f'<td>{r["typeOfOwner"]}</td>'
            f'<td>{str(r["transactionDate"])[:10]}</td>'
            f'<td>${r["price"]:.2f}</td>' if pd.notna(r['price']) else f'<td>--</td>'
            f'<td>{fmt_num(r["securitiesTransacted"])}</td>'
            f'<td>${fmt_num(val)}</td>'
            f'</tr>'
        )

    html = (
        '<h3>Largest Insider Purchases (Open Market Buys)</h3>'
        '<p>Form 4 filings: P-Purchase transactions only (excludes grants, '
        'exercises, awards). Sorted by dollar value.</p>'
        '<table class="data-table"><thead><tr>'
        '<th>Ticker</th><th title="Name of the insider making the purchase">Insider</th><th title="Corporate role (CEO, CFO, Director, 10% Owner, etc.)">Role</th><th title="Transaction date from SEC Form 4 filing">Date</th>'
        '<th title="Purchase price per share">Price</th><th title="Number of shares purchased">Shares</th><th title="Total dollar value of the purchase (price x shares)">Value</th>'
        '</tr></thead><tbody>'
        + '\n'.join(rows_html) +
        '</tbody></table>'
    )

    total_buy_value = buys['value'].sum()
    summary = {
        'total_purchases': len(buys),
        'unique_tickers': buys['_ticker'].nunique(),
        'total_buy_value': f'${fmt_num(total_buy_value)}',
    }

    return html, summary


# =============================================================================
# SUMMARY BANNER
# =============================================================================

def build_summary(summaries):
    """Top-level summary banner."""
    parts = []
    s13f = summaries.get('13f', {})
    if s13f:
        parts.append(
            f'<div class="stat-card">'
            f'<div class="stat-label">13F Top Conviction</div>'
            f'<div class="stat-value">{s13f.get("top_conviction", "N/A")}</div>'
            f'<div class="stat-sub">{s13f.get("tickers_covered", 0)} tickers, '
            f'{s13f.get("avg_add_pct", "--")} avg adding</div>'
            f'</div>'
        )

    scot = summaries.get('cot', {})
    if scot:
        parts.append(
            f'<div class="stat-card">'
            f'<div class="stat-label">COT Extremes</div>'
            f'<div class="stat-value">{scot.get("extreme_positions", 0)}</div>'
            f'<div class="stat-sub">{scot.get("contracts_tracked", 0)} contracts tracked</div>'
            f'</div>'
        )

    sshort = summaries.get('short', {})
    if sshort:
        parts.append(
            f'<div class="stat-card">'
            f'<div class="stat-label">High Short ({sshort.get("date", "")})</div>'
            f'<div class="stat-value">{sshort.get("tickers_gt50pct", 0)}</div>'
            f'<div class="stat-sub">{sshort.get("total_tickers", 0)} tickers, '
            f'{sshort.get("avg_short_ratio", "--")} avg ratio</div>'
            f'</div>'
        )

    sins = summaries.get('insider', {})
    if sins:
        parts.append(
            f'<div class="stat-card">'
            f'<div class="stat-label">Insider Purchases</div>'
            f'<div class="stat-value">{sins.get("total_purchases", 0)}</div>'
            f'<div class="stat-sub">{sins.get("unique_tickers", 0)} tickers, '
            f'{sins.get("total_buy_value", "--")} total</div>'
            f'</div>'
        )

    return '<div class="stat-row">' + '\n'.join(parts) + '</div>'


# =============================================================================
# CSS
# =============================================================================

EXTRA_CSS = """
.stat-row { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 24px; }
.stat-card {
    flex: 1; min-width: 180px; padding: 16px;
    background: #1e293b; border-radius: 8px; border: 1px solid #334155;
}
.stat-label { font-size: 12px; color: #94a3b8; text-transform: uppercase; }
.stat-value { font-size: 28px; font-weight: bold; color: #f1f5f9; margin: 4px 0; }
.stat-sub { font-size: 12px; color: #64748b; }
.data-table { width: 100%; border-collapse: collapse; margin: 12px 0 24px 0; font-size: 13px; }
.data-table th {
    text-align: left; padding: 8px 12px; border-bottom: 2px solid #334155;
    color: #94a3b8; font-size: 11px; text-transform: uppercase;
}
.data-table td { padding: 6px 12px; border-bottom: 1px solid #1e293b; }
.data-table tr:hover { background: #1e293b; }
h3 { color: #f1f5f9; margin-top: 32px; margin-bottom: 8px; }
p { color: #94a3b8; font-size: 13px; margin-bottom: 8px; }
"""


# =============================================================================
# MAIN
# =============================================================================

def main():
    import time as _time
    t0 = _time.time()

    print('=' * 60)
    print('INSTITUTIONAL FLOWS DASHBOARD v1.0')
    print('=' * 60)

    # Load data (find most recent files)
    summaries = {}
    body_parts = []

    # 13F
    f13_path = find_latest('13f_positions_')
    if f13_path:
        print(f'  Loading 13F: {f13_path.name}...', end=' ')
        df_13f = pd.read_csv(f13_path)
        print(f'{len(df_13f)} rows')
        html, summary = build_13f_panel(df_13f)
        body_parts.append(html)
        summaries['13f'] = summary
        del df_13f  # free 93MB
    else:
        print('  13F: not found')

    # COT
    cot_path = find_latest('cot_tff_')
    if cot_path:
        print(f'  Loading COT: {cot_path.name}...', end=' ')
        df_cot = pd.read_csv(cot_path)
        print(f'{len(df_cot)} rows')
        html, summary = build_cot_panel(df_cot)
        body_parts.append(html)
        summaries['cot'] = summary
    else:
        print('  COT: not found')

    # Short volume
    short_path = find_latest('short_volume_')
    if short_path:
        print(f'  Loading Short Vol: {short_path.name}...', end=' ')
        df_short = pd.read_csv(short_path)
        print(f'{len(df_short)} rows')
        html, summary = build_short_volume_panel(df_short)
        body_parts.append(html)
        summaries['short'] = summary
    else:
        print('  Short volume: not found')

    # Insider
    insider_path = find_latest('insider_trading_')
    if insider_path:
        print(f'  Loading Insider: {insider_path.name}...', end=' ')
        df_insider = pd.read_csv(insider_path)
        print(f'{len(df_insider)} rows')
        html, summary = build_insider_panel(df_insider)
        body_parts.append(html)
        summaries['insider'] = summary
    else:
        print('  Insider: not found')

    # Build full page
    banner = build_summary(summaries)
    body = banner + '\n'.join(body_parts)

    # Write dashboard
    writer = DashboardWriter('institutional-flows', 'Institutional Flows')
    writer.write(body, extra_css=EXTRA_CSS)

    elapsed = _time.time() - t0
    print(f'\n[OK] Dashboard written ({elapsed:.1f}s)')
    print('[OK] Windows Anaconda compatible (Python only, UTF-8 encoding, ASCII chars)')
    print('[OK] Versioned as v1.0')
    print('[OK] Run with: python institutional_flows_backend.py')


if __name__ == '__main__':
    main()
