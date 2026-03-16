# -*- coding: utf-8 -*-
# =============================================================================
# rsi2_dashboard_backend.py - v1.4
# Last updated: 2026-03-15
# =============================================================================
# RSI(2) Mean-Reversion Scanner Dashboard Backend
# Self-contained: simulates strategy from price_cache (no external state files).
# Quality scoring suppresses CAUTION entries (above SMA200 only).
#
# v1.4: VIX > 35 kill switch -- blocks ALL MR entries (PF < 0.7 at VIX > 35)
# v1.3: Self-contained simulation + quality scoring + CAUTION suppression
# v1.0: Read JSON state files (broken on droplet -- files never synced)
#
# Run: python rsi2_dashboard_backend.py
# Output: docs/rsi2-scanner/index.html (+ dated archive)
# =============================================================================

import os
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

from dashboard_writer import DashboardWriter, DASHBOARD_DESCRIPTIONS

# =============================================================================
# PATH SETUP
# =============================================================================

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.normpath(os.path.join(_SCRIPT_DIR, '..', 'perplexity-user-data'))
CACHE_DIR = os.path.normpath(os.path.join(_DATA_DIR, 'price_cache'))

# Strategy parameters (must match trading engine)
SMA_LONG = 200
SMA_SHORT = 5
RSI_PERIOD = 2
RSI_THRESHOLD = 10

# Quality scoring
VIX_ELEVATED = 22
VIX_KILL_SWITCH = 35  # All MR PF < 0.7 above this level (confirmed across all param combos)
LOOKBACK_DAYS = 120  # How far back to simulate for positions/trades

# =============================================================================
# EXTRA CSS
# =============================================================================

EXTRA_CSS = """
.signal-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 0.78em;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
.signal-entry { background: #dbeafe; color: #1e40af; }
.signal-exit  { background: #fef3c7; color: #92400e; }
.signal-skip  { background: #fee2e2; color: #991b1b; }
.signal-none  { background: #f3f4f6; color: #6b7280; }

.grade-strong  { color: #15803d; font-weight: 700; }
.grade-moderate { color: #ca8a04; font-weight: 600; }
.grade-caution  { color: #dc2626; font-weight: 600; }

.perf-card {
    display: inline-block;
    background: #f8fafc;
    border: 1px solid #e2e4e8;
    border-radius: 8px;
    padding: 12px 18px;
    margin: 4px;
    text-align: center;
    min-width: 110px;
}
.perf-card .value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.5em;
    font-weight: 600;
}
.perf-card .label {
    font-size: 0.78em;
    color: #888;
    margin-top: 2px;
}

.skip-detail {
    background: #fef2f2;
    border: 1px solid #fecaca;
    border-radius: 6px;
    padding: 8px 12px;
    margin: 4px 0;
    font-size: 0.82em;
    color: #991b1b;
}
"""

# =============================================================================
# DATA LOADING
# =============================================================================

def _load_json(path, default=None):
    """Safely load a JSON file."""
    if default is None:
        default = []
    if not os.path.exists(path):
        return default
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return default


def load_from_cache(ticker):
    """Load a single ticker from price_cache pkl files."""
    cache_file = os.path.join(CACHE_DIR, '{}.pkl'.format(ticker))
    if not os.path.exists(cache_file):
        return None
    try:
        with open(cache_file, 'rb') as f:
            df = pickle.load(f)
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date').sort_index()
            else:
                return None
        else:
            df = df.sort_index()
        required = ['open', 'high', 'low', 'close', 'volume', 'adjClose']
        if not all(col in df.columns for col in required):
            return None
        return df
    except Exception:
        return None


def load_vix():
    """Load latest VIX close from price_cache."""
    vix_file = os.path.join(CACHE_DIR, '^VIX.pkl')
    if not os.path.exists(vix_file):
        return None
    try:
        with open(vix_file, 'rb') as f:
            df = pickle.load(f)
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date').sort_index()
            else:
                return None
        return float(df['close'].iloc[-1])
    except Exception:
        return None


def compute_rsi(close, period=2):
    """Wilder RSI with exponential smoothing."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def compute_indicators(df):
    """Add all SMA levels and RSI(2) to dataframe."""
    df = df.copy()
    df['sma5'] = df['close'].rolling(window=SMA_SHORT, min_periods=SMA_SHORT).mean()
    df['sma50'] = df['close'].rolling(window=50, min_periods=50).mean()
    df['sma100'] = df['close'].rolling(window=100, min_periods=100).mean()
    df['sma200'] = df['close'].rolling(window=SMA_LONG, min_periods=SMA_LONG).mean()
    df['rsi2'] = compute_rsi(df['close'], period=RSI_PERIOD)
    return df


def score_entry_quality(close, sma50, sma100, sma200, vix_val):
    """
    Score entry quality based on SMA structure + VIX.
    Returns (grade, label, detail, css_class).
    """
    above_50 = close > sma50 if sma50 is not None and not np.isnan(sma50) else None
    above_100 = close > sma100 if sma100 is not None and not np.isnan(sma100) else None

    if above_50 and above_100:
        grade = 'STRONG'
        label = 'Strong trend'
        detail = 'Above SMA50/100/200 (PF 5.31, WR 72.7%)'
        css = 'grade-strong'
    elif above_100:
        grade = 'MODERATE'
        label = 'Moderate trend'
        detail = 'Above SMA100/200 (PF 3.97, WR 72.0%)'
        css = 'grade-moderate'
    else:
        grade = 'CAUTION'
        label = 'Weak trend'
        detail = 'Above SMA200 only (PF 3.49, WR 70.9%) -- all major drawdowns here'
        css = 'grade-caution'

    if vix_val is not None and vix_val > VIX_ELEVATED:
        if grade == 'STRONG':
            detail += ' | High VIX + strong = best combo'
        elif grade == 'CAUTION':
            detail += ' | High VIX + weak = worst combo'

    return grade, label, detail, css


# =============================================================================
# STRATEGY SIMULATION
# =============================================================================

def simulate_strategy(watchlist, vix_val):
    """
    Walk forward through price_cache to find current positions, recent trades,
    and today's signals. Self-contained -- no external state files needed.

    Returns (positions, trade_log, today_entries, today_exits, today_skipped, vix_latest).
    """
    positions = {}    # ticker -> {entry_date, entry_price, entry_rsi2, quality_grade, quality_detail}
    trade_log = []    # completed trades
    today_entries = []
    today_exits = []
    today_skipped = []
    vix_killed = vix_val is not None and vix_val > VIX_KILL_SWITCH

    for ticker in watchlist:
        df = load_from_cache(ticker)
        if df is None or len(df) < SMA_LONG + 10:
            continue

        df = compute_indicators(df)

        # Only simulate recent window for performance
        sim_start = max(0, len(df) - LOOKBACK_DAYS)
        sim_df = df.iloc[sim_start:]

        in_position = False
        entry_info = {}

        for i in range(len(sim_df)):
            row = sim_df.iloc[i]
            date_str = sim_df.index[i].strftime('%Y-%m-%d')
            close = float(row['close'])
            sma200 = float(row['sma200']) if not np.isnan(row['sma200']) else None
            sma100 = float(row['sma100']) if not np.isnan(row['sma100']) else None
            sma50 = float(row['sma50']) if not np.isnan(row['sma50']) else None
            sma5 = float(row['sma5']) if not np.isnan(row['sma5']) else None
            rsi2 = float(row['rsi2']) if not np.isnan(row['rsi2']) else None

            if sma200 is None or rsi2 is None:
                continue

            # Exit check
            if in_position and sma5 is not None and close > sma5:
                ret = (close / entry_info['entry_price'] - 1) * 100
                entry_dt = datetime.strptime(entry_info['entry_date'], '%Y-%m-%d')
                days_held = (datetime.strptime(date_str, '%Y-%m-%d') - entry_dt).days

                trade = {
                    'ticker': ticker,
                    'entry_date': entry_info['entry_date'],
                    'entry_price': entry_info['entry_price'],
                    'exit_date': date_str,
                    'exit_price': close,
                    'return_pct': round(ret, 2),
                    'days_held': max(days_held, 1),
                }

                is_last_bar = (i == len(sim_df) - 1)
                if is_last_bar:
                    today_exits.append(trade)
                else:
                    trade_log.append(trade)

                in_position = False
                entry_info = {}
                continue

            # Entry check
            if not in_position and close > sma200 and sma5 is not None and close < sma5 and rsi2 < RSI_THRESHOLD:
                grade, label, detail, css = score_entry_quality(close, sma50, sma100, sma200, vix_val)
                sma200_dist = (close / sma200 - 1) * 100

                is_last_bar = (i == len(sim_df) - 1)
                if is_last_bar:
                    sig = {
                        'ticker': ticker,
                        'rsi2': round(rsi2, 1),
                        'close': close,
                        'sma200': sma200,
                        'sma200_dist': round(sma200_dist, 1),
                        'quality_grade': grade,
                        'quality_label': label,
                        'quality_detail': detail,
                        'quality_css': css,
                    }
                    if vix_killed:
                        sig['quality_detail'] = 'VIX > {} KILL SWITCH -- all MR signals blocked'.format(VIX_KILL_SWITCH)
                        sig['quality_css'] = 'grade-caution'
                        today_skipped.append(sig)
                    elif grade == 'CAUTION':
                        today_skipped.append(sig)
                    else:
                        today_entries.append(sig)
                else:
                    # Historical entry -- take it regardless of grade for accurate sim
                    in_position = True
                    entry_info = {
                        'ticker': ticker,
                        'entry_date': date_str,
                        'entry_price': close,
                        'entry_rsi2': round(rsi2, 1),
                        'quality_grade': grade,
                        'quality_detail': detail,
                    }

        # If still in position at end, record as open
        if in_position:
            positions[ticker] = entry_info

    today_entries.sort(key=lambda x: x['rsi2'])
    today_skipped.sort(key=lambda x: x['rsi2'])
    trade_log.sort(key=lambda x: x['exit_date'])

    return positions, trade_log, today_entries, today_exits, today_skipped, vix_val


# =============================================================================
# HTML BUILDERS
# =============================================================================

def _fmt_pct(val, sign=True):
    """Format a percentage value."""
    if val is None or np.isnan(val):
        return 'N/A'
    prefix = '+' if sign and val >= 0 else ''
    return '{}{:.2f}%'.format(prefix, val)


def _color_class(val):
    """Return CSS class for positive/negative."""
    if val is None:
        return 'neutral'
    return 'pos' if val >= 0 else 'neg'


def build_stat_bar(positions, trade_log, today_entries, today_exits, today_skipped, vix_val):
    """Build the top stat bar."""
    n_open = len(positions)
    n_signals = len(today_entries) + len(today_exits)
    n_skipped = len(today_skipped)

    cumul_pnl = sum(t.get('return_pct', 0) for t in trade_log)
    wins = sum(1 for t in trade_log if t.get('return_pct', 0) > 0)
    win_rate = (wins / len(trade_log) * 100) if trade_log else 0

    stats = [
        ('Open Positions', str(n_open), 'neutral'),
        ('Signals', str(n_signals), 'pos' if n_signals > 0 else 'neutral'),
        ('Suppressed', str(n_skipped), 'neg' if n_skipped > 0 else 'neutral'),
        ('Win Rate', '{:.0f}%'.format(win_rate) if trade_log else '--', 'pos' if win_rate > 55 else 'neutral'),
        ('VIX', '{:.1f}'.format(vix_val) if vix_val else '--', 'neg' if vix_val and vix_val > VIX_ELEVATED else 'neutral'),
    ]
    return stats


def build_signals_section(entries, exits, skipped):
    """Build today's signals HTML including suppressed entries."""
    if not entries and not exits and not skipped:
        return '<p style="color:#888;padding:8px 0;">No signals today -- quiet day.</p>'

    rows = []
    for e in entries:
        grade = e.get('quality_grade', '?')
        css = e.get('quality_css', '')
        rows.append(
            '<tr>'
            '<td><span class="signal-badge signal-entry">BUY</span></td>'
            '<td style="font-weight:600;">{ticker}</td>'
            '<td style="font-family:\'IBM Plex Mono\',monospace;">{rsi2:.1f}</td>'
            '<td style="font-family:\'IBM Plex Mono\',monospace;">${close:.2f}</td>'
            '<td style="font-family:\'IBM Plex Mono\',monospace;">${sma200:.2f} ({sma200_dist:+.1f}%)</td>'
            '<td><span class="{css}">{grade}</span></td>'
            '</tr>'.format(css=css, grade=grade, **e)
        )
    for e in exits:
        rows.append(
            '<tr>'
            '<td><span class="signal-badge signal-exit">SELL</span></td>'
            '<td style="font-weight:600;">{}</td>'
            '<td colspan="4" style="font-family:\'IBM Plex Mono\',monospace;">Return: {}</td>'
            '</tr>'.format(e['ticker'], _fmt_pct(e.get('return_pct', 0)))
        )
    for e in skipped:
        rows.append(
            '<tr style="opacity:0.7;">'
            '<td><span class="signal-badge signal-skip">SKIP</span></td>'
            '<td style="font-weight:600;">{ticker}</td>'
            '<td style="font-family:\'IBM Plex Mono\',monospace;">{rsi2:.1f}</td>'
            '<td style="font-family:\'IBM Plex Mono\',monospace;">${close:.2f}</td>'
            '<td style="font-family:\'IBM Plex Mono\',monospace;">${sma200:.2f} ({sma200_dist:+.1f}%)</td>'
            '<td><span class="grade-caution">CAUTION</span></td>'
            '</tr>'.format(**e)
        )

    html = (
        '<table class="data-table" style="width:100%;">'
        '<thead><tr>'
        '<th title="BUY = RSI(2) &lt; 10 while above SMA200. SELL = close above SMA5. SKIP = CAUTION grade suppressed.">Signal</th>'
        '<th>Ticker</th>'
        '<th title="2-period RSI. Below 10 = deeply oversold.">RSI(2)</th>'
        '<th title="Latest closing price">Close</th>'
        '<th title="200-day SMA distance">SMA200</th>'
        '<th title="Entry quality: STRONG (above SMA50/100/200), MODERATE (above SMA100/200), CAUTION (SMA200 only, suppressed)">Grade</th>'
        '</tr></thead>'
        '<tbody>{}</tbody></table>'.format('\n'.join(rows))
    )

    # Add skip detail box if any suppressed
    if skipped:
        detail_tickers = ', '.join(e['ticker'] for e in skipped)
        html += (
            '<div class="skip-detail">'
            '<strong>Suppressed:</strong> {} -- above SMA200 only (weak trend structure). '
            'All major drawdowns occur in this bucket. PF 3.49 vs 5.31 for STRONG entries.'
            '</div>'.format(detail_tickers)
        )

    return html


def build_positions_section(positions):
    """Build open positions table."""
    if not positions:
        return '<p style="color:#888;padding:8px 0;">No open positions.</p>'

    rows = []
    for ticker, pos in sorted(positions.items()):
        df = load_from_cache(ticker)
        current_price = float(df['close'].iloc[-1]) if df is not None else None
        entry_price = pos.get('entry_price', 0)

        if current_price and entry_price > 0:
            pnl = (current_price / entry_price - 1) * 100
        else:
            pnl = None
            current_price = None

        entry_dt = pos.get('entry_date', '')
        days_held = 0
        if entry_dt:
            try:
                days_held = (datetime.now() - datetime.strptime(entry_dt, '%Y-%m-%d')).days
            except ValueError:
                pass

        grade = pos.get('quality_grade', '?')
        pnl_str = _fmt_pct(pnl) if pnl is not None else 'N/A'
        pnl_class = _color_class(pnl) if pnl is not None else ''
        cur_str = '${:.2f}'.format(current_price) if current_price else 'N/A'

        rows.append(
            '<tr>'
            '<td style="font-weight:600;">{}</td>'
            '<td>{}</td>'
            '<td style="font-family:\'IBM Plex Mono\',monospace;">${:.2f}</td>'
            '<td style="font-family:\'IBM Plex Mono\',monospace;">{}</td>'
            '<td class="{}" style="font-family:\'IBM Plex Mono\',monospace;">{}</td>'
            '<td>{}</td>'
            '<td>{}</td>'
            '</tr>'.format(
                ticker, entry_dt, entry_price, cur_str,
                pnl_class, pnl_str, days_held, grade
            )
        )

    return (
        '<table class="data-table" style="width:100%;">'
        '<thead><tr><th>Ticker</th><th>Entry Date</th><th>Entry $</th>'
        '<th>Current</th><th>P/L</th><th>Days</th><th>Grade</th></tr></thead>'
        '<tbody>{}</tbody></table>'.format('\n'.join(rows))
    )


def build_trade_log_section(trade_log, limit=10):
    """Build recent trades table."""
    if not trade_log:
        return '<p style="color:#888;padding:8px 0;">No completed trades yet.</p>'

    recent = trade_log[-limit:]
    rows = []
    for t in reversed(recent):
        ret = t.get('return_pct', 0)
        ret_class = _color_class(ret)
        rows.append(
            '<tr>'
            '<td style="font-weight:600;">{}</td>'
            '<td>{}</td>'
            '<td>{}</td>'
            '<td style="font-family:\'IBM Plex Mono\',monospace;">${:.2f}</td>'
            '<td style="font-family:\'IBM Plex Mono\',monospace;">${:.2f}</td>'
            '<td class="{}" style="font-family:\'IBM Plex Mono\',monospace;">{}</td>'
            '<td>{}</td>'
            '</tr>'.format(
                t.get('ticker', ''),
                t.get('entry_date', ''),
                t.get('exit_date', ''),
                t.get('entry_price', 0),
                t.get('exit_price', 0),
                ret_class, _fmt_pct(ret),
                t.get('days_held', 0)
            )
        )

    return (
        '<table class="data-table" style="width:100%;">'
        '<thead><tr><th>Ticker</th><th>Entry</th><th>Exit</th>'
        '<th>Entry $</th><th>Exit $</th><th>Return</th><th>Days</th></tr></thead>'
        '<tbody>{}</tbody></table>'.format('\n'.join(rows))
    )


def build_performance_section(trade_log):
    """Build performance summary cards."""
    if not trade_log:
        return '<p style="color:#888;padding:8px 0;">No trades to summarize.</p>'

    returns = [t.get('return_pct', 0) for t in trade_log]
    days = [t.get('days_held', 0) for t in trade_log]
    wins = sum(1 for r in returns if r > 0)
    losses = sum(1 for r in returns if r <= 0)
    avg_ret = np.mean(returns)
    avg_days = np.mean(days)
    cumul = sum(returns)
    avg_win = np.mean([r for r in returns if r > 0]) if wins > 0 else 0
    avg_loss = np.mean([r for r in returns if r <= 0]) if losses > 0 else 0

    # Profit factor
    gross_profit = sum(r for r in returns if r > 0)
    gross_loss = abs(sum(r for r in returns if r <= 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    pf_str = '{:.2f}'.format(pf) if pf < 100 else 'Inf'

    cards = [
        ('Total Trades', str(len(trade_log)), 'neutral'),
        ('Win Rate', '{:.0f}%'.format(wins / len(trade_log) * 100), _color_class(wins / len(trade_log) - 0.5)),
        ('Profit Factor', pf_str, 'pos' if pf > 1.5 else 'neutral'),
        ('Avg Return', _fmt_pct(avg_ret), _color_class(avg_ret)),
        ('Avg Hold', '{:.1f}d'.format(avg_days), 'neutral'),
        ('Cumulative', _fmt_pct(cumul), _color_class(cumul)),
        ('Avg Win', _fmt_pct(avg_win), 'pos'),
        ('Avg Loss', _fmt_pct(avg_loss), 'neg'),
    ]

    html_parts = []
    for label, value, css in cards:
        html_parts.append(
            '<div class="perf-card">'
            '<div class="value {}">{}</div>'
            '<div class="label">{}</div>'
            '</div>'.format(css, value, label)
        )

    return '<div style="display:flex;flex-wrap:wrap;gap:4px;">{}</div>'.format(''.join(html_parts))


def build_watchlist_section(watchlist, positions):
    """Build watchlist status table with quality grades."""
    if not watchlist:
        return '<p style="color:#888;padding:8px 0;">Watchlist is empty.</p>'

    rows = []
    for ticker in sorted(watchlist):
        df = load_from_cache(ticker)
        if df is None or len(df) < SMA_LONG + 10:
            rows.append(
                '<tr><td style="font-weight:600;">{}</td>'
                '<td colspan="5" style="color:#888;">No data</td></tr>'.format(ticker)
            )
            continue

        df = compute_indicators(df)
        last = df.iloc[-1]
        close = float(last['close'])
        sma200 = float(last['sma200']) if not np.isnan(last['sma200']) else None
        sma5 = float(last['sma5']) if not np.isnan(last['sma5']) else None
        rsi2 = float(last['rsi2']) if not np.isnan(last['rsi2']) else None

        rsi_style = ''
        if rsi2 is not None:
            if rsi2 < 10:
                rsi_style = 'color:#1e40af;font-weight:700;'
            elif rsi2 < 20:
                rsi_style = 'color:#6366f1;'

        above_200 = close > sma200 if sma200 else None
        below_5 = close < sma5 if sma5 else None
        signal_ready = above_200 and below_5 and (rsi2 is not None and rsi2 < RSI_THRESHOLD)

        if ticker in positions:
            status = '<span class="signal-badge signal-entry">IN POSITION</span>'
        elif signal_ready:
            status = '<span class="signal-badge signal-entry">SIGNAL</span>'
        else:
            status = '<span class="signal-badge signal-none">WATCH</span>'

        rows.append(
            '<tr>'
            '<td style="font-weight:600;">{}</td>'
            '<td style="font-family:\'IBM Plex Mono\',monospace;">${:.2f}</td>'
            '<td style="font-family:\'IBM Plex Mono\',monospace;{}">{}</td>'
            '<td>{}</td>'
            '<td>{}</td>'
            '</tr>'.format(
                ticker, close,
                rsi_style, '{:.1f}'.format(rsi2) if rsi2 is not None else 'N/A',
                'Yes' if above_200 else 'No' if above_200 is not None else '?',
                status
            )
        )

    return (
        '<table class="data-table" style="width:100%;">'
        '<thead><tr><th>Ticker</th><th>Close</th><th>RSI(2)</th>'
        '<th>&gt; SMA200</th><th>Status</th></tr></thead>'
        '<tbody>{}</tbody></table>'.format('\n'.join(rows))
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    print('RSI(2) Scanner Dashboard Backend v1.4')
    print('=' * 45)

    # Load watchlist
    watchlist = _load_json(os.path.join(_DATA_DIR, 'rsi2_watchlist.json'), ['SPY'])
    print('  Watchlist: {} tickers'.format(len(watchlist)))

    # Load VIX
    vix_val = load_vix()
    if vix_val is not None:
        vix_flag = ' [ELEVATED]' if vix_val > VIX_ELEVATED else ''
        if vix_val > VIX_KILL_SWITCH:
            vix_flag = ' [KILL SWITCH]'
        print('  VIX: {:.1f}{}'.format(vix_val, vix_flag))

    # Load macro data for regime context
    macro_path = os.path.join(_DATA_DIR, 'macro_data.json')
    yc_zscore = None
    yc_label = '--'
    if os.path.exists(macro_path):
        try:
            with open(macro_path, 'r', encoding='utf-8') as f:
                macro = json.load(f)
            yc_data = macro.get('yc_zscore') or {}
            yc_zscore = yc_data.get('yc_zscore_63d')
            yc_label = '{:.2f}'.format(yc_zscore) if yc_zscore is not None else '--'
            if yc_zscore is not None:
                print('  YC Z-Score: {}'.format(yc_label))
        except Exception:
            pass

    # Simulate strategy (self-contained, no state files)
    positions, trade_log, today_entries, today_exits, today_skipped, _ = simulate_strategy(watchlist, vix_val)
    print('  Simulated: {} open, {} trades, {} entries, {} exits, {} skipped'.format(
        len(positions), len(trade_log), len(today_entries), len(today_exits), len(today_skipped)))

    # Banner logic
    vix_killed = vix_val is not None and vix_val > VIX_KILL_SWITCH
    if vix_killed:
        banner_label = 'VIX KILL SWITCH'
        banner_color = '#991b1b'
        banner_detail = 'VIX {:.1f} > {} -- ALL MR entries blocked. Every MR param combo has PF < 0.7 at this level.'.format(
            vix_val, VIX_KILL_SWITCH)
    elif today_skipped and not today_entries:
        banner_label = 'SIGNALS SUPPRESSED'
        banner_color = '#dc2626'
        banner_detail = '{} signal(s) suppressed -- CAUTION grade (above SMA200 only). All major drawdowns here.'.format(
            len(today_skipped))
    elif today_entries:
        banner_label = 'ENTRY SIGNAL'
        banner_color = '#1e40af'
        grades = [e['quality_grade'] for e in today_entries]
        banner_detail = '{} signal(s): {}'.format(len(today_entries), ', '.join(grades))
    elif today_exits:
        banner_label = 'EXIT SIGNAL'
        banner_color = '#ca8a04'
        banner_detail = '{} exit(s) triggering'.format(len(today_exits))
    elif positions:
        banner_label = 'SCANNING'
        banner_color = '#22c55e'
        banner_detail = '{} open position(s) -- monitoring for exits'.format(len(positions))
    else:
        banner_label = 'NO SIGNAL'
        banner_color = '#6b7280'
        banner_detail = 'No entry or exit signals today'

    # Build dashboard
    writer = DashboardWriter('rsi2-scanner', 'RSI(2) Mean-Reversion Scanner')

    stat_items = build_stat_bar(positions, trade_log, today_entries, today_exits, today_skipped, vix_val)
    # Add regime context to stat bar
    stat_items.append(('YC Z', yc_label, 'pos' if yc_zscore and yc_zscore > 0 else 'neg' if yc_zscore and yc_zscore < 0 else 'neutral'))
    if vix_killed:
        stat_items.append(('Kill Switch', 'ACTIVE', 'neg'))
    body = writer.stat_bar(stat_items)
    body += writer.regime_banner(banner_label, banner_detail, color=banner_color)
    body += writer.build_header('RSI(2)<{} | SMA200/SMA5 | Quality Filter'.format(RSI_THRESHOLD))

    # Description block
    desc = DASHBOARD_DESCRIPTIONS.get('rsi2-scanner', '')
    if desc:
        body += '<div style="background:#f0f4ff;border:1px solid #d0d8e8;border-radius:8px;'
        body += 'padding:14px 18px;margin:16px 0;font-size:0.88em;color:#444;">'
        body += desc + '</div>'

    body += writer.section("Today's Signals", build_signals_section(today_entries, today_exits, today_skipped))
    body += writer.section('Open Positions', build_positions_section(positions))
    body += writer.section('Recent Trades (Last 10)', build_trade_log_section(trade_log, limit=10))
    body += writer.section('Performance Summary', build_performance_section(trade_log))
    body += writer.section('Watchlist Status', build_watchlist_section(watchlist, positions))
    body += writer.footer()

    writer.write(body, extra_css=EXTRA_CSS)
    print('[OK] Dashboard written')


if __name__ == '__main__':
    main()
