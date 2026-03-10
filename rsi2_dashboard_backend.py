# -*- coding: utf-8 -*-
# =============================================================================
# rsi2_dashboard_backend.py - v1.0
# Last updated: 2026-03-07
# =============================================================================
# RSI(2) Mean-Reversion Scanner Dashboard Backend
# Reads state files from perplexity-user-data/ and renders a live dashboard.
#
# Data sources (all local, no API calls):
#   - rsi2_positions.json   -- open positions
#   - rsi2_trade_log.json   -- completed trades
#   - rsi2_watchlist.json   -- active watchlist
#   - price_cache/*.pkl     -- latest prices for current RSI2/SMA values
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

# Strategy parameters
SMA_LONG = 200
SMA_SHORT = 5
RSI_PERIOD = 2
RSI_THRESHOLD = 10

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
.signal-none  { background: #f3f4f6; color: #6b7280; }

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


def compute_rsi(close, period=2):
    """Wilder RSI with exponential smoothing."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def get_current_indicators(ticker):
    """Get latest indicators for a ticker."""
    df = load_from_cache(ticker)
    if df is None or len(df) < SMA_LONG + 10:
        return None
    df['sma200'] = df['close'].rolling(window=SMA_LONG, min_periods=SMA_LONG).mean()
    df['sma5'] = df['close'].rolling(window=SMA_SHORT, min_periods=SMA_SHORT).mean()
    df['rsi2'] = compute_rsi(df['close'], period=RSI_PERIOD)
    last = df.iloc[-1]
    return {
        'close': float(last['close']),
        'sma200': float(last['sma200']) if not np.isnan(last['sma200']) else None,
        'sma5': float(last['sma5']) if not np.isnan(last['sma5']) else None,
        'rsi2': float(last['rsi2']) if not np.isnan(last['rsi2']) else None,
        'last_date': df.index[-1].strftime('%Y-%m-%d'),
    }


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


def build_stat_bar(positions, trade_log, today_entries, today_exits):
    """Build the top stat bar stats list."""
    n_open = len(positions) if isinstance(positions, dict) else len(positions)
    n_signals = len(today_entries) + len(today_exits)

    cumul_pnl = sum(t.get('return_pct', 0) for t in trade_log)
    wins = sum(1 for t in trade_log if t.get('return_pct', 0) > 0)
    win_rate = (wins / len(trade_log) * 100) if trade_log else 0

    stats = [
        ('Open Positions', str(n_open), 'neutral'),
        ('Today Signals', str(n_signals), 'pos' if n_signals > 0 else 'neutral'),
        ('Cumul P/L', _fmt_pct(cumul_pnl), _color_class(cumul_pnl)),
        ('Win Rate', '{:.0f}%'.format(win_rate) if trade_log else '--', 'pos' if win_rate > 55 else 'neutral'),
        ('Total Trades', str(len(trade_log)), 'neutral'),
    ]
    return stats


def build_signals_section(entries, exits):
    """Build today's signals HTML."""
    if not entries and not exits:
        return '<p style="color:#888;padding:8px 0;">No signals today -- quiet day.</p>'

    rows = []
    for e in entries:
        rows.append(
            '<tr>'
            '<td><span class="signal-badge signal-entry">BUY</span></td>'
            '<td style="font-weight:600;">{ticker}</td>'
            '<td style="font-family:\'IBM Plex Mono\',monospace;">{rsi2:.1f}</td>'
            '<td style="font-family:\'IBM Plex Mono\',monospace;">${close:.2f}</td>'
            '<td style="font-family:\'IBM Plex Mono\',monospace;">${sma200:.2f} ({sma200_dist:+.1f}%)</td>'
            '</tr>'.format(**e)
        )
    for e in exits:
        rows.append(
            '<tr>'
            '<td><span class="signal-badge signal-exit">SELL</span></td>'
            '<td style="font-weight:600;">{}</td>'
            '<td colspan="3" style="font-family:\'IBM Plex Mono\',monospace;">Return: {}</td>'
            '</tr>'.format(e['ticker'], _fmt_pct(e.get('return_pct', 0)))
        )

    return (
        '<table class="data-table" style="width:100%;">'
        '<thead><tr><th title="BUY = RSI(2) dropped below 10 while above SMA200. EXIT = close crossed above SMA5.">Signal</th><th>Ticker</th><th title="2-period RSI. Below 10 = deeply oversold (entry trigger). Above 50 = mean-reverted.">RSI(2)</th><th title="Latest closing price">Close</th><th title="200-day simple moving average. Stock must be above SMA200 to qualify (long-term uptrend filter).">SMA200</th></tr></thead>'
        '<tbody>{}</tbody></table>'.format('\n'.join(rows))
    )


def build_positions_section(positions_list):
    """Build open positions table."""
    if not positions_list:
        return '<p style="color:#888;padding:8px 0;">No open positions.</p>'

    rows = []
    for pos in positions_list:
        ticker = pos.get('ticker', '')
        indicators = get_current_indicators(ticker)
        current_price = indicators['close'] if indicators else None
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
            '</tr>'.format(
                ticker, entry_dt, entry_price, cur_str,
                pnl_class, pnl_str, days_held
            )
        )

    return (
        '<table class="data-table" style="width:100%;">'
        '<thead><tr><th>Ticker</th><th title="Date the RSI(2) buy signal triggered">Entry Date</th><th title="Closing price on entry date">Entry Price</th>'
        '<th title="Latest closing price">Current</th><th title="Unrealized profit/loss percentage">P/L</th><th title="Trading days since entry">Days</th></tr></thead>'
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
        '<thead><tr><th>Ticker</th><th title="Date RSI(2) buy signal fired">Entry</th><th title="Date close crossed above SMA5 (exit trigger)">Exit</th>'
        '<th title="Closing price on entry date">Entry $</th><th title="Closing price on exit date">Exit $</th><th title="Realized return percentage">Return</th><th title="Trading days held">Days</th></tr></thead>'
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

    cards = [
        ('Total Trades', str(len(trade_log)), 'neutral'),
        ('Win Rate', '{:.0f}%'.format(wins / len(trade_log) * 100), _color_class(wins / len(trade_log) - 0.5)),
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


def build_watchlist_section(watchlist):
    """Build watchlist status table."""
    if not watchlist:
        return '<p style="color:#888;padding:8px 0;">Watchlist is empty.</p>'

    rows = []
    for ticker in sorted(watchlist):
        indicators = get_current_indicators(ticker)
        if indicators is None:
            rows.append(
                '<tr><td style="font-weight:600;">{}</td>'
                '<td colspan="4" style="color:#888;">No data</td></tr>'.format(ticker)
            )
            continue

        rsi2 = indicators['rsi2']
        close = indicators['close']
        sma200 = indicators['sma200']
        sma5 = indicators['sma5']

        rsi_color = ''
        if rsi2 is not None:
            if rsi2 < 10:
                rsi_color = 'color:#1e40af;font-weight:700;'
            elif rsi2 < 20:
                rsi_color = 'color:#6366f1;'

        above_200 = close > sma200 if sma200 else None
        below_5 = close < sma5 if sma5 else None
        signal_ready = above_200 and below_5 and (rsi2 is not None and rsi2 < RSI_THRESHOLD)

        status = '<span class="signal-badge signal-entry">SIGNAL</span>' if signal_ready else \
                 '<span class="signal-badge signal-none">WATCH</span>'

        rows.append(
            '<tr>'
            '<td style="font-weight:600;">{}</td>'
            '<td style="font-family:\'IBM Plex Mono\',monospace;">${:.2f}</td>'
            '<td style="font-family:\'IBM Plex Mono\',monospace;{}">{}</td>'
            '<td>{}</td>'
            '<td>{}</td>'
            '</tr>'.format(
                ticker, close,
                rsi_color, '{:.1f}'.format(rsi2) if rsi2 is not None else 'N/A',
                'Yes' if above_200 else 'No' if above_200 is not None else '?',
                status
            )
        )

    return (
        '<table class="data-table" style="width:100%;">'
        '<thead><tr><th>Ticker</th><th title="Latest closing price">Close</th><th title="2-period RSI value. Below 10 = deeply oversold. Below 5 = extreme.">RSI(2)</th>'
        '<th title="Whether close is above the 200-day SMA (required for entry)">&gt; SMA200</th><th title="Current watchlist status: approaching entry zone, in position, or cooling off">Status</th></tr></thead>'
        '<tbody>{}</tbody></table>'.format('\n'.join(rows))
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    print('RSI(2) Scanner Dashboard Backend v1.0')
    print('=' * 45)

    # Load state files
    positions_raw = _load_json(os.path.join(_DATA_DIR, 'rsi2_positions.json'), [])
    if isinstance(positions_raw, list):
        positions = {p['ticker']: p for p in positions_raw}
    else:
        positions = positions_raw

    trade_log = _load_json(os.path.join(_DATA_DIR, 'rsi2_trade_log.json'), [])
    watchlist = _load_json(os.path.join(_DATA_DIR, 'rsi2_watchlist.json'), ['SPY'])

    print('  Watchlist: {} tickers'.format(len(watchlist)))
    print('  Open positions: {}'.format(len(positions)))
    print('  Completed trades: {}'.format(len(trade_log)))

    # Determine today's signals by scanning
    today_entries = []
    today_exits = []
    for ticker in watchlist:
        indicators = get_current_indicators(ticker)
        if indicators is None:
            continue
        close = indicators['close']
        sma200 = indicators['sma200']
        sma5 = indicators['sma5']
        rsi2 = indicators['rsi2']
        if sma200 is None or rsi2 is None:
            continue

        if close > sma200 and close < sma5 and rsi2 < RSI_THRESHOLD and ticker not in positions:
            sma200_dist = (close / sma200 - 1) * 100
            today_entries.append({
                'ticker': ticker, 'rsi2': rsi2, 'close': close,
                'sma200': sma200, 'sma200_dist': sma200_dist,
            })
        if close > sma5 and ticker in positions:
            pos = positions[ticker]
            ret = (close / pos['entry_price'] - 1) * 100
            today_exits.append({
                'ticker': ticker, 'return_pct': ret,
            })

    today_entries.sort(key=lambda x: x['rsi2'])

    # Determine regime banner
    if today_entries:
        banner_label = 'ENTRY SIGNAL'
        banner_color = '#1e40af'
        banner_detail = '{} ticker(s) triggering RSI(2) < {} pullback'.format(
            len(today_entries), RSI_THRESHOLD)
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

    body = writer.stat_bar(build_stat_bar(positions, trade_log, today_entries, today_exits))
    body += writer.regime_banner(banner_label, banner_detail, color=banner_color)
    body += writer.build_header('RSI(2)<{} | SMA200/SMA5'.format(RSI_THRESHOLD))

    # Description block
    desc = DASHBOARD_DESCRIPTIONS.get('rsi2-scanner', '')
    if desc:
        body += '<div style="background:#f0f4ff;border:1px solid #d0d8e8;border-radius:8px;'
        body += 'padding:14px 18px;margin:16px 0;font-size:0.88em;color:#444;">'
        body += desc + '</div>'

    body += writer.section("Today's Signals", build_signals_section(today_entries, today_exits))
    body += writer.section('Open Positions', build_positions_section(list(positions.values())))
    body += writer.section('Recent Trades (Last 10)', build_trade_log_section(trade_log, limit=10))
    body += writer.section('Performance Summary', build_performance_section(trade_log))
    body += writer.section('Watchlist Status', build_watchlist_section(watchlist))
    body += writer.footer()

    writer.write(body, extra_css=EXTRA_CSS)
    print('[OK] Dashboard written')


if __name__ == '__main__':
    main()
