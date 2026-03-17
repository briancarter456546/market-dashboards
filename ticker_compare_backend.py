# -*- coding: utf-8 -*-
# =============================================================================
# ticker_compare_backend.py - v1.0
# Ticker Comparison Dashboard - Backend Data Exporter
# =============================================================================
# Reads price_cache/*.pkl and exports per-ticker JSON files for the
# interactive ticker-compare dashboard.
#
# Usage:
#   python ticker_compare_backend.py              # export all tickers
#   python ticker_compare_backend.py SPY QQQ AAPL # export specific tickers
#   python ticker_compare_backend.py --list       # show available tickers
#
# Output (on minerva):
#   /var/www/ops-dashboard/ticker-compare/data/<TICKER>.json
#   /var/www/ops-dashboard/ticker-compare/tickers.json
#   /var/www/ops-dashboard/ticker-compare/index.html  (copied from docs/)
#
# Output (local/fallback):
#   docs/ticker-compare/data/<TICKER>.json
#   docs/ticker-compare/tickers.json
# =============================================================================

import json
import os
import pickle
import shutil
import sys
import time

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'perplexity-user-data')
PRICE_CACHE_DIR = os.path.join(DATA_DIR, 'price_cache')

# On minerva, deploy to Caddy-served dir (behind auth).
# Locally, fall back to docs/ for development.
MINERVA_DEPLOY_DIR = '/var/www/ops-dashboard/ticker-compare'
LOCAL_DEPLOY_DIR = os.path.join(BASE_DIR, 'docs', 'ticker-compare')

HTML_SOURCE = os.path.join(BASE_DIR, 'docs', 'ticker-compare', 'index.html')


def get_deploy_dir():
    """Return the deploy directory -- minerva if it exists, else local."""
    if os.path.isdir('/var/www/ops-dashboard'):
        return MINERVA_DEPLOY_DIR
    return LOCAL_DEPLOY_DIR


def get_available_tickers():
    """Return sorted list of tickers with pkl files."""
    tickers = []
    for f in os.listdir(PRICE_CACHE_DIR):
        if f.endswith('.pkl'):
            tickers.append(f.replace('.pkl', ''))
    return sorted(tickers)


def export_ticker(ticker, output_dir):
    """Export a single ticker's adjClose series to JSON."""
    pkl_path = os.path.join(PRICE_CACHE_DIR, '{}.pkl'.format(ticker))
    if not os.path.exists(pkl_path):
        print('[SKIP] {} - no pkl file'.format(ticker))
        return False

    try:
        with open(pkl_path, 'rb') as f:
            df = pickle.load(f)
    except Exception as e:
        print('[FAIL] {} - {}'.format(ticker, e))
        return False

    # Skip non-DataFrame pkl files (some are dicts or other types)
    if not isinstance(df, pd.DataFrame):
        print('[SKIP] {} - not a DataFrame'.format(ticker))
        return False

    # Ensure DatetimeIndex
    if not hasattr(df.index, 'strftime'):
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        else:
            print('[SKIP] {} - no date index'.format(ticker))
            return False

    # Use adjClose, fall back to close
    col = 'adjClose' if 'adjClose' in df.columns else 'close'
    series = df[col].dropna()

    if len(series) < 5:
        print('[SKIP] {} - fewer than 5 data points'.format(ticker))
        return False

    dates = [d.strftime('%Y-%m-%d') for d in series.index]
    prices = [round(float(v), 2) for v in series.values]

    out_path = os.path.join(output_dir, '{}.json'.format(ticker))
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({'dates': dates, 'adjClose': prices}, f, separators=(',', ':'))

    return True


def write_manifest(tickers, deploy_dir):
    """Write the ticker manifest (list of available tickers)."""
    manifest_path = os.path.join(deploy_dir, 'tickers.json')
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(sorted(tickers), f, separators=(',', ':'))
    print('[OK] Manifest written: {} tickers'.format(len(tickers)))


def deploy_html(deploy_dir):
    """Copy index.html from docs/ source to deploy dir."""
    dest = os.path.join(deploy_dir, 'index.html')
    if os.path.exists(HTML_SOURCE):
        shutil.copy2(HTML_SOURCE, dest)
        print('[OK] HTML deployed to {}'.format(dest))
    else:
        print('[!] HTML source not found: {}'.format(HTML_SOURCE))


def main():
    deploy_dir = get_deploy_dir()
    data_dir = os.path.join(deploy_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)

    is_minerva = deploy_dir == MINERVA_DEPLOY_DIR
    print('Deploy target: {} ({})'.format(
        deploy_dir, 'minerva' if is_minerva else 'local'))

    if '--list' in sys.argv:
        for t in get_available_tickers():
            print(t)
        return

    # Determine which tickers to export
    args = [a for a in sys.argv[1:] if not a.startswith('-')]
    if args:
        tickers = [t.upper() for t in args]
    else:
        tickers = get_available_tickers()

    print('Exporting {} tickers from price_cache...'.format(len(tickers)))
    t0 = time.time()

    exported = []
    for i, ticker in enumerate(tickers):
        if export_ticker(ticker, data_dir):
            exported.append(ticker)
        if (i + 1) % 100 == 0:
            print('  ... {}/{}'.format(i + 1, len(tickers)))

    elapsed = time.time() - t0
    print('[OK] Exported {}/{} tickers in {:.1f}s'.format(
        len(exported), len(tickers), elapsed))

    # Write manifest with ALL available exported tickers
    all_exported = []
    for f in os.listdir(data_dir):
        if f.endswith('.json'):
            all_exported.append(f.replace('.json', ''))
    write_manifest(all_exported, deploy_dir)

    # Deploy HTML to serve dir
    deploy_html(deploy_dir)

    print('[OK] Dashboard ready at {}'.format(deploy_dir))
    if is_minerva:
        print('     URL: https://ops.sortinolabs.org/ticker-compare/')


if __name__ == '__main__':
    main()
