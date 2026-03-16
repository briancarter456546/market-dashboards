# -*- coding: utf-8 -*-
"""
Macro Dashboard Backend
=======================

Fetches market regime indicators, stores history, and writes HTML dashboard.

Data sources (FMP Starter):
- VIX (^VIX) and volatility ETF proxies
- Treasury yield curve (full curve from FMP /v4/treasury)
- Sector performance (/v3/sectors-performance)
- Credit spreads (HYG/LQD ratio)
- Key commodity futures and ETF proxies
- Major market indices with MA signals
- Currency / dollar regime (UUP and FX ETFs)
- Breadth (from sector_rotation_1y_data.json)
- Market Health Score (from market_health_data.json)
- HYG/LQD Forward Returns (from hyglqd_data.json)

Outputs:
- macro_data.json          -- current snapshot (read by crash_detection_backend.py)
- macro_history.json       -- append-only daily history
- docs/macro/index.html    -- HTML dashboard via DashboardWriter
- docs/macro/archive/      -- dated archive copy

Version history:
  v1.0  2026-02-26  New file converted from macro_backend_v3.1.py.
                    Adds DashboardWriter HTML output while preserving all
                    JSON outputs and computation logic from v3.1.

Run: python macro_backend.py   (from market-dashboards/ directory)
"""

import os
import sys
import csv as _csv_mod
import json
import pickle
import requests
import numpy as np
import pandas as pd
from datetime import datetime

# ---------------------------------------------------------------------------
# PATH SETUP
# ---------------------------------------------------------------------------

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR   = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", "perplexity-user-data"))

# Make sure dashboard_writer is importable when running from any cwd
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from dashboard_writer import DashboardWriter

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

FMP_API_KEY = os.environ.get('FMP_API_KEY', '')
if not FMP_API_KEY:
    print('[FAIL] FMP_API_KEY not set in environment')
    sys.exit(1)
FMP_BASE    = 'https://financialmodelingprep.com/api'

MACRO_DATA_FILE    = os.path.join(_DATA_DIR, "macro_data.json")
MACRO_HISTORY_FILE = os.path.join(_DATA_DIR, "macro_history.json")
MARKET_HEALTH_FILE = os.path.join(_DATA_DIR, "market_health_data.json")
HYGLQD_FILE        = os.path.join(_DATA_DIR, "hyglqd_data.json")
SECROT_FILE        = os.path.join(_DATA_DIR, "sector_rotation_1y_data.json")
CACHE_DIR          = os.path.join(_DATA_DIR, "price_cache")


# ==============================================================================
# YIELD CURVE Z-SCORE (from price_cache, no API calls)
# ==============================================================================

def compute_yc_zscore():
    """Compute IEF/SHY 63-day z-score as yield curve slope proxy.

    IEF (7-10yr treasury) / SHY (1-3yr treasury) ratio captures yield curve
    slope without requiring yield data. Rolling 63-day z-score normalizes.

    Scimode finding (2026-03-15): Perfectly monotonic for D-Stock
    (Q1 PF 1.38 -> Q5 PF 1.91). Independent of VIX (r=0.128).
    """
    ief_path = os.path.join(CACHE_DIR, 'IEF.pkl')
    shy_path = os.path.join(CACHE_DIR, 'SHY.pkl')

    if not os.path.exists(ief_path) or not os.path.exists(shy_path):
        print("  [WARN] IEF.pkl or SHY.pkl not found in price_cache")
        return None

    try:
        with open(ief_path, 'rb') as f:
            ief_df = pickle.load(f)
        with open(shy_path, 'rb') as f:
            shy_df = pickle.load(f)

        # Align to DatetimeIndex
        for df in [ief_df, shy_df]:
            if not isinstance(df.index, pd.DatetimeIndex):
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                else:
                    return None

        # Use adjClose if available, else close
        ief_close = ief_df['adjClose'] if 'adjClose' in ief_df.columns else ief_df['close']
        shy_close = shy_df['adjClose'] if 'adjClose' in shy_df.columns else shy_df['close']

        # Align dates
        ratio = (ief_close / shy_close).dropna()
        if len(ratio) < 126:  # need 6 months minimum
            return None

        # Rolling 63-day z-score
        rm = ratio.rolling(63, min_periods=42).mean()
        rs = ratio.rolling(63, min_periods=42).std()
        zscore = ((ratio - rm) / rs.replace(0, np.nan)).dropna()

        if len(zscore) == 0:
            return None

        latest_z = float(zscore.iloc[-1])
        latest_ratio = float(ratio.iloc[-1])
        latest_date = str(zscore.index[-1].date())

        # Classify
        if latest_z > 1.0:
            regime = 'steep'
            label = 'Steep (YC expanding)'
        elif latest_z < -1.0:
            regime = 'flat'
            label = 'Flat/Inverted (YC compressing)'
        else:
            regime = 'neutral'
            label = 'Neutral'

        print("  YC Z-Score: {:.2f} (ratio {:.4f}, {})".format(latest_z, latest_ratio, label))

        return {
            'yc_zscore_63d': round(latest_z, 3),
            'ief_shy_ratio': round(latest_ratio, 4),
            'regime': regime,
            'regime_label': label,
            'as_of': latest_date,
        }
    except Exception as e:
        print("  [WARN] YC z-score computation failed: {}".format(e))
        return None


# ==============================================================================
# API FETCHING
# ==============================================================================

def fetch_json(url, timeout=15):
    """Fetch JSON from URL with error handling."""
    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code == 200:
            return resp.json()
        else:
            print("  Warning: HTTP {}: {}...".format(resp.status_code, url[:60]))
            return None
    except requests.exceptions.Timeout:
        print("  Warning: Timeout: {}...".format(url[:60]))
        return None
    except Exception as e:
        print("  Warning: Error: {}".format(e))
        return None


def fetch_quote(symbol):
    """Fetch single quote."""
    url = "{}/v3/quote/{}?apikey={}".format(FMP_BASE, symbol, FMP_API_KEY)
    data = fetch_json(url)
    if data and len(data) > 0:
        return data[0]
    return None


def fetch_quotes(symbols):
    """Fetch multiple quotes."""
    results = {}
    for symbol in symbols:
        quote = fetch_quote(symbol)
        if quote:
            results[symbol] = {
                'price':         quote.get('price'),
                'change':        quote.get('change'),
                'changePercent': quote.get('changesPercentage'),
                'dayHigh':       quote.get('dayHigh'),
                'dayLow':        quote.get('dayLow'),
                'previousClose': quote.get('previousClose'),
                'volume':        quote.get('volume'),
                'avgVolume':     quote.get('avgVolume'),
                'priceAvg50':    quote.get('priceAvg50'),
                'priceAvg200':   quote.get('priceAvg200'),
            }
    return results


# ==============================================================================
# DATA COLLECTION
# ==============================================================================

def collect_vix_data():
    """Collect VIX and volatility indicators."""
    print("Fetching VIX data...")

    vix_symbols = ['^VIX', '^VVIX', 'VIXY', 'UVXY', 'VXX']
    data = fetch_quotes(vix_symbols)

    vix  = data.get('^VIX',  {}).get('price')
    vvix = data.get('^VVIX', {}).get('price')

    if vix:
        if vix < 15:
            regime       = 'low'
            regime_label = 'Low (Complacent)'
        elif vix < 20:
            regime       = 'normal'
            regime_label = 'Normal'
        elif vix < 30:
            regime       = 'elevated'
            regime_label = 'Elevated'
        else:
            regime       = 'crisis'
            regime_label = 'Crisis'
    else:
        regime       = 'unknown'
        regime_label = 'Unknown'

    return {
        'vix':          vix,
        'vvix':         vvix,
        'regime':       regime,
        'regime_label': regime_label,
        'etf_proxies':  {
            'VIXY': data.get('VIXY', {}).get('price'),
            'UVXY': data.get('UVXY', {}).get('price'),
            'VXX':  data.get('VXX',  {}).get('price'),
        }
    }


def collect_treasury_data():
    """Collect treasury yield curve."""
    print("Fetching Treasury data...")

    url  = "{}/v4/treasury?apikey={}".format(FMP_BASE, FMP_API_KEY)
    data = fetch_json(url)

    if not data or len(data) == 0:
        return None

    latest = data[0]

    y10 = latest.get('year10', 0)
    y2  = latest.get('year2',  0)
    y3m = latest.get('month3', 0)
    y30 = latest.get('year30', 0)

    spread_10y_2y  = y10 - y2  if y10 and y2  else None
    spread_10y_3m  = y10 - y3m if y10 and y3m else None
    spread_30y_10y = y30 - y10 if y30 and y10 else None

    if spread_10y_2y is not None:
        if spread_10y_2y < 0:
            curve_regime = 'inverted'
            curve_label  = 'Inverted'
        elif spread_10y_2y < 0.5:
            curve_regime = 'flat'
            curve_label  = 'Flat'
        else:
            curve_regime = 'normal'
            curve_label  = 'Normal'
    else:
        curve_regime = 'unknown'
        curve_label  = 'Unknown'

    return {
        'date': latest.get('date'),
        'rates': {
            '1m':  latest.get('month1'),
            '2m':  latest.get('month2'),
            '3m':  latest.get('month3'),
            '6m':  latest.get('month6'),
            '1y':  latest.get('year1'),
            '2y':  latest.get('year2'),
            '3y':  latest.get('year3'),
            '5y':  latest.get('year5'),
            '7y':  latest.get('year7'),
            '10y': latest.get('year10'),
            '20y': latest.get('year20'),
            '30y': latest.get('year30'),
        },
        'spreads': {
            '10y_2y':  round(spread_10y_2y,  3) if spread_10y_2y  is not None else None,
            '10y_3m':  round(spread_10y_3m,  3) if spread_10y_3m  is not None else None,
            '30y_10y': round(spread_30y_10y, 3) if spread_30y_10y is not None else None,
        },
        'regime':       curve_regime,
        'regime_label': curve_label,
    }


def collect_credit_data():
    """Collect credit spread indicators."""
    print("Fetching Credit data...")

    symbols = ['HYG', 'LQD', 'JNK', 'EMB', 'AGG']
    data    = fetch_quotes(symbols)

    hyg   = data.get('HYG', {}).get('price')
    lqd   = data.get('LQD', {}).get('price')
    ratio = hyg / lqd if hyg and lqd else None

    if ratio:
        if ratio > 0.74:
            regime       = 'risk_on'
            regime_label = 'Risk-On'
        elif ratio > 0.71:
            regime       = 'neutral'
            regime_label = 'Neutral'
        else:
            regime       = 'risk_off'
            regime_label = 'Risk-Off'
    else:
        regime       = 'unknown'
        regime_label = 'Unknown'

    return {
        'hyg_lqd_ratio': round(ratio, 4) if ratio else None,
        'regime':        regime,
        'regime_label':  regime_label,
        'prices': {
            'HYG': hyg,
            'LQD': lqd,
            'JNK': data.get('JNK', {}).get('price'),
            'EMB': data.get('EMB', {}).get('price'),
            'AGG': data.get('AGG', {}).get('price'),
        },
        'changes': {
            'HYG': data.get('HYG', {}).get('changePercent'),
            'LQD': data.get('LQD', {}).get('changePercent'),
        }
    }


def collect_sector_data():
    """Collect sector performance."""
    print("Fetching Sector data...")

    url  = "{}/v3/sectors-performance?apikey={}".format(FMP_BASE, FMP_API_KEY)
    data = fetch_json(url)

    if not data:
        return None

    sectors = {}
    for sector in data:
        name       = sector.get('sector', 'Unknown')
        change_str = sector.get('changesPercentage', '0%')
        change     = float(change_str.replace('%', '')) if change_str else 0
        sectors[name] = round(change, 4)

    sorted_sectors = dict(sorted(sectors.items(), key=lambda x: x[1], reverse=True))

    sector_list = list(sorted_sectors.items())
    leaders     = sector_list[:3]  if len(sector_list) >= 3 else sector_list
    laggards    = sector_list[-3:] if len(sector_list) >= 3 else []

    return {
        'performance': sorted_sectors,
        'leaders':  [{'sector': s, 'change': c} for s, c in leaders],
        'laggards': [{'sector': s, 'change': c} for s, c in laggards],
    }


def collect_commodity_data():
    """Collect key commodity prices."""
    print("Fetching Commodity data...")

    commodity_symbols = ['GCUSD', 'SIUSD', 'CLUSD', 'NGUSD', 'HGUSD']
    commodity_names   = {
        'GCUSD': 'Gold',
        'SIUSD': 'Silver',
        'CLUSD': 'Crude Oil',
        'NGUSD': 'Natural Gas',
        'HGUSD': 'Copper',
    }

    data        = fetch_quotes(commodity_symbols)
    commodities = {}
    for symbol, name in commodity_names.items():
        if symbol in data:
            commodities[name] = {
                'price':  data[symbol].get('price'),
                'change': data[symbol].get('changePercent'),
            }

    etf_symbols = ['GLD', 'SLV', 'USO', 'UNG', 'DBC']
    etf_data    = fetch_quotes(etf_symbols)

    etf_proxies = {}
    for symbol in etf_symbols:
        if symbol in etf_data:
            etf_proxies[symbol] = {
                'price':  etf_data[symbol].get('price'),
                'change': etf_data[symbol].get('changePercent'),
            }

    return {
        'futures':     commodities,
        'etf_proxies': etf_proxies,
    }


def collect_index_data():
    """Collect major index data."""
    print("Fetching Index data...")

    symbols = ['^GSPC', '^DJI', '^IXIC', '^RUT', 'SPY', 'QQQ', 'IWM', 'DIA']
    data    = fetch_quotes(symbols)

    indices = {}
    for symbol in symbols:
        if symbol in data:
            indices[symbol] = {
                'price':       data[symbol].get('price'),
                'change':      data[symbol].get('changePercent'),
                'priceAvg50':  data[symbol].get('priceAvg50'),
                'priceAvg200': data[symbol].get('priceAvg200'),
            }

    spy         = data.get('SPY', {})
    spy_price   = spy.get('price')
    spy_50ma    = spy.get('priceAvg50')
    spy_200ma   = spy.get('priceAvg200')

    spy_above_50  = spy_price > spy_50ma  if spy_price and spy_50ma  else None
    spy_above_200 = spy_price > spy_200ma if spy_price and spy_200ma else None

    return {
        'quotes': indices,
        'spy_trend': {
            'above_50ma':  spy_above_50,
            'above_200ma': spy_above_200,
            'price':       spy_price,
            'ma50':        spy_50ma,
            'ma200':       spy_200ma,
        }
    }


def collect_currency_data():
    """Collect currency / dollar data."""
    print("Fetching Currency data...")

    symbols = ['UUP', 'FXE', 'FXY', 'FXB', 'FXA', 'FXC']
    data    = fetch_quotes(symbols)

    currencies = {}
    for symbol in symbols:
        if symbol in data:
            currencies[symbol] = {
                'price':  data[symbol].get('price'),
                'change': data[symbol].get('changePercent'),
            }

    uup       = data.get('UUP', {})
    uup_price = uup.get('price')
    uup_50ma  = uup.get('priceAvg50')
    uup_200ma = uup.get('priceAvg200')

    if uup_price and uup_50ma and uup_200ma:
        if uup_price > uup_50ma > uup_200ma:
            regime       = 'strong'
            regime_label = 'Strong Dollar'
        elif uup_price < uup_50ma < uup_200ma:
            regime       = 'weak'
            regime_label = 'Weak Dollar'
        else:
            regime       = 'neutral'
            regime_label = 'Neutral'
    else:
        regime       = 'unknown'
        regime_label = 'Unknown'

    return {
        'quotes':              currencies,
        'dollar_regime':       regime,
        'dollar_regime_label': regime_label,
    }


def collect_breadth_from_secrot():
    """Calculate breadth from sector_rotation_1y_data.json."""
    print("Calculating Breadth from sector_rotation_1y_data.json...")

    if not os.path.exists(SECROT_FILE):
        print("  Warning: sector_rotation_1y_data.json not found - run sector_rotation first")
        return None

    with open(SECROT_FILE) as f:
        secrot = json.load(f)

    data  = secrot.get('data', {})
    etfs  = list(data.values())
    total = len(etfs)

    if total == 0:
        return None

    above_200ma_proxy = 0
    above_50ma_proxy  = 0
    above_20ma_proxy  = 0
    pos_5d            = 0
    bullish_sr        = 0

    for etf in etfs:
        if etf.get('trend_1w') and etf['trend_1w'] > 5.5:
            above_200ma_proxy += 1
        if etf.get('trend_1d') and etf['trend_1d'] > 5.5:
            above_50ma_proxy += 1
        if etf.get('mom_1d') and etf['mom_1d'] > 5.5:
            above_20ma_proxy += 1
        if etf.get('ret_5d') and etf['ret_5d'] > 0:
            pos_5d += 1
        if etf.get('sr_score') and etf.get('sr_total'):
            if etf['sr_score'] / etf['sr_total'] >= 0.6:
                bullish_sr += 1

    pct_200 = round(above_200ma_proxy / total * 100, 1)
    pct_50  = round(above_50ma_proxy  / total * 100, 1)
    pct_20  = round(above_20ma_proxy  / total * 100, 1)
    pct_pos = round(pos_5d            / total * 100, 1)

    if pct_200 >= 70:
        regime       = 'strong'
        regime_label = 'Strong'
    elif pct_200 >= 50:
        regime       = 'healthy'
        regime_label = 'Healthy'
    elif pct_200 >= 30:
        regime       = 'cautious'
        regime_label = 'Cautious'
    else:
        regime       = 'weak'
        regime_label = 'Weak'

    return {
        'total_etfs':          total,
        'above_200ma_proxy':   above_200ma_proxy,
        'above_50ma_proxy':    above_50ma_proxy,
        'above_20ma_proxy':    above_20ma_proxy,
        'positive_5d':         pos_5d,
        'bullish_secrot':      bullish_sr,
        'pct_above_200ma':     pct_200,
        'pct_above_50ma':      pct_50,
        'pct_above_20ma':      pct_20,
        'pct_positive_5d':     pct_pos,
        'pct_bullish_sr':      round(bullish_sr / total * 100, 1),
        'regime':              regime,
        'regime_label':        regime_label,
    }


# ==============================================================================
# OVERALL REGIME CLASSIFICATION
# ==============================================================================

def classify_overall_regime(macro_data):
    """Classify overall market regime based on all indicators."""

    scores = {'bullish': 0, 'bearish': 0}

    vix_regime = (macro_data.get('volatility') or {}).get('regime', 'unknown')
    if vix_regime == 'low':
        scores['bullish'] += 1
    elif vix_regime == 'normal':
        scores['bullish'] += 0.5
    elif vix_regime == 'elevated':
        scores['bearish'] += 0.5
    elif vix_regime == 'crisis':
        scores['bearish'] += 1

    curve_regime = (macro_data.get('treasury') or {}).get('regime', 'unknown')
    if curve_regime == 'normal':
        scores['bullish'] += 1
    elif curve_regime == 'flat':
        scores['bearish'] += 0.5
    elif curve_regime == 'inverted':
        scores['bearish'] += 1

    credit_regime = (macro_data.get('credit') or {}).get('regime', 'unknown')
    if credit_regime == 'risk_on':
        scores['bullish'] += 1
    elif credit_regime == 'risk_off':
        scores['bearish'] += 1

    breadth_regime = (macro_data.get('breadth') or {}).get('regime', 'unknown')
    if breadth_regime == 'strong':
        scores['bullish'] += 1
    elif breadth_regime == 'healthy':
        scores['bullish'] += 0.5
    elif breadth_regime == 'cautious':
        scores['bearish'] += 0.5
    elif breadth_regime == 'weak':
        scores['bearish'] += 1

    net_score = scores['bullish'] - scores['bearish']

    if net_score >= 2:
        regime       = 'bullish'
        regime_label = 'Bullish'
        confidence   = 'high'
    elif net_score >= 1:
        regime       = 'bullish'
        regime_label = 'Leaning Bullish'
        confidence   = 'medium'
    elif net_score <= -2:
        regime       = 'bearish'
        regime_label = 'Bearish'
        confidence   = 'high'
    elif net_score <= -1:
        regime       = 'bearish'
        regime_label = 'Leaning Bearish'
        confidence   = 'medium'
    else:
        regime       = 'neutral'
        regime_label = 'Neutral/Mixed'
        confidence   = 'low'

    return {
        'regime':        regime,
        'regime_label':  regime_label,
        'confidence':    confidence,
        'net_score':     net_score,
        'bullish_score': scores['bullish'],
        'bearish_score': scores['bearish'],
    }


# ==============================================================================
# TREND CALCULATIONS
# ==============================================================================

def calculate_5d_trends(macro_data):
    """Calculate 5-day trends for key indicators from history."""
    print("Calculating 5-day trends...")

    history      = load_history()
    hist_records = history.get('history', [])

    if len(hist_records) < 2:
        print("  Not enough history for trends")
        return {}

    recent = hist_records[-6:]
    trends = {}

    # VIX trend
    current_vix    = recent[-1].get('vix')
    prev_5d_vix    = recent[0].get('vix') if len(recent) >= 6 else (recent[-2].get('vix') if len(recent) >= 2 else None)
    if current_vix and prev_5d_vix:
        trends['vix_change_5d'] = round(((current_vix - prev_5d_vix) / prev_5d_vix) * 100, 2)

    # Yield curve spread trend (in basis points)
    current_spread  = recent[-1].get('spread_10y_2y')
    prev_5d_spread  = recent[0].get('spread_10y_2y') if len(recent) >= 6 else (recent[-2].get('spread_10y_2y') if len(recent) >= 2 else None)
    if current_spread and prev_5d_spread:
        trends['yield_curve_change_5d_bps'] = round((current_spread - prev_5d_spread) * 100, 1)

    # HYG/LQD ratio trend
    current_hyglqd = recent[-1].get('hyg_lqd')
    prev_5d_hyglqd = recent[0].get('hyg_lqd') if len(recent) >= 6 else (recent[-2].get('hyg_lqd') if len(recent) >= 2 else None)
    if current_hyglqd and prev_5d_hyglqd:
        trends['hyglqd_change_5d'] = round(((current_hyglqd - prev_5d_hyglqd) / prev_5d_hyglqd) * 100, 2)

    # Breadth trend (in percentage points)
    current_breadth = recent[-1].get('breadth_200ma')
    prev_5d_breadth = recent[0].get('breadth_200ma') if len(recent) >= 6 else (recent[-2].get('breadth_200ma') if len(recent) >= 2 else None)
    if current_breadth and prev_5d_breadth:
        trends['breadth_change_5d_ppt'] = round(current_breadth - prev_5d_breadth, 1)

    return trends


def detect_regime_changes(macro_data):
    """Detect regime changes from previous day."""
    print("Detecting regime changes...")

    history      = load_history()
    hist_records = history.get('history', [])

    if len(hist_records) < 2:
        print("  Not enough history for change detection")
        return {'has_changes': False, 'changes': []}

    prev_day = hist_records[-2]
    changes  = []

    # VIX regime
    current_vix_regime = (macro_data.get('volatility') or {}).get('regime')
    prev_vix_regime    = prev_day.get('vix_regime')
    if current_vix_regime and prev_vix_regime and current_vix_regime != prev_vix_regime:
        changes.append({
            'indicator':  'VIX',
            'from':       prev_vix_regime,
            'to':         current_vix_regime,
            'from_label': prev_vix_regime.replace('_', '-').title(),
            'to_label':   current_vix_regime.replace('_', '-').title(),
        })

    # Yield curve regime
    current_curve_regime = (macro_data.get('treasury') or {}).get('regime')
    prev_curve_regime    = prev_day.get('curve_regime')
    if current_curve_regime and prev_curve_regime and current_curve_regime != prev_curve_regime:
        changes.append({
            'indicator':  'Yield Curve',
            'from':       prev_curve_regime,
            'to':         current_curve_regime,
            'from_label': prev_curve_regime.title(),
            'to_label':   current_curve_regime.title(),
        })

    # Credit regime
    current_credit_regime = (macro_data.get('credit') or {}).get('regime')
    prev_credit_regime    = prev_day.get('credit_regime')
    if current_credit_regime and prev_credit_regime and current_credit_regime != prev_credit_regime:
        changes.append({
            'indicator':  'Credit (HYG/LQD)',
            'from':       prev_credit_regime,
            'to':         current_credit_regime,
            'from_label': prev_credit_regime.replace('_', '-').title(),
            'to_label':   current_credit_regime.replace('_', '-').title(),
        })

    # Breadth regime
    current_breadth_regime = (macro_data.get('breadth') or {}).get('regime')
    prev_breadth_regime    = prev_day.get('breadth_regime')
    if current_breadth_regime and prev_breadth_regime and current_breadth_regime != prev_breadth_regime:
        changes.append({
            'indicator':  'Breadth',
            'from':       prev_breadth_regime,
            'to':         current_breadth_regime,
            'from_label': prev_breadth_regime.title(),
            'to_label':   current_breadth_regime.title(),
        })

    # Overall regime
    current_overall_regime = (macro_data.get('overall_regime') or {}).get('regime')
    prev_overall_regime    = prev_day.get('overall_regime')
    if current_overall_regime and prev_overall_regime and current_overall_regime != prev_overall_regime:
        changes.append({
            'indicator':  'Overall Market',
            'from':       prev_overall_regime,
            'to':         current_overall_regime,
            'from_label': prev_overall_regime.title(),
            'to_label':   current_overall_regime.title(),
        })

    return {
        'has_changes': len(changes) > 0,
        'count':       len(changes),
        'changes':     changes,
    }


def load_market_health():
    """Load market health score from market_health_data.json."""
    print("Loading Market Health data...")

    if not os.path.exists(MARKET_HEALTH_FILE):
        print("  Warning: market_health_data.json not found")
        return None

    try:
        with open(MARKET_HEALTH_FILE) as f:
            data = json.load(f)
        health = data.get('market_health', {})
        return {
            'overall_health':  health.get('overall_health'),
            'status':          health.get('status'),
            'status_color':    health.get('status_color'),
            'status_emoji':    health.get('status_emoji'),
            'generated_at':    data.get('generated_at'),
            'components':      health.get('component_scores', {}),
            'weights':         health.get('weights', {}),
        }
    except Exception as e:
        print("  Error loading market health: {}".format(e))
        return None


def load_hyglqd_signal():
    """Load HYG/LQD forward return expectations from hyglqd_data.json."""
    print("Loading HYG/LQD signal data...")

    if not os.path.exists(HYGLQD_FILE):
        print("  Warning: hyglqd_data.json not found")
        return None

    try:
        with open(HYGLQD_FILE) as f:
            return json.load(f)
    except Exception as e:
        print("  Error loading HYG/LQD data: {}".format(e))
        return None


def compute_alert_status(macro_data):
    """Compute alert status based on threshold breaches."""
    alerts = []

    # Market Health < 50
    market_health = macro_data.get('market_health')
    if market_health and market_health.get('overall_health') is not None:
        health_val = market_health['overall_health']
        if health_val < 50:
            alerts.append({
                'type':     'market_health',
                'severity': 'high',
                'message':  'Market Health Low: {:.1f}/100'.format(health_val),
            })

    # VIX > 18
    vix_data = macro_data.get('volatility')
    if vix_data and vix_data.get('vix'):
        vix_val = vix_data['vix']
        if vix_val > 18:
            alerts.append({
                'type':     'vix',
                'severity': 'medium' if vix_val < 25 else 'high',
                'message':  'VIX Elevated: {:.1f}'.format(vix_val),
            })

    # Breadth < 55%
    breadth_data = macro_data.get('breadth')
    if breadth_data and breadth_data.get('pct_above_200ma') is not None:
        breadth_val = breadth_data['pct_above_200ma']
        if breadth_val < 55:
            alerts.append({
                'type':     'breadth',
                'severity': 'medium' if breadth_val > 40 else 'high',
                'message':  'Breadth Weak: {:.1f}% above 200MA'.format(breadth_val),
            })

    # HYG/LQD stress
    hyglqd_data = macro_data.get('hyglqd_signal')
    if hyglqd_data and hyglqd_data.get('current'):
        ratio = hyglqd_data['current'].get('ratio')
        if ratio and ratio < 0.71:
            alerts.append({
                'type':     'credit',
                'severity': 'high',
                'message':  'Credit Stress: HYG/LQD {:.4f} (Risk-Off)'.format(ratio),
            })

    all_medium = all(a['severity'] == 'medium' for a in alerts)
    return {
        'has_alerts': len(alerts) > 0,
        'count':      len(alerts),
        'alerts':     alerts,
        'status':     'clear' if len(alerts) == 0 else ('warning' if all_medium else 'alert'),
    }


# ==============================================================================
# HISTORY MANAGEMENT
# ==============================================================================

def load_history():
    if os.path.exists(MACRO_HISTORY_FILE):
        with open(MACRO_HISTORY_FILE) as f:
            return json.load(f)
    return {'history': []}


def save_history(history):
    with open(MACRO_HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)


def append_to_history(macro_data):
    history        = load_history()
    today          = macro_data['generated_at'][:10]
    existing_dates = [h.get('date') for h in history.get('history', [])]

    record = {
        'date':           today,
        'timestamp':      macro_data['generated_at'],
        'vix':            (macro_data.get('volatility') or {}).get('vix'),
        'vix_regime':     (macro_data.get('volatility') or {}).get('regime'),
        'yield_10y':      ((macro_data.get('treasury') or {}).get('rates') or {}).get('10y'),
        'yield_2y':       ((macro_data.get('treasury') or {}).get('rates') or {}).get('2y'),
        'spread_10y_2y':  ((macro_data.get('treasury') or {}).get('spreads') or {}).get('10y_2y'),
        'curve_regime':   (macro_data.get('treasury') or {}).get('regime'),
        'hyg_lqd':        (macro_data.get('credit') or {}).get('hyg_lqd_ratio'),
        'credit_regime':  (macro_data.get('credit') or {}).get('regime'),
        'breadth_200ma':  (macro_data.get('breadth') or {}).get('pct_above_200ma'),
        'breadth_regime': (macro_data.get('breadth') or {}).get('regime'),
        'overall_regime': (macro_data.get('overall_regime') or {}).get('regime'),
        'overall_score':  (macro_data.get('overall_regime') or {}).get('net_score'),
        'spy_price':      (((macro_data.get('indices') or {}).get('quotes') or {}).get('SPY') or {}).get('price'),
        'spy_change':     (((macro_data.get('indices') or {}).get('quotes') or {}).get('SPY') or {}).get('change'),
        'yc_zscore_63d':  (macro_data.get('yc_zscore') or {}).get('yc_zscore_63d'),
        'yc_regime':      (macro_data.get('yc_zscore') or {}).get('regime'),
    }

    if today in existing_dates:
        idx = existing_dates.index(today)
        history['history'][idx] = record
        print("Updated history for {}".format(today))
    else:
        history['history'].append(record)
        print("Added history for {}".format(today))

    history['history'].sort(key=lambda x: x['date'])
    history['last_updated'] = macro_data['generated_at']
    history['total_days']   = len(history['history'])

    save_history(history)
    return history


# ==============================================================================
# HTML HELPERS
# ==============================================================================

def _fmt_val(v, decimals=2, suffix=''):
    """Format a numeric value or return N/A."""
    if v is None:
        return '<span class="muted">N/A</span>'
    try:
        tmpl = '{{:.{}f}}{{}}'.format(decimals)
        return tmpl.format(float(v), suffix)
    except (TypeError, ValueError):
        return str(v)


def _pct_class(v):
    """Return css class (pos/neg/muted) for a percent change."""
    if v is None:
        return 'muted'
    try:
        return 'pos' if float(v) >= 0 else 'neg'
    except (TypeError, ValueError):
        return 'muted'


def _trend_arrow(v):
    """Return an up/down/flat arrow span for a numeric delta."""
    if v is None:
        return '<span class="muted">--</span>'
    try:
        fv = float(v)
        if fv > 0:
            return '<span class="trend-up">&#9650; {}</span>'.format(abs(fv))
        elif fv < 0:
            return '<span class="trend-dn">&#9660; {}</span>'.format(abs(fv))
        else:
            return '<span class="trend-flat">&#9654; 0</span>'
    except (TypeError, ValueError):
        return '<span class="muted">--</span>'


def _regime_color(regime):
    """Return a hex color string for a regime key."""
    colors = {
        'bullish':  '#22c55e',
        'bearish':  '#ef4444',
        'neutral':  '#f59e0b',
        'risk_on':  '#22c55e',
        'risk_off': '#ef4444',
        'low':      '#22c55e',
        'normal':   '#22c55e',
        'elevated': '#f59e0b',
        'crisis':   '#ef4444',
        'strong':   '#22c55e',
        'healthy':  '#4ade80',
        'cautious': '#f59e0b',
        'weak':     '#ef4444',
        'inverted': '#ef4444',
        'flat':     '#f59e0b',
    }
    return colors.get(regime, '#888888')


def _stat_bar_class(regime):
    """Return a stat-bar css class for a regime key."""
    mapping = {
        'bullish':  'pos',
        'bearish':  'neg',
        'neutral':  'warn',
        'risk_on':  'pos',
        'risk_off': 'neg',
        'low':      'pos',
        'normal':   'pos',
        'elevated': 'warn',
        'crisis':   'neg',
        'strong':   'pos',
        'healthy':  'pos',
        'cautious': 'warn',
        'weak':     'neg',
        'inverted': 'neg',
        'flat':     'warn',
    }
    return mapping.get(regime, 'neutral')


# ==============================================================================
# SECTION BUILDERS
# ==============================================================================

def build_regime_changes_section(writer, regime_changes):
    """Build HTML for regime changes section (empty state if none)."""
    changes = (regime_changes or {}).get('changes', [])
    if not changes:
        content = '<p class="muted" style="padding:8px 0;">No regime changes detected since yesterday.</p>'
    else:
        rows = []
        for ch in changes:
            rows.append(
                '<tr>'
                '<td class="ticker">{indicator}</td>'
                '<td><span class="regime-pill rp-from">{from_label}</span></td>'
                '<td style="color:#888;font-size:1.2em;">&#8594;</td>'
                '<td><span class="regime-pill rp-to">{to_label}</span></td>'
                '</tr>'.format(**ch)
            )
        content = (
            '<table>'
            '<thead><tr>'
            '<th title="Market indicator that changed regime">Indicator</th><th title="Previous regime classification">From</th><th></th><th title="New regime classification as of today">To</th>'
            '</tr></thead>'
            '<tbody>{}</tbody>'
            '</table>'
        ).format(''.join(rows))

    return writer.section(
        'Regime Changes ({})'.format(len(changes)),
        content,
        hint='vs yesterday'
    )


def build_alerts_section(writer, alert_status):
    """Build HTML for alerts section."""
    alerts = (alert_status or {}).get('alerts', [])
    if not alerts:
        content = '<p class="pos" style="font-weight:600;padding:8px 0;">All Clear -- No alerts triggered.</p>'
    else:
        items = []
        for a in alerts:
            sev   = a.get('severity', 'medium')
            cls   = 'alert-high' if sev == 'high' else 'alert-medium'
            icon  = '&#9888;' if sev == 'high' else '&#9675;'
            items.append(
                '<div class="alert-row {cls}">'
                '<span class="alert-icon">{icon}</span>'
                '<span class="alert-msg">{msg}</span>'
                '<span class="alert-sev">{sev}</span>'
                '</div>'.format(cls=cls, icon=icon, msg=a.get('message', ''), sev=sev.upper())
            )
        content = ''.join(items)

    return writer.section(
        'Alerts ({})'.format(len(alerts)),
        content,
        hint='Threshold breaches'
    )


def build_trends_section(writer, trends):
    """Build 5-day trend cards."""
    if not trends:
        content = '<p class="muted">Not enough history to compute trends (need 2+ days).</p>'
        return writer.section('5-Day Trends', content)

    vix_chg      = trends.get('vix_change_5d')
    spread_chg   = trends.get('yield_curve_change_5d_bps')
    hyglqd_chg   = trends.get('hyglqd_change_5d')
    breadth_chg  = trends.get('breadth_change_5d_ppt')

    def _card(label, value, unit, note=''):
        if value is None:
            disp  = '<span class="muted">N/A</span>'
            cls   = 'muted'
            arrow = ''
        else:
            try:
                fv    = float(value)
                cls   = 'pos' if fv <= 0 else 'neg'   # VIX down = good
                disp  = '{:.2f}{}'.format(abs(fv), unit)
                arrow = '&#9650;' if fv > 0 else ('&#9660;' if fv < 0 else '&#9654;')
            except (TypeError, ValueError):
                disp  = str(value)
                cls   = 'muted'
                arrow = ''
        return (
            '<div class="card" style="border-top-color:{color};">'
            '<div class="label">{label}</div>'
            '<div class="value {cls}">{arrow} {disp}</div>'
            '{note}'
            '</div>'
        ).format(
            color=_regime_color('pos' if cls == 'pos' else 'neg' if cls == 'neg' else 'neutral'),
            label=label,
            cls=cls,
            arrow=arrow,
            disp=disp,
            note='<div class="sub">{}</div>'.format(note) if note else '',
        )

    # For VIX: rising is bad (neg), falling is good (pos)
    def _card_vix(label, value, unit, note=''):
        if value is None:
            disp  = '<span class="muted">N/A</span>'
            cls   = 'muted'
            arrow = ''
        else:
            try:
                fv    = float(value)
                cls   = 'neg' if fv > 0 else 'pos'   # VIX up = bad
                disp  = '{:.2f}{}'.format(abs(fv), unit)
                arrow = '&#9650;' if fv > 0 else ('&#9660;' if fv < 0 else '&#9654;')
            except (TypeError, ValueError):
                disp  = str(value)
                cls   = 'muted'
                arrow = ''
        return (
            '<div class="card" style="border-top-color:{color};">'
            '<div class="label">{label}</div>'
            '<div class="value {cls}">{arrow} {disp}</div>'
            '{note}'
            '</div>'
        ).format(
            color=_regime_color('pos' if cls == 'pos' else 'neg' if cls == 'neg' else 'neutral'),
            label=label,
            cls=cls,
            arrow=arrow,
            disp=disp,
            note='<div class="sub">{}</div>'.format(note) if note else '',
        )

    cards = '<div class="cards">'
    cards += _card_vix('VIX Change (5d)', vix_chg, '%', 'Down = good')
    cards += _card('Yield Curve Chg (5d)', spread_chg, 'bps', '10Y-2Y, up = steeper')
    cards += _card('HYG/LQD Chg (5d)', hyglqd_chg, '%', 'Up = risk-on')
    cards += _card('Breadth Chg (5d)', breadth_chg, 'ppt', '% above 200MA proxy')
    cards += '</div>'

    return writer.section('5-Day Trends', cards, hint='Change from 5 days ago')


def build_vix_section(writer, vol_data):
    """Build VIX & Volatility section."""
    if not vol_data:
        return writer.section('VIX and Volatility', '<p class="muted">No data</p>')

    vix   = vol_data.get('vix')
    vvix  = vol_data.get('vvix')
    rlbl  = vol_data.get('regime_label', 'Unknown')
    etfs  = vol_data.get('etf_proxies', {})

    etf_rows = ''.join(
        '<tr><td class="ticker">{}</td><td class="num">{}</td></tr>'.format(
            sym, _fmt_val(etfs.get(sym), 2)
        )
        for sym in ['VIXY', 'UVXY', 'VXX']
    )

    content = (
        '<div class="cards">'
        '<div class="card" style="border-top-color:{vix_color};">'
        '<div class="label">VIX Level</div>'
        '<div class="value {vix_cls}">{vix_val}</div>'
        '<div class="sub">{regime}</div>'
        '</div>'
        '<div class="card">'
        '<div class="label">VVIX</div>'
        '<div class="value neutral">{vvix_val}</div>'
        '<div class="sub">Vol of Vol</div>'
        '</div>'
        '</div>'
        '<table>'
        '<thead><tr><th title="Volatility-related ETF ticker symbol">ETF Proxy</th><th title="Latest traded price in USD">Price</th></tr></thead>'
        '<tbody>{etf_rows}</tbody>'
        '</table>'
    ).format(
        vix_color=_regime_color(vol_data.get('regime', 'unknown')),
        vix_cls=_stat_bar_class(vol_data.get('regime', 'unknown')),
        vix_val=_fmt_val(vix, 1),
        regime=rlbl,
        vvix_val=_fmt_val(vvix, 1),
        etf_rows=etf_rows,
    )

    return writer.section('VIX and Volatility', content)


def build_treasury_section(writer, treas_data):
    """Build Treasury Yield Curve section."""
    if not treas_data:
        return writer.section('Treasury Yield Curve', '<p class="muted">No data</p>')

    rates   = treas_data.get('rates', {})
    spreads = treas_data.get('spreads', {})
    rlbl    = treas_data.get('regime_label', 'Unknown')
    date    = treas_data.get('date', '')

    rate_order = ['1m', '2m', '3m', '6m', '1y', '2y', '3y', '5y', '7y', '10y', '20y', '30y']
    rate_rows  = ''.join(
        '<tr><td class="ticker">{}</td><td class="num">{}</td></tr>'.format(
            t.upper(), _fmt_val(rates.get(t), 3, '%')
        )
        for t in rate_order if rates.get(t) is not None
    )

    spread_rows = ''
    spread_map  = [
        ('10y_2y',  '10Y - 2Y'),
        ('10y_3m',  '10Y - 3M'),
        ('30y_10y', '30Y - 10Y'),
    ]
    for key, label in spread_map:
        val = spreads.get(key)
        if val is not None:
            css = 'pos' if val >= 0 else 'neg'
            spread_rows += (
                '<tr><td>{}</td>'
                '<td class="num {css}">{val}%</td>'
                '</tr>'
            ).format(label, css=css, val=_fmt_val(val, 3))

    content = (
        '<div class="cards">'
        '<div class="card" style="border-top-color:{color};">'
        '<div class="label">Curve Regime</div>'
        '<div class="value {cls}">{lbl}</div>'
        '<div class="sub">Date: {date}</div>'
        '</div>'
        '</div>'
        '<div class="macro-two-col">'
        '<div>'
        '<p class="pb-header">Yield Rates</p>'
        '<table><thead><tr><th title="Treasury bond maturity length">Tenor</th><th title="Current Treasury yield as percentage">Rate</th></tr></thead>'
        '<tbody>{rate_rows}</tbody></table>'
        '</div>'
        '<div>'
        '<p class="pb-header">Key Spreads</p>'
        '<table><thead><tr><th title="Yield curve spread between two tenors">Spread</th><th title="Current spread value in basis points">Value</th></tr></thead>'
        '<tbody>{spread_rows}</tbody></table>'
        '</div>'
        '</div>'
    ).format(
        color=_regime_color(treas_data.get('regime', 'unknown')),
        cls=_stat_bar_class(treas_data.get('regime', 'unknown')),
        lbl=rlbl,
        date=date,
        rate_rows=rate_rows,
        spread_rows=spread_rows or '<tr><td colspan="2" class="muted">No spread data</td></tr>',
    )

    return writer.section('Treasury Yield Curve', content)


def build_credit_section(writer, credit_data):
    """Build Credit Markets section."""
    if not credit_data:
        return writer.section('Credit Markets', '<p class="muted">No data</p>')

    ratio  = credit_data.get('hyg_lqd_ratio')
    rlbl   = credit_data.get('regime_label', 'Unknown')
    prices = credit_data.get('prices', {})
    chgs   = credit_data.get('changes', {})

    price_rows = ''.join(
        '<tr>'
        '<td class="ticker">{}</td>'
        '<td class="num">{}</td>'
        '<td class="num {cls}">{chg}</td>'
        '</tr>'.format(
            sym,
            _fmt_val(prices.get(sym), 2),
            cls=_pct_class(chgs.get(sym)),
            chg=_fmt_val(chgs.get(sym), 2, '%') if sym in chgs else _fmt_val(None)
        )
        for sym in ['HYG', 'LQD', 'JNK', 'EMB', 'AGG']
        if prices.get(sym) is not None
    )

    content = (
        '<div class="cards">'
        '<div class="card" style="border-top-color:{color};">'
        '<div class="label">HYG/LQD Ratio</div>'
        '<div class="value {cls}">{ratio}</div>'
        '<div class="sub">{lbl}</div>'
        '</div>'
        '</div>'
        '<table>'
        '<thead><tr><th title="Credit market ETF ticker symbol">ETF</th><th title="Latest traded price in USD">Price</th><th title="1-day price change as percentage">1D Chg%</th></tr></thead>'
        '<tbody>{rows}</tbody>'
        '</table>'
    ).format(
        color=_regime_color(credit_data.get('regime', 'unknown')),
        cls=_stat_bar_class(credit_data.get('regime', 'unknown')),
        ratio=_fmt_val(ratio, 4),
        lbl=rlbl,
        rows=price_rows or '<tr><td colspan="3" class="muted">No data</td></tr>',
    )

    return writer.section('Credit Markets', content)


def build_sector_section(writer, sector_data):
    """Build Sector Performance section."""
    if not sector_data:
        return writer.section('Sector Performance', '<p class="muted">No data</p>')

    performance = sector_data.get('performance', {})
    rows = ''.join(
        '<tr>'
        '<td>{}</td>'
        '<td class="num {cls}">{chg}</td>'
        '</tr>'.format(
            name,
            cls=_pct_class(chg),
            chg=_fmt_val(chg, 2, '%')
        )
        for name, chg in performance.items()
    )

    content = (
        '<table>'
        '<thead><tr><th title="S&amp;P 500 sector name from FMP API">Sector</th><th title="1-day price change as percentage">1D Change %</th></tr></thead>'
        '<tbody>{}</tbody>'
        '</table>'
    ).format(rows or '<tr><td colspan="2" class="muted">No data</td></tr>')

    return writer.section('Sector Performance', content, hint='Sorted best to worst')


def build_commodity_section(writer, commodity_data):
    """Build Commodities section."""
    if not commodity_data:
        return writer.section('Commodities', '<p class="muted">No data</p>')

    futures  = commodity_data.get('futures', {})
    etf_prox = commodity_data.get('etf_proxies', {})

    fut_rows = ''.join(
        '<tr>'
        '<td>{}</td>'
        '<td class="num">{}</td>'
        '<td class="num {cls}">{chg}</td>'
        '</tr>'.format(
            name,
            _fmt_val(d.get('price'), 2),
            cls=_pct_class(d.get('change')),
            chg=_fmt_val(d.get('change'), 2, '%')
        )
        for name, d in futures.items()
    )

    etf_rows = ''.join(
        '<tr>'
        '<td class="ticker">{}</td>'
        '<td class="num">{}</td>'
        '<td class="num {cls}">{chg}</td>'
        '</tr>'.format(
            sym,
            _fmt_val(d.get('price'), 2),
            cls=_pct_class(d.get('change')),
            chg=_fmt_val(d.get('change'), 2, '%')
        )
        for sym, d in etf_prox.items()
    )

    content = (
        '<div class="macro-two-col">'
        '<div>'
        '<p class="pb-header">Futures</p>'
        '<table><thead><tr><th title="Commodity futures contract name">Commodity</th><th title="Latest futures price in USD">Price</th><th title="1-day price change as percentage">1D %</th></tr></thead>'
        '<tbody>{fut}</tbody></table>'
        '</div>'
        '<div>'
        '<p class="pb-header">ETF Proxies</p>'
        '<table><thead><tr><th title="Commodity or currency ETF ticker symbol">Symbol</th><th title="Latest traded price in USD">Price</th><th title="1-day price change as percentage">1D %</th></tr></thead>'
        '<tbody>{etf}</tbody></table>'
        '</div>'
        '</div>'
    ).format(
        fut=fut_rows or '<tr><td colspan="3" class="muted">No data</td></tr>',
        etf=etf_rows or '<tr><td colspan="3" class="muted">No data</td></tr>',
    )

    return writer.section('Commodities', content)


def build_index_section(writer, index_data):
    """Build Indices section."""
    if not index_data:
        return writer.section('Indices', '<p class="muted">No data</p>')

    quotes    = index_data.get('quotes', {})
    spy_trend = index_data.get('spy_trend', {})

    display_order = ['^GSPC', '^DJI', '^IXIC', '^RUT', 'SPY', 'QQQ', 'IWM', 'DIA']
    rows = []
    for sym in display_order:
        d = quotes.get(sym)
        if not d:
            continue
        price  = d.get('price')
        chg    = d.get('change')
        ma50   = d.get('priceAvg50')
        ma200  = d.get('priceAvg200')
        a50    = price > ma50  if price and ma50  else None
        a200   = price > ma200 if price and ma200 else None
        ma50_s  = '<span class="trend-above">50MA</span>'  if a50  is True else ('<span class="trend-below">50MA</span>'  if a50  is False else '<span class="muted">--</span>')
        ma200_s = '<span class="trend-above">200MA</span>' if a200 is True else ('<span class="trend-below">200MA</span>' if a200 is False else '<span class="muted">--</span>')
        rows.append(
            '<tr>'
            '<td class="ticker">{sym}</td>'
            '<td class="num">{price}</td>'
            '<td class="num {cls}">{chg}</td>'
            '<td>{ma50_s}</td>'
            '<td>{ma200_s}</td>'
            '</tr>'.format(
                sym=sym,
                price=_fmt_val(price, 2),
                cls=_pct_class(chg),
                chg=_fmt_val(chg, 2, '%'),
                ma50_s=ma50_s,
                ma200_s=ma200_s,
            )
        )

    # SPY trend summary card
    spy_price = spy_trend.get('price')
    spy_50    = spy_trend.get('ma50')
    spy_200   = spy_trend.get('ma200')
    spy_card  = ''
    if spy_price:
        spy_card = (
            '<div class="cards" style="margin-bottom:16px;">'
            '<div class="card"><div class="label">SPY vs 50MA</div>'
            '<div class="value {cls50}">{lbl50}</div>'
            '<div class="sub">{price} vs {ma50}</div>'
            '</div>'
            '<div class="card"><div class="label">SPY vs 200MA</div>'
            '<div class="value {cls200}">{lbl200}</div>'
            '<div class="sub">{price} vs {ma200}</div>'
            '</div>'
            '</div>'
        ).format(
            cls50='pos'  if spy_trend.get('above_50ma')  else 'neg',
            lbl50='Above' if spy_trend.get('above_50ma') else 'Below',
            cls200='pos'  if spy_trend.get('above_200ma')  else 'neg',
            lbl200='Above' if spy_trend.get('above_200ma') else 'Below',
            price=_fmt_val(spy_price, 2),
            ma50=_fmt_val(spy_50, 2),
            ma200=_fmt_val(spy_200, 2),
        )

    content = (
        '{spy_card}'
        '<table>'
        '<thead><tr><th title="Index or ETF ticker symbol">Symbol</th><th title="Latest traded price in USD">Price</th><th title="1-day price change as percentage">1D %</th><th title="50-day moving average price level">50MA</th><th title="200-day moving average price level">200MA</th></tr></thead>'
        '<tbody>{rows}</tbody>'
        '</table>'
    ).format(
        spy_card=spy_card,
        rows=''.join(rows) or '<tr><td colspan="5" class="muted">No data</td></tr>',
    )

    return writer.section('Indices', content)


def build_breadth_section(writer, breadth_data):
    """Build Breadth section."""
    if not breadth_data:
        return writer.section('Breadth', '<p class="muted">No sector rotation data available</p>')

    total    = breadth_data.get('total_etfs', 0)
    rlbl     = breadth_data.get('regime_label', 'Unknown')
    pct_200  = breadth_data.get('pct_above_200ma')
    pct_50   = breadth_data.get('pct_above_50ma')
    pct_20   = breadth_data.get('pct_above_20ma')
    pct_pos  = breadth_data.get('pct_positive_5d')
    pct_bsr  = breadth_data.get('pct_bullish_sr')

    def _brow(label, pct):
        cls = 'pos' if (pct or 0) >= 50 else 'neg'
        return (
            '<tr><td>{}</td>'
            '<td class="num {cls}">{pct}%</td>'
            '</tr>'
        ).format(label, cls=cls, pct=_fmt_val(pct, 1))

    content = (
        '<div class="cards">'
        '<div class="card" style="border-top-color:{color};">'
        '<div class="label">Breadth Regime</div>'
        '<div class="value {cls}">{lbl}</div>'
        '<div class="sub">{n} ETFs tracked</div>'
        '</div>'
        '<div class="card">'
        '<div class="label">Above 200MA Proxy</div>'
        '<div class="value {cls200}">{pct200}%</div>'
        '<div class="sub">Strong trend>5.5</div>'
        '</div>'
        '</div>'
        '<table>'
        '<thead><tr><th title="Breadth indicator being measured">Metric</th><th title="Percentage of tracked ETFs meeting this criteria">Pct of Universe</th></tr></thead>'
        '<tbody>'
        '{r200}{r50}{r20}{rpos}{rbsr}'
        '</tbody>'
        '</table>'
    ).format(
        color=_regime_color(breadth_data.get('regime', 'unknown')),
        cls=_stat_bar_class(breadth_data.get('regime', 'unknown')),
        lbl=rlbl,
        n=total,
        cls200='pos' if (pct_200 or 0) >= 50 else 'neg',
        pct200=_fmt_val(pct_200, 1),
        r200=_brow('Above 200MA Proxy (trend_1w > 5.5)', pct_200),
        r50=_brow('Above 50MA Proxy (trend_1d > 5.5)', pct_50),
        r20=_brow('Above 20MA Proxy (mom_1d > 5.5)', pct_20),
        rpos=_brow('Positive 5D Return', pct_pos),
        rbsr=_brow('Bullish SecRot Score (>=60%)', pct_bsr),
    )

    return writer.section('Breadth (from sector_rotation_1y_data)', content)


# ==============================================================================
# EXTRA CSS / JS
# ==============================================================================

EXTRA_CSS = """
/* -- Alert rows -- */
.alert-row {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 12px 16px;
    border-radius: 6px;
    margin-bottom: 10px;
    font-weight: 600;
    font-size: 0.95em;
}
.alert-high   { background: #fef2f2; border-left: 4px solid #ef4444; color: #b91c1c; }
.alert-medium { background: #fffbeb; border-left: 4px solid #f59e0b; color: #92400e; }
.alert-icon   { font-size: 1.3em; }
.alert-msg    { flex: 1; }
.alert-sev    {
    font-size: 0.72em;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    padding: 3px 10px;
    border-radius: 20px;
    background: rgba(0,0,0,0.07);
}

/* -- Trend arrows -- */
.trend-up   { color: #ef4444; font-weight: 700; }
.trend-dn   { color: #22c55e; font-weight: 700; }
.trend-flat { color: #888;    font-weight: 600; }

/* -- Regime change table pills -- */
.regime-pill {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.85em;
    font-weight: 700;
    text-transform: capitalize;
}
.rp-from { background: #f1f5f9; color: #475569; border: 1px solid #cbd5e1; }
.rp-to   { background: #eff6ff; color: #1d4ed8; border: 1px solid #bfdbfe; }

/* Two-column grid layout */
.macro-two-col {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
}

/* ── Mobile responsive ── */
@media (max-width: 768px) {
    .alert-row { padding: 10px 12px; font-size: 0.85em; gap: 10px; }
    .alert-icon { font-size: 1.1em; }
    .alert-sev { font-size: 0.68em; padding: 2px 8px; }
    .regime-pill { font-size: 0.78em; padding: 3px 10px; }
    .macro-two-col { grid-template-columns: 1fr !important; }
}
"""

EXTRA_JS = ""


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("=" * 70)
    print("MACRO DASHBOARD BACKEND")
    print("=" * 70)
    print("Time: {}".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    print()

    macro_data = {'generated_at': datetime.now().isoformat()}

    # -- Collect core data --
    macro_data['volatility']     = collect_vix_data()
    macro_data['treasury']       = collect_treasury_data()
    macro_data['credit']         = collect_credit_data()
    macro_data['sectors']        = collect_sector_data()
    macro_data['commodities']    = collect_commodity_data()
    macro_data['indices']        = collect_index_data()
    macro_data['currencies']     = collect_currency_data()
    macro_data['breadth']        = collect_breadth_from_secrot()
    macro_data['overall_regime'] = classify_overall_regime(macro_data)

    # -- Yield curve z-score (from price_cache, no API calls) --
    macro_data['yc_zscore']      = compute_yc_zscore()

    # -- v3.0+ enrichment --
    macro_data['trends']         = calculate_5d_trends(macro_data)
    macro_data['regime_changes'] = detect_regime_changes(macro_data)
    macro_data['market_health']  = load_market_health()
    macro_data['hyglqd_signal']  = load_hyglqd_signal()
    macro_data['alert_status']   = compute_alert_status(macro_data)

    # -- Save JSON outputs (other scripts read these) --
    print("\nSaving {}...".format(MACRO_DATA_FILE))
    with open(MACRO_DATA_FILE, 'w') as f:
        json.dump(macro_data, f, indent=2)

    print("Updating {}...".format(MACRO_HISTORY_FILE))
    history = append_to_history(macro_data)

    # -- Build HTML dashboard --
    print("\nBuilding HTML dashboard...")
    writer = DashboardWriter("macro", "Macro Dashboard")

    overall  = macro_data.get('overall_regime') or {}
    vol      = macro_data.get('volatility')     or {}
    treas    = macro_data.get('treasury')        or {}
    cred     = macro_data.get('credit')          or {}
    bread    = macro_data.get('breadth')         or {}
    trends   = macro_data.get('trends')          or {}
    changes  = macro_data.get('regime_changes')  or {}
    alerts   = macro_data.get('alert_status')    or {}

    # Stat bar values
    regime_lbl      = overall.get('regime_label', 'Unknown')
    regime_regime   = overall.get('regime', 'neutral')
    vix_val         = vol.get('vix')
    spread_val      = ((treas.get('spreads') or {}).get('10y_2y'))
    hyglqd_val      = cred.get('hyg_lqd_ratio')
    breadth_pct     = bread.get('pct_above_200ma')
    yc_data         = macro_data.get('yc_zscore') or {}
    yc_z_val        = yc_data.get('yc_zscore_63d')

    stat_items = [
        ('Overall Regime', regime_lbl, _stat_bar_class(regime_regime)),
        ('VIX', _fmt_val(vix_val, 1), _stat_bar_class(vol.get('regime', 'unknown'))),
        ('10Y-2Y Spread', _fmt_val(spread_val, 3, '%'), _stat_bar_class(treas.get('regime', 'unknown'))),
        ('YC Z-Score', _fmt_val(yc_z_val, 2), _stat_bar_class(yc_data.get('regime', 'unknown'))),
        ('HYG/LQD Ratio', _fmt_val(hyglqd_val, 4), _stat_bar_class(cred.get('regime', 'unknown'))),
        ('Breadth %', _fmt_val(breadth_pct, 1, '%'), _stat_bar_class(bread.get('regime', 'unknown'))),
    ]

    # Regime banner score html
    net_score    = overall.get('net_score', 0)
    bull_score   = overall.get('bullish_score', 0)
    bear_score   = overall.get('bearish_score', 0)
    confidence   = overall.get('confidence', 'unknown')
    score_html   = (
        'Net Score: {net} &nbsp;|&nbsp; '
        'Bullish: {bull} &nbsp;|&nbsp; Bearish: {bear}'
        '<br>Confidence: {conf}'
    ).format(net=net_score, bull=bull_score, bear=bear_score, conf=confidence.title())

    banner_color = _regime_color(regime_regime)

    # Assemble page parts
    parts = []
    parts.append(writer.stat_bar(stat_items))
    parts.append(writer.build_header(
        subtitle='Generated {}'.format(datetime.now().strftime('%Y-%m-%d %H:%M'))
    ))
    parts.append(writer.regime_banner(
        label=regime_lbl.upper(),
        score_html=score_html,
        color=banner_color,
    ))

    # Alert count badge next to section title if alerts exist
    parts.append(build_regime_changes_section(writer, changes))
    parts.append(build_alerts_section(writer, alerts))
    parts.append(build_trends_section(writer, trends))
    parts.append(build_vix_section(writer, macro_data.get('volatility')))
    parts.append(build_treasury_section(writer, macro_data.get('treasury')))
    parts.append(build_credit_section(writer, macro_data.get('credit')))
    parts.append(build_sector_section(writer, macro_data.get('sectors')))
    parts.append(build_commodity_section(writer, macro_data.get('commodities')))
    parts.append(build_index_section(writer, macro_data.get('indices')))
    parts.append(build_breadth_section(writer, macro_data.get('breadth')))
    parts.append(writer.footer())

    writer.write("\n".join(parts), extra_css=EXTRA_CSS, extra_js=EXTRA_JS)

    # Write CSV
    csv_path = os.path.join(_SCRIPT_DIR, 'macro_data.csv')
    csv_row = {
        'date': datetime.now().strftime('%Y-%m-%d'),
        'overall_regime': overall.get('regime_label'),
        'net_score': overall.get('net_score'),
        'bullish_score': overall.get('bullish_score'),
        'bearish_score': overall.get('bearish_score'),
        'confidence': overall.get('confidence'),
        'vix': vol.get('vix'),
        'vix_regime': vol.get('regime_label'),
        'yield_10y': (treas.get('yields') or {}).get('10y'),
        'yield_2y': (treas.get('yields') or {}).get('2y'),
        'spread_10y_2y': (treas.get('spreads') or {}).get('10y_2y'),
        'curve_regime': treas.get('regime_label'),
        'hyg_lqd_ratio': cred.get('hyg_lqd_ratio'),
        'credit_regime': cred.get('regime_label'),
        'breadth_pct': bread.get('pct_above_200ma'),
        'breadth_regime': bread.get('regime_label'),
    }
    with open(csv_path, 'w', newline='', encoding='utf-8') as _cf:
        _w = _csv_mod.DictWriter(_cf, fieldnames=list(csv_row.keys()))
        _w.writeheader()
        _w.writerow(csv_row)
    print("CSV: {}".format(csv_path))

    # -- Console summary --
    print()
    print("=" * 70)
    print("MACRO REGIME SUMMARY")
    print("=" * 70)
    print("Overall Regime: {} (confidence: {})".format(
        overall.get('regime_label', 'Unknown'), overall.get('confidence', 'unknown')
    ))
    print("Net Score: {} (bullish: {}, bearish: {})".format(
        overall.get('net_score', 0),
        overall.get('bullish_score', 0),
        overall.get('bearish_score', 0),
    ))
    print()

    if changes.get('has_changes'):
        print("REGIME CHANGES DETECTED:")
        for ch in changes.get('changes', []):
            print("  {}: {} -> {}".format(ch['indicator'], ch['from_label'], ch['to_label']))
        print()

    if alerts.get('has_alerts'):
        print("ALERTS ({}):".format(alerts['count']))
        for a in alerts.get('alerts', []):
            marker = '[HIGH]' if a['severity'] == 'high' else '[MEDIUM]'
            print("  {} {}".format(marker, a['message']))
        print()
    else:
        print("ALL CLEAR - No alerts triggered")
        print()

    vix_chg    = trends.get('vix_change_5d', 0) or 0
    spread_chg = trends.get('yield_curve_change_5d_bps', 0) or 0
    credit_chg = trends.get('hyglqd_change_5d', 0) or 0
    bread_chg  = trends.get('breadth_change_5d_ppt', 0) or 0

    vix_arrow    = '^' if vix_chg    > 0 else ('v' if vix_chg    < 0 else '-')
    spread_arrow = '^' if spread_chg > 0 else ('v' if spread_chg < 0 else '-')
    credit_arrow = '^' if credit_chg > 0 else ('v' if credit_chg < 0 else '-')
    bread_arrow  = '^' if bread_chg  > 0 else ('v' if bread_chg  < 0 else '-')

    print("Individual Indicators (with 5-day trends):")
    print("  VIX:         {} {}{:.1f}%  -> {}".format(
        _fmt_val(vol.get('vix'), 1), vix_arrow, abs(vix_chg), vol.get('regime_label', 'Unknown')
    ))
    print("  Yield Curve: {} {}{:.0f}bps -> {}".format(
        _fmt_val(spread_val, 3, '%'), spread_arrow, abs(spread_chg), treas.get('regime_label', 'Unknown')
    ))
    print("  Credit:      {} {}{:.2f}%  -> {}".format(
        _fmt_val(hyglqd_val, 4), credit_arrow, abs(credit_chg), cred.get('regime_label', 'Unknown')
    ))
    print("  Breadth:     {} {}{:.1f}ppt -> {}".format(
        _fmt_val(breadth_pct, 1, '%'), bread_arrow, abs(bread_chg), bread.get('regime_label', 'Unknown')
    ))

    mhealth = macro_data.get('market_health')
    if mhealth and mhealth.get('overall_health') is not None:
        print()
        print("Market Health: {:.1f}/100 ({})".format(
            mhealth['overall_health'], mhealth.get('status', 'Unknown')
        ))

    print()
    print("History: {} days tracked".format(history.get('total_days', 0)))
    print()
    print("Done!")


if __name__ == '__main__':
    main()
