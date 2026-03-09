# -*- coding: utf-8 -*-
# =============================================================================
# smart_money_backend.py - v1.0
# Last updated: 2026-03-08
# =============================================================================
# v1.0: Initial market-dashboards conversion from legacy Django smart money.
#       Self-contained indicator calculations (CMF, MFI, OBV, NVI/PVI, RVOL).
#       Reads from price_cache, writes HTML via DashboardWriter.
#       Covers full price_cache universe (~1,400 tickers), not just 98 ETFs.
# =============================================================================
"""
Smart Money Dashboard Backend
=============================
Volume-based institutional flow indicators for ~1,400 tickers.

INDICATORS (6 components, 0-6 composite score):
  1. OBV trend:     OBV > 20-day MA of OBV
  2. MFI oversold:  MFI(14) < 40
  3. CMF oscillator: EMA(3) - EMA(10) of CMF(20) > 0
  4. Relative vol:  volume / 20-day avg > 1.3
  5. NVI/PVI inst:  NVI > MA(20) AND PVI < MA(20)
  6. OBV divergence: fractal-based bullish divergence

Score >= 4 = BUY signal, <= 2 = SELL signal.

NOTE: These indicators are PENDING scimode validation. The composite
weights and thresholds are from the original Django build and have not
been backtested against forward returns. Run scimode before trusting.

Author: Brian + Claude
"""

import os
import sys
import json
import pickle
import warnings
from datetime import datetime, date
from pathlib import Path

import numpy as np
import pandas as pd

from dashboard_writer import DashboardWriter

warnings.filterwarnings('ignore')

# =============================================================================
# PATHS
# =============================================================================

_SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
_DATA_DIR = _SCRIPT_DIR / '..' / 'perplexity-user-data'
CACHE_DIR = _DATA_DIR / 'price_cache'
CACHE_META = CACHE_DIR / 'cache_metadata.json'

# =============================================================================
# INDICATOR CALCULATIONS
# =============================================================================

def calc_cmf(high, low, close, volume, length=20):
    """Chaikin Money Flow."""
    hl_diff = high - low
    mfm = np.where(hl_diff == 0, 0.0,
                   ((2 * close - low - high) / hl_diff) * volume)
    mfm = pd.Series(mfm, index=close.index, dtype=np.float32)
    ad_sum = mfm.rolling(length, min_periods=length).sum()
    vol_sum = volume.rolling(length, min_periods=length).sum()
    return (ad_sum / vol_sum.replace(0, np.nan)).astype(np.float32)


def calc_cmf_oscillator(high, low, close, volume,
                        length=20, short_p=3, long_p=10):
    """CMF oscillator = EMA(short) - EMA(long) of CMF."""
    cmf = calc_cmf(high, low, close, volume, length)
    short_ema = cmf.ewm(span=short_p, adjust=False).mean()
    long_ema = cmf.ewm(span=long_p, adjust=False).mean()
    return (short_ema - long_ema).astype(np.float32)


def calc_mfi(high, low, close, volume, period=14):
    """Money Flow Index (0-100)."""
    tp = (high + low + close) / 3
    mf = tp * volume
    pos = mf.where(tp > tp.shift(1), 0)
    neg = mf.where(tp < tp.shift(1), 0)
    pos_sum = pos.rolling(period).sum()
    neg_sum = neg.rolling(period).sum()
    ratio = pos_sum / neg_sum.replace(0, np.nan)
    return (100 - (100 / (1 + ratio))).astype(np.float32)


def calc_obv(close, volume):
    """On-Balance Volume."""
    return (np.sign(close.diff()) * volume).fillna(0).cumsum()


def calc_nvi_pvi(close, volume):
    """Negative/Positive Volume Index."""
    ret = close.pct_change()
    vol_down = volume < volume.shift(1)
    vol_up = volume > volume.shift(1)
    nvi = (1 + ret.where(vol_down, 0)).cumprod() * 1000
    pvi = (1 + ret.where(vol_up, 0)).cumprod() * 1000
    return nvi, pvi


def calc_institutional_signal(nvi, pvi, ma_len=20):
    """NVI/PVI institutional flow signal: +1 accumulation, -1 distribution."""
    nvi_ma = nvi.rolling(ma_len).mean()
    pvi_ma = pvi.rolling(ma_len).mean()
    sig = pd.Series(0, index=nvi.index, dtype=np.int8)
    sig[(nvi > nvi_ma) & (pvi < pvi_ma)] = 1
    sig[(nvi < nvi_ma) & (pvi > pvi_ma)] = -1
    return sig


def detect_obv_divergence(close, obv, lookback=60):
    """Simplified divergence: price makes lower low but OBV makes higher low
    over the last `lookback` bars. Returns bool for latest bar only."""
    if len(close) < lookback + 10:
        return False, False
    c = close.iloc[-lookback:]
    o = obv.iloc[-lookback:]
    mid = lookback // 2
    # Bull div: recent price low < earlier price low, but OBV low > earlier OBV low
    price_lo1 = c.iloc[:mid].min()
    price_lo2 = c.iloc[mid:].min()
    obv_lo1 = o.iloc[:mid].min()
    obv_lo2 = o.iloc[mid:].min()
    bull_div = (price_lo2 < price_lo1) and (obv_lo2 > obv_lo1)
    # Bear div: recent price high > earlier, but OBV high < earlier
    price_hi1 = c.iloc[:mid].max()
    price_hi2 = c.iloc[mid:].max()
    obv_hi1 = o.iloc[:mid].max()
    obv_hi2 = o.iloc[mid:].max()
    bear_div = (price_hi2 > price_hi1) and (obv_hi2 < obv_hi1)
    return bull_div, bear_div


# =============================================================================
# SCORING
# =============================================================================

def score_ticker(df):
    """Compute all indicators and return a dict of signals for latest bar."""
    if df is None or len(df) < 252:
        return None

    close = df['close'].astype(np.float64)
    high = df['high'].astype(np.float64)
    low = df['low'].astype(np.float64)
    volume = df['volume'].astype(np.float64)

    # 1. OBV trend
    obv = calc_obv(close, volume)
    obv_ma20 = obv.rolling(20).mean()
    obv_above = bool(obv.iloc[-1] > obv_ma20.iloc[-1])

    # 2. MFI
    mfi = calc_mfi(high, low, close, volume, 14)
    mfi_val = float(mfi.iloc[-1]) if not pd.isna(mfi.iloc[-1]) else 50.0
    mfi_oversold = mfi_val < 40

    # 3. CMF oscillator
    cmf_osc = calc_cmf_oscillator(high, low, close, volume)
    cmf_osc_val = float(cmf_osc.iloc[-1]) if not pd.isna(cmf_osc.iloc[-1]) else 0.0
    cmf_positive = cmf_osc_val > 0

    # Multi-timeframe CMF
    cmf_21d = float(calc_cmf(high, low, close, volume, 21).iloc[-1])
    cmf_63d = float(calc_cmf(high, low, close, volume, 63).iloc[-1])

    # 4. Relative volume
    vol_ma20 = volume.rolling(20).mean()
    rvol = float(volume.iloc[-1] / vol_ma20.iloc[-1]) if vol_ma20.iloc[-1] > 0 else 1.0
    rvol_high = rvol > 1.3

    # 5. NVI/PVI institutional signal
    nvi, pvi = calc_nvi_pvi(close, volume)
    inst_sig = calc_institutional_signal(nvi, pvi)
    inst_val = int(inst_sig.iloc[-1])
    inst_accum = inst_val == 1

    # 6. OBV divergence
    obv_bull_div, obv_bear_div = detect_obv_divergence(close, obv)

    # Composite score (0-6)
    score = (int(obv_above) + int(mfi_oversold) + int(cmf_positive) +
             int(rvol_high) + int(inst_accum) + int(obv_bull_div))

    # Recent returns
    ret_5d = float(close.pct_change(5).iloc[-1] * 100) if len(close) > 5 else 0.0
    ret_21d = float(close.pct_change(21).iloc[-1] * 100) if len(close) > 21 else 0.0
    last_price = float(close.iloc[-1])

    return {
        'score': score,
        'signal': 'BUY' if score >= 4 else ('SELL' if score <= 2 else 'HOLD'),
        'obv_above': obv_above,
        'mfi': round(mfi_val, 1),
        'mfi_oversold': mfi_oversold,
        'cmf_osc': round(cmf_osc_val, 4),
        'cmf_positive': cmf_positive,
        'cmf_21d': round(cmf_21d, 4) if not np.isnan(cmf_21d) else 0.0,
        'cmf_63d': round(cmf_63d, 4) if not np.isnan(cmf_63d) else 0.0,
        'rvol': round(rvol, 2),
        'rvol_high': rvol_high,
        'inst_signal': inst_val,
        'inst_accum': inst_accum,
        'obv_bull_div': obv_bull_div,
        'obv_bear_div': obv_bear_div,
        'ret_5d': round(ret_5d, 2),
        'ret_21d': round(ret_21d, 2),
        'price': round(last_price, 2),
    }


# =============================================================================
# PRICE CACHE LOADER (streaming - memory efficient)
# =============================================================================

def load_one_pkl(ticker):
    """Load a single ticker from price_cache."""
    path = CACHE_DIR / f'{ticker}.pkl'
    try:
        with open(path, 'rb') as f:
            df = pickle.load(f)
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date').sort_index()
            else:
                return None
        else:
            df = df.sort_index()
        required = ['open', 'high', 'low', 'close', 'volume']
        if not all(c in df.columns for c in required):
            return None
        return df
    except Exception:
        return None


def load_ticker_meta():
    """Load ticker metadata (name, type) from cache_metadata.json."""
    if not CACHE_META.exists():
        return {}
    with open(CACHE_META, encoding='utf-8') as f:
        raw = json.load(f).get('assets', {})
    return {t: {'name': v.get('fund_name', t), 'type': v.get('type', 'other')}
            for t, v in raw.items()}


# =============================================================================
# HTML GENERATION
# =============================================================================

EXTRA_CSS = """
.score-pill {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-weight: 700;
    font-size: 0.85em;
}
.score-6, .score-5 { background: #dcfce7; color: #166534; }
.score-4 { background: #d1fae5; color: #065f46; }
.score-3 { background: #fef9c3; color: #854d0e; }
.score-2 { background: #fee2e2; color: #991b1b; }
.score-1, .score-0 { background: #fecaca; color: #7f1d1d; }
.signal-buy { color: #16a34a; font-weight: 700; }
.signal-sell { color: #dc2626; font-weight: 700; }
.signal-hold { color: #ca8a04; font-weight: 700; }
.check { color: #16a34a; }
.cross { color: #dc2626; }
.div-bull { color: #16a34a; font-weight: 700; }
.div-bear { color: #dc2626; font-weight: 700; }
.filter-bar { margin: 12px 0; display: flex; gap: 8px; flex-wrap: wrap; }
.filter-btn {
    padding: 4px 12px; border-radius: 6px; border: 1px solid #d1d5db;
    background: #fff; cursor: pointer; font-size: 0.85em;
}
.filter-btn.active { background: #1e40af; color: #fff; border-color: #1e40af; }
.summary-cards { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 16px; }
.summary-card {
    background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px;
    padding: 12px 20px; min-width: 140px; text-align: center;
}
.summary-card .label { font-size: 0.8em; color: #64748b; }
.summary-card .value { font-size: 1.8em; font-weight: 700; }
"""

EXTRA_JS = """
function filterTable(col, val) {
    const rows = document.querySelectorAll('#sm-table tbody tr');
    const btns = document.querySelectorAll('.filter-btn');
    // Toggle active
    btns.forEach(b => {
        if (b.dataset.col === col && b.dataset.val === val) {
            b.classList.toggle('active');
        } else if (b.dataset.col === col) {
            b.classList.remove('active');
        }
    });
    const activeBtn = document.querySelector(`.filter-btn.active[data-col="${col}"]`);
    rows.forEach(r => {
        if (!activeBtn) { r.style.display = ''; return; }
        const cell = r.querySelector(`td[data-col="${col}"]`);
        if (cell && cell.textContent.trim() === val) {
            r.style.display = '';
        } else {
            r.style.display = 'none';
        }
    });
}
"""


def build_html(results, ticker_meta, as_of):
    """Build the dashboard HTML body."""
    # Sort by score desc, then ticker
    sorted_results = sorted(results.items(),
                            key=lambda x: (-x[1]['score'], x[0]))

    n_buy = sum(1 for _, d in sorted_results if d['signal'] == 'BUY')
    n_sell = sum(1 for _, d in sorted_results if d['signal'] == 'SELL')
    n_hold = sum(1 for _, d in sorted_results if d['signal'] == 'HOLD')
    avg_score = np.mean([d['score'] for _, d in sorted_results]) if sorted_results else 0

    # Summary cards
    html = f"""
    <div class="summary-cards">
        <div class="summary-card">
            <div class="label">Tickers Scored</div>
            <div class="value">{len(sorted_results)}</div>
        </div>
        <div class="summary-card">
            <div class="label">Avg Score</div>
            <div class="value">{avg_score:.1f}</div>
        </div>
        <div class="summary-card">
            <div class="label signal-buy">BUY (>=4)</div>
            <div class="value signal-buy">{n_buy}</div>
        </div>
        <div class="summary-card">
            <div class="label signal-hold">HOLD (3)</div>
            <div class="value signal-hold">{n_hold}</div>
        </div>
        <div class="summary-card">
            <div class="label signal-sell">SELL (<=2)</div>
            <div class="value signal-sell">{n_sell}</div>
        </div>
    </div>

    <p style="color:#64748b;font-size:0.85em;">
        As of {as_of} | Indicators: OBV trend, MFI oversold, CMF oscillator,
        relative volume, NVI/PVI institutional, OBV divergence |
        <strong>PENDING VALIDATION</strong> - methods not yet scimode-tested
    </p>

    <div class="filter-bar">
        <button class="filter-btn" data-col="signal" data-val="BUY"
                onclick="filterTable('signal','BUY')">BUY only</button>
        <button class="filter-btn" data-col="signal" data-val="SELL"
                onclick="filterTable('signal','SELL')">SELL only</button>
        <button class="filter-btn" data-col="signal" data-val="HOLD"
                onclick="filterTable('signal','HOLD')">HOLD only</button>
    </div>

    <table id="sm-table" class="dash-table sortable">
    <thead><tr>
        <th>Ticker</th><th>Name</th><th>Type</th>
        <th>Score</th><th>Signal</th>
        <th>OBV</th><th>MFI</th><th>CMF Osc</th>
        <th>RVOL</th><th>Inst</th><th>Div</th>
        <th>CMF 21d</th><th>CMF 63d</th>
        <th>5d %</th><th>21d %</th><th>Price</th>
    </tr></thead>
    <tbody>
    """

    for ticker, d in sorted_results:
        meta = ticker_meta.get(ticker, {})
        name = meta.get('name', ticker)[:25]
        atype = meta.get('type', '?')

        sig_cls = f'signal-{d["signal"].lower()}'
        score_cls = f'score-{d["score"]}'

        # Component check/cross marks
        obv_mark = '<span class="check">Y</span>' if d['obv_above'] else '<span class="cross">N</span>'
        mfi_mark = f'<span class="{"check" if d["mfi_oversold"] else "cross"}">{d["mfi"]:.0f}</span>'
        cmf_mark = f'<span class="{"check" if d["cmf_positive"] else "cross"}">{d["cmf_osc"]:.3f}</span>'
        rvol_mark = f'<span class="{"check" if d["rvol_high"] else "cross"}">{d["rvol"]:.1f}x</span>'
        inst_mark = '<span class="check">+1</span>' if d['inst_accum'] else (
            '<span class="cross">-1</span>' if d['inst_signal'] == -1 else '0')

        div_str = ''
        if d['obv_bull_div']:
            div_str = '<span class="div-bull">BULL</span>'
        elif d['obv_bear_div']:
            div_str = '<span class="div-bear">BEAR</span>'

        ret5_color = '#16a34a' if d['ret_5d'] > 0 else '#dc2626'
        ret21_color = '#16a34a' if d['ret_21d'] > 0 else '#dc2626'

        html += f"""<tr>
            <td><strong>{ticker}</strong></td>
            <td>{name}</td>
            <td>{atype}</td>
            <td><span class="score-pill {score_cls}">{d['score']}</span></td>
            <td data-col="signal" class="{sig_cls}">{d['signal']}</td>
            <td>{obv_mark}</td>
            <td>{mfi_mark}</td>
            <td>{cmf_mark}</td>
            <td>{rvol_mark}</td>
            <td>{inst_mark}</td>
            <td>{div_str}</td>
            <td>{d['cmf_21d']:.3f}</td>
            <td>{d['cmf_63d']:.3f}</td>
            <td style="color:{ret5_color}">{d['ret_5d']:+.1f}%</td>
            <td style="color:{ret21_color}">{d['ret_21d']:+.1f}%</td>
            <td>{d['price']:.2f}</td>
        </tr>"""

    html += "</tbody></table>"
    return html


# =============================================================================
# MAIN
# =============================================================================

def main():
    print('=' * 60)
    print('SMART MONEY DASHBOARD BACKEND v1.0')
    print('=' * 60)

    # Load ticker list
    tickers = sorted([p.stem for p in CACHE_DIR.glob('*.pkl')])
    print(f'  {len(tickers)} tickers in price_cache')

    ticker_meta = load_ticker_meta()

    # Score all tickers (streaming from disk)
    results = {}
    from tqdm import tqdm
    for ticker in tqdm(tickers, desc='  Scoring'):
        df = load_one_pkl(ticker)
        if df is None:
            continue
        data = score_ticker(df)
        if data is not None:
            results[ticker] = data

    n_buy = sum(1 for d in results.values() if d['signal'] == 'BUY')
    n_sell = sum(1 for d in results.values() if d['signal'] == 'SELL')
    print(f'  {len(results)} scored | BUY {n_buy} | SELL {n_sell}')

    # Determine as-of date from latest data
    sample_df = load_one_pkl('SPY')
    as_of = sample_df.index[-1].strftime('%Y-%m-%d') if sample_df is not None else date.today().isoformat()

    # Build HTML
    body = build_html(results, ticker_meta, as_of)

    # Write dashboard
    writer = DashboardWriter('smart-money', 'Smart Money Flow')
    writer.write(body, extra_css=EXTRA_CSS, extra_js=EXTRA_JS)
    print(f'\n[OK] Dashboard written')


if __name__ == '__main__':
    main()
