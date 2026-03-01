"""
momentum_ranker_v1_1.py
=======================
Momentum Ranker - Backend Data Generator

Scans all tickers in price_cache, applies hard filters, computes momentum
metrics, scores and ranks every qualifying ticker, outputs JSON + CSV.

Usage:
    python momentum_ranker_v1_1.py

Output:
    momentum_ranker_data.json   (root dir, run_daily.py copies to Django)
    momentum_ranker_data.csv    (root dir)

Author: Brian + Claude
Date: 2026-02-18
Version: 1.2

============================================================================
v1.2: Static HTML output via dashboard_writer.py
v1.1: Fixed NaN serialization - sanitize_record() replaces df.to_dict()
      which was writing literal NaN into JSON (invalid JSON)
v1.0: Initial build
============================================================================
"""

import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ============================================================================
# CONFIG
# ============================================================================

BASE_DIR        = Path(__file__).resolve().parent
DATA_DIR        = BASE_DIR.parent / 'perplexity-user-data'
PRICE_CACHE_DIR = DATA_DIR / 'price_cache'
OUTPUT_JSON     = BASE_DIR / 'momentum_ranker_data.json'
OUTPUT_CSV      = BASE_DIR / 'momentum_ranker_data.csv'

BENCHMARKS = ['SPY', 'QQQ', 'IAU']

MIN_PRICE = 20.0
MIN_ROWS  = 252

PERIODS = {
    '1d':  1,
    '1w':  5,
    '1m':  21,
    '3m':  63,
    '6m':  126,
    '1y':  252,
}

RATIO_THRESHOLDS = {
    '1w_1d': 3.0,
    '1m_1w': 3.0,
    '3m_1m': 2.0,
    '6m_3m': 1.0,
    '1y_3m': 2.0,
}

SMA_WINDOWS = [30, 50, 100, 200]

BAD_SPY_LOOKBACK = 20
BAD_SPY_COUNT    = 5

MAX_WORKERS = 6

# ============================================================================
# JSON SERIALIZATION HELPER
# ============================================================================

def sanitize_record(record: dict) -> dict:
    """
    Walk a record dict and convert any float NaN/Inf to None.
    Prevents json.dump from writing literal NaN (invalid JSON).
    """
    out = {}
    for k, v in record.items():
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            out[k] = None
        elif isinstance(v, (np.floating,)):
            # Catch numpy float types too
            if np.isnan(v) or np.isinf(v):
                out[k] = None
            else:
                out[k] = float(v)
        elif isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (np.bool_,)):
            out[k] = bool(v)
        else:
            out[k] = v
    return out

# ============================================================================
# LOAD
# ============================================================================

def load_from_cache(ticker: str):
    """Load single ticker from pickle. Returns (ticker, df) or (ticker, None)."""
    cache_file = PRICE_CACHE_DIR / f'{ticker}.pkl'
    if not cache_file.exists():
        return ticker, None
    try:
        with open(cache_file, 'rb') as f:
            df = pickle.load(f)

        if not isinstance(df.index, pd.DatetimeIndex):
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date').sort_index()
            else:
                return ticker, None
        else:
            df = df.sort_index()

        required = ['open', 'high', 'low', 'close', 'volume', 'adjClose']
        if not all(col in df.columns for col in required):
            return ticker, None

        return ticker, df

    except Exception:
        return ticker, None


def load_all_parallel(tickers: list) -> dict:
    """Load all tickers in parallel using ThreadPoolExecutor."""
    cache = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(load_from_cache, t): t for t in tickers}
        for future in tqdm(as_completed(futures), total=len(futures), desc='Loading cache'):
            ticker, df = future.result()
            cache[ticker] = df
    return cache

# ============================================================================
# METRIC HELPERS
# ============================================================================

def safe_return(df: pd.DataFrame, n_bars: int) -> float:
    """Percent return over last n_bars. NaN-safe."""
    if len(df) <= n_bars:
        return np.nan
    p_now  = df['close'].iloc[-1]
    p_then = df['close'].iloc[-(n_bars + 1)]
    if p_then == 0 or np.isnan(p_then):
        return np.nan
    return (p_now - p_then) / p_then


def safe_ratio(num: float, den: float) -> float:
    if np.isnan(num) or np.isnan(den) or den == 0:
        return np.nan
    return num / den


def compute_sma_flags(df: pd.DataFrame) -> dict:
    price = df['close'].iloc[-1]
    flags = {}
    for w in SMA_WINDOWS:
        if len(df) < w:
            flags[f'sma{w}'] = False
        else:
            flags[f'sma{w}'] = bool(price > df['close'].iloc[-w:].mean())
    flags['all_above'] = all(flags[f'sma{w}'] for w in SMA_WINDOWS)
    return flags


def compute_bad_spy_score(df: pd.DataFrame, spy_df: pd.DataFrame) -> float:
    """Avg ticker return on the BAD_SPY_COUNT worst SPY days in last BAD_SPY_LOOKBACK sessions."""
    try:
        spy_ret    = spy_df['close'].iloc[-BAD_SPY_LOOKBACK:].pct_change().dropna()
        worst_dates = spy_ret.nsmallest(BAD_SPY_COUNT).index
        ticker_ret  = df['close'].pct_change()
        scores = [ticker_ret.loc[d] for d in worst_dates
                  if d in ticker_ret.index and not np.isnan(ticker_ret.loc[d])]
        return float(np.mean(scores)) if scores else np.nan
    except Exception:
        return np.nan

# ============================================================================
# MAIN COMPUTATION
# ============================================================================

def compute_all_metrics(price_cache: dict, benchmark_cache: dict) -> pd.DataFrame:
    spy_df = benchmark_cache.get('SPY')

    rows = []
    skipped_noload = skipped_rows = skipped_price = 0

    for ticker, df in tqdm(price_cache.items(), desc='Computing metrics'):

        if ticker in BENCHMARKS:
            continue

        if df is None:
            skipped_noload += 1
            continue

        if len(df) < MIN_ROWS:
            skipped_rows += 1
            continue

        price = df['close'].iloc[-1]
        if np.isnan(price) or price < MIN_PRICE:
            skipped_price += 1
            continue

        row = {'ticker': ticker, 'price': round(price, 2)}

        # Returns
        returns = {label: safe_return(df, n) for label, n in PERIODS.items()}
        for label, r in returns.items():
            row[f'ret_{label}'] = round(r * 100, 2) if not np.isnan(r) else None

        # Ratios
        ratio_map = {
            '1w_1d': safe_ratio(returns['1w'], returns['1d']),
            '1m_1w': safe_ratio(returns['1m'], returns['1w']),
            '3m_1m': safe_ratio(returns['3m'], returns['1m']),
            '6m_3m': safe_ratio(returns['6m'], returns['3m']),
            '1y_3m': safe_ratio(returns['1y'], returns['3m']),
        }
        ratio_passes = 0
        for key, val in ratio_map.items():
            row[f'ratio_{key}'] = round(val, 2) if not np.isnan(val) else None
            if not np.isnan(val) and val >= RATIO_THRESHOLDS[key]:
                ratio_passes += 1
        row['ratio_passes'] = ratio_passes

        # SMAs
        sma_flags = compute_sma_flags(df)
        for w in SMA_WINDOWS:
            row[f'sma{w}']  = sma_flags[f'sma{w}']
        row['sma_all'] = sma_flags['all_above']

        # Benchmark deltas
        bm_periods = [('1w', 5), ('1m', 21), ('1y', 252)]
        for bm in BENCHMARKS:
            bm_df = benchmark_cache.get(bm)
            for label, n in bm_periods:
                t_ret  = safe_return(df, n)
                bm_ret = safe_return(bm_df, n) if bm_df is not None else np.nan
                delta  = (t_ret - bm_ret) if not (np.isnan(t_ret) or np.isnan(bm_ret)) else None
                row[f'vs_{bm.lower()}_{label}'] = round(delta * 100, 2) if delta is not None else None

        # Bad SPY days
        bsd = compute_bad_spy_score(df, spy_df) if spy_df is not None else np.nan
        row['bad_spy_score'] = round(bsd * 100, 4) if not np.isnan(bsd) else None

        rows.append(row)

    print(f'\n  Qualified: {len(rows)}')
    print(f'  Skipped (no load): {skipped_noload}')
    print(f'  Skipped (< {MIN_ROWS} rows): {skipped_rows}')
    print(f'  Skipped (price < ${MIN_PRICE}): {skipped_price}')

    return pd.DataFrame(rows)


def add_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Percentile-rank 3 components, apply SMA gate, compute final score 0-100."""

    def pct_rank(s: pd.Series) -> pd.Series:
        return s.rank(pct=True, na_option='bottom') * 100

    # Component 1: equal-weight avg of return percentiles across all 6 periods
    ret_cols = [f'ret_{p}' for p in PERIODS.keys()]
    ret_pct  = pd.DataFrame({c: pct_rank(pd.to_numeric(df[c], errors='coerce')) for c in ret_cols})
    df['score_returns'] = ret_pct.mean(axis=1)

    # Component 2: ratio passes count percentile
    df['score_ratios'] = pct_rank(pd.to_numeric(df['ratio_passes'], errors='coerce'))

    # Component 3: bad SPY days percentile
    df['score_spy_days'] = pct_rank(pd.to_numeric(df['bad_spy_score'], errors='coerce'))

    raw_score = (df['score_returns'] + df['score_ratios'] + df['score_spy_days']) / 3.0

    # SMA gate
    df['score'] = np.where(df['sma_all'], raw_score, 0.0).round(1)
    df['rank']  = df['score'].rank(ascending=False, method='min').astype(int)

    return df.sort_values('rank').reset_index(drop=True)

# ============================================================================
# ENTRY POINT
# ============================================================================



# ============================================================================
# HTML BUILDER
# ============================================================================

EXTRA_CSS = """
.mr-table-wrap { overflow-x: auto; padding: 0; }
.mr-table {
    border-collapse: collapse;
    width: 100%;
    font-size: 0.8em;
    table-layout: fixed;
}
.mr-table colgroup col.col-rank   { width: 48px; }
.mr-table colgroup col.col-ticker { width: 72px; }
.mr-table colgroup col.col-score  { width: 68px; }
.mr-table colgroup col.col-price  { width: 68px; }
.mr-table colgroup col.col-sma    { width: 44px; }
.mr-table colgroup col.col-ret    { width: 62px; }
.mr-table colgroup col.col-ratio  { width: 52px; }
.mr-table colgroup col.col-vs     { width: 62px; }
.mr-table colgroup col.col-bsd    { width: 64px; }
.mr-table thead th {
    padding: 7px 5px;
    font-size: 0.72em;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    cursor: pointer;
    user-select: none;
    border-bottom: 2px solid #e2e4e8;
    white-space: nowrap;
    text-align: center;
    overflow: hidden;
    text-overflow: ellipsis;
    background: #f8f9fb;
}
.mr-table thead th.col-left { text-align: left; }
.mr-table thead tr.grp-row th {
    padding: 5px 4px;
    font-size: 0.72em;
    font-weight: 700;
    letter-spacing: 0.05em;
}
.mr-table tbody td {
    padding: 5px 5px;
    border-bottom: 1px solid #f0f0f0;
    vertical-align: middle;
    text-align: center;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.92em;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.mr-table tbody td.col-left {
    text-align: left;
    font-family: 'IBM Plex Sans', sans-serif;
    font-weight: 600;
}
.mr-table tbody tr:hover { background: #f5f6ff !important; }
.score-badge {
    display: inline-block;
    padding: 2px 7px;
    border-radius: 4px;
    font-size: 0.88em;
    font-weight: 700;
    font-family: 'IBM Plex Mono', monospace;
}
.cell-miss { color: #ccc !important; background: #fafafa !important; }
.filter-bar {
    background: #fff;
    border: 1px solid #e2e4e8;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 16px;
    display: flex;
    gap: 20px;
    align-items: center;
    flex-wrap: wrap;
}
.filter-bar label {
    font-size: 0.82em;
    font-weight: 600;
    color: #555;
    display: flex;
    align-items: center;
    gap: 8px;
}
.filter-bar input[type=text] {
    padding: 5px 10px;
    border: 1px solid #d0d0d8;
    border-radius: 5px;
    font-size: 0.85em;
    font-family: 'IBM Plex Mono', monospace;
    width: 120px;
}
.filter-bar input[type=range] { width: 100px; }
.filter-bar select {
    padding: 5px 8px;
    border: 1px solid #d0d0d8;
    border-radius: 5px;
    font-size: 0.85em;
}
.filter-count {
    margin-left: auto;
    font-size: 0.82em;
    color: #888;
    font-family: 'IBM Plex Mono', monospace;
}
.grp-blank   { background: #f8f9fb !important; color: transparent; border-bottom: 1px solid #e2e4e8; }
.grp-returns { background: #1e293b !important; color: #94a3b8; }
.grp-vs-spy  { background: #14532d !important; color: #86efac; }
.grp-other   { background: #1e3a5f !important; color: #93c5fd; }
"""

# JS is a plain string — no .format() substitution needed
# We splice in data_json manually in build_body_html
RENDER_JS_HEAD = """(function() {
    const DATA = """

RENDER_JS_TAIL = """;

    function getGradientColor(norm01) {
        const c = Math.max(0, Math.min(1, norm01));
        let r, g, b;
        if (c < 0.25) {
            const t=c*4; r=Math.round(211+(255-211)*t); g=Math.round(47+(152-47)*t); b=Math.round(47*(1-t));
        } else if (c < 0.5) {
            const t=(c-0.25)*4; r=255; g=Math.round(152+(235-152)*t); b=Math.round(59*t);
        } else if (c < 0.75) {
            const t=(c-0.5)*4; r=Math.round(255+(139-255)*t); g=Math.round(235+(195-235)*t); b=Math.round(59+(74-59)*t);
        } else {
            const t=(c-0.75)*4; r=Math.round(139*(1-t)); g=Math.round(195+(200-195)*t); b=Math.round(74+(83-74)*t);
        }
        return `rgb(${r},${g},${b})`;
    }
    function textColor(rgb) {
        const m = rgb.match(/rgb[(]([0-9]+),([0-9]+),([0-9]+)[)]/);
        if (!m) return '#000';
        return (0.299*m[1]+0.587*m[2]+0.114*m[3]) > 140 ? '#000' : '#fff';
    }

    const RET_COLS = ['ret_1d','ret_1w','ret_1m','ret_3m','ret_6m','ret_1y'];
    const VS_COLS  = ['vs_spy_1w','vs_spy_1m','vs_spy_1y'];
    const GRAD_COLS = [...RET_COLS, ...VS_COLS, 'bad_spy_score'];

    const colRanges = {};
    GRAD_COLS.forEach(col => {
        const vals = DATA.data.map(r => r[col]).filter(v => v != null && !isNaN(v));
        if (!vals.length) return;
        const s = [...vals].sort((a,b) => a-b);
        colRanges[col] = {
            min: s[Math.floor(s.length*0.05)],
            max: s[Math.floor(s.length*0.95)]
        };
    });

    function normVal(val, col) {
        if (val == null || isNaN(val)) return null;
        const r = colRanges[col];
        if (!r || r.max === r.min) return 0.5;
        return Math.max(0, Math.min(1, (val - r.min) / (r.max - r.min)));
    }
    function gradTd(val, col, dec) {
        dec = dec || 1;
        if (val == null) return '<td class="cell-miss">&#8212;</td>';
        const n = normVal(val, col);
        const bg = getGradientColor(n);
        const fg = textColor(bg);
        const sign = val >= 0 ? '+' : '';
        return `<td style="background:${bg};color:${fg};">${sign}${val.toFixed(dec)}%</td>`;
    }

    let sortCol = 'rank', sortDir = 'asc';
    let filterText = '', filterSma = 'all', filterMinScore = 0;

    function getSortVal(row, col) {
        if (col === 'ticker') return row.ticker || '';
        if (col === 'sma_all') return row.sma_all ? 1 : 0;
        return row[col] != null ? Number(row[col]) : -9999;
    }
    function rowVisible(row) {
        if (filterText && !row.ticker.toLowerCase().includes(filterText.toLowerCase())) return false;
        if (filterSma === 'above' && !row.sma_all) return false;
        if (filterSma === 'below' && row.sma_all) return false;
        if ((row.score || 0) < filterMinScore) return false;
        return true;
    }

    function renderTable() {
        let rows = [...DATA.data];
        rows.sort((a,b) => {
            const va = getSortVal(a, sortCol), vb = getSortVal(b, sortCol);
            if (typeof va === 'string') return sortDir==='asc' ? va.localeCompare(vb) : vb.localeCompare(va);
            return sortDir==='asc' ? va-vb : vb-va;
        });
        const visible = rows.filter(rowVisible);
        document.getElementById('filter-count').textContent =
            visible.length + ' of ' + DATA.universe_size + ' tickers';

        const arrow = col => sortCol===col ? (sortDir==='asc' ? ' \u25b2' : ' \u25bc') : '';
        const th = (col, lbl, cls) =>
            `<th class="${cls||''}" onclick="window._mrSort('${col}')">${lbl}${arrow(col)}</th>`;

        let html = `<table class="mr-table"><colgroup>
  <col class="col-rank"><col class="col-ticker"><col class="col-score">
  <col class="col-price"><col class="col-sma">
  <col class="col-ret"><col class="col-ret"><col class="col-ret">
  <col class="col-ret"><col class="col-ret"><col class="col-ret">
  <col class="col-ratio">
  <col class="col-vs"><col class="col-vs"><col class="col-vs">
  <col class="col-bsd">
</colgroup><thead>
<tr class="grp-row">
  <th class="grp-blank" colspan="5"></th>
  <th class="grp-returns" colspan="6">Returns</th>
  <th class="grp-blank"></th>
  <th class="grp-vs-spy" colspan="3">vs SPY</th>
  <th class="grp-other">Bad SPY</th>
</tr><tr>
  ${th('rank','Rank')} ${th('ticker','Ticker','col-left')} ${th('score','Score')}
  ${th('price','Price')} ${th('sma_all','SMA')}
  ${th('ret_1d','1d')} ${th('ret_1w','1w')} ${th('ret_1m','1m')}
  ${th('ret_3m','3m')} ${th('ret_6m','6m')} ${th('ret_1y','1y')}
  ${th('ratio_passes','Ratio')}
  ${th('vs_spy_1w','1w')} ${th('vs_spy_1m','1m')} ${th('vs_spy_1y','1y')}
  ${th('bad_spy_score','Score')}
</tr></thead><tbody>`;

        visible.forEach(row => {
            const sc   = row.score || 0;
            const scBg = getGradientColor(sc / 100);
            const scFg = textColor(scBg);
            const smaTd = row.sma_all
                ? '<td style="color:#16a34a;font-weight:700;">&#10003;</td>'
                : '<td style="color:#ccc;">&#8212;</td>';
            const rp   = row.ratio_passes || 0;
            const rpBg = rp >= 4 ? '#16a34a' : rp >= 2 ? '#d97706' : '#dc2626';
            html += `<tr>
  <td>${row.rank}</td>
  <td class="col-left">${row.ticker}</td>
  <td><span class="score-badge" style="background:${scBg};color:${scFg};">${sc.toFixed(1)}</span></td>
  <td>$${(row.price||0).toFixed(2)}</td>
  ${smaTd}
  ${gradTd(row.ret_1d,'ret_1d')}  ${gradTd(row.ret_1w,'ret_1w')}
  ${gradTd(row.ret_1m,'ret_1m')}  ${gradTd(row.ret_3m,'ret_3m')}
  ${gradTd(row.ret_6m,'ret_6m')}  ${gradTd(row.ret_1y,'ret_1y')}
  <td style="background:${rpBg};color:#fff;font-weight:700;">${rp}/5</td>
  ${gradTd(row.vs_spy_1w,'vs_spy_1w')}
  ${gradTd(row.vs_spy_1m,'vs_spy_1m')}
  ${gradTd(row.vs_spy_1y,'vs_spy_1y')}
  ${gradTd(row.bad_spy_score,'bad_spy_score',2)}
</tr>`;
        });
        html += '</tbody></table>';
        document.getElementById('mr-table-wrap').innerHTML = html;
    }

    window._mrSort = function(col) {
        if (sortCol === col) { sortDir = sortDir==='asc' ? 'desc' : 'asc'; }
        else { sortCol = col; sortDir = col==='rank' ? 'asc' : 'desc'; }
        renderTable();
    };

    document.getElementById('filter-text').addEventListener('input', e => {
        filterText = e.target.value; renderTable();
    });
    document.getElementById('filter-sma').addEventListener('change', e => {
        filterSma = e.target.value; renderTable();
    });
    document.getElementById('filter-score').addEventListener('input', e => {
        filterMinScore = Number(e.target.value);
        document.getElementById('score-val').textContent = e.target.value;
        renderTable();
    });

    renderTable();
})();
"""


def build_body_html(output, writer):
    import json as _json

    data   = output['data']
    gen    = output['generated_at'][:10]
    n      = output['universe_size']
    top    = data[0] if data else {}
    top_lbl = '{} {:.1f}'.format(top.get('ticker','—'), top.get('score', 0)) if top else '—'
    sma_ct  = sum(1 for r in data if r.get('sma_all'))

    parts = []
    parts.append(writer.stat_bar([
        ('Generated',      gen,          'neutral'),
        ('Universe',       str(n),       'neutral'),
        ('Above All SMAs', str(sma_ct),  'pos' if sma_ct > 0 else 'neutral'),
        ('#1 Ticker',      top_lbl,      'pos'),
    ]))
    parts.append(writer.build_header(
        'Ranked by composite momentum score \u2014 {} tickers scanned'.format(n)
    ))

    filter_html = (
        '<div class="filter-bar">'
        '<label>Search <input type="text" id="filter-text" placeholder="ticker..."></label>'
        '<label>SMA Gate '
        '<select id="filter-sma">'
        '<option value="all">All</option>'
        '<option value="above">Above all SMAs only</option>'
        '<option value="below">Below SMAs only</option>'
        '</select></label>'
        '<label>Min Score '
        '<input type="range" id="filter-score" min="0" max="100" value="0" step="1">'
        '<span id="score-val" style="font-family:\'IBM Plex Mono\',monospace;'
        'font-size:0.88em;margin-left:4px;">0</span>'
        '</label>'
        '<span class="filter-count" id="filter-count"></span>'
        '</div>'
    )

    table_html = (
        '<div class="table-section">'
        '<div class="table-section-header">'
        '<h2>Momentum Rankings</h2>'
        '<span style="font-size:0.85em;color:#888;">'
        'Click column headers to sort'
        ' &nbsp;|&nbsp; Score = percentile composite: returns + ratio passes + bad-SPY resilience'
        ' &nbsp;|&nbsp; SMA \u2713 = price above all SMAs (30/50/100/200)'
        ' &nbsp;|&nbsp; Ratio = short-term momentum consistency (out of 5 tests)'
        '</span></div>'
        + filter_html
        + '<div class="mr-table-wrap" id="mr-table-wrap"></div>'
        '</div>'
    )
    parts.append(table_html)
    parts.append(writer.footer())

    data_json = _json.dumps(output, ensure_ascii=False, default=str)
    writer._mr_extra_js = RENDER_JS_HEAD + data_json + RENDER_JS_TAIL

    return '\n'.join(parts)


def main():
    print('=' * 80)
    print('Momentum Ranker v1.2')
    print(f'Run: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print('=' * 80)

    all_tickers = [f.stem for f in sorted(PRICE_CACHE_DIR.glob('*.pkl'))]
    print(f'\nFound {len(all_tickers)} tickers in price_cache')

    price_cache     = load_all_parallel(all_tickers)
    benchmark_cache = {bm: price_cache.get(bm) for bm in BENCHMARKS}
    for bm, bm_df in benchmark_cache.items():
        print(f'  Benchmark {bm}: {"OK" if bm_df is not None else "MISSING"}')

    df = compute_all_metrics(price_cache, benchmark_cache)
    if df.empty:
        print('\nERROR: No tickers passed filters.')
        return

    df = add_scores(df)
    print(f'\nFinal ranked universe: {len(df)} tickers')

    records = [sanitize_record(r) for r in df.to_dict(orient='records')]
    output = {
        'generated_at':     datetime.now().isoformat(),
        'universe_size':    len(df),
        'filters':          {'min_price': MIN_PRICE, 'min_rows': MIN_ROWS, 'sma_windows': SMA_WINDOWS},
        'ratio_thresholds': RATIO_THRESHOLDS,
        'columns':          list(df.columns),
        'data':             records,
    }

    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(output, f, allow_nan=False)
    print(f'JSON saved: {OUTPUT_JSON}')
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
    print(f'CSV  saved: {OUTPUT_CSV}')

    try:
        from dashboard_writer import DashboardWriter
    except ImportError:
        print('ERROR: dashboard_writer.py not found — skipping HTML output')
        return

    writer   = DashboardWriter('momentum-ranker', 'Momentum Ranker')
    body     = build_body_html(output, writer)
    extra_js = getattr(writer, '_mr_extra_js', '')
    writer.write(body, extra_css=EXTRA_CSS, extra_js=extra_js)

    preview = ['rank','ticker','score','price','ret_1w','ret_1m','ret_3m','ret_1y','sma_all','ratio_passes']
    avail   = [c for c in preview if c in df.columns]
    print(f'\nTop 10:')
    print(df[avail].head(10).to_string(index=False))
    print('\n' + '=' * 80)
    print('SUCCESS')
    print('=' * 80)


if __name__ == '__main__':
    main()
