"""
momentum_ranker_v1_4.py
=======================
Momentum Ranker - Backend Data Generator

Scans all tickers in price_cache, applies hard filters, computes momentum
metrics, scores and ranks every qualifying ticker, outputs JSON + CSV.

Usage:
    python momentum_ranker_v1_4.py

Output:
    momentum_ranker_data.json   (root dir, run_daily.py copies to Django)
    momentum_ranker_data.csv    (root dir)

Author: Brian + Claude
Date: 2026-02-24
Version: 1.7

============================================================================
v1.18: Staleness filter: skip tickers with cache older than MAX_CACHE_AGE_DAYS (7d)
       Removes delisted/acquired tickers (KSU, HES, PXD, MRO, ANTM, JWN etc.)
       MIN_TOP_POLE_CORR=0.30: suppress noise assignments (FRO->QQQ, SLGL->TLT etc.)
       Removed [NO CORR] debug print
v1.17: TOP POLE column replaces PRICE in table (pole name + corr, bg by pole 21d z-score)
       Price moved to tooltip popup header
       top_pole and top_pole_corr added to row data (Python argmax of corr_vec)
v1.16: Pole panel row 4: Alignment score (corr_avg * z_score, normalised 0-100%
       using 5-95 pct scale, green=positive alignment, red=risk exposure)
v1.15: Pole panel row 3: dark/light bg interpolation by correlation strength
       high corr (#1e293b dark) -> low corr (#e2e8f0 light), text flips at 0.5
v1.14: Added pole momentum panel (3-row table between stats bar and title)
       - Row 1: CORR_UNIVERSE tickers as headers
       - Row 2: 21-day momentum z-score per pole (5-zone bg coloring)
       - Row 3: Average corr_vec across owned tickers per pole (live JS, updates on checkbox toggle)
       - pole_momentum and corr_universe added to output JSON
v1.13: M column: 5-zone background color by z-score
       green (+0.5 to +1.5), orange (+1.5 to +2.0 and -0.5 to +0.5),
       red (> +2.0 and < -0.5). White text on all zones.
v1.12: M column now shows numeric z-score instead of colored dot
v1.11: Added theme overlap detection (OL column, fourth signal dot)
       - corr_vec: 20-dim correlation fingerprint stored per ticker in JSON
       - Cosine similarity computed in JS between each ticker and owned set
       - Threshold dropdown in filter bar (0.70/0.75/0.80/0.85/0.90)
       - OL dot: amber=overlap with owned ticker, dark=clear
       - Tooltip shows which owned tickers overlap and similarity scores
       - No overlap dots fire when owned set is empty
v1.10: SPY, QQQ, IAU now rank in dashboard alongside all other tickers
v1.8: Rebuilt pullback signal as same-day entry timing alert
v1.7: Added pullback buy alert (PB column, third signal dot)
      Each column has its own sort, header tooltip, and single dot
v1.5: Added column header tooltips explaining every non-obvious column
      (Score, SMA, Ratio, Bad SPY, vs SPY, Sig/M dots)
      Extended th() JS helper to accept optional title attribute
      Updated legend dot spans with title attributes
v1.4: Added volume confirmation signal
      - vol_ratio: 5-day avg volume / 63-day avg volume
      - vol_flag: 'confirmed' (>1.5x) | 'normal' | 'thin' (<0.75x)
      - Second dot in Sig column; volume section in hover tooltip
      - Stat bar: Vol Confirmed count
v1.3: Added correlation tooltip (top 3 / bottom 3 vs 20 reference assets)
      Added momentum character signal (3-day z-score vs 63-day distribution)
      - momentum_flag: 'clean' | 'stalling' | 'overextended'
      - momentum_signal: float z-score
      - corr_top / corr_bot: list of {asset, corr} dicts
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
from datetime import datetime, date
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ============================================================================
# CONFIG
# ============================================================================

BASE_DIR        = Path(__file__).resolve().parent
DATA_DIR        = BASE_DIR.parent / 'perplexity-user-data'
PRICE_CACHE_DIR = DATA_DIR / 'price_cache'
OUTPUT_JSON     = BASE_DIR / 'momentum_ranker_data.json'
OUTPUT_CSV      = BASE_DIR / 'momentum_ranker_data_{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M'))

BENCHMARKS = ['SPY', 'QQQ', 'IAU']

MIN_PRICE = 20.0
MIN_ROWS  = 252
MAX_CACHE_AGE_DAYS = 7    # exclude tickers whose cache is older than this (likely delisted)
MIN_TOP_POLE_CORR  = 0.30 # below this, top_pole assignment is noise -- show -- instead

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

# Reference assets for correlation tooltip
# 63-day correlation computed against each; top 3 and bottom 3 shown on hover
CORR_UNIVERSE = [
    'QQQ', 'SPY',
    'XLB', 'XLC', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLRE', 'XLU', 'XLV', 'XLY',
    'TLT', 'HYG', 'LQD',
    'IAU',
    'VXUS', 'VGK', 'DBA',
]

CORR_LOOKBACK = 63   # trading days for correlation window

# Momentum character signal thresholds (3-day return z-score vs 63-day dist)
MOMENTUM_Z_OVERBOUGHT =  2.0   # above this -> overextended (don't chase)
MOMENTUM_Z_STALLING   = -1.5   # below this -> stalling (wait)

# Volume confirmation thresholds
VOL_SHORT = 5    # recent volume window (trading days)
VOL_LONG  = 63   # baseline volume window (trading days)
VOL_CONFIRMED_THRESHOLD = 1.5   # above this -> confirmed
VOL_THIN_THRESHOLD      = 0.75  # below this -> thin

# Pullback entry timing alert (v1.8)
# Fires on down days in strong momentum stocks — same-day entry signal
PULLBACK_MIN_SCORE  = 60.0  # minimum score gate (applied post add_scores)
PULLBACK_MIN_RET_3M = 0.0   # 3m return must be positive

# ============================================================================
# JSON SERIALIZATION HELPER
# ============================================================================

def sanitize_record(record: dict) -> dict:
    """
    Walk a record dict and convert any non-JSON-serializable value to a
    JSON-safe equivalent. Handles numpy scalars, NaN, Inf, nested lists.
    """
    def clean_val(v):
        # numpy booleans
        if isinstance(v, (np.bool_,)):
            return bool(v)
        # numpy integers
        if isinstance(v, (np.integer,)):
            return int(v)
        # numpy floats (includes NaN/Inf)
        if isinstance(v, (np.floating, float)):
            if np.isnan(v) or np.isinf(v):
                return None
            return float(v)
        # nested list (corr_top, corr_bot, corr_vec)
        if isinstance(v, list):
            return [clean_val(item) for item in v]
        # nested dict (items inside corr_top/corr_bot)
        if isinstance(v, dict):
            return {ik: clean_val(iv) for ik, iv in v.items()}
        return v

    return {k: clean_val(v) for k, v in record.items()}

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

        required = ['open', 'high', 'low', 'close', 'volume']  # adjClose excluded: not used in calculations
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
# NEW: CORRELATION HELPERS
# ============================================================================

def build_corr_matrix(price_cache: dict) -> pd.DataFrame:
    """
    Build a returns DataFrame for all CORR_UNIVERSE assets over CORR_LOOKBACK days.
    Returns a DataFrame of daily returns, indexed by date, columns = asset tickers.
    Missing assets are silently dropped.
    """
    series = {}
    for asset in CORR_UNIVERSE:
        df = price_cache.get(asset)
        if df is None or len(df) < CORR_LOOKBACK + 5:
            continue
        ret = df['close'].iloc[-(CORR_LOOKBACK + 1):].pct_change().dropna()
        series[asset] = ret

    if not series:
        return pd.DataFrame()

    return pd.DataFrame(series).dropna()


def compute_correlations(ticker_returns: pd.Series, corr_ref: pd.DataFrame) -> list:
    """
    Compute Pearson correlation between ticker_returns and each column in corr_ref.
    Aligns on shared dates. Returns list of {asset, corr} sorted descending.
    """
    results = []
    for asset in corr_ref.columns:
        shared = ticker_returns.index.intersection(corr_ref.index)
        if len(shared) < 20:
            continue
        t = ticker_returns.loc[shared]
        r = corr_ref[asset].loc[shared]
        if t.std() == 0 or r.std() == 0:
            continue
        c = float(np.corrcoef(t.values, r.values)[0, 1])
        if not np.isnan(c):
            results.append({'asset': asset, 'corr': round(c, 3)})

    results.sort(key=lambda x: x['corr'], reverse=True)
    return results


# ============================================================================
# NEW: MOMENTUM CHARACTER SIGNAL
# ============================================================================

def compute_momentum_signal(df: pd.DataFrame) -> tuple:
    """
    Compute 3-day return z-score vs the distribution of all 3-day returns
    over the last 63 trading days.

    Returns:
        (momentum_signal: float, momentum_flag: str)
        flag is one of: 'clean', 'stalling', 'overextended'
    """
    try:
        if len(df) < CORR_LOOKBACK + 5:
            return np.nan, 'clean'

        close = df['close'].iloc[-(CORR_LOOKBACK + 5):]
        daily_ret = close.pct_change().dropna()

        # Build all rolling 3-day returns over the lookback window
        three_day_rets = []
        for i in range(2, len(daily_ret)):
            r = (1 + daily_ret.iloc[i-2]) * (1 + daily_ret.iloc[i-1]) * (1 + daily_ret.iloc[i]) - 1
            three_day_rets.append(r)

        if len(three_day_rets) < 10:
            return np.nan, 'clean'

        dist = np.array(three_day_rets)
        mu   = float(np.mean(dist))
        sig  = float(np.std(dist))

        if sig == 0:
            return 0.0, 'clean'

        # Most recent 3-day return
        recent = three_day_rets[-1]
        z = (recent - mu) / sig
        z = round(float(z), 2)

        if z > MOMENTUM_Z_OVERBOUGHT:
            flag = 'overextended'
        elif z < MOMENTUM_Z_STALLING:
            flag = 'stalling'
        else:
            flag = 'clean'

        return z, flag

    except Exception:
        return np.nan, 'clean'


# ============================================================================
# POLE MOMENTUM SIGNAL (21-day z-score for CORR_UNIVERSE assets)
# ============================================================================

def compute_pole_momentum(df: pd.DataFrame) -> float:
    """
    Compute 21-day return z-score for a pole asset vs distribution of all
    rolling 21-day returns over the last 63 trading days.

    Returns:
        z-score (float) or None if insufficient data
    """
    try:
        needed = CORR_LOOKBACK + 25
        if len(df) < needed:
            return None

        close = df['close'].iloc[-needed:]
        daily_ret = close.pct_change().dropna()

        # Build all rolling 21-day returns
        window = 21
        twenty_one_day_rets = []
        for i in range(window - 1, len(daily_ret)):
            chunk = daily_ret.iloc[i - window + 1: i + 1]
            r = float((1 + chunk).prod() - 1)
            twenty_one_day_rets.append(r)

        if len(twenty_one_day_rets) < 10:
            return None

        dist = np.array(twenty_one_day_rets)
        mu   = float(np.mean(dist))
        sig  = float(np.std(dist))

        if sig == 0:
            return 0.0

        recent = twenty_one_day_rets[-1]
        z = round((recent - mu) / sig, 2)
        return z

    except Exception:
        return None


# ============================================================================
# NEW: VOLUME CONFIRMATION SIGNAL
# ============================================================================

def compute_volume_signal(df: pd.DataFrame) -> tuple:
    """
    Compare average daily volume over last VOL_SHORT days vs last VOL_LONG days.
    Returns:
        (vol_ratio: float, vol_flag: str)
        flag is one of: 'confirmed', 'normal', 'thin'
    """
    try:
        if len(df) < VOL_LONG + 5:
            return np.nan, 'normal'

        vol = df['volume'].iloc[-(VOL_LONG + 5):]
        vol = vol.replace(0, np.nan).dropna()

        if len(vol) < VOL_LONG:
            return np.nan, 'normal'

        avg_short = float(vol.iloc[-VOL_SHORT:].mean())
        avg_long  = float(vol.iloc[-VOL_LONG:].mean())

        if avg_long == 0 or np.isnan(avg_long) or np.isnan(avg_short):
            return np.nan, 'normal'

        ratio = round(avg_short / avg_long, 3)

        if ratio >= VOL_CONFIRMED_THRESHOLD:
            flag = 'confirmed'
        elif ratio <= VOL_THIN_THRESHOLD:
            flag = 'thin'
        else:
            flag = 'normal'

        return ratio, flag

    except Exception:
        return np.nan, 'normal'


# ============================================================================
# PULLBACK ALERT SIGNAL (v1.7)
# ============================================================================

# ============================================================================
# PULLBACK ENTRY TIMING SIGNAL (v1.8)
# ============================================================================

def compute_pullback_signal(row: dict) -> str:
    """
    Same-day entry timing alert for momentum stocks.
    Fires when the stock is down today but the broader trend is intact.
    Vol confirmed is excluded — surging volume on a down day = distribution.
    Score gate (>=60) is applied as a second pass in add_scores().

    Returns: 'buy_today' or 'none'
    """
    try:
        down_today   = (row.get('ret_1d') or 0.0) < 0
        sma_ok       = row.get('sma_all', False)
        momentum_ok  = row.get('momentum_flag', '') == 'clean'
        trend_ok     = (row.get('ret_3m') or 0.0) > PULLBACK_MIN_RET_3M
        not_dist     = row.get('vol_flag', '') != 'confirmed'  # no distribution

        if down_today and sma_ok and momentum_ok and trend_ok and not_dist:
            return 'buy_today'
        return 'none'
    except Exception:
        return 'none'


# ============================================================================
# MAIN COMPUTATION
# ============================================================================

def compute_all_metrics(price_cache: dict, benchmark_cache: dict) -> pd.DataFrame:
    spy_df = benchmark_cache.get('SPY')

    # Pre-build correlation reference matrix once (not per ticker)
    print('\nBuilding correlation reference matrix...')
    corr_ref = build_corr_matrix(price_cache)
    available_assets = list(corr_ref.columns) if not corr_ref.empty else []
    print(f'  Correlation reference: {len(available_assets)} of {len(CORR_UNIVERSE)} assets available')
    if available_assets:
        print(f'  Assets: {", ".join(available_assets)}')

    rows = []
    skipped_noload = skipped_rows = skipped_price = skipped_stale = 0
    stale_tickers = []

    for ticker, df in tqdm(price_cache.items(), desc='Computing metrics'):

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

        # Staleness check: most recent cache date vs today
        last_date = df.index[-1]
        if hasattr(last_date, 'date'):
            last_date = last_date.date()
        days_stale = (date.today() - last_date).days
        if days_stale > MAX_CACHE_AGE_DAYS:
            skipped_stale += 1
            stale_tickers.append(ticker)
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

        # Correlation to reference universe
        if not corr_ref.empty and len(df) >= CORR_LOOKBACK + 5:
            ticker_ret = df['close'].iloc[-(CORR_LOOKBACK + 1):].pct_change().dropna()
            all_corrs  = compute_correlations(ticker_ret, corr_ref)
            row['corr_top'] = all_corrs[:3]
            row['corr_bot'] = all_corrs[-3:]
            # Full ordered vector for overlap detection (matches CORR_UNIVERSE order)
            corr_map = {item['asset']: item['corr'] for item in all_corrs}
            row['corr_vec'] = [corr_map.get(a, None) for a in CORR_UNIVERSE]
            # Top pole: CORR_UNIVERSE asset with highest correlation
            valid_poles = [(v, CORR_UNIVERSE[i]) for i, v in enumerate(row['corr_vec']) if v is not None]
            if valid_poles:
                top_corr, top_name = max(valid_poles, key=lambda x: x[0])
                # Only assign if correlation is above noise threshold
                if top_corr >= MIN_TOP_POLE_CORR:
                    row['top_pole']      = top_name
                    row['top_pole_corr'] = round(top_corr, 2)
                else:
                    row['top_pole']      = None
                    row['top_pole_corr'] = None
            else:
                row['top_pole']      = None
                row['top_pole_corr'] = None
        else:
            row['corr_top']      = []
            row['corr_bot']      = []
            row['corr_vec']      = [None] * len(CORR_UNIVERSE)
            row['top_pole']      = None
            row['top_pole_corr'] = None

        # Momentum character signal
        m_signal, m_flag   = compute_momentum_signal(df)
        row['momentum_signal'] = round(m_signal, 2) if not (isinstance(m_signal, float) and np.isnan(m_signal)) else None
        row['momentum_flag']   = m_flag

        # Volume confirmation signal
        v_ratio, v_flag = compute_volume_signal(df)
        row['vol_ratio'] = round(v_ratio, 3) if not (isinstance(v_ratio, float) and np.isnan(v_ratio)) else None
        row['vol_flag']  = v_flag

        # Pullback entry timing signal (score gate applied later in add_scores)
        row['pullback_flag'] = compute_pullback_signal(row)

        rows.append(row)

    print(f'\n  Qualified: {len(rows)}')
    print(f'  Skipped (no load): {skipped_noload}')
    print(f'  Skipped (< {MIN_ROWS} rows): {skipped_rows}')
    print(f'  Skipped (price < ${MIN_PRICE}): {skipped_price}')
    print(f'  Skipped (stale > {MAX_CACHE_AGE_DAYS}d): {skipped_stale}')
    if stale_tickers:
        print(f'  Stale tickers: {", ".join(sorted(stale_tickers))}')

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

    # Pullback score gate — downgrade buy_today to none if score too low
    df['pullback_flag'] = df.apply(
        lambda r: r['pullback_flag'] if r['score'] >= PULLBACK_MIN_SCORE else 'none',
        axis=1
    )

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
.mr-table colgroup col.col-owned  { width: 28px; }
.mr-table colgroup col.col-ticker { width: 72px; }
.mr-table colgroup col.col-score  { width: 68px; }
.mr-table colgroup col.col-top-pole { width: 68px; }
.mr-table colgroup col.col-sma    { width: 44px; }
.mr-table colgroup col.col-ret    { width: 62px; }
.mr-table colgroup col.col-ratio  { width: 52px; }
.mr-table colgroup col.col-vs     { width: 62px; }
.mr-table colgroup col.col-bsd    { width: 64px; }
.mr-table colgroup col.col-mflag  { width: 28px; }
.mr-table colgroup col.col-mzscore { width: 46px; }
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

/* Owned checkbox: .own-cb and .row-owned now in shared OWNERSHIP_CSS */
.grp-blank   { background: #f8f9fb !important; color: transparent; border-bottom: 1px solid #e2e4e8; }
.grp-returns { background: #1e293b !important; color: #94a3b8; }
.grp-vs-spy  { background: #14532d !important; color: #86efac; }
.grp-other   { background: #1e3a5f !important; color: #93c5fd; }
.grp-signal  { background: #3b1e5f !important; color: #d8b4fe; }
.grp-vol     { background: #1e3a5f !important; color: #60a5fa; }
.grp-pb      { background: #14532d !important; color: #86efac; }

/* Momentum flag dot */
.mflag-dot {
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    cursor: default;
}
.mflag-clean        { background: #16a34a; }
.mflag-stalling     { background: #d97706; }
.mflag-overextended { background: #dc2626; }

/* Volume flag dot */
.vflag-dot {
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    cursor: default;
    margin-left: 3px;
}
.vflag-confirmed { background: #2563eb; }
.vflag-normal    { background: #94a3b8; }
.vflag-thin      { background: #e2e8f0; border: 1px solid #94a3b8; }

/* Pullback alert dot */
.pbflag-dot {
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    cursor: default;
}
.pbflag-buy_today {
    background: #16a34a;
    animation: pb-pulse 1.5s ease-in-out infinite;
}
.pbflag-none {
    background: #e2e8f0;
    border: 1px solid #94a3b8;
}
@keyframes pb-pulse {
    0%   { box-shadow: 0 0 0 0 rgba(22,163,74,0.7); }
    70%  { box-shadow: 0 0 0 5px rgba(22,163,74,0); }
    100% { box-shadow: 0 0 0 0 rgba(22,163,74,0); }
}

/* Theme overlap dot */
.olflag-dot {
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    cursor: default;
}
.olflag-overlap  { background: #f59e0b; }
.olflag-clear    { background: #e2e8f0; border: 1px solid #94a3b8; }
.grp-ol          { background: #78350f !important; color: #fde68a; }

/* Correlation / signal tooltip */
.mr-tooltip {
    position: fixed;
    z-index: 9999;
    background: #1a1f2e;
    color: #e2e8f0;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 10px 14px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78em;
    line-height: 1.6;
    pointer-events: none;
    min-width: 200px;
    max-width: 260px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    display: none;
}
.mr-tooltip .tt-section {
    font-size: 0.72em;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #64748b;
    margin-top: 8px;
    margin-bottom: 2px;
}
.mr-tooltip .tt-section:first-child { margin-top: 0; }
.mr-tooltip .tt-row {
    display: flex;
    justify-content: space-between;
    gap: 12px;
}
.mr-tooltip .tt-asset { color: #93c5fd; font-weight: 600; }
.mr-tooltip .tt-corr-pos { color: #4ade80; }
.mr-tooltip .tt-corr-neg { color: #f87171; }
.mr-tooltip .tt-signal-clean        { color: #4ade80; font-weight: 700; }
.mr-tooltip .tt-signal-stalling     { color: #fbbf24; font-weight: 700; }
.mr-tooltip .tt-signal-overextended { color: #f87171; font-weight: 700; }
.mr-tooltip .tt-vol-confirmed { color: #60a5fa; font-weight: 700; }
.mr-tooltip .tt-vol-normal    { color: #94a3b8; }
.mr-tooltip .tt-vol-thin      { color: #64748b; font-style: italic; }
.mr-tooltip .tt-ticker-header {
    font-size: 1.05em;
    font-weight: 700;
    color: #f1f5f9;
    border-bottom: 1px solid #334155;
    padding-bottom: 6px;
    margin-bottom: 4px;
}

/* ── Mobile responsive ── */
@media (max-width: 768px) {
    .filter-bar { padding: 10px 12px; gap: 10px; }
    .filter-bar label { font-size: 0.78em; }
    .filter-bar input[type=text] { width: 80px; font-size: 0.8em; }
    .filter-bar input[type=range] { width: 70px; }
    .mr-table-wrap { -webkit-overflow-scrolling: touch; }
    .mr-table { font-size: 0.72em; }
    .mr-table thead th { padding: 5px 3px; font-size: 0.68em; }
    .mr-table tbody td { padding: 4px 3px; font-size: 0.82em; }
}
"""

# JS is a plain string - no .format() substitution needed
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
    let filterText = '', filterSma = 'all', filterMinScore = 0, filterOwned = 'all';

    function getSortVal(row, col) {
        if (col === 'ticker')   return row.ticker || '';
        if (col === 'top_pole') return row.top_pole || '';
        if (col === 'sma_all')  return row.sma_all ? 1 : 0;
        if (col === 'owned')    return window._owned.has(row.ticker) ? 1 : 0;
        if (col === 'ol_score') return (_overlapMap[row.ticker] || {}).maxSim || 0;
        return row[col] != null ? Number(row[col]) : -9999;
    }
    function rowVisible(row) {
        if (filterText && !row.ticker.toLowerCase().includes(filterText.toLowerCase())) return false;
        if (filterSma === 'above' && !row.sma_all) return false;
        if (filterSma === 'below' && row.sma_all) return false;
        if ((row.score || 0) < filterMinScore) return false;
        if (filterOwned === 'owned' && !window._owned.has(row.ticker)) return false;
        return true;
    }

    // -------------------------------------------------------------------------
    // OWNED STATE (shared cross-dashboard API via window._owned)
    // -------------------------------------------------------------------------
    // window._owned is injected by dashboard_writer.py (OWNERSHIP_JS)
    // provides: .has(t), .toggle(t), .all(), .count()

    function toggleOwned(ticker) {
        window._owned.toggle(ticker);
        renderTable();
    }
    const tooltip = document.createElement('div');
    tooltip.className = 'mr-tooltip';
    tooltip.id = 'mr-tooltip';
    document.body.appendChild(tooltip);

    function buildTooltipHTML(row) {
        const flag  = row.momentum_flag || 'clean';
        const zscore = row.momentum_signal;
        const flagLabel = {
            'clean':        'Clean',
            'stalling':     'Stalling',
            'overextended': 'Overextended'
        }[flag] || flag;
        const flagClass = `tt-signal-${flag}`;

        const priceStr = row.price != null ? ` &nbsp;<span style="font-size:0.85em;color:#94a3b8;">$${row.price.toFixed(2)}</span>` : '';
        let html = `<div class="tt-ticker-header">${row.ticker}${priceStr}</div>`;

        // Momentum signal section
        html += `<div class="tt-section">Momentum Signal (3d z-score)</div>`;
        const zDisplay = zscore != null ? (zscore >= 0 ? '+' : '') + zscore.toFixed(2) : 'n/a';
        html += `<div class="tt-row"><span class="${flagClass}">${flagLabel}</span><span>${zDisplay}</span></div>`;

        // Volume confirmation section
        const vflag = row.vol_flag || 'normal';
        const vratio = row.vol_ratio;
        const vLabel = { 'confirmed': 'Confirmed', 'normal': 'Normal', 'thin': 'Thin' }[vflag] || vflag;
        const vClass = { 'confirmed': 'tt-vol-confirmed', 'normal': 'tt-vol-normal', 'thin': 'tt-vol-thin' }[vflag] || '';
        const vDisplay = vratio != null ? vratio.toFixed(2) + 'x' : 'n/a';
        html += `<div class="tt-section">Volume (5d vs 63d avg)</div>`;
        html += `<div class="tt-row"><span class="${vClass}">${vLabel}</span><span>${vDisplay}</span></div>`;

        // Pullback entry timing section
        const pbflag  = row.pullback_flag || 'none';
        const ret1d   = row.ret_1d;
        const pbLabel = pbflag === 'buy_today' ? 'BUY TODAY (down day)' : 'No signal';
        const pbClass = pbflag === 'buy_today' ? 'tt-signal-clean' : 'tt-vol-normal';
        const ret1dDisplay = ret1d != null ? (ret1d >= 0 ? '+' : '') + ret1d.toFixed(2) + '% today' : 'n/a';
        html += `<div class="tt-section">Entry Timing (PB)</div>`;
        html += `<div class="tt-row"><span class="${pbClass}">${pbLabel}</span><span>${ret1dDisplay}</span></div>`;

        // Theme overlap section
        const olData = _overlapMap[row.ticker] || { overlaps: [] };
        html += `<div class="tt-section">Theme Overlap (OL)</div>`;
        if (window._owned.count() === 0) {
            html += `<div class="tt-row"><span style="color:#94a3b8;">No owned tickers set</span></div>`;
        } else if (olData.overlaps.length === 0) {
            html += `<div class="tt-row"><span style="color:#16a34a;">No overlap detected</span></div>`;
        } else {
            olData.overlaps.forEach(o => {
                html += `<div class="tt-row"><span style="color:#f59e0b;">${o.ticker}</span><span>${o.sim.toFixed(2)}</span></div>`;
            });
        }
        const top = row.corr_top || [];
        const bot = row.corr_bot || [];

        if (top.length) {
            html += `<div class="tt-section">Most Correlated (63d)</div>`;
            top.forEach(item => {
                const cls = item.corr >= 0 ? 'tt-corr-pos' : 'tt-corr-neg';
                const sign = item.corr >= 0 ? '+' : '';
                html += `<div class="tt-row"><span class="tt-asset">${item.asset}</span><span class="${cls}">${sign}${item.corr.toFixed(3)}</span></div>`;
            });
        }
        if (bot.length) {
            html += `<div class="tt-section">Least Correlated (63d)</div>`;
            bot.forEach(item => {
                const cls = item.corr >= 0 ? 'tt-corr-pos' : 'tt-corr-neg';
                const sign = item.corr >= 0 ? '+' : '';
                html += `<div class="tt-row"><span class="tt-asset">${item.asset}</span><span class="${cls}">${sign}${item.corr.toFixed(3)}</span></div>`;
            });
        }

        return html;
    }

    function positionTooltip(e) {
        const tt = document.getElementById('mr-tooltip');
        const pad = 12;
        let x = e.clientX + pad;
        let y = e.clientY + pad;
        // Keep tooltip within viewport
        const ttW = tt.offsetWidth || 220;
        const ttH = tt.offsetHeight || 160;
        if (x + ttW > window.innerWidth  - pad) x = e.clientX - ttW - pad;
        if (y + ttH > window.innerHeight - pad) y = e.clientY - ttH - pad;
        tt.style.left = x + 'px';
        tt.style.top  = y + 'px';
    }

    function attachTooltipListeners() {
        var isTouchDevice = ('ontouchstart' in window) || (navigator.maxTouchPoints > 0);
        document.querySelectorAll('[data-mr-row]').forEach(el => {
            const idx = parseInt(el.getAttribute('data-mr-row'), 10);
            if (isTouchDevice) {
                el.addEventListener('click', function(e) {
                    const tt = document.getElementById('mr-tooltip');
                    if (tt.style.display === 'block' && tt._activeIdx === idx) {
                        tt.style.display = 'none';
                        tt._activeIdx = -1;
                        return;
                    }
                    tt.innerHTML = buildTooltipHTML(window._mrRows[idx]);
                    tt.style.display = 'block';
                    tt._activeIdx = idx;
                    var rect = el.getBoundingClientRect();
                    tt.style.left = '8px';
                    tt.style.top = (rect.bottom + window.scrollY + 4) + 'px';
                    tt.style.position = 'absolute';
                    tt.style.width = Math.min(360, window.innerWidth - 16) + 'px';
                    tt.style.maxWidth = (window.innerWidth - 16) + 'px';
                    e.stopPropagation();
                });
            } else {
                el.addEventListener('mouseenter', function(e) {
                    const tt = document.getElementById('mr-tooltip');
                    tt.innerHTML = buildTooltipHTML(window._mrRows[idx]);
                    tt.style.display = 'block';
                    positionTooltip(e);
                });
                el.addEventListener('mousemove', positionTooltip);
                el.addEventListener('mouseleave', function() {
                    document.getElementById('mr-tooltip').style.display = 'none';
                });
            }
        });
        if (isTouchDevice) {
            document.addEventListener('click', function(e) {
                if (!e.target.closest('[data-mr-row]') && !e.target.closest('.mr-tooltip')) {
                    document.getElementById('mr-tooltip').style.display = 'none';
                }
            });
        }
    }

    // -------------------------------------------------------------------------
    // TABLE RENDER
    // -------------------------------------------------------------------------
    function renderTable() {
        let rows = [...DATA.data];

        // Build overlap map against current owned set and threshold
        _overlapMap = buildOverlapMap(rows);

        rows.sort((a,b) => {
            const va = getSortVal(a, sortCol), vb = getSortVal(b, sortCol);
            if (typeof va === 'string') return sortDir==='asc' ? va.localeCompare(vb) : vb.localeCompare(va);
            return sortDir==='asc' ? va-vb : vb-va;
        });
        const visible = rows.filter(rowVisible);

        // Store visible rows for tooltip lookup
        window._mrRows = {};
        visible.forEach((row, i) => { window._mrRows[i] = row; });

        document.getElementById('filter-count').textContent =
            visible.length + ' of ' + DATA.universe_size + ' tickers';

        const arrow = col => sortCol===col ? (sortDir==='asc' ? ' \u25b2' : ' \u25bc') : '';
        const th = (col, lbl, cls, tip) => {
            const titleAttr = tip ? ` title="${tip}"` : '';
            return `<th class="${cls||''}" onclick="window._mrSort('${col}')"${titleAttr}>${lbl}${arrow(col)}</th>`;
        };

        const TIP = {
            rank:  'Composite rank across all scored tickers. Higher score = lower rank number.',
            score: 'Percentile composite (0-100) of three components equally weighted: '
                 + '(1) average return percentile across all 6 periods, '
                 + '(2) momentum ratio passes (out of 5), '
                 + '(3) bad-SPY day resilience. '
                 + 'Score is zeroed if price is below any SMA.',
            sma:   'Price vs simple moving averages. '
                 + 'Checkmark = price above ALL of: 30, 50, 100, 200-day SMAs. '
                 + 'Dash = below one or more. Score is zeroed when dashed.',
            ratio: 'Momentum consistency: how many of 5 short/long return ratio tests pass. '
                 + 'Tests: 1w/1d >= 3x, 1m/1w >= 3x, 3m/1m >= 2x, 6m/3m >= 1x, 1y/3m >= 2x. '
                 + 'Green=4-5 passes, amber=2-3, red=0-1.',
            badspy:'Average return on the 5 worst SPY days in the last 20 sessions. '
                 + 'Positive = stock held up or rose when the market sold off (defensive strength). '
                 + 'Negative = stock fell with or worse than SPY on bad days.',
            vsSpy: 'Return differential vs SPY over the period. '
                 + 'Positive = outperforming SPY. Negative = underperforming.',
            sig:   'LEFT dot = Momentum character (3-day return z-score vs 63-day distribution): '
                 + 'green=clean, amber=stalling (z < -1.5), red=overextended (z > +2.0). '
                 + 'Sortable by z-score. Hover ticker for full details.',
            vol:   'Volume confirmation dot. '
                 + '5-day avg volume vs 63-day avg volume ratio. '
                 + 'blue=confirmed (>1.5x, strong conviction), '
                 + 'gray=normal (0.75-1.5x), '
                 + 'white ring=thin (<0.75x, low conviction). '
                 + 'Sortable by ratio. Hover ticker for full details.',
            pb:    'Entry Timing: down-day alert on strong momentum stocks. '
                 + 'Pulsing green dot = buy_today: stock is down today (ret_1d<0) '
                 + 'but trend is intact (all SMAs, momentum clean, 3m positive). '
                 + 'Volume confirmed is excluded — surging volume on a down day signals distribution. '
                 + 'Score gate: must be >=60. Sortable by 1d return.',
            ol:    'Theme Overlap: cosine similarity of each ticker&#39;s 20-asset correlation fingerprint '
                 + 'vs your owned tickers. Amber = too similar to something you own (threshold set in filter bar). '
                 + 'No dots fire if owned set is empty. Sortable by overlap score. '
                 + 'Hover ticker for which owned names overlap and similarity scores.',
        };

        let html = `<table class="mr-table"><colgroup>
  <col class="col-owned"><col class="col-rank"><col class="col-ticker"><col class="col-score">
  <col class="col-top-pole"><col class="col-sma">
  <col class="col-ret"><col class="col-ret"><col class="col-ret">
  <col class="col-ret"><col class="col-ret"><col class="col-ret">
  <col class="col-ratio">
  <col class="col-vs"><col class="col-vs"><col class="col-vs">
  <col class="col-bsd">
  <col class="col-mzscore"><col class="col-mflag"><col class="col-mflag"><col class="col-mflag">
</colgroup><thead>
<tr class="grp-row">
  <th class="grp-blank" colspan="6"></th>
  <th class="grp-returns" colspan="6">Returns</th>
  <th class="grp-blank"></th>
  <th class="grp-vs-spy" colspan="3">vs SPY</th>
  <th class="grp-other" title="${TIP.badspy}">Bad SPY</th>
  <th class="grp-signal" title="${TIP.sig}">M</th>
  <th class="grp-vol"    title="${TIP.vol}">V</th>
  <th class="grp-pb"     title="${TIP.pb}">PB</th>
  <th class="grp-ol"     title="${TIP.ol}">OL</th>
</tr><tr>
  ${th('owned',         '\u2605',  '',        'Click to mark as owned. Sort to group owned tickers. Filter to show only owned.')}
  ${th('rank',          'Rank',  '',        TIP.rank)}
  ${th('ticker',        'Ticker','col-left')}
  ${th('score',         'Score', '',        TIP.score)}
  ${th('top_pole', 'Top Pole', '', 'Highest-correlation pole asset for this ticker')}
  ${th('sma_all',       'SMA',   '',        TIP.sma)}
  ${th('ret_1d','1d')} ${th('ret_1w','1w')} ${th('ret_1m','1m')}
  ${th('ret_3m','3m')} ${th('ret_6m','6m')} ${th('ret_1y','1y')}
  ${th('ratio_passes',  'Ratio', '',        TIP.ratio)}
  ${th('vs_spy_1w', '1w','',TIP.vsSpy)} ${th('vs_spy_1m','1m','',TIP.vsSpy)} ${th('vs_spy_1y','1y','',TIP.vsSpy)}
  ${th('bad_spy_score', 'Score', '',        TIP.badspy)}
  ${th('momentum_signal','M',    '',        TIP.sig)}
  ${th('vol_ratio',     'V',     '',        TIP.vol)}
  ${th('ret_1d',        'PB',    '',        TIP.pb)}
  ${th('ol_score',      'OL',    '',        TIP.ol)}
</tr></thead><tbody>`;

        visible.forEach((row, i) => {
            const sc   = row.score || 0;
            const scBg = getGradientColor(sc / 100);
            const scFg = textColor(scBg);
            const smaTd = row.sma_all
                ? '<td style="color:#16a34a;font-weight:700;">&#10003;</td>'
                : '<td style="color:#ccc;">&#8212;</td>';
            const rp   = row.ratio_passes || 0;
            const rpBg = rp >= 4 ? '#16a34a' : rp >= 2 ? '#d97706' : '#dc2626';

            // Momentum dot (left col) and volume dot (right col) — now separate sortable columns
            const flag  = row.momentum_flag || 'clean';
            const vflag = row.vol_flag || 'normal';
            const pbflag = row.pullback_flag || 'none';
            const ol     = _overlapMap[row.ticker] || { maxSim: 0, overlaps: [] };
            const olflag = ol.overlaps.length > 0 ? 'overlap' : 'clear';
            const mVal     = (row.momentum_signal != null) ? (row.momentum_signal >= 0 ? '+' : '') + row.momentum_signal.toFixed(2) : '—';
            const mz       = row.momentum_signal;
            const mBg      = (mz == null)        ? '#888888'
                           : (mz > 2.0)          ? '#dc2626'
                           : (mz > 1.5)          ? '#d97706'
                           : (mz >= 0.5)         ? '#16a34a'
                           : (mz >= -0.5)        ? '#d97706'
                           :                       '#dc2626';
            const mflagTd  = `<td title="Momentum z-score: ${mVal} (${flag})" style="background:${mBg};color:#fff;font-weight:700;font-size:0.78em;text-align:center;">${mVal}</td>`;
            const vflagTd  = `<td title="Volume: ${vflag}"><span class="vflag-dot vflag-${vflag}" style="margin-left:0;"></span></td>`;
            const pbflagTd = `<td title="Entry timing: ${pbflag}"><span class="pbflag-dot pbflag-${pbflag}"></span></td>`;
            const olflagTd = `<td title="Theme overlap: ${olflag}"><span class="olflag-dot olflag-${olflag}"></span></td>`;

            // Top pole cell — bg colored by that pole's 21d z-score
            const tp     = row.top_pole;
            const tpCorr = row.top_pole_corr;
            const tpZ    = (tp && DATA.pole_momentum) ? DATA.pole_momentum[tp] : null;
            const tpBg   = (tpZ == null)  ? '#888888'
                         : (tpZ > 2.0)    ? '#dc2626'
                         : (tpZ > 1.5)    ? '#d97706'
                         : (tpZ >= 0.5)   ? '#16a34a'
                         : (tpZ >= -0.5)  ? '#d97706'
                         :                  '#dc2626';
            const tpTd   = tp
                ? `<td title="Top pole: ${tp} (corr ${tpCorr}, pole 21d z: ${tpZ != null ? tpZ.toFixed(2) : '--'})" style="background:${tpBg};color:#fff;font-weight:700;font-size:0.75em;text-align:center;white-space:nowrap;">${tp} ${tpCorr != null ? tpCorr.toFixed(2) : ''}</td>`
                : `<td style="color:#ccc;text-align:center;">&#8212;</td>`;

            const isOwned = window._owned.has(row.ticker);
            const ownedClass = isOwned ? ' row-owned' : '';
            const ownedTd = `<td><input type="checkbox" class="own-cb" ${isOwned ? 'checked' : ''} onclick="window._mrToggleOwned('${row.ticker}')" title="Mark as owned"></td>`;

            html += `<tr class="${ownedClass}">
  ${ownedTd}
  <td>${row.rank}</td>
  <td class="col-left" data-mr-row="${i}" style="cursor:default;">${row.ticker}</td>
  <td><span class="score-badge" style="background:${scBg};color:${scFg};">${sc.toFixed(1)}</span></td>
  ${tpTd}
  ${smaTd}
  ${gradTd(row.ret_1d,'ret_1d')}  ${gradTd(row.ret_1w,'ret_1w')}
  ${gradTd(row.ret_1m,'ret_1m')}  ${gradTd(row.ret_3m,'ret_3m')}
  ${gradTd(row.ret_6m,'ret_6m')}  ${gradTd(row.ret_1y,'ret_1y')}
  <td style="background:${rpBg};color:#fff;font-weight:700;">${rp}/5</td>
  ${gradTd(row.vs_spy_1w,'vs_spy_1w')}
  ${gradTd(row.vs_spy_1m,'vs_spy_1m')}
  ${gradTd(row.vs_spy_1y,'vs_spy_1y')}
  ${gradTd(row.bad_spy_score,'bad_spy_score',2)}
  ${mflagTd}
  ${vflagTd}
  ${pbflagTd}
  ${olflagTd}
</tr>`;
        });
        html += '</tbody></table>';
        document.getElementById('mr-table-wrap').innerHTML = html;
        attachTooltipListeners();
    }

    window._mrToggleOwned = function(ticker) {
        toggleOwned(ticker);
        updatePoleCorrRow();
    };

    // -------------------------------------------------------------------------
    // POLE CORR ROW: average corr_vec across owned tickers per pole
    // -------------------------------------------------------------------------
    const _corrUniverse = DATA.corr_universe || [];
    const _poleMomentum = DATA.pole_momentum  || {};

    function updatePoleCorrRow() {
        const owned = [];
        for (const r of DATA.data) {
            if (window._owned.has(r.ticker) && r.corr_vec) owned.push(r.corr_vec);
        }

        // Pass 1: compute avg corr and raw alignment per pole
        const corrAvgs  = [];   // avg corr per pole (or null)
        const rawAligns = [];   // corr_avg * z_score per pole (or null)

        _corrUniverse.forEach(function(pole, i) {
            if (owned.length === 0) {
                corrAvgs.push(null);
                rawAligns.push(null);
                return;
            }
            let sum = 0, cnt = 0;
            owned.forEach(function(vec) {
                if (vec[i] != null) { sum += vec[i]; cnt++; }
            });
            if (cnt === 0) {
                corrAvgs.push(null);
                rawAligns.push(null);
                return;
            }
            const avg = sum / cnt;
            corrAvgs.push(avg);
            const z = (_poleMomentum[pole] != null) ? _poleMomentum[pole] : null;
            rawAligns.push(z != null ? avg * z : null);
        });

        // Pass 2: normalise alignment to 0-100% using 5th-95th percentile
        const validAligns = rawAligns.filter(v => v != null);
        let pct5 = 0, pct95 = 0, hasAlign = false;
        if (validAligns.length >= 2) {
            const sorted = validAligns.slice().sort((a, b) => a - b);
            const p5idx  = Math.floor(sorted.length * 0.05);
            const p95idx = Math.ceil(sorted.length  * 0.95) - 1;
            pct5  = sorted[Math.max(0, p5idx)];
            pct95 = sorted[Math.min(sorted.length - 1, p95idx)];
            hasAlign = (pct95 - pct5) > 0.001;
        }

        // Pass 3: render rows
        _corrUniverse.forEach(function(pole, i) {
            // --- Row 3: Owned Corr ---
            const corrCell = document.getElementById('pole-corr-' + pole);
            if (corrCell) {
                if (corrAvgs[i] == null) {
                    corrCell.textContent      = '--';
                    corrCell.style.color      = '#94a3b8';
                    corrCell.style.background = '#1e293b';
                } else {
                    const avg = corrAvgs[i];
                    const t   = Math.min(1, Math.max(0, avg));
                    const rC  = Math.round(226 + t * (30  - 226));
                    const gC  = Math.round(232 + t * (41  - 232));
                    const bC  = Math.round(240 + t * (59  - 240));
                    corrCell.textContent      = avg.toFixed(2);
                    corrCell.style.background = `rgb(${rC},${gC},${bC})`;
                    corrCell.style.color      = t >= 0.5 ? '#e2e8f0' : '#1e293b';
                }
            }

            // --- Row 4: Alignment ---
            const alignCell = document.getElementById('pole-align-' + pole);
            if (alignCell) {
                if (rawAligns[i] == null || !hasAlign) {
                    alignCell.textContent      = '--';
                    alignCell.style.color      = '#94a3b8';
                    alignCell.style.background = '#1e293b';
                } else {
                    const raw = rawAligns[i];

                    // Normalise to 0-100% within 5-95 percentile range
                    // Centre of scale is 0 (no alignment), positive = good, negative = risk
                    // Map symmetrically: use max abs of pct5/pct95 as scale endpoints
                    const absMax = Math.max(Math.abs(pct5), Math.abs(pct95), 0.0001);
                    const clamped = Math.min(absMax, Math.max(-absMax, raw));
                    const pct = Math.round(((clamped / absMax) + 1) / 2 * 100);  // 0-100

                    // Color: positive alignment -> green, negative -> red
                    // Intensity from centre (50%) outward
                    const intensity = Math.abs(pct - 50) / 50;  // 0 at centre, 1 at extremes
                    let rB, gB, bB;
                    if (pct >= 50) {
                        // green tint: light(226,240,226) -> dark green(22,101,52)
                        rB = Math.round(226 + intensity * (22  - 226));
                        gB = Math.round(240 + intensity * (101 - 240));
                        bB = Math.round(226 + intensity * (52  - 226));
                    } else {
                        // red tint: light(240,226,226) -> dark red(153,27,27)
                        rB = Math.round(240 + intensity * (153 - 240));
                        gB = Math.round(226 + intensity * (27  - 226));
                        bB = Math.round(226 + intensity * (27  - 226));
                    }
                    const textColor = intensity >= 0.5 ? '#e2e8f0' : '#1e293b';

                    alignCell.textContent      = pct + '%';
                    alignCell.style.background = `rgb(${rB},${gB},${bB})`;
                    alignCell.style.color      = textColor;
                }
            }
        });
    }

    // Run once on load in case owned tickers are pre-loaded
    updatePoleCorrRow();


    // -------------------------------------------------------------------------
    let olThreshold = 0.85;

    function cosineSim(a, b) {
        // a and b are arrays of numbers or nulls (length 20)
        let dot = 0, normA = 0, normB = 0;
        for (let i = 0; i < a.length; i++) {
            if (a[i] == null || b[i] == null) continue;
            dot   += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        if (normA === 0 || normB === 0) return 0;
        return dot / (Math.sqrt(normA) * Math.sqrt(normB));
    }

    // Build overlap map: ticker -> {maxSim, overlaps: [{ticker, sim}]}
    // Recomputed whenever owned set or threshold changes
    function buildOverlapMap(rows) {
        const overlapMap = {};
        if (window._owned.count() === 0) return overlapMap;

        // Gather owned row vectors
        const ownedVecs = {};
        rows.forEach(r => {
            if (window._owned.has(r.ticker) && r.corr_vec) {
                ownedVecs[r.ticker] = r.corr_vec;
            }
        });
        if (Object.keys(ownedVecs).length === 0) return overlapMap;

        rows.forEach(r => {
            if (!r.corr_vec) return;
            const overlaps = [];
            for (const [ot, ov] of Object.entries(ownedVecs)) {
                if (ot === r.ticker) continue; // skip self
                const sim = cosineSim(r.corr_vec, ov);
                if (sim >= olThreshold) {
                    overlaps.push({ ticker: ot, sim: round2(sim) });
                }
            }
            overlaps.sort((a, b) => b.sim - a.sim);
            overlapMap[r.ticker] = {
                maxSim: overlaps.length > 0 ? overlaps[0].sim : 0,
                overlaps
            };
        });
        return overlapMap;
    }

    function round2(n) { return Math.round(n * 100) / 100; }

    let _overlapMap = {};

    window._mrSort = function(col) {
        if (sortCol === col) { sortDir = sortDir==='asc' ? 'desc' : 'asc'; }
        else { sortCol = col; sortDir = (col==='rank') ? 'asc' : 'desc'; }
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
    document.getElementById('filter-owned').addEventListener('change', e => {
        filterOwned = e.target.value; renderTable();
    });
    document.getElementById('filter-ol-threshold').addEventListener('change', e => {
        olThreshold = parseFloat(e.target.value); renderTable();
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
    top_lbl = '{} {:.1f}'.format(top.get('ticker','--'), top.get('score', 0)) if top else '--'
    sma_ct  = sum(1 for r in data if r.get('sma_all'))

    # Momentum signal summary counts
    stalling_ct      = sum(1 for r in data if r.get('momentum_flag') == 'stalling')
    overextended_ct  = sum(1 for r in data if r.get('momentum_flag') == 'overextended')
    vol_confirmed_ct = sum(1 for r in data if r.get('vol_flag') == 'confirmed')
    pullback_buy_ct  = sum(1 for r in data if r.get('pullback_flag') == 'buy_today')

    parts = []
    parts.append(writer.stat_bar([
        ('Generated',      gen,                    'neutral'),
        ('Universe',       str(n),                 'neutral'),
        ('Above All SMAs', str(sma_ct),            'pos'  if sma_ct > 0          else 'neutral'),
        ('#1 Ticker',      top_lbl,                'pos'),
        ('Stalling',       str(stalling_ct),       'warn' if stalling_ct > 0     else 'neutral'),
        ('Overextended',   str(overextended_ct),   'warn' if overextended_ct > 0 else 'neutral'),
        ('Vol Confirmed',  str(vol_confirmed_ct),  'pos'  if vol_confirmed_ct > 0 else 'neutral'),
        ('Buy Today',      str(pullback_buy_ct),   'pos'  if pullback_buy_ct > 0  else 'neutral'),
    ]))
    parts.append(writer.build_header(
        'Ranked by composite momentum score \u2014 {} tickers scanned'.format(n)
    ))

    # ---- POLE PANEL (3-row table: tickers | 21d momentum z | owned corr avg) ----
    pole_momentum  = output.get('pole_momentum', {})
    corr_universe  = output.get('corr_universe', CORR_UNIVERSE)

    def mz_bg(z):
        if z is None:   return '#888888'
        if z >  2.0:    return '#dc2626'
        if z >  1.5:    return '#d97706'
        if z >= 0.5:    return '#16a34a'
        if z >= -0.5:   return '#d97706'
        return '#dc2626'

    pole_header_cells = ''.join(
        f'<th style="padding:3px 6px;font-size:0.72em;font-weight:700;'
        f'color:#e2e8f0;background:#1e293b;text-align:center;white-space:nowrap;">'
        f'{t}</th>'
        for t in corr_universe
    )
    pole_mz_cells = ''.join(
        (lambda z, bg:
            f'<td style="padding:3px 6px;font-size:0.72em;font-weight:700;'
            f'background:{bg};color:#fff;text-align:center;white-space:nowrap;">'
            f'{"+" if z is not None and z >= 0 else ""}{z if z is not None else "--"}</td>'
        )(pole_momentum.get(t), mz_bg(pole_momentum.get(t)))
        for t in corr_universe
    )

    pole_panel_html = (
        '<div id="pole-panel" style="margin:8px 0 4px 0;overflow-x:auto;">'
        '<table style="border-collapse:collapse;width:100%;">'
        '<colgroup>' + ''.join('<col>' for _ in corr_universe) + '</colgroup>'
        '<thead><tr>'
        '<th style="padding:3px 6px;font-size:0.65em;color:#94a3b8;background:#0f172a;'
        'text-align:left;white-space:nowrap;min-width:80px;">POLE</th>'
        + pole_header_cells +
        '</tr></thead>'
        '<tbody>'
        '<tr>'
        '<td style="padding:3px 6px;font-size:0.65em;color:#94a3b8;background:#0f172a;'
        'white-space:nowrap;">21d Z</td>'
        + pole_mz_cells +
        '</tr>'
        '<tr id="pole-corr-row">'
        '<td style="padding:3px 6px;font-size:0.65em;color:#94a3b8;background:#0f172a;'
        'white-space:nowrap;">Owned Corr</td>'
        + ''.join(
            f'<td id="pole-corr-{t}" style="padding:3px 6px;font-size:0.72em;'
            f'font-weight:700;color:#94a3b8;background:#1e293b;text-align:center;">--</td>'
            for t in corr_universe
        ) +
        '</tr>'
        '<tr id="pole-align-row">'
        '<td style="padding:3px 6px;font-size:0.65em;color:#94a3b8;background:#0f172a;'
        'white-space:nowrap;">Alignment</td>'
        + ''.join(
            f'<td id="pole-align-{t}" style="padding:3px 6px;font-size:0.72em;'
            f'font-weight:700;color:#94a3b8;background:#1e293b;text-align:center;">--</td>'
            for t in corr_universe
        ) +
        '</tr>'
        '</tbody>'
        '</table>'
        '</div>'
    )
    parts.append(pole_panel_html)

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
        '<label>Owned '
        '<select id="filter-owned">'
        '<option value="all">All</option>'
        '<option value="owned">Owned only</option>'
        '</select></label>'
        '<label>Overlap threshold '
        '<select id="filter-ol-threshold">'
        '<option value="0.70">0.70 (loose)</option>'
        '<option value="0.75">0.75</option>'
        '<option value="0.80">0.80</option>'
        '<option value="0.85" selected>0.85 (default)</option>'
        '<option value="0.90">0.90 (tight)</option>'
        '</select></label>'
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
        ' &nbsp;|&nbsp; M = momentum signal (sortable) \u2014 '
        ' &nbsp;<span style="color:#16a34a;font-weight:700;" title="Momentum: Clean \u2014 3-day return is within normal range (z-score between -1.5 and +2.0)">&#9679;</span> clean'
        ' &nbsp;<span style="color:#d97706;font-weight:700;" title="Momentum: Stalling \u2014 3-day return z-score below -1.5. Short-term move may be losing steam. Consider waiting for stabilization.">&#9679;</span> stalling'
        ' &nbsp;<span style="color:#dc2626;font-weight:700;" title="Momentum: Overextended \u2014 3-day return z-score above +2.0. Unusual spike relative to this stock\'s history. Risk of chasing.">&#9679;</span> overextended'
        ' &nbsp;|&nbsp; V = volume confirmation (sortable) \u2014 '
        ' &nbsp;<span style="color:#2563eb;font-weight:700;" title="Volume: Confirmed \u2014 5-day avg volume is 1.5x or more above the 63-day avg. Strong conviction behind the move.">&#9679;</span> confirmed'
        ' &nbsp;<span style="color:#94a3b8;" title="Volume: Normal \u2014 5-day avg volume is within 0.75x\u20131.5x of the 63-day avg. Nothing unusual.">&#9679;</span> normal'
        ' &nbsp;<span style="color:#cbd5e1;" title="Volume: Thin \u2014 5-day avg volume is below 0.75x the 63-day avg. Price moving on light volume. Low conviction.">&#9675;</span> thin'
        ' &nbsp;|&nbsp; PB = entry timing: down day on strong stock (sortable by 1d return) \u2014 '
        ' &nbsp;<span style="color:#16a34a;font-weight:700;" title="Buy Today \u2014 stock is down today but trend intact: all SMAs, momentum clean, 3m positive, score>=60, volume not surging.">&#9679;</span> buy today'
        ' &nbsp;<span style="color:#e2e8f0;font-weight:700;" title="No signal \u2014 conditions not met.">&#9675;</span> none'
        ' &nbsp;|&nbsp; OL = theme overlap vs owned tickers \u2014 '
        ' &nbsp;<span style="color:#f59e0b;font-weight:700;" title="Overlap \u2014 cosine similarity of 20-asset correlation fingerprint exceeds threshold vs an owned ticker.">&#9679;</span> overlap'
        ' &nbsp;<span style="color:#e2e8f0;" title="Clear \u2014 no significant overlap with owned tickers at current threshold.">&#9675;</span> clear'
        '</span></div>'
        + filter_html
        + '<div class="mr-table-wrap" id="mr-table-wrap"></div>'
        '</div>'
    )
    parts.append(table_html)
    parts.append(writer.footer())

    data_json = _json.dumps(output, ensure_ascii=False, allow_nan=False)
    writer._mr_extra_js = RENDER_JS_HEAD + data_json + RENDER_JS_TAIL

    return '\n'.join(parts)


def main():
    print('=' * 80)
    print('Momentum Ranker v1.11')
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

    # Drop list columns before CSV (not CSV-friendly), keep in JSON
    csv_df = df.drop(columns=['corr_top', 'corr_bot', 'corr_vec'], errors='ignore')

    records = [sanitize_record(r) for r in df.to_dict(orient='records')]

    # Compute 21-day momentum z-score for each CORR_UNIVERSE pole
    print('\nComputing pole momentum (21-day z-scores)...')
    pole_momentum = {}
    for pole in CORR_UNIVERSE:
        pkl_path = PRICE_CACHE_DIR / f'{pole}.pkl'
        try:
            pole_df = pd.read_pickle(pkl_path)
            z = compute_pole_momentum(pole_df)
            pole_momentum[pole] = z
            print(f'  {pole}: {z}')
        except Exception as e:
            print(f'  {pole}: ERROR ({e})')
            pole_momentum[pole] = None

    output = {
        'generated_at':     datetime.now().isoformat(),
        'universe_size':    len(df),
        'filters':          {'min_price': MIN_PRICE, 'min_rows': MIN_ROWS, 'sma_windows': SMA_WINDOWS},
        'ratio_thresholds': RATIO_THRESHOLDS,
        'columns':          list(df.columns),
        'pole_momentum':    pole_momentum,
        'corr_universe':    CORR_UNIVERSE,
        'data':             records,
    }

    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(output, f, allow_nan=False)
    print(f'JSON saved: {OUTPUT_JSON}')
    csv_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
    print(f'CSV  saved: {OUTPUT_CSV}')

    try:
        from dashboard_writer import DashboardWriter
    except ImportError:
        print('ERROR: dashboard_writer.py not found -- skipping HTML output')
        return

    writer   = DashboardWriter('momentum-ranker', 'Momentum Ranker')
    body     = build_body_html(output, writer)
    extra_js = getattr(writer, '_mr_extra_js', '')
    writer.write(body, extra_css=EXTRA_CSS, extra_js=extra_js)

    preview = ['rank','ticker','score','price','ret_1d','ret_1w','ret_1m','ret_3m','ret_1y','sma_all','ratio_passes','momentum_flag','vol_flag','vol_ratio','pullback_flag']
    avail   = [c for c in preview if c in df.columns]
    print(f'\nTop 10:')
    print(df[avail].head(10).to_string(index=False))
    print('\n' + '=' * 80)
    print('SUCCESS')
    print('=' * 80)


if __name__ == '__main__':
    main()
