"""
Turn of Month Strategy - FULL PRICE CACHE ANALYSIS (V3)
========================================================

Now includes:
- Long TOM, Short TOM, Inverse TOM strategies
- Full history, 5Y, and 1Y period analysis
- Regime change detection (1Y vs 5Y vs Full)
- Fixed infinity bug with clipping

Run: python tom_full_scan.py
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# PRICE CACHE LOADER
# ============================================================================

def load_single_pkl(pkl_file):
    ticker = pkl_file.stem
    try:
        df = pd.read_pickle(pkl_file)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        close_col = None
        for col in ['close', 'Close', 'CLOSE']:
            if col in df.columns:
                close_col = col
                break
        
        if close_col is None or len(df) < 252:
            return None
        
        return (ticker, df)
    except:
        return None


def load_price_cache(cache_dir='./price_cache', max_workers=10):
    cache_path = Path(cache_dir)
    
    if not cache_path.exists():
        print(f"❌ Price cache not found at: {cache_path.absolute()}")
        return {}
    
    pkl_files = sorted(cache_path.glob('*.pkl'))
    
    if not pkl_files:
        print(f"❌ No .pkl files found in {cache_dir}")
        return {}
    
    print(f"Loading {len(pkl_files)} assets from {cache_dir}...")
    
    price_data = {}
    excluded_count = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(load_single_pkl, f): f for f in pkl_files}
        
        for future in tqdm(as_completed(futures), total=len(pkl_files), desc="Loading cache"):
            result = future.result()
            if result:
                ticker, df = result
                price_data[ticker] = df
            else:
                excluded_count += 1
    
    print(f"✅ Loaded: {len(price_data)} assets")
    if excluded_count > 0:
        print(f"⚠️  Excluded: {excluded_count} assets")
    
    return price_data


# ============================================================================
# TOM STRATEGY
# ============================================================================

def identify_turn_of_month_days(dates):
    """Mark days in turn-of-month window (T-5 to T+3)"""
    df = pd.DataFrame(index=dates)
    df['year_month'] = df.index.to_period('M')
    df['signal'] = 0
    
    for period in df['year_month'].unique():
        month_mask = df['year_month'] == period
        month_days = df[month_mask].index
        
        if len(month_days) >= 5:
            entry_idx = month_days[-5]
            df.loc[entry_idx:month_days[-1], 'signal'] = 1
            
            next_period = period + 1
            next_mask = df['year_month'] == next_period
            next_days = df[next_mask].index
            
            if len(next_days) >= 3:
                df.loc[next_days[0]:next_days[2], 'signal'] = 1
    
    return df['signal']


def safe_cagr(equity_final, years):
    """Calculate CAGR with safety checks for extreme values"""
    if years <= 0 or equity_final <= 0:
        return 0
    equity_final = np.clip(equity_final, 1e-10, 1e10)
    cagr = (equity_final ** (1/years) - 1) * 100
    return np.clip(cagr, -99.9, 999.9)


def calculate_sortino(returns, cagr):
    """Calculate Sortino ratio (uses downside deviation only)"""
    downside_returns = returns[returns < 0]
    if len(downside_returns) == 0 or cagr == 0:
        return 0
    downside_std = downside_returns.std() * np.sqrt(252)
    if downside_std == 0:
        return 0
    return np.clip(cagr / (downside_std * 100), -10, 10)


def analyze_period(prices, signals, returns, period_name):
    """Analyze a specific time period, return stats dict"""
    
    if len(prices) < 60:  # Need at least ~3 months
        return None
    
    years = (prices.index[-1] - prices.index[0]).days / 365.25
    if years < 0.25:
        return None
    
    # Strategy returns
    long_tom_returns = returns * signals.shift(1).fillna(0)
    long_outside_returns = returns * (1 - signals.shift(1).fillna(0))
    short_tom_returns = -returns * signals.shift(1).fillna(0)
    inverse_tom_returns = short_tom_returns + long_outside_returns
    
    # Equity curves (clipped to avoid overflow)
    bh_equity = (1 + returns).cumprod().clip(1e-10, 1e10)
    long_tom_equity = (1 + long_tom_returns).cumprod().clip(1e-10, 1e10)
    long_outside_equity = (1 + long_outside_returns).cumprod().clip(1e-10, 1e10)
    short_tom_equity = (1 + short_tom_returns).cumprod().clip(1e-10, 1e10)
    inverse_tom_equity = (1 + inverse_tom_returns).cumprod().clip(1e-10, 1e10)
    
    # CAGRs
    bh_cagr = safe_cagr(bh_equity.iloc[-1], years)
    long_tom_cagr = safe_cagr(long_tom_equity.iloc[-1], years)
    long_outside_cagr = safe_cagr(long_outside_equity.iloc[-1], years)
    short_tom_cagr = safe_cagr(short_tom_equity.iloc[-1], years)
    inverse_tom_cagr = safe_cagr(inverse_tom_equity.iloc[-1], years)
    
    # Max drawdowns
    bh_dd = ((bh_equity / bh_equity.cummax() - 1) * 100).min()
    long_tom_dd = ((long_tom_equity / long_tom_equity.cummax() - 1) * 100).min()
    
    # TOM window analysis
    tom_rets = returns[signals.shift(1) == 1].dropna()
    outside_rets = returns[signals.shift(1) == 0].dropna()
    
    avg_tom_return = tom_rets.mean() * 100 if len(tom_rets) > 0 else 0
    avg_outside_return = outside_rets.mean() * 100 if len(outside_rets) > 0 else 0
    tom_win_rate = (tom_rets > 0).mean() * 100 if len(tom_rets) > 0 else 0
    
    return {
        f'{period_name}_Years': years,
        f'{period_name}_BH_CAGR': bh_cagr,
        f'{period_name}_LongTOM_CAGR': long_tom_cagr,
        f'{period_name}_LongTOM_Outperform': long_tom_cagr - bh_cagr,
        f'{period_name}_LongOutside_CAGR': long_outside_cagr,
        f'{period_name}_ShortTOM_CAGR': short_tom_cagr,
        f'{period_name}_InverseTOM_CAGR': inverse_tom_cagr,
        f'{period_name}_InverseTOM_Outperform': inverse_tom_cagr - bh_cagr,
        f'{period_name}_BH_MaxDD': bh_dd,
        f'{period_name}_LongTOM_MaxDD': long_tom_dd,
        f'{period_name}_Avg_TOM_Return': avg_tom_return,
        f'{period_name}_Avg_Outside_Return': avg_outside_return,
        f'{period_name}_TOM_vs_Outside': avg_tom_return - avg_outside_return,
        f'{period_name}_TOM_WinRate': tom_win_rate,
    }


def backtest_single(ticker, df):
    """Run backtest with Full, 5Y, and 1Y period analysis."""
    
    close_col = None
    for col in ['close', 'Close', 'CLOSE']:
        if col in df.columns:
            close_col = col
            break
    
    if close_col is None:
        return None
    
    prices = df[close_col].dropna()
    
    if len(prices) < 252:
        return None
    
    returns = prices.pct_change()
    signals = identify_turn_of_month_days(prices.index)
    signals = signals.reindex(prices.index, fill_value=0)
    
    # ========================================================================
    # FULL HISTORY ANALYSIS
    # ========================================================================
    
    full_stats = analyze_period(prices, signals, returns, 'Full')
    if full_stats is None:
        return None
    
    # ========================================================================
    # LAST 5 YEARS ANALYSIS
    # ========================================================================
    
    five_years_ago = prices.index[-1] - pd.DateOffset(years=5)
    mask_5y = prices.index >= five_years_ago
    
    if mask_5y.sum() >= 252:
        stats_5y = analyze_period(prices[mask_5y], signals[mask_5y], returns[mask_5y], '5Y')
    else:
        stats_5y = {f'5Y_{k}': np.nan for k in ['Years', 'BH_CAGR', 'LongTOM_CAGR', 'LongTOM_Outperform',
                                                  'LongOutside_CAGR', 'ShortTOM_CAGR', 'InverseTOM_CAGR',
                                                  'InverseTOM_Outperform', 'BH_MaxDD', 'LongTOM_MaxDD',
                                                  'Avg_TOM_Return', 'Avg_Outside_Return', 'TOM_vs_Outside', 'TOM_WinRate']}
    
    # ========================================================================
    # LAST 1 YEAR ANALYSIS
    # ========================================================================
    
    one_year_ago = prices.index[-1] - pd.DateOffset(years=1)
    mask_1y = prices.index >= one_year_ago
    
    if mask_1y.sum() >= 60:
        stats_1y = analyze_period(prices[mask_1y], signals[mask_1y], returns[mask_1y], '1Y')
    else:
        stats_1y = {f'1Y_{k}': np.nan for k in ['Years', 'BH_CAGR', 'LongTOM_CAGR', 'LongTOM_Outperform',
                                                  'LongOutside_CAGR', 'ShortTOM_CAGR', 'InverseTOM_CAGR',
                                                  'InverseTOM_Outperform', 'BH_MaxDD', 'LongTOM_MaxDD',
                                                  'Avg_TOM_Return', 'Avg_Outside_Return', 'TOM_vs_Outside', 'TOM_WinRate']}
    
    # ========================================================================
    # REGIME CHANGE DETECTION
    # ========================================================================
    
    # Compare 1Y vs Full
    if stats_1y and stats_5y:
        regime_1y_vs_full = (stats_1y.get('1Y_LongTOM_Outperform', 0) or 0) - (full_stats.get('Full_LongTOM_Outperform', 0) or 0)
        regime_1y_vs_5y = (stats_1y.get('1Y_LongTOM_Outperform', 0) or 0) - (stats_5y.get('5Y_LongTOM_Outperform', 0) or 0)
        regime_5y_vs_full = (stats_5y.get('5Y_LongTOM_Outperform', 0) or 0) - (full_stats.get('Full_LongTOM_Outperform', 0) or 0)
    else:
        regime_1y_vs_full = np.nan
        regime_1y_vs_5y = np.nan
        regime_5y_vs_full = np.nan
    
    # ========================================================================
    # COMBINE RESULTS
    # ========================================================================
    
    result = {'Ticker': ticker}
    result.update(full_stats)
    if stats_5y:
        result.update(stats_5y)
    if stats_1y:
        result.update(stats_1y)
    
    result['Regime_1Y_vs_Full'] = regime_1y_vs_full
    result['Regime_1Y_vs_5Y'] = regime_1y_vs_5y
    result['Regime_5Y_vs_Full'] = regime_5y_vs_full
    
    result['Start'] = prices.index[0].strftime('%Y-%m-%d')
    result['End'] = prices.index[-1].strftime('%Y-%m-%d')
    
    return result


def run_full_analysis(price_data):
    results = []
    print(f"\nBacktesting {len(price_data)} assets...")
    
    for ticker, df in tqdm(price_data.items(), desc="Backtesting"):
        try:
            stats = backtest_single(ticker, df)
            if stats:
                results.append(stats)
        except Exception as e:
            continue
    
    print(f"✅ Completed {len(results):,} backtests")
    return pd.DataFrame(results)


def fmt_pct(x, decimals=2):
    if pd.isna(x):
        return "N/A"
    return f"{x:+.{decimals}f}%" if x >= 0 or x < 0 else f"{x:.{decimals}f}%"


def fmt_pct_nosign(x, decimals=2):
    if pd.isna(x):
        return "N/A"
    return f"{x:.{decimals}f}%"


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    
    print("\n" + "="*80)
    print("TURN OF MONTH ANALYSIS - MULTI-PERIOD REGIME DETECTION")
    print("="*80)
    
    price_data = load_price_cache('./price_cache', max_workers=10)
    
    if not price_data:
        print("❌ No data to analyze")
        exit()
    
    df = run_full_analysis(price_data)
    
    if len(df) == 0:
        print("❌ No valid results")
        exit()
    
    # ========================================================================
    # SUMMARY STATISTICS
    # ========================================================================
    
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    print(f"\nTotal assets analyzed: {len(df):,}")
    
    for period in ['Full', '5Y', '1Y']:
        col = f'{period}_LongTOM_Outperform'
        if col in df.columns:
            valid = df[col].dropna()
            winners = (valid > 0).sum()
            print(f"\n📊 {period} PERIOD:")
            print(f"   Assets with data: {len(valid):,}")
            print(f"   Long TOM Winners: {winners} ({winners/len(valid)*100:.1f}%)")
            print(f"   Avg Long TOM Outperform: {valid.mean():+.2f}%")
            print(f"   Median Long TOM Outperform: {valid.median():+.2f}%")
    
    # ========================================================================
    # REGIME CHANGES (1Y vs 5Y vs Full)
    # ========================================================================
    
    print("\n" + "="*80)
    print("REGIME CHANGE DETECTION")
    print("="*80)
    
    # TOM effectiveness INCREASING recently (1Y better than history)
    df_valid = df.dropna(subset=['Regime_1Y_vs_Full', '1Y_LongTOM_Outperform'])
    
    improving = df_valid[df_valid['Regime_1Y_vs_Full'] > 5].sort_values('Regime_1Y_vs_Full', ascending=False)
    degrading = df_valid[df_valid['Regime_1Y_vs_Full'] < -5].sort_values('Regime_1Y_vs_Full', ascending=True)
    
    print(f"\n🔺 TOM EFFECTIVENESS IMPROVING ({len(improving)} assets):")
    print("   (1Y outperformance > Full history by >5%)")
    
    if len(improving) > 0:
        disp = improving.head(20)[['Ticker', 'Full_LongTOM_Outperform', '5Y_LongTOM_Outperform', 
                                   '1Y_LongTOM_Outperform', 'Regime_1Y_vs_Full']].copy()
        disp['Full_LongTOM_Outperform'] = disp['Full_LongTOM_Outperform'].apply(lambda x: fmt_pct(x))
        disp['5Y_LongTOM_Outperform'] = disp['5Y_LongTOM_Outperform'].apply(lambda x: fmt_pct(x))
        disp['1Y_LongTOM_Outperform'] = disp['1Y_LongTOM_Outperform'].apply(lambda x: fmt_pct(x))
        disp['Regime_1Y_vs_Full'] = disp['Regime_1Y_vs_Full'].apply(lambda x: fmt_pct(x))
        print(disp.to_string(index=False))
    
    print(f"\n🔻 TOM EFFECTIVENESS DEGRADING ({len(degrading)} assets):")
    print("   (1Y outperformance < Full history by >5%)")
    
    if len(degrading) > 0:
        disp = degrading.head(20)[['Ticker', 'Full_LongTOM_Outperform', '5Y_LongTOM_Outperform', 
                                   '1Y_LongTOM_Outperform', 'Regime_1Y_vs_Full']].copy()
        disp['Full_LongTOM_Outperform'] = disp['Full_LongTOM_Outperform'].apply(lambda x: fmt_pct(x))
        disp['5Y_LongTOM_Outperform'] = disp['5Y_LongTOM_Outperform'].apply(lambda x: fmt_pct(x))
        disp['1Y_LongTOM_Outperform'] = disp['1Y_LongTOM_Outperform'].apply(lambda x: fmt_pct(x))
        disp['Regime_1Y_vs_Full'] = disp['Regime_1Y_vs_Full'].apply(lambda x: fmt_pct(x))
        print(disp.to_string(index=False))
    
    # ========================================================================
    # TOP LONG TOM CANDIDATES BY PERIOD
    # ========================================================================
    
    for period in ['Full', '5Y', '1Y']:
        col = f'{period}_LongTOM_Outperform'
        if col not in df.columns:
            continue
            
        print("\n" + "="*80)
        print(f"TOP 25 LONG TOM CANDIDATES ({period})")
        print("="*80)
        
        sorted_df = df.dropna(subset=[col]).sort_values(col, ascending=False).head(25)
        
        disp_cols = ['Ticker', f'{period}_BH_CAGR', f'{period}_LongTOM_CAGR', col,
                     f'{period}_TOM_vs_Outside', f'{period}_TOM_WinRate']
        
        disp = sorted_df[disp_cols].copy()
        disp[f'{period}_BH_CAGR'] = disp[f'{period}_BH_CAGR'].apply(lambda x: fmt_pct_nosign(x))
        disp[f'{period}_LongTOM_CAGR'] = disp[f'{period}_LongTOM_CAGR'].apply(lambda x: fmt_pct_nosign(x))
        disp[col] = disp[col].apply(lambda x: fmt_pct(x))
        disp[f'{period}_TOM_vs_Outside'] = disp[f'{period}_TOM_vs_Outside'].apply(lambda x: fmt_pct(x, 4))
        disp[f'{period}_TOM_WinRate'] = disp[f'{period}_TOM_WinRate'].apply(lambda x: fmt_pct_nosign(x, 1))
        
        print(disp.to_string(index=False))
    
    # ========================================================================
    # TOP SHORT TOM CANDIDATES BY PERIOD
    # ========================================================================
    
    for period in ['Full', '5Y', '1Y']:
        col = f'{period}_ShortTOM_CAGR'
        if col not in df.columns:
            continue
            
        print("\n" + "="*80)
        print(f"TOP 20 SHORT TOM CANDIDATES ({period})")
        print("="*80)
        
        sorted_df = df.dropna(subset=[col]).sort_values(col, ascending=False).head(20)
        
        disp_cols = ['Ticker', f'{period}_BH_CAGR', col, f'{period}_Avg_TOM_Return',
                     f'{period}_TOM_vs_Outside', f'{period}_TOM_WinRate']
        
        disp = sorted_df[disp_cols].copy()
        disp[f'{period}_BH_CAGR'] = disp[f'{period}_BH_CAGR'].apply(lambda x: fmt_pct_nosign(x))
        disp[col] = disp[col].apply(lambda x: fmt_pct(x))
        disp[f'{period}_Avg_TOM_Return'] = disp[f'{period}_Avg_TOM_Return'].apply(lambda x: fmt_pct(x, 4))
        disp[f'{period}_TOM_vs_Outside'] = disp[f'{period}_TOM_vs_Outside'].apply(lambda x: fmt_pct(x, 4))
        disp[f'{period}_TOM_WinRate'] = disp[f'{period}_TOM_WinRate'].apply(lambda x: fmt_pct_nosign(x, 1))
        
        print(disp.to_string(index=False))
    
    # ========================================================================
    # CONSISTENT PERFORMERS (works across all periods)
    # ========================================================================
    
    print("\n" + "="*80)
    print("CONSISTENT LONG TOM PERFORMERS (positive in Full, 5Y, AND 1Y)")
    print("="*80)
    
    consistent = df[
        (df['Full_LongTOM_Outperform'] > 0) & 
        (df['5Y_LongTOM_Outperform'] > 0) & 
        (df['1Y_LongTOM_Outperform'] > 0)
    ].sort_values('1Y_LongTOM_Outperform', ascending=False)
    
    print(f"\nFound {len(consistent)} assets with consistent positive TOM edge:\n")
    
    if len(consistent) > 0:
        disp = consistent.head(30)[['Ticker', 'Full_LongTOM_Outperform', '5Y_LongTOM_Outperform', 
                                     '1Y_LongTOM_Outperform', '1Y_TOM_WinRate']].copy()
        disp['Full_LongTOM_Outperform'] = disp['Full_LongTOM_Outperform'].apply(lambda x: fmt_pct(x))
        disp['5Y_LongTOM_Outperform'] = disp['5Y_LongTOM_Outperform'].apply(lambda x: fmt_pct(x))
        disp['1Y_LongTOM_Outperform'] = disp['1Y_LongTOM_Outperform'].apply(lambda x: fmt_pct(x))
        disp['1Y_TOM_WinRate'] = disp['1Y_TOM_WinRate'].apply(lambda x: fmt_pct_nosign(x, 1))
        print(disp.to_string(index=False))
    
    # ========================================================================
    # NEWLY WORKING (bad history, good recent)
    # ========================================================================
    
    print("\n" + "="*80)
    print("NEWLY WORKING TOM (negative Full, but positive 1Y)")
    print("="*80)
    
    newly_working = df[
        (df['Full_LongTOM_Outperform'] < 0) & 
        (df['1Y_LongTOM_Outperform'] > 3)
    ].sort_values('1Y_LongTOM_Outperform', ascending=False)
    
    print(f"\nFound {len(newly_working)} assets where TOM recently started working:\n")
    
    if len(newly_working) > 0:
        disp = newly_working.head(20)[['Ticker', 'Full_LongTOM_Outperform', '5Y_LongTOM_Outperform', 
                                        '1Y_LongTOM_Outperform', 'Regime_1Y_vs_Full']].copy()
        disp['Full_LongTOM_Outperform'] = disp['Full_LongTOM_Outperform'].apply(lambda x: fmt_pct(x))
        disp['5Y_LongTOM_Outperform'] = disp['5Y_LongTOM_Outperform'].apply(lambda x: fmt_pct(x))
        disp['1Y_LongTOM_Outperform'] = disp['1Y_LongTOM_Outperform'].apply(lambda x: fmt_pct(x))
        disp['Regime_1Y_vs_Full'] = disp['Regime_1Y_vs_Full'].apply(lambda x: fmt_pct(x))
        print(disp.to_string(index=False))
    
    # ========================================================================
    # STOPPED WORKING (good history, bad recent)
    # ========================================================================
    
    print("\n" + "="*80)
    print("STOPPED WORKING TOM (positive Full, but negative 1Y)")
    print("="*80)
    
    stopped_working = df[
        (df['Full_LongTOM_Outperform'] > 3) & 
        (df['1Y_LongTOM_Outperform'] < 0)
    ].sort_values('1Y_LongTOM_Outperform', ascending=True)
    
    print(f"\nFound {len(stopped_working)} assets where TOM stopped working:\n")
    
    if len(stopped_working) > 0:
        disp = stopped_working.head(20)[['Ticker', 'Full_LongTOM_Outperform', '5Y_LongTOM_Outperform', 
                                          '1Y_LongTOM_Outperform', 'Regime_1Y_vs_Full']].copy()
        disp['Full_LongTOM_Outperform'] = disp['Full_LongTOM_Outperform'].apply(lambda x: fmt_pct(x))
        disp['5Y_LongTOM_Outperform'] = disp['5Y_LongTOM_Outperform'].apply(lambda x: fmt_pct(x))
        disp['1Y_LongTOM_Outperform'] = disp['1Y_LongTOM_Outperform'].apply(lambda x: fmt_pct(x))
        disp['Regime_1Y_vs_Full'] = disp['Regime_1Y_vs_Full'].apply(lambda x: fmt_pct(x))
        print(disp.to_string(index=False))
    
    # ========================================================================
    # EXPORT RESULTS
    # ========================================================================
    
    print("\n" + "="*80)
    print("EXPORTING RESULTS")
    print("="*80)
    
    # Full results
    df.to_csv('tom_ALL_results.csv', index=False)
    print(f"✅ Full results: tom_ALL_results.csv ({len(df):,} assets)")
    
    # Period-specific exports
    for period in ['Full', '5Y', '1Y']:
        col = f'{period}_LongTOM_Outperform'
        if col in df.columns:
            winners = df[df[col] > 0].sort_values(col, ascending=False)
            winners.to_csv(f'tom_LONG_winners_{period}.csv', index=False)
            print(f"✅ Long TOM winners ({period}): tom_LONG_winners_{period}.csv ({len(winners):,} assets)")
    
    # Regime change exports
    if len(improving) > 0:
        improving.to_csv('tom_REGIME_improving.csv', index=False)
        print(f"✅ TOM improving: tom_REGIME_improving.csv ({len(improving):,} assets)")
    
    if len(degrading) > 0:
        degrading.to_csv('tom_REGIME_degrading.csv', index=False)
        print(f"✅ TOM degrading: tom_REGIME_degrading.csv ({len(degrading):,} assets)")
    
    if len(consistent) > 0:
        consistent.to_csv('tom_CONSISTENT_performers.csv', index=False)
        print(f"✅ Consistent performers: tom_CONSISTENT_performers.csv ({len(consistent):,} assets)")
    
    if len(newly_working) > 0:
        newly_working.to_csv('tom_NEWLY_working.csv', index=False)
        print(f"✅ Newly working: tom_NEWLY_working.csv ({len(newly_working):,} assets)")
    
    if len(stopped_working) > 0:
        stopped_working.to_csv('tom_STOPPED_working.csv', index=False)
        print(f"✅ Stopped working: tom_STOPPED_working.csv ({len(stopped_working):,} assets)")
    
    # Short candidates by period
    for period in ['Full', '5Y', '1Y']:
        col = f'{period}_ShortTOM_CAGR'
        if col in df.columns:
            shorts = df[df[col] > 2].sort_values(col, ascending=False)
            if len(shorts) > 0:
                shorts.to_csv(f'tom_SHORT_candidates_{period}.csv', index=False)
                print(f"✅ Short candidates ({period}): tom_SHORT_candidates_{period}.csv ({len(shorts):,} assets)")
    
    # Summary
    summary = {
        'Total_Assets': len(df),
        'Full_Long_TOM_Winners': (df['Full_LongTOM_Outperform'] > 0).sum(),
        '5Y_Long_TOM_Winners': (df['5Y_LongTOM_Outperform'] > 0).sum(),
        '1Y_Long_TOM_Winners': (df['1Y_LongTOM_Outperform'] > 0).sum(),
        'Consistent_Performers': len(consistent),
        'Newly_Working': len(newly_working),
        'Stopped_Working': len(stopped_working),
        'Regime_Improving': len(improving),
        'Regime_Degrading': len(degrading),
        'Run_Date': datetime.now().strftime('%Y-%m-%d %H:%M')
    }
    
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv('tom_SUMMARY.csv', index=False)
    print(f"✅ Summary: tom_SUMMARY.csv")
    
    print("\n" + "="*80)
    print("✅ COMPLETE")
    print("="*80 + "\n")
