# -*- coding: utf-8 -*-
# =============================================================================
# INTERMARKET SPREAD MONITOR - v2.5.1
# Last updated: 2026-02-18
# =============================================================================
# v2.5.0:
#   - Replaced Django JSON output with static HTML via dashboard_writer.py
#   - Removed write_html() (now handled by DashboardWriter)
#   - Output goes to market-dashboards repo: docs/spread-monitor/
#   - CSV output preserved in output/spread_monitor/ (local archive)
#   - All compute logic unchanged from v2.4.2
# v2.4.2:
#   - Restored richer HTML/CSS styling (v2.3 look) with ASCII-only source
#   - Kept momentum-aware qualifiers (CONFIRMED / FADING / DIVERGING / HOLDING)
#   - Preserved regime scoring, force scoring, and key risk selection
# =============================================================================

import os
import pickle
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from dashboard_writer import DashboardWriter

# =============================================================================
# CONFIG
# =============================================================================

# Base dir = perplexity-user-data (one level up from this script's location)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR   = os.path.join(_SCRIPT_DIR, "..", "perplexity-user-data")

CONFIG = {
    "cache_dir":      os.path.normpath(os.path.join(_DATA_DIR, "price_cache")),
    "csv_output_dir": os.path.normpath(os.path.join(_DATA_DIR, "output", "spread_monitor")),   # local CSV archive only
    "sma_period": 50,
    "slope_lookback": 10,
    "slope_flat_threshold": 0.001,
    "max_workers": 8,
    "load_timeout_sec": 10,
    "sparkline_days": 50,
    "price_col_priority": [
        "adjClose",
        "adj_close",
        "Adj Close",
        "close",
        "Close",
    ],
}

# =============================================================================
# MARKET LEVEL CATEGORIES
# =============================================================================

FORCE_LABELS = {
    "Gravity": "I: Rates / Duration",
    "Electromagnetism": "II: Earnings / Fundamentals",
    "Strong Force": "III: Liquidity / Structure",
    "Weak Force": "IV: Sentiment / Positioning",
}

FORCE_COLORS = {
    "Gravity": "#b39ddb",
    "Electromagnetism": "#fff176",
    "Strong Force": "#4fc3f7",
    "Weak Force": "#ef9a9a",
}

FORCE_SORT_ORDER = {
    "Gravity": 1,
    "Electromagnetism": 2,
    "Strong Force": 3,
    "Weak Force": 4,
}

# =============================================================================
# SPREAD DEFINITIONS
# =============================================================================

SPREADS = [
    {
        "name": "SPY/TLT",
        "num": "SPY",
        "den": "TLT",
        "category": "Risk-On/Off",
        "force": "Gravity",
        "rising_means": "Risk-on, equities preferred over bonds",
        "falling_means": "Risk-off, flight to safety into bonds",
        "rising_playbook": (
            "Favor equity longs, underweight duration. Dip-buy equities near SMA."
        ),
        "falling_playbook": (
            "Favor long-duration Treasuries, reduce equity beta. Hedge with TLT calls."
        ),
    },
    {
        "name": "HYG/LQD",
        "num": "HYG",
        "den": "LQD",
        "category": "Credit Confidence",
        "force": "Strong Force",
        "rising_means": (
            "Credit confidence, risk appetite, junk outperforming IG"
        ),
        "falling_means": (
            "Credit stress, risk aversion, flight to quality in bonds"
        ),
        "rising_playbook": (
            "HY credit is safe to hold, favor carry trades. "
            "Add risk in credit-sensitive names."
        ),
        "falling_playbook": (
            "Reduce HY exposure, rotate to IG or Treasuries. "
            "Watch for equity follow-through."
        ),
    },
    {
        "name": "IWM/SPY",
        "num": "IWM",
        "den": "SPY",
        "category": "Breadth / Small-Cap Confidence",
        "force": "Strong Force",
        "rising_means": (
            "Small-cap confidence, broad risk-on, market breadth expanding"
        ),
        "falling_means": (
            "Large-cap hiding, risk-off, narrow leadership"
        ),
        "rising_playbook": (
            "Broaden equity exposure into small/mid caps. "
            "IWM longs, sell large-cap premium."
        ),
        "falling_playbook": (
            "Stick to mega-cap quality. Avoid small-cap longs, "
            "narrow leadership = fragile rally."
        ),
    },
    {
        "name": "EEM/SPY",
        "num": "EEM",
        "den": "SPY",
        "category": "Global Risk Appetite",
        "force": "Strong Force",
        "rising_means": "Global risk-on, EM bid, dollar weakening",
        "falling_means": "EM stress, flight to quality, dollar strengthening",
        "rising_playbook": (
            "Add EM equity/FX exposure. Short DXY or UUP. Favor commodity exporters."
        ),
        "falling_playbook": (
            "Underweight EM, favor US large cap. Long DXY, avoid commodity-linked EM."
        ),
    },
    {
        "name": "ARKK/QQQ",
        "num": "ARKK",
        "den": "QQQ",
        "category": "Speculative Sentiment",
        "force": "Weak Force",
        "rising_means": (
            "Speculative froth, maximum risk-on, unprofitable growth bid"
        ),
        "falling_means": (
            "De-risking, quality rotation within growth"
        ),
        "rising_playbook": (
            "Speculative names working -- ride momentum but tighten stops. "
            "Late-cycle warning if parabolic."
        ),
        "falling_playbook": (
            "Rotate to profitable growth (QQQ, XLK). Avoid unprofitable tech, "
            "meme stocks, SPACs."
        ),
    },
    {
        "name": "TLT/HYG",
        "num": "TLT",
        "den": "HYG",
        "category": "Credit Stress",
        "force": "Gravity",
        "rising_means": (
            "Credit stress rising, spreads widening, risk-off -- often leads equity weakness"
        ),
        "falling_means": (
            "Risk appetite, credit spreads tightening, risk-on"
        ),
        "rising_playbook": (
            "Defensive posture -- raise cash, add TLT, reduce credit exposure. "
            "Often leads SPX by 2-4 weeks."
        ),
        "falling_playbook": (
            "Credit healing -- re-engage risk. Favor HY carry, reduce hedges."
        ),
    },
    {
        "name": "JNK/LQD",
        "num": "JNK",
        "den": "LQD",
        "category": "Credit Risk Appetite",
        "force": "Strong Force",
        "rising_means": (
            "Junk outperforming IG, credit confidence, yield-seeking"
        ),
        "falling_means": (
            "Investment grade outperforming, credit stress, de-risking"
        ),
        "rising_playbook": (
            "Yield-seeking environment -- favor HY, leveraged credit, dividend growers."
        ),
        "falling_playbook": (
            "De-risk credit book. Rotate JNK to LQD or AGG. "
            "Widen stop-losses on equity."
        ),
    },
    {
        "name": "COPX/GLD",
        "num": "COPX",
        "den": "GLD",
        "category": "Global Growth (Copper/Gold)",
        "force": "Electromagnetism",
        "rising_means": (
            "Growth optimism, inflation expectations, rising rate expectations "
            "-- tracks ISM and 10Y closely"
        ),
        "falling_means": (
            "Recession risk, deflation fears, rate cuts expected"
        ),
        "rising_playbook": (
            "Favor cyclicals, commodities, and industrials. "
            "Long copper miners, short gold. Rates going higher."
        ),
        "falling_playbook": (
            "Favor gold over copper, add duration. Recession playbook -- "
            "utilities, staples, Treasuries."
        ),
    },
    {
        "name": "XLB/XLU",
        "num": "XLB",
        "den": "XLU",
        "category": "Growth Expectations (Real-Time PMI)",
        "force": "Electromagnetism",
        "rising_means": (
            "Growth expectations rising, cyclical demand, functions as "
            "daily-updated ISM PMI"
        ),
        "falling_means": (
            "Growth expectations falling, defensive rotation, recession positioning"
        ),
        "rising_playbook": (
            "ISM proxy expanding -- favor materials, industrials, cyclical value. "
            "Sell utility premium."
        ),
        "falling_playbook": (
            "ISM proxy contracting -- rotate to defensives. Long XLU, short XLB. "
            "Reduce cyclical exposure."
        ),
    },
    {
        "name": "XLY/XLP",
        "num": "XLY",
        "den": "XLP",
        "category": "Consumer Confidence / Cycle",
        "force": "Weak Force",
        "rising_means": (
            "Consumer confidence, discretionary spending strong -- parabolic readings "
            "are late-cycle warning"
        ),
        "falling_means": (
            "Recession fears, defensive rotation into staples"
        ),
        "rising_playbook": (
            "Consumer is healthy -- favor retail, travel, housing. "
            "Watch for parabolic = late-cycle top."
        ),
        "falling_playbook": (
            "Consumer weakening -- favor staples, healthcare, discount retail. "
            "Avoid discretionary longs."
        ),
    },
    {
        "name": "SMH/SPY",
        "num": "SMH",
        "den": "SPY",
        "category": "Semiconductor Leadership",
        "force": "Electromagnetism",
        "rising_means": (
            "Tech/semis leading, growth cycle intact -- semis are leading economic indicator"
        ),
        "falling_means": (
            "Semis lagging, leading indicator of economic weakness"
        ),
        "rising_playbook": (
            "Semis leading = growth intact. Overweight tech, favor NVDA/AVGO/AMD. "
            "Add on pullbacks."
        ),
        "falling_playbook": (
            "Semis rolling over = canary in the coal mine. Reduce tech, raise cash. "
            "Often leads SPX lower by 4-8 weeks."
        ),
    },
    {
        "name": "IYT/SPY",
        "num": "IYT",
        "den": "SPY",
        "category": "Transport Confirmation (Dow Theory)",
        "force": "Electromagnetism",
        "rising_means": (
            "Transports confirming equity rally, Dow Theory bullish, real economy strong"
        ),
        "falling_means": (
            "Transport divergence, economic weakness signal, Dow Theory bearish"
        ),
        "rising_playbook": (
            "Rally confirmed by real economy. Trust the uptrend, buy dips. "
            "Favor logistics, rails, truckers."
        ),
        "falling_playbook": (
            "Dow Theory non-confirmation -- do not trust equity highs. "
            "Reduce longs, raise cash on bounces."
        ),
    },
    {
        "name": "TIP/IEF",
        "num": "TIP",
        "den": "IEF",
        "category": "Inflation Expectations",
        "force": "Gravity",
        "rising_means": (
            "Inflation expectations rising, TIPS outperforming nominals, "
            "breakeven widening"
        ),
        "falling_means": (
            "Inflation expectations falling, disinflation, nominals outperforming"
        ),
        "rising_playbook": (
            "Inflation bid -- favor TIPS over nominals, commodities, real assets. "
            "Avoid long duration."
        ),
        "falling_playbook": (
            "Disinflation trade -- favor nominal Treasuries, growth over value, "
            "long duration."
        ),
    },
    {
        "name": "XLE/XLF",
        "num": "XLE",
        "den": "XLF",
        "category": "Inflation vs Credit Cycle",
        "force": "Electromagnetism",
        "rising_means": (
            "Energy outperforming financials, commodity inflation cycle, supply-driven"
        ),
        "falling_means": (
            "Financials outperforming energy, credit expansion cycle, demand-driven"
        ),
        "rising_playbook": (
            "Commodity inflation regime -- overweight energy, "
            "underweight rate-sensitive financials."
        ),
        "falling_playbook": (
            "Credit expansion regime -- favor banks, brokers, insurance. "
            "Underweight energy."
        ),
    },
    {
        "name": "GLD/SLV",
        "num": "GLD",
        "den": "SLV",
        "category": "Gold/Silver Ratio (Fear Gauge)",
        "force": "Weak Force",
        "rising_means": (
            "Fear rising, safe-haven demand, economic pessimism -- "
            "readings above 80 signal silver undervaluation"
        ),
        "falling_means": (
            "Industrial optimism, silver industrial demand rising, growth confidence"
        ),
        "rising_playbook": (
            "Fear trade -- hold gold, reduce silver. If ratio > 80, "
            "watch for silver mean-reversion long."
        ),
        "falling_playbook": (
            "Industrial metals bid -- favor silver over gold, add copper/silver miners. "
            "Growth confidence."
        ),
    },
    {
        "name": "SIL/SLV",
        "num": "SIL",
        "den": "SLV",
        "category": "Silver Miner Confidence",
        "force": "Weak Force",
        "rising_means": (
            "Miners outperforming metal, operating leverage in play, equity risk appetite"
        ),
        "falling_means": (
            "Metal outperforming miners, operational risk aversion, safety preference"
        ),
        "rising_playbook": (
            "Miners leveraging silver rally -- overweight SIL/SILJ over physical SLV. "
            "Equity risk appetite intact."
        ),
        "falling_playbook": (
            "Stick to physical silver (SLV) over miners. "
            "Miner underperformance = operational or equity risk."
        ),
    },
    {
        "name": "GDX/GLD",
        "num": "GDX",
        "den": "GLD",
        "category": "Gold Miner Confidence",
        "force": "Weak Force",
        "rising_means": (
            "Miners outperforming bullion, speculative risk-on in gold equities"
        ),
        "falling_means": (
            "Bullion outperforming miners, safety preference, operational risk aversion"
        ),
        "rising_playbook": (
            "Gold miners leveraging -- overweight GDX/GDXJ over GLD. "
            "Add junior miners for beta."
        ),
        "falling_playbook": (
            "Hold bullion (GLD/IAU) over miners. Miner underperformance = avoid GDX, "
            "trim juniors."
        ),
    },
    {
        "name": "IWF/IWD",
        "num": "IWF",
        "den": "IWD",
        "category": "Growth vs Value Rotation",
        "force": "Weak Force",
        "rising_means": (
            "Growth outperforming value, momentum/duration trade, low-rate environment favored"
        ),
        "falling_means": (
            "Value outperforming growth, rate-hiking regime, late-cycle or mean-reversion"
        ),
        "rising_playbook": (
            "Growth regime -- overweight QQQ/IWF, favor duration-sensitive tech. "
            "Underweight banks, energy."
        ),
        "falling_playbook": (
            "Value regime -- favor XLF, XLE, XLI. Underweight long-duration growth. "
            "Rates rising."
        ),
    },
    {
        "name": "XHB/SPY",
        "num": "XHB",
        "den": "SPY",
        "category": "Housing Cycle",
        "force": "Gravity",
        "rising_means": (
            "Housing leading, rate-sensitive consumer confidence strong"
        ),
        "falling_means": (
            "Housing weakness, rate sensitivity biting, consumer stress"
        ),
        "rising_playbook": (
            "Housing leading -- favor homebuilders, building materials, mortgage REITs. "
            "Consumer is strong."
        ),
        "falling_playbook": (
            "Housing rolling over -- avoid homebuilders, reduce REIT exposure. "
            "Rates hurting consumers."
        ),
    },
]

# =============================================================================
# DATA LOADING
# =============================================================================


def resolve_price_col(df):
    for col in CONFIG["price_col_priority"]:
        if col in df.columns:
            return col
    raise ValueError(
        "No price column found. Available columns: {}".format(list(df.columns))
    )


def load_one_pkl(filepath):
    try:
        ticker = os.path.splitext(os.path.basename(filepath))[0]
        with open(filepath, "rb") as f:
            df = pickle.load(f)

        if isinstance(df, pd.Series):
            df = df.to_frame(name="close")

        if (
            not isinstance(df, pd.DataFrame)
            or len(df)
            < CONFIG["sma_period"] + CONFIG["slope_lookback"]
        ):
            return None

        col = resolve_price_col(df)
        series = df[col].dropna()
        series.index = pd.to_datetime(series.index)
        series = series.sort_index()
        return ticker, series
    except Exception:
        return None


def load_price_cache(tickers_needed):
    cache_dir = CONFIG["cache_dir"]
    results = {}
    files_to_load = []

    for t in tickers_needed:
        path = os.path.join(cache_dir, "{}.pkl".format(t))
        if os.path.exists(path):
            files_to_load.append(path)

    with ThreadPoolExecutor(
        max_workers=CONFIG["max_workers"]
    ) as executor:
        futures = {
            executor.submit(load_one_pkl, fp): fp for fp in files_to_load
        }
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                ticker, series = result
                results[ticker] = series

    return results


# =============================================================================
# SPREAD CALCULATION
# =============================================================================


def compute_spread_metrics(
    num_series, den_series, sma_period, slope_lookback, slope_flat_threshold
):
    combined = pd.DataFrame({"num": num_series, "den": den_series}).dropna()

    if len(combined) < sma_period + slope_lookback + 1:
        return None

    ratio = combined["num"] / combined["den"]
    sma = ratio.rolling(window=sma_period).mean()

    latest_ratio = ratio.iloc[-1]
    latest_sma = sma.iloc[-1]

    if pd.isna(latest_sma):
        return None

    above_sma = latest_ratio > latest_sma

    sma_now = sma.iloc[-1]
    sma_prev = sma.iloc[-1 - slope_lookback]

    if pd.isna(sma_prev) or sma_prev == 0:
        return None

    sma_roc = (sma_now - sma_prev) / abs(sma_prev)

    if abs(sma_roc) < slope_flat_threshold:
        slope_label = "FLAT"
    elif sma_roc > 0:
        slope_label = "RISING"
    else:
        slope_label = "FALLING"

    if above_sma and slope_label == "RISING":
        signal_score = 2
    elif above_sma and slope_label in ("FLAT", "FALLING"):
        signal_score = 1
    elif (not above_sma) and slope_label in ("FLAT", "RISING"):
        signal_score = -1
    else:
        signal_score = -2

    dist_pct = ((latest_ratio - latest_sma) / latest_sma) * 100.0

    spark_days = CONFIG["sparkline_days"]
    if len(ratio) >= spark_days:
        sparkline = ratio.iloc[-spark_days:].tolist()
    else:
        sparkline = ratio.tolist()

    mom_5d = None
    mom_20d = None
    if len(ratio) > 5:
        mom_5d = (latest_ratio - ratio.iloc[-5]) / ratio.iloc[-5] * 100.0
    if len(ratio) > 20:
        mom_20d = (latest_ratio - ratio.iloc[-20]) / ratio.iloc[-20] * 100.0

    return {
        "latest_ratio": round(float(latest_ratio), 6),
        "sma_50": round(float(latest_sma), 6),
        "above_sma": bool(above_sma),
        "sma_slope": slope_label,
        "sma_roc_pct": round(float(sma_roc * 100.0), 4),
        "signal_score": int(signal_score),
        "dist_from_sma_pct": round(float(dist_pct), 2),
        "sparkline": sparkline,
        "mom_5d_pct": round(float(mom_5d), 2) if mom_5d is not None else None,
        "mom_20d_pct": round(float(mom_20d), 2) if mom_20d is not None else None,
    }


# =============================================================================
# INTERPRETATION (MOMENTUM-AWARE)
# =============================================================================


def interpret_spread(
    above_sma,
    sma_slope,
    rising_means,
    falling_means,
    mom_5d_pct=None,
    mom_20d_pct=None,
):
    if above_sma:
        meaning = rising_means
        if sma_slope == "RISING":
            qualifier = "CONFIRMED"
        elif sma_slope == "FLAT":
            qualifier = "HOLDING"
        else:
            qualifier = "FADING"
    else:
        meaning = falling_means
        if sma_slope == "FALLING":
            qualifier = "CONFIRMED"
        elif sma_slope == "FLAT":
            qualifier = "HOLDING BELOW"
        else:
            qualifier = "POTENTIAL_DIVERGENCE"

    if (not above_sma) and sma_slope == "RISING":
        has_pos_momentum = False
        if mom_5d_pct is not None and mom_5d_pct > 0:
            has_pos_momentum = True
        if mom_20d_pct is not None and mom_20d_pct > 0:
            has_pos_momentum = True

        if has_pos_momentum:
            qualifier = "DIVERGING"
        else:
            qualifier = "HOLDING BELOW"

    return meaning, qualifier


# =============================================================================
# REGIME SCORING
# =============================================================================

INVERTED = {"TLT/HYG", "GLD/SLV"}


def compute_regime(rows):
    total_score = 0
    max_possible = 0

    for r in rows:
        score = r["signal_score"]
        if r["ratio_name"] in INVERTED:
            score = -score
        total_score += score
        max_possible += 2

    if max_possible == 0:
        return "NO DATA", 0, 0

    normalized = total_score / float(max_possible)

    if normalized > 0.4:
        label = "RISK-ON"
    elif normalized > 0.1:
        label = "LEANING RISK-ON"
    elif normalized > -0.1:
        label = "MIXED / TRANSITIONING"
    elif normalized > -0.4:
        label = "LEANING RISK-OFF"
    else:
        label = "RISK-OFF"

    return label, total_score, max_possible


def compute_force_scores(rows):
    by_force = {}
    for r in rows:
        f = r.get("force")
        if not f:
            continue
        by_force.setdefault(f, []).append(r["signal_score"])

    out = {}
    for f, scores in by_force.items():
        max_possible = 2 * len(scores)
        total = sum(scores)
        if max_possible == 0:
            norm = 0.0
        else:
            norm = total / float(max_possible)
        out[f] = {
            "total": total,
            "max": max_possible,
            "normalized": norm,
        }
    return out


# =============================================================================
# KEY RISK SELECTION
# =============================================================================

KEY_RISK_CANDIDATES = {"TLT/HYG", "HYG/LQD", "JNK/LQD", "SPY/TLT", "SMH/SPY"}


def find_key_risks(rows):
    candidates = [
        r for r in rows if r["ratio_name"] in KEY_RISK_CANDIDATES
    ]
    if not candidates:
        return []

    ranked = sorted(
        candidates,
        key=lambda r: (abs(r["signal_score"]), abs(r["sma_roc_pct"])),
        reverse=True,
    )
    return ranked[:2]


# =============================================================================
# MARKET PLAYBOOK
# =============================================================================


def generate_market_playbook(
    rows, regime_label, regime_score, regime_max, force_scores
):
    n = len(rows)
    if n == 0:
        return "No data available."

    normalized = regime_score / float(regime_max) if regime_max else 0.0

    confirmed_bull = []
    confirmed_bear = []
    fading = []
    diverging = []
    holding = []

    for r in rows:
        above = r["above_sma"]
        meaning, qualifier = interpret_spread(
            above,
            r["sma_slope"],
            r["rising_means"],
            r["falling_means"],
            r.get("mom_5d_pct"),
            r.get("mom_20d_pct"),
        )
        r["_meaning"] = meaning
        r["_qualifier"] = qualifier

        is_inverted = r["ratio_name"] in INVERTED

        if qualifier == "CONFIRMED" and above:
            (confirmed_bear if is_inverted else confirmed_bull).append(r)
        elif qualifier == "CONFIRMED" and not above:
            (confirmed_bull if is_inverted else confirmed_bear).append(r)
        elif qualifier == "FADING":
            fading.append(r)
        elif qualifier == "DIVERGING":
            diverging.append(r)
        else:
            holding.append(r)

    n_bull = len(confirmed_bull)
    n_bear = len(confirmed_bear)
    n_fading = len(fading)
    n_diverging = len(diverging)

    parts = []

    parts.append("OVERALL POSTURE")
    if normalized > 0.4:
        parts.append(
            "{} of {} spreads are confirmed bullish. ".format(n_bull, n)
            + "This is a strong risk-on environment. Press equity longs, favor "
            "cyclicals over defensives, underweight duration and safe havens. "
            "Dips are buyable until credit or semis break."
        )
    elif normalized > 0.1:
        parts.append(
            "{} of {} spreads are confirmed bullish, but ".format(n_bull, n)
            + "{} show weakness or are fading. ".format(n_bear + n_fading)
            + "Lean risk-on but with tighter risk management. Keep equity longs "
            "but reduce position sizes on extended names. "
            "Selective dip-buying -- do not go all-in."
        )
    elif normalized > -0.1:
        parts.append(
            "The board is split -- {} bullish, {} bearish, ".format(n_bull, n_bear)
            + "{} in transition. ".format(n_fading + n_diverging)
            + "This is a stock-picker's market, not a beta market. "
            "Reduce directional bets, favor pair trades and relative value. "
            "Raise cash to 20-30% to stay nimble."
        )
    elif normalized > -0.4:
        parts.append(
            "{} of {} spreads are confirmed bearish. ".format(n_bear, n)
            + "Lean defensive. Reduce equity beta, favor quality and low-vol. "
            "Add duration (TLT/IEF) and hedges. Only buy dips in the strongest "
            "names with confirmed support."
        )
    else:
        parts.append(
            "{} of {} spreads are confirmed bearish -- broad risk-off. ".format(n_bear, n)
            + "Capital preservation mode. Raise cash aggressively, overweight "
            "Treasuries, underweight equities across the board. Fade rallies "
            "into resistance. Do not bottom-fish until credit spreads stabilize "
            "(watch HYG/LQD, JNK/LQD)."
        )

    parts.append("")
    parts.append("MARKET LEVEL SUMMARY")

    for force_key in ["Gravity", "Electromagnetism", "Strong Force", "Weak Force"]:
        force_rows = [r for r in rows if r.get("force") == force_key]
        if not force_rows:
            continue

        force_label = FORCE_LABELS[force_key]
        score_info = force_scores.get(force_key, None)

        if score_info:
            norm = score_info["normalized"]
        else:
            norm = sum(r["signal_score"] for r in force_rows) / (
                2.0 * len(force_rows)
            )

        if norm > 0.3:
            verdict = "BULLISH"
        elif norm > 0.05:
            verdict = "LEANING BULLISH"
        elif norm > -0.05:
            verdict = "MIXED"
        elif norm > -0.3:
            verdict = "LEANING BEARISH"
        else:
            verdict = "BEARISH"

        names = ", ".join(r["ratio_name"] for r in force_rows)
        parts.append(
            " {}: {} (norm {:+.2f}) -- {}".format(force_label, verdict, norm, names)
        )

    if confirmed_bull:
        parts.append("")
        parts.append("WHAT'S WORKING")
        parts.append("{} spreads with confirmed tailwinds:".format(n_bull))
        for r in confirmed_bull:
            pb = r["rising_playbook"] if r["above_sma"] else r["falling_playbook"]
            force_tag = FORCE_LABELS.get(r.get("force", ""), "")
            parts.append(
                " {} ({}) [{}]: {}".format(r["ratio_name"], r["category"], force_tag, pb)
            )

    if confirmed_bear:
        parts.append("")
        parts.append("WHAT TO AVOID")
        parts.append("{} spreads with confirmed headwinds:".format(n_bear))
        for r in confirmed_bear:
            pb = r["rising_playbook"] if r["above_sma"] else r["falling_playbook"]
            force_tag = FORCE_LABELS.get(r.get("force", ""), "")
            parts.append(
                " {} ({}) [{}]: {}".format(r["ratio_name"], r["category"], force_tag, pb)
            )

    if fading or diverging:
        parts.append("")
        parts.append("WATCH LIST (signals in transition)")

        if fading:
            names = [r["ratio_name"] for r in fading]
            parts.append(
                " FADING ({}): {} -- ".format(len(fading), ", ".join(names))
                + "still above SMA but momentum rolling over. These tailwinds are dying. "
                "Take profits in themes supported by these spreads. If they cross below "
                "SMA, the bullish case evaporates."
            )

        if diverging:
            names = [r["ratio_name"] for r in diverging]
            parts.append(
                " DIVERGING ({}): {} -- ".format(len(diverging), ", ".join(names))
                + "still below SMA but momentum recovering. These could be early "
                "reversal signals. Watch for SMA crossover to confirm. If they flip, "
                "consider early positioning in the bullish interpretation."
            )

    parts.append("")
    parts.append("KEY RISK")

    key_risks = find_key_risks(rows)
    if key_risks:
        for r in key_risks:
            meaning, _ = interpret_spread(
                r["above_sma"],
                r["sma_slope"],
                r["rising_means"],
                r["falling_means"],
                r.get("mom_5d_pct"),
                r.get("mom_20d_pct"),
            )
            parts.append(
                " {} is in focus. Currently: {}. ".format(r["ratio_name"], meaning)
                + "If this flips, it changes the picture fast -- "
                "{} is a high-impact spread for overall regime.".format(r["ratio_name"])
            )
    else:
        if normalized > 0.1 and confirmed_bear:
            worst = min(confirmed_bear, key=lambda x: x["signal_score"])
            parts.append(
                " {} is the strongest bearish signal in an ".format(worst["ratio_name"])
                + "otherwise risk-on tape. If more spreads follow it lower, the "
                "regime could shift quickly."
            )
        elif normalized < -0.1 and confirmed_bull:
            best = max(confirmed_bull, key=lambda x: x["signal_score"])
            parts.append(
                " {} is the strongest bullish signal in an ".format(best["ratio_name"])
                + "otherwise risk-off tape. Watch if it holds -- could be an early "
                "sign of regime shift, or a divergence trap."
            )
        else:
            parts.append(
                " Mixed regime -- the risk is getting chopped up in both directions. "
                "Stay nimble, reduce size."
            )

    return "\n".join(parts)


# =============================================================================
# SOCIAL SUMMARY
# =============================================================================


def generate_social_summary(regime_label, regime_score, regime_max, rows):
    normalized = regime_score / float(regime_max) if regime_max else 0.0
    pct = int(normalized * 100)
    sign = "+" if pct >= 0 else ""
    date_str = datetime.date.today().strftime("%m/%d")

    lines = []
    lines.append("Spread Monitor {}: {} ({}{}%)".format(date_str, regime_label, sign, pct))
    lines.append("")

    sorted_rows = sorted(
        rows, key=lambda x: x["signal_score"], reverse=True
    )

    for r in sorted_rows:
        above = r["above_sma"]
        meaning, qualifier = interpret_spread(
            above,
            r["sma_slope"],
            r["rising_means"],
            r["falling_means"],
            r.get("mom_5d_pct"),
            r.get("mom_20d_pct"),
        )
        arrow = "^" if above else "v"
        score = r["signal_score"]
        lines.append(
            "{} {} [{:+d}] {}: {}".format(arrow, r["ratio_name"], score, qualifier, meaning)
        )

    return "\n".join(lines)


# =============================================================================
# SPARKLINE SVG
# =============================================================================


def sparkline_svg(data, width=80, height=24, color="#4CAF50"):
    if not data or len(data) < 2:
        return ""
    mn, mx = min(data), max(data)
    rng = mx - mn if mx != mn else 1.0

    points = []
    for i, v in enumerate(data):
        x = (i / float(len(data) - 1)) * width
        y = height - ((v - mn) / float(rng)) * height
        points.append("{:.1f},{:.1f}".format(x, y))
    polyline = " ".join(points)

    svg = (
        '<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
        'xmlns="http://www.w3.org/2000/svg">'.format(w=width, h=height)
        + '<polyline fill="none" stroke="{c}" stroke-width="1.5" points="{pts}" />'
        .format(c=color, pts=polyline)
        + "</svg>"
    )
    return svg


# =============================================================================
# CSV OUTPUT  (local archive - unchanged from v2.4.2)
# =============================================================================


def write_csv(rows, filepath):
    out_rows = []

    for r in rows:
        meaning, qualifier = interpret_spread(
            r["above_sma"],
            r["sma_slope"],
            r["rising_means"],
            r["falling_means"],
            r.get("mom_5d_pct"),
            r.get("mom_20d_pct"),
        )
        playbook = r["rising_playbook"] if r["above_sma"] else r["falling_playbook"]

        out_rows.append(
            {
                "market_level": FORCE_LABELS.get(r.get("force", ""), ""),
                "category": r["category"],
                "ratio_name": r["ratio_name"],
                "numerator": r["numerator"],
                "denominator": r["denominator"],
                "rising_means": r["rising_means"],
                "falling_means": r["falling_means"],
                "latest_ratio": r["latest_ratio"],
                "sma_50": r["sma_50"],
                "above_sma": r["above_sma"],
                "sma_slope": r["sma_slope"],
                "sma_roc_pct": r["sma_roc_pct"],
                "signal_score": r["signal_score"],
                "dist_from_sma_pct": r["dist_from_sma_pct"],
                "mom_5d_pct": r.get("mom_5d_pct"),
                "mom_20d_pct": r.get("mom_20d_pct"),
                "trend_qualifier": qualifier,
                "current_trend_meaning": meaning,
                "playbook": playbook,
            }
        )

    df = pd.DataFrame(out_rows)
    df.to_csv(filepath, index=False, encoding="utf-8")
    return df


# =============================================================================
# HTML BODY BUILDER  (replaces old write_html - produces body string for writer)
# =============================================================================


def build_body_html(
    rows, regime_label, regime_score, regime_max,
    social_summary, market_playbook, writer
):
    """Build the full body HTML string for DashboardWriter.write()."""

    date_str = datetime.date.today().strftime("%Y-%m-%d")

    regime_colors = {
        "RISK-ON":               "#16a34a",
        "LEANING RISK-ON":       "#22c55e",
        "MIXED / TRANSITIONING": "#d97706",
        "LEANING RISK-OFF":      "#dc2626",
        "RISK-OFF":              "#991b1b",
        "NO DATA":               "#9ca3af",
    }
    regime_color = regime_colors.get(regime_label, "#9ca3af")

    norm = regime_score / float(regime_max) if regime_max else 0.0
    sorted_rows = sorted(rows, key=lambda x: x["signal_score"], reverse=True)

    # Count for stat bar
    bullish  = sum(1 for r in rows if r["signal_score"] > 0)
    bearish  = sum(1 for r in rows if r["signal_score"] < 0)

    # Compute force scores for cards
    force_scores = compute_force_scores(rows)

    parts = []

    # --- Stat bar ---
    parts.append(writer.stat_bar([
        ("Generated",      date_str,        "neutral"),
        ("Spreads",        str(len(rows)),   "neutral"),
        ("Bullish",        str(bullish),     "pos"),
        ("Bearish",        str(bearish),     "neg"),
        ("Regime Score",   "{:+d}".format(regime_score), "pos" if norm > 0 else "neg"),
    ]))

    # --- Page header ---
    parts.append(writer.build_header(
        "SMA Period: {} &nbsp;|&nbsp; Slope Lookback: {}d".format(
            CONFIG["sma_period"], CONFIG["slope_lookback"]
        )
    ))

    # --- Regime banner ---
    score_html = (
        "Score: {} / {} &nbsp;&bull;&nbsp; Normalized: {:+.2f}<br>"
        "+1.0 = full risk-on &nbsp;&bull;&nbsp; -1.0 = full risk-off"
    ).format(regime_score, regime_max, norm)
    parts.append(writer.regime_banner(regime_label, score_html, color=regime_color))

    # --- Force cards ---
    force_card_cfg = [
        ("Gravity",        "I: Rates / Duration",        "#8b5cf6", "border-top-color:#8b5cf6"),
        ("Electromagnetism","II: Earnings / Fundamentals","#f59e0b", "border-top-color:#f59e0b"),
        ("Strong Force",   "III: Liquidity / Structure", "#0ea5e9", "border-top-color:#0ea5e9"),
        ("Weak Force",     "IV: Sentiment / Positioning","#f87171", "border-top-color:#f87171"),
    ]

    verdicts = {
        True:  {0.3: "Bullish", 0.05: "Leaning Bullish", -0.05: "Mixed", -0.3: "Leaning Bearish"},
        False: "Bearish",
    }

    def force_verdict(norm_val):
        if norm_val > 0.3:   return "Bullish"
        if norm_val > 0.05:  return "Leaning Bullish"
        if norm_val > -0.05: return "Mixed"
        if norm_val > -0.3:  return "Leaning Bearish"
        return "Bearish"

    cards_html = []
    for force_key, label, color, border_style in force_card_cfg:
        fs = force_scores.get(force_key, {"normalized": 0.0})
        fn = fs["normalized"]
        css_class = "pos" if fn > 0.05 else ("neg" if fn < -0.05 else "warn")
        cards_html.append(
            '<div class="card" style="{border}">'
            '<div class="label">{label}</div>'
            '<div class="value {css}">{val:+.2f}</div>'
            '<div class="sub">{verdict}</div>'
            '</div>'.format(
                border=border_style,
                label=label,
                css=css_class,
                val=fn,
                verdict=force_verdict(fn),
            )
        )
    parts.append('<div class="cards">{}</div>'.format("".join(cards_html)))

    # --- Spread table ---
    force_dot_css = {
        "Gravity":        "force-gravity",
        "Electromagnetism": "force-em",
        "Strong Force":   "force-strong",
        "Weak Force":     "force-weak",
    }

    table_rows = []
    for r in sorted_rows:
        above = r["above_sma"]
        slope = r["sma_slope"]
        score = r["signal_score"]

        current_meaning, qualifier = interpret_spread(
            above, slope, r["rising_means"], r["falling_means"],
            r.get("mom_5d_pct"), r.get("mom_20d_pct")
        )

        badge_map = {
            2: "badge-2", 1: "badge-1", -1: "badge-n1", -2: "badge-n2"
        }
        q_map = {
            "CONFIRMED":    "q-confirmed",
            "FADING":       "q-fading",
            "DIVERGING":    "q-diverging",
            "HOLDING":      "q-holding",
            "HOLDING BELOW":"q-holding",
        }

        trend_pill = (
            '<span class="trend-above">ABOVE</span>' if above
            else '<span class="trend-below">BELOW</span>'
        )

        slope_class = "pos" if slope == "RISING" else ("neg" if slope == "FALLING" else "warn")

        mom5  = r.get("mom_5d_pct")
        mom20 = r.get("mom_20d_pct")
        mom5_str  = "{:+.1f}%".format(mom5)  if mom5  is not None else "--"
        mom20_str = "{:+.1f}%".format(mom20) if mom20 is not None else "--"
        mom5_class  = "pos" if (mom5  or 0) > 0 else "neg"
        mom20_class = "pos" if (mom20 or 0) > 0 else "neg"

        dist_class = "pos" if r["dist_from_sma_pct"] > 0 else "neg"
        playbook = r["rising_playbook"] if above else r["falling_playbook"]
        pb_color = "#4f46e5" if above else "#dc2626"

        force     = r.get("force", "")
        dot_css   = force_dot_css.get(force, "")
        force_short = FORCE_LABELS.get(force, force)

        spark_color = "#16a34a" if above else "#dc2626"
        spark = sparkline_svg(r.get("sparkline", []), color=spark_color)

        force_sort_val = FORCE_SORT_ORDER.get(force, 0)

        table_rows.append(
            "<tr>"
            '<td data-sort="{fs}"><span class="force-dot {dc}"></span>'
            '<span style="font-size:0.88em;color:#555;">{fl}</span></td>'
            '<td data-sort="{cat}" style="color:#555;">{cat}</td>'
            '<td data-sort="{rn}"><span class="ticker">{rn}</span><br>{spark}</td>'
            '<td data-sort="{ts}">{trend}</td>'
            '<td data-sort="{dist}" class="{distc} num">{dist:+.2f}%</td>'
            '<td data-sort="{sl}" class="{sc}">{sl} <span class="num" style="font-size:0.85em;color:#aaa;">({roc:+.3f}%)</span></td>'
            '<td data-sort="{m5raw}" class="{m5c} num">{m5s}</td>'
            '<td data-sort="{m20raw}" class="{m20c} num">{m20s}</td>'
            '<td data-sort="{score}"><span class="badge {bc}">{score:+d}</span></td>'
            '<td data-sort="{q}">'
            '<span class="qualifier {qc}">{q}</span><br>'
            '<span style="font-size:0.88em;color:#555;">{cm}</span>'
            "</td>"
            '<td data-sort="{pb}" style="font-size:0.88em;color:{pbc};max-width:240px;">{pb}</td>'
            "</tr>".format(
                fs=force_sort_val, dc=dot_css, fl=force_short,
                cat=r["category"],
                rn=r["ratio_name"], spark=spark,
                ts=1 if above else 0, trend=trend_pill,
                dist=r["dist_from_sma_pct"], distc=dist_class,
                sc=slope_class, sl=slope, roc=r["sma_roc_pct"],
                m5raw=mom5 if mom5 is not None else -999,
                m5c=mom5_class, m5s=mom5_str,
                m20raw=mom20 if mom20 is not None else -999,
                m20c=mom20_class, m20s=mom20_str,
                bc=badge_map.get(score, "badge-n2"), score=score,
                qc=q_map.get(qualifier, "q-holding"), q=qualifier,
                cm=current_meaning,
                pbc=pb_color, pb=playbook,
            )
        )

    table_html = (
        '<table class="sortable-table">'
        "<thead><tr>"
        '<th title="Force category: Rates, Earnings, Liquidity, or Sentiment">Market Level</th>'
        '<th title="Specific spread classification within the force level">Category</th>'
        '<th title="Ratio name (e.g. HYG/LQD, XLY/XLP)">Spread</th>'
        '<th title="Current trend direction and qualifier">Trend</th>'
        '<th title="Distance from mean as percentage (z-score proxy)">Dist %</th>'
        '<th title="10-day regression slope of the spread">Slope</th>'
        '<th title="5-day momentum (rate of change)">5d Mom</th>'
        '<th title="20-day momentum (rate of change)">20d Mom</th>'
        '<th title="Composite signal score, positive=bullish, negative=bearish">Score &#9660;</th>'
        '<th title="Interpretation of current spread regime">Status / Meaning</th>'
        '<th title="Suggested positioning based on spread signal">Playbook</th>'
        "</tr></thead>"
        "<tbody>" + "\n".join(table_rows) + "</tbody>"
        "</table>"
    )
    parts.append(writer.section("All Spreads", table_html, hint="Click any column to sort"))

    # --- Playbook ---
    pb_lines = []
    for line in market_playbook.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line in (
            "OVERALL POSTURE", "WHAT'S WORKING", "WHAT TO AVOID",
            "WATCH LIST (signals in transition)", "KEY RISK", "MARKET LEVEL SUMMARY"
        ):
            pb_lines.append('<div class="pb-header">{}</div>'.format(line))
        elif line.startswith(" "):
            pb_lines.append('<div class="pb-item">{}</div>'.format(line.strip()))
        else:
            pb_lines.append('<div class="pb-body">{}</div>'.format(line))

    parts.append(writer.section("General Market Playbook", "\n".join(pb_lines)))

    # --- Social / copy box ---
    social_html = (
        '<div style="background:#f8f9fb;border:1px solid #e2e4e8;border-radius:6px;'
        'padding:16px 20px;font-family:\'IBM Plex Mono\',monospace;font-size:0.88em;'
        'white-space:pre-line;cursor:pointer;position:relative;line-height:1.7;color:#333;" '
        'onclick="navigator.clipboard.writeText(this.innerText.replace(\'Click to copy\',\'\').trim())'
        '.then(function(){{this.style.borderColor=\'#16a34a\';}}.bind(this))" '
        'title="Click to copy">'
        '<span style="position:absolute;top:8px;right:12px;font-size:0.85em;color:#aaa;">'
        'Click to copy</span>'
        '{}</div>'.format(social_summary)
    )
    parts.append(writer.section("Text Summary", social_html))

    parts.append(writer.footer())

    return "\n".join(parts)


# =============================================================================
# EXTRA CSS (spread-monitor specific additions to shared theme)
# =============================================================================

EXTRA_CSS = """
.force-dot {
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    margin-right: 6px;
    vertical-align: middle;
    flex-shrink: 0;
}
.force-gravity { background: #8b5cf6; }
.force-em      { background: #f59e0b; }
.force-strong  { background: #0ea5e9; }
.force-weak    { background: #f87171; }

/* ── Mobile responsive ── */
@media (max-width: 768px) {
    .force-dot { width: 8px; height: 8px; margin-right: 4px; }
}
"""

# =============================================================================
# SORTABLE TABLE JS (unchanged from v2.4.2)
# =============================================================================

SORT_JS = """
(function() {
  // Works on any table with class 'sortable-table'.
  // Reads data-sort attribute from each <td> for comparison.
  // Numeric if parseFloat succeeds and is not NaN; string otherwise.
  document.querySelectorAll('table.sortable-table').forEach(function(table) {
    var headers = Array.from(table.querySelectorAll('thead th'));
    var currentSort = {col: -1, asc: true};
    headers.forEach(function(th, colIdx) {
      th.style.cursor = 'pointer';
      th.title = 'Click to sort';
      th.addEventListener('click', function() {
        var asc = (currentSort.col === colIdx) ? !currentSort.asc : true;
        currentSort = {col: colIdx, asc: asc};
        headers.forEach(function(h) { h.classList.remove('sorted-asc', 'sorted-desc'); });
        th.classList.add(asc ? 'sorted-asc' : 'sorted-desc');
        var tbody = table.querySelector('tbody');
        var rows = Array.from(tbody.querySelectorAll('tr'));
        rows.sort(function(a, b) {
          var aVal = (a.children[colIdx] && a.children[colIdx].getAttribute('data-sort')) || '';
          var bVal = (b.children[colIdx] && b.children[colIdx].getAttribute('data-sort')) || '';
          var aNum = parseFloat(aVal);
          var bNum = parseFloat(bVal);
          var cmp = (!isNaN(aNum) && !isNaN(bNum))
            ? aNum - bNum
            : aVal.localeCompare(bVal);
          return asc ? cmp : -cmp;
        });
        rows.forEach(function(row) { tbody.appendChild(row); });
      });
    });
  });
})();
"""


# =============================================================================
# MAIN
# =============================================================================


def main():
    print("=" * 60)
    print("INTERMARKET SPREAD MONITOR v2.5.1")
    print("=" * 60)

    tickers_needed = set()
    for s in SPREADS:
        tickers_needed.add(s["num"])
        tickers_needed.add(s["den"])

    print("Loading {} tickers from {}/...".format(len(tickers_needed), CONFIG["cache_dir"]))
    price_data = load_price_cache(tickers_needed)

    loaded = set(price_data.keys())
    missing = tickers_needed - loaded

    if missing:
        print("WARNING: Missing tickers: {}".format(sorted(missing)))

    print("Loaded: {} / {}".format(len(loaded), len(tickers_needed)))

    rows = []
    skipped = []

    for s in SPREADS:
        name = s["name"]
        num = s["num"]
        den = s["den"]

        if num not in price_data or den not in price_data:
            skipped.append(name)
            continue

        metrics = compute_spread_metrics(
            price_data[num],
            price_data[den],
            CONFIG["sma_period"],
            CONFIG["slope_lookback"],
            CONFIG["slope_flat_threshold"],
        )

        if metrics is None:
            skipped.append(name)
            continue

        row = {
            "ratio_name": name,
            "numerator": num,
            "denominator": den,
            "category": s["category"],
            "force": s["force"],
            "rising_means": s["rising_means"],
            "falling_means": s["falling_means"],
            "rising_playbook": s["rising_playbook"],
            "falling_playbook": s["falling_playbook"],
        }
        row.update(metrics)
        rows.append(row)

    if skipped:
        print("Skipped (missing or insufficient data): {}".format(skipped))

    print("Computed: {} / {} spreads".format(len(rows), len(SPREADS)))

    if not rows:
        print("ERROR: No spreads computed. Check price_cache directory.")
        return

    regime_label, regime_score, regime_max = compute_regime(rows)
    force_scores = compute_force_scores(rows)

    social_summary = generate_social_summary(
        regime_label, regime_score, regime_max, rows
    )
    market_playbook = generate_market_playbook(
        rows, regime_label, regime_score, regime_max, force_scores
    )

    print()
    print("=" * 60)
    print("REGIME: {} ({} / {})".format(regime_label, regime_score, regime_max))
    print("=" * 60)
    print(market_playbook)
    print()

    # --- Write CSV (local archive) ---
    date_compact = datetime.date.today().strftime("%Y%m%d")
    os.makedirs(CONFIG["csv_output_dir"], exist_ok=True)
    csv_dated = os.path.join(
        CONFIG["csv_output_dir"], "spread_monitor_{}.csv".format(date_compact)
    )
    csv_latest = os.path.join(CONFIG["csv_output_dir"], "spread_monitor_latest.csv")
    write_csv(rows, csv_dated)
    write_csv(rows, csv_latest)
    print("CSV: {}".format(csv_dated))

    # --- Write static HTML dashboard ---
    writer = DashboardWriter("spread-monitor", "Intermarket Spread Monitor")
    body = build_body_html(
        rows, regime_label, regime_score, regime_max,
        social_summary, market_playbook, writer
    )
    writer.write(body, extra_css=EXTRA_CSS, extra_js=SORT_JS)

    print()
    print("Done.")


if __name__ == "__main__":
    main()
