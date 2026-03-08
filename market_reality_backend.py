# ============================================================================
# market_reality_backend.py - v1.1
# Last updated: 2026-03-08
# ============================================================================
# v1.1: Context filtering (market-as-person, not people-as-emotional),
#       date column, more feeds, raised max_per_feed, synonyms,
#       actual measured values in table
# v1.0: Initial build - RSS scan, anthro detection, quant reality from cache
# ============================================================================
"""
Market Reality Check Backend
============================
Scans financial commentary RSS feeds for anthropomorphic language,
cross-references with quantitative market data from price_cache,
and generates a static dashboard via DashboardWriter.

Based on Morris et al. (2007) research: agent metaphors cause investors
to expect trend continuance. This dashboard translates emotional language
into what the data actually shows.

Run:  python market_reality_backend.py
"""

import os
import sys
import json
import pickle
import re
import time
from pathlib import Path
from datetime import datetime, timedelta
from collections import Counter

import numpy as np
import pandas as pd

try:
    import feedparser
except ImportError:
    print("[ERROR] feedparser not installed. Run: pip install feedparser")
    sys.exit(1)

from dashboard_writer import DashboardWriter

# ============================================================================
# PATH SETUP
# ============================================================================

_SCRIPT_DIR = Path(__file__).resolve().parent          # market-dashboards/
_DATA_DIR   = _SCRIPT_DIR / '..' / 'perplexity-user-data'
CACHE_DIR   = _DATA_DIR / 'price_cache'

# ============================================================================
# ANTHROPOMORPHISM LEXICON
# ============================================================================
# Maps emotional/anthropomorphic phrases -> what they claim -> what to measure

ANTHRO_LEXICON = {
    # FEAR / NERVOUSNESS
    "skittish":     {"category": "fear",        "implies": "irrational nervousness",   "measure": ["VIX", "put/call", "SKEW"]},
    "nervous":      {"category": "fear",        "implies": "anxiety without cause",     "measure": ["VIX", "put/call"]},
    "jittery":      {"category": "fear",        "implies": "fragile confidence",        "measure": ["VIX", "put/call", "SKEW"]},
    "fearful":      {"category": "fear",        "implies": "emotional selling",         "measure": ["VIX", "correlation"]},
    "scared":       {"category": "fear",        "implies": "irrational avoidance",      "measure": ["VIX", "put/call"]},
    "worried":      {"category": "fear",        "implies": "anxiety",                   "measure": ["VIX", "SKEW"]},
    "spooked":      {"category": "fear",        "implies": "startled reaction",         "measure": ["VIX", "put/call"]},
    "anxious":      {"category": "fear",        "implies": "generalized worry",         "measure": ["VIX", "SKEW"]},
    "rattled":      {"category": "fear",        "implies": "shaken confidence",         "measure": ["VIX", "breadth"]},
    "on edge":      {"category": "fear",        "implies": "ready to bolt",             "measure": ["VIX", "put/call"]},
    "uneasy":       {"category": "fear",        "implies": "discomfort, doubt",         "measure": ["VIX", "SKEW"]},
    "trembling":    {"category": "fear",        "implies": "shaking with fear",         "measure": ["VIX", "put/call"]},
    "twitchy":      {"category": "fear",        "implies": "jumpy, reactive",           "measure": ["VIX", "put/call"]},
    "apprehensive": {"category": "fear",        "implies": "dread of what's next",      "measure": ["VIX", "SKEW"]},
    "wary":         {"category": "fear",        "implies": "guarded, watchful",         "measure": ["VIX", "put/call"]},

    # PANIC
    "panicking":    {"category": "panic",       "implies": "irrational liquidation",    "measure": ["VIX term structure", "correlation", "breadth"]},
    "panic":        {"category": "panic",       "implies": "mass irrational fear",      "measure": ["VIX term structure", "correlation"]},
    "capitulation": {"category": "panic",       "implies": "giving up / surrender",     "measure": ["VIX term structure", "breadth", "volume"]},
    "capitulating": {"category": "panic",       "implies": "surrendering positions",    "measure": ["VIX term structure", "breadth", "volume"]},
    "meltdown":     {"category": "panic",       "implies": "system collapse",           "measure": ["VIX term structure", "correlation"]},
    "bloodbath":    {"category": "panic",       "implies": "violent destruction",        "measure": ["VIX", "breadth"]},
    "carnage":      {"category": "panic",       "implies": "mass destruction",          "measure": ["VIX", "breadth", "correlation"]},
    "freefall":     {"category": "panic",       "implies": "uncontrolled descent",      "measure": ["VIX term structure", "breadth"]},
    "free fall":    {"category": "panic",       "implies": "uncontrolled descent",      "measure": ["VIX term structure", "breadth"]},
    "cratering":    {"category": "panic",       "implies": "collapsing violently",      "measure": ["VIX term structure", "breadth"]},
    "imploding":    {"category": "panic",       "implies": "collapsing inward",         "measure": ["VIX term structure", "correlation"]},
    "tanking":      {"category": "panic",       "implies": "rapid uncontrolled fall",   "measure": ["VIX", "breadth"]},
    "plunging":     {"category": "panic",       "implies": "diving headfirst",          "measure": ["VIX", "breadth"]},
    "crashing":     {"category": "panic",       "implies": "violent impact",            "measure": ["VIX term structure", "correlation"]},

    # EXHAUSTION
    "tired":        {"category": "exhaustion",  "implies": "fatigue, needs rest",       "measure": ["ADX", "OBV", "breadth"]},
    "exhausted":    {"category": "exhaustion",  "implies": "energy depleted",           "measure": ["ADX", "RSI", "volume"]},
    "fatigued":     {"category": "exhaustion",  "implies": "weakening momentum",        "measure": ["ADX", "OBV"]},
    "running out of steam": {"category": "exhaustion", "implies": "decelerating",       "measure": ["ADX", "RSI"]},
    "losing steam": {"category": "exhaustion",  "implies": "momentum fading",           "measure": ["ADX", "RSI", "OBV"]},
    "out of gas":   {"category": "exhaustion",  "implies": "no fuel left",              "measure": ["ADX", "RSI", "OBV"]},
    "winded":       {"category": "exhaustion",  "implies": "needs to catch breath",     "measure": ["ADX", "RSI"]},
    "spent":        {"category": "exhaustion",  "implies": "used up all energy",        "measure": ["ADX", "volume"]},
    "running on fumes": {"category": "exhaustion", "implies": "nearly depleted",        "measure": ["ADX", "RSI", "OBV"]},
    "petering out": {"category": "exhaustion",  "implies": "gradually dying",           "measure": ["ADX", "RSI"]},

    # CONSOLIDATION
    "digesting":    {"category": "consolidation", "implies": "thoughtful processing",   "measure": ["ATR", "Bollinger width"]},
    "absorbing":    {"category": "consolidation", "implies": "taking in information",   "measure": ["ATR", "Bollinger width"]},
    "mulling":      {"category": "consolidation", "implies": "deliberating",            "measure": ["ATR", "volume"]},
    "processing":   {"category": "consolidation", "implies": "cognitive work",          "measure": ["ATR", "Bollinger width"]},
    "pausing":      {"category": "consolidation", "implies": "temporary halt",          "measure": ["ATR", "volume"]},
    "catching its breath": {"category": "consolidation", "implies": "resting before next move", "measure": ["ATR", "volume"]},
    "taking a breather": {"category": "consolidation", "implies": "brief rest",         "measure": ["ATR", "volume"]},
    "treading water": {"category": "consolidation", "implies": "staying afloat, no progress", "measure": ["ATR", "Bollinger width"]},

    # CONFUSION
    "confused":     {"category": "confusion",   "implies": "market doesn't know",       "measure": ["dispersion", "correlation"]},
    "uncertain":    {"category": "confusion",   "implies": "lack of direction",         "measure": ["VIX", "dispersion"]},
    "mixed signals": {"category": "confusion",  "implies": "contradictory data",        "measure": ["dispersion", "breadth"]},
    "indecisive":   {"category": "confusion",   "implies": "can't make up its mind",    "measure": ["ATR", "dispersion"]},
    "directionless": {"category": "confusion",  "implies": "no trend",                  "measure": ["ADX", "dispersion"]},
    "lost":         {"category": "confusion",   "implies": "doesn't know where to go",  "measure": ["ADX", "dispersion"]},
    "torn":         {"category": "confusion",   "implies": "pulled both ways",          "measure": ["dispersion", "correlation"]},
    "searching for direction": {"category": "confusion", "implies": "aimless movement", "measure": ["ADX", "dispersion"]},
    "at a crossroads": {"category": "confusion", "implies": "decision point",           "measure": ["ATR", "VIX"]},

    # EUPHORIA
    "euphoric":     {"category": "euphoria",    "implies": "irrational exuberance",     "measure": ["VIX (low)", "bullish %"]},
    "exuberant":    {"category": "euphoria",    "implies": "excessive optimism",        "measure": ["VIX (low)", "bullish %"]},
    "giddy":        {"category": "euphoria",    "implies": "childlike excitement",      "measure": ["VIX (low)", "put/call (low)"]},
    "ecstatic":     {"category": "euphoria",    "implies": "extreme joy",               "measure": ["VIX (low)", "bullish %"]},
    "frothy":       {"category": "euphoria",    "implies": "bubbly, overvalued",        "measure": ["VIX (low)", "breadth (high)"]},
    "bubbly":       {"category": "euphoria",    "implies": "unsustainable optimism",    "measure": ["VIX (low)", "bullish %"]},
    "ebullient":    {"category": "euphoria",    "implies": "enthusiastically optimistic", "measure": ["VIX (low)", "bullish %"]},
    "intoxicated":  {"category": "euphoria",    "implies": "drunk on gains",            "measure": ["VIX (low)", "bullish %"]},
    "soaring":      {"category": "euphoria",    "implies": "flying high, unstoppable",  "measure": ["VIX (low)", "RSI"]},

    # AGGRESSION
    "fighting":     {"category": "aggression",  "implies": "willful struggle",          "measure": ["volume at price", "order flow"]},
    "battling":     {"category": "aggression",  "implies": "combat at levels",          "measure": ["volume at price", "order flow"]},
    "struggling":   {"category": "aggression",  "implies": "difficulty advancing",      "measure": ["volume at price", "breadth"]},
    "wrestling":    {"category": "aggression",  "implies": "contested ground",          "measure": ["volume at price", "order flow"]},
    "clawing back": {"category": "aggression",  "implies": "desperate recovery",        "measure": ["breadth", "volume"]},
    "defending":    {"category": "aggression",  "implies": "holding a line",            "measure": ["volume at price", "breadth"]},
    "under attack": {"category": "aggression",  "implies": "being assaulted",           "measure": ["VIX", "breadth"]},
    "under siege":  {"category": "aggression",  "implies": "sustained assault",         "measure": ["VIX", "breadth"]},
    "punished":     {"category": "aggression",  "implies": "suffering consequences",    "measure": ["VIX", "breadth"]},

    # HEALTH / SICKNESS
    "healthy":      {"category": "health",      "implies": "good condition",            "measure": ["breadth", "ADX", "volume"]},
    "sick":         {"category": "health",      "implies": "diseased, broken",          "measure": ["breadth", "correlation"]},
    "recovering":   {"category": "health",      "implies": "healing from illness",      "measure": ["breadth", "RSI"]},
    "ailing":       {"category": "health",      "implies": "persistently ill",          "measure": ["breadth", "ADX"]},
    "wounded":      {"category": "health",      "implies": "injured but alive",         "measure": ["breadth", "volume"]},
    "limping":      {"category": "health",      "implies": "moving but impaired",       "measure": ["breadth", "ADX"]},
    "bruised":      {"category": "health",      "implies": "hurt but not broken",       "measure": ["breadth", "RSI"]},
    "bleeding":     {"category": "health",      "implies": "losing life force",         "measure": ["breadth", "volume"]},
    "contagion":    {"category": "health",      "implies": "spreading sickness",        "measure": ["correlation", "breadth"]},
    "infected":     {"category": "health",      "implies": "disease spreading",         "measure": ["correlation", "breadth"]},

    # FILLER (common but unfalsifiable)
    "profit-taking": {"category": "filler",     "implies": "intentional selling",       "measure": ["UNFALSIFIABLE"]},
    "profit taking": {"category": "filler",     "implies": "intentional selling",       "measure": ["UNFALSIFIABLE"]},
    "bargain hunting": {"category": "filler",   "implies": "intentional buying",        "measure": ["UNFALSIFIABLE"]},
    "sidelined":    {"category": "filler",      "implies": "waiting to act",            "measure": ["volume", "money flow"]},
    "cautious":     {"category": "filler",      "implies": "careful behavior",          "measure": ["VIX", "put/call"]},
    "resilient":    {"category": "filler",      "implies": "strong despite adversity",  "measure": ["breadth", "RSI"]},
    "shrugging off": {"category": "filler",     "implies": "ignoring bad news",         "measure": ["UNFALSIFIABLE"]},
    "betting on":   {"category": "filler",      "implies": "wagering / gambling",       "measure": ["UNFALSIFIABLE"]},
    "bracing for":  {"category": "filler",      "implies": "preparing for impact",      "measure": ["VIX", "put/call"]},
}

# ============================================================================
# MARKET SUBJECT CONTEXT FILTER
# ============================================================================
# The anthro word must be near a market subject, not describing people/consumers.
# This prevents false positives like "middle class is spending in a nervous way"

MARKET_SUBJECTS = [
    "market", "markets", "wall street", "stocks", "stock market",
    "s&p", "s&p 500", "dow", "nasdaq", "russell",
    "equities", "indices", "index", "rally", "selloff", "sell-off",
    "trading", "session", "futures",
    "bull", "bear", "correction", "rebound", "recovery",
    "sector", "sectors", "tech", "financials",
    "treasury", "treasuries", "bond", "bonds",
    "crude", "oil", "gold", "commodities",
    "bitcoin", "crypto",
    "investors", "traders", "fund", "hedge fund",
    "shares", "etf", "portfolio",
    "federal reserve", "fed ", "central bank",
    "earnings", "revenue", "gdp",
]

# People/non-market subjects -- only reject when these appear WITHOUT any market subject
NON_MARKET_SUBJECTS = [
    "middle class", "households", "families", "voters",
    "police", "capitol", "congress", "election",
    "patients", "students", "children",
    "he said", "she said",
]


def is_market_anthropomorphism(text, phrase):
    """Check if the anthro phrase is describing the market/financial context, not unrelated topics."""
    text_lower = text.lower()
    pos = text_lower.find(phrase.lower())
    if pos == -1:
        return False

    # Some phrases are almost always about markets in financial news context
    ALWAYS_MARKET = {
        "bloodbath", "carnage", "freefall", "free fall", "capitulation",
        "capitulating", "meltdown", "frothy", "bubbly", "profit-taking",
        "profit taking", "bargain hunting", "sidelined", "cratering",
        "imploding", "tanking", "contagion", "soaring", "plunging",
        "crashing", "clawing back",
    }
    if phrase.lower() in ALWAYS_MARKET:
        return True

    # Look at a window around the phrase (100 chars each side)
    window_start = max(0, pos - 100)
    window_end = min(len(text_lower), pos + len(phrase) + 100)
    window = text_lower[window_start:window_end]

    # Check for market subjects nearby
    has_market = any(subj in window for subj in MARKET_SUBJECTS)
    # Check for clearly non-market context
    has_non_market = any(subj in window for subj in NON_MARKET_SUBJECTS)

    # If clearly non-market context with no market subjects -> reject
    if has_non_market and not has_market:
        return False

    # Default: require at least one market subject nearby
    return has_market

CATEGORY_COLORS = {
    "fear":           "#e74c3c",
    "panic":          "#c0392b",
    "exhaustion":     "#e67e22",
    "consolidation":  "#27ae60",
    "confusion":      "#f1c40f",
    "euphoria":       "#3498db",
    "aggression":     "#8e44ad",
    "health":         "#1abc9c",
    "filler":         "#95a5a6",
}

# ============================================================================
# RSS FEEDS
# ============================================================================

RSS_FEEDS = {
    "Yahoo Finance":    "https://finance.yahoo.com/news/rssindex",
    "MarketWatch":      "https://feeds.marketwatch.com/marketwatch/marketpulse/",
    "CNBC":             "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114",
    "CNBC Top News":    "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100727362",
    "Google News (Markets)": "https://news.google.com/rss/search?q=stock+market+today&hl=en-US&gl=US&ceid=US:en",
    "Google News (Wall St)": "https://news.google.com/rss/search?q=wall+street+stocks&hl=en-US&gl=US&ceid=US:en",
    "Seeking Alpha":    "https://seekingalpha.com/market_currents.xml",
    "Reuters Business": "https://news.google.com/rss/search?q=site:reuters.com+stock+market&hl=en-US&gl=US&ceid=US:en",
    "Bloomberg via Google": "https://news.google.com/rss/search?q=site:bloomberg.com+markets&hl=en-US&gl=US&ceid=US:en",
}


# ============================================================================
# DATA LOADING
# ============================================================================

def load_ticker(ticker):
    """Load a single ticker from price cache."""
    cache_file = CACHE_DIR / f'{ticker}.pkl'
    if not cache_file.exists():
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
        return df
    except Exception:
        return None


# ============================================================================
# NEWS INGESTION
# ============================================================================

def parse_pub_date(date_str):
    """Parse various RSS date formats into a datetime string."""
    if not date_str:
        return ""
    from email.utils import parsedate_to_datetime
    try:
        dt = parsedate_to_datetime(date_str)
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        pass
    # Try ISO format
    for fmt in ["%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S"]:
        try:
            dt = datetime.strptime(date_str[:19], fmt[:len(date_str)])
            return dt.strftime("%Y-%m-%d %H:%M")
        except Exception:
            pass
    return date_str[:16]


def fetch_headlines(max_per_feed=50):
    """Fetch headlines + summaries from RSS feeds."""
    articles = []
    seen_titles = set()
    for source, url in RSS_FEEDS.items():
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:max_per_feed]:
                title = entry.get("title", "")
                if title in seen_titles:
                    continue
                seen_titles.add(title)
                text = title + " " + entry.get("summary", "")
                pub_date = parse_pub_date(entry.get("published", ""))
                articles.append({
                    "source": source,
                    "title": title,
                    "link": entry.get("link", ""),
                    "text": text,
                    "published": pub_date,
                })
        except Exception as e:
            print(f"  [!] Error fetching {source}: {e}")
    return articles


# ============================================================================
# ANTHROPOMORPHISM DETECTION
# ============================================================================

def detect_anthropomorphisms(text):
    """Scan text for anthropomorphic market language with context filtering."""
    text_lower = text.lower()
    matches = []
    for phrase, meta in ANTHRO_LEXICON.items():
        found = False
        if " " in phrase:
            found = phrase in text_lower
        else:
            pattern = r"\b" + re.escape(phrase) + r"\b"
            found = bool(re.search(pattern, text_lower))
        if found and is_market_anthropomorphism(text, phrase):
            matches.append({"phrase": phrase, **meta})
    return matches


def analyze_articles(articles):
    """Add anthropomorphism analysis to each article."""
    for art in articles:
        matches = detect_anthropomorphisms(art["text"])
        art["anthro_count"] = len(matches)
        art["anthro_phrases"] = [m["phrase"] for m in matches]
        art["anthro_categories"] = list(set(m["category"] for m in matches))
        art["anthro_details"] = matches
    return articles


# ============================================================================
# QUANTITATIVE REALITY ENGINE
# ============================================================================

def compute_market_reality():
    """Compute quantitative metrics from price_cache to replace anthropomorphic claims."""
    reality = {}

    # --- VIX ---
    vix_df = load_ticker("^VIX")
    if vix_df is not None and len(vix_df) > 20:
        vix_close = vix_df['close']
        reality["vix"] = round(float(vix_close.iloc[-1]), 2)
        reality["vix_5d_chg"] = round(
            (float(vix_close.iloc[-1]) / float(vix_close.iloc[-5]) - 1) * 100, 1
        )
        vix_sma20 = float(vix_close.iloc[-20:].mean())
        reality["vix_vs_sma20"] = round(float(vix_close.iloc[-1]) / vix_sma20, 2)
        # Stress level based on VIX vs its own 20d average
        if reality["vix_vs_sma20"] > 1.3:
            reality["vix_stress"] = "ACUTE STRESS (VIX 30%+ above 20d avg)"
        elif reality["vix_vs_sma20"] > 1.1:
            reality["vix_stress"] = "ELEVATED (VIX 10-30% above 20d avg)"
        else:
            reality["vix_stress"] = "NORMAL (VIX near 20d avg)"

    # --- VIX3M (term structure) ---
    vix3m_df = load_ticker("^VIX3M")
    if vix3m_df is not None and vix_df is not None:
        vix3m_close = float(vix3m_df['close'].iloc[-1])
        reality["vix3m"] = round(vix3m_close, 2)
        vix_val = reality.get("vix", 0)
        if vix_val > 0 and vix3m_close > 0:
            ratio = vix_val / vix3m_close
            reality["vix_term_ratio"] = round(ratio, 3)
            if ratio > 1.0:
                reality["vix_term_structure"] = "BACKWARDATION -- acute panic (near-term fear exceeds longer-term)"
            elif ratio > 0.9:
                reality["vix_term_structure"] = "FLAT -- mild stress, no acute panic"
            else:
                reality["vix_term_structure"] = "CONTANGO -- normal/orderly (no panic despite what commentary says)"

    # --- SKEW ---
    skew_df = load_ticker("^SKEW")
    if skew_df is not None:
        reality["skew"] = round(float(skew_df['close'].iloc[-1]), 2)

    # --- SPY realized vol ---
    spy_df = load_ticker("SPY")
    if spy_df is not None and len(spy_df) > 21:
        spy_ret = spy_df['close'].pct_change().dropna()
        reality["spy_20d_vol"] = round(float(spy_ret.iloc[-20:].std() * np.sqrt(252) * 100), 1)

    # --- Sector dispersion + correlation ---
    sectors = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLY", "XLU", "XLB", "XLRE", "XLC"]
    sector_returns = {}
    sector_period_ret = {}
    for s in sectors:
        df = load_ticker(s)
        if df is not None and len(df) > 21:
            ret = df['close'].pct_change().dropna()
            sector_returns[s] = ret.iloc[-20:]
            sector_period_ret[s] = (float(df['close'].iloc[-1]) / float(df['close'].iloc[-21]) - 1) * 100

    if len(sector_returns) >= 8:
        ret_df = pd.DataFrame(sector_returns)

        # Cross-sectional dispersion
        daily_disp = ret_df.std(axis=1)
        reality["sector_dispersion"] = round(float(daily_disp.mean() * 100), 3)

        # Average pairwise correlation
        corr = ret_df.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        avg_corr = float(corr.where(mask).stack().mean())
        reality["avg_sector_corr"] = round(avg_corr, 3)

        # Rotation vs panic classification
        if reality["sector_dispersion"] > 0.6 and avg_corr < 0.5:
            reality["flow_verdict"] = "ROTATION -- capital moving between sectors, not leaving"
        elif avg_corr > 0.8:
            reality["flow_verdict"] = "SYSTEMATIC RISK-OFF -- all sectors selling together"
        elif avg_corr > 0.6:
            reality["flow_verdict"] = "MIXED -- some broad selling, some rotation"
        else:
            reality["flow_verdict"] = "DISPERSED -- low correlation, idiosyncratic moves"

        # Best/worst sector
        if sector_period_ret:
            best = max(sector_period_ret, key=sector_period_ret.get)
            worst = min(sector_period_ret, key=sector_period_ret.get)
            reality["best_sector"] = f"{best} ({sector_period_ret[best]:+.1f}%)"
            reality["worst_sector"] = f"{worst} ({sector_period_ret[worst]:+.1f}%)"

    return reality


def generate_reality_verdicts(reality):
    """Generate plain-language, emotion-free market description."""
    verdicts = []

    vix = reality.get("vix", 0)
    if vix > 35:
        verdicts.append(f"VIX at {vix}: Implied vol elevated. Options market pricing large moves. Historically contrarian bullish at these levels.")
    elif vix > 25:
        verdicts.append(f"VIX at {vix}: Hedging demand above average but not extreme. Moderate risk-pricing.")
    elif vix > 18:
        verdicts.append(f"VIX at {vix}: Normal implied vol range. Standard risk-pricing.")
    else:
        verdicts.append(f"VIX at {vix}: Low implied vol. Hedging demand minimal -- complacency zone.")

    term = reality.get("vix_term_structure", "")
    if term:
        verdicts.append(f"VIX term structure: {term}")

    flow = reality.get("flow_verdict", "")
    disp = reality.get("sector_dispersion", 0)
    corr = reality.get("avg_sector_corr", 0)
    if flow:
        verdicts.append(f"Sector dispersion {disp}%, avg correlation {corr}: {flow}")

    best = reality.get("best_sector", "N/A")
    worst = reality.get("worst_sector", "N/A")
    if best != "N/A":
        verdicts.append(f"Rotation evidence: strongest {best}, weakest {worst}")

    return verdicts


# ============================================================================
# HTML GENERATION
# ============================================================================

EXTRA_CSS = """
.anthro-tag {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.78em;
    font-weight: 600;
    color: #fff;
    margin: 1px 2px;
}
.verdict-item {
    padding: 8px 12px;
    margin: 4px 0;
    border-left: 3px solid #2563eb;
    background: #f0f7ff;
    font-size: 0.9em;
}
.article-row td { vertical-align: top; }
.article-row:hover { background: #f8f9fa !important; }
a.article-link {
    color: #2563eb;
    text-decoration: none;
    font-weight: 600;
}
a.article-link:hover { text-decoration: underline; }
.category-bar {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin: 12px 0;
}
.cat-chip {
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 0.82em;
    font-weight: 600;
    color: #fff;
}
.reality-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 12px;
    margin: 12px 0;
}
.reality-card {
    background: #f8f9fa;
    border-radius: 8px;
    padding: 12px 16px;
    border-left: 3px solid #64748b;
}
.reality-card .label {
    font-size: 0.72em;
    font-weight: 700;
    text-transform: uppercase;
    color: #64748b;
}
.reality-card .value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.3em;
    font-weight: 600;
    color: #1e293b;
}
.filler-warning {
    background: #fef3c7;
    border: 1px solid #f59e0b;
    border-radius: 6px;
    padding: 8px 12px;
    font-size: 0.82em;
    color: #92400e;
    margin-top: 12px;
}
"""


def build_stat_bar(writer, reality, articles, flagged_count):
    """Build the top stat bar."""
    vix = reality.get("vix", "N/A")
    term_ratio = reality.get("vix_term_ratio", "N/A")
    disp = reality.get("sector_dispersion", "N/A")
    corr = reality.get("avg_sector_corr", "N/A")
    total = len(articles)
    pct = f"{flagged_count}/{total}"

    # Color VIX
    vix_class = "neg" if isinstance(vix, (int, float)) and vix > 25 else "warn" if isinstance(vix, (int, float)) and vix > 18 else "pos"
    # Color term ratio
    tr_class = "neg" if isinstance(term_ratio, (int, float)) and term_ratio > 1.0 else "pos"
    # Color correlation
    corr_class = "neg" if isinstance(corr, (int, float)) and corr > 0.7 else "warn" if isinstance(corr, (int, float)) and corr > 0.5 else "pos"

    stats = [
        ("VIX", str(vix), vix_class),
        ("VIX/VIX3M", str(term_ratio), tr_class),
        ("Sector Disp %", str(disp), "neutral"),
        ("Avg Correlation", str(corr), corr_class),
        ("Anthro Rate", pct, "warn" if flagged_count > total * 0.4 else "neutral"),
    ]
    return writer.stat_bar(stats)


def build_reality_section(writer, reality, verdicts):
    """Build the quantitative reality panel."""
    # Metric cards
    metrics = [
        ("VIX", reality.get("vix", "N/A"), reality.get("vix_stress", "")),
        ("VIX 5d Change", f"{reality.get('vix_5d_chg', 'N/A')}%", ""),
        ("VIX3M", reality.get("vix3m", "N/A"), reality.get("vix_term_structure", "")),
        ("SKEW", reality.get("skew", "N/A"), "Tail risk pricing"),
        ("SPY 20d Vol (ann.)", f"{reality.get('spy_20d_vol', 'N/A')}%", ""),
        ("Sector Dispersion", f"{reality.get('sector_dispersion', 'N/A')}%", ""),
        ("Avg Sector Corr", reality.get("avg_sector_corr", "N/A"), reality.get("flow_verdict", "")),
        ("Best Sector (20d)", reality.get("best_sector", "N/A"), ""),
        ("Worst Sector (20d)", reality.get("worst_sector", "N/A"), ""),
    ]

    cards_html = '<div class="reality-grid">'
    for label, value, note in metrics:
        cards_html += f'''
        <div class="reality-card">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
            {"<div style='font-size:0.75em;color:#64748b;margin-top:2px;'>" + note + "</div>" if note else ""}
        </div>'''
    cards_html += '</div>'

    # Verdicts
    verdict_html = '<h3 style="margin-top:16px;color:#1e40af;">Reality Verdict</h3>'
    for v in verdicts:
        verdict_html += f'<div class="verdict-item">{v}</div>'

    return writer.section("Quantitative Reality", cards_html + verdict_html,
                          hint="What the data actually shows -- no emotion, no metaphors")


def format_measured_value(measure_name, reality):
    """Turn a measure name into the actual current value from reality data."""
    mapping = {
        "VIX": ("vix", "VIX={v}"),
        "VIX (low)": ("vix", "VIX={v}"),
        "put/call": None,
        "put/call (low)": None,
        "SKEW": ("skew", "SKEW={v}"),
        "VIX term structure": ("vix_term_ratio", "VIX/VIX3M={v}"),
        "correlation": ("avg_sector_corr", "SectorCorr={v}"),
        "breadth": None,
        "breadth (high)": None,
        "volume": None,
        "volume at price": None,
        "order flow": None,
        "money flow": None,
        "dispersion": ("sector_dispersion", "Disp={v}%"),
        "ADX": None,
        "RSI": None,
        "OBV": None,
        "ATR": None,
        "Bollinger width": None,
        "bullish %": None,
    }
    info = mapping.get(measure_name)
    if info is None:
        return measure_name
    key, fmt = info
    val = reality.get(key)
    if val is None:
        return measure_name
    return fmt.format(v=val)


def generate_article_verdict(article, reality):
    """Generate a data-driven verdict for a single flagged article."""
    phrases = [d["phrase"] for d in article["anthro_details"]]
    categories = set(d["category"] for d in article["anthro_details"])
    measures_needed = set()
    for d in article["anthro_details"]:
        for m in d["measure"]:
            if m != "UNFALSIFIABLE":
                measures_needed.add(m)

    verdicts = []

    # VIX-related verdicts
    vix = reality.get("vix")
    if vix is not None and measures_needed & {"VIX", "VIX (low)", "put/call", "put/call (low)"}:
        if "fear" in categories or "panic" in categories:
            if vix > 30:
                verdicts.append(f"VIX {vix} confirms elevated stress")
            elif vix > 20:
                verdicts.append(f"VIX {vix} shows moderate hedging, not panic")
            else:
                verdicts.append(f"VIX {vix} is LOW -- no fear in options market")
        if "euphoria" in categories:
            if vix < 15:
                verdicts.append(f"VIX {vix} confirms complacency zone")
            else:
                verdicts.append(f"VIX {vix} too high for euphoria -- hedging active")

    # Term structure verdicts
    term_ratio = reality.get("vix_term_ratio")
    if term_ratio is not None and "VIX term structure" in measures_needed:
        if term_ratio > 1.0:
            verdicts.append(f"VIX/VIX3M {term_ratio} backwardated -- acute stress confirmed")
        else:
            verdicts.append(f"VIX/VIX3M {term_ratio} contango -- orderly, NOT panic")

    # Correlation/dispersion verdicts
    corr = reality.get("avg_sector_corr")
    disp = reality.get("sector_dispersion")
    if corr is not None and measures_needed & {"correlation", "dispersion"}:
        if corr > 0.7:
            verdicts.append(f"Sector corr {corr} -- broad selling together")
        elif corr < 0.4 and disp and disp > 0.5:
            verdicts.append(f"Corr {corr}, disp {disp}% -- rotation, not panic")
        else:
            verdicts.append(f"Sector corr {corr} -- mixed, not extreme")

    # SKEW
    skew = reality.get("skew")
    if skew is not None and "SKEW" in measures_needed:
        if skew > 150:
            verdicts.append(f"SKEW {skew} elevated -- tail hedging active")
        else:
            verdicts.append(f"SKEW {skew} -- normal tail risk pricing")

    # Unfalsifiable
    has_unfalsifiable = any(
        "UNFALSIFIABLE" in d["measure"] for d in article["anthro_details"]
    )
    if has_unfalsifiable:
        verdicts.append("Contains unfalsifiable claims -- no data can verify or refute")

    if not verdicts:
        verdicts.append("No cached data available for these metrics")

    return " | ".join(verdicts)


def build_commentary_section(writer, articles, reality):
    """Build the commentary scan table with actual values and verdicts."""
    flagged = [a for a in articles if a["anthro_count"] > 0]
    flagged.sort(key=lambda x: x["anthro_count"], reverse=True)

    # Category distribution
    all_cats = []
    for a in flagged:
        all_cats.extend(a["anthro_categories"])
    cat_counts = Counter(all_cats)

    cat_html = '<div class="category-bar">'
    for cat, count in cat_counts.most_common():
        color = CATEGORY_COLORS.get(cat, "#95a5a6")
        cat_html += f'<span class="cat-chip" style="background:{color}">{cat}: {count}</span>'
    cat_html += '</div>'

    # Phrase frequency
    all_phrases = []
    for a in flagged:
        all_phrases.extend(a["anthro_phrases"])
    phrase_counts = Counter(all_phrases).most_common(10)

    phrase_html = '<div style="margin:8px 0;font-size:0.85em;color:#64748b;">Top phrases: '
    phrase_html += ', '.join(f'<strong>{p}</strong> ({c})' for p, c in phrase_counts)
    phrase_html += '</div>'

    # Build article cards (not a table -- each article gets a card with verdict)
    cards_html = ''
    for a in flagged[:30]:
        # Date
        pub_date = a.get("published", "")
        date_display = pub_date[:10] if len(pub_date) >= 10 else pub_date

        # Build clickable headline
        link = a.get("link", "")
        title = a.get("title", "(no title)")
        title_safe = title.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
        if link:
            headline_html = f'<a class="article-link" href="{link}" target="_blank" rel="noopener noreferrer">{title_safe}</a>'
        else:
            headline_html = title_safe

        # Build phrase tags
        phrase_tags = ""
        measure_values = []
        for det in a["anthro_details"]:
            color = CATEGORY_COLORS.get(det["category"], "#95a5a6")
            phrase_tags += f'<span class="anthro-tag" style="background:{color}">{det["phrase"]}</span> '
            for m in det["measure"]:
                if m == "UNFALSIFIABLE":
                    continue
                val_str = format_measured_value(m, reality)
                if val_str not in measure_values:
                    measure_values.append(val_str)

        has_unfals = any("UNFALSIFIABLE" in det["measure"] for det in a["anthro_details"])
        if has_unfals:
            phrase_tags += '<span class="anthro-tag" style="background:#dc2626">UNFALSIFIABLE</span> '

        measures_str = ", ".join(measure_values) if measure_values else "N/A"

        # Generate per-article verdict
        verdict = generate_article_verdict(a, reality)

        cards_html += f'''
        <div style="background:#f8f9fa;border-radius:8px;padding:12px 16px;margin:8px 0;border-left:3px solid {CATEGORY_COLORS.get(a["anthro_categories"][0], "#95a5a6") if a["anthro_categories"] else "#95a5a6"};">
            <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:6px;">
                <div>{headline_html}</div>
                <div style="font-size:0.75em;color:#94a3b8;white-space:nowrap;margin-left:12px;">{a["source"]} | {date_display}</div>
            </div>
            <div style="margin-bottom:6px;">{phrase_tags}</div>
            <div style="font-size:0.82em;color:#475569;margin-bottom:4px;"><strong>Measured:</strong> {measures_str}</div>
            <div style="font-size:0.82em;color:#1e40af;background:#eff6ff;padding:6px 10px;border-radius:4px;"><strong>Verdict:</strong> {verdict}</div>
        </div>'''

    # Filler warning
    filler_count = sum(
        1 for a in flagged
        for d in a["anthro_details"]
        if d["category"] == "filler"
    )
    filler_html = ""
    if filler_count > 0:
        filler_html = f'''
        <div class="filler-warning">
            {filler_count} instance(s) of unfalsifiable language detected
            (e.g. "profit-taking", "bargain hunting"). These phrases sound
            explanatory but cannot be verified or disproven with any data.
        </div>'''

    total = len(articles)
    flagged_count = len(flagged)
    total_anthros = sum(a["anthro_count"] for a in articles)
    summary = (
        f"<p style='font-size:0.9em;color:#475569;'>"
        f"Scanned <strong>{total}</strong> articles | "
        f"<strong>{flagged_count}</strong> contain anthropomorphisms | "
        f"<strong>{total_anthros}</strong> total emotional phrases | "
        f"<strong>{flagged_count * 100 // max(total, 1)}%</strong> anthropomorphism rate</p>"
    )

    content = summary + cat_html + phrase_html + cards_html + filler_html
    return writer.section("Commentary Anthropomorphism Scanner", content,
                          hint="Click headlines to read the original article")


def build_methodology_section(writer):
    """Short methodology note."""
    content = '''
    <p style="font-size:0.85em;color:#475569;line-height:1.6;">
    Based on <strong>Morris et al. (2007)</strong>: agent metaphors ("the market climbed")
    cause investors to expect trend continuance, while object metaphors ("the market was pushed")
    do not. This dashboard scans live commentary for 80+ anthropomorphic phrases, classifies them
    by category (fear, panic, exhaustion, euphoria, etc.), cross-references with quantitative
    metrics from price_cache, and generates per-article verdicts comparing claims to reality.
    Context filtering ensures only market-as-person language is flagged (not "nervous consumers").
    <br><br>
    <strong>Key distinction:</strong> "skittish" implies irrational fear that should revert.
    "Rotating" implies orderly capital reallocation that may persist. These carry opposite trading
    implications, yet commentators use them interchangeably. Check the data, not the metaphor.
    </p>
    '''
    return writer.section("Methodology", content,
                          hint="Morris et al., Columbia/UCLA, 2007")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("  Market Reality Check Backend v1.1")
    print("=" * 60)
    t0 = time.time()

    # 1. Fetch RSS
    print("\n[FETCH] Scanning RSS feeds...")
    articles = fetch_headlines(max_per_feed=50)
    print(f"  Got {len(articles)} articles from {len(RSS_FEEDS)} feeds")

    if not articles:
        print("  [!] No articles fetched. Using demo data.")
        articles = [{
            "source": "Demo",
            "title": "Markets remain skittish amid rotation fears",
            "link": "",
            "text": "Markets remain skittish as investors grow nervous about sector rotation.",
            "published": datetime.now().isoformat(),
        }]

    # 2. Analyze for anthropomorphisms
    print("[SCAN] Detecting anthropomorphisms...")
    articles = analyze_articles(articles)
    flagged = [a for a in articles if a["anthro_count"] > 0]
    total_anthros = sum(a["anthro_count"] for a in articles)
    print(f"  {len(flagged)}/{len(articles)} articles flagged, {total_anthros} total phrases")

    # 3. Compute quantitative reality
    print("[DATA] Computing market reality from price_cache...")
    reality = compute_market_reality()
    verdicts = generate_reality_verdicts(reality)
    for v in verdicts:
        print(f"  --> {v}")

    # 4. Build dashboard HTML
    print("[HTML] Building dashboard...")
    writer = DashboardWriter("market-reality", "Market Reality Check")

    parts = []
    parts.append(writer.build_header("Anthropomorphism vs Quantitative Truth"))
    parts.append(build_stat_bar(writer, reality, articles, len(flagged)))
    parts.append(build_reality_section(writer, reality, verdicts))
    parts.append(build_commentary_section(writer, articles, reality))
    parts.append(build_methodology_section(writer))
    parts.append(writer.llm_block())
    parts.append(writer.footer())

    body = "\n".join(parts)
    writer.write(body, extra_css=EXTRA_CSS)

    # 5. Save data snapshot
    snapshot = {
        "generated_at": datetime.now().isoformat(),
        "reality": reality,
        "verdicts": verdicts,
        "total_articles": len(articles),
        "flagged_articles": len(flagged),
        "total_anthros": total_anthros,
        "top_phrases": Counter(
            p for a in articles for p in a["anthro_phrases"]
        ).most_common(15),
    }
    snap_path = _SCRIPT_DIR / 'market-reality' / 'market_reality_data.json'
    os.makedirs(snap_path.parent, exist_ok=True)
    with open(snap_path, 'w', encoding='utf-8') as f:
        json.dump(snapshot, f, indent=2, default=str)

    elapsed = time.time() - t0
    print(f"\n[OK] Market Reality Check complete in {elapsed:.1f}s")
    print(f"[OK] Dashboard: market-reality/index.html")
    return snapshot


if __name__ == "__main__":
    main()
