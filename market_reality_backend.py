# ============================================================================
# market_reality_backend.py - v1.2
# Last updated: 2026-03-17
# ============================================================================
# v1.2: 3-tier metaphor architecture (anthropomorphism / magnitude / narrative),
#       per-Morris et al agent vs object distinction, ADX/RSI/ATR/breadth
#       computed for full verdict coverage, subject-aware magnitude verdicts,
#       fixed ALWAYS_MARKET overmatch, updated methodology section
# v1.1: Context filtering, date column, more feeds, synonyms, measured values
# v1.0: Initial build - RSS scan, anthro detection, quant reality from cache
# ============================================================================
"""
Market Reality Check Backend
============================
Scans financial commentary RSS feeds for three types of misleading language:

  TIER 1 - ANTHROPOMORPHISM: Agent metaphors that treat markets as people
           with emotions/will (Morris et al. 2007). These cause investors
           to expect trend continuance.
  TIER 2 - MAGNITUDE HYPERBOLE: Dramatic words that exaggerate the size
           of a move (soaring, plunging, freefall). Reality check: what
           was the actual % move?
  TIER 3 - UNFALSIFIABLE NARRATIVE: Post-hoc explanations that sound
           meaningful but can't be tested (profit-taking, bargain hunting).

Cross-references all detections with quantitative market data from price_cache.

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
# 3-TIER LEXICON
# ============================================================================
# TIER 1: ANTHROPOMORPHISM (agent metaphors -- market treated as person)
# These are the Morris et al. finding: attributing will/emotion to markets
# causes investors to expect trend continuance.

TIER1_ANTHROPOMORPHISM = {
    # FEAR / NERVOUSNESS -- market treated as a scared person
    "skittish":     {"category": "fear",        "implies": "market is irrationally nervous",    "measure": ["VIX", "SKEW"]},
    "nervous":      {"category": "fear",        "implies": "market has anxiety",                "measure": ["VIX", "SKEW"]},
    "jittery":      {"category": "fear",        "implies": "market has fragile confidence",     "measure": ["VIX", "SKEW"]},
    "fearful":      {"category": "fear",        "implies": "market is emotionally selling",     "measure": ["VIX", "correlation"]},
    "scared":       {"category": "fear",        "implies": "market is irrationally avoiding",   "measure": ["VIX", "SKEW"]},
    "worried":      {"category": "fear",        "implies": "market has anxiety",                "measure": ["VIX", "SKEW"]},
    "spooked":      {"category": "fear",        "implies": "market was startled",               "measure": ["VIX", "SKEW"]},
    "anxious":      {"category": "fear",        "implies": "market has generalized worry",      "measure": ["VIX", "SKEW"]},
    "rattled":      {"category": "fear",        "implies": "market's confidence is shaken",     "measure": ["VIX", "breadth"]},
    "on edge":      {"category": "fear",        "implies": "market is ready to bolt",           "measure": ["VIX", "SKEW"]},
    "uneasy":       {"category": "fear",        "implies": "market feels discomfort",           "measure": ["VIX", "SKEW"]},
    "trembling":    {"category": "fear",        "implies": "market is shaking with fear",       "measure": ["VIX", "SKEW"]},
    "twitchy":      {"category": "fear",        "implies": "market is jumpy and reactive",      "measure": ["VIX", "SKEW"]},
    "apprehensive": {"category": "fear",        "implies": "market dreads what's next",         "measure": ["VIX", "SKEW"]},
    "wary":         {"category": "fear",        "implies": "market is guarded and watchful",    "measure": ["VIX", "SKEW"]},
    "panicking":    {"category": "fear",        "implies": "market is in irrational liquidation","measure": ["VIX term structure", "correlation", "breadth"]},
    "panic":        {"category": "fear",        "implies": "market has mass irrational fear",   "measure": ["VIX term structure", "correlation"]},
    "capitulating": {"category": "fear",        "implies": "market is surrendering",            "measure": ["VIX term structure", "breadth"]},
    "capitulation": {"category": "fear",        "implies": "market is giving up",               "measure": ["VIX term structure", "breadth"]},

    # EXHAUSTION -- market treated as a tired person
    "tired":        {"category": "exhaustion",  "implies": "market needs rest",                 "measure": ["ADX", "RSI"]},
    "exhausted":    {"category": "exhaustion",  "implies": "market's energy is depleted",       "measure": ["ADX", "RSI"]},
    "fatigued":     {"category": "exhaustion",  "implies": "market's momentum is weakening",    "measure": ["ADX", "RSI"]},
    "running out of steam": {"category": "exhaustion", "implies": "market is decelerating",     "measure": ["ADX", "RSI"]},
    "losing steam": {"category": "exhaustion",  "implies": "market's momentum is fading",       "measure": ["ADX", "RSI"]},
    "out of gas":   {"category": "exhaustion",  "implies": "market has no fuel left",           "measure": ["ADX", "RSI"]},
    "winded":       {"category": "exhaustion",  "implies": "market needs to catch its breath",  "measure": ["ADX", "RSI"]},
    "spent":        {"category": "exhaustion",  "implies": "market used up all energy",         "measure": ["ADX", "RSI"]},
    "running on fumes": {"category": "exhaustion", "implies": "market is nearly depleted",      "measure": ["ADX", "RSI"]},
    "petering out": {"category": "exhaustion",  "implies": "market is gradually dying",         "measure": ["ADX", "RSI"]},

    # CONSOLIDATION -- market treated as a thinking person
    "digesting":    {"category": "consolidation", "implies": "market is thoughtfully processing", "measure": ["ATR", "ADX"]},
    "absorbing":    {"category": "consolidation", "implies": "market is taking in information",   "measure": ["ATR", "ADX"]},
    "mulling":      {"category": "consolidation", "implies": "market is deliberating",            "measure": ["ATR", "ADX"]},
    "processing":   {"category": "consolidation", "implies": "market is doing cognitive work",    "measure": ["ATR", "ADX"]},
    "pausing":      {"category": "consolidation", "implies": "market chose to stop temporarily",  "measure": ["ATR", "ADX"]},
    "catching its breath": {"category": "consolidation", "implies": "market is resting before next move", "measure": ["ATR", "ADX"]},
    "taking a breather": {"category": "consolidation", "implies": "market chose a brief rest",    "measure": ["ATR", "ADX"]},
    "treading water": {"category": "consolidation", "implies": "market is staying afloat",        "measure": ["ATR", "ADX"]},

    # CONFUSION -- market treated as a disoriented person
    "confused":     {"category": "confusion",   "implies": "market doesn't know what to do",    "measure": ["dispersion", "correlation"]},
    "uncertain":    {"category": "confusion",   "implies": "market lacks direction",            "measure": ["VIX", "dispersion"]},
    "mixed signals": {"category": "confusion",  "implies": "market is sending contradictions",  "measure": ["dispersion", "breadth"]},
    "indecisive":   {"category": "confusion",   "implies": "market can't make up its mind",     "measure": ["ATR", "dispersion"]},
    "directionless": {"category": "confusion",  "implies": "market has no sense of purpose",    "measure": ["ADX", "dispersion"]},
    "lost":         {"category": "confusion",   "implies": "market doesn't know where to go",   "measure": ["ADX", "dispersion"]},
    "torn":         {"category": "confusion",   "implies": "market is pulled both ways",        "measure": ["dispersion", "correlation"]},
    "searching for direction": {"category": "confusion", "implies": "market is aimlessly moving", "measure": ["ADX", "dispersion"]},
    "at a crossroads": {"category": "confusion", "implies": "market faces a decision",          "measure": ["ATR", "VIX"]},

    # EUPHORIA -- market treated as an ecstatic person
    "euphoric":     {"category": "euphoria",    "implies": "market has irrational exuberance",  "measure": ["VIX", "RSI"]},
    "exuberant":    {"category": "euphoria",    "implies": "market is excessively optimistic",  "measure": ["VIX", "RSI"]},
    "giddy":        {"category": "euphoria",    "implies": "market is childlike with excitement","measure": ["VIX", "RSI"]},
    "ecstatic":     {"category": "euphoria",    "implies": "market feels extreme joy",          "measure": ["VIX", "RSI"]},
    "frothy":       {"category": "euphoria",    "implies": "market is bubbly and overvalued",   "measure": ["VIX", "RSI"]},
    "bubbly":       {"category": "euphoria",    "implies": "market has unsustainable optimism", "measure": ["VIX", "RSI"]},
    "ebullient":    {"category": "euphoria",    "implies": "market is enthusiastically optimistic", "measure": ["VIX", "RSI"]},
    "intoxicated":  {"category": "euphoria",    "implies": "market is drunk on gains",          "measure": ["VIX", "RSI"]},

    # AGGRESSION -- market treated as a fighter
    "fighting":     {"category": "aggression",  "implies": "market is willfully struggling",    "measure": ["breadth", "ADX"]},
    "battling":     {"category": "aggression",  "implies": "market is in combat at levels",     "measure": ["breadth", "ADX"]},
    "struggling":   {"category": "aggression",  "implies": "market has difficulty advancing",   "measure": ["breadth", "ADX"]},
    "wrestling":    {"category": "aggression",  "implies": "market is on contested ground",     "measure": ["breadth", "ADX"]},
    "clawing back": {"category": "aggression",  "implies": "market is desperately recovering",  "measure": ["breadth", "RSI"]},
    "defending":    {"category": "aggression",  "implies": "market is holding a line",          "measure": ["breadth", "ADX"]},
    "under attack": {"category": "aggression",  "implies": "market is being assaulted",         "measure": ["VIX", "breadth"]},
    "under siege":  {"category": "aggression",  "implies": "market faces sustained assault",    "measure": ["VIX", "breadth"]},
    "punished":     {"category": "aggression",  "implies": "market is suffering consequences",  "measure": ["VIX", "breadth"]},

    # HEALTH / SICKNESS -- market treated as a patient
    "healthy":      {"category": "health",      "implies": "market is in good condition",       "measure": ["breadth", "ADX"]},
    "sick":         {"category": "health",      "implies": "market is diseased",                "measure": ["breadth", "correlation"]},
    "recovering":   {"category": "health",      "implies": "market is healing from illness",    "measure": ["breadth", "RSI"]},
    "ailing":       {"category": "health",      "implies": "market is persistently ill",        "measure": ["breadth", "ADX"]},
    "wounded":      {"category": "health",      "implies": "market is injured but alive",       "measure": ["breadth", "RSI"]},
    "limping":      {"category": "health",      "implies": "market is moving but impaired",     "measure": ["breadth", "ADX"]},
    "bruised":      {"category": "health",      "implies": "market is hurt but not broken",     "measure": ["breadth", "RSI"]},
    "bleeding":     {"category": "health",      "implies": "market is losing life force",       "measure": ["breadth", "RSI"]},
    "contagion":    {"category": "health",      "implies": "market sickness is spreading",      "measure": ["correlation", "breadth"]},
    "infected":     {"category": "health",      "implies": "market disease is spreading",       "measure": ["correlation", "breadth"]},

    # CAUTION (agent -- market choosing to be careful)
    "cautious":     {"category": "caution",     "implies": "market chose to be careful",        "measure": ["VIX", "ATR"]},
    "resilient":    {"category": "caution",     "implies": "market is strong despite adversity","measure": ["breadth", "RSI"]},
    "bracing for":  {"category": "caution",     "implies": "market is preparing for impact",    "measure": ["VIX", "SKEW"]},
}

# TIER 2: MAGNITUDE HYPERBOLE (object/physics metaphors -- dramatic size words)
# These describe HOW MUCH something moved, not market emotions.
# The reality check: what was the actual % move?

TIER2_MAGNITUDE = {
    "soaring":      {"category": "magnitude_up",   "implies": "extremely large upward move"},
    "surging":      {"category": "magnitude_up",   "implies": "extremely large upward move"},
    "skyrocketing": {"category": "magnitude_up",   "implies": "explosive upward move"},
    "moonshot":     {"category": "magnitude_up",   "implies": "absurdly large upward move"},
    "exploding":    {"category": "magnitude_up",   "implies": "violent upward move"},

    "plunging":     {"category": "magnitude_down", "implies": "extremely large downward move"},
    "crashing":     {"category": "magnitude_down", "implies": "violent downward impact"},
    "cratering":    {"category": "magnitude_down", "implies": "collapsing violently"},
    "tanking":      {"category": "magnitude_down", "implies": "rapid uncontrolled fall"},
    "freefall":     {"category": "magnitude_down", "implies": "uncontrolled descent"},
    "free fall":    {"category": "magnitude_down", "implies": "uncontrolled descent"},
    "imploding":    {"category": "magnitude_down", "implies": "collapsing inward"},
    "meltdown":     {"category": "magnitude_down", "implies": "total system collapse"},
    "bloodbath":    {"category": "magnitude_down", "implies": "extreme destruction"},
    "carnage":      {"category": "magnitude_down", "implies": "mass destruction"},
    "hammered":     {"category": "magnitude_down", "implies": "beaten down severely"},
    "decimated":    {"category": "magnitude_down", "implies": "destroyed by large fraction"},
    "obliterated":  {"category": "magnitude_down", "implies": "completely destroyed"},
    "wiped out":    {"category": "magnitude_down", "implies": "total loss"},
    "gutted":       {"category": "magnitude_down", "implies": "hollowed out"},
}

# TIER 3: UNFALSIFIABLE NARRATIVE (post-hoc explanations that can't be tested)
TIER3_UNFALSIFIABLE = {
    "profit-taking":  {"category": "unfalsifiable", "implies": "intentional selling (unverifiable)"},
    "profit taking":  {"category": "unfalsifiable", "implies": "intentional selling (unverifiable)"},
    "bargain hunting": {"category": "unfalsifiable", "implies": "intentional buying (unverifiable)"},
    "sidelined":      {"category": "unfalsifiable", "implies": "waiting to act (unverifiable)"},
    "shrugging off":  {"category": "unfalsifiable", "implies": "ignoring bad news (unverifiable)"},
    "betting on":     {"category": "unfalsifiable", "implies": "wagering / gambling (unverifiable)"},
    "positioning for": {"category": "unfalsifiable", "implies": "strategic placement (unverifiable)"},
    "pricing in":     {"category": "unfalsifiable", "implies": "already accounted for (unverifiable)"},
    "front-running":  {"category": "unfalsifiable", "implies": "acting ahead of news (unverifiable)"},
}

# Tier label and color mapping
TIER_LABELS = {
    1: "Anthropomorphism",
    2: "Magnitude Hyperbole",
    3: "Unfalsifiable Narrative",
}

TIER_COLORS = {
    1: "#8b5cf6",   # purple -- agent metaphor
    2: "#f97316",   # orange -- dramatic size word
    3: "#6b7280",   # gray -- unfalsifiable filler
}

CATEGORY_COLORS = {
    "fear":           "#e74c3c",
    "exhaustion":     "#e67e22",
    "consolidation":  "#27ae60",
    "confusion":      "#f1c40f",
    "euphoria":       "#3498db",
    "aggression":     "#8e44ad",
    "health":         "#1abc9c",
    "caution":        "#95a5a6",
    "magnitude_up":   "#f97316",
    "magnitude_down": "#dc2626",
    "unfalsifiable":  "#6b7280",
}

# ============================================================================
# MARKET SUBJECT CONTEXT FILTER
# ============================================================================

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

NON_MARKET_SUBJECTS = [
    "middle class", "households", "families", "voters",
    "police", "capitol", "congress", "election",
    "patients", "students", "children",
    "he said", "she said",
    "rent ", "housing prices", "home prices",
    "reputation", "career", "popularity",
]

# Only words that are EXCLUSIVELY used in financial context bypass context check
# Removed: soaring, plunging, crashing, tanking, imploding (used outside finance)
ALWAYS_MARKET = {
    "bloodbath", "carnage", "capitulation", "capitulating",
    "meltdown", "frothy", "bubbly", "profit-taking",
    "profit taking", "bargain hunting", "sidelined",
    "clawing back", "pricing in",
}


def is_market_context(text, phrase):
    """Check if the phrase is in a financial context, not describing people/rent/etc."""
    text_lower = text.lower()
    pos = text_lower.find(phrase.lower())
    if pos == -1:
        return False

    if phrase.lower() in ALWAYS_MARKET:
        return True

    window_start = max(0, pos - 100)
    window_end = min(len(text_lower), pos + len(phrase) + 100)
    window = text_lower[window_start:window_end]

    has_market = any(subj in window for subj in MARKET_SUBJECTS)
    has_non_market = any(subj in window for subj in NON_MARKET_SUBJECTS)

    if has_non_market and not has_market:
        return False

    return has_market


# ============================================================================
# SUBJECT EXTRACTION (for magnitude hyperbole)
# ============================================================================

# Map subjects found in headlines to tickers we can check
SUBJECT_TICKERS = {
    "oil": "USO", "crude": "USO", "brent": "USO", "wti": "USO",
    "gold": "GLD", "silver": "SLV",
    "bitcoin": "BITO", "btc": "BITO", "crypto": "BITO",
    "nasdaq": "QQQ", "tech": "QQQ",
    "dow": "DIA", "s&p": "SPY", "s&p 500": "SPY",
    "spy": "SPY", "qqq": "QQQ", "dia": "DIA", "iwm": "IWM",
    "russell": "IWM",
    "treasury": "TLT", "treasuries": "TLT", "bond": "TLT", "bonds": "TLT",
    "india": "INDA", "china": "FXI", "japan": "EWJ", "europe": "VGK",
    "emerging": "EEM",
    "financials": "XLF", "banks": "XLF", "energy": "XLE",
    "semiconductor": "SMH", "chips": "SMH",
}


def extract_subject_ticker(text):
    """Try to find what asset is being described, return ticker or 'SPY' default."""
    text_lower = text.lower()
    for subject, ticker in SUBJECT_TICKERS.items():
        if subject in text_lower:
            return ticker, subject
    return "SPY", "market"


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
# 3-TIER DETECTION
# ============================================================================

def _phrase_in_text(phrase, text_lower):
    """Check if phrase appears in text (word-boundary for single words, substring for multi)."""
    if " " in phrase:
        return phrase in text_lower
    pattern = r"\b" + re.escape(phrase) + r"\b"
    return bool(re.search(pattern, text_lower))


def detect_all_tiers(text):
    """Scan text for all 3 tiers of misleading language. Returns list of match dicts."""
    text_lower = text.lower()
    matches = []

    # Tier 1: Anthropomorphism
    for phrase, meta in TIER1_ANTHROPOMORPHISM.items():
        if _phrase_in_text(phrase, text_lower) and is_market_context(text, phrase):
            matches.append({"phrase": phrase, "tier": 1, **meta})

    # Tier 2: Magnitude hyperbole
    for phrase, meta in TIER2_MAGNITUDE.items():
        if _phrase_in_text(phrase, text_lower) and is_market_context(text, phrase):
            matches.append({"phrase": phrase, "tier": 2, **meta})

    # Tier 3: Unfalsifiable narrative
    for phrase, meta in TIER3_UNFALSIFIABLE.items():
        if _phrase_in_text(phrase, text_lower) and is_market_context(text, phrase):
            matches.append({"phrase": phrase, "tier": 3, **meta})

    return matches


def analyze_articles(articles):
    """Add 3-tier analysis to each article."""
    for art in articles:
        matches = detect_all_tiers(art["text"])
        art["match_count"] = len(matches)
        art["match_details"] = matches
        art["match_phrases"] = [m["phrase"] for m in matches]
        art["match_tiers"] = list(set(m["tier"] for m in matches))
        art["match_categories"] = list(set(m["category"] for m in matches))
        # Back-compat aliases
        art["anthro_count"] = len(matches)
        art["anthro_phrases"] = art["match_phrases"]
        art["anthro_categories"] = art["match_categories"]
        art["anthro_details"] = matches
    return articles


# ============================================================================
# QUANTITATIVE REALITY ENGINE
# ============================================================================

def compute_market_reality():
    """Compute quantitative metrics from price_cache -- covers all 3 tiers."""
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

    # --- SPY metrics (RSI, ADX, ATR, realized vol) ---
    spy_df = load_ticker("SPY")
    if spy_df is not None and len(spy_df) > 30:
        spy_close = spy_df['close']
        spy_high = spy_df['high']
        spy_low = spy_df['low']
        spy_ret = spy_close.pct_change().dropna()

        # Realized vol
        reality["spy_20d_vol"] = round(float(spy_ret.iloc[-20:].std() * np.sqrt(252) * 100), 1)

        # RSI(14)
        delta = spy_close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta.clip(upper=0))
        avg_gain = gain.iloc[-14:].mean()
        avg_loss = loss.iloc[-14:].mean()
        if avg_loss > 0:
            rs = float(avg_gain / avg_loss)
            reality["spy_rsi"] = round(100 - (100 / (1 + rs)), 1)
        else:
            reality["spy_rsi"] = 100.0

        # ADX(14) -- simplified: average of +DI/-DI directional movement
        tr_vals = []
        plus_dm_vals = []
        minus_dm_vals = []
        for i in range(-15, 0):
            h = float(spy_high.iloc[i])
            l = float(spy_low.iloc[i])
            c_prev = float(spy_close.iloc[i - 1])
            tr_vals.append(max(h - l, abs(h - c_prev), abs(l - c_prev)))
            up_move = h - float(spy_high.iloc[i - 1])
            down_move = float(spy_low.iloc[i - 1]) - l
            plus_dm_vals.append(up_move if up_move > down_move and up_move > 0 else 0)
            minus_dm_vals.append(down_move if down_move > up_move and down_move > 0 else 0)

        atr14 = np.mean(tr_vals[-14:])
        reality["spy_atr"] = round(float(atr14), 2)
        reality["spy_atr_pct"] = round(float(atr14 / float(spy_close.iloc[-1]) * 100), 2)

        if atr14 > 0:
            plus_di = (np.mean(plus_dm_vals[-14:]) / atr14) * 100
            minus_di = (np.mean(minus_dm_vals[-14:]) / atr14) * 100
            dx = abs(plus_di - minus_di) / max(plus_di + minus_di, 0.001) * 100
            reality["spy_adx"] = round(float(dx), 1)
        else:
            reality["spy_adx"] = 0.0

        # Recent moves for magnitude calibration
        reality["spy_1d_ret"] = round(float(spy_ret.iloc[-1] * 100), 2)
        reality["spy_5d_ret"] = round(
            (float(spy_close.iloc[-1]) / float(spy_close.iloc[-5]) - 1) * 100, 2
        )
        reality["spy_20d_ret"] = round(
            (float(spy_close.iloc[-1]) / float(spy_close.iloc[-21]) - 1) * 100, 2
        )

        # Historical daily return distribution for magnitude calibration
        daily_abs = spy_ret.abs().iloc[-252:]  # last year
        reality["spy_daily_p50"] = round(float(daily_abs.quantile(0.5) * 100), 2)
        reality["spy_daily_p90"] = round(float(daily_abs.quantile(0.9) * 100), 2)
        reality["spy_daily_p95"] = round(float(daily_abs.quantile(0.95) * 100), 2)
        reality["spy_daily_p99"] = round(float(daily_abs.quantile(0.99) * 100), 2)

    # --- Breadth (% of S&P sectors above 20d SMA) ---
    sectors = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLY", "XLU", "XLB", "XLRE", "XLC"]
    sector_returns = {}
    sector_period_ret = {}
    above_sma20 = 0
    sector_count = 0
    for s in sectors:
        df = load_ticker(s)
        if df is not None and len(df) > 21:
            ret = df['close'].pct_change().dropna()
            sector_returns[s] = ret.iloc[-20:]
            sector_period_ret[s] = (float(df['close'].iloc[-1]) / float(df['close'].iloc[-21]) - 1) * 100
            sma20 = float(df['close'].iloc[-20:].mean())
            sector_count += 1
            if float(df['close'].iloc[-1]) > sma20:
                above_sma20 += 1

    if sector_count > 0:
        reality["breadth_pct"] = round(above_sma20 / sector_count * 100, 0)
        reality["breadth_label"] = f"{above_sma20}/{sector_count} sectors above 20d SMA"

    if len(sector_returns) >= 8:
        ret_df = pd.DataFrame(sector_returns)
        daily_disp = ret_df.std(axis=1)
        reality["sector_dispersion"] = round(float(daily_disp.mean() * 100), 3)

        corr = ret_df.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        avg_corr = float(corr.where(mask).stack().mean())
        reality["avg_sector_corr"] = round(avg_corr, 3)

        if reality["sector_dispersion"] > 0.6 and avg_corr < 0.5:
            reality["flow_verdict"] = "ROTATION -- capital moving between sectors, not leaving"
        elif avg_corr > 0.8:
            reality["flow_verdict"] = "SYSTEMATIC RISK-OFF -- all sectors selling together"
        elif avg_corr > 0.6:
            reality["flow_verdict"] = "MIXED -- some broad selling, some rotation"
        else:
            reality["flow_verdict"] = "DISPERSED -- low correlation, idiosyncratic moves"

        if sector_period_ret:
            best = max(sector_period_ret, key=sector_period_ret.get)
            worst = min(sector_period_ret, key=sector_period_ret.get)
            reality["best_sector"] = f"{best} ({sector_period_ret[best]:+.1f}%)"
            reality["worst_sector"] = f"{worst} ({sector_period_ret[worst]:+.1f}%)"

    # --- Pre-compute recent returns for common magnitude subjects ---
    magnitude_tickers = set(SUBJECT_TICKERS.values())
    magnitude_returns = {}
    for ticker in magnitude_tickers:
        df = load_ticker(ticker)
        if df is not None and len(df) > 21:
            c = df['close']
            ret_1d = (float(c.iloc[-1]) / float(c.iloc[-2]) - 1) * 100
            ret_5d = (float(c.iloc[-1]) / float(c.iloc[-5]) - 1) * 100
            ret_20d = (float(c.iloc[-1]) / float(c.iloc[-21]) - 1) * 100
            magnitude_returns[ticker] = {
                "1d": round(ret_1d, 2),
                "5d": round(ret_5d, 2),
                "20d": round(ret_20d, 2),
            }
    reality["magnitude_returns"] = magnitude_returns

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

    # New: RSI/ADX summary
    rsi = reality.get("spy_rsi")
    adx = reality.get("spy_adx")
    if rsi is not None:
        if rsi > 70:
            verdicts.append(f"SPY RSI(14) {rsi}: overbought territory")
        elif rsi < 30:
            verdicts.append(f"SPY RSI(14) {rsi}: oversold territory")
        else:
            verdicts.append(f"SPY RSI(14) {rsi}: neutral momentum")

    if adx is not None:
        if adx > 25:
            verdicts.append(f"SPY ADX {adx}: strong directional trend")
        else:
            verdicts.append(f"SPY ADX {adx}: weak/no trend")

    # Breadth
    breadth = reality.get("breadth_label")
    if breadth:
        pct = reality.get("breadth_pct", 0)
        if pct > 70:
            verdicts.append(f"Breadth: {breadth} -- broad participation")
        elif pct < 30:
            verdicts.append(f"Breadth: {breadth} -- narrow, most sectors below average")
        else:
            verdicts.append(f"Breadth: {breadth} -- mixed participation")

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
# PER-ARTICLE VERDICT GENERATION (3-tier aware)
# ============================================================================

def _tier1_verdict(article, reality):
    """Verdict for anthropomorphism: what emotion is claimed, what does data show?"""
    details = [d for d in article["match_details"] if d["tier"] == 1]
    if not details:
        return []

    verdicts = []
    categories = set(d["category"] for d in details)
    measures_needed = set()
    for d in details:
        for m in d.get("measure", []):
            measures_needed.add(m)

    phrases_str = ", ".join(f'"{d["phrase"]}"' for d in details)

    # VIX-based verdicts (fear, euphoria, caution)
    vix = reality.get("vix")
    if vix is not None and measures_needed & {"VIX", "SKEW"}:
        if categories & {"fear"}:
            if vix > 30:
                verdicts.append(f"Claims: {phrases_str}. VIX {vix} confirms elevated stress -- hedging IS heavy")
            elif vix > 20:
                verdicts.append(f"Claims: {phrases_str}. VIX {vix} shows moderate hedging -- concern, not panic")
            else:
                verdicts.append(f"Claims: {phrases_str}. VIX {vix} is LOW -- options market shows no fear")
        if categories & {"euphoria"}:
            rsi = reality.get("spy_rsi")
            if vix < 15 and rsi and rsi > 70:
                verdicts.append(f"Claims: {phrases_str}. VIX {vix}, RSI {rsi} -- complacency + overbought confirms euphoria signal")
            elif vix < 15:
                verdicts.append(f"Claims: {phrases_str}. VIX {vix} low (complacency zone) but RSI {rsi} not extreme")
            else:
                verdicts.append(f"Claims: {phrases_str}. VIX {vix} shows active hedging -- not consistent with euphoria")

    # Term structure (panic/capitulation)
    term_ratio = reality.get("vix_term_ratio")
    if term_ratio is not None and "VIX term structure" in measures_needed:
        if term_ratio > 1.0:
            verdicts.append(f"VIX/VIX3M {term_ratio} backwardated -- acute stress confirmed")
        else:
            verdicts.append(f"VIX/VIX3M {term_ratio} contango -- orderly, not panic")

    # ADX/RSI (exhaustion, consolidation, confusion)
    adx = reality.get("spy_adx")
    rsi = reality.get("spy_rsi")
    if adx is not None and "ADX" in measures_needed:
        if categories & {"exhaustion"}:
            if adx < 20:
                verdicts.append(f"Claims: {phrases_str}. ADX {adx} confirms weak trend -- momentum IS fading")
            else:
                verdicts.append(f"Claims: {phrases_str}. ADX {adx} shows trend still intact -- not exhausted")
        if categories & {"consolidation"}:
            atr_pct = reality.get("spy_atr_pct")
            verdicts.append(f"Claims: {phrases_str}. ADX {adx}, ATR {atr_pct}% of price -- {'low vol = consolidation plausible' if adx < 20 else 'directional trend active, not consolidating'}")
        if categories & {"confusion"}:
            disp = reality.get("sector_dispersion", 0)
            verdicts.append(f"Claims: {phrases_str}. ADX {adx}, sector dispersion {disp}% -- {'low direction confirms confusion' if adx < 20 else 'clear directional trend exists'}")

    # RSI for aggression/health
    if rsi is not None and "RSI" in measures_needed:
        if categories & {"aggression", "health"}:
            breadth = reality.get("breadth_label", "N/A")
            verdicts.append(f"Claims: {phrases_str}. RSI {rsi}, breadth: {breadth}")

    # Breadth
    if "breadth" in measures_needed:
        breadth = reality.get("breadth_label")
        if breadth and not any("breadth" in v for v in verdicts):
            verdicts.append(f"Breadth: {breadth}")

    # Correlation/dispersion
    corr = reality.get("avg_sector_corr")
    disp = reality.get("sector_dispersion")
    if corr is not None and measures_needed & {"correlation", "dispersion"}:
        if corr > 0.7:
            verdicts.append(f"Sector corr {corr} -- broad selling together")
        elif corr < 0.4 and disp and disp > 0.5:
            verdicts.append(f"Corr {corr}, disp {disp}% -- rotation, not panic")

    # SKEW
    skew = reality.get("skew")
    if skew is not None and "SKEW" in measures_needed and not any("SKEW" in v for v in verdicts):
        verdicts.append(f"SKEW {skew} -- {'elevated tail hedging' if skew > 150 else 'normal tail risk pricing'}")

    if not verdicts:
        verdicts.append(f"Agent metaphor detected ({phrases_str}) -- treats market as person with emotions")

    return verdicts


def _tier2_verdict(article, reality):
    """Verdict for magnitude hyperbole: how much did it actually move?"""
    details = [d for d in article["match_details"] if d["tier"] == 2]
    if not details:
        return []

    verdicts = []
    mag_returns = reality.get("magnitude_returns", {})

    # Find what subject this headline is about
    ticker, subject = extract_subject_ticker(article["text"])

    phrases_str = ", ".join(f'"{d["phrase"]}"' for d in details)
    is_down = any(d["category"] == "magnitude_down" for d in details)

    if ticker in mag_returns:
        rets = mag_returns[ticker]
        ret_1d = rets["1d"]
        ret_5d = rets["5d"]
        ret_20d = rets["20d"]

        # Calibrate against SPY daily distribution
        p90 = reality.get("spy_daily_p90", 1.5)
        p95 = reality.get("spy_daily_p95", 2.0)

        actual_1d = abs(ret_1d)
        if actual_1d > p95:
            calibration = "extreme (>95th percentile)"
        elif actual_1d > p90:
            calibration = "large (>90th percentile)"
        else:
            calibration = f"within normal range (90th pctl = {p90}%)"

        verdicts.append(
            f'Headline says {phrases_str} about {subject}. '
            f'Actual: {ticker} {ret_1d:+.1f}% (1d), {ret_5d:+.1f}% (5d), {ret_20d:+.1f}% (20d). '
            f'Daily move is {calibration}'
        )
    else:
        # Fallback to SPY
        spy_1d = reality.get("spy_1d_ret")
        spy_5d = reality.get("spy_5d_ret")
        if spy_1d is not None:
            verdicts.append(
                f'Headline says {phrases_str}. '
                f'No data for {subject}; SPY ref: {spy_1d:+.1f}% (1d), {spy_5d:+.1f}% (5d)'
            )
        else:
            verdicts.append(f'Headline says {phrases_str} -- magnitude claim, no price data to verify')

    return verdicts


def _tier3_verdict(article, reality):
    """Verdict for unfalsifiable: flag as empty explanation."""
    details = [d for d in article["match_details"] if d["tier"] == 3]
    if not details:
        return []

    phrases_str = ", ".join(f'"{d["phrase"]}"' for d in details)
    return [f'{phrases_str} -- sounds explanatory but no data can confirm or deny this. Treat as noise.']


def generate_article_verdict(article, reality):
    """Generate tier-aware verdict for a single flagged article."""
    all_verdicts = []
    all_verdicts.extend(_tier1_verdict(article, reality))
    all_verdicts.extend(_tier2_verdict(article, reality))
    all_verdicts.extend(_tier3_verdict(article, reality))

    if not all_verdicts:
        all_verdicts.append("Flagged but no verdict data available")

    return " | ".join(all_verdicts)


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
.tier-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 10px;
    font-size: 0.7em;
    font-weight: 700;
    color: #fff;
    margin-right: 4px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
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
.tier-legend {
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
    margin: 12px 0;
    padding: 10px 14px;
    background: #f1f5f9;
    border-radius: 6px;
    font-size: 0.82em;
}
.tier-legend .legend-item {
    display: flex;
    align-items: center;
    gap: 6px;
}
.tier-legend .dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    display: inline-block;
}
"""


def build_stat_bar(writer, reality, articles, flagged_count):
    """Build the top stat bar."""
    vix = reality.get("vix", "N/A")
    term_ratio = reality.get("vix_term_ratio", "N/A")
    rsi = reality.get("spy_rsi", "N/A")
    breadth = reality.get("breadth_pct", "N/A")
    total = len(articles)
    pct = f"{flagged_count}/{total}"

    vix_class = "neg" if isinstance(vix, (int, float)) and vix > 25 else "warn" if isinstance(vix, (int, float)) and vix > 18 else "pos"
    tr_class = "neg" if isinstance(term_ratio, (int, float)) and term_ratio > 1.0 else "pos"
    rsi_class = "neg" if isinstance(rsi, (int, float)) and rsi > 70 else "warn" if isinstance(rsi, (int, float)) and rsi < 30 else "pos"

    stats = [
        ("VIX", str(vix), vix_class),
        ("VIX/VIX3M", str(term_ratio), tr_class),
        ("SPY RSI(14)", str(rsi), rsi_class),
        ("Breadth %", f"{breadth}%" if breadth != "N/A" else "N/A", "pos" if isinstance(breadth, (int, float)) and breadth > 50 else "warn"),
        ("Flagged", pct, "warn" if flagged_count > total * 0.4 else "neutral"),
    ]
    return writer.stat_bar(stats)


def build_reality_section(writer, reality, verdicts):
    """Build the quantitative reality panel."""
    metrics = [
        ("VIX", reality.get("vix", "N/A"), reality.get("vix_stress", "")),
        ("VIX 5d Change", f"{reality.get('vix_5d_chg', 'N/A')}%", ""),
        ("VIX3M", reality.get("vix3m", "N/A"), reality.get("vix_term_structure", "")),
        ("SKEW", reality.get("skew", "N/A"), "Tail risk pricing"),
        ("SPY RSI(14)", reality.get("spy_rsi", "N/A"), ""),
        ("SPY ADX(14)", reality.get("spy_adx", "N/A"), "Trend strength"),
        ("SPY ATR(14)", f"{reality.get('spy_atr', 'N/A')} ({reality.get('spy_atr_pct', 'N/A')}%)", ""),
        ("SPY 20d Vol (ann.)", f"{reality.get('spy_20d_vol', 'N/A')}%", ""),
        ("Breadth", reality.get("breadth_label", "N/A"), ""),
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
            {"<div style='font-size:0.75em;color:#64748b;margin-top:2px;'>" + str(note) + "</div>" if note else ""}
        </div>'''
    cards_html += '</div>'

    verdict_html = '<h3 style="margin-top:16px;color:#1e40af;">Reality Verdict</h3>'
    for v in verdicts:
        verdict_html += f'<div class="verdict-item">{v}</div>'

    return writer.section("Quantitative Reality", cards_html + verdict_html,
                          hint="What the data actually shows -- no emotion, no metaphors")


def build_commentary_section(writer, articles, reality):
    """Build the commentary scan with 3-tier awareness."""
    flagged = [a for a in articles if a["match_count"] > 0]
    flagged.sort(key=lambda x: x["match_count"], reverse=True)

    # Tier distribution
    tier_counts = Counter()
    for a in flagged:
        for d in a["match_details"]:
            tier_counts[d["tier"]] += 1

    tier_html = '<div class="tier-legend">'
    for t in [1, 2, 3]:
        count = tier_counts.get(t, 0)
        color = TIER_COLORS[t]
        label = TIER_LABELS[t]
        tier_html += f'<span class="legend-item"><span class="dot" style="background:{color}"></span>{label}: {count}</span>'
    tier_html += '</div>'

    # Category distribution
    all_cats = []
    for a in flagged:
        all_cats.extend(a["match_categories"])
    cat_counts = Counter(all_cats)

    cat_html = '<div class="category-bar">'
    for cat, count in cat_counts.most_common():
        color = CATEGORY_COLORS.get(cat, "#95a5a6")
        cat_html += f'<span class="cat-chip" style="background:{color}">{cat}: {count}</span>'
    cat_html += '</div>'

    # Phrase frequency
    all_phrases = []
    for a in flagged:
        all_phrases.extend(a["match_phrases"])
    phrase_counts = Counter(all_phrases).most_common(10)

    phrase_html = '<div style="margin:8px 0;font-size:0.85em;color:#64748b;">Top phrases: '
    phrase_html += ', '.join(f'<strong>{p}</strong> ({c})' for p, c in phrase_counts)
    phrase_html += '</div>'

    # Article cards
    cards_html = ''
    for a in flagged[:30]:
        pub_date = a.get("published", "")
        date_display = pub_date[:10] if len(pub_date) >= 10 else pub_date

        link = a.get("link", "")
        title = a.get("title", "(no title)")
        title_safe = title.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
        if link:
            headline_html = f'<a class="article-link" href="{link}" target="_blank" rel="noopener noreferrer">{title_safe}</a>'
        else:
            headline_html = title_safe

        # Phrase tags with tier badges
        phrase_tags = ""
        for det in a["match_details"]:
            tier = det["tier"]
            tier_color = TIER_COLORS[tier]
            tier_label = f"T{tier}"
            cat_color = CATEGORY_COLORS.get(det["category"], "#95a5a6")
            phrase_tags += f'<span class="tier-badge" style="background:{tier_color}">{tier_label}</span>'
            phrase_tags += f'<span class="anthro-tag" style="background:{cat_color}">{det["phrase"]}</span> '

        # Per-article verdict
        verdict = generate_article_verdict(a, reality)

        # Card border color: use highest-priority tier
        tiers_present = [d["tier"] for d in a["match_details"]]
        primary_tier = min(tiers_present) if tiers_present else 1
        border_color = TIER_COLORS.get(primary_tier, "#95a5a6")

        cards_html += f'''
        <div style="background:#f8f9fa;border-radius:8px;padding:12px 16px;margin:8px 0;border-left:3px solid {border_color};">
            <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:6px;">
                <div>{headline_html}</div>
                <div style="font-size:0.75em;color:#94a3b8;white-space:nowrap;margin-left:12px;">{a["source"]} | {date_display}</div>
            </div>
            <div style="margin-bottom:6px;">{phrase_tags}</div>
            <div style="font-size:0.82em;color:#1e40af;background:#eff6ff;padding:6px 10px;border-radius:4px;"><strong>Verdict:</strong> {verdict}</div>
        </div>'''

    total = len(articles)
    flagged_count = len(flagged)
    total_matches = sum(a["match_count"] for a in articles)
    summary = (
        f"<p style='font-size:0.9em;color:#475569;'>"
        f"Scanned <strong>{total}</strong> articles | "
        f"<strong>{flagged_count}</strong> flagged | "
        f"<strong>{total_matches}</strong> total detections "
        f"(T1 anthropomorphism: {tier_counts.get(1, 0)}, "
        f"T2 magnitude: {tier_counts.get(2, 0)}, "
        f"T3 unfalsifiable: {tier_counts.get(3, 0)})</p>"
    )

    content = summary + tier_html + cat_html + phrase_html + cards_html
    return writer.section("Commentary Scanner", content,
                          hint="Click headlines to read the original article")


def build_methodology_section(writer):
    """Methodology note explaining the 3-tier system."""
    content = '''
    <p style="font-size:0.85em;color:#475569;line-height:1.6;">
    Based on <strong>Morris et al. (2007)</strong>: <em>agent</em> metaphors ("the market climbed,"
    "the market is nervous") cause investors to expect trend continuance, while <em>object</em>
    metaphors ("the market was pushed down") do not. This dashboard detects three distinct
    types of misleading financial language:
    </p>
    <div style="margin:12px 0;font-size:0.85em;">
        <div style="padding:8px 12px;margin:4px 0;border-left:3px solid #8b5cf6;background:#f5f3ff;">
            <strong style="color:#8b5cf6;">TIER 1 -- Anthropomorphism</strong> (agent metaphors):
            Words that treat the market as a person with emotions, will, or intentions
            ("skittish," "exhausted," "confused," "fighting"). These are the Morris et al. finding --
            they create a cognitive bias toward expecting the trend to continue. <em>Reality check:
            what do VIX, RSI, ADX, and breadth actually show?</em>
        </div>
        <div style="padding:8px 12px;margin:4px 0;border-left:3px solid #f97316;background:#fff7ed;">
            <strong style="color:#f97316;">TIER 2 -- Magnitude Hyperbole</strong> (object metaphors):
            Dramatic words that exaggerate the size of a move ("soaring," "plunging,"
            "freefall," "bloodbath"). These are physics/disaster metaphors, not emotions --
            they describe trajectory, not intention. <em>Reality check: what was the actual %
            move, and how does it compare to the historical distribution?</em>
        </div>
        <div style="padding:8px 12px;margin:4px 0;border-left:3px solid #6b7280;background:#f9fafb;">
            <strong style="color:#6b7280;">TIER 3 -- Unfalsifiable Narrative</strong>:
            Post-hoc explanations that sound meaningful but cannot be tested
            ("profit-taking," "bargain hunting," "pricing in"). No data can confirm
            or deny these claims. <em>Flagged as noise -- treat with skepticism.</em>
        </div>
    </div>
    <p style="font-size:0.85em;color:#475569;line-height:1.6;">
    Context filtering ensures only financial usage is flagged ("oil soaring" in a market
    headline, not "rent soaring" in a housing story). Subject extraction maps magnitude
    words to the asset being described, so verdicts compare the claimed drama to that
    specific asset's actual move.
    </p>
    '''
    return writer.section("Methodology", content,
                          hint="Morris et al., Columbia/UCLA, 2007")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("  Market Reality Check Backend v1.2")
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

    # 2. Analyze with 3-tier detection
    print("[SCAN] Detecting misleading language (3-tier)...")
    articles = analyze_articles(articles)
    flagged = [a for a in articles if a["match_count"] > 0]
    total_matches = sum(a["match_count"] for a in articles)

    tier_counts = Counter()
    for a in articles:
        for d in a.get("match_details", []):
            tier_counts[d["tier"]] += 1

    print(f"  {len(flagged)}/{len(articles)} articles flagged, {total_matches} total detections")
    print(f"  T1 anthropomorphism: {tier_counts.get(1, 0)}")
    print(f"  T2 magnitude:        {tier_counts.get(2, 0)}")
    print(f"  T3 unfalsifiable:    {tier_counts.get(3, 0)}")

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
    parts.append(writer.build_header("3-Tier Language Analysis: Agent Metaphors, Magnitude Hyperbole, Unfalsifiable Claims"))
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
        "version": "1.2",
        "reality": reality,
        "verdicts": verdicts,
        "total_articles": len(articles),
        "flagged_articles": len(flagged),
        "total_detections": total_matches,
        "tier_counts": dict(tier_counts),
        "top_phrases": Counter(
            p for a in articles for p in a.get("match_phrases", [])
        ).most_common(15),
    }
    # Remove magnitude_returns from snapshot (too large)
    if "magnitude_returns" in snapshot["reality"]:
        del snapshot["reality"]["magnitude_returns"]

    snap_path = _SCRIPT_DIR / 'market-reality' / 'market_reality_data.json'
    os.makedirs(snap_path.parent, exist_ok=True)
    with open(snap_path, 'w', encoding='utf-8') as f:
        json.dump(snapshot, f, indent=2, default=str)

    elapsed = time.time() - t0
    print(f"\n[OK] Market Reality Check v1.2 complete in {elapsed:.1f}s")
    print(f"[OK] Dashboard: market-reality/index.html")
    return snapshot


if __name__ == "__main__":
    main()
