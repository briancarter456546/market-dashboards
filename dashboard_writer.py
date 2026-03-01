# -*- coding: utf-8 -*-
# =============================================================================
# dashboard_writer.py - v1.1
# Last updated: 2026-02-18
# =============================================================================
# v1.0:
#   - Shared writer module for all static HTML dashboards
#   - Replaces Django JSON -> API -> template pipeline
#   - Handles: dated archive, index.html overwrite, CSS theme, git push
#   - Used by: spread monitor, sector rotation, macro, mirror, similar days, etc.
# =============================================================================
# Usage:
#   from dashboard_writer import DashboardWriter
#
#   writer = DashboardWriter("spread-monitor", "Intermarket Spread Monitor")
#   writer.write(html_body_string)
#
# Repo structure produced:
#   market-dashboards/
#   └── docs/
#       ├── index.html                 <- landing page (managed separately)
#       └── spread-monitor/
#           ├── index.html             <- today's dashboard (overwritten daily)
#           └── archive/
#               └── dashboard_20260218.html
# =============================================================================

import os
import shutil
import datetime
import subprocess
import sys

# =============================================================================
# CONFIG - edit these paths for your machine
# =============================================================================

# Absolute path to the separate market-dashboards git repo (for GitHub Pages)
REPO_ROOT = os.path.join(
    os.path.expanduser("~"),
    "1youtubevideopull", "market-dashboards"
)

# Dashboard HTML goes at the repo root (GitHub Pages serves from /)
DOCS_DIR = REPO_ROOT

# GitHub repo details
GITHUB_USER = "briancarter456546"
GITHUB_REPO = "market-dashboards"
GITHUB_PAGES_BASE = "https://{}.github.io/{}".format(GITHUB_USER, GITHUB_REPO)

# =============================================================================
# SHARED CSS THEME - v1.1
# Light theme - readable for traders, larger fonts, high contrast
#   - IBM Plex Sans body (sans-serif, professional)
#   - IBM Plex Mono only for numbers and tickers
#   - 17px base font (was 13px Monaco monospace)
#   - Light #f4f5f7 background, white cards
#   - Dark stat bar at top for contrast anchor
#   - Pill-shaped badges and qualifier labels
#   - Indigo #4f46e5 accent for playbook/action items
# =============================================================================

SHARED_FONT_LINK = '<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">'

SHARED_CSS = """
* { margin: 0; padding: 0; box-sizing: border-box; }

body {
    font-family: 'IBM Plex Sans', 'Segoe UI', system-ui, sans-serif;
    background: #f4f5f7;
    color: #1a1a2e;
    font-size: 17px;
    line-height: 1.55;
}

/* --- Top stat bar (dark anchor) --- */
.stat-bar {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    background: #1e1e2e;
    border-bottom: 3px solid #2d2d4a;
}

.stat-bar .stat {
    padding: 20px 26px;
    border-right: 1px solid #2d2d4a;
}
.stat-bar .stat:last-child { border-right: none; }

.stat .stat-label {
    font-size: 0.78em;
    font-weight: 600;
    color: #8888aa;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 7px;
}

.stat .stat-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2.1em;
    font-weight: 600;
    color: #f0f0f0;
    line-height: 1.1;
}

.stat .stat-value.pos     { color: #22c55e; }
.stat .stat-value.neg     { color: #ef4444; }
.stat .stat-value.warn    { color: #f59e0b; }
.stat .stat-value.neutral { color: #f0f0f0; }

/* --- Page header --- */
.page-header {
    background: #fff;
    border-bottom: 1px solid #e2e4e8;
    padding: 18px 30px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.page-header h1 {
    font-size: 1.4em;
    font-weight: 700;
    color: #1a1a2e;
}

.page-header .meta {
    font-size: 0.88em;
    color: #888;
}

.page-header .meta a { color: #4f46e5; text-decoration: none; }

/* --- Nav bar --- */
.page-nav {
    background: #fff;
    border-bottom: 1px solid #e2e4e8;
    padding: 0 30px;
    display: flex;
}

.page-nav a {
    display: inline-block;
    padding: 13px 20px;
    font-size: 0.88em;
    font-weight: 600;
    color: #666;
    text-decoration: none;
    border-bottom: 3px solid transparent;
}
.page-nav a:hover  { color: #1a1a2e; }
.page-nav a.active { color: #4f46e5; border-bottom-color: #4f46e5; }

/* --- Content area --- */
.content { padding: 26px 30px; }

/* --- Regime banner --- */
.regime-banner {
    background: #fff;
    border: 1px solid #e2e4e8;
    border-left: 6px solid #22c55e;
    border-radius: 8px;
    padding: 18px 26px;
    margin-bottom: 22px;
    display: flex;
    align-items: center;
    gap: 28px;
    flex-wrap: wrap;
}

.regime-label {
    font-size: 1.5em;
    font-weight: 700;
    letter-spacing: 0.04em;
    white-space: nowrap;
}

.regime-score {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.9em;
    color: #888;
    border-left: 1px solid #e2e4e8;
    padding-left: 28px;
    line-height: 1.8;
}

/* --- Stat/force cards --- */
.cards {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 16px;
    margin-bottom: 22px;
}

.card {
    background: #fff;
    border: 1px solid #e2e4e8;
    border-radius: 8px;
    padding: 18px 22px;
    border-top: 5px solid #ccc;
}

.card .label {
    font-size: 0.78em;
    font-weight: 700;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 10px;
}

.card .value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.9em;
    font-weight: 600;
    margin-bottom: 5px;
}

.card .sub {
    font-size: 0.85em;
    font-weight: 600;
    color: #888;
}

/* --- Table section wrapper --- */
.table-section {
    background: #fff;
    border: 1px solid #e2e4e8;
    border-radius: 8px;
    overflow: hidden;
    margin-bottom: 22px;
}

.table-section-header {
    padding: 16px 22px;
    border-bottom: 1px solid #e2e4e8;
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: #fafafa;
}

.table-section-header h2 {
    font-size: 0.95em;
    font-weight: 700;
    color: #1a1a2e;
    text-transform: uppercase;
    letter-spacing: 0.07em;
}

/* --- Tables --- */
table {
    border-collapse: collapse;
    width: 100%;
    font-size: 0.95em;
}

thead th {
    background: #f8f9fb;
    color: #555;
    padding: 13px 16px;
    text-align: left;
    border-bottom: 2px solid #e2e4e8;
    font-size: 0.82em;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    white-space: nowrap;
    cursor: pointer;
    user-select: none;
}

thead th:hover { background: #f0f2f5; color: #1a1a2e; }
thead th.sorted-asc  { color: #16a34a; }
thead th.sorted-desc { color: #dc2626; }

tbody td {
    padding: 13px 16px;
    border-bottom: 1px solid #f0f0f0;
    vertical-align: middle;
}

tbody tr:last-child td { border-bottom: none; }
tbody tr:hover { background: #fafbfd; }

/* Numbers and tickers use mono font */
.num {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.92em;
}

.ticker {
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    font-size: 1.05em;
    color: #1a1a2e;
}

/* Color classes */
.pos   { color: #16a34a; font-weight: 600; }
.neg   { color: #dc2626; font-weight: 600; }
.warn  { color: #d97706; font-weight: 600; }
.muted { color: #aaa; }
.accent { color: #4f46e5; }

/* Score badges - pill shaped */
.badge {
    display: inline-block;
    padding: 5px 14px;
    border-radius: 20px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.9em;
    font-weight: 700;
    text-align: center;
    min-width: 52px;
}

.badge-2  { background: #dcfce7; color: #15803d; border: 1px solid #bbf7d0; }
.badge-1  { background: #f0fdf4; color: #16a34a; border: 1px solid #d1fae5; }
.badge-n1 { background: #fef2f2; color: #dc2626; border: 1px solid #fecaca; }
.badge-n2 { background: #fee2e2; color: #b91c1c; border: 1px solid #fca5a5; }

/* Qualifier pills */
.qualifier {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.8em;
    font-weight: 700;
    letter-spacing: 0.04em;
    margin-bottom: 4px;
}

.q-confirmed { background: #dcfce7; color: #15803d; }
.q-fading    { background: #fff7ed; color: #c2410c; }
.q-diverging { background: #eff6ff; color: #1d4ed8; }
.q-holding   { background: #f1f5f9; color: #64748b; }

/* Trend pills */
.trend-above {
    background: #dcfce7; color: #15803d;
    padding: 4px 12px; border-radius: 4px;
    font-size: 0.85em; font-weight: 700;
    white-space: nowrap;
}
.trend-below {
    background: #fee2e2; color: #b91c1c;
    padding: 4px 12px; border-radius: 4px;
    font-size: 0.85em; font-weight: 700;
    white-space: nowrap;
}

/* --- Playbook text blocks --- */
.pb-header {
    font-size: 0.85em;
    font-weight: 700;
    color: #1a1a2e;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin: 18px 0 8px;
    padding-bottom: 6px;
    border-bottom: 1px solid #e2e4e8;
}
.pb-header:first-child { margin-top: 0; }

.pb-body {
    font-size: 0.95em;
    color: #333;
    line-height: 1.7;
    margin: 4px 0;
}

.pb-item {
    font-size: 0.9em;
    color: #4f46e5;
    margin: 5px 0 5px 16px;
    border-left: 3px solid #c7d2fe;
    padding-left: 12px;
    line-height: 1.65;
}

/* --- Footer --- */
.dash-footer {
    padding: 22px 30px;
    font-size: 0.82em;
    color: #aaa;
    border-top: 1px solid #e2e4e8;
    text-align: center;
    background: #fff;
    margin-top: 8px;
}

/* ── Mobile responsive ── */
@media (max-width: 768px) {
    body { font-size: 15px; }
    .stat-bar { grid-template-columns: repeat(2, 1fr); }
    .stat-bar .stat { padding: 14px 16px; }
    .stat .stat-value { font-size: 1.5em; }
    .stat .stat-label { font-size: 0.72em; }
    .page-header { flex-direction: column; align-items: flex-start; padding: 14px 16px; gap: 6px; }
    .page-header h1 { font-size: 1.2em; }
    .page-nav { padding: 0 12px; overflow-x: auto; -webkit-overflow-scrolling: touch; }
    .page-nav a { padding: 10px 14px; font-size: 0.82em; white-space: nowrap; }
    .content { padding: 16px 12px; }
    .regime-banner { flex-direction: column; padding: 14px 16px; gap: 12px; }
    .regime-label { font-size: 1.2em; }
    .regime-score { border-left: none; padding-left: 0; border-top: 1px solid #e2e4e8; padding-top: 10px; }
    .cards { grid-template-columns: 1fr; gap: 12px; }
    .card { padding: 14px 16px; }
    .card .value { font-size: 1.5em; }
    .table-section { border-radius: 0; margin-left: -12px; margin-right: -12px; }
    .table-section-header { padding: 12px 16px; }
    table { font-size: 0.85em; }
    thead th { padding: 8px 10px; font-size: 0.75em; }
    tbody td { padding: 8px 10px; }
    .badge { padding: 3px 10px; font-size: 0.82em; min-width: 44px; }
    .qualifier { padding: 3px 9px; font-size: 0.75em; }
    .pb-item { margin-left: 8px; padding-left: 10px; }
    .dash-footer { padding: 16px 12px; font-size: 0.78em; }
}
"""

# Gradient color helper (JS) - red to green, matches Django guide
GRADIENT_JS = """
function getGradientColor(score) {
    var clamped = Math.max(0, Math.min(1, score));
    var r, g, b;
    if (clamped < 0.25) {
        var t = clamped * 4;
        r = Math.round(211 + (255 - 211) * t);
        g = Math.round(47 + (152 - 47) * t);
        b = Math.round(47 + (0 - 47) * t);
    } else if (clamped < 0.5) {
        var t = (clamped - 0.25) * 4;
        r = 255;
        g = Math.round(152 + (235 - 152) * t);
        b = Math.round(0 + (59 - 0) * t);
    } else if (clamped < 0.75) {
        var t = (clamped - 0.5) * 4;
        r = Math.round(255 + (139 - 255) * t);
        g = Math.round(235 + (195 - 235) * t);
        b = Math.round(59 + (74 - 59) * t);
    } else {
        var t = (clamped - 0.75) * 4;
        r = Math.round(139 + (0 - 139) * t);
        g = Math.round(195 + (200 - 195) * t);
        b = Math.round(74 + (83 - 74) * t);
    }
    return 'rgb(' + r + ',' + g + ',' + b + ')';
}
"""


# =============================================================================
# DASHBOARD WRITER CLASS
# =============================================================================

class DashboardWriter(object):
    """
    Writes a static HTML dashboard to the market-dashboards repo.

    Parameters
    ----------
    slug : str
        URL-safe folder name, e.g. "spread-monitor"
    title : str
        Human-readable dashboard title, e.g. "Intermarket Spread Monitor"
    """

    def __init__(self, slug, title):
        self.slug = slug
        self.title = title
        self.date_str = datetime.date.today().strftime("%Y-%m-%d")
        self.date_compact = datetime.date.today().strftime("%Y%m%d")

        self.dash_dir = os.path.join(DOCS_DIR, slug)
        self.archive_dir = os.path.join(self.dash_dir, "archive")
        self.index_path = os.path.join(self.dash_dir, "index.html")
        self.archive_path = os.path.join(
            self.archive_dir,
            "dashboard_{}.html".format(self.date_compact)
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def write(self, body_html, extra_css="", extra_js=""):
        """
        Build and write the full HTML page.

        Parameters
        ----------
        body_html : str
            Everything that goes inside <body> after the standard header.
            Use build_header(), section(), table_html() helpers below.
        extra_css : str
            Dashboard-specific CSS (appended after shared theme).
        extra_js : str
            Dashboard-specific JS (appended after gradient helper).
        """
        html = self._build_page(body_html, extra_css, extra_js)
        self._ensure_dirs()
        self._write_file(self.archive_path, html)
        self._write_file(self.index_path, html)
        print("Written: {}".format(self.index_path))
        print("Archive: {}".format(self.archive_path))

    # ------------------------------------------------------------------
    # HTML building helpers (optional - backends can build their own body)
    # ------------------------------------------------------------------

    def build_header(self, subtitle=""):
        """
        Standard page header + nav bar.
        Produces: page-header div + page-nav div.
        subtitle goes into the meta line (e.g. "SMA 50 | Slope 10d").
        """
        lines = []
        lines.append('<div class="page-header">')
        lines.append('  <h1>{}</h1>'.format(self.title))
        lines.append('  <div class="meta">')
        if subtitle:
            lines.append('    {} &nbsp;|&nbsp;'.format(subtitle))
        lines.append('    <a href="{}">market-dashboards</a>'.format(GITHUB_PAGES_BASE))
        lines.append('  </div>')
        lines.append('</div>')
        lines.append('<div class="page-nav">')
        lines.append('  <a href="#spreads" class="active">Dashboard</a>')
        lines.append('  <a href="archive/dashboard_{}.html">Archive: {}</a>'.format(
            self.date_compact, self.date_str))
        lines.append('  <a href="{}">Home</a>'.format(GITHUB_PAGES_BASE))
        lines.append('</div>')
        lines.append('<div class="content">')
        return "\n".join(lines)

    def section(self, title, content_html, hint=""):
        """
        Wrap content in a white table-section card with a header bar.
        hint: optional small grey text on the right of the header (e.g. "Click to sort").
        """
        hint_html = (
            '<span style="font-size:0.85em;color:#888;">{}</span>'.format(hint)
            if hint else ""
        )
        return (
            '<div class="table-section">'
            '<div class="table-section-header">'
            '<h2>{}</h2>'.format(title) +
            hint_html +
            '</div>'
            '<div style="padding:20px 24px;">' +
            content_html +
            '</div>'
            '</div>'
        )

    def stat_bar(self, stats):
        """
        Dark top stat bar.
        stats: list of (label, value, css_class) tuples.
        css_class one of: pos, neg, warn, neutral
        Example: [("Bullish", "11", "pos"), ("Bearish", "3", "neg")]
        """
        items = []
        for label, value, css_class in stats:
            items.append(
                '<div class="stat">'
                '<div class="stat-label">{}</div>'
                '<div class="stat-value {}">{}</div>'
                '</div>'.format(label, css_class, value)
            )
        return '<div class="stat-bar">{}</div>'.format("".join(items))

    def regime_banner(self, label, score_html, color="#22c55e"):
        """
        Colored left-border regime banner below the stat bar.
        label: e.g. "LEANING RISK-ON"
        score_html: e.g. "Score: 14 / 36 | Normalized: +0.39"
        color: left border + label color
        """
        return (
            '<div class="regime-banner" style="border-left-color:{color};">'
            '<div class="regime-label" style="color:{color};">{label}</div>'
            '<div class="regime-score">{score}</div>'
            '</div>'.format(color=color, label=label, score=score_html)
        )

    def footer(self):
        """Close the .content div and render footer."""
        return (
            '</div>'  # close .content
            '<div class="dash-footer">'
            '{} &mdash; Generated {} &mdash; For personal research only.'
            '</div>'.format(GITHUB_PAGES_BASE, self.date_str)
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_page(self, body_html, extra_css, extra_js):
        parts = []
        parts.append('<!DOCTYPE html>')
        parts.append('<html lang="en"><head>')
        parts.append('<meta charset="utf-8">')
        parts.append('<meta name="viewport" content="width=device-width, initial-scale=1.0">')
        parts.append('<title>{} - {}</title>'.format(self.title, self.date_str))
        parts.append(SHARED_FONT_LINK)
        parts.append('<style>')
        parts.append(SHARED_CSS)
        if extra_css:
            parts.append(extra_css)
        parts.append('</style>')
        parts.append('</head><body>')
        parts.append(body_html)
        parts.append('<script>')
        parts.append(GRADIENT_JS)
        if extra_js:
            parts.append(extra_js)
        parts.append('</script>')
        parts.append('</body></html>')
        return "\n".join(parts)

    def _ensure_dirs(self):
        os.makedirs(self.dash_dir, exist_ok=True)
        os.makedirs(self.archive_dir, exist_ok=True)

    def _write_file(self, path, content):
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)


# =============================================================================
# LANDING PAGE WRITER
# Call this once after all dashboards have been written to regenerate index.html
# =============================================================================

# Registry of all live dashboards.
# Add a new entry here whenever a new dashboard is converted and confirmed working.
# Keys: slug (URL path), title, description, icon (emoji), color (accent hex), status
DASHBOARD_REGISTRY = [
    {
        "slug":        "spread-monitor",
        "title":       "Intermarket Spread Monitor",
        "description": "18 cross-asset spreads across rates, credit, equity, and commodities. "
                       "Regime scoring (risk-on/off), force breakdown by category, and a market playbook.",
        "icon":        "📡",
        "color":       "#4a9eff",
        "tag":         "Macro",
    },
    {
        "slug":        "sector-rotation",
        "title":       "Sector Rotation Deep Dive",
        "description": "Momentum z-scores + forward predictions for 500+ ETFs across 25Y / 5Y / 1Y "
                       "windows. Historical agreement scores and pattern-matched outcomes.",
        "icon":        "🔄",
        "color":       "#8b5cf6",
        "tag":         "Sectors",
    },
    {
        "slug":        "momentum-ranker",
        "title":       "Momentum Ranker",
        "description": "700+ tickers ranked by composite momentum score. "
                       "Returns across 6 periods, SMA gate, ratio consistency, "
                       "and bad-SPY-day resilience. Live filter and sort.",
        "icon":        "📈",
        "color":       "#10b981",
        "tag":         "Momentum",
    },
    {
        "slug":        "similar-days",
        "title":       "Similar Days Analyzer",
        "description": "Pattern-matched historical days with similar momentum profiles to today. "
                       "Before vs after analysis for sectors, indexes, style factors, and risk regime "
                       "across 25Y / 5Y / 1Y windows.",
        "icon":        "🔍",
        "color":       "#0ea5e9",
        "tag":         "Pattern",
    },
    {
        "slug":        "historical-mirror",
        "title":       "Historical Mirror",
        "description": "25-year x 25-period similarity heatmap comparing today's market fingerprint "
                       "against every historical bi-weekly period. Momentum correlation, spread state "
                       "matching, top analog periods, and forward outcome distributions.",
        "icon":        "🪞",
        "color":       "#06b6d4",
        "tag":         "Pattern",
    },
    {
        "slug":        "stock-secrot",
        "title":       "Stock Sector Rotation",
        "description": "Individual stocks scored across 9 momentum and sector rotation patterns. "
                       "High scorers (7/9+) with returns, RSI, ADX, and relative strength.",
        "icon":        "🏭",
        "color":       "#f59e0b",
        "tag":         "Stocks",
    },
    {
        "slug":        "hyglqd-credit",
        "title":       "HYG/LQD Credit Spread",
        "description": "Credit spread regime via HYG/LQD ratio. Historical percentile, "
                       "expected SPY forward returns at current spread level, and win rates.",
        "icon":        "💳",
        "color":       "#ec4899",
        "tag":         "Credit",
    },
    {
        "slug":        "crash-detection",
        "title":       "Crash Detection (RMT + Ising)",
        "description": "Random Matrix Theory eigenvalue analysis and Ising magnetization. "
                       "Composite crash probability score with component breakdown.",
        "icon":        "💥",
        "color":       "#ef4444",
        "tag":         "Risk",
    },
    {
        "slug":        "advanced-momentum",
        "title":       "Advanced Momentum Analyzer",
        "description": "OBV divergence, Sortino ratios, slope acceleration, and trajectory detection "
                       "for 700+ assets. Signals from STRONG_BUY to SELL with confidence scores.",
        "icon":        "🚀",
        "color":       "#6366f1",
        "tag":         "Momentum",
    },
    {
        "slug":        "conservative-momentum",
        "title":       "Conservative Momentum Qualifier",
        "description": "Pattern-based momentum qualification: trend persistence, dip structure, "
                       "sustainability scoring. Safe plays filtered for 40-300% 12M returns.",
        "icon":        "🛡️",
        "color":       "#0d9488",
        "tag":         "Momentum",
    },
    {
        "slug":        "macro",
        "title":       "Macro Dashboard",
        "description": "VIX, treasury yield curve, credit spreads, sector performance, commodities, "
                       "currencies, and breadth. Overall regime classification with 5-day trends and alerts.",
        "icon":        "🌍",
        "color":       "#7c3aed",
        "tag":         "Macro",
    },
]


def write_landing_page():
    """Write docs/index.html - the main navigation page."""
    os.makedirs(DOCS_DIR, exist_ok=True)
    date_str = datetime.date.today().strftime("%Y-%m-%d")

    cards_html = []
    for d in DASHBOARD_REGISTRY:
        url   = "./{}/index.html".format(d["slug"])
        color = d["color"]
        tag   = d.get("tag", "")
        tag_html = '<span class="dash-tag">{}</span>'.format(tag) if tag else ""
        card = (
            '<a href="{url}" class="dash-card" style="--accent:{color};">'
            '<div class="dash-card-top">'
            '<span class="dash-icon">{icon}</span>'
            '{tag_html}'
            '</div>'
            '<div class="dash-card-title">{title}</div>'
            '<div class="dash-card-desc">{description}</div>'
            '<div class="dash-card-cta">Open dashboard &rarr;</div>'
            '</a>'
        ).format(
            url=url, color=color, icon=d["icon"],
            tag_html=tag_html, title=d["title"], description=d["description"]
        )
        cards_html.append(card)

    n = len(DASHBOARD_REGISTRY)
    html = """<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Market Dashboards — Brian Carter</title>
{font_link}
<style>
*, *::before, *::after {{ margin:0; padding:0; box-sizing:border-box; }}
body {{
    font-family: 'IBM Plex Sans', 'Segoe UI', system-ui, sans-serif;
    background: #f4f5f7;
    color: #1a1a2e;
    min-height: 100vh;
}}

/* ── Header ── */
.header {{
    background: #12121e;
    padding: 36px 40px 32px;
    border-bottom: 1px solid #2a2a3e;
}}
.header-inner {{
    max-width: 960px;
    margin: 0 auto;
}}
.header h1 {{
    font-size: 1.75em;
    font-weight: 700;
    color: #f0f0f8;
    letter-spacing: -0.02em;
    margin-bottom: 6px;
}}
.header-meta {{
    display: flex;
    gap: 24px;
    align-items: center;
    flex-wrap: wrap;
}}
.header-meta .date {{
    font-size: 0.85em;
    color: #666688;
}}
.header-meta .count {{
    font-size: 0.82em;
    background: #1e1e3a;
    color: #8888cc;
    padding: 3px 10px;
    border-radius: 20px;
    border: 1px solid #2a2a4a;
}}
.header-meta .gh-link {{
    font-size: 0.82em;
    color: #5588ff;
    text-decoration: none;
    margin-left: auto;
}}
.header-meta .gh-link:hover {{ text-decoration: underline; }}

/* ── Content ── */
.content {{
    max-width: 960px;
    margin: 0 auto;
    padding: 32px 24px 60px;
}}
.section-label {{
    font-size: 0.75em;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #999;
    margin-bottom: 14px;
}}

/* ── Card grid ── */
.grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(290px, 1fr));
    gap: 18px;
    margin-bottom: 48px;
}}
.dash-card {{
    background: #fff;
    border: 1px solid #e2e4e8;
    border-top: 3px solid var(--accent);
    border-radius: 10px;
    padding: 20px 22px 18px;
    text-decoration: none;
    display: flex;
    flex-direction: column;
    gap: 10px;
    transition: box-shadow 0.15s, transform 0.12s;
    cursor: pointer;
}}
.dash-card:hover {{
    box-shadow: 0 6px 24px rgba(0,0,0,0.09);
    transform: translateY(-3px);
}}
.dash-card-top {{
    display: flex;
    justify-content: space-between;
    align-items: center;
}}
.dash-icon {{
    font-size: 1.5em;
    line-height: 1;
}}
.dash-tag {{
    font-size: 0.72em;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    background: color-mix(in srgb, var(--accent) 12%, transparent);
    color: var(--accent);
    padding: 3px 9px;
    border-radius: 20px;
    border: 1px solid color-mix(in srgb, var(--accent) 25%, transparent);
}}
.dash-card-title {{
    font-size: 1.05em;
    font-weight: 700;
    color: #1a1a2e;
    line-height: 1.2;
}}
.dash-card-desc {{
    font-size: 0.83em;
    color: #555;
    line-height: 1.6;
    flex: 1;
}}
.dash-card-cta {{
    font-size: 0.8em;
    font-weight: 600;
    color: var(--accent);
    margin-top: 4px;
}}

/* ── Coming soon placeholder cards ── */
.coming-soon {{
    background: #fafafa;
    border: 1px dashed #d0d0d8;
    border-top: 3px solid #d0d0d8;
    border-radius: 10px;
    padding: 20px 22px 18px;
    display: flex;
    flex-direction: column;
    gap: 10px;
    opacity: 0.65;
}}
.coming-soon .dash-tag {{
    background: #f0f0f5;
    color: #999;
    border-color: #ddd;
}}
.coming-soon .dash-card-title {{ color: #888; }}
.coming-soon .dash-card-desc  {{ color: #aaa; }}
.coming-soon .cs-badge {{
    font-size: 0.75em;
    font-weight: 600;
    color: #bbb;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}}

/* ── Footer ── */
.footer {{
    border-top: 1px solid #e2e4e8;
    padding-top: 20px;
    font-size: 0.8em;
    color: #bbb;
    text-align: center;
}}

/* ── Mobile responsive ── */
@media (max-width: 768px) {{
    .header {{ padding: 24px 16px 20px; }}
    .header h1 {{ font-size: 1.35em; }}
    .header-meta .gh-link {{ display: none; }}
    .content {{ padding: 20px 12px 40px; }}
    .grid {{ grid-template-columns: 1fr; gap: 12px; }}
    .dash-card {{ padding: 16px 16px 14px; }}
    .dash-card-title {{ font-size: 0.95em; }}
    .dash-card-desc {{ font-size: 0.8em; }}
}}
</style>
</head><body>

<div class="header">
  <div class="header-inner">
    <h1>Market Dashboards</h1>
    <div class="header-meta">
      <span class="date">Updated {date}</span>
      <span class="count">{n} live dashboards</span>
      <a href="{base_url}" class="gh-link">{base_url} &rarr;</a>
    </div>
  </div>
</div>

<div class="content">
  <div class="section-label">Live Dashboards</div>
  <div class="grid">
    {cards}
  </div>

  <div class="footer">{base_url} &mdash; Generated {date} &mdash; For personal research only.</div>
</div>
</body></html>""".format(
        font_link=SHARED_FONT_LINK,
        date=date_str,
        n=n,
        cards="\n    ".join(cards_html),
        base_url=GITHUB_PAGES_BASE,
    )

    index_path = os.path.join(DOCS_DIR, "index.html")
    with open(index_path, "w", encoding="utf-8") as f:
        f.write(html)
    print("Landing page: {}".format(index_path))


# =============================================================================
# GIT PUSH
# Call this once at the end of run_daily.py after all dashboards are written
# =============================================================================

def push_to_github(commit_message=None):
    """
    Stage all changes in docs/, commit, and push to GitHub.
    Equivalent to replacing copy_to_django() in run_daily.py.

    Returns True on success, False on failure.
    """
    if commit_message is None:
        commit_message = "Daily update {}".format(
            datetime.date.today().strftime("%Y-%m-%d")
        )

    print("=" * 60)
    print("PUSHING TO GITHUB PAGES")
    print("=" * 60)

    steps = [
        (["git", "-C", REPO_ROOT, "add", "."], "git add .", 30),
        (["git", "-C", REPO_ROOT, "commit", "-m", commit_message], "git commit", 30),
        (["git", "-C", REPO_ROOT, "push"], "git push", 120),
    ]

    for cmd, label, timeout_sec in steps:
        print("  Running: {} (timeout={}s)".format(label, timeout_sec))
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_sec,
            )
        except subprocess.TimeoutExpired:
            print("  TIMEOUT: '{}' hung for {}s -- killed.".format(label, timeout_sec))
            print("  This usually means git is waiting for credentials or the network is down.")
            print("  Try running 'git push' manually in a terminal to diagnose.")
            return False
        if result.returncode != 0:
            # git commit fails when there's nothing new -- not a real error.
            # Continue to git push so any prior unpushed commits still get pushed.
            combined = result.stdout + result.stderr
            if label == "git commit" and ("nothing to commit" in combined or
                                          "nothing added to commit" in combined or
                                          "no changes added to commit" in combined or
                                          "Changes not staged" in combined):
                print("  Nothing new to commit -- continuing to push.")
                continue
            print("  FAILED: {} (exit code {})".format(label, result.returncode))
            if result.stderr.strip():
                print("  stderr: {}".format(result.stderr.strip()[:300]))
            if result.stdout.strip():
                print("  stdout: {}".format(result.stdout.strip()[:300]))
            return False
        else:
            print("  OK: {}".format(result.stdout.strip()[:120]))

    print("  GitHub Pages updated: {}".format(GITHUB_PAGES_BASE))
    return True


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == "__main__":
    print("dashboard_writer.py v1.0")
    print("REPO_ROOT: {}".format(REPO_ROOT))
    print("DOCS_DIR:  {}".format(DOCS_DIR))
    print("PAGES URL: {}".format(GITHUB_PAGES_BASE))
    print()

    # Quick test - writes a minimal dashboard
    writer = DashboardWriter("test-dashboard", "Test Dashboard")
    body = writer.build_header("Self-test output")
    body += writer.section("Status", "<p class='pos'>dashboard_writer.py is working correctly.</p>")
    body += writer.footer()
    writer.write(body)
    print()
    print("Test complete. Check: {}".format(writer.index_path))
