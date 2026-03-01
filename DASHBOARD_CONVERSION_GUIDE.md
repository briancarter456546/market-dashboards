# Dashboard Conversion Guide
**Project:** Brian Carter's market dashboards — migrating from Django to static GitHub Pages  
**Last updated:** 2026-02-18  
**Read this at the start of any new conversion session.**

---

## The Big Picture

Brian runs quantitative trading dashboards built in Python. They used to write JSON → Django served HTML. We're migrating them to **static HTML files** pushed to GitHub Pages. All compute logic stays 100% unchanged. We only change the output: instead of writing JSON, each backend writes HTML via `dashboard_writer.py`.

**Repo:** `github.com/briancarter456546/market-dashboards`  
**GitHub Pages URL:** `https://briancarter456546.github.io/market-dashboards`  
**Local repo path:** `C:\Users\lynda\1youtubevideopull\perplexity-user-data\market-dashboards`  
**Scripts live in:** `market-dashboards\` (the repo root)  
**Data lives in:** `perplexity-user-data\` (one level up from scripts)

---

## File Structure

```
perplexity-user-data/
├── price_cache/              ← 700+ .pkl files, FMP API data
├── cache_loader_v1_0.py      ← shared loader used by sector rotation etc
├── output/                   ← local CSV archives (not git)
│   └── spread_monitor/
└── market-dashboards/        ← the GitHub repo
    ├── dashboard_writer.py   ← SHARED MODULE - every backend imports this
    ├── intermarket_spread_monitor_v250.py   ✅ converted
    ├── sector_rotation_v0_5y.py             ✅ converted
    ├── sector_rotation_v0_7.py              ⏳ pending
    ├── mirror_backend.py                    ⏳ pending
    ├── macro_backend_v3.1.py                ⏳ pending
    ├── similar_days_analyzer_v1_12.py       ⏳ pending
    ├── hyglqd_backend_v1.0.py               ⏳ pending
    ├── momentum_qualifier_retstabconjson3.py ⏳ pending
    ├── stock_secrot_backend.py              ⏳ pending
    ├── advanced_momentum_analyzer_v3.3.py   ⏳ pending
    ├── momentum_ranker_v1_0.py              ⏳ pending
    ├── crash_detection_backend.py           ⏳ pending
    └── docs/                 ← what GitHub Pages serves
        ├── index.html        ← landing page (written by write_landing_page())
        ├── spread-monitor/
        │   ├── index.html
        │   └── archive/
        └── sector-rotation/
            ├── index.html
            └── archive/
```

---

## dashboard_writer.py — The Shared Module

**Current version:** v1.1  
Every backend imports this. Never modify it without telling Brian.

### Key exports

```python
from dashboard_writer import DashboardWriter, write_landing_page, push_to_github

# REPO_ROOT resolves to perplexity-user-data/market-dashboards
# DOCS_DIR   resolves to perplexity-user-data/market-dashboards/docs
# GITHUB_PAGES_BASE = "https://briancarter456546.github.io/market-dashboards"
```

### DashboardWriter usage

```python
writer = DashboardWriter("spread-monitor", "Intermarket Spread Monitor")
# slug = URL-safe folder name → docs/spread-monitor/
# title = human-readable title

body = build_body_html(...)   # your function returns HTML string

writer.write(body, extra_css=EXTRA_CSS, extra_js=SORT_JS)
# Writes:  docs/spread-monitor/index.html         (overwritten daily)
# Archive: docs/spread-monitor/archive/dashboard_YYYYMMDD.html
```

### Helper methods on writer

```python
writer.stat_bar([                          # dark top bar with big numbers
    ("Label", "value", "pos|neg|warn|neutral"),
    ...
])

writer.build_header("subtitle text")       # page-header + nav + opens .content div
                                           # MUST be paired with writer.footer()

writer.regime_banner(label, score_html, color="#22c55e")   # colored left-border banner

writer.section("Title", content_html, hint="optional grey text right side")
# Wraps in white card with header bar

writer.footer()                            # closes .content div + renders footer
```

### SHARED_CSS classes available to all backends

```
body                    17px IBM Plex Sans
.num / .ticker          IBM Plex Mono font
.pos / .neg / .warn / .muted / .accent   color utilities
.badge / .badge-2 / .badge-1 / .badge-n1 / .badge-n2   pill badges (green→red)
.qualifier + .q-confirmed / .q-fading / .q-diverging / .q-holding
.trend-above / .trend-below              green/red pill
.stat-bar / .stat / .stat-label / .stat-value   dark top bar
.page-header / .page-nav                 header + nav
.content                                 main content padding wrapper
.regime-banner / .regime-label / .regime-score
.cards / .card / .card .label / .card .value / .card .sub   force/stat cards
.table-section / .table-section-header  white card wrapper
table thead th                           sortable header style
.sorted-asc / .sorted-desc              sort indicator colors
.pb-header / .pb-body / .pb-item        playbook text blocks
.dash-footer                            footer bar
```

### GRADIENT_JS

`GRADIENT_JS` is a JS string (red→orange→yellow→green) available for use in extra_js.  
`getGradientColor(norm01)` — takes 0-1, returns `rgb(r,g,b)`.

### SORT_JS (reusable)

Standard sortable table JS. Works on any `<table class="sortable-table">`.  
Reads `data-sort` attribute from each `<td>`. Auto-detects numeric vs string.  
**Important:** if the table has TWO header rows (group row + column row), use a custom  
SORT_JS that targets `thead tr:nth-child(2)` not the first row. See sector rotation example.

---

## Conversion Pattern

### For most backends (Python-colored cells)

1. **Keep all compute logic 100% unchanged** — no touching calculations, scoring, etc.
2. Add `_SCRIPT_DIR` / `_DATA_DIR` path fix (see below)
3. Remove JSON write at end of `__main__`
4. Add `build_body_html(data, writer)` function that returns an HTML body string
5. Add `EXTRA_CSS` string for dashboard-specific styles
6. Add `SORT_JS` (copy from spread monitor or write custom if dual header rows)
7. Replace `__main__` block to call `DashboardWriter` and `writer.write()`
8. Add slug to `DASHBOARD_REGISTRY` in `dashboard_writer.py`

### For data-heavy tables with per-column color gradients (JS-rendered)

Use this pattern when the original Django template colored cells using JS gradient logic  
(like sector rotation). Instead of Python generating colored `<td>` tags:

1. Serialize `dashboard` dict to JSON: `json.dumps(dashboard)`
2. Embed as `const DATA = {...}` in a JS block
3. Write `calculateColumnRanges()` using **5th-95th percentile** per column (not absolute min/max)
4. Write `renderTable()` in JS — builds the entire `<tbody>` client-side
5. Python only writes a `<div id="xyz-table-wrap">` placeholder
6. Pass the render JS as `extra_js` to `writer.write()`

See `sector_rotation_v0_5y.py` for the full working example.

---

## Critical: Path Fixes for Every Backend

Scripts run from `market-dashboards\` but data is in `perplexity-user-data\` (parent dir).  
Add this near the top of every converted backend, before any file path usage:

```python
import os, sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR   = os.path.normpath(os.path.join(_SCRIPT_DIR, ".."))  # perplexity-user-data
```

**For backends that use price_cache directly:**
```python
CONFIG = {
    "cache_dir":      os.path.normpath(os.path.join(_DATA_DIR, "price_cache")),
    "csv_output_dir": os.path.normpath(os.path.join(_DATA_DIR, "output", "your_backend")),
    ...
}
```

**For backends that import cache_loader_v1_0:**
```python
if _DATA_DIR not in sys.path:
    sys.path.insert(0, _DATA_DIR)

from cache_loader_v1_0 import load_etf_data, list_all_clusters
```

**dashboard_writer.py itself** uses `expanduser("~")` + the subdirectory path,  
so `REPO_ROOT` resolves to the correct absolute path automatically.

---

## Design System

### Light theme (v1.1)
- Background: `#f4f5f7` (page), `#fff` (cards)
- Body font: IBM Plex Sans, **17px base** (trader-readable, not developer-small)
- Data font: IBM Plex Mono (numbers, tickers only — not body text)
- Dark stat bar at top: `#1e1e2e` background — contrast anchor
- Accent: indigo `#4f46e5` for playbook/action items
- Positive: `#16a34a` green | Negative: `#dc2626` red | Warning: `#d97706` amber

### Typography scale
- Stat bar values: 2.1em Mono
- Page h1: 1.4em Sans Bold
- Section headers: 0.95em Sans Bold uppercase
- Table headers: 0.82em Sans Bold uppercase
- Table body: 0.95em (inherits body)
- Numbers in cells: 0.92em Mono

### Color-coded badges (score/signal)
```
.badge-2  → green  (strong bullish)
.badge-1  → light green
.badge-n1 → light red
.badge-n2 → red (strong bearish)
```

### Force dot colors (spread monitor / sector rotation)
```
Gravity (I: Rates)        → #8b5cf6  purple
Electromagnetism (II: Fundamentals) → #f59e0b  amber
Strong Force (III: Liquidity)       → #0ea5e9  sky blue
Weak Force (IV: Sentiment)          → #f87171  red
```

### Group header rows (wide tables like sector rotation)
```css
.grp-blank    { background: #f8f9fb; }
.grp-momentum { background: #1e293b; color: #94a3b8; }
.grp-25y      { background: #14532d; color: #86efac; }
.grp-5y       { background: #1e3a5f; color: #93c5fd; }
.grp-1y       { background: #3b1f5e; color: #c4b5fd; }
```

---

## Per-Column Gradient Coloring (JS)

For tables where each column needs its own color scale:

```javascript
function calculateColumnRanges() {
    // Collect all values per column
    // Use 5th-95th PERCENTILE (not min/max) to prevent outliers washing out color
    const sorted = [...values].sort((a, b) => a - b);
    columnRanges[key] = {
        min: sorted[Math.floor(sorted.length * 0.05)],
        max: sorted[Math.floor(sorted.length * 0.95)]
    };
}

function normalizeValue(val, columnKey) {
    const range = columnRanges[columnKey];
    if (!range || range.max === range.min) return 0.5;
    return (val - range.min) / (range.max - range.min);
}

// Then call getGradientColor(normalizeValue(val, key)) for background
// Use textColor(bgRgb) to pick black or white text based on luminance
```

---

## DASHBOARD_REGISTRY in dashboard_writer.py

Each converted backend needs an entry. Update when adding a new dashboard:

```python
DASHBOARD_REGISTRY = [
    {
        "slug": "spread-monitor",          # matches DashboardWriter slug
        "title": "Intermarket Spread Monitor",
        "description": "...",
        "icon": "I",                       # single letter shown in landing page card
        "color": "#4a9eff",               # card accent color
    },
    {
        "slug": "sector-rotation",
        "title": "SecRot Deep Dive",
        "description": "Pattern matching across 25Y/5Y/1Y history...",
        "icon": "S",
        "color": "#8b5cf6",
    },
    # add new entries here as each backend is converted
]
```

---

## Converted Backends Reference

### intermarket_spread_monitor_v250.py ✅ v2.5.1
- **Slug:** `spread-monitor`
- **Data source:** `price_cache/` PKL files loaded directly
- **Pattern:** Python-colored cells, `data-sort` attributes, `sortable-table` class
- **Key outputs:** stat bar (spreads/bullish/bearish/regime score), regime banner,  
  4 force cards, sortable spread table (11 cols), playbook section, copy box
- **EXTRA_CSS:** force dot colors only
- **SORT_JS:** standard single-header-row version

### sector_rotation_v0_5y.py ✅ v0.5y
- **Slug:** `sector-rotation`
- **Data source:** `cache_loader_v1_0.py` + `price_cache/` (sys.path patched)
- **Pattern:** JS-rendered table, data embedded as `const DATA = {...}`
- **Key outputs:** stat bar, page header, legend section, JS-rendered table  
  (ETF / Name / Score / Hist Agree / 5D Ret / Mom 3M-1M-10D-5D / Pred 25Y-5Y-1Y × 3d-5d-10d)
- **Gradient:** per-column 5th-95th percentile, red→green, black/white text by luminance
- **SORT_JS:** custom dual-header version (targets 2nd thead row)
- **Note:** v0.5 (25Y/5Y/1Y windows) preferred over v0.7 (1Y only)

---

## What NOT to Change

- Any calculation, scoring, signal logic, or data processing
- Column definitions in SPREADS list (spread monitor)
- ETF_CLUSTERS, ETF_NAMES (sector rotation)
- The `cache_loader_v1_0` interface
- Versioning — increment post-decimal (e.g. v2.5.1 → v2.5.2) for any change,  
  Brian decides when to bump major version

---

## run_daily.py

Orchestrates all backends. When a new backend is converted:
1. Add it to the `BACKENDS` list in `run_daily.py`
2. Remove its old `outputs` list (was used to copy JSON to Django)
3. `copy_to_github()` at end calls `write_landing_page()` + `push_to_github()`

---

## Setup Checklist (first time on a new machine / session)

1. Clone: `git clone https://github.com/briancarter456546/market-dashboards.git`  
   into `C:\Users\lynda\1youtubevideopull\perplexity-user-data\`
2. Enable GitHub Pages: repo Settings → Pages → Deploy from branch → main → /docs
3. `dashboard_writer.py` `REPO_ROOT` uses `expanduser("~")` + subdirectory — verify it  
   resolves to the correct path by running `python dashboard_writer.py` (self-test)
4. Run a backend standalone to test: `python intermarket_spread_monitor_v250.py`
5. Check `docs/spread-monitor/index.html` was created
6. `git push` happens automatically via `push_to_github()` at end of `run_daily.py`
