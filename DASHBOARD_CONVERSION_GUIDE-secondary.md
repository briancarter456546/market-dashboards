This conversation established a reusable system for converting your Django-style trading dashboards into static HTML dashboards served via GitHub Pages, with a shared design system and consistent path conventions. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/3383952/6da49489-e671-44e0-aea4-63b397b1f30b/paste.txt)

## Overall architecture

- All dashboards will become standalone Python scripts that read from your existing data sources (price_cache, cache_loader, CSVs) and write static HTML into a GitHub Pages repo named `market-dashboards` (served from `/docs`). [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/3383952/6da49489-e671-44e0-aea4-63b397b1f30b/paste.txt)
- A shared module `dashboard_writer.py` now owns: base CSS/JS, page layout helpers, a landing page, the archive + `index.html` pattern, and git-push integration; each dashboard imports and uses it instead of duplicating boilerplate. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/3383952/6da49489-e671-44e0-aea4-63b397b1f30b/paste.txt)
- The GitHub Pages repo lives at `C:\Users\lynda\1youtubevideopull\perplexity-user-data\market-dashboards`, and all paths in the shared writer reference that absolute base via `REPO_ROOT` so outputs go to `docs/<dashboard>/index.html` and `docs/<dashboard>/archive/dashboard_YYYYMMDD.html`. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/3383952/6da49489-e671-44e0-aea4-63b397b1f30b/paste.txt)

## GitHub Pages and repo layout

- GitHub Pages is configured to serve from `main` ? `/docs` in the `market-dashboards` repo so code and served HTML stay separated. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/3383952/6da49489-e671-44e0-aea4-63b397b1f30b/paste.txt)
- Standard structure:  
  - `docs/index.html` is a landing page linking to each dashboard.  
  - Each dashboard has its own folder under `docs/` (for example `docs/spread-monitor/`, `docs/sector-rotation/`) with `index.html` plus an `archive/` subfolder for dated snapshots. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/3383952/6da49489-e671-44e0-aea4-63b397b1f30b/paste.txt)
- The shared writer maintains a `DASHBOARD_REGISTRY` so the landing page can be auto-generated with proper names, URLs, and descriptions; newly converted dashboards just add one registry entry. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/3383952/6da49489-e671-44e0-aea4-63b397b1f30b/paste.txt)

## Shared design and UX decisions

- You decided against dark “terminal” themes and adopted a light, trader-friendly design with:  
  - Light grey page background, white cards, and a darker top stat-bar section for emphasis. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/3383952/6da49489-e671-44e0-aea4-63b397b1f30b/paste.txt)
  - Base font around 17px (IBM Plex Sans via Google Fonts) for readability, with IBM Plex Mono reserved for numbers and tickers where alignment matters. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/3383952/6da49489-e671-44e0-aea4-63b397b1f30b/paste.txt)
  - Pill-shaped badges for qualifiers (ABOVE/BELOW, etc.) and colored status tags, following a style similar to your advanced momentum dashboard. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/3383952/6da49489-e671-44e0-aea4-63b397b1f30b/paste.txt)
- The shared CSS in `dashboard_writer.py` defines:  
  - Layout for headers, nav, content cards, tables, sortable headers, stat bars, regime banners, and action “playbook” sections.  
  - Sort state styling (ascending/descending headers) and hover states for rows. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/3383952/6da49489-e671-44e0-aea4-63b397b1f30b/paste.txt)
- The helper functions in `dashboard_writer.py` encapsulate layout: `build_header`, `build_footer`, `section` (card container), `stat_bar`, `regime_banner`, and re-usable card components, so each dashboard’s Python code just provides content, not HTML strings. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/3383952/6da49489-e671-44e0-aea4-63b397b1f30b/paste.txt)

## Sorting and table behavior

- All dashboards are expected to support bi-directional column sorting on every column. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/3383952/6da49489-e671-44e0-aea4-63b397b1f30b/paste.txt)
- The spread monitor uses a generic `SORT_JS` placed in the shared writer, acting on any table with `class="sortable-table"` and using `data-sort` attributes on `<td>` cells to hold raw numeric or string values; clicking headers toggles ascending/descending. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/3383952/6da49489-e671-44e0-aea4-63b397b1f30b/paste.txt)
- CSS provides visual cues for sorting (highlighted header, caret-like background via borders), and this JS is written to be dashboard-agnostic so new dashboards can opt-in by using the same class and attributes. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/3383952/6da49489-e671-44e0-aea4-63b397b1f30b/paste.txt)
- When tables have multiple header rows (for grouped column labels), the sort logic is aware that the second header row contains the clickable labels, not the group-row. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/3383952/6da49489-e671-44e0-aea4-63b397b1f30b/paste.txt)

## Path and environment conventions

- All new dashboards are stored in `perplexity-user-data\market-dashboards` and are meant to be run from that directory (you prefer this for organization rather than running from the parent). [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/3383952/6da49489-e671-44e0-aea4-63b397b1f30b/paste.txt)
- To make relative data paths robust, each Python dashboard uses:  
  - `_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))`  
  - `_DATA_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))`  
  so it can locate shared resources (like `price_cache` or `cache_loader_v1_0.py`) one level up in `perplexity-user-data` regardless of the current working directory. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/3383952/6da49489-e671-44e0-aea4-63b397b1f30b/paste.txt)
- For dashboards that read `price_cache` directly (like the spread monitor), `cache_dir` is set using `_DATA_DIR` (for example `_DATA_DIR / "price_cache"`), and CSV/other outputs are also rooted at `_DATA_DIR` or `REPO_ROOT` to avoid brittle relative paths. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/3383952/6da49489-e671-44e0-aea4-63b397b1f30b/paste.txt)
- The sector rotation script had to be updated to:  
  - Import `os` because path helpers were added.  
  - Insert `_DATA_DIR` into `sys.path` and robustly load either `cache_loader_v1_0.py` or `cache_loader_v1.0.py` via `importlib` so it works with your actual filenames. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/3383952/6da49489-e671-44e0-aea4-63b397b1f30b/paste.txt)

## Conversion patterns

There are two main patterns for converting legacy backends to this static-dashboard system. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/3383952/6da49489-e671-44e0-aea4-63b397b1f30b/paste.txt)

### Pattern 1: Pure-Python HTML + shared writer

- Used for dashboards like the intermarket spread monitor where coloring and layout can be done entirely in Python. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/3383952/6da49489-e671-44e0-aea4-63b397b1f30b/paste.txt)
- Steps for a backend:  
  - Keep all compute logic unchanged (load data, calculate metrics).  
  - Add a `build_body_html(result_data: dict) -> str` function that:  
    - Uses `dashboard_writer` helpers to construct stat bars, headers, table HTML, playbook sections, etc.  
    - Adds `data-sort` attributes on `<td>` cells for the sorter to function. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/3383952/6da49489-e671-44e0-aea4-63b397b1f30b/paste.txt)
  - In `main()`, after computing, call a `DashboardWriter` (or equivalent) to write both dated and latest HTML under the proper `docs/<dashboard>/` path and optionally CSV outputs. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/3383952/6da49489-e671-44e0-aea4-63b397b1f30b/paste.txt)
- The spread monitor refactor (to v2.5.x) is the canonical example: logic unchanged, layout moved into `build_body_html`, HTML written via the shared writer, and all styling centralized. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/3383952/6da49489-e671-44e0-aea4-63b397b1f30b/paste.txt)

### Pattern 2: Embedded JSON + client-side JS rendering

- Used for dashboards that require per-column gradient coloring based on percentile ranges, such as the 25-year sector rotation dashboard. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/3383952/6da49489-e671-44e0-aea4-63b397b1f30b/paste.txt)
- Instead of generating colored cells in Python, the script:  
  - Serializes the entire dashboard data structure to JSON and embeds it in the HTML as a `const` (for example `const DATA = {...};`).  
  - Includes a JS block that replicates the original Django/JS logic:  
    - `calculateColumnRanges()` computes 5th–95th percentile min/max per column.  
    - `getGradientColor()` maps normalized values to red?yellow?green gradients with luminance-aware text color selection.  
    - `renderTable()` builds rows dynamically and applies per-column gradient backgrounds and win% text, exactly as your original HTML did. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/3383952/6da49489-e671-44e0-aea4-63b397b1f30b/paste.txt)
- Sorting is handled in JS using the same data structure (no reliance on `data-sort` attributes), closely mirroring the original interactive behavior. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/3383952/6da49489-e671-44e0-aea4-63b397b1f30b/paste.txt)
- Sector rotation v0.5y was converted using this pattern to keep the nuanced per-column gradients for the Momentum and Pred 25Y/5Y/1Y columns while still outputting a single static HTML file. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/3383952/6da49489-e671-44e0-aea4-63b397b1f30b/paste.txt)

## Current status and next steps

- Already converted and styled under the new system:  
  - Intermarket Spread Monitor (v2.5.x) with light theme, sortable table, force “pill” stats, and playbook section.  
  - Sector Rotation 25Y/5Y/1Y (v0.5y) with JS-driven table, per-column gradients, and the original 19-column layout preserved. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/3383952/6da49489-e671-44e0-aea4-63b397b1f30b/paste.txt)
- The `dashboard_writer` landing page registry already includes entries for spread monitor and sector rotation with descriptive text, so they appear on the index page. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/3383952/6da49489-e671-44e0-aea4-63b397b1f30b/paste.txt)
- Future conversions should:  
  - Reuse the same `_SCRIPT_DIR/_DATA_DIR` pattern.  
  - Pick Pattern 1 or Pattern 2 depending on whether the original dashboard relied heavily on JS for visual logic.  
  - Add each dashboard to `DASHBOARD_REGISTRY` for consistent naming and linking. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/3383952/6da49489-e671-44e0-aea4-63b397b1f30b/paste.txt)
- A dedicated `DASHBOARD_CONVERSION_GUIDE.md` file (already written) sits alongside your code, summarizing these conventions so future LLMs can quickly align and continue the conversion work without re-deriving the system. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/3383952/6da49489-e671-44e0-aea4-63b397b1f30b/paste.txt)