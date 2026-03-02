# -*- coding: utf-8 -*-
# ============================================================================
# dashboard_llm_descriptions.py - v1.0
# Last updated: 2026-03-02
# ============================================================================
# v1.0: Initial release
#   - Generates dynamic LLM interpretations for each dashboard
#   - Reads today's data files, sends to claude-haiku-4-5 for 3-5 sentence summary
#   - Caches results to llm_descriptions.json
#   - Injects description blocks into already-written dashboard HTML files
#   - Runs LAST in run_daily.py pipeline (after all backends have written HTML)
# ============================================================================

import os
import sys
import json
import glob
import datetime
import traceback

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.normpath(os.path.join(_SCRIPT_DIR, '..', 'perplexity-user-data'))
_LLM_CACHE = os.path.join(_SCRIPT_DIR, 'llm_descriptions.json')


# ---------------------------------------------------------------------------
# Data file mapping: slug -> (path_pattern, file_type)
# Each entry tells us where to find today's data for that dashboard.
# ---------------------------------------------------------------------------
DATA_FILES = {
    'spread-monitor': {
        'pattern': os.path.join(_DATA_DIR, 'output', 'spread_monitor', 'spread_monitor_*.csv'),
        'type': 'csv',
        'max_rows': 30,
    },
    'sector-rotation': {
        'pattern': os.path.join(_SCRIPT_DIR, 'sector_rotation_data_*.csv'),
        'type': 'csv',
        'max_rows': 20,
    },
    'momentum-ranker': {
        'pattern': os.path.join(_SCRIPT_DIR, 'momentum_ranker_data.json'),
        'type': 'json',
        'extract': 'top10',
    },
    'momentum-ranker-long': {
        'pattern': os.path.join(_SCRIPT_DIR, 'momentum_ranker_long_data.json'),
        'type': 'json',
        'extract': 'top10',
    },
    'similar-days': {
        'pattern': os.path.join(_SCRIPT_DIR, 'similar_days_data_*.csv'),
        'type': 'csv',
        'max_rows': 15,
    },
    'historical-mirror': {
        'pattern': os.path.join(_SCRIPT_DIR, 'mirror_data_*.csv'),
        'type': 'csv',
        'max_rows': 15,
    },
    'stock-secrot': {
        'pattern': os.path.join(_DATA_DIR, 'stock_secrot_scores_*.csv'),
        'type': 'csv',
        'max_rows': 20,
    },
    'hyglqd-credit': {
        'pattern': os.path.join(_SCRIPT_DIR, 'hyglqd_data_*.csv'),
        'type': 'csv',
        'max_rows': 10,
    },
    'crash-detection': {
        'pattern': os.path.join(_SCRIPT_DIR, 'crash_detection_data_*.csv'),
        'type': 'csv',
        'max_rows': 5,
    },
    'advanced-momentum': {
        'pattern': os.path.join(_DATA_DIR, 'advanced_momentum_analysis.json'),
        'type': 'json',
        'extract': 'summary',
    },
    'conservative-momentum': {
        'pattern': os.path.join(_SCRIPT_DIR, 'conservative_momentum_data_*.csv'),
        'type': 'csv',
        'max_rows': 15,
    },
    'macro': {
        'pattern': os.path.join(_DATA_DIR, 'macro_data.json'),
        'type': 'json',
        'extract': 'macro_summary',
    },
    'regime-changepoint': {
        'pattern': os.path.join(_SCRIPT_DIR, 'changepoint_summary.json'),
        'type': 'json',
        'extract': 'full',
    },
    'smart-scanner': {
        'pattern': os.path.join(_SCRIPT_DIR, 'smart_scanner_data_*.json'),
        'type': 'json',
        'extract': 'full',
    },
    'meta-dashboard': {
        'pattern': os.path.join(_SCRIPT_DIR, 'meta_dashboard_data_*.csv'),
        'type': 'csv',
        'max_rows': 20,
    },
}


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_latest_file(pattern):
    """Find most recently modified file matching glob pattern."""
    # Handle both glob patterns and direct file paths
    if '*' in pattern:
        matches = glob.glob(pattern)
        if not matches:
            return None
        return max(matches, key=os.path.getmtime)
    elif os.path.exists(pattern):
        return pattern
    return None


def _load_data_snippet(slug):
    """Load a compact data snippet for LLM consumption."""
    config = DATA_FILES.get(slug)
    if not config:
        return None

    filepath = _load_latest_file(config['pattern'])
    if not filepath:
        return None

    try:
        if config['type'] == 'csv':
            import pandas as pd
            df = pd.read_csv(filepath, encoding='utf-8')
            max_rows = config.get('max_rows', 20)
            return df.head(max_rows).to_string(index=False, max_colwidth=40)

        elif config['type'] == 'json':
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            extract = config.get('extract', 'full')

            if extract == 'top10':
                # Ranker data: show top 10 by rank
                items = data.get('data', [])
                top = sorted(items, key=lambda x: x.get('rank', 9999))[:10]
                return json.dumps(top, indent=2, default=str)[:3000]

            elif extract == 'summary':
                # Advanced momentum: show signal distribution
                results = data.get('results', [])
                signals = {}
                for r in results:
                    sig = r.get('signal', 'UNKNOWN')
                    signals[sig] = signals.get(sig, 0) + 1
                top_buys = [r for r in results if r.get('signal') in ('STRONG_BUY', 'BUY')][:10]
                summary = {
                    'signal_distribution': signals,
                    'top_buys': [{k: v for k, v in item.items()
                                  if k in ('symbol', 'signal', 'confidence', 'trajectory')}
                                 for item in top_buys],
                }
                return json.dumps(summary, indent=2, default=str)[:3000]

            elif extract == 'macro_summary':
                # Macro data: extract key metrics
                summary = {}
                if 'volatility' in data:
                    summary['vix'] = data['volatility'].get('vix')
                if 'treasury' in data:
                    summary['treasury_spreads'] = data['treasury'].get('spreads', {})
                if 'credit' in data:
                    summary['hyg_lqd_ratio'] = data['credit'].get('hyg_lqd_ratio')
                if 'breadth' in data:
                    summary['breadth_pct_above_200ma'] = data['breadth'].get('pct_above_200ma')
                if 'indices' in data:
                    summary['spy_trend'] = data['indices'].get('spy_trend', {})
                return json.dumps(summary, indent=2, default=str)[:3000]

            else:
                # Full JSON, truncated
                return json.dumps(data, indent=2, default=str)[:3000]

    except Exception as e:
        print('  WARN: could not load data for {}: {}'.format(slug, e))
        return None


# ---------------------------------------------------------------------------
# LLM interpretation generator
# ---------------------------------------------------------------------------

def generate_interpretations(slugs=None):
    """Generate dynamic LLM interpretations for each dashboard.

    Returns dict of {slug: {'dynamic': str, 'generated_at': str}}.
    """
    try:
        import anthropic
    except ImportError:
        print('  WARN: anthropic SDK not installed -- skipping LLM interpretations')
        return {}

    client = anthropic.Anthropic()
    results = {}

    if slugs is None:
        slugs = list(DATA_FILES.keys())

    for slug in slugs:
        data_snippet = _load_data_snippet(slug)
        if not data_snippet:
            print('  SKIP: no data for {}'.format(slug))
            continue

        # Import description from dashboard_writer
        from dashboard_writer import DASHBOARD_DESCRIPTIONS
        static_desc = DASHBOARD_DESCRIPTIONS.get(slug, '')

        prompt = (
            "You are a market analyst assistant. Below is the static description of "
            "a trading dashboard, followed by today's data output.\n\n"
            "Dashboard: {slug}\n"
            "Description: {desc}\n\n"
            "Today's data:\n{data}\n\n"
            "Write a 3-5 sentence interpretation of what today's data shows. "
            "Focus on notable patterns, extremes, or changes. "
            "Be specific about numbers and tickers. "
            "Do NOT give financial advice. Use plain language, no jargon. "
            "Do NOT use emoji or Unicode characters. ASCII only."
        ).format(slug=slug, desc=static_desc, data=data_snippet[:2500])

        try:
            response = client.messages.create(
                model='claude-haiku-4-5',
                max_tokens=300,
                messages=[{'role': 'user', 'content': prompt}],
            )
            interpretation = response.content[0].text.strip()
            results[slug] = {
                'dynamic': interpretation,
                'generated_at': datetime.datetime.now().isoformat(),
            }
            print('  [OK] {}: {} chars'.format(slug, len(interpretation)))
        except Exception as e:
            print('  WARN: LLM call failed for {}: {}'.format(slug, e))

    return results


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------

def save_cache(results):
    """Save LLM interpretations to llm_descriptions.json."""
    # Load existing cache and merge
    existing = {}
    if os.path.exists(_LLM_CACHE):
        try:
            with open(_LLM_CACHE, 'r', encoding='utf-8') as f:
                existing = json.load(f)
        except Exception:
            pass

    existing.update(results)

    with open(_LLM_CACHE, 'w', encoding='utf-8') as f:
        json.dump(existing, f, indent=2, ensure_ascii=True)

    print('  Cache written: {} ({} entries)'.format(_LLM_CACHE, len(existing)))


def load_cache():
    """Load cached LLM interpretations."""
    if not os.path.exists(_LLM_CACHE):
        return {}
    try:
        with open(_LLM_CACHE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# HTML injection
# ---------------------------------------------------------------------------

def inject_llm_blocks():
    """Inject LLM description blocks into already-written dashboard HTML files.

    Finds each dashboard's index.html, inserts the llm-block div after the
    page-header section.
    """
    from dashboard_writer import DASHBOARD_DESCRIPTIONS, DOCS_DIR, _LLM_DISCLAIMER

    cache = load_cache()
    injected = 0

    for slug in DASHBOARD_DESCRIPTIONS:
        index_path = os.path.join(DOCS_DIR, slug, 'index.html')
        if not os.path.exists(index_path):
            continue

        with open(index_path, 'r', encoding='utf-8') as f:
            html = f.read()

        # Skip if already injected
        if 'llm-block' in html:
            continue

        static_desc = DASHBOARD_DESCRIPTIONS.get(slug, '')
        if not static_desc:
            continue

        # Build the block
        dynamic = ''
        entry = cache.get(slug, {})
        if entry.get('dynamic'):
            dynamic = '<div class="llm-dynamic">{}</div>'.format(entry['dynamic'])

        block = (
            '<div class="llm-block">'
            '<div class="llm-block-header">About This Dashboard</div>'
            '<div class="llm-block-body">'
            '<div class="llm-static">{static}</div>'
            '{dynamic}'
            '<div class="llm-disclaimer">{disclaimer}</div>'
            '</div></div>'
        ).format(static=static_desc, dynamic=dynamic, disclaimer=_LLM_DISCLAIMER)

        # Inject after <div class="content"> (the main content area)
        marker = '<div class="content">'
        if marker in html:
            html = html.replace(marker, marker + '\n' + block, 1)
        else:
            # Fallback: inject after </div> following page-nav
            marker2 = '<div class="page-nav">'
            if marker2 in html:
                # Find the closing </div> after page-nav, then insert after it
                idx = html.find(marker2)
                close_idx = html.find('</div>', idx)
                if close_idx > 0:
                    insert_at = close_idx + len('</div>')
                    html = html[:insert_at] + '\n' + block + html[insert_at:]
            else:
                # Last resort: inject right after <body>
                html = html.replace('<body>', '<body>\n' + block, 1)

        # Also inject archive copy
        archive_dir = os.path.join(DOCS_DIR, slug, 'archive')
        today_compact = datetime.date.today().strftime('%Y%m%d')
        archive_path = os.path.join(archive_dir, 'dashboard_{}.html'.format(today_compact))

        with open(index_path, 'w', encoding='utf-8') as f:
            f.write(html)

        if os.path.exists(archive_path):
            with open(archive_path, 'w', encoding='utf-8') as f:
                f.write(html)

        injected += 1

    print('  Injected LLM blocks into {} dashboards'.format(injected))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print('=' * 68)
    print('DASHBOARD LLM DESCRIPTIONS v1.0')
    print('=' * 68)

    # 1. Generate dynamic interpretations
    print('\n[1/3] Generating LLM interpretations...')
    results = generate_interpretations()
    print('  Generated {} interpretations'.format(len(results)))

    # 2. Save to cache
    print('\n[2/3] Saving cache...')
    save_cache(results)

    # 3. Inject into HTML files
    print('\n[3/3] Injecting into dashboard HTML...')
    inject_llm_blocks()

    print('\n[DONE] LLM descriptions v1.0 complete.')


if __name__ == '__main__':
    main()
