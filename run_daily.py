"""
run_daily.py - Dashboard Suite Orchestrator
============================================
Runs all backends to generate static HTML dashboards.
Pushes output to market-dashboards GitHub Pages repo.

Usage:
    python run_daily.py           # Run all backends
    python run_daily.py --verbose # Show real-time output from each script
    python run_daily.py --quick   # Skip slow backends
    python run_daily.py --test    # Dry run, show what would execute
    python run_daily.py -v        # Short form for --verbose

Schedule with Windows Task Scheduler:
    Program:   C:/Users/lynda/anaconda3/python.exe
    Arguments: run_daily.py
    Start in:  C:/Users/lynda/1youtubevideopull/market-dashboards
    Trigger:   Daily at 4:30 PM, weekdays only

All scripts live here in market-dashboards/ except daily_price_updater which
lives in perplexity-user-data/ (location='root').
"""

import subprocess
import sys
import os
from datetime import datetime
import time

# ==============================================================================
# CONFIGURATION
# ==============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "perplexity-user-data"))

# Backends to run, in order.
# location='root' means the script lives in ROOT_DIR (perplexity-user-data/).
# location='local' (default) means it lives in SCRIPT_DIR (market-dashboards/).
BACKENDS = [
    {
        'script': 'daily_price_updater_v1.4.py',
        'name': 'Price Data Update',
        'location': 'root',
        'slow': False,
        'required': True,
    },
    {
        'script': 'fmp_pole_returns_exporter_v1_0.py',
        'name': 'FMP Pole Returns Exporter',
        'location': 'root',
        'slow': False,
        'required': True,
    },
    {
        'script': 'macro_backend.py',
        'name': 'Macro Dashboard',
        'slow': False,
        'required': False,
    },
    {
        'script': 'hyglqd_backend.py',
        'name': 'HYG/LQD Credit Spread',
        'slow': False,
        'required': False,
    },
    {
        'script': 'crash_detection_backend.py',
        'name': 'Crash Detection (RMT + Ising)',
        'slow': False,
        'required': False,
    },
    {
        'script': 'sector_rotation_v0_8.py',
        'name': 'Sector Rotation Deep Dive',
        'slow': False,
        'required': True,
    },
    {
        'script': 'similar_days_analyzer_v1_13.py',
        'name': 'Similar Days Analysis',
        'slow': False,
        'required': True,
    },
    {
        'script': 'mirror_backend.py',
        'name': 'Historical Mirror',
        'slow': True,
        'required': False,
    },
    {
        'script': 'stock_secrot_backend.py',
        'name': 'Stock Sector Rotation',
        'slow': False,
        'required': False,
    },
    {
        'script': 'momentum_ranker_v1_18.py',
        'name': 'Momentum Ranker',
        'slow': False,
        'required': True,
    },
    {
        'script': 'momentum_ranker_long_v1_18.py',
        'name': 'Momentum Ranker (Long)',
        'slow': False,
        'required': False,
    },
    {
        'script': 'advanced_momentum_backend.py',
        'name': 'Advanced Momentum Analyzer',
        'slow': True,
        'required': False,
    },
    {
        'script': 'momentum_qualifier_backend.py',
        'name': 'Conservative Momentum Qualifier',
        'slow': True,
        'required': False,
    },
    {
        'script': 'intermarket_spread_monitor_v250.py',
        'name': 'Intermarket Spread Monitor',
        'slow': False,
        'required': True,
    },
    {
        'script': 'changepoint_backend.py',
        'name': 'Regime Changepoint Detector',
        'slow': False,
        'required': False,
    },
    {
        'script': 'slope_stage_backend.py',
        'name': 'Slope Stage Scanner',
        'slow': True,
        'required': False,
    },
    {
        'script': 'pullback_health_backend.py',
        'name': 'Pullback Health Monitor',
        'slow': False,
        'required': False,
    },
    {
        'script': 'smart_scanner_v1_6.py',
        'name': 'Smart Scanner',
        'location': 'root',
        'slow': True,
        'required': False,
    },
    {
        'script': 'rsi2_dashboard_backend.py',
        'name': 'RSI2 Scanner Dashboard',
        'slow': False,
        'required': False,
    },
    {
        'script': 'meta_dashboard_backend.py',
        'name': 'Meta Dashboard',
        'slow': False,
        'required': False,
    },
    {
        'script': 'market_reality_backend.py',
        'name': 'Market Reality Check',
        'slow': False,
        'required': False,
    },
    # smart_money_backend.py REMOVED 2026-03-09: all indicators failed
    # scimode validation (CMF/OBV/MFI = coin flip vs real 13F data).
    # Archived to perplexity-user-data/archive/. Replaced by institutional_flows_backend.py.
    {
        'script': 'pole_rotation_backend.py',
        'name': 'Proven Pole Rotation',
        'slow': False,
        'required': False,
    },
    {
        'script': 'dashboard_llm_descriptions.py',
        'name': 'LLM Dashboard Descriptions',
        'slow': False,
        'required': False,
    },
]

# ==============================================================================
# HELPERS
# ==============================================================================

def log(msg, level='INFO'):
    """Log with timestamp"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    prefix = {'INFO': '[OK]', 'WARN': '[!]', 'ERROR': '[X]', 'START': '>>>'}.get(level, '   ')
    print("[{}] {} {}".format(timestamp, prefix, msg))


def run_backend(backend, dry_run=False, verbose=False):
    """Run a single backend script"""
    script = backend['script']
    name = backend['name']
    location = backend.get('location', 'local')

    # Resolve script path and working directory based on location
    if location == 'root':
        script_path = os.path.join(ROOT_DIR, script)
        cwd = ROOT_DIR
    else:
        script_path = os.path.join(SCRIPT_DIR, script)
        cwd = SCRIPT_DIR

    if not os.path.exists(script_path):
        log("{}: Script not found ({})".format(name, script_path), 'WARN')
        return False, 0

    if dry_run:
        log("{}: Would run {} (cwd={})".format(name, script_path, cwd), 'INFO')
        return True, 0

    log("{}: Starting...".format(name), 'START')
    start_time = time.time()

    try:
        if verbose:
            print()
            result = subprocess.run(
                [sys.executable, script_path],
                cwd=cwd,
                timeout=600
            )
            print()
            elapsed = time.time() - start_time
        else:
            result = subprocess.run(
                [sys.executable, script_path],
                cwd=cwd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=600
            )
            elapsed = time.time() - start_time

        if result.returncode == 0:
            log("{}: Completed in {:.1f}s".format(name, elapsed), 'INFO')
            return True, elapsed
        else:
            log("{}: Failed (exit code {})".format(name, result.returncode), 'ERROR')
            if not verbose and hasattr(result, 'stderr') and result.stderr:
                error_lines = result.stderr.strip().split('\n')[-5:]
                for line in error_lines:
                    log("  +- {}".format(line), 'ERROR')
            return False, elapsed

    except subprocess.TimeoutExpired:
        log("{}: Timeout after 10 minutes".format(name), 'ERROR')
        return False, 600
    except Exception as e:
        log("{}: Exception - {}".format(name, e), 'ERROR')
        return False, 0


def check_dependencies():
    """Check if required data files exist"""
    required_files = [
        (os.path.join(ROOT_DIR, 'regime_analysis_v2.json'), 'Run regime_backtester_v2.py first'),
        (os.path.join(ROOT_DIR, 'secrot_pattern_db.pkl'), 'Run regime_backtester_v2.py first'),
        (os.path.join(ROOT_DIR, 'price_cache'), 'Run daily_price_updater first'),
    ]
    missing = []
    for filepath, hint in required_files:
        if not os.path.exists(filepath):
            missing.append((os.path.basename(filepath), hint))
    return missing


def copy_to_github(dry_run=False):
    """
    Write landing page and push all dashboard HTML to GitHub Pages.
    Replaces copy_to_django() from the previous version.
    """
    from dashboard_writer import write_landing_page, push_to_github, DOCS_DIR

    if dry_run:
        log("Would write landing page and push to GitHub Pages", 'INFO')
        log("  +- Target: {}".format(DOCS_DIR), 'INFO')
        return True

    log("Writing landing page...", 'START')
    try:
        write_landing_page()
        log("Landing page written.", 'INFO')
    except Exception as e:
        log("Landing page failed: {}".format(e), 'ERROR')

    success = push_to_github()
    if success:
        log("GitHub Pages updated.", 'INFO')
    else:
        log("GitHub push failed - dashboards are written locally but not published.", 'WARN')

    return success


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    args = sys.argv[1:]
    quick_mode = '--quick' in args
    dry_run = '--test' in args
    verbose = '--verbose' in args or '-v' in args

    print()
    print("=" * 70)
    print("DASHBOARD SUITE - DAILY ORCHESTRATOR")
    print("=" * 70)
    log("Mode: {}".format('DRY RUN' if dry_run else ('QUICK' if quick_mode else 'FULL')))
    log("Verbose: {}".format('ON' if verbose else 'OFF'))
    log("Python: {}".format(sys.executable))
    log("Working Directory: {}".format(os.getcwd()))
    print()

    missing = check_dependencies()
    if missing:
        log("Missing required files:", 'WARN')
        for filename, hint in missing:
            log("  +- {} - {}".format(filename, hint), 'WARN')
        print()
        log("Continuing anyway - some features may not work", 'WARN')
        print()

    results = []
    total_time = 0

    for backend in BACKENDS:
        if quick_mode and backend.get('slow', False):
            log("{}: Skipped (quick mode)".format(backend['name']), 'INFO')
            continue

        success, elapsed = run_backend(backend, dry_run, verbose)
        results.append({
            'name': backend['name'],
            'success': success,
            'elapsed': elapsed,
            'required': backend.get('required', False)
        })
        total_time += elapsed
        print()

    # Reseed knowledge base with fresh data
    kb_script = os.path.join(ROOT_DIR, 'knowledge_base_v1_0.py')
    if os.path.exists(kb_script):
        if dry_run:
            log("Would reseed knowledge base", 'INFO')
        else:
            # Backup KB before reseeding (preserves manual findings)
            kb_backup = os.path.join(ROOT_DIR, 'kb_backup_v1_0.py')
            if os.path.exists(kb_backup):
                try:
                    subprocess.run(
                        [sys.executable, kb_backup],
                        cwd=ROOT_DIR, capture_output=True, timeout=30
                    )
                    log("Knowledge Base: Backup complete", 'INFO')
                except Exception:
                    log("Knowledge Base: Backup failed (continuing)", 'WARN')

            log("Knowledge Base: Reseeding...", 'START')
            kb_start = time.time()
            try:
                kb_result = subprocess.run(
                    [sys.executable, kb_script, 'seed', '--force'],
                    cwd=ROOT_DIR,
                    capture_output=True, text=True,
                    encoding='utf-8', errors='replace',
                    timeout=120
                )
                kb_elapsed = time.time() - kb_start
                if kb_result.returncode == 0:
                    log("Knowledge Base: Reseeded in {:.1f}s".format(kb_elapsed), 'INFO')
                else:
                    log("Knowledge Base: Seed failed (non-critical)", 'WARN')
                    if kb_result.stderr:
                        for line in kb_result.stderr.strip().split('\n')[-3:]:
                            log("  +- {}".format(line), 'WARN')
            except Exception as e:
                log("Knowledge Base: {} (non-critical)".format(e), 'WARN')

    # Check reminders
    reminders_file = os.path.join(ROOT_DIR, 'reminders.json')
    if os.path.exists(reminders_file):
        try:
            import json as _json
            with open(reminders_file, 'r', encoding='utf-8') as _f:
                reminders = _json.load(_f)
            today = datetime.now().strftime('%Y-%m-%d')
            remaining = []
            for r in reminders:
                if r.get('date', '') <= today:
                    print()
                    print("!" * 70)
                    print("REMINDER ({})".format(r.get('date', '')))
                    print(r.get('message', ''))
                    print("!" * 70)
                    print()
                else:
                    remaining.append(r)
            if len(remaining) < len(reminders):
                with open(reminders_file, 'w', encoding='utf-8') as _f:
                    _json.dump(remaining, _f, indent=2)
        except Exception:
            pass
    print()

    # Push to GitHub Pages (replaces copy_to_django)
    copy_to_github(dry_run=dry_run)
    print()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    succeeded = sum(1 for r in results if r['success'])
    failed = sum(1 for r in results if not r['success'])
    required_failed = sum(1 for r in results if not r['success'] and r['required'])

    log("Backends: {} succeeded, {} failed".format(succeeded, failed))
    log("Total time: {:.1f}s".format(total_time))

    if required_failed > 0:
        log("CRITICAL: {} required backend(s) failed!".format(required_failed), 'ERROR')
        print()
        return 1
    elif failed > 0:
        log("Some optional backends failed - dashboards may be stale", 'WARN')
        print()
        return 0
    else:
        log("All backends completed successfully", 'INFO')
        print()
        return 0


if __name__ == "__main__":
    sys.exit(main())
