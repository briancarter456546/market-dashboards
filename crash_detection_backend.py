# -*- coding: utf-8 -*-
# =============================================================================
# crash_detection_backend.py
# DashboardWriter-style backend for RMT + Ising crash detection metrics.
# Produces a static HTML dashboard in docs/crash-detection/.
#
# Computation logic ported from root/crash_detection_backend.py (unchanged).
# Path setup and HTML output updated for market-dashboards/ layout.
#
# Author: Brian + Claude
# Date: 2026-02-26
# =============================================================================

import os
import json
import pickle
import datetime
import time

import numpy as np
import pandas as pd
from pathlib import Path

from dashboard_writer import DashboardWriter

# =============================================================================
# PATH SETUP  (all relative to this script's location)
# =============================================================================

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR   = os.path.normpath(os.path.join(_SCRIPT_DIR, '..', 'perplexity-user-data'))
CACHE_DIR   = os.path.normpath(os.path.join(_DATA_DIR, 'price_cache'))

# =============================================================================
# STATUS COLOR MAP
# =============================================================================

STATUS_COLORS = {
    'NORMAL':   '#16a34a',   # green
    'ELEVATED': '#d97706',   # amber
    'WARNING':  '#f97316',   # orange
    'CRITICAL': '#dc2626',   # red
}

# =============================================================================
# EXTRA CSS
# =============================================================================

EXTRA_CSS = """
.component-table td,
.component-table th {
    padding: 11px 16px;
}

.risk-pill {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.82em;
    font-weight: 700;
    letter-spacing: 0.04em;
}
.risk-NORMAL   { background: #dcfce7; color: #15803d; border: 1px solid #bbf7d0; }
.risk-ELEVATED { background: #fef3c7; color: #b45309; border: 1px solid #fde68a; }
.risk-WARNING  { background: #ffedd5; color: #c2410c; border: 1px solid #fed7aa; }
.risk-CRITICAL { background: #fee2e2; color: #b91c1c; border: 1px solid #fca5a5; }

.weight-bar-bg {
    background: #f0f2f5;
    border-radius: 4px;
    height: 8px;
    width: 120px;
    display: inline-block;
    vertical-align: middle;
    margin-right: 8px;
    overflow: hidden;
    position: relative;
}
.weight-bar-fill {
    height: 8px;
    border-radius: 4px;
    background: #4f46e5;
    display: inline-block;
}

/* ── Mobile responsive ── */
@media (max-width: 768px) {
    .component-table td, .component-table th { padding: 8px 10px; }
    .weight-bar-bg { width: 80px; }
    .risk-pill { font-size: 0.75em; padding: 3px 10px; }
}
"""

# =============================================================================
# CRASH DETECTION BACKEND CLASS  (computation logic unchanged from root version)
# =============================================================================


class CrashDetectionBackend(object):
    """
    Computes crash detection metrics:
      1. RMT lambda_max (correlation structure)
      2. Ising Magnetization (market alignment)
      3. Composite crash score
    """

    def __init__(self, cache_dir=None, window=252, min_assets=50):
        self.cache_dir = cache_dir if cache_dir is not None else CACHE_DIR
        self.window = window
        self.min_assets = min_assets

    # =========================================================================
    # DATA LOADING
    # =========================================================================

    def load_price_cache(self):
        """Load all prices from pickle files in CACHE_DIR."""
        print("\n[LOAD] Loading prices from {}...".format(self.cache_dir))
        start_time = time.time()

        files = list(Path(self.cache_dir).glob('*.pkl'))
        print("Found {} files".format(len(files)))

        prices = {}
        for pkl_file in files:
            try:
                with open(str(pkl_file), 'rb') as fh:
                    df = pickle.load(fh)
                    ticker = pkl_file.stem

                    if 'close' in df.columns:
                        prices[ticker] = df['close']
                    elif 'Close' in df.columns:
                        prices[ticker] = df['Close']
            except Exception:
                continue

        if not prices:
            print("[ERROR] No valid price data found!")
            return None

        price_df = pd.DataFrame(prices)
        price_df = price_df.sort_index()

        elapsed = time.time() - start_time
        print("[OK] Loaded {} assets in {:.1f}s".format(len(price_df.columns), elapsed))

        return price_df

    # =========================================================================
    # RMT CALCULATION
    # =========================================================================

    def compute_latest_rmt(self, prices):
        """
        Compute RMT lambda_max for the latest window.

        Returns dict with: lambda_max, lambda_plus (MP threshold), ratio,
        n_significant, n_assets, risk (NORMAL/ELEVATED/WARNING/CRITICAL).
        Thresholds: ELEVATED > 270, WARNING > 300, CRITICAL > 350.
        """
        print("\n[RMT] Computing latest lambda_max...")

        if prices is None or len(prices) < self.window:
            print("[ERROR] Not enough data for RMT")
            return None

        # Get last window
        window_prices = prices.iloc[-self.window:].values

        # Filter usable assets (70%+ data)
        non_nan_counts = np.sum(~np.isnan(window_prices), axis=0)
        usable_mask = non_nan_counts >= self.window * 0.7
        usable_cols = np.where(usable_mask)[0]

        if len(usable_cols) < self.min_assets:
            print("[ERROR] Only {} usable assets (need {})".format(
                len(usable_cols), self.min_assets))
            return None

        # Clean data
        clean_prices = window_prices[:, usable_cols].copy()

        # Forward fill
        mask = np.isnan(clean_prices)
        idx = np.where(~mask, np.arange(mask.shape[0])[:, None], 0)
        np.maximum.accumulate(idx, axis=0, out=idx)
        clean_prices = clean_prices[idx, np.arange(idx.shape[1])]

        # Log returns
        log_prices = np.log(clean_prices + 1e-10)
        returns = np.diff(log_prices, axis=0)

        # Remove NaN columns
        valid_cols_mask = ~np.any(np.isnan(returns), axis=0)
        returns = returns[:, valid_cols_mask]

        if returns.shape[1] < self.min_assets:
            print("[ERROR] Only {} valid assets after cleaning".format(returns.shape[1]))
            return None

        # Standardize
        means = np.mean(returns, axis=0)
        stds  = np.std(returns, axis=0) + 1e-8
        returns_std = (returns - means) / stds

        # Correlation matrix + eigenvalues
        corr = np.corrcoef(returns_std.T)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        eigenvalues = np.linalg.eigvalsh(corr)
        eigenvalues = np.sort(eigenvalues)[::-1]

        lambda_max = float(eigenvalues[0])

        # Marchenko-Pastur upper edge
        T = len(returns)
        N = returns.shape[1]
        q = float(T) / float(N) if N > 0 else 1.0

        if q > 1:
            lambda_plus = float((1.0 + np.sqrt(1.0 / q)) ** 2)
        else:
            lambda_plus = float((1.0 + np.sqrt(q)) ** 2)

        n_significant = int(np.sum(eigenvalues > lambda_plus))

        # Assess risk
        if lambda_max > 350:
            risk = 'CRITICAL'
        elif lambda_max > 300:
            risk = 'WARNING'
        elif lambda_max > 270:
            risk = 'ELEVATED'
        else:
            risk = 'NORMAL'

        ratio = float(lambda_max / lambda_plus) if lambda_plus > 0 else 0.0

        result = {
            'date':          str(prices.index[-1].date()),
            'lambda_max':    lambda_max,
            'lambda_plus':   float(lambda_plus),
            'ratio':         ratio,
            'n_significant': n_significant,
            'n_assets':      int(N),
            'risk':          risk,
        }

        print("[OK] lambda_max = {:.2f} (threshold = {:.2f}) [{}]".format(
            lambda_max, lambda_plus, risk))

        return result

    # =========================================================================
    # ISING CALCULATION
    # =========================================================================

    def compute_latest_ising(self, prices):
        """
        Compute Ising magnetization for the latest window.

        Returns dict with: magnetization, avg_correlation, std_correlation,
        max_correlation, crash_score, n_assets, risk
        (NORMAL/ELEVATED/WARNING/CRITICAL based on thresholds 0.55/0.65/0.75).
        """
        print("\n[ISING] Computing latest magnetization...")

        if prices is None or len(prices) < self.window:
            print("[ERROR] Not enough data for Ising")
            return None

        # Get last window
        window_prices = prices.iloc[-self.window:]

        # Log returns
        returns = np.log(window_prices / window_prices.shift(1))
        returns = returns.iloc[1:]  # drop the first NaN row

        # Drop assets with too many NaNs
        non_nan_counts = returns.notna().sum()
        usable_assets = non_nan_counts[
            non_nan_counts >= len(returns) * 0.7
        ].index
        returns_clean = returns[usable_assets]

        if len(returns_clean.columns) < 10:
            print("[ERROR] Only {} usable assets".format(len(returns_clean.columns)))
            return None

        # Market return (equal-weighted)
        market_return = returns_clean.mean(axis=1)

        # Vectorized correlation
        combined = pd.concat(
            [market_return.rename('market'), returns_clean], axis=1
        )
        corr_matrix = combined.corr()

        # Extract per-asset correlations to market
        correlations = corr_matrix.loc['market', returns_clean.columns].values
        correlations = correlations[~np.isnan(correlations)]

        if len(correlations) < 10:
            print("[ERROR] Only {} valid correlations".format(len(correlations)))
            return None

        # Magnetization = mean absolute correlation
        magnetization   = float(np.mean(np.abs(correlations)))
        avg_correlation = float(np.mean(correlations))
        std_correlation = float(np.std(correlations))
        max_correlation = float(np.max(np.abs(correlations)))

        # Assess risk
        if magnetization > 0.75:
            risk = 'CRITICAL'
        elif magnetization > 0.65:
            risk = 'WARNING'
        elif magnetization > 0.55:
            risk = 'ELEVATED'
        else:
            risk = 'NORMAL'

        # Crash score: 0 at magnetization=0.40, 100 at magnetization=0.80
        crash_score = float(np.clip((magnetization - 0.4) / 0.4 * 100.0, 0.0, 100.0))

        result = {
            'date':            str(prices.index[-1].date()),
            'magnetization':   magnetization,
            'avg_correlation': avg_correlation,
            'std_correlation': std_correlation,
            'max_correlation': max_correlation,
            'crash_score':     crash_score,
            'n_assets':        int(len(correlations)),
            'risk':            risk,
        }

        print("[OK] Magnetization = {:.3f} [{}]".format(magnetization, risk))

        return result

    # =========================================================================
    # COMPOSITE SCORE
    # =========================================================================

    def compute_composite_score(self, rmt_result, ising_result, macro_data):
        """
        Compute composite crash score from multiple signals.

        Components and weights:
          RMT lambda_max    35%
          Ising             20%
          Market Breadth    20%
          VIX               15%
          Credit Spreads     5%
          Breadth Momentum   5%

        Status: NORMAL (<50), ELEVATED (50-70), WARNING (70-85), CRITICAL (>=85).
        """
        print("\n[COMPOSITE] Computing crash score...")

        # Extract macro inputs (guard against None values from API failures)
        vix = macro_data.get('volatility', {}).get('vix')
        if vix is None:
            vix = 15.0
        breadth_pct = macro_data.get('breadth', {}).get('pct_above_200ma')
        if breadth_pct is None:
            breadth_pct = 65.0
        credit_ratio = macro_data.get('credit', {}).get('hyg_lqd_ratio')
        if credit_ratio is None:
            credit_ratio = 0.73

        scores = {}

        # RMT lambda_max (35%)
        if rmt_result:
            lambda_max = rmt_result['lambda_max']
            if lambda_max < 240:
                scores['rmt'] = 0.0
            elif lambda_max > 400:
                scores['rmt'] = 100.0
            else:
                scores['rmt'] = (lambda_max - 240.0) / 160.0 * 100.0
        else:
            scores['rmt'] = 50.0   # unknown

        # Ising magnetization (20%)
        if ising_result:
            magnetization = ising_result['magnetization']
            if magnetization < 0.40:
                scores['ising'] = 0.0
            elif magnetization > 0.85:
                scores['ising'] = 100.0
            else:
                scores['ising'] = (magnetization - 0.40) / 0.45 * 100.0
        else:
            scores['ising'] = 50.0  # unknown

        # Market breadth (20%) — inverted: low breadth = high risk
        if breadth_pct > 80:
            scores['breadth'] = 0.0
        elif breadth_pct < 20:
            scores['breadth'] = 100.0
        else:
            scores['breadth'] = (80.0 - breadth_pct) / 60.0 * 100.0

        # VIX (15%)
        if vix < 10:
            scores['vix'] = 0.0
        elif vix > 40:
            scores['vix'] = 100.0
        else:
            scores['vix'] = (vix - 10.0) / 30.0 * 100.0

        # Credit spreads (5%) — proxy from HYG/LQD ratio
        credit_spread = abs(0.73 - credit_ratio) * 100.0
        scores['credit'] = min(credit_spread * 20.0, 100.0)

        # Breadth momentum (5%) — placeholder
        scores['breadth_momentum'] = 0.0

        weights = {
            'rmt':              0.35,
            'ising':            0.20,
            'breadth':          0.20,
            'vix':              0.15,
            'credit':           0.05,
            'breadth_momentum': 0.05,
        }

        overall_score = sum(scores[k] * weights[k] for k in scores)

        if overall_score >= 85:
            status = 'CRITICAL'
        elif overall_score >= 70:
            status = 'WARNING'
        elif overall_score >= 50:
            status = 'ELEVATED'
        else:
            status = 'NORMAL'

        result = {
            'overall_score':    float(overall_score),
            'status':           status,
            'component_scores': {k: float(v) for k, v in scores.items()},
            'weights':          {k: float(v) for k, v in weights.items()},
        }

        print("[OK] Composite Score = {:.1f} [{}]".format(overall_score, status))

        return result

    # =========================================================================
    # MAIN RUNNER
    # =========================================================================

    def run(self):
        """Execute all computations and return combined results dict."""
        print("=" * 70)
        print("CRASH DETECTION BACKEND")
        print("=" * 70)
        print("Generated: {}".format(datetime.datetime.now().isoformat()))

        prices = self.load_price_cache()
        if prices is None:
            return None

        rmt_result   = self.compute_latest_rmt(prices)
        ising_result = self.compute_latest_ising(prices)

        # Load macro data from parent directory
        macro_data = {}
        macro_path = os.path.join(_DATA_DIR, 'macro_data.json')
        if os.path.exists(macro_path):
            try:
                with open(macro_path, 'r') as fh:
                    macro_data = json.load(fh)
                print("[OK] Loaded macro_data.json")
            except Exception as exc:
                print("[WARN] Could not load macro_data.json: {}".format(exc))
        else:
            print("[WARN] macro_data.json not found at {}; using defaults".format(
                macro_path))

        composite_result = self.compute_composite_score(
            rmt_result, ising_result, macro_data
        )

        return {
            'generated_at': datetime.datetime.now().isoformat(),
            'rmt':       rmt_result,
            'ising':     ising_result,
            'composite': composite_result,
        }


# =============================================================================
# HTML HELPERS
# =============================================================================


def _risk_pill(risk_level):
    """Return an inline HTML pill badge for a risk level string."""
    return '<span class="risk-pill risk-{r}">{r}</span>'.format(r=risk_level)


def _status_color(status):
    return STATUS_COLORS.get(status, '#9ca3af')


def _weight_bar(score_0_100, width_px=120):
    """Inline progress bar representing a 0-100 score."""
    pct = max(0.0, min(100.0, float(score_0_100)))
    return (
        '<span class="weight-bar-bg">'
        '<span class="weight-bar-fill" style="width:{pct:.1f}%;"></span>'
        '</span>'.format(pct=pct)
    )


# =============================================================================
# DASHBOARD BODY BUILDER
# =============================================================================


def build_body_html(output, writer):
    """
    Build the full body HTML string for DashboardWriter.write().

    Parameters
    ----------
    output : dict
        Result from CrashDetectionBackend.run()
    writer : DashboardWriter
    """
    date_str = datetime.date.today().strftime("%Y-%m-%d")

    rmt        = output.get('rmt') or {}
    ising      = output.get('ising') or {}
    composite  = output.get('composite') or {}

    overall_score  = float(composite.get('overall_score', 0.0))
    overall_status = composite.get('status', 'NORMAL')
    comp_scores    = composite.get('component_scores', {})
    comp_weights   = composite.get('weights', {})

    rmt_status   = rmt.get('risk', 'N/A')
    ising_status = ising.get('risk', 'N/A')

    # Stat bar css classes
    def status_css(s):
        return {'NORMAL': 'pos', 'ELEVATED': 'warn', 'WARNING': 'warn',
                'CRITICAL': 'neg'}.get(s, 'neutral')

    parts = []

    # -------------------------------------------------------------------------
    # 1. Stat bar
    # -------------------------------------------------------------------------
    parts.append(writer.stat_bar([
        ("Date",            date_str,                          "neutral"),
        ("Composite Score", "{:.0f}/100".format(overall_score), status_css(overall_status)),
        ("RMT Status",      rmt_status,                        status_css(rmt_status)),
        ("Ising Status",    ising_status,                      status_css(ising_status)),
        ("Overall Status",  overall_status,                    status_css(overall_status)),
    ]))

    # -------------------------------------------------------------------------
    # 2. Page header
    # -------------------------------------------------------------------------
    parts.append(writer.build_header(
        "RMT + Ising &nbsp;|&nbsp; Window: 252 days"
    ))

    # -------------------------------------------------------------------------
    # 3. Regime banner
    # -------------------------------------------------------------------------
    banner_color = _status_color(overall_status)
    rmt_lmax  = float(rmt.get('lambda_max', 0.0))
    rmt_lplus = float(rmt.get('lambda_plus', 0.0))
    ising_mag = float(ising.get('magnetization', 0.0))

    score_html = (
        "Composite: {score:.1f}/100"
        " &nbsp;&bull;&nbsp; "
        "RMT &lambda;<sub>max</sub>: {lmax:.2f} (MP threshold: {lplus:.2f})"
        " &nbsp;&bull;&nbsp; "
        "Ising M: {mag:.3f}"
    ).format(
        score=overall_score,
        lmax=rmt_lmax,
        lplus=rmt_lplus,
        mag=ising_mag,
    )
    parts.append(writer.regime_banner(overall_status, score_html, color=banner_color))

    # -------------------------------------------------------------------------
    # 4. Cards section (3 cards)
    # -------------------------------------------------------------------------
    def _card(border_color, label, value_str, sub_lines):
        """Render a single .card div."""
        subs_html = "".join(
            '<div class="sub">{}</div>'.format(s) for s in sub_lines
        )
        return (
            '<div class="card" style="border-top-color:{bc};">'
            '<div class="label">{lbl}</div>'
            '<div class="value" style="color:{bc};">{val}</div>'
            '{subs}'
            '</div>'
        ).format(bc=border_color, lbl=label, val=value_str, subs=subs_html)

    rmt_color   = _status_color(rmt_status)
    ising_color = _status_color(ising_status)
    comp_color  = _status_color(overall_status)

    cards_html = []

    # RMT card
    if rmt:
        rmt_ratio = float(rmt.get('ratio', 0.0))
        rmt_nsig  = int(rmt.get('n_significant', 0))
        rmt_nass  = int(rmt.get('n_assets', 0))
        cards_html.append(_card(
            rmt_color,
            "RMT Lambda Max",
            "{:.2f}".format(rmt_lmax),
            [
                "vs MP threshold: {:.2f} (ratio {:.2f}x)".format(rmt_lplus, rmt_ratio),
                "Risk Level: {} &nbsp; Significant eigenvalues: {}".format(
                    _risk_pill(rmt_status), rmt_nsig),
                "Assets analyzed: {}".format(rmt_nass),
            ]
        ))
    else:
        cards_html.append(_card(
            '#9ca3af', "RMT Lambda Max", "N/A", ["Insufficient data"]
        ))

    # Ising card
    if ising:
        ising_crash = float(ising.get('crash_score', 0.0))
        ising_nass  = int(ising.get('n_assets', 0))
        cards_html.append(_card(
            ising_color,
            "Ising Magnetization",
            "{:.3f}".format(ising_mag),
            [
                "Crash score: {:.1f}/100".format(ising_crash),
                "Risk Level: {}".format(_risk_pill(ising_status)),
                "Assets analyzed: {}".format(ising_nass),
            ]
        ))
    else:
        cards_html.append(_card(
            '#9ca3af', "Ising Magnetization", "N/A", ["Insufficient data"]
        ))

    # Composite card
    cards_html.append(_card(
        comp_color,
        "Composite Score",
        "{:.1f} / 100".format(overall_score),
        [
            "Overall status: {}".format(_risk_pill(overall_status)),
            "Weighted blend of 6 components",
        ]
    ))

    parts.append('<div class="cards">{}</div>'.format("".join(cards_html)))

    # -------------------------------------------------------------------------
    # 5. Component breakdown table
    # -------------------------------------------------------------------------
    COMPONENT_LABELS = [
        ('rmt',              'RMT Lambda Max'),
        ('ising',            'Ising Magnetization'),
        ('breadth',          'Market Breadth'),
        ('vix',              'VIX'),
        ('credit',           'Credit Spreads (HYG/LQD)'),
        ('breadth_momentum', 'Breadth Momentum'),
    ]

    table_rows_html = []
    for key, label in COMPONENT_LABELS:
        raw   = float(comp_scores.get(key, 0.0))
        wt    = float(comp_weights.get(key, 0.0))
        wtd   = raw * wt

        # Score color
        if raw >= 85:
            score_css = 'neg'
        elif raw >= 50:
            score_css = 'warn'
        else:
            score_css = 'pos'

        table_rows_html.append((
            "<tr>"
            '<td>{lbl}</td>'
            '<td class="num {sc}">{raw:.1f}</td>'
            '<td class="num">{bar}{wt:.0f}%</td>'
            '<td class="num {sc}">{wtd:.2f}</td>'
            "</tr>"
        ).format(
            lbl=label,
            sc=score_css,
            raw=raw,
            bar=_weight_bar(wt * 100.0),
            wt=wt * 100.0,
            wtd=wtd,
        ))

    table_html = (
        '<table class="component-table">'
        '<thead><tr>'
        '<th>Component</th>'
        '<th>Raw Score (0-100)</th>'
        '<th>Weight</th>'
        '<th>Contribution</th>'
        '</tr></thead>'
        '<tbody>{rows}</tbody>'
        '</table>'
    ).format(rows="".join(table_rows_html))

    total_contribution = sum(
        float(comp_scores.get(k, 0.0)) * float(comp_weights.get(k, 0.0))
        for k, _ in COMPONENT_LABELS
    )

    table_html += (
        '<p style="margin-top:14px;font-size:0.9em;color:#555;">'
        'Total composite score: <strong class="{css}">{score:.2f} / 100</strong>'
        ' &nbsp;&mdash;&nbsp; Status: {pill}'
        '</p>'
    ).format(
        css=status_css(overall_status),
        score=total_contribution,
        pill=_risk_pill(overall_status),
    )

    parts.append(writer.section("Component Breakdown", table_html))

    # -------------------------------------------------------------------------
    # 6. Footer
    # -------------------------------------------------------------------------
    parts.append(writer.footer())

    return "\n".join(parts)


# =============================================================================
# MAIN
# =============================================================================


def main():
    print("=" * 70)
    print("CRASH DETECTION DASHBOARD  (DashboardWriter)")
    print("=" * 70)

    backend = CrashDetectionBackend(
        cache_dir=CACHE_DIR,
        window=252,
        min_assets=50,
    )

    output = backend.run()

    if output is None:
        print("[ERROR] Backend returned no data - aborting HTML write.")
        return

    # Summarise
    rmt_res  = output.get('rmt') or {}
    isg_res  = output.get('ising') or {}
    cmp_res  = output.get('composite') or {}

    print()
    print("=" * 70)
    print("CRASH DETECTION SUMMARY")
    print("=" * 70)
    if rmt_res:
        print("RMT lambda_max:  {:.2f} [{}]".format(
            float(rmt_res.get('lambda_max', 0.0)), rmt_res.get('risk', '?')))
    if isg_res:
        print("Ising M:         {:.3f} [{}]".format(
            float(isg_res.get('magnetization', 0.0)), isg_res.get('risk', '?')))
    if cmp_res:
        print("Composite Score: {:.1f}/100 [{}]".format(
            float(cmp_res.get('overall_score', 0.0)), cmp_res.get('status', '?')))
    print("=" * 70)

    # Build and write dashboard HTML
    writer = DashboardWriter("crash-detection", "Crash Detection (RMT + Ising)")
    body   = build_body_html(output, writer)
    writer.write(body, extra_css=EXTRA_CSS, extra_js="")

    # Write CSV
    csv_path = os.path.join(_SCRIPT_DIR, 'crash_detection_data.csv')
    comp_scores = cmp_res.get('component_scores', {})
    csv_row = {
        'date': datetime.date.today().strftime('%Y-%m-%d'),
        'lambda_max': rmt_res.get('lambda_max'),
        'lambda_plus': rmt_res.get('lambda_plus'),
        'rmt_ratio': rmt_res.get('ratio'),
        'n_significant': rmt_res.get('n_significant'),
        'n_assets': rmt_res.get('n_assets'),
        'rmt_risk': rmt_res.get('risk'),
        'magnetization': isg_res.get('magnetization'),
        'avg_correlation': isg_res.get('avg_correlation'),
        'ising_risk': isg_res.get('risk'),
        'composite_score': cmp_res.get('overall_score'),
        'composite_status': cmp_res.get('status'),
        'score_rmt': comp_scores.get('rmt'),
        'score_ising': comp_scores.get('ising'),
        'score_breadth': comp_scores.get('breadth'),
        'score_vix': comp_scores.get('vix'),
        'score_credit': comp_scores.get('credit'),
    }
    pd.DataFrame([csv_row]).to_csv(csv_path, index=False, encoding='utf-8')
    print("CSV: {}".format(csv_path))


if __name__ == "__main__":
    main()
