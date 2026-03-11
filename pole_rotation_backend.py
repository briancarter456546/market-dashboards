# -*- coding: utf-8 -*-
# =============================================================================
# pole_rotation_backend.py - v2.0
# Last updated: 2026-03-11
# =============================================================================
# v2.0: Complete rewrite - Entry/Exit dashboard style
#   - Uses 25 scimode-validated poles (not RSI(2) PF filter)
#   - Lead ETF per pole with SMA29 extension + zone badge
#   - VIX regime banner with scimode-validated sector recommendations
#   - Sortable table: pole name, lead ETF, classification, 1W/1M/3M/6M/YTD,
#     extension %, zone badge, coherence, # stocks, SPY beta
#   - Stat bar + regime banner matching entry/exit dashboard pattern
#
# v1.0: Heatmap cards filtered by RSI(2) PF >= 1.5
# =============================================================================

import os
import json
import pickle
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

from dashboard_writer import DashboardWriter

warnings.filterwarnings('ignore')

# =============================================================================
# PATH SETUP
# =============================================================================

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR   = os.path.normpath(os.path.join(_SCRIPT_DIR, '..', 'perplexity-user-data'))
CACHE_DIR   = os.path.normpath(os.path.join(_DATA_DIR, 'price_cache'))

FMP_RETURNS_PATH = os.path.normpath(os.path.join(
    _DATA_DIR, 'output', 'taxonomy', 'fmp_pole_returns_raw.csv'))
POLE_META_PATH   = os.path.normpath(os.path.join(
    _DATA_DIR, 'output', 'taxonomy', 'fmp_pole_metadata.json'))
VALIDATED_POLES_JSON = os.path.normpath(os.path.join(
    _DATA_DIR, 'output', 'scientist', 'validated_poles.json'))
VIX_PKL = os.path.join(CACHE_DIR, '^VIX.pkl')

# =============================================================================
# VIX REGIME (same as sma29_entry_backend v1.4)
# =============================================================================

VIX_BANDS = [
    (0,  15, '<15',   'LOW VOLATILITY'),
    (15, 20, '15-20', 'NORMAL'),
    (20, 25, '20-25', 'ELEVATED'),
    (25, 30, '25-30', 'HIGH'),
    (30, 999, '30+',  'EXTREME'),
]

VIX_SECTOR_RECS = {
    '<15':  ['Semis & Tech', 'Financials', 'Defense', 'LatAm'],
    '15-20': ['Semis & Tech', 'Financials', 'Cybersecurity', 'Consumer Disc'],
    '20-25': ['Telecom', 'Semis & Tech', 'LatAm', 'Financials'],
    '25-30': ['Telecom', 'Semis & Tech', 'LatAm', 'Energy'],
    '30+':   ['Semis & Tech', 'Energy', 'Copper & Metals', 'Defense'],
}

# Pole IDs that are favored per VIX band (for highlighting)
VIX_FAVORED_POLES = {
    '<15':  {16, 19, 11, 13},
    '15-20': {16, 19, 27, 37},
    '20-25': {18, 16, 13, 19},
    '25-30': {18, 16, 13, 9},
    '30+':   {16, 9, 45, 11},
}

# Lead ETF per pole (first/most liquid member)
LEAD_ETF = {
    1: 'VGK', 2: 'XLP', 3: 'IWM', 4: 'AGG', 5: 'GLD',
    6: 'FXI', 8: 'BITQ', 9: 'USO', 10: 'HEZU', 11: 'ITA',
    13: 'EWZ', 14: 'VNQ', 15: 'EWJ', 16: 'SMH', 18: 'XLC',
    19: 'XLF', 20: 'HYG', 27: 'CIBR', 28: 'FLTW', 29: 'URA',
    32: 'TAN', 34: 'XLB', 37: 'XLY', 42: 'GXG', 45: 'COPX',
}

# SMA29 extension zone buckets (simplified from sma29_entry_backend)
EXTENSION_ZONES = [
    (0,   5,  'OPTIMAL', '#166534', '#dcfce7'),
    (5,  10,  'OPTIMAL', '#166534', '#dcfce7'),
    (10, 15,  'GOOD',    '#075985', '#e0f2fe'),
    (15, 20,  'FAIR',    '#854d0e', '#fef9c3'),
    (20, 25,  'CAUTION', '#9a3412', '#fed7aa'),
    (25, 40,  'WARNING', '#991b1b', '#fecaca'),
    (40, 999, 'DANGER',  '#fff',    '#f87171'),
]


# =============================================================================
# DATA LOADERS
# =============================================================================

def load_validated_poles():
    """Load validated_poles.json -> usable pole IDs + details."""
    if not os.path.exists(VALIDATED_POLES_JSON):
        print("[WARN] validated_poles.json not found")
        return [], {}
    with open(VALIDATED_POLES_JSON, 'r', encoding='utf-8') as f:
        vp = json.load(f)
    usable = vp.get('usable', [])
    details = vp.get('details', {})
    return usable, details


def load_pole_metadata():
    """Load pole metadata JSON."""
    with open(POLE_META_PATH, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items() if k.isdigit()}


def load_current_vix():
    """Load current VIX level."""
    if not os.path.exists(VIX_PKL):
        return None, None, None
    try:
        vdf = pd.read_pickle(VIX_PKL)
        col = 'adjClose' if 'adjClose' in vdf.columns else 'close'
        current_vix = float(vdf[col].iloc[-1])
        for vlo, vhi, band_label, band_desc in VIX_BANDS:
            if vlo <= current_vix < vhi:
                return current_vix, band_label, band_desc
        return current_vix, '30+', 'EXTREME'
    except Exception:
        return None, None, None


def compute_pole_returns(usable_ids):
    """Compute cumulative returns for each pole from FMP log returns."""
    df = pd.read_csv(FMP_RETURNS_PATH, encoding='utf-8', parse_dates=['date'])
    df = df.set_index('date').sort_index()

    usable_set = set(usable_ids)
    col_map = {}
    for col in df.columns:
        parts = col.split('_', 1)
        if parts[0].isdigit() and int(parts[0]) in usable_set:
            col_map[int(parts[0])] = col

    today = df.index[-1]
    windows = {'1W': 5, '1M': 21, '3M': 63, '6M': 126, 'YTD': None}

    results = {}
    for pid in usable_ids:
        if pid not in col_map:
            continue
        series = df[col_map[pid]].dropna()
        if len(series) < 5:
            continue

        r = {}
        for label, days in windows.items():
            if label == 'YTD':
                year_start = pd.Timestamp(today.year, 1, 1)
                ytd = series[series.index >= year_start]
                r['YTD'] = (np.exp(ytd.sum()) - 1) * 100 if len(ytd) > 0 else 0.0
            else:
                tail = series.tail(days)
                r[label] = (np.exp(tail.sum()) - 1) * 100 if len(tail) > 0 else 0.0

        r['last_date'] = series.index[-1].strftime('%Y-%m-%d')
        results[pid] = r

    return results


def compute_lead_etf_extension(lead_etfs):
    """Compute SMA29 extension for each pole's lead ETF."""
    ext_data = {}
    for pid, ticker in lead_etfs.items():
        pkl = os.path.join(CACHE_DIR, '{}.pkl'.format(ticker))
        if not os.path.exists(pkl):
            continue
        try:
            df = pd.read_pickle(pkl)
            col = 'adjClose' if 'adjClose' in df.columns else 'close'
            close = df[col].dropna()
            if len(close) < 29:
                continue
            sma29 = close.rolling(29).mean()
            latest = float(close.iloc[-1])
            sma_val = float(sma29.iloc[-1])
            if np.isnan(sma_val) or sma_val <= 0:
                continue
            ext_pct = ((latest - sma_val) / sma_val) * 100.0
            ext_data[pid] = {
                'price': round(latest, 2),
                'sma29': round(sma_val, 2),
                'ext_pct': round(ext_pct, 2),
            }
        except Exception:
            continue
    return ext_data


def classify_extension(ext_pct):
    """Map extension % to zone label + colors."""
    if ext_pct < 0:
        return 'BELOW SMA', '#6b7280', '#e5e7eb'
    for lo, hi, label, fg, bg in EXTENSION_ZONES:
        if lo <= ext_pct < hi:
            return label, fg, bg
    return 'DANGER', '#fff', '#f87171'


# =============================================================================
# CSS
# =============================================================================

EXTRA_CSS = """
/* Classification badges */
.cls-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.72em;
    font-weight: 700;
    letter-spacing: 0.5px;
}
.cls-VALIDATED { background: #dcfce7; color: #166534; }
.cls-MARGINAL  { background: #fef9c3; color: #854d0e; }

/* Extension zone badges */
.ext-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.78em;
    font-weight: 700;
    letter-spacing: 0.5px;
}

/* VIX favored highlight */
.vix-favored {
    background: linear-gradient(90deg, #fef3c720, transparent);
}
.vix-star {
    color: #d97706;
    font-size: 0.82em;
}

/* Compact table */
#poleTable { font-size: 0.82em; }
#poleTable th, #poleTable td { padding: 5px 8px; white-space: nowrap; }
#poleTable th { cursor: pointer; border-bottom: 1px dashed #999; }

/* Filter buttons */
.filter-btn {
    margin: 2px 4px;
    padding: 4px 12px;
    border: 1px solid #ccc;
    border-radius: 4px;
    cursor: pointer;
    background: #f3f4f6;
    font-size: 0.85em;
}
.filter-btn.active {
    background: #1a1a2e;
    color: #fff;
    border-color: #1a1a2e;
}
"""

EXTRA_JS = """
// Sort by column
function sortPoleTable(colIdx, numeric) {
    var table = document.getElementById('poleTable');
    var tbody = table.tBodies[0];
    var rows = Array.from(tbody.rows);
    var asc = table.getAttribute('data-sort-col') == colIdx
              && table.getAttribute('data-sort-dir') == 'asc';
    var keyed = rows.map(function(r) {
        var v = r.cells[colIdx].getAttribute('data-val');
        if (v === null) v = r.cells[colIdx].textContent;
        if (numeric) { var p = parseFloat(v); v = isNaN(p) ? -9999 : p; }
        return {row: r, val: v};
    });
    keyed.sort(function(a, b) {
        if (a.val < b.val) return asc ? 1 : -1;
        if (a.val > b.val) return asc ? -1 : 1;
        return 0;
    });
    table.setAttribute('data-sort-col', colIdx);
    table.setAttribute('data-sort-dir', asc ? 'desc' : 'asc');
    var frag = document.createDocumentFragment();
    keyed.forEach(function(k) { frag.appendChild(k.row); });
    tbody.appendChild(frag);
}

// Filter by classification
function filterClass(cls) {
    var btns = document.querySelectorAll('.filter-btn');
    btns.forEach(function(b) { b.classList.remove('active'); });
    if (cls !== 'ALL') {
        event.target.classList.add('active');
    }
    var rows = document.getElementById('poleTable').tBodies[0].rows;
    for (var i = 0; i < rows.length; i++) {
        if (cls === 'ALL') {
            rows[i].style.display = '';
        } else if (cls === 'FAVORED') {
            rows[i].style.display = rows[i].classList.contains('vix-favored') ? '' : 'none';
        } else {
            var rc = rows[i].getAttribute('data-cls') || '';
            rows[i].style.display = (rc === cls) ? '' : 'none';
        }
    }
}
"""


# =============================================================================
# HTML BUILDER
# =============================================================================

def build_html(pole_data, vix_context):
    """Build the dashboard HTML."""
    current_vix = vix_context.get('current_vix')
    vix_band = vix_context.get('vix_band')
    vix_desc = vix_context.get('vix_desc')
    favored_ids = VIX_FAVORED_POLES.get(vix_band, set()) if vix_band else set()

    # Stats
    n_poles = len(pole_data)
    n_validated = sum(1 for p in pole_data if p['classification'] == 'VALIDATED')
    n_marginal = sum(1 for p in pole_data if p['classification'] == 'MARGINAL')
    hot_1m = sum(1 for p in pole_data if p.get('ret_1m', 0) > 0)
    cold_1m = n_poles - hot_1m
    n_optimal = sum(1 for p in pole_data if p.get('zone') == 'OPTIMAL')
    n_favored = sum(1 for p in pole_data if p['pole_id'] in favored_ids)

    # Build DashboardWriter
    dw = DashboardWriter('pole-rotation', 'Validated Pole Rotation')

    # Stat bar
    stat_bar = dw.stat_bar([
        ('Validated', str(n_validated), 'pos'),
        ('Marginal', str(n_marginal), 'warn'),
        ('Hot (1M+)', str(hot_1m), 'pos' if hot_1m > cold_1m else 'warn'),
        ('Cold (1M-)', str(cold_1m), 'neg' if cold_1m > hot_1m else 'neutral'),
        ('Optimal Zone', str(n_optimal), 'pos' if n_optimal > 5 else 'warn'),
        ('VIX Favored', str(n_favored), 'pos'),
    ])

    # Regime banner
    if hot_1m > cold_1m * 1.5:
        banner_text = "BROAD STRENGTH - {}/{} poles positive on 1M".format(hot_1m, n_poles)
        banner_color = '#22c55e'
    elif hot_1m > cold_1m:
        banner_text = "MIXED LEAN POSITIVE - {}/{} poles positive on 1M".format(hot_1m, n_poles)
        banner_color = '#f59e0b'
    else:
        banner_text = "WEAK ROTATION - Only {}/{} poles positive on 1M".format(hot_1m, n_poles)
        banner_color = '#ef4444'
    banner_score = '25 scimode-validated poles | Lead ETF extension + VIX regime overlay'

    # VIX banner
    vix_banner = ''
    if current_vix is not None:
        vix_color_map = {'<15': '#22c55e', '15-20': '#64748b', '20-25': '#f59e0b',
                         '25-30': '#ef4444', '30+': '#991b1b'}
        vix_color = vix_color_map.get(vix_band, '#64748b')
        recs = VIX_SECTOR_RECS.get(vix_band, [])
        recs_str = ', '.join(recs) if recs else 'N/A'
        vix_banner = """
        <div style="background:linear-gradient(135deg, {color}22, {color}11);
                    border-left:4px solid {color}; padding:10px 16px; margin-bottom:14px;
                    border-radius:4px; font-size:0.88em">
            <span style="font-weight:700; color:{color}">VIX {vix:.1f} ({desc})</span>
            &nbsp;|&nbsp; Best sectors: <b>{recs}</b>
            &nbsp;|&nbsp; <span style="color:#d97706">&#9733;</span> = VIX-favored pole
            <span style="float:right;font-size:0.82em;color:#888">scimode_vix_sector_v1_0</span>
        </div>""".format(color=vix_color, vix=current_vix, desc=vix_desc, recs=recs_str)

    # Filter buttons
    filter_html = '<div style="margin-bottom:14px">'
    filter_html += '<button class="filter-btn" onclick="filterClass(\'ALL\')">ALL ({})</button>'.format(n_poles)
    filter_html += '<button class="filter-btn" onclick="filterClass(\'VALIDATED\')">VALIDATED ({})</button>'.format(n_validated)
    filter_html += '<button class="filter-btn" onclick="filterClass(\'MARGINAL\')">MARGINAL ({})</button>'.format(n_marginal)
    if n_favored > 0:
        filter_html += '<button class="filter-btn" onclick="filterClass(\'FAVORED\')">VIX FAVORED ({})</button>'.format(n_favored)
    filter_html += '</div>'

    # Table
    header = """<table id="poleTable" class="dash-table" data-sort-col="0" data-sort-dir="desc">
    <thead><tr>
        <th onclick="sortPoleTable(0,false)" title="Pole name from taxonomy regression">Pole</th>
        <th onclick="sortPoleTable(1,false)" title="Lead ETF for this pole (most liquid member)">Lead ETF</th>
        <th onclick="sortPoleTable(2,false)" title="Scimode validation: VALIDATED (4/4 tests) or MARGINAL (3/4)">Class</th>
        <th onclick="sortPoleTable(3,true)" title="1-week return of pole portfolio (equal-weighted FMP)">1W</th>
        <th onclick="sortPoleTable(4,true)" title="1-month return">1M</th>
        <th onclick="sortPoleTable(5,true)" title="3-month return">3M</th>
        <th onclick="sortPoleTable(6,true)" title="6-month return">6M</th>
        <th onclick="sortPoleTable(7,true)" title="Year-to-date return">YTD</th>
        <th onclick="sortPoleTable(8,true)" title="Lead ETF % above 29-day SMA. Shows how extended this sector is.">Ext %</th>
        <th onclick="sortPoleTable(9,false)" title="SMA29 extension zone (OPTIMAL = 0-10% above SMA29)">Zone</th>
        <th onclick="sortPoleTable(10,true)" title="Pairwise correlation coherence (how tightly members move together). Higher = more reliable pole.">Coher</th>
        <th onclick="sortPoleTable(11,true)" title="Number of stocks assigned to this pole in the taxonomy">Stocks</th>
        <th onclick="sortPoleTable(12,true)" title="Average SPY beta of pole ETF members">Beta</th>
        <th title="All ETF members defining this pole">Members</th>
    </tr></thead>
    <tbody>"""

    rows_html = []
    for p in pole_data:
        pid = p['pole_id']
        is_favored = pid in favored_ids
        cls = p['classification']

        # Return cells
        def ret_td(val, col_idx):
            if val is None:
                return '<td class="tr" data-val="-9999">-</td>'
            color = '#22c55e' if val > 0 else '#ef4444' if val < 0 else '#888'
            return '<td class="tr" data-val="{v}" style="color:{c}">{v:+.1f}%</td>'.format(
                v=val, c=color)

        # Extension zone badge
        ext_pct = p.get('ext_pct')
        if ext_pct is not None:
            zone, zone_fg, zone_bg = classify_extension(ext_pct)
            ext_str = '{:+.1f}%'.format(ext_pct)
            ext_color = '#22c55e' if ext_pct >= 0 else '#ef4444'
            zone_badge = '<span class="ext-badge" style="background:{bg};color:{fg}">{z}</span>'.format(
                bg=zone_bg, fg=zone_fg, z=zone)
        else:
            ext_str = '-'
            ext_color = '#888'
            zone_badge = '-'
            zone = ''
            ext_pct = -999

        # Classification badge
        cls_badge = '<span class="cls-badge cls-{c}">{c}</span>'.format(c=cls)

        # VIX favored star
        star = ' <span class="vix-star" title="VIX-favored in current regime">&#9733;</span>' if is_favored else ''

        # Row class
        row_cls = 'vix-favored' if is_favored else ''

        tr = '<tr class="{rc}" data-cls="{cls}">'.format(rc=row_cls, cls=cls)
        tr += '<td class="tl"><b>{name}</b>{star}</td>'.format(name=p['label'], star=star)
        tr += '<td class="tc" style="font-family:monospace;font-size:0.82em">{}</td>'.format(p.get('lead_etf', '-'))
        tr += '<td class="tc">{}</td>'.format(cls_badge)
        tr += ret_td(p.get('ret_1w'), 3)
        tr += ret_td(p.get('ret_1m'), 4)
        tr += ret_td(p.get('ret_3m'), 5)
        tr += ret_td(p.get('ret_6m'), 6)
        tr += ret_td(p.get('ret_ytd'), 7)
        tr += '<td class="tr" data-val="{}" style="color:{}">{}</td>'.format(
            ext_pct, ext_color, ext_str)
        tr += '<td class="tc" data-val="{}">{}</td>'.format(
            zone if zone != 'BELOW SMA' else 'ZBELOW', zone_badge)
        tr += '<td class="tr" data-val="{:.3f}">{:.2f}</td>'.format(
            p.get('coherence', 0) or 0, p.get('coherence', 0) or 0)
        tr += '<td class="tr" data-val="{}">{}</td>'.format(p.get('n_stocks', 0), p.get('n_stocks', 0))
        tr += '<td class="tr" data-val="{:.2f}">{:.2f}</td>'.format(
            p.get('beta', 0), p.get('beta', 0))
        tr += '<td class="tl" style="font-size:0.72em;color:#888">{}</td>'.format(
            ', '.join(p.get('members', [])[:5]))
        tr += '</tr>'
        rows_html.append(tr)

    table_html = header + '\n'.join(rows_html) + '</tbody></table>'

    # Methodology
    methodology = """
    <details style="margin-bottom:16px">
    <summary style="cursor:pointer;font-weight:600;color:#4a5568">Methodology (scimode-validated)</summary>
    <div style="padding:8px 12px;font-size:0.82em;color:#555;line-height:1.6">
    <p><b>Pole Source:</b> 43 ETF poles from taxonomy_stock_regression (OLS on SPY residuals -> OLS on FMPs).
    Each pole is an equal-weighted portfolio of 2-5 ETFs that define a market factor.</p>
    <p><b>Validation:</b> scimode_pole_validation_v1_0.py tested each pole on 4 criteria:
    coherence (63d pairwise correlation), stability (half-history comparison),
    predictive value (do trending pole members outperform?), and redundancy check.
    VALIDATED = passed 4/4, MARGINAL = 3/4, NOISE = 2 or fewer (excluded).</p>
    <p><b>VIX Regime:</b> scimode_vix_sector_v1_0.py tested 21d forward returns by VIX band x pole.
    Stars mark poles historically strongest in the current VIX regime.</p>
    <p><b>Extension:</b> Lead ETF's (Close - SMA29) / SMA29. Same zones as Entry/Exit dashboard.
    OPTIMAL = 0-10% above SMA29 (best forward PF).</p>
    <p><b>Lead ETF:</b> Most liquid/common ETF in each pole's member basket.</p>
    </div>
    </details>
    """

    # Assemble
    parts = []
    parts.append(dw.build_header(subtitle='Scimode-Validated | 25 Poles | VIX Regime Overlay'))
    parts.append(stat_bar)
    parts.append(dw.regime_banner(banner_text, banner_score, color=banner_color))
    if vix_banner:
        parts.append(vix_banner)
    parts.append(dw.section('Methodology', methodology, hint='scimode_pole_validation + scimode_vix_sector'))
    parts.append(filter_html)
    parts.append(dw.section('Pole Performance', table_html, hint='Click headers to sort | Default: 1M descending'))

    # LLM block
    llm_html = dw.llm_block()
    if llm_html:
        parts.append(llm_html)

    parts.append(dw.footer())

    body = '\n'.join(parts)
    dw.write(body, extra_css=EXTRA_CSS, extra_js=EXTRA_JS)
    return dw.index_path


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("VALIDATED POLE ROTATION v2.0")
    print("=" * 70)

    # Load validated poles
    print("[1/5] Loading validated poles...")
    usable_ids, pole_details = load_validated_poles()
    if not usable_ids:
        print("[FAIL] No validated poles found")
        return 1
    print("  {} usable poles".format(len(usable_ids)))

    # Load pole metadata
    print("[2/5] Loading pole metadata...")
    pole_meta = load_pole_metadata()

    # Load VIX
    print("[3/5] Loading VIX regime...")
    current_vix, vix_band, vix_desc = load_current_vix()
    if current_vix is not None:
        print("  VIX = {:.1f} ({} / {})".format(current_vix, vix_band, vix_desc))

    # Compute pole returns
    print("[4/5] Computing pole returns from FMP data...")
    pole_returns = compute_pole_returns(usable_ids)
    print("  Returns computed for {} poles".format(len(pole_returns)))

    # Compute lead ETF extensions
    print("[5/5] Computing lead ETF extensions...")
    lead_etfs = {pid: LEAD_ETF[pid] for pid in usable_ids if pid in LEAD_ETF}
    ext_data = compute_lead_etf_extension(lead_etfs)
    print("  Extensions computed for {} ETFs".format(len(ext_data)))

    # Assemble pole data rows
    pole_data = []
    for pid in usable_ids:
        meta = pole_meta.get(pid, {})
        detail = pole_details.get(str(pid), {})
        ret = pole_returns.get(pid, {})
        ext = ext_data.get(pid, {})

        row = {
            'pole_id': pid,
            'label': detail.get('label', meta.get('pole_label', 'Pole {}'.format(pid))),
            'classification': detail.get('classification', 'MARGINAL'),
            'coherence': detail.get('coherence'),
            'n_stocks': detail.get('n_stocks', 0),
            'lead_etf': LEAD_ETF.get(pid, '-'),
            'members': meta.get('members', []),
            'beta': meta.get('avg_spy_beta', 0),
            'ret_1w': ret.get('1W'),
            'ret_1m': ret.get('1M'),
            'ret_3m': ret.get('3M'),
            'ret_6m': ret.get('6M'),
            'ret_ytd': ret.get('YTD'),
            'ext_pct': ext.get('ext_pct'),
            'zone': classify_extension(ext.get('ext_pct', -999))[0] if ext.get('ext_pct') is not None else None,
        }
        pole_data.append(row)

    # Sort by 1M return descending
    pole_data.sort(key=lambda r: r.get('ret_1m') or -999, reverse=True)

    vix_context = {
        'current_vix': current_vix,
        'vix_band': vix_band,
        'vix_desc': vix_desc,
    }

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    hot = sum(1 for p in pole_data if (p.get('ret_1m') or 0) > 0)
    print("  Poles: {} ({} hot, {} cold on 1M)".format(len(pole_data), hot, len(pole_data) - hot))
    if current_vix is not None:
        favored = VIX_FAVORED_POLES.get(vix_band, set())
        print("  VIX regime: {:.1f} ({})".format(current_vix, vix_desc))
        fav_names = [p['label'] for p in pole_data if p['pole_id'] in favored]
        print("  VIX-favored: {}".format(', '.join(fav_names)))
    print()
    print("  Top 5 by 1M:")
    for p in pole_data[:5]:
        print("    {:30s} 1M={:+.1f}%  ext={:+.1f}%  [{}]".format(
            p['label'], p.get('ret_1m') or 0, p.get('ext_pct') or 0,
            p.get('zone') or '-'))

    out_path = build_html(pole_data, vix_context)
    print()
    print("[OK] Dashboard: {}".format(out_path))
    return 0


if __name__ == '__main__':
    exit(main())
