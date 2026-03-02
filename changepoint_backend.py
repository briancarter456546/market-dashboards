# -*- coding: utf-8 -*-
# ============================================================================
# changepoint_backend.py - v1.0
# Last updated: 2026-03-01
# ============================================================================
# v1.0: DashboardWriter wrapper for regime_changepoint_detector_v1_6.py
#       Imports computation functions from v1.6, builds light-theme HTML
#       with D3 charts via DashboardWriter shared CSS/layout.
# ============================================================================

import json
import os
import sys
import datetime

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR   = os.path.normpath(os.path.join(_SCRIPT_DIR, '..', 'perplexity-user-data'))

# Add root dir so we can import the changepoint detector
sys.path.insert(0, _ROOT_DIR)

import regime_changepoint_detector_v1_6 as cpd

from dashboard_writer import DashboardWriter

# ---------------------------------------------------------------------------
# EXTRA CSS -- light-theme overrides for D3 charts
# ---------------------------------------------------------------------------
EXTRA_CSS = """
/* D3 chart containers */
#chart-container, #sim-chart-container, #pole-chart-container {
    padding: 20px 24px;
    overflow-x: auto;
}
.axis text { fill: #555; font-size: 11px; font-family: 'IBM Plex Mono', monospace; }
.axis path, .axis line { stroke: #ccc; }
.grid line { stroke: #e8e8ec; stroke-dasharray: 2,2; }

/* Legend bar under main chart */
.chart-legend {
    display: flex; flex-wrap: wrap; gap: 16px; padding: 12px 24px;
    border-top: 1px solid #e5e7eb;
}
.legend-item { display: flex; align-items: center; gap: 6px;
               font-size: 0.78em; color: #888; }
.legend-swatch { width: 24px; height: 3px; border-radius: 2px; }

/* Changepoint table */
.cp-table { width: 100%; border-collapse: collapse; font-size: 0.85em; }
.cp-table th {
    padding: 8px 14px; font-size: 0.72em; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.06em;
    background: #f4f5f7; color: #666;
    border-bottom: 2px solid #e5e7eb; text-align: left;
}
.cp-table td {
    padding: 8px 14px; border-bottom: 1px solid #f0f0f3;
    font-family: 'IBM Plex Mono', monospace; vertical-align: top;
}
.cp-table tr:hover { background: #f8f9fb; }
.cp-date { color: #4f46e5; font-weight: 700; }
.cp-cusum { color: #d97706; }

/* Window grid cards */
.window-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
    gap: 12px;
    padding: 20px 24px;
}
.window-card {
    border-radius: 6px;
    padding: 14px 16px;
    border-left: 4px solid;
    background: #fff;
}
.window-card .wname { font-size: 0.78em; font-weight: 700; text-transform: uppercase;
                      letter-spacing: 0.08em; margin-bottom: 4px; }
.window-card .wdesc { font-size: 0.82em; color: #888; margin-bottom: 6px; }
.window-card .wdates { font-size: 0.75em; font-family: 'IBM Plex Mono', monospace;
                       color: #999; }

/* Pole selector buttons */
.pole-selector {
    display: flex; align-items: center; gap: 8px;
    padding: 12px 24px 0; flex-wrap: wrap;
}
.sel-label {
    font-size: 0.68em; font-weight: 700; letter-spacing: 0.12em;
    color: #999; font-family: 'IBM Plex Mono', monospace;
}
.sel-btn {
    background: #fff; border: 1px solid #d1d5db; border-radius: 4px;
    color: #666; font-size: 0.75em; font-family: 'IBM Plex Mono', monospace;
    font-weight: 600; letter-spacing: 0.06em;
    padding: 4px 14px; cursor: pointer; transition: all 0.15s ease;
}
.sel-btn:hover { border-color: #4f46e5; color: #4f46e5; }
.sel-btn.active {
    background: #eef2ff; border-color: #4f46e5; color: #4f46e5;
}
.sel-hint {
    font-size: 0.70em; color: #999;
    font-family: 'IBM Plex Mono', monospace; margin-left: 4px;
}

/* Tooltip */
.cp-tooltip {
    position: absolute;
    background: #fff;
    border: 1px solid #d1d5db;
    border-radius: 6px;
    padding: 10px 14px;
    font-size: 0.82em;
    pointer-events: none;
    opacity: 0;
    transition: opacity 0.15s;
    max-width: 320px;
    z-index: 100;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}

/* Alert banner */
.alert-banner {
    background: #fffbeb;
    border-bottom: 1px solid #fde68a;
    padding: 10px 24px;
    font-size: 0.85em;
    color: #92400e;
    font-family: 'IBM Plex Mono', monospace;
}
.alert-banner .al-label { color: #888; margin-right: 8px; }
"""

# ---------------------------------------------------------------------------
# Light-theme window colors (same hues but lighter for white bg)
# ---------------------------------------------------------------------------
WINDOW_COLORS_LIGHT = [
    '#6b7db0', '#4d8a6d', '#b05050', '#6d6d9a', '#9a854d',
    '#4d8a8a', '#8a4d8a', '#4d6d4d', '#8a6d4d', '#9a4d6d',
]


# ---------------------------------------------------------------------------
# Build body HTML
# ---------------------------------------------------------------------------
def build_body(results, writer):
    """Build dashboard body HTML using DashboardWriter helpers."""
    r = results

    # --- Stat bar ---
    stats = [
        ('Changepoints',       str(r['n_changepoints']),  'warn'),
        ('Distance Max',       '{:.4f}'.format(r['dist_max']),    'neutral'),
        ('Distance Mean',      '{:.4f}'.format(r['dist_mean']),   'neutral'),
        ('CUSUM Threshold',    '{:.2f}'.format(r['cusum_threshold']), 'warn'),
        ('Macro Windows',      str(len(cpd.MACRO_WINDOWS)),       'neutral'),
        ('Feature Dims',       str(r['n_features']),              'neutral'),
        (r['drift_label'],     r['drift_str'],                    r['drift_css']),
        ('Most Similar',       r['top_regime'],                   'neutral'),
        ('Similarity',         r['top_sim'],                      'pos'),
    ]

    parts = []
    parts.append(writer.build_header(
        subtitle='v1.6 | Cosine distance + CUSUM | {} breakpoints'.format(r['n_changepoints'])
    ))
    parts.append(writer.stat_bar(stats))

    # Alert banner
    parts.append(
        '<div class="alert-banner">'
        '<span class="al-label">NOV 2025 - FEB 2026:</span>{}'
        '</div>'.format(r['recent_summary'])
    )

    # D3 CDN
    parts.append('<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>')

    # --- Main distance chart ---
    legend_html = """
    <div class="chart-legend">
      <div class="legend-item"><div class="legend-swatch" style="background:#3b82f6"></div><span>Distance (smoothed 3d median)</span></div>
      <div class="legend-item"><div class="legend-swatch" style="background:#d97706"></div><span>CUSUM statistic</span></div>
      <div class="legend-item"><div class="legend-swatch" style="background:#ef4444;height:1px;border-top:2px dashed #ef4444"></div><span>CUSUM threshold</span></div>
      <div class="legend-item"><div class="legend-swatch" style="background:#7c3aed"></div><span>Detected changepoints</span></div>
      <div class="legend-item"><div class="legend-swatch" style="background:#d97706"></div><span>Drift rate (normalised)</span></div>
      <div class="legend-item"><div class="legend-swatch" style="background:#10b981;border-top:2px dashed #10b981"></div><span>SPY (normalized scale)</span></div>
      <div class="legend-item"><div class="legend-swatch" style="background:#4f46e5"></div><span>Historical events</span></div>
      <div class="legend-item" style="font-style:italic"><span>Shaded bands = macro windows</span></div>
    </div>
    """
    chart_html = '<div id="chart-container"><svg id="main-chart"></svg></div>' + legend_html
    parts.append(writer.section(
        'Regime Distance Series + Changepoints',
        chart_html,
        hint='Cosine distance between consecutive 20-day fingerprints -- spikes = regime transitions'
    ))

    # --- Similarity barchart ---
    sim_html = '<div id="sim-chart-container"><svg id="sim-chart"></svg></div>'
    parts.append(writer.section(
        'Regime Similarity -- Current vs Historical Archetypes',
        sim_html,
        hint='Cosine similarity between today\'s pole fingerprint and each window\'s mean archetype'
    ))

    # --- Pole fingerprint ---
    pole_html = """
    <div class="pole-selector" id="pole-selector">
      <span class="sel-label">WINDOW</span>
      <button class="sel-btn active" data-window="20d">20d</button>
      <button class="sel-btn" data-window="63d">63d</button>
      <button class="sel-btn" data-window="126d">126d</button>
      <button class="sel-btn" data-window="252d">252d</button>
      <span class="sel-hint" id="pole-hint">20-day z-score momentum (short-term)</span>
    </div>
    <div id="pole-chart-container"><svg id="pole-chart"></svg></div>
    """
    parts.append(writer.section(
        'Current Pole Fingerprint',
        pole_html,
        hint='z-score momentum per pole -- Bull = positive / Bear = negative'
    ))

    # --- Changepoints table ---
    table_html = """
    <div style="overflow-x:auto">
      <table class="cp-table">
        <thead><tr>
          <th>Date</th><th>CUSUM</th><th>Distance</th>
          <th>Proposed Window</th><th>Pole Flips (top 6)</th>
        </tr></thead>
        <tbody id="cp-tbody"></tbody>
      </table>
    </div>
    """
    parts.append(writer.section(
        'Detected Changepoints',
        table_html,
        hint='Pole flips show which factors drove each transition (pre vs post 30 days)'
    ))

    # --- Macro windows grid ---
    parts.append(writer.section(
        'Proposed Macro Windows',
        '<div class="window-grid" id="window-grid"></div>',
        hint='Compare detected breakpoints to proposed boundaries'
    ))

    # Tooltip div
    parts.append('<div class="cp-tooltip" id="cp-tooltip"></div>')

    parts.append(writer.footer())

    return '\n'.join(parts)


# ---------------------------------------------------------------------------
# Build D3 JavaScript (extra_js)
# ---------------------------------------------------------------------------
def build_js(results):
    """Build all D3 chart JavaScript for extra_js."""
    r = results
    js_parts = []

    js_parts.append("""
// ============================================================================
// DATA (injected from Python)
// ============================================================================
const distData      = {dist_json};
const cusumData     = {cusum_json};
const cpData        = {cp_json};
const windowData    = {window_json};
const spyData       = {spy_json};
const eventsData    = {events_json};
const driftData     = {drift_json};
const normDriftData = {norm_drift_json};
const simData       = {sim_json};
const poleData      = {pole_json};
const poleWindows   = {pole_windows_json};
const DRIFT_ELEVATED = {drift_elevated};
const DRIFT_UNSTABLE = {drift_unstable};
const CUSUM_THRESH   = {cusum_threshold};
""".format(**r['js_vars']))

    # Similarity barchart (light theme)
    js_parts.append("""
// ============================================================================
// REGIME SIMILARITY BARCHART
// ============================================================================
(function() {
  if (!simData || simData.length === 0) return;
  const container = document.getElementById('sim-chart-container');
  const totalW = container.clientWidth || 900;
  const margin = { top: 10, right: 120, bottom: 10, left: 160 };
  const barH = 32, gap = 8;
  const H = simData.length * (barH + gap);
  const W = totalW - margin.left - margin.right;

  const svg = d3.select('#sim-chart')
    .attr('width', totalW).attr('height', H + margin.top + margin.bottom);
  const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

  const maxSim = d3.max(simData, d => d.sim);
  const xScale = d3.scaleLinear().domain([0, Math.max(maxSim, 100)]).range([0, W]);

  simData.forEach((d, i) => {
    const y = i * (barH + gap);
    const bw = xScale(Math.max(0, d.sim));

    // Track
    g.append('rect').attr('x', 0).attr('y', y).attr('width', W).attr('height', barH)
      .attr('fill', '#f4f5f7').attr('rx', 4);
    // Bar
    g.append('rect').attr('x', 0).attr('y', y).attr('width', bw).attr('height', barH)
      .attr('fill', d.color).attr('opacity', i === 0 ? 0.75 : 0.35).attr('rx', 4);
    // Top highlight
    if (i === 0) {
      g.append('rect').attr('x', 0).attr('y', y).attr('width', bw).attr('height', barH)
        .attr('fill', 'none').attr('stroke', d.color).attr('stroke-width', 1.5).attr('rx', 4);
    }
    // Name label
    g.append('text').attr('x', -8).attr('y', y + barH/2 + 1)
      .attr('text-anchor', 'end').attr('dominant-baseline', 'middle')
      .attr('font-size', '12px').attr('font-family', "'IBM Plex Mono', monospace")
      .attr('fill', i === 0 ? d.color : '#666').attr('font-weight', i === 0 ? '700' : '400')
      .text(d.name.replace(/_/g, ' '));
    // Sim % label
    const labelX = bw > 60 ? bw - 8 : bw + 8;
    g.append('text').attr('x', labelX).attr('y', y + barH/2 + 1)
      .attr('text-anchor', bw > 60 ? 'end' : 'start').attr('dominant-baseline', 'middle')
      .attr('font-size', '11px').attr('font-family', "'IBM Plex Mono', monospace")
      .attr('fill', i === 0 ? '#333' : '#888').attr('font-weight', i === 0 ? '700' : '400')
      .text(`${d.sim.toFixed(1)}%`);
    // Tooltip
    g.append('rect').attr('x', 0).attr('y', y).attr('width', W).attr('height', barH)
      .attr('fill', 'transparent').style('cursor', 'default')
      .on('mouseover', function(event) {
        const tip = document.getElementById('cp-tooltip');
        tip.style.opacity = '1';
        tip.innerHTML = `<div style="color:${d.color};font-weight:700;margin-bottom:4px">${d.name.replace(/_/g,' ')}</div>
          <div style="color:#555;font-size:0.85em">${d.desc}</div>
          <div style="color:#888;font-size:0.82em;margin-top:4px">Similarity: <b>${d.sim.toFixed(1)}%</b></div>`;
      })
      .on('mousemove', function(event) {
        const tip = document.getElementById('cp-tooltip');
        tip.style.left = (event.pageX + 12) + 'px';
        tip.style.top = (event.pageY - 20) + 'px';
      })
      .on('mouseout', function() { document.getElementById('cp-tooltip').style.opacity = '0'; });
  });

  // Grid lines
  [25, 50, 75, 100].forEach(v => {
    const xv = xScale(v);
    g.append('line').attr('x1', xv).attr('x2', xv).attr('y1', 0).attr('y2', H)
      .attr('stroke', '#e5e7eb').attr('stroke-dasharray', '2,2');
    g.append('text').attr('x', xv).attr('y', H + 14)
      .attr('text-anchor', 'middle').attr('font-size', '9px').attr('fill', '#aaa').text(`${v}%`);
  });
})();
""")

    # Pole fingerprint chart (light theme)
    js_parts.append("""
// ============================================================================
// POLE FINGERPRINT BARCHART
// ============================================================================
const WINDOW_HINTS = {
  '20d':  '20-day z-score -- short-term momentum, responsive to recent shifts',
  '63d':  '63-day z-score -- quarterly view, smooths noise',
  '126d': '126-day z-score -- half-year trend, regime-level signal',
  '252d': '252-day z-score -- full-year positioning, slow structural picture',
};

function renderPoleChart(windowKey) {
  const poles_all = (poleWindows && poleWindows[windowKey]) ? poleWindows[windowKey] : poleData;
  if (!poles_all || poles_all.length === 0) return;

  const sorted = [...poles_all].sort((a,b) => a.pole.toLowerCase().localeCompare(b.pole.toLowerCase()));
  const topN = Math.min(sorted.length, 32);
  const poles = sorted.slice(0, topN);

  d3.select('#pole-chart').selectAll('*').remove();

  const container = document.getElementById('pole-chart-container');
  const totalW = container.clientWidth || 900;
  const margin = { top: 12, right: 80, bottom: 30, left: 210 };
  const barH = 22, gap = 5;
  const H = poles.length * (barH + gap);
  const W = totalW - margin.left - margin.right;

  const svg = d3.select('#pole-chart')
    .attr('width', totalW).attr('height', H + margin.top + margin.bottom);
  const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

  const absMax = d3.max(poles, d => Math.abs(d.z));
  const xMax = Math.max(absMax * 1.08, 1.5);
  const xScale = d3.scaleLinear().domain([-xMax, xMax]).range([0, W]);
  const xZero = xScale(0);

  // Centre zero line
  g.append('line').attr('x1', xZero).attr('x2', xZero).attr('y1', 0).attr('y2', H)
    .attr('stroke', '#d1d5db').attr('stroke-width', 1);

  // Grid
  [-3,-2,-1,1,2,3].forEach(v => {
    const xv = xScale(v);
    if (xv < 0 || xv > W) return;
    g.append('line').attr('x1', xv).attr('x2', xv).attr('y1', 0).attr('y2', H)
      .attr('stroke', v === -2 || v === 2 ? '#e0e0e0' : '#eee').attr('stroke-dasharray', '2,3');
    g.append('text').attr('x', xv).attr('y', H + 18)
      .attr('text-anchor', 'middle').attr('font-size', '9px')
      .attr('fill', '#aaa').text(v > 0 ? `+${v}s` : `${v}s`);
  });
  g.append('text').attr('x', xZero).attr('y', H + 18)
    .attr('text-anchor', 'middle').attr('font-size', '9px').attr('fill', '#999').text('0');

  // Region labels
  g.append('text').attr('x', xZero + 18).attr('y', -2)
    .attr('font-size', '9px').attr('fill', 'rgba(16,185,129,0.4)').attr('font-weight','700').text('BULL ->');
  g.append('text').attr('x', xZero - 18).attr('y', -2)
    .attr('text-anchor','end').attr('font-size', '9px').attr('fill', 'rgba(239,68,68,0.4)').attr('font-weight','700')
    .text('<- BEAR');

  const tooltip = document.getElementById('cp-tooltip');

  poles.forEach((d, i) => {
    const y = i * (barH + gap);
    const bull = d.z >= 0;
    const intensity = Math.min(Math.abs(d.z) / 3, 1);
    const col = bull
      ? `rgba(16,${Math.round(170 + intensity*60)},129,${0.25 + intensity*0.50})`
      : `rgba(239,${Math.round(100 - intensity*60)},68,${0.25 + intensity*0.50})`;
    const strokeCol = bull ? '#10b981' : '#ef4444';
    const strong = Math.abs(d.z) > 1.5;

    // Track
    g.append('rect').attr('x',0).attr('y',y).attr('width',W).attr('height',barH)
      .attr('fill','#fafafa').attr('rx',3);
    // Bar
    const barX = bull ? xZero : xScale(d.z);
    const barW = Math.abs(xScale(d.z) - xZero);
    if (barW > 0.5) {
      g.append('rect').attr('x', barX).attr('y', y+2)
        .attr('width', barW).attr('height', barH-4).attr('fill', col).attr('rx', 2);
      g.append('rect')
        .attr('x', bull ? barX + barW - 1.5 : barX)
        .attr('y', y+2).attr('width', 1.5).attr('height', barH-4)
        .attr('fill', strokeCol).attr('opacity', strong ? 0.8 : 0.4).attr('rx',1);
    }
    // Label
    g.append('text').attr('x', -8).attr('y', y + barH/2 + 1)
      .attr('text-anchor','end').attr('dominant-baseline','middle')
      .attr('font-size','11px').attr('font-family',"'IBM Plex Mono', monospace")
      .attr('fill', strong ? (bull ? '#059669' : '#dc2626') : '#888')
      .attr('font-weight', strong ? '700' : '400')
      .text(d.pole.replace(/_/g,' '));
    // Z-value
    const labelX = bull ? xZero + barW + 6 : xZero - barW - 6;
    g.append('text').attr('x', labelX).attr('y', y + barH/2 + 1)
      .attr('text-anchor', bull ? 'start' : 'end').attr('dominant-baseline','middle')
      .attr('font-size','10px').attr('font-family',"'IBM Plex Mono', monospace")
      .attr('fill', strong ? (bull ? '#059669' : '#dc2626') : '#aaa')
      .text(d.z > 0 ? `+${d.z.toFixed(2)}` : d.z.toFixed(2));
    // Hover
    g.append('rect').attr('x',0).attr('y',y).attr('width',W).attr('height',barH)
      .attr('fill','transparent')
      .on('mouseover', function(event) {
        tooltip.style.opacity = '1';
        tooltip.innerHTML = `<div style="color:${bull?'#059669':'#dc2626'};font-weight:700">${d.pole}</div>
          <div style="color:#555;font-size:0.85em">z-score (${windowKey}): <b>${d.z > 0 ? '+' : ''}${d.z.toFixed(3)}</b></div>
          <div style="color:#888;font-size:0.82em;margin-top:2px">${bull ? 'Bull momentum' : 'Bear momentum'} -- ${Math.abs(d.z) > 2 ? 'strong' : Math.abs(d.z) > 1 ? 'moderate' : 'weak'} signal</div>`;
        tooltip.style.left = (event.pageX+12)+'px';
        tooltip.style.top  = (event.pageY-20)+'px';
      })
      .on('mousemove', function(event) {
        tooltip.style.left = (event.pageX+12)+'px';
        tooltip.style.top  = (event.pageY-20)+'px';
      })
      .on('mouseout', () => { tooltip.style.opacity='0'; });
  });
}

// Initial render + selector wiring
(function() {
  const availableWindows = Object.keys(poleWindows || {});
  let activeWindow = availableWindows.includes('20d') ? '20d' : (availableWindows[0] || null);
  if (!activeWindow && (!poleData || poleData.length === 0)) return;

  renderPoleChart(activeWindow || '20d');

  document.querySelectorAll('.sel-btn').forEach(btn => {
    const w = btn.dataset.window;
    if (!poleWindows[w] || poleWindows[w].length === 0) {
      btn.disabled = true;
      btn.style.opacity = '0.3';
      btn.style.cursor  = 'not-allowed';
      return;
    }
    btn.addEventListener('click', function() {
      document.querySelectorAll('.sel-btn').forEach(b => b.classList.remove('active'));
      this.classList.add('active');
      activeWindow = w;
      const hintEl = document.getElementById('pole-hint');
      if (hintEl) hintEl.textContent = WINDOW_HINTS[w] || w;
      renderPoleChart(w);
    });
  });
})();
""")

    # Window grid (light theme)
    js_parts.append("""
// ============================================================================
// WINDOW GRID
// ============================================================================
(function() {
  const grid = document.getElementById('window-grid');
  windowData.forEach(w => {
    const card = document.createElement('div');
    card.className = 'window-card';
    card.style.borderColor = w.color;
    card.style.background  = w.color + '12';
    card.innerHTML = `
      <div class="wname" style="color:${w.color}">${w.name}</div>
      <div class="wdesc">${w.desc}</div>
      <div class="wdates">${w.start} -> ${w.end}</div>
    `;
    grid.appendChild(card);
  });
})();
""")

    # Changepoints table
    js_parts.append("""
// ============================================================================
// CHANGEPOINTS TABLE
// ============================================================================
(function() {
  const tbody = document.getElementById('cp-tbody');
  function findWindow(dateStr) {
    for (const w of windowData) {
      if (dateStr >= w.start && dateStr <= w.end) return w;
    }
    return null;
  }
  cpData.forEach(cp => {
    const w = findWindow(cp.date);
    const wCell = w
      ? `<span style="color:${w.color};font-size:0.85em">${w.name}</span>`
      : '<span style="color:#aaa">--</span>';
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td class="cp-date">${cp.date}</td>
      <td class="cp-cusum">${cp.cusum.toFixed(3)}</td>
      <td style="color:#3b82f6">${cp.dist.toFixed(5)}</td>
      <td>${wCell}</td>
      <td class="flip-section">${cp.flip_html || '<span style="color:#aaa">no significant flips</span>'}</td>
    `;
    tbody.appendChild(tr);
  });
  if (cpData.length === 0) {
    const tr = document.createElement('tr');
    tr.innerHTML = '<td colspan="5" style="color:#aaa;text-align:center;padding:20px">No changepoints detected in this date range</td>';
    tbody.appendChild(tr);
  }
})();
""")

    # Main D3 chart (light theme colors)
    js_parts.append("""
// ============================================================================
// MAIN D3 DISTANCE CHART
// ============================================================================
(function() {
  const margin = { top: 30, right: 60, bottom: 50, left: 60 };
  const container = document.getElementById('chart-container');
  const totalW = container.clientWidth || 1100;
  const W = totalW - margin.left - margin.right;
  const H = 340;

  const svg = d3.select('#main-chart')
    .attr('width', totalW).attr('height', H + margin.top + margin.bottom);
  const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

  const parseDate = d3.timeParse('%Y-%m-%d');
  const dist  = distData.map(d => ({ date: parseDate(d.date), val: d.val }));
  const cusum = cusumData.map(d => ({ date: parseDate(d.date), val: d.val }));

  const xExt = d3.extent(dist, d => d.date);
  const xScale = d3.scaleTime().domain(xExt).range([0, W]);

  const distMax = d3.max(dist, d => d.val) * 1.1;
  const yDist   = d3.scaleLinear().domain([0, distMax]).range([H, 0]);

  const cusumMax = Math.max(d3.max(cusum, d => d.val) * 1.1, CUSUM_THRESH * 1.5);
  const yCusum   = d3.scaleLinear().domain([0, cusumMax]).range([H, 0]);

  // Grid
  g.append('g').attr('class', 'grid')
    .call(d3.axisLeft(yDist).tickSize(-W).tickFormat('').ticks(5));

  // Window bands
  const wParser = d3.timeParse('%Y-%m-%d');
  windowData.forEach(w => {
    const x0 = xScale(wParser(w.start));
    const x1 = xScale(wParser(w.end));
    if (x0 >= x1) return;
    g.append('rect').attr('x', x0).attr('y', 0)
      .attr('width', x1 - x0).attr('height', H)
      .attr('fill', w.color).attr('opacity', 0.06);
    const cx = (x0 + x1) / 2;
    if (x1 - x0 > 30) {
      g.append('text').attr('x', cx).attr('y', -8)
        .attr('text-anchor', 'middle').attr('font-size', '9px')
        .attr('fill', w.color).attr('opacity', 0.7)
        .text(w.name.replace(/_/g, ' '));
    }
  });

  // CUSUM threshold
  g.append('line').attr('x1', 0).attr('x2', W)
    .attr('y1', yCusum(CUSUM_THRESH)).attr('y2', yCusum(CUSUM_THRESH))
    .attr('stroke', '#ef4444').attr('stroke-width', 1)
    .attr('stroke-dasharray', '4,3').attr('opacity', 0.6);
  g.append('text').attr('x', W + 4).attr('y', yCusum(CUSUM_THRESH) + 4)
    .attr('font-size', '9px').attr('fill', '#ef4444').attr('opacity', 0.8).text('threshold');

  // CUSUM line
  const cusumLine = d3.line().x(d => xScale(d.date)).y(d => yCusum(d.val))
    .curve(d3.curveMonotoneX).defined(d => !isNaN(d.val));
  g.append('path').datum(cusum)
    .attr('fill', 'none').attr('stroke', '#d97706')
    .attr('stroke-width', 1.2).attr('opacity', 0.55).attr('d', cusumLine);

  // Distance area + line
  const distLine = d3.line().x(d => xScale(d.date)).y(d => yDist(d.val))
    .curve(d3.curveMonotoneX).defined(d => !isNaN(d.val));
  const distArea = d3.area().x(d => xScale(d.date)).y0(H).y1(d => yDist(d.val))
    .curve(d3.curveMonotoneX).defined(d => !isNaN(d.val));

  const areaGrad = svg.append('defs').append('linearGradient')
    .attr('id', 'distGrad').attr('gradientUnits', 'userSpaceOnUse')
    .attr('x1', 0).attr('y1', 0).attr('x2', 0).attr('y2', H);
  areaGrad.append('stop').attr('offset', '0%').attr('stop-color', '#3b82f6').attr('stop-opacity', 0.18);
  areaGrad.append('stop').attr('offset', '100%').attr('stop-color', '#3b82f6').attr('stop-opacity', 0.02);

  g.append('path').datum(dist).attr('fill', 'url(#distGrad)').attr('d', distArea);
  g.append('path').datum(dist)
    .attr('fill', 'none').attr('stroke', '#3b82f6').attr('stroke-width', 1.5).attr('d', distLine);

  // Normalised drift
  if (normDriftData.length > 0) {
    const normParsed = normDriftData.map(d => ({ date: parseDate(d.date), val: d.val }))
                                    .filter(d => !isNaN(d.val));
    const rawDriftMap = {};
    driftData.forEach(d => { rawDriftMap[d.date] = d.val; });

    const driftSorted = normParsed.map(d => d.val).sort((a,b) => a-b);
    const p90 = driftSorted[Math.floor(driftSorted.length * 0.90)];
    const yMax = Math.max(p90 * 1.1, DRIFT_ELEVATED * 1.2);
    const yDrift = d3.scaleLinear().domain([0, yMax]).range([H, 0]);

    // Tier reference lines
    [
      { val: DRIFT_UNSTABLE, col: 'rgba(239,68,68,0.2)',  label: 'UNSTABLE' },
      { val: DRIFT_ELEVATED, col: 'rgba(217,119,6,0.15)', label: 'ELEVATED' },
      { val: 0.8,            col: 'rgba(16,185,129,0.1)', label: 'STABLE'   },
    ].forEach(t => {
      if (t.val > yMax) return;
      const y = yDrift(t.val);
      g.append('line').attr('x1', 0).attr('x2', W).attr('y1', y).attr('y2', y)
        .attr('stroke', t.col).attr('stroke-width', 1).attr('stroke-dasharray', '3,4');
      g.append('text').attr('x', W - 2).attr('y', y - 3)
        .attr('text-anchor', 'end').attr('font-size', '8px')
        .attr('fill', t.col).attr('opacity', 0.7).text(t.label);
    });

    const driftAreaFn = d3.area()
      .x(d => xScale(d.date)).y0(H).y1(d => yDrift(Math.min(d.val, yMax)))
      .curve(d3.curveMonotoneX).defined(d => !isNaN(d.val));
    const driftLineFn = d3.line()
      .x(d => xScale(d.date)).y(d => yDrift(Math.min(d.val, yMax)))
      .curve(d3.curveMonotoneX).defined(d => !isNaN(d.val));

    const driftGrad = svg.select('defs').append('linearGradient')
      .attr('id', 'driftGrad').attr('gradientUnits', 'userSpaceOnUse')
      .attr('x1', 0).attr('y1', 0).attr('x2', 0).attr('y2', H);
    driftGrad.append('stop').attr('offset', '0%').attr('stop-color', '#d97706').attr('stop-opacity', 0.10);
    driftGrad.append('stop').attr('offset', '100%').attr('stop-color', '#d97706').attr('stop-opacity', 0.01);

    g.append('path').datum(normParsed).attr('fill', 'url(#driftGrad)').attr('d', driftAreaFn);
    g.append('path').datum(normParsed)
      .attr('fill', 'none').attr('stroke', '#d97706')
      .attr('stroke-width', 1.2).attr('opacity', 0.40).attr('d', driftLineFn);

    // Label current value
    const lastNorm = normParsed[normParsed.length - 1];
    if (lastNorm) {
      const tier = lastNorm.val > DRIFT_UNSTABLE ? 'UNSTABLE' :
                   lastNorm.val > DRIFT_ELEVATED ? 'ELEVATED' :
                   lastNorm.val < 0.8            ? 'STABLE' : 'NORMAL';
      const tierCol = lastNorm.val > DRIFT_UNSTABLE ? '#dc2626' :
                      lastNorm.val > DRIFT_ELEVATED ? '#d97706' :
                      lastNorm.val < 0.8            ? '#059669' : '#888';
      g.append('text')
        .attr('x', xScale(lastNorm.date) - 4)
        .attr('y', yDrift(Math.min(lastNorm.val, yMax)) - 6)
        .attr('font-size', '9px').attr('fill', tierCol).attr('opacity', 0.85)
        .attr('text-anchor', 'end')
        .text(`${lastNorm.val.toFixed(2)}x [${tier}]`);
    }

    // Hover tooltip
    const tooltip = document.getElementById('cp-tooltip');
    g.append('rect').attr('x', 0).attr('y', 0).attr('width', W).attr('height', H)
      .attr('fill', 'transparent')
      .on('mousemove', function(event) {
        const [mx] = d3.pointer(event, this);
        const hovDate = xScale.invert(mx);
        const bisect = d3.bisector(d => d.date).left;
        const idx = bisect(normParsed, hovDate, 1);
        if (idx <= 0 || idx >= normParsed.length) return;
        const d = normParsed[idx];
        const dateStr = d.date.toISOString().slice(0,10);
        const rawVal = rawDriftMap[dateStr];
        const tier2 = d.val > DRIFT_UNSTABLE ? '<span style="color:#dc2626">UNSTABLE</span>' :
                      d.val > DRIFT_ELEVATED ? '<span style="color:#d97706">ELEVATED</span>' :
                      d.val < 0.8            ? '<span style="color:#059669">STABLE</span>' :
                                               '<span style="color:#888">NORMAL</span>';
        tooltip.style.opacity = '1';
        tooltip.innerHTML = `<div style="color:#d97706;font-weight:700">${dateStr}</div>
          <div style="color:#555;font-size:0.85em">Drift rate: ${d.val.toFixed(2)}x normal -- ${tier2}</div>
          <div style="color:#888;font-size:0.82em">Raw drift: ${rawVal != null ? rawVal.toFixed(4) : '--'}</div>`;
        tooltip.style.left = (event.pageX+12)+'px';
        tooltip.style.top  = (event.pageY-20)+'px';
      })
      .on('mouseout', () => { tooltip.style.opacity='0'; });
  }

  // SPY overlay
  if (spyData.length > 0) {
    const spyParser = d3.timeParse('%Y-%m-%d');
    const spy = spyData.map(d => ({ date: spyParser(d.date), val: d.val }))
                       .filter(d => d.date !== null);
    const spyExt = d3.extent(spy, d => d.val);
    const ySpy = d3.scaleLinear().domain(spyExt).range([H, 0]);
    const spyLine = d3.line().x(d => xScale(d.date)).y(d => ySpy(d.val))
      .curve(d3.curveMonotoneX).defined(d => !isNaN(d.val));
    g.append('path').datum(spy)
      .attr('fill', 'none').attr('stroke', '#10b981')
      .attr('stroke-width', 1.2).attr('opacity', 0.5)
      .attr('stroke-dasharray', '4,2').attr('d', spyLine);
    const lastSpy = spy[spy.length - 1];
    if (lastSpy) {
      g.append('text').attr('x', xScale(lastSpy.date) + 4).attr('y', ySpy(lastSpy.val))
        .attr('font-size', '9px').attr('fill', '#10b981').attr('opacity', 0.7).text('SPY');
    }
  }

  // Historical events
  eventsData.forEach(ev => {
    const evDate = d3.timeParse('%Y-%m-%d')(ev.date);
    if (!evDate) return;
    const ex = xScale(evDate);
    if (ex < 0 || ex > W) return;
    g.append('line').attr('x1', ex).attr('x2', ex).attr('y1', H).attr('y2', H + 6)
      .attr('stroke', '#4f46e5').attr('stroke-width', 1).attr('opacity', 0.6);
    g.append('rect').attr('x', ex - 5).attr('y', 0).attr('width', 10).attr('height', H)
      .attr('fill', 'transparent').style('cursor', 'crosshair')
      .on('mouseover', function(event) {
        const tip = document.getElementById('cp-tooltip');
        tip.style.opacity = '1';
        tip.innerHTML = `<div style="color:#4f46e5;font-weight:700;margin-bottom:4px">${ev.label}</div>
          <div style="color:#888;font-size:0.82em">${ev.date}</div>
          <div style="color:#555;font-size:0.85em;margin-top:4px">${ev.desc}</div>`;
      })
      .on('mousemove', function(event) {
        const tip = document.getElementById('cp-tooltip');
        tip.style.left = (event.pageX + 12) + 'px';
        tip.style.top  = (event.pageY - 20) + 'px';
      })
      .on('mouseout', function() { document.getElementById('cp-tooltip').style.opacity = '0'; });
  });

  // Changepoint markers
  const tooltip = document.getElementById('cp-tooltip');
  cpData.forEach(cp => {
    const cpDate = parseDate(cp.date);
    const cx = xScale(cpDate);
    if (cx < 0 || cx > W) return;
    g.append('line').attr('x1', cx).attr('x2', cx).attr('y1', 0).attr('y2', H)
      .attr('stroke', '#7c3aed').attr('stroke-width', 1)
      .attr('stroke-dasharray', '3,2').attr('opacity', 0.7);
    g.append('path')
      .attr('d', `M ${cx} ${-8} L ${cx+5} ${-3} L ${cx} ${2} L ${cx-5} ${-3} Z`)
      .attr('fill', '#7c3aed').attr('opacity', 0.9)
      .style('cursor', 'pointer')
      .on('mouseover', function(event) {
        tooltip.style.opacity = '1';
        tooltip.innerHTML = `<div style="color:#7c3aed;font-weight:700;margin-bottom:6px">${cp.date}</div>
          <div style="color:#888;font-size:0.85em">CUSUM: <span style="color:#d97706">${cp.cusum.toFixed(3)}</span></div>
          <div style="color:#888;font-size:0.85em;margin-bottom:8px">Distance: <span style="color:#3b82f6">${cp.dist.toFixed(5)}</span></div>
          ${cp.flip_html}`;
      })
      .on('mousemove', function(event) {
        tooltip.style.left = (event.pageX + 12) + 'px';
        tooltip.style.top  = (event.pageY - 20) + 'px';
      })
      .on('mouseout', function() { tooltip.style.opacity = '0'; });
  });

  // Axes
  g.append('g').attr('class', 'axis').attr('transform', `translate(0,${H})`)
    .call(d3.axisBottom(xScale).ticks(d3.timeYear.every(1)).tickFormat(d3.timeFormat('%Y')));
  g.append('g').attr('class', 'axis')
    .call(d3.axisLeft(yDist).ticks(5).tickFormat(d3.format('.4f')));
  g.append('g').attr('class', 'axis').attr('transform', `translate(${W},0)`)
    .call(d3.axisRight(yCusum).ticks(5).tickFormat(d3.format('.1f')));

  // Axis labels
  g.append('text').attr('x', -H/2).attr('y', -44)
    .attr('transform', 'rotate(-90)').attr('text-anchor', 'middle')
    .attr('font-size', '10px').attr('fill', '#888').text('Cosine Distance');
  g.append('text').attr('x', W + 44).attr('y', H/2)
    .attr('transform', `rotate(90,${W+44},${H/2})`)
    .attr('text-anchor', 'middle').attr('font-size', '10px').attr('fill', '#888').text('CUSUM');
})();
""")

    return '\n'.join(js_parts)


# ---------------------------------------------------------------------------
# Prepare all data needed for HTML and JS
# ---------------------------------------------------------------------------
def prepare_results(feat_matrix, spy_prices, fingerprints, dist_smooth,
                    changepoints, cusum_series, cusum_threshold, pole_flips,
                    drift_series, norm_drift_series, regime_similarity,
                    curr_fingerprint, pole_returns_raw):
    """Prepare all data structures for the dashboard."""
    from datetime import date as _date

    today_str = _date.today().isoformat()

    # Distance stats
    ds = dist_smooth.dropna()
    dist_max  = float(ds.max()) if len(ds) > 0 else 0
    dist_mean = float(ds.mean()) if len(ds) > 0 else 0

    # Changepoint markers
    cp_data = []
    for cp_date, cp_cusum, cp_dist in changepoints:
        flips = pole_flips.get(cp_date, [])
        flip_html = ''
        for f in flips[:6]:
            color = '#dc2626' if 'BEAR' in f['direction'] or 'weak' in f['direction'] else '#059669'
            flip_html += (
                "<div style='display:flex;justify-content:space-between;"
                "padding:3px 0;border-bottom:1px solid #eee;font-size:0.78em;'>"
                "<span style='color:#555'>{}</span>"
                "<span style='color:{};font-family:monospace'>"
                "{:+.2f} -> {:+.2f} ({})</span></div>".format(
                    f['pole'][:28], color, f['pre_z'], f['post_z'], f['direction']
                )
            )
        cp_data.append({
            'date':      str(cp_date.date()),
            'cusum':     round(cp_cusum, 3),
            'dist':      round(cp_dist, 5) if not np.isnan(cp_dist) else 0,
            'flip_html': flip_html,
        })

    # Window bands
    date_min = ds.index.min() if len(ds) > 0 else pd.Timestamp('2000-01-01')
    date_max = ds.index.max() if len(ds) > 0 else pd.Timestamp.today()
    window_data = []
    for i, (wname, wstart, wend, wdesc) in enumerate(cpd.MACRO_WINDOWS):
        ws = pd.Timestamp(wstart) if wstart else date_min
        we = pd.Timestamp(wend)   if wend   else date_max
        ws = max(ws, date_min)
        we = min(we, date_max)
        if ws >= we:
            continue
        window_data.append({
            'name':  wname,
            'desc':  wdesc,
            'start': str(ws.date()),
            'end':   str(we.date()),
            'color': WINDOW_COLORS_LIGHT[i % len(WINDOW_COLORS_LIGHT)],
        })

    # Distance series JSON
    dist_data = [
        {'date': str(dt.date()), 'val': round(float(v), 5)}
        for dt, v in ds.items() if not np.isnan(v)
    ]
    # CUSUM series
    cs = cusum_series.reindex(ds.index).fillna(0)
    cusum_data = [
        {'date': str(dt.date()), 'val': round(float(v), 4)}
        for dt, v in cs.items() if not np.isnan(v)
    ]

    # Drift series
    drift_data = []
    if drift_series is not None:
        for dt, v in drift_series.dropna().items():
            drift_data.append({'date': str(dt.date()), 'val': round(float(v), 5)})

    norm_drift_data = []
    if norm_drift_series is not None:
        for dt, v in norm_drift_series.dropna().items():
            norm_drift_data.append({'date': str(dt.date()), 'val': round(float(v), 3)})

    # SPY
    spy_data = []
    if spy_prices is not None:
        spy_weekly = spy_prices.resample('W').last().dropna()
        spy_data = [{'date': str(dt.date()), 'val': round(float(v), 2)}
                    for dt, v in spy_weekly.items()]

    # Events
    events_data = []
    for ev_date, ev_label, ev_desc in cpd.HISTORICAL_EVENTS:
        events_data.append({'date': ev_date, 'label': ev_label, 'desc': ev_desc})

    # Pole fingerprints (multi-window)
    def _clean_pole_label(col_name):
        parts = str(col_name).split('_', 1)
        return parts[1] if len(parts) > 1 and parts[0].isdigit() else str(col_name)

    def _compute_window_poles(pr_df, window):
        result = []
        HISTORY_DAYS = 252
        for col in pr_df.columns:
            s = pr_df[col].dropna()
            cum = s.rolling(window, min_periods=window).sum()
            mu  = cum.rolling(HISTORY_DAYS, min_periods=60).mean()
            sig = cum.rolling(HISTORY_DAYS, min_periods=60).std()
            z_series = ((cum - mu) / sig.replace(0, np.nan)).clip(-3, 3)
            if len(z_series.dropna()) == 0:
                continue
            val = float(z_series.dropna().iloc[-1])
            if val == val:
                label = _clean_pole_label(col)
                result.append({'pole': label, 'z': round(val, 3)})
        return result

    pole_windows = {}
    if curr_fingerprint is not None:
        fp = curr_fingerprint.dropna()
        z_short_cols = [c for c in fp.index if str(c).endswith('_z_short')]
        w20 = []
        for col in z_short_cols:
            label = _clean_pole_label(str(col).replace('_z_short', ''))
            val = float(fp[col])
            if val == val:
                w20.append({'pole': label, 'z': round(val, 3)})
        w20.sort(key=lambda x: x['pole'].lower())
        pole_windows['20d'] = w20

    if pole_returns_raw is not None:
        excl = {str(p) for p in cpd.EXCLUDE_POLES}
        pr = pole_returns_raw[[c for c in pole_returns_raw.columns
                                if str(c).split('_')[0] not in excl]]
        for label, window in [('63d', 63), ('126d', 126), ('252d', 252)]:
            rows = _compute_window_poles(pr, window)
            rows.sort(key=lambda x: x['pole'].lower())
            pole_windows[label] = rows

    first_key = next(iter(pole_windows), None)
    pole_data = pole_windows.get('20d', pole_windows.get(first_key, []))

    # Drift for stat bar
    curr_norm_drift = float('nan')
    if norm_drift_series is not None:
        valid_norm = norm_drift_series.dropna()
        if len(valid_norm) > 0:
            curr_norm_drift = float(valid_norm.iloc[-1])

    if not np.isnan(curr_norm_drift):
        if curr_norm_drift > cpd.DRIFT_UNSTABLE:
            drift_tier = 'UNSTABLE'
            drift_css  = 'neg'
        elif curr_norm_drift > cpd.DRIFT_ELEVATED:
            drift_tier = 'ELEVATED'
            drift_css  = 'warn'
        elif curr_norm_drift < cpd.DRIFT_STABLE:
            drift_tier = 'STABLE'
            drift_css  = 'pos'
        else:
            drift_tier = 'NORMAL'
            drift_css  = 'neutral'
        drift_str   = '{:.2f}x'.format(curr_norm_drift)
        drift_label = 'Drift Rate ({})'.format(drift_tier)
    else:
        drift_str   = '--'
        drift_css   = 'neutral'
        drift_label = 'Drift Rate'

    top_regime = regime_similarity[0]['name'] if regime_similarity else '--'
    top_sim    = '{:.1f}%'.format(regime_similarity[0]['sim']) if regime_similarity else '--'

    # Recent summary
    nov25 = pd.Timestamp('2025-11-01')
    feb26 = pd.Timestamp('2026-02-28')
    recent_cps = [(d, cs_val, dv) for d, cs_val, dv in changepoints if nov25 <= d <= feb26]
    recent_summary = (
        '{} changepoint(s) detected between Nov 2025 and Feb 2026'.format(len(recent_cps))
        if recent_cps else
        'No significant changepoints detected between Nov 2025 and Feb 2026 -- regime appears continuous'
    )

    return {
        'n_changepoints':   len(cp_data),
        'dist_max':         dist_max,
        'dist_mean':        dist_mean,
        'cusum_threshold':  cusum_threshold,
        'n_features':       feat_matrix.shape[1],
        'drift_label':      drift_label,
        'drift_str':        drift_str,
        'drift_css':        drift_css,
        'drift_tier':       drift_tier if not np.isnan(curr_norm_drift) else 'UNKNOWN',
        'curr_norm_drift':  curr_norm_drift,
        'top_regime':       top_regime,
        'top_sim':          top_sim,
        'recent_summary':   recent_summary,
        'js_vars': {
            'dist_json':         json.dumps(dist_data),
            'cusum_json':        json.dumps(cusum_data),
            'cp_json':           json.dumps(cp_data, ensure_ascii=False),
            'window_json':       json.dumps(window_data),
            'spy_json':          json.dumps(spy_data),
            'events_json':       json.dumps(events_data, ensure_ascii=False),
            'drift_json':        json.dumps(drift_data),
            'norm_drift_json':   json.dumps(norm_drift_data),
            'sim_json':          json.dumps(regime_similarity or []),
            'pole_json':         json.dumps(pole_data),
            'pole_windows_json': json.dumps(pole_windows),
            'drift_elevated':    str(cpd.DRIFT_ELEVATED),
            'drift_unstable':    str(cpd.DRIFT_UNSTABLE),
            'cusum_threshold':   str(cusum_threshold),
        },
    }


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    print('=' * 68)
    print('CHANGEPOINT BACKEND v1.0')
    print('=' * 68)

    # 1. Feature matrix
    print('\n[1/7] Loading feature matrix...')
    feat_matrix = cpd.load_or_build_feature_matrix(force_rebuild=False)

    # 2. SPY prices
    print('\n[2/7] Loading SPY prices...')
    spy_prices = cpd.load_spy_prices()

    # 3. Fingerprints + distance
    print('\n[3/7] Computing regime fingerprints and distance...')
    fingerprints = cpd.compute_fingerprints(feat_matrix, cpd.FINGERPRINT_WINDOW)
    dist_raw, dist_smooth = cpd.compute_distance_series(fingerprints)

    # 4. CUSUM
    print('\n[4/7] Running CUSUM changepoint detection...')
    clean = dist_smooth.dropna()
    mu  = float(clean.mean())
    sig = float(clean.std())
    cusum_threshold = mu + 3.0 * sig

    changepoints, cusum_series = cpd.cusum_changepoints(
        dist_smooth, k=cpd.CUSUM_DRIFT,
        threshold=cusum_threshold, min_gap=cpd.MIN_REGIME_DAYS,
    )
    print('  {} changepoints detected'.format(len(changepoints)))

    # 5. Pole flips
    print('\n[5/7] Analysing pole flips...')
    pole_flips = {}
    for cp_date, _, _ in changepoints:
        pole_flips[cp_date] = cpd.analyze_pole_flips(feat_matrix, cp_date)

    # 6. Drift + similarity
    print('\n[6/7] Computing drift and regime similarity...')
    drift_series, norm_drift_series = cpd.compute_drift_from_baseline(
        fingerprints, changepoints
    )
    regime_similarity = cpd.compute_regime_similarity(fingerprints)
    if regime_similarity:
        print('  Top match: {} ({:.1f}%)'.format(
            regime_similarity[0]['name'], regime_similarity[0]['sim']
        ))

    # Current fingerprint
    curr_fp = fingerprints.dropna(how='all').iloc[-1] if len(fingerprints.dropna(how='all')) > 0 else None

    # Load raw pole returns for multi-window
    pole_returns_raw = None
    if cpd.POLE_RETURNS_CSV.exists():
        try:
            pole_returns_raw = pd.read_csv(
                cpd.POLE_RETURNS_CSV, index_col=0, parse_dates=True, encoding='utf-8'
            )
            pole_returns_raw.index = pd.DatetimeIndex(pole_returns_raw.index).normalize()
        except Exception as e:
            print('  Warning: could not load pole returns: {}'.format(e))

    # 7. Build dashboard
    print('\n[7/7] Building dashboard...')
    results = prepare_results(
        feat_matrix, spy_prices, fingerprints, dist_smooth,
        changepoints, cusum_series, cusum_threshold, pole_flips,
        drift_series, norm_drift_series, regime_similarity,
        curr_fp, pole_returns_raw,
    )

    writer = DashboardWriter('regime-changepoint', 'Regime Changepoint Detector')
    body   = build_body(results, writer)
    js     = build_js(results)
    writer.write(body, extra_css=EXTRA_CSS, extra_js=js)

    # Save CSVs (timestamped)
    from datetime import date as _date
    today_str = _date.today().isoformat()
    cpd.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cpd.save_csv(
        changepoints, pole_flips, dist_smooth, spy_prices,
        cpd.OUTPUT_DIR, today_str, regime_similarity,
        drift_series, norm_drift_series,
    )

    # Also save a backend CSV
    csv_path = os.path.join(_SCRIPT_DIR, 'changepoint_data_{}.csv'.format(
        datetime.datetime.now().strftime('%Y%m%d_%H%M')
    ))
    sim_rows = []
    for r in (regime_similarity or []):
        sim_rows.append({
            'name': r['name'],
            'description': r['desc'],
            'similarity_pct': r['sim'],
        })
    if sim_rows:
        pd.DataFrame(sim_rows).to_csv(csv_path, index=False, encoding='utf-8')
        print('CSV: {}'.format(csv_path))

    # Write summary JSON for meta-dashboard consumption
    summary = {
        'generated_at': datetime.datetime.now().isoformat(),
        'drift_tier': results['drift_tier'],
        'norm_drift': float(results['curr_norm_drift']) if not np.isnan(results['curr_norm_drift']) else None,
        'top_regime': results['top_regime'],
        'top_sim': float(regime_similarity[0]['sim']) if regime_similarity else 0.0,
        'regime_similarity': [
            {'name': r['name'], 'desc': r.get('desc', ''), 'sim': r['sim']}
            for r in (regime_similarity or [])[:5]
        ],
    }
    summary_path = os.path.join(_SCRIPT_DIR, 'changepoint_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print('Summary JSON: {}'.format(summary_path))

    print('\n[DONE] Changepoint dashboard written.')


if __name__ == '__main__':
    main()
