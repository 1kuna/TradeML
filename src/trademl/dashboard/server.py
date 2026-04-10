"""HTTP dashboard server with client-side live updates."""

from __future__ import annotations

import argparse
import json
import logging
import threading
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from trademl.dashboard.controller import (
    advance_collection_stage,
    collect_dashboard_live_snapshot,
    collect_dashboard_logs_snapshot,
    collect_dashboard_setup_snapshot,
    collect_dashboard_status_snapshot,
    force_release_lease,
    install_service,
    join_cluster,
    leave_cluster,
    persist_node_settings,
    rebuild_cluster_state,
    replan_coverage,
    resolve_node_settings,
    restart_node,
    rotate_cluster_passphrase,
    run_vendor_audit,
    start_node,
    stop_node,
    uninstall_worker,
    update_worker,
    update_cluster_secrets,
    reset_worker,
    NodeSettings,
)

LOGGER = logging.getLogger(__name__)

HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TradeML Operator Dashboard</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #08111c;
      --bg-alt: #0d1725;
      --panel: rgba(12, 22, 34, 0.92);
      --panel-strong: rgba(16, 28, 43, 0.98);
      --panel-alt: rgba(29, 44, 64, 0.76);
      --text: #edf4ff;
      --muted: #8fa3bf;
      --line: rgba(126, 152, 182, 0.18);
      --line-strong: rgba(126, 152, 182, 0.28);
      --accent: #20c997;
      --accent-strong: #18a67d;
      --accent-soft: rgba(32, 201, 151, 0.14);
      --warn: #f6ad55;
      --bad: #ff6b6b;
      --good: #68d391;
      --shadow: 0 26px 60px rgba(0, 0, 0, 0.35);
      --radius: 20px;
      --mono: "SFMono-Regular", Menlo, Consolas, monospace;
      font-family: "IBM Plex Sans", "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    }
    * { box-sizing: border-box; }
    html, body { min-height: 100%; }
    body {
      margin: 0;
      background:
        radial-gradient(circle at top left, rgba(32, 201, 151, 0.12), transparent 24%),
        radial-gradient(circle at top right, rgba(43, 108, 176, 0.22), transparent 30%),
        linear-gradient(180deg, #09111b 0%, var(--bg) 46%, #050b14 100%);
      color: var(--text);
      overflow-x: hidden;
    }
    .shell {
      width: min(1480px, 100%);
      margin: 0 auto;
      padding: 22px;
    }
    .hero, .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      backdrop-filter: blur(18px);
    }
    .hero {
      padding: 24px;
      margin-bottom: 18px;
    }
    .hero-grid {
      display: grid;
      grid-template-columns: minmax(0, 1.35fr) minmax(340px, 0.95fr);
      gap: 18px;
      align-items: stretch;
    }
    .hero-copy, .actions-panel {
      background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01));
      border: 1px solid var(--line);
      border-radius: 18px;
    }
    .hero-copy {
      padding: 22px;
      display: grid;
      gap: 18px;
    }
    .actions-panel {
      padding: 18px;
      display: grid;
      gap: 14px;
      background: linear-gradient(180deg, rgba(10, 19, 31, 0.95), rgba(16, 28, 43, 0.95));
    }
    .hero-top, .actions, .nav, .summary-meta {
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      align-items: center;
    }
    .hero-top { justify-content: space-between; }
    h1, h2, h3 { margin: 0; font-weight: 700; }
    h1 {
      font-size: clamp(2rem, 4vw, 3rem);
      line-height: 1;
      letter-spacing: -0.03em;
    }
    h2 {
      font-size: 1.25rem;
      letter-spacing: -0.02em;
    }
    h3 {
      font-size: 1rem;
      letter-spacing: -0.01em;
    }
    p { margin: 0; color: var(--muted); }
    .eyebrow {
      display: inline-flex;
      width: fit-content;
      border-radius: 999px;
      padding: 7px 11px;
      border: 1px solid var(--line);
      background: var(--accent-soft);
      color: #b9ffeb;
      font-size: 12px;
      letter-spacing: 0.12em;
      text-transform: uppercase;
    }
    .badge {
      border-radius: 999px;
      padding: 8px 12px;
      border: 1px solid var(--line-strong);
      background: var(--panel-alt);
      font-size: 13px;
      color: #d7fff1;
    }
    .hero-copy p {
      max-width: 70ch;
      font-size: 15px;
      line-height: 1.55;
    }
    .actions, .nav { margin-top: 0; }
    .summary-meta {
      justify-content: space-between;
    }
    .summary-meta .badge {
      background: rgba(32, 201, 151, 0.12);
    }
    .hero-status-line {
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      align-items: center;
    }
    input, select, button, textarea {
      font: inherit;
    }
    input, textarea {
      border: 1px solid var(--line-strong);
      border-radius: 12px;
      padding: 11px 12px;
      background: rgba(9, 17, 28, 0.88);
      color: var(--text);
      width: 100%;
      min-width: 0;
    }
    input::placeholder, textarea::placeholder { color: #6f86a4; }
    button {
      border: 1px solid transparent;
      border-radius: 12px;
      padding: 11px 14px;
      background: var(--accent);
      color: #071218;
      cursor: pointer;
      font-weight: 600;
      transition: transform 120ms ease, background 120ms ease, border-color 120ms ease;
    }
    button.secondary {
      background: rgba(44, 62, 84, 0.85);
      color: var(--text);
      border-color: var(--line-strong);
    }
    button.ghost {
      background: rgba(8, 17, 28, 0.2);
      color: var(--muted);
      border: 1px solid var(--line);
    }
    button:hover {
      transform: translateY(-1px);
      border-color: rgba(32, 201, 151, 0.28);
    }
    .summary-grid, .detail-grid, .setup-grid {
      display: grid;
      gap: 14px;
    }
    .summary-grid {
      grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
      margin-bottom: 18px;
    }
    .detail-grid {
      grid-template-columns: 1.2fr 1fr;
      margin-top: 18px;
    }
    .setup-grid {
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      margin-top: 18px;
    }
    .card, .section-card {
      background: linear-gradient(180deg, rgba(15, 26, 40, 0.98), rgba(10, 18, 29, 0.96));
      border: 1px solid var(--line);
      border-radius: 16px;
    }
    .card {
      padding: 18px;
      min-height: 132px;
    }
    .label {
      font-size: 12px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 10px;
    }
    .value {
      font-size: clamp(1.8rem, 3vw, 2.5rem);
      line-height: 1.1;
      font-weight: 700;
      margin-bottom: 8px;
    }
    .delta {
      font-size: 13px;
      color: var(--muted);
      line-height: 1.45;
    }
    .panel {
      padding: 20px;
    }
    .panel[hidden] { display: none; }
    .section-card {
      padding: 18px;
    }
    .stack {
      display: grid;
      gap: 14px;
    }
    .status-line {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 14px;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 10px;
      border-radius: 999px;
      background: rgba(27, 41, 59, 0.92);
      border: 1px solid var(--line-strong);
      color: #d0def2;
      font-size: 13px;
    }
    .pill.good { color: var(--good); }
    .pill.bad { color: var(--bad); }
    .pill.warn { color: var(--warn); }
    .mono, pre, code {
      font-family: var(--mono);
    }
    pre {
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      line-height: 1.45;
      background: rgba(7, 14, 24, 0.98);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 14px;
      max-height: 420px;
      overflow: auto;
      color: #d9e8ff;
    }
    .table-wrap {
      overflow-x: hidden;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: rgba(7, 14, 24, 0.62);
    }
    table {
      width: 100%;
      border-collapse: collapse;
      table-layout: fixed;
      min-width: 0;
      background: transparent;
    }
    th, td {
      padding: 10px 12px;
      border-bottom: 1px solid rgba(126, 152, 182, 0.14);
      text-align: left;
      font-size: 14px;
      word-break: break-word;
      overflow-wrap: anywhere;
      vertical-align: top;
    }
    th {
      position: sticky;
      top: 0;
      background: rgba(14, 24, 38, 0.96);
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
      font-size: 12px;
    }
    .toolbar {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      flex-wrap: wrap;
      margin-bottom: 14px;
    }
    .message {
      min-height: 20px;
      font-size: 14px;
      color: var(--muted);
    }
    .message.bad { color: var(--bad); }
    .message.good { color: var(--good); }
    .message.warn { color: var(--warn); }
    .kv {
      display: grid;
      grid-template-columns: minmax(120px, 180px) 1fr;
      gap: 8px 12px;
      font-size: 14px;
    }
    .kv strong { color: var(--text); }
    .nav button.active {
      background: var(--accent-strong);
      color: #061116;
    }
    .muted { color: var(--muted); }
    .small { font-size: 13px; }
    .forms { display: grid; gap: 12px; }
    .form-row {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      align-items: center;
    }
    @media (max-width: 1180px) {
      .hero-grid {
        grid-template-columns: 1fr;
      }
    }
    @media (max-width: 980px) {
      .shell {
        padding: 14px;
      }
      .detail-grid { grid-template-columns: 1fr; }
      .kv { grid-template-columns: 1fr; }
    }
    @media (max-width: 860px) {
      .table-wrap {
        overflow-x: visible;
        border: 0;
        background: transparent;
      }
      table, thead, tbody, tr, th, td {
        display: block;
        width: 100%;
      }
      thead {
        display: none;
      }
      tbody {
        display: grid;
        gap: 10px;
      }
      tr {
        border: 1px solid var(--line);
        border-radius: 14px;
        background: rgba(10, 18, 29, 0.94);
        overflow: hidden;
      }
      td {
        display: grid;
        grid-template-columns: minmax(110px, 132px) 1fr;
        gap: 8px;
        padding: 10px 12px;
      }
      td::before {
        content: attr(data-label);
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-size: 11px;
      }
    }
  </style>
</head>
<body>
  <div class="shell">
    <div class="hero">
      <div class="hero-grid">
        <div class="hero-copy">
          <div class="hero-top">
            <div>
              <div class="eyebrow">Operator Console</div>
              <h1>TradeML Dashboard</h1>
            </div>
            <div class="badge" id="connection-badge">Connecting</div>
          </div>
          <div class="hero-status-line">
            <div class="badge">Realtime node telemetry</div>
            <div class="badge">Canonical collection priority</div>
            <div class="badge">Training gate tracking</div>
          </div>
          <div class="nav">
            <button class="active" data-section="status">Status</button>
            <button class="ghost" data-section="budgets">Budgets</button>
            <button class="ghost" data-section="setup">Setup</button>
            <button class="ghost" data-section="logs">Logs</button>
          </div>
        </div>
        <div class="actions-panel">
          <div class="summary-meta">
            <div>
              <div class="label">Quick Control</div>
              <div class="delta">Node lifecycle and collection control live here.</div>
            </div>
            <div class="badge">Live control plane</div>
          </div>
          <input id="cluster-passphrase" type="password" placeholder="Cluster passphrase">
          <div class="form-row">
            <button data-action="start-node">Start Node</button>
            <button data-action="stop-node" class="secondary">Stop Node</button>
            <button data-action="restart-node" class="secondary">Restart Node</button>
          </div>
          <div class="form-row">
            <button data-action="run-vendor-audit" class="ghost">Run Audit</button>
            <button data-action="replan-coverage" class="ghost">Replan Coverage</button>
          </div>
          <span id="action-message" class="message"></span>
        </div>
      </div>
    </div>

    <div class="summary-grid">
      <div class="card"><div class="label">Node Status</div><div class="value" id="metric-node-status">-</div><div class="delta mono" id="metric-node-detail">-</div></div>
      <div class="card"><div class="label">Canonical Coverage</div><div class="value" id="metric-coverage">-</div><div class="delta" id="metric-coverage-detail">-</div></div>
      <div class="card"><div class="label">Canonical Datapoints</div><div class="value" id="metric-canonical">-</div><div class="delta" id="metric-remaining">-</div></div>
      <div class="card"><div class="label">Raw Vendor Rows</div><div class="value" id="metric-raw-rows">-</div><div class="delta" id="metric-updated">-</div></div>
      <div class="card"><div class="label">Phase 1 Gate</div><div class="value" id="metric-gate">-</div><div class="delta" id="metric-gate-detail">-</div></div>
      <div class="card"><div class="label">Bars ETA</div><div class="value" id="metric-eta">-</div><div class="delta" id="metric-freeze-cutoff">-</div></div>
    </div>

    <section class="panel" id="section-status">
      <div class="toolbar">
        <div>
          <h2>Status</h2>
          <div class="small muted">Training gate, collection coverage, and vendor throughput.</div>
        </div>
        <span class="small muted" id="status-updated">Waiting for first snapshot</span>
      </div>
      <div class="detail-grid">
        <div class="stack">
          <div class="section-card">
            <div class="status-line">
              <h3>Training Readiness</h3>
              <span class="pill" id="status-readiness-pill">Unknown</span>
            </div>
            <div class="kv" id="training-summary"></div>
            <div style="margin-top:14px">
              <div class="label">Phase 1 Command</div>
              <pre id="training-command"></pre>
            </div>
          </div>
          <div class="section-card">
            <h3 style="margin-bottom:12px">Training-Critical Coverage</h3>
            <div class="table-wrap"><table id="coverage-table"></table></div>
          </div>
        </div>
        <div class="stack">
          <div class="section-card">
            <h3 style="margin-bottom:12px">Vendor Utilization</h3>
            <div class="table-wrap"><table id="vendor-table"></table></div>
          </div>
          <div class="section-card">
            <h3 style="margin-bottom:12px">Readiness Detail</h3>
            <pre id="readiness-json"></pre>
          </div>
        </div>
      </div>
    </section>

    <section class="panel" id="section-budgets" hidden>
      <div class="toolbar">
        <div>
          <h2>Budgets</h2>
          <div class="small muted">Current request pressure and daily cap state per vendor.</div>
        </div>
        <span class="small muted" id="budgets-updated">Waiting for first snapshot</span>
      </div>
      <div class="stack">
        <div class="section-card">
          <div class="kv" id="budget-summary"></div>
        </div>
        <div class="section-card">
          <div class="table-wrap"><table id="budget-table"></table></div>
        </div>
      </div>
    </section>

    <section class="panel" id="section-setup" hidden>
      <div class="toolbar">
        <div>
          <h2>Setup</h2>
          <div class="small muted">Cluster membership, NAS wiring, secrets, and worker lifecycle.</div>
        </div>
        <span class="small muted" id="setup-updated">Loads on demand</span>
      </div>
      <div class="setup-grid">
        <div class="section-card forms">
          <h3>Cluster Control</h3>
          <div class="form-row">
            <button data-action="join-cluster">Join Cluster</button>
            <button data-action="rebuild-state" class="secondary">Rebuild Local State</button>
            <button data-action="leave-cluster" class="ghost">Leave Cluster</button>
          </div>
          <div class="form-row">
            <button data-action="install-service">Install Service</button>
            <button data-action="update-worker" class="secondary">Update Worker</button>
            <button data-action="reset-worker" class="secondary">Reset Worker</button>
            <button data-action="uninstall-worker" class="ghost">Uninstall Worker</button>
          </div>
          <div class="message" id="setup-message"></div>
          <div class="kv" id="setup-summary"></div>
        </div>
        <div class="section-card forms">
          <h3>NAS and Schedule</h3>
          <input id="input-nas-share" placeholder="NAS share">
          <input id="input-nas-mount" placeholder="NAS mount path">
          <input id="input-collection-time" placeholder="Collection time ET">
          <input id="input-maintenance-hour" placeholder="Maintenance hour local">
          <input id="input-fstab-path" placeholder="fstab path" value="/etc/fstab">
          <div class="form-row">
            <button id="save-settings">Save Settings</button>
          </div>
        </div>
        <div class="section-card forms">
          <h3>Secrets</h3>
          <input id="input-secret-key" placeholder="Secret key">
          <input id="input-secret-value" placeholder="Secret value">
          <div class="form-row">
            <button id="update-secret">Update Secret</button>
          </div>
          <input id="input-old-passphrase" type="password" placeholder="Old passphrase">
          <input id="input-new-passphrase" type="password" placeholder="New passphrase">
          <div class="form-row">
            <button id="rotate-passphrase">Rotate Passphrase</button>
          </div>
        </div>
        <div class="section-card forms">
          <h3>Lease Management</h3>
          <input id="input-lease-id" placeholder="Lease ID">
          <div class="form-row">
            <button id="force-release-lease" class="secondary">Force Release Lease</button>
          </div>
          <pre id="setup-json"></pre>
        </div>
      </div>
    </section>

    <section class="panel" id="section-logs" hidden>
      <div class="toolbar">
        <div>
          <h2>Logs</h2>
          <div class="small muted">Node stdout plus recent systemd journal lines.</div>
        </div>
        <span class="small muted" id="logs-updated">Loads on demand</span>
      </div>
      <div class="stack">
        <div class="section-card">
          <h3 style="margin-bottom:12px">Node Log</h3>
          <pre id="node-log"></pre>
        </div>
        <div class="section-card">
          <h3 style="margin-bottom:12px">systemd Journal</h3>
          <pre id="journal-log"></pre>
        </div>
      </div>
    </section>
  </div>

  <script>
    const sections = ['status', 'budgets', 'setup', 'logs'];
    let activeSection = 'status';
    let currentLiveSnapshot = null;
    let statusTimer = null;
    let setupLoaded = false;
    let logsLoaded = false;

    function formatNumber(value) {
      if (value === null || value === undefined || value === '') return '-';
      return new Intl.NumberFormat('en-US').format(value);
    }

    function formatEta(value) {
      if (value === null || value === undefined) return '-';
      const minutes = Math.max(0, Math.round(Number(value)));
      if (minutes < 60) return `${minutes}m`;
      const hours = Math.floor(minutes / 60);
      const remainingMinutes = minutes % 60;
      if (hours < 24) return `${hours}h ${remainingMinutes}m`;
      const days = Math.floor(hours / 24);
      const remainingHours = hours % 24;
      return `${days}d ${remainingHours}h`;
    }

    function setMessage(id, text, level='') {
      const el = document.getElementById(id);
      el.textContent = text || '';
      el.className = `message ${level}`.trim();
    }

    function renderTable(tableId, rows) {
      const table = document.getElementById(tableId);
      if (!rows || rows.length === 0) {
        table.innerHTML = '<tbody><tr><td class="muted">No data</td></tr></tbody>';
        return;
      }
      const columns = Object.keys(rows[0]);
      const head = `<thead><tr>${columns.map((column) => `<th>${column}</th>`).join('')}</tr></thead>`;
      const body = rows.map((row) => {
        return `<tr>${columns.map((column) => `<td data-label="${column}">${row[column] ?? ''}</td>`).join('')}</tr>`;
      }).join('');
      table.innerHTML = `${head}<tbody>${body}</tbody>`;
    }

    function renderKeyValue(targetId, pairs) {
      const root = document.getElementById(targetId);
      root.innerHTML = pairs.map(([label, value]) => `<strong>${label}</strong><span>${value ?? '-'}</span>`).join('');
    }

    function renderLive(snapshot) {
      currentLiveSnapshot = snapshot;
      const runtime = snapshot.runtime || {};
      const collection = snapshot.collection_status || {};
      const readiness = (snapshot.training_readiness || {}).phase1 || {};
      const freezeCutoff = (snapshot.training_readiness || {}).freeze_cutoff || {};
      const eta = (((snapshot.planner_eta || {}).canonical_bars) || {}).eta_minutes;
      document.getElementById('metric-node-status').textContent = runtime.running ? 'Running' : 'Stopped';
      document.getElementById('metric-node-detail').textContent = runtime.pid ? `PID ${runtime.pid}` : 'No active PID';
      document.getElementById('metric-coverage').textContent = `${collection.coverage_percent ?? 0}%`;
      document.getElementById('metric-coverage-detail').textContent = `${formatNumber(collection.pending_tasks ?? 0)} pending`;
      document.getElementById('metric-canonical').textContent = formatNumber(collection.canonical_completed_units ?? 0);
      const pinnedRemaining = collection.phase1_pinned_remaining_units;
      const rollingRemaining = collection.canonical_remaining_units ?? 0;
      const repairRemaining = collection.repair_remaining_units ?? 0;
      document.getElementById('metric-remaining').textContent = readiness.ready
        ? `Phase 1 complete · ${formatNumber(rollingRemaining)} rolling / ${formatNumber(repairRemaining)} repair`
        : `${formatNumber(pinnedRemaining ?? rollingRemaining)} pinned remaining`;
      document.getElementById('metric-raw-rows').textContent = formatNumber(collection.raw_vendor_rows ?? 0);
      document.getElementById('metric-updated').textContent = `Updated ${new Date().toLocaleTimeString()}`;
      document.getElementById('metric-gate').textContent = readiness.ready ? 'Ready' : 'Blocked';
      document.getElementById('metric-gate-detail').textContent = `${collection.training_critical_percent ?? 0}% critical coverage`;
      document.getElementById('metric-eta').textContent = formatEta(eta);
      document.getElementById('metric-freeze-cutoff').textContent = freezeCutoff.date ? `Freeze cutoff ${freezeCutoff.date}` : 'No freeze cutoff';
      document.getElementById('budgets-updated').textContent = `Live snapshot ${new Date().toLocaleTimeString()}`;
      renderBudgets(snapshot);
    }

    function renderBudgets(snapshot) {
      const budget = snapshot.budget_summary || {};
      renderKeyValue('budget-summary', [
        ['Available vendors', formatNumber(budget.available_vendors ?? 0)],
        ['Day capped', formatNumber(budget.day_capped_vendors ?? 0)],
        ['Minute capped', formatNumber(budget.minute_capped_vendors ?? 0)],
        ['Checked at', budget.checked_at ?? '-'],
        ['Snapshot stale', budget.stale ? `Yes (${formatNumber(budget.snapshot_age_seconds ?? 0)}s)` : 'No'],
      ]);
      renderTable('budget-table', budget.rows || []);
    }

    function renderStatus(snapshot) {
      const collection = snapshot.collection_status || {};
      const readiness = snapshot.training_readiness || {};
      const phase1 = readiness.phase1 || {};
      const freezeCutoff = readiness.freeze_cutoff || {};
      const command = (snapshot.suggested_training_commands || {}).phase1 || '';
      const readinessPill = document.getElementById('status-readiness-pill');
      readinessPill.textContent = phase1.ready ? 'Phase 1 ready' : 'Phase 1 blocked';
      readinessPill.className = `pill ${phase1.ready ? 'good' : 'warn'}`;
      renderKeyValue('training-summary', [
        ['Latest raw date', snapshot.latest_raw_date ?? '-'],
        ['Pending tasks', formatNumber(collection.pending_tasks ?? 0)],
        ['Failed tasks', formatNumber(collection.failed_tasks ?? 0)],
        ['Freeze cutoff', freezeCutoff.date ?? '-'],
        ['Pinned cutoff', freezeCutoff.pinned ? 'Yes' : 'No'],
        ['Critical coverage', `${collection.training_critical_percent ?? 0}%`],
        ['Rolling remaining', formatNumber(collection.rolling_remaining_units ?? 0)],
        ['Repair remaining', formatNumber(collection.repair_remaining_units ?? 0)],
      ]);
      document.getElementById('training-command').textContent = command || 'No training command available yet.';
      document.getElementById('readiness-json').textContent = JSON.stringify(readiness, null, 2);
      document.getElementById('status-updated').textContent = `Status refreshed ${new Date().toLocaleTimeString()}`;
      renderTable('coverage-table', snapshot.dataset_coverage ? Object.values(snapshot.dataset_coverage).map((details) => ({
        dataset: details.label,
        coverage_percent: `${Math.round((details.ratio || 0) * 1000) / 10}%`,
        remaining_units: formatNumber(details.remaining_units || 0),
        eta: formatEta(details.eta_minutes),
        blocking: details.blocking ? 'yes' : 'no',
      })) : []);
      renderTable('vendor-table', (snapshot.vendor_throughput || {}).rows || []);
    }

    function renderSetup(snapshot) {
      const nas = snapshot.nas || {};
      const cluster = snapshot.cluster || {};
      const systemd = snapshot.systemd || {};
      document.getElementById('input-nas-share').value = nas.share || '';
      document.getElementById('input-nas-mount').value = nas.mount_path || '';
      if (!document.getElementById('input-collection-time').value) {
        document.getElementById('input-collection-time').value = '16:30';
      }
      if (!document.getElementById('input-maintenance-hour').value) {
        document.getElementById('input-maintenance-hour').value = '2';
      }
      renderKeyValue('setup-summary', [
        ['Host', nas.host || '-'],
        ['Host reachable', String(nas.host_reachable)],
        ['Mount writable', String(nas.mount_writable)],
        ['Systemd', systemd.ActiveState || systemd.reason || '-'],
        ['Workers', formatNumber((cluster.workers || []).length)],
        ['Leases', formatNumber((cluster.leases || []).length)],
      ]);
      document.getElementById('setup-json').textContent = JSON.stringify(snapshot, null, 2);
      document.getElementById('setup-updated').textContent = `Setup refreshed ${new Date().toLocaleTimeString()}`;
      setupLoaded = true;
    }

    function renderLogs(snapshot) {
      document.getElementById('node-log').textContent = snapshot.log_tail || 'No node log yet.';
      document.getElementById('journal-log').textContent = snapshot.journal_tail || 'No journal entries.';
      document.getElementById('logs-updated').textContent = `Logs refreshed ${new Date().toLocaleTimeString()}`;
      logsLoaded = true;
    }

    async function fetchJson(path, options={}) {
      const response = await fetch(path, options);
      if (!response.ok) {
        throw new Error(await response.text());
      }
      return await response.json();
    }

    async function refreshStatus() {
      const snapshot = await fetchJson('/api/status');
      renderStatus(snapshot);
    }

    async function refreshSetup(force=false) {
      if (!force && setupLoaded && activeSection !== 'setup') return;
      const snapshot = await fetchJson('/api/setup');
      renderSetup(snapshot);
    }

    async function refreshLogs(force=false) {
      if (!force && logsLoaded && activeSection !== 'logs') return;
      const snapshot = await fetchJson('/api/logs');
      renderLogs(snapshot);
    }

    async function refreshActiveSection(force=false) {
      if (activeSection === 'status') {
        await refreshStatus();
        return;
      }
      if (activeSection === 'setup') {
        await refreshSetup(force);
        return;
      }
      if (activeSection === 'logs') {
        await refreshLogs(force);
      }
    }

    async function refreshLiveOnce() {
      const snapshot = await fetchJson('/api/live');
      renderLive(snapshot);
    }

    function setSection(section) {
      activeSection = section;
      for (const name of sections) {
        document.getElementById(`section-${name}`).hidden = name !== section;
      }
      document.querySelectorAll('[data-section]').forEach((button) => {
        const active = button.dataset.section === section;
        button.className = active ? 'active' : 'ghost';
      });
      refreshActiveSection(true).catch((error) => setMessage('action-message', error.message, 'bad'));
    }

    function clusterPassphrase() {
      return document.getElementById('cluster-passphrase').value.trim();
    }

    async function runAction(action, payload={}) {
      try {
        setMessage('action-message', `${action} in progress...`, 'warn');
        const result = await fetchJson(`/api/actions/${action}`, {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(payload),
        });
        setMessage('action-message', `${action} complete`, 'good');
        await refreshLiveOnce();
        await refreshActiveSection(true);
        if (activeSection === 'setup') {
          document.getElementById('setup-json').textContent = JSON.stringify(result, null, 2);
        }
      } catch (error) {
        setMessage('action-message', `${action} failed: ${error.message}`, 'bad');
        throw error;
      }
    }

    document.querySelectorAll('[data-action]').forEach((button) => {
      button.addEventListener('click', async () => {
        const action = button.dataset.action;
        const payload = {};
        if (['start-node', 'restart-node', 'join-cluster', 'reset-worker'].includes(action) && clusterPassphrase()) {
          payload.passphrase = clusterPassphrase();
        }
        if (action === 'install-service') {
          payload.service_path = '';
        }
        if (action === 'rebuild-state') {
          payload.passphrase = clusterPassphrase() || undefined;
        }
        if (activeSection === 'setup') {
          setMessage('setup-message', `${action} in progress...`, 'warn');
        }
        try {
          await runAction(action, payload);
          if (activeSection === 'setup') {
            setMessage('setup-message', `${action} complete`, 'good');
          }
        } catch (error) {
          if (activeSection === 'setup') {
            setMessage('setup-message', `${action} failed: ${error.message}`, 'bad');
          }
        }
      });
    });

    document.getElementById('save-settings').addEventListener('click', async () => {
      const payload = {
        nas_share: document.getElementById('input-nas-share').value.trim(),
        nas_mount: document.getElementById('input-nas-mount').value.trim(),
        collection_time_et: document.getElementById('input-collection-time').value.trim(),
        maintenance_hour_local: Number(document.getElementById('input-maintenance-hour').value || '0'),
        fstab_path: document.getElementById('input-fstab-path').value.trim(),
      };
      try {
        setMessage('setup-message', 'Saving settings...', 'warn');
        const result = await fetchJson('/api/actions/save-settings', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(payload),
        });
        document.getElementById('setup-json').textContent = JSON.stringify(result, null, 2);
        setMessage('setup-message', 'Settings saved', 'good');
        await refreshSetup(true);
      } catch (error) {
        setMessage('setup-message', `Save failed: ${error.message}`, 'bad');
      }
    });

    document.getElementById('update-secret').addEventListener('click', async () => {
      try {
        setMessage('setup-message', 'Updating secret...', 'warn');
        const payload = {
          passphrase: clusterPassphrase(),
          key: document.getElementById('input-secret-key').value.trim(),
          value: document.getElementById('input-secret-value').value,
        };
        const result = await fetchJson('/api/actions/update-secret', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(payload),
        });
        document.getElementById('setup-json').textContent = JSON.stringify(result, null, 2);
        setMessage('setup-message', 'Secret updated', 'good');
      } catch (error) {
        setMessage('setup-message', `Secret update failed: ${error.message}`, 'bad');
      }
    });

    document.getElementById('rotate-passphrase').addEventListener('click', async () => {
      try {
        setMessage('setup-message', 'Rotating passphrase...', 'warn');
        const payload = {
          old_passphrase: document.getElementById('input-old-passphrase').value,
          new_passphrase: document.getElementById('input-new-passphrase').value,
        };
        const result = await fetchJson('/api/actions/rotate-passphrase', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(payload),
        });
        document.getElementById('setup-json').textContent = JSON.stringify(result, null, 2);
        setMessage('setup-message', 'Passphrase rotated', 'good');
      } catch (error) {
        setMessage('setup-message', `Passphrase rotation failed: ${error.message}`, 'bad');
      }
    });

    document.getElementById('force-release-lease').addEventListener('click', async () => {
      try {
        setMessage('setup-message', 'Releasing lease...', 'warn');
        const payload = {lease_id: document.getElementById('input-lease-id').value.trim()};
        const result = await fetchJson('/api/actions/force-release-lease', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(payload),
        });
        document.getElementById('setup-json').textContent = JSON.stringify(result, null, 2);
        setMessage('setup-message', 'Lease released', 'good');
      } catch (error) {
        setMessage('setup-message', `Force release failed: ${error.message}`, 'bad');
      }
    });

    function connectLiveStream() {
      const badge = document.getElementById('connection-badge');
      badge.textContent = 'Connecting';
      const source = new EventSource('/api/live/stream');
      source.onmessage = (event) => {
        badge.textContent = 'Live';
        renderLive(JSON.parse(event.data));
      };
      source.onerror = () => {
        badge.textContent = 'Reconnecting';
        source.close();
        setTimeout(connectLiveStream, 1000);
      };
    }

    setSection('status');
    refreshLiveOnce().catch((error) => setMessage('action-message', error.message, 'bad'));
    refreshStatus().catch((error) => setMessage('action-message', error.message, 'bad'));
    connectLiveStream();
    statusTimer = setInterval(() => {
      refreshActiveSection(false).catch((error) => setMessage('action-message', error.message, 'bad'));
    }, 5000);
  </script>
</body>
</html>
"""


class DashboardHTTPServer(ThreadingHTTPServer):
    """HTTP server that holds resolved TradeML dashboard settings."""

    daemon_threads = True

    def __init__(self, server_address: tuple[str, int], *, settings: NodeSettings) -> None:
        super().__init__(server_address, DashboardRequestHandler)
        self.settings = settings


class DashboardRequestHandler(BaseHTTPRequestHandler):
    """Serve the operator dashboard HTML and JSON action endpoints."""

    server: DashboardHTTPServer

    def do_HEAD(self) -> None:  # noqa: N802
        if self.path in {"/", "/api/live", "/api/status", "/api/setup", "/api/logs", "/_stcore/health", "/_stcore/host-config"}:
            content_type = "text/html; charset=utf-8" if self.path == "/" else "application/json; charset=utf-8"
            self._write_headers(status=HTTPStatus.OK, content_type=content_type)
            return
        self.send_error(HTTPStatus.NOT_FOUND, "not found")

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/":
            self._write_html(HTML_PAGE)
            return
        if self.path == "/_stcore/health":
            self._write_json({"ok": True})
            return
        if self.path == "/_stcore/host-config":
            self._write_json({"useExternalAuthToken": False, "enableCustomParentMessages": False})
            return
        if self.path == "/api/live":
            self._write_json(collect_dashboard_live_snapshot(self.server.settings))
            return
        if self.path == "/api/status":
            self._write_json(collect_dashboard_status_snapshot(self.server.settings))
            return
        if self.path == "/api/setup":
            self._write_json(collect_dashboard_setup_snapshot(self.server.settings))
            return
        if self.path == "/api/logs":
            self._write_json(collect_dashboard_logs_snapshot(self.server.settings))
            return
        if self.path == "/api/live/stream":
            self._serve_live_stream()
            return
        self.send_error(HTTPStatus.NOT_FOUND, "not found")

    def do_POST(self) -> None:  # noqa: N802
        if not self.path.startswith("/api/actions/"):
            self.send_error(HTTPStatus.NOT_FOUND, "not found")
            return
        action = self.path.removeprefix("/api/actions/")
        try:
            payload = self._read_json_body()
            result = dispatch_dashboard_action(self.server.settings, action, payload)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("dashboard_action_failed action=%s", action)
            self._write_json({"ok": False, "error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        self._write_json({"ok": True, "result": result})

    def log_message(self, fmt: str, *args: Any) -> None:
        LOGGER.info("dashboard_http " + fmt, *args)

    def _read_json_body(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0") or "0")
        if content_length <= 0:
            return {}
        raw = self.rfile.read(content_length)
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def _write_headers(self, *, status: HTTPStatus, content_type: str, extra_headers: dict[str, str] | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Pragma", "no-cache")
        if extra_headers:
            for key, value in extra_headers.items():
                self.send_header(key, value)
        self.end_headers()

    def _write_json(self, payload: dict[str, Any], *, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, default=str).encode("utf-8")
        self._write_headers(status=status, content_type="application/json; charset=utf-8", extra_headers={"Content-Length": str(len(body))})
        self.wfile.write(body)

    def _write_html(self, html: str) -> None:
        body = html.encode("utf-8")
        self._write_headers(status=HTTPStatus.OK, content_type="text/html; charset=utf-8", extra_headers={"Content-Length": str(len(body))})
        self.wfile.write(body)

    def _serve_live_stream(self) -> None:
        self._write_headers(
            status=HTTPStatus.OK,
            content_type="text/event-stream; charset=utf-8",
            extra_headers={"Connection": "keep-alive", "X-Accel-Buffering": "no"},
        )
        while True:
            try:
                payload = json.dumps(collect_dashboard_live_snapshot(self.server.settings), default=str)
                self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                self.wfile.flush()
                time.sleep(1.0)
            except (BrokenPipeError, ConnectionResetError):
                return


def dispatch_dashboard_action(settings: NodeSettings, action: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Run a dashboard action endpoint against controller helpers."""
    if action == "start-node":
        return start_node(settings, passphrase=_optional_str(payload.get("passphrase")))
    if action == "stop-node":
        return stop_node(settings)
    if action == "restart-node":
        return restart_node(settings, passphrase=_optional_str(payload.get("passphrase")))
    if action == "join-cluster":
        return join_cluster(settings, passphrase=_optional_str(payload.get("passphrase")))
    if action == "rebuild-state":
        return rebuild_cluster_state(settings, passphrase=_optional_str(payload.get("passphrase")))
    if action == "leave-cluster":
        return leave_cluster(settings)
    if action == "install-service":
        return install_service(settings, service_path=_optional_str(payload.get("service_path")))
    if action == "update-worker":
        return update_worker(settings)
    if action == "reset-worker":
        return reset_worker(settings, passphrase=_optional_str(payload.get("passphrase")))
    if action == "uninstall-worker":
        return uninstall_worker(settings)
    if action == "run-vendor-audit":
        return run_vendor_audit(settings)
    if action == "replan-coverage":
        return replan_coverage(settings)
    if action == "save-settings":
        return persist_node_settings(
            settings,
            nas_share=str(payload.get("nas_share") or settings.nas_share),
            nas_mount=str(payload.get("nas_mount") or settings.nas_mount),
            collection_time_et=str(payload.get("collection_time_et") or settings.collection_time_et),
            maintenance_hour_local=int(payload.get("maintenance_hour_local") or settings.maintenance_hour_local),
            fstab_path=_optional_str(payload.get("fstab_path")),
        )
    if action == "update-secret":
        key = str(payload.get("key") or "").strip()
        if not key:
            raise ValueError("secret key is required")
        passphrase = _optional_str(payload.get("passphrase"))
        if not passphrase:
            raise ValueError("cluster passphrase is required")
        return update_cluster_secrets(settings, passphrase=passphrase, updates={key: str(payload.get("value") or "")})
    if action == "rotate-passphrase":
        old_passphrase = _optional_str(payload.get("old_passphrase"))
        new_passphrase = _optional_str(payload.get("new_passphrase"))
        if not old_passphrase or not new_passphrase:
            raise ValueError("old and new passphrases are required")
        return rotate_cluster_passphrase(settings, old_passphrase=old_passphrase, new_passphrase=new_passphrase)
    if action == "force-release-lease":
        lease_id = str(payload.get("lease_id") or "").strip()
        if not lease_id:
            raise ValueError("lease_id is required")
        return {"released": force_release_lease(settings, lease_id), "lease_id": lease_id}
    if action == "advance-stage":
        target_stage = int(payload.get("target_stage") or 0)
        return advance_collection_stage(
            settings,
            target_stage=target_stage,
            symbol_count=int(payload.get("symbol_count")) if payload.get("symbol_count") is not None else None,
            years=int(payload.get("years")) if payload.get("years") is not None else None,
            passphrase=_optional_str(payload.get("passphrase")),
        )
    raise ValueError(f"unsupported action: {action}")


def create_dashboard_server(host: str, port: int, *, settings: NodeSettings) -> DashboardHTTPServer:
    """Create the threaded HTTP server bound to the requested interface."""
    return DashboardHTTPServer((host, port), settings=settings)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _open_browser_later(url: str) -> None:
    time.sleep(0.4)
    with threading.Lock():
        webbrowser.open(url)


def main(argv: list[str] | None = None) -> int:
    """Run the browser-served operator dashboard."""
    parser = argparse.ArgumentParser(description="TradeML dashboard server")
    parser.add_argument("--workspace-root", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--env-file", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    settings = resolve_node_settings(
        workspace_root=args.workspace_root,
        config_path=args.config,
        env_path=args.env_file,
    )
    httpd = create_dashboard_server(args.host, args.port, settings=settings)
    url = f"http://{args.host}:{httpd.server_port}"
    LOGGER.info("dashboard_listening url=%s workspace_root=%s", url, settings.workspace_root)
    if not args.no_browser:
        threading.Thread(target=_open_browser_later, args=(url,), daemon=True).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("dashboard_shutdown requested")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
