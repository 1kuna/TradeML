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
    backtest_experiments,
    bootstrap_canonical_ledger,
    collect_dashboard_health_snapshot,
    collect_dashboard_live_snapshot,
    collect_dashboard_logs_snapshot,
    collect_dashboard_setup_snapshot,
    collect_dashboard_status_snapshot,
    force_release_lease,
    install_service,
    join_cluster,
    leave_cluster,
    pause_experiments,
    persist_node_settings,
    propose_experiment_family,
    rebuild_cluster_state,
    replan_coverage,
    resolve_node_settings,
    resume_experiments,
    restart_node,
    rotate_cluster_passphrase,
    lane_health,
    evaluate_experiments,
    repair_canonical_backlog,
    repair_status,
    run_vendor_audit,
    start_experiment_supervisor,
    start_node,
    start_training_run,
    stop_node,
    stop_experiments,
    stop_training_run,
    training_preflight_status,
    training_runtime_logs,
    training_runtime_status,
    uninstall_worker,
    update_worker,
    update_cluster_secrets,
    verify_recent_canonical_dates,
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
    .callout {
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 14px;
      background: rgba(8, 17, 28, 0.42);
    }
    #training-log-preview {
      max-height: 220px;
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
            <div class="badge">Collector live</div>
            <div class="badge">Remote training ready</div>
            <div class="badge">Experiment queue automation</div>
          </div>
        </div>
        <div class="actions-panel">
          <div class="summary-meta">
            <div>
              <div class="label">Live Control</div>
              <div class="delta">Only the controls that matter casually live here now.</div>
            </div>
            <div class="badge">Quick glance mode</div>
          </div>
          <input id="cluster-passphrase" type="password" placeholder="Cluster passphrase">
          <div class="form-row">
            <button data-action="start-node">Start Node</button>
            <button data-action="stop-node" class="secondary">Stop Node</button>
            <button data-action="restart-node" class="secondary">Restart Node</button>
          </div>
          <span id="action-message" class="message"></span>
        </div>
      </div>
    </div>

    <div class="summary-grid">
      <div class="card"><div class="label">Node</div><div class="value" id="metric-node-status">-</div><div class="delta mono" id="metric-node-detail">-</div></div>
      <div class="card"><div class="label">Total Datapoints</div><div class="value" id="metric-coverage">-</div><div class="delta" id="metric-coverage-detail">-</div></div>
      <div class="card"><div class="label">Phase 1</div><div class="value" id="metric-canonical">-</div><div class="delta" id="metric-remaining">-</div></div>
      <div class="card"><div class="label">Research Queue</div><div class="value" id="metric-raw-rows">-</div><div class="delta" id="metric-updated">-</div></div>
      <div class="card"><div class="label">Training</div><div class="value" id="metric-gate">-</div><div class="delta" id="metric-gate-detail">-</div></div>
      <div class="card"><div class="label">Best Result</div><div class="value" id="metric-eta">-</div><div class="delta" id="metric-freeze-cutoff">-</div></div>
    </div>

    <section class="panel" id="section-status">
      <div class="toolbar">
        <div>
          <h2>Quick Glance</h2>
          <div class="small muted">Collector progress, training state, and research outcomes in one view.</div>
        </div>
        <span class="small muted" id="status-updated">Waiting for first snapshot</span>
      </div>
      <div class="detail-grid">
        <div class="stack">
          <div class="section-card">
            <div class="status-line">
              <h3>Progress Board</h3>
              <span class="pill" id="status-readiness-pill">Unknown</span>
            </div>
            <div class="kv" id="progress-board"></div>
          </div>
          <div class="section-card forms">
            <div class="status-line">
              <h3>Training Desk</h3>
              <span class="pill" id="training-status-pill">Idle</span>
            </div>
            <div class="form-row">
              <select id="training-target"></select>
              <button id="training-preflight" class="secondary">Preflight</button>
              <button id="training-start">Start</button>
              <button id="training-stop" class="ghost">Stop</button>
            </div>
            <div class="kv" id="training-runtime-summary"></div>
            <div>
              <div class="label">Live Log</div>
              <pre id="training-log-preview"></pre>
            </div>
          </div>
          <div class="section-card">
            <h3 style="margin-bottom:12px">Recent Runs</h3>
            <div class="table-wrap"><table id="recent-runs-table"></table></div>
          </div>
        </div>
        <div class="stack">
          <div class="section-card">
            <h3 style="margin-bottom:12px">Best Model So Far</h3>
            <div class="kv" id="winner-board"></div>
          </div>
          <div class="section-card">
            <h3 style="margin-bottom:12px">Research Pulse</h3>
            <div class="kv" id="research-board"></div>
          </div>
          <div class="section-card">
            <h3 style="margin-bottom:12px">Next Up</h3>
            <div class="callout">
              <div class="kv" id="next-board"></div>
            </div>
          </div>
          <div class="section-card forms">
            <h3>Research Controls</h3>
            <input id="input-experiment-spec" placeholder="Experiment spec path" value="configs/experiments/phase1_remote_baseline_sweep.yml">
            <input id="input-experiment-id" placeholder="Experiment ID">
            <input id="input-experiment-poll-seconds" placeholder="Poll seconds" value="30">
            <div class="form-row">
              <button id="start-experiment-supervisor">Start Queue</button>
              <button id="pause-experiment-supervisor" class="secondary">Pause</button>
              <button id="resume-experiment-supervisor" class="secondary">Resume</button>
              <button id="stop-experiment-supervisor" class="ghost">Stop</button>
            </div>
            <div class="form-row">
              <button id="evaluate-experiment" class="secondary">Evaluate</button>
              <button id="backtest-experiment">Backtest Survivors</button>
              <button id="propose-next-experiment" class="secondary">Propose Next</button>
            </div>
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
        <div class="section-card forms">
          <h3>Ledger and Repair</h3>
          <input id="input-repair-date" placeholder="Repair date (YYYY-MM-DD)">
          <input id="input-repair-start-date" placeholder="Start date (YYYY-MM-DD)">
          <input id="input-repair-end-date" placeholder="End date (YYYY-MM-DD)">
          <input id="input-repair-symbol" placeholder="Symbol">
          <div class="form-row">
            <button id="bootstrap-ledger" class="secondary">Bootstrap Ledger</button>
            <button id="verify-canonical-repair" class="secondary">Verify Repair</button>
            <button id="run-canonical-repair">Run Repair</button>
          </div>
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
    const sections = ['status'];
    let activeSection = 'status';
    let currentLiveSnapshot = null;
    let currentStatusSnapshot = null;
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

    function formatDecimal(value, decimals=3) {
      if (value === null || value === undefined || value === '') return '-';
      const numeric = Number(value);
      if (!Number.isFinite(numeric)) return '-';
      return numeric.toFixed(decimals);
    }

    function formatPercent(value, decimals=1) {
      if (value === null || value === undefined || value === '') return '-';
      const numeric = Number(value);
      if (!Number.isFinite(numeric)) return '-';
      return `${numeric.toFixed(decimals)}%`;
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
      document.getElementById('metric-node-status').textContent = runtime.running ? 'Running' : 'Stopped';
      document.getElementById('metric-node-detail').textContent = runtime.pid ? `PID ${runtime.pid}` : 'Waiting for worker';
      document.getElementById('metric-coverage').textContent = formatNumber(collection.raw_vendor_rows ?? 0);
      document.getElementById('metric-coverage-detail').textContent = currentStatusSnapshot?.latest_raw_date
        ? `Latest raw ${currentStatusSnapshot.latest_raw_date}`
        : 'Waiting for raw data';
      const pinnedRemaining = collection.phase1_pinned_remaining_units;
      const rollingRemaining = collection.rolling_remaining_units ?? 0;
      const repairRemaining = collection.repair_remaining_units ?? 0;
      const totalRemaining = collection.canonical_remaining_units ?? (rollingRemaining + repairRemaining);
      document.getElementById('metric-canonical').textContent = readiness.ready
        ? 'Ready'
        : formatPercent(collection.training_critical_percent ?? 0);
      document.getElementById('metric-remaining').textContent = readiness.ready
        ? 'Pinned gate locked'
        : `${formatNumber(pinnedRemaining ?? rollingRemaining)} pinned left`;
      document.getElementById('metric-raw-rows').textContent = totalRemaining > 0
        ? formatNumber(totalRemaining)
        : 'Clear';
      document.getElementById('metric-updated').textContent = `${formatNumber(collection.rolling_remaining_units ?? 0)} rolling · ${formatNumber(repairRemaining)} repair`;
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
      currentStatusSnapshot = snapshot;
      const collection = snapshot.collection_status || {};
      const readiness = snapshot.training_readiness || {};
      const phase1 = readiness.phase1 || {};
      const freezeCutoff = readiness.freeze_cutoff || {};
      const health = snapshot.health || {};
      const trainingStatus = (snapshot.training_status || {}).phase1 || {};
      const trainingRuntime = trainingStatus.runtime || trainingStatus || {};
      const defaultTarget = snapshot.default_training_target || {};
      const targets = snapshot.training_targets || [];
      const repairTasks = health.repair_tasks || {};
      const experiment = snapshot.experiment_summary || {};
      const supervisor = health.experiment_supervisor || {};
      const proposal = health.proposal_summary || {};
      const readinessPill = document.getElementById('status-readiness-pill');
      readinessPill.textContent = phase1.ready ? 'Phase 1 ready' : 'Phase 1 blocked';
      readinessPill.className = `pill ${phase1.ready ? 'good' : 'warn'}`;
      const trainingPill = document.getElementById('training-status-pill');
      const runtimeStatus = String(trainingRuntime.status || 'idle').toLowerCase();
      trainingPill.textContent = trainingRuntime.running || runtimeStatus === 'running'
        ? 'Training live'
        : runtimeStatus === 'completed'
          ? 'Last run complete'
          : 'Idle';
      trainingPill.className = `pill ${
        trainingRuntime.running || runtimeStatus === 'running'
          ? 'good'
          : runtimeStatus === 'failed'
            ? 'bad'
            : 'warn'
      }`;
      const targetSelect = document.getElementById('training-target');
      const selectedTarget = targetSelect.value || defaultTarget.name || '';
      targetSelect.innerHTML = targets.map((target) => {
        const selected = (target.name === selectedTarget || (!selectedTarget && target.default)) ? 'selected' : '';
        return `<option value="${target.name}" ${selected}>${target.label || target.name}</option>`;
      }).join('');
      renderKeyValue('progress-board', [
        ['Total datapoints', formatNumber(snapshot.raw_datapoints ?? 0)],
        ['Latest raw date', snapshot.latest_raw_date ?? '-'],
        ['Phase 1', phase1.ready ? 'Ready to train' : 'Still filling'],
        ['Freeze cutoff', freezeCutoff.date ?? '-'],
        ['Critical coverage', `${collection.training_critical_percent ?? 0}%`],
        ['Pinned remaining', formatNumber(collection.phase1_pinned_remaining_units ?? 0)],
        ['Rolling remaining', formatNumber(collection.rolling_remaining_units ?? 0)],
        ['Repair remaining', formatNumber(collection.repair_remaining_units ?? 0)],
        ['Pending tasks', formatNumber(collection.pending_tasks ?? 0)],
        ['Bars ETA', formatEta((((snapshot.planner_eta || {}).canonical_bars) || {}).eta_minutes)],
      ]);
      renderKeyValue('training-runtime-summary', [
        ['Target', trainingRuntime.target || defaultTarget.name || '-'],
        ['Status', trainingRuntime.status || '-'],
        ['Host', trainingRuntime.host || defaultTarget.host || '-'],
        ['PID', trainingRuntime.pid ? formatNumber(trainingRuntime.pid) : '-'],
        ['Report date', trainingRuntime.report_date || freezeCutoff.date || '-'],
        ['Started', trainingRuntime.started_at || '-'],
        ['Finished', trainingRuntime.finished_at || '-'],
      ]);
      document.getElementById('training-log-preview').textContent = trainingStatus.log_tail || 'No training log available yet.';
      renderKeyValue('winner-board', [
        ['Experiment', experiment.experiment_id || '-'],
        ['Best candidate', experiment.best_candidate || 'No winner yet'],
        ['Primary score', formatDecimal(experiment.best_primary_score)],
        ['Backtest net', formatDecimal(experiment.best_backtest_net_return)],
        ['Decision', experiment.best_decision || 'No GO yet'],
        ['Reason', experiment.best_decision_reason || '-'],
      ]);
      const queueActive = Boolean(supervisor.status || experiment.experiment_id);
      renderKeyValue('research-board', [
        ['Queue status', queueActive ? 'Active' : 'Idle'],
        ['Runs', `${formatNumber(experiment.run_count ?? 0)} total`],
        ['Running', formatNumber((experiment.counts || {}).RUNNING ?? 0)],
        ['Completed', formatNumber((experiment.counts || {}).COMPLETED ?? 0)],
        ['Shortlisted', formatNumber(experiment.shortlist_count ?? 0)],
        ['Predictive survivors', formatNumber((experiment.evaluation_counts || {}).SURVIVES_PREDICTIVE ?? 0)],
        ['Top rejection', ((experiment.top_gate_failures || [])[0] || []).join(': ') || '-'],
        ['Repair ETA', formatEta(health.repair_eta_minutes)],
      ]);
      const runningRuns = formatNumber((experiment.counts || {}).RUNNING ?? 0);
      const plannedRuns = formatNumber((experiment.counts || {}).PLANNED ?? 0);
      let nextHeadline = 'Start the next research wave';
      let nextReason = 'No active training run or queued experiment family right now.';
      if (!phase1.ready) {
        nextHeadline = 'Finish the pinned Phase 1 tail';
        nextReason = `${formatNumber(collection.phase1_pinned_remaining_units ?? 0)} pinned units still block training readiness.`;
      } else if (trainingRuntime.running || runtimeStatus === 'running') {
        nextHeadline = 'Let the current training run finish';
        nextReason = `${trainingRuntime.target || defaultTarget.name || 'remote target'} is actively training on ${trainingRuntime.host || defaultTarget.host || 'the current host'}.`;
      } else if ((repairTasks.remaining_units ?? 0) > 0) {
        nextHeadline = 'Clear the repair tail';
        nextReason = `${formatNumber(repairTasks.remaining_units ?? 0)} repair units remain across ${(health.recent_bad_dates || []).length} flagged dates.`;
      } else if ((experiment.counts || {}).RUNNING || (experiment.counts || {}).PLANNED || supervisor.status === 'RUNNING') {
        nextHeadline = 'Drain the experiment queue';
        nextReason = `${runningRuns} running and ${plannedRuns} planned in ${experiment.experiment_id || 'the current family'}.`;
      } else if ((experiment.shortlist_count ?? 0) > 0) {
        nextHeadline = 'Review the shortlist';
        nextReason = `${formatNumber(experiment.shortlist_count ?? 0)} candidate runs survived the current research funnel.`;
      } else if (proposal.recommended_experiment_id) {
        nextHeadline = 'Launch the next bounded family';
        nextReason = proposal.recommended_experiment_id;
      }
      renderKeyValue('next-board', [
        ['Move', nextHeadline],
        ['Why', nextReason],
        ['Proposal', proposal.recommended_experiment_id || '-'],
        ['Data lane', ((proposal.data_recommendations || [])[0]) || '-'],
      ]);
      renderTable('recent-runs-table', (experiment.runs || []).slice(0, 6).map((run) => ({
        run_id: run.run_id,
        suite: run.model_suite,
        stage: run.evaluation_stage || run.status,
        shortlisted: run.shortlisted ? 'yes' : 'no',
      })));
      document.getElementById('metric-gate').textContent = trainingRuntime.running || runtimeStatus === 'running'
        ? 'Running'
        : trainingRuntime.status === 'completed'
          ? 'Complete'
          : 'Idle';
      document.getElementById('metric-gate-detail').textContent = trainingRuntime.target || defaultTarget.name || 'No target selected';
      document.getElementById('metric-eta').textContent = experiment.best_candidate || 'No winner';
      document.getElementById('metric-freeze-cutoff').textContent = experiment.best_decision
        ? `${experiment.best_decision} · ${formatDecimal(experiment.best_primary_score)}`
        : 'No promoted run yet';
      document.getElementById('status-updated').textContent = `Status refreshed ${new Date().toLocaleTimeString()}`;
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
      await refreshStatus();
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

    document.getElementById('training-preflight').addEventListener('click', async () => {
      await runAction('train-preflight', {phase: 1, target: document.getElementById('training-target').value});
    });

    document.getElementById('training-start').addEventListener('click', async () => {
      const freeze = (((currentLiveSnapshot || {}).training_readiness || {}).freeze_cutoff || {}).date || null;
      await runAction('train-start', {
        phase: 1,
        target: document.getElementById('training-target').value,
        report_date: freeze,
      });
    });

    document.getElementById('training-stop').addEventListener('click', async () => {
      await runAction('train-stop', {phase: 1, target: document.getElementById('training-target').value});
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

    function canonicalRepairPayload(verifyOnly=false) {
      const payload = {};
      const tradingDate = document.getElementById('input-repair-date').value.trim();
      const startDate = document.getElementById('input-repair-start-date').value.trim();
      const endDate = document.getElementById('input-repair-end-date').value.trim();
      const symbol = document.getElementById('input-repair-symbol').value.trim();
      if (tradingDate) payload.trading_date = tradingDate;
      if (startDate) payload.start_date = startDate;
      if (endDate) payload.end_date = endDate;
      if (symbol) payload.symbol = symbol;
      if (verifyOnly) payload.verify_only = true;
      return payload;
    }

    function experimentPayload() {
      const payload = {};
      const experimentId = document.getElementById('input-experiment-id').value.trim();
      const specPath = document.getElementById('input-experiment-spec').value.trim();
      const pollSeconds = document.getElementById('input-experiment-poll-seconds').value.trim();
      if (experimentId) payload.experiment_id = experimentId;
      if (specPath) payload.spec_path = specPath;
      if (pollSeconds) payload.poll_seconds = Number(pollSeconds);
      return payload;
    }

    document.getElementById('bootstrap-ledger').addEventListener('click', async () => {
      try {
        setMessage('setup-message', 'Bootstrapping canonical ledger...', 'warn');
        const result = await fetchJson('/api/actions/bootstrap-ledger', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({}),
        });
        document.getElementById('setup-json').textContent = JSON.stringify(result, null, 2);
        setMessage('setup-message', 'Canonical ledger bootstrapped', 'good');
        await refreshLiveOnce();
        await refreshSetup(true);
        await refreshStatus();
      } catch (error) {
        setMessage('setup-message', `Bootstrap failed: ${error.message}`, 'bad');
      }
    });

    document.getElementById('verify-canonical-repair').addEventListener('click', async () => {
      try {
        setMessage('setup-message', 'Verifying canonical repairs...', 'warn');
        const result = await fetchJson('/api/actions/repair-canonical', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(canonicalRepairPayload(true)),
        });
        document.getElementById('setup-json').textContent = JSON.stringify(result, null, 2);
        setMessage('setup-message', 'Canonical repair verification complete', 'good');
        await refreshLiveOnce();
        await refreshSetup(true);
        await refreshStatus();
      } catch (error) {
        setMessage('setup-message', `Repair verification failed: ${error.message}`, 'bad');
      }
    });

    document.getElementById('run-canonical-repair').addEventListener('click', async () => {
      try {
        setMessage('setup-message', 'Running canonical repairs...', 'warn');
        const result = await fetchJson('/api/actions/repair-canonical', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(canonicalRepairPayload(false)),
        });
        document.getElementById('setup-json').textContent = JSON.stringify(result, null, 2);
        setMessage('setup-message', 'Canonical repair run complete', 'good');
        await refreshLiveOnce();
        await refreshSetup(true);
        await refreshStatus();
      } catch (error) {
        setMessage('setup-message', `Repair run failed: ${error.message}`, 'bad');
      }
    });

    document.getElementById('start-experiment-supervisor').addEventListener('click', async () => {
      try {
        setMessage('setup-message', 'Starting experiment supervisor...', 'warn');
        const result = await fetchJson('/api/actions/experiments-supervise', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({...experimentPayload(), detach: true}),
        });
        document.getElementById('setup-json').textContent = JSON.stringify(result, null, 2);
        setMessage('setup-message', 'Experiment supervisor started', 'good');
        await refreshStatus();
      } catch (error) {
        setMessage('setup-message', `Experiment supervisor failed: ${error.message}`, 'bad');
      }
    });

    document.getElementById('pause-experiment-supervisor').addEventListener('click', async () => {
      try {
        setMessage('setup-message', 'Pausing experiment supervisor...', 'warn');
        const result = await fetchJson('/api/actions/experiments-pause', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(experimentPayload()),
        });
        document.getElementById('setup-json').textContent = JSON.stringify(result, null, 2);
        setMessage('setup-message', 'Experiment supervisor paused', 'good');
        await refreshStatus();
      } catch (error) {
        setMessage('setup-message', `Pause failed: ${error.message}`, 'bad');
      }
    });

    document.getElementById('resume-experiment-supervisor').addEventListener('click', async () => {
      try {
        setMessage('setup-message', 'Resuming experiment supervisor...', 'warn');
        const result = await fetchJson('/api/actions/experiments-resume', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(experimentPayload()),
        });
        document.getElementById('setup-json').textContent = JSON.stringify(result, null, 2);
        setMessage('setup-message', 'Experiment supervisor resumed', 'good');
        await refreshStatus();
      } catch (error) {
        setMessage('setup-message', `Resume failed: ${error.message}`, 'bad');
      }
    });

    document.getElementById('stop-experiment-supervisor').addEventListener('click', async () => {
      try {
        setMessage('setup-message', 'Stopping experiment supervisor...', 'warn');
        const result = await fetchJson('/api/actions/experiments-stop', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(experimentPayload()),
        });
        document.getElementById('setup-json').textContent = JSON.stringify(result, null, 2);
        setMessage('setup-message', 'Experiment supervisor stop requested', 'good');
        await refreshStatus();
      } catch (error) {
        setMessage('setup-message', `Stop failed: ${error.message}`, 'bad');
      }
    });

    document.getElementById('evaluate-experiment').addEventListener('click', async () => {
      try {
        setMessage('setup-message', 'Evaluating experiment...', 'warn');
        const result = await fetchJson('/api/actions/experiments-evaluate', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(experimentPayload()),
        });
        document.getElementById('setup-json').textContent = JSON.stringify(result, null, 2);
        setMessage('setup-message', 'Experiment evaluation complete', 'good');
        await refreshStatus();
      } catch (error) {
        setMessage('setup-message', `Evaluation failed: ${error.message}`, 'bad');
      }
    });

    document.getElementById('backtest-experiment').addEventListener('click', async () => {
      try {
        setMessage('setup-message', 'Running survivor backtests...', 'warn');
        const result = await fetchJson('/api/actions/experiments-backtest', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(experimentPayload()),
        });
        document.getElementById('setup-json').textContent = JSON.stringify(result, null, 2);
        setMessage('setup-message', 'Survivor backtests complete', 'good');
        await refreshStatus();
      } catch (error) {
        setMessage('setup-message', `Backtest failed: ${error.message}`, 'bad');
      }
    });

    document.getElementById('propose-next-experiment').addEventListener('click', async () => {
      try {
        setMessage('setup-message', 'Generating next experiment family...', 'warn');
        const result = await fetchJson('/api/actions/experiments-propose-next', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(experimentPayload()),
        });
        document.getElementById('setup-json').textContent = JSON.stringify(result, null, 2);
        setMessage('setup-message', 'Next experiment family ready', 'good');
        await refreshStatus();
      } catch (error) {
        setMessage('setup-message', `Proposal failed: ${error.message}`, 'bad');
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
        if self.path in {"/", "/api/live", "/api/status", "/api/health", "/api/setup", "/api/logs", "/_stcore/health", "/_stcore/host-config"}:
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
        if self.path == "/api/health":
            self._write_json(collect_dashboard_health_snapshot(self.server.settings))
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
    if action == "bootstrap-ledger":
        return bootstrap_canonical_ledger(settings)
    if action == "repair-canonical":
        return repair_canonical_backlog(
            settings,
            trading_date=_optional_str(payload.get("trading_date")),
            start_date=_optional_str(payload.get("start_date")),
            end_date=_optional_str(payload.get("end_date")),
            symbol=_optional_str(payload.get("symbol")),
            verify_only=bool(payload.get("verify_only")),
        )
    if action == "verify-recent":
        return verify_recent_canonical_dates(
            settings,
            days=int(payload.get("days") or 7),
            dataset=str(payload.get("dataset") or "equities_eod"),
            verify_only=bool(payload.get("verify_only")),
        )
    if action == "repair-status":
        return repair_status(settings)
    if action == "lane-health":
        return lane_health(settings, dataset=str(payload.get("dataset") or "equities_eod"))
    if action == "train-preflight":
        return training_preflight_status(
            settings,
            phase=int(payload.get("phase") or 1),
            target=_optional_str(payload.get("target")),
        )
    if action == "train-start":
        return start_training_run(
            settings,
            phase=int(payload.get("phase") or 1),
            report_date=_optional_str(payload.get("report_date")),
            target=_optional_str(payload.get("target")),
        )
    if action == "train-stop":
        return stop_training_run(
            settings,
            phase=int(payload.get("phase") or 1),
            target=_optional_str(payload.get("target")),
        )
    if action == "train-status":
        return training_runtime_status(
            settings,
            phase=int(payload.get("phase") or 1),
            target=_optional_str(payload.get("target")),
        )
    if action == "train-logs":
        return training_runtime_logs(
            settings,
            phase=int(payload.get("phase") or 1),
            target=_optional_str(payload.get("target")),
            tail_lines=int(payload.get("tail_lines") or 50),
        )
    if action == "experiments-supervise":
        return start_experiment_supervisor(
            settings,
            spec_path=str(payload.get("spec_path") or (settings.repo_root / "configs" / "experiments" / "phase1_remote_baseline_sweep.yml")),
            poll_seconds=int(payload["poll_seconds"]) if payload.get("poll_seconds") is not None else None,
            detach=bool(payload.get("detach", True)),
        )
    if action == "experiments-pause":
        return pause_experiments(settings, experiment_id=str(payload.get("experiment_id") or "").strip())
    if action == "experiments-resume":
        return resume_experiments(settings, experiment_id=str(payload.get("experiment_id") or "").strip())
    if action == "experiments-stop":
        return stop_experiments(settings, experiment_id=str(payload.get("experiment_id") or "").strip())
    if action == "experiments-evaluate":
        return evaluate_experiments(settings, experiment_id=str(payload.get("experiment_id") or "").strip())
    if action == "experiments-backtest":
        return backtest_experiments(settings, experiment_id=str(payload.get("experiment_id") or "").strip())
    if action == "experiments-propose-next":
        return propose_experiment_family(settings, experiment_id=str(payload.get("experiment_id") or "").strip())
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
