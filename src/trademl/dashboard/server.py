"""HTTP dashboard server with client-side live updates."""

from __future__ import annotations

import argparse
import json
import logging
import os
import threading
import time
import webbrowser
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

from trademl.dashboard.controller import (
    advance_collection_stage,
    backtest_experiments,
    bootstrap_canonical_ledger,
    collect_dashboard_game_snapshot,
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
    pause_research,
    persist_node_settings,
    propose_experiment_family,
    rebuild_cluster_state,
    replan_coverage,
    research_review_packet,
    research_status,
    resolve_node_settings,
    resume_experiments,
    resume_research,
    restart_node,
    rotate_cluster_passphrase,
    lane_health,
    evaluate_experiments,
    repair_canonical_backlog,
    repair_status,
    run_vendor_audit,
    start_experiment_supervisor,
    start_node,
    start_research_supervisor,
    start_training_run,
    stop_node,
    stop_experiments,
    stop_research,
    stop_training_run,
    steer_research,
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

OPERATOR_HTML_PAGE = """<!doctype html>
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

    <div class="toolbar" style="margin-top:18px">
      <div>
        <h2 style="margin-bottom:4px">Operator Views</h2>
        <div class="small muted">Switch between live status, vendor budgets, setup controls, and logs.</div>
      </div>
      <div class="form-row">
        <button type="button" data-section="status" class="active">Status</button>
        <button type="button" data-section="budgets" class="ghost">Budgets</button>
        <button type="button" data-section="setup" class="ghost">Setup</button>
        <button type="button" data-section="logs" class="ghost">Logs</button>
      </div>
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
            <input id="input-program-spec" placeholder="Program spec path" value="configs/research/perpetual_macmini.yml">
            <input id="input-program-id" placeholder="Program ID" value="perpetual-macmini">
            <div class="form-row">
              <button id="start-research-program">Start Autopilot</button>
              <button id="pause-research-program" class="secondary">Pause</button>
              <button id="resume-research-program" class="secondary">Resume</button>
              <button id="stop-research-program" class="ghost">Stop</button>
            </div>
            <div class="form-row">
              <button id="review-research-program" class="secondary">Write Review Packet</button>
              <button id="pivot-research-program" class="secondary">Force Pivot</button>
              <select id="input-research-breadth">
                <option value="low">Low breadth</option>
                <option value="normal" selected>Normal breadth</option>
                <option value="high">High breadth</option>
              </select>
            </div>
            <input id="input-prefer-architecture-family" placeholder="Prefer architecture family (comma-separated)">
            <input id="input-avoid-data-family" placeholder="Avoid data family (comma-separated)">
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
    const sections = ['status', 'budgets', 'setup', 'logs'];
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
      const program = health.research_program_summary || {};
      const bestCandidate = (program.best_candidate_summary || {});
      const incumbent = program.incumbent || {};
      const paperOutputs = program.latest_paper_outputs || program.paper_outputs || {};
      const driftAlerts = program.latest_drift_alerts || program.drift_alerts || [];
      const infraBlocker = program.last_infra_preflight || program.infra_blocker || {};
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
        ['Program', program.program_id || '-'],
        ['Best candidate', bestCandidate.best_candidate || experiment.best_candidate || 'No winner yet'],
        ['Research incumbent', incumbent.run_id || '-'],
        ['Primary score', formatDecimal(bestCandidate.best_primary_score ?? experiment.best_primary_score)],
        ['Backtest net', formatDecimal(bestCandidate.best_backtest_net_return ?? experiment.best_backtest_net_return)],
        ['Decision', bestCandidate.best_decision || experiment.best_decision || 'No GO yet'],
        ['Reason', bestCandidate.best_decision_reason || experiment.best_decision_reason || '-'],
      ]);
      const queueActive = Boolean(program.status || supervisor.status || experiment.experiment_id);
      renderKeyValue('research-board', [
        ['Queue status', program.status || (queueActive ? 'Active' : 'Idle')],
        ['Program phase', program.current_phase ? `Phase ${program.current_phase}` : '-'],
        ['Active family', program.current_experiment_id || experiment.experiment_id || '-'],
        ['Runs', `${formatNumber((program.budgets || {}).runs_completed ?? experiment.run_count ?? 0)} total`],
        ['Running', formatNumber((experiment.counts || {}).RUNNING ?? 0)],
        ['Completed', formatNumber((experiment.counts || {}).COMPLETED ?? 0)],
        ['Shortlisted', formatNumber(experiment.shortlist_count ?? 0)],
        ['Predictive survivors', formatNumber((experiment.evaluation_counts || {}).SURVIVES_PREDICTIVE ?? 0)],
        ['Paper orders', paperOutputs.paper_orders_path || '-'],
        ['Drift alerts', formatNumber(driftAlerts.length || 0)],
        ['Infra blocker', infraBlocker.reason || program.wait_reason || '-'],
        ['Top rejection', ((experiment.top_gate_failures || [])[0] || []).join(': ') || '-'],
        ['Budget left', formatNumber(((program.budgets || {}).max_total_runs ?? 0) - ((program.budgets || {}).runs_completed ?? 0))],
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
      } else if (program.last_transition && program.last_transition.reason) {
        nextHeadline = 'Autopilot is steering the frontier';
        nextReason = program.last_transition.reason;
      } else if (proposal.recommended_experiment_id) {
        nextHeadline = 'Launch the next bounded family';
        nextReason = proposal.recommended_experiment_id;
      }
      renderKeyValue('next-board', [
        ['Move', nextHeadline],
        ['Why', nextReason],
        ['Proposal', program.last_transition?.next_spec?.experiment_id || proposal.recommended_experiment_id || '-'],
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
      if (activeSection === 'status' || activeSection === 'budgets') {
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

    document.querySelectorAll('[data-section]').forEach((button) => {
      button.addEventListener('click', () => {
        setSection(button.dataset.section);
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

    function commaSeparatedValues(id) {
      return document.getElementById(id).value
        .split(',')
        .map((value) => value.trim())
        .filter(Boolean);
    }

    function programPayload() {
      const payload = {};
      const programId = document.getElementById('input-program-id').value.trim();
      const programPath = document.getElementById('input-program-spec').value.trim();
      const pollSeconds = document.getElementById('input-experiment-poll-seconds').value.trim();
      const explorationBreadth = document.getElementById('input-research-breadth').value;
      const preferArchitecture = commaSeparatedValues('input-prefer-architecture-family');
      const avoidDataFamily = commaSeparatedValues('input-avoid-data-family');
      if (programId) payload.program_id = programId;
      if (programPath) payload.program_path = programPath;
      if (pollSeconds) payload.poll_seconds = Number(pollSeconds);
      if (explorationBreadth) payload.exploration_breadth = explorationBreadth;
      if (preferArchitecture.length) payload.prefer_architecture_families = preferArchitecture;
      if (avoidDataFamily.length) payload.avoid_data_families = avoidDataFamily;
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

    document.getElementById('start-research-program').addEventListener('click', async () => {
      try {
        setMessage('setup-message', 'Starting perpetual research program...', 'warn');
        const result = await fetchJson('/api/actions/research-start', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({...programPayload(), detach: true}),
        });
        document.getElementById('setup-json').textContent = JSON.stringify(result, null, 2);
        setMessage('setup-message', 'Perpetual research program started', 'good');
        await refreshStatus();
      } catch (error) {
        setMessage('setup-message', `Autopilot start failed: ${error.message}`, 'bad');
      }
    });

    document.getElementById('pause-research-program').addEventListener('click', async () => {
      try {
        setMessage('setup-message', 'Pausing perpetual research...', 'warn');
        const result = await fetchJson('/api/actions/research-pause', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(programPayload()),
        });
        document.getElementById('setup-json').textContent = JSON.stringify(result, null, 2);
        setMessage('setup-message', 'Perpetual research paused', 'good');
        await refreshStatus();
      } catch (error) {
        setMessage('setup-message', `Autopilot pause failed: ${error.message}`, 'bad');
      }
    });

    document.getElementById('resume-research-program').addEventListener('click', async () => {
      try {
        setMessage('setup-message', 'Resuming perpetual research...', 'warn');
        const result = await fetchJson('/api/actions/research-resume', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(programPayload()),
        });
        document.getElementById('setup-json').textContent = JSON.stringify(result, null, 2);
        setMessage('setup-message', 'Perpetual research resumed', 'good');
        await refreshStatus();
      } catch (error) {
        setMessage('setup-message', `Autopilot resume failed: ${error.message}`, 'bad');
      }
    });

    document.getElementById('stop-research-program').addEventListener('click', async () => {
      try {
        setMessage('setup-message', 'Stopping perpetual research...', 'warn');
        const result = await fetchJson('/api/actions/research-stop', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(programPayload()),
        });
        document.getElementById('setup-json').textContent = JSON.stringify(result, null, 2);
        setMessage('setup-message', 'Perpetual research stop requested', 'good');
        await refreshStatus();
      } catch (error) {
        setMessage('setup-message', `Autopilot stop failed: ${error.message}`, 'bad');
      }
    });

    document.getElementById('review-research-program').addEventListener('click', async () => {
      try {
        setMessage('setup-message', 'Writing research review packet...', 'warn');
        const result = await fetchJson('/api/actions/research-review-packet', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(programPayload()),
        });
        document.getElementById('setup-json').textContent = JSON.stringify(result, null, 2);
        setMessage('setup-message', 'Research review packet written', 'good');
        await refreshStatus();
      } catch (error) {
        setMessage('setup-message', `Review packet failed: ${error.message}`, 'bad');
      }
    });

    document.getElementById('pivot-research-program').addEventListener('click', async () => {
      try {
        setMessage('setup-message', 'Applying research steering...', 'warn');
        const result = await fetchJson('/api/actions/research-steer', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({...programPayload(), force_pivot: true}),
        });
        document.getElementById('setup-json').textContent = JSON.stringify(result, null, 2);
        setMessage('setup-message', 'Research steering saved', 'good');
        await refreshStatus();
      } catch (error) {
        setMessage('setup-message', `Research steering failed: ${error.message}`, 'bad');
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


HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TradeML HQ</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #060b16;
      --bg-grid: rgba(93, 200, 255, 0.06);
      --panel: rgba(12, 22, 38, 0.92);
      --panel-strong: rgba(18, 30, 48, 0.98);
      --panel-outline: rgba(93, 200, 255, 0.18);
      --line: rgba(118, 155, 200, 0.18);
      --line-soft: rgba(118, 155, 200, 0.08);
      --text: #edf4ff;
      --muted: #94a9c5;
      --dim: #5e7194;
      --collector: #20c997;
      --collector-soft: rgba(32, 201, 151, 0.18);
      --brain: #5dc8ff;
      --brain-soft: rgba(93, 200, 255, 0.18);
      --gold: #f2c94c;
      --gold-soft: rgba(242, 201, 76, 0.22);
      --bad: #ff6b6b;
      --bad-soft: rgba(255, 107, 107, 0.18);
      --warn: #f6ad55;
      --warn-soft: rgba(246, 173, 85, 0.18);
      --good: #68d391;
      --shadow: 0 30px 80px rgba(0, 0, 0, 0.42);
      --radius: 22px;
      --radius-sm: 14px;
      --display: "IBM Plex Sans", "Segoe UI", "Helvetica Neue", Arial, sans-serif;
      --mono: "SFMono-Regular", Menlo, Consolas, monospace;
      font-family: var(--display);
    }
    * { box-sizing: border-box; }
    html, body { min-height: 100%; }
    body {
      margin: 0;
      color: var(--text);
      background:
        radial-gradient(circle at 18% -10%, rgba(32, 201, 151, 0.16), transparent 36%),
        radial-gradient(circle at 82% -10%, rgba(93, 200, 255, 0.18), transparent 40%),
        radial-gradient(circle at 50% 120%, rgba(242, 201, 76, 0.10), transparent 44%),
        linear-gradient(180deg, #071017 0%, var(--bg) 48%, #04080e 100%);
      background-attachment: fixed;
      overflow-x: hidden;
    }
    body::before {
      content: '';
      position: fixed;
      inset: 0;
      background-image:
        linear-gradient(transparent 97%, var(--bg-grid) 98%),
        linear-gradient(90deg, transparent 97%, var(--bg-grid) 98%);
      background-size: 38px 38px;
      pointer-events: none;
      z-index: 0;
      opacity: 0.55;
    }
    .shell {
      position: relative;
      z-index: 1;
      width: min(1360px, 100%);
      margin: 0 auto;
      padding: 28px 24px 48px;
      display: grid;
      gap: 22px;
    }
    .hero {
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(320px, 0.9fr);
      gap: 22px;
      align-items: stretch;
      padding: 26px 28px;
      border-radius: var(--radius);
      background: linear-gradient(135deg, rgba(18, 32, 52, 0.96) 0%, rgba(10, 18, 30, 0.96) 60%, rgba(6, 12, 22, 0.96) 100%);
      border: 1px solid var(--panel-outline);
      box-shadow: var(--shadow);
      position: relative;
      overflow: hidden;
    }
    .hero::after {
      content: '';
      position: absolute;
      inset: 0;
      background: radial-gradient(circle at 90% -40%, rgba(93, 200, 255, 0.25), transparent 55%);
      pointer-events: none;
    }
    .hero-copy {
      display: grid;
      gap: 14px;
      align-content: center;
      position: relative;
      z-index: 1;
    }
    .eyebrow {
      display: inline-flex;
      align-items: center;
      gap: 10px;
      width: fit-content;
      padding: 6px 14px;
      border-radius: 999px;
      border: 1px solid var(--panel-outline);
      background: rgba(93, 200, 255, 0.08);
      color: #bfe6ff;
      font-size: 12px;
      letter-spacing: 0.28em;
      text-transform: uppercase;
    }
    .eyebrow .dot {
      width: 8px; height: 8px;
      border-radius: 50%;
      background: var(--brain);
      box-shadow: 0 0 12px var(--brain);
      animation: pulse 1.8s ease-in-out infinite;
    }
    .title {
      font-family: var(--display);
      font-weight: 800;
      font-size: clamp(2.3rem, 5vw, 3.8rem);
      letter-spacing: -0.035em;
      line-height: 1;
      background: linear-gradient(90deg, #ffffff, #bfe6ff 50%, #b7ffe6);
      -webkit-background-clip: text;
      background-clip: text;
      color: transparent;
      text-shadow: 0 0 24px rgba(93, 200, 255, 0.25);
    }
    .subtitle {
      font-size: 15px;
      color: var(--muted);
      max-width: 54ch;
      line-height: 1.55;
      margin: 0;
    }
    .hero-meta {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }
    .chip {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 7px 12px;
      border-radius: 999px;
      background: rgba(18, 32, 52, 0.85);
      border: 1px solid var(--line);
      color: var(--muted);
      font-size: 12px;
      letter-spacing: 0.12em;
      text-transform: uppercase;
    }
    .chip strong { color: var(--text); font-weight: 600; }
    .hero-xp {
      position: relative;
      z-index: 1;
      display: grid;
      gap: 16px;
      padding: 22px;
      border-radius: var(--radius-sm);
      border: 1px solid rgba(242, 201, 76, 0.28);
      background: linear-gradient(180deg, rgba(242, 201, 76, 0.08), rgba(20, 34, 52, 0.88) 70%);
      align-content: center;
    }
    .hero-xp .label { font-size: 12px; letter-spacing: 0.28em; color: #fde8a3; text-transform: uppercase; }
    .hero-xp .value {
      font-family: var(--mono);
      font-size: clamp(2.1rem, 4vw, 3rem);
      font-weight: 700;
      letter-spacing: -0.02em;
      color: var(--text);
    }
    .hero-xp .value .unit { color: var(--gold); font-size: 0.6em; padding-left: 6px; }
    .xp-bar {
      position: relative;
      height: 18px;
      border-radius: 999px;
      background: rgba(5, 10, 18, 0.85);
      border: 1px solid rgba(242, 201, 76, 0.22);
      overflow: hidden;
    }
    .xp-fill {
      position: absolute;
      inset: 0;
      width: 0%;
      border-radius: inherit;
      background: linear-gradient(90deg, #68d391 0%, #5dc8ff 45%, #f2c94c 100%);
      box-shadow: 0 0 24px rgba(242, 201, 76, 0.45);
      transition: width 700ms cubic-bezier(.2,.8,.2,1);
    }
    .xp-fill::after {
      content: '';
      position: absolute;
      inset: 0;
      background: linear-gradient(90deg, transparent 45%, rgba(255,255,255,0.45) 50%, transparent 55%);
      background-size: 260% 100%;
      animation: shimmer 2.8s linear infinite;
    }
    .hero-xp.is-ready .xp-fill {
      background: linear-gradient(90deg, #f2c94c, #ffefb0, #f2c94c);
      box-shadow: 0 0 30px rgba(242, 201, 76, 0.75);
    }
    .hero-xp .status {
      display: flex;
      justify-content: space-between;
      font-size: 12px;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--muted);
    }
    .hero-xp .status .freeze { color: #d7fff1; }
    .hero-xp.is-ready .label { color: #ffe58a; }
    .hero-xp.is-ready .value { color: var(--gold); text-shadow: 0 0 20px rgba(242, 201, 76, 0.5); }
    .hero-settings {
      position: absolute;
      top: 16px;
      right: 18px;
      z-index: 2;
    }
    .gear-btn {
      appearance: none;
      border: 1px solid var(--line);
      background: rgba(5, 10, 18, 0.7);
      color: var(--muted);
      width: 38px; height: 38px;
      border-radius: 12px;
      cursor: pointer;
      font-size: 18px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      transition: color 160ms, border-color 160ms, transform 160ms;
    }
    .gear-btn:hover { color: var(--text); border-color: var(--panel-outline); transform: rotate(30deg); }

    .cards {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 22px;
    }
    @media (max-width: 960px) {
      .cards { grid-template-columns: 1fr; }
      .hero { grid-template-columns: 1fr; }
    }

    .card {
      position: relative;
      padding: 26px;
      border-radius: var(--radius);
      background: var(--panel);
      border: 1px solid var(--line);
      box-shadow: var(--shadow);
      overflow: hidden;
      display: grid;
      gap: 18px;
    }
    .card--collector { border-color: rgba(32, 201, 151, 0.32); }
    .card--brain { border-color: rgba(93, 200, 255, 0.32); }
    .card::before {
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 5px;
      background: linear-gradient(90deg, transparent, var(--accent, var(--brain)), transparent);
      opacity: 0.7;
    }
    .card--collector { --accent: var(--collector); }
    .card--brain { --accent: var(--brain); }
    .card-head {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 14px;
    }
    .card-title {
      display: grid;
      gap: 6px;
    }
    .card-title .kicker {
      font-size: 11px;
      letter-spacing: 0.3em;
      color: var(--muted);
      text-transform: uppercase;
    }
    .card-title h2 {
      margin: 0;
      font-size: clamp(1.4rem, 2.4vw, 1.9rem);
      font-weight: 700;
      letter-spacing: -0.025em;
    }
    .status-pill {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 14px;
      border-radius: 999px;
      background: rgba(5, 10, 18, 0.6);
      border: 1px solid var(--line);
      font-size: 12px;
      letter-spacing: 0.22em;
      text-transform: uppercase;
      color: var(--muted);
      font-family: var(--mono);
    }
    .status-pill .dot {
      width: 9px; height: 9px; border-radius: 50%;
      background: var(--dim);
      box-shadow: 0 0 0 0 currentColor;
    }
    .status-pill.is-online { color: #b7ffe6; border-color: rgba(32, 201, 151, 0.45); background: var(--collector-soft); }
    .status-pill.is-online .dot { background: var(--collector); animation: pulse 1.4s ease-in-out infinite; box-shadow: 0 0 14px var(--collector); }
    .status-pill.is-offline { color: #ffb4b4; border-color: rgba(255, 107, 107, 0.45); background: var(--bad-soft); }
    .status-pill.is-offline .dot { background: var(--bad); }
    .status-pill.is-running { color: #bfe6ff; border-color: rgba(93, 200, 255, 0.45); background: var(--brain-soft); }
    .status-pill.is-running .dot { background: var(--brain); animation: pulse 1.2s ease-in-out infinite; box-shadow: 0 0 14px var(--brain); }
    .status-pill.is-cooling { color: #ffd9a3; border-color: rgba(246, 173, 85, 0.45); background: var(--warn-soft); }
    .status-pill.is-cooling .dot { background: var(--warn); animation: pulse 2s ease-in-out infinite; }
    .status-pill.is-failed { color: #ffb4b4; border-color: rgba(255, 107, 107, 0.55); background: var(--bad-soft); }
    .status-pill.is-failed .dot { background: var(--bad); }
    .status-pill.is-idle { color: var(--muted); }

    .collector-body {
      display: grid;
      grid-template-columns: 180px minmax(0, 1fr);
      gap: 22px;
      align-items: center;
    }
    .ring-wrap {
      position: relative;
      width: 180px;
      height: 180px;
    }
    .ring {
      width: 100%; height: 100%;
      transform: rotate(-90deg);
    }
    .ring .track {
      stroke: rgba(93, 200, 255, 0.08);
    }
    .ring .fill {
      stroke: var(--collector);
      stroke-linecap: round;
      filter: drop-shadow(0 0 8px rgba(32, 201, 151, 0.55));
      transition: stroke-dasharray 700ms cubic-bezier(.2,.8,.2,1);
    }
    .ring-center {
      position: absolute;
      inset: 0;
      display: grid;
      place-items: center;
      text-align: center;
    }
    .ring-center .big {
      font-family: var(--mono);
      font-size: 2.4rem;
      font-weight: 700;
      line-height: 1;
      color: #b7ffe6;
    }
    .ring-center .label {
      font-size: 10px;
      letter-spacing: 0.28em;
      text-transform: uppercase;
      color: var(--muted);
      margin-top: 6px;
    }

    .metric-stack { display: grid; gap: 12px; }
    .metric {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      align-items: baseline;
      gap: 12px;
      padding: 12px 14px;
      border-radius: var(--radius-sm);
      background: rgba(5, 10, 18, 0.55);
      border: 1px solid var(--line-soft);
    }
    .metric .key { font-size: 11px; letter-spacing: 0.24em; text-transform: uppercase; color: var(--muted); }
    .metric .val {
      font-family: var(--mono);
      font-size: 1.3rem;
      font-weight: 700;
      color: var(--text);
      white-space: nowrap;
    }
    .metric .val .unit { color: var(--muted); font-size: 0.7em; padding-left: 4px; font-weight: 500; }
    .metric.is-collector .val { color: #b7ffe6; }
    .metric.is-brain .val { color: #bfe6ff; }
    .metric.is-gold .val { color: var(--gold); }

    .spark-row {
      display: grid;
      gap: 8px;
    }
    .spark-row .label {
      font-size: 11px;
      letter-spacing: 0.22em;
      text-transform: uppercase;
      color: var(--muted);
    }
    .sparkline-bars {
      display: flex;
      align-items: flex-end;
      gap: 3px;
      height: 44px;
      padding: 6px 0;
    }
    .sparkline-bars .bar {
      flex: 1;
      min-height: 2px;
      border-radius: 3px;
      background: linear-gradient(180deg, var(--collector), rgba(32, 201, 151, 0.25));
      transition: height 300ms cubic-bezier(.2,.8,.2,1);
    }
    .sparkline-svg {
      width: 100%;
      height: 64px;
    }
    .sparkline-svg path {
      fill: none;
      stroke: var(--brain);
      stroke-width: 2.5;
      stroke-linejoin: round;
      stroke-linecap: round;
      filter: drop-shadow(0 0 6px rgba(93, 200, 255, 0.6));
    }
    .sparkline-svg .area { fill: rgba(93, 200, 255, 0.12); stroke: none; }

    .brain-body { display: grid; gap: 20px; }
    .brain-headline {
      display: grid;
      grid-template-columns: auto minmax(0, 1fr);
      gap: 18px;
      align-items: center;
    }
    .latest-ic {
      padding: 18px 22px;
      border-radius: var(--radius-sm);
      background: rgba(5, 10, 18, 0.55);
      border: 1px solid var(--line-soft);
      text-align: center;
      min-width: 150px;
    }
    .latest-ic .label { font-size: 10px; letter-spacing: 0.28em; text-transform: uppercase; color: var(--muted); }
    .latest-ic .value {
      font-family: var(--mono);
      font-size: 2.4rem;
      font-weight: 700;
      line-height: 1;
      margin-top: 6px;
      color: var(--text);
    }
    .latest-ic.is-good .value { color: #b7ffe6; text-shadow: 0 0 16px rgba(32, 201, 151, 0.35); }
    .latest-ic.is-warn .value { color: var(--warn); }
    .latest-ic.is-bad .value { color: var(--bad); }
    .trophy { display: grid; gap: 10px; align-content: center; }
    .trophy .big {
      font-family: var(--mono);
      font-size: 1.4rem;
      color: var(--gold);
    }
    .trophy .big .label {
      display: inline-block;
      font-size: 10px;
      letter-spacing: 0.28em;
      color: var(--muted);
      text-transform: uppercase;
      padding-right: 8px;
    }
    .trophy .streak { font-size: 13px; color: var(--muted); letter-spacing: 0.04em; }
    .trophy .streak strong { color: var(--good); font-size: 15px; padding-right: 4px; }

    .decision-row { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }
    .decision-row .label { font-size: 11px; letter-spacing: 0.22em; text-transform: uppercase; color: var(--muted); padding-right: 4px; }
    .decision-dot {
      width: 14px; height: 14px;
      border-radius: 50%;
      border: 2px solid transparent;
      background: transparent;
    }
    .decision-dot.go { background: var(--good); border-color: rgba(104, 211, 145, 0.45); box-shadow: 0 0 10px rgba(104, 211, 145, 0.55); }
    .decision-dot.no_go { background: transparent; border-color: var(--bad); }
    .decision-dot.unknown { background: rgba(118, 155, 200, 0.25); border-color: var(--line); }

    .missions {
      padding: 24px 26px;
      border-radius: var(--radius);
      background: var(--panel);
      border: 1px solid var(--line);
      box-shadow: var(--shadow);
      display: grid;
      gap: 18px;
    }
    .missions-head {
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 12px;
      flex-wrap: wrap;
    }
    .missions-head h2 {
      font-size: 1.05rem;
      letter-spacing: 0.22em;
      text-transform: uppercase;
      color: var(--muted);
      margin: 0;
      font-weight: 600;
    }
    .missions-head .meta { font-size: 12px; color: var(--dim); letter-spacing: 0.08em; }
    .missions-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 14px;
    }
    .mission {
      display: grid;
      gap: 8px;
      padding: 14px 16px;
      border-radius: var(--radius-sm);
      border: 1px solid var(--line);
      background: rgba(5, 10, 18, 0.5);
      position: relative;
      overflow: hidden;
    }
    .mission .head {
      display: flex;
      align-items: center;
      gap: 10px;
      font-weight: 600;
      font-size: 14px;
    }
    .mission .icon {
      width: 22px; height: 22px;
      border-radius: 6px;
      display: grid;
      place-items: center;
      font-family: var(--mono);
      font-size: 13px;
      font-weight: 700;
    }
    .mission.complete { border-color: rgba(104, 211, 145, 0.45); }
    .mission.complete .icon { background: rgba(104, 211, 145, 0.2); color: var(--good); }
    .mission.in_progress { border-color: rgba(93, 200, 255, 0.4); }
    .mission.in_progress .icon { background: rgba(93, 200, 255, 0.18); color: var(--brain); animation: spin 2.8s linear infinite; }
    .mission.locked { border-color: rgba(118, 155, 200, 0.18); opacity: 0.75; }
    .mission.locked .icon { background: rgba(255, 107, 107, 0.15); color: var(--bad); }
    .mission .bar {
      position: relative;
      height: 8px;
      border-radius: 999px;
      background: rgba(5, 10, 18, 0.85);
      overflow: hidden;
    }
    .mission .bar .fill {
      position: absolute;
      inset: 0;
      width: 0%;
      border-radius: inherit;
      background: linear-gradient(90deg, var(--collector), var(--brain));
      transition: width 700ms cubic-bezier(.2,.8,.2,1);
    }
    .mission.complete .bar .fill { background: linear-gradient(90deg, var(--good), var(--gold)); }
    .mission.locked .bar .fill { background: rgba(255, 107, 107, 0.4); }
    .mission .foot {
      display: flex;
      justify-content: space-between;
      font-family: var(--mono);
      font-size: 12px;
      color: var(--muted);
    }
    .mission .foot .eta { color: var(--dim); }

    .connection-footer {
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 11px;
      letter-spacing: 0.22em;
      text-transform: uppercase;
      color: var(--dim);
      padding: 0 8px;
    }
    .connection-footer .live {
      display: inline-flex;
      align-items: center;
      gap: 8px;
    }
    .connection-footer .live .dot {
      width: 8px; height: 8px;
      border-radius: 50%;
      background: var(--dim);
    }
    .connection-footer .live.ok .dot { background: var(--collector); box-shadow: 0 0 10px var(--collector); animation: pulse 1.6s ease-in-out infinite; }
    .connection-footer .live.bad .dot { background: var(--bad); }

    dialog.operator {
      max-width: min(1280px, 92vw);
      width: 92vw;
      height: 88vh;
      padding: 0;
      border: 1px solid var(--panel-outline);
      border-radius: var(--radius);
      background: var(--panel-strong);
      color: var(--text);
      box-shadow: var(--shadow);
    }
    dialog.operator::backdrop {
      background: rgba(3, 6, 14, 0.68);
      backdrop-filter: blur(6px);
    }
    dialog.operator .frame-head {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 14px 22px;
      border-bottom: 1px solid var(--line);
      background: rgba(10, 18, 30, 0.9);
    }
    dialog.operator .frame-head h3 {
      margin: 0;
      font-size: 13px;
      letter-spacing: 0.22em;
      text-transform: uppercase;
      color: var(--muted);
    }
    dialog.operator .frame-head button {
      appearance: none;
      border: 1px solid var(--line);
      background: rgba(5, 10, 18, 0.7);
      color: var(--muted);
      font-size: 18px;
      width: 32px; height: 32px;
      border-radius: 10px;
      cursor: pointer;
    }
    dialog.operator iframe {
      width: 100%;
      height: calc(100% - 54px);
      border: 0;
      background: var(--bg);
    }

    @keyframes pulse {
      0%, 100% { opacity: 1; transform: scale(1); }
      50% { opacity: 0.55; transform: scale(0.85); }
    }
    @keyframes shimmer {
      0% { background-position: 100% 0; }
      100% { background-position: -100% 0; }
    }
    @keyframes spin {
      0% { transform: rotate(0deg); }
      100% { transform: rotate(360deg); }
    }
  </style>
</head>
<body>
  <main class="shell">
    <section class="hero">
      <div class="hero-settings">
        <button id="open-operator" class="gear-btn" aria-label="Open operator console">&#9881;</button>
      </div>
      <div class="hero-copy">
        <span class="eyebrow"><span class="dot"></span>Mission Control</span>
        <h1 class="title">TradeML HQ</h1>
        <p class="subtitle">The Raspberry Pi pulls data. The Mac Mini trains. You watch the gauges climb — nothing else to do unless the gears are calling.</p>
        <div class="hero-meta">
          <span class="chip"><span>Updated</span><strong id="updated-at">--</strong></span>
          <span class="chip"><span>Freeze</span><strong id="freeze-cutoff">--</strong></span>
        </div>
      </div>
      <div id="hero-xp" class="hero-xp">
        <div class="label">Phase 1 Gate</div>
        <div class="value" id="xp-value">--<span class="unit">%</span></div>
        <div class="xp-bar"><div class="xp-fill" id="xp-fill"></div></div>
        <div class="status">
          <span id="xp-status">Collecting data...</span>
          <span class="freeze" id="xp-freeze"></span>
        </div>
      </div>
    </section>

    <section class="cards">
      <article class="card card--collector">
        <div class="card-head">
          <div class="card-title">
            <span class="kicker">The Collector</span>
            <h2 id="collector-label">raspberry-pi</h2>
          </div>
          <div id="collector-status" class="status-pill is-offline"><span class="dot"></span><span>Offline</span></div>
        </div>
        <div class="collector-body">
          <div class="ring-wrap">
            <svg class="ring" viewBox="0 0 140 140">
              <circle class="track" cx="70" cy="70" r="60" stroke-width="12" fill="none"></circle>
              <circle id="ring-fill" class="fill" cx="70" cy="70" r="60" stroke-width="12" fill="none"
                stroke-dasharray="0 377" stroke-dashoffset="0"></circle>
            </svg>
            <div class="ring-center">
              <div>
                <div class="big" id="ring-pct">0%</div>
                <div class="label">Coverage</div>
              </div>
            </div>
          </div>
          <div class="metric-stack">
            <div class="metric is-collector">
              <span class="key">Rows / min</span>
              <span class="val" id="rows-per-min">0</span>
            </div>
            <div class="metric is-collector">
              <span class="key">Rows collected</span>
              <span class="val" id="rows-total">0</span>
            </div>
            <div class="metric">
              <span class="key">ETA to done</span>
              <span class="val" id="eta-display">--</span>
            </div>
          </div>
        </div>
        <div class="spark-row">
          <div class="label">Last minute pulse</div>
          <div class="sparkline-bars" id="collector-spark"></div>
        </div>
      </article>

      <article class="card card--brain">
        <div class="card-head">
          <div class="card-title">
            <span class="kicker">The Brain</span>
            <h2 id="brain-label">mac-mini</h2>
          </div>
          <div id="brain-status" class="status-pill is-idle"><span class="dot"></span><span>Idle</span></div>
        </div>
        <div class="brain-body">
          <div class="brain-headline">
            <div class="latest-ic is-warn" id="latest-ic">
              <div class="label">Latest Rank IC</div>
              <div class="value" id="latest-ic-value">0.000</div>
            </div>
            <div class="trophy">
              <div class="big"><span class="label">Best IC</span><span id="best-ic">0.000</span></div>
              <div class="streak"><strong id="streak-count">0</strong> GO streak &middot; <span id="total-runs">0</span> attempts</div>
            </div>
          </div>
          <div class="decision-row">
            <span class="label">Last runs</span>
            <div id="decision-dots" style="display:flex;gap:8px;"></div>
          </div>
          <div class="spark-row">
            <div class="label">Rank IC trend</div>
            <svg class="sparkline-svg" id="ic-spark" viewBox="0 0 300 64" preserveAspectRatio="none">
              <path class="area" d="M0,64 L300,64 Z"></path>
              <path class="line" d=""></path>
            </svg>
          </div>
        </div>
      </article>
    </section>

    <section class="missions">
      <div class="missions-head">
        <h2>Missions &mdash; Phase 1 datasets</h2>
        <div class="meta" id="mission-count">--</div>
      </div>
      <div class="missions-grid" id="missions-grid"></div>
    </section>

    <div class="connection-footer">
      <span id="worker-id">--</span>
      <span class="live" id="connection-badge"><span class="dot"></span><span>Connecting</span></span>
    </div>
  </main>

  <dialog id="operator-dialog" class="operator">
    <div class="frame-head">
      <h3>Operator Console</h3>
      <button id="close-operator" type="button" aria-label="Close operator console">&times;</button>
    </div>
    <iframe id="operator-frame" title="Operator Console" src="about:blank"></iframe>
  </dialog>

  <script>
    const sparkBuffer = [];
    const SPARK_MAX = 32;
    const prevNumbers = new Map();
    const ICON_MAP = { complete: '\\u2713', in_progress: '\\u27f3', locked: '\\u2717' };

    function setText(id, value) {
      const node = document.getElementById(id);
      if (node) node.textContent = value;
    }

    function formatNumber(value) {
      if (value === null || value === undefined || Number.isNaN(value)) return '--';
      const abs = Math.abs(value);
      if (abs >= 1_000_000) return (value / 1_000_000).toFixed(2) + 'M';
      if (abs >= 1_000) return (value / 1_000).toFixed(1) + 'k';
      return value.toFixed(0);
    }

    function formatEta(minutes) {
      if (minutes === null || minutes === undefined || !Number.isFinite(minutes) || minutes <= 0) return '--';
      if (minutes < 1) return '<1m';
      if (minutes < 60) return Math.round(minutes) + 'm';
      const hours = Math.floor(minutes / 60);
      const remaining = Math.round(minutes % 60);
      return hours + 'h ' + remaining + 'm';
    }

    function formatIc(value) {
      if (value === null || value === undefined || Number.isNaN(value)) return '0.000';
      return value.toFixed(3);
    }

    function tweenCounter(id, target, formatter) {
      const node = document.getElementById(id);
      if (!node) return;
      const prev = prevNumbers.get(id);
      const start = typeof prev === 'number' ? prev : (typeof target === 'number' ? target : 0);
      const end = typeof target === 'number' ? target : 0;
      prevNumbers.set(id, end);
      const duration = 450;
      const t0 = performance.now();
      const step = (now) => {
        const p = Math.min(1, (now - t0) / duration);
        const eased = 1 - Math.pow(1 - p, 3);
        const value = start + (end - start) * eased;
        node.textContent = formatter(value);
        if (p < 1) requestAnimationFrame(step);
      };
      requestAnimationFrame(step);
    }

    function renderRing(percent) {
      const fill = document.getElementById('ring-fill');
      const clamped = Math.max(0, Math.min(100, percent || 0));
      const circumference = 2 * Math.PI * 60;
      const filled = (clamped / 100) * circumference;
      if (fill) fill.setAttribute('stroke-dasharray', `${filled} ${circumference - filled}`);
      setText('ring-pct', Math.round(clamped) + '%');
    }

    function setStatusPill(id, cls, label) {
      const node = document.getElementById(id);
      if (!node) return;
      node.className = 'status-pill ' + cls;
      const span = node.querySelector('span:last-child');
      if (span) span.textContent = label;
    }

    function renderCollector(node) {
      if (!node) return;
      setText('collector-label', node.label || 'raspberry-pi');
      setText('worker-id', node.label || '--');
      const running = !!node.running;
      if (running) setStatusPill('collector-status', 'is-online', 'Online');
      else setStatusPill('collector-status', 'is-offline', 'Offline');
      renderRing(node.coverage_percent || 0);
      tweenCounter('rows-per-min', Math.round(node.rows_per_min || 0), formatNumber);
      tweenCounter('rows-total', Math.round(node.rows_total || 0), formatNumber);
      setText('eta-display', formatEta(node.eta_minutes));

      sparkBuffer.push(Math.max(0, Number(node.rows_per_min) || 0));
      while (sparkBuffer.length > SPARK_MAX) sparkBuffer.shift();
      const spark = document.getElementById('collector-spark');
      if (spark) {
        const max = Math.max(1, ...sparkBuffer);
        const html = [];
        for (const value of sparkBuffer) {
          const h = Math.max(3, (value / max) * 100);
          html.push('<div class="bar" style="height:' + h + '%"></div>');
        }
        spark.innerHTML = html.join('');
      }
    }

    function renderBrain(training) {
      if (!training) return;
      setText('brain-label', training.host || 'mac-mini');
      const state = (training.state || 'idle').toLowerCase();
      if (state === 'running' || state === 'starting') setStatusPill('brain-status', 'is-running', state.toUpperCase());
      else if (state === 'cooling_down') setStatusPill('brain-status', 'is-cooling', 'Cooling Down');
      else if (state === 'failed' || state === 'stopped' || state === 'unknown') setStatusPill('brain-status', 'is-failed', state.toUpperCase());
      else setStatusPill('brain-status', 'is-idle', 'Idle');

      const latest = Number(training.latest_rank_ic || 0);
      setText('latest-ic-value', formatIc(latest));
      const latestEl = document.getElementById('latest-ic');
      if (latestEl) {
        latestEl.classList.remove('is-good', 'is-warn', 'is-bad');
        if (latest >= 0.02) latestEl.classList.add('is-good');
        else if (latest > 0) latestEl.classList.add('is-warn');
        else latestEl.classList.add('is-bad');
      }

      setText('best-ic', formatIc(training.best_rank_ic || 0));
      setText('streak-count', String(training.streak_go || 0));
      setText('total-runs', String(training.total_runs || 0));

      const dotsWrap = document.getElementById('decision-dots');
      if (dotsWrap) {
        const decisions = Array.isArray(training.last_decisions) ? training.last_decisions : [];
        const html = [];
        for (let i = 0; i < 8; i++) {
          const decision = decisions[i];
          let cls = 'decision-dot unknown';
          if (decision) {
            const name = String(decision.decision || '').toUpperCase();
            if (name === 'GO') cls = 'decision-dot go';
            else if (name === 'NO_GO' || name === 'NO-GO') cls = 'decision-dot no_go';
          }
          const title = decision ? (decision.run_ts + ' · ' + (decision.decision || 'unknown') + ' · IC ' + formatIc(decision.mean_rank_ic)) : 'no run';
          html.push('<span class="' + cls + '" title="' + title + '"></span>');
        }
        dotsWrap.innerHTML = html.join('');
      }

      const sparkline = Array.isArray(training.sparkline) ? training.sparkline : [];
      const path = document.querySelector('#ic-spark path.line');
      const area = document.querySelector('#ic-spark path.area');
      if (path && sparkline.length >= 2) {
        const w = 300;
        const h = 64;
        const pad = 4;
        const values = sparkline.map((v) => Number(v) || 0);
        const min = Math.min(0, ...values);
        const max = Math.max(0.01, ...values, Math.abs(min));
        const range = Math.max(0.01, max - min);
        const step = (w - pad * 2) / (values.length - 1);
        const coords = values.map((v, i) => {
          const x = pad + i * step;
          const y = h - pad - ((v - min) / range) * (h - pad * 2);
          return [x, y];
        });
        const linePath = 'M' + coords.map((c) => c[0].toFixed(1) + ',' + c[1].toFixed(1)).join(' L');
        path.setAttribute('d', linePath);
        if (area) {
          const areaPath = linePath + ' L' + coords[coords.length - 1][0].toFixed(1) + ',' + h + ' L' + coords[0][0].toFixed(1) + ',' + h + ' Z';
          area.setAttribute('d', areaPath);
        }
      } else if (path) {
        path.setAttribute('d', '');
        if (area) area.setAttribute('d', '');
      }
    }

    function renderMissions(missions) {
      const grid = document.getElementById('missions-grid');
      if (!grid) return;
      const items = Array.isArray(missions) ? missions : [];
      const html = items.map((mission) => {
        const status = (mission.status || 'locked').toLowerCase();
        const icon = ICON_MAP[status] || '?';
        const percent = Math.max(0, Math.min(100, Number(mission.percent || 0)));
        const eta = formatEta(mission.eta_minutes);
        return [
          '<div class="mission ' + status + '">',
          '  <div class="head"><span class="icon">' + icon + '</span>' + (mission.label || mission.key || 'Mission') + '</div>',
          '  <div class="bar"><div class="fill" style="width:' + percent.toFixed(1) + '%"></div></div>',
          '  <div class="foot"><span>' + percent.toFixed(0) + '%</span><span class="eta">' + eta + '</span></div>',
          '</div>',
        ].join('');
      }).join('');
      grid.innerHTML = html;
      const complete = items.filter((m) => (m.status || '') === 'complete').length;
      setText('mission-count', complete + ' / ' + items.length + ' complete');
    }

    function renderHero(gate, updatedAt, meta) {
      const percent = Math.max(0, Math.min(100, Number(gate && gate.percent) || 0));
      setText('xp-value', percent.toFixed(1) + '%');
      const fill = document.getElementById('xp-fill');
      if (fill) fill.style.width = percent + '%';
      const xp = document.getElementById('hero-xp');
      if (xp) xp.classList.toggle('is-ready', !!(gate && gate.ready));
      const xpStatus = document.getElementById('xp-status');
      if (xpStatus) {
        if (gate && gate.ready) xpStatus.textContent = 'Gate unlocked';
        else if (gate && Array.isArray(gate.blockers) && gate.blockers.length) xpStatus.textContent = gate.blockers[0].replace(/_/g, ' ');
        else xpStatus.textContent = 'Collecting data...';
      }
      setText('xp-freeze', gate && gate.freeze_cutoff ? 'freeze ' + gate.freeze_cutoff : '');
      setText('freeze-cutoff', gate && gate.freeze_cutoff ? gate.freeze_cutoff : '--');
      if (updatedAt) {
        try {
          const dt = new Date(updatedAt);
          const freshness = meta && meta.stale ? ' · stale' : '';
          setText('updated-at', dt.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }) + freshness);
        } catch (err) { /* ignore */ }
      }
    }

    function applySnapshot(payload) {
      if (!payload) return;
      renderCollector(payload.node);
      renderBrain(payload.training);
      renderMissions(payload.missions);
      renderHero(payload.phase1_gate, payload.updated_at, payload.meta || {});
    }

    async function refreshOnce() {
      try {
        const response = await fetch('/api/game', { cache: 'no-store' });
        if (!response.ok) throw new Error('snapshot failed: ' + response.status);
        applySnapshot(await response.json());
      } catch (err) {
        console.error('refreshOnce failed', err);
      }
    }

    function connectStream() {
      const badge = document.getElementById('connection-badge');
      if (badge) badge.className = 'live';
      const source = new EventSource('/api/game/stream');
      source.onmessage = (event) => {
        if (badge) { badge.className = 'live ok'; badge.querySelector('span:last-child').textContent = 'Live'; }
        try { applySnapshot(JSON.parse(event.data)); } catch (err) { console.error('stream parse error', err); }
      };
      source.onerror = () => {
        if (badge) { badge.className = 'live bad'; badge.querySelector('span:last-child').textContent = 'Reconnecting'; }
        source.close();
        setTimeout(connectStream, 1500);
      };
    }

    const dialog = document.getElementById('operator-dialog');
    const frame = document.getElementById('operator-frame');
    document.getElementById('open-operator').addEventListener('click', () => {
      if (frame && frame.src === 'about:blank') frame.src = '/operator';
      if (dialog && typeof dialog.showModal === 'function') dialog.showModal();
    });
    document.getElementById('close-operator').addEventListener('click', () => {
      if (dialog) dialog.close();
    });
    if (dialog) {
      dialog.addEventListener('click', (event) => {
        if (event.target === dialog) dialog.close();
      });
    }

    refreshOnce();
    connectStream();
  </script>
</body>
</html>
"""


class DashboardSnapshotManager:
    """Build and serve dashboard snapshots outside the HTTP request path."""

    snapshot_version = 1
    stream_interval_seconds = 2.0

    def __init__(self, *, settings: NodeSettings) -> None:
        self.settings = settings
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._records: dict[str, dict[str, Any]] = {}
        self._consecutive_failures: dict[str, int] = {"game": 0, "live": 0, "status": 0}
        self._successful_builds: dict[str, bool] = {"game": False, "live": False, "status": False}
        self._refresh_intervals: dict[str, float] = {"game": 10.0, "live": 5.0, "status": 15.0}
        self._eager_channels = {"game", "live"}
        self._active_channels = set(self._eager_channels)
        self._builders: dict[str, Callable[[NodeSettings], dict[str, Any]]] = {
            "game": collect_dashboard_game_snapshot,
            "live": collect_dashboard_live_snapshot,
            "status": collect_dashboard_status_snapshot,
        }
        self._next_refresh_at: dict[str, float] = {}

        for channel in self._builders:
            self._load_snapshot_from_disk(channel)
        for channel in self._eager_channels:
            self._build_initial_snapshot(channel)
            self._next_refresh_at[channel] = time.monotonic() + self._refresh_intervals[channel]
        self._thread = threading.Thread(target=self._run_loop, name="dashboard-snapshots", daemon=True)
        self._thread.start()

    def close(self) -> None:
        """Stop the background refresh loop."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def get_latest(self, channel: str) -> dict[str, Any] | None:
        """Return the latest built snapshot envelope for one channel."""
        with self._lock:
            record = self._records.get(channel)
            if record is None:
                return None
            return json.loads(json.dumps(record, default=str))

    def get_latest_or_not_ready(self, channel: str) -> tuple[HTTPStatus, dict[str, Any]]:
        """Return the latest payload or a compact not-ready response."""
        self._ensure_channel_active(channel)
        payload = self.get_latest(channel)
        if payload is None:
            return (
                HTTPStatus.SERVICE_UNAVAILABLE,
                {"ok": False, "error": "snapshot_not_ready", "channel": channel},
            )
        return HTTPStatus.OK, payload

    def refresh_once(self, channel: str) -> None:
        """Refresh a single channel immediately."""
        self._ensure_channel_active(channel)
        self._refresh_channel(channel)
        self._next_refresh_at[channel] = time.monotonic() + self._refresh_intervals[channel]

    def health_summary(self) -> dict[str, Any]:
        """Return compact snapshot freshness information for operator status."""
        with self._lock:
            now = datetime.now(tz=UTC)
            summary: dict[str, Any] = {}
            for channel in self._builders:
                record = self._records.get(channel)
                meta = record.get("meta", {}) if isinstance(record, dict) else {}
                built_at = meta.get("built_at")
                age_seconds: int | None = None
                if built_at:
                    try:
                        age_seconds = max(0, int((now - datetime.fromisoformat(str(built_at))).total_seconds()))
                    except ValueError:
                        age_seconds = None
                summary[channel] = {
                    "built_at": built_at,
                    "stale": bool(meta.get("stale", False)),
                    "build_ms": meta.get("build_ms"),
                    "consecutive_failures": self._consecutive_failures.get(channel, 0),
                    "age_seconds": age_seconds,
                    "source": meta.get("source"),
                    "error": meta.get("error"),
                }
            return summary

    def _run_loop(self) -> None:
        while not self._stop_event.wait(0.5):
            now = time.monotonic()
            for channel, interval in self._refresh_intervals.items():
                if channel not in self._active_channels:
                    continue
                next_refresh_at = self._next_refresh_at.get(channel, now + interval)
                if now >= next_refresh_at:
                    self._refresh_channel(channel)
                    self._next_refresh_at[channel] = time.monotonic() + interval

    def _ensure_channel_active(self, channel: str) -> None:
        if channel in self._active_channels:
            return
        self._active_channels.add(channel)
        if self.get_latest(channel) is None:
            self._build_initial_snapshot(channel)
        self._next_refresh_at[channel] = time.monotonic() + self._refresh_intervals[channel]

    def _build_initial_snapshot(self, channel: str) -> None:
        try:
            self._build_fresh_snapshot(channel)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("dashboard_snapshot_initial_build_failed channel=%s", channel)
            self._mark_existing_snapshot_stale(channel, error=str(exc), source="disk_cache")

    def _refresh_channel(self, channel: str) -> None:
        try:
            self._build_fresh_snapshot(channel)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("dashboard_snapshot_refresh_failed channel=%s", channel)
            self._mark_existing_snapshot_stale(channel, error=str(exc), source="stale_last_good")

    def _build_fresh_snapshot(self, channel: str) -> None:
        builder = self._builders[channel]
        build_started_at = datetime.now(tz=UTC)
        started = time.monotonic()
        payload = builder(self.settings)
        build_ms = round((time.monotonic() - started) * 1000.0, 2)
        response = self._wrap_snapshot(
            payload,
            built_at=datetime.now(tz=UTC),
            build_started_at=build_started_at,
            build_ms=build_ms,
            stale=False,
            source="fresh",
            error=None,
        )
        with self._lock:
            self._records[channel] = response
            self._successful_builds[channel] = True
            self._consecutive_failures[channel] = 0
        self._persist_snapshot(channel, response)

    def _mark_existing_snapshot_stale(self, channel: str, *, error: str, source: str) -> None:
        with self._lock:
            current = self._records.get(channel)
            self._consecutive_failures[channel] = self._consecutive_failures.get(channel, 0) + 1
            if current is None:
                return
            response = json.loads(json.dumps(current, default=str))
            meta = response.setdefault("meta", {})
            meta["stale"] = True
            meta["source"] = source
            meta["error"] = error
            response["meta"] = meta
            self._records[channel] = response

    def _wrap_snapshot(
        self,
        payload: dict[str, Any],
        *,
        built_at: datetime,
        build_started_at: datetime,
        build_ms: float,
        stale: bool,
        source: str,
        error: str | None,
    ) -> dict[str, Any]:
        response = dict(payload)
        response["meta"] = {
            "built_at": built_at.isoformat(),
            "build_started_at": build_started_at.isoformat(),
            "build_ms": build_ms,
            "stale": stale,
            "source": source,
            "error": error,
            "version": self.snapshot_version,
        }
        return response

    def _snapshot_path(self, channel: str) -> Path:
        return self.settings.local_state / f"dashboard_{channel}_snapshot.json"

    def _persist_snapshot(self, channel: str, payload: dict[str, Any]) -> None:
        path = self._snapshot_path(channel)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(path.suffix + ".tmp")
        temp_path.write_text(json.dumps(payload, default=str), encoding="utf-8")
        os.replace(temp_path, path)

    def _load_snapshot_from_disk(self, channel: str) -> None:
        path = self._snapshot_path(channel)
        if not path.exists():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            LOGGER.warning("dashboard_snapshot_disk_load_failed channel=%s path=%s error=%s", channel, path, exc)
            return
        if not isinstance(payload, dict):
            return
        meta = payload.setdefault("meta", {})
        meta.setdefault("built_at", None)
        meta.setdefault("build_started_at", None)
        meta.setdefault("build_ms", None)
        meta["stale"] = bool(meta.get("stale", False))
        meta["source"] = "disk_cache"
        meta["error"] = meta.get("error")
        meta["version"] = self.snapshot_version
        with self._lock:
            self._records[channel] = payload


class DashboardHTTPServer(ThreadingHTTPServer):
    """HTTP server that holds resolved TradeML dashboard settings."""

    daemon_threads = True

    def __init__(self, server_address: tuple[str, int], *, settings: NodeSettings) -> None:
        super().__init__(server_address, DashboardRequestHandler)
        self.settings = settings
        self.snapshot_manager = DashboardSnapshotManager(settings=settings)

    def server_close(self) -> None:
        self.snapshot_manager.close()
        super().server_close()


class DashboardRequestHandler(BaseHTTPRequestHandler):
    """Serve the operator dashboard HTML and JSON action endpoints."""

    server: DashboardHTTPServer
    stream_interval_seconds = DashboardSnapshotManager.stream_interval_seconds

    def do_HEAD(self) -> None:  # noqa: N802
        html_paths = {"/", "/operator"}
        json_paths = {
            "/api/live",
            "/api/game",
            "/api/status",
            "/api/health",
            "/api/setup",
            "/api/logs",
            "/_stcore/health",
            "/_stcore/host-config",
        }
        if self.path in html_paths or self.path in json_paths:
            content_type = "text/html; charset=utf-8" if self.path in html_paths else "application/json; charset=utf-8"
            self._write_headers(status=HTTPStatus.OK, content_type=content_type)
            return
        self.send_error(HTTPStatus.NOT_FOUND, "not found")

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/":
            self._write_html(HTML_PAGE)
            return
        if self.path == "/operator":
            self._write_html(OPERATOR_HTML_PAGE)
            return
        if self.path == "/_stcore/health":
            self._write_json({"ok": True})
            return
        if self.path == "/_stcore/host-config":
            self._write_json({"useExternalAuthToken": False, "enableCustomParentMessages": False})
            return
        if self.path == "/api/game":
            status, payload = self.server.snapshot_manager.get_latest_or_not_ready("game")
            self._write_json(payload, status=status)
            return
        if self.path == "/api/game/stream":
            self._serve_game_stream()
            return
        if self.path == "/api/live":
            status, payload = self.server.snapshot_manager.get_latest_or_not_ready("live")
            self._write_json(payload, status=status)
            return
        if self.path == "/api/status":
            status, payload = self.server.snapshot_manager.get_latest_or_not_ready("status")
            payload = dict(payload)
            payload["snapshot_health"] = self.server.snapshot_manager.health_summary()
            self._write_json(payload, status=status)
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
        self.wfile.flush()

    def _write_html(self, html: str) -> None:
        body = html.encode("utf-8")
        self._write_headers(status=HTTPStatus.OK, content_type="text/html; charset=utf-8", extra_headers={"Content-Length": str(len(body))})
        self.wfile.write(body)
        self.wfile.flush()

    def _serve_live_stream(self) -> None:
        self._write_headers(
            status=HTTPStatus.OK,
            content_type="text/event-stream; charset=utf-8",
            extra_headers={"Connection": "keep-alive", "X-Accel-Buffering": "no"},
        )
        while True:
            try:
                payload = json.dumps(
                    self.server.snapshot_manager.get_latest("live")
                    or {"ok": False, "error": "snapshot_not_ready", "channel": "live"},
                    default=str,
                )
                self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                self.wfile.flush()
                time.sleep(self.stream_interval_seconds)
            except (BrokenPipeError, ConnectionResetError):
                return

    def _serve_game_stream(self) -> None:
        self._write_headers(
            status=HTTPStatus.OK,
            content_type="text/event-stream; charset=utf-8",
            extra_headers={"Connection": "keep-alive", "X-Accel-Buffering": "no"},
        )
        while True:
            try:
                payload = json.dumps(
                    self.server.snapshot_manager.get_latest("game")
                    or {"ok": False, "error": "snapshot_not_ready", "channel": "game"},
                    default=str,
                )
                self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                self.wfile.flush()
                time.sleep(self.stream_interval_seconds)
            except (BrokenPipeError, ConnectionResetError):
                return


def dispatch_dashboard_action(settings: NodeSettings, action: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Run a dashboard action endpoint against controller helpers."""
    handler = _DASHBOARD_ACTIONS.get(action)
    if handler is not None:
        return handler(settings, payload)
    raise ValueError(f"unsupported action: {action}")


def _cluster_secret_update(settings: NodeSettings, payload: dict[str, Any]) -> dict[str, Any]:
    key = str(payload.get("key") or "").strip()
    if not key:
        raise ValueError("secret key is required")
    passphrase = _optional_str(payload.get("passphrase"))
    if not passphrase:
        raise ValueError("cluster passphrase is required")
    return update_cluster_secrets(settings, passphrase=passphrase, updates={key: str(payload.get("value") or "")})


def _cluster_passphrase_rotation(settings: NodeSettings, payload: dict[str, Any]) -> dict[str, Any]:
    old_passphrase = _optional_str(payload.get("old_passphrase"))
    new_passphrase = _optional_str(payload.get("new_passphrase"))
    if not old_passphrase or not new_passphrase:
        raise ValueError("old and new passphrases are required")
    return rotate_cluster_passphrase(settings, old_passphrase=old_passphrase, new_passphrase=new_passphrase)


def _force_release_lease(settings: NodeSettings, payload: dict[str, Any]) -> dict[str, Any]:
    lease_id = str(payload.get("lease_id") or "").strip()
    if not lease_id:
        raise ValueError("lease_id is required")
    return {"released": force_release_lease(settings, lease_id), "lease_id": lease_id}


def _list_payload(payload: dict[str, Any], key: str) -> list[str]:
    return [str(value).strip() for value in list(payload.get(key) or []) if str(value).strip()]


DashboardAction = Callable[[NodeSettings, dict[str, Any]], dict[str, Any]]


_DASHBOARD_ACTIONS: dict[str, DashboardAction] = {
    "start-node": lambda settings, payload: start_node(settings, passphrase=_optional_str(payload.get("passphrase"))),
    "stop-node": lambda settings, _payload: stop_node(settings),
    "restart-node": lambda settings, payload: restart_node(settings, passphrase=_optional_str(payload.get("passphrase"))),
    "join-cluster": lambda settings, payload: join_cluster(settings, passphrase=_optional_str(payload.get("passphrase"))),
    "rebuild-state": lambda settings, payload: rebuild_cluster_state(settings, passphrase=_optional_str(payload.get("passphrase"))),
    "leave-cluster": lambda settings, _payload: leave_cluster(settings),
    "install-service": lambda settings, payload: install_service(settings, service_path=_optional_str(payload.get("service_path"))),
    "update-worker": lambda settings, _payload: update_worker(settings),
    "reset-worker": lambda settings, payload: reset_worker(settings, passphrase=_optional_str(payload.get("passphrase"))),
    "uninstall-worker": lambda settings, _payload: uninstall_worker(settings),
    "run-vendor-audit": lambda settings, _payload: run_vendor_audit(settings),
    "replan-coverage": lambda settings, _payload: replan_coverage(settings),
    "bootstrap-ledger": lambda settings, _payload: bootstrap_canonical_ledger(settings),
    "repair-canonical": lambda settings, payload: repair_canonical_backlog(
        settings,
        trading_date=_optional_str(payload.get("trading_date")),
        start_date=_optional_str(payload.get("start_date")),
        end_date=_optional_str(payload.get("end_date")),
        symbol=_optional_str(payload.get("symbol")),
        verify_only=bool(payload.get("verify_only")),
    ),
    "verify-recent": lambda settings, payload: verify_recent_canonical_dates(
        settings,
        days=int(payload.get("days") or 7),
        dataset=str(payload.get("dataset") or "equities_eod"),
        verify_only=bool(payload.get("verify_only")),
    ),
    "repair-status": lambda settings, _payload: repair_status(settings),
    "lane-health": lambda settings, payload: lane_health(settings, dataset=str(payload.get("dataset") or "equities_eod")),
    "train-preflight": lambda settings, payload: training_preflight_status(
        settings,
        phase=int(payload.get("phase") or 1),
        target=_optional_str(payload.get("target")),
    ),
    "train-start": lambda settings, payload: start_training_run(
        settings,
        phase=int(payload.get("phase") or 1),
        report_date=_optional_str(payload.get("report_date")),
        target=_optional_str(payload.get("target")),
    ),
    "train-stop": lambda settings, payload: stop_training_run(
        settings,
        phase=int(payload.get("phase") or 1),
        target=_optional_str(payload.get("target")),
    ),
    "train-status": lambda settings, payload: training_runtime_status(
        settings,
        phase=int(payload.get("phase") or 1),
        target=_optional_str(payload.get("target")),
    ),
    "train-logs": lambda settings, payload: training_runtime_logs(
        settings,
        phase=int(payload.get("phase") or 1),
        target=_optional_str(payload.get("target")),
        tail_lines=int(payload.get("tail_lines") or 50),
    ),
    "experiments-supervise": lambda settings, payload: start_experiment_supervisor(
        settings,
        spec_path=str(payload.get("spec_path") or (settings.repo_root / "configs" / "experiments" / "phase1_remote_baseline_sweep.yml")),
        poll_seconds=int(payload["poll_seconds"]) if payload.get("poll_seconds") is not None else None,
        detach=bool(payload.get("detach", True)),
    ),
    "experiments-pause": lambda settings, payload: pause_experiments(settings, experiment_id=str(payload.get("experiment_id") or "").strip()),
    "experiments-resume": lambda settings, payload: resume_experiments(settings, experiment_id=str(payload.get("experiment_id") or "").strip()),
    "experiments-stop": lambda settings, payload: stop_experiments(settings, experiment_id=str(payload.get("experiment_id") or "").strip()),
    "experiments-evaluate": lambda settings, payload: evaluate_experiments(settings, experiment_id=str(payload.get("experiment_id") or "").strip()),
    "experiments-backtest": lambda settings, payload: backtest_experiments(settings, experiment_id=str(payload.get("experiment_id") or "").strip()),
    "experiments-propose-next": lambda settings, payload: propose_experiment_family(settings, experiment_id=str(payload.get("experiment_id") or "").strip()),
    "research-start": lambda settings, payload: start_research_supervisor(
        settings,
        program_path=str(payload.get("program_path") or (settings.repo_root / "configs" / "research" / "perpetual_macmini.yml")),
        poll_seconds=int(payload["poll_seconds"]) if payload.get("poll_seconds") is not None else None,
        detach=bool(payload.get("detach", True)),
    ),
    "research-pause": lambda settings, payload: pause_research(settings, program_id=str(payload.get("program_id") or "").strip()),
    "research-resume": lambda settings, payload: resume_research(settings, program_id=str(payload.get("program_id") or "").strip()),
    "research-stop": lambda settings, payload: stop_research(settings, program_id=str(payload.get("program_id") or "").strip()),
    "research-status": lambda settings, payload: research_status(settings, program_id=str(payload.get("program_id") or "").strip()),
    "research-review-packet": lambda settings, payload: research_review_packet(settings, program_id=str(payload.get("program_id") or "").strip()),
    "research-steer": lambda settings, payload: steer_research(
        settings,
        program_id=str(payload.get("program_id") or "").strip(),
        prefer_architecture_families=_list_payload(payload, "prefer_architecture_families"),
        avoid_architecture_families=_list_payload(payload, "avoid_architecture_families"),
        prefer_data_families=_list_payload(payload, "prefer_data_families"),
        avoid_data_families=_list_payload(payload, "avoid_data_families"),
        freeze_phase=int(payload["freeze_phase"]) if payload.get("freeze_phase") is not None else None,
        force_pivot=bool(payload.get("force_pivot")) if payload.get("force_pivot") is not None else None,
        exploration_breadth=_optional_str(payload.get("exploration_breadth")),
    ),
    "save-settings": lambda settings, payload: persist_node_settings(
        settings,
        nas_share=str(payload.get("nas_share") or settings.nas_share),
        nas_mount=str(payload.get("nas_mount") or settings.nas_mount),
        collection_time_et=str(payload.get("collection_time_et") or settings.collection_time_et),
        maintenance_hour_local=int(payload.get("maintenance_hour_local") or settings.maintenance_hour_local),
        fstab_path=_optional_str(payload.get("fstab_path")),
    ),
    "update-secret": _cluster_secret_update,
    "rotate-passphrase": _cluster_passphrase_rotation,
    "force-release-lease": _force_release_lease,
    "advance-stage": lambda settings, payload: advance_collection_stage(
        settings,
        target_stage=int(payload.get("target_stage") or 0),
        symbol_count=int(payload.get("symbol_count")) if payload.get("symbol_count") is not None else None,
        years=int(payload.get("years")) if payload.get("years") is not None else None,
        passphrase=_optional_str(payload.get("passphrase")),
    ),
}


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
