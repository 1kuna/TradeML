"""
Rich dashboard UI for the unified Pi data-node.

Implements:
- Real-time status display with Rich Live
- Queue statistics and progress
- Vendor budget meters
- Loop status table
- Log tail panel

See updated_node_spec.md §8 for UI semantics.
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from queue import Queue, Empty, Full
from typing import Optional

from loguru import logger

try:
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
    from rich.table import Table
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    logger.warning("Rich not available, using basic console output")


@dataclass
class LoopStatus:
    """Status of a single loop."""
    name: str
    running: bool = False
    last_run: Optional[datetime] = None
    current_task: Optional[str] = None
    tasks_completed: int = 0
    tasks_failed: int = 0
    error: Optional[str] = None


@dataclass
class VendorStatus:
    """Status of a vendor's budget."""
    name: str
    spent_today: int = 0
    daily_cap: int = 0
    tokens_rpm: float = 0.0
    hard_rpm: int = 0


@dataclass
class NodeStatus:
    """Thread-safe shared state for the node dashboard."""

    # Node info
    node_id: str = ""
    env: str = "local"
    data_root: str = ""
    started_at: Optional[datetime] = None

    # Queue stats
    queue_pending: int = 0
    queue_leased: int = 0
    queue_done: int = 0
    queue_failed: int = 0

    # Stage info
    current_stage: int = 0
    universe_size: int = 0
    green_coverage: float = 0.0

    # Loop statuses
    loops: dict[str, LoopStatus] = field(default_factory=dict)

    # Vendor statuses
    vendors: dict[str, VendorStatus] = field(default_factory=dict)

    # Log lines (circular buffer) - only touched by dashboard thread
    log_lines: deque = field(default_factory=lambda: deque(maxlen=20))

    # Thread-safe log queue for worker threads to submit logs
    # Dashboard thread drains this into log_lines
    _log_queue: Queue = field(default_factory=lambda: Queue(maxsize=100))

    # Thread lock for other state
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def update_queue_stats(self, stats: dict) -> None:
        """Update queue statistics."""
        with self._lock:
            self.queue_pending = stats.get("by_status", {}).get("PENDING", 0)
            self.queue_leased = stats.get("by_status", {}).get("LEASED", 0)
            self.queue_done = stats.get("by_status", {}).get("DONE", 0)
            self.queue_failed = stats.get("by_status", {}).get("FAILED", 0)

    def update_loop(self, name: str, **kwargs) -> None:
        """Update a loop's status."""
        with self._lock:
            if name not in self.loops:
                self.loops[name] = LoopStatus(name=name)
            for key, value in kwargs.items():
                if hasattr(self.loops[name], key):
                    setattr(self.loops[name], key, value)

    def update_vendor(self, name: str, **kwargs) -> None:
        """Update a vendor's status."""
        with self._lock:
            if name not in self.vendors:
                self.vendors[name] = VendorStatus(name=name)
            for key, value in kwargs.items():
                if hasattr(self.vendors[name], key):
                    setattr(self.vendors[name], key, value)

    def add_log_line(self, line: str) -> None:
        """Add a log line directly to buffer (thread-safe)."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        with self._lock:
            self.log_lines.append(f"{timestamp} {line}")

    def queue_log_line(self, line: str) -> None:
        """Thread-safe: queue a log line for dashboard consumption.

        Worker threads should call this instead of add_log_line().
        The dashboard thread will drain the queue into log_lines.
        """
        try:
            self._log_queue.put_nowait(line)
        except Full:
            pass  # Drop if queue full - prevents blocking workers

    def drain_log_queue(self) -> None:
        """Drain log queue into log_lines (thread-safe).

        Call this from the dashboard render loop to safely transfer
        queued log lines into the deque that Rich displays.
        """
        while True:
            try:
                line = self._log_queue.get_nowait()
                timestamp = datetime.now().strftime("%H:%M:%S")
                with self._lock:
                    self.log_lines.append(f"{timestamp} {line}")
            except Empty:
                break

    def get_snapshot(self) -> dict:
        """Get a thread-safe snapshot of the status."""
        with self._lock:
            return {
                "node_id": self.node_id,
                "env": self.env,
                "data_root": self.data_root,
                "started_at": self.started_at,
                "queue_pending": self.queue_pending,
                "queue_leased": self.queue_leased,
                "queue_done": self.queue_done,
                "queue_failed": self.queue_failed,
                "current_stage": self.current_stage,
                "universe_size": self.universe_size,
                "green_coverage": self.green_coverage,
                "loops": {k: v.__dict__.copy() for k, v in self.loops.items()},
                "vendors": {k: v.__dict__.copy() for k, v in self.vendors.items()},
                "log_lines": list(self.log_lines),
            }


def make_header_panel(status: NodeStatus) -> Panel:
    """Create the header panel with node info."""
    snapshot = status.get_snapshot()

    lines = [
        f"[bold cyan]NODE[/] {snapshot['node_id']}  "
        f"[bold cyan]ENV[/] {snapshot['env']}  "
        f"[bold cyan]STAGE[/] {snapshot['current_stage']}",
        f"[bold cyan]ROOT[/] {snapshot['data_root']}",
        f"[bold cyan]QUEUE[/] "
        f"[green]{snapshot['queue_done']}[/] done  "
        f"[yellow]{snapshot['queue_pending']}[/] pending  "
        f"[blue]{snapshot['queue_leased']}[/] active  "
        f"[red]{snapshot['queue_failed']}[/] failed",
        f"[bold cyan]COVERAGE[/] {snapshot['green_coverage']:.1%} GREEN  "
        f"[bold cyan]UNIVERSE[/] {snapshot['universe_size']} symbols",
    ]

    uptime = ""
    if snapshot["started_at"]:
        elapsed = datetime.now() - snapshot["started_at"]
        hours, remainder = divmod(int(elapsed.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        uptime = f"  [dim]uptime: {hours}h {minutes}m[/]"

    return Panel(
        "\n".join(lines),
        title=f"[bold]Pi Data-Node{uptime}[/]",
        border_style="cyan",
    )


def make_loops_table(status: NodeStatus) -> Table:
    """Create the loops status table."""
    table = Table(title="Loops", box=None, expand=True)
    table.add_column("Loop", style="cyan", width=12)
    table.add_column("Status", width=10)
    table.add_column("Last Run", width=20)
    table.add_column("Current", width=30)
    table.add_column("Done/Failed", justify="right", width=12)

    snapshot = status.get_snapshot()

    for name, loop in snapshot["loops"].items():
        if loop["running"]:
            status_text = Text("● Running", style="green")
        else:
            status_text = Text("○ Stopped", style="dim")

        last_run = ""
        if loop["last_run"]:
            if isinstance(loop["last_run"], datetime):
                last_run = loop["last_run"].strftime("%H:%M:%S")
            else:
                last_run = str(loop["last_run"])[:19]

        current = loop["current_task"] or ""
        if len(current) > 28:
            current = current[:25] + "..."

        stats = f"{loop['tasks_completed']}/{loop['tasks_failed']}"

        table.add_row(name, status_text, last_run, current, stats)

    return table


def make_vendors_panel(status: NodeStatus) -> Panel:
    """Create the vendor budgets panel."""
    snapshot = status.get_snapshot()

    if not snapshot["vendors"]:
        return Panel("[dim]No vendor data[/]", title="Vendor Budgets", border_style="blue")

    lines = []
    for name, vendor in sorted(snapshot["vendors"].items()):
        if vendor["daily_cap"] > 0:
            pct = vendor["spent_today"] / vendor["daily_cap"]
            bar_width = 20
            filled = int(pct * bar_width)

            if pct < 0.7:
                color = "green"
            elif pct < 0.9:
                color = "yellow"
            else:
                color = "red"

            bar = f"[{color}]{'█' * filled}[/][dim]{'░' * (bar_width - filled)}[/]"
            lines.append(
                f"[bold]{name:8}[/] {bar} "
                f"{vendor['spent_today']:5}/{vendor['daily_cap']:5} ({pct:.0%})"
            )
        else:
            lines.append(f"[bold]{name:8}[/] [dim]no budget configured[/]")

    return Panel("\n".join(lines), title="Vendor Budgets", border_style="blue")


def make_logs_panel(status: NodeStatus) -> Panel:
    """Create the log tail panel."""
    snapshot = status.get_snapshot()
    log_lines = snapshot["log_lines"]

    if not log_lines:
        content = "[dim]No log entries yet[/]"
    else:
        # Color code log lines
        colored_lines = []
        for line in log_lines:
            if "ERROR" in line or "error" in line.lower():
                colored_lines.append(f"[red]{line}[/]")
            elif "WARNING" in line or "warn" in line.lower():
                colored_lines.append(f"[yellow]{line}[/]")
            elif "SUCCESS" in line or "complete" in line.lower():
                colored_lines.append(f"[green]{line}[/]")
            else:
                colored_lines.append(f"[dim]{line}[/]")
        content = "\n".join(colored_lines)

    return Panel(content, title="Recent Logs", border_style="dim")


class Dashboard:
    """Rich dashboard for the data node."""

    def __init__(self, status: NodeStatus):
        """
        Initialize the dashboard.

        Args:
            status: Shared NodeStatus object
        """
        self.status = status
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._console = Console() if RICH_AVAILABLE else None

    def _make_layout(self) -> Layout:
        """Create the dashboard layout."""
        layout = Layout()

        layout.split(
            Layout(name="header", size=7),
            Layout(name="body"),
            Layout(name="logs", size=12),
        )

        layout["body"].split_row(
            Layout(name="loops", ratio=2),
            Layout(name="vendors", ratio=1),
        )

        return layout

    def _update_layout(self, layout: Layout) -> None:
        """Update the layout with current status."""
        layout["header"].update(make_header_panel(self.status))
        layout["loops"].update(Panel(make_loops_table(self.status), border_style="green"))
        layout["vendors"].update(make_vendors_panel(self.status))
        layout["logs"].update(make_logs_panel(self.status))

    def start(self, threaded: bool = True) -> None:
        """Start the dashboard."""
        if not RICH_AVAILABLE:
            logger.warning("Rich not available, dashboard disabled")
            return

        self._running = True

        if threaded:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        else:
            self._run()

    def stop(self) -> None:
        """Stop the dashboard."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None

    def _run(self) -> None:
        """Run the dashboard update loop."""
        layout = self._make_layout()

        with Live(layout, console=self._console, refresh_per_second=2, screen=True) as live:
            while self._running:
                try:
                    # Drain log queue in dashboard thread (thread-safe)
                    self.status.drain_log_queue()
                    self._update_layout(layout)
                    time.sleep(0.5)
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    # Use direct add since we're in dashboard thread
                    self.status.add_log_line(f"Dashboard error: {e}")
                    time.sleep(1)


class LogHandler:
    """Loguru handler that sends logs to NodeStatus.

    Uses queue_log_line() for thread-safe log submission.
    Worker threads can safely log without blocking or corrupting Rich display.
    """

    def __init__(self, status: NodeStatus):
        self.status = status

    def write(self, message):
        """Write a log message (thread-safe via queue)."""
        # Strip ANSI codes and newlines
        text = message.strip()
        if text:
            # Extract just the message part
            parts = text.split(" | ")
            if len(parts) >= 3:
                text = " | ".join(parts[-2:])
            # Use thread-safe queue instead of direct add
            self.status.queue_log_line(text[:100])


def setup_log_handler(status: NodeStatus) -> int:
    """Set up a loguru handler that feeds the dashboard."""
    handler = LogHandler(status)
    handler_id = logger.add(
        handler.write,
        format="{time:HH:mm:ss} | {level} | {message}",
        level="INFO",
    )
    return handler_id


def print_simple_status(status: NodeStatus) -> None:
    """Print a simple text status (when Rich is not available)."""
    snapshot = status.get_snapshot()

    print("\n=== Pi Data-Node Status ===")
    print(f"Node ID: {snapshot['node_id']}")
    print(f"Environment: {snapshot['env']}")
    print(f"Stage: {snapshot['current_stage']}")
    print(f"Universe: {snapshot['universe_size']} symbols")
    print(f"Coverage: {snapshot['green_coverage']:.1%}")

    print("\n--- Queue ---")
    print(f"Pending: {snapshot['queue_pending']}")
    print(f"Active: {snapshot['queue_leased']}")
    print(f"Done: {snapshot['queue_done']}")
    print(f"Failed: {snapshot['queue_failed']}")

    print("\n--- Loops ---")
    for name, loop in snapshot["loops"].items():
        status_str = "Running" if loop["running"] else "Stopped"
        print(f"{name}: {status_str} (done: {loop['tasks_completed']})")

    print("\n--- Vendors ---")
    for name, vendor in snapshot["vendors"].items():
        pct = vendor["spent_today"] / vendor["daily_cap"] * 100 if vendor["daily_cap"] > 0 else 0
        print(f"{name}: {vendor['spent_today']}/{vendor['daily_cap']} ({pct:.0f}%)")

    print()
