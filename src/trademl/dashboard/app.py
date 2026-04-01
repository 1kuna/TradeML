"""Streamlit operator dashboard for the TradeML data node."""

from __future__ import annotations

import argparse
import json

from trademl.dashboard.controller import collect_dashboard_snapshot, persist_node_settings, resolve_node_settings, restart_node, start_node, stop_node


def main() -> None:
    """Render the operator dashboard."""
    parser = argparse.ArgumentParser(description="TradeML dashboard")
    parser.add_argument("--workspace-root", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--env-file", default=None)
    args = parser.parse_args()

    import streamlit as st

    st.set_page_config(page_title="TradeML Dashboard", page_icon="T", layout="wide")
    settings = resolve_node_settings(
        workspace_root=args.workspace_root,
        config_path=args.config,
        env_path=args.env_file,
    )

    st.title("TradeML Node Dashboard")
    st.caption("Start, stop, monitor, and reconfigure the Raspberry Pi data node from one place.")
    refresh_seconds = st.sidebar.selectbox("Auto refresh", [0, 5, 15, 30], format_func=lambda value: "Manual" if value == 0 else f"Every {value}s")
    if refresh_seconds:
        st.markdown(f"<meta http-equiv='refresh' content='{refresh_seconds}'>", unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Workspace")
        st.code(str(settings.workspace_root))
        st.caption(f"Config: {settings.config_path}")
        st.caption(f"Env: {settings.env_path}")
        st.caption(f"Local state: {settings.local_state}")

    snapshot = collect_dashboard_snapshot(settings)
    runtime = snapshot["runtime"]
    nas = snapshot["nas"]
    partition_summary = snapshot["partition_summary"]
    queue_counts = snapshot["queue_counts"]
    running = bool(runtime.get("running"))

    controls = st.columns([1, 1, 1, 2])
    if controls[0].button("Start Node", type="primary", use_container_width=True):
        start_node(settings)
        st.rerun()
    if controls[1].button("Stop Node", use_container_width=True):
        stop_node(settings)
        st.rerun()
    if controls[2].button("Restart Node", use_container_width=True):
        restart_node(settings)
        st.rerun()
    controls[3].metric("Node Status", "Running" if running else "Stopped", f"PID {runtime.get('pid', '-')}")

    top_metrics = st.columns(6)
    top_metrics[0].metric("Pending", queue_counts.get("PENDING", 0))
    top_metrics[1].metric("Failed", queue_counts.get("FAILED", 0))
    top_metrics[2].metric("Green", partition_summary["counts"].get("GREEN", 0))
    top_metrics[3].metric("Amber", partition_summary["counts"].get("AMBER", 0))
    top_metrics[4].metric("Red", partition_summary["counts"].get("RED", 0))
    top_metrics[5].metric("Coverage", _format_ratio(partition_summary.get("coverage_green")))

    lower_metrics = st.columns(6)
    lower_metrics[0].metric("Raw Dates", snapshot["raw_partitions"])
    lower_metrics[1].metric("Curated Dates", snapshot["curated_partitions"])
    lower_metrics[2].metric("Stage Symbols", snapshot["stage_symbol_count"])
    lower_metrics[3].metric("Stage Years", snapshot["stage_years"] or "-")
    lower_metrics[4].metric("Latest Raw", snapshot["latest_raw_date"] or "-")
    lower_metrics[5].metric("Latest Curated", snapshot["latest_curated_date"] or "-")

    tabs = st.tabs(["Overview", "NAS & Schedule", "Data Inventory", "Logs"])

    with tabs[0]:
        overview_cols = st.columns(2)
        with overview_cols[0]:
            st.subheader("Runtime")
            st.json(runtime, expanded=False)
        with overview_cols[1]:
            st.subheader("NAS Health")
            st.write(f"Share: `{nas['share']}`")
            st.write(f"Host: `{nas.get('host') or 'n/a'}`")
            st.write(f"Host reachable on 445: `{nas['host_reachable']}`")
            st.write(f"Mount path: `{nas['mount_path']}`")
            st.write(f"Mount writable: `{nas['mount_writable']}`")
            st.write(f"Expected stage sessions: `{snapshot['expected_stage_sessions'] or '-'}`")
            st.write(f"Stage progress ratio: `{_format_ratio(snapshot.get('stage_progress_ratio'))}`")

    with tabs[1]:
        st.subheader("NAS and Schedule Settings")
        with st.form("nas_settings"):
            nas_share = st.text_input("NAS Share", value=settings.nas_share, help="Example: //192.168.1.20/trademl")
            nas_mount = st.text_input("NAS Mount Path", value=str(settings.nas_mount))
            collection_time = st.text_input("Collection Time ET", value=settings.collection_time_et)
            maintenance_hour = st.number_input(
                "Maintenance Hour Local",
                min_value=0,
                max_value=23,
                step=1,
                value=settings.maintenance_hour_local,
            )
            fstab_path = st.text_input("Persist fstab Path", value="/etc/fstab")
            if st.form_submit_button("Save Settings"):
                result = persist_node_settings(
                    settings,
                    nas_share=nas_share,
                    nas_mount=nas_mount,
                    collection_time_et=collection_time,
                    maintenance_hour_local=int(maintenance_hour),
                    fstab_path=fstab_path,
                )
                st.success(f"Saved settings to {result['config_path']} and {result['env_path']}.")
                st.caption(f"fstab entry persisted to {result['fstab_path']}")
                st.rerun()

    with tabs[2]:
        inventory_cols = st.columns(2)
        with inventory_cols[0]:
            st.subheader("Reference Files")
            st.write(snapshot["reference_files"] or ["No reference parquet files found"])
        with inventory_cols[1]:
            st.subheader("Partition Summary")
            st.json(partition_summary, expanded=False)
        st.subheader("Queue Summary")
        st.json(queue_counts, expanded=False)
        st.subheader("Resolved Settings")
        st.json(snapshot["settings"], expanded=False)

    with tabs[3]:
        st.subheader("Node Log")
        st.text_area("Recent log lines", value=snapshot["log_tail"], height=500)
        st.download_button(
            "Download runtime snapshot",
            data=json.dumps(snapshot, indent=2, default=str),
            file_name="trademl-dashboard-snapshot.json",
            mime="application/json",
        )


def _format_ratio(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.1%}"


if __name__ == "__main__":
    main()
