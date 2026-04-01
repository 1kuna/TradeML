"""Streamlit operator dashboard for the TradeML data node."""

from __future__ import annotations

import argparse
import json
import os

from trademl.dashboard.controller import (
    _read_env_file,
    collect_dashboard_snapshot,
    force_release_lease,
    install_service,
    join_cluster,
    leave_cluster,
    persist_node_settings,
    rebuild_cluster_state,
    reset_worker,
    resolve_node_settings,
    restart_node,
    rotate_cluster_passphrase,
    start_node,
    stop_node,
    uninstall_worker,
    update_worker,
    update_cluster_secrets,
)


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
        env_values = _read_env_file(settings.env_path)
        cluster_passphrase = st.text_input(
            "Cluster Passphrase",
            type="password",
            value=env_values.get("TRADEML_CLUSTER_PASSPHRASE", os.getenv("TRADEML_CLUSTER_PASSPHRASE", "")),
        )

    snapshot = collect_dashboard_snapshot(settings)
    runtime = snapshot["runtime"]
    nas = snapshot["nas"]
    partition_summary = snapshot["partition_summary"]
    queue_counts = snapshot["queue_counts"]
    cluster = snapshot["cluster"]
    running = bool(runtime.get("running"))

    controls = st.columns([1, 1, 1, 2])
    if controls[0].button("Start Node", type="primary", use_container_width=True):
        start_node(settings, passphrase=cluster_passphrase or None)
        st.rerun()
    if controls[1].button("Stop Node", use_container_width=True):
        stop_node(settings)
        st.rerun()
    if controls[2].button("Restart Node", use_container_width=True):
        restart_node(settings, passphrase=cluster_passphrase or None)
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

    tabs = st.tabs(["Overview", "Fleet", "Cluster Config", "Data Inventory", "Logs"])

    with tabs[0]:
        overview_cols = st.columns(2)
        with overview_cols[0]:
            st.subheader("Runtime")
            st.json(runtime, expanded=False)
            service_cols = st.columns(3)
            if service_cols[0].button("Join Cluster", use_container_width=True):
                join_cluster(settings, passphrase=cluster_passphrase or None)
                st.rerun()
            if service_cols[1].button("Rebuild Local State", use_container_width=True):
                rebuild_cluster_state(settings, passphrase=cluster_passphrase or None)
                st.rerun()
            if service_cols[2].button("Leave Cluster", use_container_width=True):
                leave_cluster(settings)
                st.rerun()
        with overview_cols[1]:
            st.subheader("NAS Health")
            st.write(f"Share: `{nas['share']}`")
            st.write(f"Host: `{nas.get('host') or 'n/a'}`")
            st.write(f"Host reachable on 445: `{nas['host_reachable']}`")
            st.write(f"Mount path: `{nas['mount_path']}`")
            st.write(f"Mount writable: `{nas['mount_writable']}`")
            st.write(f"Expected stage sessions: `{snapshot['expected_stage_sessions'] or '-'}`")
            st.write(f"Stage progress ratio: `{_format_ratio(snapshot.get('stage_progress_ratio'))}`")
            st.write(f"Systemd: `{snapshot['systemd'].get('ActiveState', snapshot['systemd'].get('reason', 'unknown'))}`")

    with tabs[1]:
        worker_cols = st.columns(2)
        with worker_cols[0]:
            st.subheader("Workers")
            st.dataframe(cluster["workers"], use_container_width=True)
            st.caption(f"Active workers: {len(cluster['active_workers'])}")
        with worker_cols[1]:
            st.subheader("Leases")
            st.dataframe(cluster["leases"], use_container_width=True)
            lease_options = [lease["lease_id"] for lease in cluster["leases"] if lease]
            selected = st.selectbox("Force release lease", [""] + lease_options)
            if selected and st.button("Force Release", type="secondary"):
                force_release_lease(settings, selected)
                st.rerun()
        lower = st.columns(2)
        with lower[0]:
            st.subheader("Shard Map")
            st.dataframe(cluster["shards"], use_container_width=True)
        with lower[1]:
            st.subheader("Recent Events")
            st.dataframe(cluster["recent_events"], use_container_width=True)

    with tabs[2]:
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
        secret_cols = st.columns(2)
        with secret_cols[0]:
            st.subheader("Shared Secret Updates")
            with st.form("secret_update"):
                secret_key = st.text_input("Secret Key")
                secret_value = st.text_input("Secret Value")
                secret_passphrase = st.text_input("Cluster Passphrase", type="password", value=os.getenv("TRADEML_CLUSTER_PASSPHRASE", ""))
                if st.form_submit_button("Update Secret") and secret_key:
                    update_cluster_secrets(settings, passphrase=secret_passphrase, updates={secret_key: secret_value})
                    st.success(f"Updated {secret_key}")
                    st.rerun()
        with secret_cols[1]:
            st.subheader("Passphrase Rotation")
            with st.form("passphrase_rotate"):
                old_pass = st.text_input("Old Passphrase", type="password")
                new_pass = st.text_input("New Passphrase", type="password")
                if st.form_submit_button("Rotate Passphrase") and old_pass and new_pass:
                    rotate_cluster_passphrase(settings, old_passphrase=old_pass, new_passphrase=new_pass)
                    st.success("Cluster passphrase rotated")
                    st.rerun()
            if st.button("Install systemd Service", use_container_width=True):
                result = install_service(settings)
                st.success(f"Service written to {result['service_path']}")
                st.rerun()
        lifecycle_cols = st.columns(3)
        if lifecycle_cols[0].button("Update Worker", use_container_width=True):
            result = update_worker(settings)
            st.success(f"Updated worker at {result['wrapper_path']}")
            st.rerun()
        if lifecycle_cols[1].button("Reset Worker", use_container_width=True):
            result = reset_worker(settings, passphrase=cluster_passphrase or None)
            st.success(f"Reset worker workspace {result['workspace_root']}")
            st.rerun()
        if lifecycle_cols[2].button("Uninstall Worker", use_container_width=True):
            result = uninstall_worker(settings)
            st.success(f"Removed local worker artifacts: {len(result['removed_paths'])}")

    with tabs[3]:
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
        st.subheader("Last Success")
        st.json(cluster["last_success"], expanded=False)
        st.subheader("Manifest")
        st.json(cluster["manifest"], expanded=False)

    with tabs[4]:
        st.subheader("Node Log")
        st.text_area("Recent log lines", value=snapshot["log_tail"], height=500)
        if snapshot["journal_tail"]:
            st.subheader("systemd Journal")
            st.text_area("Recent journal lines", value=snapshot["journal_tail"], height=300)
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
