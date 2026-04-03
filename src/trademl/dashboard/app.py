"""Streamlit operator dashboard for the TradeML data node."""

from __future__ import annotations

import argparse
import json
import os

from trademl.dashboard.controller import (
    _read_env_file,
    advance_collection_stage,
    collect_dashboard_snapshot,
    force_release_lease,
    install_service,
    join_cluster,
    leave_cluster,
    persist_node_settings,
    replan_coverage,
    rebuild_cluster_state,
    reset_worker,
    resolve_node_settings,
    restart_node,
    run_vendor_audit,
    stage_one_universe_snapshot,
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

    with st.sidebar:
        st.subheader("Controls")
        refresh_seconds = st.selectbox("Auto refresh", [0, 5, 15, 30], format_func=lambda value: "Manual" if value == 0 else f"Every {value}s")
        env_values = _read_env_file(settings.env_path)
        with st.expander("Cluster Passphrase", expanded=False):
            cluster_passphrase = st.text_input(
                "Cluster Passphrase",
                type="password",
                value=env_values.get("TRADEML_CLUSTER_PASSPHRASE", os.getenv("TRADEML_CLUSTER_PASSPHRASE", "")),
            )
        with st.expander("Workspace Paths", expanded=False):
            st.code(str(settings.workspace_root))
            st.caption(f"Config: {settings.config_path}")
            st.caption(f"Env: {settings.env_path}")
            st.caption(f"Local state: {settings.local_state}")
    if refresh_seconds:
        st.markdown(f"<meta http-equiv='refresh' content='{refresh_seconds}'>", unsafe_allow_html=True)

    snapshot = collect_dashboard_snapshot(settings)
    runtime = snapshot["runtime"]
    nas = snapshot["nas"]
    partition_summary = snapshot["partition_summary"]
    queue_counts = snapshot["queue_counts"]
    collection_status = snapshot["collection_status"]
    data_readiness = snapshot["data_readiness"]
    training_readiness = snapshot["training_readiness"]
    training_status = snapshot["training_status"]
    train_operational_status = snapshot["train_operational_status"]
    suggested_training_commands = snapshot["suggested_training_commands"]
    vendor_attempt_summary = snapshot["vendor_attempt_summary"]
    budget_summary = snapshot["budget_summary"]
    cluster = snapshot["cluster"]
    audit = snapshot["audit"]
    coverage_plan = snapshot["coverage_plan"]
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

    summary_cols = st.columns(4)
    summary_cols[0].metric("Collected", f"{collection_status['coverage_percent']:.1f}%")
    summary_cols[1].metric("Left To Collect", f"{collection_status['remaining_percent']:.1f}%")
    summary_cols[2].metric("Remaining Datapoints", f"{collection_status['remaining_datapoints']:,}")
    summary_cols[3].metric("Train Status", train_operational_status.replace("_", " ").title())

    st.progress(collection_status["coverage_ratio"], text=f"Collection coverage: {collection_status['coverage_percent']:.1f}%")

    tabs = st.tabs(["Status", "Budgets", "Setup", "Logs"])

    with tabs[0]:
        if data_readiness["state"] == "complete":
            st.success(data_readiness["headline"])
        elif data_readiness["state"] == "partial":
            st.warning(data_readiness["headline"])
        else:
            st.info(data_readiness["headline"])
        status_cols = st.columns(3)
        status_cols[0].metric("Queue Pending", collection_status["pending_tasks"])
        status_cols[1].metric("Queue Failed", collection_status["failed_tasks"])
        status_cols[2].metric("Latest Raw Date", snapshot["latest_raw_date"] or "-")

        training_cols = st.columns(2)
        with training_cols[0]:
            st.subheader("Can We Train Yet?")
            if training_readiness["phase1"]["ready"]:
                st.success("Phase 1 is ready for DGX/workstation training.")
            else:
                st.warning("Phase 1 is not ready yet.")
            blockers = training_readiness["phase1"]["blockers"]
            st.write(blockers or ["No current blockers"])
        with training_cols[1]:
            st.subheader("DGX Command")
            st.caption("Run this on the workstation or DGX after mounting the NAS.")
            st.code(suggested_training_commands["phase1"])
            if training_status["phase1"]:
                st.caption(f"Latest Phase 1 training status: {training_status['phase1'].get('status', 'unknown')}")
            if training_status["phase2"]:
                st.caption(f"Latest Phase 2 training status: {training_status['phase2'].get('status', 'unknown')}")

        with st.expander("Why training is blocked", expanded=not training_readiness["phase1"]["ready"]):
            st.json(training_readiness, expanded=False)

        with st.expander("Advanced collection detail", expanded=False):
            st.write(f"Expected datapoints: `{collection_status['expected_datapoints'] or '-'}`")
            st.write(f"Collected datapoints: `{collection_status['collected_datapoints']}`")
            st.write(f"Remaining datapoints: `{collection_status['remaining_datapoints']}`")
            st.write(f"EOD QC green ratio: `{_format_ratio(partition_summary.get('coverage_green'))}`")
            st.write(f"Reference files: `{snapshot['reference_file_count']}`")
            st.write(f"Macro series: `{snapshot['macro_series_count']}`")
            st.write(f"Price checks: `{snapshot['price_check_count']}`")

        with st.expander("Advanced cluster and audit detail", expanded=False):
            action_cols = st.columns(2)
            if action_cols[0].button("Run Vendor Audit", use_container_width=True):
                result = run_vendor_audit(settings)
                st.success(f"Audit completed at {result['checked_at']}")
                st.rerun()
            if action_cols[1].button("Replan Coverage", use_container_width=True):
                result = replan_coverage(settings)
                st.success(f"Planned {result['task_count']} auxiliary tasks")
                st.rerun()
            st.caption(f"Vendor attempts tracked: {sum(vendor_attempt_summary['counts'].values())}")
            st.caption(f"Audit failures: {len(audit.get('summary', {}).get('failures', []))}")
            st.dataframe(vendor_attempt_summary["by_vendor"], use_container_width=True)
            if vendor_attempt_summary["recent_failures"]:
                st.dataframe(vendor_attempt_summary["recent_failures"], use_container_width=True)

    with tabs[1]:
        st.subheader("API Budgets")
        budget_cols = st.columns(3)
        budget_cols[0].metric("Vendors Available", budget_summary["available_vendors"])
        budget_cols[1].metric("Day Capped", budget_summary["day_capped_vendors"])
        budget_cols[2].metric("Minute Capped", budget_summary["minute_capped_vendors"])
        if budget_summary["checked_at"]:
            st.caption(f"Budget snapshot: {budget_summary['checked_at']} UTC")
        if budget_summary["rows"]:
            st.dataframe(budget_summary["rows"], use_container_width=True, hide_index=True)
        else:
            st.info("No budget snapshot has been written yet. Start the worker and let it issue requests first.")

    with tabs[2]:
        cluster_actions = st.columns(3)
        if cluster_actions[0].button("Join Cluster", use_container_width=True):
            join_cluster(settings, passphrase=cluster_passphrase or None)
            st.rerun()
        if cluster_actions[1].button("Rebuild Local State", use_container_width=True):
            rebuild_cluster_state(settings, passphrase=cluster_passphrase or None)
            st.rerun()
        if cluster_actions[2].button("Leave Cluster", use_container_width=True):
            leave_cluster(settings)
            st.rerun()

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
        with st.expander("Stage expansion", expanded=False):
            try:
                stage_preview = stage_one_universe_snapshot(settings, top_n=500)
            except Exception as exc:
                st.info(f"Stage 1 preview unavailable: {exc}")
            else:
                st.write(f"Stage 1 candidate symbols: `{stage_preview['symbol_count']}`")
                st.write(f"Primary-source history currently available: `{stage_preview['history_probe'].get('effective_years', 0)}` years")
                st.write(stage_preview["symbols_preview"])
            with st.form("stage_promotion"):
                target_stage = st.selectbox("Target Stage", [0, 1], format_func=lambda value: f"Stage {value}")
                stage_symbol_count = st.number_input("Target Symbol Count", min_value=10, max_value=1000, step=10, value=500)
                stage_years = st.number_input("History Years", min_value=1, max_value=20, step=1, value=10)
                if st.form_submit_button("Advance Collection Stage"):
                    result = advance_collection_stage(
                        settings,
                        target_stage=int(target_stage),
                        symbol_count=int(stage_symbol_count),
                        years=int(stage_years),
                        passphrase=cluster_passphrase or None,
                    )
                    st.success(
                        f"Stage {result['stage']['current']} ready with {result['stage']['symbol_count']} symbols over {result['stage']['years']} years."
                    )
                    st.rerun()

        with st.expander("Advanced setup and cluster detail", expanded=False):
            st.write(f"Share: `{nas['share']}`")
            st.write(f"Mount path: `{nas['mount_path']}`")
            st.write(f"Mount writable: `{nas['mount_writable']}`")
            st.write(f"Systemd: `{snapshot['systemd'].get('ActiveState', snapshot['systemd'].get('reason', 'unknown'))}`")
            st.dataframe(cluster["workers"], use_container_width=True)
            st.dataframe(cluster["leases"], use_container_width=True)
            lease_options = [lease["lease_id"] for lease in cluster["leases"] if lease]
            selected = st.selectbox("Force release lease", [""] + lease_options)
            if selected and st.button("Force Release", type="secondary"):
                force_release_lease(settings, selected)
                st.rerun()

    with tabs[3]:
        st.subheader("Node Log")
        st.text_area("Recent log lines", value=snapshot["log_tail"], height=500)
        if snapshot["journal_tail"]:
            st.subheader("systemd Journal")
            st.text_area("Recent journal lines", value=snapshot["journal_tail"], height=300)
        with st.expander("Raw snapshot", expanded=False):
            st.json(snapshot, expanded=False)
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
