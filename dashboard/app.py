"""
LeKiwi Depth Navigation Dashboard
==================================
Streamlit-based interactive dashboard for exploring pseudo-LiDAR scan logs
(.npz format) recorded with tools/record_scan_log.py.

Usage:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

# Allow importing project modules without installing
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import streamlit as st

# Wide layout
st.set_page_config(
    page_title="LeKiwi Depth Navigation Dashboard",
    page_icon="📡",
    layout="wide",
)

st.title("LeKiwi Depth Navigation Dashboard")
st.caption("Pseudo-LiDAR scan log explorer — heatmap, polar view, metrics over time.")

# ── Sidebar: data source ──────────────────────────────────────────────────

st.sidebar.header("Data Source")

uploaded_file = st.sidebar.file_uploader(
    "Upload a .npz scan log", type=["npz"],
    help="Log files recorded by tools/record_scan_log.py",
)

log_dir = Path(__file__).resolve().parent.parent / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
existing_logs = sorted(log_dir.glob("*.npz"))
log_options = {p.name: str(p) for p in existing_logs}
selected_log_name = st.sidebar.selectbox(
    "Or select an existing log from logs/",
    ["(none)"] + list(log_options.keys()),
)

# Resolve data
data: Optional[dict] = None
if uploaded_file is not None:
    with st.sidebar:
        st.success(f"Loaded: {uploaded_file.name}")
    data = dict(np.load(uploaded_file, allow_pickle=True))
elif selected_log_name != "(none)":
    path = log_options[selected_log_name]
    with st.sidebar:
        st.info(f"Loaded: {selected_log_name}")
    data = dict(np.load(path, allow_pickle=True))

if data is None:
    st.info("Upload a .npz scan log or select one from the logs/ directory to begin.")
    st.stop()

# ── Parse data ────────────────────────────────────────────────────────────

scans_m: np.ndarray = data.get("scans_m")  # (T, N)
scans_norm: Optional[np.ndarray] = data.get("scans_norm")  # (T, N)
timestamps: np.ndarray = data.get("timestamps")
actions: Optional[np.ndarray] = data.get("actions")  # (T, 3) optional
episode_info: Optional[dict] = data.get("episode_info", None)

if scans_m is None:
    st.error("Log must contain 'scans_m' array. Found keys: " + ", ".join(data.keys()))
    st.stop()

num_frames, num_bins = scans_m.shape
if timestamps is None:
    timestamps = np.arange(num_frames, dtype=np.float64)
else:
    timestamps = np.asarray(timestamps, dtype=np.float64)

# Relative time
t_rel = timestamps - timestamps[0]

st.sidebar.markdown("---")
st.sidebar.header("Display")
max_range_val = float(st.sidebar.slider("Max range (m)", 1.0, 10.0, 5.0, 0.5))
frame_idx = st.sidebar.slider("Frame index", 0, num_frames - 1, num_frames // 2)
fov_deg = float(st.sidebar.slider("FOV (deg)", 30.0, 180.0, 90.0, 5.0))

# ── Derived values ────────────────────────────────────────────────────────
half_fov = np.deg2rad(fov_deg / 2.0)
angles_rad = np.linspace(-half_fov, half_fov, num_bins)
angles_deg = np.rad2deg(angles_rad)
min_dists = np.min(scans_m, axis=1)

# ── Layout ────────────────────────────────────────────────────────────────

col1, col2 = st.columns(2)

# --- (a) Scan Heatmap ---
with col1:
    st.subheader("Scan Heatmap")
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4))
        extent = [angles_deg[0], angles_deg[-1], t_rel[-1], t_rel[0]]
        im = ax.imshow(
            scans_m, aspect="auto", extent=extent,
            cmap="viridis_r", vmin=0, vmax=max_range_val,
        )
        ax.set_xlabel("Angle (deg)")
        ax.set_ylabel("Time (s)")
        plt.colorbar(im, ax=ax, label="Range (m)")
        st.pyplot(fig)
        plt.close(fig)
    except Exception:
        st.warning("matplotlib needed for heatmap.")

# --- (b) Single-frame Scan Curve ---
with col2:
    st.subheader(f"Scan Curve (frame {frame_idx})")
    try:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.fill_between(angles_deg, 0, scans_m[frame_idx], alpha=0.25, color="blue")
        ax.plot(angles_deg, scans_m[frame_idx], "b-", linewidth=1.5)
        ax.axhline(y=max_range_val, color="gray", ls="--", alpha=0.4, label="max range")
        ax.set_xlabel("Angle (deg)")
        ax.set_ylabel("Range (m)")
        ax.set_ylim(0, max_range_val * 1.1)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=7)
        st.pyplot(fig)
        plt.close(fig)
    except Exception:
        st.warning("matplotlib needed.")

col3, col4 = st.columns(2)

# --- (c) Polar Scan ---
with col3:
    st.subheader(f"Polar View (frame {frame_idx})")
    try:
        fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={"projection": "polar"})
        ax.fill_between(angles_rad, 0, scans_m[frame_idx], alpha=0.25, color="blue")
        ax.plot(angles_rad, scans_m[frame_idx], "b-", linewidth=1.5)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_thetamin(-fov_deg / 2)
        ax.set_thetamax(fov_deg / 2)
        ax.set_ylim(0, max_range_val * 1.1)
        st.pyplot(fig)
        plt.close(fig)
    except Exception:
        st.warning("matplotlib needed.")

# --- (d) Min Distance Over Time ---
with col4:
    st.subheader("Min Range Over Time")
    try:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(t_rel, min_dists, "r-", linewidth=1.0)
        ax.axhline(y=0.3, color="orange", ls="--", alpha=0.5, label="safe (0.3 m)")
        ax.axhline(y=0.15, color="red", ls="--", alpha=0.5, label="danger (0.15 m)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Min Range (m)")
        ax.set_ylim(0, max_range_val * 1.1)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=7)
        st.pyplot(fig)
        plt.close(fig)
    except Exception:
        st.warning("matplotlib needed.")

# --- (e) Actions (if present) ---
if actions is not None and len(actions) == num_frames:
    st.markdown("---")
    st.subheader("Velocity Commands (vx, vy, omega)")
    actions = np.asarray(actions)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    ax1.plot(t_rel, actions[:, 0], "b-", linewidth=0.8)
    ax1.set_ylabel("vx (m/s)")
    ax1.grid(True, alpha=0.3)
    ax2.plot(t_rel, actions[:, 1], "g-", linewidth=0.8)
    ax2.set_ylabel("vy (m/s)")
    ax2.grid(True, alpha=0.3)
    ax3.plot(t_rel, actions[:, 2], "r-", linewidth=0.8)
    ax3.set_ylabel("omega (deg/s)")
    ax3.set_xlabel("Time (s)")
    ax3.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

# --- (f) Episode Summary (if present) ---
if episode_info is not None:
    st.markdown("---")
    st.subheader("Episode Summary")
    ei = dict(episode_info)
    if isinstance(ei, dict):
        cols = st.columns(3)
        status_map = {
            "success": ei.get("success", ei.get("successes", 0)),
            "collision": ei.get("collision", ei.get("collisions", 0)),
            "timeout": ei.get("timeout", ei.get("timeouts", 0)),
        }
        cols[0].metric("Success", status_map.get("success", "—"))
        cols[1].metric("Collision", status_map.get("collision", "—"))
        cols[2].metric("Timeout", status_map.get("timeout", "—"))

        # Also show all keys if they don"t match the common format
        with st.expander("Raw episode info"):
            st.json(ei)

# ── Sidebar: summary stats ────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.subheader("Log Summary")
st.sidebar.write(f"Frames: {num_frames}")
st.sidebar.write(f"Scan bins: {num_bins}")
st.sidebar.write(f"Duration: {t_rel[-1]:.1f} s")
st.sidebar.write(f"Mean FPS: {num_frames / max(t_rel[-1], 0.001):.1f}")
st.sidebar.write(f"Min range overall: {min_dists.min():.3f} m")
st.sidebar.write(f"Mean min range: {min_dists.mean():.3f} m")
st.sidebar.write(f"Has actions: {actions is not None}")
st.sidebar.write(f"Has episode info: {episode_info is not None}")
