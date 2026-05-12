#!/usr/bin/env python3
"""
Generate a top-conference-level architecture diagram for the LeKiwi Depth Navigation Framework.
Outputs both SVG and PNG formats.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Arc, Circle, Polygon
import numpy as np

# ── Global style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 8.5,
    "axes.titlesize": 14,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})

# ── Color palette ─────────────────────────────────────────────────────────
C = {
    "bg":            "#FAFAFA",
    "hw_bg":         "#E8EAF6",
    "hw_border":     "#5C6BC0",
    "hw_accent":     "#3F51B5",
    "pi_bg":         "#E0F2F1",
    "pi_border":     "#00897B",
    "pi_accent":     "#00695C",
    "pc_bg":         "#FFF3E0",
    "pc_border":     "#FB8C00",
    "pc_accent":     "#EF6C00",
    "sim_bg":        "#F3E5F5",
    "sim_border":    "#8E24AA",
    "sim_accent":    "#7B1FA2",
    "tool_bg":       "#ECEFF1",
    "tool_border":   "#607D8B",
    "tool_accent":   "#455A64",
    "arrow_scan":    "#2E7D32",
    "arrow_cmd":     "#C62828",
    "arrow_model":   "#6A1B9A",
    "node_white":    "#FFFFFF",
    "text_dark":     "#263238",
    "text_muted":    "#546E7A",
}

def rbox(ax, x, y, w, h, fc, ec, lw=1.2, r=0.08, z=3, ls="-"):
    b = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                        facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls, zorder=z)
    ax.add_patch(b)

def node(ax, x, y, w, h, text, fc=C["node_white"], ec="#B0BEC5", tc=C["text_dark"],
         fs=7.5, fw="normal", z=4, r=0.06):
    rbox(ax, x - w/2, y - h/2, w, h, fc, ec, r=r, z=z)
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, fontweight=fw, color=tc, zorder=z+1)

def title_bar(ax, x, y, w, h, text, fc, tc="white", fs=9.5):
    rbox(ax, x - w/2, y - h/2, w, h, fc, fc, lw=0, r=0.10, z=5)
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, fontweight="bold", color=tc, zorder=6)

def group(ax, x, y, w, h, fc, ec, alpha=0.15):
    rbox(ax, x, y, w, h, fc, ec, lw=1.6, r=0.15, z=0)
    # slightly more opaque fill
    rbox(ax, x, y, w, h, fc, ec, lw=0, r=0.15, z=0)
    # redraw border on top
    rbox(ax, x, y, w, h, "none", ec, lw=1.6, r=0.15, z=1)

def arrow(ax, x1, y1, x2, y2, color="#37474F", lw=1.1, z=4, ls="-", rad=0.0):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="simple", color=color, lw=lw,
                                linestyle=ls, connectionstyle=f"arc3,rad={rad}"),
                zorder=z)

def dbl_arrow(ax, x1, y1, x2, y2, c1, c2, lw=1.2, rad=0.12, z=4):
    """Two parallel-ish arrows with slight curvature for bidirectional flow."""
    arrow(ax, x1, y1 + 0.06, x2, y2 + 0.06, color=c1, lw=lw, z=z, rad=rad)
    arrow(ax, x2, y2 - 0.06, x1, y1 - 0.06, color=c2, lw=lw, z=z, rad=-rad)

def lbl(ax, x, y, text, color=C["text_muted"], fs=6.5, rot=0, ha="center"):
    ax.text(x, y, text, fontsize=fs, color=color, ha=ha, va="center", style="italic", rotation=rot)

# ── Canvas ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(18, 12.5))
ax.set_xlim(0, 18)
ax.set_ylim(0, 12.5)
ax.set_aspect("equal")
ax.axis("off")
fig.patch.set_facecolor(C["bg"])
ax.set_facecolor(C["bg"])

# ═══════════════════════════════════════════════════════════════════════════
# TITLE
# ═══════════════════════════════════════════════════════════════════════════
ax.text(9, 12.15, "LeKiwi Depth Navigation Framework — System Architecture",
        ha="center", va="center", fontsize=18, fontweight="bold", color="#1A237E")
ax.text(9, 11.72, "Pseudo-LiDAR Omnidirectional Navigation with Learned Policies  |  RealSense D435i  →  Edge Computing  →  Policy Inference",
        ha="center", va="center", fontsize=9, color=C["text_muted"], style="italic")

# ═══════════════════════════════════════════════════════════════════════════
# LAYER 1: Simulation & Training (top, full width)
# ═══════════════════════════════════════════════════════════════════════════
sim_y, sim_h = 10.1, 1.35
group(ax, 0.3, sim_y - sim_h/2, 17.4, sim_h, C["sim_bg"], C["sim_border"], alpha=0.12)
ax.text(0.6, sim_y + 0.48, "Simulation & Training Pipeline", fontsize=10, fontweight="bold", color=C["sim_accent"])

sim_nodes = [
    (2.2, sim_y, "Map Generator\nrandom_maps"),
    (5.1, sim_y, "2D Navigation Env\nGymnasium"),
    (8.0, sim_y, "SAC Training\nStable-Baselines3"),
    (11.0, sim_y, "Evaluation\nSuccess / Collision / Timeout"),
    (14.0, sim_y, "Wasserstein Gap\nSim-to-Real"),
    (16.5, sim_y, "models/\n*.zip"),
]
sx_positions = [2.2, 5.1, 8.0, 11.0, 14.0, 16.5]
s_widths   = [2.6, 2.6, 2.8, 2.8, 2.6, 1.8]
for (sx, sy, sl), sw in zip(sim_nodes, s_widths):
    node(ax, sx, sy, sw, 0.72, sl, fc="#E1BEE7", ec=C["sim_border"], tc="#4A148C", r=0.08)
for i in range(len(sim_nodes) - 2):
    arrow(ax, sx_positions[i] + s_widths[i]/2 + 0.08, sim_y,
          sx_positions[i+1] - s_widths[i+1]/2 - 0.08, sim_y, color=C["sim_accent"], lw=0.8)
# Model storage is special — dotted arrow from evaluation
arrow(ax, 14.0 + 1.3, sim_y, 16.5 - 0.9, sim_y, color=C["arrow_model"], lw=1.0, ls="dotted")

# Deploy model arrow: sim → PC policy pool
arrow(ax, 12.5, sim_y - 0.68, 13.5, 8.2, color=C["arrow_model"], lw=1.1, ls="dashed", rad=-0.2)
lbl(ax, 13.15, 9.5, "Deploy\nModel", C["arrow_model"], fs=6.5)

# ═══════════════════════════════════════════════════════════════════════════
# LAYER 2: Compute — Raspberry Pi (left)  |  ZMQ Bridge  |  PC (right)
# ═══════════════════════════════════════════════════════════════════════════

# --- Raspberry Pi Group ---
pi_x, pi_y, pi_w, pi_h = 0.4, 4.55, 5.5, 4.8
group(ax, pi_x, pi_y - pi_h/2, pi_w, pi_h, C["pi_bg"], C["pi_border"], alpha=0.12)
title_bar(ax, pi_x + pi_w/2, pi_y + pi_h/2 - 0.28, 4.8, 0.50,
          "Raspberry Pi  —  Edge Computing", C["pi_border"], fs=9.5)

pi_modules = [
    (3.15, 6.60, "Camera Node\ncamera_node.py", "RealSense or Mock"),
    (3.15, 5.70, "Depth-to-Scan\ndepth_to_scan.py", "64-D pseudo-LiDAR"),
    (3.15, 4.80, "Scan Publisher\nscan_publisher.py", "ZMQ PUB"),
    (3.15, 3.85, "Command Subscriber\ncommand_subscriber.py", "ZMQ SUB"),
    (3.15, 2.90, "Base Controller\nbase_controller.py", "Kinematics + Motor Cmd"),
]
for mx, my, mlabel, msub in pi_modules:
    node(ax, mx, my, 3.8, 0.52, mlabel, fc="#B2DFDB", ec=C["pi_border"], tc=C["pi_accent"], r=0.06)
    ax.text(mx, my - 0.33, msub, ha="center", fontsize=6, color="#607D8B", style="italic")

# Pi internal arrows
for i in range(len(pi_modules) - 1):
    arrow(ax, 3.15, pi_modules[i][1] - 0.29, 3.15, pi_modules[i+1][1] + 0.29,
          color=C["pi_accent"], lw=0.8)

# --- PC Group ---
pc_x, pc_y, pc_w, pc_h = 10.3, 4.55, 7.3, 4.8
group(ax, pc_x, pc_y - pc_h/2, pc_w, pc_h, C["pc_bg"], C["pc_border"], alpha=0.12)
title_bar(ax, pc_x + pc_w/2, pc_y + pc_h/2 - 0.28, 5.8, 0.50,
          "PC Workstation  —  Policy Server & Inference", C["pc_border"], fs=9.5)

# Policy Server
node(ax, 13.95, 6.55, 4.2, 0.52, "Policy Server  (20 Hz loop)\npolicy_server.py",
     fc="#FFE0B2", ec=C["pc_border"], tc=C["pc_accent"], fw="bold")

# ZMQ Bridge (inside PC)
node(ax, 13.95, 5.75, 3.2, 0.44, "ZMQ Bridge\nscan SUB  |  cmd PUB",
     fc="#FFE0B2", ec=C["pc_border"], tc=C["pc_accent"])

# Policy Pool label
ax.text(13.95, 5.20, "Policy Pool", fontsize=8, fontweight="bold", ha="center", color=C["pc_accent"])

policies = [
    (11.8, 4.55, "Rule-Based\nrule_policy.py"),
    (13.95, 4.55, "DWA Planner\ndwa_policy.py"),
    (16.1, 4.55, "MLP (SAC)\nmlp_policy.py"),
]
for px, py, pl in policies:
    node(ax, px, py, 2.0, 0.72, pl, fc="#FFE0B2", ec=C["pc_border"], tc=C["pc_accent"], r=0.07)

# PC internal arrows
arrow(ax, 13.95, 6.29, 13.95, 6.0, color=C["pc_accent"], lw=0.8)
for px, py, _ in policies:
    arrow(ax, 13.95, 5.75 - 0.22, px, py + 0.36, color=C["pc_accent"], lw=0.7)

# Model storage (inside PC)
node(ax, 13.95, 3.55, 2.4, 0.40, "models/*.zip (SAC weights)",
     fc="#FFCC80", ec=C["pc_border"], tc=C["pc_accent"], fs=6.5, r=0.06)
arrow(ax, 13.95, 4.19, 13.95, 3.77, color=C["arrow_model"], lw=0.8)
lbl(ax, 14.8, 3.98, "loads", C["arrow_model"], fs=6)

# --- ZMQ Bridge (Center, between Pi and PC) ---
bridge_x, bridge_y = 8.3, 5.6
# Visual bridge pillar
rbox(ax, bridge_x - 0.7, 3.15, 1.4, 4.7, C["node_white"], "#B0BEC5", lw=1.0, r=0.10, z=2)
ax.text(bridge_x, 7.1, "ZMQ", fontsize=9, fontweight="bold", ha="center", color="#455A64", zorder=3)
ax.text(bridge_x, 6.75, "PUB/SUB", fontsize=8, ha="center", color="#455A64", zorder=3)
ax.text(bridge_x, 6.20, "64-D Scan", fontsize=6.5, ha="center", color=C["arrow_scan"], style="italic", zorder=3)
ax.text(bridge_x, 5.95, "< 10 KB/s", fontsize=6, ha="center", color=C["arrow_scan"], zorder=3)
ax.text(bridge_x, 4.80, "Velocity Cmd", fontsize=6.5, ha="center", color=C["arrow_cmd"], style="italic", zorder=3)
ax.text(bridge_x, 4.55, "< 1 KB/s", fontsize=6, ha="center", color=C["arrow_cmd"], zorder=3)

# Wi-Fi label
ax.text(bridge_x, 5.30, "Wi-Fi /\nEthernet", fontsize=6.5, ha="center", color="#78909C",
        style="italic", zorder=3,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#CFD8DC", alpha=0.9))

# Cross arrows: Pi ↔ PC via ZMQ
# Scan: Pi publisher (right side) → PC subscriber
arrow(ax, 3.15 + 1.95, 4.80, bridge_x - 0.1, 5.95, color=C["arrow_scan"], lw=1.4, rad=-0.08)
arrow(ax, bridge_x + 0.1, 6.05, 13.95 - 1.65, 5.75, color=C["arrow_scan"], lw=1.2, rad=-0.06)
# Cmd: PC publisher → Pi subscriber
arrow(ax, 13.95 - 1.60, 5.55, bridge_x + 0.1, 4.70, color=C["arrow_cmd"], lw=1.4, rad=0.08)
arrow(ax, bridge_x - 0.1, 4.50, 3.15 + 1.95, 3.85, color=C["arrow_cmd"], lw=1.2, rad=0.06)

# ═══════════════════════════════════════════════════════════════════════════
# LAYER 3: Hardware / Physical (bottom, full width)
# ═══════════════════════════════════════════════════════════════════════════
hw_y, hw_h = 1.25, 1.30
group(ax, 0.3, hw_y - hw_h/2, 17.4, hw_h, C["hw_bg"], C["hw_border"], alpha=0.12)
ax.text(0.6, hw_y + 0.45, "Hardware Layer", fontsize=10, fontweight="bold", color=C["hw_accent"])

hw_items = [
    (2.2, hw_y, "RealSense D435i\nDepth Camera"),
    (5.6, hw_y, "Raspberry Pi 4/5\nEdge Computer"),
    (9.0, hw_y, "LeKiwi Robot Body\n3-Omniwheel Platform"),
    (12.4, hw_y, "Feetech STS3215\nServo Motors ×3"),
    (15.6, hw_y, "PC Workstation\nRTX 5060 GPU"),
]
for hx, hy, hl in hw_items:
    node(ax, hx, hy, 2.8, 0.62, hl, fc="#C5CAE9", ec=C["hw_border"], tc=C["hw_accent"],
         fw="bold", r=0.08)

# HW internal arrows
hw_xs = [2.2, 5.6, 9.0, 12.4, 15.6]
for i in range(len(hw_xs) - 1):
    arrow(ax, hw_xs[i] + 1.4, hw_y, hw_xs[i+1] - 1.4, hw_y, color=C["hw_accent"], lw=0.7)

# Pi ↔ hardware vertical connections
arrow(ax, 5.6, hw_y + 0.31, 3.15, 2.65, color=C["hw_accent"], lw=0.9)    # Pi ↔ RPi HW
arrow(ax, 3.15, 2.65, 2.2, hw_y + 0.31, color=C["arrow_scan"], lw=1.0)   # D435i → Camera Node
arrow(ax, 12.4, hw_y + 0.31, 13.95, 3.35, color=C["hw_accent"], lw=0.9)  # Motors ← PC (indirect)
arrow(ax, 3.15 + 1.2, 2.65, 9.0, hw_y + 0.31, color=C["arrow_cmd"], lw=1.0)  # Base Ctrl → Motors

# ═══════════════════════════════════════════════════════════════════════════
# LAYER 4: Tools & Analysis (outside main flow, bottom strip)
# ═══════════════════════════════════════════════════════════════════════════
tool_x, tool_y, tool_w, tool_h = 0.4, 0.15, 17.4, 0.85
group(ax, tool_x, tool_y, tool_w, tool_h, C["tool_bg"], C["tool_border"], alpha=0.08,)
ax.text(0.7, tool_y + tool_h - 0.22, "Tools & Analysis", fontsize=8.5, fontweight="bold", color=C["tool_accent"])

tools = [
    (2.5, tool_y + 0.38, "Record\nrecord_scan_log.py"),
    (5.0, tool_y + 0.38, "Replay\nreplay_scan_log.py"),
    (7.5, tool_y + 0.38, "Plot Scan\nplot_scan.py"),
    (10.0, tool_y + 0.38, "Latency Test\nlatency_test.py"),
    (12.5, tool_y + 0.38, "Bandwidth Test\nbandwidth_test.py"),
    (15.0, tool_y + 0.38, "Wasserstein\ncompute_wasserstein.py"),
]
for tx, ty, tl in tools:
    node(ax, tx, ty, 2.3, 0.46, tl, fc="white", ec=C["tool_border"], tc="#37474F", fs=6, r=0.05)

# ═══════════════════════════════════════════════════════════════════════════
# LEGEND
# ═══════════════════════════════════════════════════════════════════════════
lx, ly = 0.5, 9.0
rbox(ax, lx - 0.1, ly - 0.22, 14.5, 0.55, "white", "#CFD8DC", lw=0.8, r=0.06, z=8)
legends = [
    (lx, "Scan / Depth Data", C["arrow_scan"]),
    (lx + 4.0, "Velocity Command", C["arrow_cmd"]),
    (lx + 7.6, "Model Weights", C["arrow_model"]),
    (lx + 10.8, "Deploy (Sim → Real)", C["arrow_model"]),
]
for lgx, lgl, lgc in legends:
    ax.annotate("", xy=(lgx + 0.6, ly), xytext=(lgx - 0.15, ly),
                arrowprops=dict(arrowstyle="simple", color=lgc, lw=1.5), zorder=9)
    ax.text(lgx + 0.72, ly, lgl, fontsize=7, color=lgc, va="center", zorder=9)
# Dashed indicator
ax.annotate("", xy=(lx + 11.4, ly), xytext=(lx + 10.65, ly),
            arrowprops=dict(arrowstyle="simple", color=C["arrow_model"], lw=1.5, linestyle="dashed"), zorder=9)

# ═══════════════════════════════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════════════════════════════
out_svg = "/data/zbx_projects/lekiwi_depth_nav/architecture_diagram.svg"
out_png = "/data/zbx_projects/lekiwi_depth_nav/architecture_diagram.png"
fig.savefig(out_svg, format="svg", facecolor=C["bg"], edgecolor="none")
fig.savefig(out_png, format="png", facecolor=C["bg"], edgecolor="none")
plt.close(fig)
print(f"Saved:\n  {out_svg}\n  {out_png}")
