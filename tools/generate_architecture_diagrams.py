#!/usr/bin/env python3
"""
Generate 5 clean, focused architecture diagrams for the LeKiwi Depth Navigation Framework.
Each diagram is self-contained — suitable for a top-conference paper.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import os

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})

OUT = "/data/zbx_projects/lekiwi_depth_nav"

# ── Shared palette ────────────────────────────────────────────────────────
C = {
    "bg":           "#FAFAFA",
    "hw":           "#5C6BC0",
    "hw_light":     "#C5CAE9",
    "pi":           "#00897B",
    "pi_light":     "#B2DFDB",
    "pc":           "#FB8C00",
    "pc_light":     "#FFE0B2",
    "sim":          "#8E24AA",
    "sim_light":    "#E1BEE7",
    "tool":         "#607D8B",
    "tool_light":   "#CFD8DC",
    "scan":         "#2E7D32",
    "cmd":          "#C62828",
    "model":        "#6A1B9A",
    "white":        "#FFFFFF",
    "dark":         "#263238",
    "muted":        "#546E7A",
    "border":       "#B0BEC5",
}

def rbox(ax, x, y, w, h, fc, ec, lw=1.2, r=0.08, z=3, ls="-", alpha=1.0):
    b = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                        facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls, zorder=z, alpha=alpha)
    ax.add_patch(b)

def node(ax, x, y, w, h, text, fc=C["white"], ec=C["border"], tc=C["dark"],
         fs=8, fw="normal", z=5, r=0.06, ha="center", va="center"):
    rbox(ax, x - w/2, y - h/2, w, h, fc, ec, r=r, z=z)
    ax.text(x, y, text, ha=ha, va=va, fontsize=fs, fontweight=fw, color=tc, zorder=z+1)

def title_bar(ax, x, y, w, h, text, fc, tc="white", fs=10):
    rbox(ax, x - w/2, y - h/2, w, h, fc, fc, lw=0, r=0.10, z=8)
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, fontweight="bold", color=tc, zorder=9)

def group(ax, x, y, w, h, fc, ec, alpha=0.10):
    rbox(ax, x, y, w, h, fc, ec, lw=1.5, r=0.14, z=0, alpha=alpha)

def arrow(ax, x1, y1, x2, y2, color=C["dark"], lw=1.1, z=4, ls="-", rad=0.0):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="simple", color=color, lw=lw,
                                linestyle=ls, connectionstyle=f"arc3,rad={rad}"),
                zorder=z)

def albl(ax, x, y, text, color=C["muted"], fs=6.5, rot=0, ha="center"):
    ax.text(x, y, text, fontsize=fs, color=color, ha=ha, va="center", style="italic", rotation=rot)

def new_fig(w=14, h=8):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor(C["bg"])
    ax.set_facecolor(C["bg"])
    return fig, ax

def save(fig, name):
    for ext in ["svg", "png"]:
        path = os.path.join(OUT, f"arch_{name}.{ext}")
        fig.savefig(path, format=ext, facecolor=C["bg"], edgecolor="none")
    plt.close(fig)
    print(f"  arch_{name}.{{svg,png}}")

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  FIG 1 — System Overview & Physical Deployment                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝
fig, ax = new_fig(16, 7)

ax.text(8, 6.65, "System Overview — Physical Deployment & Data Flow",
        ha="center", fontsize=16, fontweight="bold", color="#1A237E")
ax.text(8, 6.25, "LeKiwi three-omniwheel robot with edge-to-server split architecture",
        ha="center", fontsize=9, color=C["muted"], style="italic")

# --- Robot Body ---
group(ax, 0.4, 1.0, 6.8, 4.8, C["pi_light"], C["pi"], alpha=0.10)
ax.text(3.8, 5.55, "LeKiwi Robot (On-Body)", fontsize=11, fontweight="bold", ha="center", color=C["pi"])

# Camera
node(ax, 1.8, 4.3, 2.4, 1.0, "RealSense D435i\nRGB-D Camera", fc=C["hw_light"], ec=C["hw"], tc="#283593", fw="bold", r=0.10)
# RPi
node(ax, 3.8, 4.3, 2.4, 1.0, "Raspberry Pi 4/5\nEdge Computer", fc=C["pi_light"], ec=C["pi"], tc="#004D40", fw="bold", r=0.10)
# Motors
node(ax, 5.8, 4.3, 1.6, 1.0, "STS3215\n×3", fc=C["hw_light"], ec=C["hw"], tc="#283593", fw="bold", r=0.10)

# On-robot arrows
arrow(ax, 3.0, 4.3, 4.6, 4.3, color=C["scan"], lw=1.6)
albl(ax, 3.8, 4.6, "USB 3.0", C["scan"])
arrow(ax, 5.0, 4.3, 5.0, 4.3, color=C["cmd"], lw=1.6)  # RPi → Motors (GPIO)
albl(ax, 5.0, 3.95, "GPIO/\nUART", C["cmd"], fs=6)

# Robot platform visual
rbox(ax, 1.0, 1.4, 5.6, 1.2, C["pi_light"], C["pi"], lw=1.2, r=0.12, z=2, alpha=0.3)
ax.text(3.8, 2.0, "3-Omniwheel Platform  |  base_radius = 0.125 m  |  wheel_radius = 0.05 m",
        ha="center", fontsize=7.5, color=C["pi"], fontweight="bold")

# Wheel icons (3 small circles)
for angle_deg in [150, -90, 30]:
    rad = np.deg2rad(angle_deg)
    wx, wy = 3.8 + 1.1 * np.cos(rad), 2.0 + 1.1 * np.sin(rad)
    wheel = plt.Circle((wx, wy), 0.25, fc=C["white"], ec=C["pi"], lw=1.2, zorder=4)
    ax.add_patch(wheel)
    ax.text(wx, wy, "W", ha="center", va="center", fontsize=6, color=C["pi"], fontweight="bold")

# --- PC (Remote) ---
group(ax, 8.5, 1.0, 7.1, 4.8, C["pc_light"], C["pc"], alpha=0.10)
ax.text(12.05, 5.55, "PC Workstation (Remote)", fontsize=11, fontweight="bold", ha="center", color=C["pc"])

node(ax, 10.4, 4.3, 2.8, 0.9, "RTX 5060 GPU\nInference Accelerator", fc=C["pc_light"], ec=C["pc"], tc="#E65100", fw="bold", r=0.10)
node(ax, 13.7, 4.3, 2.6, 0.9, "Policy Server\n20 Hz control loop", fc=C["pc_light"], ec=C["pc"], tc="#E65100", fw="bold", r=0.10)

# Policy icons on PC side
policies = [
    (10.4, 2.8, "Rule"),
    (12.05, 2.8, "DWA"),
    (13.7, 2.8, "MLP\n(SAC)"),
]
for px, py, pl in policies:
    node(ax, px, py, 1.5, 0.8, pl, fc=C["pc_light"], ec=C["pc"], tc="#E65100", r=0.08)

# PC internal
arrow(ax, 12.05, 3.85, 12.05, 3.22, color=C["pc"], lw=1.0)
for px, py, _ in policies:
    arrow(ax, 11.2, 3.35, px, py + 0.4, color=C["pc"], lw=0.7)
albl(ax, 11.3, 3.55, "policy\nselect", C["pc"])

# --- Wireless Link ---
# Lightning / wireless symbol in middle
ax.text(7.8, 4.3, "Wi-Fi /\nEthernet", ha="center", fontsize=8, fontweight="bold", color="#78909C",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#CFD8DC", lw=1.0))
arrow(ax, 5.0, 4.8, 7.2, 4.65, color=C["scan"], lw=1.8, rad=-0.05)
arrow(ax, 8.4, 4.5, 10.4, 4.8, color=C["scan"], lw=1.8, rad=-0.05)
arrow(ax, 12.05, 3.95, 8.4, 3.8, color=C["cmd"], lw=1.8, rad=-0.05)
arrow(ax, 7.2, 3.65, 4.2, 3.6, color=C["cmd"], lw=1.8, rad=-0.05)

albl(ax, 6.2, 5.1, "64-D pseudo-LiDAR scan, < 10 KB/s", C["scan"], fs=7)
albl(ax, 8.0, 3.3, "Velocity command (vx, vy, ω), < 1 KB/s", C["cmd"], fs=7)

# --- Legend ---
lx, ly = 0.6, 0.5
rbox(ax, lx - 0.1, ly - 0.15, 15.0, 0.42, C["white"], "#CFD8DC", lw=0.8, r=0.05, z=8)
for i, (ll, lc) in enumerate([("Scan Data (upstream)", C["scan"]),
                                ("Velocity Cmd (downstream)", C["cmd"])]):
    px = lx + i * 7.0
    ax.annotate("", xy=(px + 0.5, ly), xytext=(px - 0.1, ly),
                arrowprops=dict(arrowstyle="simple", color=lc, lw=1.5), zorder=9)
    ax.text(px + 0.6, ly, ll, fontsize=7, color=lc, va="center", zorder=9)

save(fig, "01_system_overview")

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  FIG 2 — Raspberry Pi Edge Computing Pipeline                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝
fig, ax = new_fig(14, 8.5)

ax.text(7, 8.15, "Edge Computing Pipeline — Raspberry Pi",
        ha="center", fontsize=16, fontweight="bold", color="#1A237E")
ax.text(7, 7.72, "Depth capture → pseudo-LiDAR scan → publish → subscribe → motor control",
        ha="center", fontsize=9, color=C["muted"], style="italic")

# Pipeline stages
stages = [
    (1.8, 6.4, "①  Camera Node\ncamera_node.py",           "RealSense D435i\nor MockCamera\n\n640×480 @ 30 fps\ndepth frame"),
    (4.2, 6.4, "②  Depth-to-Scan\ndepth_to_scan.py",        "Band extraction\nmin-pool per bin\n\nfov=90°, bins=64\nrange [0.15, 5.0] m"),
    (7.0, 6.4, "③  Scan Publisher\nscan_publisher.py",       "ZMQ PUB socket\nnon-blocking send\n\n64-D float32 array\n< 10 KB/s"),
    (9.8, 6.4, "④  Cmd Subscriber\ncommand_subscriber.py",   "ZMQ SUB socket\nnon-blocking recv\n\n(vx, vy, ω)\n3-float tuple"),
    (12.6, 6.4, "⑤  Base Controller\nbase_controller.py",    "LeKiwi kinematics\ninverse kinematics\n\nSTS3215 velocity\n±3000 raw range"),
]

for sx, sy, stitle, sdetail in stages:
    # Title box
    rbox(ax, sx - 1.1, sy + 0.2, 2.2, 0.85, C["pi_light"], C["pi"], lw=1.2, r=0.08, z=4)
    ax.text(sx, sy + 0.62, stitle, ha="center", va="center", fontsize=7.5,
            fontweight="bold", color=C["pi"])
    # Detail box
    rbox(ax, sx - 1.1, sy - 1.15, 2.2, 1.15, C["white"], C["border"], lw=0.8, r=0.06, z=3)
    ax.text(sx, sy - 0.58, sdetail, ha="center", va="center", fontsize=6.2, color=C["dark"])

# Stage arrows
for i in range(len(stages) - 1):
    arrow(ax, stages[i][0] + 1.1, 6.4, stages[i+1][0] - 1.1, 6.4, color=C["pi"], lw=1.3)

# --- Depth-to-Scan Detail (zoom) ---
group(ax, 0.5, 0.3, 13.0, 3.5, C["pi_light"], C["pi"], alpha=0.06)
ax.text(7.0, 3.55, "Depth-to-Scan Algorithm Detail (depth_to_scan.py)",
        ha="center", fontsize=10, fontweight="bold", color=C["pi"])

# Visual: depth image → bins → scan
# Mock depth image
rbox(ax, 1.0, 1.2, 2.0, 1.8, "#1B5E20", "#4CAF50", lw=1.2, r=0.05, z=3, alpha=0.6)
ax.text(2.0, 2.1, "Depth Image\n640×480", ha="center", fontsize=7, color=C["dark"])
albl(ax, 2.0, 1.0, "band rows [220:260]", C["pi"])

arrow(ax, 3.1, 2.1, 4.6, 2.1, color=C["pi"], lw=1.3)

# Binning visual
for bi in range(8):
    bx = 5.0 + bi * 0.65
    rbox(ax, bx, 1.7 - bi * 0.03, 0.55, 1.0 + bi * 0.06, C["pi_light"], C["pi"], lw=0.8, r=0.03, z=3, alpha=0.5)
ax.text(7.6, 2.1, "64 angular bins\nmin-pool reduction", ha="center", fontsize=7, color=C["dark"])
albl(ax, 7.6, 1.0, "bottleneck: ~2 ms", C["pi"])

arrow(ax, 8.9, 2.1, 10.4, 2.1, color=C["pi"], lw=1.3)

# Result scan
rbox(ax, 10.8, 1.2, 2.2, 1.8, C["white"], C["scan"], lw=1.2, r=0.06, z=3)
ax.text(11.9, 2.1, "Pseudo-LiDAR\n64-dim, float32\n\n< 256 bytes", ha="center", fontsize=7, color=C["scan"])

# Formula
ax.text(7.0, 0.55, r"$\mathrm{scan}[i] = \min_{r \in [220,260]} \;\; \mathrm{depth}\!\left(r,\;\left\lfloor \frac{i}{N_{\mathrm{bins}}} \cdot W \right\rfloor \right)$",
        ha="center", fontsize=8.5, color=C["pi"],
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=C["pi"], lw=0.8, alpha=0.8))

save(fig, "02_raspberry_pi_pipeline")

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  FIG 3 — PC Policy Server & Inference                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝
fig, ax = new_fig(15, 8.5)

ax.text(7.5, 8.15, "Policy Server & Inference — PC Workstation",
        ha="center", fontsize=16, fontweight="bold", color="#1A237E")
ax.text(7.5, 7.72, "ZMQ bridge → policy dispatch → inference → command publish (20 Hz control loop)",
        ha="center", fontsize=9, color=C["muted"], style="italic")

# --- ZMQ Bridge ---
group(ax, 0.4, 4.7, 3.2, 2.8, C["pc_light"], C["pc"], alpha=0.10)
ax.text(2.0, 7.25, "ZMQ Bridge", ha="center", fontsize=10, fontweight="bold", color=C["pc"])
node(ax, 2.0, 6.4, 2.6, 0.7, "SUB socket\nrecv scan (64-D)", fc=C["pc_light"], ec=C["pc"], tc="#E65100", r=0.08)
node(ax, 2.0, 5.4, 2.6, 0.7, "PUB socket\nsend cmd (vx,vy,ω)", fc=C["pc_light"], ec=C["pc"], tc="#E65100", r=0.08)
arrow(ax, 0.7, 6.4, 0.7, 5.4, color=C["scan"], lw=1.5)   # scan in
albl(ax, 0.25, 5.9, "scan", C["scan"], rot=90)
arrow(ax, 3.3, 5.4, 3.3, 6.4, color=C["cmd"], lw=1.5)    # cmd out
albl(ax, 3.65, 5.9, "cmd", C["cmd"], rot=90)

# --- Policy Server ---
group(ax, 4.2, 4.7, 6.2, 2.8, C["pc_light"], C["pc"], alpha=0.10)
ax.text(7.3, 7.25, "Policy Server (policy_server.py)", ha="center", fontsize=10, fontweight="bold", color=C["pc"])

# Main loop box
rbox(ax, 5.0, 5.1, 4.6, 2.1, C["white"], C["pc"], lw=1.2, r=0.10, z=3)
ax.text(7.3, 6.95, "Inference Loop @ 20 Hz", ha="center", fontsize=9, fontweight="bold", color=C["pc"])
loop_steps = [
    "1. ZMQ SUB: receive 64-D scan",
    "2. Normalize scan to [0, 1]",
    "3. Policy forward(scan) → action",
    "4. Unscale action → (vx, vy, ω)",
    "5. ZMQ PUB: send velocity cmd",
]
for i, step in enumerate(loop_steps):
    ax.text(5.3, 6.40 - i * 0.28, step, fontsize=6.8, color=C["dark"], va="center")

arrow(ax, 4.2, 6.4, 5.0, 6.4, color=C["pc"], lw=1.0)
arrow(ax, 9.6, 5.6, 10.4, 5.6, color=C["pc"], lw=1.0)

# --- Policy Pool ---
group(ax, 11.0, 4.7, 3.6, 2.8, C["pc_light"], C["pc"], alpha=0.10)
ax.text(12.8, 7.25, "Policy Pool", ha="center", fontsize=10, fontweight="bold", color=C["pc"])

policies = [
    (12.8, 6.4, "Rule-Based\nObstacle Avoidance"),
    (12.8, 5.4, "DWA Planner\nLocal Trajectory"),
]
for px, py, pl in policies:
    node(ax, px, py, 2.8, 0.7, pl, fc=C["pc_light"], ec=C["pc"], tc="#E65100", r=0.08)

# --- MLP Policy Detail (bottom) ---
group(ax, 0.4, 0.3, 14.2, 4.1, C["sim_light"], C["sim"], alpha=0.06)
ax.text(7.5, 4.15, "MLP Policy Architecture (SAC-trained, mlp_policy.py)",
        ha="center", fontsize=10, fontweight="bold", color=C["sim"])

# Network architecture
layer_y = 2.8
# Input
node(ax, 2.0, layer_y, 2.0, 0.6, "Input\n64-D scan", fc=C["sim_light"], ec=C["sim"], tc="#4A148C", r=0.08)
arrow(ax, 3.1, layer_y, 4.4, layer_y, color=C["sim"], lw=1.2)
# Hidden 1
node(ax, 5.6, layer_y, 2.0, 0.6, "Hidden FC\n256 + ReLU", fc=C["sim_light"], ec=C["sim"], tc="#4A148C", r=0.08)
arrow(ax, 6.7, layer_y, 8.0, layer_y, color=C["sim"], lw=1.2)
# Hidden 2
node(ax, 9.2, layer_y, 2.0, 0.6, "Hidden FC\n256 + ReLU", fc=C["sim_light"], ec=C["sim"], tc="#4A148C", r=0.08)
arrow(ax, 10.3, layer_y, 11.6, layer_y, color=C["sim"], lw=1.2)
# Output (mean + std for SAC)
node(ax, 12.8, layer_y + 0.35, 2.0, 0.5, "μ head\n3 (vx,vy,ω)", fc=C["sim_light"], ec=C["sim"], tc="#4A148C", r=0.06, fs=7)
node(ax, 12.8, layer_y - 0.35, 2.0, 0.5, "σ head\n3 (log std)", fc=C["sim_light"], ec=C["sim"], tc="#4A148C", r=0.06, fs=7)

# Training note
ax.text(7.5, 1.5, "Trained with Stable-Baselines3 SAC  |  Gaussian policy  |  tanh squashed actions",
        ha="center", fontsize=8, color=C["sim"], style="italic")
ax.text(7.5, 0.9, "Action: (vx, vy) ∈ [−0.3, 0.3] m/s  |  ω ∈ [−90°, 90°] deg/s  |  Saved as models/*.zip",
        ha="center", fontsize=7.5, color=C["muted"])

save(fig, "03_policy_server")

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  FIG 4 — Simulation & Training Pipeline                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝
fig, ax = new_fig(15, 8.5)

ax.text(7.5, 8.15, "Simulation & Training Pipeline",
        ha="center", fontsize=16, fontweight="bold", color="#1A237E")
ax.text(7.5, 7.72, "Gymnasium 2D env → SAC training → evaluation → sim-to-real gap analysis",
        ha="center", fontsize=9, color=C["muted"], style="italic")

# Main pipeline
pipeline = [
    (1.5, 6.5, "Map Generator\ngenerate_random_maps.py", "Random obstacle\nconfigurations\n\nWall + cylinder\nscenarios"),
    (4.3, 6.5, "2D Navigation Env\nsimple_2d_env.py", "Gymnasium API\n\nobs: 64-D scan + goal\naction: (vx, vy, ω)\nreward: distance + collision"),
    (7.5, 6.5, "SAC Training\ntrain_sac.py", "Stable-Baselines3\n\n500k steps default\nablation: 32/64/128 bins\nGPU accelerated"),
    (10.7, 6.5, "Evaluation\nevaluate_policy.py", "100 episodes\n\nMetrics:\n Success rate\n Collision rate\n Timeout rate"),
    (13.3, 6.5, "Sim-to-Real Gap\ncompute_wasserstein.py", "Wasserstein distance\nscan distribution\n\nQuantify domain gap\nbefore deployment"),
]
for px, py, ptitle, pdesc in pipeline:
    rbox(ax, px - 1.25, py + 0.2, 2.5, 0.8, C["sim_light"], C["sim"], lw=1.2, r=0.08, z=4)
    ax.text(px, py + 0.6, ptitle, ha="center", fontsize=7, fontweight="bold", color="#4A148C")
    rbox(ax, px - 1.25, py - 1.25, 2.5, 1.25, C["white"], C["border"], lw=0.8, r=0.06, z=3)
    ax.text(px, py - 0.63, pdesc, ha="center", fontsize=6.2, color=C["dark"])

for i in range(len(pipeline) - 1):
    arrow(ax, pipeline[i][0] + 1.3, 6.5, pipeline[i+1][0] - 1.3, 6.5, color=C["sim"], lw=1.2)

# --- Training Detail (bottom) ---
group(ax, 0.4, 0.3, 14.2, 3.0, C["sim_light"], C["sim"], alpha=0.06)
ax.text(7.5, 3.05, "Training Loop Detail (SAC)",
        ha="center", fontsize=10, fontweight="bold", color=C["sim"])

# RL loop
loop_items = [
    (2.5, 2.0, "Collect\nexperience"),
    (5.5, 2.0, "Update\nQ-networks"),
    (8.5, 2.0, "Update\nPolicy (MLP)"),
    (11.5, 2.0, "Update\ntarget nets"),
]
for lx, ly, ll in loop_items:
    node(ax, lx, ly, 2.2, 0.9, ll, fc=C["sim_light"], ec=C["sim"], tc="#4A148C", r=0.08)

# Cycle arrows
arrow(ax, 3.6, 2.0, 4.4, 2.0, color=C["sim"], lw=1.0)
arrow(ax, 6.6, 2.0, 7.4, 2.0, color=C["sim"], lw=1.0)
arrow(ax, 9.6, 2.0, 10.4, 2.0, color=C["sim"], lw=1.0)
# Cycle back
arrow(ax, 12.6, 1.55, 12.6, 0.85, color=C["sim"], lw=1.0, rad=0)
arrow(ax, 12.6, 0.8, 1.4, 0.8, color=C["sim"], lw=1.0, rad=0)
arrow(ax, 1.4, 0.8, 1.4, 1.55, color=C["sim"], lw=1.0, rad=0)
albl(ax, 7.5, 0.5, "Repeat for N steps (default: 500k)", C["sim"], fs=8)

# Reward formula
ax.text(7.5, 3.2, r"$r(s, a) = w_{\mathrm{dist}} \cdot \Delta d_{\mathrm{goal}} + w_{\mathrm{col}} \cdot \mathbf{1}_{\mathrm{collision}} + w_{\mathrm{time}} \cdot \mathbf{1}_{\mathrm{timeout}}$",
        ha="center", fontsize=8.5, color=C["sim"], style="italic")

save(fig, "04_simulation_training")

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  FIG 5 — Data Flow & Communication Protocol                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝
fig, ax = new_fig(14, 8.5)

ax.text(7, 8.15, "Data Flow & Communication Protocol",
        ha="center", fontsize=16, fontweight="bold", color="#1A237E")
ax.text(7, 7.72, "ZMQ PUB/SUB message formats, bandwidth, and latency breakdown",
        ha="center", fontsize=9, color=C["muted"], style="italic")

# --- Upstream: Scan ---
group(ax, 0.4, 3.8, 6.2, 3.6, C["pi_light"], C["pi"], alpha=0.06)
ax.text(3.5, 7.15, "Upstream — Pseudo-LiDAR Scan", ha="center", fontsize=11, fontweight="bold", color=C["scan"])

# Topic format
scan_detail = (
    "ZMQ Topic:  \"scan\"\n"
    "Format:     float32[64]\n"
    "Frequency:  20 Hz (50 ms period)\n"
    "Size:       64 × 4 = 256 bytes/msg\n"
    "Bandwidth:  5.12 KB/s (payload)\n"
    "            ~10 KB/s (with overhead)\n"
    "Bottleneck: Depth-to-Scan (~2 ms)"
)
ax.text(3.5, 5.6, scan_detail, fontsize=7.5, color=C["dark"], va="center", ha="left",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.6", fc="white", ec=C["scan"], lw=1.0, alpha=0.9))

# Scan visualization
scan_vis_y = 4.2
rbox(ax, 1.0, scan_vis_y, 5.0, 0.6, C["white"], C["scan"], lw=1.0, r=0.06, z=3)
for si in range(64):
    bar_h = 0.15 + 0.35 * np.random.default_rng(si).random()
    rbox(ax, 1.1 + si * 0.075, scan_vis_y + 0.1, 0.06, bar_h, C["scan"], "none", lw=0, r=0.01, z=4, alpha=0.7)
ax.text(3.5, scan_vis_y + 0.5, "64-D scan vector (example)", ha="center", fontsize=6.5, color=C["scan"], style="italic")

# --- Downstream: Command ---
group(ax, 7.4, 3.8, 6.2, 3.6, C["pc_light"], C["pc"], alpha=0.06)
ax.text(10.5, 7.15, "Downstream — Velocity Command", ha="center", fontsize=11, fontweight="bold", color=C["cmd"])

cmd_detail = (
    "ZMQ Topic:  \"cmd\"\n"
    "Format:     float32[3]  →  (vx, vy, ω)\n"
    "Frequency:  20 Hz (on-demand)\n"
    "Size:       3 × 4 = 12 bytes/msg\n"
    "Bandwidth:  240 B/s (payload)\n"
    "            < 1 KB/s (with overhead)\n"
    "Ranges:     |vx,vy| ≤ 0.3 m/s\n"
    "            |ω| ≤ 90 deg/s"
)
ax.text(10.5, 5.5, cmd_detail, fontsize=7.5, color=C["dark"], va="center", ha="left",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.6", fc="white", ec=C["cmd"], lw=1.0, alpha=0.9))

# --- Latency Breakdown (bottom) ---
group(ax, 0.4, 0.2, 13.2, 3.3, C["tool_light"], C["tool"], alpha=0.06)
ax.text(7.0, 3.30, "End-to-End Latency Budget (target: 50 ms = 20 Hz)",
        ha="center", fontsize=10, fontweight="bold", color=C["tool"])

latency_stages = [
    (1.5, 2.2, "Camera\nCapture", "~5 ms", C["hw"]),
    (3.8, 2.2, "Depth→\nScan", "~2 ms", C["pi"]),
    (6.1, 2.2, "ZMQ\nTX", "~1 ms", C["scan"]),
    (8.4, 2.2, "Policy\nInference", "~3 ms\n(GPU)", C["pc"]),
    (10.7, 2.2, "ZMQ\nTX back", "~1 ms", C["cmd"]),
    (13.0, 2.2, "Motor\nCmd", "~2 ms", C["hw"]),
]
for lx, ly, lname, lval, lc in latency_stages:
    node(ax, lx, ly, 2.0, 1.0, lname, fc=C["white"], ec=lc, tc=lc, fw="bold", r=0.08)
    ax.text(lx, ly - 0.5, lval, ha="center", fontsize=8, fontweight="bold", color=lc)

for i in range(len(latency_stages) - 1):
    arrow(ax, latency_stages[i][0] + 1.0, 2.2, latency_stages[i+1][0] - 1.0, 2.2, color=C["tool"], lw=1.0)

# Total
ax.text(7.0, 0.85, "Total: 5 + 2 + 1 + 3 + 1 + 2 ≈ 14 ms  <  50 ms period  ✓",
        ha="center", fontsize=9, fontweight="bold", color="#1B5E20",
        bbox=dict(boxstyle="round,pad=0.4", fc="#E8F5E9", ec="#4CAF50", lw=1.0))

save(fig, "05_data_flow")

print("\nAll 5 architecture diagrams generated in:", OUT)
