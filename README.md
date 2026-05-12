# LeKiwi Depth Navigation Framework

A complete engineering framework for omnidirectional mobile robot navigation using
pseudo-LiDAR scans and learned policies, targeting the **LeKiwi** three-omniwheel robot.

## Architecture

```
  ┌─────────────────────────────────────────────────────────┐
  │                   Raspberry Pi (Edge)                    │
  │                                                          │
  │  RealSense D435i ──► Depth-to-Scan ──► ZMQ PUB (scan)   │
  │                                                          │
  │  Motor Drivers ◄── Base Controller ◄── ZMQ SUB (cmd)    │
  └──────────────────────┬──────────────────────────────────┘
                         │ Wi-Fi (64-D scan, <10 KB/s)
  ┌──────────────────────▼──────────────────────────────────┐
  │                    PC (RTX 5060)                          │
  │                                                          │
  │  ZMQ SUB (scan) ──► Policy Server ──► ZMQ PUB (cmd)     │
  │                         │                                │
  │                   ┌──────┴──────┐                        │
  │                   │   Policy     │                       │
  │                   │ rule / DWA / │                       │
  │                   │ MLP (SAC)   │                        │
  │                   └─────────────┘                        │
  └─────────────────────────────────────────────────────────┘
```

## Project Structure

```
lekiwi_depth_nav/
  README.md
  requirements.txt
  config/
    robot.yaml            # Robot physical params & limits
    camera.yaml           # RealSense D435i settings
    network.yaml          # ZMQ IPs and ports
    policy.yaml           # Policy type & parameters
  raspberry_pi/
    camera_node.py        # RealSense or mock depth camera
    depth_to_scan.py      # Depth image → pseudo-LiDAR scan
    scan_publisher.py     # ZMQ PUB: scan → PC
    command_subscriber.py # ZMQ SUB: velocity cmd ← PC
    base_controller.py    # LeKiwi kinematics & motor control
    run_pi_stack.py       # Main Pi loop (launcher)
  pc/
    rule_policy.py        # Rule-based obstacle avoidance
    dwa_policy.py         # DWA local planner
    mlp_policy.py         # PyTorch MLP policy (SAC-trained)
    geometric_risk.py     # Continuous geometric risk functions
    residual_correction.py # Learned residual correction MLP
    policy_server.py      # ZMQ server + policy inference loop (3 modes)
    run_policy_server.py  # PC launcher
  sim/
    simple_2d_env.py      # Gymnasium 2D navigation env
    train_sac.py          # Train with Stable-Baselines3 SAC
    evaluate_policy.py    # Evaluate success/collision/timeout
    generate_random_maps.py
  train/
    train_residual_correction.py # Train residual correction model
  tools/
    visualize_sim.py      # Real-time simulation visualizer
    live_scan_viewer.py   # Live ZMQ scan viewer (mock/real)
    generate_residual_dataset.py # Create residual training data
    evaluate_correction.py # Compare 4 correction methods + charts
    record_scan_log.py    # Save scan data (.npz / .csv)
    replay_scan_log.py    # Replay scan log through policy
    latency_test.py       # Bench depth-to-scan, ZMQ, policy
    bandwidth_test.py     # Compare raw depth vs scan bandwidth
    plot_scan.py          # Visualize scans (linear/polar/heatmap)
    compute_wasserstein.py # Sim-to-real Wasserstein distance
  dashboard/
    app.py                # Streamlit scan log explorer
  logs/
  models/
  datasets/
```

## Installation

```bash
# Create conda environment (recommended)
conda create -n lekiwi_rl python=3.10 -y
conda activate lekiwi_rl

cd lekiwi_depth_nav

# Install all dependencies at once
pip install -r requirements.txt

# For Raspberry Pi camera (optional, only on Pi)
pip install pyrealsense2
```

## Quick Start (PC only, no hardware)

### 1. Test Depth-to-Scan

```bash
python -c "
from raspberry_pi.camera_node import MockCamera, CameraConfig
from raspberry_pi.depth_to_scan import DepthToScan, ScanConfig

cam = MockCamera(CameraConfig())
d2s = DepthToScan(ScanConfig())
depth = cam.get_depth_frame()
scan_norm, scan_m = d2s(depth)
print(f'Scan: {scan_m}')
"
```

### 2. Run Latency Benchmark

```bash
python tools/latency_test.py
```

### 3. Run Bandwidth Test

```bash
python tools/bandwidth_test.py
```

### 4. Train a Policy in Simulation

```bash
# Train with default settings (64-D scan, 500k steps)
python sim/train_sac.py --config config/

# Ablation: 32-D scan
python sim/train_sac.py --config config/ --scan-bins 32

# Ablation: 128-D scan
python sim/train_sac.py --config config/ --scan-bins 128
```

### 5. Evaluate Policies

```bash
# Evaluate rule-based policy
python sim/evaluate_policy.py --config config/ --rule-policy --episodes 100

# Evaluate DWA policy
python sim/evaluate_policy.py --config config/ --dwa-policy --episodes 100

# Compare scan dimensions
python sim/evaluate_policy.py --config config/ --rule-policy --scan-bins 32
python sim/evaluate_policy.py --config config/ --rule-policy --scan-bins 64
python sim/evaluate_policy.py --config config/ --rule-policy --scan-bins 128
```

### 6. Visualize a Scan

```bash
# Quick mock camera demo
python tools/plot_scan.py

# Live visualization from simulation
python tools/plot_scan.py --live --duration 10
```

## Running on Real Hardware

### Step 1: Configure Network

Edit `config/network.yaml`:
```yaml
network:
  pi_known_ip: "192.168.1.100"   # Raspberry Pi IP
  pc_known_ip: "192.168.1.200"   # PC IP
```

### Step 2: Start Raspberry Pi Stack

```bash
# On the Raspberry Pi:
cd lekiwi_depth_nav
python raspberry_pi/run_pi_stack.py --config config/
```

The Pi will:
- Start reading RealSense D435i depth frames
- Convert to 64-D pseudo-LiDAR scans
- Publish scans to PC via ZMQ
- Listen for velocity commands

### Step 3: Start PC Policy Server

```bash
# On the PC (with RTX 5060):
cd lekiwi_depth_nav
python pc/run_policy_server.py --config config/ --freq 20
```

### Step 4: Testing with Mock Hardware (PC only)

```bash
# Terminal 1: Simulated Pi stack (mock camera)
python raspberry_pi/run_pi_stack.py --config config/ --mock-camera

# Terminal 2: PC policy server
python pc/run_policy_server.py --config config/ --freq 20
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `scan.num_bins` | 64 | Pseudo-LiDAR resolution (32/64/128) |
| `scan.band_top` | 220 | Top row of depth band |
| `scan.band_bottom` | 260 | Bottom row of depth band |
| `scan.fov_deg` | 90.0 | Horizontal field of view |
| `scan.min_range` | 0.15 m | Minimum valid range |
| `scan.max_range` | 5.0 m | Maximum valid range |
| `robot.max_linear_vel` | 0.3 m/s | Max forward/lateral speed |
| `robot.max_angular_vel` | 90.0 deg/s | Max rotational speed |
| `robot.wheel_radius` | 0.05 m | Omniwheel radius |
| `robot.base_radius` | 0.125 m | Center-to-wheel distance |

## Kinematics Notes

The LeKiwi uses Feetech STS3215 servos in velocity mode. Wheel speeds are sent as
raw integers (±3000 range) following the lerobot convention:

- Wheel mounting angles: [240°, 0°, 120°] with -90° offset → [150°, -90°, 30°]
- Kinematic matrix: M[i] = [cos(αᵢ), sin(αᵢ), base_radius]
- omega is in **deg/s** (not rad/s)

For mock mode (no real motors), the BaseController silently stores commands.
For real hardware, inject a FeetechMotorsBus:
```python
from lerobot.motors.feetech import FeetechMotorsBus
controller.set_motors_bus(bus)
```

## Visualization

### Simulation Visualization

Real-time rendering of the 2D navigation environment with robot, obstacles,
goal, scan rays, and live metrics.

```bash
# Rule-based policy with scan rays visible
python tools/visualize_sim.py --policy rule --episodes 3 --show-rays

# DWA policy
python tools/visualize_sim.py --policy dwa --scan-bins 64

# Random policy for baseline
python tools/visualize_sim.py --policy random --episodes 5

# Save rendered frames to video
python tools/visualize_sim.py --policy rule --show-rays --save-video demo.mp4
```

### Live Scan Viewer

Real-time ZeroMQ scan subscriber with three live views.

```bash
# Mock mode (no hardware required)
python tools/live_scan_viewer.py --mock

# From a real Pi stack (reads config/network.yaml)
python tools/live_scan_viewer.py --config config/
```

Shows: linear scan curve, polar scan view, telemetry stats (FPS, seq, net delay),
and min-range history over time.

### Dashboard

Interactive Streamlit dashboard for exploring recorded scan logs (.npz files).

```bash
# Generate sample log data first (so dashboard has content)
python tools/record_scan_log.py --sim --duration 10 --output logs/sample_scans.npz

# Launch the dashboard
streamlit run dashboard/app.py
```

## Residual Policy Correction

Geometry-aware safety correction for LeRobot / learned policies.
A lightweight MLP predicts a **bounded residual** Δa that adjusts
candidate actions into safer navigation commands.

```
candidate_action (LeRobot)  ──►  ResidualCorrectionNet  ──►  final_action
                                     ↑
  scan_m + velocity + goal_heading ──┘
```

### 1. Generate Residual Dataset

Uses geometric action projection (sampling-based optimization) to
compute `safer_action` labels from a mock candidate policy.

```bash
# Small test run (5 episodes)
python tools/generate_residual_dataset.py --sim --episodes 5 --output datasets/residual_dataset.npz

# Full dataset (200+ episodes recommended)
python tools/generate_residual_dataset.py --sim --episodes 200 --output datasets/residual_dataset.npz
```

### 2. Train Residual Correction Network

```bash
# Train on generated dataset
python train/train_residual_correction.py --dataset datasets/residual_dataset.npz --epochs 50

# With GPU
python train/train_residual_correction.py --dataset datasets/residual_dataset.npz --epochs 100 --device cuda
```

### 3. Evaluate & Visualize All Methods

Compares four action sources side-by-side with **interactive matplotlib charts**:
bar chart, radar chart, trajectory overlay, and risk-over-time curves.

```bash
# Text + charts (default)
python tools/evaluate_correction.py --episodes 30

# With trained residual model
python tools/evaluate_correction.py --episodes 50 --residual-model models/residual_correction.pt

# Text-only (headless / no GUI)
python tools/evaluate_correction.py --episodes 30 --no-plot
```

Output visualization includes:
- **Bar chart**: success / collision / timeout rates per method
- **Radar chart**: normalized safety profile (higher = better)
- **Trajectory overlay**: robot paths on the same map, color-coded by method
- **Risk curve**: per-step collision risk for each method

### 4. Deploy with Policy Server

```bash
# Raw candidate policy (no correction)
python pc/run_policy_server.py --config config/ --mode lerobot_raw

# Rule-based safety shield
python pc/run_policy_server.py --config config/ --mode rule_shield

# Learned residual correction
python pc/run_policy_server.py --config config/ --mode residual_correction --residual-model models/residual_correction.pt
```

## Logging & Analysis

```bash
# Record scan log from live stream
python tools/record_scan_log.py --config config/ --duration 30 --output logs/scan_001.npz

# Record from simulation
python tools/record_scan_log.py --sim --duration 30 --output logs/sim_scans.npz

# Plot a single scan
python tools/plot_scan.py --input logs/scan_001.npz --index 50

# Plot scan heatmap over time
python tools/plot_scan.py --input logs/scan_001.npz --heatmap

# Replay scan through policy
python tools/replay_scan_log.py --input logs/scan_001.npz --policy rule

# Compute sim-to-real Wasserstein distance
python tools/compute_wasserstein.py --sim logs/sim_scans.npz --real logs/scan_001.npz --plot
```

使用

  conda activate lekiwi_rl

  # 测试环境
  python -c "from sim_mujoco.envs.lekiwi_depth_scan_env import make_env; ..."

  # 渲染视频
  python sim_mujoco/tools/render_episode.py --world lab_cluttered.xml --policy dwa --output demo.mp4

  # 训练 SAC
  python sim_mujoco/train/train_sac_mujoco.py --world lab_empty.xml --timesteps 500000

  # 训练残差安全模型
  python sim_mujoco/train/train_residual_safety.py --episodes 200 --raw-policy rule

  # 评估残差修正
  python sim_mujoco/eval/evaluate_residual.py --raw-policy rule --residual-model models/residual_correction_mujoco.pt

  # Sim-to-Real 扫描差距
  python sim_mujoco/eval/sim2real_scan_gap.py --world lab_cluttered.xml --log logs/sim_scans.npz

## License

MIT
