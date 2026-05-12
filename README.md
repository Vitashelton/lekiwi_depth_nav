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
    policy_server.py      # ZMQ server + policy inference loop
    run_policy_server.py  # PC launcher
  sim/
    simple_2d_env.py      # Gymnasium 2D navigation env
    train_sac.py          # Train with Stable-Baselines3 SAC
    evaluate_policy.py    # Evaluate success/collision/timeout
    generate_random_maps.py
  tools/
    record_scan_log.py    # Save scan data (.npz / .csv)
    replay_scan_log.py    # Replay scan log through policy
    latency_test.py       # Bench depth-to-scan, ZMQ, policy
    bandwidth_test.py     # Compare raw depth vs scan bandwidth
    plot_scan.py          # Visualize scans (linear/polar/heatmap)
    compute_wasserstein.py # Sim-to-real Wasserstein distance
  logs/
  models/
```

## Installation

```bash
cd lekiwi_depth_nav

# Core dependencies
pip install numpy pyzmq pyyaml

# For PC training & inference
pip install torch stable-baselines3 gymnasium

# For Raspberry Pi camera (optional, only on Pi)
pip install pyrealsense2

# For visualization
pip install matplotlib scipy

# Or install all at once:
pip install -r requirements.txt
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

## License

MIT
