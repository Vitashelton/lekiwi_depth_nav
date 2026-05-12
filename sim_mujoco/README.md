# LeKiwi MuJoCo Simulation — Paper-Grade Environment

This is the **primary simulation environment** for the LeKiwi Depth Navigation
paper. It replaces the toy `sim/simple_2d_env.py` with a full MuJoCo-based
simulation featuring:

- Realistic LeKiwi 3-omniwheel chassis with velocity-actuator dynamics
- D435i-mounted pseudo-LiDAR ray-cast sensor with configurable noise
- 5 structured lab environments (empty, obstacle, corridor, narrow gap, cluttered)
- Action smoothing, velocity low-pass filter, and spin/stall prevention
- Gymnasium-compatible API for direct SAC/RL training
- Full residual safety correction pipeline

**Old `sim/simple_2d_env.py`** is kept as a toy baseline only — do NOT use it
for paper experiments.

## Directory Structure

```
sim_mujoco/
  README.md
  assets/                          # MuJoCo XML models
    lekiwi_base.xml                # LeKiwi robot with 3 omniwheels
    d435i_mount.xml                # RealSense D435i camera mount
    table.xml                      # Lab table (1.2m × 0.6m)
    chair.xml                      # Lab chair (0.45m × 0.45m)
    cardboard_box.xml              # Cardboard box (0.4m × 0.3m)
  worlds/                          # Pre-built lab environments
    lab_empty.xml                  # 5m × 6m empty room
    lab_single_obstacle.xml        # One box between start and goal
    lab_corridor.xml               # 1.5m wide corridor with clutter
    lab_narrow_gap.xml             # Two tables forming a 0.65m gap
    lab_cluttered.xml              # Desks, chairs, boxes, pillar
  envs/
    lekiwi_mujoco_env.py           # Low-level MuJoCo wrapper
    lekiwi_depth_scan_env.py       # Gymnasium env with obs/reward/termination
  sensors/
    ray_scan_sensor.py             # MuJoCo ray-cast pseudo-LiDAR
    scan_noise.py                  # Configurable noise models
  controllers/
    omni_kinematics.py             # Inverse kinematics (vx,vy,ω → wheel speeds)
    velocity_controller.py         # Acceleration-limited low-pass filter
  wrappers/
    action_smoothing.py            # EMA + median action smoothing
    domain_randomization.py        # Obstacle/friction/noise randomization
  train/
    train_sac_mujoco.py            # Train SAC with Stable-Baselines3
    train_residual_safety.py       # Train residual correction on MuJoCo data
  eval/
    evaluate_mujoco.py             # Evaluate SAC on all 5 worlds
    evaluate_residual.py           # Compare raw vs residual-corrected
    sim2real_scan_gap.py           # Wasserstein distance: sim vs real scans
  tools/
    render_episode.py              # Render episode to MP4 video
    record_mujoco_dataset.py       # Record scan/action data to NPZ
    plot_mujoco_trajectory.py      # 5-panel trajectory visualization
  configs/
    lekiwi_mujoco.yaml             # Main simulation config
    train_sac.yaml                 # SAC training hyperparameters
    train_residual.yaml            # Residual training hyperparameters
```

## Quick Start

### 1. Install MuJoCo

```bash
conda activate lekiwi_rl
pip install mujoco
```

### 2. Test the Environment

```bash
python -c "
import sys; sys.path.insert(0, '.')
from sim_mujoco.envs.lekiwi_depth_scan_env import make_env

env = make_env('lab_empty.xml')
obs, _ = env.reset()
print(f'Observation shape: {obs.shape}')   # (72,)
print(f'Action space: {env.action_space}')

# Take a few steps
for _ in range(10):
    obs, reward, term, trunc, info = env.step([0.2, 0.0, 0.0])
print(f'Goal distance: {info[\"dist_to_goal\"]:.2f}m')
env.close()
"
```

### 3. Render an Episode

```bash
# Record a DWA-controlled episode to video
python sim_mujoco/tools/render_episode.py --world lab_cluttered.xml --policy dwa --output episode.mp4
```

### 4. Record Dataset

```bash
# Record scan/action data for offline analysis
python sim_mujoco/tools/record_mujoco_dataset.py --world lab_corridor.xml --episodes 10
```

## Training

### Train SAC Navigation Policy

```bash
# Train on empty room (500k steps)
python sim_mujoco/train/train_sac_mujoco.py --world lab_empty.xml --timesteps 500000

# Train on cluttered lab with domain randomization
python sim_mujoco/train/train_sac_mujoco.py --world lab_cluttered.xml --timesteps 1000000

# Resume from checkpoint
python sim_mujoco/train/train_sac_mujoco.py --world lab_empty.xml --resume models/sac_mujoco_500000_steps.zip
```

Training logs go to `logs/tensorboard/sac_mujoco/`. View with:
```bash
tensorboard --logdir logs/tensorboard/
```

### Train Residual Safety Correction

```bash
# Step 1: Generate training dataset (DWA teacher)
python sim_mujoco/train/train_residual_safety.py \
    --episodes 200 --raw-policy rule --output models/residual_correction_mujoco.pt

# Step 2: The trained model is directly loadable by pc/policy_server.py
python pc/run_policy_server.py --config config/ \
    --mode residual_correction \
    --residual-model models/residual_correction_mujoco.pt
```

## Evaluation

### Evaluate Trained SAC Policy

```bash
# Test on all 5 worlds
python sim_mujoco/eval/evaluate_mujoco.py \
    --model models/sac_mujoco_500000_steps.zip \
    --episodes 20

# Render first episode of each world
python sim_mujoco/eval/evaluate_mujoco.py \
    --model models/sac_mujoco_500000_steps.zip \
    --episodes 5 --render
```

### Compare Residual Correction Methods

```bash
# Compare 4 action modes
python sim_mujoco/eval/evaluate_residual.py \
    --raw-policy rule \
    --residual-model models/residual_correction_mujoco.pt \
    --episodes 20
```

### Sim-to-Real Scan Gap

```bash
# Record real scan log first:
python tools/record_scan_log.py --sim --duration 30 --output logs/sim_scans.npz

# Compare MuJoCo vs real scan distributions
python sim_mujoco/eval/sim2real_scan_gap.py \
    --world lab_cluttered.xml \
    --log logs/sim_scans.npz \
    --output-plot sim2real_gap.png
```

## Key Design Decisions

### Anti-Spin / Anti-Stall

The environment explicitly penalizes:
- **Spin**: angular velocity > 30 deg/s for 5+ consecutive steps
- **Stagnation**: displacement < 1cm for 10+ consecutive steps

This prevents the "原地抽搐" (spinning in place) behavior common in naive SAC-trained policies.

### Velocity Dynamics

Actions pass through:
1. Acceleration clamping (`max_linear_accel=0.5 m/s²`)
2. First-order low-pass filter (`alpha=0.3`)

This models real motor dynamics and eliminates the sim-to-real gap from
instantaneous velocity changes.

### Scan Compatibility

The `RayScanSensor.get_scan()` output format is **identical** to
`raspberry_pi/depth_to_scan.py`:
- `scan_m`: (N,) float32, metric ranges in [min_range, max_range]
- `scan_norm`: (N,) float32, normalized ranges in [0, 1]

Models trained in MuJoCo can be deployed on the real LeKiwi without
modification.

### Domain Randomization

Optional per-episode randomization:
- Obstacle position jitter: ±10cm
- Friction coefficient: [0.4, 0.9]
- Mass scaling: [0.8, 1.2]
- Scan noise scaling

Enable with:
```bash
# In code:
env = make_env('lab_cluttered.xml', apply_dr=True)
```

## Configuration

All parameters are in `configs/lekiwi_mujoco.yaml`. Override via code:

```python
from sim_mujoco.envs.lekiwi_depth_scan_env import EnvConfig
config = EnvConfig(
    world_xml='lab_cluttered.xml',
    scan_bins=128,
    max_episode_steps=1000,
)
env = LeKiwiDepthScanEnv(config=config)
```

## Connection to Real LeKiwi Stack

```
MuJoCo Sim                          Real Hardware
──────────                          ─────────────
ray_scan_sensor.get_scan()   ←──→  depth_to_scan.__call__()
  → scan_norm, scan_m                → scan_norm, scan_m

LeKiwiDepthScanEnv.step()     ←──→  PolicyServer.run()
  action = [vx, vy, ω]               action = [vx, vy, ω]

residual_correction.pt        ←──→  residual_correction.pt
  (same checkpoint format)           (loaded by policy_server.py)
```
