# Humanoid Walking using Deep Reinforcement Learning

A complete pipeline for training a humanoid robot to walk using Deep Reinforcement Learning (PPO), with pose initialization from human images.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![PyBullet](https://img.shields.io/badge/PyBullet-3.0+-green.svg)

---
## Walking Demo of Trained model
![Walking Demo](trained_walking_policy.gif)

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Training](#training)
- [Methodology](#methodology)
- [Results](#results)

---

## 🎯 Overview

This project implements an end-to-end pipeline for humanoid locomotion using:

1. **Pose Extraction**: Extract human pose from images using MediaPipe
2. **Joint Angle Mapping**: Convert 2D keypoints to robot joint angles
3. **Physics Simulation**: PyBullet-based humanoid environment
4. **RL Training**: PPO algorithm for learning walking behavior
5. **Policy Evaluation**: Visualize and evaluate trained policies

### Key Features
- MediaPipe-based pose detection from images
- Geometric angle calculation (no inverse kinematics solver needed)
- PPO (Proximal Policy Optimization) for stable training
- Pre-trained baseline model included for comparison
- Interactive PyBullet visualization

---

## 🏗️ Architecture

### System Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT PIPELINE                           │
├─────────────────────────────────────────────────────────────┤
│  Image → MediaPipe → 33 Keypoints → Geometric Calc → θ_init │
│                                         (12 joint angles)   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                  SIMULATION ENVIRONMENT                     │
├─────────────────────────────────────────────────────────────┤
│  HumanoidBulletEnv                                          │
│  • Observation Space: 44 dimensions                         │
│  • Action Space: 17 continuous actions [-1, 1]              │
│  • Joints: Torso (3) + Legs (8) + Arms (6)                 │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    RL AGENT (PPO)                           │
├─────────────────────────────────────────────────────────────┤
│  Actor Network:  obs(44) → 256 → 256 → actions(17)         │
│  Critic Network: obs(44) → 256 → 256 → value(1)            │
│  Activation: ReLU                                           │
│  Output: Tanh (bounded actions)                             │
└─────────────────────────────────────────────────────────────┘
```

### Neural Network Architecture

```
Actor-Critic Network (PPO)
═══════════════════════════════════════════════════════

ACTOR (Policy Network):
┌────────────┐    ┌────────────┐    ┌────────────┐    ┌────────────┐
│ Input (44) │───→│ Dense(256) │───→│ Dense(256) │───→│ Output(17) │
│            │    │   + ReLU   │    │   + ReLU   │    │   + Tanh   │
└────────────┘    └────────────┘    └────────────┘    └────────────┘

CRITIC (Value Network):
┌────────────┐    ┌────────────┐    ┌────────────┐    ┌────────────┐
│ Input (44) │───→│ Dense(256) │───→│ Dense(256) │───→│ Output (1) │
│            │    │   + ReLU   │    │   + ReLU   │    │  (Value)   │
└────────────┘    └────────────┘    └────────────┘    └────────────┘

Trainable log_std parameter for action distribution
```

### Humanoid Joint Structure (17 DOF)

The **training and walking** use PyBullet's built-in humanoid with 17 joints:

| Body Part | Joints | DOF |
|-----------|--------|-----|
| Torso | abdomen_z, abdomen_y, abdomen_x | 3 |
| Right Leg | right_hip_x, right_hip_z, right_hip_y, right_knee | 4 |
| Left Leg | left_hip_x, left_hip_z, left_hip_y, left_knee | 4 |
| Right Arm | right_shoulder1, right_shoulder2, right_elbow | 3 |
| Left Arm | left_shoulder1, left_shoulder2, left_elbow | 3 |

### Two Humanoid Environments

This project uses **two different PyBullet humanoid environments**:

| | `HumanoidEnv` | `HumanoidBulletEnv` |
|---|---|---|
| **Used by** | `demo.py` | `train.py`, `run_policy.py` |
| **URDF** | `humanoid/humanoid.urdf` | `humanoid_symmetric.xml` |
| **Joints** | 12 | 17 |
| **Observations** | 37 dims | 44 dims |
| **Purpose** | Pose visualization | RL training & walking |
| **Source** | Custom wrapper | pybullet_envs (gym) |

**Why two environments?**
- `HumanoidEnv`: Simple environment for visualizing poses from images
- `HumanoidBulletEnv`: Full gym environment with built-in reward function, optimized for RL training

**Pose Library Training:**
When training with `--pose-library`, the 12 joint angles extracted from images are **mapped to 17 joints** of `HumanoidBulletEnv`. This allows the model to learn walking from varied initial poses.

> **Note:** The baseline model was trained on `HumanoidBulletEnv`, so it only works with `run_policy.py`, not `demo.py`.

---

## 🔧 Installation

### Prerequisites
- Python 3.8+
- CUDA (optional, for GPU training)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd RLISL

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
```
gymnasium
pybullet
pybullet_envs
mediapipe
stable-baselines3
torch
opencv-python
numpy
matplotlib
tensorboard
```

---

## 📁 Project Structure

```
RLISL/
├── demo.py                          # Pose extraction demo
├── train.py                         # Training script
├── run_policy.py                    # Policy evaluation
│
├── humanoid_walk/                   # Main package
│   ├── __init__.py
│   │
│   ├── perception/                  # Pose extraction module
│   │   ├── __init__.py
│   │   └── pose_converter.py        # MediaPipe → Joint angles
│   │
│   ├── env/                         # Environment module
│   │   ├── __init__.py
│   │   ├── humanoid_env.py          # Custom Gymnasium env
│   │   └── humanoid_bullet_wrapper.py  # Joint mapping wrapper
│   │
│   ├── rl/                          # RL module
│   │   ├── __init__.py
│   │   ├── ppo_agent.py             # PPO implementation
│   │   └── buffers.py               # Rollout buffer
│   │
│   └── utils/                       # Utilities
│       ├── __init__.py
│       └── model_loader.py          # Model loading utilities
│
├── models/                          # Trained models
│   ├── trained_walking_policy.zip   # Baseline trained model
│   ├── trained_normalizer.pkl       # Observation normalizer
│   └── efficientdet_lite2.tflite    # Person detector
│
├── images/                          # Test images
├── logs/                            # Training logs
└── requirements.txt
```

---

## 🚀 Usage

### 1. Run Walking Policy

```bash
# Use your trained model (default)
python run_policy.py

# Use baseline model (for comparison)
python run_policy.py --baseline

# Test with random actions
python run_policy.py --random
```

### 2. Pose Detection & Humanoid Pose Copying (Without Walking)

If you just want to **extract pose from image** and **copy it to humanoid** (without walking):

```bash
# Extract pose and display humanoid in that pose
python demo.py --image images/your_image.jpg
```

**What this does:**
1. Detects person in the image
2. Extracts 33 MediaPipe landmarks
3. Calculates 12 joint angles geometrically
4. Displays humanoid in PyBullet matching the pose
5. Interactive controls (P = toggle picking, F = freeze, Q = quit)

**Use cases:**
- Pose visualization research
- Testing pose extraction accuracy
- Demonstrating image-to-robot pose transfer
- No trained model required!

### 3. Train New Model

**Option A: Standard Training (from default position)**
```bash
python train.py --use-sb3 --timesteps 500000
```

**Option B: Train with Pose Library (from varied positions)**
```bash
# Step 1: Build pose library from images
python build_pose_library.py --folder images/

# Step 2: Train with the library
python train.py --use-sb3 --pose-library --timesteps 500000
```

### Command Reference

| Command | Description |
|---------|-------------|
| `python run_policy.py` | Run your trained model |
| `python run_policy.py --baseline` | Run baseline model |
| `python run_policy.py --random` | Random actions (testing) |
| `python run_policy.py --image <path>` | Start from image pose + walk |
| `python demo.py --image <path>` | Pose detection only (no walking) |
| `python build_pose_library.py --folder images/` | Build pose library |
| `python train.py --use-sb3` | Train (standard) |
| `python train.py --use-sb3 --pose-library` | Train with pose library |

### 4. Image-Based Pose Initialization

You can initialize the humanoid with a pose extracted from an image:

```bash
# Start from image pose, then walk with baseline model
python run_policy.py --image images/person.jpg --baseline

# Start from image pose, then walk with your trained model
python run_policy.py --image images/person.jpg
```

**How it works:**
1. Pose is extracted from the image using MediaPipe
2. 12 joint angles are calculated geometrically
3. Angles are mapped to the humanoid's 17 DOF
4. Humanoid is set to that initial pose
5. Walking policy takes over

**⚠️ Important Limitation:**

The baseline model was trained from a **default standing position**. When starting from an unusual pose (e.g., mid-stride from an image), the model may struggle to recover and walk properly. This is because:

- The training didn't include varied starting positions
- The policy hasn't learned to recover from arbitrary poses

**Results when using image pose:**
- Default standing → High reward (~80-100), stable walking
- Image pose → May fall immediately or take time to stabilize

**Future Work:** To improve performance from arbitrary poses, the model should be trained with:
- Randomized initial positions
- Domain randomization
- Recovery behaviors

## 🎓 Training

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Algorithm | PPO (Proximal Policy Optimization) |
| Learning Rate | 3e-4 |
| Batch Size | 64 |
| Rollout Steps | 2048 |
| Epochs per Update | 10 |
| Discount Factor (γ) | 0.99 |
| GAE Lambda | 0.95 |
| Clip Range | 0.2 |
| Entropy Coefficient | 0.01 |
| Value Coefficient | 0.5 |

### Reward Function

```python
reward = r_alive + r_velocity + r_energy + r_stability

where:
  r_alive     = +2.0 (per step, for not falling)
  r_velocity  = +2.0 × forward_velocity (encourage movement)
  r_energy    = -0.00005 × energy (penalize excessive force)
  r_stability = -0.5 × orientation_deviation (penalize tilting)
```

### Training Command

```bash
# Full training (recommended: 500K-1M steps)
python train.py --use-sb3 --timesteps 500000

# Monitor with TensorBoard
tensorboard --logdir logs/tensorboard
```

### Advanced: Training from Varied Initial Positions

To train a model that can walk from **any starting pose** (including poses from images), you need to modify the environment to randomize initial positions during training.

**Step 1:** Create a custom environment wrapper with random initial poses:

```python
# In humanoid_walk/env/randomized_env.py
import gymnasium as gym
import numpy as np
from pybullet_envs.gym_locomotion_envs import HumanoidBulletEnv

class RandomizedHumanoidEnv(gym.Wrapper):
    """Humanoid environment with randomized initial positions."""
    
    def __init__(self, render=False):
        env = HumanoidBulletEnv(render=render)
        super().__init__(env)
    
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        
        # Apply random joint perturbations
        if hasattr(self.env, 'robot'):
            for joint in self.env.robot.ordered_joints:
                # Random angle between -0.3 and 0.3 radians
                random_angle = np.random.uniform(-0.3, 0.3)
                joint.reset_position(random_angle, 0)
        
        # Step to apply the randomization
        obs, _, _, _ = self.env.step(np.zeros(self.action_space.shape))
        return obs
```

**Step 2:** Modify `train.py` to use the randomized environment:

```python
# Replace HumanoidBulletEnv with RandomizedHumanoidEnv
from humanoid_walk.env.randomized_env import RandomizedHumanoidEnv

def make_env():
    return RandomizedHumanoidEnv(render=False)
```

**Step 3:** Train with more timesteps for better generalization:

```bash
python train.py --use-sb3 --timesteps 1000000
```

**Expected Results:**
- Model learns to recover from various poses
- Better performance when starting from image-extracted poses
- More robust walking behavior

---

## 📐 Methodology

### 1. Pose Extraction Pipeline

```
Image → Person Detection → Pose Estimation → Angle Calculation
         (EfficientDet)     (MediaPipe)      (Geometric)
```

**MediaPipe Landmarks Used:**
- Shoulders (LEFT_SHOULDER, RIGHT_SHOULDER)
- Elbows (LEFT_ELBOW, RIGHT_ELBOW)
- Wrists (LEFT_WRIST, RIGHT_WRIST)
- Hips (LEFT_HIP, RIGHT_HIP)
- Knees (LEFT_KNEE, RIGHT_KNEE)
- Ankles (LEFT_ANKLE, RIGHT_ANKLE)

**Angle Calculation:**
- Hip angles: Relative to torso vertical axis
- Shoulder angles: Relative to torso horizontal axis (T-pose reference)
- Knee/Elbow angles: Relative to parent limb segment

### 2. Joint Mapping (12 → 17)

Our pose extraction produces 12 joint angles, mapped to the 17 DOF humanoid:

| Our Joint | → | Humanoid Joint |
|-----------|---|----------------|
| chest | → | abdomen_y |
| r_shoulder | → | right_shoulder1 |
| r_elbow | → | right_elbow |
| l_shoulder | → | left_shoulder1 |
| l_elbow | → | left_elbow |
| r_hip | → | right_hip_y |
| r_knee | → | right_knee |
| l_hip | → | left_hip_y |
| l_knee | → | left_knee |

### 3. PPO Algorithm

PPO uses clipped surrogate objective for stable policy updates:

```
L^CLIP(θ) = E[min(r_t(θ)A_t, clip(r_t(θ), 1-ε, 1+ε)A_t)]

where:
  r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)  (probability ratio)
  A_t = advantage estimate (using GAE)
  ε = clip range (0.2)
```

---

## 📊 Results

### Training Performance

| Metric | Baseline Model |
|--------|----------------|
| Average Episode Reward | ~80-100 |
| Average Episode Steps | 60-80 |
| Training Time | ~2 hours (500K steps) |

### Model Files

| File | Description |
|------|-------------|
| `trained_walking_policy.zip` | Trained PPO policy weights |
| `trained_normalizer.pkl` | Observation normalization statistics |

---

## 🎮 Controls

When running the simulation:

| Key | Action |
|-----|--------|
| Close Window | Quit simulation |
| (Auto) | Episode auto-resets on fall |

---

## 📚 References

- [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347)
- [PyBullet Physics Engine](https://pybullet.org/)
- [MediaPipe Pose](https://google.github.io/mediapipe/solutions/pose.html)
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)

---

## 👤 Author

Developed as part of Reinforcement Learning coursework.

---

## 📄 License

This project is for educational purposes.
