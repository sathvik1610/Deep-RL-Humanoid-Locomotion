"""
Training Script for Humanoid Walking

Usage:
    python train.py --use-sb3                          # Standard training (default position)
    python train.py --use-sb3 --pose-library           # Train with pose library
    python train.py --use-sb3 --pose-library models/my_library.npy  # Custom library
    python train.py --timesteps 1000000                # More training steps
"""

import argparse
import os
import time
import numpy as np
import torch
import pickle

# Import environment and agent
from humanoid_walk.env.humanoid_env import HumanoidEnv
from humanoid_walk.rl.ppo_agent import PPOAgent
from humanoid_walk.rl.buffers import RolloutBuffer

# Try to import SB3 (recommended)
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from stable_baselines3.common.monitor import Monitor
    from pybullet_envs.gym_locomotion_envs import HumanoidBulletEnv
    import gymnasium as gym
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("⚠️ Stable-Baselines3 not installed. Using custom PPO.")


# ============================================================================
# ENVIRONMENT WITH POSE LIBRARY
# ============================================================================
class PoseLibraryEnv(gym.Wrapper):
    """
    HumanoidBulletEnv wrapper that randomizes initial pose from a library.
    """
    def __init__(self, pose_library_path, render=False):
        env = HumanoidBulletEnv(render=render)
        super().__init__(env)
        
        # Load pose library
        self.pose_library = np.load(pose_library_path)
        self.num_poses = len(self.pose_library)
        print(f"📚 Loaded pose library: {self.num_poses} poses")
    
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        
        # Select random pose from library
        pose_idx = np.random.randint(0, self.num_poses)
        pose_12 = self.pose_library[pose_idx]
        
        # Map 12-joint pose to 17-joint actions
        actions_17 = self._map_pose_to_actions(pose_12)
        
        # Apply pose for a few steps
        for _ in range(10):
            obs, _, _, _ = self.env.step(actions_17)
        
        return obs
    
    def _map_pose_to_actions(self, pose_12):
        """Map 12-joint pose to 17-joint action space."""
        actions_17 = np.zeros(17, dtype=np.float32)
        
        # chest → abdomen_y (index 1)
        actions_17[1] = np.clip(pose_12[0] / (np.pi/2), -1, 1)
        # r_shoulder → right_shoulder1 (index 11)
        actions_17[11] = np.clip(pose_12[2] / (np.pi/2), -1, 1)
        # r_elbow → right_elbow (index 13)
        actions_17[13] = np.clip(pose_12[3] / (np.pi/2), -1, 1)
        # l_shoulder → left_shoulder1 (index 14)
        actions_17[14] = np.clip(pose_12[4] / (np.pi/2), -1, 1)
        # l_elbow → left_elbow (index 16)
        actions_17[16] = np.clip(pose_12[5] / (np.pi/2), -1, 1)
        # r_hip → right_hip_y (index 5)
        actions_17[5] = np.clip(pose_12[6] / (np.pi/2), -1, 1)
        # r_knee → right_knee (index 6)
        actions_17[6] = np.clip(pose_12[7] / (np.pi/2), -1, 1)
        # l_hip → left_hip_y (index 9)
        actions_17[9] = np.clip(pose_12[9] / (np.pi/2), -1, 1)
        # l_knee → left_knee (index 10)
        actions_17[10] = np.clip(pose_12[10] / (np.pi/2), -1, 1)
        
        return actions_17


def make_env(render_mode=None):
    """Create standard environment."""
    def _init():
        env = HumanoidEnv(render_mode=render_mode)
        return Monitor(env) if SB3_AVAILABLE else env
    return _init


def make_bullet_env():
    """Create HumanoidBulletEnv for training."""
    def _init():
        env = HumanoidBulletEnv(render=False)
        return Monitor(env)
    return _init


def make_pose_library_env(pose_library_path):
    """Create environment with pose library."""
    def _init():
        env = PoseLibraryEnv(pose_library_path, render=False)
        return Monitor(env)
    return _init


def train_with_sb3(args):
    """Train using Stable-Baselines3 (recommended)."""
    print("\n" + "=" * 60)
    print(" TRAINING WITH STABLE-BASELINES3")
    print("=" * 60)
    
    # Check if using pose library
    if args.pose_library:
        pose_lib_path = args.pose_library if isinstance(args.pose_library, str) and args.pose_library != "True" else "models/pose_library.npy"
        
        if not os.path.exists(pose_lib_path):
            print(f"\n❌ Pose library not found: {pose_lib_path}")
            print("   Build it first with:")
            print("   python build_pose_library.py --folder images/")
            return
        
        print(f"\n📚 Mode: Training with POSE LIBRARY")
        print(f"   Library: {pose_lib_path}")
        env = DummyVecEnv([make_pose_library_env(pose_lib_path)])
        eval_env = DummyVecEnv([make_pose_library_env(pose_lib_path)])
    else:
        print(f"\n🚶 Mode: Standard training (default position)")
        env = DummyVecEnv([make_bullet_env()])
        eval_env = DummyVecEnv([make_bullet_env()])
    
    # Normalize environments
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
    
    # Model save path
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="models/",
        name_prefix="ppo_humanoid"
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="models/",
        log_path="logs/",
        eval_freq=5000,
        deterministic=True,
        render=False
    )
    
    # Create PPO model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log="logs/tensorboard/"
    )
    
    print(f"\n🚀 Starting training for {args.timesteps} timesteps...")
    print(f"   Checkpoints: models/")
    print(f"   TensorBoard: logs/tensorboard/")
    print(f"   Monitor: tensorboard --logdir logs/tensorboard\n")
    
    # Train
    model.learn(
        total_timesteps=args.timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True
    )
    
    # Save final model
    model_name = "ppo_walking_trained" if not args.pose_library else "ppo_walking_poses"
    model.save(f"models/{model_name}")
    env.save(f"models/{model_name}_normalizer.pkl")
    
    print("\n✅ Training complete!")
    print(f"   Model: models/{model_name}.zip")
    print(f"   Normalizer: models/{model_name}_normalizer.pkl")
    
    env.close()


def train_custom_ppo(args):
    """Train using custom PPO implementation."""
    print("\n" + "=" * 60)
    print(" TRAINING WITH CUSTOM PPO")
    print("=" * 60)
    
    # Create environment
    env = HumanoidEnv(render_mode=None)
    
    # Get dimensions
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    print(f"  Observation dim: {obs_dim}")
    print(f"  Action dim: {action_dim}")
    
    # Create agent
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device}")
    
    agent = PPOAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        device=device
    )
    
    # Training parameters
    rollout_steps = 2048
    total_timesteps = args.timesteps
    epochs_per_update = 10
    batch_size = 64
    
    # Create buffer
    buffer = RolloutBuffer(rollout_steps, obs_dim, action_dim)
    
    # Training loop
    os.makedirs("models", exist_ok=True)
    
    obs, _ = env.reset()
    episode_rewards = []
    current_episode_reward = 0
    timestep = 0
    update_count = 0
    
    print(f"\n🚀 Starting training for {total_timesteps} timesteps...\n")
    start_time = time.time()
    
    while timestep < total_timesteps:
        buffer.reset()
        
        # Collect rollout
        for _ in range(rollout_steps):
            # Get action
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                action, log_prob, _, value = agent.network.get_action_and_value(obs_tensor)
            
            action_np = action.cpu().numpy().squeeze()
            
            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated
            
            # Store transition
            buffer.add(
                obs, action_np, reward, done,
                value.item(), log_prob.item()
            )
            
            current_episode_reward += reward
            timestep += 1
            
            if done:
                episode_rewards.append(current_episode_reward)
                current_episode_reward = 0
                obs, _ = env.reset()
            else:
                obs = next_obs
            
            if timestep >= total_timesteps:
                break
        
        # Compute advantages
        with torch.no_grad():
            last_value = agent.network.get_value(
                torch.FloatTensor(obs).unsqueeze(0).to(device)
            ).item()
        
        rollout_data = buffer.compute_returns_and_advantages(last_value)
        
        # Update policy
        loss = agent.update(rollout_data, epochs=epochs_per_update, batch_size=batch_size)
        update_count += 1
        
        # Logging
        if len(episode_rewards) > 0:
            avg_reward = np.mean(episode_rewards[-10:])
            elapsed = time.time() - start_time
            fps = timestep / elapsed
            print(f"  Timestep: {timestep:>8} | Updates: {update_count:>4} | "
                  f"Avg Reward: {avg_reward:>8.2f} | Loss: {loss:.4f} | FPS: {fps:.0f}")
        
        # Save checkpoint
        if update_count % 50 == 0:
            agent.save(f"models/ppo_checkpoint_{update_count}.pth")
    
    # Save final model
    agent.save("models/ppo_policy.pth")
    print(f"\n✅ Training complete! Model saved to models/ppo_policy.pth")
    
    env.close()


def main():
    parser = argparse.ArgumentParser(description="Train Humanoid Walking Policy")
    parser.add_argument("--timesteps", type=int, default=500000, 
                        help="Total training timesteps")
    parser.add_argument("--use-sb3", action="store_true",
                        help="Use Stable-Baselines3 (recommended)")
    parser.add_argument("--pose-library", nargs="?", const="models/pose_library.npy",
                        help="Train with pose library (optionally specify path)")
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print(" HUMANOID WALKING - PPO TRAINING")
    print("=" * 60)
    
    if args.use_sb3:
        if not SB3_AVAILABLE:
            print("❌ Stable-Baselines3 not installed!")
            print("   Run: pip install stable-baselines3")
            return
        train_with_sb3(args)
    else:
        if args.pose_library:
            print("⚠️ Pose library only supported with --use-sb3")
            print("   Run: python train.py --use-sb3 --pose-library")
            return
        train_custom_ppo(args)


if __name__ == "__main__":
    main()
