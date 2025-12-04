"""
Model Loading Utility for Humanoid Walk Project

Supports:
1. Custom PPO (trained from scratch) - PyTorch
2. Stable-Baselines3 pre-trained models (.zip)
"""

import os
import pickle
import numpy as np

# Stable-Baselines3
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("⚠️ Stable-Baselines3 not installed. Run: pip install stable-baselines3")

# PyTorch
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class PPOActor(nn.Module):
    """Custom PPO Actor Network (PyTorch) - For training from scratch"""
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))
    
    def forward(self, obs):
        mean = self.net(obs)
        std = self.log_std.exp()
        return mean, std
    
    def get_action(self, obs, deterministic=False):
        with torch.no_grad():
            if isinstance(obs, np.ndarray):
                obs = torch.FloatTensor(obs)
            mean, std = self.forward(obs)
            if deterministic:
                return mean.numpy()
            else:
                dist = torch.distributions.Normal(mean, std)
                return dist.sample().numpy()


def load_vec_normalize(pkl_path):
    """Load VecNormalize stats from pickle file."""
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            return pickle.load(f)
    return None


def load_sb3_pretrained(model_path, vec_norm_path=None, env=None, expected_obs_dim=None):
    """
    Load SB3 pre-trained model with optional normalization.
    
    Args:
        model_path: Path to .zip model file
        vec_norm_path: Path to vec_normalize.pkl (optional but recommended)
        env: Environment instance
        expected_obs_dim: Expected observation dimension (for compatibility check)
    
    Returns:
        Tuple (model, vec_normalize or None)
    """
    if not SB3_AVAILABLE:
        raise RuntimeError("Stable-Baselines3 not installed!")
    
    print(f"🔄 Loading SB3 model from: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return None, None
    
    # Load model
    model = PPO.load(model_path)
    print("✅ SB3 model loaded")
    
    # Check observation dimension compatibility
    model_obs_dim = model.observation_space.shape[0]
    if expected_obs_dim is not None and model_obs_dim != expected_obs_dim:
        print(f"⚠️ WARNING: Model expects obs_dim={model_obs_dim}, but env has {expected_obs_dim}")
        print(f"   This model is NOT directly compatible with your environment.")
        print(f"   Use --random flag or train your own model.")
        return None, None
    
    # Load normalization stats if available
    vec_normalize = None
    if vec_norm_path and os.path.exists(vec_norm_path):
        print(f"🔄 Loading normalization from: {vec_norm_path}")
        vec_normalize = load_vec_normalize(vec_norm_path)
        print("✅ VecNormalize loaded")
    elif vec_norm_path:
        print(f"⚠️ vec_normalize.pkl not found - using raw observations")
    
    return model, vec_normalize


def normalize_obs(obs, vec_normalize):
    """Normalize observation using VecNormalize stats."""
    if vec_normalize is None:
        return obs
    
    try:
        # Apply normalization: (obs - mean) / std
        obs_rms = vec_normalize.obs_rms
        
        # Check dimension match
        if obs.shape != obs_rms.mean.shape:
            # Dimension mismatch - skip normalization
            return obs
        
        return (obs - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8)
    except Exception:
        return obs


def load_custom_ppo(model_path, obs_dim, action_dim):
    """Load custom PyTorch PPO model."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not installed!")
    
    print(f"🔄 Loading custom PPO from: {model_path}")
    
    actor = PPOActor(obs_dim, action_dim)
    
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location='cpu')
        actor.load_state_dict(state_dict)
        print("✅ Custom PPO loaded")
    else:
        print(f"⚠️ Model not found. Using random weights.")
    
    actor.eval()
    return actor


def load_model(
    custom_ppo_path="models/ppo_policy.pth",
    pretrained_path="models/ppo-HumanoidBulletEnv-v0.zip",
    vec_norm_path="models/vec_normalize.pkl",
    env=None,
    obs_dim=37,
    action_dim=12
):
    """
    Load model with priority:
    1. Custom trained PPO (.pth)
    2. SB3 pre-trained fallback (.zip)
    3. Random initialization (last resort)
    
    Returns:
        Tuple (model, model_type, get_action_fn, vec_normalize)
    """
    print("\n🔍 Searching for available models...")
    
    # Priority 1: Custom trained PPO
    if os.path.exists(custom_ppo_path):
        model = load_custom_ppo(custom_ppo_path, obs_dim, action_dim)
        get_action = lambda obs: model.get_action(obs, deterministic=True)
        print(f"📌 Using: Custom PPO ({custom_ppo_path})")
        return model, 'custom_ppo', get_action, None
    
    # Priority 2: SB3 pre-trained fallback
    if SB3_AVAILABLE and os.path.exists(pretrained_path):
        model, vec_norm = load_sb3_pretrained(pretrained_path, vec_norm_path, env)
        if model is not None:
            def get_action(obs):
                norm_obs = normalize_obs(obs, vec_norm)
                action, _ = model.predict(norm_obs, deterministic=True)
                return action
            print(f"📌 Using: Pre-trained SB3 ({pretrained_path})")
            return model, 'pretrained_sb3', get_action, vec_norm
    
    # Priority 3: Random initialization
    print("⚠️ No trained model found. Using random initialization.")
    model = PPOActor(obs_dim, action_dim) if TORCH_AVAILABLE else None
    get_action = lambda obs: model.get_action(obs) if model else np.zeros(action_dim)
    return model, 'random', get_action, None


# ============================================================================
# TEST
# ============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print(" MODEL LOADING TEST")
    print("=" * 60)
    
    model, model_type, get_action, vec_norm = load_model()
    
    print(f"\n📊 Result:")
    print(f"  Model type: {model_type}")
    print(f"  VecNormalize: {'Loaded' if vec_norm else 'Not used'}")
    
    # Test action
    dummy_obs = np.zeros(37, dtype=np.float32)
    action = get_action(dummy_obs)
    print(f"  Test action: {action[:3]}... (shape: {np.array(action).shape})")
