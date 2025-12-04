"""
Rollout Buffer for PPO Training

Stores trajectories collected during policy rollout.
"""

import numpy as np
import torch


class RolloutBuffer:
    """Buffer to store rollout data for PPO updates."""
    
    def __init__(self, buffer_size, obs_dim, action_dim):
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.reset()
    
    def reset(self):
        """Clear the buffer."""
        self.obs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.log_probs = []
        self.pos = 0
    
    def add(self, obs, action, reward, done, value, log_prob):
        """Add a transition to the buffer."""
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.pos += 1
    
    def is_full(self):
        return self.pos >= self.buffer_size
    
    def get(self):
        """Return all data as numpy arrays."""
        return {
            'obs': np.array(self.obs),
            'actions': np.array(self.actions),
            'rewards': np.array(self.rewards),
            'dones': np.array(self.dones),
            'values': np.array(self.values),
            'log_probs': np.array(self.log_probs),
        }
    
    def compute_returns_and_advantages(self, last_value, gamma=0.99, gae_lambda=0.95):
        """Compute returns and GAE advantages."""
        rewards = np.array(self.rewards)
        values = np.array(self.values)
        dones = np.array(self.dones)
        
        # Compute GAE
        advantages = np.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + gamma * gae_lambda * (1 - dones[t]) * last_gae
        
        returns = advantages + values
        
        return {
            'obs': np.array(self.obs),
            'actions': np.array(self.actions),
            'log_probs': np.array(self.log_probs),
            'advantages': torch.FloatTensor(advantages),
            'returns': torch.FloatTensor(returns),
        }
