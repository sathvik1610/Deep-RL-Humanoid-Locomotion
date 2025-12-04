"""
PPO Agent Implementation for Humanoid Walking

Based on the project spec:
- Actor: obs_dim → 256 → 256 → action_dim
- Critic: obs_dim → 256 → 256 → 1
- Trainable log_std parameter
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


class ActorCritic(nn.Module):
    """Combined Actor-Critic Network for PPO."""
    
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()
        
        # Actor network (policy)
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        
        # Trainable log standard deviation
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Critic network (value function)
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
    
    def forward(self, obs):
        return self.get_value(obs)
    
    def get_action_and_value(self, obs, action=None):
        """Get action, log_prob, entropy, and value."""
        mean = self.actor(obs)
        std = self.log_std.exp()
        dist = Normal(mean, std)
        
        if action is None:
            action = dist.sample()
        
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(obs).squeeze(-1)
        
        return action, log_prob, entropy, value
    
    def get_value(self, obs):
        return self.critic(obs).squeeze(-1)
    
    def get_action(self, obs, deterministic=False):
        """Get action for inference."""
        with torch.no_grad():
            mean = self.actor(obs)
            if deterministic:
                return mean
            std = self.log_std.exp()
            dist = Normal(mean, std)
            return dist.sample()


class PPOAgent:
    """PPO Agent with training loop."""
    
    def __init__(
        self,
        obs_dim,
        action_dim,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        device='cpu'
    ):
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        # Create actor-critic network
        self.network = ActorCritic(obs_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
    
    def get_action(self, obs, deterministic=False):
        """Get action from policy."""
        if isinstance(obs, np.ndarray):
            obs = torch.FloatTensor(obs).to(self.device)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        
        action = self.network.get_action(obs, deterministic)
        return action.cpu().numpy().squeeze()
    
    def compute_gae(self, rewards, values, dones, next_value):
        """Compute Generalized Advantage Estimation."""
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns = advantages + torch.tensor(values, dtype=torch.float32, device=self.device)
        
        return advantages, returns
    
    def update(self, rollout_buffer, epochs=10, batch_size=64):
        """Update policy using PPO."""
        obs = torch.FloatTensor(rollout_buffer['obs']).to(self.device)
        actions = torch.FloatTensor(rollout_buffer['actions']).to(self.device)
        old_log_probs = torch.FloatTensor(rollout_buffer['log_probs']).to(self.device)
        advantages = rollout_buffer['advantages'].to(self.device)
        returns = rollout_buffer['returns'].to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Training loop
        total_loss = 0
        num_updates = 0
        
        for _ in range(epochs):
            # Mini-batch updates
            indices = np.random.permutation(len(obs))
            
            for start in range(0, len(obs), batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]
                
                batch_obs = obs[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Get current values
                _, log_probs, entropy, values = self.network.get_action_and_value(
                    batch_obs, batch_actions
                )
                
                # Policy loss (PPO clipped objective)
                ratio = (log_probs - batch_old_log_probs).exp()
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = 0.5 * ((values - batch_returns) ** 2).mean()
                
                # Entropy bonus
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_loss += loss.item()
                num_updates += 1
        
        return total_loss / num_updates
    
    def save(self, path):
        """Save model weights."""
        torch.save({
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)
        print(f"✅ Model saved to {path}")
    
    def load(self, path):
        """Load model weights."""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"✅ Model loaded from {path}")
