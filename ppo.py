import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from typing import List, Dict, Tuple
import time
from collections import deque

class SharedNetwork(nn.Module):
    """Shared network for policy and value functions"""
    def __init__(self, obs_dim: int, hidden_dims: List[int]):
        super().__init__()
        
        layers = []
        prev_dim = obs_dim
        
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            prev_dim = dim
            
        self.shared = nn.Sequential(*layers)
        self.obs_dim = obs_dim
        self.output_dim = prev_dim

    def forward(self, x):
        return self.shared(x)

class PolicyNetwork(nn.Module):
    """Policy network outputting action probabilities"""
    def __init__(self, shared_net: SharedNetwork, n_actions: int):
        super().__init__()
        self.shared = shared_net
        self.policy_head = nn.Linear(shared_net.output_dim, n_actions)
    
    def forward(self, x):
        shared_out = self.shared(x)
        action_logits = self.policy_head(shared_out)
        return F.softmax(action_logits, dim=-1)

class ValueNetwork(nn.Module):
    """Value network estimating state values"""
    def __init__(self, shared_net: SharedNetwork):
        super().__init__()
        self.shared = shared_net
        self.value_head = nn.Linear(shared_net.output_dim, 1)
    
    def forward(self, x):
        shared_out = self.shared(x)
        return self.value_head(shared_out)

class PPOMemory:
    """Memory buffer for PPO"""
    def __init__(self, batch_size: int):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size
    
    def store(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)
    
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.probs.clear()
        self.vals.clear()
        self.rewards.clear()
        self.dones.clear()
    
    def get_batches(self) -> List[Tuple]:
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]
        
        return batches

class MultiAgentPPO:
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_dims: List[int] = [64, 64],
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        target_kl: float = 0.01,
        n_epochs: int = 10,
        batch_size: int = 64,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = torch.device(device)
        
        # Networks
        shared_net = SharedNetwork(obs_dim, hidden_dims).to(self.device)
        self.policy = PolicyNetwork(shared_net, n_actions).to(self.device)
        self.value = ValueNetwork(shared_net).to(self.device)
        
        # Optimizers
        self.policy_optim = optim.Adam(self.policy.parameters(), lr=lr)
        self.value_optim = optim.Adam(self.value.parameters(), lr=lr)
        
        # Hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.target_kl = target_kl
        self.n_epochs = n_epochs
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        # Memory
        self.memory = PPOMemory(batch_size)
        
        # Training metrics
        self.metrics = {
            'value_loss': deque(maxlen=100),
            'policy_loss': deque(maxlen=100),
            'entropy': deque(maxlen=100),
            'kl': deque(maxlen=100),
            'clip_fraction': deque(maxlen=100)
        }
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[torch.Tensor, float, float]:
        """Select actions for all agents"""
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            action_probs = self.policy(state)
            dist = Categorical(action_probs)
            if deterministic:
                action = torch.argmax(action_probs, dim=-1)
            else:
                action = dist.sample()
            value = self.value(state)
            
            log_prob = dist.log_prob(action)
            
        return action.cpu().numpy(), log_prob.cpu().numpy(), value.cpu().numpy().squeeze()
    
    def compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool],
        next_value: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Generalized Advantage Estimation"""
        advantages = np.zeros_like(rewards, dtype=np.float32)
        returns = np.zeros_like(rewards, dtype=np.float32)
        running_gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value_t = next_value
            else:
                next_non_terminal = 1.0 - dones[t+1]
                next_value_t = values[t+1]
            
            delta = rewards[t] + self.gamma * next_value_t * next_non_terminal - values[t]
            running_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * running_gae
            advantages[t] = running_gae
            
        returns = advantages + values
        return advantages, returns
    
    def update(self) -> Dict[str, float]:
        # Convert memory to tensors
        states = torch.FloatTensor(np.array(self.memory.states)).to(self.device)
        actions = torch.LongTensor(np.array(self.memory.actions)).to(self.device)
        old_probs = torch.FloatTensor(np.array(self.memory.probs)).to(self.device)
        values = np.array(self.memory.vals)
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(
            self.memory.rewards,
            values,
            self.memory.dones,
            values[-1]
        )
        
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Policy update
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_kl = 0
        n_updates = 0
        
        for _ in range(self.n_epochs):
            for batch_idx in self.memory.get_batches():
                states_batch = states[batch_idx]
                actions_batch = actions[batch_idx]
                advantages_batch = advantages[batch_idx]
                returns_batch = returns[batch_idx]
                old_probs_batch = old_probs[batch_idx]
                
                # Policy loss
                action_probs = self.policy(states_batch)
                dist = Categorical(action_probs)
                new_probs = dist.log_prob(actions_batch)
                entropy = dist.entropy().mean()
                
                # KL divergence
                kl = (old_probs_batch.exp() * (old_probs_batch - new_probs)).mean()
                if kl > self.target_kl:
                    break
                
                # Policy ratio and clipped objective
                ratio = (new_probs - old_probs_batch).exp()
                clip_adv = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * advantages_batch
                policy_loss = -(torch.min(ratio * advantages_batch, clip_adv)).mean()
                
                # Value loss
                value_pred = self.value(states_batch).squeeze()
                value_loss = F.mse_loss(value_pred, returns_batch)
                
                # Combined loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                # Update networks
                self.policy_optim.zero_grad()
                self.value_optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.value.parameters(), self.max_grad_norm)
                self.policy_optim.step()
                self.value_optim.step()
                
                # Track metrics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                total_kl += kl.item()
                n_updates += 1
        
        # Store average metrics
        if n_updates > 0:
            self.metrics['policy_loss'].append(total_policy_loss / n_updates)
            self.metrics['value_loss'].append(total_value_loss / n_updates)
            self.metrics['entropy'].append(total_entropy / n_updates)
            self.metrics['kl'].append(total_kl / n_updates)
        
        # Clear memory
        self.memory.clear()
        
        return {
            'policy_loss': total_policy_loss / n_updates if n_updates > 0 else 0,
            'value_loss': total_value_loss / n_updates if n_updates > 0 else 0,
            'entropy': total_entropy / n_updates if n_updates > 0 else 0,
            'kl': total_kl / n_updates if n_updates > 0 else 0
        }
    
    def save(self, path: str):
        """Save model state"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'value_state_dict': self.value.state_dict(),
            'policy_optimizer_state_dict': self.policy_optim.state_dict(),
            'value_optimizer_state_dict': self.value_optim.state_dict(),
        }, path)
    
    def load(self, path: str):
        """Load model state"""
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.value.load_state_dict(checkpoint['value_state_dict'])
        self.policy_optim.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.value_optim.load_state_dict(checkpoint['value_optimizer_state_dict'])