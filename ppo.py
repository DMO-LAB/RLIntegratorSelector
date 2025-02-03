import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class PPOConfig:
    """Configuration for PPO algorithm"""
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    target_kl: float = 0.01
    n_epochs: int = 3
    batch_size: int = 32
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    buffer_size: int = 300

class MemoryBuffer:
    """Memory buffer that grows dynamically without a fixed size limit."""
    def __init__(self):
        self._arrays: Dict[str, List[np.ndarray]] = {
            'states': [],
            'actions': [],
            'log_probs': [],
            'values': [],
            'rewards': [],
            'dones': []
        }

    def store(self, batch: Dict[str, np.ndarray]) -> None:
        """
        Store a batch of transitions in lists. 
        Ensures all data is properly shaped before storing.
        """
        for key, data in batch.items():
            # Convert to numpy array if not already
            data = np.asarray(data)
            # Ensure data is at least 1D
            if data.ndim == 0:
                # Scalar value - reshape to 1D array with length 1
                data = np.array([data])
            elif data.ndim == 1:
                # For 1D arrays, add a new axis if they're singular
                if len(data) == 1:
                    data = data.reshape(1, -1)
            self._arrays[key].append(data)

    def get_all(self) -> Optional[Tuple[np.ndarray, ...]]:
        """
        Get all stored transitions as concatenated NumPy arrays.
        Returns None if no data is stored.
        """
        if len(self._arrays['states']) == 0:
            return None
        
        try:
            return tuple(
                np.concatenate(self._arrays[key], axis=0)
                for key in self._arrays
            )
        except ValueError as e:
            # Debug information
            shapes = {key: [arr.shape for arr in self._arrays[key]] 
                     for key in self._arrays}
            print(f"Error concatenating arrays. Shapes: {shapes}")
            raise e

    def clear(self) -> None:
        """Reset buffer: clear all lists."""
        for key in self._arrays:
            self._arrays[key] = []


class SharedNetwork(nn.Module):
    """Shared feature extractor for policy and value networks"""
    def __init__(self, obs_dim: int, hidden_dims: List[int]):
        super().__init__()
        self.net = nn.Sequential()
        curr_dim = obs_dim
        
        for i, dim in enumerate(hidden_dims):
            self.net.add_module(f'layer{i}', nn.Linear(curr_dim, dim))
            self.net.add_module(f'relu{i}', nn.ReLU())
            curr_dim = dim
            
        self.output_dim = curr_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class PPOAgent:
    """Memory-efficient PPO implementation"""
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_dims: List[int],
        config: PPOConfig,
        device: str = "cpu"
    ):
        self.config = config
        self.device = torch.device(device)
        
        # Networks
        self.shared = SharedNetwork(obs_dim, hidden_dims).to(device)
        self.policy_head = nn.Linear(self.shared.output_dim, n_actions).to(device)
        self.value_head = nn.Linear(self.shared.output_dim, 1).to(device)
        
        # Optimizers
        self.optimizer = torch.optim.Adam(
            list(self.shared.parameters()) + 
            list(self.policy_head.parameters()) + 
            list(self.value_head.parameters()),
            lr=config.lr
        )
        
        # Memory buffer
        self.memory = MemoryBuffer()
        
    def _to_tensor(self, array: np.ndarray) -> torch.Tensor:
        """Convert numpy array to torch tensor on correct device"""
        return torch.as_tensor(array, device=self.device)
    
    @torch.no_grad()
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Select action using current policy"""
        state_tensor = self._to_tensor(state)
        
        # Get action distribution
        shared_features = self.shared(state_tensor)
        action_logits = self.policy_head(shared_features)
        probs = F.softmax(action_logits, dim=-1)
        dist = Categorical(probs)
        
        # Select action
        if deterministic:
            action = probs.argmax(dim=-1)
        else:
            action = dist.sample()
            
        # Get value and log prob
        value = self.value_head(shared_features)
        log_prob = dist.log_prob(action)
        
        return (
            action.cpu().numpy(),
            log_prob.cpu().numpy(),
            value.cpu().numpy().squeeze()
        )
    
    def store_transition(self, state: np.ndarray, action: np.ndarray, 
                        log_prob: np.ndarray, value: np.ndarray,
                        reward: np.ndarray, done: bool) -> None:
        """Store transition in memory buffer"""
        self.memory.store({
            'states': np.array(state, dtype=np.float32),
            'actions': np.array(action, dtype=np.int64),
            'log_probs': np.array(log_prob, dtype=np.float32),
            'values': np.array(value, dtype=np.float32),
            'rewards': np.array(reward, dtype=np.float32),
            'dones': np.array(done, dtype=bool)
        })

    def _compute_advantages(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        next_value: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute advantages using GAE"""
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)
        running_gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value_t = next_value
            else:
                next_non_terminal = 1.0 - dones[t+1]
                next_value_t = values[t+1]
                
            delta = rewards[t] + self.config.gamma * next_value_t * next_non_terminal - values[t]
            running_gae = delta + self.config.gamma * self.config.gae_lambda * next_non_terminal * running_gae
            advantages[t] = running_gae
            
        returns = advantages + values
        return advantages, returns

    def update(self) -> Dict[str, float]:
        """Update policy and value networks"""
        data = self.memory.get_all()
        if data is None:
            return {}
            
        states, actions, old_log_probs, values, rewards, dones = data
        
        # Compute advantages and returns
        advantages, returns = self._compute_advantages(rewards, values, dones, values[-1])
        
        # Convert to tensors
        states = self._to_tensor(states)
        actions = self._to_tensor(actions)
        old_log_probs = self._to_tensor(old_log_probs)
        advantages = self._to_tensor(advantages)
        returns = self._to_tensor(returns)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        metrics = {
            'policy_loss': 0.,
            'value_loss': 0.,
            'entropy': 0.,
            'kl': 0.
        }
        n_updates = 0
        
        # Mini-batch updates
        batch_size = self.config.batch_size
        indices = np.arange(len(states))
        
        for _ in range(self.config.n_epochs):
            np.random.shuffle(indices)
            
            for start_idx in range(0, len(states), batch_size):
                # Get mini-batch
                idx = indices[start_idx:start_idx + batch_size]
                mb_states = states[idx]
                mb_actions = actions[idx]
                mb_old_log_probs = old_log_probs[idx]
                mb_advantages = advantages[idx]
                mb_returns = returns[idx]
                
                # Forward pass
                shared_features = self.shared(mb_states)
                action_logits = self.policy_head(shared_features)
                values_pred = self.value_head(shared_features).squeeze()
                probs = F.softmax(action_logits, dim=-1)
                dist = Categorical(probs)
                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()
                
                # Early stopping with KL divergence
                kl = (mb_old_log_probs.exp() * (mb_old_log_probs - new_log_probs)).mean()
                if kl > self.config.target_kl:
                    break
                
                # Calculate losses
                ratio = (new_log_probs - mb_old_log_probs).exp()
                surrogate1 = ratio * mb_advantages
                surrogate2 = torch.clamp(ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio) * mb_advantages
                policy_loss = -torch.min(surrogate1, surrogate2).mean()
                value_loss = F.mse_loss(values_pred, mb_returns)
                
                # Combined loss
                loss = (
                    policy_loss + 
                    self.config.value_coef * value_loss - 
                    self.config.entropy_coef * entropy
                )
                
                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.shared.parameters()) + 
                    list(self.policy_head.parameters()) + 
                    list(self.value_head.parameters()),
                    self.config.max_grad_norm
                )
                self.optimizer.step()
                
                # Update metrics
                metrics['policy_loss'] += policy_loss.item()
                metrics['value_loss'] += value_loss.item()
                metrics['entropy'] += entropy.item()
                metrics['kl'] += kl.item()
                n_updates += 1
                
                # Clean up tensors
                del mb_states, mb_actions, mb_old_log_probs, mb_advantages, mb_returns
                del shared_features, action_logits, values_pred, probs, dist
                del new_log_probs, entropy, ratio, surrogate1, surrogate2
                del policy_loss, value_loss, loss
        
        # Clear memory and cache
        self.memory.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Average metrics
        print(f"Update complete with metrics: {metrics}")
        return {k: v/n_updates if n_updates > 0 else 0 for k, v in metrics.items()}

    def save(self, path: str) -> None:
        """Save model state"""
        torch.save({
            'shared_state': self.shared.state_dict(),
            'policy_head_state': self.policy_head.state_dict(),
            'value_head_state': self.value_head.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
        }, path)
        
    def load(self, path: str) -> None:
        """Load model state"""
        checkpoint = torch.load(path)
        self.shared.load_state_dict(checkpoint['shared_state'])
        self.policy_head.load_state_dict(checkpoint['policy_head_state'])
        self.value_head.load_state_dict(checkpoint['value_head_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])