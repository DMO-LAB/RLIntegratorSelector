import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

@dataclass
class SACConfig:
    """Configuration for discrete SAC algorithm"""
    lr: float = 3e-4
    alpha_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    buffer_size: int = 1000000
    batch_size: int = 256
    init_temperature: float = 1.0
    min_temperature: float = 0.1
    target_entropy: Optional[float] = None
    hidden_dims: List[int] = (256, 256)
    
    
class MemoryBuffer:
    """Efficient memory buffer using pre-allocated numpy arrays"""
    def __init__(self, buffer_size: int):
        self.buffer_size = buffer_size
        self.ptr = 0
        self.size = 0
        self._arrays: Dict[str, Optional[np.ndarray]] = {
            'states': None,
            'actions': None,
            'rewards': None,
            'next_states': None,  # Added for SAC
            'dones': None
        }

    def _initialize_arrays(self, sample_batch: Dict[str, np.ndarray]) -> None:
        """Initialize arrays with correct shapes from sample batch"""
        for key, sample in sample_batch.items():
            shape = sample.shape[1:] if len(sample.shape) > 1 else ()
            dtype = sample.dtype
            self._arrays[key] = np.zeros((self.buffer_size,) + shape, dtype=dtype)
            
    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Sample random batch of transitions"""
        indices = np.random.randint(0, self.size, size=batch_size)
        return {k: self._arrays[k][indices] for k in self._arrays.keys()}

    def store(self, batch: Dict[str, np.ndarray]) -> None:
        """Store a batch of transitions"""
        if self._arrays['states'] is None:
            self._initialize_arrays(batch)

        batch_size = len(next(iter(batch.values())))
        if self.ptr + batch_size > self.buffer_size:
            # Handle wrapping
            first_part = self.buffer_size - self.ptr
            for key, data in batch.items():
                self._arrays[key][self.ptr:] = data[:first_part]
                self._arrays[key][:batch_size-first_part] = data[first_part:]
            self.ptr = batch_size - first_part
        else:
            # Normal storage
            for key, data in batch.items():
                self._arrays[key][self.ptr:self.ptr+batch_size] = data
            self.ptr = (self.ptr + batch_size) % self.buffer_size

        self.size = min(self.size + batch_size, self.buffer_size)

    def get_all(self) -> Optional[Tuple[np.ndarray, ...]]:
        """Get all stored transitions"""
        if self.size == 0:
            return None
        return tuple(arr[:self.size] for arr in self._arrays.values())

    def clear(self) -> None:
        """Reset buffer pointers"""
        self.ptr = 0
        self.size = 0

class DiscretePolicy(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden_dims: List[int]):
        super().__init__()
        self.net = nn.Sequential()
        curr_dim = obs_dim
        
        for i, dim in enumerate(hidden_dims):
            self.net.add_module(f'layer{i}', nn.Linear(curr_dim, dim))
            self.net.add_module(f'relu{i}', nn.ReLU())
            curr_dim = dim
            
        self.logits = nn.Linear(curr_dim, n_actions)
        
    def forward(self, x: torch.Tensor, temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.net(x)
        logits = self.logits(features)
        
        # Gumbel-Softmax with straight-through estimator
        action_probs = F.gumbel_softmax(logits, tau=temperature, hard=True)
        
        # Calculate log probabilities
        log_probs = F.log_softmax(logits, dim=-1)
        return action_probs, log_probs

class DoubleQNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden_dims: List[int]):
        super().__init__()
        self.q1 = self._build_network(obs_dim, n_actions, hidden_dims)
        self.q2 = self._build_network(obs_dim, n_actions, hidden_dims)
        
    def _build_network(self, obs_dim: int, n_actions: int, hidden_dims: List[int]) -> nn.Sequential:
        layers = []
        curr_dim = obs_dim
        
        for dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, dim))
            layers.append(nn.ReLU())
            curr_dim = dim
            
        layers.append(nn.Linear(curr_dim, n_actions))
        return nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.q1(state), self.q2(state)

class DiscreteSAC:
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        config: SACConfig,
        device: str = "cpu"
    ):
        self.config = config
        self.device = torch.device(device)
        self.n_actions = n_actions
        
        # Initialize networks
        self.policy = DiscretePolicy(obs_dim, n_actions, config.hidden_dims).to(device)
        self.q_net = DoubleQNetwork(obs_dim, n_actions, config.hidden_dims).to(device)
        self.target_q_net = DoubleQNetwork(obs_dim, n_actions, config.hidden_dims).to(device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        
        # Initialize optimizers
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=config.lr)
        self.q_optimizer = torch.optim.Adam(self.q_net.parameters(), lr=config.lr)
        
        # Initialize temperature parameter
        self.log_alpha = torch.tensor(np.log(config.init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config.alpha_lr)
        
        # Set target entropy if not provided
        if config.target_entropy is None:
            self.target_entropy = -0.98 * np.log(1/n_actions)  # Heuristic value
        else:
            self.target_entropy = config.target_entropy
            
        # Initialize replay buffer
        self.memory = MemoryBuffer(config.buffer_size)
        
    @property
    def alpha(self):
        return torch.clamp(self.log_alpha.exp(), min=self.config.min_temperature)
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            if deterministic:
                _, logits = self.policy(state, temperature=0.1)
                action = torch.argmax(logits, dim=-1)
            else:
                action_probs, _ = self.policy(state, temperature=self.alpha.item())
                action = torch.argmax(action_probs, dim=-1)
            return action.cpu().numpy()
    
    def store_transition(self, state: np.ndarray, action: np.ndarray, 
                        reward: float, next_state: np.ndarray, done: bool):
        self.memory.store({
            'states': state.astype(np.float32),
            'actions': action.astype(np.int64),
            'rewards': np.array(reward, dtype=np.float32),
            'next_states': next_state.astype(np.float32),
            'dones': np.array(done, dtype=bool)
        })
    
    def update(self) -> Dict[str, float]:
        if self.memory.size < self.config.batch_size:
            return {}
            
        # Sample batch
        batch = self.memory.sample(self.config.batch_size)
        states = torch.FloatTensor(batch['states']).to(self.device)
        actions = torch.LongTensor(batch['actions']).to(self.device)
        rewards = torch.FloatTensor(batch['rewards']).to(self.device)
        next_states = torch.FloatTensor(batch['next_states']).to(self.device)
        dones = torch.FloatTensor(batch['dones']).to(self.device)
        
        # Update Q networks
        with torch.no_grad():
            next_action_probs, next_log_probs = self.policy(next_states, self.alpha.item())
            next_q1, next_q2 = self.target_q_net(next_states)
            next_q = torch.min(next_q1, next_q2)
            # Soft Q-learning backup
            next_value = (next_action_probs * (next_q - self.alpha * next_log_probs)).sum(dim=-1)
            target_q = rewards + (1 - dones) * self.config.gamma * next_value
            
        current_q1, current_q2 = self.q_net(states)
        q1_loss = F.mse_loss(current_q1.gather(1, actions.unsqueeze(-1)).squeeze(), target_q)
        q2_loss = F.mse_loss(current_q2.gather(1, actions.unsqueeze(-1)).squeeze(), target_q)
        q_loss = q1_loss + q2_loss
        
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        
        # Update policy
        action_probs, log_probs = self.policy(states, self.alpha.item())
        q1, q2 = self.q_net(states)
        q = torch.min(q1, q2)
        
        inside_term = self.alpha * log_probs - q
        policy_loss = (action_probs * inside_term).sum(dim=-1).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # Update temperature
        alpha_loss = -(self.log_alpha * 
                      (log_probs.detach() + self.target_entropy).mean())
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        # Update target networks
        for param, target_param in zip(self.q_net.parameters(), self.target_q_net.parameters()):
            target_param.data.copy_(
                self.config.tau * param.data + (1 - self.config.tau) * target_param.data
            )
            
        return {
            'q_loss': q_loss.item(),
            'policy_loss': policy_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'alpha': self.alpha.item(),
            'entropy': -log_probs.mean().item()
        }
        
    def save(self, path: str) -> None:
        torch.save({
            'policy_state': self.policy.state_dict(),
            'q_net_state': self.q_net.state_dict(),
            'target_q_net_state': self.target_q_net.state_dict(),
            'policy_optimizer_state': self.policy_optimizer.state_dict(),
            'q_optimizer_state': self.q_optimizer.state_dict(),
            'alpha_optimizer_state': self.alpha_optimizer.state_dict(),
            'log_alpha': self.log_alpha
        }, path)
        
    def load(self, path: str) -> None:
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy_state'])
        self.q_net.load_state_dict(checkpoint['q_net_state'])
        self.target_q_net.load_state_dict(checkpoint['target_q_net_state'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state'])
        self.q_optimizer.load_state_dict(checkpoint['q_optimizer_state'])
        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state'])
        self.log_alpha = checkpoint['log_alpha']