from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Union
import torch.nn as nn
import yaml
import os
from datetime import datetime
from environment import SimulationConfig

@dataclass
class NetworkConfig:
    """Neural network architecture configuration"""
    pi_layers: List[int] = None
    vf_layers: List[int] = None
    activation_fn: str = "Tanh"
    
    def __post_init__(self):
        self.pi_layers = self.pi_layers or [256, 128, 64]
        self.vf_layers = self.vf_layers or [256, 128, 64]

@dataclass
class PPOConfig:
    """PPO algorithm configuration"""
    learning_rate: float = 1e-3
    n_steps: int = 1000
    batch_size: int = 50000
    n_epochs: int = 8
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    clip_range_vf: Optional[float] = None
    ent_coef: float = 0.05
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    total_timesteps: int = 50_000_000

@dataclass
class FeatureConfig:
    """Feature configuration for the environment"""
    local_features: bool = True
    neighbor_features: bool = False
    gradient_features: bool = True
    temporal_features: bool = False
    window_size: int = 5

@dataclass
class RewardConfig:
    """Reward configuration for the environment"""
    weights: Dict[str, float] = None
    thresholds: Dict[str, float] = None
    scaling: Dict[str, float] = None
    use_neighbors: bool = True
    neighbor_weight: float = 0.3
    neighbor_radius: int = 4
    
    def __post_init__(self):
        self.weights = self.weights or {
            'accuracy': 1,
            'efficiency': 3,
        }
        self.thresholds = self.thresholds or {
            'time': 0.001,
            'error': 1
        }
        self.scaling = self.scaling or {
            'time': 1,
            'error': 1
        }



@dataclass
class TrainingConfig:
    """Main training configuration"""
    # Experiment settings
    exp_name: str
    output_dir: str
    use_wandb: bool = False
    wandb_project: str = "combustion_rl"
    wandb_entity: Optional[str] = None
    seed: int = 42
    cuda: bool = False
    
    # Simulation settings
    n_points: int = 50
    n_threads: int = 2
    global_timestep: float = 1e-5
    profile_interval: int = 100
    species_to_track: List[str] = None
    
    # Network and algorithm settings
    network: NetworkConfig = None
    ppo: PPOConfig = None
    features: FeatureConfig = None
    reward: RewardConfig = None
    sim_configs: List[SimulationConfig] = None
    
    # Training loop settings
    eval_freq: int = 10000
    save_freq: int = 10000
    log_interval: int = 1000
    
    def __post_init__(self):
        # Set default values if not provided
        self.network = self.network or NetworkConfig()
        self.ppo = self.ppo or PPOConfig()
        self.features = self.features or FeatureConfig()
        self.reward = self.reward or RewardConfig()
        self.species_to_track = self.species_to_track or ['CH4', 'CO2', 'HO2', 'H2O2', 'OH', 'O2', 'H2', 'H2O']
        
        if not self.sim_configs:
            # Default simulation configurations
            self.sim_configs = [
                # Non-equilibrated configurations
                SimulationConfig(T_fuel=300, T_oxidizer=1200, t_end=0.05, pressure=101325, 
                         equilibrate_counterflow=False, center_width=0, slope_width=0),
                SimulationConfig(T_fuel=600, T_oxidizer=1300, t_end=0.05, pressure=101325,
                         equilibrate_counterflow=False, center_width=0, slope_width=0),
                SimulationConfig(T_fuel=900, T_oxidizer=1100, t_end=0.05, pressure=101325,
                         equilibrate_counterflow=False, center_width=0, slope_width=0),
                SimulationConfig(T_fuel=450, T_oxidizer=1500, t_end=0.05, pressure=101325,
                         equilibrate_counterflow=False, center_width=0, slope_width=0),
                SimulationConfig(T_fuel=1500, T_oxidizer=1500, t_end=0.05, pressure=101325,
                         equilibrate_counterflow=False, center_width=0, slope_width=0),

                # Equilibrated configurations with 'TP'
                SimulationConfig(T_fuel=300, T_oxidizer=1200, t_end=0.05, pressure=101325,
                         equilibrate_counterflow='TP', center_width=0.002, slope_width=0.001),
                SimulationConfig(T_fuel=900, T_oxidizer=1100, t_end=0.05, pressure=101325,
                         equilibrate_counterflow='TP', center_width=0.001, slope_width=0.0005),
                SimulationConfig(T_fuel=1200, T_oxidizer=1000, t_end=0.05, pressure=101325,
                         equilibrate_counterflow='TP', center_width=0.005, slope_width=0.001),
                SimulationConfig(T_fuel=1050, T_oxidizer=1200, t_end=0.05, pressure=101325,
                         equilibrate_counterflow='TP', center_width=0.001, slope_width=0.0005),
                SimulationConfig(T_fuel=1350, T_oxidizer=1200, t_end=0.05, pressure=101325,
                         equilibrate_counterflow='TP', center_width=0.008, slope_width=0.003),
            ]
    
    def save(self, filepath: str):
        """Save configuration to YAML file"""
        # Convert to dictionary
        config_dict = asdict(self)
        
        # Save to YAML
        with open(filepath, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    @classmethod
    def load(cls, filepath: str) -> 'TrainingConfig':
        """Load configuration from YAML file"""
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Convert nested dictionaries to appropriate dataclasses
        network = NetworkConfig(**config_dict.pop('network'))
        ppo = PPOConfig(**config_dict.pop('ppo'))
        features = FeatureConfig(**config_dict.pop('features'))
        reward = RewardConfig(**config_dict.pop('reward'))
        
        # Convert sim_configs list
        sim_configs = [SimulationConfig(**cfg) for cfg in config_dict.pop('sim_configs')]
        
        # Create TrainingConfig instance
        return cls(
            network=network,
            ppo=ppo,
            features=features,
            reward=reward,
            sim_configs=sim_configs,
            **config_dict
        )
    
    def get_ppo_kwargs(self):
        """Get PPO configuration as kwargs dict"""
        return {
            "learning_rate": self.ppo.learning_rate,
            "n_steps": self.ppo.n_steps,
            "batch_size": self.ppo.batch_size,
            "n_epochs": self.ppo.n_epochs,
            "gamma": self.ppo.gamma,
            "gae_lambda": self.ppo.gae_lambda,
            "clip_range": self.ppo.clip_range,
            "clip_range_vf": self.ppo.clip_range_vf,
            "ent_coef": self.ppo.ent_coef,
            "vf_coef": self.ppo.vf_coef,
            "max_grad_norm": self.ppo.max_grad_norm,
            "policy_kwargs": {
                "net_arch": {
                    "pi": self.network.pi_layers,
                    "vf": self.network.vf_layers
                },
                "activation_fn": getattr(nn, self.network.activation_fn)
            }
        }

# Example usage
def create_default_config() -> TrainingConfig:
    """Create default training configuration"""
    return TrainingConfig(
        exp_name="combustion_control",
        output_dir=f"experiments/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
        network=NetworkConfig(
            pi_layers=[256, 128, 64],
            vf_layers=[256, 128, 64],
            activation_fn="Tanh"
        ),
        ppo=PPOConfig(
            learning_rate=1e-3,
            n_steps=1000,
            batch_size=50000
        ),
        features=FeatureConfig(
            local_features=True,
            gradient_features=True
        ),
        reward=RewardConfig(
            weights={'accuracy': 1, 'efficiency': 3},
            use_neighbors=True
        )
    )