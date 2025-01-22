import wandb
import time
import torch
import os
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, List
import psutil
from ppo import PPOAgent, PPOConfig
from environment import create_env, SimulationSettings
from collections import defaultdict
import matplotlib.pyplot as plt

@dataclass
class TrainerConfig:
    """Training configuration"""
    exp_name: str = "combustion_ppo_1d"
    seed: int = 1
    cuda: bool = True
    track: bool = True
    wandb_project_name: str = "combustion_control_1d"
    wandb_entity: Optional[str] = None
    
    # Environment Parameters
    output_dir: str = 'PPO_output'
    save_dir: str = f'experiments/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    
    # Training Parameters
    total_timesteps: int = 1000000
    batch_size: int = 300
    eval_freq: int = 10
    save_freq: int = 20
    max_memory_threshold: int = 2000  # MB

def get_memory_usage() -> float:
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def memory_cleanup():
    """Aggressive memory cleanup"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

@dataclass
class Args:
    exp_name: str = "combustion_ppo_1d"
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = False
    track: bool = False
    wandb_project_name: str = "combustion_control_1d"
    wandb_entity: Optional[str] = None
    
    # Environment Parameters
    output_dir: str = 'PPO_output'
    save_dir: str = f'experiments/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    t_end: float = 0.06
    n_points: int = 50
    global_timestep: float = 1e-5
    T_fuel: float = 600
    T_oxidizer: float = 1200
    pressure: float = 101325
    profile_interval: int = 20
    
class EfficientTrainer:
    def __init__(self, env, agent: PPOAgent, config: TrainerConfig, args: Args):
        self.env = env
        self.agent = agent
        self.config = config
        self.args = args
        self.device = agent.device
        
        # Create directories
        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs(self.config.save_dir, exist_ok=True)
        
        # Initialize wandb if tracking
        if self.config.track:
            self._init_wandb()
    
    def _init_wandb(self):
        """Initialize weights & biases logging"""
        wandb.init(
            project=self.config.wandb_project_name,
            entity=self.config.wandb_entity,
            config=vars(self.config),
            name=f"{self.config.exp_name}_{self.config.seed}_{int(time.time())}",
            monitor_gym=True,
            save_code=True,
        )
    
    def train(self):
        """Main training loop with memory efficiency"""
        start_time = time.time()
        global_step = 0
        episode = 0
        
        while global_step < self.config.total_timesteps:
            episode_metrics = self._run_episode(episode, global_step, start_time)
            global_step += episode_metrics['length']
            episode += 1
            
            # Cleanup between episodes
            memory_cleanup()
        
        # Save final model
        self.save_model(global_step, final=True)
        
        if self.config.track:
            wandb.finish()
        
        return self.agent
    
    def _run_episode(self, episode: int, global_step: int, start_time: float) -> Dict:
        """Run a single episode with memory-efficient batch processing"""
        print(f"\nStarting episode {episode} with memory: {get_memory_usage():.2f} MB")
        
        state, _ = self.env.reset()
        episode_reward = 0
        episode_length = 0
        episode_metrics = defaultdict(float)
        transitions_batch = []
        
        done = False
        while not done:
            # Select action
            action, log_prob, value = self.agent.select_action(state)

            # Step environment
            next_state, reward, done, truncated, info = self.env.step(action)
            
            # Store transition
            transitions_batch.append({
                'state': state.astype(np.float32),
                'action': action,
                'log_prob': log_prob,
                'value': value,
                'reward': reward,
                'done': done
            })
            
            # Process batch if ready
            if len(transitions_batch) >= self.config.batch_size or done:
                if get_memory_usage() > self.config.max_memory_threshold:
                    print("Memory threshold exceeded - performing emergency cleanup")
                    memory_cleanup()
                
                metrics = self._process_batch(transitions_batch)
                for k, v in metrics.items():
                    episode_metrics[k] += v
                transitions_batch = []
            
            # Logging
            if episode_length % 500 == 0:
                self._log_step_info(episode, episode_length, reward, info, action)
            
            # Update state and metrics
            state = next_state
            episode_reward += np.mean(reward)
            episode_length += 1
            
            if done or truncated:
                # Final batch processing
                if transitions_batch:
                    metrics = self._process_batch(transitions_batch)
                    for k, v in metrics.items():
                        episode_metrics[k] += v
                
                # Evaluation and saving
                if episode % self.config.eval_freq == 0:
                    self._evaluate(episode)
                if episode % self.config.save_freq == 0:
                    self.save_model(global_step + episode_length)
                
                break
        
        # Log episode metrics
        if self.config.track:
            self._log_episode_metrics(
                episode_reward, 
                episode_length, 
                episode_metrics, 
                global_step + episode_length, 
                start_time
            )
        
        return {
            'reward': episode_reward,
            'length': episode_length,
            'metrics': episode_metrics
        }
    
    def _process_batch(self, transitions: List[Dict]) -> Dict:
        """Process a batch of transitions efficiently"""
        # Store transitions
        for t in transitions:
            self.agent.store_transition(
                t['state'], t['action'], t['log_prob'], 
                t['value'], t['reward'], t['done']
            )
        
        # Update policy
        metrics = self.agent.update()
        metrics = {}
        print(f"Batch processed - Memory after: {get_memory_usage():.2f} MB")
        return metrics
    
    def _evaluate(self, episode: int):
        """Evaluate current policy"""
        print(f"Evaluating policy at episode {episode}")
        save_dir = os.path.join(self.config.save_dir, f"episode_{episode}")
        os.makedirs(save_dir, exist_ok=True)
        
        state, _ = self.env.reset()
        total_reward = 0
        actions_log = []
        
        with torch.no_grad():
            done = False
            while not done:
                action, _, _ = self.agent.select_action(state, deterministic=True)
                actions_log.append(action)
                next_state, reward, done, truncated, _ = self.env.step(action)
                total_reward += np.mean(reward)
                state = next_state
        
        # Save evaluation artifacts
        if hasattr(self.env, 'render'):
            self.env.render(save_path=os.path.join(save_dir, f"episode_{episode}.png"))
        
        if self.config.track:
            wandb.save(os.path.join(save_dir, f"episode_{episode}.png"))
            
        plot_actions(actions_log, episode, save_dir)
        
        if self.config.track:
            wandb.save(os.path.join(save_dir, f"actions_{episode}.png"))
        
        return total_reward, actions_log
    
    def save_model(self, step: int, final: bool = False):
        """Save model checkpoint"""
        save_dir = os.path.join(self.config.save_dir, "models")
        os.makedirs(save_dir, exist_ok=True)
        
        model_path = os.path.join(
            save_dir, 
            "model_final.pt" if final else f"model_{step}.pt"
        )
        
        self.agent.save(model_path)
        
        if self.config.track:
            wandb.save(model_path)
    
    def _log_step_info(self, episode: int, step: int, reward: float, 
                      info: Dict, action: np.ndarray):
        """Log step-level information"""
        cvode_actions = (action == 0).sum()
        rk4_actions = (action == 1).sum()
        
        print(
            f"Episode {episode} - Step: {step} - "
            f"Reward: {np.mean(reward):.2f} - "
            f"Time: {info.get('cpu_time', 0):.4f} - "
            f"Actions: CVODE: {cvode_actions}, RK4: {rk4_actions} - "
            f"Memory: {get_memory_usage():.2f} MB"
        )
    
    def _log_episode_metrics(self, episode_reward: float, episode_length: int,
                           metrics: Dict, global_step: int, start_time: float):
        """Log episode-level metrics to wandb"""
        wandb.log({
            "episode/reward": episode_reward,
            "episode/length": episode_length,
            "episode/value_loss": metrics.get('value_loss', 0),
            "episode/policy_loss": metrics.get('policy_loss', 0),
            "episode/entropy": metrics.get('entropy', 0),
            "episode/sps": int(global_step / (time.time() - start_time)),
            "global_step": global_step,
        })

def train(env, ppo_config: PPOConfig, trainer_config: TrainerConfig, args: Args):
    """Main training function"""
    # Set seeds and device
    torch.manual_seed(trainer_config.seed)
    np.random.seed(trainer_config.seed)
    device = torch.device("cuda" if trainer_config.cuda and torch.cuda.is_available() else "cpu")
    
    # Create agent
    agent = PPOAgent(
        obs_dim=env.observation_space.shape[1],
        n_actions=2,
        hidden_dims=[64, 64],
        config=ppo_config,
        device=device
    )
    
    # Create trainer and train
    trainer = EfficientTrainer(env, agent, trainer_config, args)
    trained_agent = trainer.train()
    
    return trained_agent

def plot_actions(actions, step, dir):
    """Plot actions"""
    times_to_plot = [10, 100, 1000, 2000, 3000, 4000, 5000, 5900]
    fig, axs = plt.subplots(len(times_to_plot)//2, 2, figsize=(12, 8))
    for i, time in enumerate(times_to_plot):
        axs[i//2, i%2].plot(actions[time])
        axs[i//2, i%2].set_title(f"Time: {time}")
    plt.tight_layout()
    plt.savefig(f'{dir}/actions_{step}.png')
    plt.close()
        
        
if __name__ == "__main__":
    # Create configurations
    args = Args()
    
    ppo_config = PPOConfig(
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        target_kl=0.01,
        n_epochs=10,
        batch_size=32,
        buffer_size=30000
    )

    trainer_config = TrainerConfig(
        exp_name=args.exp_name,
        cuda=True,
        track=True,
        total_timesteps=1000000,
        batch_size=300
    )

    # Create environment
    sim_settings = SimulationSettings(
        output_dir=args.output_dir,
        t_end=args.t_end,
        n_points=args.n_points,
        T_fuel=args.T_fuel,
        T_oxidizer=args.T_oxidizer,
        pressure=args.pressure,
        global_timestep=args.global_timestep,
        profile_interval=args.profile_interval
    )
    env = create_env(
        sim_settings,
        benchmark_file='env_benchmark.h5',
        species_to_track=['CH4', 'O2', 'CO2', 'H2O'],
        features_config=None,
        reward_config=None,
        save_step_data=False
    )

    # Train agent
    trained_agent = train(env, ppo_config, trainer_config, args)