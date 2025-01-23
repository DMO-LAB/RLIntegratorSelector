import wandb
import time
import torch
import os
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, List, Tuple
import psutil
from collections import defaultdict
import matplotlib.pyplot as plt
from environment import create_env, SimulationSettings
from ppo import PPOAgent, PPOConfig

def get_action_distribution(actions):
    """Get action distribution"""
    cvode_count = (actions == 0).sum()
    rk4_count = (actions == 1).sum()
    return cvode_count, rk4_count

@dataclass
class TrainerConfig:
    """Training configuration"""
    exp_name: str = "combustion_ppo_1d"
    seed: int = 1
    cuda: bool = True
    track: bool = True
    wandb_project_name: str = "combustion_control_1d"
    wandb_entity: Optional[str] = None
    output_dir: str = 'PPO_output'
    save_dir: str = f'experiments/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    total_timesteps: int = 1000000
    batch_size: int = 300
    eval_freq: int = 2
    save_freq: int = 2
    max_memory_threshold: int = 2000  # MB
    
    # Exploration parameters
    initial_temperature: float = 2.0
    final_temperature: float = 0.1
    temperature_decay_steps: int = 500000
    
    # Environment parameters
    n_points: int = 100
    T_fuel: float = 1000
    T_oxidizer: float = 1000
    pressure: float = 101325
    global_timestep: float = 1e-5
    profile_interval: float = 100
    t_end: float = 0.06

def get_memory_usage() -> float:
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def memory_cleanup():
    """Aggressive memory cleanup"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

class EfficientTrainer:
    def __init__(self, env, agent, config: TrainerConfig):
        self.env = env
        self.agent = agent
        self.config = config
        self.device = agent.device
        
        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs(self.config.save_dir, exist_ok=True)
        
        if self.config.track:
            self._init_wandb()
    
    def _init_wandb(self):
        wandb.init(
            project=self.config.wandb_project_name,
            entity=self.config.wandb_entity,
            config=vars(self.config),
            name=f"{self.config.exp_name}_{self.config.seed}_{int(time.time())}",
            monitor_gym=True,
            save_code=True,
        )
    
    def train(self):
        """Main training loop with improved exploration"""
        start_time = time.time()
        global_step = 0
        episode = 0
        
        while global_step < self.config.total_timesteps:
            # Calculate temperature for exploration
            temperature = self._get_temperature(global_step)
            
            # Run episode
            episode_metrics = self._run_episode(episode, global_step, start_time, temperature)
            global_step += episode_metrics['length']
            episode += 1
            
            memory_cleanup()
        
        self.save_model(global_step, final=True)
        if self.config.track:
            wandb.finish()
        
        return self.agent
    
    def _get_temperature(self, step: int) -> float:
        """Calculate exploration temperature based on training progress"""
        progress = min(step / self.config.temperature_decay_steps, 1.0)
        return self.config.initial_temperature + \
               (self.config.final_temperature - self.config.initial_temperature) * progress
    
    def _run_episode(self, episode: int, global_step: int, start_time: float, 
                    temperature: float) -> Dict:
        """Run single episode with temperature-based exploration"""
        print(f"\nStarting episode {episode} with memory: {get_memory_usage():.2f} MB")
        print(f"Current temperature: {temperature:.3f}")
        
        state, _ = self.env.reset()
        episode_reward = 0
        episode_length = 0
        episode_metrics = defaultdict(float)
        transitions_batch = []
        action_counts = {'CVODE': 0, 'RK4': 0}
        
        done = False
        while not done:
            # Select action with temperature
            action, log_prob, value = self._select_action_with_temperature(
                state, temperature
            )
            cvode_count, rk4_count = get_action_distribution(action)
            
            
            # Count actions
            action_counts['CVODE'] += cvode_count
            action_counts['RK4'] += rk4_count
            
            # Step environment
            next_state, reward, done, truncated, info = self.env.step(action)
            
            transitions_batch.append({
                'state': state.astype(np.float32),
                'action': action,
                'log_prob': log_prob,
                'value': value,
                'reward': reward,
                'done': done
            })
            
            if len(transitions_batch) >= self.config.batch_size or done:
                if get_memory_usage() > self.config.max_memory_threshold:
                    print("Memory threshold exceeded - performing cleanup")
                    memory_cleanup()
                
                metrics = self._process_batch(transitions_batch)
                for k, v in metrics.items():
                    episode_metrics[k] += v
                transitions_batch = []
            
            if episode_length % 500 == 0:
                self._log_step_info(episode, episode_length, reward, info, 
                                  action_counts, temperature)
                
                print(f"Step {episode_length} - CVODE: {cvode_count}, RK4: {rk4_count}")
            
            state = next_state
            episode_reward += np.mean(reward)
            episode_length += 1
            
            if done or truncated:
                if transitions_batch:
                    metrics = self._process_batch(transitions_batch)
                    for k, v in metrics.items():
                        episode_metrics[k] += v
                
                if episode % self.config.eval_freq == 0:
                    self._evaluate(episode)
                if episode % self.config.save_freq == 0:
                    self.save_model(global_step)
                
                break
        
        if self.config.track:
            self._log_episode_metrics(
                episode_reward, episode_length, episode_metrics,
                action_counts, global_step + episode_length,
                start_time, temperature
            )
        
        return {
            'reward': episode_reward,
            'length': episode_length,
            'metrics': episode_metrics
        }
    
    def _select_action_with_temperature(self, state: np.ndarray, 
                                      temperature: float) -> Tuple:
        """Select action using temperature-based exploration"""
        with torch.no_grad():
            state_tensor = torch.as_tensor(state, device=self.device)
            
            # Get logits and scale by temperature
            shared_features = self.agent.shared(state_tensor)
            action_logits = self.agent.policy_head(shared_features)
            scaled_logits = action_logits / temperature
            
            # Get probabilities and sample
            probs = torch.nn.functional.softmax(scaled_logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            
            action = dist.sample()
            log_prob = dist.log_prob(action)
            value = self.agent.value_head(shared_features)
            
            return (
                action.cpu().numpy(),
                log_prob.cpu().numpy(),
                value.cpu().numpy().squeeze()
            )
    
    def _process_batch(self, transitions: List[Dict]) -> Dict:
        """Process batch of transitions"""
        for t in transitions:
            self.agent.store_transition(
                t['state'], t['action'], t['log_prob'],
                t['value'], t['reward'], t['done']
            )
        
        metrics = self.agent.update()
        print(f"Batch processed - Memory after: {get_memory_usage():.2f} MB")
        return metrics
    
    def _evaluate(self, episode: int):
        """Evaluate current policy"""
        print(f"Evaluating policy at episode {episode}")
        save_dir = os.path.join(self.config.save_dir, f"episode_{episode}")
        os.makedirs(save_dir, exist_ok=True)
        
        eval_results = []
        # for _ in range():  # Run multiple evaluation episodes
        state, _ = self.env.reset()
        total_reward = 0
        actions_log = []
        
        with torch.no_grad():
            done = False
            step = 0
            while not done:
                action, _, _ = self.agent.select_action(state, deterministic=True)
                cvode_count, rk4_count = get_action_distribution(action)
                if step % 100 == 0:
                    print(f"Step {step} - CVODE: {cvode_count}, RK4: {rk4_count}")
                actions_log.append(action)
                next_state, reward, done, truncated, _ = self.env.step(action)
                total_reward += np.mean(reward)
                state = next_state
                step += 1
        eval_results.append(total_reward)
        
        # Save evaluation results
        if hasattr(self.env, 'render'):
            self.env.render(save_path=os.path.join(save_dir, f"episode_{episode}.png"))
        
        self._plot_actions(actions_log, episode, save_dir)
        
        if self.config.track:
            wandb.log({
                "eval/mean_reward": np.mean(eval_results),
                "eval/std_reward": np.std(eval_results),
                "eval/episode": episode
            })
            wandb.save(os.path.join(save_dir, f"episode_{episode}.png"))
            wandb.save(os.path.join(save_dir, f"actions_{episode}.png"))
        
        return np.mean(eval_results)
    
    def _plot_actions(self, actions, episode, save_dir):
        """Plot actions distribution over time"""
        times_to_plot = [10, 100, 1000, 2000, 3000, 4000, 5000, 5900]
        fig, axs = plt.subplots(len(times_to_plot)//2, 2, figsize=(12, 8))
        
        for i, time in enumerate(times_to_plot):
            if time < len(actions):
                axs[i//2, i%2].plot(actions[time])
                axs[i//2, i%2].set_title(f"Time: {time}")
                axs[i//2, i%2].set_ylim(-0.5, 1.5)
                axs[i//2, i%2].set_ylabel("Action (0=CVODE, 1=RK4)")
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"actions_{episode}.png"))
        plt.close()
    
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
                      info: Dict, action_counts: Dict, temperature: float):
        """Log step information"""
        print(
            f"Episode {episode} - Step: {step} - "
            f"Reward: {np.mean(reward):.2f} - "
            f"Time: {info.get('cpu_time', 0):.4f} - "
            f"Actions: CVODE: {action_counts['CVODE']}, RK4: {action_counts['RK4']} - "
            f"Temp: {temperature:.3f} - "
            f"Memory: {get_memory_usage():.2f} MB"
        )
    
    def _log_episode_metrics(self, episode_reward: float, episode_length: int,
                           metrics: Dict, action_counts: Dict, global_step: int,
                           start_time: float, temperature: float):
        """Log episode metrics to wandb"""
        total_actions = action_counts['CVODE'] + action_counts['RK4']
        cvode_ratio = action_counts['CVODE'] / total_actions if total_actions > 0 else 0
        
        wandb.log({
            "episode/reward": episode_reward,
            "episode/length": episode_length,
            "episode/value_loss": metrics.get('value_loss', 0),
            "episode/policy_loss": metrics.get('policy_loss', 0),
            "episode/entropy": metrics.get('entropy', 0),
            "episode/temperature": temperature,
            "episode/cvode_ratio": cvode_ratio,
            "episode/sps": int(global_step / (time.time() - start_time)),
            "global_step": global_step,
        })



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
    # PPO Configuration
    ppo_config = PPOConfig(
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        target_kl=0.015,
        n_epochs=10,
        batch_size=64,
        value_coef=0.5,
        entropy_coef=0.05,
        buffer_size=300
    )
    
    # Trainer Configuration
    trainer_config = TrainerConfig(
        exp_name="combustion_ppo_1d",
        cuda=True,
        track=True,
        total_timesteps=1000000,
        batch_size=150,
        initial_temperature=2.0,
        final_temperature=0.5,
        temperature_decay_steps=300000,
        n_points=50,
        T_fuel=600,
        T_oxidizer=1200,
        pressure=101325,
        global_timestep=1e-5,
        profile_interval=100,
        t_end=0.06
    )

    # Environment configurations
    reward_config = {
        'weights': {
            'accuracy': 1,
            'efficiency': 3,
        },
        'thresholds': {
            'time': 0.001,
            'error': 1
        },
        'scaling': {
            'time': 1,
            'error': 1
        },
        'use_neighbors': True,
        'neighbor_weight': 0.3,
        'neighbor_radius': 4
    }

    features_config = {
        'local_features': True,
        'neighbor_features': True,
        'gradient_features': True,
        'temporal_features': True,
        'window_size': 4
    }

    # Create environment
    sim_settings = SimulationSettings(
        n_threads=2,
        output_dir=trainer_config.output_dir,
        t_end=trainer_config.t_end,
        n_points=trainer_config.n_points,
        T_fuel=trainer_config.T_fuel,
        T_oxidizer=trainer_config.T_oxidizer,
        pressure=trainer_config.pressure,
        global_timestep=trainer_config.global_timestep,
        profile_interval=trainer_config.profile_interval
    )
    
    env = create_env(
        sim_settings,
        benchmark_file='env_benchmark.h5',
        species_to_track=['CH4', 'O2', 'CO2', 'H2O'],
        features_config=features_config,
        reward_config=reward_config,
        save_step_data=False
    )

    # Create agent
    agent = PPOAgent(
        obs_dim=env.observation_space.shape[1],
        n_actions=len(env.integrator_options),
        hidden_dims=[64, 64],
        config=ppo_config,
        device="cuda" if trainer_config.cuda and torch.cuda.is_available() else "cpu"
    )

    # Create trainer and train
    trainer = EfficientTrainer(env, agent, trainer_config)
    trained_agent = trainer.train()