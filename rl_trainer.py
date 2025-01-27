from dataclasses import dataclass
from typing import Literal
from ppo import PPOAgent, PPOConfig
from sac import DiscreteSAC, SACConfig
import os
import time
import numpy as np
from collections import defaultdict
from environment import VectorizedCombustionEnv
import torch
import wandb
from datetime import datetime
from typing import Optional, Dict
import matplotlib.pyplot as plt
from environment import SimulationSettings, create_env
import argparse
from typing import List, Tuple
import psutil


def get_action_distribution(action):
    cvode_count = np.sum(action == 0)
    rk4_count = np.sum(action == 1)
    return cvode_count, rk4_count

def get_memory_usage() -> float:
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

@dataclass
class TrainerConfig:
    """Training configuration"""
    algorithm: Literal["PPO", "SAC"] = "PPO"
    exp_name: str = "combustion_rl_1d"
    seed: int = 1
    cuda: bool = True
    track: bool = True
    wandb_project_name: str = "combustion_control_1d"
    wandb_entity: Optional[str] = None
    output_dir: str = 'RL_output'
    save_dir: str = f'experiments/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    total_timesteps: int = 1000000
    batch_size: int = 300
    eval_freq: int = 2
    save_freq: int = 2
    max_memory_threshold: int = 2000  # MB
    
    # PPO-specific parameters
    initial_temperature: float = 2.0  # For PPO exploration
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
    
    def plot_episode_metrics(self):
        """Plot episode metrics"""
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))
        axs[0].plot(self.episode_cumulative_rewards, label='Cummulative Rewards')
        axs[0].set_title('Cummulative Rewards')
        axs[0].set_xlabel('Episode')
        axs[0].set_ylabel('Reward')
        axs[0].legend()
        
        axs[1].plot(self.episode_total_times, label='Total Time')
        axs[1].set_title('Total Time')
        axs[1].set_xlabel('Episode')
        axs[1].set_ylabel('Time')
        axs[1].legend()
        plt.savefig(os.path.join(self.config.save_dir, 'episode_metrics.png'))
        plt.close()
    
    def _get_temperature(self, step: int) -> float:
        """Calculate exploration temperature based on training progress"""
        progress = min(step / self.config.temperature_decay_steps, 1.0)
        return self.config.initial_temperature + \
               (self.config.final_temperature - self.config.initial_temperature) * progress    

    def _run_episode(self, episode: int, global_step: int, start_time: float, 
                    temperature: float = 1) -> Dict:
        print(f"\nStarting episode {episode} with memory: {get_memory_usage():.2f} MB")
        
        state, _ = self.env.reset()
        episode_reward = 0
        episode_length = 0
        episode_time = 0
        episode_metrics = defaultdict(float)
        action_counts = defaultdict(int)
        
        done = False
        while not done:
            # Select action based on algorithm
            if isinstance(self.agent, PPOAgent):
                action, log_prob, value = self._select_action_with_temperature(
                    state, temperature
                )
    
            else:  # SAC
                action = self.agent.select_action(state)
                
            cvode_count, rk4_count = get_action_distribution(action)
            action_counts['CVODE'] = cvode_count
            action_counts['RK4'] = rk4_count
            
            # Step environment
            next_state, reward, done, truncated, info = self.env.step(action)
            
            # Store transition based on algorithm
            if isinstance(self.agent, PPOAgent):
                self.agent.store_transition(
                    state, action, log_prob, value, reward, done
                )
            else:  # SAC
                self.agent.store_transition(
                    state, action, reward, next_state, done
                )
            
            # Update based on algorithm
            if isinstance(self.agent, PPOAgent):
                if self.agent.memory.size >= self.config.batch_size or done:
                    metrics = self.agent.update()
                    for k, v in metrics.items():
                        episode_metrics[k] += v
            else:  # SAC updates every step
                if episode_length % self.agent.config.batch_size == 0:
                    metrics = self.agent.update()
                    for k, v in metrics.items():
                        episode_metrics[k] += v
            
            if episode_length % 500 == 0:
                self._log_step_info(episode, episode_length, reward, info, 
                                  action_counts, temperature)
            
            state = next_state
            episode_reward += np.mean(reward)
            episode_length += 1
            episode_time += np.sum(info.get('cpu_time', 0))
            
            if done or truncated:
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
            'metrics': episode_metrics,
            'time': episode_time
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
        return metrics
    
    def train(self):
        """Main training loop"""
        start_time = time.time()
        global_step = 0
        episode = 0
        self.episode_cumulative_rewards = []
        self.episode_total_times = []
        
        while global_step < self.config.total_timesteps:
            # Calculate temperature for PPO exploration
            temperature = 1
            if isinstance(self.agent, PPOAgent):
                temperature = self._get_temperature(global_step)
            
            # Run episode
            episode_metrics = self._run_episode(episode, global_step, start_time, temperature)
            global_step += episode_metrics['length']
            episode += 1
            
            self.episode_cumulative_rewards.append(episode_metrics['reward'])
            self.episode_total_times.append(episode_metrics['time'])
            
            if self.config.track:
                wandb.log({
                    "episode/reward": episode_metrics['reward'],
                    "episode/length": episode_metrics['length'],
                    "episode/time": episode_metrics['time']
                })
            
            if episode % 10 == 0:
                self.plot_episode_metrics()
        
        self.save_model(global_step, final=True)
        if self.config.track:
            wandb.finish()
        
        return self.agent

    def _evaluate(self, episode: int):
        """Evaluate current policy"""
        print(f"Evaluating policy at episode {episode}")
        save_dir = os.path.join(self.config.save_dir, f"episode_{episode}")
        os.makedirs(save_dir, exist_ok=True)
        
        state, _ = self.env.reset()
        total_reward = 0
        actions_log = []
        episode_rewards_per_point = []
        total_episode_rewards = []
        eval_results = []
        eval_times = []
        
        with torch.no_grad():
            done = False
            step = 0
            while not done:
                # Select action based on algorithm
                if isinstance(self.agent, PPOAgent):
                    action, _, _ = self.agent.select_action(state, deterministic=True)
                else:  # SAC
                    action = self.agent.select_action(state, deterministic=True)
                
                cvode_count, rk4_count = get_action_distribution(action)
                actions_log.append(action)
                
                next_state, reward, done, truncated, info = self.env.step(action)
                episode_rewards_per_point.append(reward)
                total_episode_rewards.append(np.mean(reward))
    
                if step % 100 == 0:
                    print(f"Step {step} - CVODE: {cvode_count}, RK4: {rk4_count} - "
                          f"Reward: {np.mean(reward):.2f}")
                
                total_reward += np.mean(reward)
                eval_times.append(np.sum(info.get('cpu_time', 0)))
                state = next_state
                step += 1
        
        # Save evaluation results
        if hasattr(self.env, 'render'):
            self.env.render(save_path=os.path.join(save_dir, f"episode_{episode}.png"))

        self._plot_rewards(episode_rewards_per_point, episode, save_dir)

        fig, ax = plt.subplots()
        total_episode_rewards = np.array(total_episode_rewards)
        ax.plot(total_episode_rewards[0:-2])
        ax.set_title('Total Episode Rewards')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        plt.savefig(os.path.join(save_dir, f"total_episode_rewards_{episode}.png"))
        plt.close()
        
        self._plot_actions(actions_log, episode, save_dir)
        print(f"Evaluation results: Total reward: {total_reward:.2f}")
        if self.config.track:
            wandb.log({
                "eval/mean_reward": np.mean(total_reward),
                "eval/episode": episode,
                "eval/mean_time": np.mean(eval_times)
            })
            wandb.save(os.path.join(save_dir, f"episode_{episode}.png"))
            wandb.save(os.path.join(save_dir, f"actions_{episode}.png"))
            wandb.save(os.path.join(save_dir, f"rewards_{episode}.png"))
            wandb.save(os.path.join(save_dir, f"total_episode_rewards_{episode}.png"))
        return np.mean(total_episode_rewards)
    
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
        
    def _plot_rewards(self, rewards, episode, save_dir):
        """Plot actions distribution over time"""
        times_to_plot = [10, 100, 1000, 2000, 3000, 4000, 5000, 5900]
        fig, axs = plt.subplots(len(times_to_plot)//2, 2, figsize=(12, 8))
        
        for i, time in enumerate(times_to_plot):
            if time < len(rewards):
                axs[i//2, i%2].plot(rewards[time])
                axs[i//2, i%2].set_title(f"Time: {time}")
                axs[i//2, i%2].set_ylim(-0.5, 1.5)
                axs[i//2, i%2].set_ylabel("Reward")
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"rewards_{episode}.png"))
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
            f"Time: {np.sum(info.get('cpu_time', 0)):.4f} - "
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
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", type=str, choices=["PPO", "SAC"], default="PPO")
    args = parser.parse_args()

    # Common trainer config
    trainer_config = TrainerConfig(
        algorithm=args.algorithm,
        exp_name=f"combustion_{args.algorithm.lower()}_1d",
        cuda=True,
        track=True,
        total_timesteps=1000000,
        batch_size=600,
        n_points=50,
        T_fuel=600,
        T_oxidizer=1200,
        pressure=101325,
        global_timestep=1e-5,
        profile_interval=100,
        t_end=0.06
    )

    # Environment configurations
    features_config = {
        'local_features': True,
        'neighbor_features': False,
        'gradient_features': True,
        'temporal_features': False,
        'window_size': 4
    }

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

    # Create environment
    env = create_env(
        sim_settings=SimulationSettings(
            n_threads=2,
            output_dir=trainer_config.output_dir,
            t_end=trainer_config.t_end,
            n_points=trainer_config.n_points,
            T_fuel=trainer_config.T_fuel,
            T_oxidizer=trainer_config.T_oxidizer,
            pressure=trainer_config.pressure,
            global_timestep=trainer_config.global_timestep,
            profile_interval=trainer_config.profile_interval
        ),
        benchmark_file='env_benchmark2.h5',
        features_config=features_config,
        reward_config=reward_config,
        save_step_data=False
    )

    # Create agent based on selected algorithm
    if args.algorithm == "PPO":
        ppo_config = PPOConfig(
            lr=1e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_ratio=0.3,
            target_kl=0.03,
            n_epochs=10,
            batch_size=512,
            value_coef=0.5,
            entropy_coef=0.1,
            buffer_size=10000
        )
        agent = PPOAgent(
            obs_dim=env.observation_space.shape[1],
            n_actions=len(env.integrator_options),
            hidden_dims=[512, 256, 256, 128],
            config=ppo_config,
            device="cuda" if trainer_config.cuda and torch.cuda.is_available() else "cpu"
        )
    else:  # SAC
        sac_config = SACConfig(
            lr=3e-4,
            alpha_lr=3e-4,
            gamma=0.99,
            tau=0.01,
            buffer_size=1000000,
            batch_size=256,
            init_temperature=2.0,  # Start with more exploration
            min_temperature=0.1,   # Minimum temperature for stability
            hidden_dims=[512, 256, 256, 128]
        )
        agent = DiscreteSAC(
            obs_dim=env.observation_space.shape[1],
            n_actions=len(env.integrator_options),
            config=sac_config,
            device="cuda" if trainer_config.cuda and torch.cuda.is_available() else "cpu"
        )

    # Create trainer and train
    trainer = EfficientTrainer(env, agent, trainer_config)
    trained_agent = trainer.train()