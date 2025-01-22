# import wandb
# import time
# import torch
# import os
# import numpy as np
# from dataclasses import dataclass, asdict
# from environment import create_env, SimulationSettings
# from ppo import MultiAgentPPO
# from typing import Dict, Optional
# import matplotlib.pyplot as plt
# from datetime import datetime

# def get_memory_usage():
#     """Get current memory usage in MB"""
#     import psutil
#     import os
#     process = psutil.Process(os.getpid())
#     return process.memory_info().rss / 1024 / 1024

# @dataclass
# class Args:
#     exp_name: str = "combustion_ppo_1d"
#     seed: int = 1
#     torch_deterministic: bool = True
#     cuda: bool = False
#     track: bool = False
#     wandb_project_name: str = "combustion_control_1d"
#     wandb_entity: Optional[str] = None
    
#     # Environment Parameters
#     output_dir: str = 'PPO_output'
#     save_dir: str = f'experiments/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
#     t_end: float = 0.06
#     n_points: int = 50
#     global_timestep: float = 1e-5
    
#     # Algorithm specific arguments
#     total_timesteps: int = 1000000
#     learning_rate: float = 2.5e-4
#     num_envs: int = 1
#     num_steps: int = 200
#     gamma: float = 0.99
#     gae_lambda: float = 0.95
#     num_minibatches: int = 128
#     update_epochs: int = 10
#     eps_clip: float = 0.2
#     entropy_coef: float = 0.01
#     value_loss_coef: float = 0.5
#     max_grad_norm: float = 0.5
    
#     save_step_data: bool = False
    
#     # Features configuration
#     features_config: Optional[Dict] = None
#     reward_config: Optional[Dict] = None
    
#     model_path: Optional[str] = None
    
#     update_freq: int = 600
    
#     save_freq: int = 20



# def train_ppo(args: Args):
#     # Set up wandb
#     run_name = f"{args.exp_name}_{args.seed}_{int(time.time())}"
#     if args.track:
#         wandb.init(
#             project=args.wandb_project_name,
#             entity=args.wandb_entity,
#             sync_tensorboard=True,
#             config=vars(args),
#             name=run_name,
#             monitor_gym=True,
#             save_code=True,
#         )

#     # Set seeds for reproducibility
#     if args.torch_deterministic:
#         torch.backends.cudnn.deterministic = True
#     torch.manual_seed(args.seed)
#     np.random.seed(args.seed)

#     # Setup device
#     device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    
#     # Create environment
#     sim_settings = SimulationSettings(
#         output_dir=args.output_dir,
#         t_end=args.t_end,
#         n_points=args.n_points,
#         global_timestep=args.global_timestep,
#     )
    
#     env = create_env(
#         sim_settings=sim_settings,
#         benchmark_file='env_benchmark.h5',
#         species_to_track=['CH4', 'O2', 'CO2', 'H2O'],
#         features_config=args.features_config,
#         reward_config=args.reward_config,
#         save_step_data=args.save_step_data
#     )
    
#     # Initialize PPO agent
#     obs_dim = env.observation_space.shape[1]
#     n_actions = 2
    
#     agent = MultiAgentPPO(
#         obs_dim=obs_dim,
#         n_actions=n_actions,
#         hidden_dims=[64, 64],
#         lr=args.learning_rate,
#         gamma=args.gamma,
#         gae_lambda=args.gae_lambda,
#         clip_ratio=args.eps_clip,
#         target_kl=0.01,
#         n_epochs=args.update_epochs,
#         batch_size=args.num_minibatches,
#         value_coef=args.value_loss_coef,
#         entropy_coef=args.entropy_coef,
#         max_grad_norm=args.max_grad_norm,
#         device=device
#     )

#     # Load model if path provided
#     if args.model_path:
#         agent.load(args.model_path)
    
#     # Training metrics
#     global_step = 0
#     start_time = time.time()
#     episode = 0
#     # Training loop
#     while global_step < args.total_timesteps:
#         state, _ = env.reset()
#         episode_reward = 0
#         episode_length = 0
#         episode_value_loss = 0
#         episode_policy_loss = 0
#         episode_entropy = 0
#         episode_accuracy = 0
#         episode_efficiency = 0
        
#         done = False
        
#         while not done:
#             # Get actions for all grid points
#             actions, log_probs, values = agent.select_action(state)
    
#             # Step environment
#             next_state, rewards, done, truncated, info = env.step(actions)
            
#             cvode, rk4 = get_action_distribution(actions)
            
#             if global_step % 500 == 0:
#                 print(f"Episode {episode} - env step: {env.current_step} - reward: {np.sum(rewards)} - cpu time: {info['cpu_time']}- action distribution: cvode: {cvode}, rk4: {rk4} - memory: {get_memory_usage()}")

#             if global_step % 100 == 0:
#                 print(f"Memory usage: {get_memory_usage()}")
            
#             # Store transition
#             agent.memory.store(state, actions, log_probs, values, rewards, done)
            
#             state = next_state
#             episode_reward += np.mean(rewards)
#             episode_length += 1
#             global_step += 1
    
#             if done or truncated:
#                 if episode % 10 == 0:
#                     print(f"Evaluating policy at episode {episode}")
#                     save_dir = os.path.join(args.save_dir, f"episode_{episode}")
#                     os.makedirs(save_dir, exist_ok=True)
#                     eval_results, episode_actions = evaluate_policy(env, agent, n_episodes=1, render=True, render_path=save_dir)
#                     env.render(save_path=os.path.join(save_dir, f"episode_{episode}.png"))
#                     plot_evaluation_results(eval_results, episode, save_dir)
#                     plot_actions(episode_actions, episode, save_dir)
#                     if args.track:
#                         wandb.save(os.path.join(save_dir, f"episode_{episode}.png"))
#                         wandb.save(os.path.join(save_dir, f"actions_{episode}.png"))
#                         wandb.save(os.path.join(save_dir, f"point_distributions_{episode}.png"))
#                 episode += 1
#                 break
            
#             if global_step % args.update_freq == 0:
#                 print(f"Updating policy at step {global_step} - memory: {get_memory_usage()}")
#                 # Update policy
#                 metrics = agent.update()
                
#                 episode_value_loss = metrics['value_loss']
#                 episode_policy_loss = metrics['policy_loss']
#                 episode_entropy = metrics['entropy']
                
#                 print(f"Memory usage after update: {get_memory_usage()}")
#                 # Log episode metrics
#                 if args.track:
#                     wandb.log({
#                         "episode/reward": episode_reward,
#                         "episode/length": episode_length,
#                         "episode/value_loss": episode_value_loss,
#                         "episode/policy_loss": episode_policy_loss,
#                         "episode/entropy": episode_entropy,
#                         "episode/accuracy": episode_accuracy,
#                         "episode/efficiency": episode_efficiency,
#                         "episode/sps": int(global_step / (time.time() - start_time)),
#                         "global_step": global_step,
#                     })
    
#         # Save model periodically
#         if episode % args.save_freq == 0:
#             save_dir = f"{args.save_dir}/models"
#             os.makedirs(save_dir, exist_ok=True)
#             model_path = os.path.join(save_dir, f"model_{global_step}.pt")
#             agent.save(model_path)
#             if args.track:
#                 wandb.save(model_path)
    
#     # Save final model
#     final_model_path = os.path.join(f"{args.save_dir}/models", "model_final.pt")
#     agent.save(final_model_path)
#     if args.track:
#         wandb.save(final_model_path)
    
#     return agent


import wandb
import time
import torch
import os
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, List
import gc
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
    track: bool = False
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
    gc.collect()
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
            
            # Store transition with explicit memory management
            transition = {
                'state': np.asarray(state, dtype=np.float32),
                'action': np.asarray(action, dtype=np.int32),
                'log_prob': np.asarray(log_prob, dtype=np.float32),
                'value': np.asarray(value, dtype=np.float32),
                'reward': np.asarray(reward, dtype=np.float32),
                'done': done
            }
            transitions_batch.append(transition)
            
            # Clear references to intermediate numpy arrays
            del state, action, log_prob, value, reward
            if len(transitions_batch) % 100 == 0:
                gc.collect()
            
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
        print(f"Processing batch - Memory before: {get_memory_usage():.2f} MB")
        
        # Store transitions
        for t in transitions:
            self.agent.store_transition(
                t['state'], t['action'], t['log_prob'], 
                t['value'], t['reward'], t['done']
            )
        
        # Update policy
        metrics = self.agent.update()
        
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

def evaluate_policy(
    env,
    agent,
    n_episodes: int = 1,
    render: bool = False,
    render_path: str = "renders"
):
    """Evaluate a trained policy"""
    episode_rewards = []
    episode_errors = []
    episode_times = []
    
    for episode in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0
        total_errors = np.zeros(env.sim_settings.n_points)
        total_time = 0
        episode_actions = []
        
        step = 0
        while True:
            # Select action
            with torch.no_grad():
                actions, _, _ = agent.select_action(state, deterministic=True)
            
            episode_actions.append(actions)
            # Step environment
            next_state, rewards, done, truncated, info = env.step(actions)
            
            # Track metrics
            episode_reward += rewards
            total_errors += info['point_errors']
            total_time += info['cpu_time']

            if step % 500 == 0:
                cvode_actions, rk4_actions = get_action_distribution(actions)
                print(f"[EVALUATION]: Episode {episode} - Step {step} - Reward: {np.sum(rewards):.2f}, Time: {total_time:.4f}, Action Distribution: cvode: {cvode_actions}, rk4: {rk4_actions}")
                print(f"Memory usage: {get_memory_usage()}")
            
            if done or truncated:
                break
                
            state = next_state
            step += 1
        
        episode_rewards.append(episode_reward)
        episode_errors.append(total_errors)
        episode_times.append(total_time)
    
    episode_rewards = np.array(episode_rewards)
    episode_errors = np.array(episode_errors)
    episode_times = np.array(episode_times)
    
    results = {
        "mean_reward_per_point": np.mean(episode_rewards, axis=0),      
        "std_reward_per_point": np.std(episode_rewards, axis=0),
        "mean_error_per_point": np.mean(episode_errors, axis=0),
        "std_error_per_point": np.std(episode_errors, axis=0),
        "mean_time_per_point": np.mean(episode_times, axis=0),
        "std_time_per_point": np.std(episode_times, axis=0)
    }

    
    return results, episode_actions

def plot_evaluation_results(results, step, dir):
    """Plot evaluation results"""
    plt.figure(figsize=(10, 8))
    
    # Reward distribution
    plt.subplot(2, 1, 1)
    plt.plot(results['mean_reward_per_point'], label='Mean Reward')
    plt.fill_between(
        range(len(results['mean_reward_per_point'])),
        results['mean_reward_per_point'] - results['std_reward_per_point'],
        results['mean_reward_per_point'] + results['std_reward_per_point'],
        alpha=0.3
    )
    plt.title('Reward Distribution Across Grid Points')
    plt.xlabel('Grid Point')
    plt.ylabel('Reward')
    plt.legend()
    
    # Error distribution
    plt.subplot(2, 1, 2)
    plt.plot(results['mean_error_per_point'], 'r-', label='Mean Error')
    plt.yscale('log')
    plt.title('Error Distribution Across Grid Points')
    plt.xlabel('Grid Point')
    plt.ylabel('Error')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{dir}/point_distributions_{step}.png')
    plt.close()

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
        n_epochs=3,
        batch_size=32,
        buffer_size=300
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