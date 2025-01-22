import wandb
import time
import torch
import os
import numpy as np
from dataclasses import dataclass, asdict
from environment import create_env, SimulationSettings
from ppo import MultiAgentPPO
from typing import Dict, Optional
import matplotlib.pyplot as plt
from datetime import datetime

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
    output_dir: str = 'run/rl_train'
    save_dir: str = f'experiments/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    t_end: float = 0.06
    n_points: int = 50
    global_timestep: float = 1e-5
    
    # Algorithm specific arguments
    total_timesteps: int = 1000000
    learning_rate: float = 2.5e-4
    num_envs: int = 1
    num_steps: int = 200
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 64
    update_epochs: int = 10
    eps_clip: float = 0.2
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    max_grad_norm: float = 0.5
    
    save_step_data: bool = False
    
    # Features configuration
    features_config: Optional[Dict] = None
    reward_config: Optional[Dict] = None
    
    model_path: Optional[str] = None
    
    update_freq: int = 600
    
    save_freq: int = 20

def get_action_distribution(actions):
    cvode = (actions == 0).sum()
    rk4 = (actions == 1).sum()
    return cvode, rk4

def train_ppo(args: Args):
    # Set up wandb
    run_name = f"{args.exp_name}_{args.seed}_{int(time.time())}"
    if args.track:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    # Set seeds for reproducibility
    if args.torch_deterministic:
        torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Setup device
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    
    # Create environment
    sim_settings = SimulationSettings(
        output_dir=args.output_dir,
        t_end=args.t_end,
        n_points=args.n_points,
        global_timestep=args.global_timestep,
    )
    
    env = create_env(
        sim_settings=sim_settings,
        benchmark_file='env_benchmark.h5',
        species_to_track=['CH4', 'O2', 'CO2', 'H2O', 'OH', 'CO'],
        features_config=args.features_config,
        reward_config=args.reward_config,
        save_step_data=args.save_step_data
    )
    
    # Initialize PPO agent
    obs_dim = env.observation_space.shape[1]
    n_actions = 2
    
    agent = MultiAgentPPO(
        obs_dim=obs_dim,
        n_actions=n_actions,
        hidden_dims=[64, 64],
        lr=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_ratio=args.eps_clip,
        target_kl=0.01,
        n_epochs=args.update_epochs,
        batch_size=args.num_minibatches,
        value_coef=args.value_loss_coef,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
        device=device
    )

    # Load model if path provided
    if args.model_path:
        agent.load(args.model_path)
    
    # Training metrics
    global_step = 0
    start_time = time.time()
    episode = 0
    # Training loop
    while global_step < args.total_timesteps:
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        episode_value_loss = 0
        episode_policy_loss = 0
        episode_entropy = 0
        episode_accuracy = 0
        episode_efficiency = 0
        
        done = False
        
        while not done:
            # Get actions for all grid points
            actions, log_probs, values = agent.select_action(state)
    
            # Step environment
            next_state, rewards, done, truncated, info = env.step(actions)
            
            cvode, rk4 = get_action_distribution(actions)
            
            if global_step % 200 == 0:
                print(f"Episode {episode} - env step: {env.current_step} - reward: {np.sum(rewards)} - cpu time: {info['cpu_time']}- action distribution: cvode: {cvode}, rk4: {rk4}")
        
            
            # Store transition
            agent.memory.store(state, actions, log_probs, values, rewards, done)
            
            state = next_state
            episode_reward += np.mean(rewards)
            episode_length += 1
            global_step += 1
    
            if done or truncated:
                if episode % 10 == 0:
                    print(f"Evaluating policy at episode {episode}")
                    save_dir = os.path.join(args.save_dir, f"episode_{episode}")
                    os.makedirs(save_dir, exist_ok=True)
                    eval_results, episode_actions = evaluate_policy(env, agent, n_episodes=1, render=True, render_path=save_dir)
                    env.render(save_path=os.path.join(save_dir, f"episode_{episode}.png"))
                    plot_evaluation_results(eval_results, episode, save_dir)
                    plot_actions(episode_actions, episode, save_dir)
                    if args.track:
                        wandb.save(os.path.join(save_dir, f"episode_{episode}.png"))
                        wandb.save(os.path.join(save_dir, f"actions_{episode}.png"))
                        wandb.save(os.path.join(save_dir, f"point_distributions_{episode}.png"))
                episode += 1
                break
            
            if global_step % args.update_freq == 0:
                # Update policy
                print(f"Updating policy at step {global_step}")
                metrics = agent.update()
                
                episode_value_loss = metrics['value_loss']
                episode_policy_loss = metrics['policy_loss']
                episode_entropy = metrics['entropy']
                
                # Log episode metrics
                if args.track:
                    wandb.log({
                        "episode/reward": episode_reward,
                        "episode/length": episode_length,
                        "episode/value_loss": episode_value_loss,
                        "episode/policy_loss": episode_policy_loss,
                        "episode/entropy": episode_entropy,
                        "episode/accuracy": episode_accuracy,
                        "episode/efficiency": episode_efficiency,
                        "episode/sps": int(global_step / (time.time() - start_time)),
                        "global_step": global_step,
                    })
    
        # Save model periodically
        if episode % args.save_freq == 0:
            save_dir = f"{args.save_dir}/models"
            os.makedirs(save_dir, exist_ok=True)
            model_path = os.path.join(save_dir, f"model_{global_step}.pt")
            agent.save(model_path)
            if args.track:
                wandb.save(model_path)
    
    # Save final model
    final_model_path = os.path.join(f"{args.save_dir}/models", "model_final.pt")
    agent.save(final_model_path)
    if args.track:
        wandb.save(final_model_path)
    
    return agent

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

            if step % 100 == 0:
                cvode_actions, rk4_actions = get_action_distribution(actions)
                print(f"[EVALUATION]: Episode {episode} - Step {step} - Reward: {np.sum(rewards):.2f}, Time: {total_time:.4f}, Action Distribution: cvode: {cvode_actions}, rk4: {rk4_actions}")
                
            
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
    args = Args()
    
    # Set default configurations if not provided
    if args.features_config is None:
        args.features_config = {
            'local_features': True,
            'neighbor_features': True,
            'gradient_features': True,
            'temporal_features': True,
            'window_size': 5
        }
    
    if args.reward_config is None:
        args.reward_config = {
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
            }
        }
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Train model
    agent = train_ppo(args)
    
    # Cleanup
    if args.track:
        wandb.finish()