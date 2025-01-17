import os
import random
import time
from dataclasses import dataclass
import torch
import numpy as np
import tyro
import wandb
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from environment import SimulationSettings, create_env
from ppo import PPO, RolloutBuffer

@dataclass
class Args:
    exp_name: str = "combustion_ppo_1d"
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = False
    track: bool = False
    wandb_project_name: str = "combustion_control_1d"
    wandb_entity: str = None
    
    # Environment Parameters
    output_dir: str = 'run/rl_train'
    t_end: float = 0.08
    n_points: int = 100
    global_timestep: float = 1e-5
    
    # Algorithm specific arguments
    total_timesteps: int = 100000
    learning_rate: float = 2.5e-3
    num_envs: int = 1
    num_steps: int = 80
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 16
    update_epochs: int = 10
    eps_clip: float = 0.2
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    max_grad_norm: float = 0.5
    
    save_step_data: bool = False
    
    # Features configuration
    features_config: dict = None
    reward_config: dict = None
    
    model_path: str = None

def make_env(args):
    """Create environment with specified settings"""
    sim_settings = SimulationSettings(
        output_dir=args.output_dir,
        t_end=args.t_end,
        n_points=args.n_points,
        global_timestep=args.global_timestep
    )
    
    env = create_env(
        sim_settings=sim_settings,
        benchmark_file="env_benchmark.h5",
        species_to_track=['CH4', 'O2', 'CO2', 'H2O'],
        features_config=args.features_config,
        reward_config=args.reward_config,
        save_step_data=args.save_step_data
    )
    return env

def log_episode_info(writer, global_step, info):
    """Log episode information to tensorboard and wandb"""
    log_dict = {}
    
    if 'point_errors' in info:
        mean_error = np.mean(info['point_errors'])
        max_error = np.max(info['point_errors'])
        log_dict.update({
            "metrics/mean_error": mean_error,
            "metrics/max_error": max_error
        })
    
    if 'point_rewards' in info:
        mean_reward = np.mean(info['point_rewards'])
        min_reward = np.min(info['point_rewards'])
        log_dict.update({
            "metrics/mean_reward": mean_reward,
            "metrics/min_reward": min_reward
        })
    
    if 'cpu_time' in info:
        log_dict["metrics/step_time"] = info['cpu_time']
    
    if 'total_time' in info:
        log_dict["metrics/total_time"] = info['total_time']
    
    # Log to both tensorboard and wandb
    for key, value in log_dict.items():
        writer.add_scalar(key, value, global_step)
    if wandb.run is not None:
        wandb.log(log_dict, step=global_step)


def train(args):
    """Main training loop"""
    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    
    # Setup run
    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
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
    writer = SummaryWriter(f"runs/{run_name}")
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    
    # Environment setup
    env = make_env(args)
    
    # PPO agent setup
    state_dim = env.observation_space.shape[1]
    action_dim = len(env.integrator_options)
    
    ppo_agent = PPO(
        state_dim=state_dim,
        action_dim=action_dim,
        lr_actor=args.learning_rate,
        lr_critic=args.learning_rate,
        gamma=args.gamma,
        K_epochs=args.update_epochs,
        eps_clip=args.eps_clip,
        has_continuous_action_space=False
    )
    
    # load model if model_path is provided
    if args.model_path is not None:
        print(f"Loading model from {args.model_path}")
        ppo_agent.load(args.model_path)
    
    # Initialize environment
    obs, _ = env.reset(seed=args.seed)
    
    # Training loop
    start_time = time.time()
    global_step = 0
    
    # Calculate number of updates
    total_timesteps = args.total_timesteps
    num_updates = total_timesteps // args.num_steps
    
    for update in range(1, num_updates + 1):
        # Collect trajectory
        for step in range(0, args.num_steps):
            global_step += 1
            
            # Get actions for all points
            actions = []
            for point_obs in obs:
                action = ppo_agent.select_action(point_obs)
                actions.append(action)
            actions = np.array(actions)
            
            # get the action distribution
            if global_step % 500 == 0:
                action_distribution = np.bincount(actions) / len(actions)
                
                print(f"[STEP {global_step}] Action Distribution: {action_distribution}")
            # Execute in environment
            next_obs, rewards, terminated, truncated, info = env.step(actions)
            
            # Store step data
            for point_idx in range(args.n_points):
                # Convert to done flag
                done = terminated or truncated
                
                # Store transition
                ppo_agent.buffer.rewards.append(rewards[point_idx])
                ppo_agent.buffer.is_terminals.append(done)
            
            # Log episode info
            if info:
                log_episode_info(writer, global_step, info)
            
            # Check if episode is done
            if (terminated or truncated):
                print(f"Episode Done {update} - resetting environment")
                if env.save_step_data:
                    env.save_episode(f"{args.output_dir}/episode_{update}.h5")
                next_obs, _ = env.reset()

        # Update PPO agent
        ppo_agent.update()
        
        # Log training metrics
        if wandb.run is not None:
            wandb.log({
                "charts/learning_rate": args.learning_rate,
                "train/mean_reward": np.mean(ppo_agent.buffer.rewards),
                "train/episode_length": len(ppo_agent.buffer.rewards)
            }, step=global_step)
        
        # Save model periodically
        if update % 10 == 0:
            save_path = f"models/{run_name}/checkpoint_{update}.pt"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            ppo_agent.save(save_path)
            if args.track:
                wandb.save(save_path)
        
        if update % 100 == 0:
            evaluate_policy(env, ppo_agent, num_episodes=1, step=update, run_name=run_name)
            obs, _ = env.reset()
    
    # Save final model
    save_path = f"models/{run_name}/model_final.pt"
    ppo_agent.save(save_path)
    if args.track:
        wandb.save(save_path)
    
    # env.close()
    
    writer.close()
    return ppo_agent, env

def evaluate_policy(env, ppo_agent, num_episodes=4, step=0, run_name=None):
    """Evaluate the trained policy"""
    if run_name is not None:
        os.makedirs(f'evaluation/{run_name}', exist_ok=True)
    else:
        os.makedirs('evaluation', exist_ok=True)
        
    dir = f'evaluation/{run_name}' if run_name is not None else 'evaluation'
    
    episode_rewards = np.zeros((num_episodes, env.sim_settings.n_points))
    episode_lengths = np.zeros(num_episodes)
    episode_errors = np.zeros((num_episodes, env.sim_settings.n_points))
    episode_cpu_times = np.zeros(num_episodes)
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_length = 0
        episode_actions = []
        global_step = 0
        while not done:
            # Get actions for all points
            actions = []
            for point_obs in obs:
                action = ppo_agent.select_action(point_obs, deterministic=True, store_in_buffer=False)
                actions.append(action)
            actions = np.array(actions)
            episode_actions.append(actions)
            
            if global_step % 100 == 0:
                print(f"[EvalSTEP {global_step}] Action Distribution: {np.bincount(actions) / len(actions)}")
            
            obs, rewards, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
            
            episode_rewards[episode] += rewards
            episode_length += 1
            global_step += 1
            if done:
                print(f"Episode Done {episode}")
                episode_lengths[episode] = episode_length
                episode_errors[episode] = info['point_errors']
                episode_cpu_times[episode] = info['cpu_time']
                
                print(f"Cummulative Episode Reward: {np.sum(episode_rewards[episode])}")
                print(f"Cummulative Episode Error: {np.sum(episode_errors[episode])}")
                print(f"Cummulative Episode CPU Time: {np.sum(episode_cpu_times[episode])}")
    
    # Compute and return statistics
    results = {
        'mean_reward_per_point': np.mean(episode_rewards, axis=0),
        'std_reward_per_point': np.std(episode_rewards, axis=0),
        'mean_error_per_point': np.mean(episode_errors, axis=0),
        'mean_episode_length': np.mean(episode_lengths),
        'mean_cpu_time': np.mean(episode_cpu_times),
        'total_rewards': np.sum(episode_rewards),
        'max_error': np.max(episode_errors)
    }
    
    # Plot results
    plot_evaluation_results(results, step, dir=dir)
    plot_actions(episode_actions, step, dir=dir)
    
    env.render(save_path=f"{dir}/render_{step}.png")
    
    return results

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
    times_to_plot = [10, 100, 1000, 2000, 3000, 5000, 6000, 7500]
    fig, axs = plt.subplots(len(times_to_plot)//2, 2, figsize=(12, 8))
    for i, time in enumerate(times_to_plot):
        axs[i//2, i%2].plot(actions[time])
        axs[i//2, i%2].set_title(f"Time: {time}")
    plt.tight_layout()
    plt.savefig(f'{dir}/actions_{step}.png')
    plt.close()

if __name__ == "__main__":
    args = Args(cuda=False, model_path="models/combustion_ppo_1d__1__1736988505/checkpoint_850.pt")

    # Set default configurations if not provided
    if args.features_config is None:
        args.features_config = {
            'local_features': True,
            'neighbor_features': False,
            'gradient_features': False,
            'temporal_features': True,
            'window_size': 4
        }

    if args.reward_config is None:
        args.reward_config = {
            'weights': {
                'accuracy': 0.6,
                'efficiency': 0.3,
                'stability': 0.0
            },
            'thresholds': {
                'time': 0.05,
                'error': 100,
                'stability': 0.1
            },
            'scaling': {
                'time': 0.1,
                'error': 1.0,
                'stability': 0.0
            }
            }
    train(args)