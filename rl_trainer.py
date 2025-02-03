import wandb
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.logger import configure
import os
from datetime import datetime
import matplotlib.pyplot as plt
from environment import create_randomized_env, SimulationSettings, SimulationConfig
from typing import Optional
import shutil
from config import TrainingConfig, create_default_config, NetworkConfig, PPOConfig, FeatureConfig, RewardConfig

class CombustionCallback(BaseCallback):
    def __init__(self, eval_env, log_dir: str, eval_freq: int = 1000):
        super().__init__(verbose=1)
        self.eval_env = eval_env
        self.log_dir = log_dir
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.n_steps = 0
        
        # Create directories
        self.plot_dir = os.path.join(log_dir, "plots")
        os.makedirs(self.plot_dir, exist_ok=True)
    
    def _on_step(self) -> bool:
        self.n_steps += self.eval_env.num_envs

        if self.n_steps % self.eval_freq == 0:
            print(f"\nEvaluating at step {self.n_steps}...")
            self._evaluate()
            path = os.path.join(self.log_dir, f"model_step_{self.n_steps}.zip")
            self.model.save(path)
            print(f"Model saved to {path}")

        return True

    def _evaluate(self):
        """Run evaluation and log results"""
        eval_rewards = []
        
        for _ in range(1):
            obs = self.eval_env.reset()
            done = False
            episode_reward = 0
            all_eval_rewards = []
            episode_actions = []
            count = 0
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                if count % 1000 == 0:
                    print(f"Action at step {count}: {action}")
                obs, reward, done, info = self.eval_env.step(action)
                all_eval_rewards.append(reward)
                episode_reward += np.mean(reward)
                episode_actions.append(action)
                count += 1
                done = np.any(done)

            self.eval_env.env.render(
                save_path=os.path.join(self.log_dir, f"plots/eval_step_{self.n_steps}.png")
            )
            eval_rewards.append(episode_reward)
        
        mean_reward = np.mean(eval_rewards)
        std_reward = np.std(eval_rewards)
        
        self.logger.record("eval/mean_reward", mean_reward)
        self.logger.record("eval/std_reward", std_reward)
        print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")
        
        # Plot evaluation results
        self._plot_actions(episode_actions, self.n_steps, self.plot_dir)
        self._plot_rewards(all_eval_rewards, self.n_steps, self.plot_dir)
        
        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            best_model_path = os.path.join(self.log_dir, "best_model")
            self.model.save(best_model_path)
            # Save the current evaluation plots with the best model
            shutil.copy(
                os.path.join(self.plot_dir, f"actions_{self.n_steps}.png"),
                os.path.join(self.plot_dir, "best_model_actions.png")
            )
            shutil.copy(
                os.path.join(self.plot_dir, f"rewards_{self.n_steps}.png"),
                os.path.join(self.plot_dir, "best_model_rewards.png")
            )

    def _plot_actions(self, actions, episode, save_dir):
        """Plot actions distribution over time"""
        steps_per_space = len(actions) // 8
        times_to_plot = [i * steps_per_space for i in range(8)]
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
        """Plot rewards distribution over time"""
        steps_per_space = len(rewards) // 8
        times_to_plot = [i * steps_per_space for i in range(8)]
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

def setup_environment(config: TrainingConfig):
    """Setup training environment based on configuration"""
    # Create simulation settings
    sim_settings = SimulationSettings(
        output_dir=config.output_dir,
        n_threads=config.n_threads,
        n_points=config.n_points,
        global_timestep=config.global_timestep,
        profile_interval=config.profile_interval,
        equilibrate_counterflow=False,
        center_width=0.002,
        slope_width=0.001
    )
    
    # Create environment
    env = create_randomized_env(
        base_settings=sim_settings,
        sim_configs=config.sim_configs,
        species_to_track=config.species_to_track,
        features_config=config.features.__dict__,
        reward_config=config.reward.__dict__
    )
    
    return env

def train_combustion_rl(config: TrainingConfig, env=None):
    """Main training function using configuration"""
    # Initialize wandb if enabled
    if config.use_wandb:
        wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=config.exp_name,
            config=config.__dict__
        )
    
    # Create log directory
    log_dir = os.path.join(
        'logs',
        f"{config.exp_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    os.makedirs(log_dir, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(log_dir, "config.yaml")
    config.save(config_path)
    print(f"Configuration saved to {config_path}")
    
    # Configure logger
    new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
    
    # Create environment if not provided
    if env is None:
        env = setup_environment(config)
    
    # Configure model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,
        **config.get_ppo_kwargs(),

    )
    
    # Set the new logger
    model.set_logger(new_logger)
    
    # Calculate evaluation frequency based on environment steps
    eval_freq = env.num_envs * config.eval_freq
    
    # Setup callbacks
    callbacks = [
        CombustionCallback(
            eval_env=env,
            log_dir=log_dir,
            eval_freq=eval_freq
        ),
        CheckpointCallback(
            save_freq=config.save_freq,
            save_path=os.path.join(log_dir, "checkpoints"),
            name_prefix="combustion_model"
        )
    ]
    
    try:
        # Train model
        model.learn(
            total_timesteps=config.ppo.total_timesteps,
            callback=callbacks,
            progress_bar=True,
            log_interval=config.log_interval,
            tb_log_name=config.exp_name
        )
        
        # Save final model
        final_model_path = os.path.join(log_dir, "final_model")
        model.save(final_model_path)
        print(f"Final model saved to {final_model_path}")
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if config.use_wandb:
            wandb.finish()
        
        # Clean up environment
        if env:
            env.close()
    
    return model, log_dir

def load_and_test_model(model_path: str, config_path: str, n_episodes: int = 5):
    """Load and test a trained model"""
    # Load configuration
    config = TrainingConfig.load(config_path)
    
    # Setup environment
    env = setup_environment(config)
    
    # Load model
    model = PPO.load(model_path, env=env)
    
    results = []
    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += np.mean(reward)
            steps += 1
            done = np.any(done)
        
        results.append({
            'episode': episode,
            'total_reward': total_reward,
            'steps': steps
        })
        print(f"Episode {episode}: Total Reward = {total_reward:.2f}, Steps = {steps}")
    
    return results

if __name__ == "__main__":
    # Create default configuration
    config = create_default_config()
    
    # Modify configuration to match original settings
    config.exp_name = "combustion_control"
    config.output_dir = f"experiments/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    config.use_wandb = False
    config.wandb_project = "combustion_rl"
    
    # Environment settings
    config.n_points = 50
    config.n_threads = 2
    config.global_timestep = 1e-5
    config.profile_interval = 100
    
    # PPO settings
    config.ppo = PPOConfig(
        learning_rate=1e-3,
        n_steps=1000,
        batch_size=50_000,
        n_epochs=8,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        ent_coef=0.05,
        vf_coef=0.5,
        max_grad_norm=0.5,
        total_timesteps=50_000_000
    )
    
    # Network settings
    config.network = NetworkConfig(
        pi_layers=[256, 128, 64],
        vf_layers=[256, 128, 64],
        activation_fn="ReLU"
    )
    
    # Feature settings
    config.features = FeatureConfig(
        local_features=True,
        neighbor_features=False,
        gradient_features=True,
        temporal_features=False,
        window_size=5
    )
    
    # Reward settings
    config.reward = RewardConfig(
        weights={
            'accuracy': 1,
            'efficiency': 3,
        },
        thresholds={
            'time': 0.001,
            'error': 1
        },
        scaling={
            'time': 1,
            'error': 1
        },
        use_neighbors=True,
        neighbor_weight=0.3,
        neighbor_radius=4
    )
    
    # Species to track
    config.species_to_track = ['CH4', 'CO2', 'HO2', 'H2O2', 'OH', 'O2', 'H2', 'H2O']
    
    print(f"Starting training with {config.ppo.total_timesteps} timesteps...")
    model, log_dir = train_combustion_rl(config)
    
    print("\nTraining complete. Testing model...")
    test_results = load_and_test_model(
        model_path=os.path.join(log_dir, "final_model"),
        config_path=os.path.join(log_dir, "config.yaml"),
        n_episodes=3
    )
    print("\nTest results:", test_results)