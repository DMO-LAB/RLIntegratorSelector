import numpy as np
import time
import psutil
import os
import gc
from environment import create_env, SimulationSettings
from dataclasses import dataclass

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def memory_cleanup():
    """Aggressive memory cleanup"""
    gc.collect()

@dataclass
class Args:
    output_dir: str = 'test_output'
    t_end: float = 0.06
    n_points: int = 50
    global_timestep: float = 1e-5
    T_fuel: float = 600
    T_oxidizer: float = 1200
    pressure: float = 101325
    profile_interval: int = 20

def test_environment_memory():
    print(f"Initial memory: {get_memory_usage():.2f} MB")
    memory_cleanup()
    
    # Create environment
    args = Args()
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
    
    print(f"Memory after env creation: {get_memory_usage():.2f} MB")
    memory_cleanup()
    
    # Run environment for multiple episodes
    n_episodes = 5
    steps_per_episode = 6000
    
    for episode in range(n_episodes):
        print(f"\nStarting episode {episode}")
        state, _ = env.reset()
        print(f"Memory after reset: {get_memory_usage():.2f} MB")
        
        for step in range(steps_per_episode):
            # Random actions
            default_action = 2
            if default_action == 0:
                action = [0] * env.sim_settings.n_points
            elif default_action == 1:
                action = [1] * env.sim_settings.n_points
            else:
                action = env.action_space.sample()
            
            # Step environment
            next_state, reward, done, truncated, info = env.step(action)
            
            
            # Log every 500 steps
            if step % 500 == 0:
                print(f"Episode {episode} - Step {step} - Memory: {get_memory_usage():.2f} MB")
                # memory_cleanup()
            
            if done:
                break
                
            state = next_state
        
        print(f"Episode {episode} completed - Final memory: {get_memory_usage():.2f} MB")

if __name__ == "__main__":
    test_environment_memory()