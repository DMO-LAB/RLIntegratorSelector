import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cantera as ct
import time
import pickle
from typing import Dict, List, Tuple, Optional, Union
from collections import deque
from ember import Config, Paths, InitialCondition, StrainParameters, General, \
    Times, TerminationCondition, ConcreteConfig, Debug, RK23Tolerances, QssTolerances, CvodeTolerances, _ember
from stable_baselines3.common.vec_env import VecEnv
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import h5py
import os
from datetime import datetime
from utils import take_step, SimulationStep, SimulationData, SimulationSettings, IntegratorOption
import hashlib
import json
from tqdm import tqdm
def _create_config(sim_settings: SimulationSettings, rtol=1e-6, atol=1e-8):
        """Create Ember configuration"""
        os.makedirs(sim_settings.output_dir, exist_ok=True)
        return ConcreteConfig(Config(
            Paths(outputDir=sim_settings.output_dir),
            General(nThreads=sim_settings.n_threads,
                    chemistryIntegrator='cvode'),
            InitialCondition(
                Tfuel=sim_settings.T_fuel,
                Toxidizer=sim_settings.T_oxidizer,
                pressure=sim_settings.pressure,
                nPoints=sim_settings.n_points,
                xLeft=sim_settings.x_left,
                xRight=sim_settings.x_right,
                flameType='diffusion',
                centerWidth=0.0,
                slopeWidth=0.0,
                equilibrateCounterflow=False,
            ),
            StrainParameters(final=100,
                     initial=100),
            Times(
                globalTimestep=sim_settings.global_timestep,
                profileStepInterval=sim_settings.profile_interval,
            ),
            TerminationCondition(tEnd=sim_settings.t_end,
                                 tolerance=0.0,
                                 abstol=0.0,
                                 steadyPeriod=1.0,
                                 dTdtTol=0),
            CvodeTolerances(
                relativeTolerance=rtol,
                momentumAbsTol=atol,
                energyAbsTol=atol,
                speciesAbsTol=atol,
                minimumTimestep=1e-18,
                maximumTimestep=1e-5
            ),
            RK23Tolerances(
                relativeTolerance=rtol,
                absoluteTolerance=atol,
                minimumTimestep=1e-10,
                maximumTimestep=1e-4,
                maxStepsNumber=100000,
            ),
            QssTolerances(
                abstol=atol
            )
        ))

def run_benchmark(sim_settings: SimulationSettings,
                 species_to_track: List[str],
                 species_index: Dict[str, int],
                 output_dir: str,
                 filename: Optional[str] = None) -> SimulationData:
    """Run benchmark simulation and return data"""
    # Create data holder
    benchmark_data = SimulationData(
        species_names=species_to_track,
        n_points=sim_settings.n_points,
        n_steps=int(sim_settings.t_end / sim_settings.global_timestep),
        output_dir=output_dir
    )
    
    if filename is not None and os.path.exists(filename):
        benchmark_data = SimulationData.load_from_hdf5(filename)
    else:
        # Create and initialize solver with tight tolerances
        solver = _ember.FlameSolver(_create_config(
            sim_settings, rtol=1e-10, atol=1e-12
        ))
        solver.initialize()
        
        integrator_types = ['cvode'] * sim_settings.n_points
        # Run simulation
        step_count = 0
        done = False
        start_time = time.time()
        print(f"[INFO] Running benchmark simulation with {sim_settings.n_points} points")
        while not done:
            current_time = step_count * sim_settings.global_timestep
            data_holder, done, cpu_time = take_step(step_count, current_time, solver, benchmark_data, integrator_types, species_index, species_to_track)
            step_count += 1
        end_time = time.time()
        benchmark_data.save_to_hdf5(filename)
        print(f"[INFO] Benchmark simulation completed in {end_time - start_time:.2f} seconds and ended at step {step_count}")
    return benchmark_data


class VectorizedCombustionEnv(gym.Env):
    """
    Vectorized 1D combustion environment where each grid point is treated independently.
    This allows the learned policy to generalize to higher dimensions.
    """
    metadata = {'render.modes': ['human']}
    
    def __init__(self, 
                 sim_settings: SimulationSettings,
                 benchmark_data: SimulationData,
                 species_to_track: List[str] = None,
                 species_index: Dict[str, int] = None,
                 error_thresholds: Dict[str, float] = None,
                 features_config: dict = None,
                 reward_config: dict = None,
                 save_step_data: bool = False):
        
        super(VectorizedCombustionEnv, self).__init__()
        
        # Store configurations
        self.sim_settings = sim_settings
        self.benchmark_data = benchmark_data
        self.species_to_track = species_to_track
        self.species_index = species_index
        self.species_indices = {spec: species_index[spec] for spec in species_to_track}
        self.save_step_data = save_step_data
        
        # Default feature configuration
        self.features_config = features_config or {
            'local_features': True,     # Temperature, species, phi at point
            'neighbor_features': False,   # Values from neighboring points
            'gradient_features': False,   # Local gradients
            'temporal_features': False,   # Historical changes
            'window_size': 5            # Size of history window
        }
        
        # Default error thresholds
        self.error_thresholds = error_thresholds or {
            'temperature': 1e-3,
            'species': 1e-4,
            'gradient': 1e-3
        }
        
        # Default reward configuration
        self.reward_config = reward_config or {
            'weights': {
                'accuracy': 0.6,    # Weight for accuracy component
                'efficiency': 0.3,  # Weight for computational efficiency
                'stability': 0.1    # Weight for numerical stability
            },
            'thresholds': {
                'time': 0.01,      # Expected computation time per step
                'error': 1e-3      # Error tolerance
            },
            'scaling': {
                'time': 0.1,       # Scaling factor for time penalty
                'error': 1.0       # Scaling factor for error penalty
            }
        }
        
        # Setup spaces
        self._setup_spaces()
        
        # Initialize episode storage
        if self.save_step_data:
            self._initialize_storage()
        
        # Initialize feature history
        self._initialize_history()
    
    def _setup_spaces(self):
        """Setup action and observation spaces"""
        n_points = self.sim_settings.n_points
        
        # Define integrator options
        self.integrator_options = [
            IntegratorOption("CVODE-Tight", "cvode", 1e-6, 1e-8, 'blue'),
            # IntegratorOption("CVODE-Loose", "cvode", 1e-6, 1e-8, 'green'),
            IntegratorOption("BoostRK-Tight", "boostRK", 1e-6, 1e-8, 'red'),
            # IntegratorOption("BoostRK-Loose", "boostRK", 1e-6, 1e-8, 'yellow')
        ]
        
        # Action space: each point can choose its own integrator
        self.action_space = spaces.MultiDiscrete([len(self.integrator_options)] * n_points)
        
        # Calculate observation size per point
        obs_size = self._calculate_observation_size()
        
        # Observation space: matrix where each row represents a point
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(n_points, obs_size),
            dtype=np.float32
        )
    
    def _calculate_observation_size(self) -> int:
        """Calculate total size of observation vector per point"""
        size = 0
        
        if self.features_config['local_features']:
            size += 1 + len(self.species_to_track)  # T + species
            size += 1  # phi
        
        if self.features_config['neighbor_features']:
            size += 2 * (1 + len(self.species_to_track))  # Values from both neighbors
        
        if self.features_config['gradient_features']:
            size += 1 + len(self.species_to_track)  # Gradients of T and species
        
        if self.features_config['temporal_features']:
            size += 2  # Previous time derivatives
        
        return size
    
    def _initialize_storage(self):
        """Initialize data storage for current episode"""
        self.episode_data = SimulationData(
            species_names=self.species_to_track,
            n_points=self.sim_settings.n_points,
            n_steps=int(self.sim_settings.t_end / self.sim_settings.global_timestep),
            output_dir=self.sim_settings.output_dir
        )
    
    def _initialize_history(self):
        """Initialize history buffers for temporal features"""
        n_points = self.sim_settings.n_points
        window_size = self.features_config['window_size']
        
        self.history = {
            'temperature': np.zeros((n_points, window_size)),
            'species': {spec: np.zeros((n_points, window_size)) 
                       for spec in self.species_to_track},
            'gradients': np.zeros((n_points, window_size)),
            'errors': np.zeros((n_points, window_size)),
            'rewards': np.zeros((n_points, window_size))
        }
        
        self.history_index = 0
    
    def _update_history(self):
        """Update history buffers with current state"""
        idx = self.history_index % self.features_config['window_size']
        
        # Store current values
        self.history['temperature'][:, idx] = self.solver.T
        for spec in self.species_to_track:
            self.history['species'][spec][:, idx] = self.solver.Y[self.species_indices[spec]]
        
        # Calculate and store gradients
        self.history['gradients'][:, idx] = np.gradient(self.solver.T)
        
        self.history_index += 1
    
    def _get_point_observation(self, point_idx: int) -> np.ndarray:
        """Get observation vector for a specific point"""
        features = []
        
        if self.features_config['local_features']:
            # Local temperature and species
            features.append(self.solver.T[point_idx] / 300)
            for spec in self.species_to_track:
                Y = self.solver.Y[self.species_indices[spec]][point_idx]
                features.append(Y)
            
            # Local phi
            phi = self.solver.phi[point_idx]
            phi = np.maximum(phi, 1e-3)
            phi = np.minimum(phi, 1)
            features.append(phi)
        
        if self.features_config['neighbor_features']:
            # Add features from neighboring points
            for offset in [-1, 1]:
                idx = max(0, min(point_idx + offset, self.sim_settings.n_points - 1))
                features.append(self.solver.T[idx] / 300)
                for spec in self.species_to_track:
                    Y = self.solver.Y[self.species_indices[spec]][idx]
                    features.append(Y)

        if self.features_config['gradient_features']:
            # Local gradients
            dT = np.gradient(self.solver.T)[point_idx]
            features.append(np.log1p(np.abs(dT)))
            
            for spec in self.species_to_track:
                dY = np.gradient(self.solver.Y[self.species_indices[spec]])[point_idx]
                features.append(np.log1p(np.abs(dY)))
        
        if self.features_config['temporal_features']:
            # Historical features
            if self.history_index >= 2:
                idx = (self.history_index - 1) % self.features_config['window_size']
                prev_idx = (self.history_index - 2) % self.features_config['window_size']
                
                dT_dt = (self.history['temperature'][point_idx, idx] - 
                        self.history['temperature'][point_idx, prev_idx]) / self.sim_settings.global_timestep
                
                features.extend([
                    np.log10(max(abs(dT_dt), 1e-20)),
                    self.history['errors'][point_idx, idx]
                ])
            else:
                features.extend([0.0, 0.0])
        
        return np.array(features, dtype=np.float32)
    
    def _get_observation(self) -> np.ndarray:
        """Get observations for all points"""
        return np.vstack([
            self._get_point_observation(i) 
            for i in range(self.sim_settings.n_points)
        ])
        
    def reward_function(self, error, cpu_time):
        # Normalize errors to typical ranges
        log_error = -np.log10(max(error, 1e-20))
        log_time = -np.log10(max(cpu_time, 1e-7))

        # Sigmoid scaling for error component
        error_component = 2.0 / (1 + np.exp(-log_error)) - 1

        # Logarithmic scaling for time
        time_component = log_time  # Normalize to ~[-1,1] range

        # Combined reward with adaptive weighting
        error_weight = 0.4 + 0.6 * (error > 10)  # Increase weight in reaction zones
        time_weight = (1.0 - error_weight) 

        reward = error_weight * error_component + time_weight * time_component

        # Additional penalty for catastrophic errors
        if error > 10:
            reward += log_error*10
                
        return reward * 0.1

    
    def _calculate_point_error(self, point_idx: int) -> Tuple[float, dict, float]:
        """Calculate error metrics for a specific point"""
        benchmark_step = self.benchmark_data.get_step(self.current_step)
        
        # Temperature relative error (as percentage)
        benchmark_T = benchmark_step.temperatures[point_idx]
        T_error = abs(self.solver.T[point_idx] - benchmark_T) / benchmark_T * 100
        
        # Species relative errors (as percentage)
        species_errors = {}
        for spec in self.species_to_track:
            current_Y = self.solver.Y[self.species_indices[spec]][point_idx]
            bench_Y = benchmark_step.species_mass_fractions[spec][point_idx]
            # Add small epsilon to avoid division by zero for trace species
            epsilon = np.max([bench_Y, 1e-5])
            species_errors[spec] = abs(current_Y - bench_Y) / epsilon * 100
        
        # Gradient relative error (as percentage)
        current_grad = np.gradient(self.solver.T)[point_idx]
        bench_grad = np.gradient(benchmark_step.temperatures)[point_idx]
        # Add small epsilon to avoid division by zero at flat regions
        epsilon = np.max([bench_grad, 1e-5])
        grad_error = abs(current_grad - bench_grad) / epsilon * 100

        return T_error, species_errors, grad_error
    
    def _compute_point_reward(self, point_idx: int, cpu_time: float, 
                         T_error: float, species_errors: dict, grad_error: float,
                         neighbor_radius: int = 4) -> float:
        """Compute reward for a specific point"""
        # Calculate total error including neighbors if enabled
        point_error = T_error + np.mean(list(species_errors.values())) + grad_error
        
        if self.reward_config.get('use_neighbors', False):
            # Get neighboring points and their errors
            n_points = self.sim_settings.n_points
            start_idx = max(0, point_idx - neighbor_radius)
            end_idx = min(n_points, point_idx + neighbor_radius + 1)
            
            neighbor_errors = []
            neighbor_weights = []
            for idx in range(start_idx, end_idx):
                if idx != point_idx:
                    T_err, spec_err, grad_err = self._calculate_point_error(idx)
                    total_err = T_err + np.mean(list(spec_err.values())) + grad_err
                    neighbor_errors.append(total_err)
                    
                    distance = abs(idx - point_idx)
                    weight = np.exp(-distance / neighbor_radius)
                    neighbor_weights.append(weight)
            
            if neighbor_errors:
                neighbor_weights = np.array(neighbor_weights)
                neighbor_weights /= np.sum(neighbor_weights)
                total_error = (1 - self.reward_config['neighbor_weight']) * point_error + \
                            self.reward_config['neighbor_weight'] * np.sum(np.array(neighbor_errors) * neighbor_weights)
            else:
                total_error = point_error
        else:
            total_error = point_error

        # Use new reward computation
        return self.reward_function(total_error, cpu_time)

    def _setup_default_reward_config(self):
        """Setup default reward configuration"""
        return {
            'use_neighbors': True,           # Whether to use neighbor influence
            'neighbor_weight': 0.3,          # Weight of neighbor influence (0 to 1)
            'neighbor_radius': 2,            # Number of neighbors to consider on each side
            'error_threshold': 1e-3,         # Minimum error threshold
            'time_threshold': 0.01,          # Time scaling threshold
            'time_exponent': 0.1,           # Exponent for time scaling
        }
    
    def step(self, action: np.ndarray):
        """Take a step using different integrators for each point"""
        try:
            # Set integrators based on action
            integrator_types = []
            for a in action:
                integrator = self.integrator_options[a]
                integrator_types.append(integrator.type)
        
            self.last_action = action
            # Take step
            start_time = time.time()
            self.solver.set_integrator_types(integrator_types)
            done = self.solver.step()

            truncated = False
            
            cpu_time = self.solver.gridPointIntegrationTimes
            
            # Calculate errors and rewards for each point
            rewards = np.zeros(self.sim_settings.n_points)
            errors = np.zeros(self.sim_settings.n_points)
            
            for i in range(self.sim_settings.n_points):
                T_error, species_errors, grad_error = self._calculate_point_error(i)
                errors[i] = T_error + np.mean(list(species_errors.values())) + grad_error
                rewards[i] = self._compute_point_reward(i, cpu_time[i], T_error, species_errors, grad_error,
                                                        neighbor_radius=self.reward_config['neighbor_radius'])
                rewards[i] = rewards[i] - np.maximum(0, np.log10(np.maximum(errors[i], 1e-10)))/10
            # Update history
            self._update_history()
            
            # Store step data
            if self.save_step_data:
                step_data = SimulationStep(
                    step=self.current_step,
                    time=self.current_step * self.sim_settings.global_timestep,
                    temperatures=self.solver.T,
                    species_mass_fractions={
                        spec: self.solver.Y[self.species_indices[spec]]
                        for spec in self.species_to_track
                    },
                    phi=self.solver.phi,
                    spatial_points=self.solver.x,
                    cpu_time=cpu_time,
                    integrator_type=str(action),
                    error=errors
                )
                self.episode_data.add_step(step_data)
            
            # Get next observation
            observation = self._get_observation()
            
            info = [{'cpu_time': cpu_time[i], 'point_errors': errors[i], 'point_rewards': rewards[i], 'action': action[i],
                    'integrator_type': self.integrator_options[action[i]].name} for i in range(self.sim_settings.n_points)]
            
            self.current_step += 1
            
            
            if self.current_step >= self.end_step:
                print(f"Episode Done {self.current_step} - resetting environment")
                done = True
                truncated = True
                rewards = rewards - np.maximum(0, np.log10(np.maximum(errors, 1e-10))) * 100
                # delete the output directory
                import shutil
                if os.path.exists(self.sim_settings.output_dir):
                   
                    shutil.rmtree(self.sim_settings.output_dir)
            else:
                truncated = False
                
            return observation, rewards, done, truncated, info
            
        except Exception as e:
            print(f"Integration failed: {e}")
            import traceback;
            traceback.print_exc()
            return (
                self._get_observation(),
                np.full(self.sim_settings.n_points, -100.0),
                True,
                True,
                {'error': float('inf'), 'cpu_time': 0.0}
            )
    
    def reset(self, seed=None, options=None):
        """Reset the environment"""
        super().reset(seed=seed)
        # # Initialize solver
        self.solver = _ember.FlameSolver(
            _create_config(self.sim_settings)
        )
        self.solver.initialize()
        
        self.solver.set_integrator_types(['cvode'] * self.sim_settings.n_points)
        self.solver.step()
        # Reset data storage
        if self.save_step_data:
            self.episode_data = None
            self._initialize_storage()
        
        # Reset tracking variables
        self.current_step = 1
        self.end_step = int(self.sim_settings.t_end / self.sim_settings.global_timestep)
        self._initialize_history()

        print(f"[CRITICAL INFO] Resetting environment with current step {self.current_step} and end step {self.end_step} - benchmark data steps {self.benchmark_data.n_steps}")
        
        return self._get_observation(), {}
    
    def render(self, save_path: str = None):
        """Visualize the current state with detailed per-point information"""
        import matplotlib.pyplot as plt
        
        # Create a figure with multiple subplots
        fig = plt.figure(figsize=(15, 10))
        gs = plt.GridSpec(3, 2)
        
        # Temperature profile
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(self.solver.x, self.solver.T, 'b-', label='Current')
        # check the max step in the benchmark data
        if self.current_step >= self.benchmark_data.n_steps:
            current_step = self.benchmark_data.n_steps - 1
            print(f"[WARNING] Current step {self.current_step} is greater than benchmark data steps {self.benchmark_data.n_steps}, using last step")
        else:
            print(f"[INFO] Current step {self.current_step} is less than benchmark data steps {self.benchmark_data.n_steps}, using current step")
            current_step = self.current_step
        ax1.plot(self.solver.x, self.benchmark_data.get_step(current_step).temperatures, 
                'k--', label='Benchmark')
        ax1.set_ylabel('Temperature (K)')
        ax1.set_title('Temperature Profile')
        ax1.legend()
        
        # Species profiles
        ax2 = fig.add_subplot(gs[1, :])
        for spec in ['CH4', 'O2', 'H2O']:
            current_Y = self.solver.Y[self.species_indices[spec]]

            benchmark_Y = self.benchmark_data.get_step(current_step).species_mass_fractions[spec]
            ax2.semilogy(self.solver.x, current_Y, '-', label=f'{spec} Current')
            ax2.semilogy(self.solver.x, benchmark_Y, '--', label=f'{spec} Benchmark')
        ax2.set_ylabel('Mass Fractions')
        ax2.set_title('Species Profiles')
        ax2.legend()
        
        # Error distribution
        ax3 = fig.add_subplot(gs[2, 0])
        T_errors = np.abs(self.solver.T - 
                         self.benchmark_data.get_step(current_step).temperatures)
        ax3.plot(self.solver.x, T_errors, 'r-', label='Temperature Error')
        ax3.set_xlabel('Position (m)')
        ax3.set_ylabel('Absolute Error')
        ax3.set_title('Error Distribution')
        ax3.set_yscale('log')
        ax3.legend()
        
        # Integrator distribution
        ax4 = fig.add_subplot(gs[2, 1])
        unique_integrators = [opt.name for opt in self.integrator_options]
        integrator_counts = np.zeros(len(unique_integrators))
        current_types = [self.integrator_options[a].name for a in self.last_action]
        for i, integ in enumerate(unique_integrators):
            integrator_counts[i] = current_types.count(integ)
        
        ax4.bar(unique_integrators, integrator_counts)
        ax4.set_xlabel('Integrator Type')
        ax4.set_ylabel('Count')
        ax4.set_title('Current Integrator Distribution')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    
    def save_episode(self, filename: str = None):
        """Save episode data with detailed statistics"""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.sim_settings.output_dir, 
                                  f'episode_data_{timestamp}.h5')
        
        # Save simulation data
        self.episode_data.save_to_hdf5(filename)
        
        # step = self.benchmark_data.get_step(self.current_step)
        # Compute and save additional statistics
        try:
            stats = {
                'total_cpu_time': self.episode_data.get_performance_metrics()['total_cpu_time'],
                'mean_error': self.episode_data.get_performance_metrics()['mean_error'],
                'max_error': self.episode_data.get_performance_metrics()['max_error'],
                'integrator_usage': self.episode_data.get_performance_metrics()['integrator_usage'],
            }
            
            # Save stats in a separate file
            stats_file = filename.replace('.h5', '_stats.pkl')
            with open(stats_file, 'wb') as f:
                pickle.dump(stats, f)
        except Exception as e:
            print(f"Error saving stats")
            
        # reset episode data
        self.episode_data = None
        self._initialize_storage()
        
        return filename
    
    def _compute_integrator_usage(self) -> Dict[str, float]:
        """Compute statistics about integrator usage"""
        total_steps = (self.current_step+1) * self.sim_settings.n_points
        usage = {}
        
        for opt in self.integrator_options:
            print(opt.name)
            count = sum(1 for step in self.episode_data.steps
                       for integ in step.integrator_type 
                       if integ == opt.name)
            usage[opt.name] = count / total_steps
        
        return usage
    
    def _compute_performance_metrics(self) -> Dict[str, float]:
        """Compute detailed performance metrics"""
        metrics = {}
        
        # Time metrics
        step_times = [step.cpu_time for step in self.episode_data.steps]
        metrics['mean_step_time'] = np.mean(step_times)
        metrics['std_step_time'] = np.std(step_times)
        metrics['max_step_time'] = np.max(step_times)
        
        # Error metrics
        errors = [step.error for step in self.episode_data.steps]
        metrics['mean_error'] = np.mean(errors)
        metrics['std_error'] = np.std(errors)
        metrics['max_error'] = np.max(errors)
        
        # Stability metrics
        temp_changes = []
        for i in range(0, len(self.episode_data.steps)):
            temp_change = np.abs(self.episode_data.steps[i].temperatures - 
                               self.episode_data.steps[i-1].temperatures)
            temp_changes.append(np.mean(temp_change))
        
        metrics['mean_temp_change'] = np.mean(temp_changes)
        metrics['max_temp_change'] = np.max(temp_changes)
        
        return metrics
    
    def close(self):
        """Clean up resources"""
        # Any cleanup needed
        pass
    
    def save_episode_data(self, filename: str = None):
        """Save episode data"""
        self.episode_data.save_to_hdf5(filename)


class VectorizedGridEnvWrapper(VecEnv):
    def __init__(self, env):
        # Create a new observation space that represents a single point
        single_point_obs_space = env.observation_space
        if len(single_point_obs_space.shape) > 1:
            # If shape is (n_points, obs_dim), take just obs_dim
            single_point_obs_space = spaces.Box(
                low=env.observation_space.low[0],
                high=env.observation_space.high[0],
                shape=(env.observation_space.shape[1],),
                dtype=env.observation_space.dtype
            )
        
        # Make sure action space is a single Discrete space per point
        if isinstance(env.action_space, spaces.MultiDiscrete):
            action_space = spaces.Discrete(env.action_space.nvec[0])
        else:
            action_space = env.action_space

        super().__init__(
            num_envs=env.sim_settings.n_points,
            observation_space=single_point_obs_space,
            action_space=action_space  # Use single point action space
        )
        self.env = env
        self.sim_settings = env.sim_settings
        self.needs_reset = False
    def step_async(self, actions):
        if self.needs_reset:
            # Reset environment if needed
            self._last_obs, _ = self.env.reset()
            self.needs_reset = False
            
        # Ensure actions are 1D array of length num_envs
        if len(actions.shape) > 1:
            actions = actions.flatten()
        self.actions = actions.astype(np.int32)
        
    def step_wait(self):
        obs, rewards, dones, truncated, infos = self.env.step(self.actions)
        # Ensure observations are properly shaped
        if len(obs.shape) == 2:
            obs = obs.reshape(self.num_envs, -1)
    
        if dones:
            self.needs_reset = True
        dones = np.full(self.num_envs, dones)
        
        # Create proper info dict
        if not isinstance(infos, list):
            infos = [infos for _ in range(self.num_envs)]
            
        return obs, rewards, dones, infos
    
    def reset(self, seed=None, options=None):
        if self.needs_reset:
            print("Resetting environment from sb3")
            obs, info = self.env.reset(seed=seed, options=options)
            self.needs_reset = False
        else:
            obs, info = self.env.reset()
        # Ensure observations are properly shaped
        if len(obs.shape) == 2:  # If obs is (n_points, obs_dim)
            obs = obs.reshape(self.num_envs, -1)  # Reshape to (n_envs, obs_dim)
        return obs
    
    def close(self):
        self.env.close()
        
    def env_is_wrapped(self, wrapper_class, indices=None):
        """Check if environment is wrapped"""
        return False
        
    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        """Call environment method"""
        return [getattr(self.env, method_name)(*method_args, **method_kwargs) 
                for _ in range(self.num_envs)]
    
    def get_attr(self, attr_name, indices=None):
        """Get environment attribute"""
        return [getattr(self.env, attr_name) for _ in range(self.num_envs)]
    
    def set_attr(self, attr_name, value, indices=None):
        """Set environment attribute"""
        setattr(self.env, attr_name, value)

    
    def seed(self, seed=None):
        """Set environment seed"""
        if hasattr(self.env, 'seed'):
            return [self.env.seed(seed) for _ in range(self.num_envs)]
        return [None for _ in range(self.num_envs)]

def create_env(sim_settings: SimulationSettings,
              benchmark_file: str = None,
              species_to_track: List[str] = None,
              features_config: dict = None,
              reward_config: dict = None,
              save_step_data: bool = False) -> VectorizedCombustionEnv:
    """Helper function to create and initialize the environment"""
    
    gas = ct.Solution('gri30.yaml')
    species_to_track = species_to_track if species_to_track is not None else ['CH4', 'O2', 'CO2', 'H2O']
    species_index = {spec: gas.species_index(spec) for spec in species_to_track} 
    
    # Run or load benchmark
    benchmark_data = run_benchmark(
        sim_settings=sim_settings,
        species_to_track=species_to_track,
        species_index=species_index,
        output_dir='benchmarks',
        filename=benchmark_file
    )
    
    # Create environment
    env = VectorizedCombustionEnv(
        sim_settings=sim_settings,
        benchmark_data=benchmark_data,
        species_to_track=species_to_track,
        species_index=species_index,
        features_config=features_config,
        reward_config=reward_config,
        save_step_data=save_step_data
    )
    
    env = VectorizedGridEnvWrapper(env)
    return env


@dataclass
class SimulationConfig:
    """Class to hold variable simulation parameters"""
    T_fuel: float
    T_oxidizer: float
    t_end: float
    pressure: float
    center_width: float
    slope_width: float
    equilibrate_counterflow: Union[bool, str]

    def get_hash(self) -> str:
        """Generate a unique hash for this configuration"""
        # Only include T_fuel and T_oxidizer in hash since t_end doesn't affect physics
        config_dict = {
            'T_fuel': self.T_fuel,
            'T_oxidizer': self.T_oxidizer,
            'pressure': self.pressure,
            'center_width': self.center_width,
            'slope_width': self.slope_width,
            'equilibrate_counterflow': self.equilibrate_counterflow
        }
        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()

class BenchmarkCache:
    """Class to manage cached benchmark simulations"""
    def __init__(self, cache_dir: str = 'benchmark_cache'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
    def get_cache_path(self, config: SimulationConfig) -> str:
        """Get path for cached benchmark file"""
        config_hash = config.get_hash()
        return os.path.join(self.cache_dir, f'benchmark_{config_hash}.h5')
    
    def is_cached(self, config: SimulationConfig) -> bool:
        """Check if benchmark exists in cache"""
        cache_path = self.get_cache_path(config)
        return os.path.exists(cache_path)
    
    def get_cached_benchmark(self, config: SimulationConfig) -> Optional[str]:
        """Get path to cached benchmark if it exists"""
        if self.is_cached(config):
            return self.get_cache_path(config)
        return None

def create_randomized_env(base_settings: SimulationSettings,
                         sim_configs: List[SimulationConfig],
                         species_to_track: List[str] = None,
                         features_config: dict = None,
                         reward_config: dict = None,
                         save_step_data: bool = False) -> VectorizedCombustionEnv:
    """Create environment with support for randomized episodes"""
    
    # Initialize benchmark cache
    cache = BenchmarkCache()
    
    # Create modified environment class with randomization support
    class RandomizedCombustionEnv(VectorizedCombustionEnv):
        def __init__(self, *args, **kwargs):
            self.sim_configs = sim_configs
            self.cache = cache
            self.current_config = None
            super().__init__(*args, **kwargs)
        
        def reset(self, seed=None, options=None):
            """Reset with random simulation config"""
            import numpy as np
            
            # Select random config
            if seed is not None:
                np.random.seed(seed)
            self.current_config = np.random.choice(self.sim_configs)
            
            # Update simulation settings
            self.sim_settings.T_fuel = self.current_config.T_fuel
            self.sim_settings.T_oxidizer = self.current_config.T_oxidizer
            self.sim_settings.t_end = self.current_config.t_end
            self.sim_settings.pressure = self.current_config.pressure
            self.sim_settings.center_width = self.current_config.center_width
            self.sim_settings.slope_width = self.current_config.slope_width
            self.sim_settings.equilibrate_counterflow = self.current_config.equilibrate_counterflow
            # Check for cached benchmark
            benchmark_path = self.cache.get_cached_benchmark(self.current_config)

            
            if benchmark_path is None:
                # Run new benchmark if not cached
                print(f"[INFO] Running new benchmark for config: T_fuel={self.current_config.T_fuel}, "
                      f"T_oxidizer={self.current_config.T_oxidizer}, "
                      f"t_end={self.current_config.t_end}, "
                      f"pressure={self.current_config.pressure}, "
                      f"center_width={self.current_config.center_width}, "
                      f"slope_width={self.current_config.slope_width}, "
                      f"equilibrate_counterflow={self.current_config.equilibrate_counterflow}")
                benchmark_path = self.cache.get_cache_path(self.current_config)
                self.benchmark_data = run_benchmark(
                    sim_settings=self.sim_settings,
                    species_to_track=self.species_to_track,
                    species_index=self.species_index,
                    output_dir='benchmarks',
                    filename=benchmark_path
                )
            else:
                # Load cached benchmark
                print(f"[INFO] Loading cached benchmark for config: T_fuel={self.current_config.T_fuel}, "
                      f"T_oxidizer={self.current_config.T_oxidizer}, "
                      f"t_end={self.current_config.t_end}, "
                      f"pressure={self.current_config.pressure}, "
                      f"center_width={self.current_config.center_width}, "
                      f"slope_width={self.current_config.slope_width}, "
                      f"equilibrate_counterflow={self.current_config.equilibrate_counterflow}")
                self.benchmark_data = SimulationData.load_from_hdf5(benchmark_path)
            
            return super().reset(seed=seed)
    

    gas = ct.Solution('gri30.yaml')
    species_to_track = species_to_track if species_to_track is not None else ['CH4', 'O2', 'CO2', 'H2O']
    species_index = {spec: gas.species_index(spec) for spec in species_to_track} 
    # Create environment instance
    env = RandomizedCombustionEnv(
        sim_settings=base_settings,
        benchmark_data=None,  # Will be set in reset()
        species_to_track=species_to_track,
        species_index=species_index,
        features_config=features_config,
        reward_config=reward_config,
        save_step_data=save_step_data
    )
    
    env = VectorizedGridEnvWrapper(env)
    return env

# Example usage:
if __name__ == "__main__":
    # Base simulation settings
    base_settings = SimulationSettings(
        output_dir='run/rl_test',
        n_points=100,
        global_timestep=1e-5,
        profile_interval=20,
        equilibrate_counterflow=False,
        center_width=0.002,
        slope_width=0.001
    )
    
    # Define possible simulation configurations
    sim_configs = [
        SimulationConfig(T_fuel=300, T_oxidizer=1200, t_end=0.05, pressure=101325, equilibrate_counterflow=False, center_width=0, slope_width=0),
        SimulationConfig(T_fuel=600, T_oxidizer=1300, t_end=0.05, pressure=101325, equilibrate_counterflow=False, center_width=0, slope_width=0),
        SimulationConfig(T_fuel=900, T_oxidizer=1100, t_end=0.05, pressure=101325, equilibrate_counterflow=False, center_width=0, slope_width=0),
        # SimulationConfig(T_fuel=1200, T_oxidizer=1000, t_end=0.025, pressure=101325, equilibrate_counterflow=False, center_width=0, slope_width=0),
        SimulationConfig(T_fuel=450, T_oxidizer=1500, t_end=0.05, pressure=101325, equilibrate_counterflow=False, center_width=0, slope_width=0),
        # SimulationConfig(T_fuel=750, T_oxidizer=1400, t_end=0.05, pressure=101325, equilibrate_counterflow=False, center_width=0, slope_width=0),
        # SimulationConfig(T_fuel=1050, T_oxidizer=1200, t_end=0.05, pressure=101325, equilibrate_counterflow=False, center_width=0, slope_width=0),
        # SimulationConfig(T_fuel=1350, T_oxidizer=1200, t_end=0.03, pressure=101325, equilibrate_counterflow=False, center_width=0, slope_width=0),
        SimulationConfig(T_fuel=1500, T_oxidizer=1500, t_end=0.05, pressure=101325, equilibrate_counterflow=False, center_width=0, slope_width=0),

        SimulationConfig(T_fuel=300, T_oxidizer=1200, t_end=0.05, pressure=101325, equilibrate_counterflow='TP', center_width=0.002, slope_width=0.001),
        # SimulationConfig(T_fuel=600, T_oxidizer=1300, t_end=0.05, pressure=101325, equilibrate_counterflow='TP', center_width=0.001, slope_width=0.0005),
        SimulationConfig(T_fuel=900, T_oxidizer=1100, t_end=0.05, pressure=101325, equilibrate_counterflow='TP', center_width=0.001, slope_width=0.0005),
        SimulationConfig(T_fuel=1200, T_oxidizer=1000, t_end=0.05, pressure=101325, equilibrate_counterflow='TP', center_width=0.005, slope_width=0.001),
        # SimulationConfig(T_fuel=450, T_oxidizer=1500, t_end=0.04, pressure=101325, equilibrate_counterflow='TP', center_width=0.001, slope_width=0.0005),
        # SimulationConfig(T_fuel=750, T_oxidizer=1400, t_end=0.04, pressure=101325, equilibrate_counterflow='TP', center_width=0.001, slope_width=0.0005),
        SimulationConfig(T_fuel=1050, T_oxidizer=1200, t_end=0.05, pressure=101325, equilibrate_counterflow='TP', center_width=0.001, slope_width=0.0005),
        SimulationConfig(T_fuel=1350, T_oxidizer=1200, t_end=0.05, pressure=101325, equilibrate_counterflow='TP', center_width=0.008, slope_width=0.003),
        #SimulationConfig(T_fuel=1500, T_oxidizer=1500, t_end=0.01, pressure=101325, equilibrate_counterflow='TP', center_width=0.001, slope_width=0.0005),
    ]
    
    # Create randomized environment
    env = create_randomized_env(
        base_settings=base_settings,
        sim_configs=sim_configs,
        species_to_track=['CH4', 'O2', 'CO2', 'H2O'],
        features_config={
            'local_features': True,
            'neighbor_features': True,
            'gradient_features': True,
            'temporal_features': True,
            'window_size': 5
        },
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
    )
    
    # Test environment
    for episode in range(3):
        obs = env.reset(seed=episode)
        done = False
        while not done:
            action = np.zeros(100, dtype=int)  # Default actions
            obs, rewards, done, info = env.step(action)
            done = np.any(done)