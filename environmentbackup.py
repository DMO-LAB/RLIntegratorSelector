import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cantera as ct
import time
import pickle
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from collections import deque
from ember import Config, Paths, InitialCondition, StrainParameters, General, \
    Times, TerminationCondition, ConcreteConfig, Debug, RK23Tolerances, QssTolerances, CvodeTolerances, _ember

from typing import Optional, Dict, List
import numpy as np
import h5py

@dataclass
class IntegratorOption:
    """Class to represent integrator options"""
    name: str
    type: str  # 'cvode' or 'boostRK'
    rtol: float
    atol: float
    color: str
    

@dataclass
class SimulationSettings:
    """General simulation settings"""
    output_dir: str = 'run/diffusion_benchmark'
    n_threads: int = 2
    global_timestep: float = 1e-6
    profile_interval: int = 20
    t_end: float = 0.08
    n_points: int = 100
    x_left: float = -0.02
    x_right: float = 0.02
    T_fuel: float = 600
    T_oxidizer: float = 1200
    pressure: float = 101325
    strain_rate: float = 100
    
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import h5py
import os
from datetime import datetime

@dataclass
class SimulationStep:
    """Class to hold data for a single simulation step"""
    step: int
    time: float
    temperatures: np.ndarray
    species_mass_fractions: Dict[str, np.ndarray]
    phi: np.ndarray
    spatial_points: np.ndarray
    cpu_time: float
    integrator_type: str
    error: np.ndarray

@dataclass
class SimulationData:
    """Class to manage and store simulation data efficiently"""
    species_names: List[str]
    n_points: int
    n_steps: int
    output_dir: str
    
    # Initialize storage arrays
    temperatures: np.ndarray = field(init=False)
    species_mass_fractions: Dict[str, np.ndarray] = field(init=False)
    phis: np.ndarray = field(init=False)
    spatial_points: np.ndarray = field(init=False)
    times: np.ndarray = field(init=False)
    cpu_times: np.ndarray = field(init=False)
    integrator_types: List[str] = field(init=False)
    errors: np.ndarray = field(init=False)
    current_step: int = field(init=False, default=0)
    #steps: List[SimulationStep] = field(init=False, default_factory=list)

    
    def __post_init__(self):
        """Initialize storage arrays after object creation"""
        self.temperatures = np.zeros((self.n_steps, self.n_points))
        self.species_mass_fractions = {
            spec: np.zeros((self.n_steps, self.n_points)) 
            for spec in self.species_names
        }
        self.phis = np.zeros((self.n_steps, self.n_points))
        self.spatial_points = np.zeros((self.n_steps, self.n_points))
        self.times = np.zeros(self.n_steps)
        self.cpu_times = np.zeros(self.n_steps)
        self.integrator_types = [''] * self.n_steps
        self.errors = np.zeros((self.n_steps, self.n_points))
    
    def add_step(self, step_data: SimulationStep) -> None:
        """Add data for a single simulation step"""
        if self.current_step >= self.n_steps:
            self._extend_arrays()
        
        idx = self.current_step
        self.temperatures[idx] = step_data.temperatures
        for spec, mass_frac in step_data.species_mass_fractions.items():
            self.species_mass_fractions[spec][idx] = mass_frac
        self.phis[idx] = step_data.phi
        self.spatial_points[idx] = step_data.spatial_points
        self.times[idx] = step_data.time
        self.cpu_times[idx] = step_data.cpu_time
        self.integrator_types[idx] = step_data.integrator_type
        #self.steps.append(step_data)
        if step_data.error is not None:
            self.errors[idx] = step_data.error
        
        self.current_step += 1
    
    def _extend_arrays(self) -> None:
        """Extend storage arrays when needed"""
        extension = self.n_steps
        self.temperatures = np.vstack([self.temperatures, np.zeros((extension, self.n_points))])
        for spec in self.species_names:
            self.species_mass_fractions[spec] = np.vstack([
                self.species_mass_fractions[spec], 
                np.zeros((extension, self.n_points))
            ])
        self.phis = np.vstack([self.phis, np.zeros((extension, self.n_points))])
        self.spatial_points = np.vstack([self.spatial_points, np.zeros((extension, self.n_points))])
        self.times = np.concatenate([self.times, np.zeros(extension)])
        self.cpu_times = np.concatenate([self.cpu_times, np.zeros(extension)])
        self.integrator_types.extend([''] * extension)
        self.errors = np.vstack([self.errors, np.zeros((extension, self.n_points))])
        #self.steps.extend([SimulationStep(step=0, time=0, temperatures=np.zeros((self.n_points,)), species_mass_fractions={}, phi=np.zeros(self.n_points), spatial_points=np.zeros(self.n_points), cpu_time=0, integrator_type='', error=np.zeros(self.n_points))] * extension)
        self.n_steps += extension
    
    def get_step(self, step: int) -> SimulationStep:
        """Get data for a specific step"""
        if step >= self.current_step:
            raise IndexError(f"Step {step} not available. Current step is {self.current_step}")
        
        return SimulationStep(
            step=step,
            time=self.times[step],
            temperatures=self.temperatures[step],
            species_mass_fractions={
                spec: self.species_mass_fractions[spec][step]
                for spec in self.species_names
            },
            phi=self.phis[step],
            spatial_points=self.spatial_points[step],
            cpu_time=self.cpu_times[step],
            integrator_type=self.integrator_types[step],
            error=self.errors[step]
        )
    
    def save_to_hdf5(self, filename: Optional[str] = None) -> str:
        """Save simulation data to HDF5 file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.output_dir, f"simulation_data_{timestamp}.h5")
        
        # os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with h5py.File(filename, 'w') as f:
            # Create groups
            meta = f.create_group('metadata')
            data = f.create_group('data')
            
            # Store metadata
            meta.attrs['n_points'] = self.n_points
            meta.attrs['n_steps'] = self.n_steps
            meta.attrs['current_step'] = self.current_step
            meta.create_dataset('species_names', data=np.array(self.species_names, dtype='S'))
            
            # Store simulation data
            data.create_dataset('temperatures', data=self.temperatures[:self.current_step])
            data.create_dataset('phis', data=self.phis[:self.current_step])
            data.create_dataset('spatial_points', data=self.spatial_points[:self.current_step])
            data.create_dataset('times', data=self.times[:self.current_step])
            data.create_dataset('cpu_times', data=self.cpu_times[:self.current_step])
            data.create_dataset('errors', data=self.errors[:self.current_step])
            data.create_dataset('integrator_types', 
                              data=np.array(self.integrator_types[:self.current_step], dtype='S'))
            
            # Store species data in a separate group
            species_group = data.create_group('species')
            for spec in self.species_names:
                species_group.create_dataset(
                    spec, 
                    data=self.species_mass_fractions[spec][:self.current_step]
                )
        
        return filename
    
    @classmethod
    def load_from_hdf5(cls, filename: str) -> 'SimulationData':
        """Load simulation data from HDF5 file"""
        with h5py.File(filename, 'r') as f:
            # Load metadata
            meta = f['metadata']
            n_points = meta.attrs['n_points']
            n_steps = meta.attrs['n_steps']
            species_names = [name.decode() for name in meta['species_names']]
            
            # Create instance
            output_dir = os.path.dirname(filename)
            instance = cls(species_names, n_points, n_steps, output_dir)
            
            # Load data
            data = f['data']
            instance.temperatures = data['temperatures'][:]
            instance.phis = data['phis'][:]
            instance.spatial_points = data['spatial_points'][:]
            instance.times = data['times'][:]
            instance.cpu_times = data['cpu_times'][:]
            instance.errors = data['errors'][:]
            instance.integrator_types = [
                name for name in data['integrator_types'][:]
            ]
            
            # Load species data
            species_group = data['species']
            for spec in species_names:
                instance.species_mass_fractions[spec] = species_group[spec][:]
            
            instance.current_step = meta.attrs['current_step']
        
        return instance
    
    def get_species_profile(self, species_name: str, step: Optional[int] = None) -> np.ndarray:
        """Get mass fraction profile for a specific species"""
        if species_name not in self.species_names:
            raise ValueError(f"Species {species_name} not found in simulation data")
        
        if step is None:
            return self.species_mass_fractions[species_name][:self.current_step]
        return self.species_mass_fractions[species_name][step]
    
    def get_temperature_profile(self, step: Optional[int] = None) -> np.ndarray:
        """Get temperature profile"""
        if step is None:
            return self.temperatures[:self.current_step]
        return self.temperatures[step]
    
    def get_phi_profile(self, step: Optional[int] = None) -> np.ndarray:
        """Get equivalence ratio profile"""
        if step is None:
            return self.phis[:self.current_step]
        return self.phis[step]
    
    def get_performance_metrics(self) -> Dict:
        """Calculate and return performance metrics"""
        return {
            'total_cpu_time': np.sum(self.cpu_times[:self.current_step]),
            'mean_cpu_time': np.mean(self.cpu_times[:self.current_step]),
            'max_error': np.max(self.errors[:self.current_step]),
            'mean_error': np.mean(self.errors[:self.current_step]),
            'integrator_usage': dict(zip(
                *np.unique(self.integrator_types[:self.current_step], return_counts=True)
            ))
        }

def take_step(step_count, current_time, solver, data_holder, integrator_types, species_index, species_names):
    start_time = time.time()
    done = solver.step()
    end_time = time.time()
    cpu_time = end_time - start_time
    step_data = SimulationStep(
        step=step_count,
        time=current_time,
        temperatures=solver.T,
        species_mass_fractions={
            spec: solver.Y[species_index[spec]] for spec in species_names
        },
        phi=solver.phi,
        spatial_points=solver.x,
        cpu_time=cpu_time,
        integrator_type=integrator_types,
        error=None
    )
    data_holder.add_step(step_data)
    return data_holder, done, cpu_time

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
        n_steps=sim_settings.n_points,
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
        while not done:
            current_time = step_count * sim_settings.global_timestep
            data_holder, done, cpu_time = take_step(step_count, current_time, solver, benchmark_data, integrator_types, species_index, species_to_track)
            step_count += 1
        benchmark_data.save_to_hdf5(filename)
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
            features.append(self.solver.T[point_idx] / self.sim_settings.T_oxidizer)
            for spec in self.species_to_track:
                Y = self.solver.Y[self.species_indices[spec]][point_idx]
                features.append(np.log10(max(abs(Y), 1e-20)))
            
            # Local phi
            phi = self.solver.phi[point_idx]
            phi = np.maximum(phi, 1e-3)
            phi = np.minimum(phi, 1)
            features.append(phi)
        
        if self.features_config['neighbor_features']:
            # Add features from neighboring points
            for offset in [-1, 1]:
                idx = max(0, min(point_idx + offset, self.sim_settings.n_points - 1))
                features.append(self.solver.T[idx] / self.sim_settings.T_oxidizer)
                for spec in self.species_to_track:
                    Y = self.solver.Y[self.species_indices[spec]][idx]
                    features.append(np.log10(max(abs(Y), 1e-10)))

        if self.features_config['gradient_features']:
            # Local gradients
            dT = np.gradient(self.solver.T)[point_idx]
            features.append(np.log10(max(abs(dT), 1e-10)))
            
            for spec in self.species_to_track:
                dY = np.gradient(self.solver.Y[self.species_indices[spec]])[point_idx]
                features.append(np.log10(max(abs(dY), 1e-20)))
        
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
    
    def _calculate_point_error(self, point_idx: int) -> Tuple[float, dict]:
        """Calculate error metrics for a specific point"""
        benchmark_step = self.benchmark_data.get_step(self.current_step)
        
        # Temperature error
        T_error = abs(self.solver.T[point_idx] - 
                     benchmark_step.temperatures[point_idx])
        
        # Species errors
        species_errors = {}
        for spec in self.species_to_track:
            current_Y = self.solver.Y[self.species_indices[spec]][point_idx]
            bench_Y = benchmark_step.species_mass_fractions[spec][point_idx]
            species_errors[spec] = abs(current_Y - bench_Y)
        
        # Gradient error
        current_grad = np.gradient(self.solver.T)[point_idx]
        bench_grad = np.gradient(benchmark_step.temperatures)[point_idx]
        grad_error = abs(current_grad - bench_grad)
        
        return T_error, species_errors, grad_error
    
    def _compute_point_reward(self, point_idx: int, cpu_time: float, 
                            T_error: float, species_errors: dict, grad_error: float) -> float:
        """Compute reward for a specific point"""
        weights = self.reward_config['weights']
        thresholds = self.reward_config['thresholds']
        scaling = self.reward_config['scaling']
        
        # Accuracy reward
        #temp_reward = np.exp(-scaling['error'] * T_error / thresholds['error'])
        temp_reward = np.exp(-T_error/thresholds['error'])
        # species_reward = np.mean([
        #     np.exp(-scaling['error'] * error / thresholds['error'])
        #     for error in species_errors.values()
        # ])
        species_reward = np.mean([
            np.exp(-error/thresholds['error'])
            for error in species_errors.values()
        ])
        #grad_reward = np.exp(-scaling['error'] * grad_error / thresholds['error'])
        grad_reward = np.exp(-grad_error/thresholds['error'])
        
        accuracy_reward = weights['accuracy'] * (temp_reward + species_reward + grad_reward) / 3
        
        # Efficiency reward
        # time_reward = weights['efficiency'] * np.exp(-scaling['time'] * cpu_time / thresholds['time'])
        time_reward = np.exp(-cpu_time/thresholds['time']) * weights['efficiency']
        
        # Stability reward
        if self.history_index >= 2:
            idx = (self.history_index - 1) % self.features_config['window_size']
            stability = np.std(self.history['temperature'][point_idx, max(0, idx-5):idx+1])
            # stability_reward = weight s['stability'] * np.exp(-stability)
            stability_reward = np.exp(-stability/thresholds['stability']) * weights['stability']
        else:
            stability_reward = 0
        
        return accuracy_reward + time_reward + stability_reward
    
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
            cpu_time = time.time() - start_time
            
            # Calculate errors and rewards for each point
            rewards = np.zeros(self.sim_settings.n_points)
            errors = np.zeros(self.sim_settings.n_points)
            
            for i in range(self.sim_settings.n_points):
                T_error, species_errors, grad_error = self._calculate_point_error(i)
                errors[i] = T_error + np.mean(list(species_errors.values())) + grad_error
                rewards[i] = self._compute_point_reward(i, cpu_time, T_error, species_errors, grad_error)
            
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
            
            # Prepare info dictionary
            info = {
                'cpu_time': cpu_time,
                'point_errors': errors,
                'point_rewards': rewards,
                'total_time': self.episode_data.get_performance_metrics()['total_cpu_time'] if self.save_step_data else 0,
                'action': action
            }
            
            self.current_step += 1
            
            if self.current_step >= self.end_step:
                print(f"Episode Done {self.current_step} - resetting environment")
                done = True
                truncated = True
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
    
    def reset(self, seed=None):
        """Reset the environment"""
        super().reset(seed=seed)
        
        # Initialize solver
        self.solver = _ember.FlameSolver(
            _create_config(self.sim_settings)
        )
        self.solver.initialize()
        
        self.solver.set_integrator_types(['cvode'] * 100)
        self.solver.step()
        # Reset data storage
        if self.save_step_data:
            self.episode_data = None
            self._initialize_storage()
        
        # Reset tracking variables
        self.current_step = 1
        self.end_step = int(self.sim_settings.t_end / self.sim_settings.global_timestep)
        self._initialize_history()
        
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
        ax1.plot(self.solver.x, self.benchmark_data.get_step(self.current_step).temperatures, 
                'k--', label='Benchmark')
        ax1.set_ylabel('Temperature (K)')
        ax1.set_title('Temperature Profile')
        ax1.legend()
        
        # Species profiles
        ax2 = fig.add_subplot(gs[1, :])
        for spec in self.species_to_track:
            current_Y = self.solver.Y[self.species_indices[spec]]
            benchmark_Y = self.benchmark_data.get_step(self.current_step).species_mass_fractions[spec]
            ax2.semilogy(self.solver.x, current_Y, '-', label=f'{spec} Current')
            ax2.semilogy(self.solver.x, benchmark_Y, '--', label=f'{spec} Benchmark')
        ax2.set_ylabel('Mass Fractions')
        ax2.set_title('Species Profiles')
        ax2.legend()
        
        # Error distribution
        ax3 = fig.add_subplot(gs[2, 0])
        T_errors = np.abs(self.solver.T - 
                         self.benchmark_data.get_step(self.current_step).temperatures)
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





class DataChunkManager:
    """Manages data in chunks to reduce memory usage"""
    def __init__(self, chunk_size: int = 1000):
        self.chunk_size = chunk_size
        self.current_chunk = []
        self.chunk_files = []
        self.total_items = 0
        
    def add_item(self, item: SimulationStep):
        """Add item to current chunk, write to disk if chunk is full"""
        self.current_chunk.append(item)
        self.total_items += 1
        
        if len(self.current_chunk) >= self.chunk_size:
            self._write_chunk_to_disk()
            
    def _write_chunk_to_disk(self):
        """Write current chunk to disk and clear memory"""
        if not self.current_chunk:
            return
            
        chunk_file = f'temp_chunk_{len(self.chunk_files)}.h5'
        with h5py.File(chunk_file, 'w') as f:
            # Store chunk data
            chunk_group = f.create_group('steps')
            for i, step in enumerate(self.current_chunk):
                step_group = chunk_group.create_group(f'step_{i}')
                step_group.create_dataset('temperatures', data=step.temperatures)
                step_group.create_dataset('time', data=step.time)
                # Store species data
                species_group = step_group.create_group('species')
                for name, data in step.species_mass_fractions.items():
                    species_group.create_dataset(name, data=data)
                # Store other attributes
                step_group.create_dataset('phi', data=step.phi)
                step_group.create_dataset('spatial_points', data=step.spatial_points)
                step_group.attrs['cpu_time'] = step.cpu_time
                step_group.attrs['integrator_type'] = step.integrator_type
                if step.error is not None:
                    step_group.create_dataset('error', data=step.error)
        
        self.chunk_files.append(chunk_file)
        self.current_chunk.clear()
        gc.collect()  # Force garbage collection

class MemoryEfficientSimulationData:
    """Memory efficient version of SimulationData using disk-based storage"""
    def __init__(self, species_names: List[str], n_points: int, n_steps: int, output_dir: str):
        self.species_names = species_names
        self.n_points = n_points
        self.n_steps = n_steps
        self.output_dir = output_dir
        self.current_step = 0
        
        # Use chunk manager for step data
        self.chunk_manager = DataChunkManager()
        
        # Keep minimal arrays in memory
        self._initialize_summary_arrays()
    
    def _initialize_summary_arrays(self):
        """Initialize small summary arrays kept in memory"""
        self.times = np.zeros(self.n_steps)
        self.cpu_times = np.zeros(self.n_steps)
        self.max_errors = np.zeros(self.n_steps)
        self.mean_temps = np.zeros(self.n_steps)
    
    def add_step(self, step_data: SimulationStep):
        """Add step data using chunk manager"""
        if self.current_step >= self.n_steps:
            self._extend_arrays()
        
        # Update summary arrays
        self.times[self.current_step] = step_data.time
        self.cpu_times[self.current_step] = step_data.cpu_time
        if step_data.error is not None:
            self.max_errors[self.current_step] = np.max(step_data.error)
        self.mean_temps[self.current_step] = np.mean(step_data.temperatures)
        
        # Add to chunk manager
        self.chunk_manager.add_item(step_data)
        
        self.current_step += 1
    
    def _extend_arrays(self):
        """Extend summary arrays when needed"""
        extension = self.n_steps
        self.times = np.concatenate([self.times, np.zeros(extension)])
        self.cpu_times = np.concatenate([self.cpu_times, np.zeros(extension)])
        self.max_errors = np.concatenate([self.max_errors, np.zeros(extension)])
        self.mean_temps = np.concatenate([self.mean_temps, np.zeros(extension)])
        self.n_steps += extension
    
    def save_to_hdf5(self, filename: Optional[str] = None) -> str:
        """Save all data to a single HDF5 file"""
        if filename is None:
            filename = os.path.join(self.output_dir, f"simulation_data_{time.strftime('%Y%m%d_%H%M%S')}.h5")
        
        with h5py.File(filename, 'w') as f:
            # Store metadata
            meta = f.create_group('metadata')
            meta.attrs['n_points'] = self.n_points
            meta.attrs['n_steps'] = self.n_steps
            meta.attrs['current_step'] = self.current_step
            meta.create_dataset('species_names', data=np.array(self.species_names, dtype='S'))
            
            # Store summary data
            summary = f.create_group('summary')
            summary.create_dataset('times', data=self.times[:self.current_step])
            summary.create_dataset('cpu_times', data=self.cpu_times[:self.current_step])
            summary.create_dataset('max_errors', data=self.max_errors[:self.current_step])
            summary.create_dataset('mean_temps', data=self.mean_temps[:self.current_step])
            
            # Merge chunk data
            steps = f.create_group('steps')
            step_idx = 0
            
            # First write any remaining data in current chunk
            self.chunk_manager._write_chunk_to_disk()
            
            # Then merge all chunks
            for chunk_file in self.chunk_manager.chunk_files:
                with h5py.File(chunk_file, 'r') as chunk_f:
                    chunk_steps = chunk_f['steps']
                    for step_name in chunk_steps:
                        step_data = chunk_steps[step_name]
                        step_group = steps.create_group(f'step_{step_idx}')
                        for key in step_data.keys():
                            step_data.copy(key, step_group)
                        for attr_name, attr_value in step_data.attrs.items():
                            step_group.attrs[attr_name] = attr_value
                        step_idx += 1
                
                # Remove temporary chunk file
                print(f"Removing chunk file")
                os.remove(chunk_file)
        
        return filename

class MemoryEfficientEnvironment(VectorizedCombustionEnv):
    """Memory efficient version of the environment"""
    def __init__(self, *args, **kwargs):
        # Initialize base environment
        super().__init__(*args, **kwargs)
        
        # Store solver directly since it doesn't support weak references
        self._solver = None
        
        # Configure garbage collection
        gc.enable()
        gc.set_threshold(100, 5, 5)  # Aggressive collection
    
    @property
    def solver(self):
        """Getter for solver"""
        return self._solver
    
    @solver.setter
    def solver(self, value):
        """Setter for solver"""
        self._solver = value
        if value is None:
            gc.collect()
            
    @property
    def episode_data(self):
        """Getter for episode data - maintains compatibility with parent class"""
        return self._episode_data
    
    def _initialize_storage(self):
        """Initialize data storage using memory efficient version"""
        self._episode_data = MemoryEfficientSimulationData(
            species_names=self.species_to_track,
            n_points=self.sim_settings.n_points,
            n_steps=int(self.sim_settings.t_end / self.sim_settings.global_timestep),
            output_dir=self.sim_settings.output_dir
        )
    
    def step(self, action: np.ndarray):
        """Memory efficient step implementation"""
        try:
            # Set integrators based on action
            integrator_types = []
            for a in action:
                integrator = self.integrator_options[a]
                integrator_types.append(integrator.type)
            
            # Take step
            start_time = time.time()
            self.solver.set_integrator_types(integrator_types)
            done = self.solver.step()
            cpu_time = time.time() - start_time
            
            # Calculate errors and rewards for each point
            rewards = np.zeros(self.sim_settings.n_points)
            errors = np.zeros(self.sim_settings.n_points)
            
            for i in range(self.sim_settings.n_points):
                T_error, species_errors, grad_error = self._calculate_point_error(i)
                errors[i] = T_error + np.mean(list(species_errors.values())) + grad_error
                rewards[i] = self._compute_point_reward(i, cpu_time, T_error, species_errors, grad_error)
            
            # Update history
            self._update_history()
            
            # Create step data
            step_data = SimulationStep(
                step=self.current_step,
                time=self.current_step * self.sim_settings.global_timestep,
                temperatures=self.solver.T.copy(),  # Make copies to ensure data independence
                species_mass_fractions={
                    spec: self.solver.Y[self.species_indices[spec]].copy()
                    for spec in self.species_to_track
                },
                phi=self.solver.phi.copy(),
                spatial_points=self.solver.x.copy(),
                cpu_time=cpu_time,
                integrator_type=str(action),
                error=errors.copy()
            )
            
            # Store step data
            self._episode_data.add_step(step_data)
            
            # Get next observation
            observation = self._get_observation()
            
            # Prepare info dictionary
            info = {
                'cpu_time': cpu_time,
                'point_errors': errors,
                'point_rewards': rewards,
                'total_time': self._episode_data.cpu_times.sum(),
                'action': action
            }
            
            self.current_step += 1
            
            if self.current_step >= self.end_step:
                done = True
                truncated = True
            else:
                truncated = False
            
            # Force garbage collection periodically
            if self.current_step % 10 == 0:
                print(f"Garbage collection")
                gc.collect()
                
            return observation, rewards, done, truncated, info
            
        except Exception as e:
            print(f"Integration failed: {e}")
            import traceback
            traceback.print_exc()
            return (
                self._get_observation(),
                np.full(self.sim_settings.n_points, -100.0),
                True,
                True,
                {'error': float('inf'), 'cpu_time': 0.0}
            )
    
    def reset(self, seed=None):
        """Memory efficient reset implementation"""
        # Clear existing solver
        if self._solver is not None:
            self._solver = None
            gc.collect()
        
        # Initialize new solver
        self._solver = _ember.FlameSolver(_create_config(self.sim_settings))
        self._solver.initialize()
        
        # Set initial integrator types
        self._solver.set_integrator_types(['cvode'] * self.sim_settings.n_points)
        self._solver.step()
        
        # Reset data storage
        self._initialize_storage()
        
        # Reset tracking variables
        self.current_step = 1
        self.end_step = int(self.sim_settings.t_end / self.sim_settings.global_timestep)
        self._initialize_history()
        
        return self._get_observation(), {}
    
    def close(self):
        """Clean up resources"""
        if hasattr(self, '_solver') and self._solver is not None:
            self._solver = None
        
        if hasattr(self, '_episode_data') and self._episode_data is not None:
            self._episode_data = None
            
        gc.collect()

def create_memory_efficient_env(sim_settings: SimulationSettings,
                              benchmark_file: str = None,
                              species_to_track: List[str] = None,
                              features_config: dict = None,
                              reward_config: dict = None) -> MemoryEfficientEnvironment:
    """Create memory efficient version of the environment"""
    
    gas = ct.Solution('gri30.yaml')
    species_to_track = species_to_track if species_to_track is not None else ['CH4', 'O2', 'CO2', 'H2O']
    species_index = {spec: gas.species_index(spec) for spec in species_to_track}
    
    benchmark_data = run_benchmark(
        sim_settings=sim_settings,
        species_to_track=species_to_track,
        species_index=species_index,
        output_dir='benchmarks',
        filename=benchmark_file
    )
    
    env = MemoryEfficientEnvironment(
        sim_settings=sim_settings,
        benchmark_data=benchmark_data,
        species_to_track=species_to_track,
        species_index=species_index,
        features_config=features_config,
        reward_config=reward_config
    )
    
    return env



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
    
    return env





if __name__ == "__main__":
    # Create simulation settings
    sim_settings = SimulationSettings(
        output_dir='run/rl_test',
        t_end=0.06,
        n_points=100,
        T_fuel=600,
        T_oxidizer=1200,
        pressure=101325,
        global_timestep=1e-5,
        profile_interval=20
    )

    # Create environment
    env = create_env(
        sim_settings=sim_settings,
        benchmark_file='run/rl_test/benchmark.h5',
        species_to_track=['CH4', 'O2', 'CO2', 'H2O'],
        features_config={
            'local_features': True,
            'neighbor_features': False,
            'gradient_features': False,
            'temporal_features': False,
            'window_size': 5
        },
        reward_config={
            'weights': {
                'accuracy': 0.4,
                'efficiency': 0.3,
                'stability': 0.3
            },
            'thresholds': {
                'time': 0.01,
                'error': 1e-3
            },
            'scaling': {
                'time': 0.1,
                'error': 1.0
            }
        }
    )
    
    obs, info = env.reset()
    done = False
    total_errors = np.zeros(sim_settings.n_points)
    total_rewards = np.zeros(sim_settings.n_points)
    while not done:
        action = [1] * 100
        obs, rewards, done, truncated, info = env.step(action)
        total_errors += info['point_errors']
        total_rewards += info['point_rewards']
    
    print(f"Total errors: {total_errors}")
    print(f"Total rewards: {total_rewards}")
    
    np.savetxt('run/rl_test/cvode_total_errors.txt', total_errors)
    np.savetxt('run/rl_test/cvode_total_rewards.txt', total_rewards)
    
    env.save_episode_data('run/rl_test/episode_data_cvode.h5')