import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import h5py
import os
from datetime import datetime
import time


def get_memory_usage():
    """Get current memory usage in MB"""
    import psutil
    import os
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

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
    center_width: float = 0.000
    slope_width: float = 0.0000
    equilibrate_counterflow: bool = False

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
        self.cpu_times = np.zeros((self.n_steps, self.n_points))
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
        if step_data.error is not None:
            self.errors[idx] = step_data.error
        
        self.current_step += 1

    
    def _extend_arrays(self) -> None:
        """Extend storage arrays when needed"""
        extension = 1
        self.temperatures = np.vstack([self.temperatures, np.zeros((extension, self.n_points))])
        for spec in self.species_names:
            self.species_mass_fractions[spec] = np.vstack([
                self.species_mass_fractions[spec], 
                np.zeros((extension, self.n_points))
            ])
        self.phis = np.vstack([self.phis, np.zeros((extension, self.n_points))])
        self.spatial_points = np.vstack([self.spatial_points, np.zeros((extension, self.n_points))])
        self.times = np.concatenate([self.times, np.zeros(extension)])
        self.cpu_times = np.concatenate([self.cpu_times, np.zeros((extension, self.n_points))])
        self.integrator_types.extend([''] * extension)
        self.errors = np.vstack([self.errors, np.zeros((extension, self.n_points))])
        self.n_steps += extension
    
    def get_step(self, step: int) -> SimulationStep:
        """Get data for a specific step"""
        if step > self.current_step:
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
        cpu_time=solver.gridPointIntegrationTimes,
        integrator_type=integrator_types,
        error=None
    )
    data_holder.add_step(step_data)
    return data_holder, done, cpu_time