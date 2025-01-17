from dataclasses import dataclass
from typing import Literal, Optional, Union
from pyFlame.strain import ConstantStrain, RampedStrain, OscillatingStrain
from pyFlame.integrators import IntegratorType, IntegratorAPI

@dataclass
class Config:
    """Configuration parameters for the flame solver"""
    # Grid parameters
    grid_min: float = 1e-6  # Minimum grid spacing
    grid_max: float = 1e-3  # Maximum grid spacing
    grid_points: int = 100  # Initial number of grid points
    uniformity_tol: float = 1.2  # Maximum ratio of adjacent grid spacings
    
    # Domain parameters
    x_min: float = 0.0  # Left boundary of domain
    x_max: float = 0.02  # Right boundary of domain [m]
    
    # Chemistry parameters
    mechanism: str = "gri30.yaml"  # Chemical mechanism file
    fuel: str = "CH4"
    oxidizer: str = "O2:1, N2:3.76"
    pressure: float = 101325.0  # [Pa]
    
    # Time stepping
    dt: float = 1e-6  # Global timestep [s]
    t_start: float = 0.0  # Start time [s]
    t_end: float = 0.1  # End time [s]
    splitting_method: Literal["balanced", "strang"] = "balanced"
    
    # Flame configuration
    flame_type: Literal["premixed", "diffusion"] = "premixed"
    phi: float = 1.0  # Equivalence ratio for premixed flames
    T_fuel: float = 300.0  # Fuel temperature [K]
    T_oxidizer: float = 300.0  # Oxidizer temperature [K]
    fuel_left: bool = True  # Fuel stream on left side
    
    disc_flame: bool = False  # Discontinuity in flame speed
    flame_geometry: Literal['cylinder', 'disc', 'planar'] = 'planar'
    Rs: float = 0.0  # Stagnation surface radius (for cylindrical flames)
    
    # Solver parameters
    rel_tol: float = 1e-6  # Relative tolerance
    abs_tol: float = 1e-8  # Absolute tolerance
    max_step_size: Optional[float] = 1e-5 # Maximum step size
    
    # Grid adaptation parameters
    adaptation_interval: int = 5  # Adapt every N steps
    grad_tol: float = 0.2
    curv_tol: float = 0.1
    max_ratio: float = 1.5
    min_points: int = 50
    max_points: int = 500
    flame_res: float = 0.1  # mm
    
    save_interval: int = 10  # Save every N steps
    save_filename: str = "flame_solution.npz"
    
    # Ignition parameters
    ignition_energy: float = 0 #1e5  # [W/m³]
    ignition_duration: float = 0 #1e-3  # [s]
    ignition_start_time: float = 0.0  # [s]
    ignition_center: float = 0.01  # [m]
    ignition_width: float = 0.002  # [m]
    
    # Strain rate parameters
    strain_rate: int = 100  # [1/s]
    strain_function: Union[ConstantStrain, RampedStrain, OscillatingStrain] = RampedStrain(
        initial_strain=100,  # [1/s]
        final_strain=1000,   # [1/s]
        start_time=0.01,     # [s]
        ramp_duration=0.1    # [s]
    )
    
    # Output parameters
    output_dir: str = "output"
    
    # time stepper parameters
    dt_min: float = 1e-8 
    dt_max: float = 1e-4
    
    max_T_change: float = 50.0
    max_Y_change: float = 0.1
    timestep_safety: float = 0.8
    
    max_timestep_growth: float = 2.0
    min_timestep_reduction: float = 0.1
    
    use_mixing_layer: bool = False
    use_ignition_zone: bool = False
    mixing_width: float = 0.002 # [m]
    ignition_T_peak: float = 300
    
    use_cantera: bool = False
    use_parallel: bool = False
    
    default_integrator: str = "VODE"
    integrator_api: IntegratorAPI = IntegratorAPI.ODE