import numpy as np
from typing import Tuple, Dict, Any, List
import cantera as ct
from scipy.integrate import solve_ivp
from numba import jit, float64, int64
from .base import BaseSystem
from pyFlame.integrators import IntegratorType, IntegratorConfig, create_integrator, IntegratorSelector, DefaultSelector
from scipy.interpolate import interp1d


class ChemistrySystem(BaseSystem):
    def __init__(self, config, grid):
        super().__init__(config, grid)
        
        # Initialize Cantera with optimized settings
        self.gas = ct.Solution(config.mechanism)
        self.n_species = self.gas.n_species
        
        # Pre-compute constant arrays and cache them as numpy arrays
        self.molecular_weights = np.array(self.gas.molecular_weights, dtype=np.float64)
        
        # State indices as class constants
        self.i_T = 0
        self.i_U = 1
        self.i_Y_start = 2
        self.i_Y_end = 2 + self.n_species
        
        # Pre-allocate arrays with specified data types
        self._initialize_arrays(grid.n_points)
        
        # Optimized cache structure
        self._setup_cache(grid.n_points)
        
        # Store initial grid for reference
        self._previous_grid = grid.x.copy()
        
        # Initialize integrator system
        self._setup_integrators(config)
    
    def _interpolate_properties(self, old_x: np.ndarray, old_data: dict):
        """Interpolate properties from old grid to new grid with robust error handling"""
        try:
            new_x = self.grid.x
            
            # Validate input arrays
            if len(old_x) < 2:
                print("Warning: Old grid too small for interpolation, using direct property calculation")
                return
                
            # Check for NaN or inf in old_x
            if np.any(np.isnan(old_x)) or np.any(np.isinf(old_x)):
                print("Warning: Invalid values in old grid coordinates, using direct property calculation")
                return
                
            # Ensure old_x is strictly increasing
            if not np.all(np.diff(old_x) > 0):
                old_x, unique_indices = np.unique(old_x, return_index=True)
                # Update old_data to match unique old_x points
                for prop_name in old_data:
                    if old_data[prop_name].ndim == 1:
                        old_data[prop_name] = old_data[prop_name][unique_indices]
                    else:
                        old_data[prop_name] = old_data[prop_name][:, unique_indices]
            
            # Handle each property
            for prop_name, old_values in old_data.items():
                if old_values is None:
                    continue
                    
                try:
                    if prop_name in ['h', 'wdot']:  # 2D arrays
                        n_species, n_points = old_values.shape
                        if n_points != len(old_x):
                            print(f"Warning: Shape mismatch for {prop_name}: {old_values.shape} vs {len(old_x)} points")
                            continue
                            
                        new_values = np.zeros((n_species, len(new_x)))
                        for i in range(n_species):
                            # Create and apply interpolation with bounds handling
                            interp = interp1d(
                                old_x,
                                old_values[i],
                                kind='linear',
                                bounds_error=False,
                                fill_value=(old_values[i,0], old_values[i,-1])
                            )
                            new_values[i] = interp(new_x)
                        setattr(self, prop_name, new_values)
                        
                    else:  # 1D arrays
                        if len(old_values) != len(old_x):
                            print(f"Warning: Length mismatch for {prop_name}: {len(old_values)} vs {len(old_x)} points")
                            continue
                            
                        interp = interp1d(
                            old_x,
                            old_values,
                            kind='linear',
                            bounds_error=False,
                            fill_value=(old_values[0], old_values[-1])
                        )
                        new_values = interp(new_x)
                        setattr(self, prop_name, new_values)
                    
                    # Ensure physical bounds for specific properties
                    if prop_name == 'rho':
                        self.rho = np.maximum(self.rho, 1e-10)
                    elif prop_name == 'cp':
                        self.cp = np.maximum(self.cp, 1e-10)
                        
                except Exception as e:
                    print(f"Error interpolating {prop_name}: {str(e)}")
                    # Initialize with zeros and let property update handle it
                    if prop_name in ['h', 'wdot']:
                        setattr(self, prop_name, np.zeros((self.n_species, len(new_x))))
                    else:
                        setattr(self, prop_name, np.zeros(len(new_x)))
                    
        except Exception as e:
            print(f"Error during property interpolation: {str(e)}")
            # Properties will be recalculated in _update_all_properties

    def resize(self, n_points: int):
        """Resize system arrays and update properties for new grid"""
        if n_points == len(self._previous_grid):
            print(f"Warning: No resize needed - old point {self.n_points} matches new point {n_points}")
            return  # No resize needed
            
        try:
            # Store old grid data for interpolation
            old_x = self._previous_grid
            old_data = {}
            
            # Only store properties that exist and have correct shapes
            for prop_name in ['rho', 'cp', 'h', 'wdot']:
                if hasattr(self, prop_name):
                    prop = getattr(self, prop_name)
                    if prop is not None and (
                        (prop.ndim == 1 and len(prop) == len(old_x)) or
                        (prop.ndim == 2 and prop.shape[1] == len(old_x))
                    ):
                        old_data[prop_name] = prop.copy()
                    else:
                        print(f"Warning: Skipping {prop_name} due to shape mismatch")
            
            # Call parent resize
            super().resize(n_points)
            
            # Resize property arrays
            self._initialize_arrays(n_points)
            
            # Resize cache with proper initialization
            self._setup_cache(n_points)
            
            # Interpolate properties if we have valid old data
            if old_x is not None and len(old_data) > 0:
                self._interpolate_properties(old_x, old_data)
            
            # Force property update for all points
            self._cache['needs_update'] = np.ones(n_points, dtype=bool)
            self._update_all_properties()
            
            # Update stored grid for next resize
            self._previous_grid = self.grid.x.copy()
            
        except Exception as e:
            print(f"Error during resize: {str(e)}")
            # Attempt recovery
            self._initialize_arrays(n_points)
            self._setup_cache(n_points)
            self._update_all_properties()
            
            self._previous_grid = self.grid.x.copy()
            
    def _validate_properties(self):
        """Validate properties after updates"""
        invalid_points = []
        
        # Check for invalid values
        invalid_mask = (
            (self.rho <= 0) |
            (self.cp <= 0) |
            np.isnan(self.rho) |
            np.isnan(self.cp) |
            np.isinf(self.rho) |
            np.isinf(self.cp)
        )
        
        if np.any(invalid_mask):
            invalid_points = np.where(invalid_mask)[0]
            print(f"Found {len(invalid_points)} points with invalid properties")
            
            # Try to recover invalid points
            self._fix_zero_properties(invalid_points)
            
            # Verify fix worked
            still_invalid = (
                (self.rho <= 0) |
                (self.cp <= 0) |
                np.isnan(self.rho) |
                np.isnan(self.cp) |
                np.isinf(self.rho) |
                np.isinf(self.cp)
            )
            
            if np.any(still_invalid):
                raise ValueError("Unable to recover valid properties after resize")

            
    def _setup_integrators(self, config):
        """Initialize integrator system"""
        self.integrator_selector = DefaultSelector(
            IntegratorType[config.default_integrator]
        )
        self.integrator_configs = self._create_integrator_configs()
        self._integrator_cache = {}
        
    def _create_integrator_configs(self) -> Dict[IntegratorType, IntegratorConfig]:
        """Create configurations for each integrator type"""
        configs = {}
        
        # BDF configuration for stiff systems
        configs[IntegratorType.BDF] = IntegratorConfig(
            method=IntegratorType.BDF,
            rtol=self.config.rel_tol,
            atol=self.config.abs_tol,
        )
        
        # LSODA configuration for adaptive stiff/non-stiff systems
        configs[IntegratorType.LSODA] = IntegratorConfig(
            method=IntegratorType.LSODA,
            rtol=self.config.rel_tol,
            atol=self.config.abs_tol
        )
        
        configs[IntegratorType.RK23] = IntegratorConfig(
            method=IntegratorType.RK23,
            rtol=max(self.config.rel_tol * 10, 1e-6),  # Less strict for non-stiff regions
            atol=max(self.config.abs_tol * 10, 1e-8),
        )
        
        # RK45 configuration for non-stiff systems
        configs[IntegratorType.RK45] = IntegratorConfig(
            method=IntegratorType.RK45,
            rtol=max(self.config.rel_tol * 10, 1e-6),  # Less strict for non-stiff regions
            atol=max(self.config.abs_tol * 10, 1e-8),
        )
        
        configs[IntegratorType.Radau] = IntegratorConfig(
            method=IntegratorType.Radau,
            rtol=self.config.rel_tol,
            atol=self.config.abs_tol
        )
        
        # Explicit Euler for very simple regions
        configs[IntegratorType.EULER] = IntegratorConfig(
            method=IntegratorType.EULER,
            rtol=max(self.config.rel_tol * 100, 1e-5),  # Even less strict
            atol=max(self.config.abs_tol * 100, 1e-7),
            extra_params={
                'max_step': min(1e-6, self.config.max_step_size if hasattr(self.config, 'max_step_size') else 1e-6)
            }
        )
        
        return configs
    
    def _get_integrator(self, integrator_type: IntegratorType):
        """Get or create integrator from cache"""
        if integrator_type not in self._integrator_cache:
            if integrator_type not in self.integrator_configs:
                raise ValueError(f"No configuration found for integrator type: {integrator_type}")
            
            config = self.integrator_configs[integrator_type]
            self._integrator_cache[integrator_type] = create_integrator(config)
            
        return self._integrator_cache[integrator_type]
    
    def _get_jacobian_sparsity(self) -> np.ndarray:
        """Compute Jacobian sparsity pattern for the chemical system"""
        n_vars = 2 + self.n_species
        sparsity = np.zeros((n_vars, n_vars), dtype=bool)
        
        # Temperature affects all variables
        sparsity[self.i_T, :] = True
        
        # Species equations dependencies
        for i in range(self.n_species):
            idx = self.i_Y_start + i
            # Each species potentially affects temperature and other species
            sparsity[idx, self.i_T] = True
            sparsity[idx, self.i_Y_start:self.i_Y_end] = True
        
        return sparsity
    
    def _get_state_dict(self, point_idx: int) -> Dict[str, Any]:
        """Get state dictionary for integrator selection"""
        return {
            'T': self.state[self.i_T, point_idx],
            'P': self.config.pressure,
            'Y': self.state[self.i_Y_start:self.i_Y_end, point_idx],
            # 'rho': self.rho[point_idx],
            # 'cp': self.cp[point_idx],
            # 'wdot_norm': np.linalg.norm(self.wdot[:, point_idx]),
            # 'point_location': point_idx / (self.n_points - 1)  # Normalized position
        }
        
    def _setup_cache(self, n_points: int):
        """Initialize cache with pre-allocated arrays"""
        self._cache = {
            'T': np.zeros(n_points, dtype=np.float64),
            'Y': np.zeros((self.n_species, n_points), dtype=np.float64),
            'needs_update': np.ones(n_points, dtype=bool),
            'last_update': np.zeros(n_points, dtype=np.float64)
        }
    
    
    def _initialize_arrays(self, n_points: int):
        """Initialize arrays with proper data types"""
        self.rho = np.zeros(n_points, dtype=np.float64)
        self.cp = np.zeros(n_points, dtype=np.float64)
        self.h = np.zeros((self.n_species, n_points), dtype=np.float64)
        self.wdot = np.zeros((self.n_species, n_points), dtype=np.float64)
        
    @staticmethod
    @jit(nopython=True, cache=True)
    def _check_state_change_numba(T: float64, Y: np.ndarray, 
                                cached_T: float64, cached_Y: np.ndarray,
                                tolerance: float64) -> bool:
        """Numba-optimized state change check"""
        if abs(T - cached_T) > tolerance:
            return True
        
        for i in range(Y.shape[0]):
            if abs(Y[i] - cached_Y[i]) > tolerance:
                return True
        return False
    
    def _update_all_properties(self):
        """Update properties for all points after resize"""
        for j in range(self.n_points):
            self.update_properties(j)
            
        # Verify no zero values in critical properties
        zero_mask = (self.rho == 0) | (self.cp == 0)
        if np.any(zero_mask):
            problem_points = np.where(zero_mask)[0]
            self._fix_zero_properties(problem_points)
    
    def update_properties(self, j: int):
        """Optimized property update with safety checks"""
        if j >= self.state.shape[1]:
            raise IndexError(f"Point index {j} out of bounds for state array with shape {self.state.shape}")
        T = self.state[self.i_T, j] 
        Y = self.state[self.i_Y_start:self.i_Y_end, j]
        
        # Add safety checks
        if T <= 0:
            T = max(T, 300.0)  # Set minimum temperature
            self.state[self.i_T, j] = T
        
        # Ensure mass fractions sum to 1 and are non-negative
        Y = np.maximum(Y, 0)  # Ensure non-negative
        Y_sum = Y.sum()
        if Y_sum > 0:
            Y = Y / Y_sum
        else:
            Y = np.zeros_like(Y)
            Y[0] = 1.0  # Set first species to 1 if all are zero
        
        self.state[self.i_Y_start:self.i_Y_end, j] = Y
        
        if self._check_state_change_numba(
            T, Y,
            self._cache['T'][j],
            self._cache['Y'][:, j],
            1e-10
        ):
            # Update cache
            self._cache['T'][j] = T
            self._cache['Y'][:, j] = Y
            
            # Update Cantera state
            try:
                self.gas.TPY = T, self.config.pressure, Y
                
                # Get properties with error checking
                rho = self.gas.density
                cp = self.gas.cp_mass
                h = self.gas.partial_molar_enthalpies
                wdot = self.gas.net_production_rates
                
                # Update only if values are valid
                if rho > 0 and cp > 0:
                    self.rho[j] = rho
                    self.cp[j] = cp
                    self.h[:, j] = h
                    self.wdot[:, j] = wdot
                else:
                    raise ValueError(f"Invalid properties at point {j}: rho={rho}, cp={cp}")
                    
            except Exception as e:
                # Log error and attempt recovery
                print(f"Error updating properties at point {j}: {str(e)}")
                self._recover_properties(j)
    
    def _recover_properties(self, j: int):
        """Attempt to recover valid properties at a problematic point"""
        # Try to interpolate from neighboring points
        if j > 0 and j < self.n_points - 1:
            # Use average of neighboring points
            self.rho[j] = (self.rho[j-1] + self.rho[j+1]) / 2
            self.cp[j] = (self.cp[j-1] + self.cp[j+1]) / 2
            self.h[:, j] = (self.h[:, j-1] + self.h[:, j+1]) / 2
            self.wdot[:, j] = (self.wdot[:, j-1] + self.wdot[:, j+1]) / 2
        elif j == 0:
            # Use next point
            self.rho[j] = self.rho[j+1]
            self.cp[j] = self.cp[j+1]
            self.h[:, j] = self.h[:, j+1]
            self.wdot[:, j] = self.wdot[:, j+1]
        else:
            # Use previous point
            self.rho[j] = self.rho[j-1]
            self.cp[j] = self.cp[j-1]
            self.h[:, j] = self.h[:, j-1]
            self.wdot[:, j] = self.wdot[:, j-1]
    
    def _fix_zero_properties(self, problem_points: np.ndarray):
        """Fix zero values in properties by interpolation or extrapolation"""
        # Find valid points (non-zero properties)
        valid_mask = (self.rho > 0) & (self.cp > 0)
        valid_points = np.where(valid_mask)[0]
        
        if len(valid_points) == 0:
            raise RuntimeError("No valid points found for property interpolation")
        
        # Create interpolation functions
        x_valid = self.grid.x[valid_points]
        rho_interp = interp1d(
            x_valid,
            self.rho[valid_points],
            kind='linear',
            bounds_error=False,
            fill_value='extrapolate'
        )
        cp_interp = interp1d(
            x_valid,
            self.cp[valid_points],
            kind='linear',
            bounds_error=False,
            fill_value='extrapolate'
        )
        
        # Fix properties at problem points
        x_problem = self.grid.x[problem_points]
        self.rho[problem_points] = rho_interp(x_problem)
        self.cp[problem_points] = cp_interp(x_problem)
        
        # Force property update at these points
        for j in problem_points:
            self._cache['needs_update'][j] = True
            self.update_properties(j)
    
    @staticmethod
    @jit(nopython=True, parallel=True, cache=True)
    def _compute_source_terms_parallel(ddt, state, rho, cp, h, wdot, molecular_weights,
                                     i_T, i_U, i_Y_start, split_const, points):
        """Parallelized source term computation"""
        n_species = wdot.shape[0]
        
        for j in points:
            # Species equations - vectorized
            ddt[i_Y_start:i_Y_start+n_species, j] = (
                wdot[:, j] * molecular_weights / rho[j] + 
                split_const[i_Y_start:i_Y_start+n_species, j]
            )
            
            # Energy equation - vectorized
            q = -(wdot[:, j] * h[:, j]).sum()
            ddt[i_T, j] = q/(rho[j] * cp[j]) + split_const[i_T, j]
            
            # Momentum
            ddt[i_U, j] = split_const[i_U, j]
        
        return ddt
    
    def evaluate(self, t: float, points: List[int] = None) -> np.ndarray:
        """Optimized chemical source term evaluation"""
        points = points if points is not None else range(self.n_points)
        points = np.array(points, dtype=np.int64)
        
        # Pre-allocate output with proper type
        ddt = np.zeros_like(self.state, dtype=np.float64)
        
        # Batch update properties
        for j in points:
            self.update_properties(j)
        
        # Compute source terms in parallel
        return self._compute_source_terms_parallel(
            ddt, self.state, self.rho, self.cp, self.h, self.wdot,
            self.molecular_weights, self.i_T, self.i_U, self.i_Y_start,
            self.split_const, points
        )
    
    
    def integrate(self, t_start: float, t_end: float) -> Tuple[np.ndarray, bool]:
        """Optimized integration with batched processing"""
        success = True
        result = self.state.copy()
        
        # Pre-allocate groups dictionary
        integrator_groups: Dict[IntegratorType, List[int]] = {}
        
        # Batch integrator selection
        for j in range(self.n_points):
            state_dict = self._get_state_dict(j)
            integrator_type = self.integrator_selector.select(state_dict)
            if integrator_type not in integrator_groups:
                integrator_groups[integrator_type] = []
            integrator_groups[integrator_type].append(j)
        
        # Process each group efficiently
        for integrator_type, points in integrator_groups.items():
            points = np.array(points, dtype=np.int64)
            integrator = self._get_integrator(integrator_type) 
            
            # Optimize RHS function
            def rhs(t: float, y: np.ndarray) -> np.ndarray:
                self.state = y.reshape(self.n_vars, -1)
                return self.evaluate(t, points).ravel()
            
            # Integrate with minimal copying
            y0 = self.state[:, points].ravel()
            y_final, group_success = integrator.integrate(rhs, y0, (t_start, t_end))
            
            # Efficient update
            result[:, points] = y_final.reshape(self.n_vars, -1)
            success = success and group_success
        
        return result, success
    
    
    def integrate_with_cantera(self, t_start: float, t_end: float) -> Tuple[np.ndarray, bool]:
        """
        Alternative integration using Cantera's direct reactor integration.
        Used for testing and validation purposes.
        """
        result = self.state.copy()
        success = True
        dt = t_end - t_start
        
        # Process each point independently using Cantera
        for j in range(self.n_points):
            try:
                # Get current state
                T = self.state[self.i_T, j]
                Y = self.state[self.i_Y_start:self.i_Y_end, j]
                
                # Create a constant pressure reactor
                gas = ct.Solution(self.config.mechanism)
                gas.TPY = T, self.config.pressure, Y
                
                # Create and configure reactor
                reactor = ct.IdealGasConstPressureReactor(gas)
                sim = ct.ReactorNet([reactor])
                
                # Set tolerances (use same as main integrator)
                sim.rtol = self.config.rel_tol
                sim.atol = self.config.abs_tol
                
                # Advance the reactor
                sim.advance(dt)
                
                # Update result
                result[self.i_T, j] = reactor.thermo.T
                result[self.i_Y_start:self.i_Y_end, j] = reactor.thermo.Y
                
                # Velocity stays constant
                result[self.i_U, j] = self.state[self.i_U, j]
                
            except Exception as e:
                print(f"Cantera integration failed at point {j}: {str(e)}")
                success = False
                # Keep original values at this point
                result[:, j] = self.state[:, j]
        
        return result, success

    @staticmethod
    def _integrate_point_cantera(args):
        """
        Worker function for single point Cantera integration
        args: tuple of (state_vector, t_start, t_end, mechanism, pressure)
        """
        state_vector, t_start, t_end, mechanism, pressure = args
        try:
            # Unpack state
            T = state_vector[0]
            Y = state_vector[2:]  # Skip U
            
            # Setup and run reactor
            gas = ct.Solution(mechanism)
            gas.TPY = T, pressure, Y
            reactor = ct.IdealGasConstPressureReactor(gas)
            sim = ct.ReactorNet([reactor])
            sim.advance(t_end - t_start)
            
            # Return updated state
            new_state = state_vector.copy()
            new_state[0] = reactor.thermo.T
            new_state[2:] = reactor.thermo.Y
            return new_state, True
            
        except Exception as e:
            print(f"Parallel integration failed: {str(e)}")
            return state_vector, False

    def integrate_with_cantera_parallel(self, t_start: float, t_end: float) -> Tuple[np.ndarray, bool]:
        """
        Parallel version of Cantera integration using multiple reactors.
        Used for testing and validation purposes.
        """
        from multiprocessing import Pool
        
        result = self.state.copy()
        
        # Prepare arguments for each point
        args_list = [
            (
                self.state[:, j].copy(),
                t_start,
                t_end,
                self.config.mechanism,
                self.config.pressure
            )
            for j in range(self.n_points)
        ]
        
        # Run parallel integration
        try:
            with Pool() as pool:
                results = pool.map(self._integrate_point_cantera, args_list)
            
            # Unpack results
            success = True
            for j, (new_state, point_success) in enumerate(results):
                result[:, j] = new_state
                success = success and point_success
                
        except Exception as e:
            print(f"Parallel integration failed: {str(e)}")
            success = False
        
        return result, success
    
    
# integrators.py
from enum import Enum
from dataclasses import dataclass
from typing import Tuple, Callable, Dict, Any, Optional
import numpy as np
from scipy.integrate import solve_ivp

class IntegratorType(Enum):
    """Available integrator types"""
    BDF = "BDF"
    LSODA = "LSODA" 
    RK45 = "RK45"
    RK23 = "RK23"
    Radau = "Radau"
    EULER = "EULER"
    EQUILIBRIUM = "EQUILIBRIUM"

@dataclass
class IntegratorConfig:
    """Integrator configuration parameters"""
    method: IntegratorType
    rtol: float
    atol: float
    max_step: Optional[float] = None
    min_step: Optional[float] = None
    extra_params: Optional[Dict[str, Any]] = None

class BaseIntegrator:
    """Base integrator interface"""
    def init(self, config: IntegratorConfig):
        self.config = config

    def integrate(self, rhs: Callable, y0: np.ndarray, t_span: Tuple[float, float]) -> Tuple[np.ndarray, bool]:
        raise NotImplementedError

class ScipyIntegrator(BaseIntegrator):
    """Efficient scipy-based integrator"""
    def integrate(self, rhs: Callable, y0: np.ndarray, t_span: Tuple[float, float]) -> Tuple[np.ndarray, bool]:
        solution = solve_ivp(
            rhs,
            t_span,
            y0,
            method=self.config.method.value,
            rtol=self.config.rtol,
            atol=self.config.atol,
            jac_sparsity=self.config.extra_params.get('jac_sparsity') if self.config.extra_params else None
        )
        return solution.y[:,-1], solution.success

class ExplicitEuler(BaseIntegrator):
    """Fast explicit Euler for non-stiff regions"""
    def integrate(self, rhs: Callable, y0: np.ndarray, t_span: Tuple[float, float]) -> Tuple[np.ndarray, bool]:
        t_start, t_end = t_span
        dt = self.config.max_step or (t_end - t_start) / 100
        y = y0.copy()
        t = t_start

        while t < t_end:
            dt = min(dt, t_end - t)
            y += dt * rhs(t, y)
            t += dt

        return y, True

# Factory for creating integrators
INTEGRATOR_MAP = {
    IntegratorType.BDF: lambda cfg: ScipyIntegrator(cfg),
    IntegratorType.LSODA: lambda cfg: ScipyIntegrator(cfg),
    IntegratorType.RK45: lambda cfg: ScipyIntegrator(cfg),
    IntegratorType.RK23: lambda cfg: ScipyIntegrator(cfg),
    IntegratorType.Radau: lambda cfg: ScipyIntegrator(cfg),
    IntegratorType.EULER: lambda cfg: ExplicitEuler(cfg),
    IntegratorType.EQUILIBRIUM: lambda cfg: ScipyIntegrator(cfg)
}

def create_integrator(config: IntegratorConfig) -> BaseIntegrator:
    """Factory function to create integrators"""
    return INTEGRATOR_MAP[config.method](config)

class IntegratorSelector:
    """Base class for integrator selection strategies"""
    def select(self, state: Dict[str, Any]) -> IntegratorType:
        raise NotImplementedError

class DefaultSelector(IntegratorSelector):
    """Simple selector that always returns the default integrator"""
    def init(self, default_type: IntegratorType):
        self.default_type = default_type

    def select(self, state: Dict[str, Any]) -> IntegratorType:
        T = state.get("T", 300)
        if T > 2500:
            return IntegratorType.BDF
        else:
            return self.default_type

class RLPolicySelector(IntegratorSelector):
    """Selector that uses RL policy for integrator selection"""
    def init(self, policy_path: str):
        # Initialize RL policy here
        self.policy = None  # Load your trained policy

    def select(self, state: Dict[str, Any]) -> IntegratorType:
        # Convert state to policy input format
        policy_input = self._prepare_state(state)
        # Get policy prediction
        return self._predict(policy_input)

    def _prepare_state(self, state: Dict[str, Any]) -> np.ndarray:
        # Implement state preprocessing for your policy
        pass

    def _predict(self, policy_input: np.ndarray) -> IntegratorType:
        pass