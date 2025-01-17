## CORRECT CHEMISTY IMPLEMENTATION

import numpy as np
from typing import Tuple, Dict, Any, List
import cantera as ct
from scipy.integrate import solve_ivp
from numba import jit, float64, int64
from .base import BaseSystem
from pyFlame.integrators import IntegratorType, IntegratorConfig, create_integrator, IntegratorSelector, DefaultSelector, IntegratorAPI
from scipy.interpolate import interp1d
from multiprocessing import Pool, shared_memory, cpu_count

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
            IntegratorType[config.default_integrator],
            config.integrator_api
            )
        self.integrator_configs = self._create_integrator_configs()
        self._integrator_cache = {}
        
    def _create_integrator_configs(self) -> Dict[Tuple[IntegratorType, IntegratorAPI], IntegratorConfig]:
        """Create configurations for each integrator type and API combination"""
        configs = {}
        
        # Common parameters
        base_config = {
            'rtol': self.config.rel_tol,
            'atol': self.config.abs_tol,
            # 'max_step': getattr(self.config, 'max_step_size', None),
            # 'min_step': getattr(self.config, 'min_step_size', None),
        }
        
        # solve_ivp configurations
        configs[(IntegratorType.BDF, IntegratorAPI.SOLVE_IVP)] = IntegratorConfig(
            method=IntegratorType.BDF,
            api=IntegratorAPI.SOLVE_IVP,
            **base_config,
            #extra_params={'jac_sparsity': self._get_jacobian_sparsity()}
        )
        
        configs[(IntegratorType.LSODA, IntegratorAPI.SOLVE_IVP)] = IntegratorConfig(
            method=IntegratorType.LSODA,
            api=IntegratorAPI.SOLVE_IVP,
            **base_config
        )
        
        configs[(IntegratorType.RK23, IntegratorAPI.SOLVE_IVP)] = IntegratorConfig(
            method=IntegratorType.RK23,
            api=IntegratorAPI.SOLVE_IVP,
            rtol=max(base_config['rtol'] * 10, 1e-6),
            atol=max(base_config['atol'] * 10, 1e-8),
            **{k: v for k, v in base_config.items() if k not in ['rtol', 'atol']}
        )
        
        configs[(IntegratorType.RK45, IntegratorAPI.SOLVE_IVP)] = IntegratorConfig(
            method=IntegratorType.RK45,
            api=IntegratorAPI.SOLVE_IVP,
            rtol=max(base_config['rtol'] * 10, 1e-6),
            atol=max(base_config['atol'] * 10, 1e-8),
            **{k: v for k, v in base_config.items() if k not in ['rtol', 'atol']}
        )
        
        configs[(IntegratorType.Radau, IntegratorAPI.SOLVE_IVP)] = IntegratorConfig(
            method=IntegratorType.Radau,
            api=IntegratorAPI.SOLVE_IVP,
            **base_config,
            extra_params={'jac_sparsity': self._get_jacobian_sparsity()}
        )
        
        # ODE configurations
        configs[(IntegratorType.VODE, IntegratorAPI.ODE)] = IntegratorConfig(
            method=IntegratorType.VODE,
            api=IntegratorAPI.ODE,
            **base_config,
            extra_params={
                'method': 'bdf',  # Use BDF for stiff problems
                'with_jacobian': True,
                'nsteps': 500,
                'order': 5,  # Maximum order for BDF
            }
        )
        
        configs[(IntegratorType.ZVODE, IntegratorAPI.ODE)] = IntegratorConfig(
            method=IntegratorType.ZVODE,
            api=IntegratorAPI.ODE,
            **base_config,
            extra_params={
                'method': 'bdf',
                'with_jacobian': True,
                'nsteps': 500,
                'order': 5
            }
        )
        
        configs[(IntegratorType.LSODA_ODE, IntegratorAPI.ODE)] = IntegratorConfig(
            method=IntegratorType.LSODA_ODE,
            api=IntegratorAPI.ODE,
            **base_config,
            extra_params={
                'with_jacobian': True,
                'nsteps': 500,
                'max_order_ns': 12,  # Maximum order for non-stiff
                'max_order_s': 5,    # Maximum order for stiff
            }
        )
        
        configs[(IntegratorType.DOPRI5, IntegratorAPI.ODE)] = IntegratorConfig(
            method=IntegratorType.DOPRI5,
            api=IntegratorAPI.ODE,
            rtol=max(base_config['rtol'] * 10, 1e-6),
            atol=max(base_config['atol'] * 10, 1e-8),
            **{k: v for k, v in base_config.items() if k not in ['rtol', 'atol']},
            extra_params={
                'nsteps': 500,
                'safety': 0.9,
                'ifactor': 10.0,
                'dfactor': 0.2,
                'beta': 0.0,
            }
        )
        
        configs[(IntegratorType.DOP853, IntegratorAPI.ODE)] = IntegratorConfig(
            method=IntegratorType.DOP853,
            api=IntegratorAPI.ODE,
            rtol=max(base_config['rtol'] * 10, 1e-6),
            atol=max(base_config['atol'] * 10, 1e-8),
            **{k: v for k, v in base_config.items() if k not in ['rtol', 'atol']},
            extra_params={
                'nsteps': 500,
                'safety': 0.9,
                'ifactor': 6.0,
                'dfactor': 0.3,
                'beta': 0.0,
            }
        )
        
        # Explicit Euler (kept for simple cases)
        configs[(IntegratorType.EULER, IntegratorAPI.SOLVE_IVP)] = IntegratorConfig(
            method=IntegratorType.EULER,
            api=IntegratorAPI.SOLVE_IVP,
            rtol=max(base_config['rtol'] * 100, 1e-5),
            atol=max(base_config['atol'] * 100, 1e-7),
            extra_params={
                'max_step': min(1e-6, self.config.max_step_size if hasattr(self.config, 'max_step_size') else 1e-6)
            }
        )
        
        return configs
    
    def _get_integrator(self, integrator_type: IntegratorType, api: IntegratorAPI):
        """Get or create integrator from cache with API support"""
        cache_key = (integrator_type, api)
        if cache_key not in self._integrator_cache:
            if cache_key not in self.integrator_configs:
                raise ValueError(f"No configuration found for integrator type: {integrator_type} with API: {api}")
            
            config = self.integrator_configs[cache_key]
            self._integrator_cache[cache_key] = create_integrator(config)
            
        return self._integrator_cache[cache_key]
    
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
            'Y': self.state[self.i_Y_start:self.i_Y_end, point_idx]
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
            #print(f"Point: {j} - Updated")
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
        """Integrate chemistry system using scipy.integrate.ode"""
        from scipy.integrate import ode
        
        success = True
        y0 = self.state.copy().ravel()  # Flatten state for ODE solver
        
        def rhs(t: float, y: np.ndarray) -> np.ndarray:
            """RHS function for ODE solver"""
            try:
                # Reshape state to 2D array (n_vars x n_points)
                self.state = y.reshape(self.n_vars, -1)
                
                # Compute derivatives
                dydt = self.evaluate(t)
                
                return dydt.ravel()  # Return flattened array
                
            except Exception as e:
                print(f"RHS evaluation failed: {str(e)}")
                # Return zeros if evaluation fails
                return np.zeros_like(y)
        
        try:
            # Create ODE solver
            solver = ode(rhs)
            
            # Set integration method and parameters
            solver.set_integrator('vode', 
                                method='bdf',  # Use BDF for stiff problems
                                with_jacobian=False,  # No analytical Jacobian
                                rtol=self.config.rel_tol,
                                atol=self.config.abs_tol,
                                order=5,  # Maximum order for BDF
                                nsteps=500)  # Maximum number of internal steps
            
            # Set initial conditions
            solver.set_initial_value(y0, t_start)
            
            # Integrate to final time
            result = solver.integrate(t_end)
            
            success = solver.successful()
            
            if success:
                # Reshape result back to state shape
                result = result.reshape(self.n_vars, -1)
                
                # Ensure mass fractions are valid
                Y_sum = result[self.i_Y_start:].sum(axis=0)
                result[self.i_Y_start:] /= Y_sum[np.newaxis, :]
                
                # Update properties with final state
                self.state = result
                self._update_all_properties()
                
            else:
                print("Integration failed")
                result = self.state.copy()
                
        except Exception as e:
            print(f"Integration failed with error: {str(e)}")
            success = False
            result = self.state.copy()
        
        return result, success
    
    def integrate_(self, t_start: float, t_end: float) -> Tuple[np.ndarray, bool]:
        """Enhanced integration with API support and batched processing"""
        success = True
        result = self.state.copy()
        
        # Pre-allocate groups dictionary with API support
        integrator_groups: Dict[Tuple[IntegratorType, IntegratorAPI], List[int]] = {}
        
        # Batch integrator selection
        for j in range(self.n_points):
            state_dict = self._get_state_dict(j)
            integrator_type, api = self.integrator_selector.select(state_dict)
            group_key = (integrator_type, api)
            
            if group_key not in integrator_groups:
                integrator_groups[group_key] = []
            integrator_groups[group_key].append(j)
        
        # Process each group efficiently
        for (integrator_type, api), points in integrator_groups.items():
            points = np.array(points, dtype=np.int64)
            integrator = self._get_integrator(integrator_type, api)
            
            # Optimize RHS function
            def rhs(t: float, y: np.ndarray) -> np.ndarray:
                self.state = y.reshape(self.n_vars, -1)
                return self.evaluate(t, points).ravel()
            
            # Integrate with minimal copying
            y0 = self.state[:, points].ravel()
            #print(f"Points: {points} -  Temperature: {y0[0]} - start: {t_start} - end: {t_end}")
            y_final, group_success = integrator.integrate(rhs, y0, (t_start, t_end))
            #print(f"Points: {points} -  Temperature: {y_final[0]} - start: {t_start} - end: {t_end}")
            # Efficient update
            result[:, points] = y_final.reshape(self.n_vars, -1)
            success = success and group_success
        
        return result, success
    
    def integrate_parallel(self, t_start: float, t_end: float) -> Tuple[np.ndarray, bool]:
        """Optimized parallel integration with better state management"""
        success = True
        result = self.state.copy()
        
        # Group points by integrator type
        integrator_groups: Dict[Tuple[IntegratorType, IntegratorAPI], List[int]] = {}
        
        # Batch integrator selection
        for j in range(self.n_points):
            state_dict = self._get_state_dict(j)
            integrator_type, api = self.integrator_selector.select(state_dict)
            group_key = (integrator_type, api)
            
            if group_key not in integrator_groups:
                integrator_groups[group_key] = []
            integrator_groups[group_key].append(j)
        
        # Prepare work chunks
        chunk_size = max(1, self.n_points // (cpu_count() * 2))
        work_items = []
        
        for (integrator_type, api), points in integrator_groups.items():
            points = np.array(points)
            # Split points into chunks
            for i in range(0, len(points), chunk_size):
                chunk = points[i:i + chunk_size]
                work_items.append((
                    chunk,
                    self.state[:, chunk].copy(),  # Pass full state slice
                    {
                        'rho': self.rho[chunk].copy(),
                        'cp': self.cp[chunk].copy(),
                        'h': self.h[:, chunk].copy(),
                        'wdot': self.wdot[:, chunk].copy()
                    },
                    integrator_type,
                    api,
                    t_start,
                    t_end,
                    self.config.mechanism,
                    self.config.pressure,
                    self.molecular_weights,
                    self.split_const[:, chunk].copy(),
                    self.config.rel_tol,
                    self.config.abs_tol
                ))
        
        # Process chunks in parallel
        try:
            with Pool() as pool:
                results = pool.map(self._process_chunk, work_items)
                
                # Combine results
                for points, final_state, thermo_props, chunk_success in results:
                    result[:, points] = final_state
                    
                    # Update thermodynamic properties
                    self.rho[points] = thermo_props['rho']
                    self.cp[points] = thermo_props['cp']
                    self.h[:, points] = thermo_props['h']
                    self.wdot[:, points] = thermo_props['wdot']
                    
                    success = success and chunk_success
                    
        except Exception as e:
            print(f"Parallel integration failed: {str(e)}")
            success = False
        
        return result, success

    @staticmethod 
    def _process_chunk(args):
        """Process a single chunk of points"""
        (points, initial_state, initial_props, integrator_type, api, 
        t_start, t_end, mechanism, pressure, molecular_weights, 
        split_const, rtol, atol) = args
        
        try:
            # Create local Cantera solution
            gas = ct.Solution(mechanism)
            n_species = gas.n_species
            
            # Current properties
            props = {
                'rho': initial_props['rho'].copy(),
                'cp': initial_props['cp'].copy(),
                'h': initial_props['h'].copy(),
                'wdot': initial_props['wdot'].copy()
            }
            
            def update_properties(state_vec, j):
                """Update properties for a single point"""
                T = state_vec[0]
                Y = state_vec[2:]
                
                # Safety checks
                T = max(T, 300.0)
                Y = np.maximum(Y, 0)
                Y_sum = Y.sum()
                if Y_sum > 0:
                    Y = Y / Y_sum
                else:
                    Y = np.zeros_like(Y)
                    Y[0] = 1.0
                
                try:
                    gas.TPY = T, pressure, Y
                    props['rho'][j] = gas.density
                    props['cp'][j] = gas.cp_mass
                    props['h'][:, j] = gas.partial_molar_enthalpies
                    props['wdot'][:, j] = gas.net_production_rates
                    return True
                except Exception as e:
                    print(f"Property update failed: {str(e)}")
                    return False
            
            # Create integrator
            integrator_config = IntegratorConfig(
                method=integrator_type,
                api=api,
                rtol=rtol,
                atol=atol
            )
            integrator = create_integrator(integrator_config)
            
            # RHS function for integration
            def rhs(t, y):
                state = y.reshape(-1, len(points))
                ddt = np.zeros_like(state)
                
                # Update properties and compute derivatives
                for j in range(len(points)):
                    if update_properties(state[:, j], j):
                        # Species equations
                        ddt[2:, j] = (props['wdot'][:, j] * molecular_weights / 
                                    props['rho'][j] + split_const[2:, j])
                        
                        # Energy equation
                        q = -(props['wdot'][:, j] * props['h'][:, j]).sum()
                        ddt[0, j] = q/(props['rho'][j] * props['cp'][j]) + split_const[0, j]
                        
                        # Momentum
                        ddt[1, j] = split_const[1, j]
                
                return ddt.ravel()
            
            # Perform integration
            y0 = initial_state.ravel()
            y_final, success = integrator.integrate(rhs, y0, (t_start, t_end))
            
            # Final state update
            final_state = y_final.reshape(-1, len(points))
            for j in range(len(points)):
                update_properties(final_state[:, j], j)
            
            return points, final_state, props, success
            
        except Exception as e:
            print(f"Chunk processing failed: {str(e)}")
            return points, initial_state, initial_props, False

    # Helper class to share large arrays between processes
    class SharedArrayWrapper:
        def __init__(self, shape, dtype=np.float64):
            from multiprocessing import shared_memory
            self.shape = shape
            self.dtype = dtype
            
            # Create shared memory block
            size = int(np.prod(shape)) * np.dtype(dtype).itemsize
            self.shm = shared_memory.SharedMemory(create=True, size=size)
            
            # Create NumPy array backed by shared memory
            self.array = np.ndarray(shape, dtype=dtype, buffer=self.shm.buf)
        
        def __del__(self):
            # Clean up
            self.shm.close()
            self.shm.unlink()
    
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