# import numpy as np
# from typing import Tuple
# import cantera as ct
# from scipy.integrate import solve_ivp
# from .base import BaseSystem

# class ChemistrySystem(BaseSystem):
#     """
#     System representing chemical reactions and heat release
#     """
#     def __init__(self, config, grid):
#         super().__init__(config, grid)
        
#         # Create Cantera gas object
#         self.gas = ct.Solution(config.mechanism)
#         self.n_species = self.gas.n_species
        
#         # State variables indices
#         self.i_T = 0  # Temperature index
#         self.i_U = 1  # Velocity index
#         self.i_Y = slice(2, 2 + self.n_species)  # Species indices
        
#         # Properties
#         self.rho = np.zeros(self.n_points)  # Density
#         self.cp = np.zeros(self.n_points)   # Heat capacity
#         self.h = np.zeros((self.n_species, self.n_points))  # Species enthalpies
#         self.wdot = np.zeros((self.n_species, self.n_points))  # Reaction rates
        
#     def resize(self, n_points: int):
#         """Resize system arrays for new grid size"""
#         super().resize(n_points)
        
#         # Resize property arrays
#         self.rho = np.zeros(n_points)
#         self.cp = np.zeros(n_points)
#         self.h = np.zeros((self.n_species, n_points))
#         self.wdot = np.zeros((self.n_species, n_points))
        
#     def initialize(self, n_vars: int):
#         """Initialize chemistry system"""
#         super().initialize(n_vars)
#         assert n_vars == 2 + self.n_species
        
#     def update_properties(self, j: int):
#         """Update thermodynamic properties at point j"""
#         # Get state at this point
#         T = self.state[self.i_T, j]
#         Y = self.state[self.i_Y, j]
        
#         # Update gas object
#         self.gas.TPY = T, self.config.pressure, Y
        
#         # Get properties
#         self.rho[j] = self.gas.density
#         self.cp[j] = self.gas.cp_mass
#         self.h[:,j] = self.gas.partial_molar_enthalpies
#         self.wdot[:,j] = self.gas.net_production_rates
        
#     def evaluate(self, t: float) -> np.ndarray:
#         """Evaluate chemical source terms"""
#         ddt = np.zeros_like(self.state)
        
#         # Loop over points
#         for j in range(self.n_points):
#             # Update properties
#             self.update_properties(j)
            
#             # Species equations
#             ddt[self.i_Y,j] = (self.wdot[:,j] * self.gas.molecular_weights / 
#                               self.rho[j] + self.split_const[self.i_Y,j])
            
#             # Energy equation - heat release
#             q = -(self.wdot[:,j] * self.h[:,j]).sum()
#             ddt[self.i_T,j] = q/(self.rho[j] * self.cp[j]) + self.split_const[self.i_T,j]
            
#             # Momentum - no chemical source
#             ddt[self.i_U,j] = self.split_const[self.i_U,j]
            
#         return ddt
        
#     def integrate(self, t_start: float, t_end: float) -> Tuple[np.ndarray, bool]:
#         """
#         Integrate the chemistry system 
#         Returns: (final_state, success)
#         """
#         def rhs(t, y):
#             self.state = y.reshape(self.n_vars, -1)
#             return self.evaluate(t).ravel()
            
#         # Integrate using stiff solver
#         solution = solve_ivp(
#             rhs,
#             (t_start, t_end),
#             self.state.ravel(),
#             method='BDF',
#             rtol=self.config.rel_tol,
#             atol=self.config.abs_tol
#         )
        
#         success = solution.success
#         if success:
#             result = solution.y[:,-1].reshape(self.n_vars, -1)
#         else:
#             result = self.state
            
#         return result, success


import numpy as np
from typing import Tuple, Dict, Any, List
import cantera as ct
from scipy.integrate import solve_ivp
from numba import jit
from .base import BaseSystem
from ..integrators import IntegratorType, IntegratorConfig, create_integrator, IntegratorSelector, DefaultSelector

class ChemistrySystem(BaseSystem):
    """
    Optimized system representing chemical reactions and heat release
    """
    def __init__(self, config, grid):
        super().__init__(config, grid)
        
        # Create Cantera gas object with optimized settings
        self.gas = ct.Solution(config.mechanism)
        self.n_species = self.gas.n_species
        
        # Pre-compute constant arrays
        self.molecular_weights = self.gas.molecular_weights
        
        # State variables indices (use integers instead of slice objects)
        self.i_T = 0  # Temperature index
        self.i_U = 1  # Velocity index
        self.i_Y_start = 2  # Start of species indices
        self.i_Y_end = 2 + self.n_species  # End of species indices
        
        # Pre-allocate arrays
        self._initialize_arrays(grid.n_points)
        
        # Cache for property updates
        self._cache = {
            'T': np.zeros(grid.n_points),
            'Y': np.zeros((self.n_species, grid.n_points)),
            'needs_update': np.ones(grid.n_points, dtype=bool)
        }
        
        # Initialize integrator selector
        self.integrator_selector = DefaultSelector(
            IntegratorType[config.default_integrator]
        )
        
        # Initialize integrator configs
        self.integrator_configs = self._create_integrator_configs()
        
        # Cache for integrators
        self._integrator_cache = {}
        
    def _create_integrator_configs(self) -> Dict[IntegratorType, IntegratorConfig]:
        """Create configurations for each integrator type"""
        return {
            IntegratorType.BDF: IntegratorConfig(
                method=IntegratorType.BDF,
                rtol=self.config.rel_tol,
                atol=self.config.abs_tol,
                extra_params={'jac_sparsity': self._get_jacobian_sparsity()}
            ),
            IntegratorType.LSODA: IntegratorConfig(
                method=IntegratorType.LSODA,
                rtol=self.config.rel_tol,
                atol=self.config.abs_tol
            ),
            # ... other configurations ...
        }
        
    def _get_integrator(self, integrator_type: IntegratorType):
        """Get or create integrator from cache"""
        if integrator_type not in self._integrator_cache:
            config = self.integrator_configs[integrator_type]
            self._integrator_cache[integrator_type] = create_integrator(config)
        return self._integrator_cache[integrator_type]
    
    def _get_state_dict(self, point_idx: int) -> Dict[str, Any]:
        """Get state dictionary for integrator selection"""
        return {
            'T': self.state[self.i_T, point_idx],
            'P': self.config.pressure,
            'Y': self.state[self.i_Y_start:self.i_Y_end, point_idx],
            # Add other relevant state information
        }
        
    def _initialize_arrays(self, n_points: int):
        """Initialize all arrays at once"""
        self.rho = np.zeros(n_points)
        self.cp = np.zeros(n_points)
        self.h = np.zeros((self.n_species, n_points))
        self.wdot = np.zeros((self.n_species, n_points))
        
    def resize(self, n_points: int):
        """Resize system arrays efficiently"""
        super().resize(n_points)
        self._initialize_arrays(n_points)
        self._cache['T'] = np.zeros(n_points)
        self._cache['Y'] = np.zeros((self.n_species, n_points))
        self._cache['needs_update'] = np.ones(n_points, dtype=bool)
        
    def initialize(self, n_vars: int):
        """Initialize chemistry system"""
        super().initialize(n_vars)
        assert n_vars == 2 + self.n_species
        
    def _check_state_change(self, j: int) -> bool:
        """Check if state has changed at point j"""
        T = self.state[self.i_T, j]
        Y = self.state[self.i_Y_start:self.i_Y_end, j]
        
        if (abs(T - self._cache['T'][j]) > 1e-10 or 
            np.any(abs(Y - self._cache['Y'][:, j]) > 1e-10)):
            self._cache['T'][j] = T
            self._cache['Y'][:, j] = Y
            return True
        return False
        
    def update_properties(self, j: int):
        """Update thermodynamic properties with caching"""
        if not self._check_state_change(j):
            return
            
        # Get state at this point
        T = self.state[self.i_T, j]
        Y = self.state[self.i_Y_start:self.i_Y_end, j]
        
        # Update gas object (minimize Cantera calls)
        self.gas.TPY = T, self.config.pressure, Y
        
        # Get properties in batch
        self.rho[j] = self.gas.density
        self.cp[j] = self.gas.cp_mass
        self.h[:, j] = self.gas.partial_molar_enthalpies
        self.wdot[:, j] = self.gas.net_production_rates
        
    @staticmethod
    @jit(nopython=True)
    def _compute_source_terms(ddt, state, rho, cp, h, wdot, molecular_weights, i_T, i_U, i_Y_start, split_const):
        """Numba-optimized source term computation"""
        n_points = state.shape[1]
        n_species = wdot.shape[0]
        
        for j in range(n_points):
            # Species equations
            for k in range(n_species):
                ddt[k + i_Y_start, j] = (wdot[k, j] * molecular_weights[k] / 
                                       rho[j] + split_const[k + i_Y_start, j])
            
            # Energy equation - heat release
            q = 0.0
            for k in range(n_species):
                q -= wdot[k, j] * h[k, j]
            ddt[i_T, j] = q/(rho[j] * cp[j]) + split_const[i_T, j]
            
            # Momentum - no chemical source
            ddt[i_U, j] = split_const[i_U, j]
            
        return ddt
        
    def evaluate(self, t: float) -> np.ndarray:
        """Evaluate chemical source terms efficiently"""
        # Pre-allocate output array
        ddt = np.zeros_like(self.state)
        
        # Update properties where needed
        for j in range(self.n_points):
            self.update_properties(j)
        
        # Compute source terms using optimized function
        ddt = self._compute_source_terms(
            ddt, self.state, self.rho, self.cp, self.h, self.wdot,
            self.molecular_weights, self.i_T, self.i_U, self.i_Y_start,
            self.split_const
        )
        
        return ddt
        
    # def integrate(self, t_start: float, t_end: float) -> Tuple[np.ndarray, bool]:
    #     """Integrate the chemistry system efficiently"""
    #     # Prepare initial state
    #     y0 = self.state.ravel()
        
    #     # Define RHS function with minimal overhead
    #     def rhs(t, y):
    #         self.state = y.reshape(self.n_vars, -1)
    #         return self.evaluate(t).ravel()
        
    #     # Integrate using stiff solver with optimized settings
    #     solution = solve_ivp(
    #         rhs,
    #         (t_start, t_end),
    #         y0,
    #         method='BDF',
    #         rtol=self.config.rel_tol,
    #         atol=self.config.abs_tol,
    #         max_step=t_end - t_start  # Allow larger steps when possible
    #     )
        
    #     if solution.success:
    #         return solution.y[:,-1].reshape(self.n_vars, -1), True
    #     return self.state, False
    
    def integrate(self, t_start: float, t_end: float) -> Tuple[np.ndarray, bool]:
        """Integrate using selected integrators for each point"""
        success = True
        result = self.state.copy()
        
        # Group points by selected integrator for batch processing
        integrator_groups: Dict[IntegratorType, List[int]] = {}
        
        # Select integrators for each point
        for j in range(self.n_points):
            state_dict = self._get_state_dict(j)
            integrator_type = self.integrator_selector.select(state_dict)
            integrator_groups.setdefault(integrator_type, []).append(j)
        
        # Integrate each group
        for integrator_type, points in integrator_groups.items():
            integrator = self._get_integrator(integrator_type)
            
            # Create RHS function for these points
            def rhs(t, y):
                self.state = y.reshape(self.n_vars, -1)
                return self.evaluate(t, points).ravel()
            
            # Integrate
            y0 = self.state[:, points].ravel()
            y_final, group_success = integrator.integrate(rhs, y0, (t_start, t_end))
            
            # Update result
            result[:, points] = y_final.reshape(self.n_vars, -1)
            success = success and group_success
        
        return result, success