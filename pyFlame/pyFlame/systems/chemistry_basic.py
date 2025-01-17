import numpy as np
import cantera as ct
from typing import Tuple, Optional
from .base import BaseSystem

class ChemistrySystem(BaseSystem):
    def __init__(self, config, grid):
        super().__init__(config, grid)
        
        # Initialize Cantera
        self.gas = ct.Solution(config.mechanism)
        self.n_species = self.gas.n_species
        self.n_vars = 2 + self.n_species  # T, U, Y1...Yn
        
        # State indices
        self.i_T = 0       # Temperature index
        self.i_U = 1       # Velocity index
        self.i_Y_start = 2 # First species index
        
        # Reaction rate control
        self.rate_multiplier = 1.0
        self._last_rate_multiplier = None
        
        # Initialize state arrays
        self.initialize(self.n_vars)
        
        # Property arrays
        self._initialize_properties()
        
        # Store initial grid and pressure
        self._previous_grid = grid.x.copy()
        self.pressure = config.pressure
        
        # Error tracking
        self.last_error = None
        
    def _initialize_properties(self):
        """Initialize property arrays"""
        self.rho = np.zeros(self.n_points)
        self.cp = np.zeros(self.n_points)
        self.h = np.zeros((self.n_species, self.n_points))
        self.wdot = np.zeros((self.n_species, self.n_points))
        self.wdot_creation = np.zeros((self.n_species, self.n_points))
        self.wdot_destruction = np.zeros((self.n_species, self.n_points))
        self.molecular_weights = np.array(self.gas.molecular_weights)
        
    def resize(self, n_points: int):
        """Handle grid resizing with property interpolation"""
        if n_points == self.n_points:
            return
            
        # Store old grid and properties
        old_x = self._previous_grid
        old_properties = {
            'T': self.state[self.i_T].copy(),
            'U': self.state[self.i_U].copy(),
            'Y': self.state[self.i_Y_start:].copy(),
            'rho': self.rho.copy(),
            'cp': self.cp.copy(),
            'h': self.h.copy(),
            'wdot': self.wdot.copy(),
            'wdot_creation': self.wdot_creation.copy(),
            'wdot_destruction': self.wdot_destruction.copy()
        }
        
        # Resize base arrays
        super().resize(n_points)
        
        # Resize property arrays
        self._initialize_properties()
        
        # Interpolate to new grid
        new_x = self.grid.x
        
        try:
            # Interpolate state variables
            self.state[self.i_T] = np.interp(new_x, old_x, old_properties['T'])
            self.state[self.i_U] = np.interp(new_x, old_x, old_properties['U'])
            
            # Interpolate species mass fractions
            for i in range(self.n_species):
                self.state[self.i_Y_start + i] = np.interp(new_x, old_x, old_properties['Y'][i])
                
            # Normalize mass fractions
            Y_sum = np.sum(self.state[self.i_Y_start:], axis=0)
            self.state[self.i_Y_start:] /= Y_sum[np.newaxis, :]
            
            # Interpolate other properties
            self.rho = np.interp(new_x, old_x, old_properties['rho'])
            self.cp = np.interp(new_x, old_x, old_properties['cp'])
            
            for i in range(self.n_species):
                self.h[i] = np.interp(new_x, old_x, old_properties['h'][i])
                self.wdot[i] = np.interp(new_x, old_x, old_properties['wdot'][i])
                self.wdot_creation[i] = np.interp(new_x, old_x, old_properties['wdot_creation'][i])
                self.wdot_destruction[i] = np.interp(new_x, old_x, old_properties['wdot_destruction'][i])
            
        except Exception as e:
            self.last_error = str(e)
            # Initialize with safe values
            self.rho.fill(1.0)
            self.cp.fill(1000.0)
        
        # Update stored grid
        self._previous_grid = new_x.copy()
        
        # Update properties
        self._update_all_properties()
        
    def set_rate_multiplier(self, multiplier):
        """Set the reaction rate multiplier"""
        if multiplier != self._last_rate_multiplier:
            for i in range(self.gas.n_reactions):
                self.gas.set_multiplier(i, multiplier)
            self._last_rate_multiplier = multiplier
    
    def _update_point_properties(self, j: int):
        """Update properties at a single point"""
        try:
            # Get state
            T = self.state[self.i_T, j]
            Y = self.state[self.i_Y_start:, j]
            
            # Safety checks
            T = max(T, 300.0)
            Y = np.maximum(Y, 0)
            Y_sum = np.sum(Y)
            if Y_sum > 0:
                Y = Y / Y_sum
            else:
                Y = np.zeros_like(Y)
                Y[0] = 1.0
            
            # Update state
            self.state[self.i_T, j] = T
            self.state[self.i_Y_start:, j] = Y
            
            # Update Cantera state
            self.gas.TPY = T, self.pressure, Y
            
            # Update properties
            self.rho[j] = self.gas.density
            self.cp[j] = self.gas.cp_mass
            self.h[:, j] = self.gas.partial_molar_enthalpies
            
            # Update reaction rates
            self.wdot[:, j] = self.gas.net_production_rates
            self.wdot_creation[:, j] = self.gas.creation_rates
            self.wdot_destruction[:, j] = self.gas.destruction_rates
            
            return True
            
        except Exception as e:
            self.last_error = str(e)
            if j > 0:
                self._recover_properties(j)
            return False
    
    def _update_all_properties(self):
        """Update properties at all points"""
        for j in range(self.n_points):
            self._update_point_properties(j)
            
    def _recover_properties(self, j: int):
        """Recover properties at a point using neighboring values"""
        if j > 0 and j < self.n_points - 1:
            for arr in [self.rho, self.cp]:
                arr[j] = (arr[j-1] + arr[j+1]) / 2
            for arr in [self.h, self.wdot, self.wdot_creation, self.wdot_destruction]:
                arr[:, j] = (arr[:, j-1] + arr[:, j+1]) / 2
        elif j == 0 and self.n_points > 1:
            for arr in [self.rho, self.cp]:
                arr[j] = arr[j+1]
            for arr in [self.h, self.wdot, self.wdot_creation, self.wdot_destruction]:
                arr[:, j] = arr[:, j+1]
        elif j == self.n_points - 1 and j > 0:
            for arr in [self.rho, self.cp]:
                arr[j] = arr[j-1]
            for arr in [self.h, self.wdot, self.wdot_creation, self.wdot_destruction]:
                arr[:, j] = arr[:, j-1]

    # def integrate(self, t_start: float, t_end: float) -> Tuple[np.ndarray, bool]:
    #     """Integrate using Cantera reactors with proper error handling"""
    #     dt = t_end - t_start
    #     result = self.state.copy()
    #     success = True
        
    #     for j in range(self.n_points):
    #         try:
    #             # Get state
    #             T = self.state[self.i_T, j]
    #             Y = self.state[self.i_Y_start:, j]
    #             U = self.state[self.i_U, j]
                
    #             # Create new solution object using the same mechanism
    #             gas = ct.Solution(self.config.mechanism)
    #             gas.TPY = T, self.pressure, Y
                
    #             # Set up reactor
    #             reactor = ct.IdealGasConstPressureReactor(gas)
    #             sim = ct.ReactorNet([reactor])
                
    #             # Set tolerances
    #             sim.rtol = self.config.rel_tol
    #             sim.atol = self.config.abs_tol
                
    #             # Set rate multiplier if needed
    #             if self.rate_multiplier != self._last_rate_multiplier:
    #                 for i in range(gas.n_reactions):
    #                     gas.set_multiplier(i, self.rate_multiplier)
                        
    #             # Advance reactor
    #             sim.advance(dt)
                
    #             # Update result
    #             result[self.i_T, j] = reactor.thermo.T
    #             result[self.i_Y_start:, j] = reactor.thermo.Y
    #             result[self.i_U, j] = U
                
    #             # Update properties
    #             gas.TPY = reactor.thermo.T, self.pressure, reactor.thermo.Y
    #             self.rho[j] = gas.density
    #             self.cp[j] = gas.cp_mass
    #             self.h[:, j] = gas.partial_molar_enthalpies
    #             self.wdot[:, j] = gas.net_production_rates
    #             self.wdot_creation[:, j] = gas.creation_rates
    #             self.wdot_destruction[:, j] = gas.destruction_rates
                
    #         except Exception as e:
    #             print(f"Integration failed at point {j}: {str(e)}")
    #             success = False
    #             self._recover_properties(j)
        
    #     return result, success 
    
    def evaluate(self, t: float) -> np.ndarray:
        """Evaluate chemical source terms for ODE integration"""
        ddt = np.zeros_like(self.state)
        
        # Update properties and compute source terms for each point
        for j in range(self.n_points):
            # Get state for this point
            T = self.state[self.i_T, j]
            U = self.state[self.i_U, j]  # Velocity doesn't change
            Y = self.state[self.i_Y_start:, j]
            
            # Update Cantera state and get properties
            self._update_point_properties(j)
            
            # Species equations - mass fraction rate of change
            # Convert from kmol/m³/s to mass fraction rate
            ddt[self.i_Y_start:, j] = self.wdot[:, j] * self.molecular_weights / self.rho[j] + self.split_const[self.i_Y_start:, j]
            
            # Energy equation
            # q = -Σ(h_k * wdot_k)  [J/m³/s]
            q = -(self.wdot[:, j] * self.h[:, j]).sum()
            ddt[self.i_T, j] = q/(self.rho[j] * self.cp[j])
            
            # Add splitting terms if any
            ddt[:, j] += self.split_const[:, j]
        
        return ddt

    def integrate(self, t_start: float, t_end: float) -> Tuple[np.ndarray, bool]:
        """Integrate chemistry system using scipy.integrate.solve_ivp"""
        from scipy.integrate import solve_ivp
        
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
            # Solve ODE system
            solution = solve_ivp(
                fun=rhs,
                t_span=(t_start, t_end),
                y0=y0,
                method='BDF',  # Use BDF for stiff chemical systems
                rtol=self.config.rel_tol,
                atol=self.config.abs_tol,
                # Additional settings for stiff system
                jac_sparsity=None,  # Could be optimized with sparsity pattern
                first_step=min((t_end - t_start)/100, 1e-6),  # Conservative first step
                max_step=t_end - t_start
            )
            
            success = solution.success
            
            if success:
                # Reshape result back to state shape
                result = solution.y[:, -1].reshape(self.n_vars, -1)
                
                # Ensure mass fractions are valid
                Y_sum = result[self.i_Y_start:].sum(axis=0)
                result[self.i_Y_start:] /= Y_sum[np.newaxis, :]
                
                # Update properties with final state
                self.state = result
                self._update_all_properties()
                
            else:
                print(f"Integration failed: {solution.message}")
                result = self.state.copy()
                
        except Exception as e:
            print(f"Integration failed with error: {str(e)}")
            success = False
            result = self.state.copy()
        
        return result, success