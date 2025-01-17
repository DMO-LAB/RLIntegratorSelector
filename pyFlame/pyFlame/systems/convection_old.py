
    
import numpy as np
from typing import Tuple, Optional
from scipy.integrate import solve_ivp
from numba import jit, float64, prange, vectorize
from .base import BaseSystem
from pyFlame.grid import BoundaryCondition

class ConvectionSystem(BaseSystem):
    """
    System representing convective transport with strain field
    Based on Ember's ConvectionSystemSplit
    """
    def __init__(self, config, grid):
        super().__init__(config, grid)
        
        # Flow fields
        self.v = np.zeros(grid.n_points)  # Normal velocity
        self.rho = np.zeros(grid.n_points)  # Density
        self.rho_unburned = None  # Unburned density
        self.rV = np.zeros(grid.n_points)  # r*V for cylindrical
        self.rVzero = 0.0  # BC value for rV
        
        # Strain field
        self.strain_function = config.strain_function
        
        # Working arrays
        self.dUdx = np.zeros(grid.n_points)
        self.dTdx = np.zeros(grid.n_points)
        self.dWdx = np.zeros(grid.n_points)  # For molecular weights
        self.drhodt = np.zeros(grid.n_points)
        
        # Boundary conditions
        self.T_left = None
        self.W_left = None
        self.jContBC = 0
        self.xVzero = 0.0
        
        # Parallel optimization
        self._prev_dt = None
        self._prev_max_v = None
        self._prev_min_dx = None
        self._cached_max_dt = None
        
    def set_rho_unburned(self, rho_u: float):
        """Set unburned density for strain calculations"""
        self.rho_unburned = rho_u
        
    def _calculate_V(self, t: float):
        """Calculate normal velocity field"""
        if self.grid.left_bc == BoundaryCondition.FIXED_VALUE:
            # Left boundary condition
            self.rV[0] = self.rVzero
            
            # Forward integration of continuity
            for j in range(self.n_points-1):
                self.rV[j+1] = (self.rV[j] - 
                               self.grid.metrics.dx[j] * 
                               self.grid.metrics.r_half[j] * 
                               (self.drhodt[j] + 
                                self.rho[j] * self.grid.beta * 
                                self.state[self.i_U, j]))
                                
        elif self.grid.left_bc == BoundaryCondition.ZERO_GRADIENT:
            # Stagnation point condition
            j = self.jContBC
            dVdx0 = -self.grid.metrics.r_half[j] * (
                self.drhodt[j] - self.rho[j] * self.grid.beta * self.state[self.i_U, j]
            )
            
            if self.jContBC != 0:
                dVdx0 = (0.5 * dVdx0 - 
                        0.5 * self.grid.metrics.r_half[j-1] * 
                        (self.drhodt[j-1] + 
                         self.rho[j-1] * self.grid.beta * 
                         self.state[self.i_U, j-1]))
                
            # Set velocity at stagnation point
            self.rV[j] = (self.grid.x[j] - self.xVzero) * dVdx0
            
            # Integrate away from stagnation point
            for j in range(self.jContBC, self.n_points-1):
                self.rV[j+1] = (self.rV[j] - 
                               self.grid.metrics.dx[j] * self.grid.metrics.r_half[j] * 
                               (self.drhodt[j] + 
                                self.rho[j] * self.grid.beta * 
                                self.state[self.i_U, j]))
                                
            if self.jContBC != 0:
                for j in range(self.jContBC-1, 0, -1):
                    self.rV[j-1] = (self.rV[j] + 
                                   self.grid.metrics.dx[j-1] * self.grid.metrics.r_half[j-1] * 
                                   (self.drhodt[j-1] + 
                                    self.rho[j-1] * self.grid.beta * 
                                    self.state[self.i_U, j-1]))
                                    
        # Convert rV to V
        if self.grid.alpha == 0:  # Planar
            self.v = self.rV.copy()
        else:  # Cylindrical
            self.v = self.rV / self.grid.r
            
    def _compute_derivatives(self):
        """Compute upwinded derivatives"""
        # Interior points
        for j in range(self.n_points-1):
            if self.rV[j] < 0 or j == 0:
                # Forward difference
                self.dTdx[j] = ((self.state[self.i_T, j+1] - 
                                self.state[self.i_T, j]) / self.grid.metrics.dx[j])
                self.dUdx[j] = ((self.state[self.i_U, j+1] - 
                                self.state[self.i_U, j]) / self.grid.metrics.dx[j])
            else:
                # Backward difference
                self.dTdx[j] = ((self.state[self.i_T, j] - 
                                self.state[self.i_T, j-1]) / self.grid.metrics.dx[j-1])
                self.dUdx[j] = ((self.state[self.i_U, j] - 
                                self.state[self.i_U, j-1]) / self.grid.metrics.dx[j-1])
                
    def evaluate(self, t: float) -> np.ndarray:
        """Evaluate convection terms"""
        # Update velocity field
        self._calculate_V(t)
        
        # Compute derivatives
        self._compute_derivatives()
        
        # Get strain rate
        a = self.strain_function.a(t)
        dadt = self.strain_function.dadt(t)
        
        # Initialize derivatives
        ddt = np.zeros_like(self.state)
        
        # Left boundary
        if (self.grid.left_bc == BoundaryCondition.CONTROL_VOLUME or 
            self.grid.left_bc == BoundaryCondition.WALL_FLUX):
            
            center_vol = pow(self.grid.x[1], self.grid.alpha + 1) / (self.grid.alpha + 1)
            rVzero_mod = max(self.rV[0], 0.0)
            
            ddt[self.i_T, 0] = (-rVzero_mod * 
                               (self.state[self.i_T, 0] - self.T_left) / 
                               (self.rho[0] * center_vol) + 
                               self.split_const[self.i_T, 0])
                               
        else:  # Fixed value or zero gradient
            ddt[self.i_T, 0] = self.split_const[self.i_T, 0]
            
        # Add momentum equation strain terms
        ddt[self.i_U, 0] = (self.split_const[self.i_U, 0] - 
                           self.state[self.i_U, 0]**2 + 
                           self.rho_unburned/self.rho[0] * 
                           (dadt/self.grid.beta + 
                            a**2/(self.grid.beta**2)))
                            
        # Interior points
        for j in range(1, self.n_points-1):
            # Temperature
            ddt[self.i_T, j] = (-self.v[j] * self.dTdx[j] / self.rho[j] + 
                               self.split_const[self.i_T, j])
                               
            # Momentum with strain
            ddt[self.i_U, j] = (-self.v[j] * self.dUdx[j] / self.rho[j] - 
                               self.state[self.i_U, j]**2 +
                               self.rho_unburned/self.rho[j] * 
                               (dadt/self.grid.beta + 
                                a**2/(self.grid.beta**2)) + 
                               self.split_const[self.i_U, j])
                               
            # Species
            for k in range(self.n_species):
                species_idx = self.i_Y.start + k  # Get absolute index for species k
                if self.v[j] > 0:  # Upwind
                    dYdx = ((self.state[species_idx, j] - 
                            self.state[species_idx, j-1]) / 
                           self.grid.metrics.dx[j-1])
                else:
                    dYdx = ((self.state[species_idx, j+1] - 
                            self.state[species_idx, j]) / 
                           self.grid.metrics.dx[j])
                            
                ddt[species_idx, j] = (-self.v[j] * dYdx / self.rho[j] + 
                                       self.split_const[species_idx, j])
                                       
        # Right boundary
        if self.v[-1] < 0 or self.grid.right_bc == BoundaryCondition.FIXED_VALUE:
            ddt[:, -1] = self.split_const[:, -1]
        else:  # Outflow
            dx = self.grid.metrics.dx[-1]
            # All variables use same treatment at outflow
            for k in range(self.n_vars):
                ddt[k, -1] = (self.split_const[k, -1] - 
                             self.v[-1] * (self.state[k, -1] - 
                                         self.state[k, -2]) / 
                             (dx * self.rho[-1]))
                             
        return ddt

    def _estimate_timestep(self, t_start: float, t_end: float) -> float:
        """Get maximum stable timestep"""
        dt = t_end - t_start
        max_v = np.max(np.abs(self.v))
        min_dx = np.min(self.grid.metrics.dx)
        
        if max_v > 0:
            return min(dt, 0.8 * min_dx / max_v)  # CFL condition
        return dt
    
    def resize(self, n_points: int):
        """Resize system arrays for new grid size"""
        super().resize(n_points)
        
        # Resize flow fields
        self.v = np.zeros(n_points)
        self.rho = np.zeros(n_points)
        self.rV = np.zeros(n_points)
        
        # Reset work arrays
        self.dUdx = np.zeros(n_points)
        self.dTdx = np.zeros(n_points)
        self.dWdx = np.zeros(n_points)
        self.drhodt = np.zeros(n_points)
        
        # Reset cache
        self._prev_dt = None
        self._prev_max_v = None
        self._prev_min_dx = None
        self._cached_max_dt = None
    
    def integrate(self, t_start: float, t_end: float) -> Tuple[np.ndarray, bool]:
        """
        Integrate convection terms explicitly
        Returns: (final_state, success)
        """
        # Prepare state
        y0 = np.ascontiguousarray(self.state.ravel())
        
        # Get timestep estimate
        max_dt = self._estimate_timestep(t_start, t_end)
        
        def rhs(t: float, y: np.ndarray) -> np.ndarray:
            """RHS function for the ODE solver"""
            self.state = y.reshape(self.n_vars, -1)
            return self.evaluate(t).ravel()
        
        # Use explicit RK23 method for hyperbolic equations
        try:
            solution = solve_ivp(
                rhs,
                (t_start, t_end),
                y0,
                method='RK23',          # Explicit method good for hyperbolic equations
                rtol=self.config.rel_tol,
                atol=self.config.abs_tol,
                max_step=max_dt,        # CFL condition
                first_step=max_dt/10,   # Start with smaller step for stability
                vectorized=False
            )
            
            if solution.success:
                return solution.y[:,-1].reshape(self.n_vars, -1), True
                
        except Exception as e:
            import traceback;
            traceback.print_exc()
            print(f"Convection integration failed: {str(e)}")
            
        # Return original state if integration fails
        return self.state, False


