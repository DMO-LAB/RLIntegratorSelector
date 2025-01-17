from typing import Tuple
import numpy as np
from numba import jit, njit
from .base import BaseSystem
from pyFlame.grid import BoundaryCondition

@njit
def _compute_derivatives_kernel(rV, state, dx, dTdx, dUdx, dYdx, i_T, i_U, i_Y_start, n_species):
    """JIT-compiled kernel for computing derivatives"""
    n_points = len(rV)
    
    for j in range(n_points-1):
        if rV[j] < 0 or j == 0:
            # Forward difference
            dTdx[j] = (state[i_T, j+1] - state[i_T, j]) / dx[j]
            dUdx[j] = (state[i_U, j+1] - state[i_U, j]) / dx[j]
            
            # Species derivatives
            for k in range(n_species):
                species_idx = i_Y_start + k
                dYdx[k,j] = (state[species_idx, j+1] - state[species_idx, j]) / dx[j]
        else:
            # Backward difference
            dTdx[j] = (state[i_T, j] - state[i_T, j-1]) / dx[j-1]
            dUdx[j] = (state[i_U, j] - state[i_U, j-1]) / dx[j-1]
            
            # Species derivatives
            for k in range(n_species):
                species_idx = i_Y_start + k
                dYdx[k,j] = (state[species_idx, j] - state[species_idx, j-1]) / dx[j-1]

@njit
def _calculate_V_kernel(rV, drhodt, rho, state, dx, r_half, beta, i_U, jContBC):
    """JIT-compiled kernel for velocity field calculation"""
    n_points = len(rV)
    
    # Forward integration
    for j in range(jContBC, n_points-1):
        rV[j+1] = (rV[j] - dx[j] * r_half[j] * 
                   (drhodt[j] + rho[j] * beta * state[i_U, j]))
        
    # Backward integration if needed
    if jContBC > 0:
        for j in range(jContBC-1, -1, -1):
            rV[j] = (rV[j+1] + dx[j] * r_half[j] * 
                    (drhodt[j] + rho[j] * beta * state[i_U, j]))

@njit
def _evaluate_species_kernel(v, Y, Y_left_value, dx, split_const, center_vol, is_control_vol):
    """JIT-compiled kernel for species evaluation"""
    dYdt = np.zeros_like(Y)
    n_points = len(Y)
    
    # Left boundary
    if is_control_vol:
        rvzero_mod = max(v[0], 0.0)
        if Y_left_value is not None:  # Handle None case outside JIT
            dYdt[0] = -rvzero_mod * (Y[0] - Y_left_value) / center_vol + split_const[0]
        else:
            dYdt[0] = split_const[0]
    else:
        dYdt[0] = split_const[0]
    
    # Interior points
    for j in range(1, n_points-1):
        if v[j] < 0:
            dYdx = (Y[j+1] - Y[j]) / dx[j]
        else:
            dYdx = (Y[j] - Y[j-1]) / dx[j-1]
        dYdt[j] = -v[j] * dYdx + split_const[j]
    
    # Right boundary
    if v[-1] < 0:
        dYdt[-1] = split_const[-1]
    else:
        dYdt[-1] = split_const[-1] - v[-1] * (Y[-1] - Y[-2]) / dx[-1]
        
    return dYdt

class ConvectionSystemY:
    def __init__(self, grid, n_points: int):
        self.grid = grid
        self.v = np.zeros(n_points)
        self.Y_left = None
        self.split_const = np.zeros(n_points)
        
        # Cache grid metrics
        self._dx = grid.metrics.dx
        self._alpha = grid.alpha
        self._x = grid.x
        
    def evaluate(self, t: float, Y: np.ndarray) -> np.ndarray:
        center_vol = pow(self._x[1], self._alpha + 1) / (self._alpha + 1)
        is_control_vol = (self.grid.left_bc == BoundaryCondition.CONTROL_VOLUME or
                         self.grid.left_bc == BoundaryCondition.WALL_FLUX)
        
        # Handle None values outside JIT
        Y_left_value = self.Y_left if self.Y_left is not None else Y[0]
        
        return _evaluate_species_kernel(
            self.v, Y, Y_left_value, self._dx, 
            self.split_const, center_vol, is_control_vol
        )

class ConvectionSystem(BaseSystem):
    """Optimized convection system for 1D flames"""
    def __init__(self, config, grid):
        super().__init__(config, grid)
        
        # Pre-allocate arrays
        n = grid.n_points
        self.V = np.zeros(n)
        self.v = np.zeros(n)
        self.rho = np.zeros(n)
        self.rV = np.zeros(n)
        self.rVzero = 0.0
        
        # Properties
        self.Wmx = np.zeros(n)
        self.dWdt = np.zeros(n)
        self.dUdt = np.zeros(n)
        self.dTdt = np.zeros(n)
        
        # Working arrays
        self.dUdx = np.zeros(n)
        self.dTdx = np.zeros(n)
        self.dYdx = None  # Will be initialized when n_species is known
        self.drhodt = np.zeros(n)
        
        # Work arrays for calculations
        self._work1 = np.zeros(n)
        self._work2 = np.zeros(n)
        
        # Cache grid metrics
        self._dx = grid.metrics.dx
        self._r_half = grid.metrics.r_half
        self._beta = grid.beta
        self._x = grid.x
        self._alpha = grid.alpha
        self._r = grid.r
        
        # Strain
        self.strain_function = config.strain_function
        self.rho_unburned = None
        
        # Cache for strain rates
        self._last_t = None
        self._last_a = None
        self._last_dadt = None
        
        # Boundary conditions
        self.T_left = None
        self.Y_left = None
        self.jContBC = 0
        self.xVzero = 0.0
        
        self._prev_dt = None
        
        # Species systems
        self.species_systems = []

    def initialize(self, n_vars: int, n_species: int):
        """Initialize with correct number of variables"""
        super().initialize(n_vars, n_species)
        self.dYdx = np.zeros((n_species, self.n_points))
        
        # Create species systems
        self.species_systems = [
            ConvectionSystemY(self.grid, self.n_points)
            for _ in range(n_species)
        ]
    
    def set_rho_unburned(self, rho_u: float):
        """Set unburned density"""
        self.rho_unburned = rho_u
    
    def _calculate_V(self, t: float):
        """Calculate velocity field"""
        # Set boundary condition
        self.rV[0] = self.rVzero
        
        # Calculate velocity field
        _calculate_V_kernel(
            self.rV, self.drhodt, self.rho, self.state,
            self._dx, self._r_half, self._beta,
            self.i_U, self.jContBC
        )
        
        # Convert rV to V and v (vectorized)
        if self._alpha == 0:
            self.V = self.rV
        else:
            np.divide(self.rV, self._r, out=self.V)
            
        np.divide(self.V, self.rho, out=self.v)
        
        # Update species system velocities
        for system in self.species_systems:
            system.v[:] = self.v
            
    def _compute_derivatives(self):
        """Compute spatial derivatives"""
        _compute_derivatives_kernel(
            self.rV, self.state, self._dx,
            self.dTdx, self.dUdx, self.dYdx,
            self.i_T, self.i_U, self.i_Y.start, self.n_species
        )

    def _get_strain_rates(self, t: float) -> Tuple[float, float]:
        """Get cached strain rates"""
        if t != self._last_t:
            self._last_t = t
            self._last_a = self.strain_function.a(t)
            self._last_dadt = self.strain_function.dadt(t)
        return self._last_a, self._last_dadt
                
    def evaluate(self, t: float) -> np.ndarray:
        """Optimized evaluation of convection terms"""
        # Update velocity field
        self._calculate_V(t)
        
        # Compute derivatives
        self._compute_derivatives()
        
        # Get strain rates
        a, dadt = self._get_strain_rates(t)
        
        # Initialize derivatives
        ddt = np.zeros_like(self.state)
        
        # Compute strain term once (vectorized)
        np.divide(self.rho_unburned, self.rho, out=self._work1)
        strain_term = self._work1 * (dadt/self._beta + a*a/(self._beta*self._beta))
        
        # Left boundary
        if (self.grid.left_bc == BoundaryCondition.CONTROL_VOLUME or 
            self.grid.left_bc == BoundaryCondition.WALL_FLUX):
            
            center_vol = pow(self._x[1], self._alpha + 1) / (self._alpha + 1)
            rVzero_mod = max(self.rV[0], 0.0)
            
            # Temperature
            ddt[self.i_T, 0] = (-rVzero_mod * 
                               (self.state[self.i_T, 0] - self.T_left) / 
                               (self.rho[0] * center_vol) + 
                               self.split_const[self.i_T, 0])
        else:
            ddt[self.i_T, 0] = self.split_const[self.i_T, 0]
            
        # Add momentum equation strain terms
        ddt[self.i_U, 0] = (self.split_const[self.i_U, 0] - 
                           self.state[self.i_U, 0]**2 + strain_term[0])
                            
        # Interior points (vectorized)
        j = slice(1, -1)
        ddt[self.i_T, j] = -self.v[j] * self.dTdx[j] + self.split_const[self.i_T, j]
        ddt[self.i_U, j] = (-self.v[j] * self.dUdx[j] - 
                           self.state[self.i_U, j]**2 + 
                           strain_term[j] + 
                           self.split_const[self.i_U, j])
        
        # Species equations (parallel evaluation)
        for k in range(self.n_species):
            species_idx = self.i_Y.start + k
            ddt[species_idx] = self.species_systems[k].evaluate(
                t, self.state[species_idx])
        
        # Right boundary (vectorized)
        if self.v[-1] < 0 or self.grid.right_bc == BoundaryCondition.FIXED_VALUE:
            ddt[:, -1] = self.split_const[:, -1]
            ddt[self.i_U, -1] += -self.state[self.i_U, -1]**2 + strain_term[-1]
        else:
            # Outflow
            dx_inv = 1.0 / self._dx[-1]
            grad = (self.state[:, -1] - self.state[:, -2]) * dx_inv
            ddt[:, -1] = self.split_const[:, -1] - self.v[-1] * grad
            ddt[self.i_U, -1] += (-self.state[self.i_U, -1]**2 + strain_term[-1])
        
        return ddt

    def update_bc(self, qdot: np.ndarray):
        """Update boundary conditions with vectorized operations"""
        j_qmax = np.argmax(qdot)
        T_mid = 0.5 * (np.max(self.state[self.i_T]) + np.min(self.state[self.i_T]))
        
        if self.grid.left_bc == BoundaryCondition.FIXED_VALUE:
            self.jContBC = 0
        elif self.grid.left_bc == BoundaryCondition.CONTROL_VOLUME:
            # Find T crossings using vectorized operations
            T_diff = self.state[self.i_T, 1:] - T_mid
            T_diff_prev = self.state[self.i_T, :-1] - T_mid
            crossings = np.where(T_diff * T_diff_prev <= 0)[0]
            self.jContBC = crossings[0] + 1 if len(crossings) > 0 else j_qmax
        else:  # Zero gradient
            # Find stagnation point
            if self._x[-1] > self.xVzero:
                j_start = np.searchsorted(self._x, self.xVzero)
            else:
                j_start = self.n_points - 1
                
            V_signs = np.sign(self.V)
            V_sign = V_signs[j_start]
            
            # Find sign change
            sign_changes = np.where(V_signs != V_sign)[0]
            if len(sign_changes) > 0:
                if j_start >= sign_changes[0]:
                    self.jContBC = j_start - sign_changes[0] + 1
                else:
                    self.jContBC = j_start + sign_changes[0]
            else:
                self.jContBC = j_start
            
            # Update xVzero
            if self.jContBC == 0:
                self.xVzero = (self._x[0] - 
                              self.V[0] * self._dx[0] / 
                              (self.V[1] - self.V[0]))
            elif self.jContBC == self.n_points - 1:
                self.xVzero = (self._x[-1] - 
                              self.V[-1] * self._dx[-1] / 
                              (self.V[-1] - self.V[-2]))
            else:
                self.xVzero = (self._x[self.jContBC] - 
                              self.V[self.jContBC] * self._dx[self.jContBC-1] / 
                              (self.V[self.jContBC] - self.V[self.jContBC-1]))

    def set_split_constants(self, split_const: np.ndarray):
        """Set split constants with optimized access"""
        self.split_const = split_const
        
        # Update species systems
        for k in range(self.n_species):
            self.species_systems[k].split_const = split_const[self.i_Y.start + k]
            
    def resize(self, n_points: int):
        """Resize arrays with optimized allocation"""
        super().resize(n_points)
        
        # Batch allocate arrays
        arrays = [
            'V', 'v', 'rho', 'rV', 'Wmx', 'dWdt', 'dUdt', 'dTdt',
            'dUdx', 'dTdx', 'dWdx', 'drhodt', '_work1', '_work2'
        ]
        
        for arr_name in arrays:
            setattr(self, arr_name, np.zeros(n_points))
            
        if self.n_species is not None:
            self.dYdx = np.zeros((self.n_species, n_points))
            
        # Resize species systems
        for system in self.species_systems:
            system.v = np.zeros(n_points)
            system.split_const = np.zeros(n_points)
            
    def _estimate_timestep(self, t_start: float, t_end: float) -> float:
        """Optimized timestep estimation"""
        dt = t_end - t_start
        
        # Cache max velocity and min grid spacing
        if (self._prev_dt == dt and 
            self._prev_max_v is not None and 
            self._prev_min_dx is not None):
            max_v = self._prev_max_v
            min_dx = self._prev_min_dx
        else:
            max_v = np.max(np.abs(self.v))
            min_dx = np.min(self._dx)
            self._prev_dt = dt
            self._prev_max_v = max_v
            self._prev_min_dx = min_dx
            
        # CFL condition
        if max_v > 0:
            dt_cfl = 0.5 * min_dx / max_v
            dt = min(dt, dt_cfl)
            
        return dt
    
    @staticmethod
    @jit(nopython=True)
    def _euler_step(y: np.ndarray, dydt: np.ndarray, dt: float) -> np.ndarray:
        """JIT-compiled Euler step"""
        return y + dt * dydt

    def integrate(self, t_start: float, t_end: float) -> Tuple[np.ndarray, bool]:
        """
        Optimized integration using adaptive timestepping and JIT compilation
        """
        try:
            from scipy.integrate import solve_ivp
            
            # Get initial timestep
            dt = self._estimate_timestep(t_start, t_end)
            
            # Initialize state array views
            y = self.state.ravel()
            shape = self.state.shape
            
            def rhs(t: float, y: np.ndarray) -> np.ndarray:
                """Optimized RHS evaluation"""
                self.state = y.reshape(shape)
                return self.evaluate(t).ravel()
                
            # Try adaptive RK23 integration first
            solution = solve_ivp(
                rhs,
                (t_start, t_end),
                y,
                method='RK23',
                rtol=self.config.rel_tol,
                atol=self.config.abs_tol,
                max_step=dt,
                first_step=dt/10,
                vectorized=False,
                dense_output=True
            )
            
            if solution.success:
                # Use dense output for final state
                y_final = solution.sol(t_end)
                self.state = y_final.reshape(shape)
                return self.state.copy(), True
                
            # Fallback to stabilized explicit Euler
            return self._integrate_euler(t_start, t_end)
            
        except Exception as e:
            print(f"Convection integration failed: {str(e)}")
            return self.state.copy(), False
            
    def _integrate_euler(self, t_start: float, t_end: float) -> Tuple[np.ndarray, bool]:
        """
        Fallback integration using stabilized explicit Euler
        """
        try:
            dt = self._estimate_timestep(t_start, t_end) / 4  # Conservative timestep
            t = t_start
            state_prev = self.state.copy()
            
            while t < t_end:
                # Adjust timestep
                dt = min(dt, t_end - t)
                
                # Evaluate RHS
                dydt = self.evaluate(t)
                
                # Take Euler step
                self.state = self._euler_step(state_prev, dydt, dt)
                
                # Check stability
                if np.any(np.isnan(self.state)) or np.any(np.isinf(self.state)):
                    # Restore state and reduce timestep
                    self.state = state_prev
                    dt *= 0.5
                    if dt < 1e-14:  # Minimum timestep threshold
                        return self.state.copy(), False
                    continue
                
                # Step succeeded
                t += dt
                state_prev = self.state.copy()
                
                # Try increasing timestep
                dt *= 1.1
                
            return self.state.copy(), True
            
        except Exception as e:
            print(f"Euler integration failed: {str(e)}")
            return self.state.copy(), False

    def set_state(self, state: np.ndarray, t: float):
        """Set solver state with validation"""
        if state.shape != self.state.shape:
            raise ValueError(f"Invalid state shape: {state.shape} != {self.state.shape}")
            
        self.state = state.copy()
        self._last_t = None  # Reset strain cache
