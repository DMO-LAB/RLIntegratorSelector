# flame_solver.py

from typing import Optional, Tuple
import numpy as np
import cantera as ct
from dataclasses import dataclass
import time

from pyFlame.config import Config
from pyFlame.grid import Grid, BoundaryCondition
from pyFlame.split_solver import SplitSolver
from pyFlame.systems.diffusion import DiffusionSystem
from pyFlame.systems.convection import ConvectionSystem
# from pyFlame.systems.chemistry import ChemistrySystem
from pyFlame.systems.chemistry import ChemistrySystem
from pyFlame.strain import ConstantStrain, StrainFunction, RampedStrain, OscillatingStrain
from pyFlame.time_stepper import AdaptiveTimeStepper

from scipy.interpolate import interp1d

@dataclass
class SolutionHistory:
    """Container for solution time history with adaptive grid support"""
    times: list  # List of time points
    
    # State variables
    T: list      # Temperature histories [K]
    U: list      # Velocity histories [m/s]
    Y: list      # Species mass fraction histories
    x: list      # Grid point locations
    
    # Derived quantities
    rho: list    # Density histories [kg/m³]
    cp: list     # Specific heat histories [J/kg-K]
    heat_release: list  # Heat release rate histories [W/m³]
    
    def __init__(self):
        """Initialize empty history lists"""
        self.times = []
        self.T = []
        self.U = []
        self.Y = []
        self.x = []
        self.rho = []
        self.cp = []
        self.heat_release = []
        
    def save_state(self, t: float, solver: 'FlameSolver'):
        """Save current solver state"""
        self.times.append(t)
        self.T.append(solver.state[solver.i_T].copy())
        self.U.append(solver.state[solver.i_U].copy())
        self.Y.append(solver.state[solver.i_Y].copy())
        self.x.append(solver.grid.x.copy())
        self.rho.append(solver.flame_state.rho.copy())
        self.cp.append(solver.flame_state.cp.copy())
        
        # Compute heat release rate
        q = -(solver.flame_state.wdot * solver.flame_state.h).sum(axis=0)
        self.heat_release.append(q)
        
    def get_arrays_for_saving(self):
        """
        Convert data to a format suitable for saving with varying grid sizes.
        Returns a dictionary of arrays and lists that can be saved using np.savez.
        """
        # Convert time points to array
        times_array = np.array(self.times)
        
        # Keep the varying-size arrays as lists of arrays
        return {
            'times': times_array,
            'T': np.array(self.T, dtype=object),
            'U': np.array(self.U, dtype=object),
            'Y': np.array(self.Y, dtype=object),
            'x': np.array(self.x, dtype=object),
            'rho': np.array(self.rho, dtype=object),
            'cp': np.array(self.cp, dtype=object),
            'heat_release': np.array(self.heat_release, dtype=object)
        }
        
    def save(self, filename: str):
        """Save history to file, preserving varying grid sizes"""
        np.savez(
            filename,
            **self.get_arrays_for_saving()
        )
        
    @classmethod
    def load(cls, filename: str):
        """Load history from file"""
        data = np.load(filename, allow_pickle=True)
        history = cls()
        
        history.times = data['times'].tolist()
        history.T = data['T'].tolist()
        history.U = data['U'].tolist()
        history.Y = data['Y'].tolist()
        history.x = data['x'].tolist()
        history.rho = data['rho'].tolist()
        history.cp = data['cp'].tolist()
        history.heat_release = data['heat_release'].tolist()
        
        return history

@dataclass
class FlameState:
    """Container for flame solution state"""
    T: np.ndarray  # Temperature [K]
    U: np.ndarray  # Velocity [m/s]
    Y: np.ndarray  # Species mass fractions [nSpec, nPoints]
    
    # Derived quantities
    rho: np.ndarray  # Density [kg/m³]
    cp: np.ndarray   # Specific heat [J/kg-K]
    k: np.ndarray    # Thermal conductivity [W/m-K]
    D: np.ndarray    # Species diffusion coefficients [m²/s]
    h: np.ndarray    # Species enthalpies [J/kg]
    wdot: np.ndarray # Species production rates [kg/m³-s]

class FlameSolver(SplitSolver):
    """
    Main solver class for 1D flames using operator splitting
    
    Handles:
    - Grid adaptation
    - Property evaluation
    - Integration of split systems
    - Boundary conditions
    """
    def __init__(self, config: Config):
        super().__init__(config)
        
        # Create grid
        self.grid = Grid(config)
        
        # Create gas object and initialize properties
        self.gas = ct.Solution(config.mechanism)
        self.n_species = self.gas.n_species
        self.n_vars = self.n_species + 2  # T, U, Y[:]
        
        # Create physical systems BEFORE initializing them
        self.diffusion = DiffusionSystem(config, self.grid, self.n_vars)
        self.convection = ConvectionSystem(config, self.grid)
        self.chemistry = ChemistrySystem(config, self.grid)
        
        # Initialize the systems with correct number of variables
        self.diffusion.initialize(self.n_vars, self.n_species)
        self.convection.initialize(self.n_vars, self.n_species)
        self.chemistry.initialize(self.n_vars, self.n_species)
        
                
        # Index mapping for state vector
        self.i_T = 0  # Temperature index
        self.i_U = 1  # Velocity index  
        self.i_Y = slice(2, 2 + self.n_species)  # Species mass fractions
            
        # Add history storage
        self.history = SolutionHistory()
        self.save_interval = config.save_interval if hasattr(config, 'save_interval') else 10
        
        # Set up strain function
        if config.flame_type == "diffusion":
            if hasattr(config, 'strain_function'):
                self.strain_function = config.strain_function
            else:
                self.strain_function = ConstantStrain(config.strain_rate)
        
        # Initialize arrays
        self.initialize_arrays()
        
        # Current flame state
        self.flame_state: Optional[FlameState] = None
        
        # Boundary conditions
        self._setup_boundary_conditions()
        
        # For diffusion flames, set rho_unburned from boundary conditions
        if config.flame_type == "diffusion":
            if config.fuel_left:
                self.convection.set_rho_unburned(self.rho_left)  # Fuel side density
            else:
                self.convection.set_rho_unburned(self.rho_right)  # Oxidizer side density
        
    def smooth_profile(self, profile: np.ndarray) -> np.ndarray:
        """
        Implement profile smoothing matching C++ implementation.
        This matches the utils::smooth function in the C++ code.
        """
        smoothed = profile.copy()
        n = len(profile)
        
        if profile.ndim == 1:
            # For 1D arrays (T, U, etc)
            for j in range(1, n-1):
                smoothed[j] = 0.25 * profile[j-1] + 0.5 * profile[j] + 0.25 * profile[j+1]
        else:
            # For 2D arrays (species mass fractions)
            for k in range(profile.shape[0]):
                for j in range(1, n-1):
                    smoothed[k,j] = 0.25 * profile[k,j-1] + 0.5 * profile[k,j] + 0.25 * profile[k,j+1]
                    
        return smoothed

    def initialize_strain_field(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Initialize strain-dependent velocity fields (U and V)
        matching C++ implementation
        """
        x = self.grid.x
        n = len(x)
        U = np.zeros(n)
        V = np.zeros(n)
        
        # Get initial strain rate
        a0 = self.strain_function.a(self.t)
        beta = self.grid.beta
        
        # Calculate density field
        rho = np.zeros(n)
        for j in range(n):
            self.gas.TPY = self.state[self.i_T, j], self.config.pressure, self.state[self.i_Y][:,j]
            rho[j] = self.gas.density
            # Set tangential velocity matching C++ implementation
            U[j] = a0 / beta * np.sqrt(self.rho_unburned/rho[j])
        
        # Smooth velocity profile
        for _ in range(2):  # C++ uses 2 smoothing passes
            U = self.smooth_profile(U)
            
        # Calculate normal velocity (V) based on geometry
        if self.config.flame_type == "diffusion":
            if self.grid.alpha == 0:  # Planar/disc flame
                V = -a0 * x
            else:  # Cylindrical flame
                Rs = self.config.Rs  # Stagnation radius
                alpha = self.grid.alpha
                V = -a0/(alpha + 1) * (
                    np.abs(Rs) * np.power(Rs, alpha) / 
                    np.power(x, alpha) - x
                )
                
            # Set stagnation point velocity
            if self.config.fuel_left:
                j_stag = len(x) // 4  # Stagnation point on fuel side
            else:
                j_stag = 3 * len(x) // 4  # Stagnation point on oxidizer side
                
            V[j_stag] = 0.0
            
            # Integrate continuity equation for V
            for j in range(j_stag + 1, n):
                dx = x[j] - x[j-1]
                V[j] = V[j-1] - beta * rho[j] * U[j] * dx
                
            for j in range(j_stag - 1, -1, -1):
                dx = x[j+1] - x[j]
                V[j] = V[j+1] + beta * rho[j] * U[j] * dx
                
        return U, V

    def _setup_boundary_conditions(self):
        """
        Setup boundary conditions exactly matching C++ implementation
        """
        if self.config.flame_type == "diffusion":
            # Fuel stream properties
            self.gas.TPX = self.config.T_fuel, self.config.pressure, f"{self.config.fuel}:1.0"
            self.Y_fuel = self.gas.Y.copy()
            self.T_fuel = self.config.T_fuel
            rho_fuel = self.gas.density
            
            # Oxidizer stream properties
            self.gas.TPX = self.config.T_oxidizer, self.config.pressure, self.config.oxidizer
            self.Y_oxidizer = self.gas.Y.copy()
            self.T_oxidizer = self.config.T_oxidizer
            rho_oxidizer = self.gas.density
            
            # Set boundary conditions based on fuel position
            if self.config.fuel_left:
                self.rho_left = rho_fuel
                self.rho_right = rho_oxidizer
                self.T_left = self.T_fuel
                self.T_right = self.T_oxidizer
                self.Y_left = self.Y_fuel
                self.Y_right = self.Y_oxidizer
                self.rho_unburned = rho_fuel  # For strain calculations
            else:
                self.rho_left = rho_oxidizer
                self.rho_right = rho_fuel
                self.T_left = self.T_oxidizer
                self.T_right = self.T_fuel
                self.Y_left = self.Y_oxidizer
                self.Y_right = self.Y_fuel
                self.rho_unburned = rho_oxidizer  # For strain calculations
                
            # Apply boundary conditions to state vector
            self.state[self.i_T, 0] = self.T_left
            self.state[self.i_T, -1] = self.T_right
            self.state[self.i_Y][:, 0] = self.Y_left
            self.state[self.i_Y][:, -1] = self.Y_right

    def initialize_solution(self):
        """
        Initialize solution matching C++ implementation
        """
        x = self.grid.x
        n = len(x)
        
        # Setup boundary conditions first
        self._setup_boundary_conditions()
        
        if self.config.flame_type == "diffusion":
            # Initialize mixing layer
            x_center = 0.0
            width = self.config.mixing_width if hasattr(self.config, 'mixing_width') else 0.002
            
            # Temperature profile
            xi = (x - x_center) / width
            phi = 0.5 * (1 + np.tanh(2*xi))
            
            if self.config.fuel_left:
                self.state[self.i_T] = self.T_fuel + (self.T_oxidizer - self.T_fuel) * phi
                for k in range(self.n_species):
                    self.state[self.i_Y][k] = (self.Y_fuel[k] + 
                                             (self.Y_oxidizer[k] - self.Y_fuel[k]) * phi)
            else:
                self.state[self.i_T] = self.T_oxidizer + (self.T_fuel - self.T_oxidizer) * phi
                for k in range(self.n_species):
                    self.state[self.i_Y][k] = (self.Y_oxidizer[k] + 
                                             (self.Y_fuel[k] - self.Y_oxidizer[k]) * phi)
                    
            # Apply smoothing matching C++ implementation
            smooth_count = self.config.smooth_count if hasattr(self.config, 'smooth_count') else 4
            for _ in range(smooth_count):
                self.state[self.i_T] = self.smooth_profile(self.state[self.i_T])
                self.state[self.i_Y] = self.smooth_profile(self.state[self.i_Y])
                
            # Initialize velocity fields
            U, V = self.initialize_strain_field()
            self.state[self.i_U] = U
            self.convection.v = V
            
            # Enforce boundary conditions again after smoothing
            self.state[self.i_T, 0] = self.T_left
            self.state[self.i_T, -1] = self.T_right
            self.state[self.i_Y][:, 0] = self.Y_left
            self.state[self.i_Y][:, -1] = self.Y_right
            
            # Correct mass fractions
            self.correct_mass_fractions()
            
            # Final property update
            self.update_properties()
            
            # Fixed value boundary conditions on both sides
            self.grid.left_bc = BoundaryCondition.FIXED_VALUE
            self.grid.right_bc = BoundaryCondition.FIXED_VALUE
            
    def update_properties(self):
        """Update thermodynamic and transport properties"""
        n = self.grid.n_points
        
        # Initialize property arrays
        rho = np.zeros(n)
        cp = np.zeros(n)
        k = np.zeros(n)
        D = np.zeros((self.n_species, n))
        h = np.zeros((self.n_species, n))
        wdot = np.zeros((self.n_species, n))
        
        # Check temperature bounds
        if np.any(self.state[self.i_T] <= 0):
            raise ValueError("Negative or zero temperature detected")
        
        # Loop over points
        for j in range(n):
            T = self.state[self.i_T, j]
            Y = self.state[self.i_Y, j]
            
            # Ensure mass fractions sum to 1
            Y_sum = np.sum(Y)
            if abs(Y_sum - 1.0) > 1e-10:
                Y = Y / Y_sum
                self.state[self.i_Y, j] = Y
                
            # Update properties
            self.gas.TPY = T, self.config.pressure, Y
            rho[j] = max(self.gas.density, 1e-10)  # Ensure positive density
            cp[j] = max(self.gas.cp_mass, 1e-10)   # Ensure positive cp
            k[j] = max(self.gas.thermal_conductivity, 1e-10)
            
            D[:, j] = np.maximum(self.gas.mix_diff_coeffs, 1e-10)
            h[:, j] = self.gas.partial_molar_enthalpies
            wdot[:, j] = self.gas.net_production_rates
            
        # Store in flame state
        self.flame_state = FlameState(
            T=self.state[self.i_T],
            U=self.state[self.i_U],
            Y=self.state[self.i_Y],
            rho=rho,
            cp=cp,
            k=k,
            D=D,
            h=h,
            wdot=wdot
        )
        
        # Update system properties
        self._update_diffusion_coeffs()
        self._update_convection_coeffs()
        
    def _update_diffusion_coeffs(self):
        """Update diffusion coefficients"""
        # Temperature equation
        k = self.flame_state.k  # Thermal conductivity
        rho = self.flame_state.rho
        cp = self.flame_state.cp
        
        self.diffusion.D[self.i_T] = k
        self.diffusion.B[self.i_T] = 1.0 / (rho * cp)
        
        # Species equations 
        D = self.flame_state.D  # Species diffusion coefficients
        for k in range(self.n_species):
            self.diffusion.D[self.i_Y][k] = D[k]
            self.diffusion.B[self.i_Y][k] = 1.0 / rho
            
        # Velocity equation (no diffusion)
        self.diffusion.D[self.i_U] = np.zeros(self.grid.n_points)
        self.diffusion.B[self.i_U] = np.zeros(self.grid.n_points)
        
    def _update_convection_coeffs(self):
        """Update convection system coefficients"""
        if self.config.flame_type == "diffusion":
            # Get current strain rate
            a = self.strain_function.a(self.t)
            dadt = self.strain_function.dadt(self.t)
            
            # Update velocity field based on strain
            x = self.grid.x
            beta = self.grid.beta  # Get from grid
            
            # Tangential velocity (U)
            self.state[self.i_U] = a * x / beta
            
            # Normal velocity (V) for potential flow
            if self.grid.alpha == 0:  # Planar/disc
                self.convection.v = -a * x
            else:  # Cylindrical
                Rs = self.config.Rs  # Stagnation surface radius
                alpha = self.grid.alpha
                self.convection.v = -a/(alpha + 1) * (
                    np.abs(Rs) * np.power(Rs, alpha) / 
                    np.power(x, alpha) - x
                )
            
            # Add strain contribution to momentum equation
            rhou = self.rho_left  # Unburned density
            beta = self.grid.beta
            a = self.strain_function.a(self.t)
            self.chemistry.momentum_source = (
                rhou/self.flame_state.rho * (
                    dadt/beta + a*a/(beta*beta)
                )
            )
            # self.chemistry.momentum_source = (
            #     rhou/self.flame_state.rho * (
            #         dadt/beta + a*a/(beta*beta)
            #     )
            # )
            self.convection.rho = self.flame_state.rho
        
        else:
            self.convection.v = self.flame_state.U
            self.convection.rho = self.flame_state.rho
        
    def setup_step(self):
        """Prepare for the next timestep"""
        # Update properties
        self.update_properties()
        
        # Check if grid adaptation is needed
        # Only adapt every N steps to avoid excessive adaptation
        if (hasattr(self.config, 'adaptation_interval') and  # Check if parameter exists
            self.step_number % self.config.adaptation_interval == 0):
            if self.grid.adapt_grid(self.state):
                # Interpolate solution to new grid
                self._interpolate_solution()
                
                # Update properties on new grid
                self.update_properties()
                
                # Update system matrices for new grid
                self.diffusion._build_matrices()
                
        # self.step_number += 1
        
    def correct_mass_fractions(self):
        """Correct mass fractions using Cantera"""
        for j in range(self.state.shape[1]):
            # Set the gas state with current mass fractions and temperature
            self.gas.TPY = self.state[self.i_T, j], self.config.pressure, self.state[self.i_Y, j]
            # Get the corrected mass fractions
            self.state[self.i_Y, j] = self.gas.Y
        
    def finish_step(self) -> bool:
        """
        Cleanup at the end of a timestep
        Returns: True if step was successful and can continue, False otherwise
        """
        # First correct mass fractions
        self.correct_mass_fractions()
        
        # Check for numerical stability
        if (np.any(np.isnan(self.state)) or 
            np.any(np.isinf(self.state))):
            print("Solution contains NaN or Inf values")
            return False
            
        # Check for physical constraints
        if np.any(self.state[self.i_T] <= 0):  # Temperature must be positive
            print("Negative temperature detected")
            return False
            
        if np.any(self.state[self.i_Y] < -1e-8):  # Mass fractions must be non-negative
            print("Negative mass fractions detected")
            return False
            
        # Check mass fraction sum
        Y_sum = np.sum(self.state[self.i_Y], axis=0)
        if np.any(np.abs(Y_sum - 1.0) > 1e-8):
            print("Mass fractions do not sum to 1")
            return False
            
        # Update properties if step was successful
        self.update_properties()
        
        return True
            
    def prepare_integrators(self):
        """Prepare split integrators"""
        # Set states
        self.diffusion.set_state(self.state, self.t)
        self.convection.set_state(self.state, self.t)
        self.chemistry.set_state(self.state, self.t)
        
    def _integrate_diffusion(self, t_start: float, t_end: float, stage: int) -> bool:
        """Integrate diffusion terms"""
        start_time = time.time()
        self.diffusion.set_state(self.state, t_start)
        result, success = self.diffusion.integrate(t_start, t_end)
        
        if success:
            self.delta_diff += result - self.state
            self.state = result
        #print(f"Diffusion stage {stage} took {time.time() - start_time:.2f} s")
        return success
        
    def _integrate_convection(self, t_start: float, t_end: float, stage: int) -> bool:
        """Integrate convection terms"""
        start_time = time.time()
        self.convection.set_state(self.state, t_start)
        result, success = self.convection.integrate(t_start, t_end)
        
        if success:
            self.delta_conv += result - self.state
            self.state = result
        #if time.time() - start_time > 5:
        #print(f"Convection stage {stage} took {time.time() - start_time:.2f} s")
        return success
        
    def _integrate_production(self, t_start: float, t_end: float) -> bool:
        """Integrate chemical source terms"""
        start_time = time.time()
        self.chemistry.set_state(self.state, t_start)

        if self.config.use_cantera:
            result, success = self.chemistry.integrate_with_cantera(t_start, t_end)
        elif self.config.use_parallel:
            result, success = self.chemistry.integrate_parallel(t_start, t_end)
        else:
            result, success = self.chemistry.integrate(t_start, t_end)
        
        if success:
            self.delta_prod += result - self.state
            self.state = result
        
        #if time.time() - start_time > 5:
        #print(f"Chemistry stage took {time.time() - start_time:.2f} s")
        
        return success
        
    def _interpolate_solution(self):
        """Interpolate solution after grid adaptation"""
        x_old = self.grid.x_old
        x_new = self.grid.x
        n_points_new = len(x_new)
        
        # Create interpolation functions for each variable
        interpolators = []
        for i in range(self.n_vars):
            interpolators.append(
                interp1d(
                    x_old,
                    self.state[i],
                    kind='cubic',
                    bounds_error=False,
                    fill_value='extrapolate'
                )
            )
        
        # Interpolate to new grid
        new_state = np.zeros((self.n_vars, n_points_new))
        for i in range(self.n_vars):
            new_state[i] = interpolators[i](x_new)
            
        # Ensure mass fractions sum to 1
        Y_sum = new_state[self.i_Y].sum(axis=0)
        new_state[self.i_Y] /= Y_sum[np.newaxis, :]
        
        self.correct_mass_fractions()
        
        # Update state
        self.state = new_state
        
        # Resize system arrays
        self.diffusion.resize(n_points_new)
        self.convection.resize(n_points_new)
        self.chemistry.resize(n_points_new)
        
        # Resize split solver arrays
        shape = (self.n_vars, n_points_new)
        self.start_state = np.zeros(shape)
        self.delta_conv = np.zeros(shape)
        self.delta_diff = np.zeros(shape)
        self.delta_prod = np.zeros(shape)
        self.ddt_conv = np.zeros(shape)
        self.ddt_diff = np.zeros(shape)
        self.ddt_prod = np.zeros(shape)
        self.ddt_cross = np.zeros(shape)
        self.split_const_conv = np.zeros(shape)
        self.split_const_diff = np.zeros(shape)
        self.split_const_prod = np.zeros(shape)
            
    def solve(self) -> bool:
        """
        Solve the flame from initial condition to steady state
        Returns: True if successful
        """
        # Initialize solution
        self.initialize_solution()
        
         # Save initial state
        self.history.save_state(self.t, self)
        
        while self.t < self.t_end:
            # Take a step
            success = self.step()
            if not success:
                print(f"Solution failed at t = {self.t}")
                return False
                
            # Check for steady state
            if self.check_steady_state():
                print(f"Reached steady state at t = {self.t}")
                break

            # Save final state if needed
            if self.step_number % self.save_interval == 0:
                self.history.save_state(self.t, self)
              
            if self.step_number % 50 == 0:  
                print(f"t = {self.t:.4e} s - max T = {np.max(self.state[self.i_T]):.2f} K - dt = {self.dt:.2e} s - flame speed = {self.calculate_flame_speed():.2e} m/s")
                grid_metrics, conservation = self.get_solution_metrics()
                print(f"Grid quality: {grid_metrics}")
                print(f"Conservation errors: {conservation}")
                print(self.analyze_heat_release())
                # self.history.save("flame_history.npz")
                
        return True
        
    def check_steady_state(self) -> bool:
        """Check if steady state has been reached"""
        # Compute RMS change in temperature
        dTdt = (self.ddt_conv[self.i_T] + 
                self.ddt_diff[self.i_T] + 
                self.ddt_prod[self.i_T])
        
        T_rms = np.sqrt(np.mean(dTdt**2))
        T_ref = np.mean(self.state[self.i_T])
        
        #print(f"RMS change in T: {T_rms:.2e} K/s - T_ref = {T_ref:.2f} K - rel_tol = {T_rms/T_ref:.2e}")
        
        return T_rms/T_ref < self.config.rel_tol
    
    def get_solution_metrics(self):
        """Calculate solution quality metrics"""
        # Grid quality
        dx = np.diff(self.grid.x)
        grid_metrics = {
            'min_dx': dx.min(),
            'max_dx': dx.max(),
            'mean_dx': dx.mean(),
            'grid_points_in_flame': np.sum(dx < self.config.flame_res)
        }
        
        # Conservation errors
        mass_in = self.flame_state.rho[0] * self.flame_state.U[0]
        mass_out = self.flame_state.rho[-1] * self.flame_state.U[-1]
        conservation = {
            'mass_flux_error': (mass_out - mass_in)/mass_in,
            'species_sum_error': np.max(np.abs(np.sum(self.state[self.i_Y], axis=0) - 1.0))
        }
        
        return grid_metrics, conservation
    
    
    def analyze_heat_release(self):
        """Analyze heat release rate and flame thickness"""
        # Heat release rate
        q = -(self.flame_state.wdot * self.flame_state.h).sum(axis=0)
        
        # Flame thickness based on temperature gradient
        dTdx = np.gradient(self.state[self.i_T], self.grid.x)
        max_grad = np.max(np.abs(dTdx))
        delta_T = (self.state[self.i_T].max() - self.state[self.i_T].min()) / max_grad
        # get maximum temperature and location of maximum temperature gradient
        max_T = np.max(self.state[self.i_T])
        max_grad_loc = np.argmax(np.abs(dTdx))
        
        mid_point = len(self.grid.x) // 2
        mid_point_temperature = self.state[self.i_T][mid_point]
        # check if H2O is present in the flame
        if self.gas.species_index('H2O') != -1:
            H2O_index = self.gas.species_index('H2O')
            H2O_mass_fraction = self.state[self.i_Y][H2O_index][mid_point]
        else:
            H2O_mass_fraction = 0.0 # no H2O in the flame

        return {
            'total_heat_release': np.trapz(q, self.grid.x),
            'peak_heat_release': np.max(q),
            'flame_thickness': delta_T,
            'max_temperature': max_T,
            'max_temperature_gradient': max_grad,
            'max_temperature_gradient_location': self.grid.x[max_grad_loc],
            'mid_point_temperature': mid_point_temperature,
            'H2O_mass_fraction at mid-point': H2O_mass_fraction
        }
        
    def calculate_flame_speed(self):
        """Calculate flame speed based on mass flux"""
        # Find location of maximum temperature gradient
        dTdx = np.gradient(self.state[self.i_T], self.grid.x)
        j_flame = np.argmax(np.abs(dTdx))
        
        # Calculate mass flux at flame location
        mass_flux = self.flame_state.rho[j_flame] * self.flame_state.U[j_flame]
        
        return mass_flux / self.flame_state.rho[0]  # [m/s]