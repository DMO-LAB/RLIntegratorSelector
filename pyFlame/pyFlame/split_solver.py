from typing import Optional
import numpy as np
from pyFlame.config import Config
from pyFlame.time_stepper import AdaptiveTimeStepper
import time

class SplitSolver:
    """
    Base class implementing operator splitting for solving PDEs
    
    The solver splits the problem into three operators:
    - Diffusion (D)
    - Convection (C)
    - Production/Source terms (P)
    
    Using either Strang splitting or balanced splitting
    """
    def __init__(self, config: Config):
        self.config = config
        
        # Time information
        self.t: float = config.t_start
        self.dt: float = config.dt
        self.t_end: float = config.t_end
        
        # State variables and derivatives
        self.n_vars: int = 0  # Number of variables - set by derived class
        self.n_points: int = config.grid_points
        
        # Arrays will be initialized when n_vars is set
        self.state: Optional[np.ndarray] = None
        self.start_state: Optional[np.ndarray] = None
        
        self.step_number: int = 0
        self.time_stepper = AdaptiveTimeStepper(config)
        
        # Storage for operator contributions
        self.delta_conv: Optional[np.ndarray] = None  # Change due to convection
        self.delta_diff: Optional[np.ndarray] = None  # Change due to diffusion
        self.delta_prod: Optional[np.ndarray] = None  # Change due to production
        
        # Time derivatives from each operator
        self.ddt_conv: Optional[np.ndarray] = None
        self.ddt_diff: Optional[np.ndarray] = None
        self.ddt_prod: Optional[np.ndarray] = None
        self.ddt_cross: Optional[np.ndarray] = None
        
        # Split constants
        self.split_const_conv: Optional[np.ndarray] = None
        self.split_const_diff: Optional[np.ndarray] = None
        self.split_const_prod: Optional[np.ndarray] = None

    def initialize_arrays(self):
        """Initialize solution arrays once n_vars is set"""
        shape = (self.n_vars, self.n_points)
        
        # Initialize state arrays
        self.state = np.zeros(shape)
        self.start_state = np.zeros(shape)
        
        # Initialize operator contributions
        self.delta_conv = np.zeros(shape)
        self.delta_diff = np.zeros(shape)
        self.delta_prod = np.zeros(shape)
        
        # Initialize time derivatives
        self.ddt_conv = np.zeros(shape)
        self.ddt_diff = np.zeros(shape)
        self.ddt_prod = np.zeros(shape)
        self.ddt_cross = np.zeros(shape)
        
        # Initialize split constants
        self.split_const_conv = np.zeros(shape)
        self.split_const_diff = np.zeros(shape)
        self.split_const_prod = np.zeros(shape)

    def step(self) -> bool:
        """
        Take one global timestep using operator splitting
        Returns: True if step was successful, False otherwise
        """
        self.setup_step()
        
        # Get adaptive time step
        self.dt = self.time_stepper.compute_timestep(self)
        
        # Store initial state
        initial_state = self.state.copy()
        
        # Compute split constants
        if self.config.splitting_method == "balanced":
            self.split_const_diff = 0.25 * (self.ddt_prod + self.ddt_conv + 
                                          self.ddt_cross - 3*self.ddt_diff)
            self.split_const_conv = 0.25 * (self.ddt_prod + self.ddt_diff + 
                                          self.ddt_cross - 3*self.ddt_conv)
            self.split_const_prod = 0.5 * (self.ddt_conv + self.ddt_diff + 
                                         self.ddt_cross - self.ddt_prod)
        else:
            self.split_const_diff = self.ddt_cross
            self.split_const_conv = np.zeros_like(self.ddt_cross)
            self.split_const_prod = np.zeros_like(self.ddt_cross)
            
        # Prepare integrators
        self.prepare_integrators()
        
        # Integrate split terms
        success = self.integrate_split_terms(self.t, self.dt)
        if not success:
            return False
            
        # Calculate time derivatives for next step
        self.calculate_time_derivatives()
        
        # Update time
        self.t += self.dt
        self.step_number += 1
        
        # accepted_step = self.time_stepper.check_step(self, self.state)
        # if not accepted_step:
        #     print(f"[WARNING] - Step {self.step_number} should be rejected")
        
        return self.finish_step()

    def integrate_split_terms(self, t: float, dt: float) -> bool:
        """
        Integrate the split terms using Strang splitting sequence
        """
        # Reset deltas
        self.delta_conv[:] = 0
        self.delta_diff[:] = 0
        self.delta_prod[:] = 0
        
        # Strang splitting sequence
        success = (
            self._integrate_diffusion(t, t + 0.25*dt, 1) and  # D1 (1/4)
            self._integrate_convection(t, t + 0.5*dt, 1) and  # C1 (1/2)
            self._integrate_diffusion(t + 0.25*dt, t + 0.5*dt, 2) and  # D2 (2/4)
            self._integrate_production(t, t + dt) and  # P (full)
            self._integrate_diffusion(t + 0.5*dt, t + 0.75*dt, 3) and  # D3 (3/4)
            self._integrate_convection(t + 0.5*dt, t + dt, 2) and  # C2 (2/2)
            self._integrate_diffusion(t + 0.75*dt, t + dt, 4)  # D4 (4/4)
        )
        
        return success

    def calculate_time_derivatives(self):
        """Calculate time derivatives based on the changes over the timestep"""
        self.ddt_conv = self.delta_conv/self.dt - self.split_const_conv
        self.ddt_diff = self.delta_diff/self.dt - self.split_const_diff
        self.ddt_prod = self.delta_prod/self.dt - self.split_const_prod

    # Methods to be implemented by derived classes
    def setup_step(self):
        """Setup for the beginning of a timestep"""
        raise NotImplementedError
        
    def prepare_integrators(self):
        """Prepare integrators before split integration"""
        raise NotImplementedError
        
    def finish_step(self) -> bool:
        """Cleanup at the end of a timestep"""
        raise NotImplementedError
        
    def _integrate_diffusion(self, t_start: float, t_end: float, stage: int) -> bool:
        """Integrate diffusion terms"""
        raise NotImplementedError
        
    def _integrate_convection(self, t_start: float, t_end: float, stage: int) -> bool:
        """Integrate convection terms"""
        raise NotImplementedError
        
    def _integrate_production(self, t_start: float, t_end: float) -> bool:
        """Integrate production/source terms"""
        raise NotImplementedError