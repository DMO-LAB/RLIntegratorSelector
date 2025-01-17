from typing import Optional, Dict
import numpy as np
from config import Config
from grid import Grid

class BaseSystem:
    """Base class for physical systems (diffusion, convection, source)"""
    def __init__(self, config: Dict, grid: Grid, **kwargs):
        self.config = config
        self.grid = grid
        
        # System properties
        self.n_vars: int = 0  # Number of variables
        self.n_points: int = grid.n_points
        self.n_species: int = 0  # Number of species
        
        # State and derivatives
        self.state: Optional[np.ndarray] = None  # Current state
        self.ddt: Optional[np.ndarray] = None  # Time derivatives
        
        # Split constant from operator splitting
        self.split_const: Optional[np.ndarray] = None
    
        
    def resize(self, n_points: int):
        """Resize system arrays for new grid size"""
        self.n_points = n_points
        shape = (self.n_vars, n_points)
        
        # Resize arrays
        self.state = np.zeros(shape)
        self.ddt = np.zeros(shape)
        self.split_const = np.zeros(shape)
        
    def initialize(self, n_vars: int, n_species: int):
        """Initialize system arrays"""
        self.n_vars = n_vars
        shape = (n_vars, self.n_points)
        self.n_species = n_species
        
        self.state = np.zeros(shape)
        self.ddt = np.zeros(shape)
        self.split_const = np.zeros(shape)
        self.dYdx = np.zeros((n_species, self.grid.n_points))
        
        # Index mapping for state vector
        self.i_T = 0  # Temperature index
        self.i_U = 1  # Velocity index  
        self.i_Y = slice(2, 2 + n_species)  # Species mass fractions
        
    def set_state(self, state: np.ndarray, t: float):
        """Update the system state"""
        self.state = state.copy()
        
    def evaluate(self, t: float) -> np.ndarray:
        """
        Evaluate time derivatives at current state
        To be implemented by derived classes
        """
        raise NotImplementedError