from dataclasses import dataclass
from enum import Enum
import numpy as np
from typing import Optional, List, Dict, Any


class BoundaryCondition(Enum):
    """Enumeration of possible boundary conditions"""
    FIXED_VALUE = "fixed_value"
    ZERO_GRADIENT = "zero_gradient"
    CONTROL_VOLUME = "control_volume"
    WALL_FLUX = "wall_flux"

@dataclass
class GridMetrics:
    """Container for grid metrics used in finite difference calculations"""
    dx: np.ndarray  # Grid spacing
    dx_left: np.ndarray  # Distance to left neighbor
    dx_right: np.ndarray  # Distance to right neighbor
    
    # Finite difference coefficients for first derivatives
    alpha: np.ndarray  # Coefficient for left point
    beta: np.ndarray   # Coefficient for center point
    gamma: np.ndarray  # Coefficient for right point
    r: np.ndarray  # Geometric radius at each point
    r_half: np.ndarray  # Geometric radius at half points (cell faces)
    
@dataclass
class AdaptationCriteria:
    """Parameters controlling grid adaptation"""
    grad_tol: float = 0.2  # Relative gradient tolerance
    curv_tol: float = 0.1  # Relative curvature tolerance
    max_ratio: float = 1.5  # Maximum ratio between adjacent cells
    min_points: int = 200   # Minimum number of points
    max_points: int = 10000  # Maximum number of points
    flame_res: float = 0.1 # Minimum resolution in flame (points per mm)


class Grid:
    """
    Class handling the computational grid and its metrics
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        
        # Grid points
        self.x: np.ndarray = np.linspace(config.x_min, config.x_max, config.grid_points)
        self.x_old: np.ndarray = self.x.copy()
        self.n_points = config.grid_points
        
        # Geometry parameters (defaults to planar)
        self.alpha = 0  # Power for r-coordinate 
        self.beta = 1   # Velocity scaling
        self.r = 1      # Geometric factor: 1=planar, 2=cylindrical, 3=spherical
        
        # Set geometry based on configuration
        if hasattr(config, 'flame_geometry'):
            if config.flame_geometry == 'cylindrical':
                self.alpha = 1
                self.beta = 1
                self.r = 2
            elif config.flame_geometry == 'spherical':
                self.alpha = 2
                self.beta = 1
                self.r = 3
            elif config.flame_geometry == 'disc':
                self.alpha = 0
                self.beta = 2
                self.r = 2
            elif config.flame_geometry == 'planar':
                self.alpha = 0
                self.beta = 1
                self.r = 1
        
        # Grid metrics
        self.metrics = self.compute_metrics()
        
        # Adaptation parameters
        self.criteria = AdaptationCriteria()
        
        # Grid history for debugging
        self.adaptation_history: List[np.ndarray] = []
        
        self.left_bc = BoundaryCondition.FIXED_VALUE
        self.right_bc = BoundaryCondition.FIXED_VALUE

        
    def update_metrics(self):
        """Update grid metrics after any change to the grid"""
        # Grid spacing
        dx = np.diff(self.x)
        dx_left = np.zeros_like(self.x)
        dx_right = np.zeros_like(self.x)
        
        dx_left[1:] = dx
        dx_right[:-1] = dx
        
        # Finite difference coefficients for interior points
        alpha = np.zeros_like(self.x)
        beta = np.zeros_like(self.x)
        gamma = np.zeros_like(self.x)
        
        # Geometric radius
        if hasattr(self.config, 'flame_geometry') and self.config.flame_geometry in ['cylindrical', 'spherical', 'disc']:
            r = np.abs(self.x)  # Use absolute distance from centerline as radius
        else:
            r = np.ones_like(self.x)  # Planar case
            
        # Compute r at half points (cell faces)
        r_half = np.zeros(len(r)-1)  # One less point than cell centers
        for j in range(len(r)-1):
            r_half[j] = 0.5 * (r[j] + r[j+1])  # Linear interpolation
            
        # Interior points
        for i in range(1, self.n_points-1):
            dl = dx_left[i]
            dr = dx_right[i]
            alpha[i] = -dr / (dl * (dl + dr))
            beta[i] = (dr - dl) / (dl * dr)
            gamma[i] = dl / (dr * (dl + dr))
            
        self.metrics = GridMetrics(dx, dx_left, dx_right, alpha, beta, gamma, r, r_half)
    
    # def _find_points_to_add(self, indicators: np.ndarray) -> List[int]:
    #     """Find indices where points should be added"""
    #     points_to_add = []
        
    #     # Don't add points if we're already at max_points
    #     if self.n_points >= self.criteria.max_points:
    #         return points_to_add
            
    #     # Calculate how many points we can add
    #     points_available = self.criteria.max_points - self.n_points
        
    #     # Check gradient criteria
    #     for i in range(self.n_points - 1):
    #         if indicators[i] > self.criteria.grad_tol:
    #             dx = self.x[i+1] - self.x[i]
                
    #             # Check if spacing is too large
    #             if dx > self.criteria.flame_res:
    #                 points_to_add.append(i)
                    
    #     # Check grid stretching
    #     for i in range(1, self.n_points - 1):
    #         dx_left = self.x[i] - self.x[i-1]
    #         dx_right = self.x[i+1] - self.x[i]
            
    #         if max(dx_left/dx_right, dx_right/dx_left) > self.criteria.max_ratio:
    #             points_to_add.append(i)
        
    #     # Limit number of points to add based on available space
    #     if len(points_to_add) > points_available:
    #         # Sort by indicator value to keep the most important points
    #         points_with_indicators = [(i, indicators[i]) for i in points_to_add]
    #         points_with_indicators.sort(key=lambda x: x[1], reverse=True)
    #         points_to_add = [p[0] for p in points_with_indicators[:points_available]]
            
    #     return points_to_add
        
    # def _find_points_to_remove(self, indicators: np.ndarray) -> List[int]:
    #     """Find indices where points can be removed"""
    #     points_to_remove = []
        
    #     # Check if we can remove points while respecting min_points
    #     points_removable = self.n_points - self.criteria.min_points
        
    #     if points_removable > 0:
    #         # Check gradient criteria
    #         candidates = []
    #         for i in range(1, self.n_points - 1):
    #             if (indicators[i] < 0.5 * self.criteria.grad_tol and
    #                 indicators[i-1] < 0.5 * self.criteria.grad_tol and
    #                 indicators[i+1] < 0.5 * self.criteria.grad_tol):
                    
    #                 # Check if removing point maintains adequate resolution
    #                 dx = self.x[i+1] - self.x[i-1]
    #                 if dx < 2 * self.criteria.flame_res:
    #                     candidates.append((i, indicators[i]))
            
    #         # Sort candidates by indicator value (remove points with lowest indicators first)
    #         candidates.sort(key=lambda x: x[1])
            
    #         # Take only as many points as we can remove
    #         points_to_remove = [p[0] for p in candidates[:points_removable]]
                        
    #     return points_to_remove

    def adapt_grid(self, solution: np.ndarray) -> bool:
        """
        Adapt grid based on solution features
        Returns: True if grid was modified
        """
        # Early return if we're already at min_points and can't remove any
        if self.n_points <= self.criteria.min_points:
            print("[INFO] - Grid already at minimum points")
            return False
            
        # Store old grid
        self.x_old = self.x.copy()
        
        # Get temperature and species mass fractions
        T = solution[0]  # Temperature is first component
        Y = solution[2:] # Species mass fractions
        
        # 1. Compute adaptation indicators
        indicators = self._compute_indicators(T, Y)
        
        # 2. Determine points to add/remove
        points_to_add = self._find_points_to_add(indicators)
        points_to_remove = self._find_points_to_remove(indicators)
        
        # 3. Apply modifications if needed
        modified = False
        
        if len(points_to_add) > 0:
            print(f"[INFO] - Adding {len(points_to_add)} points")
            self._add_points(points_to_add)
            modified = True
            
        if len(points_to_remove) > 0:
            print(f"[INFO] - Removing {len(points_to_remove)} points")
            self._remove_points(points_to_remove)
            modified = True
            
        # 4. Update grid metrics if modified
        if modified:
            print(f"[INFO] - Grid adapted: {self.n_points} points")
            self.metrics = self.compute_metrics()
            self.adaptation_history.append(self.x.copy())
            
        return modified
        
    def _compute_indicators(self, T: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Compute adaptation indicators based on solution features
        Returns array of indicators for each point
        """
        # Initialize indicators
        indicators = np.zeros(self.n_points)
        
        # 1. Temperature gradients
        dTdx = np.gradient(T, self.x)
        d2Tdx2 = np.gradient(dTdx, self.x)
        
        # Normalize temperature indicators
        T_range = T.max() - T.min()
        grad_T = np.abs(dTdx) / (T_range / (self.x[-1] - self.x[0]))
        curv_T = np.abs(d2Tdx2) / (T_range / (self.x[-1] - self.x[0])**2)
        
        # 2. Species gradients
        grad_Y = np.zeros_like(indicators)
        curv_Y = np.zeros_like(indicators)
        
        for k in range(Y.shape[0]):
            if Y[k].max() > 1e-4:  # Only consider significant species
                dYdx = np.gradient(Y[k], self.x)
                d2Ydx2 = np.gradient(dYdx, self.x)
                
                # Normalize species indicators
                Y_range = Y[k].max() - Y[k].min()
                if Y_range > 0:
                    grad_Y += np.abs(dYdx) / (Y_range / (self.x[-1] - self.x[0]))
                    curv_Y += np.abs(d2Ydx2) / (Y_range / (self.x[-1] - self.x[0])**2)
        
        # Combine indicators
        indicators = np.maximum(grad_T, grad_Y) + self.criteria.curv_tol * np.maximum(curv_T, curv_Y)
        
        return indicators
        
    def _find_points_to_add(self, indicators: np.ndarray) -> List[int]:
        """Find indices where points should be added"""
        points_to_add = []
        
        # Check gradient criteria
        for i in range(self.n_points - 1):
            if indicators[i] > self.criteria.grad_tol:
                dx = self.x[i+1] - self.x[i]
                
                # Check if spacing is too large
                if dx > self.criteria.flame_res:
                    points_to_add.append(i)
                    
        # Check grid stretching
        for i in range(1, self.n_points - 1):
            dx_left = self.x[i] - self.x[i-1]
            dx_right = self.x[i+1] - self.x[i]
            
            if max(dx_left/dx_right, dx_right/dx_left) > self.criteria.max_ratio:
                points_to_add.append(i)
                
        # Limit number of points to add based on available space
        if self.n_points + len(points_to_add) > self.criteria.max_points:
            print(f"[INFO] - Not enough space to add {len(points_to_add)} points")
            points_to_add = points_to_add[:self.criteria.max_points - self.n_points]
                
        return points_to_add
        
    def _find_points_to_remove(self, indicators: np.ndarray) -> List[int]:
        """Find indices where points can be removed"""
        points_to_remove = []
        
        if self.n_points > self.criteria.min_points:
            # Check gradient criteria
            for i in range(1, self.n_points - 1):
                if (indicators[i] < 0.5 * self.criteria.grad_tol and
                    indicators[i-1] < 0.5 * self.criteria.grad_tol and
                    indicators[i+1] < 0.5 * self.criteria.grad_tol):
                    
                    # Check if removing point maintains adequate resolution
                    dx = self.x[i+1] - self.x[i-1]
                    if dx < 2 * self.criteria.flame_res:
                        points_to_remove.append(i)
        
        if points_to_remove and self.n_points - len(points_to_remove) < self.criteria.min_points:
            #print(f"[INFO] - Not enough {len(points_to_remove)} removable points to reach min_points")
            points_to_remove = []
        return points_to_remove
        
    def _add_points(self, indices: List[int]):
        """Add points at specified indices"""
        # Sort indices in descending order to maintain indexing
        indices = sorted(indices, reverse=True)
        
        for i in indices:
            # New point location
            x_new = 0.5 * (self.x[i] + self.x[i+1])
            
            # Insert new point
            self.x = np.insert(self.x, i+1, x_new)
            
        self.n_points = len(self.x)
        
    def _remove_points(self, indices: List[int]):
        """Remove points at specified indices"""
        # Sort indices in descending order to maintain indexing
        indices = sorted(indices, reverse=True)
        
        # Remove points
        self.x = np.delete(self.x, indices)
        self.n_points = len(self.x)
        
    def compute_metrics(self) -> GridMetrics:
        """Compute grid metrics"""
        # Grid spacing
        dx = np.diff(self.x)
        dx_left = np.zeros_like(self.x)
        dx_right = np.zeros_like(self.x)
        
        dx_left[1:] = dx
        dx_right[:-1] = dx
        
        # Finite difference coefficients
        alpha = np.zeros_like(self.x)
        beta = np.zeros_like(self.x)
        gamma = np.zeros_like(self.x)
        
        # Geometric radius
        if hasattr(self.config, 'flame_geometry') and self.config.flame_geometry in ['cylindrical', 'spherical', 'disc']:
            r = np.abs(self.x)  # Use absolute distance from centerline as radius
        else:
            r = np.ones_like(self.x)  # Planar case
            
        # Compute r at half points (cell faces)
        r_half = np.zeros(len(r)-1)  # One less point than cell centers
        for j in range(len(r)-1):
            r_half[j] = 0.5 * (r[j] + r[j+1])  # Linear interpolation
        
        # Interior points
        for i in range(1, self.n_points-1):
            dl = dx_left[i]
            dr = dx_right[i]
            alpha[i] = -dr / (dl * (dl + dr))
            beta[i] = (dr - dl) / (dl * dr)
            gamma[i] = dl / (dr * (dl + dr))
        
        return GridMetrics(dx, dx_left, dx_right, alpha, beta, gamma, r, r_half)
            
    def refine_flame_zone(self, T: np.ndarray, threshold: float = 0.1):
        """
        Add points in the flame zone based on temperature gradient
        threshold: fraction of maximum temperature gradient
        """
        # Compute temperature gradient
        dTdx = np.gradient(T, self.x)
        
        # Find flame zone
        grad_max = np.max(np.abs(dTdx))
        flame_zone = np.abs(dTdx) > threshold * grad_max
        
        # Add points in flame zone
        points_to_add = []
        for i in range(self.n_points - 1):
            if flame_zone[i] or flame_zone[i+1]:
                dx = self.x[i+1] - self.x[i]
                if dx > self.criteria.flame_res:
                    points_to_add.append(i)
                    
        self._add_points(points_to_add)
        
    def plot_adaptation_history(self):
        """Plot grid adaptation history"""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        
        # Plot grid point locations at each adaptation step
        for i, grid in enumerate(self.adaptation_history):
            plt.plot(grid, np.ones_like(grid) * i, 'k.', alpha=0.5)
            
        plt.xlabel('Position [m]')
        plt.ylabel('Adaptation step')
        plt.title('Grid Adaptation History')
        plt.grid(True)
        plt.show()