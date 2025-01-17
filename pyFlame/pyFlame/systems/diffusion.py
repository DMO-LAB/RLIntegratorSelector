# from typing import Tuple, Optional
# import numpy as np
# from scipy import sparse
# from scipy.sparse.linalg import spsolve
# from .base import BaseSystem

# class DiffusionSystem(BaseSystem):
#     """
#     System representing species and energy diffusion
    
#     For each component, solves equation of form:
#     dy/dt = (1/r^α) * d/dx(r^α * D * dy/dx) + split_const
#     """
#     def __init__(self, config, grid):
#         super().__init__(config, grid)
        
#         # Diffusion coefficients
#         self.D: Optional[np.ndarray] = None  # [n_vars, n_points]
        
#         # Equation scaling factors
#         self.B: Optional[np.ndarray] = None  # [n_vars, n_points]
        
#         # System matrices
#         self.A: Optional[sparse.csr_matrix] = None
        
#     def resize(self, n_points: int):
#         """Resize system arrays for new grid size"""
#         super().resize(n_points)
        
#         # Resize coefficient arrays
#         self.D = np.zeros((self.n_vars, n_points))
#         self.B = np.zeros((self.n_vars, n_points))
        
#         # Rebuild matrices for new size
#         self._build_matrices()
        
#     def initialize(self, n_vars: int):
#         """Initialize diffusion system"""
#         super().initialize(n_vars)
        
#         # Initialize coefficients
#         self.D = np.zeros((n_vars, self.n_points))
#         self.B = np.zeros((n_vars, self.n_points))
        
#         # Build system matrices
#         self._build_matrices()
        
#     def _build_matrices(self):
#         """Build sparse matrices for implicit integration"""
#         n = self.n_points
        
#         # Allocate matrix diagonals
#         diag = np.zeros(n)
#         lower = np.zeros(n-1)
#         upper = np.zeros(n-1)
        
#         # Fill matrix entries based on finite difference scheme
#         dx = self.grid.metrics.dx
#         alpha = self.grid.metrics.alpha
#         beta = self.grid.metrics.beta
#         gamma = self.grid.metrics.gamma
        
#         # Interior points
#         for i in range(1, n-1):
#             # Second order central difference approximation
#             diag[i] = beta[i]
#             lower[i-1] = alpha[i]
#             upper[i] = gamma[i]
            
#         # Boundary conditions
#         if self.grid.left_bc.FIXED_VALUE:
#             diag[0] = 1.0
#         else:  # Zero gradient
#             diag[0] = -1.0
#             upper[0] = 1.0
            
#         if self.grid.right_bc.FIXED_VALUE:
#             diag[-1] = 1.0
#         else:  # Zero gradient
#             diag[-1] = -1.0
#             lower[-1] = 1.0
            
#         # Create sparse matrix
#         self.A = sparse.diags(
#             [lower, diag, upper],
#             [-1, 0, 1],
#             format='csr'
#         )
        
#     def evaluate(self, t: float) -> np.ndarray:
#         """Evaluate diffusion terms"""
#         ddt = np.zeros_like(self.state)
        
#         # Loop over variables
#         for k in range(self.n_vars):
#             # Get coefficients for this variable
#             D = self.D[k]
#             B = self.B[k]
            
#             # Compute fluxes
#             flux = -D * self._gradient(self.state[k])
            
#             # Compute divergence of fluxes
#             ddt[k] = B * self._divergence(flux) + self.split_const[k]
            
#         return ddt
        
#     def _gradient(self, y: np.ndarray) -> np.ndarray:
#         """Compute gradient of y using finite differences"""
#         dx = self.grid.metrics.dx
#         alpha = self.grid.metrics.alpha
#         beta = self.grid.metrics.beta
#         gamma = self.grid.metrics.gamma
        
#         grad = np.zeros_like(y)
        
#         # Interior points
#         for i in range(1, self.n_points-1):
#             grad[i] = (alpha[i]*y[i-1] + beta[i]*y[i] + gamma[i]*y[i+1])
            
#         # Boundary conditions
#         if self.grid.left_bc.FIXED_VALUE:
#             grad[0] = (y[1] - y[0])/dx[0]
#         else:
#             grad[0] = 0
            
#         if self.grid.right_bc.FIXED_VALUE:
#             grad[-1] = (y[-1] - y[-2])/dx[-1]
#         else:
#             grad[-1] = 0
            
#         return grad
        
#     def _divergence(self, flux: np.ndarray) -> np.ndarray:
#         """Compute divergence of flux"""
#         dx = self.grid.metrics.dx
#         div = np.zeros_like(flux)
        
#         # Interior points using central differences
#         for i in range(1, self.n_points-1):
#             div[i] = (flux[i+1] - flux[i-1])/(dx[i-1] + dx[i])
            
#         # Boundary points using one-sided differences
#         div[0] = (flux[1] - flux[0])/dx[0]
#         div[-1] = (flux[-1] - flux[-2])/dx[-1]
        
#         return div

#     def integrate(self, t_start: float, t_end: float) -> Tuple[np.ndarray, bool]:
#         """
#         Integrate the diffusion system implicitly
#         Returns: (final_state, success)
#         """
#         dt = t_end - t_start
#         success = True
        
#         # Create RHS for each variable
#         result = np.zeros_like(self.state)
        
#         for k in range(self.n_vars):
#             # Get coefficients
#             D = self.D[k]
#             B = self.B[k]
            
#             # Build system matrix with coefficients
#             A = sparse.diags(B * D) @ self.A
            
#             # RHS includes current state and split constant
#             rhs = self.state[k] + dt * self.split_const[k]
            
#             try:
#                 # Solve system
#                 result[k] = spsolve(
#                     sparse.eye(self.n_points) - dt*A,
#                     rhs
#                 )
#             except:
#                 success = False
#                 break
                
#         return result, success


from typing import Tuple, Optional
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from .base import BaseSystem
from pyFlame.grid import BoundaryCondition

class DiffusionSystem(BaseSystem):
    def __init__(self, config, grid, nvars):
        super().__init__(config, grid)
        
        # Initialize coefficient arrays with correct shapes
        self.D = np.zeros((nvars, self.n_points))  # [n_vars x n_points]
        self.B = np.zeros((nvars, self.n_points))  # [n_vars x n_points]
        
        # Working arrays for each variable
        self.c1 = np.zeros(self.n_points)
        self.c2 = np.zeros(self.n_points)
        # Wall conditions (for heat loss)
        self.y_inf = 0.0
        self.wall_const = 0.0
        
        
        # System matrix
        self.A = None

    def _build_matrices(self):
        """Build diffusion matrices following Ember's implementation"""
        n = self.n_points
        dx = self.grid.metrics.dx
        r = self.grid.metrics.r
        
        # For each variable
        for k in range(self.n_vars):
            # Initialize coefficient arrays
            self.c1 = np.zeros(n)
            self.c2 = np.zeros(n)
            
            # Compute coefficients for interior points
            for j in range(1, n-1):
                self.c1[j] = 0.5 * self.B[k,j] / (dx[j] * r[j])
                self.c2[j] = self.grid.metrics.r_half[j] * (self.D[k,j] + self.D[k,j+1]) / dx[j]
                
            # Initialize tridiagonal matrix elements
            diag = np.zeros(n)  # Main diagonal
            lower = np.zeros(n-1)  # Lower diagonal 
            upper = np.zeros(n-1)  # Upper diagonal

            # Left boundary
            if self.grid.left_bc == BoundaryCondition.FIXED_VALUE:
                j_start = 1
                diag[0] = 1.0
            elif self.grid.left_bc == BoundaryCondition.CONTROL_VOLUME:
                j_start = 1
                c0 = self.B[k,0] * (self.grid.alpha + 1) * (self.D[k,0] + self.D[k,1]) / (2 * dx[0] * dx[0])
                diag[0] = -c0
                upper[0] = c0
            elif self.grid.left_bc == BoundaryCondition.WALL_FLUX:
                j_start = 1
                c0 = self.B[k,0] * (self.grid.alpha + 1) / dx[0]
                d = 0.5 * (self.D[k,0] + self.D[k,1])
                diag[0] = -c0 * (d/dx[0] + self.wall_const)
                upper[0] = d * c0 / dx[0]
            else:  # ZERO_GRADIENT
                j_start = 2
                diag[1] = -self.c1[1] * self.c2[1]
                upper[1] = self.c1[1] * self.c2[1]

            # Right boundary
            if self.grid.right_bc == BoundaryCondition.FIXED_VALUE:
                j_stop = n-1
                diag[-1] = 1.0
            else:  # ZERO_GRADIENT
                j_stop = n-2
                lower[n-2] = self.c1[n-2] * self.c2[n-3]
                diag[n-2] = -self.c1[n-2] * self.c2[n-3]

            # Interior points
            for j in range(j_start, j_stop):
                lower[j-1] = self.c1[j] * self.c2[j-1]
                diag[j] = -self.c1[j] * (self.c2[j-1] + self.c2[j])
                upper[j] = self.c1[j] * self.c2[j]

            # Create sparse matrix for this variable
            self.A = sparse.diags(
                [lower, diag, upper],
                [-1, 0, 1],
                format='csr'
            )

    def get_k(self) -> np.ndarray:
        """Get RHS vector"""
        k = self.split_const.copy()
        
        if self.grid.left_bc == BoundaryCondition.WALL_FLUX:
            k[0] += (self.B[0] * (self.grid.alpha + 1) / 
                    self.grid.metrics.dx[0] * self.wall_const * self.y_inf)
            
        return k

    def integrate(self, t_start: float, t_end: float) -> Tuple[np.ndarray, bool]:
        """Integrate the diffusion system implicitly"""
        dt = t_end - t_start
        success = True
        
        # Build matrices if needed
        self._build_matrices()
        
        # Use Crank-Nicolson method
        theta = 0.5
        result = np.zeros_like(self.state)
        
        try:
            for k in range(self.n_vars):
                # Get coefficients for this variable
                D_k = np.diag(self.B[k] * self.D[k])
                A_k = D_k @ self.A
                
                # RHS vector
                k_vec = self.get_k()[k]
                
                # Solve system
                M = sparse.eye(self.n_points) - theta * dt * A_k
                rhs = (self.state[k] + 
                      dt * ((1-theta) * A_k @ self.state[k] + k_vec))
                
                result[k] = spsolve(M, rhs)
                
        except:
            success = False
            result = self.state
            
        return result, success

    def resize(self, n_points: int):
        """Resize system arrays"""
        super().resize(n_points)
        
        if self.n_vars > 0:
            # Resize coefficient arrays
            self.D = np.zeros((self.n_vars, n_points))
            self.B = np.zeros((self.n_vars, n_points))
            
            # Reset working arrays
            self.c1 = None
            self.c2 = None
            
            # Force matrix rebuild
            self.A = None