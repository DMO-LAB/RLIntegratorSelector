import numpy as np



class AdaptiveTimeStepper:
    """
    Manages adaptive time stepping based on solution features
    """
    def __init__(self, config):
        # Time step bounds
        self.dt_min = getattr(config, 'dt_min', 1e-8)
        self.dt_max = getattr(config, 'dt_max', 1e-4)
        self.dt = getattr(config, 'dt', 1e-6)  # Initial time step
        self.old_dt = self.dt
        
        # Control parameters
        self.max_T_change = getattr(config, 'max_T_change', 50.0)  # Maximum temperature change per step [K]
        self.max_Y_change = getattr(config, 'max_Y_change', 0.1)   # Maximum species mass fraction change
        self.safety_factor = getattr(config, 'timestep_safety', 0.8)  # Safety factor for time step
        
        # Growth limits
        self.max_growth = getattr(config, 'max_timestep_growth', 2.0)  # Maximum time step increase
        self.min_reduction = getattr(config, 'min_timestep_reduction', 0.1)  # Minimum time step reduction
        
        # History tracking
        self.rejected_steps = 0
        self.accepted_steps = 0
        
    def compute_timestep(self, solver) -> float:
        """
        Compute next time step based on solution features
        """
        # Current solution state
        T = solver.state[solver.i_T]
        Y = solver.state[solver.i_Y]
        
        # Get rates of change from chemistry system
        dT_dt = solver.ddt_prod[solver.i_T]
        dY_dt = solver.ddt_prod[solver.i_Y]
        
        # Compute time step constraints
        dt_T = self.max_T_change / (np.abs(dT_dt) + 1e-10)
        dt_Y = self.max_Y_change / (np.abs(dY_dt) + 1e-10)
        
        # Get minimum across all points and species
        dt_T_min = np.min(dt_T)
        dt_Y_min = np.min(dt_Y)
        
        # Additional constraint from flame propagation
        if hasattr(solver, 'flame_state'):
            # Estimate CFL condition for flame propagation
            dx_min = np.min(np.diff(solver.grid.x))
            u_max = np.max(np.abs(solver.state[solver.i_U]))
            if u_max > 0:
                dt_cfl = 0.5 * dx_min / u_max  # CFL = 0.5
            else:
                dt_cfl = self.dt_max
        else:
            dt_cfl = self.dt_max
        
        # Choose minimum of all constraints
        dt_new = min(dt_T_min, dt_Y_min, dt_cfl)
        
        # Apply safety factor and limits
        dt_new *= self.safety_factor
        dt_new = min(max(dt_new, self.dt_min), self.dt_max)
        
        # Limit growth rate
        dt_new = min(dt_new, self.dt * self.max_growth)
        
        # print infomation if dt_new is less than dt
        if dt_new < self.old_dt:
            print(f"[INFO] - Time step reduced to {dt_new:.2e} s from {self.old_dt:.2e} s")

        # Update time step
        self.old_dt = dt_new
        return dt_new
    
    def check_step(self, solver, state_new) -> bool:
        """
        Check if the step should be accepted
        Returns: True if step is acceptable
        """
        # Get changes in solution
        dT = np.abs(state_new[solver.i_T] - solver.state[solver.i_T])
        dY = np.abs(state_new[solver.i_Y] - solver.state[solver.i_Y])
        
        # Check against maximum allowed changes
        T_ok = np.all(dT <= self.max_T_change)
        Y_ok = np.all(dY <= self.max_Y_change)
        
        # Additional checks for physical constraints
        physical = (
            np.all(state_new[solver.i_T] > 0) and  # Positive temperature
            np.all(state_new[solver.i_Y] >= -1e-7) and  # Non-negative mass fractions
            np.all(np.abs(np.sum(state_new[solver.i_Y], axis=0) - 1.0) < 1e-6)  # Mass conservation
        )
        
        step_ok = T_ok and Y_ok and physical
        #print(f"[INFO] - Step accepted: {step_ok} - T_ok: {T_ok} - Y_ok: {Y_ok} - Physical: {physical}")
        
        
        # Update statistics
        if step_ok:
            self.accepted_steps += 1
        else:
            self.rejected_steps += 1
            # Reduce time step for next attempt
            self.dt *= max(self.min_reduction, 
                          min(1.0, self.max_T_change/np.max(dT),
                              self.max_Y_change/np.max(dY)))
            
        return step_ok