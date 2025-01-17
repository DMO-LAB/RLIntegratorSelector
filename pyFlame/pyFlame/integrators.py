# integrators.py
from enum import Enum
from dataclasses import dataclass
from typing import Tuple, Callable, Dict, Any, Optional
import numpy as np
from scipy.integrate import solve_ivp,  odeint, ode


class IntegratorType(Enum):
    """Available integrator types"""
    # solve_ivp methods
    BDF = "BDF"
    LSODA = "LSODA" 
    RK45 = "RK45"
    RK23 = "RK23"
    Radau = "Radau"
    EULER = "EULER"
    EQUILIBRIUM = "EQUILIBRIUM"
    # odeint methods
    VODE = "vode"
    ZVODE = "zvode"
    LSODA_ODE = "lsoda"
    DOPRI5 = "dopri5"
    DOP853 = "dop853"
    
class IntegratorAPI(Enum):
    """Available integrator APIs"""
    SOLVE_IVP = "solve_ivp"
    ODE = "ode"


@dataclass
class IntegratorConfig:
    """Integrator configuration parameters"""
    method: IntegratorType
    api: IntegratorAPI
    rtol: float
    atol: float
    max_step: Optional[float] = None
    min_step: Optional[float] = None
    extra_params: Optional[Dict[str, Any]] = None

class BaseIntegrator:
    """Base integrator interface"""
    def __init__(self, config: IntegratorConfig):
        self.config = config
    
    def integrate(self, rhs: Callable, y0: np.ndarray, t_span: Tuple[float, float]) -> Tuple[np.ndarray, bool]:
        raise NotImplementedError

class ScipyIVPIntegrator(BaseIntegrator):
    """Efficient scipy solve_ivp-based integrator"""
    def integrate(self, rhs: Callable, y0: np.ndarray, t_span: Tuple[float, float]) -> Tuple[np.ndarray, bool]:
        solution = solve_ivp(
            rhs,
            t_span,
            y0,
            method=self.config.method.value,
            rtol=self.config.rtol,
            atol=self.config.atol,
            # max_step=self.config.max_step,
            # min_step=self.config.min_step,
            # jac_sparsity=self.config.extra_params.get('jac_sparsity') if self.config.extra_params else None
        )
        #print(f"Solution shape: {solution.y[:, -1].shape}")
        return solution.y[:,-1], solution.success


class ScipyODEIntegrator(BaseIntegrator):
    """Efficient scipy ode-based integrator"""
    def integrate(self, rhs: Callable, y0: np.ndarray, t_span: Tuple[float, float]) -> Tuple[np.ndarray, bool]:
        t_start, t_end = t_span
        
        # Create ODE instance
        r = ode(rhs)
        
        # Configure the integrator
        integrator_params = {
            'rtol': self.config.rtol,
            'atol': self.config.atol,
        }
        
        # Add method-specific parameters
        if self.config.method in [IntegratorType.VODE, IntegratorType.ZVODE]:
            # For VODE/ZVODE, specify stiff/non-stiff method
            integrator_params['method'] = 'bdf'# if self.config.extra_params.get('stiff', True) else 'adams'
            
            # # Add band parameters if provided
            # if 'lband' in self.config.extra_params:
            #     integrator_params['lband'] = self.config.extra_params['lband']
            # if 'uband' in self.config.extra_params:
            #     integrator_params['uband'] = self.config.extra_params['uband']
                
            # # Add order if specified
            # if 'order' in self.config.extra_params:
            #     integrator_params['order'] = self.config.extra_params['order']
        
        # Add step size limits
        # if self.config.max_step:
        #     integrator_params['max_step'] = self.config.max_step
        # if self.config.min_step:
        #     integrator_params['min_step'] = self.config.min_step
        
        # #Add any remaining extra parameters
        # if self.config.extra_params:
        #     for key, value in self.config.extra_params.items():
        #         if key not in ['stiff', 'lband', 'uband', 'order']:
        #             integrator_params[key] = value
        
        # Set up the integrator
        r.set_integrator(self.config.method.value, **integrator_params)
        
        # Set initial conditions
        r.set_initial_value(y0, t_start)
        
        # Integrate to end time
        y_final = r.integrate(t_end)
        
        # Check success
        success = r.successful()
        
        if success:
            print(f"Final data shape: {y_final.shape}")
            return y_final, True
        else:
            print(f"ODE integration failed at t={r.t}")
            return y0, False

class ExplicitEuler(BaseIntegrator):
    """Fast explicit Euler for non-stiff regions"""
    def integrate(self, rhs: Callable, y0: np.ndarray, t_span: Tuple[float, float]) -> Tuple[np.ndarray, bool]:
        t_start, t_end = t_span
        dt = self.config.max_step or (t_end - t_start) / 100
        y = y0.copy()
        t = t_start
        
        while t < t_end:
            dt = min(dt, t_end - t)
            y += dt * rhs(t, y)
            t += dt
        
        return y, True

# Factory for creating integrators
SOLVE_IVP_INTEGRATORS = {
    IntegratorType.BDF: ScipyIVPIntegrator,
    IntegratorType.LSODA: ScipyIVPIntegrator,
    IntegratorType.RK45: ScipyIVPIntegrator,
    IntegratorType.RK23: ScipyIVPIntegrator,
    IntegratorType.Radau: ScipyIVPIntegrator,
    IntegratorType.EULER: ExplicitEuler,
    IntegratorType.EQUILIBRIUM: ScipyIVPIntegrator
}

ODE_INTEGRATORS = {
    IntegratorType.VODE: ScipyODEIntegrator,
    IntegratorType.ZVODE: ScipyODEIntegrator,
    IntegratorType.LSODA_ODE: ScipyODEIntegrator,
    IntegratorType.DOPRI5: ScipyODEIntegrator,
    IntegratorType.DOP853: ScipyODEIntegrator
}

def create_integrator(config: IntegratorConfig) -> BaseIntegrator:
    """Factory function to create integrators based on API and method"""
    if config.api == IntegratorAPI.SOLVE_IVP:
        if config.method not in SOLVE_IVP_INTEGRATORS:
            raise ValueError(f"Unsupported method {config.method} for solve_ivp API")
        return SOLVE_IVP_INTEGRATORS[config.method](config)
    
    elif config.api == IntegratorAPI.ODE:
        if config.method not in ODE_INTEGRATORS:
            raise ValueError(f"Unsupported method {config.method} for odeint API")
        return ODE_INTEGRATORS[config.method](config)
    
    else:
        raise ValueError(f"Unsupported integrator API: {config.api}")



class IntegratorSelector:
    """Base class for integrator selection strategies"""
    def select(self, state: Dict[str, Any]) -> Tuple[IntegratorType, IntegratorAPI]:
        raise NotImplementedError

class DefaultSelector(IntegratorSelector):
    """Simple selector that returns the default integrator and API"""
    def __init__(self, default_type: IntegratorType, default_api: IntegratorAPI = IntegratorAPI.ODE):
        self.default_type = default_type
        self.default_api = default_api
    
    def select(self, state: Dict[str, Any]) -> Tuple[IntegratorType, IntegratorAPI]:
        T = state.get("T", 300)
        return self.default_type, self.default_api

class AdaptiveSelector(IntegratorSelector):
    """More sophisticated selector that chooses between APIs and methods based on problem characteristics"""
    def select(self, state: Dict[str, Any]) -> Tuple[IntegratorType, IntegratorAPI]:
        T = state.get("T", 300)
        
        # Estimate problem stiffness (this is a simple heuristic)
        stiff = T > 2500 or state.get("wdot_norm", 0) > 1e3
        
        if stiff:
            # For stiff problems, prefer VODE with odeint
            return IntegratorType.VODE, IntegratorAPI.ODE
        else:
            # For non-stiff problems, use DOP853 with odeint (usually faster)
            return IntegratorType.DOP853, IntegratorAPI.ODE


class RLPolicySelector(IntegratorSelector):
    """Selector that uses RL policy for integrator selection"""
    def __init__(self, policy_path: str):
        self.policy = None  # Load your trained policy
    
    def select(self, state: Dict[str, Any]) -> Tuple[IntegratorType, IntegratorAPI]:
        policy_input = self._prepare_state(state)
        return self._predict(policy_input)
    
    def _prepare_state(self, state: Dict[str, Any]) -> np.ndarray:
        # Implement state preprocessing for your policy
        pass
    
    def _predict(self, policy_input: np.ndarray) -> Tuple[IntegratorType, IntegratorAPI]:
        # Implement policy prediction
        pass