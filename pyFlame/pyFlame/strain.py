from abc import ABC, abstractmethod

class StrainFunction(ABC):
    """Base class for strain rate functions"""
    @abstractmethod
    def a(self, t: float) -> float:
        """Get strain rate at time t"""
        pass
    
    @abstractmethod
    def dadt(self, t: float) -> float:
        """Get strain rate time derivative at time t"""
        pass

class ConstantStrain(StrainFunction):
    """Constant strain rate"""
    def __init__(self, strain_rate: float):
        self.a0 = strain_rate
        
    def a(self, t: float) -> float:
        return self.a0
        
    def dadt(self, t: float) -> float:
        return 0.0

class RampedStrain(StrainFunction):
    """Linearly ramped strain rate"""
    def __init__(self, initial_strain: float, final_strain: float, 
                 start_time: float, ramp_duration: float):
        self.a0 = initial_strain
        self.a1 = final_strain
        self.t0 = start_time
        self.dt = ramp_duration
        self.rate = (final_strain - initial_strain) / ramp_duration
        
    def a(self, t: float) -> float:
        if t < self.t0:
            return self.a0
        elif t > self.t0 + self.dt:
            return self.a1
        else:
            return self.a0 + self.rate * (t - self.t0)
            
    def dadt(self, t: float) -> float:
        if self.t0 <= t <= self.t0 + self.dt:
            return self.rate
        return 0.0

class OscillatingStrain(StrainFunction):
    """Oscillating strain rate"""
    def __init__(self, mean_strain: float, amplitude: float, frequency: float):
        self.a0 = mean_strain
        self.A = amplitude
        self.f = frequency
        self.omega = 2 * np.pi * frequency
        
    def a(self, t: float) -> float:
        return self.a0 + self.A * np.sin(self.omega * t)
        
    def dadt(self, t: float) -> float:
        return self.A * self.omega * np.cos(self.omega * t)