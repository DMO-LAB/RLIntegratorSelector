import numpy as np
import cantera as ct
import time
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Tuple
from ember import ZeroDCombustion, Config, Paths, General, ConcreteConfig, CVODETolerancesZeroD, RK23Tolerances, QssTolerances

@dataclass
class IntegratorConfig:
    name: str
    config: ConcreteConfig
    dt: float
    color: str  # For plotting

@dataclass
class SimulationResult:
    times: np.ndarray
    temperatures: np.ndarray
    species: Dict[int, np.ndarray]
    total_time: float
    steps: int
    time_per_step: np.ndarray

def create_configs() -> Tuple[IntegratorConfig, List[IntegratorConfig]]:
    """Create configurations for baseline and test integrators"""
    # Baseline configuration (CVODE with tight tolerances)
    baseline_config = IntegratorConfig(
        name="CVODE-baseline",
        config=ConcreteConfig(Config(
            Paths(outputDir='run/zerod_test'),
            General(nThreads=1, chemistryIntegrator='cvode'),
            CVODETolerancesZeroD(
                relativeTolerance=1e-8,
                abstol_temperature=1e-20,
                abstol_species=1e-20,
                minStep=1e-20,
                maxStep=1e-6
            )
        )),
        dt=1e-6,
        color='k'
    )
    
    # Test configurations
    test_configs = [
        IntegratorConfig(
            name="CVODE",
            config=ConcreteConfig(Config(
                Paths(outputDir='run/zerod_test'),
                General(nThreads=1, chemistryIntegrator='cvode'),
                CVODETolerancesZeroD(
                    relativeTolerance=1e-6,
                    abstol_temperature=1e-12,
                    abstol_species=1e-12,
                    minStep=1e-18,
                    maxStep=1e-5
                )
            )),
            dt=1e-6,
            color='b'
        ),
        IntegratorConfig(
            name="BoostRK-tight",
            config=ConcreteConfig(Config(
                Paths(outputDir='run/zerod_test'),
                General(nThreads=1, chemistryIntegrator='boostRK'),
                RK23Tolerances(
                    relativeTolerance=1e-6,
                    absoluteTolerance=1e-12,
                    maxStepsNumber=10000,
                    minval=1e-18
                )
            )),
            dt=1e-6,
            color='g'
        ),
        IntegratorConfig(
            name="BoostRK-loose",
            config=ConcreteConfig(Config(
                Paths(outputDir='run/zerod_test'),
                General(nThreads=1, chemistryIntegrator='boostRK'),
                RK23Tolerances(
                    relativeTolerance=1e-6,
                    absoluteTolerance=1e-8,
                    maxStepsNumber=10000,
                    minval=1e-18
                )
            )),
            dt=1e-6,
            color='y'
        ),
        IntegratorConfig(
            name="QSS",
            config=ConcreteConfig(Config(
                Paths(outputDir='run/zerod_test'),
                General(nThreads=1, chemistryIntegrator='qss'),
                QssTolerances(
                    abstol=1e-12,
                )
            )),
            dt=1e-6,
            color='r'
        )
    ]
    
    return baseline_config, test_configs

def setup_initial_state() -> Tuple[float, np.ndarray, Dict[int, str]]:
    """Setup the initial state and return important parameters"""
    gas = ct.Solution('gri30.yaml')
    T0 = 1200.0
    gas.TPY = T0, ct.one_atm, 'CH4:1, O2:1, N2:3.76'
    Y0 = np.array(gas.Y)
    
    # Track important species
    species_dict = {
        gas.species_index('CH4'): 'CH4',
        gas.species_index('O2'): 'O2',
        gas.species_index('CO2'): 'CO2',
        gas.species_index('H2O'): 'H2O'
    }
    
    return T0, Y0, species_dict

def run_simulation(config: IntegratorConfig, T0: float, Y0: np.ndarray, 
                  species_indices: Dict[int, str], t_end: float) -> SimulationResult:
    """Run simulation with given configuration and return results"""
    solver = ZeroDCombustion(config.config)
    solver.initialize()
    solver.set_state(T0, Y0)
    
    times = []
    temperatures = []
    species_data = {idx: [] for idx in species_indices}
    time_per_step = []
    
    t_now = 0.0
    step_count = 0
    start_time = time.time()
    
    while t_now < t_end:
        try:
            solver.integrate_to_time(t_now + config.dt)
            time_per_step.append(time.time() - start_time)
            start_time = time.time()
            times.append(t_now)
            temperatures.append(solver.T)
            for idx in species_indices:
                species_data[idx].append(solver.Y[idx])
                
            t_now += config.dt
            step_count += 1
            
        except Exception as e:
            print(f"Integration failed for {config.name} at t = {t_now}: {e}")
            break
    
    total_time = np.sum(time_per_step)
    
    return SimulationResult(
        times=np.array(times),
        temperatures=np.array(temperatures),
        species=species_data,
        total_time=total_time,
        steps=step_count,
        time_per_step=np.array(time_per_step)
    )

def calculate_errors(baseline: SimulationResult, result: SimulationResult) -> Dict[str, float]:
    """Calculate various error metrics compared to baseline"""
    # Interpolate result to baseline time points for comparison
    temp_interp = np.interp(baseline.times, result.times, result.temperatures)
    
    errors = {
        'temp_max_error': np.max(np.abs((temp_interp - baseline.temperatures) / baseline.temperatures)),
        'temp_mean_error': np.mean(np.abs((temp_interp - baseline.temperatures) / baseline.temperatures)),
        'error_per_step': np.mean(np.abs((temp_interp - baseline.temperatures) / baseline.temperatures)) / result.steps
    }
    
    return errors

def plot_results(baseline: SimulationResult, results: Dict[str, SimulationResult], 
                species_names: Dict[int, str], configs: List[IntegratorConfig]):
    """Create comparison plots"""
    # Temperature plot
    plt.figure(figsize=(10, 6))
    plt.plot(baseline.times, baseline.temperatures, 'k--', label='Baseline')
    
    for name, result in results.items():
        config = next(c for c in configs if c.name == name)
        plt.plot(result.times, result.temperatures, color=config.color, label=name)
    
    plt.xlabel('Time (s)')
    plt.ylabel('Temperature (K)')
    plt.legend()
    plt.savefig('temperature_comparison.png')
    plt.close()
    
    # Species plots
    plt.figure(figsize=(12, 8))
    for idx, species_name in species_names.items():
        plt.subplot(2, 2, list(species_names.keys()).index(idx) + 1)
        plt.plot(baseline.times, baseline.species[idx], 'k--', label='Baseline')
        
        for name, result in results.items():
            config = next(c for c in configs if c.name == name)
            plt.plot(result.times, result.species[idx], color=config.color, label=name)
        
        plt.xlabel('Time (s)')
        plt.ylabel(f'{species_name} Mass Fraction')
        if idx == list(species_names.keys())[0]:  # Only show legend once
            plt.legend()
    
    plt.tight_layout()
    plt.savefig('species_comparison.png')
    plt.close()
    
    # Time per step plot
    plt.figure(figsize=(10, 6))
    plt.plot(baseline.times, baseline.time_per_step, 'k--', label='Baseline')
    for name, result in results.items():
        config = next(c for c in configs if c.name == name)
        plt.plot(result.times, result.time_per_step, color=config.color, label=name)
    plt.xlabel('Time (s)')
    plt.ylabel('Time per Step (s)')
    plt.legend()
    plt.savefig('time_per_step.png')
    plt.close()

def main():
    # Setup
    t_end = 1e-1
    baseline_config, test_configs = create_configs()
    T0, Y0, species_dict = setup_initial_state()
    
    # Run baseline
    print("Running baseline simulation...")
    baseline_result = run_simulation(baseline_config, T0, Y0, species_dict, t_end)
    
    # Run test cases
    results = {}
    errors = {}
    print("\nRunning test simulations...")
    for config in test_configs:
        print(f"\nRunning {config.name}...")
        result = run_simulation(config, T0, Y0, species_dict, t_end)
        results[config.name] = result
        errors[config.name] = calculate_errors(baseline_result, result)
    
    # Print performance comparison
    print("\nPerformance Comparison:")
    print(f"{'Integrator':<15} {'Total Time (s)':<15} {'Time/Step (ms)':<15} {'Max Error (%)':<15} {'Mean Error (%)':<15}")
    print("-" * 75)
    
    for name, result in results.items():
        error = errors[name]
        print(f"{name:<15} {result.total_time:<15.3f} {np.mean(result.time_per_step)*1000:<15.3f} "
              f"{error['temp_max_error']*100:<15.3f} {error['temp_mean_error']*100:<15.3f}")
    
    # print the baseline result
    print(f"Baseline: {baseline_result.total_time:.3f} {np.mean(baseline_result.time_per_step)*1000:.3f}")
    # Plot results
    plot_results(baseline_result, results, species_dict, test_configs)

if __name__ == "__main__":
    main()