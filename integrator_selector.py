import numpy as np
import cantera as ct
import time
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from ember import Config, Paths, InitialCondition, StrainParameters, General, \
    Times, TerminationCondition, ConcreteConfig, Debug, RK23Tolerances, QssTolerances, CvodeTolerances, _ember
import os

@dataclass
class ToleranceSettings:
    """Settings for different integrator tolerances"""
    # CVODE settings
    cvode_rtol: float = 1e-6
    cvode_momentum_atol: float = 1e-7
    cvode_energy_atol: float = 1e-8
    cvode_species_atol: float = 1e-13
    cvode_min_timestep: float = 1e-18
    cvode_max_timestep: float = 1e-5
    
    # RK23 settings
    rk_rtol: float = 1e-6
    rk_atol: float = 1e-8
    rk_min_timestep: float = 1e-10
    rk_max_timestep: float = 1e-4
    rk_max_steps: int = 100000
    
    # QSS settings
    qss_eps_min: float = 1e-2
    qss_eps_max: float = 1e1
    qss_min_timestep: float = 1e-16
    qss_max_timestep: float = 1e-6
    qss_atol: float = 1e-11
    qss_min_val: float = 1e-60

@dataclass
class SimulationSettings:
    """General simulation settings"""
    output_dir: str = 'run/diffusion_benchmark'
    n_threads: int = 2
    global_timestep: float = 1e-6
    profile_interval: int = 20
    t_end: float = 0.08
    n_points: int = 100
    x_left: float = -0.02
    x_right: float = 0.02
    T_fuel: float = 600
    T_oxidizer: float = 1200
    pressure: float = 101325
    strain_rate: float = 100

@dataclass
class IntegratorConfig:
    name: str
    config: ConcreteConfig
    constant_integrator: Optional[str]  # For constant integration mode
    color: str  # For plotting

@dataclass
class SimulationResult:
    x: np.ndarray  # Spatial coordinates
    times: np.ndarray
    temperatures: np.ndarray
    species: Dict[int, np.ndarray]
    total_time: float
    integrator_distribution: Dict[str, int]  # Count of each integrator type used

def create_base_config(sim_settings: SimulationSettings, tol_settings: ToleranceSettings) -> Config:
    """Create base configuration with given settings"""
    return Config(
        Paths(outputDir=sim_settings.output_dir),
        General(nThreads=sim_settings.n_threads, chemistryIntegrator='cvode'),
        InitialCondition(
            Tfuel=sim_settings.T_fuel,
            Toxidizer=sim_settings.T_oxidizer,
            centerWidth=0.0,
            equilibrateCounterflow=False,
            flameType='diffusion',
            slopeWidth=0.0,
            xLeft=sim_settings.x_left,
            xRight=sim_settings.x_right,
            pressure=sim_settings.pressure,
            nPoints=sim_settings.n_points
        ),
        StrainParameters(final=sim_settings.strain_rate, initial=sim_settings.strain_rate),
        Times(globalTimestep=sim_settings.global_timestep, 
              profileStepInterval=sim_settings.profile_interval),
        TerminationCondition(
            abstol=0.0,
            dTdtTol=0,
            steadyPeriod=1.0,
            tEnd=sim_settings.t_end,
            tolerance=0.0
        ),
        RK23Tolerances(
            absoluteTolerance=tol_settings.rk_atol,
            relativeTolerance=tol_settings.rk_rtol,
            minimumTimestep=tol_settings.rk_min_timestep,
            maximumTimestep=tol_settings.rk_max_timestep,
            maxStepsNumber=tol_settings.rk_max_steps
        ),
        QssTolerances(
            epsmin=tol_settings.qss_eps_min,
            epsmax=tol_settings.qss_eps_max,
            dtmin=tol_settings.qss_min_timestep,
            dtmax=tol_settings.qss_max_timestep,
            iterationCount=1,
            abstol=tol_settings.qss_atol,
            minval=tol_settings.qss_min_val,
            stabilityCheck=False
        ),
        CvodeTolerances(
            relativeTolerance=tol_settings.cvode_rtol,
            momentumAbsTol=tol_settings.cvode_momentum_atol,
            energyAbsTol=tol_settings.cvode_energy_atol,
            speciesAbsTol=tol_settings.cvode_species_atol,
            minimumTimestep=tol_settings.cvode_min_timestep,
            maximumTimestep=tol_settings.cvode_max_timestep
        ),
        Debug(veryVerbose=False)
    )

def create_configs(sim_settings: SimulationSettings) -> Tuple[IntegratorConfig, List[IntegratorConfig]]:
    """Create configurations for baseline and test integrators"""
    # Create stricter tolerance settings for baseline
    baseline_tol = ToleranceSettings(
        cvode_rtol=1e-8,
        cvode_momentum_atol=1e-8,
        cvode_energy_atol=1e-8,
        cvode_species_atol=1e-13
    )
    
    # Regular tolerance settings for test cases
    test_tol = ToleranceSettings()
    
    # Baseline configuration (CVODE with tight tolerances)
    baseline_config = IntegratorConfig(
        name="CVODE-baseline",
        config=ConcreteConfig(create_base_config(sim_settings, baseline_tol)),
        constant_integrator='cvode',
        color='k'
    )
    
    # Test configurations
    base_config = create_base_config(sim_settings, test_tol)
    test_configs = [
        IntegratorConfig(
            name="CVODE",
            config=ConcreteConfig(base_config),
            constant_integrator='cvode',
            color='b'
        ),
        IntegratorConfig(
            name="BoostRK",
            config=ConcreteConfig(base_config),
            constant_integrator='boostRK',
            color='g'
        ),
        IntegratorConfig(
            name="QSS",
            config=ConcreteConfig(base_config),
            constant_integrator='qss',
            color='r'
        ),
        IntegratorConfig(
            name="Heuristic",
            config=ConcreteConfig(base_config),
            constant_integrator=None,  # None means use heuristic selection
            color='m'
        )
    ]
    
    return baseline_config, test_configs

def setup_species_indices() -> Dict[int, str]:
    """Setup the species indices to track"""
    gas = ct.Solution('gri30.yaml')
    names = gas.species_names
    
    return {
        names.index('CH4'): 'CH4',
        names.index('O2'): 'O2',
        names.index('CO2'): 'CO2',
        names.index('H2O'): 'H2O',
        names.index('CO'): 'CO',
        names.index('OH'): 'OH'
    }

def set_integrator_heuristic(solver) -> List[str]:
    """Set integrator type based on temperature and equivalence ratio"""
    nPoints = len(solver.T)
    
    # Start with temperature-based decision
    integ = np.where(solver.T <= 600.0, 'boostRK', 'cvode')
    
    try:
        # Get equivalence ratio
        phi = solver.phi
        
        # Use boostRK for extreme conditions
        integ = np.where(phi == -1, 'boostRK', integ)  # invalid phi
        integ = np.where(phi <= 1e-8, 'boostRK', integ)  # oxidizer-dominated
        integ = np.where(phi >= 1e4, 'boostRK', integ)   # fuel-dominated
        
        # Create boolean mask for CVODE points
        cvode_mask = (integ == 'cvode')
        
        # Include neighboring points
        cvode_mask_left = np.roll(cvode_mask, 1)
        cvode_mask_right = np.roll(cvode_mask, -1)
        cvode_mask_left[0] = False
        cvode_mask_right[-1] = False
        
        use_cvode = cvode_mask | cvode_mask_left | cvode_mask_right
        integ = np.where(use_cvode, 'cvode', 'boostRK')
        
    except Exception as e:
        print(f"Warning: Could not calculate phi for integrator selection: {e}")
    
    return integ.tolist()

def run_simulation(config: IntegratorConfig) -> SimulationResult:
    """Run simulation with given configuration and return results"""
    solver = _ember.FlameSolver(config.config)
    solver.initialize()
    
    start_time = time.time()
    integrator_counts = {'cvode': 0, 'boostRK': 0, 'qss': 0}
    
    done = False
    times = []
    while not done:
        if config.constant_integrator:
            integrator_types = [config.constant_integrator] * len(solver.T)
        else:
            integrator_types = set_integrator_heuristic(solver)
            
        # Update integrator counts
        for int_type in integrator_types:
            integrator_counts[int_type] += 1
            
        solver.set_integrator_types(integrator_types)
        done = solver.step()
        times.append(solver.t)
    total_time = time.time() - start_time
    
    return SimulationResult(
        x=solver.x,
        times=np.array([solver.t]),  # Single time point for steady-state
        temperatures=solver.T,
        species={idx: solver.Y[idx] for idx in setup_species_indices()},
        total_time=total_time,
        integrator_distribution=integrator_counts
    )

def calculate_errors(baseline: SimulationResult, result: SimulationResult) -> Dict[str, float]:
    """Calculate various error metrics compared to baseline"""
    # Interpolate result to baseline spatial points for comparison
    temp_interp = np.interp(baseline.x, result.x, result.temperatures)
    
    errors = {
        'temp_max_error': np.max(np.abs((temp_interp - baseline.temperatures) / baseline.temperatures)),
        'temp_mean_error': np.mean(np.abs((temp_interp - baseline.temperatures) / baseline.temperatures)),
        'temp_rms_error': np.sqrt(np.mean(((temp_interp - baseline.temperatures) / baseline.temperatures) ** 2))
    }
    
    return errors

def plot_results(baseline: SimulationResult, results: Dict[str, SimulationResult], 
                species_indices: Dict[int, str], configs: List[IntegratorConfig],
                output_dir: str):
    """Create comparison plots"""
    # Temperature plot
    plt.figure(figsize=(10, 6))
    plt.plot(baseline.x, baseline.temperatures, 'k--', label='Baseline')
    
    for name, result in results.items():
        config = next(c for c in configs if c.name == name)
        plt.plot(result.x, result.temperatures, color=config.color, label=name)
    
    plt.xlabel('Position (m)')
    plt.ylabel('Temperature (K)')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'temperature_comparison_1d.png'))
    plt.close()
    
    # Species plots
    plt.figure(figsize=(12, 8))
    for idx, species_name in species_indices.items():
        plt.subplot(2, 3, list(species_indices.keys()).index(idx) + 1)
        plt.plot(baseline.x, baseline.species[idx], 'k--', label='Baseline')
        
        for name, result in results.items():
            config = next(c for c in configs if c.name == name)
            plt.plot(result.x, result.species[idx], color=config.color, label=name)
        
        plt.xlabel('Position (m)')
        plt.ylabel(f'{species_name} Mass Fraction')
        if idx == list(species_indices.keys())[0]:  # Only show legend once
            plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'species_comparison_1d.png'))
    plt.close()

def main():
    # Create simulation settings
    sim_settings = SimulationSettings(
        output_dir='run/diffusion_benchmark_test',
        n_threads=2,
        global_timestep=1e-6,
        profile_interval=20,
        t_end=0.08,
        n_points=100
    )
    
    # Setup
    baseline_config, test_configs = create_configs(sim_settings)
    species_indices = setup_species_indices()
    
    # Create output directory if it doesn't exist
    os.makedirs(sim_settings.output_dir, exist_ok=True)
    
    # Run baseline
    print("Running baseline simulation...")
    baseline_result = run_simulation(baseline_config)
    
    # Run test cases
    results = {}
    errors = {}
    print("\nRunning test simulations...")
    for config in test_configs:
        print(f"\nRunning {config.name}...")
        result = run_simulation(config)
        results[config.name] = result
        errors[config.name] = calculate_errors(baseline_result, result)
    
    # Print performance comparison
    print("\nPerformance Comparison:")
    print(f"{'Integrator':<15} {'Total Time (s)':<15} {'Max Error (%)':<15} {'Mean Error (%)':<15} {'RMS Error (%)':<15}")
    print("-" * 75)
    
    for name, result in results.items():
        error = errors[name]
        print(f"{name:<15} {result.total_time:<15.3f} "
              f"{error['temp_max_error']*100:<15.3f} "
              f"{error['temp_mean_error']*100:<15.3f} "
              f"{error['temp_rms_error']*100:<15.3f}")
    
    # Print integrator distribution for heuristic case
    if 'Heuristic' in results:
        print("\nHeuristic Integrator Distribution:")
        dist = results
        print(dist) 

    plot_results(baseline_result, results, species_indices, test_configs, sim_settings.output_dir)

if __name__ == "__main__":
    main()