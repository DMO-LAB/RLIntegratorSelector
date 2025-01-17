from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
import numpy as np

def plot_evolution(solver, species_to_plot=None, save_path=None):
    """Plot the evolution of temperature, species, and density with adaptive grid"""
    plt.figure(figsize=(15, 12), dpi=200)
    plt.suptitle('Flame Evolution', fontsize=16)

    history = solver.history
    
    # Create a reference grid for interpolation (use last grid point)
    x_ref = history.x[-1]

    # Add timestamp text
    time_text = plt.figtext(0.5, 0.95, f'Time: {history.times[-1]*1000:.2f} ms', 
                           ha='center', fontsize=12, wrap=True)

    # Plot species evolution
    plt.subplot(3, 1, 1)
    if species_to_plot is None:
        # Default to first two species
        species_to_plot = [0, 1]
    
    specie_names = solver.gas.species_names
    colors = ['b', 'orange', 'g', 'r']
    for k, spec_names in enumerate(species_to_plot):
        spec_idx = solver.gas.species_index(spec_names)
        for i, (Y, x) in enumerate(zip(history.Y, history.x)):
            alpha = (i + 1) / len(history.Y)
            
            # Interpolate to reference grid if grid sizes don't match
            if len(x) != len(x_ref):
                interp = interp1d(x, Y[spec_idx], kind='linear', bounds_error=False, 
                                fill_value='extrapolate')
                y_plot = interp(x_ref)
                x_plot = x_ref
            else:
                y_plot = Y[spec_idx]
                x_plot = x
            
            if i == len(history.Y) - 1:
                label = f"{spec_idx} ({specie_names[spec_idx]})"
            else:
                label = None
            plt.plot(x_plot, y_plot, color=colors[k], alpha=alpha, label=label)

    plt.xlabel('Position (m)')
    plt.ylabel('Mass Fraction')
    plt.legend()
    plt.grid(True)
    plt.title('Species Evolution')

    # Plot temperature evolution
    plt.subplot(3, 1, 2)
    for i, (T, x) in enumerate(zip(history.T, history.x)):
        alpha = (i + 1) / len(history.T)
        
        # Interpolate if needed
        if len(x) != len(x_ref):
            interp = interp1d(x, T, kind='linear', bounds_error=False, 
                            fill_value='extrapolate')
            T_plot = interp(x_ref)
            x_plot = x_ref
        else:
            T_plot = T
            x_plot = x
            
        plt.plot(x_plot, T_plot, 'r-', alpha=alpha, 
                 label=f't={history.times[i]:.2e}s' if i == 0 else None)

    plt.xlabel('Position (m)')
    plt.ylabel('Temperature (K)')
    plt.legend()
    plt.grid(True)
    plt.title('Temperature Evolution')

    # Plot density evolution
    plt.subplot(3, 1, 3)
    for i, (rho, x) in enumerate(zip(history.rho, history.x)):
        alpha = (i + 1) / len(history.rho)
        
        # Interpolate if needed
        if len(x) != len(x_ref):
            interp = interp1d(x, rho, kind='linear', bounds_error=False, 
                            fill_value='extrapolate')
            rho_plot = interp(x_ref)
            x_plot = x_ref
        else:
            rho_plot = rho
            x_plot = x
            
        plt.plot(x_plot, rho_plot, 'b-', alpha=alpha, 
                 label=f't={history.times[i]:.2e}s' if i == 0 else None)

    plt.xlabel('Position (m)')
    plt.ylabel('Density (kg/m³)')
    plt.legend()
    plt.grid(True)
    plt.title('Density Evolution')

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_results(solver, species_to_plot=None, save_path=None):
    """Plot initial and final states"""
    plt.figure(figsize=(15, 12), dpi=200)
    
    history = solver.history
    x_final = history.x[-1]
    
    # Plot species mass fractions
    plt.subplot(4, 1, 1)
    if species_to_plot is None:
        species_to_plot = [0, 1, 2]  # Default to first three species
    
    colors = ['b', 'orange', 'g', 'r']
    species_names = solver.gas.species_names
    for k, spec_name in enumerate(species_to_plot):
        # Final state
        spec_idx = solver.gas.species_index(spec_name)
        Y_final = history.Y[-1][spec_idx]
        plt.plot(x_final, Y_final, color=colors[k], 
                label=f"{spec_idx} ({species_names[spec_idx]}) (tfinal)")
        
        # Initial state
        x_init = history.x[0]
        Y_init = history.Y[0][spec_idx]
        if len(x_init) != len(x_final):
            interp = interp1d(x_init, Y_init, kind='linear', 
                            bounds_error=False, fill_value='extrapolate')
            Y_init = interp(x_final)
            
        plt.plot(x_final, Y_init, '-o', color=colors[k], 
                label=f'{spec_idx} ({species_names[spec_idx]}) (t=0s)')
    
    plt.xlabel('Position (m)')
    plt.ylabel('Mass Fraction')
    plt.legend()
    plt.grid(True)
    
    # Plot velocity
    plt.subplot(4, 1, 2)
    plt.plot(x_final, history.U[-1], 'k-', label='Velocity (tfinal)')
    
    # Initial velocity
    U_init = history.U[0]
    x_init = history.x[0]
    if len(x_init) != len(x_final):
        interp = interp1d(x_init, U_init, kind='linear', 
                        bounds_error=False, fill_value='extrapolate')
        U_init = interp(x_final)
    plt.plot(x_final, U_init, '-o', color='k', label='Velocity (t=0s)')
    
    plt.xlabel('Position (m)')
    plt.ylabel('Velocity (m/s)')
    plt.legend()
    plt.grid(True)
    
    # Plot temperature
    plt.subplot(4, 1, 3)
    plt.plot(x_final, history.T[-1], 'r-', label='Temperature (tfinal)')
    
    # Initial temperature
    T_init = history.T[0]
    x_init = history.x[0]
    if len(x_init) != len(x_final):
        interp = interp1d(x_init, T_init, kind='linear', 
                        bounds_error=False, fill_value='extrapolate')
        T_init = interp(x_final)
    plt.plot(x_final, T_init, '-o', color='r', label='Temperature (t=0s)')
    
    plt.xlabel('Position (m)')
    plt.ylabel('Temperature (K)')
    plt.legend()
    plt.grid(True)
    
    # Plot density
    plt.subplot(4, 1, 4)
    plt.plot(x_final, history.rho[-1], 'b-', label='Density (tfinal)')
    
    # Initial density
    rho_init = history.rho[0]
    x_init = history.x[0]
    if len(x_init) != len(x_final):
        interp = interp1d(x_init, rho_init, kind='linear', 
                        bounds_error=False, fill_value='extrapolate')
        rho_init = interp(x_final)
    plt.plot(x_final, rho_init, '-o', color='b', label='Density (t=0s)')
    
    plt.xlabel('Position (m)')
    plt.ylabel('Density (kg/m³)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    

def plot_results_(solver):
    """Plot the flame solution"""
    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), dpi=200)
    
    # Get solution
    x = solver.grid.x * 100  # Convert to cm
    T = solver.flame_state.T
    Y = solver.flame_state.Y
    
    # Temperature profile
    ax1.plot(x, T, 'r-', linewidth=2)
    ax1.set_xlabel('Position [cm]')
    ax1.set_ylabel('Temperature [K]')
    ax1.grid(True)
    
    # Major species profiles
    species_indices = {
        'CH4': solver.gas.species_index('CH4'),
        'O2': solver.gas.species_index('O2'),
        'CO2': solver.gas.species_index('CO2'),
        'H2O': solver.gas.species_index('H2O'),
    }
    
    for species, idx in species_indices.items():
        ax2.plot(x, Y[idx], label=species, linewidth=2)
    
    ax2.set_xlabel('Position [cm]')
    ax2.set_ylabel('Mass Fraction [-]')
    ax2.legend()
    ax2.grid(True)
    
    # Grid point distribution
    ax3.plot(x, np.zeros_like(x), 'k.', label='Grid points')
    ax3.set_xlabel('Position [cm]')
    ax3.set_ylabel('Grid Points')
    ax3.set_yticks([])
    ax3.grid(True)
    
    # Plot adaptation history if available
    if solver.grid.adaptation_history:
        for i, grid in enumerate(solver.grid.adaptation_history):
            ax3.plot(grid * 100, np.ones_like(grid) * i, 'k.', alpha=0.3)
        ax3.set_ylim(-1, len(solver.grid.adaptation_history))
        
    plt.tight_layout()
    plt.show()
    
    # Print some statistics
    print("\nSimulation Statistics:")
    print(f"Final time: {solver.t:.3e} s")
    print(f"Number of grid points: {solver.grid.n_points}")
    print(f"Number of grid adaptations: {len(solver.grid.adaptation_history)}")
    print(f"Maximum temperature: {T.max():.1f} K")
    print(f"Flame position: {x[np.argmax(np.gradient(T, x))]:.2f} cm")