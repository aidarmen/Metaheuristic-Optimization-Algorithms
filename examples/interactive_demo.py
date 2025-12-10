"""
Interactive PSO Demonstration
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from algorithms.particle_swarm_optimization import ParticleSwarmOptimization
from problems.function_optimization import (
    sphere_function,
    rastrigin_function,
    ackley_function,
    FUNCTION_INFO
)
from visualization.interactive_pso import InteractivePSOVisualizer


def run_interactive_demo(function_name: str = 'rastrigin'):
    """
    Run interactive PSO demonstration.
    
    Parameters:
    -----------
    function_name : str
        Name of the benchmark function to optimize
    """
    # Get function info
    if function_name not in FUNCTION_INFO:
        print(f"Unknown function: {function_name}")
        print(f"Available functions: {list(FUNCTION_INFO.keys())}")
        return
    
    func_info = FUNCTION_INFO[function_name]
    objective_func = func_info['function']
    bounds = func_info['bounds']
    
    print(f"\n{'='*60}")
    print(f"Interactive PSO Demo - {func_info['name']} Function")
    print(f"{'='*60}")
    print(f"Description: {func_info['description']}")
    print(f"Global Minimum: {func_info['global_min']}")
    print(f"Global Minimum Position: {func_info['global_min_pos']}")
    print(f"Bounds: {bounds}")
    print(f"{'='*60}\n")
    
    # Create PSO optimizer
    pso = ParticleSwarmOptimization(
        objective_func=objective_func,
        bounds=bounds,
        n_particles=30,
        w=0.7,
        c1=1.5,
        c2=1.5,
        max_iter=50,
        verbose=True
    )
    
    # Run optimization
    print("\nRunning PSO optimization...\n")
    results = pso.optimize()
    
    # Display results
    print(f"\n{'='*60}")
    print("Optimization Results:")
    print(f"{'='*60}")
    print(f"Best Position: {results['best_position']}")
    print(f"Best Fitness: {results['best_fitness']:.6f}")
    print(f"Global Minimum: {func_info['global_min']}")
    print(f"Error: {abs(results['best_fitness'] - func_info['global_min']):.6f}")
    print(f"Number of Iterations: {results['n_iterations']}")
    print(f"Number of Function Evaluations: {results['n_evaluations']}")
    print(f"{'='*60}\n")
    
    # Create visualizer
    visualizer = InteractivePSOVisualizer(
        objective_func=objective_func,
        bounds=bounds,
        resolution=50
    )
    
    # Plot convergence
    print("Generating convergence plot...")
    visualizer.plot_convergence(results['history'], show_plot=False)
    plt.savefig(f'media/{function_name}_convergence.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Convergence plot saved as media/{function_name}_convergence.png")
    
    # Create 3D GIF animation
    print("\nGenerating 3D GIF animation...")
    gif_path = visualizer.create_3d_gif(
        positions_history=results['history']['positions'],
        global_best_history=results['history']['global_best_position'],
        save_path=f'media/{function_name}_pso_animation.gif',
        fps=5,
        dpi=100
    )
    print(f"3D GIF saved as {gif_path}")
    
    return results


if __name__ == "__main__":
    # Run demo with different functions
    functions = ['sphere', 'rastrigin', 'ackley']
    
    for func_name in functions:
        try:
            run_interactive_demo(func_name)
            input("\nPress Enter to continue to next function...")
        except KeyboardInterrupt:
            print("\nDemo interrupted by user.")
            break

