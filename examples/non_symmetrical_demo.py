"""
Non-Symmetrical Optimization Functions Demo
Demonstrates PSO on non-symmetrical optimization problems
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from algorithms.particle_swarm_optimization import ParticleSwarmOptimization
from problems.function_optimization import FUNCTION_INFO
from visualization.interactive_pso import InteractivePSOVisualizer


def run_non_symmetrical_demo(function_name: str = 'beale'):
    """
    Run PSO demonstration on non-symmetrical functions.
    
    Parameters:
    -----------
    function_name : str
        Name of the non-symmetrical benchmark function to optimize
    """
    # Get function info
    if function_name not in FUNCTION_INFO:
        print(f"Unknown function: {function_name}")
        print(f"Available functions: {list(FUNCTION_INFO.keys())}")
        return
    
    func_info = FUNCTION_INFO[function_name]
    objective_func = func_info['function']
    bounds = func_info['bounds']
    
    print(f"\n{'='*70}")
    print(f"Non-Symmetrical PSO Demo - {func_info['name']} Function")
    print(f"{'='*70}")
    print(f"Description: {func_info['description']}")
    print(f"Global Minimum: {func_info['global_min']}")
    print(f"Global Minimum Position: {func_info['global_min_pos']}")
    print(f"Bounds: {bounds}")
    print(f"{'='*70}\n")
    
    # Create PSO optimizer with adjusted parameters for non-symmetrical functions
    pso = ParticleSwarmOptimization(
        objective_func=objective_func,
        bounds=bounds,
        n_particles=40,  # More particles for better exploration
        w=0.8,           # Higher inertia for more exploration
        c1=2.0,          # Higher cognitive component
        c2=2.0,          # Higher social component
        max_iter=80,     # More iterations for complex landscapes
        verbose=True
    )
    
    # Run optimization
    print("\nRunning PSO optimization...\n")
    results = pso.optimize()
    
    # Display results
    print(f"\n{'='*70}")
    print("Optimization Results:")
    print(f"{'='*70}")
    print(f"Best Position: {results['best_position']}")
    print(f"Best Fitness: {results['best_fitness']:.6f}")
    print(f"Global Minimum: {func_info['global_min']}")
    print(f"Global Minimum Position: {func_info['global_min_pos']}")
    print(f"Position Error: {np.linalg.norm(results['best_position'] - func_info['global_min_pos']):.6f}")
    print(f"Fitness Error: {abs(results['best_fitness'] - func_info['global_min']):.6f}")
    print(f"Number of Iterations: {results['n_iterations']}")
    print(f"Number of Function Evaluations: {results['n_evaluations']}")
    print(f"{'='*70}\n")
    
    # Create visualizer
    visualizer = InteractivePSOVisualizer(
        objective_func=objective_func,
        bounds=bounds,
        resolution=80  # Higher resolution for better visualization
    )
    
    # Plot convergence
    print("Generating convergence plot...")
    visualizer.plot_convergence(results['history'], show_plot=False)
    plt.savefig(f'media/{function_name}_convergence.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Convergence plot saved as media/{function_name}_convergence.png")
    
    # Create 3D GIF animation
    print("\nGenerating 3D GIF animation...")
    gif_path = visualizer.create_3d_gif(
        positions_history=results['history']['positions'],
        global_best_history=results['history']['global_best_position'],
        save_path=f'media/{function_name}_pso_animation.gif',
        fps=6,  # Slightly faster for non-symmetrical functions
        dpi=120  # Higher resolution
    )
    print(f"3D GIF saved as {gif_path}")
    
    return results


def compare_non_symmetrical_functions():
    """Compare PSO performance on multiple non-symmetrical functions."""
    # Non-symmetrical functions
    non_sym_functions = [
        'beale',
        'goldstein_price',
        'three_hump_camel',
        'rotated_ellipsoid',
        'shifted_sphere',
        'schwefel',
        'easom'
    ]
    
    print(f"\n{'#'*70}")
    print("# Non-Symmetrical Optimization Functions Comparison")
    print(f"{'#'*70}\n")
    
    all_results = {}
    
    for func_name in non_sym_functions:
        if func_name not in FUNCTION_INFO:
            print(f"Skipping {func_name} - not available")
            continue
        
        try:
            print(f"\n{'='*70}")
            print(f"Testing: {FUNCTION_INFO[func_name]['name']} Function")
            print(f"{'='*70}")
            
            results = run_non_symmetrical_demo(func_name)
            all_results[func_name] = results
            
            print(f"\n✓ Completed {func_name}")
            
        except Exception as e:
            print(f"\n✗ Error with {func_name}: {e}")
            continue
    
    # Summary
    print(f"\n\n{'#'*70}")
    print("# Summary of Non-Symmetrical Functions")
    print(f"{'#'*70}\n")
    print(f"{'Function':<25} {'Best Fitness':<20} {'Position Error':<20} {'Fitness Error':<20}")
    print("-" * 85)
    
    for func_name, results in all_results.items():
        func_info = FUNCTION_INFO[func_name]
        pos_error = np.linalg.norm(results['best_position'] - func_info['global_min_pos'])
        fit_error = abs(results['best_fitness'] - func_info['global_min'])
        print(f"{func_info['name']:<25} {results['best_fitness']:<20.6f} "
              f"{pos_error:<20.6f} {fit_error:<20.6f}")
    
    print(f"\n{'#'*70}\n")
    
    return all_results


if __name__ == "__main__":
    # Run individual demo
    # print("Running individual non-symmetrical function demo...")
    # run_non_symmetrical_demo('goldstein_price')
    
    # Uncomment to run comparison of all non-symmetrical functions
    print("\n\nRunning comparison of all non-symmetrical functions...")
    compare_non_symmetrical_functions()

