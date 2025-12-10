"""
PSO vs Traditional Optimization Methods Comparison
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from algorithms.particle_swarm_optimization import ParticleSwarmOptimization
from algorithms.gradient_descent import GradientDescent
from algorithms.random_search import RandomSearch
from algorithms.hill_climbing import HillClimbing
from problems.function_optimization import (
    sphere_function,
    rastrigin_function,
    ackley_function,
    get_function_gradient,
    FUNCTION_INFO
)
from visualization.interactive_pso import InteractivePSOVisualizer


def compare_methods(function_name: str = 'rastrigin', max_iter: int = 50):
    """
    Compare PSO with traditional optimization methods.
    
    Parameters:
    -----------
    function_name : str
        Name of the benchmark function
    max_iter : int
        Maximum number of iterations
    """
    # Get function info
    if function_name not in FUNCTION_INFO:
        print(f"Unknown function: {function_name}")
        return
    
    func_info = FUNCTION_INFO[function_name]
    objective_func = func_info['function']
    bounds = func_info['bounds']
    
    print(f"\n{'='*70}")
    print(f"Comparison: PSO vs Traditional Methods - {func_info['name']} Function")
    print(f"{'='*70}\n")
    
    results = {}
    
    # Run PSO
    print("Running PSO...")
    pso = ParticleSwarmOptimization(
        objective_func=objective_func,
        bounds=bounds,
        n_particles=30,
        w=0.7,
        c1=1.5,
        c2=1.5,
        max_iter=max_iter,
        verbose=False
    )
    pso_results = pso.optimize()
    results['PSO'] = pso_results
    print(f"PSO - Best Fitness: {pso_results['best_fitness']:.6f}, "
          f"Evaluations: {pso_results['n_evaluations']}")
    
    # Run Random Search
    print("\nRunning Random Search...")
    random_search = RandomSearch(
        objective_func=objective_func,
        bounds=bounds,
        max_iter=max_iter * 30,  # Same number of evaluations as PSO
        verbose=False
    )
    rs_results = random_search.optimize()
    results['Random Search'] = rs_results
    print(f"Random Search - Best Fitness: {rs_results['best_fitness']:.6f}, "
          f"Evaluations: {rs_results['n_evaluations']}")
    
    # Run Hill Climbing
    print("\nRunning Hill Climbing...")
    hill_climbing = HillClimbing(
        objective_func=objective_func,
        bounds=bounds,
        step_size=0.1,
        max_iter=max_iter * 30,  # Same number of evaluations
        verbose=False
    )
    hc_results = hill_climbing.optimize()
    results['Hill Climbing'] = hc_results
    print(f"Hill Climbing - Best Fitness: {hc_results['best_fitness']:.6f}, "
          f"Evaluations: {hc_results['n_evaluations']}")
    
    # Run Gradient Descent (if gradient available)
    try:
        print("\nRunning Gradient Descent...")
        gradient_func = get_function_gradient(function_name)
        initial_pos = np.random.uniform(bounds[:, 0], bounds[:, 1], len(bounds))
        
        gradient_descent = GradientDescent(
            objective_func=objective_func,
            gradient_func=gradient_func,
            bounds=bounds,
            learning_rate=0.01,
            max_iter=max_iter * 30,
            verbose=False
        )
        gd_results = gradient_descent.optimize(initial_position=initial_pos)
        results['Gradient Descent'] = gd_results
        print(f"Gradient Descent - Best Fitness: {gd_results['best_fitness']:.6f}, "
              f"Evaluations: {gd_results['n_evaluations']}")
    except Exception as e:
        print(f"Gradient Descent skipped: {e}")
    
    # Display comparison summary
    print(f"\n{'='*70}")
    print("Comparison Summary:")
    print(f"{'='*70}")
    print(f"{'Method':<20} {'Best Fitness':<20} {'Evaluations':<15} {'Error':<15}")
    print("-" * 70)
    
    global_min = func_info['global_min']
    for method_name, method_results in results.items():
        error = abs(method_results['best_fitness'] - global_min)
        print(f"{method_name:<20} {method_results['best_fitness']:<20.6f} "
              f"{method_results['n_evaluations']:<15} {error:<15.6f}")
    print(f"{'='*70}\n")
    
    # Create comparison visualization
    print("Generating comparison plots...")
    visualizer = InteractivePSOVisualizer(
        objective_func=objective_func,
        bounds=bounds,
        resolution=50
    )
    
    # Prepare histories for comparison
    other_histories = {}
    for method_name, method_results in results.items():
        if method_name != 'PSO':
            other_histories[method_name] = method_results['history']
    
    # Create comparison plot
    visualizer.plot_comparison(
        pso_history=pso_results['history'],
        other_histories=other_histories,
        show_plot=True
    )
    
    # Create interactive Plotly comparison
    try:
        fig = visualizer.create_interactive_plotly_comparison(
            pso_history=pso_results['history'],
            other_histories=other_histories
        )
        print("\nDisplaying interactive Plotly comparison...")
        fig.show()
    except Exception as e:
        print(f"Plotly visualization skipped: {e}")
    
    return results


def run_multiple_function_comparison():
    """Compare methods across multiple benchmark functions."""
    functions = ['sphere', 'rastrigin', 'ackley']
    
    all_results = {}
    
    for func_name in functions:
        print(f"\n\n{'#'*70}")
        print(f"# Testing on {FUNCTION_INFO[func_name]['name']} Function")
        print(f"{'#'*70}")
        all_results[func_name] = compare_methods(func_name, max_iter=30)
        input("\nPress Enter to continue to next function...")
    
    # Summary across all functions
    print(f"\n\n{'='*70}")
    print("Overall Summary Across All Functions")
    print(f"{'='*70}")
    
    for func_name, func_results in all_results.items():
        print(f"\n{func_name.upper()} Function:")
        for method_name, method_results in func_results.items():
            print(f"  {method_name}: {method_results['best_fitness']:.6f}")
    
    return all_results


if __name__ == "__main__":
    # Run single function comparison
    # compare_methods('rastrigin', max_iter=50)
    
    # Uncomment to run multiple function comparison
    run_multiple_function_comparison()

