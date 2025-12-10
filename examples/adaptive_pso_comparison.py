"""
Adaptive PSO vs Standard PSO vs Gradient Descent Comparison
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from algorithms.adaptive_pso import AdaptiveParticleSwarmOptimization
from algorithms.particle_swarm_optimization import ParticleSwarmOptimization
from algorithms.gradient_descent import GradientDescent
from problems.function_optimization import (
    get_function_gradient,
    FUNCTION_INFO
)
from visualization.interactive_pso import InteractivePSOVisualizer


def compare_adaptive_pso(function_name: str = 'rastrigin', max_iter: int = 50):
    """
    Compare Adaptive PSO with Standard PSO and Gradient Descent.
    
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
        print(f"Available functions: {list(FUNCTION_INFO.keys())}")
        return
    
    func_info = FUNCTION_INFO[function_name]
    objective_func = func_info['function']
    bounds = func_info['bounds']
    
    print(f"\n{'='*70}")
    print(f"Comparison: Adaptive PSO vs Standard PSO vs Gradient Descent")
    print(f"Function: {func_info['name']}")
    print(f"{'='*70}\n")
    
    results = {}
    
    # Set random seed for fair comparison (same initial conditions)
    np.random.seed(42)
    
    # Run Adaptive PSO
    print("Running Adaptive PSO...")
    adaptive_pso = AdaptiveParticleSwarmOptimization(
        objective_func=objective_func,
        bounds=bounds,
        n_particles=30,
        w_max=0.9,
        w_min=0.4,
        c1_max=2.5,
        c1_min=0.5,
        c2_min=0.5,
        c2_max=2.5,
        max_iter=max_iter,
        verbose=False
    )
    adaptive_results = adaptive_pso.optimize()
    results['Adaptive PSO'] = adaptive_results
    print(f"Adaptive PSO - Best Fitness: {adaptive_results['best_fitness']:.6f}, "
          f"Evaluations: {adaptive_results['n_evaluations']}")
    
    # Reset seed for Standard PSO
    np.random.seed(42)
    
    # Run Standard PSO
    print("\nRunning Standard PSO...")
    standard_pso = ParticleSwarmOptimization(
        objective_func=objective_func,
        bounds=bounds,
        n_particles=30,
        w=0.7,
        c1=1.5,
        c2=1.5,
        max_iter=max_iter,
        verbose=False
    )
    standard_results = standard_pso.optimize()
    results['Standard PSO'] = standard_results
    print(f"Standard PSO - Best Fitness: {standard_results['best_fitness']:.6f}, "
          f"Evaluations: {standard_results['n_evaluations']}")
    
    # Run Gradient Descent (if gradient available)
    gd_results = None
    try:
        print("\nRunning Gradient Descent...")
        gradient_func = get_function_gradient(function_name)
        
        # Run multiple trials with different starting points
        best_gd_fitness = float('inf')
        best_gd_history = None
        best_gd_position = None
        
        for trial in range(10):
            initial_pos = np.random.uniform(bounds[:, 0], bounds[:, 1], len(bounds))
            learning_rate = 0.01 * (0.5 + np.random.rand())  # Vary learning rate
            
            gradient_descent = GradientDescent(
                objective_func=objective_func,
                gradient_func=gradient_func,
                bounds=bounds,
                learning_rate=learning_rate,
                max_iter=max_iter * 30,
                verbose=False
            )
            trial_results = gradient_descent.optimize(initial_position=initial_pos)
            
            if trial_results['best_fitness'] < best_gd_fitness:
                best_gd_fitness = trial_results['best_fitness']
                best_gd_history = trial_results['history']
                best_gd_position = trial_results['best_position']
        
        gd_results = {
            'best_position': best_gd_position,
            'best_fitness': best_gd_fitness,
            'history': best_gd_history,
            'n_iterations': max_iter * 30,
            'n_evaluations': max_iter * 30
        }
        results['Gradient Descent'] = gd_results
        print(f"Gradient Descent (best of 10 trials) - Best Fitness: {gd_results['best_fitness']:.6f}, "
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
    
    # Create visualizer
    print("Generating visualizations...")
    visualizer = InteractivePSOVisualizer(
        objective_func=objective_func,
        bounds=bounds,
        resolution=50
    )
    
    # 1. Convergence Comparison Plot
    print("Creating convergence comparison plot...")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot Adaptive PSO
    adaptive_iterations = range(len(adaptive_results['history']['global_best_fitness']))
    ax.plot(adaptive_iterations, adaptive_results['history']['global_best_fitness'], 
           label='Adaptive PSO', linewidth=2, marker='o', markersize=3, color='blue')
    
    # Plot Standard PSO
    standard_iterations = range(len(standard_results['history']['global_best_fitness']))
    ax.plot(standard_iterations, standard_results['history']['global_best_fitness'], 
           label='Standard PSO', linewidth=2, marker='s', markersize=3, color='green')
    
    # Plot Gradient Descent if available
    if gd_results:
        gd_iterations = range(len(gd_results['history']['fitness']))
        ax.plot(gd_iterations, gd_results['history']['fitness'], 
               label='Gradient Descent', linewidth=2, marker='^', markersize=3, color='red')
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Best Fitness', fontsize=12)
    ax.set_title('Convergence Comparison: Adaptive PSO vs Standard PSO vs Gradient Descent', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Use log scale if all values are positive
    all_positive = True
    if np.any(np.array(adaptive_results['history']['global_best_fitness']) <= 0):
        all_positive = False
    if np.any(np.array(standard_results['history']['global_best_fitness']) <= 0):
        all_positive = False
    if gd_results and np.any(np.array(gd_results['history']['fitness']) <= 0):
        all_positive = False
    
    if all_positive:
        ax.set_yscale('log')
    
    plt.tight_layout()
    comparison_path = f'media/{function_name}_adaptive_comparison.png'
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Convergence comparison saved as {comparison_path}")
    
    # 2. Parameter Adaptation Plot
    print("Creating parameter adaptation plot...")
    visualizer.plot_parameter_adaptation(
        parameter_history=adaptive_results['history'],
        show_plot=False
    )
    param_path = f'media/{function_name}_adaptive_parameters.png'
    plt.savefig(param_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Parameter adaptation plot saved as {param_path}")
    
    # 3. 3D GIF Animations
    print("\nGenerating 3D GIF animations...")
    
    # Adaptive PSO animation
    adaptive_gif_path = f'media/{function_name}_adaptive_pso_animation.gif'
    visualizer.create_3d_gif(
        positions_history=adaptive_results['history']['positions'],
        global_best_history=adaptive_results['history']['global_best_position'],
        save_path=adaptive_gif_path,
        fps=5,
        dpi=100
    )
    print(f"Adaptive PSO animation saved as {adaptive_gif_path}")
    
    # Standard PSO animation
    standard_gif_path = f'media/{function_name}_standard_pso_animation.gif'
    visualizer.create_3d_gif(
        positions_history=standard_results['history']['positions'],
        global_best_history=standard_results['history']['global_best_position'],
        save_path=standard_gif_path,
        fps=5,
        dpi=100
    )
    print(f"Standard PSO animation saved as {standard_gif_path}")
    
    # 4. Performance Metrics Comparison
    print("\nGenerating performance metrics comparison...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = list(results.keys())
    final_fitness = [results[m]['best_fitness'] for m in methods]
    colors = ['blue', 'green', 'red', 'orange', 'purple']
    
    bars = ax.bar(methods, final_fitness, color=colors[:len(methods)])
    ax.set_ylabel('Final Best Fitness', fontsize=12)
    ax.set_title('Final Solution Quality Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Use log scale if all values are positive
    if np.all(np.array(final_fitness) > 0):
        ax.set_yscale('log')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2e}', ha='center', va='bottom', fontsize=9)
        else:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2e}', ha='center', va='top', fontsize=9)
    
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    metrics_path = f'media/{function_name}_adaptive_metrics.png'
    plt.savefig(metrics_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Performance metrics saved as {metrics_path}")
    
    print(f"\n{'='*70}")
    print("All visualizations generated successfully!")
    print(f"{'='*70}\n")
    
    return results


def run_multiple_function_comparison():
    """Compare Adaptive PSO across multiple benchmark functions."""
    functions = ['sphere', 'rastrigin', 'ackley', 'beale', 'goldstein_price']
    
    all_results = {}
    
    for func_name in functions:
        if func_name not in FUNCTION_INFO:
            print(f"Skipping {func_name} - not available")
            continue
        
        print(f"\n\n{'#'*70}")
        print(f"# Testing on {FUNCTION_INFO[func_name]['name']} Function")
        print(f"{'#'*70}")
        all_results[func_name] = compare_adaptive_pso(func_name, max_iter=50)
    
    # Summary across all functions
    print(f"\n\n{'='*70}")
    print("Overall Summary Across All Functions")
    print(f"{'='*70}")
    
    for func_name, func_results in all_results.items():
        print(f"\n{FUNCTION_INFO[func_name]['name']} Function:")
        for method_name, method_results in func_results.items():
            print(f"  {method_name}: {method_results['best_fitness']:.6f}")
    
    return all_results


if __name__ == "__main__":
    # Run single function comparison
    compare_adaptive_pso('rastrigin', max_iter=50)
    
    # Uncomment to run multiple function comparison
    # run_multiple_function_comparison()

