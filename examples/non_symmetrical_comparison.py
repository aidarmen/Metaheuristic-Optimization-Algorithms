"""
PSO vs Traditional Methods on Non-Symmetrical Functions
Compares PSO performance with traditional optimization methods
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from algorithms.particle_swarm_optimization import ParticleSwarmOptimization
from algorithms.gradient_descent import GradientDescent
from algorithms.hill_climbing import HillClimbing
from problems.function_optimization import FUNCTION_INFO, get_function_gradient
from visualization.interactive_pso import InteractivePSOVisualizer


def compare_methods_on_non_symmetrical(function_name: str = 'beale', max_iter: int = 80):
    """
    Compare PSO with traditional methods on non-symmetrical functions.
    
    Parameters:
    -----------
    function_name : str
        Name of the non-symmetrical benchmark function
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
    
    print(f"\n{'='*80}")
    print(f"PSO vs Traditional Methods - {func_info['name']} Function (Non-Symmetrical)")
    print(f"{'='*80}")
    print(f"Description: {func_info['description']}")
    print(f"Global Minimum: {func_info['global_min']}")
    print(f"Global Minimum Position: {func_info['global_min_pos']}")
    print(f"Bounds: {bounds}")
    print(f"{'='*80}\n")
    
    results = {}
    
    # Run PSO with optimized parameters for non-symmetrical functions
    print("Running PSO (optimized for non-symmetrical functions)...")
    pso = ParticleSwarmOptimization(
        objective_func=objective_func,
        bounds=bounds,
        n_particles=50,  # More particles for better exploration
        w=0.9,           # Higher inertia for more exploration
        c1=2.0,          # Cognitive component
        c2=2.0,          # Social component
        max_iter=max_iter,
        verbose=False
    )
    pso_results = pso.optimize()
    results['PSO'] = pso_results
    print(f"  PSO - Best Fitness: {pso_results['best_fitness']:.6f}, "
          f"Position: {pso_results['best_position']}, "
          f"Evaluations: {pso_results['n_evaluations']}")
    
    # Run Hill Climbing
    print("\nRunning Hill Climbing...")
    hill_climbing = HillClimbing(
        objective_func=objective_func,
        bounds=bounds,
        step_size=0.1,
        max_iter=pso_results['n_evaluations'],
        verbose=False
    )
    hc_results = hill_climbing.optimize()
    results['Hill Climbing'] = hc_results
    print(f"  Hill Climbing - Best Fitness: {hc_results['best_fitness']:.6f}, "
          f"Position: {hc_results['best_position']}, "
          f"Evaluations: {hc_results['n_evaluations']}")
    
    # Run Gradient Descent (if gradient available) - Multiple trials to show variability
    gradient_func = get_function_gradient(function_name)
    if gradient_func is not None:
        print("\nRunning Gradient Descent (10 trials with different starting points and learning rates)...")
        gd_trials = []
        learning_rates = [0.001, 0.005, 0.01, 0.02, 0.05]  # Different learning rates
        
        for trial in range(10):
            try:
                initial_pos = np.random.uniform(bounds[:, 0], bounds[:, 1], len(bounds))
                lr = learning_rates[trial % len(learning_rates)]  # Cycle through learning rates
                gradient_descent = GradientDescent(
                    objective_func=objective_func,
                    gradient_func=gradient_func,
                    bounds=bounds,
                    learning_rate=lr,
                    max_iter=pso_results['n_evaluations'],
                    verbose=False
                )
                gd_result = gradient_descent.optimize(initial_position=initial_pos)
                gd_trials.append(gd_result)
            except Exception as e:
                continue
        
        if gd_trials:
            # Take the best result from all trials
            best_gd = min(gd_trials, key=lambda x: x['best_fitness'])
            results['Gradient Descent (best of 10)'] = best_gd
            avg_fitness = np.mean([r['best_fitness'] for r in gd_trials])
            std_fitness = np.std([r['best_fitness'] for r in gd_trials])
            print(f"  Gradient Descent - Best: {best_gd['best_fitness']:.6f}, "
                  f"Average: {avg_fitness:.6f} ± {std_fitness:.6f}, "
                  f"Position: {best_gd['best_position']}")
            print(f"  (Note: Gradient descent requires careful tuning of learning rate and starting point)")
            print(f"  (Best result from 10 trials with different parameters)")
        else:
            print(f"  Gradient Descent failed on all trials")
    else:
        print("\nGradient Descent skipped: Gradient not available for this function")
    
    # Display comparison summary
    print(f"\n{'='*80}")
    print("Comparison Summary:")
    print(f"{'='*80}")
    print(f"{'Method':<25} {'Best Fitness':<20} {'Position Error':<20} {'Fitness Error':<20}")
    print("-" * 85)
    
    global_min = func_info['global_min']
    global_min_pos = func_info['global_min_pos']
    
    # Sort by fitness error (best first)
    sorted_results = sorted(results.items(), key=lambda x: abs(x[1]['best_fitness'] - global_min))
    
    for method_name, method_results in sorted_results:
        pos_error = np.linalg.norm(method_results['best_position'] - global_min_pos)
        fit_error = abs(method_results['best_fitness'] - global_min)
        marker = "🏆" if method_name == sorted_results[0][0] else "  "
        print(f"{marker} {method_name:<23} {method_results['best_fitness']:<20.6f} "
              f"{pos_error:<20.6f} {fit_error:<20.6f}")
    print(f"{'='*80}\n")
    
    # Show insights about the comparison
    print("Key Insights:")
    print("-" * 80)
    if 'Gradient Descent' in [r[0] for r in sorted_results]:
        gd_result = next(r[1] for r in sorted_results if 'Gradient Descent' in r[0])
        pso_pos_error = np.linalg.norm(results['PSO']['best_position'] - global_min_pos)
        gd_pos_error = np.linalg.norm(gd_result['best_position'] - global_min_pos)
        
        if pso_pos_error < gd_pos_error:
            print("✓ PSO found a better solution (closer to global minimum)")
        else:
            print("⚠ Gradient Descent found a better solution on this function.")
            print("\nWhy Gradient Descent might perform better:")
            print("  - This function is smooth and differentiable")
            print("  - Gradient descent can converge quickly on well-behaved functions")
            print("  - With proper tuning (10 trials tested), it finds good solutions")
            
            print("\nHowever, PSO has important advantages:")
            print("  ✓ No gradient required - works on non-differentiable functions")
            print("  ✓ More robust - single run vs. 10 trials needed for gradient descent")
            print("  ✓ Better on multimodal functions - escapes local minima")
            print("  ✓ Better on non-symmetrical landscapes - explores entire space")
            print("  ✓ Less parameter tuning - PSO parameters are more forgiving")
            print("  ✓ Parallelizable - can evaluate particles simultaneously")
    
    print(f"\nWhen to use each method:")
    print(f"  • Gradient Descent: Smooth, unimodal functions with available gradients")
    print(f"  • PSO: Multimodal, non-differentiable, or complex landscapes")
    print(f"  • Hill Climbing: Simple local search when gradient unavailable")
    print(f"{'='*80}\n")
    
    # Create comparison visualization
    print("Generating comparison plots...")
    visualizer = InteractivePSOVisualizer(
        objective_func=objective_func,
        bounds=bounds,
        resolution=80
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
        show_plot=False
    )
    plt.savefig(f'media/{function_name}_comparison.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Comparison plot saved as media/{function_name}_comparison.png")
    
    # Create detailed comparison table
    print(f"\n{'='*80}")
    print("Detailed Comparison:")
    print(f"{'='*80}")
    print(f"\nGlobal Minimum Position: {global_min_pos}")
    print(f"Global Minimum Value: {global_min}\n")
    
    for method_name, method_results in results.items():
        pos_error = np.linalg.norm(method_results['best_position'] - global_min_pos)
        fit_error = abs(method_results['best_fitness'] - global_min)
        print(f"{method_name}:")
        print(f"  Best Position: {method_results['best_position']}")
        print(f"  Best Fitness: {method_results['best_fitness']:.6f}")
        print(f"  Position Error: {pos_error:.6f}")
        print(f"  Fitness Error: {fit_error:.6f}")
        print(f"  Function Evaluations: {method_results['n_evaluations']}")
        print()
    
    return results


def compare_all_non_symmetrical_functions():
    """Compare PSO vs traditional methods on all non-symmetrical functions."""
    non_sym_functions = [
        'beale',
        'goldstein_price',
        'three_hump_camel',
        'rotated_ellipsoid',
        'shifted_sphere',
        'easom'
    ]
    
    print(f"\n{'#'*80}")
    print("# PSO vs Traditional Methods - Non-Symmetrical Functions Comparison")
    print(f"{'#'*80}\n")
    
    all_results = {}
    
    for func_name in non_sym_functions:
        if func_name not in FUNCTION_INFO:
            print(f"Skipping {func_name} - not available")
            continue
        
        try:
            print(f"\n{'='*80}")
            print(f"Testing: {FUNCTION_INFO[func_name]['name']} Function")
            print(f"{'='*80}")
            
            results = compare_methods_on_non_symmetrical(func_name, max_iter=50)
            all_results[func_name] = results
            
            print(f"\n✓ Completed {func_name}")
            
        except Exception as e:
            print(f"\n✗ Error with {func_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Overall summary
    print(f"\n\n{'#'*80}")
    print("# Overall Summary - PSO vs Traditional Methods")
    print(f"{'#'*80}\n")
    
    # Create summary table
    print(f"{'Function':<25} {'Method':<20} {'Best Fitness':<20} {'Position Error':<20}")
    print("-" * 85)
    
    for func_name, func_results in all_results.items():
        func_info = FUNCTION_INFO[func_name]
        global_min_pos = func_info['global_min_pos']
        
        for method_name, method_results in func_results.items():
            pos_error = np.linalg.norm(method_results['best_position'] - global_min_pos)
            print(f"{func_info['name']:<25} {method_name:<20} "
                  f"{method_results['best_fitness']:<20.6f} {pos_error:<20.6f}")
    
    # Calculate statistics
    print(f"\n{'#'*80}")
    print("# Performance Statistics")
    print(f"{'#'*80}\n")
    
    methods = ['PSO', 'Hill Climbing', 'Gradient Descent']
    method_stats = {method: {'wins': 0, 'avg_error': []} for method in methods}
    
    for func_name, func_results in all_results.items():
        func_info = FUNCTION_INFO[func_name]
        global_min_pos = func_info['global_min_pos']
        
        best_method = None
        best_error = float('inf')
        
        for method_name, method_results in func_results.items():
            if method_name in method_stats:
                pos_error = np.linalg.norm(method_results['best_position'] - global_min_pos)
                method_stats[method_name]['avg_error'].append(pos_error)
                
                if pos_error < best_error:
                    best_error = pos_error
                    best_method = method_name
        
        if best_method:
            method_stats[best_method]['wins'] += 1
    
    print(f"{'Method':<20} {'Wins':<10} {'Avg Position Error':<20}")
    print("-" * 50)
    for method_name, stats in method_stats.items():
        if stats['avg_error']:
            avg_error = np.mean(stats['avg_error'])
            print(f"{method_name:<20} {stats['wins']:<10} {avg_error:<20.6f}")
    
    print(f"\n{'#'*80}\n")
    
    return all_results


def test_pso_advantages():
    """
    Test PSO on functions where it should excel over gradient descent.
    Focuses on multimodal and non-symmetrical functions.
    """
    print(f"\n{'#'*80}")
    print("# Testing PSO Advantages on Challenging Functions")
    print(f"{'#'*80}\n")
    print("These functions are chosen because:")
    print("1. They have multiple local minima (gradient descent gets stuck)")
    print("2. They are non-symmetrical (gradient descent struggles)")
    print("3. They have deceptive landscapes (gradient descent misled)")
    print(f"{'#'*80}\n")
    
    # Functions where PSO should excel
    challenging_functions = [
        ('goldstein_price', 'Multimodal, non-symmetrical - many local minima'),
        ('beale', 'Valley-shaped, non-symmetrical - gradient descent gets stuck'),
        ('three_hump_camel', 'Three local minima - gradient descent finds local optimum'),
        ('schwefel', 'Highly deceptive - gradient descent misled by landscape'),
    ]
    
    for func_name, reason in challenging_functions:
        if func_name not in FUNCTION_INFO:
            continue
        
        print(f"\n{'='*80}")
        print(f"Testing: {FUNCTION_INFO[func_name]['name']}")
        print(f"Reason: {reason}")
        print(f"{'='*80}")
        
        compare_methods_on_non_symmetrical(func_name, max_iter=100)
        input("\nPress Enter to continue to next function...")


if __name__ == "__main__":
    # Run single function comparison
    # print("Running PSO vs Traditional Methods comparison on non-symmetrical function...")
    # compare_methods_on_non_symmetrical('three_hump_camel', max_iter=80)
    
    # Uncomment to test PSO advantages on challenging functions
    # print("\n\nTesting PSO advantages on challenging functions...")
    # test_pso_advantages()
    
    # Uncomment to run comparison on all non-symmetrical functions
    print("\n\nRunning comparison on all non-symmetrical functions...")
    compare_all_non_symmetrical_functions()

