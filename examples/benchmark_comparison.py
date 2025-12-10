"""
Compare PSO Performance Across Different Benchmark Functions
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from algorithms.particle_swarm_optimization import ParticleSwarmOptimization
from problems.function_optimization import FUNCTION_INFO
from visualization.interactive_pso import InteractivePSOVisualizer


def benchmark_pso_across_functions(max_iter: int = 50):
    """
    Test PSO performance on multiple benchmark functions.
    
    Parameters:
    -----------
    max_iter : int
        Maximum number of iterations
    """
    print(f"\n{'='*70}")
    print("PSO Benchmark Comparison Across Functions")
    print(f"{'='*70}\n")
    
    results = {}
    
    for func_name, func_info in FUNCTION_INFO.items():
        print(f"\nTesting on {func_info['name']} Function...")
        print(f"Description: {func_info['description']}")
        
        objective_func = func_info['function']
        bounds = func_info['bounds']
        
        # Run PSO
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
        results[func_name] = {
            'results': pso_results,
            'info': func_info
        }
        
        error = abs(pso_results['best_fitness'] - func_info['global_min'])
        print(f"  Best Fitness: {pso_results['best_fitness']:.6f}")
        print(f"  Global Minimum: {func_info['global_min']}")
        print(f"  Error: {error:.6f}")
        print(f"  Evaluations: {pso_results['n_evaluations']}")
    
    # Create comparison plot
    print("\n\nGenerating comparison plots...")
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Convergence comparison
    ax1 = axes[0]
    for func_name, data in results.items():
        history = data['results']['history']
        iterations = range(len(history['global_best_fitness']))
        ax1.plot(iterations, history['global_best_fitness'], 
                label=data['info']['name'], linewidth=2, marker='o', markersize=3)
    
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Best Fitness', fontsize=12)
    ax1.set_title('PSO Convergence Comparison Across Functions', 
                 fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Final solution comparison
    ax2 = axes[1]
    func_names = [data['info']['name'] for data in results.values()]
    final_fitness = [data['results']['best_fitness'] for data in results.values()]
    errors = [abs(f - data['info']['global_min']) 
              for f, data in zip(final_fitness, results.values())]
    
    x_pos = np.arange(len(func_names))
    bars = ax2.bar(x_pos, errors, color=['blue', 'green', 'orange', 'red', 'purple'][:len(func_names)])
    ax2.set_xlabel('Function', fontsize=12)
    ax2.set_ylabel('Error from Global Minimum', fontsize=12)
    ax2.set_title('Final Solution Quality Comparison', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(func_names, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_yscale('log')
    
    # Add value labels
    for bar, error in zip(bars, errors):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{error:.2e}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary table
    print(f"\n{'='*70}")
    print("Summary Table")
    print(f"{'='*70}")
    print(f"{'Function':<15} {'Best Fitness':<20} {'Error':<20} {'Evaluations':<15}")
    print("-" * 70)
    
    for func_name, data in results.items():
        r = data['results']
        error = abs(r['best_fitness'] - data['info']['global_min'])
        print(f"{data['info']['name']:<15} {r['best_fitness']:<20.6f} "
              f"{error:<20.6f} {r['n_evaluations']:<15}")
    print(f"{'='*70}\n")
    
    return results


if __name__ == "__main__":
    benchmark_pso_across_functions(max_iter=50)

