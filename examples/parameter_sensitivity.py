"""
PSO Parameter Sensitivity Analysis
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from algorithms.particle_swarm_optimization import ParticleSwarmOptimization
from problems.function_optimization import rastrigin_function, FUNCTION_INFO


def analyze_parameter_sensitivity(function_name: str = 'rastrigin', max_iter: int = 30):
    """
    Analyze sensitivity of PSO to different parameter settings.
    
    Parameters:
    -----------
    function_name : str
        Name of the benchmark function
    max_iter : int
        Maximum number of iterations
    """
    func_info = FUNCTION_INFO[function_name]
    objective_func = func_info['function']
    bounds = func_info['bounds']
    
    print(f"\n{'='*70}")
    print(f"PSO Parameter Sensitivity Analysis - {func_info['name']} Function")
    print(f"{'='*70}\n")
    
    # Test different inertia weights
    print("Testing different inertia weights (w)...")
    w_values = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    w_results = {}
    
    for w in w_values:
        pso = ParticleSwarmOptimization(
            objective_func=objective_func,
            bounds=bounds,
            n_particles=20,
            w=w,
            c1=1.5,
            c2=1.5,
            max_iter=max_iter,
            verbose=False
        )
        results = pso.optimize()
        w_results[w] = results
        print(f"  w={w:.1f}: Best Fitness = {results['best_fitness']:.6f}")
    
    # Test different cognitive/social coefficients
    print("\nTesting different cognitive/social coefficients (c1, c2)...")
    c_values = [(1.0, 1.0), (1.5, 1.5), (2.0, 2.0), (1.0, 2.0), (2.0, 1.0)]
    c_results = {}
    
    for c1, c2 in c_values:
        pso = ParticleSwarmOptimization(
            objective_func=objective_func,
            bounds=bounds,
            n_particles=20,
            w=0.7,
            c1=c1,
            c2=c2,
            max_iter=max_iter,
            verbose=False
        )
        results = pso.optimize()
        c_results[(c1, c2)] = results
        print(f"  c1={c1:.1f}, c2={c2:.1f}: Best Fitness = {results['best_fitness']:.6f}")
    
    # Test different swarm sizes
    print("\nTesting different swarm sizes...")
    n_particles_values = [10, 20, 30, 50, 100]
    n_results = {}
    
    for n_particles in n_particles_values:
        pso = ParticleSwarmOptimization(
            objective_func=objective_func,
            bounds=bounds,
            n_particles=n_particles,
            w=0.7,
            c1=1.5,
            c2=1.5,
            max_iter=max_iter,
            verbose=False
        )
        results = pso.optimize()
        n_results[n_particles] = results
        print(f"  n_particles={n_particles}: Best Fitness = {results['best_fitness']:.6f}, "
              f"Evaluations = {results['n_evaluations']}")
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Inertia weight sensitivity
    ax1 = axes[0]
    w_list = list(w_results.keys())
    fitness_list = [w_results[w]['best_fitness'] for w in w_list]
    ax1.plot(w_list, fitness_list, 'o-', linewidth=2, markersize=8)
    ax1.set_xlabel('Inertia Weight (w)', fontsize=12)
    ax1.set_ylabel('Best Fitness', fontsize=12)
    ax1.set_title('Sensitivity to Inertia Weight', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Cognitive/Social coefficients sensitivity
    ax2 = axes[1]
    c_labels = [f'({c1},{c2})' for c1, c2 in c_results.keys()]
    c_fitness = [c_results[c]['best_fitness'] for c in c_results.keys()]
    x_pos = np.arange(len(c_labels))
    ax2.bar(x_pos, c_fitness, color='green', alpha=0.7)
    ax2.set_xlabel('(c1, c2)', fontsize=12)
    ax2.set_ylabel('Best Fitness', fontsize=12)
    ax2.set_title('Sensitivity to Cognitive/Social Coefficients', 
                 fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(c_labels, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_yscale('log')
    
    # Swarm size sensitivity
    ax3 = axes[2]
    n_list = list(n_results.keys())
    n_fitness = [n_results[n]['best_fitness'] for n in n_list]
    n_evals = [n_results[n]['n_evaluations'] for n in n_list]
    
    ax3_twin = ax3.twinx()
    line1 = ax3.plot(n_list, n_fitness, 'o-', color='blue', 
                    linewidth=2, markersize=8, label='Best Fitness')
    ax3.set_xlabel('Swarm Size', fontsize=12)
    ax3.set_ylabel('Best Fitness', fontsize=12, color='blue')
    ax3.tick_params(axis='y', labelcolor='blue')
    ax3.set_yscale('log')
    
    line2 = ax3_twin.plot(n_list, n_evals, 's-', color='red', 
                         linewidth=2, markersize=8, label='Evaluations')
    ax3_twin.set_ylabel('Function Evaluations', fontsize=12, color='red')
    ax3_twin.tick_params(axis='y', labelcolor='red')
    
    ax3.set_title('Sensitivity to Swarm Size', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc='upper left')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print(f"\n{'='*70}")
    print("Parameter Sensitivity Summary")
    print(f"{'='*70}")
    print(f"\nBest Inertia Weight: {min(w_results, key=lambda w: w_results[w]['best_fitness']):.1f}")
    print(f"Best (c1, c2): {min(c_results, key=lambda c: c_results[c]['best_fitness'])}")
    print(f"Best Swarm Size: {min(n_results, key=lambda n: n_results[n]['best_fitness'])}")
    print(f"{'='*70}\n")
    
    return {
        'w_results': w_results,
        'c_results': c_results,
        'n_results': n_results
    }


if __name__ == "__main__":
    analyze_parameter_sensitivity('rastrigin', max_iter=30)

