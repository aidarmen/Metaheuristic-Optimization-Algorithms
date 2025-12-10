"""
Resource Allocation Optimization using PSO
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from algorithms.particle_swarm_optimization import ParticleSwarmOptimization
from typing import List, Dict, Callable


class ResourceAllocator:
    """
    Resource allocation optimization using PSO.
    
    Example: Allocating limited resources to projects to maximize utility.
    """
    
    def __init__(
        self,
        projects: List[dict],
        total_budget: float,
        project_utilities: List[Callable]
    ):
        """
        Initialize resource allocator.
        
        Parameters:
        -----------
        projects : List[dict]
            List of projects
        total_budget : float
            Total available budget
        project_utilities : List[Callable]
            Utility functions for each project (takes allocation, returns utility)
        """
        self.projects = projects
        self.total_budget = total_budget
        self.project_utilities = project_utilities
        self.n_projects = len(projects)
        
    def _normalize_allocation(self, position: np.ndarray) -> np.ndarray:
        """
        Normalize position to valid budget allocation.
        
        Parameters:
        -----------
        position : np.ndarray
            Continuous position vector
        
        Returns:
        --------
        np.ndarray : Normalized allocation (sums to total_budget)
        """
        # Normalize to [0, 1]
        normalized = (position - position.min()) / (position.max() - position.min() + 1e-10)
        
        # Scale to budget
        allocation = normalized * self.total_budget / normalized.sum()
        
        return allocation
    
    def _objective(self, position: np.ndarray) -> float:
        """
        Objective function: negative total utility (for minimization).
        
        Parameters:
        -----------
        position : np.ndarray
            Continuous position vector
        
        Returns:
        --------
        float : Negative total utility
        """
        allocation = self._normalize_allocation(position)
        
        # Calculate total utility
        total_utility = sum(
            utility(alloc)
            for alloc, utility in zip(allocation, self.project_utilities)
        )
        
        # Return negative (for minimization)
        return -total_utility
    
    def optimize(
        self,
        n_particles: int = 30,
        max_iter: int = 100,
        **pso_kwargs
    ) -> dict:
        """
        Optimize resource allocation using PSO.
        
        Parameters:
        -----------
        n_particles : int
            Number of particles
        max_iter : int
            Maximum iterations
        **pso_kwargs : dict
            Additional PSO parameters
        
        Returns:
        --------
        dict : Optimization results with allocation
        """
        # Bounds: one dimension per project
        bounds = np.array([[0.0, 1.0] for _ in range(self.n_projects)])
        
        # Create PSO optimizer
        pso = ParticleSwarmOptimization(
            objective_func=self._objective,
            bounds=bounds,
            n_particles=n_particles,
            max_iter=max_iter,
            **pso_kwargs
        )
        
        # Optimize
        results = pso.optimize()
        
        # Decode best solution
        best_allocation = self._normalize_allocation(results['best_position'])
        
        results['allocation'] = {
            i: {
                'project': self.projects[i],
                'budget': best_allocation[i],
                'utility': self.project_utilities[i](best_allocation[i])
            }
            for i in range(self.n_projects)
        }
        results['total_utility'] = -results['best_fitness']
        results['total_allocated'] = best_allocation.sum()
        
        return results


# Example usage
def example_resource_allocation():
    """
    Example: Allocating budget to 5 projects.
    """
    # Create projects
    projects = [{'id': i, 'name': f'Project {i}'} for i in range(5)]
    
    # Define utility functions (diminishing returns)
    def utility_func(project_id):
        def func(allocation):
            return np.sqrt(allocation) * (project_id + 1)  # Different utility per project
        return func
    
    project_utilities = [utility_func(i) for i in range(5)]
    
    # Create allocator
    allocator = ResourceAllocator(
        projects=projects,
        total_budget=1000.0,
        project_utilities=project_utilities
    )
    
    # Optimize
    results = allocator.optimize(n_particles=30, max_iter=50)
    
    print("Resource Allocation Results:")
    print(f"Total Utility: {results['total_utility']:.2f}")
    print(f"Total Allocated: {results['total_allocated']:.2f}")
    print("\nProject Allocations:")
    for i, alloc_info in results['allocation'].items():
        print(f"{alloc_info['project']['name']}: "
              f"Budget: {alloc_info['budget']:.2f}, "
              f"Utility: {alloc_info['utility']:.2f}")
    
    return results


if __name__ == "__main__":
    example_resource_allocation()

