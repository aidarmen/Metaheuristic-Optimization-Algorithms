"""
Hill Climbing Optimization Implementation
"""

import numpy as np
from typing import Callable, Optional


class HillClimbing:
    """
    Hill Climbing optimization algorithm.
    
    A local search algorithm that starts with an arbitrary solution and
    iteratively improves it by making small changes.
    """
    
    def __init__(
        self,
        objective_func: Callable,
        bounds: np.ndarray,
        step_size: float = 0.1,
        max_iter: int = 100,
        verbose: bool = True
    ):
        """
        Initialize Hill Climbing algorithm.
        
        Parameters:
        -----------
        objective_func : Callable
            Objective function to minimize
        bounds : np.ndarray
            Bounds for each dimension, shape (n_dim, 2)
        step_size : float
            Size of step for neighbor generation
        max_iter : int
            Maximum number of iterations
        verbose : bool
            Print progress information
        """
        self.objective_func = objective_func
        self.bounds = np.array(bounds)
        self.step_size = step_size
        self.max_iter = max_iter
        self.verbose = verbose
        
        self.n_dim = len(bounds)
        self.dim_min = self.bounds[:, 0]
        self.dim_max = self.bounds[:, 1]
        
        # History tracking
        self.history = {
            'positions': [],
            'fitness': []
        }
        
    def optimize(self, initial_position: Optional[np.ndarray] = None) -> dict:
        """
        Run hill climbing optimization.
        
        Parameters:
        -----------
        initial_position : Optional[np.ndarray]
            Starting position. If None, random initialization.
        
        Returns:
        --------
        dict : Optimization results
        """
        # Initialize position
        if initial_position is None:
            self.position = np.random.uniform(
                self.dim_min,
                self.dim_max,
                self.n_dim
            )
        else:
            self.position = np.array(initial_position).copy()
        
        self.best_position = self.position.copy()
        self.best_fitness = self.objective_func(self.position)
        
        n_evaluations = 1
        no_improvement_count = 0
        
        for iteration in range(self.max_iter):
            # Generate neighbor
            step = np.random.uniform(-self.step_size, self.step_size, self.n_dim)
            neighbor = self.position + step
            
            # Apply bounds
            neighbor = np.clip(neighbor, self.dim_min, self.dim_max)
            
            # Evaluate neighbor
            neighbor_fitness = self.objective_func(neighbor)
            n_evaluations += 1
            
            # Accept if better
            if neighbor_fitness < self.best_fitness:
                self.position = neighbor
                self.best_position = neighbor.copy()
                self.best_fitness = neighbor_fitness
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            # Store history
            self.history['positions'].append(self.position.copy())
            self.history['fitness'].append(self.best_fitness)
            
            # Early stopping if no improvement
            if no_improvement_count > 20:
                if self.verbose:
                    print(f"No improvement for 20 iterations, stopping at {iteration + 1}")
                break
            
            if self.verbose and (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{self.max_iter}, "
                      f"Best fitness: {self.best_fitness:.6f}")
        
        return {
            'best_position': self.best_position,
            'best_fitness': self.best_fitness,
            'history': self.history,
            'n_iterations': iteration + 1,
            'n_evaluations': n_evaluations
        }

