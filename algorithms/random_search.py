"""
Random Search Optimization Implementation
"""

import numpy as np
from typing import Callable, Optional


class RandomSearch:
    """
    Random Search optimization algorithm.
    
    A simple baseline optimization method that randomly samples the search space.
    """
    
    def __init__(
        self,
        objective_func: Callable,
        bounds: np.ndarray,
        max_iter: int = 100,
        verbose: bool = True
    ):
        """
        Initialize Random Search algorithm.
        
        Parameters:
        -----------
        objective_func : Callable
            Objective function to minimize
        bounds : np.ndarray
            Bounds for each dimension, shape (n_dim, 2)
        max_iter : int
            Maximum number of iterations (random samples)
        verbose : bool
            Print progress information
        """
        self.objective_func = objective_func
        self.bounds = np.array(bounds)
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
        
    def optimize(self) -> dict:
        """
        Run random search optimization.
        
        Returns:
        --------
        dict : Optimization results
        """
        self.best_position = None
        self.best_fitness = float('inf')
        
        for iteration in range(self.max_iter):
            # Random sample
            position = np.random.uniform(
                self.dim_min,
                self.dim_max,
                self.n_dim
            )
            
            # Evaluate
            fitness = self.objective_func(position)
            
            # Update best
            if fitness < self.best_fitness:
                self.best_fitness = fitness
                self.best_position = position.copy()
            
            # Store history
            self.history['positions'].append(position.copy())
            self.history['fitness'].append(fitness)
            
            if self.verbose and (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{self.max_iter}, "
                      f"Best fitness: {self.best_fitness:.6f}")
        
        return {
            'best_position': self.best_position,
            'best_fitness': self.best_fitness,
            'history': self.history,
            'n_iterations': self.max_iter,
            'n_evaluations': self.max_iter
        }

