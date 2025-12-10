"""
Gradient Descent Optimization Implementation
"""

import numpy as np
from typing import Callable, Tuple, Optional


class GradientDescent:
    """
    Gradient Descent optimization algorithm.
    
    A first-order iterative optimization algorithm for finding a local minimum
    of a differentiable function.
    """
    
    def __init__(
        self,
        objective_func: Callable,
        gradient_func: Callable,
        bounds: np.ndarray,
        learning_rate: float = 0.01,
        max_iter: int = 100,
        tolerance: float = 1e-6,
        verbose: bool = True
    ):
        """
        Initialize Gradient Descent algorithm.
        
        Parameters:
        -----------
        objective_func : Callable
            Objective function to minimize
        gradient_func : Callable
            Gradient function of the objective
        bounds : np.ndarray
            Bounds for each dimension, shape (n_dim, 2)
        learning_rate : float
            Step size for gradient descent
        max_iter : int
            Maximum number of iterations
        tolerance : float
            Convergence tolerance
        verbose : bool
            Print progress information
        """
        self.objective_func = objective_func
        self.gradient_func = gradient_func
        self.bounds = np.array(bounds)
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tolerance = tolerance
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
        Run gradient descent optimization.
        
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
        
        for iteration in range(self.max_iter):
            # Compute gradient
            gradient = self.gradient_func(self.position)
            
            # Update position
            self.position = self.position - self.learning_rate * gradient
            
            # Apply bounds
            self.position = np.clip(
                self.position,
                self.dim_min,
                self.dim_max
            )
            
            # Evaluate
            fitness = self.objective_func(self.position)
            n_evaluations += 1
            
            # Update best
            if fitness < self.best_fitness:
                self.best_fitness = fitness
                self.best_position = self.position.copy()
            
            # Store history
            self.history['positions'].append(self.position.copy())
            self.history['fitness'].append(fitness)
            
            # Check convergence
            if np.linalg.norm(gradient) < self.tolerance:
                if self.verbose:
                    print(f"Converged at iteration {iteration + 1}")
                break
            
            if self.verbose and (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{self.max_iter}, "
                      f"Fitness: {fitness:.6f}")
        
        return {
            'best_position': self.best_position,
            'best_fitness': self.best_fitness,
            'history': self.history,
            'n_iterations': iteration + 1,
            'n_evaluations': n_evaluations
        }

