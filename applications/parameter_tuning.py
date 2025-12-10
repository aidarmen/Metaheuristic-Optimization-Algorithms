"""
Hyperparameter Tuning Application using PSO
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from algorithms.particle_swarm_optimization import ParticleSwarmOptimization
from typing import Dict, Callable


class HyperparameterOptimizer:
    """
    Hyperparameter optimization using PSO.
    
    Example: Optimizing learning rate and regularization for a model.
    """
    
    def __init__(
        self,
        model_train_func: Callable,
        param_bounds: Dict[str, tuple],
        validation_func: Callable
    ):
        """
        Initialize hyperparameter optimizer.
        
        Parameters:
        -----------
        model_train_func : Callable
            Function that trains a model given hyperparameters
        param_bounds : Dict[str, tuple]
            Dictionary mapping parameter names to (min, max) bounds
        validation_func : Callable
            Function that evaluates model performance (returns score to maximize)
        """
        self.model_train_func = model_train_func
        self.param_bounds = param_bounds
        self.validation_func = validation_func
        self.param_names = list(param_bounds.keys())
        
    def _objective(self, params: np.ndarray) -> float:
        """
        Objective function: negative validation score (for minimization).
        
        Parameters:
        -----------
        params : np.ndarray
            Parameter values
        
        Returns:
        --------
        float : Negative validation score
        """
        # Convert array to dictionary
        param_dict = {
            name: params[i] 
            for i, name in enumerate(self.param_names)
        }
        
        # Train model
        model = self.model_train_func(param_dict)
        
        # Evaluate
        score = self.validation_func(model)
        
        # Return negative (for minimization)
        return -score
    
    def optimize(
        self,
        n_particles: int = 20,
        max_iter: int = 50,
        **pso_kwargs
    ) -> Dict:
        """
        Optimize hyperparameters using PSO.
        
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
        Dict : Optimization results
        """
        # Create bounds array
        bounds = np.array([self.param_bounds[name] for name in self.param_names])
        
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
        
        # Convert best position back to dictionary
        best_params = {
            name: results['best_position'][i]
            for i, name in enumerate(self.param_names)
        }
        
        results['best_params'] = best_params
        results['best_score'] = -results['best_fitness']  # Convert back to positive
        
        return results


# Example usage
def example_hyperparameter_tuning():
    """
    Example: Optimizing learning rate and regularization for a simple model.
    """
    # Mock model training function
    def train_model(params):
        """Mock model training."""
        class MockModel:
            def __init__(self, lr, reg):
                self.lr = lr
                self.reg = reg
                # Simulate training
                self.accuracy = 0.9 - abs(lr - 0.01) * 10 - abs(reg - 0.001) * 100
        return MockModel(params['learning_rate'], params['regularization'])
    
    # Mock validation function
    def validate_model(model):
        """Mock validation."""
        return max(0, model.accuracy)  # Ensure non-negative
    
    # Define parameter bounds
    param_bounds = {
        'learning_rate': (0.001, 0.1),
        'regularization': (0.0001, 0.01)
    }
    
    # Create optimizer
    optimizer = HyperparameterOptimizer(
        model_train_func=train_model,
        param_bounds=param_bounds,
        validation_func=validate_model
    )
    
    # Optimize
    results = optimizer.optimize(n_particles=20, max_iter=30)
    
    print("Optimization Results:")
    print(f"Best Parameters: {results['best_params']}")
    print(f"Best Score: {results['best_score']:.4f}")
    
    return results


if __name__ == "__main__":
    example_hyperparameter_tuning()

