"""
Particle Swarm Optimization (PSO) Implementation
"""

import numpy as np
from typing import Callable, Tuple, List, Optional
import json


class ParticleSwarmOptimization:
    """
    Particle Swarm Optimization algorithm implementation.
    
    PSO is a population-based metaheuristic optimization algorithm inspired
    by the social behavior of bird flocking or fish schooling.
    """
    
    def __init__(
        self,
        objective_func: Callable,
        bounds: np.ndarray,
        n_particles: int = 30,
        w: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5,
        max_iter: int = 100,
        verbose: bool = True
    ):
        """
        Initialize PSO algorithm.
        
        Parameters:
        -----------
        objective_func : Callable
            Objective function to minimize
        bounds : np.ndarray
            Bounds for each dimension, shape (n_dim, 2)
        n_particles : int
            Number of particles in the swarm
        w : float
            Inertia weight
        c1 : float
            Cognitive coefficient (personal best influence)
        c2 : float
            Social coefficient (global best influence)
        max_iter : int
            Maximum number of iterations
        verbose : bool
            Print progress information
        """
        self.objective_func = objective_func
        self.bounds = np.array(bounds)
        self.n_particles = n_particles
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.max_iter = max_iter
        self.verbose = verbose
        
        self.n_dim = len(bounds)
        self.dim_min = self.bounds[:, 0]
        self.dim_max = self.bounds[:, 1]
        
        # Initialize swarm
        self.positions = None
        self.velocities = None
        self.personal_best_positions = None
        self.personal_best_fitness = None
        self.global_best_position = None
        self.global_best_fitness = float('inf')
        
        # History tracking
        self.history = {
            'positions': [],
            'velocities': [],
            'fitness': [],
            'global_best_fitness': [],
            'global_best_position': []
        }
        
    def _initialize_swarm(self):
        """Initialize particle positions and velocities."""
        # Initialize positions randomly within bounds
        self.positions = np.random.uniform(
            self.dim_min,
            self.dim_max,
            (self.n_particles, self.n_dim)
        )
        
        # Initialize velocities (typically set to 10-20% of search space)
        velocity_range = (self.dim_max - self.dim_min) * 0.1
        self.velocities = np.random.uniform(
            -velocity_range,
            velocity_range,
            (self.n_particles, self.n_dim)
        )
        
        # Initialize personal bests
        self.personal_best_positions = self.positions.copy()
        self.personal_best_fitness = np.array([
            self.objective_func(pos) for pos in self.positions
        ])
        
        # Initialize global best
        best_idx = np.argmin(self.personal_best_fitness)
        self.global_best_position = self.personal_best_positions[best_idx].copy()
        self.global_best_fitness = self.personal_best_fitness[best_idx]
        
    def _update_velocity(self):
        """Update particle velocities."""
        r1 = np.random.rand(self.n_particles, self.n_dim)
        r2 = np.random.rand(self.n_particles, self.n_dim)
        
        cognitive = self.c1 * r1 * (self.personal_best_positions - self.positions)
        social = self.c2 * r2 * (self.global_best_position - self.positions)
        
        self.velocities = (
            self.w * self.velocities + cognitive + social
        )
        
    def _update_position(self):
        """Update particle positions."""
        self.positions += self.velocities
        
        # Apply bounds
        self.positions = np.clip(
            self.positions,
            self.dim_min,
            self.dim_max
        )
        
    def _update_bests(self):
        """Update personal and global bests."""
        # Evaluate current positions
        current_fitness = np.array([
            self.objective_func(pos) for pos in self.positions
        ])
        
        # Update personal bests
        improved = current_fitness < self.personal_best_fitness
        self.personal_best_positions[improved] = self.positions[improved].copy()
        self.personal_best_fitness[improved] = current_fitness[improved]
        
        # Update global best
        best_idx = np.argmin(self.personal_best_fitness)
        if self.personal_best_fitness[best_idx] < self.global_best_fitness:
            self.global_best_position = self.personal_best_positions[best_idx].copy()
            self.global_best_fitness = self.personal_best_fitness[best_idx]
    
    def optimize(self, callback: Optional[Callable] = None) -> dict:
        """
        Run PSO optimization.
        
        Parameters:
        -----------
        callback : Optional[Callable]
            Callback function called after each iteration with current state
        
        Returns:
        --------
        dict : Optimization results
        """
        self._initialize_swarm()
        
        for iteration in range(self.max_iter):
            # Update velocity and position
            self._update_velocity()
            self._update_position()
            self._update_bests()
            
            # Store history
            self.history['positions'].append(self.positions.copy())
            self.history['velocities'].append(self.velocities.copy())
            self.history['fitness'].append(self.personal_best_fitness.copy())
            self.history['global_best_fitness'].append(self.global_best_fitness)
            self.history['global_best_position'].append(self.global_best_position.copy())
            
            # Callback for visualization
            if callback:
                callback(iteration, self)
            
            if self.verbose and (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{self.max_iter}, "
                      f"Best fitness: {self.global_best_fitness:.6f}")
        
        return {
            'best_position': self.global_best_position,
            'best_fitness': self.global_best_fitness,
            'history': self.history,
            'n_iterations': self.max_iter,
            'n_evaluations': self.max_iter * self.n_particles
        }
    
    def get_current_state(self) -> dict:
        """Get current state of the swarm."""
        return {
            'positions': self.positions.copy(),
            'velocities': self.velocities.copy(),
            'personal_best_positions': self.personal_best_positions.copy(),
            'personal_best_fitness': self.personal_best_fitness.copy(),
            'global_best_position': self.global_best_position.copy(),
            'global_best_fitness': self.global_best_fitness
        }

