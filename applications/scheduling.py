"""
Job Scheduling Optimization using PSO
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from algorithms.particle_swarm_optimization import ParticleSwarmOptimization
from typing import List, Tuple


class JobScheduler:
    """
    Job scheduling optimization using PSO.
    
    Example: Scheduling jobs on machines to minimize makespan.
    """
    
    def __init__(
        self,
        jobs: List[dict],
        machines: int,
        job_processing_times: List[float]
    ):
        """
        Initialize job scheduler.
        
        Parameters:
        -----------
        jobs : List[dict]
            List of jobs with properties
        machines : int
            Number of available machines
        job_processing_times : List[float]
            Processing time for each job
        """
        self.jobs = jobs
        self.machines = machines
        self.job_processing_times = np.array(job_processing_times)
        self.n_jobs = len(jobs)
        
    def _decode_solution(self, position: np.ndarray) -> List[int]:
        """
        Decode continuous position to discrete machine assignments.
        
        Parameters:
        -----------
        position : np.ndarray
            Continuous position vector
        
        Returns:
        --------
        List[int] : Machine assignments for each job
        """
        # Map continuous values to machine indices
        assignments = (position * (self.machines - 1)).astype(int)
        assignments = np.clip(assignments, 0, self.machines - 1)
        return assignments.tolist()
    
    def _calculate_makespan(self, assignments: List[int]) -> float:
        """
        Calculate makespan (total completion time) for given assignments.
        
        Parameters:
        -----------
        assignments : List[int]
            Machine assignment for each job
        
        Returns:
        --------
        float : Makespan
        """
        machine_times = np.zeros(self.machines)
        
        for job_idx, machine in enumerate(assignments):
            machine_times[machine] += self.job_processing_times[job_idx]
        
        return np.max(machine_times)  # Makespan is max machine time
    
    def _objective(self, position: np.ndarray) -> float:
        """
        Objective function: makespan to minimize.
        
        Parameters:
        -----------
        position : np.ndarray
            Continuous position vector
        
        Returns:
        --------
        float : Makespan
        """
        assignments = self._decode_solution(position)
        return self._calculate_makespan(assignments)
    
    def optimize(
        self,
        n_particles: int = 30,
        max_iter: int = 100,
        **pso_kwargs
    ) -> dict:
        """
        Optimize job schedule using PSO.
        
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
        dict : Optimization results with schedule
        """
        # Bounds: one dimension per job, values in [0, 1]
        bounds = np.array([[0.0, 1.0] for _ in range(self.n_jobs)])
        
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
        best_assignments = self._decode_solution(results['best_position'])
        
        results['schedule'] = {
            job_idx: {
                'job': self.jobs[job_idx],
                'machine': machine,
                'processing_time': self.job_processing_times[job_idx]
            }
            for job_idx, machine in enumerate(best_assignments)
        }
        results['makespan'] = results['best_fitness']
        
        return results


# Example usage
def example_job_scheduling():
    """
    Example: Scheduling 10 jobs on 3 machines.
    """
    # Create jobs
    jobs = [{'id': i, 'priority': np.random.randint(1, 5)} for i in range(10)]
    processing_times = np.random.uniform(1, 10, 10)
    
    # Create scheduler
    scheduler = JobScheduler(
        jobs=jobs,
        machines=3,
        job_processing_times=processing_times
    )
    
    # Optimize
    results = scheduler.optimize(n_particles=30, max_iter=50)
    
    print("Scheduling Results:")
    print(f"Makespan: {results['makespan']:.2f}")
    print("\nJob Assignments:")
    for job_idx, schedule in results['schedule'].items():
        print(f"Job {job_idx}: Machine {schedule['machine']}, "
              f"Time: {schedule['processing_time']:.2f}")
    
    return results


if __name__ == "__main__":
    example_job_scheduling()

