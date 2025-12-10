# Particle Swarm Optimization (PSO) - Lecture Materials

Comprehensive lecture package on Particle Swarm Optimization with interactive visualizations, Python implementations, and real-world applications. Includes comparison with traditional optimization methods.

## Overview

This package provides a complete educational resource for understanding Particle Swarm Optimization, including:

- **Complete PSO Implementation**: Standard PSO with configurable parameters
- **Interactive Visualizations**: Real-time particle movement, convergence plots, and comparison tools
- **Benchmark Functions**: Multiple optimization test functions (Sphere, Rastrigin, Ackley, Rosenbrock, Griewank)
- **Traditional Methods Comparison**: Gradient Descent, Random Search, Hill Climbing
- **Real-World Applications**: Hyperparameter tuning, job scheduling, resource allocation

## Installation

1. Clone the repository:
```bash
git clone https://github.com/aidarmen/Metaheuristic-Optimization-Algorithms.git
cd Metaheuristic-Optimization-Algorithms
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Install Node.js dependencies for the presentation:
```bash
cd presentation
npm install
cd ..
```

## Project Structure

```
├── algorithms/
│   ├── __init__.py
│   ├── particle_swarm_optimization.py    # Standard PSO implementation
│   ├── adaptive_pso.py                  # Adaptive PSO implementation
│   ├── gradient_descent.py              # Gradient descent for comparison
│   ├── random_search.py                  # Random search baseline
│   └── hill_climbing.py                  # Hill climbing algorithm
├── visualization/
│   ├── __init__.py
│   └── interactive_pso.py                # Interactive visualization tools
├── problems/
│   ├── __init__.py
│   └── function_optimization.py          # Benchmark functions
├── applications/
│   ├── __init__.py
│   ├── parameter_tuning.py               # Hyperparameter optimization
│   ├── scheduling.py                     # Job scheduling
│   └── resource_allocation.py            # Resource allocation
├── examples/
│   ├── interactive_demo.py               # Main interactive demonstration
│   ├── pso_vs_traditional.py             # PSO vs traditional methods
│   ├── adaptive_pso_comparison.py        # Adaptive PSO comparison
│   ├── benchmark_comparison.py            # Compare across functions
│   ├── parameter_sensitivity.py          # Parameter analysis
│   ├── non_symmetrical_demo.py           # Non-symmetrical functions demo
│   └── non_symmetrical_comparison.py     # Non-symmetrical comparison
├── presentation/
│   ├── slides.md                         # Slidev presentation
│   ├── slides.config.ts                  # Slidev configuration
│   ├── package.json                      # Node.js dependencies
│   └── README_PRESENTATION.md            # Presentation instructions
├── media/
│   ├── *.png                             # Convergence plots, comparisons
│   ├── *.gif                             # PSO animations
│   └── *.jpg                             # Formula diagrams
├── requirements.txt
├── convert_to_pptx.py                    # Convert slides to PowerPoint
└── README.md
```

## Quick Start

### Basic PSO Usage

```python
import numpy as np
from algorithms.particle_swarm_optimization import ParticleSwarmOptimization
from problems.function_optimization import rastrigin_function, FUNCTION_INFO

# Get function and bounds
func_info = FUNCTION_INFO['rastrigin']
objective_func = func_info['function']
bounds = func_info['bounds']

# Create PSO optimizer
pso = ParticleSwarmOptimization(
    objective_func=objective_func,
    bounds=bounds,
    n_particles=30,
    w=0.7,          # Inertia weight
    c1=1.5,         # Cognitive coefficient
    c2=1.5,         # Social coefficient
    max_iter=50,
    verbose=True
)

# Run optimization
results = pso.optimize()

print(f"Best Position: {results['best_position']}")
print(f"Best Fitness: {results['best_fitness']}")
```

### Interactive Visualization

```python
from examples.interactive_demo import run_interactive_demo

# Run interactive demo with visualization
run_interactive_demo('rastrigin')
```

### Compare PSO with Traditional Methods

```python
from examples.pso_vs_traditional import compare_methods

# Compare PSO vs Gradient Descent, Random Search, Hill Climbing
results = compare_methods('rastrigin', max_iter=50)
```

## Algorithm Explanation

### Particle Swarm Optimization (PSO)

PSO is a population-based metaheuristic optimization algorithm inspired by the social behavior of bird flocking or fish schooling. Each particle in the swarm represents a potential solution and moves through the search space based on:

1. **Inertia**: Current velocity
2. **Cognitive Component**: Attraction to particle's personal best position
3. **Social Component**: Attraction to swarm's global best position

**Update Equations:**
```
v(t+1) = w * v(t) + c1 * r1 * (pbest - x(t)) + c2 * r2 * (gbest - x(t))
x(t+1) = x(t) + v(t+1)
```

Where:
- `w`: Inertia weight
- `c1, c2`: Cognitive and social coefficients
- `r1, r2`: Random numbers in [0, 1]
- `pbest`: Personal best position
- `gbest`: Global best position

### Advantages of PSO

- **No gradient required**: Works on non-differentiable functions
- **Global search**: Can escape local minima
- **Simple implementation**: Easy to understand and code
- **Few parameters**: Only requires tuning of w, c1, c2
- **Parallelizable**: Particles can be evaluated independently

## Benchmark Functions

The package includes several standard benchmark functions:

1. **Sphere Function**: Unimodal, convex - easy to optimize
2. **Rastrigin Function**: Highly multimodal - many local minima
3. **Ackley Function**: Multimodal with many local minima
4. **Rosenbrock Function**: Valley-shaped, hard to optimize
5. **Griewank Function**: Multimodal with many local minima

## Examples

### 1. Interactive Demo
```bash
python examples/interactive_demo.py
```
Shows PSO in action with animated particle movement and convergence plots.

### 2. PSO vs Traditional Methods
```bash
python examples/pso_vs_traditional.py
```
Compares PSO performance against Gradient Descent, Random Search, and Hill Climbing.

### 3. Benchmark Comparison
```bash
python examples/benchmark_comparison.py
```
Tests PSO performance across different benchmark functions.

### 4. Parameter Sensitivity Analysis
```bash
python examples/parameter_sensitivity.py
```
Analyzes how PSO parameters affect performance.

### 5. Non-Symmetrical Functions Demo
```bash
python examples/non_symmetrical_demo.py
```
Demonstrates PSO on non-symmetrical optimization functions.

### 6. PSO vs Traditional Methods (Non-Symmetrical)
```bash
python examples/non_symmetrical_comparison.py
```
Compares PSO with traditional methods on non-symmetrical functions.

### 7. Adaptive PSO Comparison
```bash
python examples/adaptive_pso_comparison.py
```
Compares Adaptive PSO with Standard PSO and Gradient Descent.

## Real-World Applications

### Hyperparameter Tuning
```python
from applications.parameter_tuning import HyperparameterOptimizer

# Optimize learning rate and regularization
optimizer = HyperparameterOptimizer(
    model_train_func=train_model,
    param_bounds={
        'learning_rate': (0.001, 0.1),
        'regularization': (0.0001, 0.01)
    },
    validation_func=validate_model
)

results = optimizer.optimize()
```

### Job Scheduling
```python
from applications.scheduling import JobScheduler

# Schedule jobs on machines
scheduler = JobScheduler(
    jobs=jobs,
    machines=3,
    job_processing_times=processing_times
)

results = scheduler.optimize()
```

### Resource Allocation
```python
from applications.resource_allocation import ResourceAllocator

# Allocate budget to projects
allocator = ResourceAllocator(
    projects=projects,
    total_budget=1000.0,
    project_utilities=utility_functions
)

results = allocator.optimize()
```

## Visualization Features

The interactive visualization module provides:

- **Real-time Animation**: Watch particles move through the search space
- **Convergence Plots**: Track fitness improvement over iterations
- **Comparison Visualizations**: Side-by-side comparison of different methods
- **3D Surface Plots**: Visualize the objective function landscape
- **Interactive Controls**: Adjust parameters and view different perspectives

## Comparison with Traditional Methods

### When to Use PSO vs Traditional Methods

**Use PSO when:**
- Function is non-differentiable or noisy
- Multiple local minima exist
- Gradient information is unavailable
- Parallel evaluation is possible
- **Non-symmetrical search spaces** (PSO excels here)

**Use Gradient Descent when:**
- Function is smooth and differentiable
- Gradient can be computed efficiently
- Local minimum is sufficient
- Fast convergence is needed
- Search space is symmetrical

**Use Random Search when:**
- Simple baseline is needed
- No structure in the problem
- Very limited computational budget

### Non-Symmetrical Functions

PSO particularly excels on non-symmetrical optimization problems where:
- The global minimum is not at the origin
- The landscape is rotated or shifted
- Traditional gradient-based methods struggle
- The search space has deceptive local minima

Available non-symmetrical functions:
- **Beale**: Valley-shaped, non-symmetrical
- **Goldstein-Price**: Multimodal, non-symmetrical
- **Three-Hump Camel**: Three local minima
- **Rotated Ellipsoid**: Rotated search space
- **Shifted Sphere**: Shifted from origin
- **Schwefel**: Highly deceptive
- **Easom**: Flat with sharp minimum

## Parameter Tuning Guide

### Inertia Weight (w)
- **High (0.9-1.0)**: More exploration, slower convergence
- **Low (0.4-0.6)**: More exploitation, faster convergence
- **Adaptive**: Start high, decrease over time

### Cognitive Coefficient (c1)
- Controls attraction to personal best
- Typical range: 1.0-2.0
- Higher values: More exploration

### Social Coefficient (c2)
- Controls attraction to global best
- Typical range: 1.0-2.0
- Higher values: Faster convergence

### Swarm Size
- **Small (10-20)**: Faster, less exploration
- **Medium (30-50)**: Good balance
- **Large (100+)**: Better exploration, slower

## Performance Tips

1. **Start with default parameters**: w=0.7, c1=1.5, c2=1.5
2. **Use appropriate swarm size**: 20-50 particles for most problems
3. **Normalize search space**: Scale bounds to similar ranges
4. **Use early stopping**: Stop if no improvement for N iterations
5. **Run multiple times**: PSO is stochastic, average over runs

## References

1. Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization. Proceedings of IEEE international conference on neural networks.
2. Shi, Y., & Eberhart, R. (1998). A modified particle swarm optimizer. IEEE international conference on evolutionary computation.

## License

This educational material is provided for learning and teaching purposes.

## Contributing

Feel free to extend this package with:
- Additional benchmark functions
- More optimization algorithms
- Enhanced visualizations
- New application examples

## Presentation

A comprehensive Slidev presentation is included in the `presentation/` folder.

### Running the Presentation

1. Navigate to the presentation directory:
```bash
cd presentation
```

2. Install dependencies (if not already done):
```bash
npm install
```

3. Start the development server:
```bash
npm run dev
```

The presentation will open in your browser at `http://localhost:3030`

### Exporting the Presentation

**Export to PDF:**
```bash
cd presentation
npm run export
```

**Export to PowerPoint:**
```bash
cd presentation
npm run export -- --format png --output slides-export
# Then use PowerPoint or other tools to create PPTX from PNG images
```

For more details, see [presentation/README_PRESENTATION.md](presentation/README_PRESENTATION.md)

## Repository

GitHub: https://github.com/aidarmen/Metaheuristic-Optimization-Algorithms

## Contact

For questions or suggestions, please open an issue or contact the maintainer.

