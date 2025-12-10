---
theme: default
background: https://source.unsplash.com/1920x1080/?swarm,particles
class: text-center
highlighter: shiki
lineNumbers: false
info: |
  ## Particle Swarm Optimization (PSO)
  
  A comprehensive presentation on PSO algorithm, its applications, and comparisons with traditional optimization methods.
drawings:
  persist: false
transition: slide-left
title: Particle Swarm Optimization
mdc: true
---

# Particle Swarm Optimization
## Metaheuristic Optimization Algorithms and Their Real-World Applications

<div class="pt-12">
  <span @click="$slidev.nav.next" class="px-2 py-1 rounded cursor-pointer" hover="bg-white bg-opacity-10">
    Press Space for next page <carbon:arrow-right class="inline"/>
  </span>
</div>

---
layout: center
class: text-center
---

# What is PSO?

<div class="text-lg mt-8 space-y-4">

**Particle Swarm Optimization** is a search algorithm

Inspired by how **birds flock** or **fish school**

Each particle = one possible solution

Particles work together to find the best solution

</div>

---
layout: center
class: text-center
---

# Key Idea

<div class="text-lg mt-8 space-y-6">

**Swarm Intelligence**

Many particles search together

They share information about good solutions

This helps them find the best answer faster

</div>

---
layout: center
class: text-center
---

# Main Advantages

<div class="grid grid-cols-2 gap-6 mt-8 text-left">

<div>

✅ No gradient needed  
✅ Works on any function  
✅ Finds global minimum  
✅ Escapes local minima  
✅ Simple to use  

</div>

<div>

✅ Few parameters  
✅ Fast convergence  
✅ Parallel processing  
✅ Robust results  
✅ Easy to code  

</div>

</div>

---
layout: default
---

# The PSO Formula

<div class="text-center mt-8">

## Velocity Update

$$v_{t+1} = w \cdot v_t + c_1 r_1 (p_{best} - x_t) + c_2 r_2 (g_{best} - x_t)$$

## Position Update

$$x_{t+1} = x_t + v_{t+1}$$

</div>

<div class="text-sm mt-6 opacity-70 text-center">
Works for any number of dimensions (1D, 2D, 3D, ...)
</div>

---
layout: center
class: text-center
---

# Multi-Dimensional Formula

<div class="text-lg mt-8 space-y-4">

## Works for Any Dimension

**The formula is the same for:**
- 1D: $x = [x_1]$
- 2D: $x = [x_1, x_2]$
- 3D: $x = [x_1, x_2, x_3]$
- N-D: $x = [x_1, x_2, ..., x_n]$

**All operations are vector operations:**
- $v_t$, $x_t$, $p_{best}$, $g_{best}$ are all vectors
- Each dimension updated independently
- Same formula applies to all dimensions

</div>

---
layout: center
class: text-center
---

# Part 1: Inertia

<div class="text-lg mt-8 space-y-4">

## $w \cdot v_t$

**What it does:**
- Keeps particle moving in same direction
- Like momentum in physics

**Inertia Weight ($w$):**
- High $w$ (0.9) = keeps moving = explores more
- Low $w$ (0.4) = stops quickly = converges fast

</div>

---
layout: center
class: text-center
---

# Part 2: Cognitive

<div class="text-lg mt-8 space-y-4">

## $c_1 r_1 (p_{best} - x_t)$

**What it does:**
- Pulls particle toward its **own best position**
- Each particle remembers where it did best

**Cognitive Coefficient ($c_1$):**
- Higher = stronger pull to personal best
- Typical value: 1.5 to 2.0

</div>

---
layout: center
class: text-center
---

# Part 3: Social

<div class="text-lg mt-8 space-y-4">

## $c_2 r_2 (g_{best} - x_t)$

**What it does:**
- Pulls particle toward **swarm's best position**
- All particles know the best solution found

**Social Coefficient ($c_2$):**
- Higher = faster convergence
- Typical value: 1.5 to 2.0

</div>

---
layout: default
---

# How They Work Together

<div class="grid grid-cols-2 gap-4">

<div class="text-lg space-y-4">

**New velocity =**

Keep going (Inertia)  
+  
Go to my best (Cognitive)  
+  
Go to swarm's best (Social)

**Result:** Particle moves toward better solutions

</div>

<div class="flex items-center justify-center">
  <img src="pso Formula.jpg" class="max-w-full max-h-[70vh] rounded-lg shadow-lg" />
</div>

</div>

---
layout: default
---

# Step-by-Step Algorithm

<div class="text-sm space-y-3 mt-4">

## Step 1: Initialize

- Create N particles
- Random positions
- Random velocities
- Find initial best

## Step 2: For Each Particle

- Update velocity using formula
- Update position
- Evaluate fitness

</div>

---
layout: default
---

# Step-by-Step Algorithm (continued)

<div class="text-sm space-y-3 mt-4">

## Step 3: Update Best Positions

- If current is better than personal best:
  - Update personal best
- If current is better than global best:
  - Update global best

## Step 4: Repeat

- Go back to Step 2
- Continue until convergence
- Or maximum iterations reached

</div>

---
layout: two-cols
---

# PSO Parameters

<div class="text-sm space-y-4">

## Inertia Weight ($w$)

- Controls exploration
- High (0.9): Explore more
- Low (0.4): Converge faster
- Typical: 0.7

## Cognitive ($c_1$)

- Personal best attraction
- Typical: 1.5 - 2.0

</div>

::right::

<div class="text-sm space-y-4">

## Social ($c_2$)

- Global best attraction
- Typical: 1.5 - 2.0

## Swarm Size

- Number of particles
- Small (10-20): Fast
- Medium (30-50): Balanced
- Large (100+): Better search

</div>

---
layout: center
class: text-center
---

<div class="text-lg mb-2 font-semibold">PSO in Action: Sphere Function</div>

<div class="flex justify-center items-center mt-2">
  <img src="sphere_pso_animation.gif" class="w-4/5 max-h-[70vh] rounded-lg shadow-lg" />
</div>

<div class="mt-2 text-xs opacity-70">
  Particles converge to global minimum at (0, 0)
</div>

---
layout: center
class: text-center
---

<div class="text-lg mb-2 font-semibold">PSO on Rastrigin Function</div>

<div class="flex justify-center items-center mt-2">
  <img src="rastrigin_pso_animation.gif" class="w-4/5 max-h-[70vh] rounded-lg shadow-lg" />
</div>

<div class="mt-2 text-xs opacity-70">
  Many local minima - PSO navigates through them
</div>

---
layout: center
class: text-center
---

<div class="text-lg mb-2 font-semibold">PSO on Ackley Function</div>

<div class="flex justify-center items-center mt-2">
  <img src="ackley_pso_animation.gif" class="w-4/5 max-h-[70vh] rounded-lg shadow-lg" />
</div>

<div class="mt-2 text-xs opacity-70">
  Multimodal function - PSO finds global minimum
</div>

---
layout: two-cols
---

# Convergence Analysis

<div>
  <img src="sphere_convergence.png" class="rounded shadow-lg max-h-48" />
  <div class="text-xs mt-1 opacity-70">Sphere</div>
</div>

<div>
  <img src="rastrigin_convergence.png" class="rounded shadow-lg max-h-48" />
  <div class="text-xs mt-1 opacity-70">Rastrigin</div>
</div>

::right::

<div>
  <img src="ackley_convergence.png" class="rounded shadow-lg max-h-64" />
  <div class="text-xs mt-1 opacity-70">Ackley</div>
</div>

<div class="mt-4 text-sm">

## Observations

- Fast on simple functions
- Steady on complex functions
- Finds global minimum
- Robust performance

</div>

---
layout: center
class: text-center
---

<div class="text-lg mb-2 font-semibold">Non-Symmetrical: Beale Function</div>

<div class="flex justify-center items-center mt-2">
  <img src="beale_pso_animation.gif" class="w-4/5 max-h-[70vh] rounded-lg shadow-lg" />
</div>

<div class="mt-2 text-xs opacity-70">
  Valley-shaped, non-symmetrical - PSO finds minimum at (3, 0.5)
</div>

---
layout: center
class: text-center
---

<div class="text-lg mb-2 font-semibold">Non-Symmetrical: Goldstein-Price</div>

<div class="flex justify-center items-center mt-2">
  <img src="goldstein_price_pso_animation.gif" class="w-4/5 max-h-[70vh] rounded-lg shadow-lg" />
</div>

<div class="mt-2 text-xs opacity-70">
  Multiple local minima - PSO explores and finds best
</div>

---
layout: center
class: text-center
---

<div class="text-lg mb-2 font-semibold">Non-Symmetrical: Rotated Ellipsoid</div>

<div class="flex justify-center items-center mt-2">
  <img src="rotated_ellipsoid_pso_animation.gif" class="w-4/5 max-h-[70vh] rounded-lg shadow-lg" />
</div>

<div class="mt-2 text-xs opacity-70">
  Rotated search space - PSO adapts to non-symmetrical landscape
</div>

---
layout: two-cols
---

# Convergence: Non-Symmetrical

<div>
  <img src="beale_convergence.png" class="rounded shadow-lg max-h-40" />
  <div class="text-xs mt-1 opacity-70">Beale</div>
</div>

<div>
  <img src="goldstein_price_convergence.png" class="rounded shadow-lg max-h-40" />
  <div class="text-xs mt-1 opacity-70">Goldstein-Price</div>
</div>

::right::

<div>
  <img src="rotated_ellipsoid_convergence.png" class="rounded shadow-lg max-h-48" />
  <div class="text-xs mt-1 opacity-70">Rotated Ellipsoid</div>
</div>

<div class="mt-3 text-sm">

## PSO Advantages

✅ No gradient needed  
✅ Explores entire space  
✅ Handles local minima  
✅ Single run works  

</div>

---
layout: default
---

# PSO vs Traditional Methods

<div class="grid grid-cols-2 gap-4">

<div>
  <img src="beale_comparison.png" class="rounded shadow-lg max-h-56" />
  <div class="text-xs mt-1 opacity-70">Beale Function</div>
</div>

<div>
  <img src="goldstein_price_comparison.png" class="rounded shadow-lg max-h-56" />
  <div class="text-xs mt-1 opacity-70">Goldstein-Price</div>
</div>

</div>

---
layout: default
---

# More Comparisons

<div class="grid grid-cols-2 gap-4">

<div>
  <img src="three_hump_camel_comparison.png" class="rounded shadow-lg max-h-56" />
  <div class="text-xs mt-1 opacity-70">Three-Hump Camel</div>
</div>

<div>
  <img src="rotated_ellipsoid_comparison.png" class="rounded shadow-lg max-h-56" />
  <div class="text-xs mt-1 opacity-70">Rotated Ellipsoid</div>
</div>

</div>

---
layout: default
---

# Additional Comparisons

<div class="grid grid-cols-2 gap-4">

<div>
  <img src="shifted_sphere_comparison.png" class="rounded shadow-lg max-h-64" />
  <div class="text-xs mt-1 opacity-70">Shifted Sphere</div>
</div>

<div>
  <img src="easom_comparison.png" class="rounded shadow-lg max-h-64" />
  <div class="text-xs mt-1 opacity-70">Easom Function</div>
</div>

</div>

---
layout: default
---

# When Gradient Descent is Better

<div class="text-lg space-y-4 mt-6">

✅ Smooth functions  
✅ Gradient available  
✅ Unimodal problems  
✅ Well-tuned parameters  

**But needs:**
- Multiple trials (10+)
- Careful tuning
- Gradient computation

</div>

---
layout: default
---

# When PSO is Better

<div class="text-lg space-y-4 mt-6">

✅ **Multimodal functions** - escapes local minima  
✅ **No gradient needed** - works on any function  
✅ **Non-symmetrical** - explores entire space  
✅ **Single run** - no multiple trials  
✅ **Less tuning** - more forgiving  
✅ **Parallelizable** - can use all cores  

</div>

---
layout: default
---

# Why PSO is Fast: Reason 1

<div class="text-center mt-8">

## Parallel Evaluation

**All particles evaluated at once**

- Traditional: One at a time
- PSO: All together
- **Result:** Up to N× faster
- Can use GPU acceleration

</div>

---
layout: default
---

# Why PSO is Fast: Reason 2

<div class="text-center mt-8">

## No Gradient Computation

**Only function evaluation needed**

- Gradient Descent: Must compute derivatives
- PSO: Just evaluate function
- **Result:** 2-10× faster per step
- Works on black-box functions

</div>

---
layout: default
---

# Why PSO is Fast: Reason 3

<div class="text-center mt-8">

## Efficient Search Strategy

**Swarm intelligence = smart search**

- Single point: Random walk
- Swarm: Coordinated search
- **Result:** Finds solutions faster
- Less wasted computation

</div>

---
layout: default
---

# Why PSO is Fast: Reason 4

<div class="text-center mt-8">

## Fewer Iterations Needed

**Global search converges quickly**

- Local methods: Many iterations
- PSO: Fewer iterations
- **Result:** Faster overall
- Especially on complex functions

</div>

---
layout: center
class: text-center
---

# Next Level: Adaptive PSO

<div class="text-lg mt-8 space-y-4">

**Improvement over Standard PSO**

Parameters **adapt** during optimization

**Early stage:** Explore more  
**Late stage:** Converge faster

Better balance of exploration and exploitation

</div>

---
layout: center
class: text-center
---

# Key Idea: Adaptive Parameters

<div class="text-lg mt-8 space-y-4">

**Standard PSO:** Fixed parameters

**Adaptive PSO:** Parameters change over time

- Inertia weight: **decreases** (0.9 → 0.4)
- Cognitive coefficient: **decreases** (2.5 → 0.5)
- Social coefficient: **increases** (0.5 → 2.5)

</div>

---
layout: default
---

# Adaptive PSO Formula

<div class="text-center mt-6">

## Velocity Update (Same Structure)

$$v_{t+1} = w(t) \cdot v_t + c_1(t) r_1 (p_{best} - x_t) + c_2(t) r_2 (g_{best} - x_t)$$

</div>

<div class="text-center mt-4">

## Position Update

$$x_{t+1} = x_t + v_{t+1}$$

</div>

<div class="text-sm mt-4 opacity-70 text-center">
Parameters w(t), c₁(t), c₂(t) change at each iteration t
</div>

---
layout: center
class: text-center
---

# Parameter Adaptation Formulas

<div class="text-lg mt-8 space-y-6">

## Inertia Weight

$$w(t) = w_{max} - (w_{max} - w_{min}) \cdot \frac{t}{T}$$

**Decreases** from $w_{max}$ to $w_{min}$

## Cognitive Coefficient

$$c_1(t) = c_{1,max} - (c_{1,max} - c_{1,min}) \cdot \frac{t}{T}$$

**Decreases** from $c_{1,max}$ to $c_{1,min}$

</div>

---
layout: center
class: text-center
---

# Parameter Adaptation Formulas (continued)

<div class="text-lg mt-8 space-y-6">

## Social Coefficient

$$c_2(t) = c_{2,min} + (c_{2,max} - c_{2,min}) \cdot \frac{t}{T}$$

**Increases** from $c_{2,min}$ to $c_{2,max}$

## Where

- $t$ = current iteration
- $T$ = total iterations
- $\frac{t}{T}$ = progress (0 to 1)

</div>

---
layout: default
---

# Example: Parameter Values

<div class="text-sm space-y-4 mt-4">

## At Iteration 0 (Start)

- $w(0) = 0.9$ (high exploration)
- $c_1(0) = 2.5$ (strong personal best)
- $c_2(0) = 0.5$ (weak global best)

## At Iteration 50 (Middle)

- $w(50) = 0.65$ (moderate)
- $c_1(50) = 1.5$ (balanced)
- $c_2(50) = 1.5$ (balanced)

## At Iteration 100 (End)

- $w(100) = 0.4$ (low, fast convergence)
- $c_1(100) = 0.5$ (weak personal best)
- $c_2(100) = 2.5$ (strong global best)

</div>

---
layout: default
---

# How Parameters Adapt

<div class="grid grid-cols-2 gap-6 mt-6">

<div>

## Early Stage (Exploration)

- High inertia (0.9) → explore more
- High cognitive (2.5) → personal best
- Low social (0.5) → less convergence

**Goal:** Find good regions

</div>

<div>

## Late Stage (Exploitation)

- Low inertia (0.4) → converge faster
- Low cognitive (0.5) → less personal focus
- High social (2.5) → strong convergence

**Goal:** Refine solution

</div>

</div>

---
layout: center
class: text-center
---

# Parameter Adaptation Visualization

<div class="flex justify-center items-center">
  <img src="rastrigin_adaptive_parameters.png" class="w-4/5 max-h-[70vh] rounded-lg shadow-lg" />
</div>

<div class="mt-2 text-xs opacity-70">
  Parameters change smoothly over iterations
</div>

---
layout: center
class: text-center
---

# Adaptive PSO in Action

<div class="flex justify-center items-center">
  <img src="rastrigin_adaptive_pso_animation.gif" class="w-4/5 max-h-[70vh] rounded-lg shadow-lg" />
</div>

<div class="mt-2 text-xs opacity-70">
  Adaptive PSO on Rastrigin function
</div>

---
layout: default
---

# Convergence Comparison

<div class="flex justify-center items-center">
  <img src="rastrigin_adaptive_comparison.png" class="w-4/5 max-h-[70vh] rounded-lg shadow-lg" />
</div>

<div class="mt-2 text-xs opacity-70 text-center">
  Adaptive PSO converges faster and finds better solutions
</div>

---
layout: default
---

# More Comparisons

<div class="grid grid-cols-2 gap-4">

<div>
  <img src="sphere_adaptive_comparison.png" class="rounded shadow-lg max-h-64" />
  <div class="text-xs mt-1 opacity-70 text-center">Sphere Function</div>
</div>

<div>
  <img src="ackley_adaptive_comparison.png" class="rounded shadow-lg max-h-64" />
  <div class="text-xs mt-1 opacity-70 text-center">Ackley Function</div>
</div>

</div>

---
layout: default
---

# Performance Metrics

<div class="grid grid-cols-2 gap-4">

<div>
  <img src="rastrigin_adaptive_metrics.png" class="rounded shadow-lg max-h-64" />
  <div class="text-xs mt-1 opacity-70 text-center">Rastrigin</div>
</div>

<div>
  <img src="ackley_adaptive_metrics.png" class="rounded shadow-lg max-h-64" />
  <div class="text-xs mt-1 opacity-70 text-center">Ackley</div>
</div>

</div>

---
layout: center
class: text-center
---

# Adaptive PSO Advantages

<div class="grid grid-cols-2 gap-8 mt-8 text-left text-lg">

<div>

✅ Better convergence  
✅ Fewer iterations  
✅ Better solutions  
✅ Self-tuning  

</div>

<div>

✅ Works on complex functions  
✅ No manual tuning  
✅ Automatic adaptation  
✅ Production ready  

</div>

</div>

---
layout: default
---

# Real-World Applications

<div class="grid grid-cols-3 gap-4 mt-6">

<div class="border rounded p-4 text-sm">

## Hyperparameter Tuning

- Machine learning models
- Neural network parameters
- No gradient needed
- Works with any model

</div>

<div class="border rounded p-4 text-sm">

## Job Scheduling

- Assign jobs to machines
- Minimize completion time
- Handle constraints
- Optimize resources

</div>

<div class="border rounded p-4 text-sm">

## Resource Allocation

- Budget optimization
- Project allocation
- Maximize utility
- Multi-objective problems

</div>

</div>

---
layout: center
class: text-center
---

# Key Advantages Summary

<div class="grid grid-cols-2 gap-8 mt-8 text-left text-lg">

<div>

✅ No gradient required  
✅ Global search  
✅ Escapes local minima  
✅ Works on any function  
✅ Robust single run  

</div>

<div>

✅ Simple to implement  
✅ Few parameters  
✅ Parallelizable  
✅ Fast convergence  
✅ Real-world ready  

</div>

</div>

---
layout: center
class: text-center
---

# Best Use Cases

<div class="grid grid-cols-2 gap-8 mt-8 text-left text-lg">

<div>

🎯 Multimodal optimization  
🎯 Non-symmetrical landscapes  
🎯 Black-box functions  
🎯 Complex search spaces  

</div>

<div>

🎯 No gradient available  
🎯 Robust solutions needed  
🎯 Parallel computing  
🎯 Quick results  

</div>

</div>

---
layout: center
class: text-center
---

# Algorithm Summary

<div class="text-lg mt-8 space-y-4">

## The Formula

$$v_{t+1} = w \cdot v_t + c_1 r_1 (p_{best} - x_t) + c_2 r_2 (g_{best} - x_t)$$

$$x_{t+1} = x_t + v_{t+1}$$

## Typical Settings

- Swarm: 30-50 particles
- Inertia ($w$): 0.7
- Cognitive ($c_1$): 1.5
- Social ($c_2$): 1.5

</div>

---
layout: center
class: text-center
---

# Key Takeaways

<div class="text-left text-xl mt-8 space-y-3">

1. **Powerful algorithm** for global optimization

2. **Works without gradients** - black-box problems

3. **Robust single run** - no multiple trials

4. **Excellent on multimodal** - escapes local minima

5. **Adapts to any landscape** - explores entire space

6. **Simple to use** - few parameters

7. **Real applications** - tuning, scheduling, allocation

</div>

---
layout: center
class: text-center
---

# Thank You!

<div class="mt-8 text-2xl">

## Questions?

</div>

<div class="mt-12 text-sm opacity-70">

**Particle Swarm Optimization**  
Metaheuristic Optimization Algorithms and Their Real-World Applications

</div>

<div class="mt-8 text-base">

**Aidar Batyrbekov**

</div>
