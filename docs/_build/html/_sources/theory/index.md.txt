# Theory: Numerical Calculus for Real Systems

> *"We cannot compute exact derivatives from data. But by approximating the underlying system, we transform an intractable problem into a tractable one."*

## The Central Thesis

Every physical, biological, financial, and engineered system evolves according to some underlying dynamics—equations of motion we rarely know exactly. What we have instead are discrete, noisy observations.

**PyDelt's approach**: Fit a differentiable surrogate to the data. The surrogate yields to standard calculus tools, giving us approximate derivatives with quantifiable error.

This documentation teaches you:
1. **Why exact derivatives are impossible** from discrete, noisy data
2. **How approximation methods work** and when they fail
3. **How to quantify and control error** in derivative estimates
4. **How to apply these tools** across domains

## Who This Is For

- **Data scientists** who know basic calculus but need numerical methods
- **Engineers** working with sensor data and dynamical systems
- **Researchers** in physics, biology, or finance dealing with noisy observations
- **ML practitioners** who want to understand gradients beyond autodiff

We assume you've completed undergraduate calculus (derivatives, integrals, chain rule). We'll introduce multivariate calculus, numerical analysis, and stochastic processes as needed.

---

## Chapters

### Part I: Foundations

| Chapter | Topic | Key Question |
|---------|-------|--------------|
| [0. Introduction](00_introduction.md) | The Approximation Paradigm | Why can't we compute exact derivatives? |
| [1. Numerical Differentiation](01_numerical_differentiation.md) | Finite Differences | How do classical methods work and fail? |
| [2. Noise and Smoothing](02_noise_and_smoothing.md) | Bias-Variance Tradeoff | How do we handle noisy data? |
| [3. Interpolation Methods](03_interpolation_methods.md) | Surrogates | What functions can we fit? |

### Part II: Extensions

| Chapter | Topic | Key Question |
|---------|-------|--------------|
| [4. Multivariate Derivatives](04_multivariate_derivatives.md) | Gradients, Jacobians, Hessians | How do we handle multiple dimensions? |
| [5. Approximation Theory](05_approximation_theory.md) | Error Bounds | When does approximation work? |
| [6. Differential Equations](06_differential_equations.md) | System Identification | How do we discover dynamics from data? |
| [7. Stochastic Calculus](07_stochastic_calculus.md) | SDEs and Itô | What about random systems? |

### Part III: Practice

| Chapter | Topic | Key Question |
|---------|-------|--------------|
| [8. Applications](08_applications.md) | Real-World Use Cases | How do we apply this under error? |
| [Bibliography](bibliography.md) | References | Where can I learn more? |

---

## The Narrative Arc

**Chapter 0**: Sets up the central problem—we observe systems but don't know their equations.

**Chapters 1-3**: Build the toolkit—finite differences fail on noise, smoothing helps but biases, interpolation creates differentiable surrogates.

**Chapters 4-5**: Extend to harder problems—multiple dimensions, theoretical guarantees.

**Chapters 6-7**: Handle dynamics—ODEs from data, stochastic systems that break classical calculus.

**Chapter 8**: Apply to real domains—sensors, finance, physics, ML—each with its own error sources.

---

## Quick Reference

### PyDelt's Core Pattern

```python
from pydelt.interpolation import SplineInterpolator

# 1. Create interpolator with smoothing
interp = SplineInterpolator(smoothing=0.1)

# 2. Fit to data
interp.fit(time_data, signal_data)

# 3. Get derivative function
derivative = interp.differentiate(order=1)

# 4. Evaluate anywhere
dy_dt = derivative(query_points)
```

### Method Selection Guide

| Scenario | Method | Smoothing |
|----------|--------|-----------|
| Low noise, smooth function | Spline | Low (0.01-0.1) |
| Moderate noise | Spline | Medium (0.1-1.0) |
| High noise | LOWESS | frac=0.1-0.3 |
| Outliers present | LOWESS | Robust by default |
| Complex patterns | Neural Network | Architecture-dependent |
| Stochastic system | Estimate drift/diffusion | Heavy smoothing |

### Error Scaling

| Quantity | Error Scaling | Notes |
|----------|---------------|-------|
| Function value | O(n^{-4/5}) | For C² functions, optimal |
| First derivative | O(n^{-3/5}) | One order worse |
| Second derivative | O(n^{-2/5}) | Two orders worse |
| Noise amplification | O(1/h) | Smaller step = more noise |

---

## Learning Paths

### Quick Start (2 hours)
For those who need to use PyDelt now:
1. [Introduction](00_introduction.md) — 15 min
2. [Interpolation Methods](03_interpolation_methods.md) — 45 min
3. [Applications](08_applications.md) — 60 min

### Standard Path (8-10 hours)
For solid understanding:
- Work through chapters 0-5 in order
- Skim 6-7 based on your domain
- Study 8 for your application area

### Deep Dive (20+ hours)
For mastery:
- All chapters with exercises
- Implement methods from scratch
- Read bibliography references
- Apply to your own data

---

## Prerequisites Assumed

- **Calculus**: Derivatives, integrals, chain rule, Taylor series
- **Linear Algebra**: Vectors, matrices, eigenvalues
- **Statistics**: Mean, variance, Gaussian distribution
- **Python**: NumPy, basic plotting

## Concepts Introduced

- **Numerical Analysis**: Truncation error, rounding error, stability
- **Multivariate Calculus**: Gradients, Jacobians, Hessians
- **Approximation Theory**: Function spaces, convergence rates
- **Stochastic Processes**: Brownian motion, Itô calculus
- **System Identification**: ODEs from data, SINDy

---

*Start your journey: [Introduction →](00_introduction.md)*
