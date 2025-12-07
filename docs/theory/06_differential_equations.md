# Chapter 6: Differential Equations from Data

> *"The universe is written in the language of differential equations. But we rarely know the equations—only the trajectories."*

## Connection to the Central Thesis

This chapter brings together everything we've learned. The ultimate goal of numerical differentiation is often to understand **dynamical systems**:

$$\frac{dx}{dt} = f(x, t)$$

We observe trajectories x(t). We want to:
1. **Estimate derivatives** dx/dt from noisy samples
2. **Identify the dynamics** f(x, t) from derivative estimates
3. **Predict future states** by integrating the identified system
4. **Analyze stability** and long-term behavior

This is the payoff of the approximation paradigm: transforming raw observations into mathematical understanding of the underlying system.

## The System Identification Problem

### Setup

You observe a system at discrete times:
- States: x(t₁), x(t₂), ..., x(tₙ)
- Possibly with noise: yᵢ = x(tᵢ) + εᵢ

You want to find f such that dx/dt = f(x, t) explains the observations.

### The Two-Step Approach

**Step 1**: Estimate derivatives
$$\hat{v}_i \approx \frac{dx}{dt}\bigg|_{t=t_i}$$

using interpolation + differentiation (PyDelt's core functionality).

**Step 2**: Fit dynamics
$$\hat{v}_i \approx f(x_i, t_i)$$

by regression, sparse regression, or neural networks.

### Why This Is Hard

1. **Derivative estimation amplifies noise** (Chapter 2)
2. **The form of f is unknown** (could be anything)
3. **Data may not cover the state space** (limited observations)
4. **Multiple f's may fit the data** (non-identifiability)

## Estimating Derivatives for ODEs

### The Noise Problem

For ODEs, derivative errors propagate:
- Small errors in dx/dt → errors in identified f
- Errors in f → errors in predictions
- Predictions diverge exponentially for chaotic systems

### Recommended Approaches

| Noise Level | Method | Notes |
|-------------|--------|-------|
| Very low | Finite differences | Fast, simple |
| Low | Spline interpolation | Smooth derivatives |
| Moderate | Smoothing splines | Balance bias/variance |
| High | LOWESS or GP | Robust, uncertainty quantification |

### PyDelt for ODE Derivatives

```python
from pydelt.interpolation import SplineInterpolator
import numpy as np

# Observed trajectory (e.g., from sensors)
t_obs = np.linspace(0, 10, 100)
x_obs = np.sin(t_obs) + 0.05 * np.random.randn(100)  # Noisy observations

# Fit smooth surrogate
interp = SplineInterpolator(smoothing=0.5)
interp.fit(t_obs, x_obs)

# Estimate derivatives
x_smooth = interp(t_obs)
dx_dt = interp.differentiate(order=1)(t_obs)
d2x_dt2 = interp.differentiate(order=2)(t_obs)

# Now (x_smooth, dx_dt) pairs can be used for system identification
```

## Sparse Identification of Nonlinear Dynamics (SINDy)

### The Idea

Assume f is a sparse combination of candidate functions:

$$\frac{dx}{dt} = \sum_k \xi_k \phi_k(x)$$

where φₖ are candidates (1, x, x², sin(x), ...) and most ξₖ = 0.

### Algorithm

1. Build library matrix Θ with columns [1, x, x², x³, sin(x), ...]
2. Estimate derivatives: ẋ
3. Solve sparse regression: ẋ ≈ Θξ with ‖ξ‖₀ small

### Example: Discovering a Pendulum

```python
import numpy as np
from scipy.integrate import odeint

# True system: d²θ/dt² = -sin(θ) (pendulum)
def true_dynamics(y, t):
    theta, omega = y
    return [omega, -np.sin(theta)]

# Generate data
t = np.linspace(0, 20, 500)
y0 = [np.pi/4, 0]
sol = odeint(true_dynamics, y0, t)
theta, omega = sol[:, 0], sol[:, 1]

# Add noise
theta_noisy = theta + 0.01 * np.random.randn(len(t))
omega_noisy = omega + 0.01 * np.random.randn(len(t))

# Estimate derivatives using PyDelt
from pydelt.interpolation import SplineInterpolator

interp_theta = SplineInterpolator(smoothing=0.1)
interp_theta.fit(t, theta_noisy)
dtheta_dt = interp_theta.differentiate(order=1)(t)

interp_omega = SplineInterpolator(smoothing=0.1)
interp_omega.fit(t, omega_noisy)
domega_dt = interp_omega.differentiate(order=1)(t)

# Build library: [1, θ, ω, θ², θω, ω², sin(θ), cos(θ), ...]
Theta = np.column_stack([
    np.ones_like(theta),
    theta_noisy,
    omega_noisy,
    theta_noisy**2,
    np.sin(theta_noisy),
    np.cos(theta_noisy)
])

# Sparse regression for dω/dt
from sklearn.linear_model import Lasso
lasso = Lasso(alpha=0.01)
lasso.fit(Theta, domega_dt)

print("Discovered coefficients:")
print(f"  1: {lasso.coef_[0]:.4f}")
print(f"  θ: {lasso.coef_[1]:.4f}")
print(f"  ω: {lasso.coef_[2]:.4f}")
print(f"  θ²: {lasso.coef_[3]:.4f}")
print(f"  sin(θ): {lasso.coef_[4]:.4f}")  # Should be ≈ -1
print(f"  cos(θ): {lasso.coef_[5]:.4f}")
```

## Neural Network Dynamics

### Neural ODEs

Learn f directly as a neural network:

$$\frac{dx}{dt} = f_\theta(x, t)$$

where f_θ is a neural network with parameters θ.

### Training

Minimize prediction error:
$$L(\theta) = \sum_i \|x(t_i) - \hat{x}(t_i; \theta)\|^2$$

where x̂ is obtained by integrating the neural ODE.

### Advantages

- No need to specify functional form
- Can capture complex, nonlinear dynamics
- Automatic differentiation handles gradients

### Disadvantages

- Black box (hard to interpret)
- May not generalize outside training data
- Computationally expensive

## Phase Space Analysis

### Reconstructing Phase Space

From a single time series x(t), reconstruct the full state using **delay embedding**:

$$\mathbf{y}(t) = [x(t), x(t-\tau), x(t-2\tau), ..., x(t-(d-1)\tau)]$$

**Takens' theorem**: For generic systems, this reconstruction is diffeomorphic to the true attractor if d > 2n (where n is the true dimension).

### Estimating Derivatives in Reconstructed Space

```python
from pydelt.interpolation import GllaInterpolator

# GLLA is designed for delay embedding
glla = GllaInterpolator(embedding=5, n=3)
glla.fit(t, x_obs)

# Get derivatives in embedded space
dx_dt = glla.differentiate(order=1)(t)
```

## Stability Analysis

### Linearization

Near an equilibrium x*, the dynamics are approximately linear:

$$\frac{d\delta x}{dt} = J \cdot \delta x$$

where J = ∂f/∂x|_{x*} is the Jacobian.

### Stability from Eigenvalues

| Eigenvalues of J | Stability |
|------------------|-----------|
| All Re(λ) < 0 | Asymptotically stable |
| Any Re(λ) > 0 | Unstable |
| All Re(λ) ≤ 0, some = 0 | Marginal (need higher order) |

### Computing Jacobians from Data

```python
from pydelt.multivariate import MultivariateDerivatives

# Fit multivariate surrogate to vector field
mv = MultivariateDerivatives(SplineInterpolator, smoothing=0.1)
mv.fit(state_data, derivative_data)

# Jacobian at equilibrium
jacobian_func = mv.jacobian()
J = jacobian_func(equilibrium_point)

# Eigenvalue analysis
eigenvalues = np.linalg.eigvals(J[0])
print(f"Eigenvalues: {eigenvalues}")
print(f"Stable: {all(np.real(eigenvalues) < 0)}")
```

## Partial Differential Equations

### From ODEs to PDEs

PDEs involve derivatives in space and time:

$$\frac{\partial u}{\partial t} = F\left(u, \frac{\partial u}{\partial x}, \frac{\partial^2 u}{\partial x^2}, ...\right)$$

### Estimating Spatial Derivatives

Same techniques apply, but now in space:

```python
# u(x, t) observed on a grid
# Estimate ∂u/∂x at fixed t

from pydelt.interpolation import SplineInterpolator

# For each time slice
for t_idx in range(n_times):
    interp = SplineInterpolator(smoothing=0.1)
    interp.fit(x_grid, u[:, t_idx])
    du_dx[:, t_idx] = interp.differentiate(order=1)(x_grid)
```

### Physics-Informed Approaches

If you know the PDE form, use it as a constraint:

$$L_{\text{physics}} = \left\|\frac{\partial u}{\partial t} - F\left(u, \frac{\partial u}{\partial x}, ...\right)\right\|^2$$

This is the basis of Physics-Informed Neural Networks (PINNs).

## Practical Workflow

### Step-by-Step System Identification

1. **Visualize data**: Plot trajectories, look for patterns
2. **Estimate derivatives**: Use PyDelt with appropriate smoothing
3. **Check derivative quality**: Do they make physical sense?
4. **Build candidate library**: Based on domain knowledge
5. **Fit sparse model**: SINDy or similar
6. **Validate**: Integrate identified system, compare to data
7. **Analyze**: Stability, bifurcations, long-term behavior

### Common Pitfalls

1. **Over-smoothing**: Misses fast dynamics
2. **Under-smoothing**: Noise dominates derivatives
3. **Wrong library**: Missing the true terms
4. **Overfitting**: Too many terms, poor generalization
5. **Extrapolation**: Predictions outside training region fail

## Key Takeaways

1. **System identification requires good derivative estimates**
2. **SINDy discovers sparse, interpretable dynamics**
3. **Neural ODEs learn complex dynamics without functional form**
4. **Phase space reconstruction works from single time series**
5. **Stability analysis uses Jacobians from data**
6. **Validation is essential**—integrate and compare

## Exercises

1. **Harmonic oscillator**: Generate data from ẍ + x = 0. Use PyDelt to estimate ẋ and ẍ. Verify the relationship ẍ ≈ -x.

2. **Lorenz system**: Generate data from the Lorenz equations. Use SINDy to recover the equations from noisy observations.

3. **Stability analysis**: For the damped pendulum θ̈ + 0.1θ̇ + sin(θ) = 0, estimate the Jacobian at θ = 0 from data. Verify stability.

4. **Neural ODE**: Train a neural ODE on pendulum data. Compare predictions to the true trajectory.

---

*Previous: [← Approximation Theory](05_approximation_theory.md) | Next: [Stochastic Calculus →](07_stochastic_calculus.md)*
