# Chapter 7: Stochastic Calculus

> *"When the path is nowhere differentiable, classical calculus fails. Itô calculus provides the rules for random worlds."*

## Connection to the Central Thesis

So far, we've assumed the underlying system is deterministic—noisy observations of a smooth trajectory. But many systems are **inherently stochastic**:

- Financial markets (prices driven by random shocks)
- Molecular dynamics (thermal fluctuations)
- Population biology (random births and deaths)
- Turbulent fluids (chaotic, effectively random)

For these systems, the trajectory itself is random. The governing equation is a **stochastic differential equation (SDE)**:

$$dX_t = \mu(X_t, t) dt + \sigma(X_t, t) dW_t$$

where W_t is a Wiener process (Brownian motion).

**The problem**: W_t is continuous but **nowhere differentiable**. Classical derivatives don't exist. We need new tools.

## Brownian Motion

### Definition

A Wiener process W_t is a continuous stochastic process with:
1. W_0 = 0
2. Independent increments: W_t - W_s independent of W_r for r ≤ s < t
3. Gaussian increments: W_t - W_s ~ N(0, t-s)
4. Continuous paths (almost surely)

### Key Properties

- **Nowhere differentiable**: dW/dt doesn't exist
- **Unbounded variation**: Total up-and-down movement is infinite
- **Quadratic variation**: [W, W]_t = t (this is crucial for Itô calculus)
- **Self-similar**: W_{ct} has same distribution as √c · W_t

### Simulation

```python
import numpy as np
import matplotlib.pyplot as plt

def simulate_brownian(T, n_steps, n_paths=1):
    """Simulate Brownian motion paths."""
    dt = T / n_steps
    dW = np.sqrt(dt) * np.random.randn(n_paths, n_steps)
    W = np.cumsum(dW, axis=1)
    W = np.hstack([np.zeros((n_paths, 1)), W])
    t = np.linspace(0, T, n_steps + 1)
    return t, W

t, W = simulate_brownian(1.0, 1000, n_paths=5)
plt.plot(t, W.T)
plt.xlabel('Time')
plt.ylabel('W(t)')
plt.title('Brownian Motion Sample Paths')
plt.show()
```

## Why Classical Calculus Fails

### The Chain Rule Breaks

For smooth functions, d[f(X_t)] = f'(X_t) dX_t.

For Brownian motion, this is wrong. Consider f(x) = x²:

$$d[W_t^2] \neq 2W_t \, dW_t$$

The correct answer (Itô's lemma) includes an extra term:

$$d[W_t^2] = 2W_t \, dW_t + dt$$

### The Culprit: Quadratic Variation

In classical calculus, (dx)² → 0 as dx → 0.

For Brownian motion, (dW)² → dt ≠ 0.

This non-vanishing quadratic variation is why stochastic calculus has different rules.

## Itô Calculus

### Itô's Lemma

For a smooth function f(X_t, t) where dX_t = μ dt + σ dW_t:

$$df = \frac{\partial f}{\partial t} dt + \frac{\partial f}{\partial x} dX_t + \frac{1}{2} \frac{\partial^2 f}{\partial x^2} (dX_t)^2$$

Using (dW)² = dt, (dt)² = 0, dt·dW = 0:

$$df = \left(\frac{\partial f}{\partial t} + \mu \frac{\partial f}{\partial x} + \frac{\sigma^2}{2} \frac{\partial^2 f}{\partial x^2}\right) dt + \sigma \frac{\partial f}{\partial x} dW_t$$

The extra term (σ²/2) ∂²f/∂x² is the **Itô correction**.

### Example: Geometric Brownian Motion

Stock prices often modeled as:
$$dS_t = \mu S_t \, dt + \sigma S_t \, dW_t$$

Apply Itô's lemma to f(S) = log(S):

$$d[\log S_t] = \left(\mu - \frac{\sigma^2}{2}\right) dt + \sigma \, dW_t$$

So log(S_t) is Brownian motion with drift μ - σ²/2.

**Solution**:
$$S_t = S_0 \exp\left[\left(\mu - \frac{\sigma^2}{2}\right)t + \sigma W_t\right]$$

## Estimating Drift and Diffusion

### The Estimation Problem

Given observations X_{t₁}, X_{t₂}, ..., X_{tₙ} from:
$$dX_t = \mu(X_t) dt + \sigma(X_t) dW_t$$

Estimate μ(x) and σ(x).

### Infinitesimal Moments

For small Δt:
$$\mathbb{E}[X_{t+\Delta t} - X_t | X_t = x] \approx \mu(x) \Delta t$$
$$\mathbb{E}[(X_{t+\Delta t} - X_t)^2 | X_t = x] \approx \sigma^2(x) \Delta t$$

### Nonparametric Estimation

```python
import numpy as np
from pydelt.interpolation import LowessInterpolator

def estimate_drift_diffusion(t, X, bandwidth=0.1):
    """
    Estimate drift and diffusion from SDE observations.
    
    Uses local averaging of increments.
    """
    dt = np.diff(t)
    dX = np.diff(X)
    X_mid = (X[:-1] + X[1:]) / 2
    
    # Estimate drift: E[dX]/dt
    drift_raw = dX / dt
    
    # Estimate diffusion: E[(dX)²]/dt
    diffusion_raw = dX**2 / dt
    
    # Smooth estimates
    lowess_drift = LowessInterpolator(frac=bandwidth)
    lowess_drift.fit(X_mid, drift_raw)
    
    lowess_diff = LowessInterpolator(frac=bandwidth)
    lowess_diff.fit(X_mid, diffusion_raw)
    
    def mu(x):
        return lowess_drift(np.atleast_1d(x))
    
    def sigma_sq(x):
        return np.maximum(lowess_diff(np.atleast_1d(x)), 0)
    
    return mu, sigma_sq

# Example: Ornstein-Uhlenbeck process
# dX = -θX dt + σ dW
theta, sigma_true = 1.0, 0.5
dt = 0.01
t = np.arange(0, 100, dt)
X = np.zeros(len(t))
for i in range(1, len(t)):
    X[i] = X[i-1] - theta * X[i-1] * dt + sigma_true * np.sqrt(dt) * np.random.randn()

mu_est, sigma_sq_est = estimate_drift_diffusion(t, X, bandwidth=0.2)

# Check estimates
x_test = np.linspace(-2, 2, 50)
print(f"True drift at x=1: {-theta * 1:.2f}")
print(f"Estimated drift at x=1: {mu_est(1.0)[0]:.2f}")
print(f"True σ²: {sigma_true**2:.2f}")
print(f"Estimated σ² at x=0: {sigma_sq_est(0.0)[0]:.2f}")
```

## The Fokker-Planck Equation

### From SDE to PDE

The probability density p(x, t) of X_t evolves according to:

$$\frac{\partial p}{\partial t} = -\frac{\partial}{\partial x}[\mu(x) p] + \frac{1}{2}\frac{\partial^2}{\partial x^2}[\sigma^2(x) p]$$

This is the **Fokker-Planck** (or Kolmogorov forward) equation.

### Stationary Distribution

If the system reaches equilibrium, ∂p/∂t = 0:

$$p_{\text{eq}}(x) \propto \frac{1}{\sigma^2(x)} \exp\left(\int^x \frac{2\mu(y)}{\sigma^2(y)} dy\right)$$

### Application: Estimating Potential

For gradient systems μ(x) = -∂U/∂x with constant σ:

$$p_{\text{eq}}(x) \propto \exp\left(-\frac{2U(x)}{\sigma^2}\right)$$

The stationary distribution reveals the potential landscape.

## Numerical Methods for SDEs

### Euler-Maruyama

Simplest discretization:
$$X_{n+1} = X_n + \mu(X_n) \Delta t + \sigma(X_n) \sqrt{\Delta t} \, Z_n$$

where Z_n ~ N(0, 1).

**Strong convergence**: O(√Δt)
**Weak convergence**: O(Δt)

### Milstein Method

Includes Itô correction:
$$X_{n+1} = X_n + \mu \Delta t + \sigma \sqrt{\Delta t} Z_n + \frac{1}{2}\sigma \sigma' (\Delta t Z_n^2 - \Delta t)$$

**Strong convergence**: O(Δt)

### When to Use What

| Goal | Method | Convergence |
|------|--------|-------------|
| Distribution statistics | Euler-Maruyama | Weak O(Δt) sufficient |
| Path-dependent quantities | Milstein | Need strong O(Δt) |
| High accuracy | Higher-order Runge-Kutta | O(Δt^{1.5}) or better |

## Applications

### Finance: Option Pricing

The Black-Scholes model:
$$dS_t = rS_t \, dt + \sigma S_t \, dW_t$$

Option price V(S, t) satisfies:
$$\frac{\partial V}{\partial t} + rS\frac{\partial V}{\partial S} + \frac{\sigma^2 S^2}{2}\frac{\partial^2 V}{\partial S^2} = rV$$

**Greeks** (sensitivities) are derivatives:
- Delta: ∂V/∂S
- Gamma: ∂²V/∂S²
- Theta: ∂V/∂t
- Vega: ∂V/∂σ

### Physics: Langevin Dynamics

Particle in potential U(x) with thermal noise:
$$m\ddot{x} = -\gamma \dot{x} - \nabla U(x) + \sqrt{2\gamma k_B T} \, \xi(t)$$

In overdamped limit:
$$dx = -\frac{1}{\gamma}\nabla U(x) \, dt + \sqrt{\frac{2k_B T}{\gamma}} \, dW_t$$

### Biology: Population Dynamics

Stochastic logistic growth:
$$dN = rN\left(1 - \frac{N}{K}\right) dt + \sigma N \, dW_t$$

Noise can cause extinction even when deterministic model predicts survival.

## PyDelt for Stochastic Systems

### Challenges

1. **Paths are rough**: Standard smoothing may over-smooth
2. **Derivatives don't exist**: Estimate drift/diffusion instead
3. **Noise is intrinsic**: Can't be "removed"

### Recommended Approach

```python
from pydelt.interpolation import LowessInterpolator
import numpy as np

# For SDE data, estimate drift and diffusion rather than derivatives
def analyze_sde_data(t, X):
    """
    Analyze SDE observations.
    
    Returns drift and diffusion estimates.
    """
    dt = np.diff(t)
    dX = np.diff(X)
    X_mid = (X[:-1] + X[1:]) / 2
    
    # Local estimates
    drift_local = dX / dt
    diff_local = dX**2 / dt
    
    # Smooth with robust method (handles outliers from large jumps)
    lowess = LowessInterpolator(frac=0.1)
    
    lowess.fit(X_mid, drift_local)
    drift_smooth = lowess(X_mid)
    
    lowess.fit(X_mid, diff_local)
    diff_smooth = lowess(X_mid)
    
    return X_mid, drift_smooth, np.sqrt(np.maximum(diff_smooth, 0))
```

## Key Takeaways

1. **Stochastic systems need different calculus** (Itô, not Newton-Leibniz)
2. **Itô's lemma has an extra term** from quadratic variation
3. **Estimate drift and diffusion**, not pointwise derivatives
4. **Fokker-Planck connects SDEs to PDEs** for probability evolution
5. **Numerical methods exist** but convergence is slower than for ODEs
6. **Applications span finance, physics, biology**

## Exercises

1. **Itô's lemma**: Verify that for geometric Brownian motion, d[S²] = (2μS² + σ²S²)dt + 2σS² dW.

2. **Drift estimation**: Simulate an Ornstein-Uhlenbeck process. Estimate drift μ(x) = -θx and verify you recover θ.

3. **Euler-Maruyama convergence**: Simulate geometric Brownian motion with decreasing Δt. Verify weak convergence is O(Δt).

4. **Stationary distribution**: For dX = -X dt + dW, the stationary distribution is N(0, 1/2). Simulate long trajectory and verify.

---

*Previous: [← Differential Equations](06_differential_equations.md) | Next: [Applications →](08_applications.md)*
