# Chapter 1: Numerical Differentiation

> *"The derivative is a limit. In practice, we can't take limits—we can only approximate them."*

## Connection to the Central Thesis

Recall our core problem: we observe a system governed by unknown dynamics dx/dt = f(x,t), but we only have discrete, noisy samples. **We cannot compute the exact derivative**—we can only approximate it.

Finite differences are the simplest approximation. They work directly on the data without fitting a surrogate. This makes them fast and assumption-free, but also fragile: they amplify noise and require careful step size selection.

Understanding finite differences is essential because:
1. They reveal **why differentiation is hard** (ill-posed inverse problem)
2. They establish **error analysis** techniques used throughout
3. They motivate **why we need smoothing and interpolation**

## The Fundamental Problem

You have discrete samples (tᵢ, yᵢ) and want to estimate dy/dt. The textbook definition:

$$f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$

is useless because:
1. You can't take h → 0 with finite precision arithmetic
2. Your data has gaps—you can't evaluate f at arbitrary points
3. Your measurements have noise

This chapter covers the classical approaches and their limitations.

## Finite Difference Formulas

### Forward, Backward, and Central Differences

Given evenly-spaced data with step h:

| Formula | Expression | Error Order |
|---------|------------|-------------|
| Forward | (f(x+h) - f(x)) / h | O(h) |
| Backward | (f(x) - f(x-h)) / h | O(h) |
| Central | (f(x+h) - f(x-h)) / 2h | O(h²) |

The central difference is more accurate because errors cancel symmetrically.

### Derivation via Taylor Series

Expand f(x+h) and f(x-h) around x:

$$f(x+h) = f(x) + hf'(x) + \frac{h^2}{2}f''(x) + \frac{h^3}{6}f'''(x) + O(h^4)$$

$$f(x-h) = f(x) - hf'(x) + \frac{h^2}{2}f''(x) - \frac{h^3}{6}f'''(x) + O(h^4)$$

Subtracting:

$$f(x+h) - f(x-h) = 2hf'(x) + \frac{h^3}{3}f'''(x) + O(h^5)$$

So:

$$\frac{f(x+h) - f(x-h)}{2h} = f'(x) + \frac{h^2}{6}f'''(x) + O(h^4)$$

The error is O(h²), not O(h).

### Higher-Order Formulas

More points = higher accuracy:

**Five-point stencil for f'(x):**
$$f'(x) \approx \frac{-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h)}{12h} + O(h^4)$$

**Second derivative (central):**
$$f''(x) \approx \frac{f(x+h) - 2f(x) + f(x-h)}{h^2} + O(h^2)$$

### Implementation

```python
import numpy as np

def finite_diff_derivative(y, h, order=1, accuracy=2):
    """
    Compute derivative using finite differences.
    
    Parameters:
        y: array of function values
        h: step size (assumed uniform)
        order: derivative order (1 or 2)
        accuracy: 2 or 4 (determines stencil width)
    """
    n = len(y)
    dy = np.zeros(n)
    
    if order == 1 and accuracy == 2:
        # Central difference O(h²)
        dy[1:-1] = (y[2:] - y[:-2]) / (2*h)
        dy[0] = (y[1] - y[0]) / h  # Forward at boundary
        dy[-1] = (y[-1] - y[-2]) / h  # Backward at boundary
        
    elif order == 1 and accuracy == 4:
        # Five-point stencil O(h⁴)
        dy[2:-2] = (-y[4:] + 8*y[3:-1] - 8*y[1:-3] + y[:-4]) / (12*h)
        # Fall back to lower order at boundaries
        dy[0] = (y[1] - y[0]) / h
        dy[1] = (y[2] - y[0]) / (2*h)
        dy[-2] = (y[-1] - y[-3]) / (2*h)
        dy[-1] = (y[-1] - y[-2]) / h
        
    elif order == 2 and accuracy == 2:
        # Second derivative O(h²)
        dy[1:-1] = (y[2:] - 2*y[1:-1] + y[:-2]) / h**2
        dy[0] = dy[1]  # Extrapolate at boundaries
        dy[-1] = dy[-2]
        
    return dy
```

## The Two Sources of Error

### Truncation Error

From dropping higher-order Taylor terms. For central difference:

$$\epsilon_{\text{trunc}} \approx \frac{h^2}{6} |f'''(x)|$$

**Decreases** as h → 0.

### Rounding Error

From finite precision arithmetic. When subtracting nearly-equal numbers:

$$\epsilon_{\text{round}} \approx \frac{\epsilon_{\text{machine}} |f(x)|}{h}$$

where ε_machine ≈ 10⁻¹⁶ for float64.

**Increases** as h → 0.

### The Optimal Step Size

Total error ≈ truncation + rounding:

$$\epsilon_{\text{total}} \approx \frac{h^2}{6}|f'''| + \frac{\epsilon_{\text{machine}}|f|}{h}$$

Minimize by taking derivative with respect to h and setting to zero:

$$h_{\text{opt}} \approx \left(\frac{3\epsilon_{\text{machine}}|f|}{|f'''|}\right)^{1/3}$$

For typical functions and float64: **h_opt ≈ 10⁻⁵ to 10⁻⁶**.

```python
import numpy as np
import matplotlib.pyplot as plt

def test_optimal_h():
    """Demonstrate optimal step size."""
    f = np.sin
    f_prime = np.cos
    x = 1.0
    
    h_values = np.logspace(-15, 0, 100)
    errors = []
    
    for h in h_values:
        approx = (f(x + h) - f(x - h)) / (2 * h)
        true = f_prime(x)
        errors.append(abs(approx - true))
    
    plt.loglog(h_values, errors)
    plt.xlabel('Step size h')
    plt.ylabel('Absolute error')
    plt.title('Finite Difference Error vs Step Size')
    plt.axvline(1e-5, color='r', linestyle='--', label='Optimal h ≈ 10⁻⁵')
    plt.legend()
    plt.show()
    
    # Find optimal
    opt_idx = np.argmin(errors)
    print(f"Optimal h: {h_values[opt_idx]:.2e}, Error: {errors[opt_idx]:.2e}")
```

## The Noise Amplification Problem

With noisy data y = f(x) + ε where ε ~ N(0, σ²):

$$\frac{y(x+h) - y(x-h)}{2h} = f'(x) + \frac{\epsilon(x+h) - \epsilon(x-h)}{2h}$$

The noise term has variance:

$$\text{Var}\left(\frac{\epsilon(x+h) - \epsilon(x-h)}{2h}\right) = \frac{2\sigma^2}{4h^2} = \frac{\sigma^2}{2h^2}$$

**Noise in derivative scales as σ/h**. Smaller h = more noise amplification.

### Demonstration

```python
import numpy as np
from pydelt.interpolation import SplineInterpolator

# True function and derivative
t = np.linspace(0, 2*np.pi, 200)
y_true = np.sin(t)
dy_true = np.cos(t)

# Add noise
noise_level = 0.02
y_noisy = y_true + noise_level * np.random.randn(len(t))

# Finite difference (naive)
h = t[1] - t[0]
dy_fd = np.gradient(y_noisy, h)

# Compare noise levels
print(f"Signal noise std: {np.std(y_noisy - y_true):.4f}")
print(f"Derivative noise std: {np.std(dy_fd - dy_true):.4f}")
print(f"Amplification factor: {np.std(dy_fd - dy_true) / np.std(y_noisy - y_true):.1f}x")
```

## Irregular Grids

Real data often has non-uniform spacing. The formulas change:

### Two-Point (Non-uniform)

$$f'(x_i) \approx \frac{f(x_{i+1}) - f(x_{i-1})}{x_{i+1} - x_{i-1}}$$

### Three-Point (Non-uniform)

For points x₀, x₁, x₂ with spacings h₁ = x₁ - x₀ and h₂ = x₂ - x₁:

$$f'(x_1) \approx \frac{h_1^2 f(x_2) - h_2^2 f(x_0) + (h_2^2 - h_1^2) f(x_1)}{h_1 h_2 (h_1 + h_2)}$$

### Fornberg's Algorithm

For arbitrary stencils, use Fornberg's algorithm to compute optimal weights:

```python
def fornberg_weights(z, x, m):
    """
    Compute finite difference weights for derivative of order m
    at point z using function values at points x.
    
    Based on Fornberg (1988).
    """
    n = len(x) - 1
    c = np.zeros((n+1, m+1))
    c[0, 0] = 1
    c1 = 1
    
    for i in range(1, n+1):
        mn = min(i, m)
        c2 = 1
        for j in range(i):
            c3 = x[i] - x[j]
            c2 *= c3
            if i <= m:
                c[i-1, i] = 0
            for k in range(mn, -1, -1):
                c[i, k] = c1 * (k * c[i-1, k-1] - (x[i-1] - z) * c[i-1, k]) / c2
                c[j, k] = ((x[i] - z) * c[j, k] - k * c[j, k-1]) / c3
        c1 = c2
    
    return c[:, m]
```

## PyDelt's Finite Difference Methods

PyDelt provides robust finite difference implementations:

```python
from pydelt import finite_difference

# Basic usage
t = np.linspace(0, 1, 100)
y = np.sin(2 * np.pi * t)

# First derivative
dy = finite_difference(t, y, order=1)

# Second derivative
d2y = finite_difference(t, y, order=2)

# With specified accuracy
dy_high = finite_difference(t, y, order=1, accuracy=4)
```

## When Finite Differences Fail

Finite differences are inadequate when:

1. **Noise level is significant** (σ/h becomes large)
2. **Data is sparse** (can't use higher-order stencils)
3. **Function has sharp features** (truncation error dominates)
4. **You need derivatives at non-sample points**

The solution: **smoothing** and **interpolation**, covered in the next chapters.

## Key Takeaways

1. **Central differences** are O(h²), better than forward/backward O(h)
2. **Optimal h** balances truncation and rounding error (~10⁻⁵ for float64)
3. **Noise amplifies** as 1/h—smaller steps = more noise
4. **Irregular grids** require modified formulas (Fornberg's algorithm)
5. **Finite differences alone** are insufficient for noisy data

## Exercises

1. **Optimal step size**: For f(x) = exp(x) at x = 0, find the step size that minimizes total error. Compare to the theoretical prediction.

2. **Noise amplification**: Generate sin(x) with Gaussian noise σ = 0.01. Plot derivative error vs. step size h. At what h does noise dominate?

3. **Higher-order stencils**: Implement the 5-point stencil for f'(x). Compare accuracy to 3-point for f(x) = x⁵ at x = 1.

4. **Non-uniform grid**: Generate data on a random grid. Implement the 3-point non-uniform formula and compare to linear interpolation + uniform finite difference.

---

*Previous: [← The Numerical Calculus Problem](00_why_calculus.md) | Next: [Noise and Smoothing →](02_noise_and_smoothing.md)*
