# Chapter 3: Interpolation Methods

> *"To differentiate discrete data, first make it continuous. The choice of interpolation method determines everything."*

## Connection to the Central Thesis

This chapter is the heart of the approximation paradigm. **Interpolation creates the differentiable surrogate** that makes everything else possible.

The true system evolves according to unknown dynamics. We observe discrete samples. By fitting an interpolant, we construct a continuous, differentiable function that approximates the true trajectory. This surrogate:

- **Exists everywhere** in the domain (not just at data points)
- **Has well-defined derivatives** (by construction)
- **Encodes smoothness assumptions** about the true system
- **Yields to standard calculus** tools

The choice of interpolation method is the most consequential decision in numerical differentiation. It determines what class of functions you're searching over, what smoothness you assume, and what errors you'll incur.

## The Interpolation-Differentiation Pipeline

The standard approach to numerical differentiation:

1. **Fit** a continuous function f̂(x) to discrete data (xᵢ, yᵢ)
2. **Differentiate** f̂(x) analytically or numerically
3. **Evaluate** f̂'(x) at desired points

The interpolation method determines:
- Smoothness of derivatives
- Bias-variance tradeoff
- Computational cost
- Behavior at boundaries

## Polynomial Interpolation

### Lagrange Interpolation

Given n+1 points, there's a unique polynomial of degree n passing through all of them:

$$p(x) = \sum_{i=0}^{n} y_i \prod_{j \neq i} \frac{x - x_j}{x_i - x_j}$$

**Problem**: High-degree polynomials oscillate wildly (Runge's phenomenon).

```python
import numpy as np
import matplotlib.pyplot as plt

# Runge's function
f = lambda x: 1 / (1 + 25*x**2)

# Interpolate with increasing degree
x = np.linspace(-1, 1, 1000)
for n in [5, 10, 15]:
    xi = np.linspace(-1, 1, n+1)
    yi = f(xi)
    
    # Lagrange interpolation
    from numpy.polynomial import polynomial as P
    coeffs = np.polyfit(xi, yi, n)
    p = np.polyval(coeffs, x)
    
    plt.plot(x, p, label=f'degree {n}')

plt.plot(x, f(x), 'k--', label='true', linewidth=2)
plt.ylim(-1, 2)
plt.legend()
plt.title("Runge's Phenomenon")
plt.show()
```

**Lesson**: Global polynomial interpolation is unstable. Use piecewise methods instead.

## Spline Interpolation

### What is a Spline?

A spline of degree k is a piecewise polynomial where:
- Each piece is degree k
- Function is Cᵏ⁻¹ continuous (k-1 continuous derivatives)
- Pieces join at **knots**

### Cubic Splines (k=3)

Most common choice. Between knots xᵢ and xᵢ₊₁:

$$S_i(x) = a_i + b_i(x-x_i) + c_i(x-x_i)^2 + d_i(x-x_i)^3$$

Constraints:
- Interpolation: S(xᵢ) = yᵢ
- C¹ continuity: S'ᵢ(xᵢ₊₁) = S'ᵢ₊₁(xᵢ₊₁)
- C² continuity: S''ᵢ(xᵢ₊₁) = S''ᵢ₊₁(xᵢ₊₁)
- Boundary conditions (natural, clamped, not-a-knot)

### Natural Cubic Splines

Boundary condition: S''(x₀) = S''(xₙ) = 0

This minimizes the "total curvature":

$$\int_{x_0}^{x_n} (S''(x))^2 dx$$

among all C² interpolants.

### Smoothing Splines

Don't interpolate exactly—balance fit and smoothness:

$$\min_f \sum_{i=1}^n (y_i - f(x_i))^2 + \lambda \int (f''(x))^2 dx$$

The solution is a natural cubic spline with knots at all data points.

```python
from scipy.interpolate import UnivariateSpline
import numpy as np

t = np.linspace(0, 2*np.pi, 100)
y = np.sin(t) + 0.1 * np.random.randn(100)

# Interpolating spline (s=0)
spline_interp = UnivariateSpline(t, y, s=0)

# Smoothing spline
spline_smooth = UnivariateSpline(t, y, s=1.0)

# Derivatives are analytical
dy_smooth = spline_smooth.derivative()(t)
d2y_smooth = spline_smooth.derivative(n=2)(t)
```

### B-Splines

Splines represented in the B-spline basis:

$$S(x) = \sum_{i} c_i B_{i,k}(x)$$

where Bᵢ,ₖ are B-spline basis functions of degree k.

**Advantages**:
- Numerically stable
- Local support (changing one coefficient affects limited region)
- Efficient algorithms (de Boor's algorithm)

```python
from scipy.interpolate import BSpline, make_interp_spline

# Create B-spline interpolant
spline = make_interp_spline(t, y, k=3)

# Evaluate and differentiate
y_interp = spline(t_new)
dy_interp = spline.derivative()(t_new)
```

## Local Polynomial Regression

### The Idea

Instead of fitting one global function, fit local polynomials around each query point.

At query point x₀:
1. Select nearby data points
2. Fit weighted polynomial (weights decrease with distance)
3. Use polynomial value/derivative at x₀

### LOWESS/LOESS

**LO**cally **W**eighted **S**catterplot **S**moothing:

$$\min_{\beta} \sum_{i=1}^n w_i(x_0) (y_i - \beta_0 - \beta_1(x_i - x_0) - ...)^2$$

where weights w_i(x₀) decrease with |xᵢ - x₀|.

Common weight function (tricube):

$$w(u) = \begin{cases} (1 - |u|^3)^3 & |u| < 1 \\ 0 & |u| \geq 1 \end{cases}$$

```python
from pydelt.interpolation import LowessInterpolator

# LOWESS with 10% of data in each local fit
lowess = LowessInterpolator(frac=0.1)
lowess.fit(t, y)

# Smooth values and derivatives
y_smooth = lowess(t)
dy_smooth = lowess.differentiate(order=1)(t)
```

### Local Linear Approximation (LLA)

PyDelt's LLA fits local polynomials with explicit derivative estimation:

```python
from pydelt.interpolation import LlaInterpolator

# Window-based local fitting
lla = LlaInterpolator(window_size=11, poly_order=2)
lla.fit(t, y)

# First and second derivatives
dy = lla.differentiate(order=1)(t)
d2y = lla.differentiate(order=2)(t)
```

### Comparison: Global vs. Local

| Aspect | Global (Splines) | Local (LOWESS/LLA) |
|--------|------------------|-------------------|
| Smoothness | C² everywhere | May have discontinuities |
| Adaptivity | Same smoothing everywhere | Can adapt to local features |
| Computation | O(n) after setup | O(n²) naive, O(n log n) with trees |
| Extrapolation | Dangerous | Very dangerous |
| Outlier robustness | Poor | Good (with robust weights) |

## Kernel Methods

### Kernel Smoothing

Generalization of local averaging:

$$\hat{f}(x) = \frac{\sum_i K_h(x - x_i) y_i}{\sum_i K_h(x - x_i)}$$

where Kₕ(u) = K(u/h)/h is a scaled kernel.

### Common Kernels

| Kernel | Formula | Properties |
|--------|---------|------------|
| Gaussian | exp(-u²/2) | Infinitely smooth |
| Epanechnikov | (3/4)(1-u²) for |u|<1 | Optimal MSE |
| Tricube | (1-|u|³)³ for |u|<1 | Used in LOESS |
| Uniform | 1 for |u|<1 | Moving average |

### Bandwidth Selection

The bandwidth h controls bias-variance:
- Small h: Low bias, high variance
- Large h: High bias, low variance

Optimal bandwidth (for MSE) scales as n^(-1/5) for kernel regression.

## Radial Basis Functions (RBF)

### The Approach

Represent the function as:

$$f(x) = \sum_{i=1}^n c_i \phi(\|x - x_i\|)$$

where φ is a radial basis function.

### Common RBFs

| RBF | Formula | Properties |
|-----|---------|------------|
| Gaussian | exp(-r²/ε²) | Infinitely smooth |
| Multiquadric | √(1 + (εr)²) | Smooth, good for interpolation |
| Thin plate spline | r² log(r) | Minimizes bending energy |
| Polyharmonic | rᵏ or rᵏ log(r) | Generalizes thin plate |

### Interpolation with RBFs

Solve the linear system:

$$\begin{pmatrix} \Phi & P \\ P^T & 0 \end{pmatrix} \begin{pmatrix} c \\ \lambda \end{pmatrix} = \begin{pmatrix} y \\ 0 \end{pmatrix}$$

where Φᵢⱼ = φ(‖xᵢ - xⱼ‖) and P contains polynomial terms.

```python
from scipy.interpolate import RBFInterpolator
import numpy as np

# 1D example
x = np.linspace(0, 2*np.pi, 20).reshape(-1, 1)
y = np.sin(x.flatten())

# RBF interpolation
rbf = RBFInterpolator(x, y, kernel='thin_plate_spline')

# Evaluate
x_new = np.linspace(0, 2*np.pi, 100).reshape(-1, 1)
y_new = rbf(x_new)
```

## Gaussian Process Regression

### Probabilistic Interpolation

Model the function as a Gaussian process:

$$f(x) \sim \mathcal{GP}(m(x), k(x, x'))$$

where m(x) is the mean function and k(x, x') is the covariance kernel.

### Posterior Mean and Variance

Given data (X, y), the posterior at new point x* is:

$$f(x^*) | X, y \sim \mathcal{N}(\mu^*, \sigma^{*2})$$

where:
$$\mu^* = k(x^*, X) [K + \sigma_n^2 I]^{-1} y$$
$$\sigma^{*2} = k(x^*, x^*) - k(x^*, X) [K + \sigma_n^2 I]^{-1} k(X, x^*)$$

### Derivatives of GPs

If f ~ GP, then f' ~ GP with:
- Mean: derivative of posterior mean
- Covariance: derivative of kernel

```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

# GP regression
kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
gp = GaussianProcessRegressor(kernel=kernel)
gp.fit(t.reshape(-1, 1), y)

# Predictions with uncertainty
y_pred, y_std = gp.predict(t_new.reshape(-1, 1), return_std=True)
```

## PyDelt's Interpolation Classes

### Unified Interface

All interpolators share the same API:

```python
from pydelt.interpolation import (
    SplineInterpolator,
    LlaInterpolator,
    GllaInterpolator,
    LowessInterpolator,
    LoessInterpolator,
    FdaInterpolator,
    NeuralNetworkInterpolator
)

# Same pattern for all
interp = SplineInterpolator(smoothing=0.1)
interp.fit(t, y)

# Evaluate
y_smooth = interp(t_query)

# Differentiate
dy = interp.differentiate(order=1)(t_query)
d2y = interp.differentiate(order=2)(t_query)
```

### Method Comparison

| Method | Best For | Smoothing Control | Derivative Quality |
|--------|----------|-------------------|-------------------|
| Spline | Smooth functions | `smoothing` parameter | Excellent (analytical) |
| LLA | Local features | `window_size` | Good (polynomial) |
| GLLA | Noisy + local | `embedding`, `n` | Good |
| LOWESS | Outliers | `frac` | Moderate (numerical) |
| FDA | Functional data | Basis functions | Excellent |
| Neural | Complex patterns | Architecture | Exact (autodiff) |

## Choosing an Interpolation Method

### Decision Tree

1. **Is data very noisy?**
   - Yes → Use smoothing spline or LOWESS
   - No → Interpolating spline may work

2. **Are there outliers?**
   - Yes → LOWESS/LOESS (robust)
   - No → Splines are fine

3. **Do you need derivatives at non-data points?**
   - Yes → Splines or GP (continuous derivatives)
   - No → Local methods are fine

4. **Is the function highly nonlinear/complex?**
   - Yes → Neural network or GP
   - No → Splines or local polynomials

5. **Is computational speed critical?**
   - Yes → Splines (O(n) after setup)
   - No → Any method

## Key Takeaways

1. **Avoid high-degree global polynomials** (Runge's phenomenon)
2. **Splines** are the workhorse: piecewise polynomials with continuity
3. **Local methods** (LOWESS, LLA) adapt to local structure
4. **Smoothing parameter** controls bias-variance tradeoff
5. **PyDelt provides unified interface** across all methods

## Exercises

1. **Runge's phenomenon**: Interpolate 1/(1+25x²) with polynomials of degree 5, 10, 15, 20. Then use cubic splines. Compare.

2. **Spline vs. LOWESS**: Generate sin(x) + outliers. Compare derivative estimates from smoothing spline vs. LOWESS.

3. **Bandwidth selection**: Implement cross-validation for LOWESS bandwidth. Find optimal for sin(x) + noise.

4. **GP derivatives**: Use sklearn's GP to fit noisy data. Extract derivative by finite-differencing the posterior mean. Compare to analytical GP derivative.

---

*Previous: [← Noise and Smoothing](02_noise_and_smoothing.md) | Next: [Multivariate Derivatives →](04_multivariate_derivatives.md)*
