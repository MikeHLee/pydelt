# Chapter 5: Approximation Theory

> *"Every approximation has an error. The question is: can we bound it, minimize it, and understand when it fails?"*

## Connection to the Central Thesis

We've established that exact derivatives from data are impossible. We approximate instead. But how good is the approximation? When does it work? When does it fail?

**Approximation theory** provides the mathematical framework to answer these questions. It tells us:
- **Convergence rates**: How fast does error decrease with more data?
- **Error bounds**: What's the worst-case error for a given method?
- **Optimality**: Is there a better method, or are we at the limit?

This chapter gives you the tools to reason rigorously about approximation quality—essential for trusting your derivative estimates.

## The Approximation Problem

### Setup

We want to approximate an unknown function f from a class F (e.g., smooth functions) using a function f̂ from a class G (e.g., polynomials, splines).

**Key quantities**:
- **Approximation error**: ‖f - f̂‖ in some norm
- **Best approximation**: inf_{g ∈ G} ‖f - g‖
- **Achievable rate**: How does error scale with complexity of G?

### For Derivatives

If we approximate f with f̂, how well does f̂' approximate f'?

**Key insight**: Derivative approximation is harder than function approximation. If ‖f - f̂‖ = O(hᵏ), then typically ‖f' - f̂'‖ = O(hᵏ⁻¹).

Differentiation "costs" one order of accuracy.

## Function Spaces

### Smoothness Classes

| Class | Definition | Example |
|-------|------------|---------|
| C⁰ | Continuous | |x| |
| C¹ | Continuous first derivative | x|x| |
| Cᵏ | k continuous derivatives | Polynomials |
| C^∞ | Infinitely differentiable | sin(x), exp(x) |
| Analytic | Equals its Taylor series | sin(x), exp(x) |

Smoother functions are easier to approximate.

### Sobolev Spaces

Functions with derivatives in Lᵖ:

$$W^{k,p} = \{f : D^\alpha f \in L^p \text{ for } |\alpha| \leq k\}$$

These spaces are natural for PDE theory and provide sharp approximation results.

### Why Smoothness Matters

**Theorem (Jackson)**: If f ∈ Cᵏ[a,b], then the best polynomial approximation of degree n satisfies:

$$\inf_{\deg p \leq n} \|f - p\|_\infty \leq C \cdot \frac{\|f^{(k)}\|_\infty}{n^k}$$

Smoother functions (larger k) → faster convergence (higher power of n).

## Polynomial Approximation

### Weierstrass Theorem

Any continuous function on [a,b] can be uniformly approximated by polynomials.

**Implication**: Polynomials are "universal approximators" for continuous functions.

### Rate of Convergence

For f ∈ Cᵏ:
- Best polynomial of degree n: error O(n⁻ᵏ)
- Interpolating polynomial at n+1 points: error O(n⁻ᵏ) if points chosen well

### Runge's Phenomenon

Equally-spaced interpolation points can cause divergence:

$$\max_{x \in [-1,1]} |f(x) - p_n(x)| \to \infty \text{ as } n \to \infty$$

for some smooth functions (e.g., 1/(1+25x²)).

**Solution**: Use Chebyshev points (clustered near endpoints) or piecewise methods.

## Spline Approximation

### Why Splines Work

Instead of one high-degree polynomial, use many low-degree pieces.

**Theorem**: For f ∈ Cᵏ and cubic splines with n knots:

$$\|f - S_n\|_\infty = O(h^4), \quad \|f' - S_n'\|_\infty = O(h^3)$$

where h = 1/n is the knot spacing.

### Optimal Rates

For smoothing splines minimizing ∫(f'')² + λ⁻¹∑(yᵢ - f(xᵢ))²:

$$\mathbb{E}[\|f - \hat{f}\|^2] = O(n^{-4/5})$$

when f ∈ C² and noise is Gaussian. This is **minimax optimal**—no method can do better without additional assumptions.

### Derivative Approximation

For cubic splines:
- Function: O(h⁴)
- First derivative: O(h³)
- Second derivative: O(h²)

Each derivative costs one order.

## Local Polynomial Regression

### Bias-Variance for Local Methods

For kernel regression with bandwidth h:

$$\text{Bias}^2 \approx h^{2(p+1)} \cdot |f^{(p+1)}|^2$$
$$\text{Variance} \approx \frac{\sigma^2}{nh^d}$$

where p is polynomial order, d is dimension, σ² is noise variance.

### Optimal Bandwidth

Minimize MSE = Bias² + Variance:

$$h_{\text{opt}} \propto n^{-1/(2p+2+d)}$$

For local linear (p=1) in 1D (d=1):

$$h_{\text{opt}} \propto n^{-1/5}, \quad \text{MSE} = O(n^{-4/5})$$

Same rate as smoothing splines—this is the optimal rate for C² functions.

### For Derivatives

Estimating f' with local polynomials:

$$\text{MSE}(f') = O(n^{-2(p-1)/(2p+3)})$$

For local quadratic (p=2): MSE(f') = O(n^{-2/7}) ≈ O(n^{-0.29})

Slower than function estimation—derivatives are harder.

## Neural Network Approximation

### Universal Approximation

**Theorem (Cybenko, Hornik)**: A feedforward network with one hidden layer can approximate any continuous function on a compact set to arbitrary accuracy.

**Caveat**: Says nothing about:
- How many neurons needed
- How to find the weights
- Generalization to unseen data

### Approximation Rates

For ReLU networks with L layers and W total weights:

$$\inf_{\text{networks}} \|f - f_{\text{NN}}\| = O(W^{-2/d})$$

for f ∈ C² in d dimensions. Deep networks can achieve better rates for certain function classes.

### Derivatives via Autodiff

Neural network derivatives are exact (via automatic differentiation), but the network itself is an approximation. The derivative of an approximate function is an approximate derivative.

## Error Analysis Framework

### Total Error Decomposition

$$\text{Total Error} = \text{Approximation Error} + \text{Estimation Error} + \text{Computational Error}$$

- **Approximation**: Best possible in function class (bias)
- **Estimation**: From finite, noisy data (variance)
- **Computational**: From numerical precision (usually negligible)

### For Derivative Estimation

$$\|\hat{f}' - f'\| \leq \|\hat{f}' - \hat{f}_{\text{true}}'\| + \|\hat{f}_{\text{true}}' - f'\|$$

where f̂_true is the best approximation to f in the function class.

First term: estimation error (from noise)
Second term: approximation error (from function class)

## Convergence Diagnostics

### How to Know If Approximation Is Good

1. **Residual analysis**: Do residuals look like noise?
2. **Cross-validation**: Does error decrease with more data?
3. **Sensitivity analysis**: Do results change with smoothing parameter?
4. **Physical plausibility**: Do derivatives make sense?

### Warning Signs

- Derivatives change sign rapidly (noise dominating)
- Derivatives are very different at nearby points (instability)
- Cross-validation error increases with model complexity (overfitting)
- Results sensitive to smoothing parameter (ill-conditioned)

## Practical Guidelines

### Choosing Approximation Method

| Data Characteristics | Recommended Method | Expected Rate |
|---------------------|-------------------|---------------|
| Smooth, low noise | Splines | O(n^{-4/5}) |
| Smooth, high noise | Smoothing splines | O(n^{-4/5}) |
| Non-smooth, low noise | Local linear | O(n^{-2/3}) |
| Complex patterns | Neural networks | Depends on architecture |
| High dimensional | Neural networks | Avoids curse of dimensionality |

### Rules of Thumb

1. **More data always helps** (until computational limits)
2. **Smoother functions are easier** (higher convergence rates)
3. **Derivatives are harder than values** (lose one order)
4. **Higher dimensions are harder** (curse of dimensionality)
5. **Noise hurts more for derivatives** (amplification)

## Key Takeaways

1. **Approximation error is unavoidable** but can be bounded
2. **Smoothness determines convergence rate** (smoother = faster)
3. **Derivatives cost one order of accuracy** compared to function values
4. **Optimal rates exist** (n^{-4/5} for C² functions in 1D)
5. **Method choice matters** but optimal methods achieve similar rates
6. **Diagnostics are essential** to verify approximation quality

## Exercises

1. **Jackson's theorem**: Verify numerically that polynomial approximation to sin(x) converges as O(n^{-k}) where k depends on which derivative you're approximating.

2. **Spline convergence**: Generate data from a known C² function. Fit splines with increasing numbers of knots. Verify O(h⁴) convergence for function, O(h³) for derivative.

3. **Optimal bandwidth**: For local linear regression on sin(x) + noise, find the bandwidth that minimizes cross-validation error. Compare to theoretical n^{-1/5}.

4. **Curse of dimensionality**: Repeat exercise 3 in 2D and 3D. How does optimal bandwidth and error scale with dimension?

---

*Previous: [← Multivariate Derivatives](04_multivariate_derivatives.md) | Next: [Differential Equations →](06_differential_equations.md)*
