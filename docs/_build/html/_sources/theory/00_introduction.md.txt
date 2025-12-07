# Introduction: The Approximation Paradigm

> *"We cannot compute exact derivatives from data. But by approximating the underlying system, we transform an intractable problem into a tractable one."*

## A Quick Refresher: What Calculus Promises

Calculus gives us powerful tools for analyzing change:

- **Derivatives** tell us instantaneous rates of change
- **Integrals** accumulate quantities over time
- **Differential equations** describe how systems evolve

If you have a function f(x), calculus lets you compute f'(x) exactly. For f(x) = x², you get f'(x) = 2x. Done.

**The problem**: In the real world, we don't have f(x). We have *measurements*.

## The Real-World Gap

Every physical, biological, financial, and engineered system evolves according to some underlying dynamics:

$$\frac{dx}{dt} = f(x, t)$$

The function f encodes the "equations of motion." If we knew f exactly, calculus would give us everything.

**But we almost never know f.** What we have instead:
- **Discrete observations** at scattered times
- **Noisy measurements** corrupted by sensors
- **No closed-form expression** for the dynamics

This is the gap between textbook calculus and real-world data.

## The Simplest Solution: Linear Regression

Let's start with the most basic case. You have data that looks roughly linear:

```python
import numpy as np
from sklearn.linear_model import LinearRegression

# Noisy observations of a linear system
t = np.array([0, 1, 2, 3, 4, 5]).reshape(-1, 1)
x = np.array([1.1, 2.9, 5.2, 6.8, 9.1, 10.9])  # True: x = 2t + 1

# Fit a line
model = LinearRegression()
model.fit(t, x)

print(f"Estimated slope (dx/dt): {model.coef_[0]:.2f}")  # ≈ 2.0
print(f"Estimated intercept: {model.intercept_:.2f}")    # ≈ 1.0
```

**The slope of the regression line is our derivative estimate.**

This works because:
1. We assume the true relationship is linear: x(t) = mt + b
2. The derivative of a line is its slope: dx/dt = m
3. Linear regression finds the best-fit slope

**Key insight**: We approximated the unknown function with a line, then differentiated the approximation. This is the core idea behind all numerical differentiation.

## From Lines to Curves: The Approximation Paradigm

Linear regression works when the true function is (approximately) linear. But what about curves?

The same principle applies:

> **Fit a differentiable function to the data, then differentiate the fitted function.**

| Data Pattern | Approximation | Derivative |
|--------------|---------------|------------|
| Linear | Line: y = mx + b | m (constant) |
| Quadratic | Parabola: y = ax² + bx + c | 2ax + b |
| Smooth curve | Spline | Spline derivative |
| Complex pattern | Neural network | Autodiff gradient |

The approximation is not the true function. But if it's close, its derivatives approximate the true derivatives.

## Multiple Regression: Partial Derivatives

What if your system depends on multiple variables? For z = f(x, y):

```python
import numpy as np
from sklearn.linear_model import LinearRegression

# Observations of a system depending on two variables
# True: z = 3x + 2y + 1
np.random.seed(42)
n = 100
x = np.random.randn(n)
y = np.random.randn(n)
z = 3*x + 2*y + 1 + 0.1*np.random.randn(n)  # Add noise

# Multiple linear regression
X = np.column_stack([x, y])
model = LinearRegression()
model.fit(X, z)

print(f"∂z/∂x ≈ {model.coef_[0]:.2f}")  # ≈ 3.0
print(f"∂z/∂y ≈ {model.coef_[1]:.2f}")  # ≈ 2.0
```

**The regression coefficients are partial derivative estimates.**

This extends to gradients, Jacobians, and Hessians—the multivariate derivatives essential for optimization and machine learning.

## Beyond Linear: Why We Need More

Linear regression assumes the underlying function is linear. Real systems are often:

- **Nonlinear**: Oscillations, exponential growth, saturation
- **Locally varying**: Different behavior in different regions
- **Discontinuous**: Phase transitions, regime changes
- **Stochastic**: Inherent randomness, not just measurement noise

For these, we need more sophisticated approximations:

| System Type | Challenge | PyDelt Solution |
|-------------|-----------|------------------|
| Smooth nonlinear | Curves, not lines | Splines, local polynomials |
| Locally varying | Different slopes in different regions | LOWESS, kernel methods |
| Noisy | Noise amplified by differentiation | Smoothing, regularization |
| High-dimensional | Curse of dimensionality | Neural networks, separable methods |
| Stochastic | No classical derivative exists | Itô calculus, drift estimation |

## The Bias-Variance Tradeoff

Every approximation method faces a fundamental tension:

- **Too simple** (e.g., linear fit to curved data): High bias, misses true structure
- **Too complex** (e.g., interpolating every noisy point): High variance, amplifies noise

For derivatives, this is especially severe because **differentiation amplifies noise**. A small wiggle in the function becomes a large spike in the derivative.

The art is finding the right balance. PyDelt provides tools to navigate this tradeoff.

## The Progression: From Simple to Complex

This documentation follows a natural progression:

### Level 1: Linear Systems
Linear regression gives slopes. Multiple regression gives partial derivatives. This is where we start.

### Level 2: Smooth Nonlinear Systems
Splines and local polynomials handle curves. We fit piecewise smooth functions and differentiate them analytically.

```python
from pydelt.interpolation import SplineInterpolator

interp = SplineInterpolator(smoothing=0.1)
interp.fit(t, x)
dx_dt = interp.differentiate(order=1)(t)
```

### Level 3: Noisy Systems
Smoothing methods (LOWESS, Savitzky-Golay) balance fidelity to data against noise amplification.

### Level 4: High-Dimensional Systems
Gradients, Jacobians, and Hessians from scattered multivariate data. Neural networks scale where classical methods fail.

```python
from pydelt.multivariate import MultivariateDerivatives

mv = MultivariateDerivatives(SplineInterpolator, smoothing=0.1)
mv.fit(inputs, outputs)
gradient = mv.gradient()(query_points)
```

### Level 5: Dynamical Systems
ODEs from data. System identification. Predicting future states from learned dynamics.

### Level 6: Stochastic Systems
When the path itself is random, classical derivatives don't exist. Itô calculus provides the framework for drift and diffusion estimation.

## What This Buys You

Once you have a differentiable approximation, you can:

1. **Compute derivatives at any point**—not just at data points
2. **Estimate higher derivatives**—acceleration, curvature, jerk
3. **Compute multivariate derivatives**—gradients, Jacobians, Hessians
4. **Identify governing equations**—discover dx/dt = f(x) from data
5. **Quantify uncertainty**—error bounds on derivative estimates

## The Limitations

Approximation is powerful but not magic:

- **Garbage in, garbage out**: Too sparse or noisy data defeats any method
- **Extrapolation fails**: Surrogates are only valid within the data domain
- **Sharp features get smoothed**: Discontinuities are blurred
- **High dimensions are hard**: Curse of dimensionality limits classical methods
- **Stochastic systems need special treatment**: Classical smoothing doesn't apply

## The Structure of This Documentation

Each chapter addresses a piece of the approximation paradigm:

| Chapter | Question | Answer |
|---------|----------|--------|
| 1. Numerical Differentiation | How do finite differences work and fail? | Truncation vs. rounding error, optimal step sizes |
| 2. Noise and Smoothing | How do we handle noisy data? | Bias-variance tradeoff, regularization |
| 3. Interpolation Methods | What surrogates can we use? | Splines, local regression, kernels, neural nets |
| 4. Multivariate Derivatives | How do we handle multiple dimensions? | Gradients, Jacobians, Hessians |
| 5. Approximation Theory | When does approximation work? | Error bounds, convergence rates |
| 6. Differential Equations | How do we work with ODEs from data? | System identification, prediction |
| 7. Stochastic Calculus | What about random systems? | Itô calculus, SDEs, drift/diffusion estimation |
| 8. Applications | Where is this used? | Finance, physics, sensors, ML |

## Prerequisites

We assume you have:
- **Calculus**: Derivatives, integrals, chain rule, Taylor series
- **Linear algebra**: Vectors, matrices, eigenvalues
- **Basic statistics**: Mean, variance, distributions
- **Python/NumPy**: Arrays, plotting

We'll introduce:
- **Numerical analysis**: Error, stability, convergence
- **Multivariate calculus**: Gradients, Jacobians, Hessians
- **Stochastic processes**: Brownian motion, SDEs
- **Approximation theory**: Function spaces, basis expansions

## The Payoff

By the end of this documentation, you'll understand:

1. **Why exact derivatives from data are impossible**—and why that's okay
2. **How to choose the right approximation method** for your problem
3. **How to quantify and control error** in derivative estimates
4. **How to work with multivariate and stochastic systems**
5. **How to apply these tools** to real problems in science and engineering

The goal is not to make you a numerical analyst. It's to give you the **working understanding** needed to use PyDelt effectively and interpret its outputs correctly.

---

*Next: [Numerical Differentiation →](01_numerical_differentiation.md)*
