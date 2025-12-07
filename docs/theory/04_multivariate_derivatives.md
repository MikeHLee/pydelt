# Chapter 4: Multivariate Derivatives

> *"In one dimension, the derivative is a number. In n dimensions, it's a vector, a matrix, or a tensor."*

## Connection to the Central Thesis

Most real systems are multivariate. A robot arm has joint angles. A neural network has millions of weights. A climate model has temperature, pressure, and humidity at every grid point.

The governing dynamics are:
$$\frac{d\mathbf{x}}{dt} = \mathbf{f}(\mathbf{x}, t)$$

where **x** ∈ ℝⁿ is the state vector. To understand this system from data, we need:
- **Gradients**: How does a scalar output depend on vector input?
- **Jacobians**: How does a vector output depend on vector input?
- **Hessians**: What is the curvature of the landscape?

The approximation paradigm extends naturally: fit a multivariate surrogate, then differentiate it. But the **curse of dimensionality** makes this harder—we need exponentially more data to cover high-dimensional spaces.

PyDelt addresses this with separable fitting strategies and neural network surrogates that scale to high dimensions.

## From Scalar to Vector

Your calculus course covered f: ℝ → ℝ. Real problems involve:
- f: ℝⁿ → ℝ (scalar field, e.g., loss function)
- f: ℝⁿ → ℝᵐ (vector field, e.g., neural network layer)
- f: ℝⁿ → ℝᵐˣᵖ (tensor field)

Each requires different derivative objects.

## Partial Derivatives

### Definition

The partial derivative of f(x₁, ..., xₙ) with respect to xᵢ:

$$\frac{\partial f}{\partial x_i} = \lim_{h \to 0} \frac{f(x_1, ..., x_i + h, ..., x_n) - f(x_1, ..., x_i, ..., x_n)}{h}$$

Hold all other variables fixed, differentiate with respect to one.

### Notation

| Notation | Meaning |
|----------|---------|
| ∂f/∂xᵢ | Partial derivative |
| fₓᵢ | Subscript notation |
| ∂ᵢf | Index notation |
| Dᵢf | Operator notation |

### Computing from Data

For discrete data, use finite differences along each dimension:

```python
import numpy as np

def partial_derivative(f_values, grid, dim, order=1):
    """
    Compute partial derivative along dimension dim.
    
    f_values: n-dimensional array of function values
    grid: list of 1D arrays for each dimension
    dim: which dimension to differentiate
    """
    h = grid[dim][1] - grid[dim][0]  # Assume uniform
    
    # Central difference along specified axis
    return np.gradient(f_values, h, axis=dim)
```

## The Gradient

### Definition

For f: ℝⁿ → ℝ, the gradient is the vector of partial derivatives:

$$\nabla f = \begin{pmatrix} \partial f/\partial x_1 \\ \partial f/\partial x_2 \\ \vdots \\ \partial f/\partial x_n \end{pmatrix}$$

### Geometric Interpretation

- **Direction**: Points toward steepest ascent
- **Magnitude**: Rate of change in that direction
- **Perpendicular** to level sets (contours)

### The Gradient in ML

Gradient descent: move opposite to gradient to minimize loss.

$$\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)$$

### Computing Gradients from Data

```python
from pydelt.multivariate import MultivariateDerivatives
from pydelt.interpolation import SplineInterpolator
import numpy as np

# Generate 2D data
np.random.seed(42)
n_points = 500
x = np.random.randn(n_points, 2)  # 2D input
y = x[:, 0]**2 + x[:, 1]**2  # f(x,y) = x² + y²

# Fit and compute gradient
mv = MultivariateDerivatives(SplineInterpolator, smoothing=0.1)
mv.fit(x, y)

gradient_func = mv.gradient()

# Evaluate gradient at test points
test_points = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
gradients = gradient_func(test_points)

# For f = x² + y², gradient = (2x, 2y)
print("Computed gradients:")
print(gradients)
print("\nTrue gradients:")
print(2 * test_points)
```

## The Jacobian

### Definition

For f: ℝⁿ → ℝᵐ, the Jacobian is the m×n matrix of partial derivatives:

$$J_f = \begin{pmatrix} 
\partial f_1/\partial x_1 & \cdots & \partial f_1/\partial x_n \\
\vdots & \ddots & \vdots \\
\partial f_m/\partial x_1 & \cdots & \partial f_m/\partial x_n
\end{pmatrix}$$

Row i contains the gradient of fᵢ.

### Interpretation

The Jacobian is the **best linear approximation** to f near a point:

$$f(x + \delta) \approx f(x) + J_f(x) \cdot \delta$$

### The Jacobian in ML

- **Backpropagation**: Chain rule with Jacobians
- **Normalizing flows**: Jacobian determinant for density transformation
- **Sensitivity analysis**: How outputs depend on inputs

### Computing Jacobians from Data

```python
# Vector-valued function: f(x,y) = (x+y, x*y)
x = np.random.randn(500, 2)
y = np.column_stack([x[:, 0] + x[:, 1], x[:, 0] * x[:, 1]])

mv = MultivariateDerivatives(SplineInterpolator, smoothing=0.1)
mv.fit(x, y)

jacobian_func = mv.jacobian()

# Evaluate at a point
point = np.array([[1.0, 2.0]])
J = jacobian_func(point)

# True Jacobian at (1, 2):
# [[1, 1], [2, 1]]  (df1/dx=1, df1/dy=1, df2/dx=y=2, df2/dy=x=1)
print(f"Computed Jacobian:\n{J[0]}")
```

## The Hessian

### Definition

For f: ℝⁿ → ℝ, the Hessian is the n×n matrix of second partial derivatives:

$$H_f = \begin{pmatrix}
\partial^2 f/\partial x_1^2 & \partial^2 f/\partial x_1 \partial x_2 & \cdots \\
\partial^2 f/\partial x_2 \partial x_1 & \partial^2 f/\partial x_2^2 & \cdots \\
\vdots & \vdots & \ddots
\end{pmatrix}$$

For smooth functions, the Hessian is symmetric (mixed partials are equal).

### Interpretation

The Hessian captures **curvature**:
- Eigenvalues > 0: Convex (bowl-shaped)
- Eigenvalues < 0: Concave (dome-shaped)
- Mixed signs: Saddle point

### The Hessian in ML

- **Newton's method**: Uses H⁻¹∇f for faster convergence
- **Second-order optimization**: Adam, L-BFGS approximate Hessian
- **Loss landscape analysis**: Characterize critical points

### Critical Point Classification

At a critical point (∇f = 0):

| Hessian Eigenvalues | Classification |
|---------------------|----------------|
| All positive | Local minimum |
| All negative | Local maximum |
| Mixed signs | Saddle point |
| Some zero | Degenerate (need higher order) |

**In high dimensions, saddle points dominate.** For a random function in n dimensions, the probability of a local minimum vs. saddle point decreases exponentially with n.

### Computing Hessians from Data

```python
# Scalar function
x = np.random.randn(500, 2)
y = x[:, 0]**2 + 2*x[:, 1]**2  # f = x² + 2y²

mv = MultivariateDerivatives(SplineInterpolator, smoothing=0.1)
mv.fit(x, y)

hessian_func = mv.hessian()

# Evaluate at origin
point = np.array([[0.0, 0.0]])
H = hessian_func(point)

# True Hessian: [[2, 0], [0, 4]]
print(f"Computed Hessian:\n{H[0]}")
```

## The Laplacian

### Definition

The Laplacian is the trace of the Hessian (sum of second derivatives):

$$\nabla^2 f = \Delta f = \sum_{i=1}^n \frac{\partial^2 f}{\partial x_i^2} = \text{tr}(H_f)$$

### Interpretation

- **Diffusion**: Heat equation ∂u/∂t = α∇²u
- **Smoothness**: Laplacian measures deviation from local average
- **Graph Laplacian**: Discrete analog on networks

### The Laplacian in ML

- **Graph neural networks**: Message passing uses graph Laplacian
- **Regularization**: Laplacian smoothing
- **Physics-informed ML**: PDEs often involve Laplacian

```python
laplacian_func = mv.laplacian()
lap = laplacian_func(point)

# For f = x² + 2y², Laplacian = 2 + 4 = 6
print(f"Computed Laplacian: {lap[0]}")
```

## Directional Derivatives

### Definition

The derivative of f in direction v (unit vector):

$$D_v f = \nabla f \cdot v = \sum_i \frac{\partial f}{\partial x_i} v_i$$

### Interpretation

Rate of change of f as you move in direction v.

### Maximum Rate of Change

The gradient direction gives maximum rate of change:
- Direction: v = ∇f / ‖∇f‖
- Rate: ‖∇f‖

## The Chain Rule in Multiple Dimensions

### Scalar Composition

If h(t) = f(g(t)) where g: ℝ → ℝⁿ and f: ℝⁿ → ℝ:

$$\frac{dh}{dt} = \nabla f \cdot \frac{dg}{dt} = \sum_i \frac{\partial f}{\partial x_i} \frac{dg_i}{dt}$$

### Vector Composition

If h = f ∘ g where g: ℝᵖ → ℝⁿ and f: ℝⁿ → ℝᵐ:

$$J_h = J_f \cdot J_g$$

This is **backpropagation**: multiply Jacobians in reverse order.

## Numerical Challenges

### Curse of Dimensionality

To estimate gradients in n dimensions with k points per dimension:
- Total points needed: kⁿ
- For n=10, k=10: 10¹⁰ points!

**Solutions**:
- Sparse sampling + interpolation
- Automatic differentiation (if you have the function)
- Dimensionality reduction

### Mixed Partial Derivatives

Computing ∂²f/∂x∂y from data is hard:
- Requires 2D local fitting
- Noise amplifies twice
- Most methods approximate as zero

PyDelt's `MultivariateDerivatives` computes diagonal Hessian elements (∂²f/∂xᵢ²) but approximates mixed partials.

### Boundary Effects

Gradients near the boundary of the data domain are unreliable:
- Fewer neighbors for local fitting
- Extrapolation artifacts
- Consider trimming boundary regions

## PyDelt's Multivariate Interface

```python
from pydelt.multivariate import MultivariateDerivatives
from pydelt.interpolation import SplineInterpolator, LlaInterpolator

# Choose any interpolator
mv = MultivariateDerivatives(
    interpolator_class=SplineInterpolator,
    smoothing=0.1
)

# Fit to data
mv.fit(input_data, output_data)

# Get derivative functions
gradient = mv.gradient()      # For scalar outputs
jacobian = mv.jacobian()      # For vector outputs
hessian = mv.hessian()        # Second derivatives
laplacian = mv.laplacian()    # Trace of Hessian

# Evaluate at query points
grad_values = gradient(query_points)
```

### Limitations

1. **Separable fitting**: Fits 1D interpolators for each input-output pair
2. **Mixed partials**: Approximated as zero
3. **High dimensions**: Accuracy degrades with dimension
4. **Data density**: Needs sufficient coverage of input space

For exact multivariate derivatives, use automatic differentiation with neural networks.

## Key Takeaways

1. **Gradient** (∇f): Direction of steepest ascent, used in optimization
2. **Jacobian** (J): Matrix of partials, used in backprop and flows
3. **Hessian** (H): Curvature matrix, classifies critical points
4. **Laplacian** (∇²f): Sum of second derivatives, appears in PDEs
5. **Chain rule**: Jacobian multiplication, foundation of backprop
6. **Curse of dimensionality**: High-D derivatives are expensive

## Exercises

1. **Gradient field**: For f(x,y) = sin(x)cos(y), compute and plot the gradient field. Verify gradients point perpendicular to contours.

2. **Saddle point**: For f(x,y) = x² - y², compute the Hessian at (0,0). Verify it's a saddle point (mixed eigenvalue signs).

3. **Jacobian of rotation**: For f(x,y) = (x cos θ - y sin θ, x sin θ + y cos θ), compute the Jacobian. Verify det(J) = 1 (area-preserving).

4. **Laplacian smoothing**: Generate a noisy 2D image. Apply Laplacian smoothing: u_new = u + α∇²u. Observe the effect.

---

*Previous: [← Interpolation Methods](03_interpolation_methods.md) | Next: [Approximation Theory →](05_approximation_theory.md)*
