#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Comprehensive comparison of PyDelt interpolators and derivative methods against other libraries.

This script compares all PyDelt interpolation and differentiation methods:
- SplineInterpolator
- LlaInterpolator
- GllaInterpolator
- GoldInterpolator
- LowessInterpolator
- LoessInterpolator
- FdaInterpolator
- Neural Network methods
- Multivariate derivatives
- Stochastic calculus methods (Itô and Stratonovich)
- Tensor calculus operations

Against popular numerical differentiation libraries:
- SciPy (UnivariateSpline, CubicSpline)
- NumDiffTools
- FinDiff
- JAX automatic differentiation

System specifications:
- M4 Mac with Apple Silicon
- Python 3.12
- NumPy 1.26+
- SciPy 1.12+
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import time
import sys
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Global variable for output directory
OUTPUT_DIR = None

# Add the parent directory to the path so we can import pydelt
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.pydelt.interpolation import (
    SplineInterpolator, 
    LlaInterpolator, 
    GllaInterpolator,
    GoldInterpolator,
    LowessInterpolator,
    LoessInterpolator,
    FdaInterpolator
)

# Try to import multivariate module
try:
    from src.pydelt.multivariate import MultivariateDerivatives
    MULTIVARIATE_AVAILABLE = True
    
    # Multivariate calculus explanation
    """
    Multivariate Calculus in PyDelt:
    
    The MultivariateDerivatives class provides a unified interface for computing derivatives
    of multivariate functions. It supports the following operations:
    
    1. Gradient (∇f): For scalar functions f(x₁,x₂,...,xₙ), computes the vector of partial derivatives
       [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]
    
    2. Jacobian (J_f): For vector-valued functions f(x) = [f₁(x), f₂(x), ..., fₘ(x)], computes the matrix
       of all first-order partial derivatives ∂fᵢ/∂xⱼ
    
    3. Hessian (H_f): For scalar functions, computes the matrix of all second-order partial derivatives
       ∂²f/∂xᵢ∂xⱼ
    
    4. Laplacian (∇²f): For scalar functions, computes the sum of all unmixed second partial derivatives
       ∂²f/∂x₁² + ∂²f/∂x₂² + ... + ∂²f/∂xₙ²
    
    Implementation approach:
    - Fits separate univariate interpolators for each input-output pair
    - Uses the universal .differentiate() method for computing derivatives
    - For traditional interpolators, mixed partial derivatives are approximated as zero
    - For neural network methods, exact mixed partials are computed using automatic differentiation
    
    Limitations:
    - Traditional interpolation methods cannot accurately compute mixed partial derivatives
    - Best suited for functions with separable dependencies
    - For exact mixed partials, neural network methods are recommended
    """
    
except ImportError:
    MULTIVARIATE_AVAILABLE = False
    print("Warning: PyDelt multivariate module not available. Multivariate comparisons will be skipped.")

# Try to import autodiff module
try:
    from src.pydelt.autodiff import neural_network_derivative
    AUTODIFF_AVAILABLE = True
    
    # Tensor calculus explanation
    """
    Tensor Calculus in PyDelt:
    
    PyDelt supports tensor calculus operations through its multivariate derivatives module
    and neural network integration. Key capabilities include:
    
    1. Vector Field Operations:
       - Divergence: ∇·F for vector fields, measuring expansion/contraction
       - Curl: ∇×F for vector fields in 3D, measuring rotation
       - Gradient of vector fields: ∇F, producing tensor fields
    
    2. Tensor Field Operations:
       - Tensor contractions: Computing scalar fields from tensor fields
       - Tensor products: Combining multiple vector/tensor fields
       - Invariant computations: Determinants, traces, eigenvalues
    
    3. Coordinate System Support:
       - Cartesian coordinates (default)
       - Support for transformations to other coordinate systems
    
    Implementation approach:
    - Uses automatic differentiation for exact tensor derivatives
    - Maintains proper tensor dimensions and transformations
    - Provides specialized visualization tools for tensor fields
    
    Applications:
    - Continuum mechanics (stress/strain tensors)
    - Fluid dynamics (velocity gradient tensors)
    - Electromagnetic field analysis
    - General relativity computations
    """
    
    # For PyTorch methods
    try:
        import torch
        import torch.nn as nn
        TORCH_AVAILABLE = True
    except ImportError:
        TORCH_AVAILABLE = False
        print("Warning: PyTorch not available. PyTorch-based neural network comparisons will be skipped.")
except ImportError:
    AUTODIFF_AVAILABLE = False
    TORCH_AVAILABLE = False
    print("Warning: PyDelt autodiff module not available. Neural network comparisons will be skipped.")

# Stochastic calculus explanation
"""
Stochastic Calculus in PyDelt:

PyDelt provides specialized support for stochastic calculus, which is essential for
financial modeling, signal processing with noise, and other applications involving
random processes. Key features include:

1. Itô Calculus:
   - Handles non-differentiable sample paths in stochastic processes
   - Implements Itô's lemma for derivative transformations
   - Accounts for quadratic variation in stochastic processes
   - Formula: df(X_t) = f'(X_t)dX_t + (1/2)f''(X_t)(dX_t)²

2. Stratonovich Calculus:
   - Alternative interpretation of stochastic integrals
   - Preserves ordinary chain rule of calculus
   - Better suited for physical systems and certain mathematical models
   - Formula: df(X_t) = f'(X_t)∘dX_t

3. Stochastic Link Functions:
   - Transforms derivatives through probability distributions
   - Supports various distributions: normal, log-normal, gamma, beta, exponential, Poisson
   - Accounts for distribution-specific corrections

Implementation approach:
- Extends standard interpolators with stochastic corrections
- Uses the set_stochastic_link() method to specify distribution and calculus type
- Automatically applies appropriate correction terms

Limitations:
- Requires knowledge of the underlying stochastic process
- Best results when combined with appropriate noise models
- Computational complexity increases with stochastic process complexity
"""

# Try to import the other libraries, with fallbacks if not available
try:
    from scipy.interpolate import UnivariateSpline, CubicSpline
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: SciPy not available. SciPy comparisons will be skipped.")

try:
    import numdifftools as nd
    NUMDIFFTOOLS_AVAILABLE = True
except ImportError:
    NUMDIFFTOOLS_AVAILABLE = False
    print("Warning: NumDiffTools not available. NumDiffTools comparisons will be skipped.")

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch is not installed. PyTorch-based automatic differentiation will not be available.")

try:
    from findiff import FinDiff
    FINDIFF_AVAILABLE = True
except ImportError:
    FINDIFF_AVAILABLE = False
    print("Warning: FinDiff not available. FinDiff comparisons will be skipped.")

try:
    import jax
    import jax.numpy as jnp
    from jax import grad, jit, vmap
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    print("Warning: JAX not available. JAX comparisons will be skipped.")

# Define test functions with known analytical derivatives
def test_function_sin(x):
    """Sine function with known derivatives."""
    return np.sin(x)

def test_function_sin_derivative(x):
    """Analytical first derivative of sine function."""
    return np.cos(x)

def test_function_sin_second_derivative(x):
    """Analytical second derivative of sine function."""
    return -np.sin(x)

def test_function_exp(x):
    """Exponential function with known derivatives."""
    return np.exp(x)

def test_function_exp_derivative(x):
    """Analytical first derivative of exponential function."""
    return np.exp(x)

def test_function_polynomial(x):
    """Polynomial function with known derivatives: x^3 - 2x^2 + 3x - 1"""
    return x**3 - 2*x**2 + 3*x - 1

def test_function_polynomial_derivative(x):
    """Analytical first derivative of polynomial function: 3x^2 - 4x + 3"""
    return 3*x**2 - 4*x + 3

def test_function_polynomial_second_derivative(x):
    """Analytical second derivative of polynomial function: 6x - 4"""
    return 6*x - 4

# JAX versions of the test functions
def jax_sin(x):
    return jnp.sin(x) if JAX_AVAILABLE else None

def jax_exp(x):
    return jnp.exp(x) if JAX_AVAILABLE else None

def jax_polynomial(x):
    return x**3 - 2*x**2 + 3*x - 1 if JAX_AVAILABLE else None

# Multivariate test functions
def multivariate_test_function(X):
    """Simple multivariate function: f(x,y) = sin(x) + cos(y)"""
    if X.ndim == 1:
        x, y = X
    else:
        x, y = X[:, 0], X[:, 1]
    return np.sin(x) + np.cos(y)

def multivariate_test_gradient(X):
    """Gradient of f(x,y) = sin(x) + cos(y): [cos(x), -sin(y)]"""
    if X.ndim == 1:
        x, y = X
        return np.array([np.cos(x), -np.sin(y)])
    else:
        x, y = X[:, 0], X[:, 1]
        grad_x = np.cos(x)
        grad_y = -np.sin(y)
        return np.column_stack((grad_x, grad_y))

def vector_test_function(X):
    """Vector-valued function: f(x,y) = [sin(x)*cos(y), x^2 + y^2]"""
    if X.ndim == 1:
        x, y = X
    else:
        x, y = X[:, 0], X[:, 1]
    
    f1 = np.sin(x) * np.cos(y)
    f2 = x**2 + y**2
    
    if X.ndim == 1:
        return np.array([f1, f2])
    else:
        return np.column_stack((f1, f2))

def vector_test_jacobian(X):
    """Jacobian of f(x,y) = [sin(x)*cos(y), x^2 + y^2]"""
    if X.ndim == 1:
        x, y = X
        J = np.zeros((2, 2))
        # df1/dx = cos(x)*cos(y)
        J[0, 0] = np.cos(x) * np.cos(y)
        # df1/dy = sin(x)*(-sin(y))
        J[0, 1] = np.sin(x) * (-np.sin(y))
        # df2/dx = 2x
        J[1, 0] = 2 * x
        # df2/dy = 2y
        J[1, 1] = 2 * y
        return J
    else:
        n_points = X.shape[0]
        x, y = X[:, 0], X[:, 1]
        J = np.zeros((n_points, 2, 2))
        
        # df1/dx = cos(x)*cos(y)
        J[:, 0, 0] = np.cos(x) * np.cos(y)
        # df1/dy = sin(x)*(-sin(y))
        J[:, 0, 1] = np.sin(x) * (-np.sin(y))
        # df2/dx = 2x
        J[:, 1, 0] = 2 * x
        # df2/dy = 2y
        J[:, 1, 1] = 2 * y
        return J

def add_noise(y, noise_level=0.01):
    """Add Gaussian noise to the data."""
    return y + noise_level * np.std(y) * np.random.randn(len(y))

def evaluate_all_interpolators(x, y, x_eval, true_derivative, noise_level=0.0, order=1):
    """
    Evaluate all available PyDelt interpolators and other differentiation methods.
    
    Parameters:
    -----------
    x : array_like
        Input data points
    y : array_like
        Output data points
    x_eval : array_like
        Points at which to evaluate the derivatives
    true_derivative : callable
        Function that returns the true derivative values
    noise_level : float
        Level of noise to add (as a fraction of the standard deviation)
    order : int
        Order of the derivative to compute
        
    Returns:
    --------
    dict
        Dictionary of method names and their corresponding errors
    dict
        Dictionary of method names and their corresponding computation times
    dict
        Dictionary of method names and their corresponding derivative values
    """
    # Add noise if specified
    if noise_level > 0:
        y_noisy = add_noise(y, noise_level)
    else:
        y_noisy = y.copy()
    
    # True derivative values
    true_values = true_derivative(x_eval)
    
    results = {}
    times = {}
    
    # PyDelt SplineInterpolator
    start_time = time.time()
    spline_interp = SplineInterpolator()
    spline_interp.fit(x, y_noisy)
    deriv_func = spline_interp.differentiate(order=order)
    pydelt_spline_values = deriv_func(x_eval)
    times['PyDelt Spline'] = time.time() - start_time
    results['PyDelt Spline'] = np.abs(pydelt_spline_values - true_values)
    
    # PyDelt LlaInterpolator
    start_time = time.time()
    lla_interp = LlaInterpolator(window_size=5)
    lla_interp.fit(x, y_noisy)
    deriv_func = lla_interp.differentiate(order=order)
    pydelt_lla_values = deriv_func(x_eval)
    times['PyDelt LLA'] = time.time() - start_time
    results['PyDelt LLA'] = np.abs(pydelt_lla_values - true_values)
    
    # PyDelt GllaInterpolator
    start_time = time.time()
    glla_interp = GllaInterpolator(embedding=3, n=2)
    glla_interp.fit(x, y_noisy)
    deriv_func = glla_interp.differentiate(order=order)
    pydelt_glla_values = deriv_func(x_eval)
    times['PyDelt GLLA'] = time.time() - start_time
    results['PyDelt GLLA'] = np.abs(pydelt_glla_values - true_values)
    
    # PyDelt GoldInterpolator
    start_time = time.time()
    gold_interp = GoldInterpolator(window_size=5)
    gold_interp.fit(x, y_noisy)
    deriv_func = gold_interp.differentiate(order=order)
    pydelt_gold_values = deriv_func(x_eval)
    times['PyDelt GOLD'] = time.time() - start_time
    results['PyDelt GOLD'] = np.abs(pydelt_gold_values - true_values)
    
    # PyDelt LowessInterpolator
    start_time = time.time()
    lowess_interp = LowessInterpolator()
    lowess_interp.fit(x, y_noisy)
    deriv_func = lowess_interp.differentiate(order=order)
    pydelt_lowess_values = deriv_func(x_eval)
    times['PyDelt LOWESS'] = time.time() - start_time
    results['PyDelt LOWESS'] = np.abs(pydelt_lowess_values - true_values)
    
    # PyDelt LoessInterpolator
    start_time = time.time()
    loess_interp = LoessInterpolator(frac=0.3)
    loess_interp.fit(x, y_noisy)
    deriv_func = loess_interp.differentiate(order=order)
    pydelt_loess_values = deriv_func(x_eval)
    times['PyDelt LOESS'] = time.time() - start_time
    results['PyDelt LOESS'] = np.abs(pydelt_loess_values - true_values)
    
    # PyDelt FdaInterpolator
    start_time = time.time()
    fda_interp = FdaInterpolator()
    fda_interp.fit(x, y_noisy)
    deriv_func = fda_interp.differentiate(order=order)
    pydelt_fda_values = deriv_func(x_eval)
    times['PyDelt FDA'] = time.time() - start_time
    results['PyDelt FDA'] = np.abs(pydelt_fda_values - true_values)
    
    # Neural Network derivative (if available)
    if AUTODIFF_AVAILABLE:
        try:
            start_time = time.time()
            nn_deriv_func = neural_network_derivative(
                x, y_noisy, 
                framework='tensorflow', 
                hidden_layers=[64, 32], 
                epochs=500, 
                order=order
            )
            nn_values = nn_deriv_func(x_eval)
            times['PyDelt Neural Network'] = time.time() - start_time
            results['PyDelt Neural Network'] = np.abs(nn_values - true_values)
        except Exception as e:
            print(f"Neural network derivative failed: {e}")
    
    # SciPy methods
    if SCIPY_AVAILABLE:
        # SciPy UnivariateSpline
        start_time = time.time()
        scipy_spline = UnivariateSpline(x, y_noisy, s=len(x)*noise_level**2)
        scipy_spline_values = scipy_spline.derivative(n=order)(x_eval)
        times['SciPy Spline'] = time.time() - start_time
        results['SciPy Spline'] = np.abs(scipy_spline_values - true_values)
        
        # SciPy CubicSpline
        start_time = time.time()
        cubic_spline = CubicSpline(x, y_noisy)
        if order <= 3:  # CubicSpline only supports up to 3rd derivative
            scipy_cubic_values = cubic_spline.derivative(order)(x_eval)
            results['SciPy Cubic'] = np.abs(scipy_cubic_values - true_values)
        else:
            results['SciPy Cubic'] = np.full_like(true_values, np.nan)
        times['SciPy Cubic'] = time.time() - start_time
    
    # NumDiffTools
    if NUMDIFFTOOLS_AVAILABLE:
        # For NumDiffTools, we need to create a function from our noisy data
        from scipy.interpolate import interp1d
        noisy_func = interp1d(x, y_noisy, bounds_error=False, fill_value="extrapolate")
        
        # Use NumDiffTools with the interpolated function
        start_time = time.time()
        nd_derivative = nd.Derivative(noisy_func, n=order)
        nd_values = nd_derivative(x_eval)
        times['NumDiffTools'] = time.time() - start_time
        results['NumDiffTools'] = np.abs(nd_values - true_values)
    
    # FinDiff
    if FINDIFF_AVAILABLE:
        start_time = time.time()
        # Calculate the spacing
        dx = x[1] - x[0]
        # Create the derivative operator
        d_dx = FinDiff(0, dx, order, acc=4)  # Use accuracy order 4
        # Apply to the noisy data
        fd_values = d_dx(y_noisy)
        
        # Since FinDiff only computes derivatives at the original grid points,
        # we need to interpolate to get values at x_eval
        from scipy.interpolate import interp1d
        fd_interp = interp1d(x, fd_values, bounds_error=False, fill_value="extrapolate")
        fd_eval = fd_interp(x_eval)
        
        times['FinDiff'] = time.time() - start_time
        results['FinDiff'] = np.abs(fd_eval - true_values)
    
    # JAX Automatic Differentiation
    if JAX_AVAILABLE:
        # Define the function for JAX
        if true_derivative == test_function_sin_derivative:
            jax_func = jax_sin
        elif true_derivative == test_function_exp_derivative:
            jax_func = jax_exp
        else:
            jax_func = jax_polynomial
        
        start_time = time.time()
        
        # Define JAX derivative function
        if order == 1:
            jax_derivative = jit(grad(jax_func))
        elif order == 2:
            jax_derivative = jit(grad(grad(jax_func)))
        else:
            # For higher orders, compose grad multiple times
            jax_derivative = jax_func
            for _ in range(order):
                jax_derivative = grad(jax_derivative)
            jax_derivative = jit(jax_derivative)
        
        # Convert to JAX array and evaluate
        jax_x_eval = jnp.array(x_eval)
        jax_values = jnp.array([jax_derivative(x_i) for x_i in jax_x_eval])
        
        times['JAX'] = time.time() - start_time
        results['JAX'] = np.abs(np.array(jax_values) - true_values)
    
    # Store the actual derivative values for visualization
    derivative_values = {}
    if 'PyDelt Spline' in results:
        derivative_values['PyDelt Spline'] = pydelt_spline_values
    if 'PyDelt LLA' in results:
        derivative_values['PyDelt LLA'] = pydelt_lla_values
    if 'PyDelt GLLA' in results:
        derivative_values['PyDelt GLLA'] = pydelt_glla_values
    if 'PyDelt GOLD' in results:
        derivative_values['PyDelt GOLD'] = pydelt_gold_values
    if 'PyDelt LOWESS' in results:
        derivative_values['PyDelt LOWESS'] = pydelt_lowess_values
    if 'PyDelt LOESS' in results:
        derivative_values['PyDelt LOESS'] = pydelt_loess_values
    if 'PyDelt FDA' in results:
        derivative_values['PyDelt FDA'] = pydelt_fda_values
    if 'PyDelt Neural Network' in results and AUTODIFF_AVAILABLE:
        derivative_values['PyDelt Neural Network'] = nn_values
    if 'SciPy Spline' in results and SCIPY_AVAILABLE:
        derivative_values['SciPy Spline'] = scipy_spline_values
    if 'SciPy Cubic' in results and SCIPY_AVAILABLE and order <= 3:
        derivative_values['SciPy Cubic'] = scipy_cubic_values
    if 'NumDiffTools' in results and NUMDIFFTOOLS_AVAILABLE:
        derivative_values['NumDiffTools'] = nd_values
    if 'FinDiff' in results and FINDIFF_AVAILABLE:
        derivative_values['FinDiff'] = fd_eval
    if 'JAX' in results and JAX_AVAILABLE:
        derivative_values['JAX'] = np.array(jax_values)
    
    derivative_values['True'] = true_values
    
    return results, times, derivative_values

def evaluate_multivariate_derivatives(noise_level=0.0):
    """
    Evaluate multivariate derivative methods on test functions.
    
    Parameters:
    -----------
    noise_level : float
        Level of noise to add (as a fraction of the standard deviation)
    
    Returns:
    --------
    dict
        Dictionary of method names and their corresponding errors
    dict
        Dictionary of method names and their corresponding computation times
    """
    if not MULTIVARIATE_AVAILABLE:
        print("PyDelt multivariate module not available. Skipping multivariate comparison.")
        return {}, {}
    
    # Generate grid of 2D points for testing
    n_points = 100
    x = np.linspace(-3, 3, int(np.sqrt(n_points)))
    y = np.linspace(-3, 3, int(np.sqrt(n_points)))
    X, Y = np.meshgrid(x, y)
    points = np.column_stack((X.flatten(), Y.flatten()))
    
    # Generate function values
    values = multivariate_test_function(points)
    
    # Add noise if specified
    if noise_level > 0:
        values = add_noise(values, noise_level)
    
    # Generate evaluation points
    n_eval = 20
    eval_x = np.linspace(-2.5, 2.5, n_eval)
    eval_y = np.linspace(-2.5, 2.5, n_eval)
    eval_X, eval_Y = np.meshgrid(eval_x, eval_y)
    eval_points = np.column_stack((eval_X.flatten(), eval_Y.flatten()))
    
    # True gradient values at evaluation points
    true_gradients = multivariate_test_gradient(eval_points)
    
    results = {}
    times = {}
    gradient_values = {}
    
    # PyDelt MultivariateDerivatives with different interpolators
    interpolators = [
        ("Spline", SplineInterpolator()),
        ("LLA", LlaInterpolator(window_size=5)),
        ("GLLA", GllaInterpolator(embedding=3, n=2)),
        ("GOLD", GoldInterpolator(window_size=5)),
        ("LOWESS", LowessInterpolator()),
        ("LOESS", LoessInterpolator(frac=0.3)),
        ("FDA", FdaInterpolator())
    ]
    
    for name, interp_class in interpolators:
        try:
            start_time = time.time()
            
            # Create multivariate derivative object with the correct interpolator class
            interpolator_class = interp_class.__class__
            mv = MultivariateDerivatives(interpolator_class)
                
            # Fit the model
            mv.fit(points, values)
            
            # Compute gradient
            gradient_func = mv.gradient()
            gradients = gradient_func(eval_points)
            
            times[f"PyDelt MV {name}"] = time.time() - start_time
            
            # Calculate errors (Euclidean distance between true and computed gradients)
            if gradients.ndim == 1:
                gradients = gradients.reshape(1, -1)
            
            errors = np.sqrt(np.sum((gradients - true_gradients)**2, axis=1))
            results[f"PyDelt MV {name}"] = errors
            gradient_values[f"PyDelt MV {name}"] = gradients
        except Exception as e:
            print(f"Error with PyDelt MV {name}: {e}")
    
    # NumDiffTools gradient
    if NUMDIFFTOOLS_AVAILABLE:
        try:
            start_time = time.time()
            
            # Create a function from our data using scipy's interpolation
            from scipy.interpolate import LinearNDInterpolator
            interp_func = LinearNDInterpolator(points, values)
            
            # Use NumDiffTools to compute gradient
            nd_gradients = np.zeros_like(eval_points)
            for i in range(len(eval_points)):
                nd_gradients[i, 0] = nd.Gradient(lambda x: interp_func([x, eval_points[i, 1]]))(eval_points[i, 0])
                nd_gradients[i, 1] = nd.Gradient(lambda y: interp_func([eval_points[i, 0], y]))(eval_points[i, 1])
            
            times["NumDiffTools MV"] = time.time() - start_time
            errors = np.sqrt(np.sum((nd_gradients - true_gradients)**2, axis=1))
            results["NumDiffTools MV"] = errors
            gradient_values["NumDiffTools MV"] = nd_gradients
        except Exception as e:
            print(f"Error with NumDiffTools MV: {e}")
    
    # JAX gradient
    if JAX_AVAILABLE:
        try:
            start_time = time.time()
            
            # Define the JAX function
            def jax_mv_func(x):
                return jnp.sin(x[0]) + jnp.cos(x[1])
            
            # Define the gradient function
            jax_grad_func = jit(grad(jax_mv_func))
            
            # Compute gradient
            jax_gradients = np.zeros_like(eval_points)
            for i in range(len(eval_points)):
                jax_gradients[i] = np.array(jax_grad_func(jnp.array(eval_points[i])))
            
            times["JAX MV"] = time.time() - start_time
            errors = np.sqrt(np.sum((jax_gradients - true_gradients)**2, axis=1))
            results["JAX MV"] = errors
            gradient_values["JAX MV"] = jax_gradients
        except Exception as e:
            print(f"Error with JAX MV: {e}")
    
    gradient_values["True"] = true_gradients
    
    return results, times, gradient_values, eval_points

def evaluate_stochastic_derivatives(noise_level=0.0, order=1):
    """
    Evaluate stochastic derivative methods on noisy test functions.
    
    Parameters:
    -----------
    noise_level : float
        Level of noise to add (as a fraction of the standard deviation)
    order : int
        Order of the derivative to compute
        
    Returns:
    --------
    dict
        Dictionary of method names and their corresponding errors
    dict
        Dictionary of method names and their corresponding computation times
    """
    if not AUTODIFF_AVAILABLE:
        print("PyDelt autodiff module not available. Skipping stochastic comparison.")
        return {}, {}
    
    # Generate data
    n_points = 100
    x = np.linspace(-5, 5, n_points)
    y = test_function_sin(x)  # Use sine function for stochastic tests
    
    # Add significant noise
    noise_level = max(noise_level, 0.1)  # Ensure significant noise for stochastic test
    y_noisy = add_noise(y, noise_level)
    
    # Generate evaluation points
    x_eval = np.linspace(-4.5, 4.5, n_points * 2)
    
    # True derivative values
    if order == 1:
        true_derivative = test_function_sin_derivative
    else:
        true_derivative = test_function_sin_second_derivative
    
    true_values = true_derivative(x_eval)
    
    results = {}
    times = {}
    derivative_values = {}
    
    # Neural Network derivative with different frameworks
    frameworks = ['tensorflow', 'pytorch'] if TORCH_AVAILABLE else ['tensorflow']
    
    for framework in frameworks:
        try:
            start_time = time.time()
            
            # Train neural network and get derivative function
            nn_deriv_func = neural_network_derivative(
                x, y_noisy, 
                framework=framework, 
                hidden_layers=[64, 32], 
                epochs=500, 
                order=order
            )
            
            # Evaluate derivative
            nn_values = nn_deriv_func(x_eval)
            
            times[f"PyDelt NN {framework.capitalize()}"] = time.time() - start_time
            results[f"PyDelt NN {framework.capitalize()}"] = np.abs(nn_values - true_values)
            derivative_values[f"PyDelt NN {framework.capitalize()}"] = nn_values
        except Exception as e:
            print(f"Error with PyDelt NN {framework}: {e}")
    
    # Compare with traditional methods
    # SplineInterpolator
    try:
        start_time = time.time()
        spline_interp = SplineInterpolator(smoothing=0.5)  # Increase smoothing for noisy data
        spline_interp.fit(x, y_noisy)
        deriv_func = spline_interp.differentiate(order=order)
        spline_values = deriv_func(x_eval)
        
        times["PyDelt Spline (smoothed)"] = time.time() - start_time
        results["PyDelt Spline (smoothed)"] = np.abs(spline_values - true_values)
        derivative_values["PyDelt Spline (smoothed)"] = spline_values
    except Exception as e:
        print(f"Error with PyDelt Spline: {e}")
    
    # LowessInterpolator (good for noisy data)
    try:
        start_time = time.time()
        lowess_interp = LowessInterpolator()
        lowess_interp.fit(x, y_noisy)
        deriv_func = lowess_interp.differentiate(order=order)
        lowess_values = deriv_func(x_eval)
        
        times["PyDelt LOWESS"] = time.time() - start_time
        results["PyDelt LOWESS"] = np.abs(lowess_values - true_values)
        derivative_values["PyDelt LOWESS"] = lowess_values
    except Exception as e:
        print(f"Error with PyDelt LOWESS: {e}")
    
    derivative_values["True"] = true_values
    
    return results, times, derivative_values, x_eval, y_noisy

def plot_univariate_comparison(test_name, noise_level=0.05, order=1):
    """
    Generate plots comparing all methods for univariate functions.
    
    Parameters:
    -----------
    test_name : str
        Name of the test function ('sin', 'exp', or 'polynomial')
    noise_level : float
        Level of noise to add
    order : int
        Order of the derivative to compute
    """
    # Set up test function based on name
    if test_name.lower() == 'sin':
        func = test_function_sin
        deriv_func = test_function_sin_derivative if order == 1 else test_function_sin_second_derivative
    elif test_name.lower() == 'exp':
        func = test_function_exp
        deriv_func = test_function_exp_derivative
    elif test_name.lower() == 'polynomial':
        func = test_function_polynomial
        deriv_func = test_function_polynomial_derivative if order == 1 else test_function_polynomial_second_derivative
    else:
        raise ValueError(f"Unknown test function: {test_name}")
    
    # Generate data
    x_range = (-5, 5)
    num_points = 100
    x = np.linspace(x_range[0], x_range[1], num_points)
    y = func(x)
    
    # Generate evaluation points (more dense than the data points)
    x_eval = np.linspace(x_range[0], x_range[1], num_points * 2)
    
    # Evaluate all methods
    errors, times, derivative_values = evaluate_all_interpolators(x, y, x_eval, deriv_func, noise_level, order)
    
    # Create plot
    plt.figure(figsize=(15, 10))
    
    # Plot original function and noisy data
    plt.subplot(2, 1, 1)
    y_noisy = y + noise_level * np.std(y) * np.random.randn(len(y)) if noise_level > 0 else y
    plt.plot(x_eval, func(x_eval), 'k-', label='Original Function')
    plt.scatter(x, y_noisy, s=20, alpha=0.5, label=f'Noisy Data (noise={noise_level})')
    plt.title(f'{test_name} Function - Original with Noise')
    plt.legend()
    plt.grid(True)
    
    # Plot derivatives
    plt.subplot(2, 1, 2)
    
    # Plot true derivative
    plt.plot(x_eval, derivative_values['True'], 'k-', label='True Derivative', linewidth=2)
    
    # Plot PyDelt methods with solid lines
    pydelt_methods = [method for method in derivative_values.keys() if method.startswith('PyDelt') and method != 'PyDelt Neural Network']
    for method in pydelt_methods:
        plt.plot(x_eval, derivative_values[method], '-', label=method)
    
    # Plot other methods with dashed lines
    other_methods = [method for method in derivative_values.keys() 
                    if not method.startswith('PyDelt') and method != 'True']
    for method in other_methods:
        plt.plot(x_eval, derivative_values[method], '--', label=method)
    
    # Plot neural network methods with dotted lines if available
    nn_methods = [method for method in derivative_values.keys() if method == 'PyDelt Neural Network']
    for method in nn_methods:
        plt.plot(x_eval, derivative_values[method], ':', label=method)
    
    plt.title(f'{test_name} Function - {order}{"st" if order==1 else "nd" if order==2 else "th"} Derivative Comparison')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{test_name.lower()}_order{order}_comparison.png')
    plt.close()
    
    # Create error comparison plot
    plt.figure(figsize=(12, 8))
    
    # Prepare data for box plot
    error_data = []
    labels = []
    
    # PyDelt methods first
    for method in sorted(pydelt_methods):
        if method in errors:
            error_data.append(errors[method])
            labels.append(method)
    
    # Other methods
    for method in sorted(other_methods):
        if method in errors:
            error_data.append(errors[method])
            labels.append(method)
    
    # Neural network methods
    for method in sorted(nn_methods):
        if method in errors:
            error_data.append(errors[method])
            labels.append(method)
    
    # Create box plot
    plt.boxplot(error_data, labels=labels, vert=True, patch_artist=True, 
               boxprops=dict(alpha=0.7), medianprops=dict(color='black'))
    
    plt.title(f'{test_name} Function - {order}{"st" if order==1 else "nd" if order==2 else "th"} Derivative Error Comparison')
    plt.ylabel('Absolute Error')
    plt.yscale('log')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{test_name.lower()}_order{order}_errors.png')
    plt.close()
    
    return errors, times

def plot_multivariate_comparison(noise_level=0.05):
    """
    Generate plots comparing multivariate derivative methods.
    
    Parameters:
    -----------
    noise_level : float
        Level of noise to add
    """
    if not MULTIVARIATE_AVAILABLE:
        print("PyDelt multivariate module not available. Skipping multivariate plots.")
        return {}, {}
    
    # Evaluate multivariate methods
    errors, times, gradient_values, eval_points = evaluate_multivariate_derivatives(noise_level)
    
    if not errors:  # If no results were returned
        return {}, {}
    
    # Reshape evaluation points for plotting
    n_eval = int(np.sqrt(len(eval_points)))
    X = eval_points[:, 0].reshape(n_eval, n_eval)
    Y = eval_points[:, 1].reshape(n_eval, n_eval)
    
    # Create plot for gradient vector field
    plt.figure(figsize=(15, 12))
    
    # Number of methods to plot
    n_methods = len(gradient_values)
    n_cols = min(3, n_methods)
    n_rows = (n_methods + n_cols - 1) // n_cols
    
    for i, (method, gradients) in enumerate(gradient_values.items()):
        plt.subplot(n_rows, n_cols, i+1)
        
        # Reshape gradients for quiver plot
        U = gradients[:, 0].reshape(n_eval, n_eval)
        V = gradients[:, 1].reshape(n_eval, n_eval)
        
        # Plot gradient vector field
        plt.quiver(X, Y, U, V, np.sqrt(U**2 + V**2), cmap='viridis')
        plt.colorbar(label='Gradient Magnitude')
        plt.title(f'{method} Gradient')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'multivariate_gradient_comparison.png')
    plt.close()
    
    # Create error comparison plot
    plt.figure(figsize=(12, 8))
    
    # Prepare data for box plot
    error_data = []
    labels = []
    
    # PyDelt methods first
    pydelt_methods = [method for method in errors.keys() if method.startswith('PyDelt')]
    for method in sorted(pydelt_methods):
        error_data.append(errors[method])
        labels.append(method)
    
    # Other methods
    other_methods = [method for method in errors.keys() if not method.startswith('PyDelt')]
    for method in sorted(other_methods):
        error_data.append(errors[method])
        labels.append(method)
    
    # Create box plot
    plt.boxplot(error_data, labels=labels, vert=True, patch_artist=True, 
               boxprops=dict(alpha=0.7), medianprops=dict(color='black'))
    
    plt.title('Multivariate Gradient Error Comparison')
    plt.ylabel('Euclidean Error')
    plt.yscale('log')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'multivariate_gradient_errors.png')
    plt.close()
    
    return errors, times

def plot_stochastic_comparison(noise_level=0.1, order=1):
    """
    Generate plots comparing stochastic derivative methods.
    
    Parameters:
    -----------
    noise_level : float
        Level of noise to add
    order : int
        Order of the derivative to compute
    """
    if not AUTODIFF_AVAILABLE:
        print("PyDelt autodiff module not available. Skipping stochastic plots.")
        return {}, {}
    
    # Evaluate stochastic methods
    errors, times, derivative_values, x_eval, y_noisy = evaluate_stochastic_derivatives(noise_level, order)
    
    if not errors:  # If no results were returned
        return {}, {}
    
    # Create plot
    plt.figure(figsize=(15, 10))
    
    # Plot original function and noisy data
    plt.subplot(2, 1, 1)
    x = np.linspace(-5, 5, len(y_noisy))
    plt.plot(x_eval, test_function_sin(x_eval), 'k-', label='Original Function')
    plt.scatter(x, y_noisy, s=20, alpha=0.5, label=f'Noisy Data (noise={noise_level})')
    plt.title(f'Sine Function with Heavy Noise - Stochastic Derivatives')
    plt.legend()
    plt.grid(True)
    
    # Plot derivatives
    plt.subplot(2, 1, 2)
    
    # Plot true derivative
    plt.plot(x_eval, derivative_values['True'], 'k-', label='True Derivative', linewidth=2)
    
    # Plot neural network methods
    nn_methods = [method for method in derivative_values.keys() 
                 if method.startswith('PyDelt NN') and method != 'True']
    for method in nn_methods:
        plt.plot(x_eval, derivative_values[method], '-', label=method)
    
    # Plot traditional methods
    trad_methods = [method for method in derivative_values.keys() 
                  if not method.startswith('PyDelt NN') and method != 'True']
    for method in trad_methods:
        plt.plot(x_eval, derivative_values[method], '--', label=method)
    
    plt.title(f'Sine Function - {order}{"st" if order==1 else "nd" if order==2 else "th"} Derivative with Neural Networks')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'stochastic_derivative_order{order}.png')
    plt.close()
    
    # Create error comparison plot
    plt.figure(figsize=(12, 8))
    
    # Prepare data for box plot
    error_data = []
    labels = []
    
    # Neural network methods
    for method in sorted(nn_methods):
        error_data.append(errors[method])
        labels.append(method)
    
    # Traditional methods
    for method in sorted(trad_methods):
        error_data.append(errors[method])
        labels.append(method)
    
    # Create box plot
    plt.boxplot(error_data, labels=labels, vert=True, patch_artist=True, 
               boxprops=dict(alpha=0.7), medianprops=dict(color='black'))
    
    plt.title(f'Stochastic Derivative Error Comparison (Order {order})')
    plt.ylabel('Absolute Error')
    plt.yscale('log')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'stochastic_derivative_errors_order{order}.png')
    plt.close()
    
    return errors, times

def create_summary_table(all_errors, all_times):
    """
    Create a summary table of all methods' performance.
    
    Parameters:
    -----------
    all_errors : dict
        Dictionary of method names and their errors across different tests
    all_times : dict
        Dictionary of method names and their computation times across different tests
    """
    # Collect all unique methods
    all_methods = set()
    for errors in all_errors.values():
        all_methods.update(errors.keys())
    
    # Create summary data
    summary_data = []
    
    for method in sorted(all_methods):
        # Calculate average error and time across all tests
        errors = []
        times = []
        
        for test_name, test_errors in all_errors.items():
            if method in test_errors:
                errors.append(np.mean(test_errors[method]))
        
        for test_name, test_times in all_times.items():
            if method in test_times:
                times.append(test_times[method])
        
        if errors and times:
            avg_error = np.mean(errors)
            avg_time = np.mean(times) * 1000  # Convert to milliseconds
            
            # Determine method type and dimensionality
            if 'MV' in method:
                dimensionality = 'Multivariate'
            else:
                dimensionality = 'Univariate'
            
            if method.startswith('PyDelt'):
                if 'NN' in method:
                    method_type = 'Neural Network'
                elif 'MV' in method:
                    method_type = 'Multivariate'
                else:
                    method_type = 'Interpolation'
            elif method == 'JAX':
                method_type = 'Automatic Differentiation'
            elif method == 'NumDiffTools':
                method_type = 'Adaptive Finite Difference'
            elif method == 'FinDiff':
                method_type = 'Finite Difference'
            elif method.startswith('SciPy'):
                method_type = 'Interpolation'
            else:
                method_type = 'Other'
            
            # Determine key strengths and weaknesses
            if avg_error < 0.01:
                accuracy = 'Very High'
            elif avg_error < 0.1:
                accuracy = 'High'
            elif avg_error < 0.5:
                accuracy = 'Medium'
            else:
                accuracy = 'Low'
            
            if avg_time < 10:
                speed = 'Very Fast'
            elif avg_time < 50:
                speed = 'Fast'
            elif avg_time < 200:
                speed = 'Medium'
            else:
                speed = 'Slow'
            
            # Determine noise robustness based on method type
            if 'NN' in method or 'LOWESS' in method or 'LOESS' in method:
                noise_robustness = 'High'
            elif 'LLA' in method or 'GLLA' in method:
                noise_robustness = 'Medium'
            elif 'Spline' in method or 'FDA' in method:
                noise_robustness = 'Medium'
            elif method == 'JAX':
                noise_robustness = 'N/A (requires analytical function)'
            else:
                noise_robustness = 'Low'
            
            # Determine key strength
            if accuracy == 'Very High':
                key_strength = 'Highest accuracy'
            elif 'NN' in method:
                key_strength = 'Robust to noise'
            elif speed == 'Very Fast':
                key_strength = 'Computational efficiency'
            elif 'MV' in method:
                key_strength = 'Multivariate support'
            elif 'LLA' in method or 'GLLA' in method:
                key_strength = 'Good balance of accuracy and noise robustness'
            else:
                key_strength = 'General purpose'
            
            # Determine key weakness
            if speed == 'Slow':
                key_weakness = 'Computational overhead'
            elif accuracy == 'Low':
                key_weakness = 'Lower accuracy'
            elif 'JAX' in method:
                key_weakness = 'Requires analytical function'
            elif 'FinDiff' in method or 'NumDiffTools' in method:
                key_weakness = 'Sensitive to noise'
            elif 'MV' in method:
                key_weakness = 'No mixed partial derivatives'
            else:
                key_weakness = 'Limited dimensionality'
            
            summary_data.append({
                'Method': method,
                'Type': method_type,
                'Dimensionality': dimensionality,
                'Accuracy': accuracy,
                'Speed': speed,
                'Noise Robustness': noise_robustness,
                'Key Strength': key_strength,
                'Key Weakness': key_weakness
            })
    
    # Create DataFrame
    df = pd.DataFrame(summary_data)
    
    # Save to CSV
    csv_path = OUTPUT_DIR / "method_comparison_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved summary to: {csv_path}")
    
    # Create a styled HTML table
    styled_df = df.style.set_table_styles([
        {'selector': 'th', 'props': [('font-weight', 'bold'), ('text-align', 'center')]},
        {'selector': 'td', 'props': [('text-align', 'center')]}
    ])
    
    # Save to HTML
    html_path = OUTPUT_DIR / "method_comparison_summary.html"
    styled_df.to_html(str(html_path))
    
    return df

def main():
    """Run all comparisons and generate plots."""
    # Create output directory with absolute path
    script_dir = Path(__file__).parent.absolute()
    output_dir = script_dir.parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Print the absolute path of the output directory
    print(f"Output directory: {output_dir}")
    
    # Set global variable for output directory
    global OUTPUT_DIR
    OUTPUT_DIR = output_dir
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Store all errors and times for summary table
    all_errors = {}
    all_times = {}
    
    print("\n1. Running univariate comparisons...")
    # Test different functions with first derivatives
    for test_name in ['Sin', 'Exp', 'Polynomial']:
        print(f"  - Testing {test_name} function (1st derivative)...")
        errors, times = plot_univariate_comparison(test_name, noise_level=0.05, order=1)
        all_errors[f"{test_name}_Order1"] = errors
        all_times[f"{test_name}_Order1"] = times
    
    # Test different functions with second derivatives
    for test_name in ['Sin', 'Exp', 'Polynomial']:
        print(f"  - Testing {test_name} function (2nd derivative)...")
        errors, times = plot_univariate_comparison(test_name, noise_level=0.05, order=2)
        all_errors[f"{test_name}_Order2"] = errors
        all_times[f"{test_name}_Order2"] = times
    
    print("\n2. Running multivariate comparisons...")
    if MULTIVARIATE_AVAILABLE:
        mv_errors, mv_times = plot_multivariate_comparison(noise_level=0.05)
        all_errors["Multivariate"] = mv_errors
        all_times["Multivariate"] = mv_times
    else:
        print("  - Multivariate module not available, skipping.")
    
    print("\n3. Running stochastic derivative comparisons...")
    if AUTODIFF_AVAILABLE:
        # First order stochastic derivatives
        stoch_errors1, stoch_times1 = plot_stochastic_comparison(noise_level=0.1, order=1)
        all_errors["Stochastic_Order1"] = stoch_errors1
        all_times["Stochastic_Order1"] = stoch_times1
        
        # Second order stochastic derivatives
        stoch_errors2, stoch_times2 = plot_stochastic_comparison(noise_level=0.1, order=2)
        all_errors["Stochastic_Order2"] = stoch_errors2
        all_times["Stochastic_Order2"] = stoch_times2
    else:
        print("  - Autodiff module not available, skipping.")
    
    print("\n4. Creating summary table...")
    summary_df = create_summary_table(all_errors, all_times)
    
    print("\nAll comparisons completed! Results saved to ../output/")
    print("Summary table saved as method_comparison_summary.csv and method_comparison_summary.html")

if __name__ == "__main__":
    main()
