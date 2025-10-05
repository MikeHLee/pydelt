#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Comparison between PyDelt and JAX for numerical differentiation.

This script compares the accuracy and performance of PyDelt's differentiation methods
with JAX's automatic differentiation for computing derivatives of known test functions.
"""

import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
import time
import sys
import os

# Add the parent directory to the path so we can import pydelt
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.pydelt.interpolation import SplineInterpolator, LlaInterpolator, GllaInterpolator
from src.pydelt.autodiff import neural_network_derivative

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
    return jnp.sin(x)

def jax_exp(x):
    return jnp.exp(x)

def jax_polynomial(x):
    return x**3 - 2*x**2 + 3*x - 1

def add_noise(y, noise_level=0.01):
    """Add Gaussian noise to the data."""
    return y + noise_level * np.std(y) * np.random.randn(len(y))

def evaluate_methods(x, y, x_eval, true_derivative, jax_func, noise_level=0.0, order=1):
    """
    Evaluate different differentiation methods and return their errors.
    
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
    jax_func : callable
        JAX version of the test function
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
    times['PyDelt SplineInterpolator'] = time.time() - start_time
    results['PyDelt SplineInterpolator'] = np.abs(pydelt_spline_values - true_values)
    
    # PyDelt LlaInterpolator
    start_time = time.time()
    lla_interp = LlaInterpolator(window_size=5)
    lla_interp.fit(x, y_noisy)
    deriv_func = lla_interp.differentiate(order=order)
    pydelt_lla_values = deriv_func(x_eval)
    times['PyDelt LlaInterpolator'] = time.time() - start_time
    results['PyDelt LlaInterpolator'] = np.abs(pydelt_lla_values - true_values)
    
    # PyDelt GllaInterpolator
    start_time = time.time()
    glla_interp = GllaInterpolator(embedding=3, n=2)
    glla_interp.fit(x, y_noisy)
    deriv_func = glla_interp.differentiate(order=order)
    pydelt_glla_values = deriv_func(x_eval)
    times['PyDelt GllaInterpolator'] = time.time() - start_time
    results['PyDelt GllaInterpolator'] = np.abs(pydelt_glla_values - true_values)
    
    # PyDelt Neural Network Derivative
    try:
        start_time = time.time()
        nn_derivative = neural_network_derivative(x, y_noisy, framework='pytorch')
        nn_values = nn_derivative(x_eval)
        times['PyDelt Neural Network'] = time.time() - start_time
        results['PyDelt Neural Network'] = np.abs(nn_values - true_values)
    except Exception as e:
        print(f"Warning: PyDelt Neural Network failed with error: {e}")
        times['PyDelt Neural Network'] = float('nan')
        results['PyDelt Neural Network'] = np.full_like(true_values, float('nan'))
    
    # JAX Automatic Differentiation
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
    
    times['JAX Automatic Differentiation'] = time.time() - start_time
    results['JAX Automatic Differentiation'] = np.abs(np.array(jax_values) - true_values)
    
    # JAX Vectorized (for performance comparison)
    start_time = time.time()
    jax_derivative_vmap = jit(vmap(jax_derivative))
    jax_values_vmap = jax_derivative_vmap(jax_x_eval)
    times['JAX Vectorized'] = time.time() - start_time
    results['JAX Vectorized'] = np.abs(np.array(jax_values_vmap) - true_values)
    
    return results, times

def run_comparison(test_name, func, deriv_func, second_deriv_func, jax_func,
                  x_range=(-5, 5), num_points=100, noise_levels=[0.0, 0.01, 0.05, 0.1]):
    """
    Run a comprehensive comparison for a test function.
    
    Parameters:
    -----------
    test_name : str
        Name of the test function for display
    func : callable
        The test function
    deriv_func : callable
        The analytical first derivative function
    second_deriv_func : callable
        The analytical second derivative function
    jax_func : callable
        JAX version of the test function
    x_range : tuple
        Range of x values to test
    num_points : int
        Number of data points to use
    noise_levels : list
        List of noise levels to test
    """
    print(f"\n{'='*80}\nTesting {test_name}\n{'='*80}")
    
    # Generate data
    x = np.linspace(x_range[0], x_range[1], num_points)
    y = func(x)
    
    # Generate evaluation points (more dense than the data points)
    x_eval = np.linspace(x_range[0], x_range[1], num_points * 2)
    
    # First derivative tests
    print("\nFirst Derivative Tests:")
    print(f"{'Noise Level':<15} {'Method':<25} {'Max Error':<15} {'Mean Error':<15} {'Time (ms)':<15}")
    print("-" * 85)
    
    for noise_level in noise_levels:
        errors, times = evaluate_methods(x, y, x_eval, deriv_func, jax_func, noise_level=noise_level, order=1)
        
        for method, error in errors.items():
            if np.isnan(error).any():
                continue  # Skip methods that failed
            max_error = np.max(error)
            mean_error = np.mean(error)
            time_ms = times[method] * 1000  # Convert to milliseconds
            print(f"{noise_level:<15.2f} {method:<25} {max_error:<15.6f} {mean_error:<15.6f} {time_ms:<15.2f}")
    
    # Second derivative tests
    print("\nSecond Derivative Tests:")
    print(f"{'Noise Level':<15} {'Method':<25} {'Max Error':<15} {'Mean Error':<15} {'Time (ms)':<15}")
    print("-" * 85)
    
    for noise_level in noise_levels:
        errors, times = evaluate_methods(x, y, x_eval, second_deriv_func, jax_func, noise_level=noise_level, order=2)
        
        for method, error in errors.items():
            if np.isnan(error).any():
                continue  # Skip methods that failed
            max_error = np.max(error)
            mean_error = np.mean(error)
            time_ms = times[method] * 1000  # Convert to milliseconds
            print(f"{noise_level:<15.2f} {method:<25} {max_error:<15.6f} {mean_error:<15.6f} {time_ms:<15.2f}")

def plot_comparison(test_name, func, deriv_func, jax_func, x_range=(-5, 5), num_points=100, noise_level=0.05):
    """
    Plot a visual comparison of different methods.
    
    Parameters:
    -----------
    test_name : str
        Name of the test function for display
    func : callable
        The test function
    deriv_func : callable
        The analytical first derivative function
    jax_func : callable
        JAX version of the test function
    x_range : tuple
        Range of x values to test
    num_points : int
        Number of data points to use
    noise_level : float
        Level of noise to add
    """
    # Generate data
    x = np.linspace(x_range[0], x_range[1], num_points)
    y = func(x)
    y_noisy = add_noise(y, noise_level)
    
    # Generate evaluation points (more dense than the data points)
    x_eval = np.linspace(x_range[0], x_range[1], num_points * 2)
    true_deriv = deriv_func(x_eval)
    
    # PyDelt SplineInterpolator
    spline_interp = SplineInterpolator()
    spline_interp.fit(x, y_noisy)
    pydelt_spline_values = spline_interp.differentiate(order=1)(x_eval)
    
    # PyDelt LlaInterpolator
    lla_interp = LlaInterpolator(window_size=5)
    lla_interp.fit(x, y_noisy)
    pydelt_lla_values = lla_interp.differentiate(order=1)(x_eval)
    
    # PyDelt GllaInterpolator
    glla_interp = GllaInterpolator(embedding=3, n=2)
    glla_interp.fit(x, y_noisy)
    pydelt_glla_values = glla_interp.differentiate(order=1)(x_eval)
    
    # PyDelt Neural Network
    try:
        nn_derivative = neural_network_derivative(x, y_noisy, framework='pytorch')
        nn_values = nn_derivative(x_eval)
    except Exception as e:
        print(f"Warning: PyDelt Neural Network failed with error: {e}")
        nn_values = np.full_like(x_eval, np.nan)
    
    # JAX Automatic Differentiation
    jax_derivative = jit(grad(jax_func))
    jax_x_eval = jnp.array(x_eval)
    jax_values = jnp.array([jax_derivative(x_i) for x_i in jax_x_eval])
    
    # JAX Vectorized
    jax_derivative_vmap = jit(vmap(jax_derivative))
    jax_values_vmap = jax_derivative_vmap(jax_x_eval)
    
    # Create plot
    plt.figure(figsize=(15, 10))
    
    # Plot original function and noisy data
    plt.subplot(2, 1, 1)
    plt.plot(x_eval, func(x_eval), 'k-', label='Original Function')
    plt.scatter(x, y_noisy, s=20, alpha=0.5, label=f'Noisy Data (noise={noise_level})')
    plt.title(f'{test_name} - Original Function with Noise')
    plt.legend()
    plt.grid(True)
    
    # Plot derivatives
    plt.subplot(2, 1, 2)
    plt.plot(x_eval, true_deriv, 'k-', label='True Derivative')
    plt.plot(x_eval, pydelt_spline_values, label='PyDelt Spline')
    plt.plot(x_eval, pydelt_lla_values, label='PyDelt LLA')
    plt.plot(x_eval, pydelt_glla_values, label='PyDelt GLLA')
    
    if not np.isnan(nn_values).any():
        plt.plot(x_eval, nn_values, label='PyDelt Neural Network')
    
    plt.plot(x_eval, np.array(jax_values), label='JAX Automatic Differentiation')
    plt.title(f'{test_name} - First Derivative Comparison')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'../output/{test_name.lower().replace(" ", "_")}_jax_comparison.png')
    plt.close()

def dimensionality_scaling_test():
    """
    Test how different methods scale with increasing dimensionality.
    """
    print("\n" + "=" * 80)
    print("Dimensionality Scaling Test")
    print("=" * 80)
    
    dimensions = [2, 5, 10, 20]
    num_points = 100
    
    print(f"{'Dimensions':<15} {'Method':<25} {'Time (ms)':<15}")
    print("-" * 55)
    
    for dim in dimensions:
        # Generate random data for the given dimensionality
        np.random.seed(42)  # For reproducibility
        X = np.random.rand(num_points, dim)
        
        # Simple quadratic function: sum of squares
        def quad_func(x):
            return np.sum(x**2)
        
        def jax_quad_func(x):
            return jnp.sum(x**2)
        
        Y = np.array([quad_func(x) for x in X])
        
        # Evaluation points
        X_eval = np.random.rand(10, dim)  # Fewer points for evaluation
        
        # PyDelt multivariate approach (if available)
        try:
            from src.pydelt.multivariate import MultivariateDerivatives
            
            start_time = time.time()
            mv = MultivariateDerivatives(SplineInterpolator)
            mv.fit(X, Y)
            gradient_func = mv.gradient()
            _ = gradient_func(X_eval)
            pydelt_time = time.time() - start_time
            print(f"{dim:<15d} {'PyDelt Multivariate':<25} {pydelt_time * 1000:<15.2f}")
        except (ImportError, Exception) as e:
            print(f"{dim:<15d} {'PyDelt Multivariate':<25} {'Failed: ' + str(e):<15}")
        
        # JAX gradient
        start_time = time.time()
        jax_grad_func = grad(jax_quad_func)
        _ = [jax_grad_func(jnp.array(x)) for x in X_eval]
        jax_time = time.time() - start_time
        print(f"{dim:<15d} {'JAX Gradient':<25} {jax_time * 1000:<15.2f}")
        
        # JAX vectorized gradient (should be faster)
        start_time = time.time()
        jax_grad_vmap = jit(vmap(grad(jax_quad_func)))
        _ = jax_grad_vmap(jnp.array(X_eval))
        jax_vmap_time = time.time() - start_time
        print(f"{dim:<15d} {'JAX Vectorized Gradient':<25} {jax_vmap_time * 1000:<15.2f}")

def main():
    """Run all comparisons."""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run numerical comparisons
    run_comparison("Sine Function", test_function_sin, test_function_sin_derivative, 
                  test_function_sin_second_derivative, jax_sin)
    
    run_comparison("Exponential Function", test_function_exp, test_function_exp_derivative, 
                  test_function_exp_derivative, jax_exp)
    
    run_comparison("Polynomial Function", test_function_polynomial, test_function_polynomial_derivative, 
                  test_function_polynomial_second_derivative, jax_polynomial)
    
    # Run visual comparisons
    plot_comparison("Sine Function", test_function_sin, test_function_sin_derivative, jax_sin)
    plot_comparison("Exponential Function", test_function_exp, test_function_exp_derivative, jax_exp)
    plot_comparison("Polynomial Function", test_function_polynomial, test_function_polynomial_derivative, jax_polynomial)
    
    # Run dimensionality scaling test
    dimensionality_scaling_test()

if __name__ == "__main__":
    main()
