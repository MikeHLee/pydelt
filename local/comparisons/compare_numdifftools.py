#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Comparison between PyDelt and NumDiffTools for numerical differentiation.

This script compares the accuracy and performance of PyDelt's differentiation methods
with NumDiffTools' adaptive finite difference methods for computing derivatives
of known test functions.
"""

import numpy as np
import matplotlib.pyplot as plt
import numdifftools as nd
import time
import sys
import os

# Add the parent directory to the path so we can import pydelt
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.pydelt.interpolation import SplineInterpolator, LlaInterpolator, GllaInterpolator

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

def add_noise(y, noise_level=0.01):
    """Add Gaussian noise to the data."""
    return y + noise_level * np.std(y) * np.random.randn(len(y))

def evaluate_methods(x, y, x_eval, true_derivative, func, noise_level=0.0, order=1):
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
    func : callable
        Original function (needed for NumDiffTools)
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
    
    # NumDiffTools - Derivative
    start_time = time.time()
    # For NumDiffTools, we need to create a function from our noisy data
    # We'll use a simple linear interpolation for this
    from scipy.interpolate import interp1d
    noisy_func = interp1d(x, y_noisy, bounds_error=False, fill_value="extrapolate")
    
    # Use NumDiffTools with the interpolated function
    nd_derivative = nd.Derivative(noisy_func, n=order)
    nd_values = nd_derivative(x_eval)
    times['NumDiffTools Derivative'] = time.time() - start_time
    results['NumDiffTools Derivative'] = np.abs(nd_values - true_values)
    
    # NumDiffTools with the original function (for comparison)
    start_time = time.time()
    nd_exact = nd.Derivative(func, n=order)
    nd_exact_values = nd_exact(x_eval)
    times['NumDiffTools (Exact Function)'] = time.time() - start_time
    results['NumDiffTools (Exact Function)'] = np.abs(nd_exact_values - true_values)
    
    return results, times

def run_comparison(test_name, func, deriv_func, second_deriv_func=None, 
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
    second_deriv_func : callable, optional
        The analytical second derivative function
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
        errors, times = evaluate_methods(x, y, x_eval, deriv_func, func, noise_level=noise_level, order=1)
        
        for method, error in errors.items():
            max_error = np.max(error)
            mean_error = np.mean(error)
            time_ms = times[method] * 1000  # Convert to milliseconds
            print(f"{noise_level:<15.2f} {method:<25} {max_error:<15.6f} {mean_error:<15.6f} {time_ms:<15.2f}")
    
    # Second derivative tests if provided
    if second_deriv_func is not None:
        print("\nSecond Derivative Tests:")
        print(f"{'Noise Level':<15} {'Method':<25} {'Max Error':<15} {'Mean Error':<15} {'Time (ms)':<15}")
        print("-" * 85)
        
        for noise_level in noise_levels:
            errors, times = evaluate_methods(x, y, x_eval, second_deriv_func, func, noise_level=noise_level, order=2)
            
            for method, error in errors.items():
                max_error = np.max(error)
                mean_error = np.mean(error)
                time_ms = times[method] * 1000  # Convert to milliseconds
                print(f"{noise_level:<15.2f} {method:<25} {max_error:<15.6f} {mean_error:<15.6f} {time_ms:<15.2f}")

def plot_comparison(test_name, func, deriv_func, x_range=(-5, 5), num_points=100, noise_level=0.05):
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
    
    # NumDiffTools with noisy data
    from scipy.interpolate import interp1d
    noisy_func = interp1d(x, y_noisy, bounds_error=False, fill_value="extrapolate")
    nd_derivative = nd.Derivative(noisy_func, n=1)
    nd_values = nd_derivative(x_eval)
    
    # NumDiffTools with exact function
    nd_exact = nd.Derivative(func, n=1)
    nd_exact_values = nd_exact(x_eval)
    
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
    plt.plot(x_eval, nd_values, label='NumDiffTools (Noisy Data)')
    plt.plot(x_eval, nd_exact_values, label='NumDiffTools (Exact Function)')
    plt.title(f'{test_name} - First Derivative Comparison')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'../output/{test_name.lower().replace(" ", "_")}_numdifftools_comparison.png')
    plt.close()

def main():
    """Run all comparisons."""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run numerical comparisons
    run_comparison("Sine Function", test_function_sin, test_function_sin_derivative, 
                  test_function_sin_second_derivative)
    
    run_comparison("Exponential Function", test_function_exp, test_function_exp_derivative, 
                  test_function_exp_derivative)
    
    run_comparison("Polynomial Function", test_function_polynomial, test_function_polynomial_derivative, 
                  test_function_polynomial_second_derivative)
    
    # Run visual comparisons
    plot_comparison("Sine Function", test_function_sin, test_function_sin_derivative)
    plot_comparison("Exponential Function", test_function_exp, test_function_exp_derivative)
    plot_comparison("Polynomial Function", test_function_polynomial, test_function_polynomial_derivative)

if __name__ == "__main__":
    main()
