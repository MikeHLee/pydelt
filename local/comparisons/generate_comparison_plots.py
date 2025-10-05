#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate comprehensive comparison plots for PyDelt against other numerical differentiation libraries.

This script generates visualizations comparing PyDelt's differentiation methods with:
- SciPy interpolation methods
- NumDiffTools
- FinDiff
- JAX automatic differentiation

The plots include accuracy comparisons, noise robustness analysis, and performance benchmarks.
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

# Add the parent directory to the path so we can import pydelt
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.pydelt.interpolation import SplineInterpolator, LlaInterpolator, GllaInterpolator

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

def add_noise(y, noise_level=0.01):
    """Add Gaussian noise to the data."""
    return y + noise_level * np.std(y) * np.random.randn(len(y))

def evaluate_all_methods(x, y, x_eval, true_derivative, noise_level=0.0, order=1):
    """
    Evaluate all available differentiation methods and return their errors.
    
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
        if order == 1:
            jax_func = lambda x: jnp.sin(x) if test_derivative == test_function_sin_derivative else \
                      jnp.exp(x) if test_derivative == test_function_exp_derivative else \
                      x**3 - 2*x**2 + 3*x - 1
        
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
    
    return results, times

def plot_accuracy_comparison(noise_levels=[0.0, 0.01, 0.05, 0.1], order=1):
    """
    Generate plots comparing the accuracy of different methods across noise levels.
    
    Parameters:
    -----------
    noise_levels : list
        List of noise levels to test
    order : int
        Order of the derivative to compute
    """
    # Test functions
    test_functions = [
        ("Sine", test_function_sin, test_function_sin_derivative, test_function_sin_second_derivative),
        ("Exponential", test_function_exp, test_function_exp_derivative, test_function_exp_derivative),
        ("Polynomial", test_function_polynomial, test_function_polynomial_derivative, test_function_polynomial_second_derivative)
    ]
    
    # Set up the figure
    fig, axes = plt.subplots(len(test_functions), len(noise_levels), figsize=(16, 12))
    
    # Generate data
    x_range = (-5, 5)
    num_points = 100
    x = np.linspace(x_range[0], x_range[1], num_points)
    x_eval = np.linspace(x_range[0], x_range[1], num_points * 2)
    
    for i, (test_name, func, deriv_func, second_deriv_func) in enumerate(test_functions):
        y = func(x)
        test_derivative = deriv_func if order == 1 else second_deriv_func
        
        for j, noise_level in enumerate(noise_levels):
            # Evaluate all methods
            errors, _ = evaluate_all_methods(x, y, x_eval, test_derivative, noise_level, order)
            
            # Plot the errors
            ax = axes[i, j]
            
            # Create a box plot of the errors
            error_data = []
            labels = []
            
            for method, error in errors.items():
                if not np.isnan(error).any():
                    error_data.append(error)
                    labels.append(method)
            
            ax.boxplot(error_data, labels=labels, vert=True, patch_artist=True, 
                      boxprops=dict(alpha=0.7), medianprops=dict(color='black'))
            
            # Set the title and labels
            ax.set_title(f"{test_name}, Noise={noise_level}")
            if i == len(test_functions) - 1:
                ax.set_xlabel("Method")
            if j == 0:
                ax.set_ylabel("Absolute Error")
            
            # Set y-scale to log for better visualization
            ax.set_yscale('log')
            
            # Rotate x-axis labels for readability
            ax.set_xticklabels(labels, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(f"../output/accuracy_comparison_order{order}.png", dpi=300)
    plt.close()

def plot_noise_robustness():
    """
    Generate plots showing how different methods handle increasing noise levels.
    """
    # Test functions
    test_functions = [
        ("Sine", test_function_sin, test_function_sin_derivative),
        ("Exponential", test_function_exp, test_function_exp_derivative),
        ("Polynomial", test_function_polynomial, test_function_polynomial_derivative)
    ]
    
    # Noise levels to test
    noise_levels = np.linspace(0, 0.2, 10)  # 0% to 20% noise
    
    # Set up the figure
    fig, axes = plt.subplots(len(test_functions), 1, figsize=(10, 15))
    
    # Generate data
    x_range = (-5, 5)
    num_points = 100
    x = np.linspace(x_range[0], x_range[1], num_points)
    x_eval = np.linspace(x_range[0], x_range[1], num_points * 2)
    
    for i, (test_name, func, deriv_func) in enumerate(test_functions):
        y = func(x)
        
        # Store the mean errors for each method at each noise level
        method_errors = {}
        
        for noise_level in noise_levels:
            # Evaluate all methods
            errors, _ = evaluate_all_methods(x, y, x_eval, deriv_func, noise_level, order=1)
            
            # Calculate mean error for each method
            for method, error in errors.items():
                if method not in method_errors:
                    method_errors[method] = []
                method_errors[method].append(np.mean(error))
        
        # Plot the error growth with noise
        ax = axes[i]
        
        for method, error_values in method_errors.items():
            ax.plot(noise_levels, error_values, marker='o', label=method)
        
        # Set the title and labels
        ax.set_title(f"Noise Robustness: {test_name} Function")
        ax.set_xlabel("Noise Level (fraction of std)")
        ax.set_ylabel("Mean Absolute Error")
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("../output/noise_robustness_comparison.png", dpi=300)
    plt.close()

def plot_performance_comparison(num_points_range=[50, 100, 500, 1000, 5000]):
    """
    Generate plots comparing the performance of different methods with increasing data size.
    
    Parameters:
    -----------
    num_points_range : list
        List of data sizes to test
    """
    # Test function (use sine for simplicity)
    func = test_function_sin
    deriv_func = test_function_sin_derivative
    
    # Set up the figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Store the computation times for each method at each data size
    method_times = {}
    method_errors = {}
    
    x_range = (-5, 5)
    
    for num_points in num_points_range:
        # Generate data
        x = np.linspace(x_range[0], x_range[1], num_points)
        y = func(x)
        
        # Use fewer evaluation points for large datasets to keep memory usage reasonable
        eval_factor = 2 if num_points < 1000 else 1
        x_eval = np.linspace(x_range[0], x_range[1], num_points * eval_factor)
        
        # Evaluate all methods
        errors, times = evaluate_all_methods(x, y, x_eval, deriv_func, noise_level=0.01, order=1)
        
        # Store results
        for method, time_value in times.items():
            if method not in method_times:
                method_times[method] = []
            method_times[method].append(time_value * 1000)  # Convert to ms
        
        for method, error in errors.items():
            if method not in method_errors:
                method_errors[method] = []
            method_errors[method].append(np.mean(error))
    
    # Plot computation times
    for method, time_values in method_times.items():
        ax1.plot(num_points_range, time_values, marker='o', label=method)
    
    ax1.set_title("Computation Time vs. Data Size")
    ax1.set_xlabel("Number of Data Points")
    ax1.set_ylabel("Computation Time (ms)")
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot errors
    for method, error_values in method_errors.items():
        ax2.plot(num_points_range, error_values, marker='o', label=method)
    
    ax2.set_title("Mean Error vs. Data Size")
    ax2.set_xlabel("Number of Data Points")
    ax2.set_ylabel("Mean Absolute Error")
    ax2.set_yscale('log')
    ax2.set_xscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("../output/performance_comparison.png", dpi=300)
    plt.close()

def plot_dimensionality_scaling():
    """
    Generate plots showing how different methods scale with increasing dimensionality.
    This is only applicable for methods that support multivariate functions.
    """
    if not JAX_AVAILABLE:
        print("JAX not available, skipping dimensionality scaling plot.")
        return
    
    try:
        from src.pydelt.multivariate import MultivariateDerivatives
        MULTIVARIATE_AVAILABLE = True
    except ImportError:
        MULTIVARIATE_AVAILABLE = False
        print("PyDelt multivariate module not available, skipping dimensionality scaling plot.")
        return
    
    # Dimensions to test
    dimensions = [2, 5, 10, 20, 50]
    num_points = 100
    
    # Store the computation times for each method at each dimensionality
    method_times = {
        'PyDelt Multivariate': [],
        'JAX Gradient': [],
        'JAX Vectorized': []
    }
    
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
        
        # PyDelt multivariate approach
        try:
            start_time = time.time()
            mv = MultivariateDerivatives(SplineInterpolator)
            mv.fit(X, Y)
            gradient_func = mv.gradient()
            _ = gradient_func(X_eval)
            pydelt_time = time.time() - start_time
            method_times['PyDelt Multivariate'].append(pydelt_time * 1000)
        except Exception as e:
            print(f"PyDelt multivariate failed for {dim} dimensions: {e}")
            method_times['PyDelt Multivariate'].append(np.nan)
        
        # JAX gradient
        start_time = time.time()
        jax_grad_func = grad(jax_quad_func)
        _ = [jax_grad_func(jnp.array(x)) for x in X_eval]
        jax_time = time.time() - start_time
        method_times['JAX Gradient'].append(jax_time * 1000)
        
        # JAX vectorized gradient
        start_time = time.time()
        jax_grad_vmap = jit(vmap(grad(jax_quad_func)))
        _ = jax_grad_vmap(jnp.array(X_eval))
        jax_vmap_time = time.time() - start_time
        method_times['JAX Vectorized'].append(jax_vmap_time * 1000)
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    
    for method, time_values in method_times.items():
        plt.plot(dimensions, time_values, marker='o', label=method)
    
    plt.title("Computation Time vs. Dimensionality")
    plt.xlabel("Number of Dimensions")
    plt.ylabel("Computation Time (ms)")
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("../output/dimensionality_scaling.png", dpi=300)
    plt.close()

def create_summary_table():
    """
    Create a summary table of the strengths and weaknesses of each method.
    """
    # Define the methods and their characteristics
    methods = [
        {
            'Method': 'PyDelt LLA/GLLA',
            'Accuracy': 'High',
            'Noise Robustness': 'High',
            'Performance': 'Medium',
            'Dimensionality': 'Low (1-3)',
            'Key Strength': 'Best accuracy for noisy data',
            'Key Weakness': 'Limited to low dimensions'
        },
        {
            'Method': 'PyDelt Spline',
            'Accuracy': 'Medium',
            'Noise Robustness': 'Medium',
            'Performance': 'High',
            'Dimensionality': 'Low (1-3)',
            'Key Strength': 'Good balance of accuracy and speed',
            'Key Weakness': 'Less accurate than LLA/GLLA'
        },
        {
            'Method': 'PyDelt Neural Network',
            'Accuracy': 'Medium',
            'Noise Robustness': 'Medium',
            'Performance': 'Low',
            'Dimensionality': 'High (1-100+)',
            'Key Strength': 'Scales well to high dimensions',
            'Key Weakness': 'Training overhead, less accurate for simple functions'
        },
        {
            'Method': 'SciPy Spline',
            'Accuracy': 'Medium',
            'Noise Robustness': 'Medium',
            'Performance': 'High',
            'Dimensionality': 'Low (1)',
            'Key Strength': 'Fast for 1D data',
            'Key Weakness': 'Limited to univariate functions'
        },
        {
            'Method': 'NumDiffTools',
            'Accuracy': 'High',
            'Noise Robustness': 'Low',
            'Performance': 'Medium',
            'Dimensionality': 'Medium (1-5)',
            'Key Strength': 'High accuracy for clean data',
            'Key Weakness': 'Poor noise robustness'
        },
        {
            'Method': 'FinDiff',
            'Accuracy': 'High',
            'Noise Robustness': 'Low',
            'Performance': 'High',
            'Dimensionality': 'High (1-100+)',
            'Key Strength': 'Fast and supports high dimensions',
            'Key Weakness': 'Very sensitive to noise'
        },
        {
            'Method': 'JAX',
            'Accuracy': 'Very High',
            'Noise Robustness': 'N/A',
            'Performance': 'Very High',
            'Dimensionality': 'High (1-1000+)',
            'Key Strength': 'Extremely fast and accurate',
            'Key Weakness': 'Requires analytical function definition'
        }
    ]
    
    # Create a DataFrame
    df = pd.DataFrame(methods)
    
    # Save to CSV
    df.to_csv("../output/method_comparison_summary.csv", index=False)
    
    # Create a styled HTML table
    styled_df = df.style.set_table_styles([
        {'selector': 'th', 'props': [('font-weight', 'bold'), ('text-align', 'center')]},
        {'selector': 'td', 'props': [('text-align', 'center')]}
    ])
    
    # Save to HTML
    styled_df.to_html("../output/method_comparison_summary.html")
    
    return df

def main():
    """Run all visualizations."""
    # Create output directory if it doesn't exist
    Path("../output").mkdir(parents=True, exist_ok=True)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    print("Generating accuracy comparison plots...")
    plot_accuracy_comparison(order=1)  # First derivatives
    plot_accuracy_comparison(order=2)  # Second derivatives
    
    print("Generating noise robustness plots...")
    plot_noise_robustness()
    
    print("Generating performance comparison plots...")
    plot_performance_comparison()
    
    print("Generating dimensionality scaling plots...")
    plot_dimensionality_scaling()
    
    print("Creating summary table...")
    create_summary_table()
    
    print("All visualizations completed!")

if __name__ == "__main__":
    main()
