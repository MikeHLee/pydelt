"""
Visual tests for automatic differentiation methods in PyDelt.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path

# Add the src directory to the path so we can import pydelt
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from pydelt.autodiff import (
    neural_network_derivative
)

# Check if dependencies are available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch is not installed. Some tests will be skipped.")

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow is not installed. Some tests will be skipped.")


def generate_sine_data(n_points=50, noise_level=0.05):
    """Generate sine wave data with optional noise for testing."""
    np.random.seed(42)  # For reproducibility
    time = np.linspace(0, 2*np.pi, n_points)
    clean_signal = np.sin(time)
    
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, n_points)
        noisy_signal = clean_signal + noise
        return time, noisy_signal, clean_signal
    else:
        return time, clean_signal, clean_signal


def test_neural_network_derivative_visual():
    """Visual test for neural network derivative calculation."""
    if not TORCH_AVAILABLE and not TF_AVAILABLE:
        print("Skipping neural network derivative visual test - no frameworks available.")
        return
    
    # Generate data
    time, signal, clean_signal = generate_sine_data(10000, 0.05)
    
    # Create dense time points for smooth plotting
    dense_time = np.linspace(0, 2*np.pi, 200)
    true_signal = np.sin(dense_time)
    true_derivative = np.cos(dense_time)  # First derivative of sin(x) is cos(x)
    true_second_derivative = -np.sin(dense_time)  # Second derivative of sin(x) is -sin(x)
    
    # Create figure
    plt.figure(figsize=(15, 10))
    
    plot_count = 0
    
    # Test PyTorch if available
    if TORCH_AVAILABLE:
        # First derivative with PyTorch
        plot_count += 1
        plt.subplot(2, 2, plot_count)
        
        # Create derivative function
        deriv_func = neural_network_derivative(
            time, signal, framework='pytorch', 
            hidden_layers=[64, 32, 16], dropout=0.1, epochs=10,
            order=1
        )
        
        # Calculate derivative at dense time points
        calculated_derivative = deriv_func(dense_time)
        
        plt.scatter(time, np.cos(time), color='red', alpha=0.5, label='True Derivative at Data Points')
        plt.plot(dense_time, true_derivative, 'g--', label='True Derivative')
        plt.plot(dense_time, calculated_derivative, 'b-', label='PyTorch NN Derivative')
        plt.title('Neural Network First Derivative (PyTorch)')
        plt.xlabel('Time')
        plt.ylabel('Derivative')
        plt.legend()
        plt.grid(True)
        
        # Second derivative with PyTorch
        plot_count += 1
        plt.subplot(2, 2, plot_count)
        
        # Create second derivative function
        second_deriv_func = neural_network_derivative(
            time, signal, framework='pytorch', 
            hidden_layers=[64, 32, 16], dropout=0.1, epochs=10,
            order=2
        )
        
        # Calculate second derivative at dense time points
        calculated_second_derivative = second_deriv_func(dense_time)
        
        plt.scatter(time, -np.sin(time), color='red', alpha=0.5, label='True 2nd Derivative at Data Points')
        plt.plot(dense_time, true_second_derivative, 'g--', label='True 2nd Derivative')
        plt.plot(dense_time, calculated_second_derivative, 'b-', label='PyTorch NN 2nd Derivative')
        plt.title('Neural Network Second Derivative (PyTorch)')
        plt.xlabel('Time')
        plt.ylabel('Second Derivative')
        plt.legend()
        plt.grid(True)
    
    # Test TensorFlow if available
    if TF_AVAILABLE:
        # First derivative with TensorFlow
        plot_count += 1
        plt.subplot(2, 2, plot_count)
        
        # Create derivative function
        deriv_func = neural_network_derivative(
            time, signal, framework='tensorflow', 
            hidden_layers=[64, 32, 16], dropout=0.1, epochs=10,
            order=1
        )
        
        # Calculate derivative at dense time points
        calculated_derivative = deriv_func(dense_time)
        
        plt.scatter(time, np.cos(time), color='red', alpha=0.5, label='True Derivative at Data Points')
        plt.plot(dense_time, true_derivative, 'g--', label='True Derivative')
        plt.plot(dense_time, calculated_derivative, 'b-', label='TensorFlow NN Derivative')
        plt.title('Neural Network First Derivative (TensorFlow)')
        plt.xlabel('Time')
        plt.ylabel('Derivative')
        plt.legend()
        plt.grid(True)
        
        # Second derivative with TensorFlow
        if plot_count < 4:  # Only if we have space in the 2x2 grid
            plot_count += 1
            plt.subplot(2, 2, plot_count)
            
            # Create second derivative function
            second_deriv_func = neural_network_derivative(
                time, signal, framework='tensorflow', 
                hidden_layers=[64, 32, 16], dropout=0.1, epochs=10,
                order=2
            )
            
            # Calculate second derivative at dense time points
            calculated_second_derivative = second_deriv_func(dense_time)
            
            plt.scatter(time, -np.sin(time), color='red', alpha=0.5, label='True 2nd Derivative at Data Points')
            plt.plot(dense_time, true_second_derivative, 'g--', label='True 2nd Derivative')
            plt.plot(dense_time, calculated_second_derivative, 'b-', label='TensorFlow NN 2nd Derivative')
            plt.title('Neural Network Second Derivative (TensorFlow)')
            plt.xlabel('Time')
            plt.ylabel('Second Derivative')
            plt.legend()
            plt.grid(True)
    
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    # Save figure
    plt.savefig(output_dir / "neural_network_derivative.png")

    # Save Plotly HTML
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dense_time, y=true_derivative, mode='lines', name='True Derivative', line=dict(dash='dash', color='green')))
    if TORCH_AVAILABLE:
        fig.add_trace(go.Scatter(x=dense_time, y=calculated_derivative, mode='lines', name='PyTorch NN Derivative', line=dict(color='blue')))
    if TF_AVAILABLE:
        fig.add_trace(go.Scatter(x=dense_time, y=calculated_derivative, mode='lines', name='TensorFlow NN Derivative', line=dict(color='red')))
    fig.update_layout(title="Neural Network Derivative (PyTorch/TensorFlow)", xaxis_title="Time", yaxis_title="First Derivative")
    fig.write_html(str(output_dir / "neural_network_derivative.html"))
    plt.close()
    
    print("Neural network derivative visual test completed.")


def test_combined_methods_comparison():
    """Visual comparison of different derivative methods on the same data."""
    if not TORCH_AVAILABLE:
        print("Skipping combined methods comparison - PyTorch not available.")
        return
    
    # Generate data with more noise to test robustness
    time, signal, clean_signal = generate_sine_data(10000, 0.1)
    
    # Create dense time points for smooth plotting
    dense_time = np.linspace(0, 2*np.pi, 200)
    true_derivative = np.cos(dense_time)
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Plot original data
    plt.subplot(2, 1, 1)
    plt.scatter(time, signal, color='red', label='Noisy Data')
    plt.plot(dense_time, np.sin(dense_time), 'g--', label='True Signal')
    plt.title('Original Signal')
    plt.xlabel('Time')
    plt.ylabel('Signal')
    plt.legend()
    plt.grid(True)
    
    # Plot derivatives from different methods
    plt.subplot(2, 1, 2)
    
    # True derivative
    plt.plot(dense_time, true_derivative, 'g--', label='True Derivative')
    
    # Neural Network derivative
    nn_deriv_func = neural_network_derivative(
        time, signal, framework='pytorch', 
        hidden_layers=[64, 32, 16], dropout=0.1, epochs=10
    )
    nn_derivative = nn_deriv_func(dense_time)
    plt.plot(dense_time, nn_derivative, 'b-', label='Neural Network')
    
    # Placeholder for additional methods if needed
    
    # Add numerical derivative for comparison
    # Import here to avoid circular imports
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
    from pydelt.derivatives import lla
    
    # Calculate LLA derivative
    lla_derivative, _ = lla(time.tolist(), signal.tolist())
    
    # Create interpolation function for LLA derivative
    from scipy.interpolate import interp1d
    lla_interp = interp1d(time, lla_derivative, kind='cubic', bounds_error=False, fill_value='extrapolate')
    lla_dense = lla_interp(dense_time)
    
    plt.plot(dense_time, lla_dense, 'y-', label='LLA (Numerical)')
    
    plt.title('Derivative Comparison')
    plt.xlabel('Time')
    plt.ylabel('First Derivative')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    # Save figure
    plt.savefig(output_dir / "derivative_methods_comparison.png")
    plt.close()
    
    print("Combined methods comparison test completed.")


if __name__ == "__main__":
    print("Running visual tests for automatic differentiation methods...")
    test_neural_network_derivative_visual()

    test_combined_methods_comparison()
    print("All visual tests completed.")
