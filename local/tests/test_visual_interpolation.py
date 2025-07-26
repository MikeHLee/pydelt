"""
Visual tests for advanced interpolation methods in PyDelt.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path

# Add the src directory to the path so we can import pydelt
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from pydelt.interpolation import (
    derivative_based_interpolation,
    neural_network_interpolation
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


def generate_noisy_sine_data(n_points=50, noise_level=0.1):
    """Generate sine wave data with noise for testing."""
    np.random.seed(42)  # For reproducibility
    time = np.linspace(0, 2*np.pi, n_points)
    clean_signal = np.sin(time)
    noise = np.random.normal(0, noise_level, n_points)
    noisy_signal = clean_signal + noise
    return time, noisy_signal, clean_signal


def test_derivative_based_interpolation_visual():
    """Visual test for derivative-based interpolation methods."""
    # Generate data
    time, noisy_signal, clean_signal = generate_noisy_sine_data(30, 0.2)
    
    # Create dense time points for smooth plotting
    dense_time = np.linspace(0, 2*np.pi, 200)
    true_signal = np.sin(dense_time)
    
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # Test each derivative method
    methods = ['lla', 'glla', 'gold', 'fda']
    
    for i, method in enumerate(methods):
        # Create interpolation function
        interp_func = derivative_based_interpolation(time, noisy_signal, method=method)
        
        # Interpolate at dense time points
        interpolated = interp_func(dense_time)
        
        # Plot in subplot
        plt.subplot(2, 2, i+1)
        plt.scatter(time, noisy_signal, color='red', label='Noisy Data')
        plt.plot(dense_time, true_signal, 'g--', label='True Signal')
        plt.plot(dense_time, interpolated, 'b-', label=f'{method.upper()} Interpolation')
        plt.title(f'Derivative-Based Interpolation ({method.upper()})')
        plt.xlabel('Time')
        plt.ylabel('Signal')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    # Save figure
    plt.savefig(output_dir / "derivative_based_interpolation.png")
    plt.close()

    # Save Plotly HTML
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=2, cols=2, subplot_titles=[m.upper() for m in methods])
    for i, method in enumerate(methods):
        interp_func = derivative_based_interpolation(time, noisy_signal, method=method)
        interpolated = interp_func(dense_time)
        row = i // 2 + 1
        col = i % 2 + 1
        fig.add_trace(go.Scatter(x=time, y=noisy_signal, mode='markers', name='Noisy Data'), row=row, col=col)
        fig.add_trace(go.Scatter(x=dense_time, y=true_signal, mode='lines', name='True Signal', line=dict(dash='dash', color='green')), row=row, col=col)
        fig.add_trace(go.Scatter(x=dense_time, y=interpolated, mode='lines', name=f'{method.upper()} Interpolation', line=dict(color='blue')), row=row, col=col)
        fig.update_xaxes(title_text='Time', row=row, col=col)
        fig.update_yaxes(title_text='Signal', row=row, col=col)
    fig.update_layout(title="Derivative-Based Interpolation (All Methods)", height=800, width=1200)
    fig.write_html(str(output_dir / "derivative_based_interpolation.html"))
    print("Derivative-based interpolation visual test completed.")


def test_neural_network_interpolation_visual():
    """Visual test for neural network interpolation."""
    if not TORCH_AVAILABLE and not TF_AVAILABLE:
        print("Skipping neural network interpolation visual test - no frameworks available.")
        return
    
    # Generate data
    time, noisy_signal, clean_signal = generate_noisy_sine_data(30, 0.2)
    
    # Create dense time points for smooth plotting
    dense_time = np.linspace(0, 2*np.pi, 200)
    true_signal = np.sin(dense_time)
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Plot count
    plot_count = 0
    
    # Test PyTorch if available
    if TORCH_AVAILABLE:
        plot_count += 1
        plt.subplot(2, 2, plot_count)
        
        # Create interpolation function with small network and few epochs for testing
        interp_func = neural_network_interpolation(
            time, noisy_signal, framework='pytorch', 
            hidden_layers=[64, 32, 16], dropout=0.1, epochs=10
        )
        
        # Interpolate at dense time points
        interpolated = interp_func(dense_time)
        
        plt.scatter(time, noisy_signal, color='red', label='Noisy Data')
        plt.plot(dense_time, true_signal, 'g--', label='True Signal')
        plt.plot(dense_time, interpolated, 'b-', label='PyTorch NN Interpolation')
        plt.title('Neural Network Interpolation (PyTorch)')
        plt.xlabel('Time')
        plt.ylabel('Signal')
        plt.legend()
        plt.grid(True)
        
        # Test with holdout
        plot_count += 1
        plt.subplot(2, 2, plot_count)
        
        # Create interpolation function with holdout
        interp_func = neural_network_interpolation(
            time, noisy_signal, framework='pytorch',
            hidden_layers=[64, 32, 16], dropout=0.1, epochs=10,
            holdout_fraction=0.2
        )
        
        # Interpolate at dense time points
        interpolated = interp_func(dense_time)
        
        plt.scatter(time, noisy_signal, color='red', label='Noisy Data')
        plt.plot(dense_time, true_signal, 'g--', label='True Signal')
        plt.plot(dense_time, interpolated, 'b-', label='PyTorch NN with Holdout')
        plt.title('Neural Network Interpolation with Holdout (PyTorch)')
        plt.xlabel('Time')
        plt.ylabel('Signal')
        plt.legend()
        plt.grid(True)
    
    # Test TensorFlow if available
    if TF_AVAILABLE:
        plot_count += 1
        plt.subplot(2, 2, plot_count)
        
        # Create interpolation function with small network and few epochs for testing
        interp_func = neural_network_interpolation(
            time, noisy_signal, framework='tensorflow', 
            hidden_layers=[64, 32, 16], dropout=0.1, epochs=10
        )
        
        # Interpolate at dense time points
        interpolated = interp_func(dense_time)
        
        plt.scatter(time, noisy_signal, color='red', label='Noisy Data')
        plt.plot(dense_time, true_signal, 'g--', label='True Signal')
        plt.plot(dense_time, interpolated, 'b-', label='TensorFlow NN Interpolation')
        plt.title('Neural Network Interpolation (TensorFlow)')
        plt.xlabel('Time')
        plt.ylabel('Signal')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    # Save figure
    plt.savefig(output_dir / "neural_network_interpolation.png")
    plt.close()

    # Save Plotly HTML
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=2, cols=2, subplot_titles=["PyTorch", "PyTorch Holdout", "TensorFlow"])
    plot_count = 0
    if TORCH_AVAILABLE:
        plot_count += 1
        row, col = (plot_count - 1) // 2 + 1, (plot_count - 1) % 2 + 1
        interp_func = neural_network_interpolation(time, noisy_signal, framework='pytorch', hidden_layers=[64, 32, 16], dropout=0.1, epochs=10)
        interpolated = interp_func(dense_time)
        fig.add_trace(go.Scatter(x=time, y=noisy_signal, mode='markers', name='Noisy Data'), row=row, col=col)
        fig.add_trace(go.Scatter(x=dense_time, y=true_signal, mode='lines', name='True Signal', line=dict(dash='dash', color='green')), row=row, col=col)
        fig.add_trace(go.Scatter(x=dense_time, y=interpolated, mode='lines', name='PyTorch NN Interpolation', line=dict(color='blue')), row=row, col=col)
    if TORCH_AVAILABLE:
        plot_count += 1
        row, col = (plot_count - 1) // 2 + 1, (plot_count - 1) % 2 + 1
        interp_func = neural_network_interpolation(time, noisy_signal, framework='pytorch', hidden_layers=[64, 32, 16], dropout=0.1, epochs=10, holdout_fraction=0.2)
        interpolated = interp_func(dense_time)
        fig.add_trace(go.Scatter(x=time, y=noisy_signal, mode='markers', name='Noisy Data'), row=row, col=col)
        fig.add_trace(go.Scatter(x=dense_time, y=true_signal, mode='lines', name='True Signal', line=dict(dash='dash', color='green')), row=row, col=col)
        fig.add_trace(go.Scatter(x=dense_time, y=interpolated, mode='lines', name='PyTorch NN with Holdout', line=dict(color='blue')), row=row, col=col)
    if TF_AVAILABLE:
        plot_count += 1
        row, col = (plot_count - 1) // 2 + 1, (plot_count - 1) % 2 + 1
        interp_func = neural_network_interpolation(time, noisy_signal, framework='tensorflow', hidden_layers=[64, 32, 16], dropout=0.1, epochs=10)
        interpolated = interp_func(dense_time)
        fig.add_trace(go.Scatter(x=time, y=noisy_signal, mode='markers', name='Noisy Data'), row=row, col=col)
        fig.add_trace(go.Scatter(x=dense_time, y=true_signal, mode='lines', name='True Signal', line=dict(dash='dash', color='green')), row=row, col=col)
        fig.add_trace(go.Scatter(x=dense_time, y=interpolated, mode='lines', name='TensorFlow NN Interpolation', line=dict(color='blue')), row=row, col=col)
    fig.update_layout(title="Neural Network Interpolation (PyTorch/TensorFlow)", height=800, width=1200)
    fig.write_html(str(output_dir / "neural_network_interpolation.html"))
    print("Neural network interpolation visual test completed.")


def test_classical_interpolation_visual():
    """Visual test for classical interpolation methods (lsl, spline, lowess, loess)."""
    from pydelt.interpolation import local_segmented_linear, spline_interpolation, lowess_interpolation, loess_interpolation
    time, noisy_signal, clean_signal = generate_noisy_sine_data(30, 0.2)
    dense_time = np.linspace(0, 2*np.pi, 200)
    true_signal = np.sin(dense_time)
    methods = [
        ("LSL", local_segmented_linear(time, noisy_signal, window_size=7)),
        ("Spline", spline_interpolation(time, noisy_signal)),
        ("LOWESS", lowess_interpolation(time, noisy_signal)),
        ("LOESS", loess_interpolation(time, noisy_signal))
    ]
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 8))
    plt.scatter(time, noisy_signal, color='red', label='Noisy Data', alpha=0.7)
    plt.plot(dense_time, true_signal, 'g--', label='True Signal')
    for name, func in methods:
        plt.plot(dense_time, func(dense_time), label=f'{name} Interpolation')
    plt.title('Classical Interpolation Methods')
    plt.xlabel('Time')
    plt.ylabel('Signal')
    plt.legend()
    plt.grid(True)
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "classical_interpolation.png")
    plt.close()
    # Plotly HTML
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=noisy_signal, mode='markers', name='Noisy Data'))
    fig.add_trace(go.Scatter(x=dense_time, y=true_signal, mode='lines', name='True Signal', line=dict(dash='dash', color='green')))
    for name, func in methods:
        fig.add_trace(go.Scatter(x=dense_time, y=func(dense_time), mode='lines', name=f'{name} Interpolation'))
    fig.update_layout(title="Classical Interpolation Methods", xaxis_title="Time", yaxis_title="Signal", width=900, height=600)
    fig.write_html(str(output_dir / "classical_interpolation.html"))
    print("Classical interpolation visual test completed.")

def test_neural_network_derivative_visual():
    """Visual test for neural network derivatives (PyTorch and TensorFlow)."""
    from pydelt.autodiff import neural_network_derivative
    time = np.linspace(0, 2*np.pi, 1000)
    signal = np.sin(time)
    true_derivative = np.cos(time)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.plot(time, true_derivative, 'g--', label='True Derivative (cos)')
    plot_count = 0
    try:
        deriv_func = neural_network_derivative(time, signal, framework='pytorch', hidden_layers=[64, 32, 16], dropout=0.025, epochs=100)
        nn_deriv = deriv_func(time)
        plt.plot(time, nn_deriv, 'b-', label='PyTorch NN Derivative')
        plot_count += 1
    except Exception as e:
        print(f"PyTorch NN derivative failed: {e}")
    try:
        deriv_func = neural_network_derivative(time, signal, framework='tensorflow', hidden_layers=[64, 32, 16], dropout=0.1, epochs=50)
        nn_deriv = deriv_func(time)
        plt.plot(time, nn_deriv, 'r-', label='TensorFlow NN Derivative')
        plot_count += 1
    except Exception as e:
        print(f"TensorFlow NN derivative failed: {e}")
    plt.title('Neural Network Derivative vs True Derivative')
    plt.xlabel('Time')
    plt.ylabel('d/dt sin(t)')
    plt.legend()
    plt.grid(True)
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "neural_network_derivative.png")
    plt.close()
    # Plotly HTML
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=true_derivative, mode='lines', name='True Derivative (cos)', line=dict(dash='dash', color='green')))
    try:
        deriv_func = neural_network_derivative(time, signal, framework='pytorch', hidden_layers=[64, 32, 16], dropout=0.025, epochs=100)
        nn_deriv = deriv_func(time)
        fig.add_trace(go.Scatter(x=time, y=nn_deriv, mode='lines', name='PyTorch NN Derivative', line=dict(color='blue')))
    except Exception as e:
        print(f"PyTorch NN derivative failed for Plotly: {e}")
    try:
        deriv_func = neural_network_derivative(time, signal, framework='tensorflow', hidden_layers=[64, 32, 16], dropout=0.1, epochs=50)
        nn_deriv = deriv_func(time)
        fig.add_trace(go.Scatter(x=time, y=nn_deriv, mode='lines', name='TensorFlow NN Derivative', line=dict(color='red')))
    except Exception as e:
        print(f"TensorFlow NN derivative failed for Plotly: {e}")
    fig.update_layout(title="Neural Network Derivative vs True Derivative", xaxis_title="Time", yaxis_title="d/dt sin(t)", width=900, height=500)
    fig.write_html(str(output_dir / "neural_network_derivative.html"))
    print("Neural network derivative visual test completed.")

if __name__ == "__main__":
    print("Running visual tests for advanced interpolation methods...")
    test_derivative_based_interpolation_visual()
    test_neural_network_interpolation_visual()
    test_classical_interpolation_visual()
    test_neural_network_derivative_visual()
    print("All visual tests completed.")
