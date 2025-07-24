"""
Visual tests for the derivatives module.
Creates plotly visualizations to compare expected vs actual results.
Includes tests with varying levels of noise.
"""

import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pydelt.derivatives import lla, gold, glla, fda

# Import neural network and DNDF derivatives if available
try:
    from pydelt.autodiff import neural_network_derivative
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# Ensure output directory exists
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_plot(fig, filename):
    """Save a plotly figure to the output directory."""
    filepath = os.path.join(OUTPUT_DIR, filename)
    fig.write_html(filepath)
    print(f"Plot saved to {filepath}")

def visual_test_lla_sine():
    """Visual test of LLA derivative on sine function."""
    time = np.linspace(0, 10, 100)
    signal = np.sin(time)
    derivative, steps = lla(time.tolist(), signal.tolist(), window_size=5)
    
    # Expected derivative of sine is cosine
    expected = np.cos(time)
    
    # Create plot
    fig = make_subplots(rows=2, cols=1, 
                        subplot_titles=("Derivative vs Expected", "Difference"),
                        shared_xaxes=True)
    
    fig.add_trace(
        go.Scatter(x=time, y=derivative, mode='lines', name='LLA Derivative'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=time, y=expected, mode='lines', name='Expected (cosine)'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=time, y=derivative-expected, mode='lines', name='Difference'),
        row=2, col=1
    )
    
    fig.update_layout(title="LLA Derivative of Sine",
                     height=800, width=1000)
    
    save_plot(fig, "lla_sine_derivative.html")

def visual_test_gold_sine():
    """Visual test of GOLD derivative on sine function."""
    time = np.linspace(0, 10, 100)
    signal = np.sin(time)
    result = gold(signal, time, embedding=5, n=2)
    
    # Extract first derivative and corresponding time points
    derivative = result['dsignal'][:, 1]
    # Account for boundary effects
    valid_time = time[2:-2][:derivative.shape[0]]
    
    # Expected derivative of sine is cosine
    expected = np.cos(valid_time)
    
    # Create plot
    fig = make_subplots(rows=2, cols=1, 
                        subplot_titles=("Derivative vs Expected", "Difference"),
                        shared_xaxes=True)
    
    fig.add_trace(
        go.Scatter(x=valid_time, y=derivative, mode='lines', name='GOLD Derivative'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=valid_time, y=expected, mode='lines', name='Expected (cosine)'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=valid_time, y=derivative-expected, mode='lines', name='Difference'),
        row=2, col=1
    )
    
    fig.update_layout(title="GOLD Derivative of Sine",
                     height=800, width=1000)
    
    save_plot(fig, "gold_sine_derivative.html")

def visual_test_glla_sine():
    """Visual test of GLLA derivative on sine function."""
    time = np.linspace(0, 10, 100)
    signal = np.sin(time)
    result = glla(signal, time, embedding=5, n=2)
    
    # Extract first derivative and corresponding time points
    derivative = result['dsignal'][:, 1]
    # Account for boundary effects
    valid_time = time[2:-2][:derivative.shape[0]]
    
    # Expected derivative of sine is cosine
    expected = np.cos(valid_time)
    
    # Create plot
    fig = make_subplots(rows=2, cols=1, 
                        subplot_titles=("Derivative vs Expected", "Difference"),
                        shared_xaxes=True)
    
    fig.add_trace(
        go.Scatter(x=valid_time, y=derivative, mode='lines', name='GLLA Derivative'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=valid_time, y=expected, mode='lines', name='Expected (cosine)'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=valid_time, y=derivative-expected, mode='lines', name='Difference'),
        row=2, col=1
    )
    
    fig.update_layout(title="GLLA Derivative of Sine",
                     height=800, width=1000)
    
    save_plot(fig, "glla_sine_derivative.html")

def visual_test_fda_sine():
    """Visual test of FDA derivative on sine function."""
    time = np.linspace(0, 10, 100)
    signal = np.sin(time)
    result = fda(signal, time)
    
    # Extract first derivative
    derivative = result['dsignal'][:, 1]
    
    # Expected derivative of sine is cosine
    expected = np.cos(time)
    
    # Create plot
    fig = make_subplots(rows=2, cols=1, 
                        subplot_titles=("Derivative vs Expected", "Difference"),
                        shared_xaxes=True)
    
    fig.add_trace(
        go.Scatter(x=time, y=derivative, mode='lines', name='FDA Derivative'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=time, y=expected, mode='lines', name='Expected (cosine)'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=time, y=derivative-expected, mode='lines', name='Difference'),
        row=2, col=1
    )
    
    fig.update_layout(title="FDA Derivative of Sine",
                     height=800, width=1000)
    
    save_plot(fig, "fda_sine_derivative.html")

def visual_test_algorithm_comparison():
    """Visual comparison of all derivative algorithms on sine function."""
    time = np.linspace(0, 10, 100)
    signal = np.sin(time)
    
    # Calculate derivatives using different methods
    lla_derivative, _ = lla(time.tolist(), signal.tolist(), window_size=5)
    
    gold_result = gold(signal, time, embedding=5, n=2)
    gold_derivative = gold_result['dsignal'][:, 1]
    gold_time = time[2:-2][:gold_derivative.shape[0]]
    
    glla_result = glla(signal, time, embedding=5, n=2)
    glla_derivative = glla_result['dsignal'][:, 1]
    glla_time = time[2:-2][:glla_derivative.shape[0]]
    
    fda_result = fda(signal, time)
    fda_derivative = fda_result['dsignal'][:, 1]
    
    # Expected derivative of sine is cosine
    expected = np.cos(time)
    
    # Create plot
    fig = make_subplots(rows=2, cols=1, 
                        subplot_titles=("Derivative Methods Comparison", "Difference from Expected"),
                        shared_xaxes=True)
    
    # Plot derivatives
    fig.add_trace(
        go.Scatter(x=time, y=lla_derivative, mode='lines', name='LLA'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=gold_time, y=gold_derivative, mode='lines', name='GOLD'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=glla_time, y=glla_derivative, mode='lines', name='GLLA'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=time, y=fda_derivative, mode='lines', name='FDA'),
        row=1, col=1
    )
    # Add neural network derivatives if available
    if TORCH_AVAILABLE:
        nn_deriv_func = neural_network_derivative(
            time, signal, framework='pytorch', hidden_layers=[64,32,16], dropout=0.025, epochs=300
        )
        nn_derivative = nn_deriv_func(time)
        fig.add_trace(
            go.Scatter(x=time, y=nn_derivative, mode='lines', name='NN (PyTorch)'),
            row=1, col=1
        )
    if TF_AVAILABLE:
        nn_deriv_func_tf = neural_network_derivative(
            time, signal, framework='tensorflow', hidden_layers=[64,32,16], dropout=0.025, epochs=300
        )
        nn_derivative_tf = nn_deriv_func_tf(time)
        fig.add_trace(
            go.Scatter(x=time, y=nn_derivative_tf, mode='lines', name='NN (TensorFlow)'),
            row=1, col=1
        )
    fig.add_trace(
        go.Scatter(x=time, y=expected, mode='lines', name='Expected (cosine)'),
        row=1, col=1
    )
    
    # Plot differences
    fig.add_trace(
        go.Scatter(x=time, y=lla_derivative-expected, mode='lines', name='LLA Difference'),
        row=2, col=1
    )
    # For GOLD and GLLA, we need to compare with the expected values at the valid time points
    gold_expected = np.cos(gold_time)
    fig.add_trace(
        go.Scatter(x=gold_time, y=gold_derivative-gold_expected, mode='lines', name='GOLD Difference'),
        row=2, col=1
    )
    glla_expected = np.cos(glla_time)
    fig.add_trace(
        go.Scatter(x=glla_time, y=glla_derivative-glla_expected, mode='lines', name='GLLA Difference'),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=time, y=fda_derivative-expected, mode='lines', name='FDA Difference'),
        row=2, col=1
    )
    
    fig.update_layout(title="Comparison of Derivative Methods on Sine",
                     height=800, width=1000)
    
    save_plot(fig, "derivative_methods_comparison.html")

def add_noise(signal, noise_level):
    """Add Gaussian noise to a signal."""
    np.random.seed(42)  # For reproducibility
    noise = np.random.normal(0, noise_level, size=signal.shape)
    return signal + noise

def visual_test_noise_comparison():
    """Visual test of how different algorithms handle noisy data."""
    time = np.linspace(0, 10, 100)
    clean_signal = np.sin(time)
    
    # Create signals with different noise levels
    noise_levels = [0.01, 0.05, 0.1, 0.2]
    noisy_signals = [add_noise(clean_signal, level) for level in noise_levels]
    
    # Create a figure with subplots for each noise level
    fig = make_subplots(
        rows=len(noise_levels), 
        cols=2,
        subplot_titles=[f"Noise Level: {level} - Derivatives" for level in noise_levels] + 
                      [f"Noise Level: {level} - Original Signal" for level in noise_levels],
        shared_xaxes=True,
        vertical_spacing=0.1,
        horizontal_spacing=0.05,
        column_widths=[0.7, 0.3]
    )
    
    # Expected derivative of sine is cosine
    expected = np.cos(time)
    
    # Process each noise level
    for i, (noisy_signal, noise_level) in enumerate(zip(noisy_signals, noise_levels)):
        row = i + 1
        
        # Calculate derivatives using different methods
        lla_derivative, _ = lla(time.tolist(), noisy_signal.tolist(), window_size=5)
        
        gold_result = gold(noisy_signal, time, embedding=5, n=2)
        gold_derivative = gold_result['dsignal'][:, 1]
        gold_time = time[2:-2][:gold_derivative.shape[0]]
        gold_expected = np.cos(gold_time)
        
        glla_result = glla(noisy_signal, time, embedding=5, n=2)
        glla_derivative = glla_result['dsignal'][:, 1]
        glla_time = time[2:-2][:glla_derivative.shape[0]]
        
        fda_result = fda(noisy_signal, time)
        fda_derivative = fda_result['dsignal'][:, 1]
        
        # Add traces for each method in the first column
        fig.add_trace(
            go.Scatter(x=time, y=lla_derivative, mode='lines', name=f'LLA (noise={noise_level})'),
            row=row, col=1
        )
        fig.add_trace(
            go.Scatter(x=gold_time, y=gold_derivative, mode='lines', name=f'GOLD (noise={noise_level})'),
            row=row, col=1
        )
        fig.add_trace(
            go.Scatter(x=glla_time, y=glla_derivative, mode='lines', name=f'GLLA (noise={noise_level})'),
            row=row, col=1
        )
        fig.add_trace(
            go.Scatter(x=time, y=fda_derivative, mode='lines', name=f'FDA (noise={noise_level})'),
            row=row, col=1
        )
        # Neural network derivatives (if available)
        if TORCH_AVAILABLE:
            nn_deriv_func = neural_network_derivative(
                time, noisy_signal, framework='pytorch', hidden_layers=[64,32,16], dropout=0.025, epochs=300
            )
            nn_derivative = nn_deriv_func(time)
            fig.add_trace(
                go.Scatter(x=time, y=nn_derivative, mode='lines', name=f'NN (PyTorch, noise={noise_level})'),
                row=row, col=1
            )
            try:
                dndf_deriv_func = deep_neural_decision_forest_derivative(
                    time, noisy_signal, num_trees=5, depth=3, hidden_layers=[64,32,16], dropout=0.025, epochs=300
                )
                dndf_derivative = dndf_deriv_func(time)
                fig.add_trace(
                    go.Scatter(x=time, y=dndf_derivative, mode='lines', name=f'DNDF (PyTorch, noise={noise_level})'),
                    row=row, col=1
                )
            except Exception:
                pass
        if TF_AVAILABLE:
            nn_deriv_func_tf = neural_network_derivative(
                time, noisy_signal, framework='tensorflow', hidden_layers=[64,32,16], dropout=0.025, epochs=300
            )
            nn_derivative_tf = nn_deriv_func_tf(time)
            fig.add_trace(
                go.Scatter(x=time, y=nn_derivative_tf, mode='lines', name=f'NN (TensorFlow, noise={noise_level})'),
                row=row, col=1
            )
        # Add expected derivative
        fig.add_trace(
            go.Scatter(x=time, y=expected, mode='lines', 
                      name=f'Expected (noise={noise_level})', 
                      line=dict(color='black', dash='dash')),
            row=row, col=1
        )
        
        # Add original signal and clean signal in the second column
        fig.add_trace(
            go.Scatter(x=time, y=noisy_signal, mode='lines', 
                      name=f'Noisy Signal (noise={noise_level})',
                      line=dict(color='blue')),
            row=row, col=2
        )
        fig.add_trace(
            go.Scatter(x=time, y=clean_signal, mode='lines', 
                      name=f'Clean Signal', 
                      line=dict(color='black', dash='dash')),
            row=row, col=2
        )
    
    fig.update_layout(
        title="Effect of Noise on Different Derivative Methods",
        height=1200, 
        width=1000,
        showlegend=True
    )
    
    save_plot(fig, "noise_effect_on_derivatives.html")

def visual_test_window_size_comparison():
    """Visual test of how window size affects derivative calculation with noise."""
    time = np.linspace(0, 10, 100)
    clean_signal = np.sin(time)
    
    # Add moderate noise
    noise_level = 0.1
    noisy_signal = add_noise(clean_signal, noise_level)
    
    # Try different window sizes for LLA
    window_sizes = [3, 5, 9, 15]
    
    # Create a figure with subplots for each window size
    fig = make_subplots(
        rows=len(window_sizes), 
        cols=1,
        subplot_titles=[f"Window Size: {size}" for size in window_sizes],
        shared_xaxes=True,
        vertical_spacing=0.1
    )
    
    # Expected derivative of sine is cosine
    expected = np.cos(time)
    
    # Process each window size
    for i, window_size in enumerate(window_sizes):
        row = i + 1
        
        # Calculate derivative with current window size
        lla_derivative, _ = lla(time.tolist(), noisy_signal.tolist(), window_size=window_size)
        
        # Add traces
        fig.add_trace(
            go.Scatter(x=time, y=lla_derivative, mode='lines', 
                      name=f'LLA (window={window_size})'),
            row=row, col=1
        )
        fig.add_trace(
            go.Scatter(x=time, y=expected, mode='lines', 
                      name=f'Expected (window={window_size})', 
                      line=dict(color='black', dash='dash')),
            row=row, col=1
        )
        
        # Add original noisy signal (scaled to match derivative amplitude)
        signal_scaled = noisy_signal * max(np.abs(expected)) / max(np.abs(noisy_signal))
        fig.add_trace(
            go.Scatter(x=time, y=signal_scaled, mode='lines', 
                      name=f'Noisy Signal (scaled, window={window_size})',
                      opacity=0.3),
            row=row, col=1
        )
    
    fig.update_layout(
        title=f"Effect of Window Size on LLA Derivative (Noise Level: {noise_level})",
        height=1200, 
        width=1000,
        showlegend=True
    )
    
    save_plot(fig, "window_size_effect_on_derivatives.html")

if __name__ == "__main__":
    print("Running visual tests for derivatives module...")
    visual_test_lla_sine()
    visual_test_gold_sine()
    visual_test_glla_sine()
    visual_test_fda_sine()
    visual_test_algorithm_comparison()
    visual_test_noise_comparison()
    visual_test_window_size_comparison()
    print("All visual tests completed!")
