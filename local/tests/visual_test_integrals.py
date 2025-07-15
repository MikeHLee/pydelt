"""
Visual tests for the integrals module.
Creates plotly visualizations to compare expected vs actual results.
Includes tests with varying levels of noise.
"""

import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pydelt.integrals import integrate_derivative, integrate_derivative_with_error
from pydelt.derivatives import lla

# Ensure output directory exists
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_plot(fig, filename):
    """Save a plotly figure to the output directory."""
    filepath = os.path.join(OUTPUT_DIR, filename)
    fig.write_html(filepath)
    print(f"Plot saved to {filepath}")

def visual_test_integrate_constant_derivative():
    """Visual test of integration of a constant derivative (should give linear function)."""
    time = np.linspace(0, 10, 100)
    derivative = np.ones_like(time)  # constant derivative = 1
    integral = integrate_derivative(time, derivative)
    
    # Should approximate y = x
    expected = time
    
    # Create plot
    fig = make_subplots(rows=2, cols=1, 
                        subplot_titles=("Integrated vs Expected", "Difference"),
                        shared_xaxes=True)
    
    fig.add_trace(
        go.Scatter(x=time, y=integral, mode='lines', name='Integrated'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=time, y=expected, mode='lines', name='Expected (y=x)'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=time, y=integral-expected, mode='lines', name='Difference'),
        row=2, col=1
    )
    
    fig.update_layout(title="Integration of Constant Derivative",
                     height=800, width=1000)
    
    save_plot(fig, "integrate_constant_derivative.html")

def visual_test_integrate_sine():
    """Visual test of integration of cosine (derivative of sine) should give sine."""
    time = np.linspace(0, 10, 500)
    derivative = np.cos(time)
    integral = integrate_derivative(time, derivative)
    
    # Should approximate sine
    expected = np.sin(time)
    
    # Create plot
    fig = make_subplots(rows=2, cols=1, 
                        subplot_titles=("Integrated vs Expected", "Difference"),
                        shared_xaxes=True)
    
    fig.add_trace(
        go.Scatter(x=time, y=integral, mode='lines', name='Integrated'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=time, y=expected, mode='lines', name='Expected (sine)'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=time, y=integral-expected, mode='lines', name='Difference'),
        row=2, col=1
    )
    
    fig.update_layout(title="Integration of Cosine to Sine",
                     height=800, width=1000)
    
    save_plot(fig, "integrate_sine.html")

def visual_test_integrate_with_initial_value():
    """Visual test of integration with non-zero initial value."""
    time = np.linspace(0, 10, 100)
    derivative = np.ones_like(time)
    initial_value = 5.0
    integral = integrate_derivative(time, derivative, initial_value=initial_value)
    
    # Should approximate y = x + 5
    expected = time + initial_value
    
    # Create plot
    fig = make_subplots(rows=2, cols=1, 
                        subplot_titles=("Integrated vs Expected", "Difference"),
                        shared_xaxes=True)
    
    fig.add_trace(
        go.Scatter(x=time, y=integral, mode='lines', name='Integrated'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=time, y=expected, mode='lines', name='Expected (y=x+5)'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=time, y=integral-expected, mode='lines', name='Difference'),
        row=2, col=1
    )
    
    fig.update_layout(title="Integration with Initial Value",
                     height=800, width=1000)
    
    save_plot(fig, "integrate_with_initial_value.html")

def visual_test_integrate_with_error():
    """Visual test of error estimation in integration."""
    time = np.linspace(0, 10, 500)
    signal = np.sin(time)
    derivative, _ = lla(time.tolist(), signal.tolist(), window_size=5)
    
    reconstructed, error = integrate_derivative_with_error(time, derivative, initial_value=signal[0])
    
    # Create plot
    fig = make_subplots(rows=2, cols=1, 
                        subplot_titles=("Reconstructed vs Original", "Error"),
                        shared_xaxes=True)
    
    fig.add_trace(
        go.Scatter(x=time, y=reconstructed, mode='lines', name='Reconstructed'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=time, y=signal, mode='lines', name='Original Signal'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=time, y=error, mode='lines', name='Error Estimate'),
        row=2, col=1
    )
    
    fig.update_layout(title="Integration with Error Estimation",
                     height=800, width=1000)
    
    save_plot(fig, "integrate_with_error.html")

def visual_test_input_types():
    """Visual test that functions accept both lists and numpy arrays."""
    time = [0, 1, 2, 3, 4]
    derivative = [1, 1, 1, 1, 1]
    
    # Test with lists
    result_list = integrate_derivative(time, derivative)
    
    # Test with numpy arrays
    result_array = integrate_derivative(np.array(time), np.array(derivative))
    
    # Create plot
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(x=time, y=result_list, mode='lines+markers', name='Result from Lists')
    )
    fig.add_trace(
        go.Scatter(x=time, y=result_array, mode='lines+markers', name='Result from Arrays')
    )
    
    fig.update_layout(title="Integration with Different Input Types",
                     height=600, width=1000)
    
    save_plot(fig, "integrate_input_types.html")

def add_noise(signal, noise_level):
    """Add Gaussian noise to a signal."""
    np.random.seed(42)  # For reproducibility
    noise = np.random.normal(0, noise_level, size=signal.shape)
    return signal + noise

def visual_test_noise_effect_on_integration():
    """Visual test of how noise affects integration results."""
    time = np.linspace(0, 10, 500)
    clean_derivative = np.cos(time)  # Derivative of sine
    
    # Create derivatives with different noise levels
    noise_levels = [0.01, 0.05, 0.1, 0.2]
    noisy_derivatives = [add_noise(clean_derivative, level) for level in noise_levels]
    
    # Create a figure with subplots for each noise level
    fig = make_subplots(
        rows=len(noise_levels), 
        cols=2,
        subplot_titles=[f"Noise Level: {level} - Integration Results" for level in noise_levels] + 
                      [f"Noise Level: {level} - Original Derivatives" for level in noise_levels],
        shared_xaxes=True,
        vertical_spacing=0.1,
        horizontal_spacing=0.05,
        column_widths=[0.7, 0.3]
    )
    
    # Expected integral of cosine is sine
    expected = np.sin(time)
    
    # Process each noise level
    for i, (noisy_derivative, noise_level) in enumerate(zip(noisy_derivatives, noise_levels)):
        row = i + 1
        
        # Integrate the noisy derivative
        integral = integrate_derivative(time, noisy_derivative)
        
        # Add traces for integration results in the first column
        fig.add_trace(
            go.Scatter(x=time, y=integral, mode='lines', name=f'Integrated (noise={noise_level})'),
            row=row, col=1
        )
        fig.add_trace(
            go.Scatter(x=time, y=expected, mode='lines', 
                      name=f'Expected (sine, noise={noise_level})', 
                      line=dict(color='black', dash='dash')),
            row=row, col=1
        )
        fig.add_trace(
            go.Scatter(x=time, y=integral-expected, mode='lines', 
                      name=f'Difference (noise={noise_level})',
                      line=dict(color='red')),
            row=row, col=1
        )
        
        # Add original derivative data in the second column
        fig.add_trace(
            go.Scatter(x=time, y=noisy_derivative, mode='lines', 
                      name=f'Noisy Derivative (noise={noise_level})',
                      line=dict(color='blue')),
            row=row, col=2
        )
        fig.add_trace(
            go.Scatter(x=time, y=clean_derivative, mode='lines', 
                      name=f'Clean Derivative', 
                      line=dict(color='black', dash='dash')),
            row=row, col=2
        )
    
    fig.update_layout(
        title="Effect of Noise on Integration Results",
        height=1200, 
        width=1000,
        showlegend=True
    )
    
    save_plot(fig, "noise_effect_on_integration.html")

def visual_test_derivative_reconstruction_with_noise():
    """Visual test of derivative calculation and reconstruction with noise."""
    time = np.linspace(0, 10, 500)
    
    # Create original signals with different noise levels
    clean_signal = np.sin(time)
    noise_levels = [0.01, 0.05, 0.1, 0.2]
    noisy_signals = [add_noise(clean_signal, level) for level in noise_levels]
    
    # Create a figure with subplots for each noise level
    fig = make_subplots(
        rows=len(noise_levels), 
        cols=1,
        subplot_titles=[f"Noise Level: {level}" for level in noise_levels],
        shared_xaxes=True,
        vertical_spacing=0.1
    )
    
    # Process each noise level
    for i, (noisy_signal, noise_level) in enumerate(zip(noisy_signals, noise_levels)):
        row = i + 1
        
        # Calculate derivative
        derivative, _ = lla(time.tolist(), noisy_signal.tolist(), window_size=5)
        
        # Reconstruct signal by integrating the derivative
        reconstructed, error = integrate_derivative_with_error(time, derivative, initial_value=noisy_signal[0])
        
        # Add traces
        fig.add_trace(
            go.Scatter(x=time, y=noisy_signal, mode='lines', 
                      name=f'Original (noise={noise_level})',
                      line=dict(color='blue')),
            row=row, col=1
        )
        fig.add_trace(
            go.Scatter(x=time, y=reconstructed, mode='lines', 
                      name=f'Reconstructed (noise={noise_level})',
                      line=dict(color='green')),
            row=row, col=1
        )
        fig.add_trace(
            go.Scatter(x=time, y=clean_signal, mode='lines', 
                      name=f'Clean Signal (noise={noise_level})',
                      line=dict(color='black', dash='dash')),
            row=row, col=1
        )
        fig.add_trace(
            go.Scatter(x=time, y=error, mode='lines', 
                      name=f'Error Estimate (noise={noise_level})',
                      line=dict(color='red'),
                      opacity=0.5),
            row=row, col=1
        )
    
    fig.update_layout(
        title="Signal Reconstruction from Noisy Derivatives",
        height=1200, 
        width=1000,
        showlegend=True
    )
    
    save_plot(fig, "reconstruction_with_noise.html")

if __name__ == "__main__":
    print("Running visual tests for integrals module...")
    visual_test_integrate_constant_derivative()
    visual_test_integrate_sine()
    visual_test_integrate_with_initial_value()
    visual_test_integrate_with_error()
    visual_test_input_types()
    visual_test_noise_effect_on_integration()
    visual_test_derivative_reconstruction_with_noise()
    print("All visual tests completed!")
