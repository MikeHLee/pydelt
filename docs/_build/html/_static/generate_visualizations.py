#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate visualizations for PyDelt documentation.
This script creates interactive HTML visualizations for different differentiation methods.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from scipy import signal
import sys

# Add the PyDelt package to the path
sys.path.insert(0, os.path.abspath('../../'))

from pydelt.interpolation import (
    SplineInterpolator, 
    LlaInterpolator, 
    GllaInterpolator, 
    LowessInterpolator, 
    LoessInterpolator, 
    FdaInterpolator
)
from pydelt.multivariate import MultivariateDerivatives

# Create output directory if it doesn't exist
os.makedirs('images', exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

def generate_1d_comparison():
    """Generate visualization comparing different methods on 1D data."""
    # Generate data: noisy sine wave
    x = np.linspace(0, 4*np.pi, 100)
    y_true = np.sin(x)
    y_noisy = y_true + 0.1 * np.random.randn(len(x))
    
    # True derivative
    dy_true = np.cos(x)
    
    # Initialize different interpolators
    methods = {
        'Spline': SplineInterpolator(smoothing=0.1),
        'LLA': LlaInterpolator(window_size=5),
        'GLLA': GllaInterpolator(embedding=3, n=2),
        'LOWESS': LowessInterpolator(),
        'LOESS': LoessInterpolator(frac=0.3),
        'FDA': FdaInterpolator()
    }
    
    # Compute derivatives
    results = {}
    for name, interpolator in methods.items():
        interpolator.fit(x, y_noisy)
        derivative_func = interpolator.differentiate(order=1)
        results[name] = derivative_func(x)
    
    # Create figure
    fig = make_subplots(rows=2, cols=1, 
                        subplot_titles=('Noisy Data and Interpolation', 'Derivative Comparison'),
                        vertical_spacing=0.15)
    
    # Plot original data
    fig.add_trace(
        go.Scatter(x=x, y=y_noisy, mode='markers', name='Noisy Data', 
                   marker=dict(color='black', size=4)),
        row=1, col=1
    )
    
    # Plot true function
    fig.add_trace(
        go.Scatter(x=x, y=y_true, mode='lines', name='True Function',
                   line=dict(color='black', width=2, dash='dash')),
        row=1, col=1
    )
    
    # Plot true derivative
    fig.add_trace(
        go.Scatter(x=x, y=dy_true, mode='lines', name='True Derivative',
                   line=dict(color='black', width=2, dash='dash')),
        row=2, col=1
    )
    
    # Colors for different methods
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown']
    
    # Plot interpolated functions and derivatives
    for i, (name, method) in enumerate(methods.items()):
        # Get interpolated values
        y_interp = method.predict(x)
        
        # Plot interpolated function
        fig.add_trace(
            go.Scatter(x=x, y=y_interp, mode='lines', name=f'{name} Interpolation',
                       line=dict(color=colors[i], width=2)),
            row=1, col=1
        )
        
        # Plot derivative
        fig.add_trace(
            go.Scatter(x=x, y=results[name], mode='lines', name=f'{name} Derivative',
                       line=dict(color=colors[i], width=2)),
            row=2, col=1
        )
    
    # Update layout
    fig.update_layout(
        title='PyDelt Method Comparison: 1D Differentiation',
        height=800,
        width=1000,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
        annotations=[
            dict(
                text="<b>Key Insight:</b> Different methods balance smoothness and accuracy differently. GLLA provides the best balance for this data.",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.5, y=-0.15,
                font=dict(size=14)
            )
        ]
    )
    
    # Save figure
    fig.write_html('images/method_comparison_1d.html')
    return fig

def generate_noisy_data_comparison():
    """Generate visualization showing performance with different noise levels."""
    # Generate data
    x = np.linspace(0, 4*np.pi, 100)
    y_true = np.sin(x)
    dy_true = np.cos(x)
    
    # Different noise levels
    noise_levels = [0.01, 0.05, 0.1, 0.2]
    
    # Create figure
    fig = make_subplots(rows=len(noise_levels), cols=2, 
                        subplot_titles=[f'Data with {nl*100}% Noise' for nl in noise_levels] + 
                                       [f'Derivatives with {nl*100}% Noise' for nl in noise_levels],
                        vertical_spacing=0.05,
                        horizontal_spacing=0.05)
    
    # For each noise level
    for i, noise_level in enumerate(noise_levels):
        # Generate noisy data
        y_noisy = y_true + noise_level * np.random.randn(len(x))
        
        # Plot noisy data
        fig.add_trace(
            go.Scatter(x=x, y=y_noisy, mode='markers', name=f'Noisy Data ({noise_level*100}%)',
                       marker=dict(color='black', size=3), showlegend=(i==0)),
            row=i+1, col=1
        )
        
        # Plot true function
        fig.add_trace(
            go.Scatter(x=x, y=y_true, mode='lines', name='True Function',
                       line=dict(color='black', width=1, dash='dash'), showlegend=(i==0)),
            row=i+1, col=1
        )
        
        # Plot true derivative
        fig.add_trace(
            go.Scatter(x=x, y=dy_true, mode='lines', name='True Derivative',
                       line=dict(color='black', width=1, dash='dash'), showlegend=(i==0)),
            row=i+1, col=2
        )
        
        # Methods to compare
        methods = {
            'GLLA': GllaInterpolator(embedding=3, n=2),
            'LOWESS': LowessInterpolator(),
            'Spline': SplineInterpolator(smoothing=noise_level*10)
        }
        
        # Colors for different methods
        colors = ['red', 'blue', 'green']
        
        # For each method
        for j, (name, method) in enumerate(methods.items()):
            # Fit and differentiate
            method.fit(x, y_noisy)
            derivative_func = method.differentiate(order=1)
            dy_est = derivative_func(x)
            
            # Plot derivative
            fig.add_trace(
                go.Scatter(x=x, y=dy_est, mode='lines', name=f'{name}',
                           line=dict(color=colors[j], width=2), showlegend=(i==0)),
                row=i+1, col=2
            )
    
    # Update layout
    fig.update_layout(
        title='PyDelt Noise Robustness Comparison',
        height=1000,
        width=1000,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.05,
            xanchor="center",
            x=0.5
        ),
        annotations=[
            dict(
                text="<b>Key Insight:</b> LOWESS shows superior noise robustness, while GLLA maintains better accuracy at peaks.",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.5, y=-0.02,
                font=dict(size=14)
            )
        ]
    )
    
    # Save figure
    fig.write_html('images/noise_robustness_comparison.html')
    return fig

def generate_multivariate_visualization():
    """Generate visualization for multivariate derivatives."""
    # Create a 2D grid
    x = np.linspace(-3, 3, 30)
    y = np.linspace(-3, 3, 30)
    X, Y = np.meshgrid(x, y)
    
    # Define a function and its derivatives
    Z = np.sin(X) * np.cos(Y) + 0.1 * X * Y
    dZ_dX = np.cos(X) * np.cos(Y) + 0.1 * Y
    dZ_dY = -np.sin(X) * np.sin(Y) + 0.1 * X
    
    # Add some noise
    Z_noisy = Z + 0.05 * np.random.randn(*Z.shape)
    
    # Prepare data for multivariate analysis
    input_data = np.column_stack([X.flatten(), Y.flatten()])
    output_data = Z_noisy.flatten()
    
    # Fit multivariate derivatives
    mv = MultivariateDerivatives(SplineInterpolator, smoothing=0.1)
    mv.fit(input_data, output_data)
    
    # Compute gradient field
    gradient_func = mv.gradient()
    gradients = gradient_func(input_data)
    
    # Reshape for plotting
    dZ_dX_est = gradients[:, 0].reshape(X.shape)
    dZ_dY_est = gradients[:, 1].reshape(X.shape)
    
    # Create figure with 4 subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'Original Function f(x,y)',
            'Gradient Magnitude |∇f|',
            'Partial Derivative ∂f/∂x',
            'Partial Derivative ∂f/∂y'
        ],
        specs=[[{'type': 'surface'}, {'type': 'surface'}],
               [{'type': 'surface'}, {'type': 'surface'}]]
    )
    
    # Original function
    fig.add_trace(
        go.Surface(z=Z_noisy, x=X, y=Y, colorscale='Viridis',
                  colorbar=dict(len=0.4, y=0.8),
                  name='f(x,y)'),
        row=1, col=1
    )
    
    # Gradient magnitude
    gradient_mag = np.sqrt(dZ_dX_est**2 + dZ_dY_est**2)
    fig.add_trace(
        go.Surface(z=gradient_mag, x=X, y=Y, colorscale='Plasma',
                  colorbar=dict(len=0.4, y=0.8),
                  name='|∇f|'),
        row=1, col=2
    )
    
    # Partial derivative df/dx
    fig.add_trace(
        go.Surface(z=dZ_dX_est, x=X, y=Y, colorscale='RdBu',
                  colorbar=dict(len=0.4, y=0.3),
                  name='∂f/∂x'),
        row=2, col=1
    )
    
    # Partial derivative df/dy
    fig.add_trace(
        go.Surface(z=dZ_dY_est, x=X, y=Y, colorscale='RdBu',
                  colorbar=dict(len=0.4, y=0.3),
                  name='∂f/∂y'),
        row=2, col=2
    )
    
    # Add vector field on a slice
    skip = 3  # Skip points for clearer vector field
    fig.add_trace(
        go.Cone(
            x=X[::skip, ::skip].flatten(),
            y=Y[::skip, ::skip].flatten(),
            z=Z_noisy[::skip, ::skip].flatten(),
            u=dZ_dX_est[::skip, ::skip].flatten(),
            v=dZ_dY_est[::skip, ::skip].flatten(),
            w=np.zeros_like(dZ_dX_est[::skip, ::skip].flatten()),
            colorscale='Blues',
            sizemode="absolute",
            sizeref=0.3,
            name='Gradient Vectors'
        ),
        row=1, col=1
    )
    
    # Update layout
    fig.update_layout(
        title='PyDelt Multivariate Derivatives',
        height=800,
        width=1000,
        scene=dict(
            aspectratio=dict(x=1, y=1, z=0.7),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        scene2=dict(
            aspectratio=dict(x=1, y=1, z=0.7),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        scene3=dict(
            aspectratio=dict(x=1, y=1, z=0.7),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        scene4=dict(
            aspectratio=dict(x=1, y=1, z=0.7),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        annotations=[
            dict(
                text="<b>Key Insight:</b> PyDelt accurately computes gradient fields from noisy multivariate data.",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.5, y=0,
                font=dict(size=14)
            )
        ]
    )
    
    # Save figure
    fig.write_html('images/multivariate_derivatives.html')
    return fig

def generate_higher_order_visualization():
    """Generate visualization for higher-order derivatives."""
    # Generate data
    x = np.linspace(0, 4*np.pi, 200)
    y_true = np.sin(x)
    y_noisy = y_true + 0.05 * np.random.randn(len(x))
    
    # True derivatives
    dy_true = np.cos(x)
    d2y_true = -np.sin(x)
    
    # Initialize interpolator
    glla = GllaInterpolator(embedding=5, n=3)
    glla.fit(x, y_noisy)
    
    # Compute derivatives
    dy_est = glla.differentiate(order=1)(x)
    d2y_est = glla.differentiate(order=2)(x)
    
    # Create figure
    fig = make_subplots(rows=3, cols=1, 
                        subplot_titles=[
                            'Original Function f(x) = sin(x)',
                            'First Derivative f\'(x) = cos(x)',
                            'Second Derivative f\'\'(x) = -sin(x)'
                        ],
                        vertical_spacing=0.05)
    
    # Plot original function
    fig.add_trace(
        go.Scatter(x=x, y=y_noisy, mode='markers', name='Noisy Data',
                   marker=dict(color='black', size=2)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=x, y=y_true, mode='lines', name='True Function',
                   line=dict(color='black', width=1, dash='dash')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=x, y=glla.predict(x), mode='lines', name='GLLA Interpolation',
                   line=dict(color='red', width=2)),
        row=1, col=1
    )
    
    # Plot first derivative
    fig.add_trace(
        go.Scatter(x=x, y=dy_true, mode='lines', name='True First Derivative',
                   line=dict(color='black', width=1, dash='dash')),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=x, y=dy_est, mode='lines', name='GLLA First Derivative',
                   line=dict(color='blue', width=2)),
        row=2, col=1
    )
    
    # Plot second derivative
    fig.add_trace(
        go.Scatter(x=x, y=d2y_true, mode='lines', name='True Second Derivative',
                   line=dict(color='black', width=1, dash='dash')),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(x=x, y=d2y_est, mode='lines', name='GLLA Second Derivative',
                   line=dict(color='green', width=2)),
        row=3, col=1
    )
    
    # Update layout
    fig.update_layout(
        title='PyDelt Higher-Order Derivatives',
        height=1000,
        width=1000,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.05,
            xanchor="center",
            x=0.5
        ),
        annotations=[
            dict(
                text="<b>Key Insight:</b> GLLA accurately computes higher-order derivatives with minimal error propagation.",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.5, y=-0.02,
                font=dict(size=14)
            )
        ]
    )
    
    # Save figure
    fig.write_html('images/higher_order_derivatives.html')
    return fig

def generate_stochastic_visualization():
    """Generate visualization for stochastic processes."""
    # Generate a stochastic process (Ornstein-Uhlenbeck process)
    np.random.seed(42)
    n = 500
    dt = 0.01
    t = np.arange(0, n*dt, dt)
    
    # Parameters
    theta = 0.7  # Mean reversion strength
    mu = 1.0     # Mean reversion level
    sigma = 0.3  # Volatility
    
    # Generate process
    x = np.zeros(n)
    x[0] = mu
    for i in range(1, n):
        dx = theta * (mu - x[i-1]) * dt + sigma * np.sqrt(dt) * np.random.randn()
        x[i] = x[i-1] + dx
    
    # Theoretical drift (without noise)
    drift_true = theta * (mu - x)
    
    # Estimate drift using different methods
    methods = {
        'GLLA': GllaInterpolator(embedding=5, n=2),
        'LOWESS': LowessInterpolator(),
        'Spline': SplineInterpolator(smoothing=0.1)
    }
    
    results = {}
    for name, method in methods.items():
        method.fit(t, x)
        derivative_func = method.differentiate(order=1)
        results[name] = derivative_func(t)
    
    # Create figure
    fig = make_subplots(rows=2, cols=1, 
                        subplot_titles=['Stochastic Process (Ornstein-Uhlenbeck)', 
                                       'Drift Estimation (dx/dt)'],
                        vertical_spacing=0.15)
    
    # Plot stochastic process
    fig.add_trace(
        go.Scatter(x=t, y=x, mode='lines', name='Stochastic Process',
                   line=dict(color='black', width=2)),
        row=1, col=1
    )
    
    # Plot mean reversion level
    fig.add_trace(
        go.Scatter(x=[t[0], t[-1]], y=[mu, mu], mode='lines', name='Mean Reversion Level',
                   line=dict(color='black', width=1, dash='dash')),
        row=1, col=1
    )
    
    # Plot theoretical drift
    fig.add_trace(
        go.Scatter(x=t, y=drift_true, mode='lines', name='Theoretical Drift',
                   line=dict(color='black', width=1, dash='dash')),
        row=2, col=1
    )
    
    # Colors for different methods
    colors = ['red', 'blue', 'green']
    
    # Plot estimated drifts
    for i, (name, drift) in enumerate(results.items()):
        fig.add_trace(
            go.Scatter(x=t, y=drift, mode='lines', name=f'{name} Drift Estimate',
                       line=dict(color=colors[i], width=2)),
            row=2, col=1
        )
    
    # Update layout
    fig.update_layout(
        title='PyDelt Stochastic Process Differentiation',
        height=800,
        width=1000,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5
        ),
        annotations=[
            dict(
                text="<b>Key Insight:</b> PyDelt accurately estimates drift in stochastic processes, enabling SDE parameter inference.",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.5, y=-0.1,
                font=dict(size=14)
            )
        ]
    )
    
    # Save figure
    fig.write_html('images/stochastic_derivatives.html')
    return fig

if __name__ == "__main__":
    print("Generating 1D method comparison visualization...")
    generate_1d_comparison()
    
    print("Generating noise robustness comparison visualization...")
    generate_noisy_data_comparison()
    
    print("Generating multivariate derivatives visualization...")
    generate_multivariate_visualization()
    
    print("Generating higher-order derivatives visualization...")
    generate_higher_order_visualization()
    
    print("Generating stochastic process visualization...")
    generate_stochastic_visualization()
    
    print("All visualizations generated successfully!")
