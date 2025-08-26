#!/usr/bin/env python3
"""
Generate beautiful Plotly visualizations for pydelt documentation.
This script creates interactive plots demonstrating key features and saves them as HTML.
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from pathlib import Path

# Import pydelt modules
from pydelt.interpolation import SplineInterpolator, LlaInterpolator, GllaInterpolator
from pydelt.multivariate import MultivariateDerivatives

def create_universal_api_demo():
    """Create visualization showing universal differentiation API across methods."""
    
    # Generate test data: f(x) = sin(x) + 0.1*sin(10*x) + noise
    np.random.seed(42)
    x = np.linspace(0, 2*np.pi, 100)
    y_clean = np.sin(x) + 0.1*np.sin(10*x)
    noise = 0.05 * np.random.randn(len(x))
    y_noisy = y_clean + noise
    
    # True derivative: cos(x) + cos(10*x)
    x_eval = np.linspace(0, 2*np.pi, 200)
    y_true = np.cos(x_eval) + np.cos(10*x_eval)
    
    # Test different interpolators
    methods = {
        'Spline': SplineInterpolator(smoothing=0.1),
        'LLA': LlaInterpolator(window_size=7),
        'GLLA': GllaInterpolator(embedding=3, n=2)
    }
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Original Data with Noise', 'Derivative Comparison', 
                       'Error Analysis', 'Method Performance'),
        specs=[[{"colspan": 2}, None],
               [{"type": "scatter"}, {"type": "bar"}]]
    )
    
    # Plot 1: Original data
    fig.add_trace(go.Scatter(
        x=x, y=y_noisy, mode='markers', name='Noisy Data',
        marker=dict(size=4, color='lightblue', opacity=0.6)
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=x, y=y_clean, mode='lines', name='True Function',
        line=dict(color='black', width=2)
    ), row=1, col=1)
    
    # Plot 2: Derivatives
    fig.add_trace(go.Scatter(
        x=x_eval, y=y_true, mode='lines', name='True Derivative',
        line=dict(color='black', width=3, dash='dash')
    ), row=2, col=1)
    
    colors = ['red', 'blue', 'green']
    errors = []
    
    for i, (name, interpolator) in enumerate(methods.items()):
        # Fit and compute derivative
        interpolator.fit(x, y_noisy)
        derivative_func = interpolator.differentiate(order=1)
        y_pred = derivative_func(x_eval)
        
        # Calculate error
        error = np.mean(np.abs(y_pred - y_true))
        errors.append(error)
        
        fig.add_trace(go.Scatter(
            x=x_eval, y=y_pred, mode='lines', name=f'{name} Derivative',
            line=dict(color=colors[i], width=2)
        ), row=2, col=1)
    
    # Plot 3: Error bars
    fig.add_trace(go.Bar(
        x=list(methods.keys()), y=errors, name='Mean Absolute Error',
        marker_color=['red', 'blue', 'green']
    ), row=2, col=2)
    
    fig.update_layout(
        title="Universal Differentiation API: Consistent Interface Across Methods",
        height=800,
        showlegend=True
    )
    
    fig.update_xaxes(title_text="x", row=1, col=1)
    fig.update_yaxes(title_text="f(x)", row=1, col=1)
    fig.update_xaxes(title_text="x", row=2, col=1)
    fig.update_yaxes(title_text="f'(x)", row=2, col=1)
    fig.update_xaxes(title_text="Method", row=2, col=2)
    fig.update_yaxes(title_text="Error", row=2, col=2)
    
    return fig

def create_multivariate_surface():
    """Create 3D surface plot for multivariate calculus demonstration."""
    
    # Generate 2D function: f(x,y) = sin(x)*cos(y) + 0.1*x*y
    x = np.linspace(-3, 3, 30)
    y = np.linspace(-3, 3, 30)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(X) * np.cos(Y) + 0.1 * X * Y
    
    # Prepare data for multivariate derivatives
    input_data = np.column_stack([X.flatten(), Y.flatten()])
    output_data = Z.flatten()
    
    # Compute gradient using pydelt
    mv = MultivariateDerivatives(SplineInterpolator, smoothing=0.1)
    mv.fit(input_data, output_data)
    gradient_func = mv.gradient()
    
    # Evaluate gradient on a grid
    x_grad = np.linspace(-2, 2, 15)
    y_grad = np.linspace(-2, 2, 15)
    X_grad, Y_grad = np.meshgrid(x_grad, y_grad)
    grad_points = np.column_stack([X_grad.flatten(), Y_grad.flatten()])
    gradients = gradient_func(grad_points)
    
    # Reshape gradients
    grad_x = gradients[:, 0].reshape(X_grad.shape)
    grad_y = gradients[:, 1].reshape(Y_grad.shape)
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('3D Function Surface', 'Gradient Vector Field', 
                       'Gradient X-Component', 'Gradient Y-Component'),
        specs=[[{"type": "surface"}, {"type": "scatter"}],
               [{"type": "heatmap"}, {"type": "heatmap"}]]
    )
    
    # 3D surface
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z,
        colorscale='Viridis',
        name='f(x,y)'
    ), row=1, col=1)
    
    # Vector field
    fig.add_trace(go.Scatter(
        x=X_grad.flatten(), y=Y_grad.flatten(),
        mode='markers',
        marker=dict(
            size=8,
            color=np.sqrt(grad_x.flatten()**2 + grad_y.flatten()**2),
            colorscale='Plasma',
            showscale=True,
            colorbar=dict(title="Gradient Magnitude")
        ),
        name='Gradient Field'
    ), row=1, col=2)
    
    # Add arrows for vector field
    for i in range(0, len(x_grad), 2):
        for j in range(0, len(y_grad), 2):
            fig.add_annotation(
                x=X_grad[i,j], y=Y_grad[i,j],
                ax=X_grad[i,j] + 0.3*grad_x[i,j], ay=Y_grad[i,j] + 0.3*grad_y[i,j],
                xref="x2", yref="y2", axref="x2", ayref="y2",
                arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="red"
            )
    
    # Gradient components
    fig.add_trace(go.Heatmap(
        x=x_grad, y=y_grad, z=grad_x,
        colorscale='RdBu', name='∂f/∂x'
    ), row=2, col=1)
    
    fig.add_trace(go.Heatmap(
        x=x_grad, y=y_grad, z=grad_y,
        colorscale='RdBu', name='∂f/∂y'
    ), row=2, col=2)
    
    fig.update_layout(
        title="Multivariate Calculus: Gradient Computation for f(x,y) = sin(x)cos(y) + 0.1xy",
        height=900
    )
    
    return fig

def create_method_comparison():
    """Create comparison of different interpolation methods."""
    
    # Generate challenging test function with multiple features
    np.random.seed(42)
    x = np.linspace(0, 4*np.pi, 80)
    y_true = np.sin(x) * np.exp(-x/8) + 0.3*np.sin(5*x)
    noise = 0.1 * np.random.randn(len(x))
    y_noisy = y_true + noise
    
    # Evaluation points
    x_eval = np.linspace(0, 4*np.pi, 200)
    y_eval_true = np.sin(x_eval) * np.exp(-x_eval/8) + 0.3*np.sin(5*x_eval)
    
    methods = {
        'Spline (s=0.1)': SplineInterpolator(smoothing=0.1),
        'Spline (s=1.0)': SplineInterpolator(smoothing=1.0),
        'LLA (w=5)': LlaInterpolator(window_size=5),
        'LLA (w=15)': LlaInterpolator(window_size=15),
    }
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Method Comparison', 'Residuals', 
                       'Parameter Sensitivity', 'Computational Performance'),
        specs=[[{"colspan": 2}, None],
               [{"type": "scatter"}, {"type": "bar"}]]
    )
    
    # Original data
    fig.add_trace(go.Scatter(
        x=x, y=y_noisy, mode='markers', name='Noisy Data',
        marker=dict(size=4, color='lightgray', opacity=0.7)
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=x_eval, y=y_eval_true, mode='lines', name='True Function',
        line=dict(color='black', width=3)
    ), row=1, col=1)
    
    colors = px.colors.qualitative.Set1
    rmse_values = []
    
    for i, (name, interpolator) in enumerate(methods.items()):
        # Fit interpolator
        interpolator.fit(x, y_noisy)
        
        # Get interpolated values (not derivatives for this comparison)
        if hasattr(interpolator, 'interpolate'):
            y_pred = interpolator.interpolate(x_eval)
        else:
            # For methods without direct interpolation, use the fitted function
            try:
                y_pred = interpolator.spline(x_eval)
            except:
                # Fallback: evaluate at training points and interpolate
                from scipy.interpolate import interp1d
                y_fit = interpolator.fit(x, y_noisy)
                if hasattr(y_fit, '__call__'):
                    y_pred = y_fit(x_eval)
                else:
                    # Simple linear interpolation as fallback
                    interp = interp1d(x, y_noisy, kind='linear', fill_value='extrapolate')
                    y_pred = interp(x_eval)
        
        # Calculate RMSE
        rmse = np.sqrt(np.mean((y_pred - y_eval_true)**2))
        rmse_values.append(rmse)
        
        fig.add_trace(go.Scatter(
            x=x_eval, y=y_pred, mode='lines', name=name,
            line=dict(color=colors[i % len(colors)], width=2)
        ), row=1, col=1)
        
        # Residuals
        residuals = y_pred - y_eval_true
        fig.add_trace(go.Scatter(
            x=x_eval, y=residuals, mode='lines', name=f'{name} Residuals',
            line=dict(color=colors[i % len(colors)], width=1, dash='dot'),
            showlegend=False
        ), row=2, col=1)
    
    # RMSE comparison
    fig.add_trace(go.Bar(
        x=list(methods.keys()), y=rmse_values, name='RMSE',
        marker_color=colors[:len(methods)]
    ), row=2, col=2)
    
    fig.update_layout(
        title="Method Comparison: Interpolation Performance on Complex Function",
        height=800,
        showlegend=True
    )
    
    fig.update_xaxes(title_text="x", row=1, col=1)
    fig.update_yaxes(title_text="f(x)", row=1, col=1)
    fig.update_xaxes(title_text="x", row=2, col=1)
    fig.update_yaxes(title_text="Residual", row=2, col=1)
    fig.update_xaxes(title_text="Method", row=2, col=2)
    fig.update_yaxes(title_text="RMSE", row=2, col=2)
    
    return fig

def main():
    """Generate all visualizations and save to docs directory."""
    
    # Create output directory
    output_dir = Path("docs/_static/images")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating visualizations...")
    
    # Generate plots
    plots = {
        "universal_api_demo.html": create_universal_api_demo(),
        "multivariate_surface.html": create_multivariate_surface(),
        "method_comparison.html": create_method_comparison()
    }
    
    # Save plots
    for filename, fig in plots.items():
        filepath = output_dir / filename
        fig.write_html(str(filepath))
        print(f"Saved: {filepath}")
    
    print("\nAll visualizations generated successfully!")
    print(f"Files saved to: {output_dir}")

if __name__ == "__main__":
    main()
