#!/usr/bin/env python3
"""
Visual tests for multivariate derivatives functionality.
"""

import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.subplots import make_subplots
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.pydelt.multivariate import MultivariateDerivatives
from src.pydelt.interpolation import SplineInterpolator, LlaInterpolator

# Output directory
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def test_2d_scalar_gradient_surface():
    """Visualize 2D scalar function with gradient vectors."""
    print("Creating 2D scalar gradient surface visualization...")
    
    # Create test data: f(x,y) = x^2 + y^2 - 2*x*y
    x = np.linspace(-3, 3, 25)
    y = np.linspace(-3, 3, 25)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + Y**2 - 2*X*Y
    
    # Fit multivariate derivatives
    input_data = np.column_stack([X.flatten(), Y.flatten()])
    output_data = Z.flatten()
    
    mv = MultivariateDerivatives(SplineInterpolator, smoothing=0.1)
    mv.fit(input_data, output_data)
    
    # Evaluation grid
    x_eval = np.linspace(-3, 3, 15)
    y_eval = np.linspace(-3, 3, 15)
    X_eval, Y_eval = np.meshgrid(x_eval, y_eval)
    eval_points = np.column_stack([X_eval.flatten(), Y_eval.flatten()])
    
    # Compute gradients
    gradient_func = mv.gradient()
    gradients = gradient_func(eval_points)
    grad_x = gradients[:, 0].reshape(X_eval.shape)
    grad_y = gradients[:, 1].reshape(X_eval.shape)
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Create visualization with detailed annotations
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Function Surface f(x,y) = x² + y² - 2xy<br><sub>Original scalar function - shows the "landscape"</sub>', 
            'Gradient Magnitude |∇f| = √((∂f/∂x)² + (∂f/∂y)²)<br><sub>Shows steepness at each point (higher = steeper slope)</sub>',
            'Gradient Vector Field ∇f = (∂f/∂x, ∂f/∂y)<br><sub>Red arrows point in direction of steepest ascent</sub>', 
            'Gradient Components vs Position<br><sub>Blue: ∂f/∂x (x-direction slope), Red: ∂f/∂y (y-direction slope)</sub>'
        ),
        specs=[[{'type': 'surface'}, {'type': 'surface'}],
               [{'type': 'scatter'}, {'type': 'scatter'}]],
        vertical_spacing=0.12
    )
    
    # Function surface
    fig.add_trace(go.Surface(x=X, y=Y, z=Z, colorscale='Viridis', showscale=False), row=1, col=1)
    
    # Gradient magnitude surface
    fig.add_trace(go.Surface(x=X_eval, y=Y_eval, z=grad_magnitude, 
                           colorscale='Plasma', showscale=False), row=1, col=2)
    
    # Vector field
    fig.add_trace(go.Scatter(x=X_eval.flatten(), y=Y_eval.flatten(), mode='markers',
                           marker=dict(size=6, color=grad_magnitude.flatten(), 
                                     colorscale='Plasma', showscale=True)), row=2, col=1)
    
    # Add gradient vectors (subsampled)
    for i in range(0, len(eval_points), 4):
        x_pos, y_pos = eval_points[i]
        dx, dy = gradients[i] * 0.15
        fig.add_trace(go.Scatter(x=[x_pos, x_pos + dx], y=[y_pos, y_pos + dy],
                               mode='lines', line=dict(color='red', width=2),
                               showlegend=False), row=2, col=1)
    
    # Gradient components
    fig.add_trace(go.Scatter(x=X_eval.flatten(), y=grad_x.flatten(), mode='markers',
                           marker=dict(size=4, color='blue'), name='∂f/∂x'), row=2, col=2)
    fig.add_trace(go.Scatter(x=Y_eval.flatten(), y=grad_y.flatten(), mode='markers',
                           marker=dict(size=4, color='red'), name='∂f/∂y'), row=2, col=2)
    
    # Add comprehensive title and annotations
    fig.update_layout(
        title={
            'text': "2D Scalar Function: Gradient Analysis<br><sub>" +
                   "INTERPRETATION GUIDE: Top-left shows original function surface. " +
                   "Top-right shows gradient magnitude (steepness). " +
                   "Bottom-left shows gradient vectors (direction of steepest ascent). " +
                   "Bottom-right shows how each gradient component varies with position.</sub>",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16}
        },
        height=900,
        annotations=[
            dict(text="Higher values = steeper slopes", x=0.75, y=0.95, xref="paper", yref="paper",
                 showarrow=False, font=dict(size=10, color="blue")),
            dict(text="Arrow length ∝ gradient magnitude", x=0.25, y=0.45, xref="paper", yref="paper",
                 showarrow=False, font=dict(size=10, color="red")),
            dict(text="Scatter shows component values", x=0.75, y=0.45, xref="paper", yref="paper",
                 showarrow=False, font=dict(size=10, color="green"))
        ]
    )
    
    # Update 3D scene properties
    fig.update_scenes(camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)))
    
    # Update axis labels
    fig.update_xaxes(title_text="x position", row=2, col=1)
    fig.update_yaxes(title_text="y position", row=2, col=1)
    fig.update_xaxes(title_text="Position coordinate", row=2, col=2)
    fig.update_yaxes(title_text="Partial derivative value", row=2, col=2)
    
    output_file = os.path.join(OUTPUT_DIR, 'multivariate_2d_scalar_gradient.html')
    fig.write_html(output_file)
    print(f"✅ Saved to {output_file}")

def test_2d_vector_field_jacobian():
    """Visualize 2D vector field with Jacobian analysis."""
    print("Creating 2D vector field Jacobian visualization...")
    
    # Vector field: F(x,y) = [x^2 - y^2, 2xy]
    x = np.linspace(-2, 2, 20)
    y = np.linspace(-2, 2, 20)
    X, Y = np.meshgrid(x, y)
    F1 = X**2 - Y**2
    F2 = 2*X*Y
    
    input_data = np.column_stack([X.flatten(), Y.flatten()])
    output_data = np.column_stack([F1.flatten(), F2.flatten()])
    
    mv = MultivariateDerivatives(SplineInterpolator, smoothing=0.1)
    mv.fit(input_data, output_data)
    
    # Evaluation
    x_eval = np.linspace(-2, 2, 12)
    y_eval = np.linspace(-2, 2, 12)
    X_eval, Y_eval = np.meshgrid(x_eval, y_eval)
    eval_points = np.column_stack([X_eval.flatten(), Y_eval.flatten()])
    
    jacobian_func = mv.jacobian()
    jacobians = jacobian_func(eval_points)
    
    # Extract components
    dF1_dx = jacobians[:, 0, 0].reshape(X_eval.shape)
    dF1_dy = jacobians[:, 0, 1].reshape(X_eval.shape)
    dF2_dx = jacobians[:, 1, 0].reshape(X_eval.shape)
    dF2_dy = jacobians[:, 1, 1].reshape(X_eval.shape)
    
    divergence = dF1_dx + dF2_dy
    curl = dF2_dx - dF1_dy
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Vector Field F(x,y) = [x²-y², 2xy]<br><sub>Red arrows show vector direction & magnitude</sub>',
            'Divergence ∇·F = ∂F1/∂x + ∂F2/∂y<br><sub>Positive (red) = expansion, Negative (blue) = contraction</sub>',
            'Curl ∇×F = ∂F2/∂x - ∂F1/∂y<br><sub>Positive (red) = counterclockwise rotation</sub>',
            'Jacobian Component ∂F1/∂x<br><sub>Shows how F1 component changes with x</sub>'
        ),
        specs=[[{'type': 'scatter'}, {'type': 'heatmap'}],
               [{'type': 'heatmap'}, {'type': 'heatmap'}]],
        vertical_spacing=0.12
    )
    
    # Vector field
    step = 2
    X_sub = X_eval[::step, ::step]
    Y_sub = Y_eval[::step, ::step]
    F1_eval = (X_eval**2 - Y_eval**2)[::step, ::step]
    F2_eval = (2*X_eval*Y_eval)[::step, ::step]
    
    fig.add_trace(go.Scatter(x=X_sub.flatten(), y=Y_sub.flatten(), mode='markers',
                           marker=dict(size=4, color='blue')), row=1, col=1)
    
    # Add vectors
    for i in range(X_sub.shape[0]):
        for j in range(X_sub.shape[1]):
            x_pos, y_pos = X_sub[i, j], Y_sub[i, j]
            dx, dy = F1_eval[i, j] * 0.08, F2_eval[i, j] * 0.08
            fig.add_trace(go.Scatter(x=[x_pos, x_pos + dx], y=[y_pos, y_pos + dy],
                                   mode='lines', line=dict(color='red', width=2),
                                   showlegend=False), row=1, col=1)
    
    # Heatmaps
    fig.add_trace(go.Heatmap(x=x_eval, y=y_eval, z=divergence, colorscale='RdBu'), row=1, col=2)
    fig.add_trace(go.Heatmap(x=x_eval, y=y_eval, z=curl, colorscale='RdBu'), row=2, col=1)
    fig.add_trace(go.Heatmap(x=x_eval, y=y_eval, z=dF1_dx, colorscale='Viridis'), row=2, col=2)
    
    # Add comprehensive title and annotations
    fig.update_layout(
        title={
            'text': "2D Vector Field: Jacobian Analysis F(x,y) = [x²-y², 2xy]<br><sub>" +
                   "INTERPRETATION GUIDE: Top-left shows vector field (arrows). " +
                   "Top-right shows divergence (expansion/contraction). " +
                   "Bottom-left shows curl (rotation). " +
                   "Bottom-right shows one Jacobian component.</sub>",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16}
        },
        height=800,
        annotations=[
            dict(text="Arrow length ∝ vector magnitude", x=0.25, y=0.95, xref="paper", yref="paper",
                 showarrow=False, font=dict(size=10, color="red")),
            dict(text="Red = sources, Blue = sinks", x=0.75, y=0.95, xref="paper", yref="paper",
                 showarrow=False, font=dict(size=10, color="blue")),
            dict(text="Measures local rotation", x=0.25, y=0.45, xref="paper", yref="paper",
                 showarrow=False, font=dict(size=10, color="green")),
            dict(text="Jacobian matrix element", x=0.75, y=0.45, xref="paper", yref="paper",
                 showarrow=False, font=dict(size=10, color="purple"))
        ]
    )
    
    output_file = os.path.join(OUTPUT_DIR, 'multivariate_2d_vector_jacobian.html')
    fig.write_html(output_file)
    print(f"✅ Saved to {output_file}")

def test_3d_scalar_derivatives():
    """Visualize 3D scalar function derivatives."""
    print("Creating 3D scalar derivatives visualization...")
    
    # 3D function sampled points
    np.random.seed(42)
    n_samples = 300
    x_samples = np.random.uniform(-2, 2, n_samples)
    y_samples = np.random.uniform(-2, 2, n_samples)
    z_samples = np.random.uniform(-2, 2, n_samples)
    f_samples = x_samples**2 + y_samples**2 + z_samples**2 - x_samples*y_samples
    
    input_data = np.column_stack([x_samples, y_samples, z_samples])
    mv = MultivariateDerivatives(LlaInterpolator, window_size=10)
    mv.fit(input_data, f_samples)
    
    # Evaluate on z=0 slice
    x_eval = np.linspace(-2, 2, 20)
    y_eval = np.linspace(-2, 2, 20)
    X_eval, Y_eval = np.meshgrid(x_eval, y_eval)
    z_eval = np.zeros_like(X_eval)
    eval_points_2d = np.column_stack([X_eval.flatten(), Y_eval.flatten(), z_eval.flatten()])
    
    gradient_func = mv.gradient()
    laplacian_func = mv.laplacian()
    gradients = gradient_func(eval_points_2d)
    
    laplacians = laplacian_func(eval_points_2d)
    
    grad_magnitude = np.sqrt(np.sum(gradients**2, axis=1)).reshape(X_eval.shape)
    laplacian_2d = laplacians.reshape(X_eval.shape)
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=(
            '3D Sample Points f(x,y,z) = x²+y²+z²-xy<br><sub>Color shows function values at random 3D points</sub>',
            'Gradient Magnitude |∇f| at z=0 slice<br><sub>Shows steepness in the z=0 plane</sub>',
            'Laplacian ∇²f = ∂²f/∂x² + ∂²f/∂y² + ∂²f/∂z²<br><sub>Red = local maxima, Blue = local minima</sub>'
        ),
        specs=[[{'type': 'scatter3d'}, {'type': 'heatmap'}, {'type': 'heatmap'}]],
        horizontal_spacing=0.08
    )
    
    fig.add_trace(go.Scatter3d(x=x_samples, y=y_samples, z=z_samples, mode='markers',
                             marker=dict(size=3, color=f_samples, colorscale='Viridis', showscale=True)), 
                 row=1, col=1)
    fig.add_trace(go.Heatmap(x=x_eval, y=y_eval, z=grad_magnitude, colorscale='Plasma'), row=1, col=2)
    fig.add_trace(go.Heatmap(x=x_eval, y=y_eval, z=laplacian_2d, colorscale='RdBu'), row=1, col=3)
    
    # Add comprehensive title and annotations
    fig.update_layout(
        title={
            'text': "3D Scalar Function: f(x,y,z) = x²+y²+z²-xy<br><sub>" +
                   "INTERPRETATION GUIDE: Left shows 3D training data points. " +
                   "Center shows gradient magnitude (steepness) on z=0 slice. " +
                   "Right shows Laplacian (curvature) indicating local extrema.</sub>",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16}
        },
        height=700,
        annotations=[
            dict(text="Point color = function value", x=0.17, y=0.95, xref="paper", yref="paper",
                 showarrow=False, font=dict(size=10, color="blue")),
            dict(text="Higher values = steeper slopes", x=0.5, y=0.95, xref="paper", yref="paper",
                 showarrow=False, font=dict(size=10, color="red")),
            dict(text="Positive = concave up (minima)", x=0.83, y=0.95, xref="paper", yref="paper",
                 showarrow=False, font=dict(size=10, color="green"))
        ]
    )
    
    fig.update_scenes(camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)))
    
    output_file = os.path.join(OUTPUT_DIR, 'multivariate_3d_scalar_derivatives.html')
    fig.write_html(output_file)
    print(f"✅ Saved to {output_file}")

def test_interpolator_comparison():
    """Compare different interpolators for multivariate derivatives."""
    print("Creating interpolator comparison...")
    
    # Test function: f(x,y) = sin(x) * cos(y)
    x = np.linspace(0, 2*np.pi, 20)
    y = np.linspace(0, 2*np.pi, 20)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(X) * np.cos(Y)
    
    input_data = np.column_stack([X.flatten(), Y.flatten()])
    output_data = Z.flatten()
    
    # Evaluation grid
    x_eval = np.linspace(0, 2*np.pi, 15)
    y_eval = np.linspace(0, 2*np.pi, 15)
    X_eval, Y_eval = np.meshgrid(x_eval, y_eval)
    eval_points = np.column_stack([X_eval.flatten(), Y_eval.flatten()])
    
    # Analytical solutions
    analytical_grad_x = np.cos(X_eval) * np.cos(Y_eval)
    analytical_laplacian = -2 * np.sin(X_eval) * np.cos(Y_eval)
    
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            'Original f(x,y) = sin(x)cos(y)<br><sub>Test function with known analytical derivatives</sub>',
            'Analytical ∂f/∂x = cos(x)cos(y)<br><sub>True gradient component for comparison</sub>',
            'Analytical ∇²f = -2sin(x)cos(y)<br><sub>True Laplacian for reference</sub>',
            'Spline Interpolator ∂f/∂x<br><sub>Computed using spline-based derivatives</sub>',
            'LLA Interpolator ∂f/∂x<br><sub>Computed using local linear approximation</sub>',
            'Error Distribution<br><sub>Histogram of absolute errors vs analytical solution</sub>'
        ),
        specs=[[{'type': 'heatmap'} for _ in range(3)] for _ in range(2)],
        vertical_spacing=0.12
    )
    
    # Original and analytical
    fig.add_trace(go.Heatmap(x=x_eval, y=y_eval, z=np.sin(X_eval) * np.cos(Y_eval), 
                           colorscale='Viridis'), row=1, col=1)
    fig.add_trace(go.Heatmap(x=x_eval, y=y_eval, z=analytical_grad_x, 
                           colorscale='RdBu'), row=1, col=2)
    fig.add_trace(go.Heatmap(x=x_eval, y=y_eval, z=analytical_laplacian, 
                           colorscale='RdBu'), row=1, col=3)
    
    # Compare interpolators
    interpolators = [("Spline", SplineInterpolator, {"smoothing": 0.1}),
                    ("LLA", LlaInterpolator, {"window_size": 8})]
    errors = []
    
    for idx, (name, interp_class, kwargs) in enumerate(interpolators):
        mv = MultivariateDerivatives(interp_class, **kwargs)
        mv.fit(input_data, output_data)
        
        gradient_func = mv.gradient()
        gradients = gradient_func(eval_points)
        grad_x = gradients[:, 0].reshape(X_eval.shape)
        
        error = np.abs(grad_x - analytical_grad_x)
        errors.append(error.flatten())
        
        fig.add_trace(go.Heatmap(x=x_eval, y=y_eval, z=grad_x, colorscale='RdBu'), 
                     row=2, col=idx+1)
    
    # Error comparison
    fig.add_trace(go.Histogram(x=errors[0], name='Spline Error', opacity=0.7, nbinsx=20), row=2, col=3)
    fig.add_trace(go.Histogram(x=errors[1], name='LLA Error', opacity=0.7, nbinsx=20), row=2, col=3)
    
    # Add comprehensive title and annotations
    fig.update_layout(
        title={
            'text': "Interpolator Comparison: f(x,y) = sin(x)cos(y)<br><sub>" +
                   "INTERPRETATION GUIDE: Top row shows original function and analytical derivatives. " +
                   "Bottom row compares numerical methods against analytical solutions. " +
                   "Error histogram shows accuracy distribution.</sub>",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16}
        },
        height=800,
        annotations=[
            dict(text="Ground truth for validation", x=0.5, y=0.95, xref="paper", yref="paper",
                 showarrow=False, font=dict(size=12, color="blue")),
            dict(text="Numerical approximations", x=0.33, y=0.45, xref="paper", yref="paper",
                 showarrow=False, font=dict(size=12, color="red")),
            dict(text="Lower errors = better accuracy", x=0.83, y=0.45, xref="paper", yref="paper",
                 showarrow=False, font=dict(size=10, color="green"))
        ]
    )
    
    output_file = os.path.join(OUTPUT_DIR, 'multivariate_interpolator_comparison.html')
    fig.write_html(output_file)
    print(f"✅ Saved to {output_file}")
