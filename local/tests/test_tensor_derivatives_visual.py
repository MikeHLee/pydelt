#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visual tests for tensor derivatives functionality.

This module provides visual tests for the tensor derivatives functionality,
including directional derivatives, divergence, curl, strain tensor, and stress tensor.
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

from src.pydelt.tensor_derivatives import TensorDerivatives
from src.pydelt.interpolation import SplineInterpolator, LlaInterpolator

# Output directory
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def test_directional_derivatives():
    """Visualize directional derivatives of a scalar function."""
    print("Creating directional derivatives visualization...")
    
    # Create test data: f(x,y) = x^2 + y^2 - 2*x*y
    x = np.linspace(-3, 3, 30)
    y = np.linspace(-3, 3, 30)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + Y**2 - 2*X*Y
    
    # Fit tensor derivatives
    input_data = np.column_stack([X.flatten(), Y.flatten()])
    output_data = Z.flatten()
    
    td = TensorDerivatives(SplineInterpolator, smoothing=0.1)
    td.fit(input_data, output_data)
    
    # Evaluation grid
    x_eval = np.linspace(-3, 3, 20)
    y_eval = np.linspace(-3, 3, 20)
    X_eval, Y_eval = np.meshgrid(x_eval, y_eval)
    eval_points = np.column_stack([X_eval.flatten(), Y_eval.flatten()])
    
    # Compute directional derivatives in different directions
    directions = [
        ([1, 0], "X Direction (1,0)"),
        ([0, 1], "Y Direction (0,1)"),
        ([1, 1], "Diagonal Direction (1,1)"),
        ([1, -1], "Anti-diagonal Direction (1,-1)")
    ]
    
    # Create visualization
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            'Original Function f(x,y) = x² + y² - 2xy<br><sub>Shows the scalar field surface</sub>', 
            'Gradient Vector Field ∇f<br><sub>Shows direction and magnitude of steepest ascent</sub>',
            'Directional Derivative: X Direction<br><sub>Rate of change along x-axis</sub>',
            'Directional Derivative: Y Direction<br><sub>Rate of change along y-axis</sub>',
            'Directional Derivative: Diagonal (1,1)<br><sub>Rate of change along diagonal</sub>',
            'Directional Derivative: Anti-diagonal (1,-1)<br><sub>Rate of change along anti-diagonal</sub>'
        ),
        specs=[[{'type': 'surface'}, {'type': 'scatter'}, {'type': 'contour'}],
               [{'type': 'contour'}, {'type': 'contour'}, {'type': 'contour'}]],
        vertical_spacing=0.12,
        horizontal_spacing=0.05
    )
    
    # Original function surface
    fig.add_trace(go.Surface(x=X, y=Y, z=Z, colorscale='Viridis', showscale=False), row=1, col=1)
    
    # Gradient vector field
    gradient_func = td.mv.gradient()
    gradients = gradient_func(eval_points)
    grad_x = gradients[:, 0].reshape(X_eval.shape)
    grad_y = gradients[:, 1].reshape(X_eval.shape)
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    fig.add_trace(go.Scatter(x=X_eval.flatten(), y=Y_eval.flatten(), mode='markers',
                           marker=dict(size=5, color=grad_magnitude.flatten(), 
                                     colorscale='Plasma', showscale=True,
                                     colorbar=dict(title="Gradient<br>Magnitude"))), 
                 row=1, col=2)
    
    # Add gradient vectors (subsampled)
    skip = 3
    for i in range(0, len(eval_points), skip*skip):
        x_pos, y_pos = eval_points[i]
        dx, dy = gradients[i] * 0.15
        fig.add_trace(go.Scatter(x=[x_pos, x_pos + dx], y=[y_pos, y_pos + dy],
                               mode='lines', line=dict(color='red', width=1.5),
                               showlegend=False), row=1, col=2)
    
    # Directional derivatives
    for i, (direction, title) in enumerate(directions):
        dir_deriv_func = td.directional_derivative(direction)
        dir_derivs = dir_deriv_func(eval_points).reshape(X_eval.shape)
        
        row, col = (1, 3) if i == 0 else (2, i) if i < 3 else (2, 3)
        
        fig.add_trace(go.Contour(
            x=x_eval, y=y_eval, z=dir_derivs,
            colorscale='RdBu', showscale=(i == 0),
            colorbar=dict(title="Derivative<br>Value") if i == 0 else None,
            contours=dict(start=-6, end=6, size=0.5)
        ), row=row, col=col)
        
        # Add direction vector visualization
        arrow_len = 0.8
        dx, dy = np.array(direction) * arrow_len / np.linalg.norm(direction)
        fig.add_trace(go.Scatter(
            x=[0, dx], y=[0, dy],
            mode='lines+markers',
            line=dict(color='black', width=3),
            marker=dict(size=[0, 8], symbol=['circle', 'arrow-bar-up']),
            showlegend=False
        ), row=row, col=col)
    
    # Add comprehensive title and annotations
    fig.update_layout(
        title={
            'text': "Directional Derivatives Analysis<br><sub>" +
                   "INTERPRETATION GUIDE: Directional derivatives show rate of change along specific directions. " +
                   "Red areas indicate positive change (increasing), blue areas indicate negative change (decreasing). " +
                   "The arrows show the direction vector along which the derivative is computed.</sub>",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16}
        },
        height=900,
        width=1200,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='f(X,Y)',
            camera=dict(eye=dict(x=1.5, y=-1.5, z=1))
        ),
        annotations=[
            dict(text="Original scalar function", x=0.16, y=0.95, xref="paper", yref="paper",
                 showarrow=False, font=dict(size=12, color="black")),
            dict(text="Gradient field shows steepest ascent", x=0.5, y=0.95, xref="paper", yref="paper",
                 showarrow=False, font=dict(size=12, color="black")),
            dict(text="Compare derivatives along different directions", x=0.5, y=0.45, xref="paper", yref="paper",
                 showarrow=False, font=dict(size=12, color="black"))
        ]
    )
    
    output_file = os.path.join(OUTPUT_DIR, 'tensor_directional_derivatives.html')
    fig.write_html(output_file)
    print("✅ Saved to {}".format(output_file))


def test_vector_field_divergence():
    """Visualize divergence of a vector field."""
    print("Creating vector field divergence visualization...")
    
    # Create a grid for evaluation
    x = np.linspace(-2, 2, 20)
    y = np.linspace(-2, 2, 20)
    X, Y = np.meshgrid(x, y)
    
    # Create input data points
    input_data = np.column_stack([X.flatten(), Y.flatten()])
    
    # Create vector field: [x^2 - y, x + y^2]
    # This field has both sources and sinks
    vector_field_x = (X**2 - Y).flatten()
    vector_field_y = (X + Y**2).flatten()
    vector_output = np.column_stack([vector_field_x, vector_field_y])
    
    # Analytical divergence: div(F) = ∂F₁/∂x + ∂F₂/∂y = 2x + 2y
    analytical_div = 2*X + 2*Y
    
    # Fit tensor derivatives
    td = TensorDerivatives(SplineInterpolator, smoothing=0.1)
    td.fit(input_data, vector_output)
    
    # Compute divergence
    divergence_func = td.divergence()
    divergence = divergence_func(input_data).reshape(X.shape)
    
    # Create visualization
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Vector Field F = [x² - y, x + y²]<br><sub>Arrows show direction and magnitude</sub>',
            'Analytical Divergence: div(F) = 2x + 2y<br><sub>Red = sources (expansion), Blue = sinks (contraction)</sub>',
            'Numerical Divergence<br><sub>Computed using TensorDerivatives</sub>',
            'Error: |Analytical - Numerical|<br><sub>Darker areas indicate higher error</sub>'
        ),
        specs=[[{'type': 'scatter'}, {'type': 'contour'}],
               [{'type': 'contour'}, {'type': 'contour'}]],
        vertical_spacing=0.12
    )
    
    # Vector field plot
    skip = 2  # Skip some points for clarity
    fig.add_trace(go.Scatter(
        x=X[::skip, ::skip].flatten(),
        y=Y[::skip, ::skip].flatten(),
        mode='markers+text',
        marker=dict(
            symbol='arrow',
            size=10,
            angle=np.arctan2(
                vector_field_y.reshape(X.shape)[::skip, ::skip],
                vector_field_x.reshape(X.shape)[::skip, ::skip]
            ).flatten() * 180/np.pi,
            color=np.sqrt(
                vector_field_x.reshape(X.shape)[::skip, ::skip]**2 +
                vector_field_y.reshape(X.shape)[::skip, ::skip]**2
            ).flatten(),
            colorscale='Viridis',
            colorbar=dict(title="Magnitude", x=-0.15)
        ),
        text='→',
        textposition='middle center',
        name='Vector Field'
    ), row=1, col=1)
    
    # Analytical divergence
    fig.add_trace(go.Contour(
        x=x, y=y, z=analytical_div,
        colorscale='RdBu_r',
        contours=dict(start=-8, end=8, size=0.5),
        colorbar=dict(title="Divergence", x=0.45)
    ), row=1, col=2)
    
    # Numerical divergence
    fig.add_trace(go.Contour(
        x=x, y=y, z=divergence,
        colorscale='RdBu_r',
        contours=dict(start=-8, end=8, size=0.5),
        colorbar=dict(title="Divergence", x=1.05)
    ), row=2, col=1)
    
    # Error plot
    error = np.abs(analytical_div - divergence)
    fig.add_trace(go.Contour(
        x=x, y=y, z=error,
        colorscale='Viridis',
        contours=dict(start=0, end=2, size=0.1),
        colorbar=dict(title="Error")
    ), row=2, col=2)
    
    # Add zero contour lines to divergence plots
    for row, col in [(1, 2), (2, 1)]:
        fig.add_trace(go.Contour(
            x=x, y=y, z=analytical_div if row == 1 else divergence,
            contours=dict(start=0, end=0, coloring='lines'),
            line=dict(width=2, color='black'),
            showscale=False,
            name='Zero Divergence'
        ), row=row, col=col)
    
    # Add comprehensive title and annotations
    fig.update_layout(
        title={
            'text': "Vector Field Divergence Analysis<br><sub>" +
                   "INTERPRETATION GUIDE: Divergence measures the 'outgoingness' of a vector field. " +
                   "Positive values (red) indicate sources/expansion, negative values (blue) indicate sinks/contraction. " +
                   "Zero divergence (black contour) indicates incompressible flow.</sub>",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16}
        },
        height=800,
        width=1000,
        annotations=[
            dict(text="Vector field visualization", x=0.2, y=0.95, xref="paper", yref="paper",
                 showarrow=False, font=dict(size=12, color="black")),
            dict(text="Ground truth for validation", x=0.8, y=0.95, xref="paper", yref="paper",
                 showarrow=False, font=dict(size=12, color="black")),
            dict(text="Numerical approximation", x=0.2, y=0.45, xref="paper", yref="paper",
                 showarrow=False, font=dict(size=12, color="black")),
            dict(text="Error distribution", x=0.8, y=0.45, xref="paper", yref="paper",
                 showarrow=False, font=dict(size=12, color="black"))
        ]
    )
    
    # Update axes
    for i in range(1, 3):
        for j in range(1, 3):
            fig.update_xaxes(title="x", row=i, col=j)
            fig.update_yaxes(title="y", row=i, col=j)
    
    output_file = os.path.join(OUTPUT_DIR, 'tensor_vector_field_divergence.html')
    fig.write_html(output_file)
    print("✅ Saved to {}".format(output_file))


def test_vector_field_curl():
    """Visualize curl of a 3D vector field."""
    print("Creating vector field curl visualization...")
    
    # Create a 3D grid for evaluation
    x = np.linspace(-2, 2, 10)
    y = np.linspace(-2, 2, 10)
    z = np.linspace(-2, 2, 10)
    X, Y, Z = np.meshgrid(x, y, z)
    
    # Create input data points
    input_data = np.column_stack([X.flatten(), Y.flatten(), Z.flatten()])
    
    # Create vector field: [y, -x, z]
    # This field has a rotational component in the xy-plane
    vector_field_x = Y.flatten()
    vector_field_y = -X.flatten()
    vector_field_z = Z.flatten()
    vector_output = np.column_stack([vector_field_x, vector_field_y, vector_field_z])
    
    # Analytical curl: curl(F) = [∂F₃/∂y - ∂F₂/∂z, ∂F₁/∂z - ∂F₃/∂x, ∂F₂/∂x - ∂F₁/∂y]
    # For this field: curl(F) = [0, 0, -2]
    analytical_curl_x = np.zeros_like(X.flatten())
    analytical_curl_y = np.zeros_like(Y.flatten())
    analytical_curl_z = -2 * np.ones_like(Z.flatten())
    analytical_curl = np.column_stack([analytical_curl_x, analytical_curl_y, analytical_curl_z])
    
    # Fit tensor derivatives
    td = TensorDerivatives(SplineInterpolator, smoothing=0.1)
    td.fit(input_data, vector_output)
    
    # Compute curl
    curl_func = td.curl()
    curl = curl_func(input_data)
    
    # Reshape for visualization
    curl_x = curl[:, 0].reshape(X.shape)
    curl_y = curl[:, 1].reshape(Y.shape)
    curl_z = curl[:, 2].reshape(Z.shape)
    
    # Calculate error
    error_x = np.abs(curl_x - analytical_curl_x.reshape(X.shape))
    error_y = np.abs(curl_y - analytical_curl_y.reshape(Y.shape))
    error_z = np.abs(curl_z - analytical_curl_z.reshape(Z.shape))
    error_magnitude = np.sqrt(error_x**2 + error_y**2 + error_z**2)
    
    # Extract a slice for 2D visualization (z=0 plane)
    z_slice_idx = len(z) // 2
    X_slice = X[:, :, z_slice_idx]
    Y_slice = Y[:, :, z_slice_idx]
    
    # Vector field on slice
    vf_x_slice = vector_field_x.reshape(X.shape)[:, :, z_slice_idx]
    vf_y_slice = vector_field_y.reshape(Y.shape)[:, :, z_slice_idx]
    vf_z_slice = vector_field_z.reshape(Z.shape)[:, :, z_slice_idx]
    
    # Curl on slice
    curl_x_slice = curl_x[:, :, z_slice_idx]
    curl_y_slice = curl_y[:, :, z_slice_idx]
    curl_z_slice = curl_z[:, :, z_slice_idx]
    
    # Error on slice
    error_z_slice = error_z[:, :, z_slice_idx]
    
    # Create visualization
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Vector Field F = [y, -x, z] at z=0<br><sub>Arrows show direction and magnitude</sub>',
            'Analytical Curl: curl(F) = [0, 0, -2]<br><sub>Arrows show direction and magnitude</sub>',
            'Numerical Curl (z-component)<br><sub>Computed using TensorDerivatives</sub>',
            'Error: |Analytical - Numerical|<br><sub>Darker areas indicate higher error</sub>'
        ),
        specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
               [{'type': 'contour'}, {'type': 'contour'}]],
        vertical_spacing=0.12
    )
    
    # Vector field plot
    skip = 1  # Skip some points for clarity
    fig.add_trace(go.Scatter(
        x=X_slice[::skip, ::skip].flatten(),
        y=Y_slice[::skip, ::skip].flatten(),
        mode='markers',
        marker=dict(
            symbol='arrow',
            size=12,
            angle=np.arctan2(
                vf_y_slice[::skip, ::skip],
                vf_x_slice[::skip, ::skip]
            ).flatten() * 180/np.pi,
            color=np.sqrt(
                vf_x_slice[::skip, ::skip]**2 +
                vf_y_slice[::skip, ::skip]**2
            ).flatten(),
            colorscale='Viridis',
            colorbar=dict(title="Magnitude", x=-0.15)
        ),
        name='Vector Field'
    ), row=1, col=1)
    
    # Analytical curl plot (only z component is non-zero)
    fig.add_trace(go.Scatter(
        x=X_slice[::skip, ::skip].flatten(),
        y=Y_slice[::skip, ::skip].flatten(),
        mode='markers',
        marker=dict(
            symbol='circle',
            size=10,
            color=analytical_curl_z.reshape(X.shape)[:, :, z_slice_idx][::skip, ::skip].flatten(),
            colorscale='RdBu_r',
            colorbar=dict(title="Curl Z", x=0.45)
        ),
        name='Analytical Curl'
    ), row=1, col=2)
    
    # Numerical curl z-component
    fig.add_trace(go.Contour(
        x=x, y=y, z=curl_z_slice,
        colorscale='RdBu_r',
        contours=dict(start=-3, end=0, size=0.2),
        colorbar=dict(title="Curl Z", x=1.05)
    ), row=2, col=1)
    
    # Error plot
    fig.add_trace(go.Contour(
        x=x, y=y, z=error_z_slice,
        colorscale='Viridis',
        contours=dict(start=0, end=0.5, size=0.05),
        colorbar=dict(title="Error")
    ), row=2, col=2)
    
    # Add comprehensive title and annotations
    fig.update_layout(
        title={
            'text': "Vector Field Curl Analysis<br><sub>" +
                   "INTERPRETATION GUIDE: Curl measures the rotational tendency of a vector field. " +
                   "The magnitude indicates rotation strength, and the direction (via right-hand rule) " +
                   "shows the axis of rotation. This example shows a vector field with constant curl in z-direction.</sub>",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16}
        },
        height=800,
        width=1000,
        annotations=[
            dict(text="Original vector field", x=0.2, y=0.95, xref="paper", yref="paper",
                 showarrow=False, font=dict(size=12, color="black")),
            dict(text="Analytical curl (z-component)", x=0.8, y=0.95, xref="paper", yref="paper",
                 showarrow=False, font=dict(size=12, color="black")),
            dict(text="Numerical curl approximation", x=0.2, y=0.45, xref="paper", yref="paper",
                 showarrow=False, font=dict(size=12, color="black")),
            dict(text="Error distribution", x=0.8, y=0.45, xref="paper", yref="paper",
                 showarrow=False, font=dict(size=12, color="black"))
        ]
    )
    
    # Update axes
    for i in range(1, 3):
        for j in range(1, 3):
            fig.update_xaxes(title="x", row=i, col=j)
            fig.update_yaxes(title="y", row=i, col=j)
    
    # Create 3D visualization of vector field and curl
    fig3d = go.Figure()
    
    # Subsample for 3D visualization
    subsample = 2
    X_sub = X[::subsample, ::subsample, ::subsample]
    Y_sub = Y[::subsample, ::subsample, ::subsample]
    Z_sub = Z[::subsample, ::subsample, ::subsample]
    
    # Vector field components
    u = vector_field_x.reshape(X.shape)[::subsample, ::subsample, ::subsample]
    v = vector_field_y.reshape(Y.shape)[::subsample, ::subsample, ::subsample]
    w = vector_field_z.reshape(Z.shape)[::subsample, ::subsample, ::subsample]
    
    # Curl components
    curl_x_sub = curl_x[::subsample, ::subsample, ::subsample]
    curl_y_sub = curl_y[::subsample, ::subsample, ::subsample]
    curl_z_sub = curl_z[::subsample, ::subsample, ::subsample]
    
    # Add vector field as 3D cones
    fig3d.add_trace(go.Cone(
        x=X_sub.flatten(),
        y=Y_sub.flatten(),
        z=Z_sub.flatten(),
        u=u.flatten(),
        v=v.flatten(),
        w=w.flatten(),
        colorscale='Blues',
        showscale=False,
        name='Vector Field',
        sizemode='absolute',
        sizeref=0.3
    ))
    
    # Add curl vectors as red cones
    fig3d.add_trace(go.Cone(
        x=X_sub.flatten(),
        y=Y_sub.flatten(),
        z=Z_sub.flatten(),
        u=curl_x_sub.flatten(),
        v=curl_y_sub.flatten(),
        w=curl_z_sub.flatten(),
        colorscale='Reds',
        showscale=True,
        colorbar=dict(title="Curl<br>Magnitude"),
        name='Curl',
        sizemode='absolute',
        sizeref=0.15,
        opacity=0.7
    ))
    
    # Update layout
    fig3d.update_layout(
        title={
            'text': "3D Vector Field and Curl Visualization<br><sub>" +
                   "Blue cones: Original vector field F = [y, -x, z]<br>" +
                   "Red cones: Curl of the vector field (should point in -z direction)</sub>",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16}
        },
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='cube'
        ),
        height=800,
        width=1000
    )
    
    # Save both visualizations
    output_file = os.path.join(OUTPUT_DIR, 'tensor_vector_field_curl.html')
    fig.write_html(output_file)
    print("✅ Saved to {}".format(output_file))
    
    output_file_3d = os.path.join(OUTPUT_DIR, 'tensor_vector_field_curl_3d.html')
    fig3d.write_html(output_file_3d)
    print("✅ Saved 3D visualization to {}".format(output_file_3d))


def test_strain_tensor():
    """Visualize strain tensor computed from displacement fields."""
    print("Creating strain tensor visualization...")
    
    # Create a grid for evaluation
    x = np.linspace(-2, 2, 20)
    y = np.linspace(-2, 2, 20)
    X, Y = np.meshgrid(x, y)
    
    # Create input data points
    input_data = np.column_stack([X.flatten(), Y.flatten()])
    
    # Create displacement field: u = [x*y, x^2 - y^2]
    # This field has both normal and shear strain components
    displacement_x = (X * Y).flatten()
    displacement_y = (X**2 - Y**2).flatten()
    displacement = np.column_stack([displacement_x, displacement_y])
    
    # Analytical strain tensor components
    # ε_xx = ∂u_x/∂x = y
    # ε_yy = ∂u_y/∂y = -2y
    # ε_xy = 0.5*(∂u_x/∂y + ∂u_y/∂x) = 0.5*(x + 2x) = 1.5x
    analytical_strain_xx = Y.flatten()
    analytical_strain_yy = (-2 * Y).flatten()
    analytical_strain_xy = (1.5 * X).flatten()
    
    # Fit tensor derivatives
    td = TensorDerivatives(SplineInterpolator, smoothing=0.1)
    td.fit(input_data, displacement)
    
    # Compute strain tensor
    strain_func = td.strain_tensor()
    strain = strain_func(input_data)
    
    # Extract strain components
    strain_xx = strain[:, 0].reshape(X.shape)
    strain_yy = strain[:, 1].reshape(Y.shape)
    strain_xy = strain[:, 2].reshape(X.shape)
    
    # Calculate error
    error_xx = np.abs(strain_xx - analytical_strain_xx.reshape(X.shape))
    error_yy = np.abs(strain_yy - analytical_strain_yy.reshape(Y.shape))
    error_xy = np.abs(strain_xy - analytical_strain_xy.reshape(X.shape))
    
    # Calculate volumetric strain (trace of strain tensor)
    volumetric_strain_analytical = analytical_strain_xx.reshape(X.shape) + analytical_strain_yy.reshape(Y.shape)
    volumetric_strain_numerical = strain_xx + strain_yy
    
    # Create visualization
    fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=(
            'Displacement Field<br><sub>u = [xy, x² - y²]</sub>',
            'Normal Strain ε_xx (Analytical)<br><sub>∂u_x/∂x = y</sub>',
            'Normal Strain ε_xx (Numerical)<br><sub>Computed using TensorDerivatives</sub>',
            'Normal Strain ε_yy (Analytical)<br><sub>∂u_y/∂y = -2y</sub>',
            'Normal Strain ε_yy (Numerical)<br><sub>Computed using TensorDerivatives</sub>',
            'Error in ε_yy<br><sub>|Analytical - Numerical|</sub>',
            'Shear Strain ε_xy (Analytical)<br><sub>0.5*(∂u_x/∂y + ∂u_y/∂x) = 1.5x</sub>',
            'Shear Strain ε_xy (Numerical)<br><sub>Computed using TensorDerivatives</sub>',
            'Volumetric Strain<br><sub>ε_xx + ε_yy</sub>'
        ),
        specs=[
            [{'type': 'scatter'}, {'type': 'contour'}, {'type': 'contour'}],
            [{'type': 'contour'}, {'type': 'contour'}, {'type': 'contour'}],
            [{'type': 'contour'}, {'type': 'contour'}, {'type': 'contour'}]
        ],
        vertical_spacing=0.08,
        horizontal_spacing=0.05
    )
    
    # Displacement field plot
    skip = 2  # Skip some points for clarity
    fig.add_trace(go.Scatter(
        x=X[::skip, ::skip].flatten(),
        y=Y[::skip, ::skip].flatten(),
        mode='markers+text',
        marker=dict(
            symbol='circle',
            size=8,
            color=np.sqrt(
                displacement_x.reshape(X.shape)[::skip, ::skip]**2 +
                displacement_y.reshape(Y.shape)[::skip, ::skip]**2
            ).flatten(),
            colorscale='Viridis',
            colorbar=dict(title="|u|", x=-0.15)
        ),
        text='',
        name='Grid Points'
    ), row=1, col=1)
    
    # Add displacement vectors
    for i in range(0, len(X[::skip, ::skip].flatten())):
        x_pos = X[::skip, ::skip].flatten()[i]
        y_pos = Y[::skip, ::skip].flatten()[i]
        dx = displacement_x.reshape(X.shape)[::skip, ::skip].flatten()[i] * 0.2
        dy = displacement_y.reshape(Y.shape)[::skip, ::skip].flatten()[i] * 0.2
        
        fig.add_trace(go.Scatter(
            x=[x_pos, x_pos + dx],
            y=[y_pos, y_pos + dy],
            mode='lines+markers',
            line=dict(color='red', width=1),
            marker=dict(size=[0, 5], symbol=['circle', 'arrow-bar-up']),
            showlegend=False
        ), row=1, col=1)
    
    # Normal strain xx (analytical)
    fig.add_trace(go.Contour(
        x=x, y=y, z=analytical_strain_xx.reshape(X.shape),
        colorscale='RdBu_r',
        contours=dict(start=-2, end=2, size=0.2),
        colorbar=dict(title="ε_xx", x=0.45)
    ), row=1, col=2)
    
    # Normal strain xx (numerical)
    fig.add_trace(go.Contour(
        x=x, y=y, z=strain_xx,
        colorscale='RdBu_r',
        contours=dict(start=-2, end=2, size=0.2),
        colorbar=dict(title="ε_xx", x=1.05)
    ), row=1, col=3)
    
    # Normal strain yy (analytical)
    fig.add_trace(go.Contour(
        x=x, y=y, z=analytical_strain_yy.reshape(Y.shape),
        colorscale='RdBu_r',
        contours=dict(start=-4, end=4, size=0.5),
        colorbar=dict(title="ε_yy", x=-0.15)
    ), row=2, col=1)
    
    # Normal strain yy (numerical)
    fig.add_trace(go.Contour(
        x=x, y=y, z=strain_yy,
        colorscale='RdBu_r',
        contours=dict(start=-4, end=4, size=0.5),
        colorbar=dict(title="ε_yy", x=0.45)
    ), row=2, col=2)
    
    # Error in strain yy
    fig.add_trace(go.Contour(
        x=x, y=y, z=error_yy,
        colorscale='Viridis',
        contours=dict(start=0, end=0.5, size=0.05),
        colorbar=dict(title="Error", x=1.05)
    ), row=2, col=3)
    
    # Shear strain xy (analytical)
    fig.add_trace(go.Contour(
        x=x, y=y, z=analytical_strain_xy.reshape(X.shape),
        colorscale='RdBu_r',
        contours=dict(start=-3, end=3, size=0.3),
        colorbar=dict(title="ε_xy", x=-0.15)
    ), row=3, col=1)
    
    # Shear strain xy (numerical)
    fig.add_trace(go.Contour(
        x=x, y=y, z=strain_xy,
        colorscale='RdBu_r',
        contours=dict(start=-3, end=3, size=0.3),
        colorbar=dict(title="ε_xy", x=0.45)
    ), row=3, col=2)
    
    # Volumetric strain
    fig.add_trace(go.Contour(
        x=x, y=y, z=volumetric_strain_numerical,
        colorscale='RdBu_r',
        contours=dict(start=-4, end=4, size=0.5),
        colorbar=dict(title="ε_v")
    ), row=3, col=3)
    
    # Add zero contour lines to strain plots
    for row, col, z_data in [
        (1, 2, analytical_strain_xx.reshape(X.shape)),
        (1, 3, strain_xx),
        (2, 1, analytical_strain_yy.reshape(Y.shape)),
        (2, 2, strain_yy),
        (3, 1, analytical_strain_xy.reshape(X.shape)),
        (3, 2, strain_xy),
        (3, 3, volumetric_strain_numerical)
    ]:
        fig.add_trace(go.Contour(
            x=x, y=y, z=z_data,
            contours=dict(start=0, end=0, coloring='lines'),
            line=dict(width=2, color='black'),
            showscale=False,
            name='Zero Strain'
        ), row=row, col=col)
    
    # Add comprehensive title and annotations
    fig.update_layout(
        title={
            'text': "Strain Tensor Analysis<br><sub>" +
                   "INTERPRETATION GUIDE: Strain tensor describes deformation of a material. " +
                   "Normal components (ε_xx, ε_yy) show stretching/compression along axes. " +
                   "Shear component (ε_xy) shows angular distortion. " +
                   "Red indicates positive strain (stretching), blue indicates negative strain (compression).</sub>",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16}
        },
        height=1000,
        width=1200,
        annotations=[
            dict(text="Original displacement field", x=0.16, y=0.95, xref="paper", yref="paper",
                 showarrow=False, font=dict(size=12, color="black")),
            dict(text="Compare analytical vs. numerical results", x=0.5, y=0.95, xref="paper", yref="paper",
                 showarrow=False, font=dict(size=12, color="black")),
            dict(text="Volumetric strain = trace of strain tensor", x=0.84, y=0.35, xref="paper", yref="paper",
                 showarrow=False, font=dict(size=12, color="black"))
        ]
    )
    
    # Update axes
    for i in range(1, 4):
        for j in range(1, 4):
            fig.update_xaxes(title="x", row=i, col=j)
            fig.update_yaxes(title="y", row=i, col=j)
    
    output_file = os.path.join(OUTPUT_DIR, 'tensor_strain_tensor.html')
    fig.write_html(output_file)
    print("✅ Saved to {}".format(output_file))


def test_stress_tensor():
    """Visualize stress tensor computed from displacement fields."""
    print("Creating stress tensor visualization...")
    
    # Create a grid for evaluation
    x = np.linspace(-2, 2, 20)
    y = np.linspace(-2, 2, 20)
    X, Y = np.meshgrid(x, y)
    
    # Create input data points
    input_data = np.column_stack([X.flatten(), Y.flatten()])
    
    # Create displacement field: u = [x*y, x^2 - y^2]
    # This field has both normal and shear strain components
    displacement_x = (X * Y).flatten()
    displacement_y = (X**2 - Y**2).flatten()
    displacement = np.column_stack([displacement_x, displacement_y])
    
    # Lamé parameters (for linear elastic material)
    lambda_param = 1.0  # First Lamé parameter
    mu_param = 0.5      # Second Lamé parameter (shear modulus)
    
    # Analytical strain tensor components
    # ε_xx = ∂u_x/∂x = y
    # ε_yy = ∂u_y/∂y = -2y
    # ε_xy = 0.5*(∂u_x/∂y + ∂u_y/∂x) = 0.5*(x + 2x) = 1.5x
    analytical_strain_xx = Y.flatten()
    analytical_strain_yy = (-2 * Y).flatten()
    analytical_strain_xy = (1.5 * X).flatten()
    
    # Analytical stress tensor components (using linear elasticity)
    # σ_xx = λ(ε_xx + ε_yy) + 2με_xx = λ(y - 2y) + 2μ*y = λ(-y) + 2μ*y = (-λ + 2μ)y
    # σ_yy = λ(ε_xx + ε_yy) + 2με_yy = λ(y - 2y) + 2μ*(-2y) = λ(-y) - 4μ*y = (-λ - 4μ)y
    # σ_xy = 2με_xy = 2μ*1.5x = 3μ*x
    analytical_stress_xx = ((-lambda_param + 2*mu_param) * Y).flatten()
    analytical_stress_yy = ((-lambda_param - 4*mu_param) * Y).flatten()
    analytical_stress_xy = ((3*mu_param) * X).flatten()
    
    # Von Mises stress (analytical)
    analytical_von_mises = np.sqrt(
        analytical_stress_xx**2 + analytical_stress_yy**2 - 
        analytical_stress_xx*analytical_stress_yy + 3*analytical_stress_xy**2
    )
    
    # Fit tensor derivatives
    td = TensorDerivatives(SplineInterpolator, smoothing=0.1)
    td.fit(input_data, displacement)
    
    # Compute stress tensor
    stress_func = td.stress_tensor(lambda_param=lambda_param, mu_param=mu_param)
    stress = stress_func(input_data)
    
    # Extract stress components
    stress_xx = stress[:, 0].reshape(X.shape)
    stress_yy = stress[:, 1].reshape(Y.shape)
    stress_xy = stress[:, 2].reshape(X.shape)
    
    # Calculate error
    error_xx = np.abs(stress_xx - analytical_stress_xx.reshape(X.shape))
    error_yy = np.abs(stress_yy - analytical_stress_yy.reshape(Y.shape))
    error_xy = np.abs(stress_xy - analytical_stress_xy.reshape(X.shape))
    
    # Calculate von Mises stress (numerical)
    von_mises = np.sqrt(
        stress_xx**2 + stress_yy**2 - stress_xx*stress_yy + 3*stress_xy**2
    )
    
    # Create visualization
    fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=(
            'Normal Stress σ_xx<br><sub>(-λ + 2μ)y</sub>',
            'Normal Stress σ_yy<br><sub>(-λ - 4μ)y</sub>',
            'Shear Stress σ_xy<br><sub>3μ*x</sub>',
            'Von Mises Stress (Analytical)<br><sub>Equivalent stress measure</sub>',
            'Von Mises Stress (Numerical)<br><sub>Computed using TensorDerivatives</sub>',
            'Error in Von Mises<br><sub>|Analytical - Numerical|</sub>',
            'Principal Stress σ₁<br><sub>Maximum normal stress</sub>',
            'Principal Stress σ₂<br><sub>Minimum normal stress</sub>',
            'Maximum Shear Stress<br><sub>τ_max = (σ₁ - σ₂)/2</sub>'
        ),
        specs=[
            [{'type': 'contour'}, {'type': 'contour'}, {'type': 'contour'}],
            [{'type': 'contour'}, {'type': 'contour'}, {'type': 'contour'}],
            [{'type': 'contour'}, {'type': 'contour'}, {'type': 'contour'}]
        ],
        vertical_spacing=0.08,
        horizontal_spacing=0.05
    )
    
    # Normal stress xx
    fig.add_trace(go.Contour(
        x=x, y=y, z=stress_xx,
        colorscale='RdBu_r',
        contours=dict(start=-2, end=2, size=0.2),
        colorbar=dict(title="σ_xx", x=-0.15)
    ), row=1, col=1)
    
    # Normal stress yy
    fig.add_trace(go.Contour(
        x=x, y=y, z=stress_yy,
        colorscale='RdBu_r',
        contours=dict(start=-4, end=4, size=0.5),
        colorbar=dict(title="σ_yy", x=0.45)
    ), row=1, col=2)
    
    # Shear stress xy
    fig.add_trace(go.Contour(
        x=x, y=y, z=stress_xy,
        colorscale='RdBu_r',
        contours=dict(start=-3, end=3, size=0.3),
        colorbar=dict(title="σ_xy", x=1.05)
    ), row=1, col=3)
    
    # Von Mises stress (analytical)
    fig.add_trace(go.Contour(
        x=x, y=y, z=analytical_von_mises.reshape(X.shape),
        colorscale='Viridis',
        contours=dict(start=0, end=5, size=0.5),
        colorbar=dict(title="σ_vm", x=-0.15)
    ), row=2, col=1)
    
    # Von Mises stress (numerical)
    fig.add_trace(go.Contour(
        x=x, y=y, z=von_mises,
        colorscale='Viridis',
        contours=dict(start=0, end=5, size=0.5),
        colorbar=dict(title="σ_vm", x=0.45)
    ), row=2, col=2)
    
    # Error in von Mises
    error_vm = np.abs(von_mises - analytical_von_mises.reshape(X.shape))
    fig.add_trace(go.Contour(
        x=x, y=y, z=error_vm,
        colorscale='Viridis',
        contours=dict(start=0, end=0.5, size=0.05),
        colorbar=dict(title="Error", x=1.05)
    ), row=2, col=3)
    
    # Calculate principal stresses
    sigma_1 = 0.5 * (stress_xx + stress_yy + np.sqrt((stress_xx - stress_yy)**2 + 4*stress_xy**2))
    sigma_2 = 0.5 * (stress_xx + stress_yy - np.sqrt((stress_xx - stress_yy)**2 + 4*stress_xy**2))
    tau_max = 0.5 * np.sqrt((stress_xx - stress_yy)**2 + 4*stress_xy**2)
    
    # Principal stress 1
    fig.add_trace(go.Contour(
        x=x, y=y, z=sigma_1,
        colorscale='RdBu_r',
        contours=dict(start=-4, end=4, size=0.5),
        colorbar=dict(title="σ₁", x=-0.15)
    ), row=3, col=1)
    
    # Principal stress 2
    fig.add_trace(go.Contour(
        x=x, y=y, z=sigma_2,
        colorscale='RdBu_r',
        contours=dict(start=-4, end=4, size=0.5),
        colorbar=dict(title="σ₂", x=0.45)
    ), row=3, col=2)
    
    # Maximum shear stress
    fig.add_trace(go.Contour(
        x=x, y=y, z=tau_max,
        colorscale='Viridis',
        contours=dict(start=0, end=3, size=0.3),
        colorbar=dict(title="τ_max")
    ), row=3, col=3)
    
    # Add zero contour lines to stress plots
    for row, col, z_data in [
        (1, 1, stress_xx),
        (1, 2, stress_yy),
        (1, 3, stress_xy),
        (3, 1, sigma_1),
        (3, 2, sigma_2)
    ]:
        fig.add_trace(go.Contour(
            x=x, y=y, z=z_data,
            contours=dict(start=0, end=0, coloring='lines'),
            line=dict(width=2, color='black'),
            showscale=False,
            name='Zero Stress'
        ), row=row, col=col)
    
    # Add comprehensive title and annotations
    fig.update_layout(
        title={
            'text': "Stress Tensor Analysis<br><sub>" +
                   "INTERPRETATION GUIDE: Stress tensor describes internal forces in a material. " +
                   "Normal components (σ_xx, σ_yy) show tension/compression forces. " +
                   "Shear component (σ_xy) shows tangential forces. " +
                   "Von Mises stress is used to predict yielding of materials under complex loading.</sub>",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16}
        },
        height=1000,
        width=1200,
        annotations=[
            dict(text="Stress components", x=0.5, y=0.95, xref="paper", yref="paper",
                 showarrow=False, font=dict(size=12, color="black")),
            dict(text="Von Mises equivalent stress", x=0.5, y=0.65, xref="paper", yref="paper",
                 showarrow=False, font=dict(size=12, color="black")),
            dict(text="Principal stress analysis", x=0.5, y=0.35, xref="paper", yref="paper",
                 showarrow=False, font=dict(size=12, color="black"))
        ]
    )
    
    # Update axes
    for i in range(1, 4):
        for j in range(1, 4):
            fig.update_xaxes(title="x", row=i, col=j)
            fig.update_yaxes(title="y", row=i, col=j)
    
    output_file = os.path.join(OUTPUT_DIR, 'tensor_stress_tensor.html')
    fig.write_html(output_file)
    print("✅ Saved to {}".format(output_file))


if __name__ == "__main__":
    print("Running tensor derivatives visual tests...")
    try:
        test_directional_derivatives()
        test_vector_field_divergence()
        test_vector_field_curl()
        test_strain_tensor()
        test_stress_tensor()
        print("\n✅ All tensor derivative tests completed successfully!")
    except Exception as e:
        print("\n❌ Error running tensor derivative tests: {}".format(e))
        raise
