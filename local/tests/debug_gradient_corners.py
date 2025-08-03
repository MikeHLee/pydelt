#!/usr/bin/env python3
"""
Debug script to compare numerical vs analytical gradients for the corner issue.
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
from src.pydelt.interpolation import SplineInterpolator

# Output directory
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def debug_gradient_corners():
    """Compare numerical vs analytical gradients to debug corner issue."""
    print("Debugging gradient corner issue and domain coverage importance...")
    
    # Create test data: f(x,y) = x^2 + y^2 - 2*x*y = (x-y)^2
    # Test with different sampling densities to show domain coverage importance
    
    # Dense sampling (good coverage)
    x_dense = np.linspace(-3, 3, 25)
    y_dense = np.linspace(-3, 3, 25)
    X_dense, Y_dense = np.meshgrid(x_dense, y_dense)
    Z_dense = X_dense**2 + Y_dense**2 - 2*X_dense*Y_dense
    
    # Sparse sampling (poor coverage) - only corners and center
    sparse_points = np.array([[-3, -3], [-3, 3], [3, -3], [3, 3], [0, 0], 
                             [-1.5, -1.5], [1.5, 1.5], [-1.5, 1.5], [1.5, -1.5]])
    Z_sparse = sparse_points[:, 0]**2 + sparse_points[:, 1]**2 - 2*sparse_points[:, 0]*sparse_points[:, 1]
    
    # Fit multivariate derivatives with both datasets
    input_data_dense = np.column_stack([X_dense.flatten(), Y_dense.flatten()])
    output_data_dense = Z_dense.flatten()
    
    mv_dense = MultivariateDerivatives(SplineInterpolator, smoothing=0.1)
    mv_dense.fit(input_data_dense, output_data_dense)
    
    mv_sparse = MultivariateDerivatives(SplineInterpolator, smoothing=0.1)
    mv_sparse.fit(sparse_points, Z_sparse)
    
    # Evaluation grid
    x_eval = np.linspace(-3, 3, 15)
    y_eval = np.linspace(-3, 3, 15)
    X_eval, Y_eval = np.meshgrid(x_eval, y_eval)
    eval_points = np.column_stack([X_eval.flatten(), Y_eval.flatten()])
    
    # Compute numerical gradients for both dense and sparse cases
    gradient_func_dense = mv_dense.gradient()
    gradient_func_sparse = mv_sparse.gradient()
    
    gradients_dense = gradient_func_dense(eval_points)
    gradients_sparse = gradient_func_sparse(eval_points)
    
    grad_magnitude_dense = np.sqrt(gradients_dense[:, 0]**2 + gradients_dense[:, 1]**2).reshape(X_eval.shape)
    grad_magnitude_sparse = np.sqrt(gradients_sparse[:, 0]**2 + gradients_sparse[:, 1]**2).reshape(X_eval.shape)
    
    # Compute analytical gradients
    grad_x_ana = 2*X_eval - 2*Y_eval
    grad_y_ana = 2*Y_eval - 2*X_eval
    grad_magnitude_ana = np.sqrt(grad_x_ana**2 + grad_y_ana**2)
    
    # Create comparison visualization
    fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=(
            'Function f(x,y) = (x-y)²<br><sub>Original function with valley along x=y</sub>',
            'Dense Sampling (625 points)<br><sub>Good domain coverage</sub>',
            'Sparse Sampling (9 points)<br><sub>Poor domain coverage</sub>',
            'Dense: Gradient Magnitude<br><sub>Reasonable approximation</sub>',
            'Sparse: Gradient Magnitude<br><sub>Poor approximation</sub>',
            'Analytical: Gradient Magnitude<br><sub>True mathematical gradient</sub>',
            'Dense Error: |Num - Ana|<br><sub>Moderate errors from smoothing</sub>',
            'Sparse Error: |Num - Ana|<br><sub>Large errors from poor coverage</sub>',
            'Sampling Points Overlay<br><sub>Dense vs Sparse comparison</sub>'
        ),
        specs=[[{'type': 'surface'}, {'type': 'scatter'}, {'type': 'scatter'}],
               [{'type': 'heatmap'}, {'type': 'heatmap'}, {'type': 'heatmap'}],
               [{'type': 'heatmap'}, {'type': 'heatmap'}, {'type': 'scatter'}]],
        vertical_spacing=0.08
    )
    
    # Function surface
    fig.add_trace(go.Surface(x=X_dense, y=Y_dense, z=Z_dense, colorscale='Viridis', showscale=False), row=1, col=1)
    
    # Sampling point visualizations
    fig.add_trace(go.Scatter(x=X_dense.flatten()[::25], y=Y_dense.flatten()[::25], 
                           mode='markers', marker=dict(size=3, color='blue'),
                           name='Dense sampling'), row=1, col=2)
    fig.add_trace(go.Scatter(x=sparse_points[:, 0], y=sparse_points[:, 1], 
                           mode='markers', marker=dict(size=8, color='red'),
                           name='Sparse sampling'), row=1, col=3)
    
    # Gradient magnitudes
    fig.add_trace(go.Heatmap(x=x_eval, y=y_eval, z=grad_magnitude_dense, 
                           colorscale='Plasma', showscale=False), row=2, col=1)
    fig.add_trace(go.Heatmap(x=x_eval, y=y_eval, z=grad_magnitude_sparse, 
                           colorscale='Plasma', showscale=False), row=2, col=2)
    fig.add_trace(go.Heatmap(x=x_eval, y=y_eval, z=grad_magnitude_ana, 
                           colorscale='Plasma', showscale=False), row=2, col=3)
    
    # Errors
    error_dense = np.abs(grad_magnitude_dense - grad_magnitude_ana)
    error_sparse = np.abs(grad_magnitude_sparse - grad_magnitude_ana)
    fig.add_trace(go.Heatmap(x=x_eval, y=y_eval, z=error_dense, 
                           colorscale='Reds', showscale=False), row=3, col=1)
    fig.add_trace(go.Heatmap(x=x_eval, y=y_eval, z=error_sparse, 
                           colorscale='Reds', showscale=False), row=3, col=2)
    
    # Comparison overlay
    fig.add_trace(go.Scatter(x=X_dense.flatten()[::50], y=Y_dense.flatten()[::50], 
                           mode='markers', marker=dict(size=2, color='blue', opacity=0.6),
                           name='Dense (every 50th point)'), row=3, col=3)
    fig.add_trace(go.Scatter(x=sparse_points[:, 0], y=sparse_points[:, 1], 
                           mode='markers', marker=dict(size=12, color='red', symbol='x'),
                           name='Sparse (all points)'), row=3, col=3)
    
    # Add comprehensive title and annotations
    fig.update_layout(
        title={
            'text': "Domain Coverage Importance in Multivariate Derivatives: f(x,y) = (x-y)²<br><sub>" +
                   "KEY INSIGHT: Dense sampling (625 points) gives reasonable gradient approximations, while " +
                   "sparse sampling (9 points) fails catastrophically. You need data coverage over the ENTIRE " +
                   "domain where you want reliable derivatives, not just at boundaries or critical points.</sub>",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 14}
        },
        height=1200,
        annotations=[
            dict(text="Original function surface", x=0.17, y=0.95, xref="paper", yref="paper",
                 showarrow=False, font=dict(size=10, color="blue")),
            dict(text="625 sample points", x=0.5, y=0.95, xref="paper", yref="paper",
                 showarrow=False, font=dict(size=10, color="blue")),
            dict(text="Only 9 sample points", x=0.83, y=0.95, xref="paper", yref="paper",
                 showarrow=False, font=dict(size=10, color="red")),
            dict(text="Good approximation", x=0.17, y=0.65, xref="paper", yref="paper",
                 showarrow=False, font=dict(size=10, color="green")),
            dict(text="Poor approximation", x=0.5, y=0.65, xref="paper", yref="paper",
                 showarrow=False, font=dict(size=10, color="red")),
            dict(text="True gradient", x=0.83, y=0.65, xref="paper", yref="paper",
                 showarrow=False, font=dict(size=10, color="purple")),
            dict(text="Moderate errors", x=0.17, y=0.35, xref="paper", yref="paper",
                 showarrow=False, font=dict(size=10, color="orange")),
            dict(text="Large errors!", x=0.5, y=0.35, xref="paper", yref="paper",
                 showarrow=False, font=dict(size=10, color="red")),
            dict(text="Coverage comparison", x=0.83, y=0.35, xref="paper", yref="paper",
                 showarrow=False, font=dict(size=10, color="black"))
        ]
    )
    
    fig.update_scenes(camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)))
    
    output_file = os.path.join(OUTPUT_DIR, 'debug_gradient_corners.html')
    fig.write_html(output_file)
    print(f"✅ Debug visualization saved to {output_file}")
    
    # Print corner analysis comparing dense vs sparse
    corners = [(-3, -3), (-3, 3), (3, -3), (3, 3)]
    corner_points = np.array(corners)
    gradients_dense_corners = gradient_func_dense(corner_points)
    gradients_sparse_corners = gradient_func_sparse(corner_points)
    
    print("\nCorner Analysis - Dense vs Sparse Sampling:")
    print("Corner\t\tFunction\tDense |∇f|\tSparse |∇f|\tAnalytical |∇f|\tDense Error\tSparse Error")
    print("-" * 95)
    
    for i, (x, y) in enumerate(corners):
        f_val = x**2 + y**2 - 2*x*y
        grad_mag_dense = np.sqrt(gradients_dense_corners[i, 0]**2 + gradients_dense_corners[i, 1]**2)
        grad_mag_sparse = np.sqrt(gradients_sparse_corners[i, 0]**2 + gradients_sparse_corners[i, 1]**2)
        grad_mag_ana = np.sqrt((2*x - 2*y)**2 + (2*y - 2*x)**2)
        error_dense = abs(grad_mag_dense - grad_mag_ana)
        error_sparse = abs(grad_mag_sparse - grad_mag_ana)
        
        print(f"({x:2},{y:2})\t\t{f_val:6.1f}\t\t{grad_mag_dense:7.3f}\t\t{grad_mag_sparse:7.3f}\t\t{grad_mag_ana:7.3f}\t\t{error_dense:7.3f}\t\t{error_sparse:7.3f}")
    
    print("\n🔍 KEY OBSERVATIONS:")
    print("1. Dense sampling provides much better gradient approximations")
    print("2. Sparse sampling fails especially badly away from sample points")
    print("3. Domain coverage is MORE important than just having critical points")
    print("4. For reliable derivatives, sample densely across the entire domain of interest")

if __name__ == "__main__":
    debug_gradient_corners()
