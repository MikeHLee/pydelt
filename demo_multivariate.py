#!/usr/bin/env python3
"""
Demonstration of the multivariate derivatives functionality in pydelt.

This script shows how to use the MultivariateDerivatives class to compute
gradients, Jacobians, Hessians, and Laplacians for multivariate functions.
"""

import numpy as np
import matplotlib.pyplot as plt
from src.pydelt.multivariate import MultivariateDerivatives
from src.pydelt.interpolation import SplineInterpolator, LlaInterpolator

def main():
    print("🚀 Multivariate Derivatives Demonstration")
    print("=" * 50)
    
    # Create test data for a 2D scalar function: f(x,y) = x^2 + y^2
    print("\n1. Setting up test data...")
    x = np.linspace(-2, 2, 20)
    y = np.linspace(-2, 2, 20)
    X, Y = np.meshgrid(x, y)
    
    # Input data: (x, y) coordinates
    input_data = np.column_stack([X.flatten(), Y.flatten()])
    
    # Scalar function: f(x,y) = x^2 + y^2
    scalar_output = (X**2 + Y**2).flatten()
    
    # Vector function: [x^2 + y^2, x + y]
    vector_output = np.column_stack([
        (X**2 + Y**2).flatten(),
        (X + Y).flatten()
    ])
    
    print(f"Input shape: {input_data.shape}")
    print(f"Scalar output shape: {scalar_output.shape}")
    print(f"Vector output shape: {vector_output.shape}")
    
    # Test with different interpolators
    interpolators = [
        ("Spline", SplineInterpolator, {"smoothing": 0.1}),
        ("LLA", LlaInterpolator, {"window_size": 5})
    ]
    
    for name, interp_class, kwargs in interpolators:
        print(f"\n2. Testing with {name} Interpolator")
        print("-" * 30)
        
        # Initialize multivariate derivatives
        mv = MultivariateDerivatives(interp_class, **kwargs)
        mv.fit(input_data, scalar_output)
        
        # Test point
        test_point = np.array([[1.0, 1.0]])
        print(f"Test point: {test_point[0]}")
        
        # Compute gradient
        gradient_func = mv.gradient()
        grad = gradient_func(test_point)
        print(f"Gradient: {grad.flatten()}")
        print(f"Expected: [2.0, 2.0] (analytical: [2x, 2y])")
        
        # Compute Hessian
        hessian_func = mv.hessian()
        hess = hessian_func(test_point)
        print(f"Hessian diagonal: {np.diag(hess)}")
        print(f"Expected: [2.0, 2.0] (analytical: [2, 2])")
        
        # Compute Laplacian
        laplacian_func = mv.laplacian()
        lap = laplacian_func(test_point)
        print(f"Laplacian: {lap.flatten()[0]:.4f}")
        print(f"Expected: 4.0 (analytical: ∇²f = 2 + 2 = 4)")
        
        # Now test with vector output
        print("\n3. Testing with Vector Function")
        print("-" * 30)
        
        # Fit with vector output
        mv.fit(input_data, vector_output)
        
        # Compute Jacobian
        jacobian_func = mv.jacobian()
        jac = jacobian_func(test_point)
        print(f"Jacobian at {test_point[0]}:")
        print(jac[0])
        print("Expected:")
        print("[[2.0, 2.0], [1.0, 1.0]] (analytical: [[2x, 2y], [1, 1]])")

def visualize_results():
    """Create visualizations of multivariate derivatives"""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    print("\n4. Creating visualization...")
    
    # Create a finer grid for visualization
    x = np.linspace(-2, 2, 50)
    y = np.linspace(-2, 2, 50)
    X, Y = np.meshgrid(x, y)
    
    # Input data: (x, y) coordinates
    input_data = np.column_stack([X.flatten(), Y.flatten()])
    
    # Scalar function: f(x,y) = x^2 + y^2
    scalar_output = (X**2 + Y**2).flatten()
    
    # Initialize multivariate derivatives with SplineInterpolator
    mv = MultivariateDerivatives(SplineInterpolator, smoothing=0.1)
    mv.fit(input_data, scalar_output)
    
    # Compute gradient and laplacian
    gradient_func = mv.gradient()
    laplacian_func = mv.laplacian()
    
    # Evaluate on the grid
    gradients = gradient_func(input_data)
    laplacians = laplacian_func(input_data)
    
    # Reshape for plotting
    Z = scalar_output.reshape(X.shape)
    grad_x = gradients[:, 0].reshape(X.shape)
    grad_y = gradients[:, 1].reshape(X.shape)
    lap = laplacians.reshape(X.shape)
    
    # Create figure with subplots
    fig = make_subplots(rows=2, cols=2, 
                       subplot_titles=('Function f(x,y) = x² + y²', 
                                      'Gradient Magnitude ||∇f||',
                                      'Gradient Vector Field',
                                      'Laplacian ∇²f'),
                       specs=[[{'type': 'surface'}, {'type': 'surface'}],
                              [{'type': 'contour'}, {'type': 'contour'}]])
    
    # Add surface plot of function
    fig.add_trace(go.Surface(x=X, y=Y, z=Z, colorscale='Viridis',
                            name='f(x,y) = x² + y²'), row=1, col=1)
    
    # Add surface plot of gradient magnitude
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    fig.add_trace(go.Surface(x=X, y=Y, z=grad_mag, colorscale='Plasma',
                            name='||∇f||'), row=1, col=2)
    
    # Add contour plot with gradient vector field
    fig.add_trace(go.Contour(x=x, y=y, z=Z, colorscale='Viridis',
                            name='f(x,y)'), row=2, col=1)
    
    # Add vector field (subsample for clarity)
    skip = 4
    fig.add_trace(go.Scatter(x=X[::skip, ::skip].flatten(), 
                           y=Y[::skip, ::skip].flatten(),
                           mode='markers+text',
                           marker=dict(symbol='arrow', size=10,
                                     angle=np.arctan2(grad_y[::skip, ::skip], 
                                                     grad_x[::skip, ::skip]).flatten()*180/np.pi),
                           text='→', textposition='middle center',
                           name='∇f'), row=2, col=1)
    
    # Add Laplacian contour plot
    fig.add_trace(go.Contour(x=x, y=y, z=lap, colorscale='RdBu',
                            contours=dict(start=3.9, end=4.1, size=0.01),
                            name='∇²f'), row=2, col=2)
    
    # Update layout
    fig.update_layout(title_text='Multivariate Derivatives Visualization',
                     height=800, width=1000,
                     scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='f(x,y)'),
                     scene2=dict(xaxis_title='x', yaxis_title='y', zaxis_title='||∇f||'))
    
    # Save to HTML file
    fig.write_html('multivariate_derivatives_demo.html')
    print("Visualization saved to 'multivariate_derivatives_demo.html'")

if __name__ == "__main__":
    main()
    try:
        visualize_results()
    except ImportError:
        print("\nVisualization requires plotly. Install with: pip install plotly")
    
    print(f"\n4. Multiple Point Evaluation")
    print("-" * 30)
    
    # Create test data again for this section
    x = np.linspace(-2, 2, 20)
    y = np.linspace(-2, 2, 20)
    X, Y = np.meshgrid(x, y)
    
    # Input data: (x, y) coordinates
    grid_input_data = np.column_stack([X.flatten(), Y.flatten()])
    
    # Scalar function: f(x,y) = x^2 + y^2
    grid_scalar_output = (X**2 + Y**2).flatten()
    
    # Test with multiple points
    test_points = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    mv = MultivariateDerivatives(SplineInterpolator, smoothing=0.1)
    mv.fit(grid_input_data, grid_scalar_output)
    
    gradient_func = mv.gradient()
    gradients = gradient_func(test_points)
    
    print(f"Test points shape: {test_points.shape}")
    print(f"Gradients shape: {gradients.shape}")
    print("Gradients at multiple points:")
    for i, (point, grad) in enumerate(zip(test_points, gradients)):
        print(f"  Point {point}: Gradient {grad}")
    
    print(f"\n✅ Multivariate derivatives demonstration complete!")
    print("The module successfully computes gradients, Jacobians, Hessians, and Laplacians")
    print("for both scalar and vector-valued multivariate functions.")

if __name__ == "__main__":
    main()
