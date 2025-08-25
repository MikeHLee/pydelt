Multivariate Calculus & Vector Operations
========================================

Multivariate calculus extends differentiation to functions of multiple variables, enabling analysis of vector fields, optimization landscapes, and tensor operations. This section covers gradients, Jacobians, Hessians, and their applications in scientific computing and machine learning.

🎯 **Core Concepts**
-------------------

**Scalar Functions**: f: ℝⁿ → ℝ (multiple inputs, single output)
- **Gradient**: ∇f = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ] - direction of steepest ascent
- **Hessian**: H_f = ∂²f/∂xᵢ∂xⱼ - matrix of second derivatives
- **Laplacian**: ∇²f = tr(H_f) - sum of diagonal Hessian elements

**Vector Functions**: f: ℝⁿ → ℝᵐ (multiple inputs, multiple outputs)
- **Jacobian**: J_f = ∂fᵢ/∂xⱼ - matrix of all first-order partial derivatives

**Physical Interpretations**:
- **Gradient**: Electric field from potential, temperature gradient
- **Jacobian**: Velocity field transformation, system linearization
- **Hessian**: Curvature, optimization landscape
- **Laplacian**: Heat diffusion, wave propagation

🔧 **MultivariateDerivatives Class**
-----------------------------------

The ``MultivariateDerivatives`` class provides a unified interface for multivariate calculus operations:

.. code-block:: python

   from pydelt.multivariate import MultivariateDerivatives
   from pydelt.interpolation import SplineInterpolator
   
   # Create multivariate derivatives object
   mv = MultivariateDerivatives(
       interpolator_class=SplineInterpolator,
       smoothing=0.1  # Parameters passed to interpolator
   )
   
   # Fit to data
   mv.fit(input_data, output_data)
   
   # Compute various derivatives
   gradient_func = mv.gradient()      # For scalar functions
   jacobian_func = mv.jacobian()      # For vector functions  
   hessian_func = mv.hessian()        # Second derivatives
   laplacian_func = mv.laplacian()    # Scalar from Hessian trace

🌋 **Example 1: Optimization Landscape Analysis**
------------------------------------------------

**Classic Example: Rosenbrock Function**

The Rosenbrock function is a classic optimization test case with a curved valley:

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from pydelt.multivariate import MultivariateDerivatives
   from pydelt.interpolation import SplineInterpolator
   
   # Rosenbrock function: f(x,y) = (a-x)² + b(y-x²)²
   # Global minimum at (a,a²) with a=1, b=100
   def rosenbrock(x, y, a=1, b=100):
       return (a - x)**2 + b * (y - x**2)**2
   
   def rosenbrock_gradient(x, y, a=1, b=100):
       df_dx = -2*(a - x) - 4*b*x*(y - x**2)
       df_dy = 2*b*(y - x**2)
       return np.array([df_dx, df_dy])
   
   # Generate training data on a grid
   x = np.linspace(-2, 2, 25)
   y = np.linspace(-1, 3, 25)
   X, Y = np.meshgrid(x, y)
   
   # Flatten for input
   input_data = np.column_stack([X.flatten(), Y.flatten()])
   output_data = rosenbrock(X.flatten(), Y.flatten())
   
   # Fit multivariate derivatives
   mv = MultivariateDerivatives(SplineInterpolator, smoothing=0.1)
   mv.fit(input_data, output_data)
   
   # Compute gradient function
   gradient_func = mv.gradient()
   
   # Evaluate gradient at test points
   test_points = np.array([
       [0.0, 0.0],    # Away from minimum
       [1.0, 1.0],    # At minimum
       [0.5, 0.25],   # On the valley floor
       [-1.0, 1.0]    # Another test point
   ])
   
   gradients_numerical = gradient_func(test_points)
   gradients_analytical = np.array([rosenbrock_gradient(p[0], p[1]) for p in test_points])
   
   print("Gradient Analysis:")
   for i, point in enumerate(test_points):
       num_grad = gradients_numerical[i]
       ana_grad = gradients_analytical[i]
       error = np.linalg.norm(num_grad - ana_grad)
       print(f"Point {point}: Numerical {num_grad}, Analytical {ana_grad}, Error: {error:.4f}")
   
   # Find critical points (where gradient ≈ 0)
   # Create dense evaluation grid
   x_dense = np.linspace(-2, 2, 100)
   y_dense = np.linspace(-1, 3, 100)
   X_dense, Y_dense = np.meshgrid(x_dense, y_dense)
   eval_points = np.column_stack([X_dense.flatten(), Y_dense.flatten()])
   
   gradients_dense = gradient_func(eval_points)
   gradient_magnitudes = np.linalg.norm(gradients_dense, axis=1)
   
   # Find minimum gradient magnitude (critical point)
   min_idx = np.argmin(gradient_magnitudes)
   critical_point = eval_points[min_idx]
   min_gradient_mag = gradient_magnitudes[min_idx]
   
   print(f"\nCritical point found at: ({critical_point[0]:.3f}, {critical_point[1]:.3f})")
   print(f"Gradient magnitude: {min_gradient_mag:.6f}")
   print(f"True minimum at: (1.000, 1.000)")

🌊 **Example 2: Fluid Dynamics - Vector Field Analysis**
-------------------------------------------------------

**Application**: Analysis of 2D fluid flow with vorticity and divergence computation.

.. code-block:: python

   # Double gyre flow - classic fluid dynamics example
   def double_gyre_velocity(x, y, t=0, A=0.1, omega=0.2, epsilon=0.25):
       """Double gyre velocity field - chaotic mixing flow"""
       a = epsilon * np.sin(omega * t)
       b = 1 - 2 * epsilon * np.sin(omega * t)
       f = a * x**2 + b * x
       
       u = -np.pi * A * np.sin(np.pi * f) * np.cos(np.pi * y)
       v = np.pi * A * np.cos(np.pi * f) * np.sin(np.pi * y) * (2*a*x + b)
       
       return u, v
   
   # Generate velocity field data
   x = np.linspace(0, 2, 30)
   y = np.linspace(0, 1, 15)
   X, Y = np.meshgrid(x, y)
   
   U, V = double_gyre_velocity(X, Y)
   
   # Prepare data for multivariate analysis
   input_data = np.column_stack([X.flatten(), Y.flatten()])
   output_data = np.column_stack([U.flatten(), V.flatten()])  # Vector field
   
   # Fit multivariate derivatives for vector function
   mv_vector = MultivariateDerivatives(SplineInterpolator, smoothing=0.05)
   mv_vector.fit(input_data, output_data)
   
   # Compute Jacobian matrix for flow analysis
   jacobian_func = mv_vector.jacobian()
   
   # Evaluate at analysis points
   analysis_points = np.array([
       [0.5, 0.5],   # Center of left gyre
       [1.5, 0.5],   # Center of right gyre
       [1.0, 0.5],   # Saddle point
       [0.2, 0.8],   # Edge region
   ])
   
   jacobians = jacobian_func(analysis_points)
   
   print("Flow Field Analysis:")
   for i, point in enumerate(analysis_points):
       J = jacobians[i]  # 2x2 Jacobian matrix
       
       # Extract components: J = [[∂u/∂x, ∂u/∂y], [∂v/∂x, ∂v/∂y]]
       du_dx, du_dy = J[0, :]
       dv_dx, dv_dy = J[1, :]
       
       # Compute flow properties
       divergence = du_dx + dv_dy  # ∇·v (expansion/contraction)
       vorticity = dv_dx - du_dy   # ∇×v (rotation)
       
       # Strain rate tensor components
       strain_rate = 0.5 * (J + J.T)  # Symmetric part
       rotation_rate = 0.5 * (J - J.T)  # Antisymmetric part
       
       # Eigenvalues for stability analysis
       eigenvals = np.linalg.eigvals(J)
       
       print(f"\nPoint ({point[0]:.1f}, {point[1]:.1f}):")
       print(f"  Divergence: {divergence:.4f}")
       print(f"  Vorticity:  {vorticity:.4f}")
       print(f"  Eigenvalues: {eigenvals[0]:.4f} + {eigenvals[1]:.4f}i")
       
       # Classify flow behavior
       if np.abs(divergence) < 0.01 and np.abs(vorticity) > 0.1:
           flow_type = "Rotational (vortex)"
       elif np.abs(vorticity) < 0.01 and divergence > 0.1:
           flow_type = "Divergent (source)"
       elif np.abs(vorticity) < 0.01 and divergence < -0.1:
           flow_type = "Convergent (sink)"
       elif np.real(eigenvals).prod() < 0:
           flow_type = "Saddle point"
       else:
           flow_type = "Complex flow"
       
       print(f"  Flow type: {flow_type}")

🔬 **Example 3: Heat Diffusion Analysis**
----------------------------------------

**Application**: Temperature field analysis with Laplacian computation for heat equation.

.. code-block:: python

   # 2D heat distribution with multiple sources
   def temperature_field(x, y):
       """Temperature field with multiple heat sources"""
       # Heat source at (0.3, 0.3)
       T1 = 100 * np.exp(-((x - 0.3)**2 + (y - 0.3)**2) / 0.1**2)
       
       # Heat source at (0.7, 0.7)  
       T2 = 80 * np.exp(-((x - 0.7)**2 + (y - 0.7)**2) / 0.15**2)
       
       # Heat sink at (0.5, 0.8)
       T3 = -60 * np.exp(-((x - 0.5)**2 + (y - 0.8)**2) / 0.08**2)
       
       # Background temperature
       T_bg = 20
       
       return T1 + T2 + T3 + T_bg
   
   def temperature_laplacian_analytical(x, y):
       """Analytical Laplacian for validation"""
       # This would be complex to compute analytically
       # Using finite differences for validation
       h = 0.001
       d2T_dx2 = (temperature_field(x+h, y) - 2*temperature_field(x, y) + temperature_field(x-h, y)) / h**2
       d2T_dy2 = (temperature_field(x, y+h) - 2*temperature_field(x, y) + temperature_field(x, y-h)) / h**2
       return d2T_dx2 + d2T_dy2
   
   # Generate temperature measurement data
   x = np.linspace(0, 1, 40)
   y = np.linspace(0, 1, 40)
   X, Y = np.meshgrid(x, y)
   
   # Add measurement noise
   T_true = temperature_field(X, Y)
   T_measured = T_true + 0.5 * np.random.randn(*T_true.shape)
   
   # Prepare data
   input_data = np.column_stack([X.flatten(), Y.flatten()])
   output_data = T_measured.flatten()
   
   # Fit multivariate derivatives
   mv_temp = MultivariateDerivatives(SplineInterpolator, smoothing=0.2)
   mv_temp.fit(input_data, output_data)
   
   # Compute gradient and Laplacian
   gradient_func = mv_temp.gradient()
   laplacian_func = mv_temp.laplacian()
   
   # Analysis points
   analysis_points = np.array([
       [0.3, 0.3],   # Near heat source 1
       [0.7, 0.7],   # Near heat source 2
       [0.5, 0.8],   # Near heat sink
       [0.5, 0.5],   # Center region
       [0.1, 0.1],   # Edge region
   ])
   
   gradients = gradient_func(analysis_points)
   laplacians = laplacian_func(analysis_points)
   
   print("Heat Diffusion Analysis:")
   for i, point in enumerate(analysis_points):
       grad = gradients[i]
       lapl = laplacians[i]
       
       # Heat flux (proportional to negative gradient)
       heat_flux_magnitude = np.linalg.norm(grad)
       heat_flux_direction = -grad / (heat_flux_magnitude + 1e-10)
       
       print(f"\nPoint ({point[0]:.1f}, {point[1]:.1f}):")
       print(f"  Temperature gradient: [{grad[0]:.2f}, {grad[1]:.2f}] K/m")
       print(f"  Heat flux magnitude: {heat_flux_magnitude:.2f}")
       print(f"  Heat flux direction: [{heat_flux_direction[0]:.3f}, {heat_flux_direction[1]:.3f}]")
       print(f"  Laplacian (∇²T): {lapl:.2f} K/m²")
       
       # Interpret Laplacian for heat equation ∂T/∂t = α∇²T
       if lapl > 1:
           diffusion_behavior = "Net heat accumulation (heating up)"
       elif lapl < -1:
           diffusion_behavior = "Net heat loss (cooling down)"
       else:
           diffusion_behavior = "Near thermal equilibrium"
       
       print(f"  Diffusion behavior: {diffusion_behavior}")

⚙️ **Advanced Features**
-----------------------

**Mixed Partial Derivatives**

For functions requiring cross-derivatives:

.. code-block:: python

   # Compute mixed partials ∂²f/∂x∂y
   hessian_func = mv.hessian()
   hessians = hessian_func(test_points)
   
   for i, H in enumerate(hessians):
       mixed_partial = H[0, 1]  # ∂²f/∂x∂y = ∂²f/∂y∂x
       print(f"Mixed partial at point {i}: {mixed_partial:.4f}")

**Tensor Operations**

For higher-dimensional tensor calculus:

.. code-block:: python

   # Vector field curl in 3D (requires 3D vector field)
   # ∇×F = [∂Fz/∂y - ∂Fy/∂z, ∂Fx/∂z - ∂Fz/∂x, ∂Fy/∂x - ∂Fx/∂y]
   
   # For 2D vector fields, scalar curl:
   jacobian = jacobian_func(point)
   curl_2d = jacobian[1, 0] - jacobian[0, 1]  # ∂v/∂x - ∂u/∂y

**Interpolator Method Comparison**

Different interpolators for different applications:

.. code-block:: python

   from pydelt.interpolation import LowessInterpolator, NeuralNetworkInterpolator
   
   # Robust to noise
   mv_robust = MultivariateDerivatives(LowessInterpolator, frac=0.3)
   
   # High accuracy with automatic differentiation
   mv_neural = MultivariateDerivatives(NeuralNetworkInterpolator, 
                                      hidden_layers=[128, 64], epochs=2000)

🎓 **Best Practices**
--------------------

**Data Preparation**:
1. **Grid vs Scattered Data**: Regular grids work best, but scattered data is supported
2. **Data Density**: Ensure sufficient sampling in regions of interest
3. **Noise Handling**: Use robust interpolators (LOWESS) for noisy data
4. **Scaling**: Normalize input coordinates to similar ranges

**Method Selection**:
- **Smooth Functions**: SplineInterpolator for best accuracy
- **Noisy Data**: LowessInterpolator for robustness
- **High Precision**: NeuralNetworkInterpolator with automatic differentiation
- **Large Datasets**: Consider computational cost vs accuracy trade-offs

**Validation**:
Always validate against analytical solutions when available:

.. code-block:: python

   # Compare numerical vs analytical gradients
   grad_numerical = gradient_func(test_points)
   grad_analytical = analytical_gradient(test_points)
   
   errors = np.linalg.norm(grad_numerical - grad_analytical, axis=1)
   print(f"RMS gradient error: {np.sqrt(np.mean(errors**2)):.6f}")

**Physical Interpretation**:
- **Gradient**: Points in direction of steepest increase
- **Divergence > 0**: Source behavior (expansion)
- **Divergence < 0**: Sink behavior (contraction)  
- **Vorticity ≠ 0**: Rotational flow
- **Laplacian > 0**: Concave up (local minimum tendency)
- **Laplacian < 0**: Concave down (local maximum tendency)

⚠️ **Limitations**
-----------------

**Mixed Partials**: Traditional interpolation methods approximate mixed partials as zero. For exact mixed derivatives, use neural networks with automatic differentiation.

**Boundary Effects**: Derivatives near data boundaries may be less accurate due to extrapolation.

**Curse of Dimensionality**: Accuracy decreases with increasing input dimensions. Consider dimensionality reduction for high-dimensional problems.

🔗 **Next Steps**
----------------

Multivariate calculus provides the foundation for advanced stochastic computing. The next level introduces:

- **Stochastic Computing**: Probabilistic derivatives with uncertainty quantification
- **Stochastic Link Functions**: Model derivatives under different probability distributions
- **Financial Applications**: Option pricing, risk analysis, and volatility modeling

The gradient and Jacobian computations become especially powerful when combined with stochastic transformations for uncertainty propagation and risk analysis.
