Basic Interpolation & Derivatives
==================================

This section covers the fundamental interpolation methods and derivative computation in pydelt. These classical methods form the foundation for all advanced features and provide reliable, well-understood approaches to numerical differentiation.

🎯 **Core Concepts**
-------------------

**Interpolation** is the process of constructing new data points within the range of known data points. **Numerical differentiation** computes derivatives by approximating the rate of change using these interpolated functions.

**Universal API Pattern**
All interpolation methods in pydelt follow the same consistent interface:

.. code-block:: python

   # Universal pattern for all interpolators
   interpolator = InterpolatorClass(**parameters)
   interpolator.fit(input_data, output_data)
   derivative_func = interpolator.differentiate(order=1, mask=None)
   derivatives = derivative_func(evaluation_points)

📊 **Method Comparison**
-----------------------

+------------------+-------------+---------------+----------------+------------------+
| Method           | Accuracy    | Noise Robust  | Speed          | Best Use Case    |
+==================+=============+===============+================+==================+
| SplineInterpolator| High        | Low           | Fast           | Smooth data      |
+------------------+-------------+---------------+----------------+------------------+
| LowessInterpolator| Medium      | High          | Medium         | Noisy data       |
+------------------+-------------+---------------+----------------+------------------+
| LoessInterpolator | Medium      | High          | Slow           | Outlier-prone    |
+------------------+-------------+---------------+----------------+------------------+
| LlaInterpolator  | High        | Medium        | Fast           | Local analysis   |
+------------------+-------------+---------------+----------------+------------------+
| GllaInterpolator | Very High   | Medium        | Medium         | High precision   |
+------------------+-------------+---------------+----------------+------------------+

🌟 **1. Spline Interpolation**
-----------------------------

**Concept**: Piecewise polynomial functions that are smooth at connection points. Excellent for smooth data with minimal noise.

**Mathematical Foundation**: Cubic splines minimize the integrated squared second derivative, providing the "smoothest" possible interpolation.

**Classic Example: Projectile Motion**

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from pydelt.interpolation import SplineInterpolator
   
   # Projectile motion: y = x*tan(θ) - (g*x²)/(2*v₀²*cos²(θ))
   # Parameters: v₀=20 m/s, θ=45°, g=9.81 m/s²
   x = np.linspace(0, 40, 20)  # Horizontal distance
   y = x * np.tan(np.pi/4) - (9.81 * x**2) / (2 * 20**2 * np.cos(np.pi/4)**2)
   
   # Add small amount of measurement noise
   y_noisy = y + 0.1 * np.random.randn(len(y))
   
   # Fit spline interpolator
   spline = SplineInterpolator(smoothing=0.1)
   spline.fit(x, y_noisy)
   
   # Compute velocity (first derivative) and acceleration (second derivative)
   velocity_func = spline.differentiate(order=1)
   acceleration_func = spline.differentiate(order=2)
   
   # Evaluate at dense points for smooth curves
   x_dense = np.linspace(0, 40, 200)
   y_smooth = spline.predict(x_dense)
   velocity = velocity_func(x_dense)
   acceleration = acceleration_func(x_dense)
   
   # Theoretical values for comparison
   v_theoretical = np.tan(np.pi/4) - (9.81 * x_dense) / (20**2 * np.cos(np.pi/4)**2)
   a_theoretical = -9.81 / (20**2 * np.cos(np.pi/4)**2) * np.ones_like(x_dense)
   
   print(f"Velocity error (RMS): {np.sqrt(np.mean((velocity - v_theoretical)**2)):.3f}")
   print(f"Acceleration error (RMS): {np.sqrt(np.mean((acceleration - a_theoretical)**2)):.3f}")

**Key Parameters**:
- ``smoothing``: Controls trade-off between fitting data exactly vs. smoothness
- ``degree``: Polynomial degree (default: 3 for cubic splines)

📈 **2. Local Linear Approximation (LLA)**
------------------------------------------

**Concept**: Fits local linear models in sliding windows. Provides excellent balance between accuracy and computational efficiency.

**Mathematical Foundation**: Uses weighted least squares with kernel functions to emphasize nearby points.

**Classic Example: Population Growth Analysis**

.. code-block:: python

   from pydelt.interpolation import LlaInterpolator
   
   # Logistic population growth: P(t) = K / (1 + A*exp(-r*t))
   # Parameters: K=1000 (carrying capacity), r=0.1 (growth rate), A=9
   t = np.linspace(0, 50, 25)
   population = 1000 / (1 + 9 * np.exp(-0.1 * t))
   
   # Add realistic measurement noise
   pop_noisy = population + 20 * np.random.randn(len(population))
   
   # Fit LLA interpolator
   lla = LlaInterpolator(window_size=7, polynomial_degree=1)
   lla.fit(t, pop_noisy)
   
   # Compute growth rate (first derivative)
   growth_rate_func = lla.differentiate(order=1)
   growth_rate = growth_rate_func(t)
   
   # Theoretical growth rate: dP/dt = r*P*(1 - P/K)
   theoretical_rate = 0.1 * population * (1 - population/1000)
   
   print(f"Growth rate error (RMS): {np.sqrt(np.mean((growth_rate - theoretical_rate)**2)):.3f}")
   
   # Find maximum growth rate
   max_growth_idx = np.argmax(growth_rate)
   print(f"Maximum growth rate at t={t[max_growth_idx]:.1f}, P={population[max_growth_idx]:.0f}")

**Key Parameters**:
- ``window_size``: Number of points in local fitting window
- ``polynomial_degree``: Degree of local polynomial (1=linear, 2=quadratic)

🔧 **3. LOWESS (Locally Weighted Scatterplot Smoothing)**
--------------------------------------------------------

**Concept**: Robust regression method that handles outliers and noise effectively. Uses iterative reweighting to downweight outliers.

**Mathematical Foundation**: Combines local polynomial fitting with robust M-estimation techniques.

**Classic Example: Economic Time Series with Outliers**

.. code-block:: python

   from pydelt.interpolation import LowessInterpolator
   
   # Economic indicator with trend, seasonality, and outliers
   t = np.linspace(0, 4*np.pi, 100)
   trend = 0.5 * t  # Linear trend
   seasonal = 2 * np.sin(t)  # Seasonal component
   noise = 0.3 * np.random.randn(len(t))
   
   # Add some outliers (economic shocks)
   outlier_indices = [20, 45, 75]
   economic_data = trend + seasonal + noise
   economic_data[outlier_indices] += [-3, 4, -2.5]  # Outliers
   
   # Fit LOWESS interpolator (robust to outliers)
   lowess = LowessInterpolator(frac=0.3, it=3)
   lowess.fit(t, economic_data)
   
   # Compare with spline (not robust)
   spline = SplineInterpolator(smoothing=0.1)
   spline.fit(t, economic_data)
   
   # Compute derivatives
   lowess_deriv_func = lowess.differentiate(order=1)
   spline_deriv_func = spline.differentiate(order=1)
   
   t_eval = np.linspace(0, 4*np.pi, 200)
   lowess_smooth = lowess.predict(t_eval)
   spline_smooth = spline.predict(t_eval)
   lowess_deriv = lowess_deriv_func(t_eval)
   spline_deriv = spline_deriv_func(t_eval)
   
   # Theoretical derivative (without outliers)
   theoretical_deriv = 0.5 + 2 * np.cos(t_eval)
   
   print(f"LOWESS derivative error: {np.sqrt(np.mean((lowess_deriv - theoretical_deriv)**2)):.3f}")
   print(f"Spline derivative error: {np.sqrt(np.mean((spline_deriv - theoretical_deriv)**2)):.3f}")

**Key Parameters**:
- ``frac``: Fraction of data used in each local regression (0.2-0.8)
- ``it``: Number of robustifying iterations (2-5)

⚙️ **Advanced Features**
-----------------------

**Higher-Order Derivatives**

All interpolators support arbitrary-order derivatives:

.. code-block:: python

   # Compute up to 3rd derivative
   first_deriv = interpolator.differentiate(order=1)
   second_deriv = interpolator.differentiate(order=2)
   third_deriv = interpolator.differentiate(order=3)

**Masking for Partial Analysis**

Compute derivatives only for specific data points:

.. code-block:: python

   # Boolean mask
   mask = np.array([True, False, True, True, False])
   partial_deriv = interpolator.differentiate(order=1, mask=mask)
   
   # Index mask
   indices = [0, 2, 3]  # Only these points
   indexed_deriv = interpolator.differentiate(order=1, mask=indices)

**Callable Functions for Flexible Evaluation**

Derivative functions can be evaluated at any points:

.. code-block:: python

   deriv_func = interpolator.differentiate(order=1)
   
   # Evaluate at original points
   derivs_original = deriv_func(input_data)
   
   # Evaluate at new points
   new_points = np.linspace(input_data.min(), input_data.max(), 1000)
   derivs_dense = deriv_func(new_points)

🎓 **Best Practices**
--------------------

**Method Selection Guidelines**:

1. **Clean, smooth data**: Use ``SplineInterpolator`` for best accuracy
2. **Noisy data**: Use ``LowessInterpolator`` or ``LoessInterpolator``
3. **Local analysis**: Use ``LlaInterpolator`` for computational efficiency
4. **High precision**: Use ``GllaInterpolator`` for critical applications
5. **Outliers present**: Always use ``LowessInterpolator`` or ``LoessInterpolator``

**Parameter Tuning**:

- Start with default parameters
- Increase smoothing/window size for noisy data
- Decrease smoothing for detailed features
- Use cross-validation for optimal parameter selection

**Error Assessment**:

.. code-block:: python

   # Always validate against known analytical solutions when possible
   analytical_deriv = np.cos(x)  # For f(x) = sin(x)
   numerical_deriv = derivative_func(x)
   error = np.sqrt(np.mean((numerical_deriv - analytical_deriv)**2))
   print(f"RMS Error: {error:.6f}")

🔗 **Next Steps**
----------------

Once you're comfortable with basic interpolation methods, explore:

- **Neural Networks**: Deep learning-based interpolation with automatic differentiation
- **Multivariate Calculus**: Gradients, Jacobians, and tensor operations
- **Stochastic Computing**: Probabilistic derivatives with uncertainty quantification

The universal API ensures that transitioning between methods requires only changing the interpolator class while keeping all other code identical.
