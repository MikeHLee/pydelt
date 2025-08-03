Quick Start Guide
=================

This guide will help you get started with pydelt quickly.

Basic Derivative Calculation
----------------------------

The most common use case is calculating derivatives of time series data:

.. code-block:: python

   import numpy as np
   from pydelt.derivatives import lla, fda
   
   # Create sample data
   time = np.linspace(0, 2*np.pi, 100)
   signal = np.sin(time)
   
   # Method 1: Local Linear Approximation (recommended for smooth data)
   lla_result = lla(time.tolist(), signal.tolist(), window_size=5)
   derivative_lla = lla_result[0]  # Extract derivatives
   
   # Method 2: Functional Data Analysis (spline-based, good for smooth data)
   fda_result = fda(signal, time)
   derivative_fda = fda_result['dsignal'][:, 1]  # Extract first derivative
   
   # Compare with analytical derivative
   analytical = np.cos(time)
   print(f"LLA Error: {np.mean(np.abs(derivative_lla - analytical)):.4f}")
   print(f"FDA Error: {np.mean(np.abs(derivative_fda - analytical)):.4f}")

Advanced Methods
---------------

For more sophisticated analysis, try neural network-based derivatives:

.. code-block:: python

   from pydelt.autodiff import neural_network_derivative
   
   # Neural network derivative (requires PyTorch or TensorFlow)
   try:
       derivative_nn = neural_network_derivative(
           time, signal, 
           framework='pytorch',  # or 'tensorflow'
           epochs=500,
           hidden_layers=[64, 32]
       )
       
       # Evaluate at specific points
       query_points = np.linspace(0.5, 5.5, 20)
       derivatives_at_points = derivative_nn(query_points)
       print(f"Neural network derivative shape: {derivatives_at_points.shape}")
       
   except ImportError:
       print("PyTorch or TensorFlow not installed - skipping neural network example")

Interpolation
------------

pydelt also provides advanced interpolation methods:

.. code-block:: python

   from pydelt.interpolation import spline_interpolation, lowess_interpolation
   
   # Create sparse, noisy data
   sparse_time = np.linspace(0, 2*np.pi, 20)
   noisy_signal = np.sin(sparse_time) + 0.1 * np.random.randn(20)
   
   # Dense evaluation points
   dense_time = np.linspace(0, 2*np.pi, 100)
   
   # Spline interpolation
   spline_interp = spline_interpolation(sparse_time, noisy_signal)
   spline_values = spline_interp(dense_time)
   
   # LOWESS (locally weighted regression)
   lowess_interp = lowess_interpolation(sparse_time, noisy_signal, frac=0.3)
   lowess_values = lowess_interp(dense_time)

Integration
----------

Calculate integrals with error estimation:

.. code-block:: python

   from pydelt.integrals import integrate_derivative
   
   # If you have derivative data, integrate it back
   time = np.linspace(0, np.pi, 50)
   derivative_data = np.cos(time)  # derivative of sin(x)
   
   # Integrate with initial condition
   integral, error = integrate_derivative(time, derivative_data, initial_value=0.0)
   
   # Compare with analytical integral (sin(x))
   analytical_integral = np.sin(time)
   integration_error = np.mean(np.abs(integral - analytical_integral))
   print(f"Integration error: {integration_error:.4f}")
   print(f"Estimated numerical error: {error:.6f}")

Multivariate Data
----------------

Handle multi-dimensional time series:

.. code-block:: python

   # Create 2D signal: [sin(t), cos(t)]
   time = np.linspace(0, 2*np.pi, 100)
   signal_2d = np.column_stack([np.sin(time), np.cos(time)])
   
   # Calculate derivatives using LLA (supports multivariate data)
   from pydelt.derivatives import lla
   
   result = lla(time.tolist(), signal_2d.tolist(), window_size=5)
   derivatives_2d = result[0]  # Shape: (N, 2) for 2D signal
   
   print(f"Input shape: {signal_2d.shape}")
   print(f"Derivative shape: {derivatives_2d.shape}")
   
   # Expected: [cos(t), -sin(t)]
   expected = np.column_stack([np.cos(time), -np.sin(time)])
   error = np.mean(np.abs(derivatives_2d - expected))
   print(f"Multivariate derivative error: {error:.4f}")

Important Limitations
--------------------

When working with multivariate derivatives, be aware of these numerical limitations:

**Critical Point Smoothing**

Numerical interpolation methods smooth out sharp mathematical features:

.. code-block:: python

   # Example: f(x,y) = (x-y)² has zero gradient along x=y line
   # But numerical methods will give non-zero gradients everywhere
   
   from pydelt.multivariate import MultivariateDerivatives
   from pydelt.interpolation import SplineInterpolator
   
   # At point (-3,-3), gradient should be [0, 0] mathematically
   # But numerical result will be non-zero due to smoothing
   
   # Always validate against analytical solutions when possible
   analytical_gradient = [2*x - 2*y, 2*y - 2*x]  # For f(x,y) = x² + y² - 2xy

**When to Be Cautious:**

* Functions with sharp valleys or ridges
* Optimization problems where exact critical points matter
* Near boundaries of the interpolation domain
* Functions with discontinuities or sharp transitions

**Mitigation Strategies:**

* Use higher resolution sampling near critical points
* Reduce smoothing parameters (but beware of overfitting)
* Compare numerical results with analytical solutions
* Consider neural network methods for exact derivatives

Error Handling
-------------

pydelt provides comprehensive error checking:

.. code-block:: python

   import numpy as np
   from pydelt.derivatives import lla
   
   # These will raise informative errors:
   try:
       # Mismatched array lengths
       lla([1, 2, 3], [1, 2])
   except ValueError as e:
       print(f"Caught expected error: {e}")
   
   try:
       # NaN values
       lla([1, 2, np.nan], [1, 4, 9])
   except ValueError as e:
       print(f"Caught expected error: {e}")

Next Steps
----------

* Check out the :doc:`examples` for more detailed use cases
* Browse the :doc:`api` reference for all available functions
* See the :doc:`changelog` for recent improvements
