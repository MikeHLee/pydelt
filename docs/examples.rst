Examples
========

This page contains detailed examples showing how to use pydelt for various tasks.

Example 1: Comparing Derivative Methods
---------------------------------------

Let's compare different derivative calculation methods on a known function:

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from pydelt.derivatives import lla, fda, gold, glla
   
   # Generate test signal: f(x) = sin(2x) + 0.5*cos(5x)
   x = np.linspace(0, 2*np.pi, 200)
   signal = np.sin(2*x) + 0.5*np.cos(5*x)
   
   # Analytical derivative: f'(x) = 2*cos(2x) - 2.5*sin(5x)
   analytical = 2*np.cos(2*x) - 2.5*np.sin(5*x)
   
   # Compare different methods
   lla_result = lla(x.tolist(), signal.tolist(), window_size=5)
   fda_result = fda(signal, x)
   gold_result = gold(signal, x, embedding=3)
   glla_result = glla(signal, x, embedding=3)
   
   methods = {
       'LLA': lla_result[0],
       'FDA': fda_result['dsignal'][:, 1],
       'GOLD': gold_result['dsignal'][:, 0],
       'GLLA': glla_result['dsignal'][:, 0]
   }
   
   # Calculate errors for each method
   errors = {}
   
   for name, derivative in methods.items():
       try:
           errors[name] = np.mean(np.abs(derivative - analytical))
           print(f"{name}: Mean absolute error = {errors[name]:.6f}")
       except Exception as e:
           print(f"{name}: Failed with error {e}")
   
   # Plot results
   plt.figure(figsize=(12, 8))
   
   plt.subplot(2, 1, 1)
   plt.plot(x, signal, 'k-', label='Original signal', linewidth=2)
   plt.plot(x, analytical, 'r--', label='Analytical derivative', linewidth=2)
   plt.legend()
   plt.title('Original Signal and Analytical Derivative')
   
   plt.subplot(2, 1, 2)
   for name, derivative in methods.items():
       plt.plot(x, derivative, label=f'{name} (error: {errors[name]:.4f})')
   plt.plot(x, analytical, 'r--', label='Analytical', linewidth=2)
   plt.legend()
   plt.title('Derivative Comparison')
   plt.xlabel('x')
   
   plt.tight_layout()
   plt.show()

Example 2: Neural Network Derivatives
-------------------------------------

Using neural networks for derivative calculation:

.. code-block:: python

   import numpy as np
   from pydelt.autodiff import neural_network_derivative
   
   # Create training data
   x_train = np.linspace(0, 4*np.pi, 100)
   y_train = np.exp(-0.1*x_train) * np.sin(x_train)
   
   # Train neural network
   try:
       # Using PyTorch backend
       derivative_func = neural_network_derivative(
           x_train, y_train,
           framework='pytorch',
           hidden_layers=[128, 64, 32],
           epochs=1000,
           learning_rate=0.001,
           dropout=0.1
       )
       
       # Evaluate on test points
       x_test = np.linspace(0.5, 3.5*np.pi, 50)
       derivatives = derivative_func(x_test)
       
       # Analytical derivative for comparison
       analytical = -0.1*np.exp(-0.1*x_test)*np.sin(x_test) + np.exp(-0.1*x_test)*np.cos(x_test)
       
       error = np.mean(np.abs(derivatives.flatten() - analytical))
       print(f"Neural network derivative error: {error:.4f}")
       
       # Plot results
       import matplotlib.pyplot as plt
       plt.figure(figsize=(10, 6))
       plt.plot(x_test, analytical, 'r-', label='Analytical derivative', linewidth=2)
       plt.plot(x_test, derivatives.flatten(), 'b--', label='Neural network', linewidth=2)
       plt.legend()
       plt.title('Neural Network vs Analytical Derivative')
       plt.xlabel('x')
       plt.ylabel("f'(x)")
       plt.show()
       
   except ImportError:
       print("PyTorch not available - install with: pip install torch")

Example 3: Noisy Data Processing
--------------------------------

Handling noisy time series data:

.. code-block:: python

   import numpy as np
   from pydelt.derivatives import lla
   from pydelt.interpolation import lowess_interpolation, spline_interpolation
   
   # Create noisy data
   np.random.seed(42)
   t = np.linspace(0, 10, 100)
   clean_signal = np.sin(t) * np.exp(-0.1*t)
   noise = 0.2 * np.random.randn(len(t))
   noisy_signal = clean_signal + noise
   
   # Method 1: Direct derivative of noisy data
   direct_result = lla(t.tolist(), noisy_signal.tolist(), window_size=5)
   direct_derivative = direct_result[0]
   
   # Method 2: Smooth first, then differentiate
   smoother = lowess_interpolation(t, noisy_signal, frac=0.1)
   smoothed_signal = smoother(t)
   smooth_result = lla(t.tolist(), smoothed_signal.tolist(), window_size=5)
   smooth_derivative = smooth_result[0]
   
   # Method 3: Spline smoothing
   spline_smoother = spline_interpolation(t, noisy_signal, smoothing_factor=1.0)
   spline_signal = spline_smoother(t)
   spline_result = lla(t.tolist(), spline_signal.tolist(), window_size=5)
   spline_derivative = spline_result[0]
   
   # Analytical derivative for comparison
   analytical_derivative = np.cos(t)*np.exp(-0.1*t) - 0.1*np.sin(t)*np.exp(-0.1*t)
   
   # Calculate errors
   errors = {
       'Direct': np.mean(np.abs(direct_derivative - analytical_derivative)),
       'LOWESS': np.mean(np.abs(smooth_derivative - analytical_derivative)),
       'Spline': np.mean(np.abs(spline_derivative - analytical_derivative))
   }
   
   for method, error in errors.items():
       print(f"{method} derivative error: {error:.4f}")
   
   # Plotting
   import matplotlib.pyplot as plt
   fig, axes = plt.subplots(2, 2, figsize=(15, 10))
   
   # Original signals
   axes[0,0].plot(t, clean_signal, 'g-', label='Clean signal', linewidth=2)
   axes[0,0].plot(t, noisy_signal, 'k.', alpha=0.5, label='Noisy signal')
   axes[0,0].plot(t, smoothed_signal, 'b-', label='LOWESS smoothed')
   axes[0,0].plot(t, spline_signal, 'r-', label='Spline smoothed')
   axes[0,0].legend()
   axes[0,0].set_title('Signal Comparison')
   
   # Derivatives
   axes[0,1].plot(t, analytical_derivative, 'g-', label='Analytical', linewidth=3)
   axes[0,1].plot(t, direct_derivative, 'k-', alpha=0.7, label=f'Direct (err: {errors["Direct"]:.3f})')
   axes[0,1].plot(t, smooth_derivative, 'b-', label=f'LOWESS (err: {errors["LOWESS"]:.3f})')
   axes[0,1].plot(t, spline_derivative, 'r-', label=f'Spline (err: {errors["Spline"]:.3f})')
   axes[0,1].legend()
   axes[0,1].set_title('Derivative Comparison')
   
   # Error plots
   axes[1,0].plot(t, np.abs(direct_derivative - analytical_derivative), 'k-', label='Direct')
   axes[1,0].plot(t, np.abs(smooth_derivative - analytical_derivative), 'b-', label='LOWESS')
   axes[1,0].plot(t, np.abs(spline_derivative - analytical_derivative), 'r-', label='Spline')
   axes[1,0].set_yscale('log')
   axes[1,0].legend()
   axes[1,0].set_title('Absolute Errors (log scale)')
   axes[1,0].set_xlabel('Time')
   
   # Histogram of errors
   axes[1,1].hist(np.abs(direct_derivative - analytical_derivative), alpha=0.5, label='Direct', bins=20)
   axes[1,1].hist(np.abs(smooth_derivative - analytical_derivative), alpha=0.5, label='LOWESS', bins=20)
   axes[1,1].hist(np.abs(spline_derivative - analytical_derivative), alpha=0.5, label='Spline', bins=20)
   axes[1,1].legend()
   axes[1,1].set_title('Error Distribution')
   axes[1,1].set_xlabel('Absolute Error')
   
   plt.tight_layout()
   plt.show()

Example 4: Integration and Roundtrip Accuracy
---------------------------------------------

Testing integration accuracy by doing derivative → integral roundtrips:

.. code-block:: python

   import numpy as np
   from pydelt.derivatives import lla
   from pydelt.integrals import integrate_derivative
   
   # Original function: f(x) = x^3 - 2x^2 + x + 5
   x = np.linspace(0, 5, 100)
   original_function = x**3 - 2*x**2 + x + 5
   
   # Analytical derivative: f'(x) = 3x^2 - 4x + 1
   analytical_derivative = 3*x**2 - 4*x + 1
   
   # Step 1: Calculate derivative numerically
   result = lla(x.tolist(), original_function.tolist(), window_size=5)
   numerical_derivative = result[0]
   
   # Step 2: Integrate the derivative back
   integrated_function, integration_error = integrate_derivative(
       x, numerical_derivative, 
       initial_value=original_function[0]
   )
   
   # Step 3: Calculate errors
   derivative_error = np.mean(np.abs(numerical_derivative - analytical_derivative))
   roundtrip_error = np.mean(np.abs(integrated_function - original_function))
   
   print(f"Derivative calculation error: {derivative_error:.6f}")
   print(f"Integration roundtrip error: {roundtrip_error:.6f}")
   print(f"Estimated integration error: {integration_error:.6f}")
   
   # Plotting
   import matplotlib.pyplot as plt
   fig, axes = plt.subplots(2, 2, figsize=(12, 10))
   
   # Original function
   axes[0,0].plot(x, original_function, 'b-', linewidth=2, label='Original f(x)')
   axes[0,0].plot(x, integrated_function, 'r--', linewidth=2, label='Integrated f(x)')
   axes[0,0].legend()
   axes[0,0].set_title('Function Comparison')
   axes[0,0].set_ylabel('f(x)')
   
   # Derivatives
   axes[0,1].plot(x, analytical_derivative, 'b-', linewidth=2, label='Analytical f\'(x)')
   axes[0,1].plot(x, numerical_derivative, 'r--', linewidth=2, label='Numerical f\'(x)')
   axes[0,1].legend()
   axes[0,1].set_title('Derivative Comparison')
   axes[0,1].set_ylabel('f\'(x)')
   
   # Function error
   axes[1,0].plot(x, np.abs(integrated_function - original_function), 'g-', linewidth=2)
   axes[1,0].set_title(f'Function Roundtrip Error (mean: {roundtrip_error:.4f})')
   axes[1,0].set_ylabel('|Error|')
   axes[1,0].set_xlabel('x')
   
   # Derivative error
   axes[1,1].plot(x, np.abs(numerical_derivative - analytical_derivative), 'orange', linewidth=2)
   axes[1,1].set_title(f'Derivative Error (mean: {derivative_error:.4f})')
   axes[1,1].set_ylabel('|Error|')
   axes[1,1].set_xlabel('x')
   
   plt.tight_layout()
   plt.show()

Example 5: Multivariate Time Series
-----------------------------------

Working with multi-dimensional data:

.. code-block:: python

   import numpy as np
   from pydelt.derivatives import lla
   
   # Create 3D trajectory data (e.g., particle motion)
   t = np.linspace(0, 4*np.pi, 200)
   
   # Parametric equations for a 3D spiral
   x = np.cos(t) * np.exp(-0.1*t)
   y = np.sin(t) * np.exp(-0.1*t)  
   z = 0.1 * t
   
   # Combine into multivariate signal
   trajectory = np.column_stack([x, y, z])
   
   # Calculate velocity (derivative of position) using multivariate support
   velocity_result = lla(t.tolist(), trajectory.tolist(), window_size=5)
   velocity = velocity_result[0]  # Shape: (N, 3) for 3D trajectory
   
   # Calculate speed (magnitude of velocity)
   speed = np.sqrt(np.sum(velocity**2, axis=1))
   
   # Analytical derivatives for comparison
   dx_dt = -np.sin(t)*np.exp(-0.1*t) - 0.1*np.cos(t)*np.exp(-0.1*t)
   dy_dt = np.cos(t)*np.exp(-0.1*t) - 0.1*np.sin(t)*np.exp(-0.1*t)
   dz_dt = 0.1 * np.ones_like(t)
   
   analytical_velocity = np.column_stack([dx_dt, dy_dt, dz_dt])
   analytical_speed = np.sqrt(np.sum(analytical_velocity**2, axis=1))
   
   # Calculate errors
   velocity_error = np.mean(np.abs(velocity - analytical_velocity))
   speed_error = np.mean(np.abs(speed - analytical_speed))
   
   print(f"Velocity calculation error: {velocity_error:.6f}")
   print(f"Speed calculation error: {speed_error:.6f}")
   
   # 3D plotting
   import matplotlib.pyplot as plt
   from mpl_toolkits.mplot3d import Axes3D
   
   fig = plt.figure(figsize=(15, 5))
   
   # 3D trajectory
   ax1 = fig.add_subplot(131, projection='3d')
   ax1.plot(x, y, z, 'b-', linewidth=2, label='Trajectory')
   ax1.set_xlabel('X')
   ax1.set_ylabel('Y')
   ax1.set_zlabel('Z')
   ax1.set_title('3D Trajectory')
   
   # Velocity components
   ax2 = fig.add_subplot(132)
   ax2.plot(t, velocity[:, 0], 'r-', label='vx (numerical)')
   ax2.plot(t, velocity[:, 1], 'g-', label='vy (numerical)')
   ax2.plot(t, velocity[:, 2], 'b-', label='vz (numerical)')
   ax2.plot(t, dx_dt, 'r--', alpha=0.7, label='vx (analytical)')
   ax2.plot(t, dy_dt, 'g--', alpha=0.7, label='vy (analytical)')
   ax2.plot(t, dz_dt, 'b--', alpha=0.7, label='vz (analytical)')
   ax2.legend()
   ax2.set_title('Velocity Components')
   ax2.set_xlabel('Time')
   ax2.set_ylabel('Velocity')
   
   # Speed comparison
   ax3 = fig.add_subplot(133)
   ax3.plot(t, speed, 'k-', linewidth=2, label='Numerical speed')
   ax3.plot(t, analytical_speed, 'r--', linewidth=2, label='Analytical speed')
   ax3.legend()
   ax3.set_title(f'Speed Comparison (error: {speed_error:.4f})')
   ax3.set_xlabel('Time')
   ax3.set_ylabel('Speed')
   
   plt.tight_layout()
   plt.show()

Example 6: Understanding Numerical Limitations
----------------------------------------------

Demonstrating critical point smoothing in multivariate derivatives:

.. code-block:: python

   import numpy as np
   from pydelt.multivariate import MultivariateDerivatives
   from pydelt.interpolation import SplineInterpolator
   
   # Function with known critical points: f(x,y) = (x-y)²
   x = np.linspace(-3, 3, 25)
   y = np.linspace(-3, 3, 25)
   X, Y = np.meshgrid(x, y)
   Z = (X - Y)**2  # Zero gradient along x=y line
   
   # Fit multivariate derivatives
   input_data = np.column_stack([X.flatten(), Y.flatten()])
   output_data = Z.flatten()
   
   mv = MultivariateDerivatives(SplineInterpolator, smoothing=0.1)
   mv.fit(input_data, output_data)
   
   # Test at critical points where gradient should be zero
   critical_points = np.array([[-3, -3], [3, 3]])  # Should have zero gradient
   
   gradient_func = mv.gradient()
   grad_critical = gradient_func(critical_points)
   
   print("Critical Points (should be ~[0, 0]):")
   for i, point in enumerate(critical_points):
       print(f"  Point {point}: Numerical gradient {grad_critical[i]:.3f}")
   
   print("\nKey Insight:")
   print("  Numerical methods smooth out zero gradients at critical points.")
   print("  This is a fundamental limitation of interpolation-based derivatives.")

**Key Takeaways:**

* Numerical methods cannot perfectly capture sharp mathematical features
* Critical points often show non-zero numerical gradients due to smoothing
* Always validate numerical results against analytical solutions when possible
* Consider neural network methods for exact derivatives at critical points

Example 7: Tensor Calculus
--------------------------

Directional derivative, divergence, curl, and strain/stress

.. code-block:: python

   import numpy as np
   from pydelt.multivariate import MultivariateDerivatives
   from pydelt.tensor_derivatives import TensorDerivatives
   from pydelt.interpolation import SplineInterpolator

   # 2D vector field: rotation F(x, y) = [-y, x]
   xs = ys = np.linspace(-2, 2, 25)
   X, Y = np.meshgrid(xs, ys)
   U = -Y
   V = X
   inputs = np.column_stack([X.flatten(), Y.flatten()])
   outputs = np.column_stack([U.flatten(), V.flatten()])

   mv = MultivariateDerivatives(SplineInterpolator, smoothing=0.0)
   mv.fit(inputs, outputs)

   q = np.array([[1.0, 0.5]])
   e = np.array([0.0, 1.0])
   dF_e = mv.directional_derivative(e)(q)  # shape (1, 2)
   divF = mv.divergence()(q)               # ≈ 0
   curlF = mv.curl()(q)                    # ≈ 2
   print("Directional derivative along e:", dF_e)
   print("div(F):", float(divF), " curl(F):", float(curlF))

   # 2D displacement field for strain/stress tensors
   U = 0.05 * X**2 - 0.02 * Y
   V = 0.10 * X * Y
   displacement = np.column_stack([U.flatten(), V.flatten()])

   td = TensorDerivatives(SplineInterpolator, smoothing=0.0)
   td.fit(inputs, displacement)

   strain = td.strain_tensor()(inputs)  # (N, 2, 2)
   stress = td.stress_tensor(lambda_param=1.0, mu_param=0.5)(inputs)  # (N, 2, 2)

   # Correct component extraction using three indices
   exx = strain[:, 0, 0].reshape(X.shape)
   eyy = strain[:, 1, 1].reshape(X.shape)
   exy = strain[:, 0, 1].reshape(X.shape)
   sxy = stress[:, 0, 1].reshape(X.shape)
   print("mean |ε_xy|:", float(np.mean(np.abs(exy))))
   print("mean |σ_xy|:", float(np.mean(np.abs(sxy))))

Example 8: Stochastic Derivatives
----------------------------------

Computing derivatives with stochastic corrections for financial applications:

.. code-block:: python

   import numpy as np
   from pydelt.interpolation import SplineInterpolator
   
   # Simulate stock price following geometric Brownian motion
   # dS_t = μS_t dt + σS_t dW_t
   np.random.seed(42)
   T = 1.0          # 1 year
   N = 252          # Daily observations
   dt = T / N
   mu = 0.05        # Expected return (5%)
   sigma = 0.2      # Volatility (20%)
   S0 = 100         # Initial stock price
   
   # Generate price path
   t = np.linspace(0, T, N+1)
   W = np.random.randn(N+1).cumsum() * np.sqrt(dt)  # Brownian motion
   S = S0 * np.exp((mu - 0.5*sigma**2)*t + sigma*W)  # GBM solution
   
   # Fit interpolator
   spline = SplineInterpolator(smoothing=0.01)
   spline.fit(t, S)
   
   # Regular derivative (deterministic)
   regular_deriv_func = spline.differentiate(order=1)
   regular_derivatives = regular_deriv_func(t)
   
   # Stochastic derivative with lognormal correction (Itô)
   spline.set_stochastic_link('lognormal', sigma=sigma, method='ito')
   stochastic_deriv_func = spline.differentiate(order=1)
   stochastic_derivatives = stochastic_deriv_func(t)
   
   # Compare corrections
   correction = stochastic_derivatives - regular_derivatives
   print(f"Mean stochastic correction: {np.mean(correction):.4f}")
   print(f"Std of correction: {np.std(correction):.4f}")
   
   # Option Greeks approximation
   # Delta ≈ ∂S/∂t (rate of change)
   delta_approx = stochastic_derivatives[-1]
   print(f"Approximate Delta at maturity: {delta_approx:.4f}")
   
   # Compare different stochastic methods
   spline.set_stochastic_link('lognormal', sigma=sigma, method='stratonovich')
   stratonovich_deriv = spline.differentiate(order=1)(t)
   
   method_diff = stratonovich_deriv - stochastic_derivatives
   print(f"Itô vs Stratonovich difference: {np.mean(np.abs(method_diff)):.6f}")


Example 7: Window Functions for Segmented Data
----------------------------------------------

Window functions are useful when working with data that has natural boundaries,
such as sensor data with measurement gaps or financial data with trading sessions.
This example shows how to apply window functions to handle segmented time series.

.. code-block:: python

   import numpy as np
   from pydelt.interpolation import SplineInterpolator
   
   # Simulate sensor data with measurement sessions separated by gaps
   np.random.seed(42)
   
   # Session 1: 0-60 minutes
   session1_time = np.linspace(0, 60, 30)
   session1_signal = 100 + 10 * np.sin(session1_time / 10) + np.random.randn(30) * 0.5
   
   # Session 2: 100-160 minutes (40-minute gap, ~15 value drop)
   session2_time = np.linspace(100, 160, 30)
   session2_signal = 85 + 10 * np.sin(session2_time / 10) + np.random.randn(30) * 0.5
   
   # Session 3: 200-260 minutes (40-minute gap, ~15 value drop)
   session3_time = np.linspace(200, 260, 30)
   session3_signal = 70 + 10 * np.sin(session3_time / 10) + np.random.randn(30) * 0.5
   
   # Combine all sessions
   time = np.concatenate([session1_time, session2_time, session3_time])
   signal = np.concatenate([session1_signal, session2_signal, session3_signal])
   
   # Create a custom window generator that detects session boundaries
   def create_session_window(time_gap_threshold=30, value_drop_threshold=10):
       """
       Window generator that tapers at session boundaries.
       Detects boundaries based on time gaps and value drops.
       """
       def window_func(n):
           # Create Tukey window with 10% taper at edges
           weights = np.ones(n)
           taper_length = max(1, n // 10)
           
           # Taper at beginning
           for i in range(taper_length):
               weights[i] = 0.5 * (1 - np.cos(np.pi * i / taper_length))
           
           # Taper at end
           for i in range(taper_length):
               weights[-(i+1)] = 0.5 * (1 - np.cos(np.pi * i / taper_length))
           
           return weights
       
       return window_func
   
   # Fit interpolator with window function
   window_gen = create_session_window(time_gap_threshold=30, value_drop_threshold=10)
   interp = SplineInterpolator(smoothing=1.0)
   interp.fit(time, signal, window_func=window_gen)
   
   # Compute derivatives with and without normalization
   deriv_func = interp.differentiate(order=1, normalize_by_observations=False)
   derivatives = deriv_func(time)
   
   deriv_func_norm = interp.differentiate(order=1, normalize_by_observations=True)
   derivatives_normalized = deriv_func_norm(time)
   
   print(f"Window function applied: {interp.window_func is not None}")
   print(f"Number of observations: {interp.n_observations}")
   print(f"Derivative range: [{derivatives.min():.3f}, {derivatives.max():.3f}]")
   print(f"Normalized derivative range: [{derivatives_normalized.min():.6f}, "
         f"{derivatives_normalized.max():.6f}]")
   
   # Compare with standard numpy windows
   interp_hanning = SplineInterpolator(smoothing=1.0)
   interp_hanning.fit(time, signal, window_func=np.hanning)
   
   deriv_hanning = interp_hanning.differentiate(order=1)(time)
   print(f"\\nHanning window derivative range: [{deriv_hanning.min():.3f}, "
         f"{deriv_hanning.max():.3f}]")

**Key Points:**

- Window functions modify the signal before interpolation, reducing edge effects
- Custom window generators can detect session boundaries based on time gaps and value drops
- The ``normalize_by_observations`` parameter scales derivatives by 1/N when a window was applied
- Standard NumPy windows (hanning, hamming, blackman, bartlett) are supported
- Window functions are particularly useful for:
  
  - Sensor data with measurement gaps
  - Financial data with trading sessions
  - Any time series with natural segmentation
  - Reducing artifacts at data boundaries
