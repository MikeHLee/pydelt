Quick Start Guide
=================

This guide introduces pydelt's progressive feature set, from basic interpolation to advanced stochastic computing. Follow the examples below to get started quickly.

🚀 **Universal API Pattern**
---------------------------

All pydelt interpolators follow the same consistent interface:

.. code-block:: python

   # Universal pattern for all methods
   interpolator = InterpolatorClass(**parameters)
   interpolator.fit(input_data, output_data)
   derivative_func = interpolator.differentiate(order=1, mask=None)
   derivatives = derivative_func(evaluation_points)

**Level 1: Basic Interpolation**
-------------------------------

Start with classical interpolation methods:

.. code-block:: python

   import numpy as np
   from pydelt.interpolation import SplineInterpolator, LlaInterpolator
   
   # Create sample data: f(t) = sin(t)
   time = np.linspace(0, 2*np.pi, 50)
   signal = np.sin(time) + 0.1 * np.random.randn(len(time))  # Add noise
   
   # Method 1: Spline interpolation (best for smooth data)
   spline = SplineInterpolator(smoothing=0.1)
   spline.fit(time, signal)
   spline_deriv_func = spline.differentiate(order=1)
   spline_derivatives = spline_deriv_func(time)
   
   # Method 2: Local Linear Approximation (efficient, robust)
   lla = LlaInterpolator(window_size=7)
   lla.fit(time, signal)
   lla_deriv_func = lla.differentiate(order=1)
   lla_derivatives = lla_deriv_func(time)
   
   # Compare with analytical derivative
   analytical = np.cos(time)
   print(f"Spline Error: {np.sqrt(np.mean((spline_derivatives - analytical)**2)):.4f}")
   print(f"LLA Error: {np.sqrt(np.mean((lla_derivatives - analytical)**2)):.4f}")

**Level 2: Neural Networks & Automatic Differentiation**
--------------------------------------------------------

For complex nonlinear functions with exact derivatives:

.. code-block:: python

   from pydelt.interpolation import NeuralNetworkInterpolator
   
   # Neural network with automatic differentiation
   nn_interp = NeuralNetworkInterpolator(
       hidden_layers=[128, 64, 32],
       activation='tanh',
       learning_rate=0.002,
       epochs=1000,
       backend='pytorch'
   )
   nn_interp.fit(time, signal)
   
   # Exact derivatives via backpropagation
   nn_deriv_func = nn_interp.differentiate(order=1)
   nn_derivatives = nn_deriv_func(time)
   
   print(f"Neural Network Error: {np.sqrt(np.mean((nn_derivatives - analytical)**2)):.4f}")

**Level 3: Multivariate Calculus**
----------------------------------

For functions of multiple variables:

.. code-block:: python

   from pydelt.multivariate import MultivariateDerivatives
   
   # 2D function: f(x,y) = x² + y²
   x = np.linspace(-2, 2, 20)
   y = np.linspace(-2, 2, 20)
   X, Y = np.meshgrid(x, y)
   Z = X**2 + Y**2
   
   # Prepare data
   input_data = np.column_stack([X.flatten(), Y.flatten()])
   output_data = Z.flatten()
   
   # Fit multivariate derivatives
   mv = MultivariateDerivatives(SplineInterpolator, smoothing=0.1)
   mv.fit(input_data, output_data)
   
   # Compute gradient: ∇f = [2x, 2y]
   gradient_func = mv.gradient()
   test_point = np.array([[1.0, 1.0]])
   gradient = gradient_func(test_point)
   print(f"Gradient at (1,1): {gradient[0]} (expected: [2, 2])")

**Level 4: Stochastic Computing** ⭐ *New Feature*
--------------------------------------------------

For probabilistic derivatives with uncertainty quantification:

.. code-block:: python

   # Stock price data with geometric Brownian motion
   np.random.seed(42)
   T = 1.0  # 1 year
   N = 252  # Daily data
   dt = T / N
   S0 = 100  # Initial price
   mu = 0.05  # Expected return
   sigma = 0.2  # Volatility
   
   # Generate stock price path
   t = np.linspace(0, T, N+1)
   W = np.random.randn(N+1).cumsum() * np.sqrt(dt)
   stock_prices = S0 * np.exp((mu - 0.5*sigma**2)*t + sigma*W)
   
   # Fit with stochastic link function
   stock_interp = SplineInterpolator(smoothing=0.01)
   stock_interp.fit(t, stock_prices)
   
   # Set log-normal stochastic link (appropriate for stock prices)
   stock_interp.set_stochastic_link('lognormal', sigma=sigma, method='ito')
   
   # Compute stochastic derivatives (includes Itô correction)
   stochastic_deriv_func = stock_interp.differentiate(order=1)
   stochastic_derivatives = stochastic_deriv_func(t)
   
   # Compare with regular derivatives
   stock_interp_regular = SplineInterpolator(smoothing=0.01)
   stock_interp_regular.fit(t, stock_prices)
   regular_deriv_func = stock_interp_regular.differentiate(order=1)
   regular_derivatives = regular_deriv_func(t)
   
   correction = np.mean(stochastic_derivatives - regular_derivatives)
   print(f"Stochastic correction: {correction:.2f}")
   print(f"Theoretical drift (μS): {mu * np.mean(stock_prices):.2f}")

⚠️ **Important Considerations**
------------------------------

**Numerical Limitations**: Interpolation-based methods can smooth critical points and sharp features. This affects:

- Optimization landscape analysis (finding exact minima/maxima)
- Bifurcation detection in dynamical systems
- Phase transition identification
- Sharp boundary detection

**Mitigation Strategies**:

1. **Increase data resolution** in critical regions
2. **Reduce smoothing parameters** (trade-off with noise sensitivity)
3. **Use neural networks** for exact automatic differentiation
4. **Validate against analytical solutions** when available
5. **Apply domain knowledge** for result interpretation

**Method Selection for Critical Applications**:

- **Exact derivatives needed**: ``NeuralNetworkInterpolator`` with automatic differentiation
- **Optimization problems**: Low smoothing + validation
- **Noisy data**: ``LowessInterpolator`` with appropriate ``frac`` parameter
- **Financial modeling**: Stochastic link functions for proper risk assessment

🎓 **Progressive Learning Path**
-------------------------------

Follow this sequence to master pydelt:

1. **Start with Basic Interpolation**: Master splines, LLA, and LOWESS for fundamental understanding
2. **Advance to Neural Networks**: Learn automatic differentiation for complex nonlinear functions  
3. **Explore Multivariate Calculus**: Compute gradients, Jacobians, and Hessians for optimization
4. **Master Stochastic Computing**: Apply probabilistic derivatives for uncertainty quantification

**Quick Method Selection Guide**:

- **Clean, smooth data**: ``SplineInterpolator``
- **Noisy data with outliers**: ``LowessInterpolator`` 
- **Complex nonlinear functions**: ``NeuralNetworkInterpolator``
- **Multiple variables**: ``MultivariateDerivatives``
- **Financial/risk modeling**: Add stochastic link functions
- **High precision needed**: ``GllaInterpolator``

🔗 **Next Steps**
----------------

Explore the progressive learning path:

- **Basic Interpolation**: Master fundamental methods and universal API
- **Neural Networks**: Learn automatic differentiation and deep learning integration
- **Multivariate Calculus**: Compute gradients, Jacobians, and tensor operations
- **Stochastic Computing**: Apply probabilistic derivatives for uncertainty quantification

Each section builds on the previous, providing a complete framework for numerical differentiation from basic applications to cutting-edge research.
