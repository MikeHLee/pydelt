Neural Networks & Automatic Differentiation
============================================

Neural networks provide powerful interpolation capabilities with exact automatic differentiation. Unlike traditional methods that approximate derivatives numerically, neural networks can compute exact gradients through backpropagation, making them ideal for optimization and machine learning applications.

🧠 **Core Concepts**
-------------------

**Automatic Differentiation (AutoDiff)** computes derivatives by applying the chain rule systematically to elementary operations. This provides machine-precision accuracy for derivatives, unlike numerical approximation methods.

**Universal Function Approximation**: Neural networks can approximate any continuous function to arbitrary precision given sufficient capacity, making them extremely versatile interpolators.

**Backpropagation**: The algorithm that efficiently computes gradients by propagating error signals backward through the network.

🔧 **Neural Network Interpolator**
---------------------------------

The ``NeuralNetworkInterpolator`` combines the universal pydelt API with deep learning backends:

.. code-block:: python

   from pydelt.interpolation import NeuralNetworkInterpolator
   import numpy as np
   
   # Create neural network interpolator
   nn_interp = NeuralNetworkInterpolator(
       hidden_layers=[64, 32, 16],  # Network architecture
       activation='relu',           # Activation function
       learning_rate=0.001,         # Optimizer learning rate
       epochs=1000,                 # Training iterations
       backend='pytorch'            # 'pytorch' or 'tensorflow'
   )

**Key Parameters**:
- ``hidden_layers``: List of hidden layer sizes [64, 32] creates 2 hidden layers
- ``activation``: 'relu', 'tanh', 'sigmoid', 'swish', 'gelu'
- ``learning_rate``: Adam optimizer learning rate (0.001-0.01 typical)
- ``epochs``: Training iterations (500-5000 depending on complexity)
- ``backend``: 'pytorch' (default) or 'tensorflow'

🎯 **Example 1: Nonlinear Function Approximation**
-------------------------------------------------

**Classic Example: Runge Function**

The Runge function is notoriously difficult for polynomial interpolation but neural networks handle it well:

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from pydelt.interpolation import NeuralNetworkInterpolator, SplineInterpolator
   
   # Runge function: f(x) = 1 / (1 + 25x²)
   def runge_function(x):
       return 1 / (1 + 25 * x**2)
   
   def runge_derivative(x):
       return -50 * x / (1 + 25 * x**2)**2
   
   # Training data (sparse sampling)
   x_train = np.linspace(-1, 1, 15)
   y_train = runge_function(x_train)
   
   # Neural network interpolator
   nn_interp = NeuralNetworkInterpolator(
       hidden_layers=[128, 64, 32],
       activation='tanh',  # Good for smooth functions
       learning_rate=0.005,
       epochs=2000
   )
   nn_interp.fit(x_train, y_train)
   
   # Compare with spline
   spline = SplineInterpolator(smoothing=0.0)
   spline.fit(x_train, y_train)
   
   # Evaluation points
   x_test = np.linspace(-1, 1, 200)
   y_true = runge_function(x_test)
   dy_true = runge_derivative(x_test)
   
   # Predictions
   y_nn = nn_interp.predict(x_test)
   y_spline = spline.predict(x_test)
   
   # Derivatives (automatic vs numerical)
   nn_deriv_func = nn_interp.differentiate(order=1)
   spline_deriv_func = spline.differentiate(order=1)
   
   dy_nn = nn_deriv_func(x_test)
   dy_spline = spline_deriv_func(x_test)
   
   # Error analysis
   nn_func_error = np.sqrt(np.mean((y_nn - y_true)**2))
   nn_deriv_error = np.sqrt(np.mean((dy_nn - dy_true)**2))
   spline_func_error = np.sqrt(np.mean((y_spline - y_true)**2))
   spline_deriv_error = np.sqrt(np.mean((dy_spline - dy_true)**2))
   
   print("Function Approximation Errors:")
   print(f"Neural Network: {nn_func_error:.6f}")
   print(f"Spline:         {spline_func_error:.6f}")
   print("\nDerivative Errors:")
   print(f"Neural Network: {nn_deriv_error:.6f}")
   print(f"Spline:         {spline_deriv_error:.6f}")

**Expected Results**: Neural networks typically achieve 10-100x better accuracy than splines for the Runge function, especially near the boundaries where polynomial methods struggle.

🌊 **Example 2: Fluid Dynamics - Velocity Field**
------------------------------------------------

**Application**: Reconstructing velocity fields from particle tracking data in fluid mechanics.

.. code-block:: python

   # Simulate 2D fluid flow around a cylinder (potential flow)
   def potential_flow_velocity(x, y, U_inf=1.0, R=0.5):
       """Velocity field around a cylinder in cross-flow"""
       r_sq = x**2 + y**2
       # Avoid singularity at origin
       r_sq = np.maximum(r_sq, 1e-10)
       
       # Velocity components for flow around cylinder
       u = U_inf * (1 - R**2 * (x**2 - y**2) / r_sq**2)
       v = U_inf * (-R**2 * 2 * x * y / r_sq**2)
       return u, v
   
   # Generate training data (sparse particle tracking)
   np.random.seed(42)
   n_particles = 200
   x_particles = np.random.uniform(-2, 2, n_particles)
   y_particles = np.random.uniform(-2, 2, n_particles)
   
   # Remove particles inside cylinder
   mask = (x_particles**2 + y_particles**2) > 0.6**2
   x_particles = x_particles[mask]
   y_particles = y_particles[mask]
   
   # Get velocity components
   u_true, v_true = potential_flow_velocity(x_particles, y_particles)
   
   # Add measurement noise
   u_measured = u_true + 0.05 * np.random.randn(len(u_true))
   v_measured = v_true + 0.05 * np.random.randn(len(v_true))
   
   # Prepare input data (x,y positions) and output data (u,v velocities)
   input_data = np.column_stack([x_particles, y_particles])
   output_data = np.column_stack([u_measured, v_measured])
   
   # Neural network for vector-valued function
   nn_flow = NeuralNetworkInterpolator(
       hidden_layers=[128, 128, 64],
       activation='swish',  # Good for fluid dynamics
       learning_rate=0.002,
       epochs=3000
   )
   nn_flow.fit(input_data, output_data)
   
   # Create evaluation grid
   x_grid = np.linspace(-2, 2, 50)
   y_grid = np.linspace(-2, 2, 50)
   X, Y = np.meshgrid(x_grid, y_grid)
   
   # Remove points inside cylinder
   mask_grid = (X**2 + Y**2) > 0.6**2
   x_eval = X[mask_grid]
   y_eval = Y[mask_grid]
   eval_points = np.column_stack([x_eval, y_eval])
   
   # Predict velocity field
   velocity_pred = nn_flow.predict(eval_points)
   u_pred = velocity_pred[:, 0]
   v_pred = velocity_pred[:, 1]
   
   # Compute derivatives for vorticity analysis
   # ∂u/∂y and ∂v/∂x for vorticity ω = ∂v/∂x - ∂u/∂y
   deriv_func = nn_flow.differentiate(order=1)
   derivatives = deriv_func(eval_points)
   
   print(f"Reconstructed {len(eval_points)} velocity vectors from {len(x_particles)} measurements")
   print(f"Velocity field ready for vorticity and strain rate analysis")

📊 **Example 3: Time Series with Complex Dynamics**
--------------------------------------------------

**Application**: Chaotic time series analysis (Lorenz attractor).

.. code-block:: python

   from scipy.integrate import odeint
   
   # Lorenz system parameters
   def lorenz_system(state, t, sigma=10, rho=28, beta=8/3):
       x, y, z = state
       dxdt = sigma * (y - x)
       dydt = x * (rho - z) - y
       dzdt = x * y - beta * z
       return [dxdt, dydt, dzdt]
   
   # Generate Lorenz attractor data
   t = np.linspace(0, 20, 2000)
   initial_state = [1.0, 1.0, 1.0]
   trajectory = odeint(lorenz_system, initial_state, t)
   
   # Extract x-component time series
   x_series = trajectory[:, 0]
   
   # Subsample for training (simulate sparse measurements)
   indices = np.arange(0, len(t), 10)  # Every 10th point
   t_train = t[indices]
   x_train = x_series[indices]
   
   # Add measurement noise
   x_noisy = x_train + 0.5 * np.random.randn(len(x_train))
   
   # Neural network interpolator
   nn_chaos = NeuralNetworkInterpolator(
       hidden_layers=[256, 128, 64, 32],  # Deep network for complex dynamics
       activation='gelu',  # Good for chaotic systems
       learning_rate=0.001,
       epochs=4000
   )
   nn_chaos.fit(t_train, x_noisy)
   
   # Predict full time series
   x_pred = nn_chaos.predict(t)
   
   # Compute instantaneous rate of change
   rate_func = nn_chaos.differentiate(order=1)
   dx_dt_pred = rate_func(t)
   
   # True derivative from Lorenz equations
   dx_dt_true = 10 * (trajectory[:, 1] - trajectory[:, 0])
   
   # Analysis
   reconstruction_error = np.sqrt(np.mean((x_pred - x_series)**2))
   derivative_error = np.sqrt(np.mean((dx_dt_pred - dx_dt_true)**2))
   
   print(f"Time series reconstruction error: {reconstruction_error:.3f}")
   print(f"Derivative reconstruction error: {derivative_error:.3f}")
   
   # Phase space reconstruction quality
   correlation = np.corrcoef(x_pred, x_series)[0, 1]
   print(f"Correlation with true attractor: {correlation:.4f}")

⚡ **Advantages of Neural Networks**
----------------------------------

**1. Exact Derivatives**
- Automatic differentiation provides machine-precision gradients
- No numerical approximation errors
- Consistent accuracy across all derivative orders

**2. Universal Approximation**
- Can represent any continuous function
- Handles highly nonlinear relationships
- Scales to high-dimensional problems

**3. Noise Robustness**
- Implicit regularization through architecture
- Dropout and batch normalization for stability
- Learns underlying patterns despite measurement noise

**4. Scalability**
- GPU acceleration for large datasets
- Batch processing for efficiency
- Parallel computation of derivatives

🔧 **Advanced Configuration**
----------------------------

**Custom Architecture Design**:

.. code-block:: python

   # Deep network for complex functions
   complex_nn = NeuralNetworkInterpolator(
       hidden_layers=[512, 256, 128, 64, 32],
       activation='swish',
       learning_rate=0.0005,
       epochs=5000,
       batch_size=64,
       dropout_rate=0.1
   )
   
   # Wide network for high-frequency components
   wide_nn = NeuralNetworkInterpolator(
       hidden_layers=[1024, 1024],
       activation='relu',
       learning_rate=0.002,
       epochs=2000
   )

**Training Monitoring**:

.. code-block:: python

   # Enable training progress tracking
   nn_interp = NeuralNetworkInterpolator(
       hidden_layers=[128, 64],
       epochs=1000,
       verbose=True,        # Print training progress
       early_stopping=True, # Stop when validation loss plateaus
       validation_split=0.2 # Use 20% of data for validation
   )

**Backend Selection**:

.. code-block:: python

   # PyTorch backend (default, recommended)
   nn_pytorch = NeuralNetworkInterpolator(backend='pytorch')
   
   # TensorFlow backend
   nn_tensorflow = NeuralNetworkInterpolator(backend='tensorflow')

⚠️ **Limitations & Considerations**
----------------------------------

**Computational Cost**:
- Training time scales with network size and data complexity
- GPU recommended for large networks (>1000 parameters)
- Memory usage grows with batch size and network depth

**Hyperparameter Sensitivity**:
- Learning rate requires tuning (too high: instability, too low: slow convergence)
- Architecture choice affects approximation quality
- Overfitting possible with insufficient data

**Reproducibility**:
- Random initialization affects results
- Set random seeds for reproducible results:

.. code-block:: python

   import torch
   import numpy as np
   
   # Set seeds for reproducibility
   torch.manual_seed(42)
   np.random.seed(42)
   
   nn_interp = NeuralNetworkInterpolator(...)

🎓 **Best Practices**
--------------------

**Architecture Guidelines**:
1. **Start simple**: Begin with [64, 32] hidden layers
2. **Go deeper for complexity**: Add layers for highly nonlinear functions
3. **Go wider for detail**: Increase layer sizes for high-frequency components
4. **Use appropriate activations**: 'relu' (general), 'tanh' (smooth), 'swish' (modern)

**Training Tips**:
1. **Monitor convergence**: Use validation split to track overfitting
2. **Adjust learning rate**: Decrease if training is unstable
3. **Early stopping**: Prevent overfitting with patience parameter
4. **Data normalization**: Scale inputs to [-1, 1] or [0, 1] range

**Derivative Accuracy**:
- Neural network derivatives are exact (no approximation error)
- Accuracy depends on function approximation quality
- Higher-order derivatives may amplify approximation errors

🔗 **Next Steps**
----------------

Neural networks excel at univariate and simple multivariate problems. For advanced multivariate calculus operations, continue to:

- **Multivariate Calculus**: Gradients, Jacobians, and Hessians for vector-valued functions
- **Stochastic Computing**: Probabilistic neural networks with uncertainty quantification

The automatic differentiation capabilities of neural networks become especially powerful when combined with multivariate operations and stochastic link functions.
