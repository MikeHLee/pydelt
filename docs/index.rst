.. pydelt documentation master file, created by
   sphinx-quickstart on Sun Jul 27 15:58:03 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

pydelt: Advanced Numerical Differentiation & Stochastic Computing
===============================================================

**pydelt** is a comprehensive Python library for numerical differentiation, interpolation, and stochastic computing. From basic polynomial derivatives to advanced stochastic calculus, pydelt provides a unified framework that scales from simple data analysis to cutting-edge research in dynamical systems, financial modeling, and scientific computing.

🎯 **Key Strengths & Applications**
----------------------------------

**Scientific Computing**
* **Dynamical Systems**: Reconstruct differential equations from noisy time series data
* **Phase Space Analysis**: Compute derivatives for embedding dimension analysis and chaos detection
* **Fluid Dynamics**: Analyze velocity fields, compute vorticity and strain tensors
* **Signal Processing**: Extract instantaneous frequency and phase derivatives from complex signals

**Financial & Risk Modeling**
* **Stochastic Derivatives**: Apply Itô's lemma and Stratonovich corrections to derivative computations
* **Option Pricing**: Compute Greeks (delta, gamma, theta) with stochastic link functions
* **Risk Management**: Model volatility surfaces and correlation structures with multivariate derivatives
* **Algorithmic Trading**: Real-time derivative computation for momentum and mean-reversion strategies

**Engineering & Control**
* **System Identification**: Extract governing equations from experimental data
* **Control Theory**: Design controllers using derivative information from system responses
* **Optimization**: Gradient-based methods with automatic differentiation support
* **Machine Learning**: Custom loss functions with higher-order derivative constraints

🚀 **Progressive Feature Set**
-----------------------------

**Level 1: Basic Interpolation & Derivatives**
* **Classical Methods**: Splines, polynomial fitting, local linear approximation
* **Universal API**: Consistent `.fit().differentiate()` interface across all methods
* **Higher-Order Support**: Analytical and numerical derivatives up to arbitrary order

**Level 2: Advanced Interpolation**
* **Robust Methods**: LOWESS, LOESS for noisy data and outlier resistance
* **Neural Networks**: Deep learning-based interpolation with automatic differentiation
* **Adaptive Methods**: LLA, GLLA with local bandwidth selection

**Level 3: Multivariate Calculus**
* **Vector Calculus**: Gradient (∇f), Jacobian (∂f/∂x), Hessian (∂²f/∂x²), Laplacian (∇²f)
* **Tensor Operations**: Full support for vector-valued functions and tensor calculus
* **Mixed Derivatives**: Cross-partial derivatives for multivariate analysis

**Level 4: Stochastic Computing** ⭐ *New Feature*
* **Stochastic Link Functions**: 6 probability distributions (Normal, Log-Normal, Gamma, Beta, Exponential, Poisson)
* **Stochastic Calculus**: Itô's lemma and Stratonovich integral corrections
* **Financial Applications**: Geometric Brownian motion, volatility modeling, option pricing
* **Risk Analysis**: Uncertainty propagation through derivative computations

📦 **Installation**
------------------

Install pydelt from PyPI:

.. code-block:: bash

   pip install pydelt

🔧 **Quick Start Examples**
---------------------------

**1. Universal Differentiation Interface**

.. code-block:: python

   import numpy as np
   from pydelt.interpolation import SplineInterpolator
   
   # Generate sample data: f(t) = sin(t)
   time = np.linspace(0, 2*np.pi, 100)
   signal = np.sin(time)
   
   # Universal API: fit interpolator and compute derivatives
   interpolator = SplineInterpolator(smoothing=0.1)
   interpolator.fit(time, signal)
   derivative_func = interpolator.differentiate(order=1)
   
   # Evaluate derivative at any points
   derivatives = derivative_func(time)
   print(f"Max error vs cos(t): {np.max(np.abs(derivatives - np.cos(time))):.4f}")

**2. Multivariate Calculus**

.. code-block:: python

   from pydelt.multivariate import MultivariateDerivatives
   
   # Generate 2D data: f(x,y) = x² + y²
   x = np.linspace(-2, 2, 50)
   y = np.linspace(-2, 2, 50)
   X, Y = np.meshgrid(x, y)
   Z = X**2 + Y**2
   
   # Fit multivariate derivatives
   input_data = np.column_stack([X.flatten(), Y.flatten()])
   output_data = Z.flatten()
   
   mv = MultivariateDerivatives(SplineInterpolator, smoothing=0.1)
   mv.fit(input_data, output_data)
   
   # Compute gradient: ∇f = [2x, 2y]
   gradient_func = mv.gradient()
   test_point = np.array([[1.0, 1.0]])
   gradient = gradient_func(test_point)
   print(f"Gradient at (1,1): {gradient[0]} (expected: [2, 2])")

**3. Stochastic Derivatives for Financial Modeling**

.. code-block:: python

   from pydelt.interpolation import SplineInterpolator
   
   # Stock price data following geometric Brownian motion
   time = np.linspace(0, 1, 252)  # 1 year of daily data
   stock_prices = 100 * np.exp(0.1*time + 0.2*np.random.randn(252).cumsum()/np.sqrt(252))
   
   # Fit interpolator with log-normal stochastic link
   interpolator = SplineInterpolator(smoothing=0.1)
   interpolator.fit(time, stock_prices)
   interpolator.set_stochastic_link('lognormal', sigma=0.2, method='ito')
   
   # Compute stochastic derivatives (automatically applies Itô correction)
   stochastic_deriv = interpolator.differentiate(order=1)
   derivatives = stochastic_deriv(time)
   
   print(f"Regular vs Stochastic derivative difference: {np.mean(np.abs(derivatives - regular_derivatives)):.2f}")

🌌 **Applications in Dynamical Systems**
-----------------------------------------

**pydelt** excels in analyzing dynamical systems and differential equations from data:

* **System Identification**: Reconstruct differential equations from time series observations
* **Phase Space Reconstruction**: Compute derivatives for embedding dimension analysis
* **Stability Analysis**: Calculate Jacobians and eigenvalues for equilibrium point classification
* **Bifurcation Analysis**: Track parameter-dependent changes in system behavior
* **Control Theory**: Design controllers using derivative information from system responses
* **Fluid Dynamics**: Analyze velocity fields and compute vorticity, divergence, and strain tensors
* **Continuum Mechanics**: Calculate stress and strain derivatives for material property estimation
* **Signal Processing**: Extract instantaneous frequency and phase derivatives from complex signals

**Time Series as a Special Case**: Traditional time series derivative analysis is just one application of our broader dynamical systems framework. The universal differentiation interface seamlessly handles both temporal data and general multivariate functions.

📚 **Documentation Contents**
----------------------------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started:

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: Progressive Learning Path:

   basic_interpolation
   neural_networks
   multivariate_calculus
   stochastic_computing
   
.. toctree::
   :maxdepth: 2
   :caption: Reference:

   examples
   api
   faq
   changelog

🔗 **Links**
-----------

* **PyPI**: https://pypi.org/project/pydelt/
* **Source Code**: https://github.com/MikeHLee/pydelt
* **Issues**: https://github.com/MikeHLee/pydelt/issues

📋 **Indices and Tables**
------------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
