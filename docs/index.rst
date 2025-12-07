.. pydelt documentation master file, created by
   sphinx-quickstart on Sun Jul 27 15:58:03 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PyDelt: Advanced Numerical Function Interpolation & Differentiation
====================================================================

**PyDelt** transforms raw data into mathematical insights through advanced numerical interpolation and differentiation. Whether you're analyzing experimental measurements, financial time series, or complex dynamical systems, PyDelt provides the tools to extract derivatives, gradients, and higher-order mathematical properties with precision and reliability.

🎯 **What Makes PyDelt Special**
---------------------------------

**Universal Interface**
All interpolation methods share the same `.fit().differentiate()` API, making it easy to switch between techniques and compare results without rewriting code.

**From Simple to Sophisticated**
Start with basic spline interpolation and scale up to neural networks with automatic differentiation - all within the same framework.

**Real-World Ready**
Built-in noise handling, robust error estimation, and comprehensive validation ensure your results are reliable even with imperfect data.

**Applications Across Domains**

• **Scientific Computing**: Reconstruct differential equations from experimental data, analyze phase spaces, compute fluid dynamics properties
• **Financial Modeling**: Calculate option Greeks, model volatility surfaces, apply stochastic calculus corrections
• **Engineering**: System identification, control design, optimization with gradient information
• **Data Science**: Feature engineering, signal processing, time series analysis with mathematical rigor

🚀 **Progressive Feature Set**
-------------------------------

**Level 1: Foundation Methods**

• **Spline Interpolation**: Smooth curves through your data with analytical derivatives
• **Local Linear Approximation (LLA)**: Robust sliding-window approach for noisy data
• **Functional Data Analysis (FDA)**: Sophisticated smoothing with optimal parameter selection

**Level 2: Advanced Techniques**

• **LOWESS/LOESS**: Non-parametric methods resistant to outliers and varying noise levels
• **Neural Networks**: Deep learning with automatic differentiation for complex patterns
• **Generalized LLA (GLLA)**: Higher-order local approximations for enhanced accuracy

**Level 3: Multivariate Calculus**

• **Gradient Computation**: ∇f for scalar functions of multiple variables
• **Jacobian Matrices**: ∂f/∂x for vector-valued functions
• **Hessian Analysis**: Second-order derivatives for optimization and stability
• **Laplacian Operations**: ∇²f for diffusion and field analysis

**Level 4: Stochastic Extensions** ⭐

• **Stochastic Link Functions**: Transform derivatives through probability distributions
• **Itô and Stratonovich Corrections**: Proper stochastic calculus for financial modeling
• **Risk Propagation**: Uncertainty quantification through derivative computations

📦 **Installation**
--------------------

Install pydelt from PyPI:

.. code-block:: bash

   pip install pydelt

🚀 **Quick Start: See PyDelt in Action**
-----------------------------------------

**The Universal Interface**

Every interpolation method in PyDelt follows the same simple pattern:

.. code-block:: python

   import numpy as np
   from pydelt.interpolation import SplineInterpolator
   
   # Your data: noisy measurements of f(t) = sin(t)
   time = np.linspace(0, 2*np.pi, 100)
   signal = np.sin(time) + 0.1 * np.random.randn(100)
   
   # Three-step process: create, fit, differentiate
   interpolator = SplineInterpolator(smoothing=0.1)
   interpolator.fit(time, signal)
   derivative_func = interpolator.differentiate(order=1)
   
   # Evaluate derivatives anywhere you need them
   new_points = np.linspace(0, 2*np.pi, 50)
   derivatives = derivative_func(new_points)
   
   # Compare with analytical result: d/dt[sin(t)] = cos(t)
   analytical = np.cos(new_points)
   error = np.mean(np.abs(derivatives - analytical))
   print(f"Average error: {error:.4f}")

.. raw:: html

   <iframe src="_static/images/universal_api_demo.html" width="100%" height="600px" frameborder="0"></iframe>

**Beyond 1D: Multivariate Functions**

PyDelt extends naturally to functions of multiple variables:

.. code-block:: python

   from pydelt.multivariate import MultivariateDerivatives
   
   # 2D surface: f(x,y) = sin(x)cos(y) + 0.1xy
   x = np.linspace(-3, 3, 30)
   y = np.linspace(-3, 3, 30)
   X, Y = np.meshgrid(x, y)
   Z = np.sin(X) * np.cos(Y) + 0.1 * X * Y
   
   # Prepare data for multivariate analysis
   input_data = np.column_stack([X.flatten(), Y.flatten()])
   output_data = Z.flatten()
   
   # Same universal interface, now for gradients
   mv = MultivariateDerivatives(SplineInterpolator, smoothing=0.1)
   mv.fit(input_data, output_data)
   
   # Compute gradient field: ∇f = [∂f/∂x, ∂f/∂y]
   gradient_func = mv.gradient()
   test_points = np.array([[0.0, 0.0], [1.0, 1.0]])
   gradients = gradient_func(test_points)
   
   print(f"Gradient at origin: {gradients[0]}")
   print(f"Gradient at (1,1): {gradients[1]}")

.. raw:: html

   <iframe src="_static/images/multivariate_surface.html" width="100%" height="700px" frameborder="0"></iframe>

**Method Comparison: Choose the Right Tool**

Different methods excel in different scenarios:

.. code-block:: python

   # Complex function with multiple scales and noise
   x = np.linspace(0, 4*np.pi, 80)
   y_true = np.sin(x) * np.exp(-x/8) + 0.3*np.sin(5*x)
   y_noisy = y_true + 0.1 * np.random.randn(len(x))
   
   # Compare different approaches
   methods = {
       'Spline (smooth)': SplineInterpolator(smoothing=1.0),
       'Spline (detailed)': SplineInterpolator(smoothing=0.1),
       'LLA (adaptive)': LlaInterpolator(window_size=5),
       'LLA (stable)': LlaInterpolator(window_size=15)
   }
   
   results = {}
   for name, interpolator in methods.items():
       interpolator.fit(x, y_noisy)
       # Each method automatically handles the complexity differently
       derivative_func = interpolator.differentiate(order=1)
       results[name] = derivative_func(x)
   
   # PyDelt makes it easy to compare and choose
   print("Method comparison complete - see visualization below")

.. raw:: html

   <iframe src="_static/images/method_comparison.html" width="100%" height="600px" frameborder="0"></iframe>

🌍 **Real-World Impact**
-------------------------

**Scientific Discovery**
Extract governing equations from experimental data, analyze phase spaces in nonlinear dynamics, compute fluid properties from velocity measurements.

**Financial Engineering**
Calculate option Greeks with proper stochastic corrections, model volatility surfaces, implement risk management strategies with mathematical precision.

**Engineering Design**
Identify system dynamics from sensor data, design controllers using derivative feedback, optimize processes with gradient-based methods.

**Data Science Excellence**
Transform time series analysis with mathematical rigor, engineer features with derivative information, validate models through mathematical consistency.

**Why PyDelt Matters**

Traditional numerical differentiation is notoriously unstable - small changes in data can cause large changes in derivatives. PyDelt solves this through:

• **Smart smoothing** that preserves important features while reducing noise
• **Multiple methods** so you can choose the best approach for your data
• **Robust validation** to ensure your results are mathematically sound
• **Unified interface** that makes comparison and validation straightforward

📚 **Learn PyDelt**
--------------------

.. toctree::
   :maxdepth: 2
   :caption: Start Here:

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: Mathematical Foundations:

   theory_index
   theory

.. toctree::
   :maxdepth: 2
   :caption: Master the Methods:

   basic_interpolation
   neural_networks
   multivariate_calculus
   stochastic_computing
   
.. toctree::
   :maxdepth: 2
   :caption: Reference & Help:

   examples
   visual_examples
   feature_comparison
   api
   faq
   changelog

🔗 **Links**
-------------

* **PyPI**: https://pypi.org/project/pydelt/
* **Source Code**: https://github.com/MikeHLee/pydelt
* **Issues**: https://github.com/MikeHLee/pydelt/issues

📋 **Indices and Tables**
--------------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
