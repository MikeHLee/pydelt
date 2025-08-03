.. pydelt documentation master file, created by
   sphinx-quickstart on Sun Jul 27 15:58:03 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

pydelt: Dynamical Systems & Differential Equations Approximation
================================================================

**pydelt** is a comprehensive Python package for approximating derivatives and solving differential equations from data, with applications in dynamical systems analysis, time series modeling, and scientific computing.

🚀 **Core Capabilities**
-----------------------

* **Universal Differentiation Interface**: Consistent API across all interpolation methods with `.differentiate(order, mask)` support
* **Multivariate Calculus**: Gradient (∇f), Jacobian (∂f/∂x), Hessian (∂²f/∂x²), and Laplacian (∇²f) computation
* **Higher-Order Derivatives**: Support for arbitrary-order derivatives with analytical and numerical methods
* **Advanced Interpolation**: Splines, LOWESS, LOESS, Local Linear Approximation (LLA), and neural network-based methods
* **Automatic Differentiation**: PyTorch and TensorFlow backends for exact gradient computation
* **Vector & Tensor Operations**: Full support for vector-valued functions and tensor calculus
* **Time Series Applications**: Specialized methods for temporal data analysis and dynamical systems identification

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

**3. Time Series Application**

.. code-block:: python

   from pydelt.derivatives import lla
   
   # Traditional time series derivative (legacy API)
   time = np.linspace(0, 2*np.pi, 100)
   signal = np.sin(time)
   result = lla(time.tolist(), signal.tolist(), window_size=5)
   derivative = result[0]  # Extract derivatives

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
   :caption: Contents:

   installation
   quickstart
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
