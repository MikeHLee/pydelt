.. pydelt documentation master file, created by
   sphinx-quickstart on Sun Jul 27 15:58:03 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

pydelt: Python Derivatives for Time Series
==========================================

**pydelt** is a comprehensive Python package for calculating derivatives of time series data using various numerical and machine learning methods.

🚀 **Key Features**
------------------

* **Multiple derivative methods**: Finite differences, local linear approximation, Gaussian processes, and neural networks
* **Advanced interpolation**: Splines, LOWESS, LOESS, and neural network-based interpolation
* **Automatic differentiation**: PyTorch and TensorFlow backends for gradient computation
* **Integration capabilities**: Numerical integration with error estimation
* **Multivariate support**: Handle multi-dimensional time series data
* **Robust error handling**: Comprehensive input validation and error messages

📦 **Installation**
------------------

Install pydelt from PyPI:

.. code-block:: bash

   pip install pydelt

🔧 **Quick Start**
-----------------

.. code-block:: python

   import numpy as np
   from pydelt.derivatives import lla
   
   # Generate sample data
   time = np.linspace(0, 2*np.pi, 100)
   signal = np.sin(time)
   
   # Calculate derivative using Local Linear Approximation
   result = lla(time.tolist(), signal.tolist(), window_size=5)
   derivative = result[0]  # Extract derivatives
   
   # The derivative of sin(x) should be approximately cos(x)
   expected = np.cos(time)
   print(f"Max error: {np.max(np.abs(derivative - expected)):.4f}")

📚 **Documentation Contents**
----------------------------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   examples
   api
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
