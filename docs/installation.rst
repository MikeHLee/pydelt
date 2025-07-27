Installation
============

Requirements
------------

pydelt requires Python 3.8 or higher and the following dependencies:

* numpy >= 1.20.0
* pandas >= 1.3.0
* scipy >= 1.7.0
* plotly >= 5.0.0
* statsmodels >= 0.13.0

Optional Dependencies
--------------------

For neural network functionality:

* **PyTorch**: For neural network-based derivatives and interpolation
* **TensorFlow**: Alternative backend for neural networks

Install from PyPI
-----------------

The easiest way to install pydelt is from PyPI:

.. code-block:: bash

   pip install pydelt

Install with Optional Dependencies
---------------------------------

To install with PyTorch support:

.. code-block:: bash

   pip install pydelt torch

To install with TensorFlow support:

.. code-block:: bash

   pip install pydelt tensorflow

Install from Source
------------------

To install the latest development version:

.. code-block:: bash

   git clone https://github.com/MikeHLee/pydelt.git
   cd pydelt
   pip install -e .

Verify Installation
------------------

To verify your installation:

.. code-block:: python

   import pydelt
   print(f"pydelt version: {pydelt.__version__}")
   
   # Test basic functionality
   import numpy as np
   from pydelt.derivatives import lla_derivative
   
   time = np.linspace(0, 1, 10)
   signal = time**2
   derivative = lla_derivative(time, signal)
   print("Installation successful!")
