.. _visual_examples:

Visual Examples
==============

This page provides interactive visualizations of PyDelt's differentiation capabilities across various scenarios.

1D Method Comparison
-------------------

This visualization compares different PyDelt interpolation methods on a simple 1D function (sine wave with noise).
Each method balances smoothness and accuracy differently, with GLLA providing the best overall performance.

.. raw:: html

   <iframe src="_static/images/method_comparison_1d.html" width="100%" height="800px" frameborder="0"></iframe>

Noise Robustness Comparison
--------------------------

This visualization demonstrates how different PyDelt methods perform with increasing levels of noise.
LOWESS and LOESS show superior noise robustness, while GLLA maintains better accuracy at peaks.

.. raw:: html

   <iframe src="_static/images/noise_robustness_comparison.html" width="100%" height="1000px" frameborder="0"></iframe>

Multivariate Derivatives
----------------------

This visualization shows PyDelt's capabilities for computing derivatives of multivariate functions.
The example demonstrates gradient computation for a 2D scalar function, showing the original function,
gradient magnitude, and partial derivatives.

.. raw:: html

   <iframe src="_static/images/multivariate_derivatives.html" width="100%" height="800px" frameborder="0"></iframe>

Higher-Order Derivatives
----------------------

This visualization demonstrates PyDelt's ability to compute higher-order derivatives (up to 2nd order)
with minimal error propagation. GLLA is particularly effective for higher-order derivatives.

.. raw:: html

   <iframe src="_static/images/higher_order_derivatives.html" width="100%" height="1000px" frameborder="0"></iframe>

Stochastic Process Differentiation
--------------------------------

This visualization shows PyDelt's application to stochastic processes, demonstrating drift estimation
in an Ornstein-Uhlenbeck process. This capability is particularly useful for SDE parameter inference.

.. raw:: html

   <iframe src="_static/images/stochastic_derivatives.html" width="100%" height="800px" frameborder="0"></iframe>

Generating Your Own Visualizations
--------------------------------

The visualizations on this page were generated using the ``generate_visualizations.py`` script in the
``docs/_static`` directory. You can modify this script to create your own visualizations for your specific data.

.. code-block:: python

   # Example: Generate 1D method comparison visualization
   from docs._static.generate_visualizations import generate_1d_comparison
   
   # Generate and save the visualization
   fig = generate_1d_comparison()
   
   # Display the figure in a Jupyter notebook
   from IPython.display import IFrame
   IFrame('_static/images/method_comparison_1d.html', width=1000, height=800)
