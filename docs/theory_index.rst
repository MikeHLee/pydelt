==============================================
Theory: Numerical Calculus for Real Systems
==============================================

.. note::
   This documentation teaches numerical differentiation from a practical perspective:
   approximating unknown dynamical systems from discrete, noisy observations.

The Central Thesis
==================

We cannot compute exact derivatives from data. But by approximating the underlying 
system with a differentiable surrogate, we transform an intractable problem into 
a tractable one—trading exactness for practicality.

Chapters
========

.. toctree::
   :maxdepth: 1
   :caption: Part I: Foundations

   theory/00_introduction
   theory/01_numerical_differentiation
   theory/02_noise_and_smoothing
   theory/03_interpolation_methods

.. toctree::
   :maxdepth: 1
   :caption: Part II: Extensions

   theory/04_multivariate_derivatives
   theory/05_approximation_theory
   theory/06_differential_equations
   theory/07_stochastic_calculus

.. toctree::
   :maxdepth: 1
   :caption: Part III: Practice

   theory/08_applications
   theory/bibliography
   theory/index

Quick Reference
===============

.. list-table:: PyDelt Methods
   :header-rows: 1
   :widths: 25 35 40

   * - Concept
     - Use Case
     - PyDelt Method
   * - First derivative
     - Velocity, rate of change
     - ``.differentiate(order=1)``
   * - Second derivative
     - Acceleration, curvature
     - ``.differentiate(order=2)``
   * - Gradient (∇f)
     - Optimization, sensitivity
     - ``MultivariateDerivatives.gradient()``
   * - Jacobian
     - Vector field analysis
     - ``MultivariateDerivatives.jacobian()``
   * - Hessian
     - Curvature, stability
     - ``MultivariateDerivatives.hessian()``
   * - Laplacian
     - Diffusion, PDEs
     - ``MultivariateDerivatives.laplacian()``

Who This Is For
===============

- **Data scientists** who know basic calculus but need numerical methods
- **Engineers** working with sensor data and dynamical systems
- **Researchers** in physics, biology, or finance dealing with noisy observations
- **ML practitioners** who want to understand gradients beyond autodiff

Prerequisites: Undergraduate calculus, basic linear algebra, Python/NumPy.

*Start your journey:* :doc:`theory/00_introduction`
