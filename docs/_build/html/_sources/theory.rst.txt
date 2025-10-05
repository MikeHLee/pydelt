=============================
Mathematical Theory
=============================

This section explains the mathematical foundations behind PyDelt's numerical differentiation methods, from classical finite differences to modern interpolation-based approaches.

.. contents:: Table of Contents
   :local:
   :depth: 3

Notation and Problem Setup
===========================

Continuous vs. Discrete Functions
----------------------------------

In numerical differentiation, we work with **discrete data points** rather than continuous functions:

* **Continuous case**: We have a function :math:`f: \mathbb{R} \to \mathbb{R}` and want :math:`\frac{df}{dx}`
* **Discrete case**: We have data points :math:`\{(t_i, s_i)\}_{i=1}^N` and want to estimate :math:`\frac{ds}{dt}` at these or other points

PyDelt addresses the discrete case by:

1. **Interpolation**: Constructing a smooth function :math:`\hat{f}` that approximates the data
2. **Differentiation**: Computing derivatives of :math:`\hat{f}` analytically or numerically

Grid Types
----------

PyDelt supports both uniform and non-uniform grids:

* **Uniform grid**: :math:`t_i = a + i \cdot \Delta t` for constant spacing :math:`\Delta t`
* **Non-uniform grid**: Arbitrary spacing :math:`t_{i+1} - t_i \neq \text{constant}`

Most real-world data uses non-uniform grids, which PyDelt handles naturally through interpolation.

Method Categories
=================

PyDelt implements three fundamental approaches to numerical differentiation, applicable to both **known analytical functions** (where we want numerical approximations) and **unknown functions** (where we only have discrete data):

1. **Finite Difference Methods**: Direct approximation using neighboring points
2. **Interpolation-Based Methods**: Fit smooth functions, then differentiate
3. **Learning-Based Methods**: Neural networks with automatic differentiation

Each has distinct advantages and theoretical properties.

Applicability to Known vs. Unknown Functions
--------------------------------------------

**Known Functions** (analytical form available):

* **Use Case**: When you have f(x) = sin(x) or other explicit formulas but need numerical derivatives
* **Why Numerical Methods**: Automatic differentiation, avoiding symbolic complexity, or validating analytical derivatives
* **Best Methods**: Neural networks with autodiff (exact), splines (high accuracy), LLA/GLLA (robust)

**Unknown Functions** (only discrete data available):

* **Use Case**: Experimental measurements, sensor data, financial time series, physical observations
* **Challenge**: Must reconstruct smooth function from noisy, sparse, or irregularly sampled data
* **Best Methods**: LOWESS/LOESS (robust to noise), FDA (functional data), splines (smooth data), neural networks (complex patterns)

**PyDelt's Universal Approach**: All methods work seamlessly for both cases through the unified `.fit(input_data, output_data).differentiate(order)` interface.

Finite Difference Methods
==========================

Basic Principle
---------------

Finite differences approximate derivatives as linear combinations of function values at neighboring points:

.. math::

   \frac{d^n f}{dt^n}\bigg|_{t=t_k} \approx \sum_{i \in A} c_i \cdot f(t_{k+i})

where :math:`A` is a set of offsets (the **stencil**) and :math:`c_i` are **finite difference coefficients**.

Central Difference (2nd Order)
-------------------------------

The most common scheme is the **central difference** for first derivatives:

.. math::

   f'(t_k) \approx \frac{f(t_{k+1}) - f(t_{k-1})}{2\Delta t}

**Derivation from Taylor Series**:

Expand :math:`f(t_{k+1})` and :math:`f(t_{k-1})` around :math:`t_k`:

.. math::

   f(t_{k+1}) &= f(t_k) + \Delta t \cdot f'(t_k) + \frac{(\Delta t)^2}{2} f''(t_k) + O(\Delta t^3) \\
   f(t_{k-1}) &= f(t_k) - \Delta t \cdot f'(t_k) + \frac{(\Delta t)^2}{2} f''(t_k) + O(\Delta t^3)

Subtracting:

.. math::

   f(t_{k+1}) - f(t_{k-1}) = 2\Delta t \cdot f'(t_k) + O(\Delta t^3)

Therefore:

.. math::

   f'(t_k) = \frac{f(t_{k+1}) - f(t_{k-1})}{2\Delta t} + O(\Delta t^2)

The error is :math:`O(\Delta t^2)`, making this a **second-order accurate** method.

**Stencil Visualization**:

.. code-block:: text

   Points:    t_{k-1}    t_k    t_{k+1}
   Weights:     -1/2Δt     0      +1/2Δt

Higher-Order Derivatives
------------------------

For second derivatives, the central difference formula is:

.. math::

   f''(t_k) \approx \frac{f(t_{k+1}) - 2f(t_k) + f(t_{k-1})}{(\Delta t)^2}

**Stencil**:

.. code-block:: text

   Points:    t_{k-1}      t_k      t_{k+1}
   Weights:    1/Δt²     -2/Δt²      1/Δt²

**Iterative Approach** (used in PyDelt):

For :math:`n`-th order derivatives, apply central differences recursively:

.. math::

   f^{(n)}(t) \approx \frac{f^{(n-1)}(t+h) - f^{(n-1)}(t-h)}{2h}

This is implemented in ``LowessInterpolator`` and ``LoessInterpolator``.

Accuracy and Stability
----------------------

**Accuracy**: Central differences are :math:`O(\Delta t^2)` accurate

**Stability Issues**:

* **Noise amplification**: Differentiation amplifies high-frequency noise
* **Truncation error**: Finite :math:`\Delta t` introduces approximation errors
* **Cancellation error**: Subtracting similar numbers loses precision

**PyDelt's Solution**: Use interpolation to smooth data before applying finite differences.

Interpolation-Based Methods
============================

Core Idea
---------

Instead of directly using finite differences on raw data:

1. **Fit** a smooth interpolating function :math:`\hat{f}(t)` to the data
2. **Differentiate** :math:`\hat{f}(t)` analytically or numerically

This approach:

* Reduces noise sensitivity
* Provides derivatives at arbitrary points (not just grid points)
* Enables higher-order derivatives with better stability

Spline Interpolation
--------------------

**Method**: Fit piecewise cubic polynomials with continuous second derivatives

**Mathematical Form**:

On interval :math:`[t_i, t_{i+1}]`, the spline is:

.. math::

   S(t) = a_i + b_i(t-t_i) + c_i(t-t_i)^2 + d_i(t-t_i)^3

**Derivative Computation**:

Analytical differentiation of the spline:

.. math::

   S'(t) &= b_i + 2c_i(t-t_i) + 3d_i(t-t_i)^2 \\
   S''(t) &= 2c_i + 6d_i(t-t_i)

**Advantages**:

* Smooth interpolation (C² continuity)
* Analytical derivatives (no numerical errors)
* Efficient computation via scipy's ``UnivariateSpline``

**Implementation**: ``SplineInterpolator`` class

**Smoothing Parameter**:

PyDelt uses smoothing splines that minimize:

.. math::

   \sum_{i=1}^N (s_i - S(t_i))^2 + \lambda \int (S''(t))^2 dt

where :math:`\lambda` is the smoothing parameter (``smoothing`` argument).

Local Linear Approximation (LLA)
---------------------------------

**Method**: Fit local polynomials in sliding windows

**Mathematical Form**:

At each point :math:`t_k`, fit a polynomial to nearby points:

.. math::

   p(t) = a_0 + a_1(t-t_k) + a_2(t-t_k)^2 + \ldots

using points :math:`\{t_i : |i-k| \leq w\}` where :math:`w` is the window size.

**Derivative Estimation**:

The derivative at :math:`t_k` is the polynomial coefficient:

.. math::

   f'(t_k) \approx a_1

For higher orders:

.. math::

   f''(t_k) \approx 2a_2, \quad f'''(t_k) \approx 6a_3

**Hermite Polynomial Representation**:

PyDelt uses Hermite polynomials for analytical derivatives:

.. math::

   H(t) = f(t_k) + f'(t_k)(t-t_k) + \frac{f''(t_k)}{2}(t-t_k)^2 + \ldots

**Implementation**: ``LlaInterpolator`` class

**Advantages**:

* Adaptive to local data behavior
* Robust to varying noise levels
* Analytical derivatives from polynomial coefficients

Generalized LLA (GLLA)
----------------------

**Extension**: Uses Takens' embedding theorem for time series

**Mathematical Form**:

For a time series, construct delay-coordinate embedding:

.. math::

   \mathbf{x}_i = [s_i, s_{i+\tau}, s_{i+2\tau}, \ldots, s_{i+(m-1)\tau}]

where:

* :math:`m` = embedding dimension
* :math:`\tau` = time delay

Then apply local polynomial fitting in the embedded space.

**Derivative Computation**:

Same as LLA but in higher-dimensional space, capturing temporal dependencies.

**Implementation**: ``GllaInterpolator`` class

**Use Cases**:

* Chaotic time series
* Nonlinear dynamical systems
* Data with temporal correlations

LOWESS/LOESS Methods
--------------------

**Method**: Locally weighted regression with robust weights

**Mathematical Form**:

At each point :math:`t_k`, minimize weighted least squares:

.. math::

   \min_{a,b} \sum_{i=1}^N w_i(t_k) \cdot (s_i - a - b(t_i - t_k))^2

where :math:`w_i(t_k)` are tricube weights:

.. math::

   w_i(t_k) = \left(1 - \left|\frac{t_i - t_k}{d(t_k)}\right|^3\right)^3

and :math:`d(t_k)` is the distance to the :math:`k`-th nearest neighbor.

**Derivative Computation**:

PyDelt uses two approaches:

1. **Numerical differentiation** of the smoothed curve using central differences
2. **Direct estimation** from local regression slope :math:`b`

**Implementation**: ``LowessInterpolator`` and ``LoessInterpolator`` classes

**Advantages**:

* Robust to outliers
* Automatic bandwidth selection
* No global parametric assumptions

Functional Data Analysis (FDA)
-------------------------------

**Method**: Represent data as smooth functions using basis expansions

**Mathematical Form**:

Represent the function as a linear combination of basis functions:

.. math::

   f(t) = \sum_{k=1}^K c_k \phi_k(t)

Common bases:

* **B-splines**: Piecewise polynomials
* **Fourier**: Trigonometric functions
* **Wavelets**: Localized oscillations

**Derivative Computation**:

Differentiate the basis functions:

.. math::

   f'(t) = \sum_{k=1}^K c_k \phi_k'(t)

**Implementation**: ``FdaInterpolator`` class (uses B-spline basis)

**Advantages**:

* Principled statistical framework
* Optimal smoothing parameter selection
* Handles functional data naturally

Neural Network Methods
=======================

Automatic Differentiation
--------------------------

**Method**: Train neural networks, then use automatic differentiation (autodiff)

**Mathematical Form**:

Neural network function:

.. math::

   \hat{f}(t; \theta) = W_L \sigma(W_{L-1} \sigma(\ldots \sigma(W_1 t + b_1) \ldots) + b_{L-1}) + b_L

where :math:`\theta = \{W_i, b_i\}` are learned parameters and :math:`\sigma` is an activation function.

**Derivative Computation**:

Use automatic differentiation (backpropagation):

.. math::

   \frac{d\hat{f}}{dt} = \frac{\partial \hat{f}}{\partial t}\bigg|_{\theta=\theta^*}

computed via chain rule through the network.

**Implementation**: ``NeuralNetworkInterpolator`` class

**Advantages**:

* **Exact derivatives** (no numerical approximation)
* Scales to high dimensions
* Captures complex nonlinear patterns
* Automatic higher-order derivatives

**Frameworks**: PyTorch and TensorFlow support

Multivariate Derivatives
=========================

PyDelt provides comprehensive multivariate calculus operations through the ``MultivariateDerivatives`` class, supporting gradient, Jacobian, Hessian, and Laplacian computations. These operations are **fully implemented and production-ready** with extensive documentation and examples.

Gradient (Scalar Functions)
----------------------------

For :math:`f: \mathbb{R}^n \to \mathbb{R}`, the gradient is:

.. math::

   \nabla f = \left[\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \ldots, \frac{\partial f}{\partial x_n}\right]^T

**PyDelt Approach**:

1. Fit separate 1D interpolators for each partial derivative :math:`\frac{\partial f}{\partial x_i}`
2. Evaluate at query points to get gradient vector

**Implementation**: ``MultivariateDerivatives.gradient()`` ✅ **IMPLEMENTED**

**Use Cases**:

* **Optimization**: Finding critical points, gradient descent algorithms
* **Physics**: Electric field from potential, temperature gradients
* **Machine Learning**: Loss function gradients, feature importance

Jacobian (Vector Functions)
----------------------------

For :math:`\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m`, the Jacobian is:

.. math::

   J_{\mathbf{f}} = \begin{bmatrix}
   \frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\
   \vdots & \ddots & \vdots \\
   \frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n}
   \end{bmatrix}

**PyDelt Approach**:

Fit interpolators for each output-input pair :math:`\frac{\partial f_i}{\partial x_j}`

**Implementation**: ``MultivariateDerivatives.jacobian()`` ✅ **IMPLEMENTED**

**Use Cases**:

* **Fluid Dynamics**: Velocity field analysis, vorticity and divergence computation
* **Robotics**: Kinematic transformations, manipulator Jacobians
* **Dynamical Systems**: Linearization around equilibrium points, stability analysis

Hessian (Second Derivatives)
-----------------------------

For :math:`f: \mathbb{R}^n \to \mathbb{R}`, the Hessian is:

.. math::

   H_f = \begin{bmatrix}
   \frac{\partial^2 f}{\partial x_1^2} & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \\
   \vdots & \ddots & \vdots \\
   \frac{\partial^2 f}{\partial x_n \partial x_1} & \cdots & \frac{\partial^2 f}{\partial x_n^2}
   \end{bmatrix}

**PyDelt Approach**:

* **Diagonal elements**: Second-order differentiation of 1D interpolators
* **Off-diagonal (mixed partials)**: Approximated as zero for traditional methods

**Note**: For exact mixed partials, use neural network methods with autodiff.

**Implementation**: ``MultivariateDerivatives.hessian()`` ✅ **IMPLEMENTED**

**Use Cases**:

* **Optimization**: Newton's method, trust region algorithms, curvature analysis
* **Statistics**: Fisher information matrix, covariance estimation
* **Physics**: Stability analysis, potential energy surfaces

Laplacian (Divergence of Gradient)
-----------------------------------

The Laplacian is the trace of the Hessian:

.. math::

   \nabla^2 f = \text{tr}(H_f) = \sum_{i=1}^n \frac{\partial^2 f}{\partial x_i^2}

**PyDelt Approach**:

Sum diagonal elements of the Hessian.

**Implementation**: ``MultivariateDerivatives.laplacian()`` ✅ **IMPLEMENTED**

**Applications**:

* Heat equation: :math:`\frac{\partial u}{\partial t} = \alpha \nabla^2 u`
* Poisson equation: :math:`\nabla^2 \phi = \rho`
* Diffusion processes, wave propagation, quantum mechanics

Tensor Calculus Operations
==========================

PyDelt provides advanced tensor calculus operations through the ``TensorDerivatives`` class for continuum mechanics, fluid dynamics, and physics applications.

Divergence (Vector Fields)
--------------------------

For a vector field :math:`\mathbf{F}: \mathbb{R}^n \to \mathbb{R}^n`, the divergence is:

.. math::

   \nabla \cdot \mathbf{F} = \sum_{i=1}^n \frac{\partial F_i}{\partial x_i}

**Implementation**: ``TensorDerivatives.divergence()`` ✅ **IMPLEMENTED**

**Physical Interpretation**: Measures expansion (positive) or contraction (negative) of the field.

Curl (Vector Fields)
--------------------

For 3D vector fields :math:`\mathbf{F}: \mathbb{R}^3 \to \mathbb{R}^3`, the curl is:

.. math::

   \nabla \times \mathbf{F} = \begin{bmatrix}
   \frac{\partial F_z}{\partial y} - \frac{\partial F_y}{\partial z} \\
   \frac{\partial F_x}{\partial z} - \frac{\partial F_z}{\partial x} \\
   \frac{\partial F_y}{\partial x} - \frac{\partial F_x}{\partial y}
   \end{bmatrix}

For 2D fields, scalar curl: :math:`\nabla \times \mathbf{F} = \frac{\partial F_y}{\partial x} - \frac{\partial F_x}{\partial y}`

**Implementation**: ``TensorDerivatives.curl()`` ✅ **IMPLEMENTED**

**Physical Interpretation**: Measures rotation or vorticity of the field.

Strain and Stress Tensors
-------------------------

For displacement fields :math:`\mathbf{u}`, the strain tensor is:

.. math::

   \epsilon_{ij} = \frac{1}{2}\left(\frac{\partial u_i}{\partial x_j} + \frac{\partial u_j}{\partial x_i}\right)

The stress tensor (linear elasticity):

.. math::

   \sigma_{ij} = \lambda \delta_{ij} \epsilon_{kk} + 2\mu \epsilon_{ij}

where :math:`\lambda, \mu` are Lamé parameters.

**Implementation**: 

* ``TensorDerivatives.strain_tensor()`` ✅ **IMPLEMENTED**
* ``TensorDerivatives.stress_tensor()`` ✅ **IMPLEMENTED**

**Applications**: Continuum mechanics, structural analysis, material deformation.

Directional Derivatives
-----------------------

Derivative along a specific direction :math:`\mathbf{v}`:

.. math::

   D_{\mathbf{v}}f = \nabla f \cdot \mathbf{v}

**Implementation**: ``TensorDerivatives.directional_derivative()`` ✅ **IMPLEMENTED**

Stochastic Calculus Extensions
===============================

PyDelt supports stochastic derivatives for probabilistic modeling and uncertainty quantification.

Itô vs. Stratonovich
---------------------

For stochastic differential equations (SDEs), derivatives transform differently:

**Itô Calculus**:

.. math::

   dY_t = f(X_t)dX_t \implies \frac{dY}{dX} = f'(X_t)

**Stratonovich Calculus**:

.. math::

   dY_t = f(X_t) \circ dX_t \implies \frac{dY}{dX} = f'(X_t) + \frac{1}{2}f''(X_t)\sigma^2

where :math:`\sigma^2` is the diffusion coefficient.

**PyDelt Implementation**: ✅ **IMPLEMENTED**

The ``set_stochastic_link()`` method applies these corrections automatically.

Stochastic Link Functions
-------------------------

Transform derivatives through probability distributions:

.. math::

   \frac{d}{dt}g(f(t)) = g'(f(t)) \cdot f'(t) + \text{correction terms}

For stochastic processes, additional correction terms appear based on the distribution.

**Supported Distributions**: ✅ **IMPLEMENTED**

* **Normal**: Symmetric, unbounded (interest rates, errors)
* **Lognormal**: Positive, right-skewed (stock prices, volumes)
* **Gamma**: Positive, flexible shape (waiting times, rates)
* **Beta**: Bounded [0,1] (proportions, ratios)
* **Exponential**: Memoryless, decreasing (survival times)
* **Poisson**: Discrete, non-negative (count processes)

**Implementation**: ``interpolator.set_stochastic_link(link_function, **params)``

**Applications**:

* **Finance**: Option Greeks, risk analysis, volatility modeling
* **Physics**: Brownian motion, diffusion processes
* **Biology**: Population dynamics with stochastic effects

Error Analysis and Accuracy
============================

Sources of Error
----------------

1. **Truncation Error**: From finite difference approximations (:math:`O(\Delta t^2)`)
2. **Interpolation Error**: From fitting smooth functions to discrete data
3. **Numerical Error**: From floating-point arithmetic
4. **Noise Amplification**: Differentiation amplifies measurement noise

Error Bounds
------------

**Finite Differences**:

For central differences with step size :math:`h`:

.. math::

   \text{Error} \approx \frac{h^2}{6}f'''(\xi) + \frac{\epsilon}{h}

where :math:`\epsilon` is the noise level. Optimal :math:`h \approx (\epsilon)^{1/3}`.

**Spline Interpolation**:

For smoothing splines with parameter :math:`\lambda`:

.. math::

   \text{Error} \propto \lambda^{-1/2} + \lambda^{1/2}

Optimal :math:`\lambda` balances bias and variance.

Method Selection Guidelines
---------------------------

+------------------+------------------------+------------------+-------------------+
| Method           | Best For               | Accuracy         | Computational Cost|
+==================+========================+==================+===================+
| Spline           | Smooth data            | High             | Low               |
+------------------+------------------------+------------------+-------------------+
| LLA/GLLA         | Noisy data             | Medium-High      | Medium            |
+------------------+------------------------+------------------+-------------------+
| LOWESS/LOESS     | Outliers, varying noise| Medium           | Medium-High       |
+------------------+------------------------+------------------+-------------------+
| FDA              | Functional data        | High             | Medium            |
+------------------+------------------------+------------------+-------------------+
| Neural Networks  | Complex patterns, high-dim | Very High    | High (training)   |
+------------------+------------------------+------------------+-------------------+

Theoretical Guarantees
======================

Convergence Properties
----------------------

**Splines**: Converge to true function as :math:`N \to \infty` with rate :math:`O(N^{-2})`

**Local Polynomials**: Converge with rate :math:`O(h^{p+1})` for degree :math:`p` polynomials

**Neural Networks**: Universal approximation theorem guarantees convergence

Consistency
-----------

All PyDelt methods are **consistent estimators**: as sample size increases, estimates converge to true derivatives (under regularity conditions).

Asymptotic Normality
--------------------

For smooth functions and sufficient data, derivative estimates are asymptotically normal:

.. math::

   \sqrt{N}(\hat{f}'(t) - f'(t)) \xrightarrow{d} \mathcal{N}(0, \sigma^2(t))

This enables confidence intervals and hypothesis testing.

References and Further Reading
===============================

**Finite Differences**:

* Fornberg, B. (1988). "Generation of finite difference formulas on arbitrarily spaced grids"
* LeVeque, R. (2007). "Finite Difference Methods for Ordinary and Partial Differential Equations"

**Spline Methods**:

* de Boor, C. (2001). "A Practical Guide to Splines"
* Wahba, G. (1990). "Spline Models for Observational Data"

**Local Polynomial Regression**:

* Fan, J. & Gijbels, I. (1996). "Local Polynomial Modelling and Its Applications"
* Cleveland, W. S. (1979). "Robust locally weighted regression and smoothing scatterplots"

**Functional Data Analysis**:

* Ramsay, J. O. & Silverman, B. W. (2005). "Functional Data Analysis"

**Neural Networks & Autodiff**:

* Baydin, A. G. et al. (2018). "Automatic differentiation in machine learning: a survey"
* Raissi, M. et al. (2019). "Physics-informed neural networks"

**Stochastic Calculus**:

* Øksendal, B. (2003). "Stochastic Differential Equations"
* Kloeden, P. E. & Platen, E. (1992). "Numerical Solution of Stochastic Differential Equations"

Summary
=======

PyDelt combines classical numerical analysis with modern machine learning to provide:

* **Multiple theoretical frameworks**: Finite differences, interpolation, and learning-based methods
* **Rigorous mathematical foundations**: Convergence guarantees and error bounds
* **Practical implementations**: Optimized algorithms with unified API
* **Advanced features**: Multivariate calculus and stochastic extensions

The choice of method depends on your data characteristics, accuracy requirements, and computational constraints. The theory presented here guides that selection.

Implementation Status
=====================

All theoretical methods described in this document are **fully implemented and production-ready**:

**Core Differentiation Methods** ✅:

* Finite differences (central, forward, backward)
* Spline interpolation (cubic, smoothing)
* Local Linear Approximation (LLA)
* Generalized LLA (GLLA) with Takens embedding
* LOWESS/LOESS robust regression
* Functional Data Analysis (FDA)
* Neural networks with automatic differentiation (PyTorch/TensorFlow)

**Multivariate Calculus** ✅:

* Gradient computation (``MultivariateDerivatives.gradient()``)
* Jacobian matrices (``MultivariateDerivatives.jacobian()``)
* Hessian matrices (``MultivariateDerivatives.hessian()``)
* Laplacian operator (``MultivariateDerivatives.laplacian()``)

**Tensor Operations** ✅:

* Divergence (``TensorDerivatives.divergence()``)
* Curl (``TensorDerivatives.curl()``)
* Strain tensor (``TensorDerivatives.strain_tensor()``)
* Stress tensor (``TensorDerivatives.stress_tensor()``)
* Directional derivatives (``TensorDerivatives.directional_derivative()``)

**Stochastic Calculus** ✅:

* Itô and Stratonovich corrections
* Six probability distributions (normal, lognormal, gamma, beta, exponential, Poisson)
* Stochastic link functions (``set_stochastic_link()``)

**Universal API**: All methods support the consistent ``.fit(input_data, output_data).differentiate(order, mask)`` interface.

Citing PyDelt
=============

If you use PyDelt in your research or applications, please cite it as follows:

**BibTeX Entry**::

   @software{pydelt2025,
     title = {PyDelt: Advanced Numerical Function Interpolation and Differentiation},
     author = {Lee, Michael},
     year = {2025},
     url = {https://github.com/MikeHLee/pydelt},
     version = {0.6.1},
     note = {Python package for numerical differentiation with multivariate calculus, 
             tensor operations, and stochastic computing support}
   }

**APA Style**::

   Lee, M. (2025). PyDelt: Advanced Numerical Function Interpolation and Differentiation 
   (Version 0.6.1) [Computer software]. https://github.com/MikeHLee/pydelt

**IEEE Style**::

   M. Lee, "PyDelt: Advanced Numerical Function Interpolation and Differentiation," 
   version 0.6.1, 2025. [Online]. Available: https://github.com/MikeHLee/pydelt

**Key Features to Cite**:

* **Universal differentiation interface** across multiple interpolation methods
* **Multivariate calculus operations** (gradient, Jacobian, Hessian, Laplacian)
* **Tensor calculus** for continuum mechanics and fluid dynamics
* **Stochastic derivatives** with Itô/Stratonovich corrections
* **Neural network integration** with automatic differentiation

License
=======

PyDelt is released under the **MIT License**, which permits:

* ✅ Commercial use
* ✅ Modification
* ✅ Distribution
* ✅ Private use

**Full License Text**::

   MIT License
   
   Copyright (c) 2025
   
   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:
   
   The above copyright notice and this permission notice shall be included in all
   copies or substantial portions of the Software.
   
   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
   SOFTWARE.

For the complete license, see the `LICENSE <https://github.com/MikeHLee/pydelt/blob/main/LICENSE>`_ file in the repository.

Contributing
============

Contributions are welcome! Please see the project repository for contribution guidelines and development setup instructions.
