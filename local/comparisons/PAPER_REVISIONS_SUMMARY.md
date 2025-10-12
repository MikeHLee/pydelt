# Paper Revisions Summary

## Terminology Changes

### "Ill-Posed" → "Poorly Conditioned"

**Rationale**: The term "ill-posed" is highly technical and may not be immediately clear to all readers. "Poorly conditioned" or "noise-sensitive" are more accessible while maintaining mathematical rigor.

**Changes Made**:

1. **Abstract** (Line 37): Changed "ill-posed problem" → "poorly conditioned problem"

2. **Section Title** (Line 76): Changed "The Ill-Posed Nature of Numerical Differentiation" → "The Poorly Conditioned Nature of Numerical Differentiation"

3. **Section 1.2** (Line 78): Changed "ill-posed nature" → "poorly conditioned nature"

4. **Section 1.2** (Line 148): Changed "ill-posed problem" → "poorly conditioned problem"

5. **Section 2.1.1** (Line 256): Changed "ill-posed problem" → "poorly conditioned problem" and "well-posed" → "well-conditioned"

## Mathematical Symbol Explanations

All formulas now follow the pattern: "...where {variable} represents {quantity}..." as commonly used in formal sciences.

### Key Additions:

#### **Condition Number** (Line 84):
- Added: "where $\|\cdot\|$ denotes a suitable function norm, $\sup$ represents the supremum (least upper bound), and $L^{-1}$ is the inverse operator (integration)"

#### **Bias-Variance Decomposition** (Line 148):
- Added: "where $\text{Bias}[\hat{f}']$ represents the systematic error (deviation from the true derivative) and $\text{Var}[\hat{f}']$ represents the variance (sensitivity to noise)"

#### **Gaussian Filter** (Line 110):
- Added: "where $g(x) = \frac{1}{\sigma_g\sqrt{2\pi}}e^{-\frac{x^2}{2\sigma_g^2}}$ is the Gaussian kernel, $\sigma_g$ controls the smoothing bandwidth, and $*$ denotes the convolution operator"

#### **Polynomial Fitting** (Line 118):
- Added: "where $a_j$ are polynomial coefficients, $w$ is the window half-width, and the coefficients are determined by minimizing..."

#### **Spline Interpolation** (Line 131):
- Added: "where each $S_i(x) = a_i + b_i(x-x_i) + c_i(x-x_i)^2 + d_i(x-x_i)^3$ is a cubic polynomial on interval $[x_i, x_{i+1}]$, and $a_i, b_i, c_i, d_i$ are coefficients determined by continuity constraints"

#### **Regularization** (Line 139):
- Added: "where $J[f]$ is the objective functional, $\lambda > 0$ is the regularization parameter controlling the trade-off between data fidelity (first term) and smoothness (second term), and $p$ determines the type of regularization"

#### **Smoothing Spline** (Line 169):
- Added: "where $E[S]$ is the energy functional, $y_i$ are observed data values, $S(x_i)$ are spline values at data points, $\lambda \geq 0$ is the smoothing parameter (automatically selected using generalized cross-validation), and the integral penalizes curvature"

#### **LLA Method** (Line 177):
- Added: "where $a_i$ is the local intercept and $b_i$ is the local slope, determined using weighted least squares within a window of size $w$"

#### **GLLA Method** (Line 185):
- Added: "where $m$ is the polynomial degree (embedding dimension), $a_{i,j}$ are local polynomial coefficients, and derivatives are computed as $f^{(n)}(x_i) = n! \cdot a_{i,n}$ where $n!$ is the factorial"

#### **GOLD Method** (Line 193):
- Added: "where $H_j$ are Hermite polynomials of order $j$, $c_{i,j}$ are expansion coefficients, $h > 0$ is a scale parameter, and the orthogonality of Hermite polynomials improves numerical stability"

#### **LOWESS Method** (Line 201):
- Added: "where $w_j(x)$ is the weight for point $x_j$ when evaluating at $x$, $W$ is a weight function (typically tri-cubic), $d(x)$ is the distance to the $q$-th nearest neighbor of $x$, with $q = \lfloor f \cdot n \rfloor$ where $f \in (0,1]$ is the smoothing parameter (fraction of points used) and $\lfloor \cdot \rfloor$ is the floor function"

#### **LOESS Method** (Line 209):
- Added: "where $\hat{f}(x)$ is the estimated function value, $\arg\min$ denotes the minimizing argument, $\mathcal{P}_d$ is the space of polynomials of degree $d$, $w_i(x)$ are distance-based weights, and $\rho$ is a robust loss function (e.g., bisquare)"

#### **FDA Method** (Line 217):
- Added: "where $K$ is the number of basis functions, $\phi_k(x)$ are basis functions (typically B-splines), and coefficients $c_k$ are determined by penalized least squares"

#### **Neural Networks** (Line 225):
- Added: "where $L(\theta)$ is the loss function, $\theta$ represents network parameters (weights and biases), $f_{\theta}(x_i)$ is the network output at $x_i$, $\lambda \geq 0$ is the regularization strength, and $R(\theta)$ is a regularization term (e.g., $L_2$ norm)"

#### **Universal Differentiation Interface** (Line 343):
- Added: "where $D^n[I]$ denotes the $n$-th order derivative operator applied to interpolator $I$, and $n \in \mathbb{N}$ is the derivative order"
- Added: "where $\mathbf{m} = [m_1, m_2, \ldots, m_n]$ specifies the input dimensions for differentiation, and $\mathbf{x} \in \mathbb{R}^d$ is the evaluation point in $d$-dimensional space"

#### **Noise Robustness** (Line 412):
- Added: "where $R(\alpha)$ is the error amplification factor at noise level $\alpha$, and MAE denotes mean absolute error"

#### **Gradient Error** (Line 422):
- Added: "where $E_{\nabla f}$ is the gradient error, $\hat{\nabla}f$ is the estimated gradient, $\nabla f$ is the true gradient, and $\|\cdot\|_2$ denotes the Euclidean (L2) norm"

#### **Jacobian Error** (Line 428):
- Added: "where $E_{J_f}$ is the Jacobian error, $\hat{J}_f$ is the estimated Jacobian matrix, $J_f$ is the true Jacobian, $\|\cdot\|_F$ denotes the Frobenius norm, and the sum is over all matrix elements $i,j$"

#### **GLLA Error Bound** (Line 451):
- Added: "where $f'(x)$ is the true derivative, $\hat{f}'(x)$ is the estimated derivative, $C^{m+1}[a,b]$ denotes the space of $(m+1)$-times continuously differentiable functions on interval $[a,b]$, $h > 0$ is the effective window size, $m \geq 1$ is the embedding dimension (polynomial degree), $\sigma \geq 0$ is the noise standard deviation, $n$ is the number of points in the local window, and $C, K > 0$ are constants depending on the function's smoothness properties"

#### **LOWESS/LOESS Robustness** (Line 465):
- Added: "where $\hat{f}(x)$ is the estimated function value at $x$, $\arg\min$ denotes the minimizing argument, $\mathcal{P}_d$ is the space of polynomials of degree $d$, $w_i(x)$ are distance-based weights, $\rho$ is a robust loss function (typically bisquare: $\rho(u) = (1-u^2)^2$ for $|u| < 1$ and 0 otherwise), $s > 0$ is a scale parameter estimated from the data, and $y_i$ are observed values"

#### **Neural Network Regularization** (Line 473):
- Added: "where $\min_{\theta}$ denotes minimization over network parameters $\theta$, the first term is the data fidelity (mean squared error), $\lambda \geq 0$ is the regularization strength, and $\|\theta\|_2^2 = \sum_j \theta_j^2$ is the squared L2 norm of parameters"

#### **MSE Decomposition** (Line 483):
- Added: "where $\text{MSE}[\hat{f}'] = \mathbb{E}[(\hat{f}' - f')^2]$ is the expected squared error, $\text{Bias}[\hat{f}'] = \mathbb{E}[\hat{f}'] - f'$ is the systematic error, $\text{Var}[\hat{f}'] = \mathbb{E}[(\hat{f}' - \mathbb{E}[\hat{f}'])^2]$ is the variance, and $\mathbb{E}[\cdot]$ denotes expectation"

#### **Gradient Computation** (Line 545):
- Added: "where $\nabla f(\mathbf{x}) \in \mathbb{R}^n$ is the gradient vector, $\frac{\partial f}{\partial x_j}$ denotes the partial derivative with respect to the $j$-th input dimension, and $[\cdot]^T$ denotes the transpose operation"

#### **Influence Function** (Line 555):
- Added: "where $IF(\mathbf{x}, \delta)$ is the influence function measuring sensitivity to data perturbations, $\delta$ represents a data perturbation (e.g., outlier), $M > 0$ is a finite constant, and $\|\cdot\|_2$ denotes the Euclidean norm"

#### **Jacobian Matrix** (Line 570):
- Added: "where $\mathbf{J}_f(\mathbf{x}) \in \mathbb{R}^{m \times n}$ is the Jacobian matrix, $f_i$ denotes the $i$-th output component of $\mathbf{f}$, and $\frac{\partial f_i}{\partial x_j}$ is the partial derivative of the $i$-th output with respect to the $j$-th input"

#### **Jacobian Error Bound** (Line 578):
- Added: "where $\mathbf{J}_f(\mathbf{x})$ is the true Jacobian, $\hat{\mathbf{J}}_f(\mathbf{x})$ is the estimated Jacobian, $\|\cdot\|_F$ denotes the Frobenius norm, and the error in each partial derivative $\hat{\frac{\partial f_i}{\partial x_j}}$ is bounded by the corresponding univariate error bound"

#### **Mixed Partials** (Line 590):
- Added: "where $\frac{\partial^2 f_{\theta}}{\partial x_i \partial x_j}$ is the mixed second-order partial derivative with respect to inputs $x_i$ and $x_j$ (where $i \neq j$), and the computation is performed using the chain rule of automatic differentiation"

#### **Method Selection** (Line 694):
- Added: "where $M^*$ is the optimal method, $\arg\max$ denotes the maximizing argument, $\mathcal{M}$ is the set of available methods, $\mathcal{A}(f, M) \in [0,1]$ is a normalized accuracy metric, $\mathcal{R}(f, M, \sigma) \in [0,1]$ is a normalized robustness metric, $\mathcal{C}(f, M, n) \geq 0$ is the computational cost, and $w_A, w_R, w_C \geq 0$ are weights (with $w_A + w_R + w_C = 1$) reflecting the relative importance of each criterion"

#### **GLLA Optimal Parameters** (Line 725):
- Added: "where $m_{\text{optimal}}$ is the optimal embedding dimension, $\arg\min_m$ denotes minimization over $m$, $C_1, C_2 > 0$ are constants depending on the function, $h > 0$ is the effective window size, $\sigma \geq 0$ is the noise level, $n$ is the number of local points, and $\binom{m}{n}$ is the binomial coefficient"

#### **GOLD Optimal Window** (Line 733):
- Added: "where $w_{\text{optimal}}$ is the optimal window size (number of points), $C_3, C_4 > 0$ are method-specific constants, $\sigma^2$ is the noise variance, $f^{(m+1)}$ is the $(m+1)$-th derivative of the function (measuring smoothness), and $m$ is the polynomial degree"

#### **LOESS/LOWESS Optimal Span** (Line 741):
- Added: "where $\alpha_{\text{optimal}} \in (0,1]$ is the optimal span parameter (fraction of points used), $C_5, C_6 > 0$ are constants, $\sigma^2$ is the noise variance, $n$ is the total number of data points, $f^{(p+1)}$ is the $(p+1)$-th derivative (measuring curvature), and $p \geq 1$ is the degree of the local polynomial"

#### **Spline Optimal Smoothing** (Line 749):
- Added: "where $s_{\text{optimal}} \geq 0$ is the optimal smoothing parameter, $n$ is the number of data points, $\sigma^2$ is the noise variance, $\text{trace}(\cdot)$ denotes the matrix trace, $I$ is the identity matrix, and $A(s) \in \mathbb{R}^{n \times n}$ is the influence (hat) matrix of the spline that maps observed values to fitted values"

#### **Divergence** (Line 769):
- Added: "where $\nabla \cdot \mathbf{F}$ is the divergence (a scalar field), $\mathbf{F} = [F_1, F_2, \ldots, F_n]^T$ is the vector field, $F_i$ is the $i$-th component of $\mathbf{F}$, and $\frac{\partial F_i}{\partial x_i}$ is the partial derivative of the $i$-th component with respect to the $i$-th coordinate"

#### **Curl** (Line 783):
- Added: "where $\nabla \times \mathbf{F}$ is the curl (a vector field), $\mathbf{F} = [F_1, F_2, F_3]^T$ is the 3D vector field with components $(F_1, F_2, F_3)$ corresponding to coordinates $(x_1, x_2, x_3)$. For 2D fields, scalar curl: $\nabla \times \mathbf{F} = \frac{\partial F_y}{\partial x} - \frac{\partial F_x}{\partial y}$ where $F_x, F_y$ are the $x$ and $y$ components"

#### **Strain and Stress Tensors** (Line 794, 800):
- Added: "where $\epsilon_{ij}$ is the $(i,j)$ component of the strain tensor, $\mathbf{u} = [u_1, u_2, \ldots, u_n]^T$ is the displacement field, and the symmetrization ensures $\epsilon_{ij} = \epsilon_{ji}$"
- Added: "where $\sigma_{ij}$ is the $(i,j)$ component of the stress tensor, $\lambda, \mu$ are Lamé parameters (material properties), $\delta_{ij}$ is the Kronecker delta ($\delta_{ij} = 1$ if $i=j$, else $0$), and $\epsilon_{kk} = \sum_k \epsilon_{kk}$ is the trace of the strain tensor (Einstein summation convention)"

#### **Directional Derivatives** (Line 811):
- Added: "where $D_{\mathbf{v}}f$ is the directional derivative of scalar function $f$ in direction $\mathbf{v}$, $\nabla f$ is the gradient vector, $\mathbf{v} \in \mathbb{R}^n$ is the direction vector (typically normalized: $\|\mathbf{v}\|_2 = 1$), and $\cdot$ denotes the dot product"

#### **Stochastic Calculus** (Line 828):
- Added: "where $Y_t$ and $X_t$ are stochastic processes, $dY_t$ and $dX_t$ are stochastic differentials, $f'(X_t)$ is the derivative of transformation function $f$, $\circ$ denotes the Stratonovich product, $f''(X_t)$ is the second derivative, and $\sigma^2$ is the diffusion coefficient (variance rate)"

#### **Mixed Partials with Neural Networks** (Line 858):
- Added: "where the mixed partial derivative is computed exactly using the chain rule of automatic differentiation, with $i \neq j$ denoting different input dimensions"

## Summary

**Total Changes**: 40+ mathematical formulas now have complete symbol explanations
**Terminology Updates**: 5 instances of "ill-posed" replaced with "poorly conditioned"
**Consistency**: All formulas follow the formal sciences pattern for variable explanation

The paper now provides clear, accessible explanations while maintaining mathematical rigor, making it suitable for both expert and general scientific audiences.
