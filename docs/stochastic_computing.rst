Stochastic Computing & Probabilistic Derivatives
===============================================

Stochastic computing extends numerical differentiation to probabilistic settings, enabling uncertainty quantification and risk analysis. By applying stochastic link functions and calculus corrections, derivatives can reflect the underlying probability distributions of the data, making this essential for financial modeling, risk management, and scientific applications with inherent uncertainty.

🎲 **Core Concepts**
-------------------

**Stochastic Link Functions** transform deterministic derivatives to account for the probabilistic nature of the underlying data. Instead of computing derivatives of f(x), we compute derivatives that reflect the expected behavior when data follows a specific probability distribution.

**Itô's Lemma** provides correction terms for stochastic calculus when the underlying process follows a stochastic differential equation. For a function f(X_t) where X_t follows a diffusion process:

.. math::
   df = f'(X_t)dX_t + \frac{1}{2}f''(X_t)(dX_t)^2

**Stratonovich Integral** offers an alternative interpretation of stochastic integration that preserves the classical chain rule, useful when the noise is "physical" rather than mathematical.

**Financial Applications**: Essential for option pricing, risk management, and volatility modeling where asset prices follow geometric Brownian motion or other stochastic processes.

🔧 **Stochastic Link Functions**
-------------------------------

pydelt supports six probability distributions as stochastic link functions:

.. code-block:: python

   from pydelt.interpolation import SplineInterpolator
   
   # Create interpolator
   interpolator = SplineInterpolator(smoothing=0.1)
   interpolator.fit(time_data, price_data)
   
   # Set stochastic link function
   interpolator.set_stochastic_link(
       link_function='lognormal',  # Distribution type
       sigma=0.2,                  # Distribution parameter
       method='ito'                # Stochastic calculus method
   )
   
   # Derivatives now include stochastic corrections
   stochastic_deriv = interpolator.differentiate(order=1)
   derivatives = stochastic_deriv(evaluation_points)

**Available Distributions**:

+----------------+------------------+------------------------+-------------------------+
| Distribution   | Parameters       | Typical Applications   | Key Properties          |
+================+==================+========================+=========================+
| ``normal``     | ``sigma``        | Interest rates, errors | Symmetric, unbounded    |
+----------------+------------------+------------------------+-------------------------+
| ``lognormal``  | ``sigma``        | Stock prices, volumes  | Positive, right-skewed  |
+----------------+------------------+------------------------+-------------------------+
| ``gamma``      | ``alpha, beta``  | Waiting times, rates   | Positive, flexible shape|
+----------------+------------------+------------------------+-------------------------+
| ``beta``       | ``alpha, beta``  | Proportions, ratios    | Bounded [0,1]           |
+----------------+------------------+------------------------+-------------------------+
| ``exponential``| ``lambda``       | Survival times         | Memoryless, decreasing  |
+----------------+------------------+------------------------+-------------------------+
| ``poisson``    | ``lambda``       | Count processes        | Discrete, non-negative  |
+----------------+------------------+------------------------+-------------------------+

💰 **Example 1: Stock Price Derivatives (Geometric Brownian Motion)**
--------------------------------------------------------------------

**Classic Application**: Computing option Greeks with stochastic corrections.

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from pydelt.interpolation import SplineInterpolator
   
   # Simulate geometric Brownian motion stock price
   # dS_t = μS_t dt + σS_t dW_t
   np.random.seed(42)
   T = 1.0          # 1 year
   N = 252          # Daily observations
   dt = T / N
   mu = 0.05        # Expected return (5%)
   sigma = 0.2      # Volatility (20%)
   S0 = 100         # Initial stock price
   
   # Generate price path
   t = np.linspace(0, T, N+1)
   W = np.random.randn(N+1).cumsum() * np.sqrt(dt)  # Brownian motion
   S = S0 * np.exp((mu - 0.5*sigma**2)*t + sigma*W)  # GBM solution
   
   # Fit interpolator
   spline = SplineInterpolator(smoothing=0.01)
   spline.fit(t, S)
   
   # Compare regular vs stochastic derivatives
   regular_deriv_func = spline.differentiate(order=1)
   regular_derivatives = regular_deriv_func(t)
   
   # Set log-normal stochastic link (appropriate for stock prices)
   spline.set_stochastic_link('lognormal', sigma=sigma, method='ito')
   stochastic_deriv_func = spline.differentiate(order=1)
   stochastic_derivatives = stochastic_deriv_func(t)
   
   # Also try Stratonovich method
   spline.set_stochastic_link('lognormal', sigma=sigma, method='stratonovich')
   stratonovich_deriv_func = spline.differentiate(order=1)
   stratonovich_derivatives = stratonovich_deriv_func(t)
   
   # Analysis
   print("Stock Price Derivative Analysis:")
   print(f"Regular derivative mean: {np.mean(regular_derivatives):.2f}")
   print(f"Itô stochastic derivative mean: {np.mean(stochastic_derivatives):.2f}")
   print(f"Stratonovich derivative mean: {np.mean(stratonovich_derivatives):.2f}")
   
   # Theoretical expectation: E[dS/dt] = μS for GBM
   theoretical_mean = mu * np.mean(S)
   print(f"Theoretical mean (μS): {theoretical_mean:.2f}")
   
   # Compute differences
   ito_correction = np.mean(stochastic_derivatives - regular_derivatives)
   stratonovich_correction = np.mean(stratonovich_derivatives - regular_derivatives)
   
   print(f"\nStochastic Corrections:")
   print(f"Itô correction: {ito_correction:.2f}")
   print(f"Stratonovich correction: {stratonovich_correction:.2f}")
   
   # Option Greeks approximation
   # Delta (price sensitivity) ≈ derivative w.r.t. underlying
   current_price = S[-1]
   current_delta_regular = regular_derivatives[-1] / current_price
   current_delta_stochastic = stochastic_derivatives[-1] / current_price
   
   print(f"\nOption Greeks Approximation:")
   print(f"Regular Delta: {current_delta_regular:.4f}")
   print(f"Stochastic Delta: {current_delta_stochastic:.4f}")

🏦 **Example 2: Interest Rate Modeling**
---------------------------------------

**Application**: Modeling interest rate derivatives with mean reversion.

.. code-block:: python

   # Vasicek interest rate model simulation
   # dr_t = κ(θ - r_t)dt + σ dW_t
   # κ: mean reversion speed, θ: long-term mean, σ: volatility
   
   def vasicek_simulation(r0, kappa, theta, sigma, T, N):
       """Simulate Vasicek interest rate model"""
       dt = T / N
       t = np.linspace(0, T, N+1)
       r = np.zeros(N+1)
       r[0] = r0
       
       for i in range(N):
           dW = np.random.randn() * np.sqrt(dt)
           r[i+1] = r[i] + kappa*(theta - r[i])*dt + sigma*dW
       
       return t, r
   
   # Model parameters
   r0 = 0.03      # Initial rate (3%)
   kappa = 0.5    # Mean reversion speed
   theta = 0.04   # Long-term mean (4%)
   sigma = 0.01   # Volatility (1%)
   T = 5.0        # 5 years
   N = 1000       # Time steps
   
   # Simulate interest rate path
   np.random.seed(123)
   t, r = vasicek_simulation(r0, kappa, theta, sigma, T, N)
   
   # Fit with different interpolators
   spline_rates = SplineInterpolator(smoothing=0.001)
   spline_rates.fit(t, r)
   
   # Normal distribution appropriate for interest rates (can be negative)
   spline_rates.set_stochastic_link('normal', sigma=sigma, method='ito')
   
   # Compute rate derivatives (duration-like measures)
   rate_deriv_func = spline_rates.differentiate(order=1)
   rate_derivatives = rate_deriv_func(t)
   
   # Second derivatives (convexity-like measures)
   rate_second_deriv_func = spline_rates.differentiate(order=2)
   rate_second_derivatives = rate_second_deriv_func(t)
   
   # Theoretical drift: E[dr/dt] = κ(θ - r)
   theoretical_drift = kappa * (theta - r)
   
   print("Interest Rate Analysis:")
   print(f"Mean rate: {np.mean(r):.4f}")
   print(f"Rate volatility: {np.std(r):.4f}")
   print(f"Mean derivative: {np.mean(rate_derivatives):.6f}")
   print(f"Theoretical mean drift: {np.mean(theoretical_drift):.6f}")
   
   # Duration and convexity approximations
   current_rate = r[-1]
   duration_approx = -rate_derivatives[-1] / current_rate
   convexity_approx = rate_second_derivatives[-1] / current_rate
   
   print(f"\nBond Risk Measures (approximations):")
   print(f"Modified duration: {duration_approx:.4f}")
   print(f"Convexity: {convexity_approx:.4f}")

🧬 **Example 3: Population Dynamics with Uncertainty**
-----------------------------------------------------

**Application**: Biological population modeling with environmental stochasticity.

.. code-block:: python

   # Stochastic logistic growth with environmental noise
   # dN/dt = rN(1 - N/K) + σN ξ(t)
   # where ξ(t) is white noise
   
   def stochastic_logistic(N0, r, K, sigma, T, N_steps):
       """Simulate stochastic logistic growth"""
       dt = T / N_steps
       t = np.linspace(0, T, N_steps+1)
       N = np.zeros(N_steps+1)
       N[0] = N0
       
       for i in range(N_steps):
           # Deterministic growth
           growth = r * N[i] * (1 - N[i]/K) * dt
           # Stochastic perturbation
           noise = sigma * N[i] * np.random.randn() * np.sqrt(dt)
           N[i+1] = max(0, N[i] + growth + noise)  # Prevent negative population
       
       return t, N
   
   # Population parameters
   N0 = 10        # Initial population
   r = 0.5        # Growth rate
   K = 1000       # Carrying capacity
   sigma = 0.1    # Environmental noise strength
   T = 20         # Time horizon
   N_steps = 500  # Time steps
   
   # Simulate population
   np.random.seed(456)
   t, N = stochastic_logistic(N0, r, K, sigma, T, N_steps)
   
   # Fit interpolator
   population_interp = SplineInterpolator(smoothing=0.1)
   population_interp.fit(t, N)
   
   # Use gamma distribution (positive values, flexible shape)
   # Gamma parameters chosen to match population characteristics
   alpha = 4.0  # Shape parameter
   beta = alpha / np.mean(N)  # Rate parameter (scale = 1/beta)
   
   population_interp.set_stochastic_link('gamma', alpha=alpha, beta=beta, method='ito')
   
   # Compute growth rates
   growth_rate_func = population_interp.differentiate(order=1)
   stochastic_growth_rates = growth_rate_func(t)
   
   # Compare with deterministic logistic growth rate
   deterministic_growth_rates = r * N * (1 - N/K)
   
   # Per capita growth rates
   per_capita_stochastic = stochastic_growth_rates / N
   per_capita_deterministic = deterministic_growth_rates / N
   
   print("Population Dynamics Analysis:")
   print(f"Final population: {N[-1]:.0f}")
   print(f"Carrying capacity: {K}")
   print(f"Mean growth rate: {np.mean(stochastic_growth_rates):.2f}")
   print(f"Growth rate volatility: {np.std(stochastic_growth_rates):.2f}")
   
   # Find maximum growth rate period
   max_growth_idx = np.argmax(stochastic_growth_rates)
   max_growth_time = t[max_growth_idx]
   max_growth_pop = N[max_growth_idx]
   
   print(f"\nMaximum growth rate:")
   print(f"Time: {max_growth_time:.1f}")
   print(f"Population: {max_growth_pop:.0f}")
   print(f"Growth rate: {stochastic_growth_rates[max_growth_idx]:.2f}")
   
   # Environmental impact analysis
   stochastic_correction = np.mean(stochastic_growth_rates - deterministic_growth_rates)
   print(f"\nEnvironmental stochasticity effect:")
   print(f"Mean correction: {stochastic_correction:.3f}")

⚙️ **Advanced Stochastic Features**
----------------------------------

**Custom Link Functions**

Create custom probability distributions:

.. code-block:: python

   from pydelt.stochastic import StochasticLinkFunction
   
   class WeibullLink(StochasticLinkFunction):
       def __init__(self, shape, scale):
           self.shape = shape
           self.scale = scale
       
       def transform(self, x):
           return self.scale * (-np.log(1 - x))**(1/self.shape)
       
       def inverse_transform(self, y):
           return 1 - np.exp(-(y/self.scale)**self.shape)
       
       def derivative_transform(self, x, f_prime, f_double_prime=None, method='ito'):
           # Implement Weibull-specific corrections
           pass
   
   # Use custom link
   weibull_link = WeibullLink(shape=2.0, scale=1.0)
   interpolator.set_stochastic_link(weibull_link, method='ito')

**Method Comparison**

Compare Itô vs Stratonovich corrections:

.. code-block:: python

   # Fit same data with both methods
   interp_ito = SplineInterpolator(smoothing=0.1)
   interp_ito.fit(data_x, data_y)
   interp_ito.set_stochastic_link('lognormal', sigma=0.2, method='ito')
   
   interp_strat = SplineInterpolator(smoothing=0.1)
   interp_strat.fit(data_x, data_y)
   interp_strat.set_stochastic_link('lognormal', sigma=0.2, method='stratonovich')
   
   # Compare derivatives
   ito_deriv = interp_ito.differentiate(order=1)(eval_points)
   strat_deriv = interp_strat.differentiate(order=1)(eval_points)
   
   difference = np.mean(np.abs(ito_deriv - strat_deriv))
   print(f"Itô vs Stratonovich difference: {difference:.4f}")

**Parameter Sensitivity Analysis**

Analyze sensitivity to distribution parameters:

.. code-block:: python

   # Test different volatility levels
   sigmas = [0.1, 0.2, 0.3, 0.4]
   derivative_means = []
   
   for sigma in sigmas:
       interp = SplineInterpolator(smoothing=0.1)
       interp.fit(data_x, data_y)
       interp.set_stochastic_link('lognormal', sigma=sigma, method='ito')
       
       deriv_func = interp.differentiate(order=1)
       derivatives = deriv_func(eval_points)
       derivative_means.append(np.mean(derivatives))
   
   print("Sensitivity to volatility parameter:")
   for sigma, mean_deriv in zip(sigmas, derivative_means):
       print(f"σ = {sigma:.1f}: Mean derivative = {mean_deriv:.3f}")

🎓 **Best Practices**
--------------------

**Distribution Selection**:
1. **Stock Prices**: Log-normal (positive, multiplicative noise)
2. **Interest Rates**: Normal (can be negative, additive noise)
3. **Waiting Times**: Exponential or Gamma (positive, memoryless or aging)
4. **Proportions**: Beta (bounded between 0 and 1)
5. **Count Data**: Poisson (discrete, non-negative)

**Method Selection**:
- **Itô Calculus**: Mathematical finance, theoretical models
- **Stratonovich Calculus**: Physical systems, engineering applications
- **When in doubt**: Try both and compare results

**Parameter Estimation**:
Use maximum likelihood or method of moments to estimate distribution parameters from data:

.. code-block:: python

   from scipy import stats
   
   # Estimate log-normal parameters
   sigma_est = np.std(np.log(price_data))
   print(f"Estimated volatility: {sigma_est:.3f}")
   
   # Use estimated parameter
   interpolator.set_stochastic_link('lognormal', sigma=sigma_est, method='ito')

**Validation**:
Always compare with known analytical solutions when available:

.. code-block:: python

   # For geometric Brownian motion: E[dS/dt] = μS
   theoretical_drift = mu * stock_prices
   numerical_drift = stochastic_derivatives
   
   error = np.sqrt(np.mean((numerical_drift - theoretical_drift)**2))
   print(f"Drift estimation error: {error:.4f}")

⚠️ **Limitations & Considerations**
----------------------------------

**Numerical Stability**:
- Second derivatives required for Itô corrections may amplify noise
- Use appropriate smoothing parameters
- Consider robust interpolation methods for noisy data

**Model Assumptions**:
- Stochastic link functions assume specific probability distributions
- Validate distribution assumptions with data
- Consider model uncertainty in critical applications

**Computational Cost**:
- Stochastic corrections require additional derivative computations
- Itô method needs second derivatives (more expensive)
- Consider computational budget for real-time applications

**Interpretation**:
- Stochastic derivatives reflect expected behavior under distributional assumptions
- Results depend on chosen probability distribution and parameters
- Always validate against domain knowledge and empirical data

🔬 **Research Applications**
---------------------------

**Financial Engineering**:
- Option pricing with stochastic volatility
- Risk-neutral measure transformations
- Credit risk modeling with jump processes

**Scientific Computing**:
- Uncertainty quantification in differential equations
- Stochastic partial differential equations
- Climate modeling with random forcing

**Machine Learning**:
- Bayesian neural networks with stochastic derivatives
- Uncertainty-aware optimization
- Robust control with distributional assumptions

🔗 **Integration with Other Features**
------------------------------------

Stochastic computing combines powerfully with other pydelt features:

**Multivariate Stochastic Derivatives**:
.. code-block:: python

   # Apply stochastic links to multivariate functions
   from pydelt.multivariate import MultivariateDerivatives
   
   mv = MultivariateDerivatives(SplineInterpolator, smoothing=0.1)
   mv.fit(input_data, output_data)
   
   # Set stochastic link for underlying interpolator
   mv.interpolator_class.set_stochastic_link('lognormal', sigma=0.2)
   
   # Gradients now include stochastic corrections
   stochastic_gradient = mv.gradient()

**Neural Networks with Stochastic Links**:
.. code-block:: python

   # Combine automatic differentiation with stochastic corrections
   nn_stochastic = NeuralNetworkInterpolator(hidden_layers=[128, 64])
   nn_stochastic.fit(data_x, data_y)
   nn_stochastic.set_stochastic_link('gamma', alpha=2.0, beta=1.0)
   
   # Exact derivatives with probabilistic corrections
   exact_stochastic_deriv = nn_stochastic.differentiate(order=1)

This completes the progressive learning path from basic interpolation to advanced stochastic computing, providing a comprehensive framework for numerical differentiation with uncertainty quantification.
