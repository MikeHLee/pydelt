# Chapter 8: Applications Under Error

> *"Theory is clean. Reality is messy. The art is knowing how much error you can tolerate."*

## Connection to the Central Thesis

We've built the theoretical foundation. Now we apply it to real problems—each with its own error sources, constraints, and tolerances.

The unifying theme: **every application involves approximating unknown dynamics from imperfect data**. Success depends on:
1. Understanding the error sources
2. Choosing appropriate methods
3. Quantifying uncertainty
4. Validating against domain knowledge

This chapter surveys applications across domains, emphasizing practical considerations.

## Sensor Fusion and State Estimation

### The Problem

Multiple sensors measure the same system:
- GPS: Position, noisy, 1 Hz
- IMU: Acceleration, less noisy, 100 Hz
- Wheel encoders: Velocity, very noisy

How do you combine them to estimate position, velocity, and acceleration?

### The Kalman Filter Approach

Model the system as:
$$x_{k+1} = A x_k + B u_k + w_k$$
$$y_k = C x_k + v_k$$

The Kalman filter optimally fuses measurements when noise is Gaussian.

### Where PyDelt Fits

PyDelt complements Kalman filtering:
1. **Preprocessing**: Smooth individual sensor streams before fusion
2. **Derivative estimation**: Get velocity from position, acceleration from velocity
3. **Model identification**: Discover A, B, C from data

```python
from pydelt.interpolation import SplineInterpolator
import numpy as np

# GPS position data (1 Hz, noisy)
t_gps = np.arange(0, 100, 1.0)
pos_gps = true_position(t_gps) + 2.0 * np.random.randn(len(t_gps))

# Smooth and differentiate
interp = SplineInterpolator(smoothing=10.0)  # Heavy smoothing for GPS
interp.fit(t_gps, pos_gps)

# Estimate velocity and acceleration
vel_est = interp.differentiate(order=1)
acc_est = interp.differentiate(order=2)

# Evaluate at high frequency for fusion with IMU
t_fine = np.arange(0, 100, 0.01)
vel_from_gps = vel_est(t_fine)
acc_from_gps = acc_est(t_fine)
```

### Error Considerations

| Sensor | Error Source | Typical Magnitude |
|--------|--------------|-------------------|
| GPS | Multipath, atmospheric | 2-5 m position |
| IMU | Bias drift, noise | Accumulates over time |
| Encoders | Wheel slip, quantization | 1-5% velocity |

**Key insight**: Differentiation amplifies GPS noise. Heavy smoothing is essential.

## Financial Time Series

### The Problem

Estimate instantaneous volatility, drift, and Greeks from price data.

### Challenges Specific to Finance

1. **Non-stationarity**: Parameters change over time
2. **Jumps**: Prices can gap (earnings, news)
3. **Microstructure noise**: Bid-ask bounce at high frequency
4. **Fat tails**: Extreme events more common than Gaussian

### Volatility Estimation

```python
from pydelt.interpolation import LowessInterpolator
import numpy as np

# Log returns
log_prices = np.log(prices)
returns = np.diff(log_prices)

# Realized volatility (rolling window)
window = 20
realized_vol = np.array([
    np.std(returns[max(0,i-window):i]) * np.sqrt(252)
    for i in range(1, len(returns)+1)
])

# Smooth volatility estimate
lowess = LowessInterpolator(frac=0.1)
lowess.fit(np.arange(len(realized_vol)), realized_vol)
vol_smooth = lowess(np.arange(len(realized_vol)))
```

### Greeks from Market Data

Option prices V(S, t, σ, r) depend on underlying S, time t, volatility σ, rate r.

**Greeks** are partial derivatives:
- Delta = ∂V/∂S (hedge ratio)
- Gamma = ∂²V/∂S² (convexity)
- Vega = ∂V/∂σ (volatility sensitivity)
- Theta = ∂V/∂t (time decay)

```python
from pydelt.multivariate import MultivariateDerivatives
from pydelt.interpolation import SplineInterpolator

# Option prices at different strikes and maturities
# strikes: array of strike prices
# maturities: array of times to expiration
# prices: 2D array of option prices

# Flatten for multivariate fitting
inputs = np.column_stack([
    strikes.flatten(),
    maturities.flatten()
])
outputs = prices.flatten()

mv = MultivariateDerivatives(SplineInterpolator, smoothing=0.01)
mv.fit(inputs, outputs)

# Delta = ∂V/∂K (approximately ∂V/∂S with sign flip)
gradient = mv.gradient()
delta_gamma = gradient(inputs)  # [:, 0] is ∂V/∂K
```

### Error Considerations

| Greek | Error Source | Mitigation |
|-------|--------------|------------|
| Delta | Price noise, discrete strikes | Smooth across strikes |
| Gamma | Second derivative amplifies noise | Heavy smoothing |
| Vega | Implied vol uncertainty | Use vol surface |
| Theta | Time discretization | Interpolate in time |

## Physics Simulations

### The Problem

Extract physical quantities (forces, energies, transport coefficients) from simulation trajectories.

### Molecular Dynamics

Atoms move according to Newton's equations:
$$m_i \ddot{r}_i = -\nabla_i U(r_1, ..., r_N)$$

From trajectories r_i(t), estimate:
- Velocities: ṙ_i
- Forces: F_i = mᵢr̈_i
- Diffusion coefficients: from velocity autocorrelation

```python
from pydelt.interpolation import SplineInterpolator
import numpy as np

# Particle trajectory from MD simulation
# positions: (n_frames, n_atoms, 3)
# times: (n_frames,)

def compute_velocities(times, positions):
    """Compute velocities for all atoms."""
    n_frames, n_atoms, n_dims = positions.shape
    velocities = np.zeros_like(positions)
    
    for atom in range(n_atoms):
        for dim in range(n_dims):
            interp = SplineInterpolator(smoothing=0.01)
            interp.fit(times, positions[:, atom, dim])
            velocities[:, atom, dim] = interp.differentiate(order=1)(times)
    
    return velocities

def compute_diffusion_coefficient(times, positions):
    """Compute diffusion coefficient from MSD."""
    # Mean squared displacement
    msd = np.mean(np.sum((positions - positions[0])**2, axis=2), axis=1)
    
    # D = lim_{t→∞} MSD / (6t) in 3D
    # Fit linear region
    interp = SplineInterpolator(smoothing=1.0)
    interp.fit(times, msd)
    slope = interp.differentiate(order=1)(times[-1])
    
    return slope / 6
```

### Fluid Dynamics

From velocity fields u(x, t), compute:
- Vorticity: ω = ∇ × u
- Strain rate: S = (∇u + ∇uᵀ)/2
- Dissipation: ε = 2ν S:S

```python
from pydelt.multivariate import MultivariateDerivatives

# Velocity field on grid
# u_x, u_y: (nx, ny) arrays

# Compute velocity gradients
def compute_vorticity(x, y, u_x, u_y):
    """Compute vorticity ω = ∂u_y/∂x - ∂u_x/∂y."""
    # Flatten grid
    X, Y = np.meshgrid(x, y)
    inputs = np.column_stack([X.flatten(), Y.flatten()])
    
    # Fit u_x and u_y
    mv_ux = MultivariateDerivatives(SplineInterpolator, smoothing=0.1)
    mv_ux.fit(inputs, u_x.flatten())
    
    mv_uy = MultivariateDerivatives(SplineInterpolator, smoothing=0.1)
    mv_uy.fit(inputs, u_y.flatten())
    
    # Gradients
    grad_ux = mv_ux.gradient()(inputs)  # [:, 0] = ∂u_x/∂x, [:, 1] = ∂u_x/∂y
    grad_uy = mv_uy.gradient()(inputs)  # [:, 0] = ∂u_y/∂x, [:, 1] = ∂u_y/∂y
    
    # Vorticity
    vorticity = grad_uy[:, 0] - grad_ux[:, 1]
    return vorticity.reshape(u_x.shape)
```

## Biomedical Signals

### The Problem

Extract physiological information from noisy biosignals (ECG, EEG, EMG).

### ECG Analysis

Heart rate variability (HRV) requires:
1. Detect R-peaks
2. Compute RR intervals
3. Estimate instantaneous heart rate
4. Compute HRV metrics (SDNN, RMSSD, frequency bands)

```python
from pydelt.interpolation import SplineInterpolator
import numpy as np

def analyze_hrv(r_peak_times):
    """Compute HRV metrics from R-peak times."""
    # RR intervals
    rr_intervals = np.diff(r_peak_times)
    rr_times = r_peak_times[:-1] + rr_intervals / 2
    
    # Instantaneous heart rate
    hr = 60 / rr_intervals  # beats per minute
    
    # Smooth for continuous HR estimate
    interp = SplineInterpolator(smoothing=0.5)
    interp.fit(rr_times, hr)
    
    # Heart rate variability = derivative of HR
    hr_variability = interp.differentiate(order=1)
    
    # Time-domain metrics
    sdnn = np.std(rr_intervals)
    rmssd = np.sqrt(np.mean(np.diff(rr_intervals)**2))
    
    return {
        'hr_interp': interp,
        'hr_variability': hr_variability,
        'sdnn': sdnn,
        'rmssd': rmssd
    }
```

### EEG Processing

Brain signals require:
- Artifact removal (eye blinks, muscle)
- Frequency band extraction (alpha, beta, gamma)
- Phase and amplitude estimation

**Key insight**: EEG is inherently noisy. Heavy smoothing is appropriate for slow features (< 30 Hz), but fast oscillations require careful bandpass filtering first.

## Machine Learning

### Gradient-Based Optimization

All of deep learning is gradient descent:
$$\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)$$

PyDelt's relevance:
1. **Loss landscape analysis**: Estimate Hessian eigenvalues from loss samples
2. **Hyperparameter sensitivity**: ∂(validation loss)/∂(hyperparameter)
3. **Neural network analysis**: Jacobian of network outputs

### Physics-Informed Neural Networks (PINNs)

Embed physical laws in the loss:
$$L = L_{\text{data}} + \lambda L_{\text{physics}}$$

where:
$$L_{\text{physics}} = \left\|\frac{\partial u}{\partial t} - F\left(u, \frac{\partial u}{\partial x}, ...\right)\right\|^2$$

PyDelt can:
1. **Precompute derivatives** for training data
2. **Validate PINN predictions** against numerical derivatives
3. **Estimate derivatives** where PINN is uncertain

### Sensitivity Analysis

How do model outputs depend on inputs?

```python
from pydelt.multivariate import MultivariateDerivatives
from pydelt.interpolation import SplineInterpolator

# Model predictions at various input points
# inputs: (n_samples, n_features)
# outputs: (n_samples,) or (n_samples, n_outputs)

mv = MultivariateDerivatives(SplineInterpolator, smoothing=0.1)
mv.fit(inputs, outputs)

# Sensitivity = gradient magnitude
gradient = mv.gradient()
sensitivities = gradient(inputs)

# Feature importance = average |∂output/∂feature|
feature_importance = np.mean(np.abs(sensitivities), axis=0)
```

## Error Budget Framework

### Systematic Approach

For any application, construct an **error budget**:

| Error Source | Magnitude | Mitigation |
|--------------|-----------|------------|
| Measurement noise | σ_meas | Smoothing, filtering |
| Sampling rate | Δt | Interpolation |
| Model mismatch | ε_model | Better model class |
| Numerical precision | ε_num | Usually negligible |
| **Total** | √(Σε²) | Propagate through pipeline |

### Propagation Through Derivatives

If measurement error is σ and you estimate f' via finite differences with step h:

$$\sigma_{f'} \approx \frac{\sqrt{2} \sigma}{h}$$

For smoothing splines with n points:

$$\sigma_{f'} \approx C \cdot n^{-2/5} \cdot \sigma$$

### Validation Strategies

1. **Synthetic data**: Generate from known system, verify recovery
2. **Cross-validation**: Hold out data, predict derivatives
3. **Physical constraints**: Do derivatives satisfy conservation laws?
4. **Sensitivity analysis**: How do results change with parameters?

## Practical Checklist

Before applying PyDelt to a new problem:

### 1. Understand Your Data
- [ ] What is the sampling rate?
- [ ] What is the noise level?
- [ ] Are there outliers or gaps?
- [ ] Is the underlying process smooth?

### 2. Choose Appropriate Method
- [ ] Low noise → Splines or finite differences
- [ ] High noise → LOWESS or heavy smoothing
- [ ] Outliers → Robust methods (LOWESS)
- [ ] Complex patterns → Neural networks

### 3. Validate Results
- [ ] Do derivatives have correct sign and magnitude?
- [ ] Are results stable to smoothing parameter?
- [ ] Do they satisfy physical constraints?
- [ ] Cross-validation error acceptable?

### 4. Quantify Uncertainty
- [ ] Estimate error from noise propagation
- [ ] Bootstrap confidence intervals
- [ ] Sensitivity to method choice

## Key Takeaways

1. **Every application has domain-specific error sources**
2. **Smoothing parameters should reflect noise level**
3. **Validation against domain knowledge is essential**
4. **Error budgets help quantify uncertainty**
5. **No single method works for all applications**

## Exercises

1. **Sensor fusion**: Simulate GPS (1 Hz, σ=2m) and IMU (100 Hz, σ=0.1 m/s²) data. Use PyDelt to estimate velocity from GPS. Compare to integrated IMU.

2. **Option Greeks**: Download option chain data. Estimate Delta and Gamma across strikes. Compare to Black-Scholes theoretical values.

3. **HRV analysis**: Use a public ECG dataset. Compute instantaneous heart rate and HRV metrics. Validate against published norms.

4. **Error budget**: For your favorite application, construct an error budget. Identify the dominant error source.

---

*Previous: [← Stochastic Calculus](07_stochastic_calculus.md) | Back to: [Introduction](00_introduction.md)*

---

## Further Reading

### General Numerical Methods
- Press et al., *Numerical Recipes*
- Trefethen, *Approximation Theory and Approximation Practice*

### Domain-Specific
- **Finance**: Hull, *Options, Futures, and Other Derivatives*
- **Physics**: Frenkel & Smit, *Understanding Molecular Simulation*
- **Biomedical**: Rangayyan, *Biomedical Signal Analysis*
- **ML**: Goodfellow et al., *Deep Learning*

### PyDelt Documentation
- [API Reference](../api.rst)
- [Examples](../examples.rst)
- [FAQ](../faq.rst)
