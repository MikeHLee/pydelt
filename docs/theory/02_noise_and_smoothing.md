# Chapter 2: Noise and Smoothing

> *"Differentiation amplifies noise. Smoothing reduces noise but introduces bias. There is no free lunch."*

## Connection to the Central Thesis

Real measurements are never clean. Sensors have precision limits. Environments fluctuate. Sampling introduces quantization. The data we observe is:

$$y_i = x(t_i) + \epsilon_i$$

where x(t) is the true state and ε is measurement noise.

This creates a fundamental tension: **differentiation amplifies noise**, but **smoothing distorts the signal**. The approximation paradigm addresses this by treating numerical differentiation as a **regularized inverse problem**—we accept some bias to control variance.

The smoothing parameter encodes our belief about the true system: "the underlying dynamics are probably smooth, so extreme oscillations in the derivative are likely noise, not signal."

## The Fundamental Tradeoff

Every numerical differentiation method faces the same dilemma:

- **No smoothing**: Derivatives are noisy (high variance)
- **Heavy smoothing**: Derivatives are biased (systematic error)

This is the **bias-variance tradeoff** applied to differentiation.

## Why Differentiation Amplifies Noise

### Frequency Domain Perspective

Differentiation in the frequency domain is multiplication by iω:

$$\mathcal{F}[f'(t)] = i\omega \cdot \mathcal{F}[f(t)]$$

High-frequency components get amplified. Noise is typically high-frequency. Therefore, differentiation amplifies noise.

### Quantitative Analysis

If your signal is y(t) = f(t) + ε(t) where ε is white noise with power spectral density S_ε:

- Signal derivative power: |iω|² S_f(ω) = ω² S_f(ω)
- Noise derivative power: |iω|² S_ε = ω² S_ε

The signal-to-noise ratio after differentiation:

$$\text{SNR}_{\text{derivative}} = \frac{\omega^2 S_f(\omega)}{\omega^2 S_\epsilon} = \frac{S_f(\omega)}{S_\epsilon}$$

The SNR is unchanged only if signal and noise have the same spectrum. In practice, noise dominates at high frequencies, so differentiation destroys SNR.

## Smoothing as Regularization

### The Regularization Framework

Instead of solving:
$$\text{Find } f' \text{ such that } f \text{ matches data}$$

Solve:
$$\text{Minimize } \|f - \text{data}\|^2 + \lambda \cdot \text{Roughness}(f)$$

where λ controls the bias-variance tradeoff:
- λ = 0: Interpolate exactly (high variance)
- λ → ∞: Maximally smooth (high bias)

### Common Roughness Penalties

| Penalty | Formula | Effect |
|---------|---------|--------|
| First derivative | ∫(f')² dt | Penalizes slopes |
| Second derivative | ∫(f'')² dt | Penalizes curvature |
| Total variation | ∫|f'| dt | Preserves edges |

The second derivative penalty is most common—it produces **cubic smoothing splines**.

## Smoothing Methods

### 1. Moving Average

The simplest smoother: replace each point with the average of its neighbors.

```python
import numpy as np

def moving_average(y, window):
    """Simple moving average."""
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode='same')
```

**Properties**:
- Reduces variance by factor of 1/window
- Introduces phase lag
- Blurs sharp features
- Not optimal for any particular noise model

### 2. Savitzky-Golay Filter

Fit a polynomial locally, use the polynomial's value (or derivative) at center.

```python
from scipy.signal import savgol_filter

# Smooth and differentiate in one step
y_smooth = savgol_filter(y, window_length=11, polyorder=3)
dy_smooth = savgol_filter(y, window_length=11, polyorder=3, deriv=1, delta=dt)
```

**Properties**:
- Preserves moments up to polynomial order
- Better edge preservation than moving average
- Analytical derivatives from fitted polynomial
- Window size and polynomial order are tuning parameters

### 3. Gaussian Smoothing

Convolve with a Gaussian kernel:

$$y_{\text{smooth}}(t) = \int y(\tau) \cdot \frac{1}{\sqrt{2\pi}\sigma} e^{-(t-\tau)^2/2\sigma^2} d\tau$$

```python
from scipy.ndimage import gaussian_filter1d

y_smooth = gaussian_filter1d(y, sigma=2.0)
```

**Properties**:
- Optimal for Gaussian noise (in MSE sense)
- No ringing artifacts
- σ controls smoothing strength
- Derivative of Gaussian = smoothed derivative

### 4. Kernel Regression (Nadaraya-Watson)

Weighted average with kernel weights:

$$\hat{f}(x) = \frac{\sum_i K\left(\frac{x - x_i}{h}\right) y_i}{\sum_i K\left(\frac{x - x_i}{h}\right)}$$

where K is a kernel (Gaussian, Epanechnikov, etc.) and h is the bandwidth.

```python
from sklearn.kernel_ridge import KernelRidge

# Kernel regression
kr = KernelRidge(kernel='rbf', gamma=1.0)
kr.fit(t.reshape(-1, 1), y)
y_smooth = kr.predict(t.reshape(-1, 1))
```

### 5. Spline Smoothing

Minimize:

$$\sum_i (y_i - f(t_i))^2 + \lambda \int (f''(t))^2 dt$$

The solution is a **natural cubic spline** with knots at data points.

```python
from scipy.interpolate import UnivariateSpline

# Smoothing spline (s controls smoothing)
spline = UnivariateSpline(t, y, s=len(y)*noise_variance)
y_smooth = spline(t)
dy_smooth = spline.derivative()(t)
```

## The Bias-Variance Decomposition

For any estimator f̂ of true function f:

$$\mathbb{E}[(f̂ - f)^2] = \text{Bias}^2 + \text{Variance}$$

where:
- **Bias** = E[f̂] - f (systematic error from smoothing)
- **Variance** = E[(f̂ - E[f̂])²] (random error from noise)

### Visualizing the Tradeoff

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

# True function
t = np.linspace(0, 2*np.pi, 200)
y_true = np.sin(t)
dy_true = np.cos(t)

# Noisy observations
np.random.seed(42)
noise = 0.1
y_noisy = y_true + noise * np.random.randn(len(t))

# Different smoothing levels
smoothing_values = [0.01, 0.1, 1.0, 10.0]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for ax, s in zip(axes, smoothing_values):
    spline = UnivariateSpline(t, y_noisy, s=s*len(t))
    dy_est = spline.derivative()(t)
    
    ax.plot(t, dy_true, 'k-', label='True derivative', linewidth=2)
    ax.plot(t, dy_est, 'r-', label=f'Estimated (s={s})', alpha=0.8)
    ax.set_title(f'Smoothing = {s}')
    ax.legend()
    ax.set_xlabel('t')
    ax.set_ylabel("f'(t)")

plt.tight_layout()
plt.show()
```

## Choosing the Smoothing Parameter

### 1. Cross-Validation

Leave out data points, predict them, minimize prediction error:

$$\text{CV}(\lambda) = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{f}_{-i}(x_i))^2$$

where f̂₋ᵢ is fitted without point i.

### 2. Generalized Cross-Validation (GCV)

Efficient approximation to leave-one-out CV:

$$\text{GCV}(\lambda) = \frac{\frac{1}{n}\sum_i (y_i - \hat{f}(x_i))^2}{(1 - \text{tr}(S)/n)^2}$$

where S is the smoothing matrix (ŷ = Sy).

### 3. AIC/BIC

Information criteria that penalize complexity:

$$\text{AIC} = n \log(\text{RSS}/n) + 2 \cdot \text{df}$$
$$\text{BIC} = n \log(\text{RSS}/n) + \log(n) \cdot \text{df}$$

where df is the effective degrees of freedom.

### 4. Known Noise Level

If you know σ², set smoothing so residual variance ≈ σ²:

```python
# For UnivariateSpline, s ≈ n * sigma^2
spline = UnivariateSpline(t, y, s=len(t) * sigma**2)
```

## Differentiation-Specific Considerations

### Smoothing for Derivatives vs. Values

The optimal smoothing for estimating f(x) differs from optimal smoothing for f'(x):

- **For values**: Moderate smoothing
- **For derivatives**: More smoothing needed (noise amplification)
- **For second derivatives**: Even more smoothing

Rule of thumb: Increase smoothing parameter by ~2-4x per derivative order.

### Boundary Effects

Smoothing methods often perform poorly near boundaries:
- Less data available for local averaging
- Splines may oscillate
- Derivatives especially unreliable

**Solutions**:
- Extend data with padding (reflection, extrapolation)
- Use boundary-aware methods
- Trim boundary regions from analysis

### Preserving Features

Heavy smoothing can destroy important features:
- Peaks get flattened
- Edges get blurred
- Oscillations get damped

**Solutions**:
- Adaptive smoothing (less smoothing where curvature is high)
- Edge-preserving methods (total variation, bilateral filter)
- Physics-informed constraints

## PyDelt's Smoothing Options

```python
from pydelt.interpolation import SplineInterpolator, LowessInterpolator

# Spline with explicit smoothing
spline = SplineInterpolator(smoothing=0.1)
spline.fit(t, y)
dy = spline.differentiate(order=1)(t)

# LOWESS (robust to outliers)
lowess = LowessInterpolator(frac=0.1)
lowess.fit(t, y)
dy = lowess.differentiate(order=1)(t)
```

## Practical Guidelines

### When to Use What

| Situation | Recommended Method |
|-----------|-------------------|
| Gaussian noise, smooth function | Spline smoothing |
| Outliers present | LOWESS/LOESS |
| Sharp features to preserve | Savitzky-Golay or adaptive |
| Very high noise | Heavy smoothing + accept bias |
| Unknown noise level | Cross-validation |

### Diagnostic Checks

1. **Residual analysis**: Residuals should look like noise, not signal
2. **Sensitivity analysis**: How much do results change with smoothing parameter?
3. **Physical plausibility**: Do derivatives make sense (sign, magnitude)?

## Key Takeaways

1. **Differentiation amplifies noise** by factor ~1/h or ~ω in frequency domain
2. **Smoothing reduces variance** but introduces bias
3. **Optimal smoothing** depends on noise level and derivative order
4. **Cross-validation** provides data-driven smoothing selection
5. **Boundaries and features** require special attention

## Exercises

1. **Bias-variance visualization**: For sin(x) + noise, plot bias² and variance of derivative estimate vs. smoothing parameter. Find the optimal smoothing.

2. **Cross-validation**: Implement leave-one-out CV for spline smoothing. Compare to GCV.

3. **Derivative order**: For the same data, find optimal smoothing for f, f', and f''. How do they differ?

4. **Outlier robustness**: Add outliers to your data. Compare spline vs. LOWESS derivative estimates.

---

*Previous: [← Numerical Differentiation](01_numerical_differentiation.md) | Next: [Interpolation Methods →](03_interpolation_methods.md)*
