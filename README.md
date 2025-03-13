# PyDelt

A Python package for calculating derivatives and integrals of time series data using various methods:

- Local Linear Approximation (LLA)
- Generalized Orthogonal Local Derivative (GOLD)
- Generalized Local Linear Approximation (GLLA)
- Functional Data Analysis (FDA)
- Integration with Error Estimation

## Installation

```bash
pip install pydelt
```

## Usage

```python
import numpy as np
from pydelt import lla, gold, glla, fda, integrate_derivative, integrate_derivative_with_error

# Generate sample data
time = np.linspace(0, 10, 500)
signal = np.sin(time) + np.random.normal(0, 0.1, size=time.shape)

# Calculate derivatives using different methods
derivative, steps = lla(time.tolist(), signal.tolist(), window_size=5)
result_gold = gold(signal, time, embedding=5, n=2)
result_glla = glla(signal, time, embedding=5, n=2)
result_fda = fda(signal, time)

# Reconstruct signal through integration
reconstructed = integrate_derivative(time, derivative, initial_value=signal[0])

# Get integration with error estimates
reconstructed_with_error, error = integrate_derivative_with_error(time, derivative, initial_value=signal[0])
```

## Methods
Implements the method described in:
- https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4940142/
- https://www.tandfonline.com/doi/abs/10.1080/00273171.2010.498294

### LLA (Local Linear Approximation)
A sliding window approach that uses min-normalization and linear regression to estimate derivatives. By normalizing the data within each window relative to its minimum value, LLA reduces the impact of local offsets and trends. The method is particularly effective for data with varying baselines or drift, and provides robust first-order derivative estimates even in the presence of moderate noise.

### GLLA (Generalized Local Linear Approximation)
An extension of the LLA method that enables calculation of higher-order derivatives using a generalized linear approximation framework. GLLA uses a local polynomial fit of arbitrary order and combines it with a sliding window approach. This method is particularly useful when you need consistent estimates of multiple orders of derivatives simultaneously, and it maintains good numerical stability even for higher-order derivatives.

### GOLD (Generalized Orthogonal Local Derivative)
A robust method for calculating derivatives using orthogonal polynomials. GOLD constructs a local coordinate system at each point using orthogonal polynomials, which helps reduce the impact of noise and provides accurate estimates of higher-order derivatives. The method is particularly effective for noisy time series data and can estimate multiple orders of derivatives simultaneously.

### FDA (Functional Data Analysis)
A sophisticated approach that uses spline-based smoothing to represent the time series as a continuous function. FDA automatically determines an optimal smoothing parameter based on the data characteristics, balancing the trade-off between smoothness and fidelity to the original data. This method is particularly well-suited for smooth underlying processes and can provide consistent derivatives up to the order of the chosen spline basis.

### Integration Methods
The package provides two integration methods:

#### Basic Integration (integrate_derivative)
Uses the trapezoidal rule to integrate a derivative signal and reconstruct the original time series. You can specify an initial value to match known boundary conditions.

#### Integration with Error Estimation (integrate_derivative_with_error)
Performs integration using both trapezoidal and rectangular rules to provide an estimate of the integration error. This is particularly useful when working with noisy or uncertain derivative data.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
