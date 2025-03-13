# Time Series Derivatives

A Python package for calculating derivatives of time series data using various methods:

- Local Linear Approximation (LLA)
- Generalized Orthogonal Local Derivative (GOLD)
- Generalized Local Linear Approximation (GLLA)
- Functional Data Analysis (FDA)

## Installation

```bash
pip install timeseries_derivatives
```

## Usage

```python
import numpy as np
from timeseries_derivatives import lla, gold, glla, fda

# Generate sample data
time = np.linspace(0, 10, 500)
signal = np.sin(time) + np.random.normal(0, 0.1, size=time.shape)

# Calculate derivatives using different methods
derivative, steps = lla(time.tolist(), signal.tolist(), window_size=5)
result_gold = gold(signal, time, embedding=5, n=2)
result_glla = glla(signal, time, embedding=5, n=2)
result_fda = fda(signal, time)
```

## Methods

### LLA (Local Linear Approximation)
Uses min-normalization and linear regression within a sliding window to estimate derivatives.

### GOLD (Generalized Orthogonal Local Derivative)
Implements the method described in:
- https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4940142/
- https://www.tandfonline.com/doi/abs/10.1080/00273171.2010.498294

### GLLA (Generalized Local Linear Approximation)
A generalized version of local linear approximation that can calculate higher-order derivatives.

### FDA (Functional Data Analysis)
Uses spline-based smoothing to calculate derivatives.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
