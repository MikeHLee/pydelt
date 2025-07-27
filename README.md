# pydelt: Python Derivatives for Time Series

[![PyPI version](https://badge.fury.io/py/pydelt.svg)](https://badge.fury.io/py/pydelt)
[![Documentation Status](https://readthedocs.org/projects/pydelt/badge/?version=latest)](https://pydelt.readthedocs.io/en/latest/?badge=latest)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**pydelt** is a comprehensive Python package for calculating derivatives of time series data using various numerical and machine learning methods.

## 🚀 Key Features

- **Multiple derivative methods**: Finite differences, local linear approximation, Gaussian processes, and neural networks
- **Advanced interpolation**: Splines, LOWESS, LOESS, and neural network-based interpolation  
- **Automatic differentiation**: PyTorch and TensorFlow backends for gradient computation
- **Integration capabilities**: Numerical integration with error estimation
- **Multivariate support**: Handle multi-dimensional time series data
- **Robust error handling**: Comprehensive input validation and error messages

## Installation

```bash
pip install pydelt
```

## 📚 Quick Start

```python
import numpy as np
from pydelt.derivatives import lla, fda
from pydelt.interpolation import spline_interpolation
from pydelt.integrals import integrate_derivative

# Generate sample data
time = np.linspace(0, 2*np.pi, 100)
signal = np.sin(time)

# Calculate derivative using Local Linear Approximation
result = lla(time.tolist(), signal.tolist(), window_size=5)
derivative = result[0]  # Extract derivatives

# The derivative of sin(x) should be approximately cos(x)
expected = np.cos(time)
print(f"Max error: {np.max(np.abs(derivative - expected)):.4f}")

# Advanced: Neural network derivatives (requires PyTorch/TensorFlow)
try:
    from pydelt.autodiff import neural_network_derivative
    nn_derivative = neural_network_derivative(
        time, signal, 
        framework='pytorch',
        epochs=500
    )
    # Evaluate at specific points
    test_points = np.linspace(0.5, 5.5, 20)
    derivatives_at_points = nn_derivative(test_points)
except ImportError:
    print("Install PyTorch or TensorFlow for neural network support")
```

## 📚 Documentation

For detailed documentation, examples, and API reference, visit:

**🔗 [https://pydelt.readthedocs.io/](https://pydelt.readthedocs.io/)**

### Quick Links

- **[Installation Guide](https://pydelt.readthedocs.io/en/latest/installation.html)** - Detailed installation instructions
- **[Quick Start](https://pydelt.readthedocs.io/en/latest/quickstart.html)** - Get up and running quickly
- **[Examples](https://pydelt.readthedocs.io/en/latest/examples.html)** - Comprehensive usage examples
- **[API Reference](https://pydelt.readthedocs.io/en/latest/api.html)** - Complete function documentation
- **[Changelog](https://pydelt.readthedocs.io/en/latest/changelog.html)** - Version history and updates

## 🛠️ Methods
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

## 🧪 Testing

PyDelt includes a comprehensive test suite to verify the correctness of its implementations. To run the tests:

```bash
# Activate your virtual environment (if using one)
source venv/bin/activate

# Install pytest if not already installed
pip install pytest

# Run all tests
python -m pytest src/pydelt/tests/

# Run specific test files
python -m pytest src/pydelt/tests/test_derivatives.py
python -m pytest src/pydelt/tests/test_integrals.py
```

The test suite includes verification of:
- Derivative calculation accuracy for various methods
- Integration accuracy and error estimation
- Input validation and error handling
- Edge cases and boundary conditions

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/MikeHLee/pydelt.git
cd pydelt

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .

# Install development dependencies
pip install pytest sphinx sphinx-rtd-theme

# Run tests
python -m pytest src/pydelt/tests/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

- **Documentation**: [https://pydelt.readthedocs.io/](https://pydelt.readthedocs.io/)
- **Issues**: [GitHub Issues](https://github.com/MikeHLee/pydelt/issues)
- **PyPI**: [https://pypi.org/project/pydelt/](https://pypi.org/project/pydelt/)
