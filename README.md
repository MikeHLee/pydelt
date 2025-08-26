# PyDelt: Advanced Numerical Function Interpolation & Differentiation

[![PyPI version](https://badge.fury.io/py/pydelt.svg)](https://badge.fury.io/py/pydelt)
[![Documentation Status](https://readthedocs.org/projects/pydelt/badge/?version=latest)](https://pydelt.readthedocs.io/en/latest/?badge=latest)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**PyDelt** transforms raw data into mathematical insights through advanced numerical interpolation and differentiation. Whether you're analyzing experimental measurements, financial time series, or complex dynamical systems, PyDelt provides the tools to extract derivatives, gradients, and higher-order mathematical properties with precision and reliability.

## Why PyDelt?

Traditional numerical differentiation is notoriously unstable - small changes in data can cause large changes in derivatives. PyDelt solves this through smart smoothing that preserves important features while reducing noise, multiple methods so you can choose the best approach for your data, and a unified interface that makes comparison and validation straightforward.

## 🎯 Key Features

• **Universal Interface**: Every method uses the same `.fit().differentiate()` pattern - easy to learn, easy to switch
• **From Simple to Sophisticated**: Start with splines, scale to neural networks with automatic differentiation
• **Multivariate Ready**: Gradients, Jacobians, Hessians, and Laplacians for functions of multiple variables
• **Noise Robust**: Built-in smoothing and validation ensure reliable results from imperfect data
• **Stochastic Extensions**: Proper handling of financial derivatives with Itô and Stratonovich corrections
• **Production Ready**: Comprehensive error handling, extensive testing, and clear documentation

## Installation

```bash
pip install pydelt
```

## 📚 Quick Start

### Universal Differentiation Interface

```python
import numpy as np
from pydelt.interpolation import SplineInterpolator, LlaInterpolator

# Generate sample data
x = np.linspace(0, 2*np.pi, 100)
y = np.sin(x)

# Modern API: Universal differentiation interface
interpolator = SplineInterpolator(smoothing=0.1)
interpolator.fit(x, y)
derivative_func = interpolator.differentiate(order=1)

# Evaluate derivative at any points
eval_points = np.linspace(0, 2*np.pi, 50)
derivatives = derivative_func(eval_points)
print(f"Max error: {np.max(np.abs(derivatives - np.cos(eval_points))):.4f}")
```

### Multivariate Calculus

```python
from pydelt.multivariate import MultivariateDerivatives

# Generate 2D data: f(x,y) = x² + y²
x = np.linspace(-2, 2, 20)
y = np.linspace(-2, 2, 20)
X, Y = np.meshgrid(x, y)
Z = X**2 + Y**2

# Prepare data for multivariate derivatives
input_data = np.column_stack([X.flatten(), Y.flatten()])
output_data = Z.flatten()

# Compute gradient: ∇f = [2x, 2y]
mv = MultivariateDerivatives(SplineInterpolator, smoothing=0.1)
mv.fit(input_data, output_data)
gradient_func = mv.gradient()

# Evaluate at specific points
test_points = np.array([[1.0, 1.0], [0.5, -0.5]])
gradients = gradient_func(test_points)
print(f"Gradient at (1,1): {gradients[0]} (expected: [2, 2])")
```

### Legacy API (Traditional Methods)

```python
from pydelt.derivatives import lla, fda

# Calculate derivative using Local Linear Approximation
time = np.linspace(0, 2*np.pi, 100)
signal = np.sin(time)
result = lla(time.tolist(), signal.tolist(), window_size=5)
derivative = result[0]  # Extract derivatives

# Neural network derivatives (requires PyTorch/TensorFlow)
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

## 🧮 Methods & Mathematical Foundations

### Universal Differentiation Interface
All interpolators in pydelt implement a consistent `.differentiate(order, mask)` method that returns a callable function for evaluating derivatives at any points. This unified API allows seamless switching between different methods while maintaining consistent behavior.

```python
# Universal pattern for all interpolators
interpolator = InterpolatorClass(**params)
interpolator.fit(input_data, output_data)
derivative_func = interpolator.differentiate(order=1, mask=None)
derivatives = derivative_func(eval_points)
```

### LLA (Local Linear Approximation)
A sliding window approach that uses min-normalization and linear regression to estimate derivatives. By normalizing the data within each window relative to its minimum value, LLA reduces the impact of local offsets and trends. The method is particularly effective for data with varying baselines or drift, and provides robust first-order derivative estimates even in the presence of moderate noise.

**Implementation**: `LlaInterpolator` uses analytical Hermite polynomial derivatives for 1st and 2nd order derivatives, achieving the highest accuracy among all methods (~0.003 max error).

**References**: 
- [Estimating Derivatives of Function-Valued Parameters in a Class of Moment Condition Models](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4940142/)

### GLLA (Generalized Local Linear Approximation)
An extension of the LLA method that enables calculation of higher-order derivatives using a generalized linear approximation framework. GLLA uses a local polynomial fit of arbitrary order and combines it with a sliding window approach. This method is particularly useful when you need consistent estimates of multiple orders of derivatives simultaneously, and it maintains good numerical stability even for higher-order derivatives.

**Implementation**: `GllaInterpolator` uses analytical Hermite polynomial derivatives for 1st and 2nd order derivatives, with similar accuracy to LLA.

### FDA (Functional Data Analysis)
A sophisticated approach that uses spline-based smoothing to represent the time series as a continuous function. FDA automatically determines an optimal smoothing parameter based on the data characteristics, balancing the trade-off between smoothness and fidelity to the original data. This method is particularly well-suited for smooth underlying processes and can provide consistent derivatives up to the order of the chosen spline basis.

**Implementation**: `FdaInterpolator` uses scipy's UnivariateSpline.derivative() for analytical derivatives, achieving good accuracy (~0.1-0.5 max error).

**References**:
- [Functional Data Analysis with R and MATLAB](https://www.tandfonline.com/doi/abs/10.1080/00273171.2010.498294)

### Spline Interpolation
Uses cubic splines to create a smooth, continuous function that passes through or near the data points. Splines are piecewise polynomial functions that maintain continuity up to a specified derivative order at the knots (connection points).

**Implementation**: `SplineInterpolator` uses scipy's UnivariateSpline.derivative() for analytical derivatives, achieving good accuracy (~0.1-0.5 max error).

### LOWESS/LOESS (Locally Weighted Scatterplot Smoothing)
Non-parametric regression methods that fit simple models to localized subsets of the data. These methods are particularly robust to outliers and can handle data with varying noise levels across the domain.

**Implementation**: `LowessInterpolator` and `LoessInterpolator` use numerical differentiation with central differences, achieving moderate accuracy (~0.5-0.6 max error).

### Neural Network Methods
Leverages deep learning to learn complex functional relationships and their derivatives. Neural networks can automatically learn features from the data and provide derivatives through automatic differentiation.

**Implementation**: `NeuralNetworkInterpolator` uses automatic differentiation with PyTorch/TensorFlow, achieving moderate accuracy (~0.6 max error) but with superior performance for high-dimensional data.

### Multivariate Calculus Operations
The `multivariate` module provides comprehensive support for vector calculus operations:

- **Gradient (∇f)**: For scalar functions, computes the vector of partial derivatives
- **Jacobian (J_f)**: For vector-valued functions, computes the matrix of all first-order partial derivatives
- **Hessian (H_f)**: For scalar functions, computes the matrix of all second-order partial derivatives
- **Laplacian (∇²f)**: For scalar functions, computes the sum of all unmixed second partial derivatives

```python
# Multivariate API pattern
mv_derivatives = MultivariateDerivatives(SplineInterpolator, smoothing=0.1)
mv_derivatives.fit(input_data, output_data)
gradient_func = mv_derivatives.gradient()
gradients = gradient_func(eval_points)
```

### Integration Methods
The package provides two integration methods:

#### Basic Integration (integrate_derivative)
Uses the trapezoidal rule to integrate a derivative signal and reconstruct the original time series. You can specify an initial value to match known boundary conditions.

#### Integration with Error Estimation (integrate_derivative_with_error)
Performs integration using both trapezoidal and rectangular rules to provide an estimate of the integration error. This is particularly useful when working with noisy or uncertain derivative data.

## 🔬 Numerical vs. Analytical Methods: Limitations & Considerations

### The Fundamental Trade-offs

Numerical derivative approximation involves inherent trade-offs that users should understand:

#### 1. Smoothing Effects

All numerical methods apply some degree of smoothing, which:
- **Reduces noise** but also **smooths out legitimate sharp features**
- **Rounds critical points** where derivatives should be exactly zero
- **Blurs discontinuities** that should be sharp in the analytical solution

#### 2. Domain Coverage & Data Density

The accuracy of numerical derivatives depends critically on data coverage:

- **Sparse sampling** leads to poor approximation of high-frequency components
- **Boundary effects** reduce accuracy near the edges of your data domain
- **Irregular sampling** can cause inconsistent accuracy across the domain
- **Higher-order derivatives** require progressively denser sampling

#### 3. The Curse of Dimensionality

As the dimensionality of the input space increases:

- **Data requirements grow exponentially** for traditional methods
- **Computational complexity increases** dramatically
- **Accuracy decreases** without corresponding increases in data density

### When to Use Each Approach

#### Traditional Methods (LLA, GLLA, Splines)

**Best for**:
- Low-dimensional problems (1-3 dimensions)
- Smooth functions with moderate complexity
- Cases where interpretability is important
- When analytical derivatives are unavailable

**Limitations**:
- Poor scaling to high dimensions
- Difficulty with sharp features or discontinuities
- Mixed partial derivatives approximated as zero

#### Neural Network Methods with Automatic Differentiation

**Best for**:
- High-dimensional problems (4+ dimensions)
- Complex functions with many interactions
- When exact mixed partial derivatives are needed
- Large datasets that would overwhelm traditional methods

**Limitations**:
- Requires more data for training
- Less interpretable than traditional methods
- Training variability affects reproducibility

### The Crossover Point

The transition point where neural networks become more efficient than traditional methods occurs around 3-4 input dimensions. This is because:

1. **Traditional methods** scale exponentially with dimensions (O(n^d) where d is dimensionality)
2. **Neural networks** scale linearly with parameters, regardless of input dimensions
3. **Automatic differentiation** provides exact derivatives without numerical approximation errors

For a typical problem with moderate complexity:
- **1-3 dimensions**: Traditional methods are faster and often more accurate
- **4-10 dimensions**: Neural networks become competitive and often superior
- **10+ dimensions**: Neural networks are significantly more efficient and accurate

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
python -m pytest src/pydelt/tests/test_multivariate.py
```

The test suite includes verification of:
- Universal differentiation API consistency
- Multivariate calculus operations (gradient, Jacobian, Hessian, Laplacian)
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

## 👨‍💻 Developer Guide

### GitHub Actions Workflows

The pydelt repository uses GitHub Actions for continuous integration and documentation building. The following workflows are configured:

#### 1. Python Tests (`python-tests.yml`)

```yaml
name: Python Tests

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, '3.10', '3.11']

    steps:
    - uses: actions/checkout@v3
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install pytest
        pip install -e .
    - name: Test with pytest
      run: |
        pytest src/pydelt/tests/
```

This workflow runs the test suite on multiple Python versions whenever code is pushed to the main branch or a pull request is created.

#### 2. Documentation Build (`docs.yml`)

```yaml
name: Documentation

on:
  push:
    branches: [ main ]
    paths:
      - 'docs/**'
      - 'src/pydelt/**'

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e .
        pip install -r docs/requirements.txt
    - name: Build documentation
      run: |
        cd docs
        make html
    - name: Check for warnings
      run: |
        cd docs
        ! grep -r "WARNING:" _build/html/output.txt
```

This workflow builds the documentation whenever changes are made to the docs directory or the source code.

### Documentation Build Process

1. **Local Development**:
   ```bash
   # Install Sphinx and dependencies
   pip install -r docs/requirements.txt
   
   # Build documentation locally
   cd docs
   make html
   
   # View in browser
   open _build/html/index.html  # On macOS
   ```

2. **Read the Docs Integration**:
   - The repository is connected to [Read the Docs](https://readthedocs.org/)
   - Configuration is specified in `.readthedocs.yaml`
   - Documentation is automatically built and hosted at https://pydelt.readthedocs.io/
   - Versioned documentation is available for each release

3. **Documentation Structure**:
   - `docs/index.rst`: Main landing page
   - `docs/installation.rst`: Installation instructions
   - `docs/quickstart.rst`: Getting started guide
   - `docs/examples.rst`: Detailed examples
   - `docs/api.rst`: API reference (auto-generated from docstrings)
   - `docs/faq.rst`: Frequently asked questions
   - `docs/changelog.rst`: Version history

### PyPI Release Process

1. **Update Version**:
   - Increment version in `pyproject.toml`
   - Update `CHANGELOG.md` with release notes

2. **Build Distribution Packages**:
   ```bash
   python -m build
   ```

3. **Upload to PyPI**:
   ```bash
   python -m twine upload dist/*
   ```

4. **Create GitHub Release**:
   - Tag the release in git
   - Create a GitHub release with release notes

5. **Documentation Update**:
   - Read the Docs will automatically build documentation for the new version
   - Set the default version in the Read the Docs admin panel

## 📞 Support

- **Documentation**: [https://pydelt.readthedocs.io/](https://pydelt.readthedocs.io/)
- **Issues**: [GitHub Issues](https://github.com/MikeHLee/pydelt/issues)
- **PyPI**: [https://pypi.org/project/pydelt/](https://pypi.org/project/pydelt/)
