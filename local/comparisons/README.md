# PyDelt Comparison Scripts

This directory contains scripts for comparing PyDelt's numerical differentiation methods with other popular libraries mentioned in the research paper. These scripts help validate PyDelt's performance claims and provide benchmarks for accuracy, noise robustness, and computational efficiency.

## Available Comparison Scripts

1. **`compare_scipy.py`**: Compares PyDelt with SciPy's interpolation methods (UnivariateSpline, CubicSpline)
2. **`compare_numdifftools.py`**: Compares PyDelt with NumDiffTools' adaptive finite difference methods
3. **`compare_findiff.py`**: Compares PyDelt with FinDiff's finite difference approximations
4. **`compare_jax.py`**: Compares PyDelt with JAX's automatic differentiation capabilities
5. **`generate_comparison_plots.py`**: Creates comprehensive visualizations comparing all methods

## Prerequisites

To run these comparison scripts, you need the following packages installed:

```bash
pip install numpy matplotlib scipy
pip install numdifftools findiff jax jaxlib
pip install pandas seaborn
```

## Running the Scripts

Each comparison script can be run independently:

```bash
python compare_scipy.py
python compare_numdifftools.py
python compare_findiff.py
python compare_jax.py
```

To generate comprehensive comparison plots across all methods:

```bash
python generate_comparison_plots.py
```

## Output

The scripts generate both console output (showing numerical results) and visual plots saved to the `../output/` directory. The generated plots include:

- Function and derivative visualizations for each test function
- Accuracy comparisons across different noise levels
- Performance benchmarks for varying data sizes
- Dimensionality scaling tests (for multivariate methods)

## Test Functions

All scripts use the same set of test functions with known analytical derivatives:

1. **Sine function**: sin(x) with derivatives cos(x) and -sin(x)
2. **Exponential function**: exp(x) with derivative exp(x)
3. **Polynomial function**: x³ - 2x² + 3x - 1 with derivatives 3x² - 4x + 3 and 6x - 4

## Comparison Metrics

The scripts evaluate methods based on:

1. **Accuracy**: Maximum and mean absolute error compared to analytical derivatives
2. **Noise Robustness**: How error increases with different noise levels (0%, 1%, 5%, 10%)
3. **Computational Efficiency**: Time required to compute derivatives
4. **Dimensionality Scaling**: How methods perform as dimensionality increases (JAX comparison only)

## Summary Table

The `generate_comparison_plots.py` script also creates a summary table (`method_comparison_summary.csv` and `.html`) that provides a high-level comparison of all methods based on:

- Accuracy
- Noise robustness
- Performance
- Dimensionality support
- Key strengths and weaknesses

## Citations

The comparison scripts reference the following libraries:

- **SciPy**: Virtanen, P., Gommers, R., Oliphant, T. E., et al. (2020). SciPy 1.0: fundamental algorithms for scientific computing in Python. Nature Methods, 17(3), 261-272.
- **NumDiffTools**: https://github.com/pbrod/numdifftools
- **FinDiff**: https://github.com/maroba/findiff
- **JAX**: https://github.com/jax-ml/jax

## Integration with Research Paper

These scripts provide the empirical foundation for the claims made in the research paper, particularly the performance evaluations in Section 5 (Accuracy Comparison, Noise Robustness, Computational Efficiency, and Dimensionality Scaling).
