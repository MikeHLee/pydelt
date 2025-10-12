#!/usr/bin/env python3
"""
Script to add window function support to all remaining interpolator classes.
"""

import re

# Read the interpolation file
with open('src/pydelt/interpolation.py', 'r') as f:
    content = f.read()

# List of interpolator classes to update (excluding BaseInterpolator, SplineInterpolator, LowessInterpolator which are done)
interpolators_to_update = [
    'LoessInterpolator',
    'FdaInterpolator',
    'LlaInterpolator',
    'GllaInterpolator',
    'GoldInterpolator',
    'NeuralNetworkInterpolator'
]

# Pattern to find fit method signatures
fit_pattern = r'(    def fit\(self, time: Union\[List\[float\], np\.ndarray\], signal: Union\[List\[float\], np\.ndarray\]\):)'
fit_replacement = r'    def fit(self, time: Union[List[float], np.ndarray], signal: Union[List[float], np.ndarray],\n            window_func: Optional[Callable[[int], np.ndarray]] = None):'

# Pattern to add window function application after sorting in fit methods
# We'll add after the sorting logic

# For each interpolator, we need to:
# 1. Update fit() signature to include window_func parameter
# 2. Add window function application code after data preparation
# 3. Update differentiate() signature to include normalize_by_observations parameter
# 4. Add normalization code in derivative_func before returning results

print("Updating interpolator classes with window function support...")
print("=" * 60)

# Update fit method signatures
content = re.sub(fit_pattern, fit_replacement, content)

# Pattern to find differentiate method signatures (excluding those already updated)
diff_pattern = r'(    def differentiate\(self, order: int = 1, mask: Optional\[Union\[np\.ndarray, List\[bool\], List\[int\]\]\] = None\) -> Callable:)'
diff_replacement = r'    def differentiate(self, order: int = 1, mask: Optional[Union[np.ndarray, List[bool], List[int]]] = None,\n                     normalize_by_observations: bool = False) -> Callable:'

# Update differentiate signatures
original_count = content.count('def differentiate(self, order: int = 1, mask: Optional[Union[np.ndarray, List[bool], List[int]]] = None) -> Callable:')
content = re.sub(diff_pattern, diff_replacement, content)
new_count = content.count('def differentiate(self, order: int = 1, mask: Optional[Union[np.ndarray, List[bool], List[int]]] = None,')

print("Updated {} differentiate() method signatures".format(new_count - original_count))

# Save the updated content
with open('src/pydelt/interpolation.py', 'w') as f:
    f.write(content)

print("✓ Updated fit() and differentiate() signatures")
print("\nNote: Manual updates still needed for:")
print("  1. Adding window function application code in each fit() method")
print("  2. Adding normalization code in each derivative_func")
print("  3. Storing n_observations and window_weights in each fit() method")
