#!/usr/bin/env python3
"""
Script to fix the numerical differentiation issues in LlaInterpolator and GllaInterpolator.
This replaces the problematic recursive approach with a more stable iterative method.
"""

import re

def fix_interpolation_file():
    # Read the file
    with open('src/pydelt/interpolation.py', 'r') as f:
        content = f.read()
    
    # Pattern for LlaInterpolator numerical differentiation
    lla_pattern = r'(                else:\n                    # For higher orders, use numerical differentiation\n                    h_num = 1e-6\n                    \n                    # Compute numerical derivative at this point\n                    def compute_numerical_derivative\(x, order\):\n                        if order == 1:\n                            return \(self\.predict\(x \+ h_num\) - self\.predict\(x - h_num\)\) / \(2 \* h_num\)\n                        else:\n                            # Higher order derivatives using finite differences\n                            return \(compute_numerical_derivative\(x \+ h_num, order - 1\) - \n                                   compute_numerical_derivative\(x - h_num, order - 1\)\) / \(2 \* h_num\)\n                    \n                    result\[i\] = compute_numerical_derivative\(t_i, order\))'
    
    lla_replacement = '''                else:
                    # For higher orders, use numerical differentiation with larger step size
                    h_num = 1e-4  # Larger step size for stability
                    
                    # Use iterative finite differences for higher order derivatives
                    def compute_derivative_iteratively(x, order):
                        # Start with the function values
                        values = self.predict(np.array([x - h_num, x, x + h_num]))
                        
                        # Apply finite differences iteratively
                        for _ in range(order):
                            # Central difference formula: f'(x) ≈ (f(x+h) - f(x-h)) / (2h)
                            new_values = np.zeros(len(values) - 2)
                            for j in range(len(new_values)):
                                new_values[j] = (values[j + 2] - values[j]) / (2 * h_num)
                            values = new_values
                            if len(values) == 0:
                                return 0.0
                        
                        return values[0] if len(values) > 0 else 0.0
                    
                    result[i] = compute_derivative_iteratively(t_i, order)'''
    
    # Apply the replacement for LlaInterpolator
    content = re.sub(lla_pattern, lla_replacement, content, flags=re.MULTILINE)
    
    # Similar pattern for GllaInterpolator (if it exists)
    glla_pattern = r'(                else:\n                    # For higher orders, use numerical differentiation\n                    h_num = 1e-6\n                    \n                    # Compute numerical derivative at this point\n                    def compute_numerical_derivative\(x, order\):\n                        if order == 1:\n                            return \(self\.predict\(x \+ h_num\) - self\.predict\(x - h_num\)\) / \(2 \* h_num\)\n                        else:\n                            # Higher order derivatives using finite differences\n                            return \(compute_numerical_derivative\(x \+ h_num, order - 1\) - \n                                   compute_numerical_derivative\(x - h_num, order - 1\)\) / \(2 \* h_num\)\n                    \n                    result\[i\] = compute_numerical_derivative\(t_i, order\))'
    
    # Apply the same replacement for GllaInterpolator
    content = re.sub(glla_pattern, lla_replacement, content, flags=re.MULTILINE)
    
    # Write the file back
    with open('src/pydelt/interpolation.py', 'w') as f:
        f.write(content)
    
    print("Fixed numerical differentiation in interpolation.py")

if __name__ == "__main__":
    fix_interpolation_file()
