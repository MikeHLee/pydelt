Frequently Asked Questions
==========================

This section addresses common questions and issues when using pydelt.

Multivariate Derivatives
------------------------

**Q: Why do I get non-zero gradients at points where the gradient should be zero?**

A: This is due to numerical smoothing in interpolation methods. For example, with the function f(x,y) = (x-y)², the gradient should be zero along the line x=y, but numerical methods will show small non-zero values due to smoothing effects.

**Solution:**
- Use higher resolution sampling near critical points
- Reduce smoothing parameters (but beware of overfitting)
- Validate against analytical solutions when possible
- Consider neural network methods with automatic differentiation

**Q: How accurate are the multivariate derivatives?**

A: Accuracy depends on several factors:
- **Function smoothness**: Smooth functions give better results
- **Sampling density**: More data points improve accuracy
- **Interpolation method**: LLA/GLLA typically most accurate, LOWESS/LOESS moderate
- **Distance from boundaries**: Accuracy decreases near domain edges

**Q: When should I use different interpolation methods for multivariate derivatives?**

A: Choose based on your needs:
- **SplineInterpolator**: Good balance of speed and accuracy
- **LlaInterpolator/GllaInterpolator**: Best accuracy for smooth functions
- **LowessInterpolator**: Robust to outliers, moderate accuracy
- **NeuralNetworkInterpolator**: Best for exact derivatives and complex functions

**Q: Can I compute mixed partial derivatives?**

A: Traditional interpolation methods approximate mixed partials as zero. For exact mixed partials, use neural network methods with automatic differentiation:

.. code-block:: python

   from pydelt.multivariate import NeuralNetworkMultivariateDerivatives
   
   # This will provide exact mixed partial derivatives
   nn_mv = NeuralNetworkMultivariateDerivatives()
   nn_mv.fit(input_data, output_data)

Numerical Accuracy
------------------

**Q: Why do my derivatives look "smoothed out" compared to the analytical solution?**

A: All numerical interpolation methods apply some degree of smoothing to handle noise and ensure stability. This smoothing:
- Reduces sharp features and discontinuities
- Makes critical points appear "rounded"
- Can mask important mathematical properties

**Q: How can I improve derivative accuracy?**

A: Try these strategies:
1. **Increase sampling density** near regions of interest
2. **Reduce smoothing parameters** (e.g., lower `smoothing` in SplineInterpolator)
3. **Use analytical derivatives** when available for validation
4. **Choose appropriate interpolation method** for your function type
5. **Consider neural networks** for complex multivariate functions

**Q: What causes boundary effects in derivatives?**

A: Near the edges of your data domain:
- Interpolation has fewer neighboring points to work with
- Extrapolation may be required, reducing accuracy
- Edge effects from smoothing become more pronounced

**Solution:** Extend your sampling domain beyond the region where you need accurate derivatives.

Performance and Memory
----------------------

**Q: My multivariate derivative computation is slow. How can I speed it up?**

A: Performance optimization strategies:
1. **Reduce data size** if possible while maintaining accuracy
2. **Use SplineInterpolator** for fastest computation
3. **Pre-compute derivatives** at fixed evaluation points
4. **Consider neural networks** for batch processing large datasets

**Q: I'm running out of memory with large datasets. What should I do?**

A: Memory management approaches:
1. **Process data in chunks** rather than all at once
2. **Use lower-resolution sampling** for initial exploration
3. **Consider neural network methods** which can handle larger datasets more efficiently

Error Messages and Troubleshooting
-----------------------------------

**Q: I get "Interpolation failed" errors. What's wrong?**

A: Common causes and solutions:
- **Duplicate input points**: Remove or average duplicate x-values
- **Insufficient data**: Ensure you have enough points for the chosen method
- **Extreme smoothing**: Try reducing smoothing parameters
- **Data scaling issues**: Consider normalizing your input/output data

**Q: My Hessian computation returns unexpected results. Why?**

A: The Hessian implementation computes only diagonal elements (pure second derivatives). Mixed partial derivatives are approximated as zero for traditional methods. For full Hessian matrices, use neural network methods.

**Q: Neural network derivatives are inconsistent between runs. Is this normal?**

A: Yes, this is expected due to:
- Random weight initialization
- Stochastic training process
- Local minima in optimization

For more consistent results, set random seeds or use multiple training runs with averaging.

Integration with Other Libraries
---------------------------------

**Q: Can I use pydelt with pandas DataFrames?**

A: Yes, convert to numpy arrays:

.. code-block:: python

   import pandas as pd
   import numpy as np
   
   # Convert DataFrame to numpy arrays
   input_data = df[['x', 'y']].values
   output_data = df['z'].values

**Q: How do I integrate with scipy optimization routines?**

A: Use pydelt derivatives as objective function gradients:

.. code-block:: python

   from scipy.optimize import minimize
   from pydelt.multivariate import MultivariateDerivatives
   
   # Fit derivatives
   mv = MultivariateDerivatives(SplineInterpolator)
   mv.fit(input_data, output_data)
   gradient_func = mv.gradient()
   
   # Use in optimization
   result = minimize(objective_func, x0, jac=gradient_func)

**Q: Can I use pydelt with JAX or other autodiff libraries?**

A: pydelt focuses on interpolation-based derivatives. For automatic differentiation, use JAX, PyTorch, or TensorFlow directly. However, you can use pydelt to validate autodiff results or handle cases where analytical functions aren't available.

Getting Help
------------

**Q: I found a bug or have a feature request. Where should I report it?**

A: Please report issues on the GitHub repository: https://github.com/yourusername/pydelt

**Q: How can I contribute to pydelt?**

A: Contributions are welcome! See the contributing guidelines in the repository for details on:
- Code style and testing requirements
- Documentation standards
- Pull request process

**Q: Where can I find more examples?**

A: Check out:
- The examples section in this documentation
- The `/local/tests/` directory for visual test examples
- The GitHub repository for additional demos and notebooks
