"""
Helper functions for adding stochastic derivative support to interpolation methods.
"""

import numpy as np
from typing import Dict, Union, Optional, List, Callable


def add_stochastic_transform_to_derivative(
    interpolator, 
    eval_points: np.ndarray, 
    result: np.ndarray, 
    order: int,
    scalar_input: bool,
    h: float = 1e-6
) -> np.ndarray:
    """
    Add stochastic transformation to derivative results if a link function is set.
    
    Args:
        interpolator: The interpolator instance with potential stochastic link
        eval_points: Points where derivatives are evaluated
        result: Current derivative results
        order: Order of derivative
        scalar_input: Whether input was scalar
        h: Step size for numerical derivatives (if needed)
    
    Returns:
        Transformed derivative results
    """
    if interpolator.stochastic_link is None:
        return result
    
    # Get function values at eval points for stochastic transform
    func_values = interpolator.predict(eval_points)
    if func_values.ndim == 0:
        func_values = np.array([func_values])
    
    # Prepare derivatives dictionary for transformation
    derivatives_dict = {order: result}
    
    # Get higher order derivatives if needed for Itô correction
    if interpolator.stochastic_method == "ito" and order == 1:
        try:
            # Try to get second derivative for Itô correction
            second_deriv_func = interpolator.differentiate(order=2)
            second_result = second_deriv_func(eval_points)
            if second_result.ndim == 0:
                second_result = np.array([second_result])
            derivatives_dict[2] = second_result
        except:
            # If second derivative not available, compute numerically
            try:
                f_plus = interpolator.predict(eval_points + h)
                f_center = interpolator.predict(eval_points)
                f_minus = interpolator.predict(eval_points - h)
                second_result = (f_plus - 2 * f_center + f_minus) / (h * h)
                if second_result.ndim == 0:
                    second_result = np.array([second_result])
                derivatives_dict[2] = second_result
            except:
                pass  # Second derivative not available
    
    # Apply stochastic transformation
    transformed = interpolator._apply_stochastic_transform(func_values, derivatives_dict)
    return transformed[order]


def create_stochastic_derivative_wrapper(original_derivative_func: Callable, interpolator) -> Callable:
    """
    Create a wrapper around a derivative function that applies stochastic transformations.
    
    Args:
        original_derivative_func: Original derivative function
        interpolator: Interpolator instance with stochastic link support
    
    Returns:
        Wrapped derivative function with stochastic support
    """
    def stochastic_derivative_func(eval_points: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        # Get original derivative result
        result = original_derivative_func(eval_points)
        
        # Apply stochastic transformation if link function is set
        if interpolator.stochastic_link is not None:
            eval_points = np.asarray(eval_points)
            scalar_input = eval_points.ndim == 0
            if scalar_input:
                eval_points = np.array([eval_points])
            
            result = add_stochastic_transform_to_derivative(
                interpolator, eval_points, result, 1, scalar_input
            )
            
            if scalar_input and result.size == 1:
                result = result.item()
        
        return result
    
    return stochastic_derivative_func
