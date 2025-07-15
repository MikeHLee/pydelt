"""
Functions for integrating time series data using calculated derivatives.
"""

import numpy as np
from typing import List, Tuple, Union, Optional

def integrate_derivative(
    time: Union[List[float], np.ndarray],
    derivative: Union[List[float], np.ndarray],
    initial_value: Optional[float] = 0.0
) -> np.ndarray:
    """
    Integrate a time series derivative to reconstruct the original signal.
    
    Args:
        time: Time points corresponding to the derivative values.
        derivative: Derivative values at each time point.
        initial_value: Initial value of the integral at time[0]. Defaults to 0.0.
        
    Returns:
        np.ndarray: Reconstructed signal through integration.
        
    Example:
        >>> time = np.linspace(0, 10, 500)
        >>> signal = np.sin(time)
        >>> derivative, _ = lla(time.tolist(), signal.tolist(), window_size=5)
        >>> reconstructed = integrate_derivative(time, derivative, initial_value=signal[0])
        >>> # reconstructed should be close to original signal
    """
    # Convert inputs to numpy arrays for efficient computation
    t = np.asarray(time)
    deriv = np.asarray(derivative)
    
    # Calculate time differences between consecutive points
    # This gives us the width of each integration interval
    dt = np.diff(t)
    
    # Initialize array to store integrated values with same shape as time
    integral = np.zeros_like(t)
    
    # Set the first value to the specified initial value
    # This represents the integration constant C in ∫f'(x)dx = f(x) + C
    integral[0] = initial_value
    
    # Perform cumulative integration using the trapezoidal rule
    # The trapezoidal rule approximates the area under the curve as a series of trapezoids
    # For each interval [t_i-1, t_i], the area is approximately (f(t_i-1) + f(t_i))/2 * (t_i - t_i-1)
    for i in range(1, len(t)):
        # Add the area of the current trapezoid to the previous cumulative sum
        # 0.5 * (deriv[i] + deriv[i-1]) is the average height of the trapezoid
        # dt[i-1] is the width of the trapezoid
        integral[i] = integral[i-1] + 0.5 * (deriv[i] + deriv[i-1]) * dt[i-1]
    
    return integral

def integrate_derivative_with_error(
    time: Union[List[float], np.ndarray],
    derivative: Union[List[float], np.ndarray],
    initial_value: Optional[float] = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Integrate a time series derivative and estimate integration error.
    
    Args:
        time: Time points corresponding to the derivative values.
        derivative: Derivative values at each time point.
        initial_value: Initial value of the integral at time[0]. Defaults to 0.0.
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (reconstructed signal, estimated error)
        
    Example:
        >>> time = np.linspace(0, 10, 500)
        >>> signal = np.sin(time)
        >>> derivative, _ = lla(time.tolist(), signal.tolist(), window_size=5)
        >>> reconstructed, error = integrate_derivative_with_error(time, derivative, initial_value=signal[0])
    """
    # Convert inputs to numpy arrays for efficient computation
    t = np.asarray(time)
    deriv = np.asarray(derivative)
    
    # Calculate time differences between consecutive points
    # This gives us the width of each integration interval
    dt = np.diff(t)
    
    # Initialize arrays to store integrated values using two different methods
    # We'll use both trapezoidal and rectangular rules to estimate error
    integral_trap = np.zeros_like(t)  # For trapezoidal rule integration
    integral_rect = np.zeros_like(t)  # For rectangular rule integration
    
    # Set the first value of both methods to the specified initial value
    integral_trap[0] = initial_value
    integral_rect[0] = initial_value
    
    # Perform cumulative integration using both methods
    for i in range(1, len(t)):
        # Trapezoidal rule: approximates area as trapezoid
        # Average of function values at endpoints multiplied by interval width
        integral_trap[i] = integral_trap[i-1] + 0.5 * (deriv[i] + deriv[i-1]) * dt[i-1]
        
        # Rectangular rule: approximates area as rectangle
        # Uses only the left endpoint function value multiplied by interval width
        integral_rect[i] = integral_rect[i-1] + deriv[i-1] * dt[i-1]
    
    # Estimate error as the absolute difference between the two methods
    # This provides a rough approximation of the numerical integration error
    # The difference between methods is proportional to the true error
    error = np.abs(integral_trap - integral_rect)
    
    # Return the trapezoidal rule result (more accurate) and the error estimate
    return integral_trap, error
