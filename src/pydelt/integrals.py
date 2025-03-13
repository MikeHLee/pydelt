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
    # Convert inputs to numpy arrays
    t = np.asarray(time)
    deriv = np.asarray(derivative)
    
    # Calculate time steps
    dt = np.diff(t)
    
    # Integrate using the trapezoidal rule
    integral = np.zeros_like(t)
    integral[0] = initial_value
    
    # Cumulative integration using trapezoidal rule
    for i in range(1, len(t)):
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
    # Convert inputs to numpy arrays
    t = np.asarray(time)
    deriv = np.asarray(derivative)
    
    # Calculate time steps
    dt = np.diff(t)
    
    # Integrate using both trapezoidal and rectangular rules to estimate error
    integral_trap = np.zeros_like(t)
    integral_rect = np.zeros_like(t)
    integral_trap[0] = initial_value
    integral_rect[0] = initial_value
    
    for i in range(1, len(t)):
        # Trapezoidal rule
        integral_trap[i] = integral_trap[i-1] + 0.5 * (deriv[i] + deriv[i-1]) * dt[i-1]
        # Rectangular rule
        integral_rect[i] = integral_rect[i-1] + deriv[i-1] * dt[i-1]
    
    # Estimate error as difference between methods
    error = np.abs(integral_trap - integral_rect)
    
    return integral_trap, error
