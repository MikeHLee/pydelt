"""
Functions for integrating time series data using calculated derivatives.
"""

import numpy as np
from typing import List, Tuple, Union, Optional

def integrate_derivative(
    time: Union[List[float], np.ndarray],
    derivative: Union[List[float], np.ndarray],
    initial_value: Optional[Union[float, np.ndarray, List[float]]] = 0.0
) -> np.ndarray:
    """
    Integrate a time series derivative to reconstruct the original signal.
    Supports vector-valued derivatives and initial values.
    
    Args:
        time: Time points corresponding to the derivative values.
        derivative: Derivative values at each time point, shape (N,) or (N, n_out)
        initial_value: Initial value(s) of the integral at time[0]. Scalar or shape (n_out,). Defaults to 0.0.
    Returns:
        np.ndarray: Reconstructed signal through integration, shape (N, n_out) or (N,)
    Example:
        >>> time = np.linspace(0, 10, 500)
        >>> signal = np.stack([np.sin(time), np.cos(time)], axis=-1)
        >>> derivative, _ = lla(time, signal, window_size=5)
        >>> reconstructed = integrate_derivative(time, derivative, initial_value=signal[0])
        >>> # reconstructed should be close to original signal
    """
    t = np.asarray(time)
    deriv = np.asarray(derivative)
    if deriv.ndim == 1:
        deriv = deriv[:, None]
    n_out = deriv.shape[1]
    if np.isscalar(initial_value):
        init = np.full(n_out, initial_value)
    else:
        init = np.asarray(initial_value)
        if init.shape == ():
            init = np.full(n_out, float(init))
        elif init.shape[0] != n_out:
            raise ValueError("initial_value must be scalar or shape (n_out,)")
    dt = np.diff(t)
    integral = np.zeros((len(t), n_out), dtype=deriv.dtype)
    integral[0, :] = init
    for i in range(1, len(t)):
        integral[i, :] = integral[i-1, :] + 0.5 * (deriv[i, :] + deriv[i-1, :]) * dt[i-1]
    if integral.shape[1] == 1:
        return integral[:, 0]
    return integral

def integrate_derivative_with_error(
    time: Union[List[float], np.ndarray],
    derivative: Union[List[float], np.ndarray],
    initial_value: Optional[Union[float, np.ndarray, List[float]]] = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Integrate a time series derivative and estimate integration error.
    Supports vector-valued derivatives and initial values.
    
    Args:
        time: Time points corresponding to the derivative values.
        derivative: Derivative values at each time point, shape (N,) or (N, n_out)
        initial_value: Initial value(s) of the integral at time[0]. Scalar or shape (n_out,). Defaults to 0.0.
    Returns:
        Tuple[np.ndarray, np.ndarray]: (reconstructed signal, estimated error), each shape (N, n_out) or (N,)
    Example:
        >>> time = np.linspace(0, 10, 500)
        >>> signal = np.stack([np.sin(time), np.cos(time)], axis=-1)
        >>> derivative, _ = lla(time, signal, window_size=5)
        >>> reconstructed, error = integrate_derivative_with_error(time, derivative, initial_value=signal[0])
    """
    t = np.asarray(time)
    deriv = np.asarray(derivative)
    if deriv.ndim == 1:
        deriv = deriv[:, None]
    n_out = deriv.shape[1]
    if np.isscalar(initial_value):
        init = np.full(n_out, initial_value)
    else:
        init = np.asarray(initial_value)
        if init.shape == ():
            init = np.full(n_out, float(init))
        elif init.shape[0] != n_out:
            raise ValueError("initial_value must be scalar or shape (n_out,)")
    dt = np.diff(t)
    integral_trap = np.zeros((len(t), n_out), dtype=deriv.dtype)
    integral_rect = np.zeros((len(t), n_out), dtype=deriv.dtype)
    integral_trap[0, :] = init
    integral_rect[0, :] = init
    for i in range(1, len(t)):
        integral_trap[i, :] = integral_trap[i-1, :] + 0.5 * (deriv[i, :] + deriv[i-1, :]) * dt[i-1]
        integral_rect[i, :] = integral_rect[i-1, :] + deriv[i-1, :] * dt[i-1]
    error = np.abs(integral_trap - integral_rect)
    if integral_trap.shape[1] == 1:
        return integral_trap[:, 0], error[:, 0]
    return integral_trap, error
