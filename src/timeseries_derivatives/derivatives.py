import numpy as np
import pandas as pd
from scipy.special import factorial
from scipy.interpolate import UnivariateSpline
from scipy.stats import linregress
from typing import List, Optional, Dict, Tuple, Union

def lla(time_data: List[int], signal_data: List[float], window_size: Optional[int] = 5) -> Tuple[List[float], List[float]]:
    '''
    Local Linear Approximation (LLA) method for estimating the derivative of a time series.
    Uses min-normalization and linear regression within a sliding window.
    
    Args:
        time_data: List of time values (epoch seconds)
        signal_data: List of signal values
        window_size: Number of points to consider for derivative calculation
    
    Returns:
        Tuple containing:
        - List of derivative values
        - List of step sizes used for each calculation
    '''
    if len(time_data) != len(signal_data):
        raise ValueError("Time and Signal data must have the same length")
    
    def slope_calc(i: int) -> Tuple[float, float]:
        window_start = int(max(0, i - (window_size - 0.5) // 2))
        shift = 0 if window_size % 2 == 0 else 1
        window_end = int(min(len(time_data), i + (window_size - 0.5) // 2 + shift))
        
        time_window = np.array(time_data[window_start:window_end])
        signal_window = np.array(signal_data[window_start:window_end])
        
        # Min normalization
        min_time = np.min(time_window)
        min_signal = np.min(signal_window)
        time_window = time_window - min_time
        signal_window = signal_window - min_signal
        
        fit = linregress(time_window, signal_window)
        step = (window_end - window_start)/window_size
        return fit.slope, step
    
    results = [slope_calc(i) for i in range(len(time_data))]
    derivative = [r[0] for r in results]
    steps = [r[1] for r in results]
    
    return derivative, steps

def gold(signal: np.ndarray, time: np.ndarray, embedding: int = 3, n: int = 2) -> Dict[str, Union[np.ndarray, int]]:
    """
    Calculate derivatives using the Generalized Orthogonal Local Derivative (GOLD) method.
    
    Args:
        signal: Array of signal values
        time: Array of time values corresponding to the signal
        embedding: Number of points to consider for derivative calculation
        n: Maximum order of derivative to estimate
    
    Returns:
        Dictionary containing:
        - dtime: Time values for derivatives
        - dsignal: Matrix of derivatives (0th to nth order)
        - embedding: Embedding dimension used
        - n: Maximum order of derivatives calculated
    """
    if len(signal) != len(time):
        raise ValueError("Signal and time vectors should have the same length.")
    if len(signal) <= embedding:
        raise ValueError("Signal and time vectors should have a length greater than embedding.")
    if n >= embedding:
        raise ValueError("The embedding dimension should be higher than the maximum order of the derivative, n.")
    
    tembed = np.column_stack([time[i:len(time)-embedding+i+1] for i in range(embedding)])
    Xembed = np.column_stack([signal[i:len(signal)-embedding+i+1] for i in range(embedding)])
    
    derivatives = np.zeros((tembed.shape[0], n+1))
    
    for k in range(tembed.shape[0]):
        t = tembed[k] - tembed[k, embedding // 2]
        Xi = np.vstack([t**q for q in range(n+1)])
        for q in range(1, n+1):
            for p in range(q):
                Xi[q] -= np.dot(Xi[p], t**q) / np.dot(Xi[p], t**p) * Xi[p]
        
        D = np.diag(1 / factorial(np.arange(n+1)))
        L = D @ Xi
        W = L.T @ np.linalg.inv(L @ L.T)
        derivatives[k] = Xembed[k] @ W
    
    time_derivative = np.convolve(time, np.ones(embedding)/embedding, mode='valid')
    
    return {'dtime': time_derivative, 'dsignal': derivatives, 'embedding': embedding, 'n': n}

def glla(signal: np.ndarray, time: np.ndarray, embedding: int = 3, n: int = 2) -> Dict[str, Union[np.ndarray, int]]:
    """
    Calculate derivatives using the Generalized Local Linear Approximation (GLLA) method.
    
    Args:
        signal: Array of signal values
        time: Array of time values corresponding to the signal
        embedding: Number of points to consider for derivative calculation
        n: Maximum order of derivative to calculate
    
    Returns:
        Dictionary containing:
        - dtime: Time values for derivatives
        - dsignal: Matrix of derivatives (0th to nth order)
        - embedding: Embedding dimension used
        - n: Maximum order of derivatives calculated
    """
    if len(signal) != len(time):
        raise ValueError("Signal and time vectors should have the same length.")
    if len(signal) <= embedding:
        raise ValueError("Signal and time vectors should have a length greater than embedding.")
    if n >= embedding:
        raise ValueError("The embedding dimension should be higher than the maximum order of the derivative, n.")
    
    deltat = np.min(np.diff(time))
    L = np.column_stack([(np.arange(1, embedding+1) - np.mean(np.arange(1, embedding+1)))**i / factorial(i) for i in range(n+1)])
    W = L @ np.linalg.inv(L.T @ L)
    
    Xembed = np.column_stack([signal[i:len(signal)-embedding+i+1] for i in range(embedding)])
    derivatives = Xembed @ W
    derivatives[:, 1:] /= deltat**np.arange(1, n+1)[None, :]
    
    time_derivative = np.convolve(time, np.ones(embedding)/embedding, mode='valid')
    
    return {'dtime': time_derivative, 'dsignal': derivatives, 'embedding': embedding, 'n': n}

def fda(signal: np.ndarray, time: np.ndarray, spar: Optional[float] = None) -> Dict[str, Union[np.ndarray, float, None]]:
    """
    Calculate derivatives using the Functional Data Analysis (FDA) method.
    
    Args:
        signal: Array of signal values
        time: Array of time values corresponding to the signal
        spar: Smoothing parameter for the spline. If None, automatically determined
    
    Returns:
        Dictionary containing:
        - dtime: Time values for derivatives
        - dsignal: Matrix of derivatives (0th to 2nd order)
        - spar: Smoothing parameter used
    """
    # If spar is None, estimate it based on data characteristics
    if spar is None:
        # Use a heuristic based on data length and range
        n = len(signal)
        range_y = np.ptp(signal)
        spar = n * (0.01 * range_y) ** 2

    spline = UnivariateSpline(time, signal, s=spar)
    
    # Evaluate derivatives at time points
    d0 = spline(time)
    d1 = spline.derivative(n=1)(time)
    d2 = spline.derivative(n=2)(time)
    
    derivatives = np.column_stack([d0, d1, d2])
    
    return {'dtime': time, 'dsignal': derivatives, 'spar': spar}
