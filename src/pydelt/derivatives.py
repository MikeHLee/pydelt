import numpy as np
import pandas as pd
from scipy.special import factorial
from scipy.interpolate import UnivariateSpline
from scipy.stats import linregress
from typing import List, Optional, Dict, Tuple, Union, Callable

# Import interpolation methods
from pydelt.interpolation import (
    local_segmented_linear,
    spline_interpolation,
    lowess_interpolation,
    loess_interpolation,
    get_best_interpolation,
    calculate_fit_quality
)

def lla(time_data: List[int], signal_data: List[float], window_size: Optional[int] = 5, 
         normalization: str = 'min', zero_mean: bool = False,
         r2_threshold: Optional[float] = None,
         resample_method: Optional[str] = None) -> Tuple[List[float], List[float]]:
    '''
    Local Linear Approximation (LLA) method for estimating the derivative of a time series.
    Uses configurable normalization and linear regression within a sliding window.
    
    Args:
        time_data: List of time values (epoch seconds)
        signal_data: List of signal values
        window_size: Number of points to consider for derivative calculation
        normalization: Type of normalization to apply ('min', 'none')
        zero_mean: Whether to center the data by subtracting the mean
        r2_threshold: If provided, only keep derivatives where local fit R² exceeds this value
        resample_method: If provided with r2_threshold, resample filtered derivatives using this method
                        ('linear', 'spline', 'lowess', 'loess', or 'best')
    
    Returns:
        Tuple containing:
        - List of derivative values
        - List of step sizes used for each calculation
    '''
    # Validate input data has matching lengths
    if len(time_data) != len(signal_data):
        raise ValueError("Time and Signal data must have the same length")
    
    # Validate normalization parameter
    if normalization not in ['min', 'none']:
        raise ValueError("Normalization must be 'min' or 'none'")
    
    def slope_calc(i: int) -> Tuple[float, float, float]:
        # Calculate window boundaries centered around index i
        window_start = int(max(0, i - (window_size - 0.5) // 2))
        # Adjust for even/odd window sizes
        shift = 0 if window_size % 2 == 0 else 1
        # Ensure window end doesn't exceed data length
        window_end = int(min(len(time_data), i + (window_size - 0.5) // 2 + shift))
        
        # Extract window data
        time_window = np.array(time_data[window_start:window_end])
        signal_window = np.array(signal_data[window_start:window_end])
        
        # Apply normalization if requested
        if normalization == 'min':
            # Min normalization - shift data to start at zero
            min_time = np.min(time_window)
            min_signal = np.min(signal_window)
            time_window = time_window - min_time
            signal_window = signal_window - min_signal
        
        # Apply zero-mean centering if requested
        if zero_mean:
            # Center data by subtracting mean
            time_window = time_window - np.mean(time_window)
            signal_window = signal_window - np.mean(signal_window)
        
        # Perform linear regression on the window
        fit = linregress(time_window, signal_window)
        # Calculate effective step size
        step = (window_end - window_start)/window_size
        # Return slope, step size, and R² value
        return fit.slope, step, fit.rvalue**2
    
    # Calculate slopes for each point in the time series
    results = [slope_calc(i) for i in range(len(time_data))]
    # Extract derivative values and R² values
    derivative = [r[0] for r in results]
    steps = [r[1] for r in results]
    r_squared = [r[2] for r in results] if len(results[0]) > 2 else [1.0] * len(results)
    
    # Apply R² threshold filtering if requested
    if r2_threshold is not None:
        # Create mask for points that meet the threshold
        mask = np.array(r_squared) >= r2_threshold
        valid_indices = np.where(mask)[0]
        
        if len(valid_indices) == 0:
            # If no points meet threshold, return original results
            return derivative, steps
        
        # Get valid time points, derivatives, and steps
        valid_times = np.array(time_data)[valid_indices]
        valid_derivatives = np.array(derivative)[valid_indices]
        
        # If resampling is requested
        if resample_method and len(valid_indices) > 1:
            # Create interpolation function for derivatives
            if resample_method == 'best':
                interp_func, _, _ = get_best_interpolation(valid_times, valid_derivatives)
            elif resample_method == 'linear':
                interp_func = local_segmented_linear(valid_times, valid_derivatives)
            elif resample_method == 'spline':
                interp_func = spline_interpolation(valid_times, valid_derivatives)
            elif resample_method == 'lowess':
                interp_func = lowess_interpolation(valid_times, valid_derivatives)
            elif resample_method == 'loess':
                interp_func = loess_interpolation(valid_times, valid_derivatives)
            else:
                # Default to spline if method not recognized
                interp_func = spline_interpolation(valid_times, valid_derivatives)
            
            # Resample derivatives at all original time points
            derivative = interp_func(time_data).tolist()
    
    return derivative, steps

def gold(signal: np.ndarray, time: np.ndarray, embedding: int = 3, n: int = 2,
         r2_threshold: Optional[float] = None,
         resample_method: Optional[str] = None) -> Dict[str, Union[np.ndarray, int]]:
    """
    Calculate derivatives using the Generalized Orthogonal Local Derivative (GOLD) method.
    
    Args:
        signal: Array of signal values
        time: Array of time values corresponding to the signal
        embedding: Number of points to consider for derivative calculation
        n: Maximum order of derivative to estimate
        r2_threshold: If provided, only keep derivatives where local fit R² exceeds this value
        resample_method: If provided with r2_threshold, resample filtered derivatives using this method
                        ('linear', 'spline', 'lowess', 'loess', or 'best')
    
    Returns:
        Dictionary containing:
        - dtime: Time values for derivatives
        - dsignal: Matrix of derivatives (0th to nth order)
        - embedding: Embedding dimension used
        - n: Maximum order of derivatives calculated
        - r_squared: R² values for each point (if calculated)
    """
    # Validate input dimensions
    if len(signal) != len(time):
        raise ValueError("Signal and time vectors should have the same length.")
    # Ensure sufficient data points for embedding
    if len(signal) <= embedding:
        raise ValueError("Signal and time vectors should have a length greater than embedding.")
    # Ensure embedding dimension is sufficient for derivative order
    if n >= embedding:
        raise ValueError("The embedding dimension should be higher than the maximum order of the derivative, n.")
    
    # Create time and signal embedding matrices
    # Each column represents a shifted version of the original time/signal
    tembed = np.column_stack([time[i:len(time)-embedding+i+1] for i in range(embedding)])
    Xembed = np.column_stack([signal[i:len(signal)-embedding+i+1] for i in range(embedding)])
    
    # Initialize matrix to store derivatives
    derivatives = np.zeros((tembed.shape[0], n+1))
    
    # Initialize array to store R² values
    r_squared = np.zeros(tembed.shape[0])
    
    # Calculate derivatives for each window
    for k in range(tembed.shape[0]):
        # Center time values around the middle point of the window
        t = tembed[k] - tembed[k, embedding // 2]
        # Create basis functions (powers of t)
        Xi = np.vstack([t**q for q in range(n+1)])
        
        # Gram-Schmidt orthogonalization of the basis functions
        for q in range(1, n+1):
            for p in range(q):
                # Project higher order basis onto lower order and subtract
                Xi[q] -= np.dot(Xi[p], t**q) / np.dot(Xi[p], t**p) * Xi[p]
        
        # Scale basis functions by factorial for derivative calculation
        D = np.diag(1 / factorial(np.arange(n+1)))
        # Apply scaling to orthogonalized basis
        L = D @ Xi
        # Calculate weights for derivative estimation
        W = L.T @ np.linalg.inv(L @ L.T)
        # Compute derivatives by applying weights to signal values
        derivatives[k] = Xembed[k] @ W
        
        # Calculate R² for this window's fit
        # Use the 0th derivative (function value) to reconstruct the signal
        # W.T is the matrix that transforms derivatives back to signal values
        predicted = np.dot(derivatives[k, 0], np.ones(embedding))  # 0th derivative is constant
        ss_total = np.sum((Xembed[k] - np.mean(Xembed[k]))**2)
        ss_residual = np.sum((Xembed[k] - predicted)**2)
        r_squared[k] = 1 - (ss_residual / ss_total) if ss_total > 0 else 0
    
    # Calculate time points corresponding to derivatives (centered moving average)
    time_derivative = np.convolve(time, np.ones(embedding)/embedding, mode='valid')
    
    # Apply R² threshold filtering if requested
    if r2_threshold is not None:
        # Create mask for points that meet the threshold
        mask = r_squared >= r2_threshold
        valid_indices = np.where(mask)[0]
        
        if len(valid_indices) > 0:
            # Get valid time points and derivatives
            valid_times = time_derivative[valid_indices]
            valid_derivatives = derivatives[valid_indices]
            
            # If resampling is requested and we have enough valid points
            if resample_method and len(valid_indices) > 1:
                # Create interpolated derivatives for each order
                resampled_derivatives = np.zeros_like(derivatives)
                
                for j in range(n+1):
                    # Create interpolation function for this derivative order
                    if resample_method == 'best':
                        interp_func, _, _ = get_best_interpolation(valid_times, valid_derivatives[:, j])
                    elif resample_method == 'linear':
                        interp_func = local_segmented_linear(valid_times, valid_derivatives[:, j])
                    elif resample_method == 'spline':
                        interp_func = spline_interpolation(valid_times, valid_derivatives[:, j])
                    elif resample_method == 'lowess':
                        interp_func = lowess_interpolation(valid_times, valid_derivatives[:, j])
                    elif resample_method == 'loess':
                        interp_func = loess_interpolation(valid_times, valid_derivatives[:, j])
                    else:
                        # Default to spline if method not recognized
                        interp_func = spline_interpolation(valid_times, valid_derivatives[:, j])
                    
                    # Resample derivatives at all original time points
                    resampled_derivatives[:, j] = interp_func(time_derivative)
                
                # Replace derivatives with resampled values
                derivatives = resampled_derivatives
    
    # Return results as dictionary
    return {
        'dtime': time_derivative, 
        'dsignal': derivatives, 
        'embedding': embedding, 
        'n': n,
        'r_squared': r_squared
    }

def glla(signal: np.ndarray, time: np.ndarray, embedding: int = 3, n: int = 2,
         r2_threshold: Optional[float] = None,
         resample_method: Optional[str] = None) -> Dict[str, Union[np.ndarray, int]]:
    """
    Calculate derivatives using the Generalized Local Linear Approximation (GLLA) method.
    
    Args:
        signal: Array of signal values
        time: Array of time values corresponding to the signal
        embedding: Number of points to consider for derivative calculation
        n: Maximum order of derivative to calculate
        r2_threshold: If provided, only keep derivatives where local fit R² exceeds this value
        resample_method: If provided with r2_threshold, resample filtered derivatives using this method
                        ('linear', 'spline', 'lowess', 'loess', or 'best')
    
    Returns:
        Dictionary containing:
        - dtime: Time values for derivatives
        - dsignal: Matrix of derivatives (0th to nth order)
        - embedding: Embedding dimension used
        - n: Maximum order of derivatives calculated
        - r_squared: R² values for each point (if calculated)
    """
    # Validate input dimensions
    if len(signal) != len(time):
        raise ValueError("Signal and time vectors should have the same length.")
    # Ensure sufficient data points for embedding
    if len(signal) <= embedding:
        raise ValueError("Signal and time vectors should have a length greater than embedding.")
    # Ensure embedding dimension is sufficient for derivative order
    if n >= embedding:
        raise ValueError("The embedding dimension should be higher than the maximum order of the derivative, n.")
    
    # Calculate minimum time step for scaling
    deltat = np.min(np.diff(time))
    
    # Create design matrix with centered time indices raised to powers
    # Each column represents a different power (0 to n)
    # Each power is divided by factorial for Taylor series representation
    L = np.column_stack([(np.arange(1, embedding+1) - np.mean(np.arange(1, embedding+1)))**i / factorial(i) for i in range(n+1)])
    
    # Calculate weights matrix for derivative estimation
    W = L @ np.linalg.inv(L.T @ L)
    
    # Create signal embedding matrix (sliding window)
    Xembed = np.column_stack([signal[i:len(signal)-embedding+i+1] for i in range(embedding)])
    
    # Calculate derivatives by applying weights to signal values
    derivatives = Xembed @ W
    
    # Scale derivatives by appropriate powers of time step
    derivatives[:, 1:] /= deltat**np.arange(1, n+1)[None, :]
    
    # Calculate time points corresponding to derivatives (centered moving average)
    time_derivative = np.convolve(time, np.ones(embedding)/embedding, mode='valid')
    
    # Calculate R² for each window's fit
    r_squared = np.zeros(len(time_derivative))
    for k in range(len(time_derivative)):
        # Calculate R² for this window's fit using 0th derivative (function value)
        predicted = np.dot(derivatives[k, 0], np.ones(embedding))  # 0th derivative is constant
        ss_total = np.sum((Xembed[k] - np.mean(Xembed[k]))**2)
        ss_residual = np.sum((Xembed[k] - predicted)**2)
        r_squared[k] = 1 - (ss_residual / ss_total) if ss_total > 0 else 0
    
    # Apply R² threshold filtering if requested
    if r2_threshold is not None:
        # Create mask for points that meet the threshold
        mask = r_squared >= r2_threshold
        valid_indices = np.where(mask)[0]
        
        if len(valid_indices) > 0:
            # Get valid time points and derivatives
            valid_times = time_derivative[valid_indices]
            valid_derivatives = derivatives[valid_indices]
            
            # If resampling is requested and we have enough valid points
            if resample_method and len(valid_indices) > 1:
                # Create interpolated derivatives for each order
                resampled_derivatives = np.zeros_like(derivatives)
                
                for j in range(n+1):
                    # Create interpolation function for this derivative order
                    if resample_method == 'best':
                        interp_func, _, _ = get_best_interpolation(valid_times, valid_derivatives[:, j])
                    elif resample_method == 'linear':
                        interp_func = local_segmented_linear(valid_times, valid_derivatives[:, j])
                    elif resample_method == 'spline':
                        interp_func = spline_interpolation(valid_times, valid_derivatives[:, j])
                    elif resample_method == 'lowess':
                        interp_func = lowess_interpolation(valid_times, valid_derivatives[:, j])
                    elif resample_method == 'loess':
                        interp_func = loess_interpolation(valid_times, valid_derivatives[:, j])
                    else:
                        # Default to spline if method not recognized
                        interp_func = spline_interpolation(valid_times, valid_derivatives[:, j])
                    
                    # Resample derivatives at all original time points
                    resampled_derivatives[:, j] = interp_func(time_derivative)
                
                # Replace derivatives with resampled values
                derivatives = resampled_derivatives
    
    # Return results as dictionary
    return {
        'dtime': time_derivative, 
        'dsignal': derivatives, 
        'embedding': embedding, 
        'n': n,
        'r_squared': r_squared
    }

def fda(signal: np.ndarray, time: np.ndarray, spar: Optional[float] = None,
         r2_threshold: Optional[float] = None,
         resample_method: Optional[str] = None) -> Dict[str, Union[np.ndarray, float, None]]:
    """
    Calculate derivatives using the Functional Data Analysis (FDA) method.
    
    Args:
        signal: Array of signal values
        time: Array of time values corresponding to the signal
        spar: Smoothing parameter for the spline. If None, automatically determined
        r2_threshold: If provided, only keep derivatives where local fit R² exceeds this value
        resample_method: If provided with r2_threshold, resample filtered derivatives using this method
                        ('linear', 'spline', 'lowess', 'loess', or 'best')
    
    Returns:
        Dictionary containing:
        - dtime: Time values for derivatives
        - dsignal: Matrix of derivatives (0th to 2nd order)
        - spar: Smoothing parameter used
        - r_squared: R² values for each point (if calculated)
    """
    # If spar is None, estimate it based on data characteristics
    if spar is None:
        # Use a heuristic based on data length and range
        n = len(signal)
        # Calculate peak-to-peak range of signal
        range_y = np.ptp(signal)
        # Set smoothing parameter proportional to data length and squared range
        spar = n * (0.01 * range_y) ** 2

    # Create univariate spline with specified smoothing parameter
    spline = UnivariateSpline(time, signal, s=spar)
    
    # Evaluate the spline (0th derivative) at original time points
    d0 = spline(time)
    # Evaluate first derivative at original time points
    d1 = spline.derivative(n=1)(time)
    # Evaluate second derivative at original time points
    d2 = spline.derivative(n=2)(time)
    
    # Combine all derivatives into a single matrix
    derivatives = np.column_stack([d0, d1, d2])
    
    # Calculate R² for the spline fit
    ss_total = np.sum((signal - np.mean(signal))**2)
    ss_residual = np.sum((signal - d0)**2)
    r_squared_global = 1 - (ss_residual / ss_total) if ss_total > 0 else 0
    
    # Create array of R² values (same for all points since it's a global fit)
    r_squared = np.full(len(time), r_squared_global)
    
    # Apply R² threshold filtering if requested
    if r2_threshold is not None and r_squared_global < r2_threshold:
        # If global R² is below threshold and resampling is requested
        if resample_method:
            # Try a different interpolation method for each derivative order
            for j in range(3):  # 0th, 1st, and 2nd derivatives
                if resample_method == 'best':
                    interp_func, _, _ = get_best_interpolation(time, derivatives[:, j])
                elif resample_method == 'linear':
                    interp_func = local_segmented_linear(time, derivatives[:, j])
                elif resample_method == 'lowess':
                    interp_func = lowess_interpolation(time, derivatives[:, j])
                elif resample_method == 'loess':
                    interp_func = loess_interpolation(time, derivatives[:, j])
                else:
                    # Default to spline with different smoothing
                    alt_spar = spar * 0.5 if spar is not None else None
                    interp_func = spline_interpolation(time, derivatives[:, j], smoothing=alt_spar)
                
                # Replace derivatives with resampled values
                derivatives[:, j] = interp_func(time)
    
    # Return results as dictionary
    return {
        'dtime': time, 
        'dsignal': derivatives, 
        'spar': spar,
        'r_squared': r_squared
    }
