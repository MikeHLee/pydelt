"""
Functions for interpolating time series data using various methods.
"""

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.stats import linregress
from statsmodels.nonparametric.smoothers_lowess import lowess
from typing import List, Tuple, Dict, Union, Optional, Callable

def local_segmented_linear(
    time: Union[List[float], np.ndarray],
    signal: Union[List[float], np.ndarray],
    window_size: int = 5,
    force_continuous: bool = True
) -> Callable[[Union[float, np.ndarray]], np.ndarray]:
    """
    Interpolate a time series using local segmented linear regression.
    
    Args:
        time: Time points of the original signal
        signal: Signal values
        window_size: Size of the window for local linear regression
        force_continuous: If True, ensures the interpolation is continuous at segment boundaries
        
    Returns:
        Callable function that interpolates the signal at any time point
    """
    # Convert inputs to numpy arrays
    t = np.asarray(time)
    s = np.asarray(signal)
    
    # Ensure data is sorted by time
    sort_idx = np.argsort(t)
    t = t[sort_idx]
    s = s[sort_idx]
    
    # Define segment boundaries
    if window_size >= len(t):
        # If window is larger than data, use a single segment
        segments = [(0, len(t) - 1)]
    else:
        # Create overlapping segments
        step = max(1, window_size // 2)
        starts = list(range(0, len(t) - window_size + 1, step))
        # Ensure the last segment covers the end of the data
        if starts[-1] + window_size < len(t):
            starts.append(len(t) - window_size)
        segments = [(i, i + window_size - 1) for i in starts]
    
    # Calculate linear regression for each segment
    segment_models = []
    for start, end in segments:
        segment_t = t[start:end+1]
        segment_s = s[start:end+1]
        
        # Linear regression for this segment
        slope, intercept, r_value, p_value, std_err = linregress(segment_t, segment_s)
        segment_models.append({
            'start_time': segment_t[0],
            'end_time': segment_t[-1],
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_value**2
        })
    
    # Function to interpolate at a given time point
    def interpolate(query_time):
        query_time = np.asarray(query_time)
        scalar_input = query_time.ndim == 0
        if scalar_input:
            query_time = np.array([query_time])
        
        result = np.zeros_like(query_time, dtype=float)
        
        for i, t_i in enumerate(query_time):
            # Find applicable segments
            applicable_segments = [
                m for m in segment_models 
                if m['start_time'] <= t_i <= m['end_time']
            ]
            
            if not applicable_segments:
                # If outside all segments, use the nearest segment
                distances = [
                    min(abs(t_i - m['start_time']), abs(t_i - m['end_time']))
                    for m in segment_models
                ]
                nearest_segment = segment_models[np.argmin(distances)]
                result[i] = nearest_segment['slope'] * t_i + nearest_segment['intercept']
            elif len(applicable_segments) == 1:
                # If only one segment applies, use it directly
                segment = applicable_segments[0]
                result[i] = segment['slope'] * t_i + segment['intercept']
            else:
                # If multiple segments apply, use weighted average based on distance from segment centers
                if force_continuous:
                    # Calculate weights based on distance from segment boundaries
                    weights = []
                    for segment in applicable_segments:
                        # Distance from boundaries (closer to center = higher weight)
                        dist_from_start = (t_i - segment['start_time']) / (segment['end_time'] - segment['start_time'])
                        dist_from_end = (segment['end_time'] - t_i) / (segment['end_time'] - segment['start_time'])
                        # Weight is higher when point is further from boundaries
                        weight = min(dist_from_start, dist_from_end) * 2  # Scale to [0, 1]
                        weights.append(weight)
                    
                    # Normalize weights
                    weights = np.array(weights)
                    weights = weights / np.sum(weights)
                    
                    # Weighted average of segment predictions
                    segment_predictions = [
                        segment['slope'] * t_i + segment['intercept']
                        for segment in applicable_segments
                    ]
                    result[i] = np.sum(np.array(segment_predictions) * weights)
                else:
                    # Use the segment with highest r-squared
                    best_segment = max(applicable_segments, key=lambda x: x['r_squared'])
                    result[i] = best_segment['slope'] * t_i + best_segment['intercept']
        
        return result[0] if scalar_input else result
    
    return interpolate

def spline_interpolation(
    time: Union[List[float], np.ndarray],
    signal: Union[List[float], np.ndarray],
    smoothing: Optional[float] = None,
    k: int = 3
) -> Callable[[Union[float, np.ndarray]], np.ndarray]:
    """
    Interpolate a time series using spline interpolation.
    
    Args:
        time: Time points of the original signal
        signal: Signal values
        smoothing: Smoothing factor for the spline. If None, it's automatically determined
        k: Degree of the spline (1=linear, 2=quadratic, 3=cubic)
        
    Returns:
        Callable function that interpolates the signal at any time point
    """
    # Convert inputs to numpy arrays
    t = np.asarray(time)
    s = np.asarray(signal)
    
    # Ensure data is sorted by time
    sort_idx = np.argsort(t)
    t = t[sort_idx]
    s = s[sort_idx]
    
    # If smoothing is None, estimate it based on data characteristics
    if smoothing is None:
        n = len(signal)
        range_y = np.ptp(signal)
        smoothing = n * (0.01 * range_y) ** 2
    
    # Create the spline
    spline = UnivariateSpline(t, s, s=smoothing, k=k)
    
    # Return the spline as the interpolation function
    return spline

def lowess_interpolation(
    time: Union[List[float], np.ndarray],
    signal: Union[List[float], np.ndarray],
    frac: float = 0.3,
    it: int = 3
) -> Callable[[Union[float, np.ndarray]], np.ndarray]:
    """
    Interpolate a time series using LOWESS (Locally Weighted Scatterplot Smoothing).
    
    Args:
        time: Time points of the original signal
        signal: Signal values
        frac: Between 0 and 1. The fraction of the data used when estimating each y-value
        it: Number of robustifying iterations
        
    Returns:
        Callable function that interpolates the signal at any time point
    """
    # Convert inputs to numpy arrays
    t = np.asarray(time)
    s = np.asarray(signal)
    
    # Ensure data is sorted by time
    sort_idx = np.argsort(t)
    t = t[sort_idx]
    s = s[sort_idx]
    
    # Apply LOWESS smoothing
    smoothed = lowess(s, t, frac=frac, it=it, return_sorted=True)
    
    # Create an interpolation function from the smoothed data
    # Use linear interpolation between the LOWESS points
    interp_func = interp1d(
        smoothed[:, 0], smoothed[:, 1], 
        bounds_error=False, 
        fill_value=(smoothed[0, 1], smoothed[-1, 1])
    )
    
    return interp_func

def loess_interpolation(
    time: Union[List[float], np.ndarray],
    signal: Union[List[float], np.ndarray],
    degree: int = 2,
    frac: float = 0.3,
    it: int = 3
) -> Callable[[Union[float, np.ndarray]], np.ndarray]:
    """
    Interpolate a time series using LOESS (LOcally Estimated Scatterplot Smoothing).
    This is similar to LOWESS but uses higher-degree local polynomials.
    
    Args:
        time: Time points of the original signal
        signal: Signal values
        degree: Degree of local polynomials (1=linear, 2=quadratic)
        frac: Between 0 and 1. The fraction of the data used when estimating each y-value
        it: Number of robustifying iterations
        
    Returns:
        Callable function that interpolates the signal at any time point
    """
    # Convert inputs to numpy arrays
    t = np.asarray(time)
    s = np.asarray(signal)
    
    # Ensure data is sorted by time
    sort_idx = np.argsort(t)
    t = t[sort_idx]
    s = s[sort_idx]
    
    # For now, we'll implement LOESS as a wrapper around LOWESS
    # In a future version, this could be replaced with a true LOESS implementation
    # that supports higher-degree local polynomials
    
    # Apply LOWESS smoothing
    smoothed = lowess(s, t, frac=frac, it=it, return_sorted=True)
    
    # Create an interpolation function from the smoothed data
    # Use cubic spline interpolation to better approximate higher-degree polynomials
    interp_func = interp1d(
        smoothed[:, 0], smoothed[:, 1], 
        kind='cubic' if len(t) > 3 else 'linear',
        bounds_error=False, 
        fill_value=(smoothed[0, 1], smoothed[-1, 1])
    )
    
    return interp_func

def calculate_fit_quality(
    time: Union[List[float], np.ndarray],
    signal: Union[List[float], np.ndarray],
    interpolation_func: Callable[[Union[float, np.ndarray]], np.ndarray]
) -> Dict[str, float]:
    """
    Calculate the quality of fit for an interpolation function.
    
    Args:
        time: Time points of the original signal
        signal: Signal values
        interpolation_func: Interpolation function to evaluate
        
    Returns:
        Dictionary with quality metrics (r_squared, rmse)
    """
    # Convert inputs to numpy arrays
    t = np.asarray(time)
    s = np.asarray(signal)
    
    # Predict values using the interpolation function
    predicted = interpolation_func(t)
    
    # Calculate R-squared
    ss_total = np.sum((s - np.mean(s))**2)
    ss_residual = np.sum((s - predicted)**2)
    r_squared = 1 - (ss_residual / ss_total) if ss_total > 0 else 0
    
    # Calculate RMSE
    rmse = np.sqrt(np.mean((s - predicted)**2))
    
    return {
        'r_squared': r_squared,
        'rmse': rmse
    }

def get_best_interpolation(
    time: Union[List[float], np.ndarray],
    signal: Union[List[float], np.ndarray],
    methods: List[str] = ['linear', 'spline', 'lowess', 'loess'],
    metric: str = 'r_squared'
) -> Tuple[Callable[[Union[float, np.ndarray]], np.ndarray], str, float]:
    """
    Find the best interpolation method based on fit quality.
    
    Args:
        time: Time points of the original signal
        signal: Signal values
        methods: List of interpolation methods to try
        metric: Metric to use for comparison ('r_squared' or 'rmse')
        
    Returns:
        Tuple containing:
        - Best interpolation function
        - Name of the best method
        - Value of the quality metric for the best method
    """
    # Convert inputs to numpy arrays
    t = np.asarray(time)
    s = np.asarray(signal)
    
    # Dictionary of interpolation methods
    method_funcs = {
        'linear': lambda: local_segmented_linear(t, s, window_size=5, force_continuous=True),
        'spline': lambda: spline_interpolation(t, s),
        'lowess': lambda: lowess_interpolation(t, s),
        'loess': lambda: loess_interpolation(t, s)
    }
    
    # Try each method and calculate fit quality
    results = {}
    for method in methods:
        if method in method_funcs:
            try:
                interp_func = method_funcs[method]()
                quality = calculate_fit_quality(t, s, interp_func)
                results[method] = {
                    'function': interp_func,
                    'r_squared': quality['r_squared'],
                    'rmse': quality['rmse']
                }
            except Exception as e:
                print(f"Error with {method} interpolation: {e}")
    
    # Find the best method based on the specified metric
    if metric == 'r_squared':
        # Higher is better for R-squared
        best_method = max(results.items(), key=lambda x: x[1]['r_squared'])
    else:
        # Lower is better for RMSE
        best_method = min(results.items(), key=lambda x: x[1]['rmse'])
    
    method_name = best_method[0]
    method_info = best_method[1]
    
    return method_info['function'], method_name, method_info[metric]
