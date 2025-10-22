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

def lla(input_data: Union[List[float], np.ndarray], output_data: Union[List[float], np.ndarray], 
        window_size: Optional[int] = 5, normalization: str = 'min', zero_mean: bool = False,
        r2_threshold: Optional[float] = None, resample_method: Optional[str] = None,
        # Backward compatibility parameters
        time_data: Optional[Union[List[int], np.ndarray]] = None, 
        signal_data: Optional[Union[List[float], np.ndarray]] = None) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Local Linear Approximation (LLA) method for estimating derivatives.
    Supports univariate and multivariate input/output data.
    Uses configurable normalization and linear regression within a sliding window.
    
    Args:
        input_data: Array of input values, shape (N,) for univariate or (N, n_in) for multivariate
        output_data: Array of output values, shape (N,) for univariate or (N, n_out) for multivariate
        window_size: Number of points to consider for derivative calculation
        normalization: Type of normalization to apply ('min', 'none')
        zero_mean: Whether to center the data by subtracting the mean
        r2_threshold: If provided, only keep derivatives where local fit R² exceeds this value
        resample_method: If provided with r2_threshold, resample filtered derivatives using this method
                        ('linear', 'spline', 'lowess', 'loess', or 'best')
        time_data: DEPRECATED - use input_data instead
        signal_data: DEPRECATED - use output_data instead
    
    Returns:
        Tuple containing:
        - Array of derivative values, shape (N, n_out) for univariate input or (N, n_out, n_in) for multivariate
        - Array of step sizes used for each calculation
    '''
    # Handle backward compatibility
    if time_data is not None and signal_data is not None:
        warnings.warn("time_data and signal_data parameters are deprecated. Use input_data and output_data instead.", 
                     DeprecationWarning, stacklevel=2)
        input_data = time_data
        output_data = signal_data
    elif time_data is not None or signal_data is not None:
        raise ValueError("If using deprecated parameters, both time_data and signal_data must be provided")
    
    # Convert to numpy arrays
    input_data = np.asarray(input_data)
    output_data = np.asarray(output_data)
    
    # Handle input dimensions
    if input_data.ndim == 1:
        input_data = input_data[:, None]  # Shape: (N, 1)
    n_in = input_data.shape[1]
    
    # Handle output dimensions  
    if output_data.ndim == 1:
        output_data = output_data[:, None]  # Shape: (N, 1)
    n_out = output_data.shape[1]
    
    if input_data.shape[0] != output_data.shape[0]:
        raise ValueError("Input and output data must have the same number of samples")
    
    # For univariate input, compute derivatives with respect to the single input dimension
    if n_in == 1:
        input_col = input_data[:, 0]
        derivatives = []
        steps = []
        
        for j in range(n_out):
            output_col = output_data[:, j]
            def slope_calc(i: int) -> Tuple[float, float, float]:
                window_start = int(max(0, i - (window_size - 0.5) // 2))
                shift = 0 if window_size % 2 == 0 else 1
                window_end = int(min(len(input_col), i + (window_size - 0.5) // 2 + shift))
                input_window = np.array(input_col[window_start:window_end])
                output_window = np.array(output_col[window_start:window_end])
                if normalization == 'min':
                    min_input = np.min(input_window)
                    min_output = np.min(output_window)
                    input_window = input_window - min_input
                    output_window = output_window - min_output
                if zero_mean:
                    input_window = input_window - np.mean(input_window)
                    output_window = output_window - np.mean(output_window)
                fit = linregress(input_window, output_window)
                step = (window_end - window_start)/window_size
                return fit.slope, step, fit.rvalue**2
            
            results = [slope_calc(i) for i in range(len(input_col))]
            deriv_col = [r[0] for r in results]
            step_col = [r[1] for r in results]
            r_squared = [r[2] for r in results] if len(results[0]) > 2 else [1.0] * len(results)
            
            if r2_threshold is not None:
                mask = np.array(r_squared) >= r2_threshold
                valid_indices = np.where(mask)[0]
                if resample_method and len(valid_indices) > 0:
                    methods = {
                        'best': get_best_interpolation,
                        'linear': local_segmented_linear,
                        'spline': spline_interpolation,
                        'lowess': lowess_interpolation,
                        'loess': loess_interpolation
                    }
                    valid_inputs = np.array(input_col)[valid_indices]
                    valid_derivatives = np.array(deriv_col)[valid_indices]
                    interp_func = methods.get(resample_method, spline_interpolation)(
                        valid_inputs, valid_derivatives
                    )
                    deriv_col = interp_func(input_col)
        
            derivatives.append(deriv_col)
            steps.append(step_col)
        
        # Convert lists to numpy arrays
        derivatives = np.array(derivatives).T  # Transpose to get shape (N, n_out)
        steps = np.array(steps).T  # Transpose to get shape (N, n_out)
        
        # If we have only one output dimension, flatten the arrays
        if n_out == 1:
            derivatives = derivatives.flatten()
            steps = steps.flatten()
            
        return derivatives, steps
    
    else:
        # Multivariate input case - compute partial derivatives
        # For now, raise NotImplementedError as this requires more complex implementation
        raise NotImplementedError("Multivariate input support for LLA is not yet implemented. "
                                "Use neural network methods for multivariate derivatives.")

def gold(input_data: np.ndarray, output_data: np.ndarray, embedding: int = 3, n: int = 2,
         r2_threshold: Optional[float] = None, resample_method: Optional[str] = None,
         # Backward compatibility parameters
         signal: Optional[np.ndarray] = None, time: Optional[np.ndarray] = None) -> Dict[str, Union[np.ndarray, int]]:
    """
    Calculate derivatives using the Generalized Orthogonal Local Derivative (GOLD) method.
    
    Args:
        input_data: Array of input values, shape (N,) for univariate or (N, n_in) for multivariate
        output_data: Array of output values, shape (N,) for univariate or (N, n_out) for multivariate
        embedding: Number of points to consider for derivative calculation
        n: Maximum order of derivative to estimate
        r2_threshold: If provided, only keep derivatives where local fit R² exceeds this value
        resample_method: If provided with r2_threshold, resample filtered derivatives using this method
                        ('linear', 'spline', 'lowess', 'loess', or 'best')
        signal: DEPRECATED - use output_data instead
        time: DEPRECATED - use input_data instead
    
    Returns:
        Dictionary containing:
        - dinput: Input values for derivatives
        - doutput: Matrix of derivatives (0th to nth order)
        - embedding: Embedding dimension used
        - n: Maximum order of derivatives calculated
        - r_squared: R² values for each point (if calculated)
    """
    # Handle backward compatibility
    if signal is not None and time is not None:
        import warnings
        warnings.warn("signal and time parameters are deprecated. Use input_data and output_data instead.", 
                     DeprecationWarning, stacklevel=2)
        input_data = time
        output_data = signal
    elif signal is not None or time is not None:
        raise ValueError("If using deprecated parameters, both signal and time must be provided")
    
    # Convert to numpy arrays and handle dimensions
    input_data = np.asarray(input_data)
    output_data = np.asarray(output_data)
    
    # For now, only support univariate input
    if input_data.ndim > 1:
        raise NotImplementedError("Multivariate input support for GOLD is not yet implemented. "
                                "Use neural network methods for multivariate derivatives.")
    
    # Validate input dimensions
    if len(output_data) != len(input_data):
        raise ValueError("Output and input vectors should have the same length.")
    # Ensure sufficient data points for embedding
    if len(output_data) <= embedding:
        raise ValueError("Output and input vectors should have a length greater than embedding.")
    # Ensure embedding dimension is sufficient for derivative order
    if n >= embedding:
        raise ValueError("The embedding dimension should be higher than the maximum order of the derivative, n.")
    
    # Create input and output embedding matrices
    # Each column represents a shifted version of the original input/output
    tembed = np.column_stack([input_data[i:len(input_data)-embedding+i+1] for i in range(embedding)])
    Xembed = np.column_stack([output_data[i:len(output_data)-embedding+i+1] for i in range(embedding)])
    
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
    
    # Calculate input points corresponding to derivatives (centered moving average)
    input_derivative = np.convolve(input_data, np.ones(embedding)/embedding, mode='valid')
    
    # Apply R² threshold filtering if requested
    if r2_threshold is not None:
        # Create mask for points that meet the threshold
        mask = r_squared >= r2_threshold
        valid_indices = np.where(mask)[0]
        
        if len(valid_indices) > 0:
            # Get valid time points and derivatives
            valid_inputs = input_derivative[valid_indices]
            valid_derivatives = derivatives[valid_indices]
            
            # If resampling is requested and we have enough valid points
            if resample_method and len(valid_indices) > 1:
                # Create interpolated derivatives for each order
                resampled_derivatives = np.zeros_like(derivatives)
                
                for j in range(n+1):
                    # Create interpolation function for this derivative order
                    if resample_method == 'best':
                        interp_func, _, _ = get_best_interpolation(valid_inputs, valid_derivatives[:, j])
                    elif resample_method == 'linear':
                        interp_func = local_segmented_linear(valid_inputs, valid_derivatives[:, j])
                    elif resample_method == 'spline':
                        interp_func = spline_interpolation(valid_inputs, valid_derivatives[:, j])
                    elif resample_method == 'lowess':
                        interp_func = lowess_interpolation(valid_inputs, valid_derivatives[:, j])
                    elif resample_method == 'loess':
                        interp_func = loess_interpolation(valid_inputs, valid_derivatives[:, j])
                    else:
                        # Default to spline if method not recognized
                        interp_func = spline_interpolation(valid_inputs, valid_derivatives[:, j])
                    
                    # Resample derivatives at all original time points
                    resampled_derivatives[:, j] = interp_func(input_derivative)
                
                # Replace derivatives with resampled values
                derivatives = resampled_derivatives
    
    # Return results as dictionary
    return {
        'dinput': input_derivative, 
        'doutput': derivatives, 
        'embedding': embedding, 
        'n': n,
        'r_squared': r_squared
    }

def glla(input_data: np.ndarray, output_data: np.ndarray, embedding: int = 3, n: int = 2,
         r2_threshold: Optional[float] = None, resample_method: Optional[str] = None,
         # Backward compatibility parameters
         signal: Optional[np.ndarray] = None, time: Optional[np.ndarray] = None) -> Dict[str, Union[np.ndarray, int]]:
    """
    Calculate derivatives using the Generalized Local Linear Approximation (GLLA) method.
    
    Args:
        input_data: Array of input values, shape (N,) for univariate or (N, n_in) for multivariate
        output_data: Array of output values, shape (N,) for univariate or (N, n_out) for multivariate
        embedding: Number of points to consider for derivative calculation
        n: Maximum order of derivative to calculate
        r2_threshold: If provided, only keep derivatives where local fit R² exceeds this value
        resample_method: If provided with r2_threshold, resample filtered derivatives using this method
                        ('linear', 'spline', 'lowess', 'loess', or 'best')
        signal: DEPRECATED - use output_data instead
        time: DEPRECATED - use input_data instead
    
    Returns:
        Dictionary containing:
        - dinput: Input values for derivatives
        - doutput: Matrix of derivatives (0th to nth order)
        - embedding: Embedding dimension used
        - n: Maximum order of derivatives calculated
        - r_squared: R² values for each point (if calculated)
    """
    # Validate input dimensions
    # Handle backward compatibility
    if signal is not None and time is not None:
        warnings.warn("signal and time parameters are deprecated. Use input_data and output_data instead.", 
                     DeprecationWarning, stacklevel=2)
        input_data = time
        output_data = signal
    elif signal is not None or time is not None:
        raise ValueError("If using deprecated parameters, both signal and time must be provided")
    
    # Convert to numpy arrays and handle dimensions
    input_data = np.asarray(input_data)
    output_data = np.asarray(output_data)
    
    # For now, only support univariate input
    if input_data.ndim > 1:
        raise NotImplementedError("Multivariate input support for GLLA is not yet implemented. "
                                "Use neural network methods for multivariate derivatives.")
    
    if len(output_data) != len(input_data):
        raise ValueError("Output and input vectors should have the same length.")
    # Ensure sufficient data points for embedding
    if len(output_data) <= embedding:
        raise ValueError("Output and input vectors should have a length greater than embedding.")
    # Ensure embedding dimension is sufficient for derivative order
    if n >= embedding:
        raise ValueError("The embedding dimension should be higher than the maximum order of the derivative, n.")
    
    # Calculate minimum input step for scaling
    deltat = np.min(np.diff(input_data))
    
    # Create design matrix with centered time indices raised to powers
    # Each column represents a different power (0 to n)
    # Each power is divided by factorial for Taylor series representation
    L = np.column_stack([(np.arange(1, embedding+1) - np.mean(np.arange(1, embedding+1)))**i / factorial(i) for i in range(n+1)])
    
    # Calculate weights matrix for derivative estimation
    W = L @ np.linalg.inv(L.T @ L)
    
    # Create input and output embedding matrices (sliding windows)
    tembed = np.column_stack([input_data[i:len(input_data)-embedding+i+1] for i in range(embedding)])
    Xembed = np.column_stack([output_data[i:len(output_data)-embedding+i+1] for i in range(embedding)])
    
    # Calculate derivatives by applying weights to signal values
    derivatives = Xembed @ W
    
    # Scale derivatives by appropriate powers of time step
    derivatives[:, 1:] /= deltat**np.arange(1, n+1)[None, :]
    
    # Calculate time points corresponding to derivatives (centered moving average)
    input_derivative = np.convolve(input_data, np.ones(embedding)/embedding, mode='valid')
    
    # Calculate R² for each window's fit
    r_squared = np.zeros(len(input_derivative))
    for k in range(len(input_derivative)):
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
            valid_inputs = input_derivative[valid_indices]
            valid_derivatives = derivatives[valid_indices]
            
            # If resampling is requested and we have enough valid points
            if resample_method and len(valid_indices) > 1:
                # Create interpolated derivatives for each order
                resampled_derivatives = np.zeros_like(derivatives)
                
                for j in range(n+1):
                    # Create interpolation function for this derivative order
                    if resample_method == 'best':
                        interp_func, _, _ = get_best_interpolation(valid_inputs, valid_derivatives[:, j])
                    elif resample_method == 'linear':
                        interp_func = local_segmented_linear(valid_inputs, valid_derivatives[:, j])
                    elif resample_method == 'spline':
                        interp_func = spline_interpolation(valid_inputs, valid_derivatives[:, j])
                    elif resample_method == 'lowess':
                        interp_func = lowess_interpolation(valid_inputs, valid_derivatives[:, j])
                    elif resample_method == 'loess':
                        interp_func = loess_interpolation(valid_inputs, valid_derivatives[:, j])
                    else:
                        # Default to spline if method not recognized
                        interp_func = spline_interpolation(valid_inputs, valid_derivatives[:, j])
                    
                    # Resample derivatives at all original time points
                    resampled_derivatives[:, j] = interp_func(input_derivative)
                
                # Replace derivatives with resampled values
                derivatives = resampled_derivatives
    
    # Return results as dictionary
    return {
        'dinput': input_derivative, 
        'doutput': derivatives, 
        'embedding': embedding, 
        'n': n,
        'r_squared': r_squared
    }

def fda(input_data: np.ndarray, output_data: np.ndarray, spar: Optional[float] = None,
         r2_threshold: Optional[float] = None, resample_method: Optional[str] = None,
         # Backward compatibility parameters
         signal: Optional[np.ndarray] = None, time: Optional[np.ndarray] = None) -> Dict[str, Union[np.ndarray, float, None]]:
    """
    Calculate derivatives using the Functional Data Analysis (FDA) method.
    Supports vector-valued outputs (multiple columns).
    
    Args:
        input_data: Array of input values, shape (N,) for univariate or (N, n_in) for multivariate
        output_data: Array of output values, shape (N,) or (N, n_out)
        spar: Smoothing parameter for the spline. If None, automatically determined
        r2_threshold: If provided, only keep derivatives where local fit R² exceeds this value
        resample_method: If provided with r2_threshold, resample filtered derivatives using this method
                        ('linear', 'spline', 'lowess', 'loess', or 'best')
        signal: DEPRECATED - use output_data instead
        time: DEPRECATED - use input_data instead
    
    Returns:
        Dictionary containing:
        - dinput: Input values for derivatives
        - doutput: Matrix of derivatives (0th to 2nd order, shape (N, n_out, 3))
        - spar: Smoothing parameter used
        - r_squared: R² values for each point (if calculated)
    """
    # Handle backward compatibility
    if signal is not None and time is not None:
        import warnings
        warnings.warn("signal and time parameters are deprecated. Use input_data and output_data instead.", 
                     DeprecationWarning, stacklevel=2)
        input_data = time
        output_data = signal
    elif signal is not None or time is not None:
        raise ValueError("If using deprecated parameters, both signal and time must be provided")
    
    # Convert to numpy arrays and handle dimensions
    input_data = np.asarray(input_data)
    output_data = np.asarray(output_data)
    
    # For now, only support univariate input
    if input_data.ndim > 1:
        raise NotImplementedError("Multivariate input support for FDA is not yet implemented. "
                                "Use neural network methods for multivariate derivatives.")
    
    if output_data.ndim == 1:
        output_data = output_data[:, None]
    n_out = output_data.shape[1]
    derivatives = []
    r_squared_list = []
    for j in range(n_out):
        sig_col = output_data[:, j]
        # If spar is None, estimate it based on data characteristics
        spar_j = spar
        if spar_j is None:
            n = len(sig_col)
            range_y = np.ptp(sig_col)
            spar_j = n * (0.01 * range_y) ** 2
        spline = UnivariateSpline(input_data, sig_col, s=spar_j)
        d0 = spline(input_data)
        d1 = spline.derivative(n=1)(input_data)
        d2 = spline.derivative(n=2)(input_data)
        deriv_mat = np.column_stack([d0, d1, d2])
        derivatives.append(deriv_mat)
        ss_total = np.sum((sig_col - np.mean(sig_col))**2)
        ss_residual = np.sum((sig_col - d0)**2)
        r_squared_global = 1 - (ss_residual / ss_total) if ss_total > 0 else 0
        r_squared_list.append(np.full(len(input_data), r_squared_global))
        # Apply R² threshold filtering if requested
        if r2_threshold is not None and r_squared_global < r2_threshold:
            if resample_method:
                methods = {
                    'best': get_best_interpolation,
                    'linear': local_segmented_linear,
                    'spline': spline_interpolation,
                    'lowess': lowess_interpolation,
                    'loess': loess_interpolation
                }
                for k in range(3):
                    interp_func = methods.get(resample_method, spline_interpolation)(input_data, deriv_mat[:, k])
                    deriv_mat[:, k] = interp_func(input_data)
        derivatives[-1] = deriv_mat
    derivatives = np.stack(derivatives, axis=1)  # (N, n_out, 3)
    r_squared = np.stack(r_squared_list, axis=1)  # (N, n_out)
    if derivatives.shape[1] == 1:
        derivatives = derivatives[:, 0, :]
        r_squared = r_squared[:, 0]
    return {
        'dinput': input_data,
        'doutput': derivatives,
        'spar': spar,
        'r_squared': r_squared
    }
