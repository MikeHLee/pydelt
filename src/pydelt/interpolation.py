"""
Functions for interpolating time series data using various methods.
"""

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.stats import linregress
from statsmodels.nonparametric.smoothers_lowess import lowess
from typing import List, Tuple, Dict, Union, Optional, Callable, Any
from scipy.interpolate import UnivariateSpline
import numpy as np

class BaseInterpolator:
    """
    Abstract base class for all interpolators.
    """
    def fit(self, time, signal):
        raise NotImplementedError("fit() must be implemented in subclasses.")
    def predict(self, new_time):
        raise NotImplementedError("predict() must be implemented in subclasses.")

class SplineInterpolator(BaseInterpolator):
    def __init__(self, smoothing: Optional[float] = None, k: int = 5):
        self.smoothing = smoothing
        self.k = k
        self.spline = None

    def fit(self, time: Union[List[float], np.ndarray], signal: Union[List[float], np.ndarray]):
        t = np.asarray(time)
        s = np.asarray(signal)
        sort_idx = np.argsort(t)
        t = t[sort_idx]
        s = s[sort_idx]
        smoothing = self.smoothing
        if smoothing is None:
            n = len(s)
            range_y = np.ptp(s)
            noise_estimate = np.std(np.diff(s)) / np.sqrt(2)
            smoothing = n * (0.005 * range_y + 0.1 * noise_estimate) ** 2
        self.spline = UnivariateSpline(t, s, s=smoothing, k=self.k)
        return self

    def predict(self, new_time: Union[float, np.ndarray]):
        if self.spline is None:
            raise RuntimeError("SplineInterpolator must be fit before calling predict().")
        return self.spline(new_time)

# DEPRECATED: Old function-based interface. Use SplineInterpolator instead.

class LowessInterpolator(BaseInterpolator):
    def __init__(self, frac: float = 0.1, it: int = 3):
        self.frac = frac
        self.it = it
        self.smoothed = None
        self.interp_func = None

    def fit(self, time: Union[List[float], np.ndarray], signal: Union[List[float], np.ndarray]):
        t = np.asarray(time)
        s = np.asarray(signal)
        sort_idx = np.argsort(t)
        t = t[sort_idx]
        s = s[sort_idx]
        from statsmodels.nonparametric.smoothers_lowess import lowess
        self.smoothed = lowess(s, t, frac=self.frac, it=self.it, return_sorted=True)
        from scipy.interpolate import interp1d
        self.interp_func = interp1d(
            self.smoothed[:, 0], self.smoothed[:, 1],
            bounds_error=False,
            fill_value=(self.smoothed[0, 1], self.smoothed[-1, 1])
        )
        return self

    def predict(self, new_time: Union[float, np.ndarray]):
        if self.interp_func is None:
            raise RuntimeError("LowessInterpolator must be fit before calling predict().")
        return self.interp_func(new_time)

class LoessInterpolator(BaseInterpolator):
    def __init__(self, degree: int = 2, frac: float = 0.1, it: int = 3):
        self.degree = degree
        self.frac = frac
        self.it = it
        self.smoothed = None
        self.interp_func = None

    def fit(self, time: Union[List[float], np.ndarray], signal: Union[List[float], np.ndarray]):
        t = np.asarray(time)
        s = np.asarray(signal)
        sort_idx = np.argsort(t)
        t = t[sort_idx]
        s = s[sort_idx]
        from statsmodels.nonparametric.smoothers_lowess import lowess
        self.smoothed = lowess(s, t, frac=self.frac, it=self.it, return_sorted=True)
        from scipy.interpolate import interp1d
        kind = 'cubic' if len(t) > 3 else 'linear'
        self.interp_func = interp1d(
            self.smoothed[:, 0], self.smoothed[:, 1],
            kind=kind,
            bounds_error=False,
            fill_value=(self.smoothed[0, 1], self.smoothed[-1, 1])
        )
        return self

    def predict(self, new_time: Union[float, np.ndarray]):
        if self.interp_func is None:
            raise RuntimeError("LoessInterpolator must be fit before calling predict().")
        return self.interp_func(new_time)

class FdaInterpolator(BaseInterpolator):
    def __init__(self, smoothing: Optional[float] = None, k: int = 3):
        self.smoothing = smoothing
        self.k = k
        self.spline = None

    def fit(self, time: Union[List[float], np.ndarray], signal: Union[List[float], np.ndarray]):
        t = np.asarray(time)
        s = np.asarray(signal)
        idx = np.argsort(t)
        t, s = t[idx], s[idx]
        smoothing = self.smoothing
        if smoothing is None:
            n = len(s)
            range_y = np.ptp(s)
            noise_estimate = np.std(np.diff(s)) / np.sqrt(2)
            smoothing = n * (0.005 * range_y + 0.1 * noise_estimate) ** 2
        from scipy.interpolate import UnivariateSpline
        self.spline = UnivariateSpline(t, s, s=smoothing, k=self.k)
        return self

    def predict(self, new_time: Union[float, np.ndarray]):
        if self.spline is None:
            raise RuntimeError("FdaInterpolator must be fit before calling predict().")
        return self.spline(new_time)

class LlaInterpolator(BaseInterpolator):
    def __init__(self, window_size: int = 5, normalization: str = 'min', zero_mean: bool = False):
        self.window_size = window_size
        self.normalization = normalization
        self.zero_mean = zero_mean
        self.t = None
        self.s = None
        self.d = None

    def fit(self, time, signal):
        t = np.asarray(time)
        s = np.asarray(signal)
        sort_idx = np.argsort(t)
        t = t[sort_idx]
        s = s[sort_idx]
        # Derivative estimation (central difference)
        d = np.gradient(s, t)
        self.t, self.s, self.d = t, s, d
        return self

    def predict(self, query_time):
        t, s, d = self.t, self.s, self.d
        query_time = np.atleast_1d(query_time)
        result = np.empty_like(query_time, dtype=float)
        for i, t_i in enumerate(query_time):
            idx_next = np.searchsorted(t, t_i, side='right')
            if idx_next == 0:
                idx_prev = idx_next
            elif idx_next == len(t):
                idx_prev = idx_next - 1
            else:
                idx_prev = idx_next - 1
            s_prev, s_next = s[idx_prev], s[min(idx_next, len(s)-1)]
            d_prev, d_next = d[idx_prev], d[min(idx_next, len(d)-1)]
            t_prev, t_next = t[idx_prev], t[min(idx_next, len(t)-1)]
            h = t_next - t_prev
            u = (t_i - t_prev) / h if h != 0 else 0
            h00 = 2*u**3 - 3*u**2 + 1
            h10 = u**3 - 2*u**2 + u
            h01 = -2*u**3 + 3*u**2
            h11 = u**3 - u**2
            result[i] = h00*s_prev + h10*h*d_prev + h01*s_next + h11*h*d_next
        return result if query_time.shape else result.item()

class GllaInterpolator(BaseInterpolator):
    def __init__(self, embedding: int = 3, n: int = 2):
        self.embedding = embedding
        self.n = n
        self.t = None
        self.s = None
        self.d = None

    def fit(self, time, signal):
        t = np.asarray(time)
        s = np.asarray(signal)
        sort_idx = np.argsort(t)
        t = t[sort_idx]
        s = s[sort_idx]
        # Derivative estimation (central difference)
        d = np.gradient(s, t)
        self.t, self.s, self.d = t, s, d
        return self

    def predict(self, query_time):
        t, s, d = self.t, self.s, self.d
        query_time = np.atleast_1d(query_time)
        result = np.empty_like(query_time, dtype=float)
        for i, t_i in enumerate(query_time):
            idx_next = np.searchsorted(t, t_i, side='right')
            if idx_next == 0:
                idx_prev = idx_next
            elif idx_next == len(t):
                idx_prev = idx_next - 1
            else:
                idx_prev = idx_next - 1
            s_prev, s_next = s[idx_prev], s[min(idx_next, len(s)-1)]
            d_prev, d_next = d[idx_prev], d[min(idx_next, len(d)-1)]
            t_prev, t_next = t[idx_prev], t[min(idx_next, len(t)-1)]
            h = t_next - t_prev
            u = (t_i - t_prev) / h if h != 0 else 0
            h00 = 2*u**3 - 3*u**2 + 1
            h10 = u**3 - 2*u**2 + u
            h01 = -2*u**3 + 3*u**2
            h11 = u**3 - u**2
            result[i] = h00*s_prev + h10*h*d_prev + h01*s_next + h11*h*d_next
        return result if query_time.shape else result.item()

class GoldInterpolator(BaseInterpolator):
    def __init__(self, window_size: int = 5, normalization: str = 'min', zero_mean: bool = False):
        self.window_size = window_size
        self.normalization = normalization
        self.zero_mean = zero_mean
        self.t = None
        self.s = None
        self.d = None

    def fit(self, time, signal):
        t = np.asarray(time)
        s = np.asarray(signal)
        sort_idx = np.argsort(t)
        t = t[sort_idx]
        s = s[sort_idx]
        # Derivative estimation (central difference)
        d = np.gradient(s, t)
        self.t, self.s, self.d = t, s, d
        return self

    def predict(self, query_time):
        t, s, d = self.t, self.s, self.d
        query_time = np.atleast_1d(query_time)
        result = np.empty_like(query_time, dtype=float)
        for i, t_i in enumerate(query_time):
            idx_next = np.searchsorted(t, t_i, side='right')
            if idx_next == 0:
                idx_prev = idx_next
            elif idx_next == len(t):
                idx_prev = idx_next - 1
            else:
                idx_prev = idx_next - 1
            s_prev, s_next = s[idx_prev], s[min(idx_next, len(s)-1)]
            d_prev, d_next = d[idx_prev], d[min(idx_next, len(d)-1)]
            t_prev, t_next = t[idx_prev], t[min(idx_next, len(t)-1)]
            h = t_next - t_prev
            u = (t_i - t_prev) / h if h != 0 else 0
            h00 = 2*u**3 - 3*u**2 + 1
            h10 = u**3 - 2*u**2 + u
            h01 = -2*u**3 + 3*u**2
            h11 = u**3 - u**2
            result[i] = h00*s_prev + h10*h*d_prev + h01*s_next + h11*h*d_next
        return result if query_time.shape else result.item()

# Neural network interpolators
class NeuralNetworkInterpolator(BaseInterpolator):
    def __init__(self, framework: str = 'pytorch', hidden_layers: list = [64, 32], epochs: int = 1000, dropout: float = 0.1):
        self.framework = framework
        self.hidden_layers = hidden_layers
        self.epochs = epochs
        self.dropout = dropout
        self.model = None
        self.t_min = None
        self.t_max = None
        self.s_min = None
        self.s_max = None

    def fit(self, time, signal):
        import numpy as np
        time = np.asarray(time)
        signal = np.asarray(signal)
        if np.isnan(time).any() or np.isnan(signal).any():
            raise ValueError('Input time and signal must not contain NaN values. Please impute or remove missing data before fitting.')
        self.t_min = np.min(time)
        self.t_max = np.max(time)
        self.s_min = np.min(signal)
        self.s_max = np.max(signal)
        t_norm = (time - self.t_min) / (self.t_max - self.t_min) if self.t_max > self.t_min else time
        s_norm = (signal - self.s_min) / (self.s_max - self.s_min) if self.s_max > self.s_min else signal
        if self.framework == 'pytorch':
            import torch
            import torch.nn as nn
            import torch.optim as optim
            class PyTorchMLP(nn.Module):
                def __init__(self, hidden_layers=[64, 32], dropout=0.1):
                    super().__init__()
                    layers = []
                    input_dim = 1
                    for h in hidden_layers:
                        layers.append(nn.Linear(input_dim, h))
                        layers.append(nn.ReLU())
                        layers.append(nn.Dropout(dropout))
                        input_dim = h
                    layers.append(nn.Linear(input_dim, 1))
                    self.net = nn.Sequential(*layers)
                def forward(self, x):
                    return self.net(x)
            model = PyTorchMLP(self.hidden_layers, self.dropout)
            optimizer = optim.Adam(model.parameters(), lr=0.01)
            loss_fn = nn.MSELoss()
            X = torch.tensor(t_norm.reshape(-1, 1), dtype=torch.float32)
            y = torch.tensor(s_norm.reshape(-1, 1), dtype=torch.float32)
            model.train()
            for _ in range(self.epochs):
                optimizer.zero_grad()
                output = model(X)
                loss = loss_fn(output, y)
                loss.backward()
                optimizer.step()
            self.model = model
        elif self.framework == 'tensorflow':
            import tensorflow as tf
            from tensorflow import keras
            from tensorflow.keras import layers
            model = keras.Sequential()
            model.add(keras.Input(shape=(1,)))
            for h in self.hidden_layers:
                model.add(layers.Dense(h, activation='relu'))
                model.add(layers.Dropout(self.dropout))
            model.add(layers.Dense(1))
            model.compile(optimizer='adam', loss='mse')
            model.fit(t_norm.reshape(-1, 1), s_norm.reshape(-1, 1), epochs=self.epochs, verbose=0)
            self.model = model
        else:
            raise ValueError(f"Unknown framework: {self.framework}")
        return self

    def predict(self, query_time):
        import numpy as np
        query_time = np.asarray(query_time)
        t_norm = (query_time - self.t_min) / (self.t_max - self.t_min) if self.t_max > self.t_min else query_time
        if self.framework == 'pytorch':
            import torch
            self.model.eval()
            with torch.no_grad():
                X = torch.tensor(t_norm.reshape(-1, 1), dtype=torch.float32)
                pred_norm = self.model(X).numpy().flatten()
        elif self.framework == 'tensorflow':
            pred_norm = self.model.predict(t_norm.reshape(-1, 1), verbose=0).flatten()
        else:
            raise ValueError(f"Unknown framework: {self.framework}")
        pred = pred_norm * (self.s_max - self.s_min) + self.s_min if self.s_max > self.s_min else pred_norm
        if query_time.shape:
            return pred
        else:
            # pred could be a numpy array, tensorflow tensor, or python scalar
            if hasattr(pred, "numpy"):
                pred = pred.numpy()
            if hasattr(pred, "item"):
                return pred.item()
            elif isinstance(pred, (list, tuple)) and len(pred) == 1:
                return float(pred[0])
            else:
                return float(pred)




import warnings

# For neural network methods
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch is not installed. Neural network interpolation methods will not be available.")

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    warnings.warn("TensorFlow is not installed. Some neural network interpolation methods will not be available.")

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
        
        return result.item() if scalar_input and np.ndim(result) == 0 else result
    
    return interpolate

def spline_interpolation(
    time: Union[List[float], np.ndarray],
    signal: Union[List[float], np.ndarray],
    smoothing: Optional[float] = None,
    k: int = 5,
    return_model: bool = False
) -> Union[Callable[[Union[float, np.ndarray]], np.ndarray], Tuple[Callable[[Union[float, np.ndarray]], np.ndarray], object]]:
    """
    Interpolate a time series using spline interpolation.
    
    Args:
        time: Time points of the original signal
        signal: Signal values
        smoothing: Smoothing factor for the spline. If None, it's automatically determined
        k: Degree of the spline (1=linear, 2=quadratic, 3=cubic)
        return_model: If True, also return the fitted UnivariateSpline object for further predictions/interpolations.
    
    Returns:
        - If return_model is False (default): Callable function (UnivariateSpline) that interpolates the signal at any time point
        - If return_model is True: (UnivariateSpline, UnivariateSpline) tuple (the spline is both the interpolator and the fitted object)
    
    Example:
        >>> interp, spline_obj = spline_interpolation(time, signal, return_model=True)
        >>> new_vals = interp(new_time)
        >>> # Or use spline_obj directly for further predictions:
        >>> spline_obj(new_time)
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
        # More precise smoothing factor calculation
        noise_estimate = np.std(np.diff(signal)) / np.sqrt(2)
        smoothing = n * (0.005 * range_y + 0.1 * noise_estimate) ** 2
    
    # Create the spline
    spline = UnivariateSpline(t, s, s=smoothing, k=k)
    
    # Return the spline as the interpolation function
    if return_model:
        return spline, spline
    return spline

def lowess_interpolation(
    time: Union[List[float], np.ndarray],
    signal: Union[List[float], np.ndarray],
    frac: float = 0.3,
    it: int = 3,
    return_model: bool = False
) -> Union[Callable[[Union[float, np.ndarray]], np.ndarray], Tuple[Callable[[Union[float, np.ndarray]], np.ndarray], object]]:
    """
    Interpolate a time series using LOWESS (Locally Weighted Scatterplot Smoothing).
    
    Args:
        time: Time points of the original signal
        signal: Signal values
        frac: Between 0 and 1. The fraction of the data used when estimating each y-value
        it: Number of robustifying iterations
        return_model: If True, also return the smoothed data array used for interpolation (for further analysis or custom interpolators).
    
    Returns:
        - If return_model is False (default): Callable function (interp1d) that interpolates the signal at any time point
        - If return_model is True: (interp1d, smoothed_data) tuple
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
    it: int = 3,
    return_model: bool = False
) -> Union[Callable[[Union[float, np.ndarray]], np.ndarray], Tuple[Callable[[Union[float, np.ndarray]], np.ndarray], object]]:
    """
    Interpolate a time series using LOESS (LOcally Estimated Scatterplot Smoothing).
    This is similar to LOWESS but uses higher-degree local polynomials.
    
    Args:
        time: Time points of the original signal
        signal: Signal values
        degree: Degree of local polynomials (1=linear, 2=quadratic)
        frac: Between 0 and 1. The fraction of the data used when estimating each y-value
        it: Number of robustifying iterations
        return_model: If True, also return the smoothed data array used for interpolation (for further analysis or custom interpolators).
    
    Returns:
        - If return_model is False (default): Callable function (interp1d) that interpolates the signal at any time point
        - If return_model is True: (interp1d, smoothed_data) tuple
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
    methods: List[str] = ['linear', 'spline', 'lowess', 'loess', 'lla', 'glla', 'gold', 'fda'],
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
    
    results = {}
    for method in methods:
        try:
            if method == 'linear':
                func = interp1d(time, signal, kind='linear', bounds_error=False, fill_value=(signal[0], signal[-1]))
            elif method == 'spline':
                func = spline_interpolation(time, signal)
            elif method == 'lowess':
                func = lowess_interpolation(time, signal)
            elif method == 'loess':
                func = loess_interpolation(time, signal)
            elif method == 'lla':
                func = lla_interpolation(time, signal)
            elif method == 'glla':
                func = glla_interpolation(time, signal)
            elif method == 'gold':
                func = gold_interpolation(time, signal)
            elif method == 'fda':
                func = fda_interpolation(time, signal)
            else:
                raise ValueError(f"Unknown interpolation method: {method}")

            metrics = calculate_fit_quality(time, signal, func)
            results[method] = {
                'function': func,
                'r_squared': metrics['r_squared'],
                'rmse': metrics['rmse']
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


def lla_interpolation(
    time: Union[List[float], np.ndarray],
    signal: Union[List[float], np.ndarray],
    window_size: int = 5,
    normalization: str = 'min',
    zero_mean: bool = False,
    r2_threshold: Optional[float] = None,
    resample_method: Optional[str] = None
) -> Callable[[Union[float, np.ndarray]], np.ndarray]:
    """
    Interpolate a time series using Local Linear Approximation (LLA) with Hermite interpolation.

    Args:
        time: Time points of the original signal
        signal: Signal values
        window_size: Size of the window for local linear regression (default: 5)
        normalization: Normalization method for window ('min', 'none', etc.)
        zero_mean: Whether to zero-mean center the window before regression
        r2_threshold: Optional R^2 threshold for filtering low-quality windows
        resample_method: Method for resampling derivatives ('spline', 'lowess', 'loess', etc.)

    Returns:
        Callable function that interpolates the signal at any time point
    """
    from pydelt.derivatives import lla
    # Get derivatives at original time points
    derivs, _ = lla(time, signal, window_size=window_size, normalization=normalization, zero_mean=zero_mean, r2_threshold=r2_threshold, resample_method=resample_method)
    t = np.asarray(time)
    s = np.asarray(signal)
    d = np.asarray(derivs)
    # Sort by time
    idx = np.argsort(t)
    t, s, d = t[idx], s[idx], d[idx]
    def interpolate(query_time):
        query_time = np.asarray(query_time)
        scalar_input = query_time.ndim == 0
        if scalar_input:
            query_time = np.array([query_time])
        result = np.zeros_like(query_time, dtype=float)
        for i, t_i in enumerate(query_time):
            # Find closest interval
            if t_i <= t[0]:
                s_prev, s_next = s[0], s[1]
                d_prev, d_next = d[0], d[1]
                t_prev, t_next = t[0], t[1]
            elif t_i >= t[-1]:
                s_prev, s_next = s[-2], s[-1]
                d_prev, d_next = d[-2], d[-1]
                t_prev, t_next = t[-2], t[-1]
            else:
                idx_next = np.searchsorted(t, t_i)
                idx_prev = idx_next - 1
                s_prev, s_next = s[idx_prev], s[idx_next]
                d_prev, d_next = d[idx_prev], d[idx_next]
                t_prev, t_next = t[idx_prev], t[idx_next]
            h = t_next - t_prev
            u = (t_i - t_prev) / h if h != 0 else 0
            h00 = 2*u**3 - 3*u**2 + 1
            h10 = u**3 - 2*u**2 + u
            h01 = -2*u**3 + 3*u**2
            h11 = u**3 - u**2
            result[i] = h00*s_prev + h10*h*d_prev + h01*s_next + h11*h*d_next
        return result.item() if scalar_input and np.ndim(result) == 0 else result
    return interpolate

def glla_interpolation(
    time: Union[List[float], np.ndarray],
    signal: Union[List[float], np.ndarray],
    embedding: int = 3,
    n: int = 2,
    r2_threshold: Optional[float] = None,
    resample_method: Optional[str] = None
) -> Callable[[Union[float, np.ndarray]], np.ndarray]:
    """
    Interpolate a time series using Generalized Local Linear Approximation (GLLA) with Hermite interpolation.

    Args:
        time: Time points of the original signal
        signal: Signal values
        embedding: Number of points to consider for derivative calculation (default: 3)
        n: Maximum order of derivative to calculate (default: 2)
        r2_threshold: Optional R^2 threshold for filtering low-quality windows
        resample_method: Method for resampling derivatives ('spline', 'lowess', 'loess', etc.)

    Returns:
        Callable function that interpolates the signal at any time point
    """
    from pydelt.derivatives import glla
    result = glla(signal, time, embedding=embedding, n=n, r2_threshold=r2_threshold, resample_method=resample_method)
    derivs = result['derivatives'][:,0] if result['derivatives'].ndim > 1 else result['derivatives']
    t = np.asarray(time)
    s = np.asarray(signal)
    d = np.asarray(derivs)
    idx = np.argsort(t)
    t, s, d = t[idx], s[idx], d[idx]
    def interpolate(query_time):
        query_time = np.asarray(query_time)
        scalar_input = query_time.ndim == 0
        if scalar_input:
            query_time = np.array([query_time])
        result = np.zeros_like(query_time, dtype=float)
        for i, t_i in enumerate(query_time):
            if t_i <= t[0]:
                s_prev, s_next = s[0], s[1]
                d_prev, d_next = d[0], d[1]
                t_prev, t_next = t[0], t[1]
            elif t_i >= t[-1]:
                s_prev, s_next = s[-2], s[-1]
                d_prev, d_next = d[-2], d[-1]
                t_prev, t_next = t[-2], t[-1]
            else:
                idx_next = np.searchsorted(t, t_i)
                idx_prev = idx_next - 1
                s_prev, s_next = s[idx_prev], s[idx_next]
                d_prev, d_next = d[idx_prev], d[idx_next]
                t_prev, t_next = t[idx_prev], t[idx_next]
            h = t_next - t_prev
            u = (t_i - t_prev) / h if h != 0 else 0
            h00 = 2*u**3 - 3*u**2 + 1
            h10 = u**3 - 2*u**2 + u
            h01 = -2*u**3 + 3*u**2
            h11 = u**3 - u**2
            result[i] = h00*s_prev + h10*h*d_prev + h01*s_next + h11*h*d_next
        return result.item() if scalar_input and np.ndim(result) == 0 else result
    return interpolate

def gold_interpolation(
    time: Union[List[float], np.ndarray],
    signal: Union[List[float], np.ndarray],
    window_size: int = 5,
    normalization: str = 'min',
    zero_mean: bool = False,
    r2_threshold: Optional[float] = None,
    resample_method: Optional[str] = None
) -> Callable[[Union[float, np.ndarray]], np.ndarray]:
    """
    Interpolate a time series using Generalized Orthogonal Local Derivative (GOLD) with Hermite interpolation.

    Args:
        time: Time points of the original signal
        signal: Signal values
        window_size: Size of the window for local regression (default: 5)
        normalization: Normalization method for window ('min', 'none', etc.)
        zero_mean: Whether to zero-mean center the window before regression
        r2_threshold: Optional R^2 threshold for filtering low-quality windows
        resample_method: Method for resampling derivatives ('spline', 'lowess', 'loess', etc.)

    Returns:
        Callable function that interpolates the signal at any time point
    """
    from pydelt.derivatives import gold
    derivs, _ = gold(time, signal, window_size=window_size, normalization=normalization, zero_mean=zero_mean, r2_threshold=r2_threshold, resample_method=resample_method)
    t = np.asarray(time)
    s = np.asarray(signal)
    d = np.asarray(derivs)
    idx = np.argsort(t)
    t, s, d = t[idx], s[idx], d[idx]
    def interpolate(query_time):
        query_time = np.asarray(query_time)
        scalar_input = query_time.ndim == 0
        if scalar_input:
            query_time = np.array([query_time])
        result = np.zeros_like(query_time, dtype=float)
        for i, t_i in enumerate(query_time):
            if t_i <= t[0]:
                s_prev, s_next = s[0], s[1]
                d_prev, d_next = d[0], d[1]
                t_prev, t_next = t[0], t[1]
            elif t_i >= t[-1]:
                s_prev, s_next = s[-2], s[-1]
                d_prev, d_next = d[-2], d[-1]
                t_prev, t_next = t[-2], t[-1]
            else:
                idx_next = np.searchsorted(t, t_i)
                idx_prev = idx_next - 1
                s_prev, s_next = s[idx_prev], s[idx_next]
                d_prev, d_next = d[idx_prev], d[idx_next]
                t_prev, t_next = t[idx_prev], t[idx_next]
            h = t_next - t_prev
            u = (t_i - t_prev) / h if h != 0 else 0
            h00 = 2*u**3 - 3*u**2 + 1
            h10 = u**3 - 2*u**2 + u
            h01 = -2*u**3 + 3*u**2
            h11 = u**3 - u**2
            result[i] = h00*s_prev + h10*h*d_prev + h01*s_next + h11*h*d_next
        return result.item() if scalar_input and np.ndim(result) == 0 else result
    return interpolate

def fda_interpolation(
    time: Union[List[float], np.ndarray],
    signal: Union[List[float], np.ndarray],
    smoothing: Optional[float] = None,
    k: int = 3,
    return_model: bool = False
) -> Union[Callable[[Union[float, np.ndarray]], np.ndarray], Tuple[Callable[[Union[float, np.ndarray]], np.ndarray], object]]:
    """
    Interpolate a time series using Functional Data Analysis (FDA) spline smoothing.

    Args:
        time: Time points of the original signal
        signal: Signal values
        smoothing: Smoothing factor for the spline. If None, it is automatically determined.
        k: Degree of the spline (default: 3, cubic)
        return_model: If True, also return the fitted UnivariateSpline object for further predictions/interpolations.

    Returns:
        - If return_model is False (default): Callable function (UnivariateSpline) that interpolates the signal at any time point
        - If return_model is True: (UnivariateSpline, UnivariateSpline) tuple (the spline is both the interpolator and the fitted object)
    """
    from scipy.interpolate import UnivariateSpline
    t = np.asarray(time)
    s = np.asarray(signal)
    idx = np.argsort(t)
    t, s = t[idx], s[idx]
    if smoothing is None:
        n = len(s)
        range_y = np.ptp(s)
        noise_estimate = np.std(np.diff(s)) / np.sqrt(2)
        smoothing = n * (0.005 * range_y + 0.1 * noise_estimate) ** 2
    spline = UnivariateSpline(t, s, s=smoothing, k=k)
    return spline

def derivative_based_interpolation(
    time: Union[List[float], np.ndarray],
    signal: Union[List[float], np.ndarray],
    method: str = 'lla',
    **kwargs
) -> Callable[[Union[float, np.ndarray]], np.ndarray]:
    """
    Interpolate a time series using derivative estimation and signal reconstruction.
    
    Args:
        time: Time points of the original signal
        signal: Signal values
        method: Derivative estimation method ('lla', 'glla', 'gold', 'fda')
        **kwargs: Additional parameters for the derivative method
        
    Returns:
        Callable function that interpolates the signal at any time point
    """
    # Import here to avoid circular imports
    from pydelt.derivatives import lla, glla, gold, fda
    from pydelt.integrals import integrate_derivative
    
    # Convert inputs to numpy arrays
    t = np.asarray(time)
    s = np.asarray(signal)
    
    # Ensure data is sorted by time
    sort_idx = np.argsort(t)
    t = t[sort_idx]
    s = s[sort_idx]
    
    # Calculate derivatives using the specified method
    if method.lower() == 'lla':
        derivative, _ = lla(t.tolist(), s.tolist(), **kwargs)
    elif method.lower() == 'glla':
        result = glla(s, t, **kwargs)
        derivative = result['dsignal'][:, 1]  # First derivative
    elif method.lower() == 'gold':
        result = gold(s, t, **kwargs)
        derivative = result['dsignal'][:, 1]  # First derivative
    elif method.lower() == 'fda':
        result = fda(s, t, **kwargs)
        derivative = result['dsignal'][:, 1]  # First derivative
    else:
        raise ValueError(f"Unknown derivative method: {method}")
    
    # Ensure derivative array has the same length as time and signal arrays
    derivative = np.array(derivative)
    if len(derivative) != len(t):
        # If derivatives are missing for some points, use linear interpolation to fill them
        from scipy.interpolate import interp1d
        x_known = np.arange(len(derivative))
        x_all = np.arange(len(t))
        if len(derivative) > 0:
            deriv_interp = interp1d(x_known, derivative, bounds_error=False, fill_value=(derivative[0], derivative[-1]))
            derivative = deriv_interp(x_all)
        else:
            # If no derivatives were calculated, use zeros
            derivative = np.zeros_like(t)
    
    # Create interpolation function
    def interpolate(query_time):
        query_time = np.asarray(query_time)
        scalar_input = query_time.ndim == 0
        if scalar_input:
            query_time = np.array([query_time])
        
        result = np.zeros_like(query_time, dtype=float)
        
        for i, t_i in enumerate(query_time):
            # Find the appropriate segment
            if t_i <= t[0]:
                # Extrapolate before the first point
                result[i] = s[0] + derivative[0] * (t_i - t[0])
            elif t_i >= t[-1]:
                # Extrapolate after the last point
                result[i] = s[-1] + derivative[-1] * (t_i - t[-1])
            else:
                # Find the nearest time points
                idx = np.searchsorted(t, t_i)
                t_prev, t_next = t[idx-1], t[idx]
                s_prev, s_next = s[idx-1], s[idx]
                d_prev, d_next = derivative[idx-1], derivative[idx]
                
                # Use Hermite interpolation
                h = t_next - t_prev
                u = (t_i - t_prev) / h
                
                # Hermite basis functions
                h00 = 2*u**3 - 3*u**2 + 1
                h10 = u**3 - 2*u**2 + u
                h01 = -2*u**3 + 3*u**2
                h11 = u**3 - u**2
                
                # Interpolate
                result[i] = h00*s_prev + h10*h*d_prev + h01*s_next + h11*h*d_next
        
        return result.item() if scalar_input and np.ndim(result) == 0 else result
    
    return interpolate


class PyTorchMLP(nn.Module):
    """PyTorch Multi-Layer Perceptron for time series interpolation."""
    
    def __init__(self, hidden_layers=[128, 96, 64, 48, 32], dropout=0.1):
        super(PyTorchMLP, self).__init__()
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(1, hidden_layers[0]))
        self.dropout = nn.Dropout(dropout)
        
        # Hidden layers
        for i in range(len(hidden_layers) - 1):
            self.layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
        # Output layer
        self.layers.append(nn.Linear(hidden_layers[-1], 1))
        self.activation = nn.ReLU()
    
    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = self.activation(layer(x))
            x = self.dropout(x)
        x = self.layers[-1](x)  # No activation on output layer
        return x


class TensorFlowModel:
    """TensorFlow model wrapper for time series interpolation."""
    
    def __init__(self, hidden_layers=[128, 96, 64, 48, 32], dropout=0.1):
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is not installed.")
            
        self.model = keras.Sequential()
        
        # Input layer
        self.model.add(keras.Input(shape=(1,)))
        self.model.add(layers.Dense(hidden_layers[0], activation='relu'))
        self.model.add(layers.Dropout(dropout))
        
        # Hidden layers
        for units in hidden_layers[1:]:
            self.model.add(layers.Dense(units, activation='relu'))
            self.model.add(layers.Dropout(dropout))
        
        # Output layer
        self.model.add(layers.Dense(1))
        # Compile the model
        self.model.compile(optimizer='adam', loss='mse')
    
    def fit(self, x, y, epochs=100, batch_size=32, verbose=0):
        return self.model.fit(x, y, epochs=epochs, batch_size=batch_size, verbose=verbose)
    
    def predict(self, x):
        return self.model.predict(x, verbose=0)


def neural_network_interpolation(
    time: Union[list, np.ndarray],
    signal: Union[list, np.ndarray],
    framework: str = 'pytorch',
    hidden_layers: list = [64, 32],
    epochs: int = 1000,
    holdout_fraction: float = 0.0,
    return_model: bool = False,
    **kwargs
) -> Union[callable, tuple]:
    # Convert to numpy arrays
    time = np.asarray(time)
    signal = np.asarray(signal)
    if np.isnan(time).any() or np.isnan(signal).any():
        raise ValueError('Input time and signal must not contain NaN values. Please impute or remove missing data before fitting.')

    """
    Interpolate a time series using a neural network.
    
    Args:
        time: Time points of the original signal
        signal: Signal values
        framework: Neural network framework ('pytorch' or 'tensorflow')
        hidden_layers: List of hidden layer sizes
        epochs: Number of training epochs
        holdout_fraction: Fraction of data to hold out for evaluation (0.0 to 0.9)
        return_model: If True, return the trained model along with the interpolation function
        **kwargs: Additional parameters for the neural network
        
    Returns:
        If return_model is False:
            Callable function that interpolates the signal at any time point
        If return_model is True:
            Tuple containing:
            - Callable function that interpolates the signal at any time point
            - Trained neural network model
    """
    if framework == 'pytorch' and not TORCH_AVAILABLE:
        raise ImportError("PyTorch is not installed.")
    if framework == 'tensorflow' and not TF_AVAILABLE:
        raise ImportError("TensorFlow is not installed.")
    
    # Convert inputs to numpy arrays
    t = np.asarray(time)
    s = np.asarray(signal)
    
    # Ensure data is sorted by time
    sort_idx = np.argsort(t)
    t = t[sort_idx]
    s = s[sort_idx]
    
    # Normalize time to [0, 1] for better training
    t_min, t_max = t.min(), t.max()
    t_norm = (t - t_min) / (t_max - t_min) if t_max > t_min else t
    
    # Normalize signal to [0, 1] for better training
    s_min, s_max = s.min(), s.max()
    s_norm = (s - s_min) / (s_max - s_min) if s_max > s_min else s
    
    # Split data into training and holdout sets if requested
    if 0.0 < holdout_fraction < 0.9:
        n_holdout = int(len(t) * holdout_fraction)
        if n_holdout > 0:
            # Randomly select indices for holdout
            holdout_indices = np.random.choice(len(t), n_holdout, replace=False)
            train_indices = np.array([i for i in range(len(t)) if i not in holdout_indices])
            
            t_train, s_train = t_norm[train_indices], s_norm[train_indices]
        else:
            t_train, s_train = t_norm, s_norm
    else:
        t_train, s_train = t_norm, s_norm
    
    # Train the neural network
    if framework == 'pytorch':
        # Prepare data for PyTorch
        X = torch.tensor(t_train.reshape(-1, 1), dtype=torch.float32)
        y = torch.tensor(s_train.reshape(-1, 1), dtype=torch.float32)
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=min(32, len(t_train)), shuffle=True)
        
        # Create and train the model
        model = PyTorchMLP(hidden_layers=hidden_layers)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        loss_fn = nn.MSELoss()
        
        model.train()
        for epoch in range(epochs):
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = loss_fn(outputs, batch_y)
                loss.backward()
                optimizer.step()
        
        model.eval()
        
        # Create interpolation function
        def interpolate(query_time):
            query_time = np.asarray(query_time)
            scalar_input = query_time.ndim == 0
            if scalar_input:
                query_time = np.array([query_time])
            
            # Normalize query time
            query_norm = (query_time - t_min) / (t_max - t_min) if t_max > t_min else query_time
            
            # Convert to PyTorch tensor
            query_tensor = torch.tensor(query_norm.reshape(-1, 1), dtype=torch.float32)
            
            # Get predictions
            with torch.no_grad():
                pred_norm = model(query_tensor).numpy().flatten()
            
            # Denormalize predictions
            pred = pred_norm * (s_max - s_min) + s_min if s_max > s_min else pred_norm
            
            return pred.item() if scalar_input and np.ndim(pred) == 0 else pred
    
    elif framework == 'tensorflow':
        # Create and train the model
        model = TensorFlowModel(hidden_layers=hidden_layers)
        model.fit(t_train.reshape(-1, 1), s_train.reshape(-1, 1), epochs=epochs)
        
        # Create interpolation function
        def interpolate(query_time):
            query_time = np.asarray(query_time)
            scalar_input = query_time.ndim == 0
            if scalar_input:
                query_time = np.array([query_time])
            
            # Normalize query time
            query_norm = (query_time - t_min) / (t_max - t_min) if t_max > t_min else query_time
            
            # Get predictions
            pred_norm = model.predict(query_norm.reshape(-1, 1)).flatten()
            
            # Denormalize predictions
            pred = pred_norm * (s_max - s_min) + s_min if s_max > s_min else pred_norm
            
            return pred.item() if scalar_input and pred.size == 1 else pred
    
    else:
        raise ValueError(f"Unknown framework: {framework}")
    
    if return_model:
        return interpolate, model
    else:
        return interpolate
