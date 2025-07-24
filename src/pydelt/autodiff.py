"""
Functions for calculating derivatives using automatic differentiation with trained models.
"""

import numpy as np
from typing import List, Tuple, Dict, Union, Optional, Callable, Any
import warnings

# For PyTorch methods
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch is not installed. PyTorch-based automatic differentiation will not be available.")

# For TensorFlow methods
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    warnings.warn("TensorFlow is not installed. TensorFlow-based automatic differentiation will not be available.")


def neural_network_derivative(
    time: Union[List[float], np.ndarray],
    signal: Union[List[float], np.ndarray],
    framework: str = 'pytorch',
    hidden_layers: List[int] = [64, 32, 16],
    epochs: int = 1000,
    holdout_fraction: float = 0.0,
    return_model: bool = False,
    order: int = 1,
    dropout: float = 0.1,
    **kwargs
) -> Union[Callable[[Union[float, np.ndarray]], np.ndarray], Tuple[Callable[[Union[float, np.ndarray]], np.ndarray], Any]]:
    import numpy as np
    if np.isnan(time).any() or np.isnan(signal).any():
        raise ValueError("Input time and signal must not contain NaN values")
    """
    Calculate derivatives using automatic differentiation with a neural network.
    
    Args:
        time: Time points of the original signal
        signal: Signal values
        framework: Neural network framework ('pytorch' or 'tensorflow')
        hidden_layers: List of hidden layer sizes
        epochs: Number of training epochs
        holdout_fraction: Fraction of data to hold out for evaluation (0.0 to 0.9)
        return_model: If True, return the trained model along with the derivative function
        order: Order of the derivative to calculate (1 for first derivative, 2 for second, etc.)
        dropout: Dropout rate for the neural network
        **kwargs: Additional parameters for the neural network
        
    Returns:
        If return_model is False:
            Callable function that calculates the derivative at any time point
        If return_model is True:
            Tuple containing:
            - Callable function that calculates the derivative at any time point
            - Trained neural network model
    """
    if framework == 'pytorch':
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not installed.")
        return _pytorch_derivative(time, signal, hidden_layers, epochs, holdout_fraction, return_model, order, dropout=dropout)
    elif framework == 'tensorflow':
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is not installed.")
        return _tensorflow_derivative(time, signal, hidden_layers, epochs, holdout_fraction, return_model, order, dropout=dropout)
    else:
        raise ValueError(f"Unknown framework: {framework}")


def _pytorch_derivative(
    time: Union[List[float], np.ndarray],
    signal: Union[List[float], np.ndarray],
    hidden_layers: List[int] = [64, 32, 16],
    epochs: int = 1000,
    holdout_fraction: float = 0.0,
    return_model: bool = False,
    order: int = 1,
    dropout: float = 0.1,
) -> Union[Callable[[Union[float, np.ndarray]], np.ndarray], Tuple[Callable[[Union[float, np.ndarray]], np.ndarray], Any]]:
    """
    Calculate derivatives using automatic differentiation with PyTorch.
    """
    from pydelt.interpolation import PyTorchMLP
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    
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
    
    # Prepare data for PyTorch
    X = torch.tensor(t_train.reshape(-1, 1), dtype=torch.float32)
    y = torch.tensor(s_train.reshape(-1, 1), dtype=torch.float32)
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=min(32, len(t_train)), shuffle=True)
    
    # Create and train the model
    model = PyTorchMLP(hidden_layers=hidden_layers, dropout=dropout)
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
    
    # Create derivative function using automatic differentiation
    def derivative_func(query_time):
        query_time = np.asarray(query_time)
        scalar_input = query_time.ndim == 0
        if scalar_input:
            query_time = np.array([query_time])
        
        # Normalize query time
        query_norm = (query_time - t_min) / (t_max - t_min) if t_max > t_min else query_time
        
        # Calculate derivative for each query point
        results = np.zeros_like(query_norm)
        
        for i, t_i in enumerate(query_norm):
            # Create tensor with requires_grad=True for autodiff
            t_tensor = torch.tensor([[t_i]], dtype=torch.float32, requires_grad=True)
            
            # Forward pass
            y_pred = model(t_tensor)
            
            # Initialize gradient calculation
            grad = torch.ones_like(y_pred)
            
            # Calculate derivative of the specified order
            for _ in range(order):
                # Backward pass to get gradient
                y_pred.backward(grad, retain_graph=True)
                
                # Get the gradient
                grad_value = t_tensor.grad.item()
                
                # Reset gradients
                t_tensor.grad.zero_()
                
                if _ < order - 1:
                    # For higher-order derivatives, create a new tensor with the gradient
                    y_pred = torch.tensor([[grad_value]], dtype=torch.float32, requires_grad=True)
                    grad = torch.ones_like(y_pred)
                else:
                    # For the final derivative, store the result
                    results[i] = grad_value
        
        # Scale the derivative based on the normalization
        scale_factor = (s_max - s_min) / (t_max - t_min) if t_max > t_min and s_max > s_min else 1.0
        for _ in range(order):
            results *= scale_factor
        
        return results[0] if scalar_input else results
    
    if return_model:
        return derivative_func, model
    else:
        return derivative_func


def _tensorflow_derivative(
    time: Union[List[float], np.ndarray],
    signal: Union[List[float], np.ndarray],
    hidden_layers: List[int] = [64, 32, 16],
    epochs: int = 1000,
    holdout_fraction: float = 0.0,
    return_model: bool = False,
    order: int = 1,
    dropout: float = 0.1,
) -> Union[Callable[[Union[float, np.ndarray]], np.ndarray], Tuple[Callable[[Union[float, np.ndarray]], np.ndarray], Any]]:
    """
    Calculate derivatives using automatic differentiation with TensorFlow.
    """
    from pydelt.interpolation import TensorFlowModel
    
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
    
    # Create and train the model
    model = TensorFlowModel(hidden_layers=hidden_layers, dropout=dropout)
    model.fit(t_train.reshape(-1, 1), s_train.reshape(-1, 1), epochs=epochs)
    
    # Create a TensorFlow function for computing derivatives
    @tf.function
    def get_derivative(x, order=1):
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        
        with tf.GradientTape(persistent=True) as tape:
            # Watch the input tensor
            tape.watch(x)
            
            # Forward pass
            y = model.model(x)
            
            # For higher-order derivatives
            for i in range(1, order):
                # Get the gradient
                grad = tape.gradient(y, x)
                
                # For higher orders, we need to watch the gradient
                tape.watch(grad)
                
                # Update y to be the gradient for the next iteration
                y = grad
        
        # Get the final derivative
        derivative = tape.gradient(y, x)
        return derivative
    
    # Create derivative function
    def derivative_func(query_time):
        query_time = np.asarray(query_time)
        scalar_input = query_time.ndim == 0
        if scalar_input:
            query_time = np.array([query_time])
        
        # Normalize query time
        query_norm = (query_time - t_min) / (t_max - t_min) if t_max > t_min else query_time
        
        # Reshape for TensorFlow
        query_tensor = query_norm.reshape(-1, 1).astype(np.float32)
        
        # Calculate derivative
        with tf.GradientTape(persistent=True) as tape:
            x = tf.convert_to_tensor(query_tensor)
            tape.watch(x)
            y = model.model(x)
            
            # For higher-order derivatives
            for i in range(1, order):
                grad = tape.gradient(y, x)
                tape.watch(grad)
                y = grad
        
        derivative = tape.gradient(y, x).numpy().flatten()
        
        # Scale the derivative based on the normalization
        scale_factor = (s_max - s_min) / (t_max - t_min) if t_max > t_min and s_max > s_min else 1.0
        for _ in range(order):
            derivative *= scale_factor
        
        return derivative[0] if scalar_input else derivative
    
    if return_model:
        return derivative_func, model
    else:
        return derivative_func
