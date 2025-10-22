"""
Functions for calculating derivatives using automatic differentiation with trained models.

Note: TensorFlow retracing warnings and general UserWarnings are suppressed in this module.
"""

import numpy as np
from typing import List, Tuple, Dict, Union, Optional, Callable, Any
import warnings

# Suppress TensorFlow retracing and UserWarnings
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")
try:
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
except ImportError:
    pass

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
    framework: str = 'tensorflow',
    hidden_layers: List[int] = [128, 96, 64, 48, 32],
    epochs: int = 1000,
    holdout_fraction: float = 0.0,
    return_model: bool = False,
    order: int = 1,
    dropout: float = 0.1,
    learning_rate: float = 0.001,
    batch_size: int = 32,
    early_stopping: bool = True,
    patience: int = 50,
) -> Union[Callable[[Union[float, np.ndarray]], np.ndarray], Tuple[Callable[[Union[float, np.ndarray]], np.ndarray], Any]]:
    """
    Calculate derivatives using automatic differentiation with a neural network.
    
    Args:
        time: Input array, shape (N,) for univariate or (N, n_in) for multivariate
        signal: Output array, shape (N,) for univariate or (N, n_out) for multivariate
        framework: Neural network framework ('pytorch' or 'tensorflow')
        hidden_layers: List of hidden layer sizes
        epochs: Number of training epochs
        holdout_fraction: Fraction of data to use for validation
        return_model: Whether to return the model along with the derivative function
        order: Order of the derivative to compute
        dropout: Dropout rate for regularization
        learning_rate: Learning rate for optimizer
        batch_size: Batch size for training
        early_stopping: Whether to use early stopping
        patience: Number of epochs with no improvement after which training will be stopped
    
    Returns:
        If return_model is False:
            Callable function that calculates the gradient (for scalar output) or Jacobian (for vector output) at any input point
        If return_model is True:
            Tuple containing:
                - Callable function that calculates derivatives
                - Trained neural network model
    """
    time = np.asarray(time)
    signal = np.asarray(signal)
    if np.isnan(time).any() or np.isnan(signal).any():
        raise ValueError("Input time and signal must not contain NaN values")
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


def _pytorch_derivative_legacy(
    time: Union[List[float], np.ndarray],
    signal: Union[List[float], np.ndarray],
    hidden_layers: List[int] = [128, 96, 64, 48, 32],
    epochs: int = 1000,
    holdout_fraction: float = 0.0,
    return_model: bool = False,
    order: int = 1,
    dropout: float = 0.1,
    learning_rate: float = 0.001,
    batch_size: int = 32,
    early_stopping: bool = True,
    patience: int = 50,
) -> Union[Callable[[Union[float, np.ndarray]], np.ndarray], Tuple[Callable[[Union[float, np.ndarray]], np.ndarray], Any]]:
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
        return _pytorch_derivative(time, signal, hidden_layers, epochs, holdout_fraction, return_model, order, 
                                  dropout=dropout, learning_rate=learning_rate, batch_size=batch_size, 
                                  early_stopping=early_stopping, patience=patience)
    elif framework == 'tensorflow':
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is not installed.")
        return _tensorflow_derivative(time, signal, hidden_layers, epochs, holdout_fraction, return_model, order, 
                                    dropout=dropout, learning_rate=learning_rate, batch_size=batch_size, 
                                    early_stopping=early_stopping, patience=patience)
    else:
        raise ValueError(f"Unknown framework: {framework}")


def _pytorch_derivative(
    X: Union[List[float], np.ndarray],
    Y: Union[List[float], np.ndarray],
    hidden_layers: List[int] = [128, 96, 64, 48, 32],
    epochs: int = 1000,
    holdout_fraction: float = 0.0,
    return_model: bool = False,
    order: int = 1,
    dropout: float = 0.1,
    learning_rate: float = 0.001,
    batch_size: int = 32,
    early_stopping: bool = True,
    patience: int = 50,
) -> Union[Callable[[Union[float, np.ndarray]], np.ndarray], Tuple[Callable[[Union[float, np.ndarray]], np.ndarray], Any]]:
    """
    Calculate derivatives using automatic differentiation with PyTorch for vector-valued input/output.
    """
    from pydelt.interpolation import PyTorchMLP
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    X = np.asarray(X)
    Y = np.asarray(Y)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    # Ensure data is sorted by first input dimension for consistency
    sort_idx = np.argsort(X[:, 0])
    X = X[sort_idx]
    Y = Y[sort_idx]

    # Normalize X and Y for better training
    X_min, X_max = X.min(axis=0), X.max(axis=0)
    X_norm = (X - X_min) / (X_max - X_min + 1e-12)
    Y_min, Y_max = Y.min(axis=0), Y.max(axis=0)
    Y_norm = (Y - Y_min) / (Y_max - Y_min + 1e-12)

    # Split data into training and holdout sets if requested
    N = X.shape[0]
    if 0.0 < holdout_fraction < 0.9:
        n_holdout = int(N * holdout_fraction)
        if n_holdout > 0:
            holdout_indices = np.random.choice(N, n_holdout, replace=False)
            train_indices = np.array([i for i in range(N) if i not in holdout_indices])
            X_train, Y_train = X_norm[train_indices], Y_norm[train_indices]
        else:
            X_train, Y_train = X_norm, Y_norm
    else:
        X_train, Y_train = X_norm, Y_norm

    # Prepare data for PyTorch
    X_train_torch = torch.tensor(X_train, dtype=torch.float32)
    Y_train_torch = torch.tensor(Y_train, dtype=torch.float32)
    dataset = TensorDataset(X_train_torch, Y_train_torch)
    dataloader = DataLoader(dataset, batch_size=min(batch_size, len(X_train)), shuffle=True)

    # Create and train the model
    model = PyTorchMLP(input_dim=X.shape[1], output_dim=Y.shape[1], hidden_layers=hidden_layers, dropout=dropout)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()
    
    # For early stopping
    best_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_X, batch_Y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = loss_fn(outputs, batch_Y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        # Early stopping check
        if early_stopping:
            avg_loss = epoch_loss / len(dataloader)
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if best_model_state is not None:
                        model.load_state_dict(best_model_state)
                    break
    
    model.eval()

    # Create derivative function
    def derivative_func(query_X):
        query_X = np.asarray(query_X)
        original_shape = query_X.shape
        
        # Determine input type
        scalar_input = query_X.ndim == 0
        vector_input = query_X.ndim == 1
        
        # Reshape for processing
        if scalar_input:
            # Single scalar value
            query_X = np.array([[query_X]])
        elif vector_input and len(query_X) > X.shape[1]:
            # Vector of points for 1D input (e.g., multiple time points)
            query_X = query_X.reshape(-1, 1)
        elif vector_input:
            # Single vector input for multivariate case
            query_X = query_X.reshape(1, -1)
        
        # Normalize query_X
        query_norm = (query_X - X_min) / (X_max - X_min + 1e-12)
        
        # Process each point individually
        jacobians = []
        for i in range(query_norm.shape[0]):
            # Get single sample and ensure correct shape
            sample = query_norm[i:i+1]
            sample_tensor = torch.tensor(sample, dtype=torch.float32, requires_grad=True)
            
            # Calculate derivatives based on order
            if order == 1:
                # First-order derivative (Jacobian)
                jac = torch.autograd.functional.jacobian(model, sample_tensor)
                jac = jac.detach().numpy()
                
                # Reshape to ensure consistent dimensions
                if jac.ndim > 2:
                    jac = jac.reshape(Y.shape[1], X.shape[1])
                
                # Scale Jacobian based on normalization
                scale_y = (Y_max - Y_min + 1e-12).reshape(-1, 1)
                scale_x = (X_max - X_min + 1e-12).reshape(1, -1)
                jac = jac * (scale_y / scale_x)
                jacobians.append(jac)
            else:
                # Higher-order derivatives (for scalar output only)
                if Y.shape[1] == 1:
                    # Create a function that returns scalar output for autograd
                    def scalar_model(x):
                        return model(x).squeeze()
                    
                    # Use PyTorch's hessian for 2nd order or manual nesting for higher
                    if order == 2:
                        hess = torch.autograd.functional.hessian(scalar_model, sample_tensor)
                        hess = hess.detach().numpy()
                        # Reshape and scale
                        hess = hess.reshape(X.shape[1], X.shape[1])
                        scale_factor = (Y_max - Y_min) / ((X_max - X_min + 1e-12) ** 2)
                        hess = hess * scale_factor
                        jacobians.append(hess)
                    else:
                        # For higher orders, approximate with finite differences
                        # This is a placeholder - higher orders need custom implementation
                        zeros = np.zeros((Y.shape[1], X.shape[1]))
                        jacobians.append(zeros)
                else:
                    # For vector outputs, higher-order derivatives are tensors
                    # Return zeros as placeholder
                    zeros = np.zeros((Y.shape[1], X.shape[1]))
                    jacobians.append(zeros)
        
        # Stack results
        result = np.stack(jacobians, axis=0)
        
        # Return appropriate shape based on input
        if scalar_input:
            if Y.shape[1] == 1 and X.shape[1] == 1:
                return result[0, 0, 0]  # Return scalar for scalar I/O
            else:
                return result[0]  # Return matrix for vector I/O with scalar input
        elif vector_input and len(original_shape) == 1 and original_shape[0] > X.shape[1]:
            if Y.shape[1] == 1 and X.shape[1] == 1:
                return result[:, 0, 0]  # Return vector for 1D case
            else:
                return result  # Return batch of matrices
        else:
            return result[0]  # Return single matrix for vector input

    if return_model:
        return derivative_func, model
    else:
        return derivative_func

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
    if 0.00 < holdout_fraction < 0.99:
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
            # Always use torch.tensor for input when requires_grad is needed
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
        
        return results.item() if scalar_input and np.ndim(results) == 0 else results
    
    if return_model:
        return derivative_func, model
    else:
        return derivative_func


def _tensorflow_derivative(
    X: Union[List[float], np.ndarray],
    Y: Union[List[float], np.ndarray],
    hidden_layers: List[int] = [128, 96, 64, 48, 32],
    epochs: int = 1000,
    holdout_fraction: float = 0.0,
    return_model: bool = False,
    order: int = 1,
    dropout: float = 0.1,
    learning_rate: float = 0.001,
    batch_size: int = 32,
    early_stopping: bool = True,
    patience: int = 50,
) -> Union[Callable[[Union[float, np.ndarray]], np.ndarray], Tuple[Callable[[Union[float, np.ndarray]], np.ndarray], Any]]:
    """
    Calculate derivatives using automatic differentiation with TensorFlow for vector-valued input/output.
    """
    from pydelt.interpolation import TensorFlowModel

    X = np.asarray(X)
    Y = np.asarray(Y)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    # Ensure data is sorted by first input dimension for consistency
    sort_idx = np.argsort(X[:, 0])
    X = X[sort_idx]
    Y = Y[sort_idx]

    # Normalize X and Y for better training
    X_min, X_max = X.min(axis=0), X.max(axis=0)
    X_norm = (X - X_min) / (X_max - X_min + 1e-12)
    Y_min, Y_max = Y.min(axis=0), Y.max(axis=0)
    Y_norm = (Y - Y_min) / (Y_max - Y_min + 1e-12)

    # Split data into training and holdout sets if requested
    N = X.shape[0]
    if 0.0 < holdout_fraction < 0.9:
        n_holdout = int(N * holdout_fraction)
        if n_holdout > 0:
            holdout_indices = np.random.choice(N, n_holdout, replace=False)
            train_indices = np.array([i for i in range(N) if i not in holdout_indices])
            X_train, Y_train = X_norm[train_indices], Y_norm[train_indices]
        else:
            X_train, Y_train = X_norm, Y_norm
    else:
        X_train, Y_train = X_norm, Y_norm

    # Create and train the model
    model = TensorFlowModel(input_dim=X.shape[1], output_dim=Y.shape[1], hidden_layers=hidden_layers, dropout=dropout)
    
    # Setup callbacks for early stopping if requested
    callbacks = []
    if early_stopping:
        try:
            from tensorflow.keras.callbacks import EarlyStopping
            es_callback = EarlyStopping(monitor='loss', patience=patience, restore_best_weights=True)
            callbacks.append(es_callback)
        except ImportError:
            # Fallback for older TensorFlow versions
            pass
    
    # Train the model with appropriate batch size and learning rate
    from tensorflow.keras.optimizers import Adam
    model.model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    # Create derivative function
    def derivative_func(query_X):
        query_X = np.asarray(query_X)
        original_shape = query_X.shape
        
        # Determine input type
        scalar_input = query_X.ndim == 0
        vector_input = query_X.ndim == 1
        
        # Reshape for processing
        if scalar_input:
            # Single scalar value
            query_X = np.array([[query_X]])
        elif vector_input and len(query_X) > X.shape[1]:
            # Vector of points for 1D input (e.g., multiple time points)
            # Each value needs to be treated as a separate sample
            query_X = query_X.reshape(-1, 1)
        elif vector_input:
            # Single vector input for multivariate case
            query_X = query_X.reshape(1, -1)
        
        # Normalize query_X
        query_norm = (query_X - X_min) / (X_max - X_min + 1e-12)
        
        # Process each point individually to avoid shape issues
        jacobians = []
        for i in range(query_norm.shape[0]):
            # Get single sample and ensure correct shape
            sample = query_norm[i:i+1]
            sample_tensor = tf.convert_to_tensor(sample, dtype=tf.float32)
            
            # Calculate derivatives
            if order == 1:
                # First-order derivative (Jacobian)
                with tf.GradientTape() as tape:
                    tape.watch(sample_tensor)
                    y = model.model(sample_tensor)
                
                # Get Jacobian and reshape
                jac = tape.jacobian(y, sample_tensor)
                jac = jac.numpy().reshape(Y.shape[1], X.shape[1])
                
                # Scale Jacobian based on normalization
                scale_y = (Y_max - Y_min + 1e-12).reshape(-1, 1)
                scale_x = (X_max - X_min + 1e-12).reshape(1, -1)
                jac = jac * (scale_y / scale_x)
                jacobians.append(jac)
            else:
                # Higher-order derivatives
                # For higher orders, we need to use nested GradientTape
                def get_nth_derivative(x, n):
                    if n == 0:
                        return model.model(x)
                    
                    with tf.GradientTape() as g:
                        g.watch(x)
                        y = get_nth_derivative(x, n-1)
                    return g.gradient(y, x)
                
                # Get the nth derivative
                deriv = get_nth_derivative(sample_tensor, order)
                
                # Scale for normalization
                scale_factor = (Y_max - Y_min) / (X_max - X_min + 1e-12)
                for _ in range(order):
                    scale_factor = scale_factor / (X_max - X_min + 1e-12)
                
                if deriv is not None:
                    deriv = deriv.numpy() * scale_factor
                    jacobians.append(deriv.reshape(Y.shape[1], X.shape[1]))
                else:
                    # If derivative is None (e.g., for constant functions)
                    jacobians.append(np.zeros((Y.shape[1], X.shape[1])))
        
        # Stack results
        result = np.stack(jacobians, axis=0)
        
        # Return appropriate shape based on input
        if scalar_input:
            if Y.shape[1] == 1 and X.shape[1] == 1:
                return result[0, 0, 0]  # Return scalar for scalar I/O
            else:
                return result[0]  # Return matrix for vector I/O with scalar input
        elif vector_input and len(original_shape) == 1 and original_shape[0] > X.shape[1]:
            if Y.shape[1] == 1 and X.shape[1] == 1:
                return result[:, 0, 0]  # Return vector for 1D case
            else:
                return result  # Return batch of matrices
        else:
            return result[0]  # Return single matrix for vector input

    if return_model:
        return derivative_func, model
    else:
        return derivative_func

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
        
        return derivative.item() if scalar_input and np.ndim(derivative) == 0 else derivative
    
    if return_model:
        return derivative_func, model
    else:
        return derivative_func
