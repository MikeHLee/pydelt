"""
Multivariate derivative computation for pydelt.

This module provides classes for computing multivariate derivatives including:
- Gradient (∇f): For scalar functions of multiple variables
- Jacobian (∂f/∂x): For vector-valued functions
- Hessian (∂²f/∂x²): Second derivatives for scalar functions
- Laplacian (∇²f): Trace of Hessian for scalar functions

The module supports both traditional interpolation methods and neural networks
with automatic differentiation for superior performance in high dimensions.
"""

import numpy as np
from typing import List, Tuple, Dict, Union, Optional, Callable, Any
from .interpolation import BaseInterpolator, SplineInterpolator, LlaInterpolator
import warnings

try:
    import torch
    import torch.nn as nn
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


class MultivariateDerivatives:
    """
    Compute multivariate derivatives using traditional interpolation methods.
    
    This class provides methods to compute gradient, Jacobian, Hessian, and Laplacian
    for multivariate functions using fitted interpolators. It works by fitting separate
    1D interpolators for each output-input dimension pair and computing derivatives
    along each dimension independently.
    
    Key Features:
        - Gradient computation for scalar functions
        - Jacobian matrix computation for vector-valued functions
        - Hessian matrix computation for scalar functions (diagonal elements only)
        - Laplacian computation for scalar functions
        - Support for any interpolator from the pydelt library
    
    Limitations:
        - Mixed partial derivatives are approximated as zero
        - Assumes separable dependencies between input dimensions
        - Best suited for functions where each output depends primarily on one input
        - **Critical Point Smoothing**: Interpolation methods smooth out sharp mathematical
          features, leading to non-zero gradients at points where they should be zero
        - **Boundary Effects**: Edge artifacts can distort derivative calculations
        - **Regularization Impact**: Smoothing parameters blur sharp transitions and critical points
        - **Finite Sampling**: Cannot capture infinitely sharp transitions or discontinuities
    
    **Example of Critical Point Issue:**
        For f(x,y) = (x-y)², the mathematical gradient is zero along x=y line,
        especially at corners like (-3,-3) and (3,3). However, numerical interpolation
        will give non-zero gradients everywhere due to smoothing effects.
    
    **Mitigation Strategies:**
        - Use higher resolution sampling near critical points
        - Reduce smoothing parameters (but beware of overfitting)
        - Validate against analytical solutions when available
        - Consider neural network methods with automatic differentiation for exact derivatives
    
    For exact mixed partial derivatives and better handling of critical points,
    consider using NeuralNetworkMultivariateDerivatives with automatic differentiation.
    
    Example:
        >>> from pydelt.interpolation import SplineInterpolator
        >>> mv = MultivariateDerivatives(SplineInterpolator, smoothing=0.1)
        >>> mv.fit(input_data, output_data)
        >>> gradient_func = mv.gradient()
        >>> grad = gradient_func(test_points)
    
    Args:
        interpolator_class: The interpolation class to use (e.g., SplineInterpolator, LlaInterpolator)
        **interpolator_kwargs: Keyword arguments to pass to the interpolator constructor
    """
    
    def __init__(self, interpolator_class: type = SplineInterpolator, **interpolator_kwargs):
        self.interpolator_class = interpolator_class
        self.interpolator_kwargs = interpolator_kwargs
        self.interpolators = None
        self.input_data = None
        self.output_data = None
        self.n_inputs = None
        self.n_outputs = None
        self.fitted = False
    
    def fit(self, input_data: np.ndarray, output_data: np.ndarray):
        """
        Fit interpolators for multivariate derivative computation.
        
        This method fits separate 1D interpolators for each output-input dimension pair.
        For each combination of output dimension i and input dimension j, an interpolator
        is fitted using input_data[:, j] as x-values and output_data[:, i] as y-values.
        
        The data is automatically sorted and duplicate x-values are handled by averaging
        the corresponding y-values to ensure proper interpolation.
        
        Args:
            input_data: Input data of shape (n_samples, n_input_dims)
                       Must be 2D array where each row is a sample and each column
                       is an input dimension.
            output_data: Output data of shape (n_samples,) for scalar functions or 
                        (n_samples, n_output_dims) for vector-valued functions.
                        For scalar functions, can be 1D array.
        
        Returns:
            Self for method chaining
        
        Raises:
            ValueError: If input_data is not 2D or if input/output sample counts don't match
        
        Note:
            After fitting, the interpolators are stored in self.interpolators as a nested list
            where interpolators[i][j] contains the interpolator for output i, input j.
        """
        input_data = np.asarray(input_data)
        output_data = np.asarray(output_data)
        
        # Handle 1D input data by reshaping to 2D
        if input_data.ndim == 1:
            input_data = input_data.reshape(-1, 1)
        elif input_data.ndim != 2:
            raise ValueError(f"Input data must be 1D or 2D. Got shape {input_data.shape}")
        
        self.n_samples, self.n_inputs = input_data.shape
        
        # Handle scalar vs vector output
        if output_data.ndim == 1:
            self.n_outputs = 1
            output_data = output_data.reshape(-1, 1)
        else:
            self.n_outputs = output_data.shape[1]
        
        if output_data.shape[0] != self.n_samples:
            raise ValueError(f"Input and output data must have same number of samples. "
                           f"Got {self.n_samples} and {output_data.shape[0]}")
        
        # Store data for derivative computation
        self.input_data = input_data.copy()
        self.output_data = output_data.copy()
        
        # Create interpolators for each output-input pair
        self.interpolators = []
        for output_dim in range(self.n_outputs):
            output_interpolators = []
            for input_dim in range(self.n_inputs):
                # Create interpolator for this output-input pair
                interpolator = self.interpolator_class(**self.interpolator_kwargs)
                
                # Fit interpolator using input dimension as x and output as y
                # Sort by input dimension to ensure proper interpolation
                x_values = input_data[:, input_dim]
                y_values = output_data[:, output_dim]
                
                # Sort data by x_values for proper interpolation
                sort_indices = np.argsort(x_values)
                x_sorted = x_values[sort_indices]
                y_sorted = y_values[sort_indices]
                
                # Remove duplicate x values by averaging y values
                unique_x, inverse_indices = np.unique(x_sorted, return_inverse=True)
                unique_y = np.zeros_like(unique_x)
                for i in range(len(unique_x)):
                    mask = inverse_indices == i
                    unique_y[i] = np.mean(y_sorted[mask])
                
                try:
                    interpolator.fit(unique_x, unique_y)
                    output_interpolators.append(interpolator)
                except Exception as e:
                    warnings.warn(f"Failed to fit interpolator for output {output_dim}, input {input_dim}: {e}")
                    # Create a dummy interpolator that returns zeros
                    class DummyInterpolator:
                        def differentiate(self, order=1):
                            def dummy_func(x):
                                return np.zeros_like(x)
                            return dummy_func
                    output_interpolators.append(DummyInterpolator())
            
            self.interpolators.append(output_interpolators)
        
        self.fitted = True
        return self
    
    def gradient(self, eval_points: Optional[np.ndarray] = None) -> Callable:
        """
        Compute the gradient (∇f) for scalar functions.
        
        The gradient is computed by fitting separate 1D interpolators for each input dimension
        and evaluating their first derivatives. This approach works well for functions where
        each output depends primarily on one input dimension.
        
        Args:
            eval_points: Points at which to evaluate the gradient. If None, uses training points.
        
        Returns:
            Callable function that takes input points and returns gradient vectors.
            
            For single evaluation point:
                - Input: array of shape (n_input_dims,) or (1, n_input_dims)
                - Output: array of shape (n_input_dims,)
            
            For multiple evaluation points:
                - Input: array of shape (n_points, n_input_dims)
                - Output: array of shape (n_points, n_input_dims)
        
        Raises:
            ValueError: If the function is not scalar (n_outputs != 1)
            RuntimeError: If the model has not been fitted
        
        Note:
            This method assumes that partial derivatives can be approximated by fitting
            1D interpolators along each input dimension independently. Mixed partial
            derivatives are not computed with this approach.
            
        **Critical Point Warning:**
            Numerical interpolation smooths out sharp mathematical features. For functions
            with critical points (where gradient should be zero), the computed gradient
            may be non-zero due to smoothing effects. Always validate against analytical
            solutions when possible, especially near minima, maxima, and saddle points.
        """
        if not self.fitted:
            raise RuntimeError("MultivariateDerivatives must be fit before computing gradient.")
        
        if self.n_outputs != 1:
            raise ValueError(f"Gradient is only defined for scalar functions. Got {self.n_outputs} outputs.")
        
        def gradient_func(query_points: np.ndarray) -> np.ndarray:
            """
            Evaluate gradient at query points.
            
            Args:
                query_points: Points of shape (n_points, n_input_dims)
            
            Returns:
                Gradient vectors of shape (n_points, n_input_dims)
            """
            query_points = np.asarray(query_points)
            if query_points.ndim == 1:
                query_points = query_points.reshape(1, -1)
            
            n_points = query_points.shape[0]
            gradients = np.zeros((n_points, self.n_inputs))
            
            # Compute partial derivative with respect to each input dimension
            for input_dim in range(self.n_inputs):
                interpolator = self.interpolators[0][input_dim]  # First (and only) output
                derivative_func = interpolator.differentiate(order=1)
                
                # Evaluate derivative at the corresponding input dimension
                input_values = query_points[:, input_dim]
                try:
                    partial_derivatives = derivative_func(input_values)
                    # Handle scalar output from derivative function
                    if np.isscalar(partial_derivatives):
                        partial_derivatives = np.array([partial_derivatives])
                    gradients[:, input_dim] = partial_derivatives
                except Exception as e:
                    warnings.warn(f"Could not compute partial derivative for dimension {input_dim}: {e}")
                    gradients[:, input_dim] = 0.0
            
            # For single point, return shape (n_inputs,), for multiple points (n_points, n_inputs)
            return gradients[0] if n_points == 1 else gradients
        
        return gradient_func
    
    def jacobian(self, eval_points: Optional[np.ndarray] = None) -> Callable:
        """
        Compute the Jacobian matrix (∂f/∂x) for vector-valued functions.
        
        The Jacobian matrix contains all first-order partial derivatives of a vector-valued
        function. Element (i,j) represents ∂f_i/∂x_j. This is computed by fitting separate
        1D interpolators for each output-input dimension pair.
        
        Args:
            eval_points: Points at which to evaluate the Jacobian. If None, uses training points.
        
        Returns:
            Callable function that takes input points and returns Jacobian matrices.
            
            For single evaluation point:
                - Input: array of shape (n_input_dims,) or (1, n_input_dims)
                - Output: array of shape (n_outputs, n_inputs)
            
            For multiple evaluation points:
                - Input: array of shape (n_points, n_input_dims)
                - Output: array of shape (n_points, n_outputs, n_inputs)
        
        Raises:
            RuntimeError: If the model has not been fitted
        
        Note:
            For scalar functions (n_outputs=1), the Jacobian is equivalent to the gradient
            transposed. Mixed partial derivatives are not computed with traditional
            interpolation methods.
        """
        if not self.fitted:
            raise RuntimeError("MultivariateDerivatives must be fit before computing Jacobian.")
        
        def jacobian_func(query_points: np.ndarray) -> np.ndarray:
            """
            Evaluate Jacobian at query points.
            
            Args:
                query_points: Points of shape (n_points, n_input_dims)
            
            Returns:
                Jacobian matrices of shape (n_points, n_outputs, n_inputs)
            """
            query_points = np.asarray(query_points)
            if query_points.ndim == 1:
                query_points = query_points.reshape(1, -1)
            
            n_points = query_points.shape[0]
            jacobians = np.zeros((n_points, self.n_outputs, self.n_inputs))
            
            # Compute partial derivatives for each output with respect to each input
            for output_dim in range(self.n_outputs):
                for input_dim in range(self.n_inputs):
                    interpolator = self.interpolators[output_dim][input_dim]
                    derivative_func = interpolator.differentiate(order=1)
                    
                    # Evaluate derivative at the corresponding input dimension
                    input_values = query_points[:, input_dim]
                    try:
                        partial_derivatives = derivative_func(input_values)
                        # Handle scalar output from derivative function
                        if np.isscalar(partial_derivatives):
                            partial_derivatives = np.array([partial_derivatives])
                        jacobians[:, output_dim, input_dim] = partial_derivatives
                    except Exception as e:
                        warnings.warn(f"Could not compute partial derivative for output {output_dim}, input {input_dim}: {e}")
                        jacobians[:, output_dim, input_dim] = 0.0
            
            # For single point, return shape (n_outputs, n_inputs), for multiple points (n_points, n_outputs, n_inputs)
            return jacobians[0] if n_points == 1 else jacobians
        
        return jacobian_func
    
    def hessian(self, eval_points: Optional[np.ndarray] = None) -> Callable:
        """
        Compute the Hessian matrix (∂²f/∂x²) for scalar functions.
        
        The Hessian matrix contains all second-order partial derivatives of a scalar function.
        Element (i,j) represents ∂²f/∂x_i∂x_j. For traditional interpolation methods, only
        diagonal elements (pure second derivatives) are computed; off-diagonal elements
        (mixed partials) are approximated as zero.
        
        Args:
            eval_points: Points at which to evaluate the Hessian. If None, uses training points.
        
        Returns:
            Callable function that takes input points and returns Hessian matrices.
            
            For single evaluation point:
                - Input: array of shape (n_input_dims,) or (1, n_input_dims)
                - Output: array of shape (n_inputs, n_inputs)
            
            For multiple evaluation points:
                - Input: array of shape (n_points, n_input_dims)
                - Output: array of shape (n_points, n_inputs, n_inputs)
        
        Raises:
            ValueError: If the function is not scalar (n_outputs != 1)
            RuntimeError: If the model has not been fitted
        
        Warning:
            Traditional interpolation methods cannot compute mixed partial derivatives
            (off-diagonal elements). These are set to zero, which may not be accurate
            for functions with significant cross-dependencies. Consider using neural
            network methods with automatic differentiation for exact mixed partials.
        """
        if not self.fitted:
            raise RuntimeError("MultivariateDerivatives must be fit before computing Hessian.")
        
        if self.n_outputs != 1:
            raise ValueError(f"Hessian is only defined for scalar functions. Got {self.n_outputs} outputs.")
        
        def hessian_func(query_points: np.ndarray) -> np.ndarray:
            """
            Evaluate Hessian at query points.
            
            Args:
                query_points: Points of shape (n_points, n_input_dims)
            
            Returns:
                Hessian matrices of shape (n_points, n_inputs, n_inputs)
            """
            query_points = np.asarray(query_points)
            if query_points.ndim == 1:
                query_points = query_points.reshape(1, -1)
            
            n_points = query_points.shape[0]
            hessians = np.zeros((n_points, self.n_inputs, self.n_inputs))
            
            # Compute second partial derivatives (diagonal elements)
            for input_dim in range(self.n_inputs):
                interpolator = self.interpolators[0][input_dim]  # First (and only) output
                try:
                    second_derivative_func = interpolator.differentiate(order=2)
                    input_values = query_points[:, input_dim]
                    partial_derivatives = second_derivative_func(input_values)
                    # Handle scalar output from derivative function
                    if np.isscalar(partial_derivatives):
                        partial_derivatives = np.array([partial_derivatives])
                    hessians[:, input_dim, input_dim] = partial_derivatives
                except Exception as e:
                    warnings.warn(f"Could not compute second derivative for dimension {input_dim}: {e}")
                    hessians[:, input_dim, input_dim] = 0.0
            
            # Mixed partial derivatives (off-diagonal elements)
            # Note: For traditional interpolation methods, mixed partials are approximated as zero
            # This is a limitation that neural networks with automatic differentiation can overcome
            
            # For single point, return shape (n_inputs, n_inputs), for multiple points (n_points, n_inputs, n_inputs)
            return hessians[0] if n_points == 1 else hessians
        
        return hessian_func
    
    def laplacian(self, eval_points: Optional[np.ndarray] = None) -> Callable:
        """
        Compute the Laplacian (∇²f = tr(H)) for scalar functions.
        
        The Laplacian is the trace (sum of diagonal elements) of the Hessian matrix,
        representing the sum of all pure second partial derivatives: ∇²f = ∂²f/∂x₁² + ∂²f/∂x₂² + ...
        This is a scalar measure of the "curvature" of the function.
        
        Args:
            eval_points: Points at which to evaluate the Laplacian. If None, uses training points.
        
        Returns:
            Callable function that takes input points and returns Laplacian values.
            
            For single evaluation point:
                - Input: array of shape (n_input_dims,) or (1, n_input_dims)
                - Output: scalar value
            
            For multiple evaluation points:
                - Input: array of shape (n_points, n_input_dims)
                - Output: array of shape (n_points,)
        
        Raises:
            ValueError: If the function is not scalar (n_outputs != 1)
            RuntimeError: If the model has not been fitted
        
        Note:
            Since mixed partial derivatives are not computed in traditional interpolation
            methods, this Laplacian only includes pure second derivatives. For functions
            with significant cross-dependencies, consider neural network methods.
        """
        if not self.fitted:
            raise RuntimeError("MultivariateDerivatives must be fit before computing Laplacian.")
        
        if self.n_outputs != 1:
            raise ValueError(f"Laplacian is only defined for scalar functions. Got {self.n_outputs} outputs.")
        
        def laplacian_func(query_points: np.ndarray) -> np.ndarray:
            """
            Evaluate Laplacian at query points.
            
            Args:
                query_points: Points of shape (n_points, n_input_dims)
            
            Returns:
                Laplacian values of shape (n_points,)
            """
            query_points = np.asarray(query_points)
            if query_points.ndim == 1:
                query_points = query_points.reshape(1, -1)
            
            n_points = query_points.shape[0]
            laplacians = np.zeros(n_points)
            
            # Laplacian is the trace of the Hessian (sum of diagonal elements)
            for input_dim in range(self.n_inputs):
                interpolator = self.interpolators[0][input_dim]  # First (and only) output
                try:
                    second_derivative_func = interpolator.differentiate(order=2)
                    input_values = query_points[:, input_dim]
                    partial_derivatives = second_derivative_func(input_values)
                    # Handle scalar output from derivative function
                    if np.isscalar(partial_derivatives):
                        partial_derivatives = np.array([partial_derivatives])
                    laplacians += partial_derivatives
                except Exception as e:
                    warnings.warn(f"Could not compute second derivative for dimension {input_dim}: {e}")
            
            # For single point, return scalar, for multiple points return array
            return laplacians[0] if n_points == 1 else laplacians
        
        return laplacian_func


class NeuralNetworkMultivariateDerivatives:
    """
    Neural network-based multivariate derivatives with automatic differentiation.
    
    This class provides true multivariate derivative computation using automatic
    differentiation, which is superior to traditional methods for high-dimensional
    problems and can compute exact mixed partial derivatives.
    
    Args:
        framework: Deep learning framework to use ('pytorch' or 'tensorflow')
        hidden_layers: List of hidden layer sizes
        epochs: Number of training epochs
        learning_rate: Learning rate for optimization
    """
    
    def __init__(self, 
                 framework: str = 'pytorch',
                 hidden_layers: List[int] = [64, 32],
                 epochs: int = 1000,
                 learning_rate: float = 0.01):
        
        if framework == 'pytorch' and not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch is required for neural network derivatives. Install with: pip install torch")
        elif framework == 'tensorflow' and not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for neural network derivatives. Install with: pip install tensorflow")
        
        self.framework = framework
        self.hidden_layers = hidden_layers
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.model = None
        self.input_data = None
        self.output_data = None
        self.n_inputs = None
        self.n_outputs = None
        self.fitted = False
        
        # Normalization parameters
        self.input_mean = None
        self.input_std = None
        self.output_mean = None
        self.output_std = None
    
    def fit(self, input_data: np.ndarray, output_data: np.ndarray) -> 'NeuralNetworkMultivariateDerivatives':
        """
        Fit the neural network model.
        
        Args:
            input_data: Input data of shape (n_samples, n_input_dims)
            output_data: Output data of shape (n_samples,) for scalar functions or 
                        (n_samples, n_output_dims) for vector functions
        
        Returns:
            Self for method chaining
        """
        input_data = np.asarray(input_data)
        output_data = np.asarray(output_data)
        
        if input_data.ndim == 1:
            input_data = input_data.reshape(-1, 1)
        if output_data.ndim == 1:
            output_data = output_data.reshape(-1, 1)
        
        self.input_data = input_data
        self.output_data = output_data
        self.n_inputs = input_data.shape[1]
        self.n_outputs = output_data.shape[1]
        
        # Normalize data for better training
        self.input_mean = np.mean(input_data, axis=0)
        self.input_std = np.std(input_data, axis=0) + 1e-8  # Avoid division by zero
        self.output_mean = np.mean(output_data, axis=0)
        self.output_std = np.std(output_data, axis=0) + 1e-8
        
        input_normalized = (input_data - self.input_mean) / self.input_std
        output_normalized = (output_data - self.output_mean) / self.output_std
        
        if self.framework == 'pytorch':
            self._fit_pytorch(input_normalized, output_normalized)
        elif self.framework == 'tensorflow':
            self._fit_tensorflow(input_normalized, output_normalized)
        else:
            raise ValueError(f"Unknown framework: {self.framework}")
        
        self.fitted = True
        return self
    
    def _fit_pytorch(self, input_data: np.ndarray, output_data: np.ndarray):
        """Fit PyTorch model."""
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
        
        # Create neural network architecture
        layers = []
        prev_size = self.n_inputs
        
        for hidden_size in self.hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, self.n_outputs))
        
        self.model = nn.Sequential(*layers)
        
        # Prepare data
        X = torch.tensor(input_data, dtype=torch.float32)
        y = torch.tensor(output_data, dtype=torch.float32)
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=min(32, len(input_data)), shuffle=True)
        
        # Train model
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()
        
        self.model.train()
        for epoch in range(self.epochs):
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = loss_fn(outputs, batch_y)
                loss.backward()
                optimizer.step()
        
        self.model.eval()
    
    def _fit_tensorflow(self, input_data: np.ndarray, output_data: np.ndarray):
        """Fit TensorFlow model."""
        import tensorflow as tf
        
        # Create neural network architecture
        layers = [tf.keras.layers.Input(shape=(self.n_inputs,))]
        
        for hidden_size in self.hidden_layers:
            layers.append(tf.keras.layers.Dense(hidden_size, activation='relu'))
        
        layers.append(tf.keras.layers.Dense(self.n_outputs))
        
        self.model = tf.keras.Sequential(layers)
        
        # Compile and train
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                          loss='mse')
        
        self.model.fit(input_data, output_data, epochs=self.epochs, verbose=0)
    
    def gradient(self) -> Callable:
        """
        Compute gradient using automatic differentiation.
        
        Returns:
            Callable function that takes input points and returns gradient vectors
        """
        if not self.fitted:
            raise RuntimeError("NeuralNetworkMultivariateDerivatives must be fit before computing gradient.")
        
        if self.n_outputs != 1:
            raise ValueError(f"Gradient is only defined for scalar functions. Got {self.n_outputs} outputs.")
        
        if self.framework == 'pytorch':
            return self._pytorch_gradient()
        elif self.framework == 'tensorflow':
            return self._tensorflow_gradient()
    
    def _pytorch_gradient(self) -> Callable:
        """PyTorch gradient computation."""
        import torch
        
        def gradient_func(query_points: np.ndarray) -> np.ndarray:
            query_points = np.asarray(query_points)
            if query_points.ndim == 1:
                query_points = query_points.reshape(1, -1)
            
            # Normalize input
            query_normalized = (query_points - self.input_mean) / self.input_std
            query_tensor = torch.tensor(query_normalized, dtype=torch.float32, requires_grad=True)
            
            # Forward pass
            output = self.model(query_tensor)
            
            # Compute gradients
            gradients = []
            for i in range(query_tensor.shape[0]):
                grad = torch.autograd.grad(output[i], query_tensor, create_graph=True, retain_graph=True)[0]
                gradients.append(grad[i].detach().numpy())
            
            gradients = np.array(gradients)
            
            # Denormalize gradients
            gradients = gradients * (self.output_std / self.input_std)
            
            return gradients.squeeze() if gradients.shape[0] == 1 else gradients
        
        return gradient_func
    
    def _tensorflow_gradient(self) -> Callable:
        """TensorFlow gradient computation."""
        import tensorflow as tf
        
        def gradient_func(query_points: np.ndarray) -> np.ndarray:
            query_points = np.asarray(query_points)
            if query_points.ndim == 1:
                query_points = query_points.reshape(1, -1)
            
            # Normalize input
            query_normalized = (query_points - self.input_mean) / self.input_std
            query_tensor = tf.Variable(query_normalized, dtype=tf.float32)
            
            with tf.GradientTape() as tape:
                output = self.model(query_tensor)
            
            # Compute gradients
            gradients = tape.gradient(output, query_tensor).numpy()
            
            # Denormalize gradients
            gradients = gradients * (self.output_std / self.input_std)
            
            return gradients.squeeze() if gradients.shape[0] == 1 else gradients
        
        return gradient_func
    
    def jacobian(self) -> Callable:
        """
        Compute Jacobian using automatic differentiation.
        
        Returns:
            Callable function that takes input points and returns Jacobian matrices
        """
        if not self.fitted:
            raise RuntimeError("NeuralNetworkMultivariateDerivatives must be fit before computing Jacobian.")
        
        if self.framework == 'pytorch':
            return self._pytorch_jacobian()
        elif self.framework == 'tensorflow':
            return self._tensorflow_jacobian()
    
    def _pytorch_jacobian(self) -> Callable:
        """PyTorch Jacobian computation."""
        import torch
        
        def jacobian_func(query_points: np.ndarray) -> np.ndarray:
            query_points = np.asarray(query_points)
            if query_points.ndim == 1:
                query_points = query_points.reshape(1, -1)
            
            # Normalize input
            query_normalized = (query_points - self.input_mean) / self.input_std
            query_tensor = torch.tensor(query_normalized, dtype=torch.float32, requires_grad=True)
            
            # Forward pass
            output = self.model(query_tensor)
            
            # Compute Jacobian
            jacobians = []
            for i in range(query_tensor.shape[0]):
                point_jacobian = []
                for j in range(self.n_outputs):
                    grad = torch.autograd.grad(output[i, j], query_tensor, 
                                             create_graph=True, retain_graph=True)[0]
                    point_jacobian.append(grad[i].detach().numpy())
                jacobians.append(np.array(point_jacobian))
            
            jacobians = np.array(jacobians)
            
            # Denormalize Jacobian
            jacobians = jacobians * (self.output_std.reshape(1, -1, 1) / self.input_std.reshape(1, 1, -1))
            
            return jacobians.squeeze() if jacobians.shape[0] == 1 else jacobians
        
        return jacobian_func
    
    def _tensorflow_jacobian(self) -> Callable:
        """TensorFlow Jacobian computation."""
        import tensorflow as tf
        
        def jacobian_func(query_points: np.ndarray) -> np.ndarray:
            query_points = np.asarray(query_points)
            if query_points.ndim == 1:
                query_points = query_points.reshape(1, -1)
            
            # Normalize input
            query_normalized = (query_points - self.input_mean) / self.input_std
            query_tensor = tf.Variable(query_normalized, dtype=tf.float32)
            
            with tf.GradientTape() as tape:
                output = self.model(query_tensor)
            
            # Compute Jacobian
            jacobians = []
            for i in range(self.n_outputs):
                grad = tape.gradient(output[:, i], query_tensor)
                jacobians.append(grad.numpy())
            
            jacobians = np.array(jacobians).transpose(1, 0, 2)  # (n_points, n_outputs, n_inputs)
            
            # Denormalize Jacobian
            jacobians = jacobians * (self.output_std.reshape(1, -1, 1) / self.input_std.reshape(1, 1, -1))
            
            return jacobians.squeeze() if jacobians.shape[0] == 1 else jacobians
        
        return jacobian_func
    
    def hessian(self) -> Callable:
        """
        Compute Hessian using automatic differentiation.
        
        Returns:
            Callable function that takes input points and returns Hessian matrices
        """
        if not self.fitted:
            raise RuntimeError("NeuralNetworkMultivariateDerivatives must be fit before computing Hessian.")
        
        if self.n_outputs != 1:
            raise ValueError(f"Hessian is only defined for scalar functions. Got {self.n_outputs} outputs.")
        
        if self.framework == 'pytorch':
            return self._pytorch_hessian()
        elif self.framework == 'tensorflow':
            return self._tensorflow_hessian()
    
    def _pytorch_hessian(self) -> Callable:
        """PyTorch Hessian computation."""
        import torch
        
        def hessian_func(query_points: np.ndarray) -> np.ndarray:
            query_points = np.asarray(query_points)
            if query_points.ndim == 1:
                query_points = query_points.reshape(1, -1)
            
            # Normalize input
            query_normalized = (query_points - self.input_mean) / self.input_std
            query_tensor = torch.tensor(query_normalized, dtype=torch.float32, requires_grad=True)
            
            # Forward pass
            output = self.model(query_tensor)
            
            # Compute Hessian
            hessians = []
            for i in range(query_tensor.shape[0]):
                # First derivatives
                first_grad = torch.autograd.grad(output[i], query_tensor, 
                                               create_graph=True, retain_graph=True)[0]
                
                # Second derivatives
                hessian_matrix = []
                for j in range(self.n_inputs):
                    second_grad = torch.autograd.grad(first_grad[i, j], query_tensor, 
                                                    create_graph=True, retain_graph=True)[0]
                    hessian_matrix.append(second_grad[i].detach().numpy())
                
                hessians.append(np.array(hessian_matrix))
            
            hessians = np.array(hessians)
            
            # Denormalize Hessian
            hessians = hessians * (self.output_std / (self.input_std.reshape(1, -1) * self.input_std.reshape(-1, 1)))
            
            return hessians.squeeze() if hessians.shape[0] == 1 else hessians
        
        return hessian_func
    
    def _tensorflow_hessian(self) -> Callable:
        """TensorFlow Hessian computation."""
        import tensorflow as tf
        
        def hessian_func(query_points: np.ndarray) -> np.ndarray:
            query_points = np.asarray(query_points)
            if query_points.ndim == 1:
                query_points = query_points.reshape(1, -1)
            
            # Normalize input
            query_normalized = (query_points - self.input_mean) / self.input_std
            query_tensor = tf.Variable(query_normalized, dtype=tf.float32)
            
            with tf.GradientTape() as tape2:
                with tf.GradientTape() as tape1:
                    output = self.model(query_tensor)
                first_grad = tape1.gradient(output, query_tensor)
            
            # Compute Hessian
            hessians = []
            for i in range(self.n_inputs):
                second_grad = tape2.gradient(first_grad[:, i], query_tensor)
                hessians.append(second_grad.numpy())
            
            hessians = np.array(hessians).transpose(1, 0, 2)  # (n_points, n_inputs, n_inputs)
            
            # Denormalize Hessian
            hessians = hessians * (self.output_std / (self.input_std.reshape(1, 1, -1) * self.input_std.reshape(1, -1, 1)))
            
            return hessians.squeeze() if hessians.shape[0] == 1 else hessians
        
        return hessian_func
    
    def laplacian(self) -> Callable:
        """
        Compute Laplacian using automatic differentiation.
        
        Returns:
            Callable function that takes input points and returns Laplacian values
        """
        if not self.fitted:
            raise RuntimeError("NeuralNetworkMultivariateDerivatives must be fit before computing Laplacian.")
        
        if self.n_outputs != 1:
            raise ValueError(f"Laplacian is only defined for scalar functions. Got {self.n_outputs} outputs.")
        
        def laplacian_func(query_points: np.ndarray) -> np.ndarray:
            hessian_func = self.hessian()
            hessian_matrices = hessian_func(query_points)
            
            if hessian_matrices.ndim == 2:  # Single point
                return np.trace(hessian_matrices)
            else:  # Multiple points
                return np.array([np.trace(h) for h in hessian_matrices])
        
        return laplacian_func
