"""
Tensor derivatives computation for pydelt.

This module provides classes and functions for computing derivatives with respect to
vectors and tensors, extending the multivariate derivatives functionality:

- Directional derivatives: Derivative along a specific direction vector
- Tensor gradients: Gradient of tensor-valued functions
- Divergence: Divergence of vector fields
- Curl: Curl of vector fields
- Strain tensor: Symmetric part of the deformation gradient
- Stress tensor: Based on strain tensor and material properties

These operations are particularly useful in continuum mechanics, fluid dynamics,
and other physics applications requiring tensor calculus.
"""

import numpy as np
from typing import List, Tuple, Dict, Union, Optional, Callable, Any
from .multivariate import MultivariateDerivatives
from .interpolation import SplineInterpolator, LlaInterpolator
import warnings


class TensorDerivatives:
    """
    Compute derivatives with respect to vectors and tensors.
    
    This class extends the MultivariateDerivatives class to handle more complex
    tensor operations commonly used in continuum mechanics, fluid dynamics, and
    other physics applications.
    
    Key Features:
        - Directional derivatives: Derivative along a specific direction vector
        - Tensor gradients: Gradient of tensor-valued functions
        - Divergence: Divergence of vector fields
        - Curl: Curl of vector fields (3D only)
        - Strain tensor: Symmetric part of the deformation gradient
        - Stress tensor: Based on strain tensor and material properties
    
    Limitations:
        - Mixed partial derivatives are approximated as zero with traditional methods
        - Best suited for functions where each output depends primarily on one input
        - Critical point smoothing occurs due to interpolation methods
    
    Example:
        >>> from pydelt.tensor_derivatives import TensorDerivatives
        >>> from pydelt.interpolation import SplineInterpolator
        >>> td = TensorDerivatives(SplineInterpolator, smoothing=0.1)
        >>> td.fit(input_data, output_data)
        >>> div_func = td.divergence()
        >>> div = div_func(test_points)
    """
    
    def __init__(self, interpolator_class=SplineInterpolator, **interpolator_kwargs):
        """
        Initialize TensorDerivatives with specified interpolator.
        
        Args:
            interpolator_class: The interpolation class to use
            **interpolator_kwargs: Keyword arguments to pass to the interpolator constructor
        """
        self.mv = MultivariateDerivatives(interpolator_class, **interpolator_kwargs)
        self.fitted = False
        self.n_inputs = None
        self.n_outputs = None
    
    def fit(self, input_data: np.ndarray, output_data: np.ndarray):
        """
        Fit interpolators for tensor derivative computation.
        
        Args:
            input_data: Input data of shape (n_samples, n_inputs)
            output_data: Output data of shape (n_samples, n_outputs) or (n_samples,)
        
        Returns:
            self: The fitted instance
        """
        self.mv.fit(input_data, output_data)
        self.fitted = True
        self.n_inputs = self.mv.n_inputs
        self.n_outputs = self.mv.n_outputs
        return self
    
    def directional_derivative(self, direction: np.ndarray, normalize: bool = True) -> Callable:
        """
        Compute directional derivative along a specified direction.
        
        The directional derivative represents the instantaneous rate of change of the function
        in the direction of the vector. It is computed as the dot product of the gradient
        with the direction vector.
        
        Args:
            direction: Direction vector of shape (n_inputs,)
            normalize: Whether to normalize the direction vector to unit length
        
        Returns:
            Callable function that takes input points and returns directional derivatives
        """
        if not self.fitted:
            raise RuntimeError("TensorDerivatives must be fit before computing directional derivative.")
        
        if self.n_outputs != 1:
            raise ValueError(f"Directional derivative is only defined for scalar functions. Got {self.n_outputs} outputs.")
        
        direction = np.asarray(direction)
        if direction.shape != (self.n_inputs,):
            raise ValueError(f"Direction vector must have shape ({self.n_inputs},). Got {direction.shape}.")
        
        # Normalize direction vector if requested
        if normalize:
            direction = direction / np.linalg.norm(direction)
        
        gradient_func = self.mv.gradient()
        
        def directional_derivative_func(query_points: np.ndarray) -> np.ndarray:
            """
            Evaluate directional derivative at query points.
            
            Args:
                query_points: Points of shape (n_points, n_input_dims)
            
            Returns:
                Directional derivatives of shape (n_points,)
            """
            gradients = gradient_func(query_points)
            
            # Compute dot product of gradient with direction vector
            if gradients.ndim == 1:  # Single point, single output
                return np.dot(gradients, direction)
            else:  # Multiple points
                return np.dot(gradients, direction)
        
        return directional_derivative_func
    
    def divergence(self, eval_points: Optional[np.ndarray] = None) -> Callable:
        """
        Compute the divergence of a vector field.
        
        The divergence measures the "outgoingness" of a vector field at each point.
        Positive divergence indicates sources (expansion), while negative divergence
        indicates sinks (contraction).
        
        Args:
            eval_points: Optional points at which to evaluate the divergence
        
        Returns:
            Callable function that takes input points and returns divergence values
        """
        if not self.fitted:
            raise RuntimeError("TensorDerivatives must be fit before computing divergence.")
        
        if self.n_outputs != self.n_inputs:
            raise ValueError(f"Divergence requires vector field with n_outputs = n_inputs. Got {self.n_outputs} outputs and {self.n_inputs} inputs.")
        
        jacobian_func = self.mv.jacobian()
        
        def divergence_func(query_points: np.ndarray) -> np.ndarray:
            """
            Evaluate divergence at query points.
            
            Args:
                query_points: Points of shape (n_points, n_input_dims)
            
            Returns:
                Divergence values of shape (n_points,)
            """
            jacobians = jacobian_func(query_points)
            
            # Compute trace of Jacobian (sum of diagonal elements)
            if jacobians.ndim == 2:  # Single point
                return np.trace(jacobians)
            else:  # Multiple points
                return np.array([np.trace(jac) for jac in jacobians])
        
        if eval_points is not None:
            return divergence_func(eval_points)
        return divergence_func
    
    def curl(self, eval_points: Optional[np.ndarray] = None) -> Callable:
        """
        Compute the curl of a 3D vector field.
        
        The curl measures the "rotationality" of a vector field at each point.
        It is only defined for 3D vector fields (3 inputs, 3 outputs).
        
        Args:
            eval_points: Optional points at which to evaluate the curl
        
        Returns:
            Callable function that takes input points and returns curl vectors
        """
        if not self.fitted:
            raise RuntimeError("TensorDerivatives must be fit before computing curl.")
        
        if self.n_inputs != 3 or self.n_outputs != 3:
            raise ValueError(f"Curl is only defined for 3D vector fields (3 inputs, 3 outputs). Got {self.n_inputs} inputs and {self.n_outputs} outputs.")
        
        jacobian_func = self.mv.jacobian()
        
        def curl_func(query_points: np.ndarray) -> np.ndarray:
            """
            Evaluate curl at query points.
            
            Args:
                query_points: Points of shape (n_points, 3)
            
            Returns:
                Curl vectors of shape (n_points, 3)
            """
            jacobians = jacobian_func(query_points)
            
            # Compute curl from Jacobian
            if jacobians.ndim == 2:  # Single point
                curl = np.array([
                    jacobians[2, 1] - jacobians[1, 2],  # dF3/dy - dF2/dz
                    jacobians[0, 2] - jacobians[2, 0],  # dF1/dz - dF3/dx
                    jacobians[1, 0] - jacobians[0, 1]   # dF2/dx - dF1/dy
                ])
                return curl
            else:  # Multiple points
                curls = np.zeros((len(jacobians), 3))
                for i, jac in enumerate(jacobians):
                    curls[i] = np.array([
                        jac[2, 1] - jac[1, 2],  # dF3/dy - dF2/dz
                        jac[0, 2] - jac[2, 0],  # dF1/dz - dF3/dx
                        jac[1, 0] - jac[0, 1]   # dF2/dx - dF1/dy
                    ])
                return curls
        
        if eval_points is not None:
            return curl_func(eval_points)
        return curl_func
    
    def strain_tensor(self, eval_points: Optional[np.ndarray] = None) -> Callable:
        """
        Compute the strain tensor (symmetric part of the deformation gradient).
        
        The strain tensor represents the symmetric part of the deformation gradient
        and is commonly used in continuum mechanics to describe deformation.
        
        Args:
            eval_points: Optional points at which to evaluate the strain tensor
        
        Returns:
            Callable function that takes input points and returns strain tensors
        """
        if not self.fitted:
            raise RuntimeError("TensorDerivatives must be fit before computing strain tensor.")
        
        if self.n_outputs != self.n_inputs:
            raise ValueError(f"Strain tensor requires vector field with n_outputs = n_inputs. Got {self.n_outputs} outputs and {self.n_inputs} inputs.")
        
        jacobian_func = self.mv.jacobian()
        
        def strain_tensor_func(query_points: np.ndarray) -> np.ndarray:
            """
            Evaluate strain tensor at query points.
            
            Args:
                query_points: Points of shape (n_points, n_input_dims)
            
            Returns:
                Strain tensors of shape (n_points, n_inputs, n_inputs)
            """
            jacobians = jacobian_func(query_points)
            
            # Compute strain tensor as symmetric part of Jacobian: (J + J^T) / 2
            if jacobians.ndim == 2:  # Single point
                return 0.5 * (jacobians + jacobians.T)
            else:  # Multiple points
                return np.array([0.5 * (jac + jac.T) for jac in jacobians])
        
        if eval_points is not None:
            return strain_tensor_func(eval_points)
        return strain_tensor_func
    
    def stress_tensor(self, lambda_param: float = 1.0, mu_param: float = 0.5, 
                     eval_points: Optional[np.ndarray] = None) -> Callable:
        """
        Compute the stress tensor based on strain tensor and Lamé parameters.
        
        The stress tensor is computed using linear elasticity theory:
        σ = λ tr(ε) I + 2μ ε
        where:
        - λ and μ are Lamé parameters
        - ε is the strain tensor
        - I is the identity tensor
        - tr(ε) is the trace of the strain tensor
        
        Args:
            lambda_param: First Lamé parameter (λ)
            mu_param: Second Lamé parameter (μ, shear modulus)
            eval_points: Optional points at which to evaluate the stress tensor
        
        Returns:
            Callable function that takes input points and returns stress tensors
        """
        if not self.fitted:
            raise RuntimeError("TensorDerivatives must be fit before computing stress tensor.")
        
        if self.n_outputs != self.n_inputs:
            raise ValueError(f"Stress tensor requires vector field with n_outputs = n_inputs. Got {self.n_outputs} outputs and {self.n_inputs} inputs.")
        
        strain_tensor_func = self.strain_tensor()
        
        def stress_tensor_func(query_points: np.ndarray) -> np.ndarray:
            """
            Evaluate stress tensor at query points.
            
            Args:
                query_points: Points of shape (n_points, n_input_dims)
            
            Returns:
                Stress tensors of shape (n_points, n_inputs, n_inputs)
            """
            strain_tensors = strain_tensor_func(query_points)
            
            # Compute stress tensor using linear elasticity formula
            if strain_tensors.ndim == 2:  # Single point
                trace_strain = np.trace(strain_tensors)
                identity = np.eye(self.n_inputs)
                return lambda_param * trace_strain * identity + 2 * mu_param * strain_tensors
            else:  # Multiple points
                stress_tensors = np.zeros_like(strain_tensors)
                identity = np.eye(self.n_inputs)
                for i, strain in enumerate(strain_tensors):
                    trace_strain = np.trace(strain)
                    stress_tensors[i] = lambda_param * trace_strain * identity + 2 * mu_param * strain
                return stress_tensors
        
        if eval_points is not None:
            return stress_tensor_func(eval_points)
        return stress_tensor_func
    
    def tensor_gradient(self, eval_points: Optional[np.ndarray] = None) -> Callable:
        """
        Compute the gradient of a tensor-valued function.
        
        For a tensor-valued function, the gradient is a higher-order tensor.
        This method computes the gradient tensor by extending the Jacobian computation.
        
        Args:
            eval_points: Optional points at which to evaluate the tensor gradient
        
        Returns:
            Callable function that takes input points and returns tensor gradients
        """
        if not self.fitted:
            raise RuntimeError("TensorDerivatives must be fit before computing tensor gradient.")
        
        # For now, we'll use the Jacobian as the tensor gradient
        # In the future, this could be extended to handle higher-order tensors
        jacobian_func = self.mv.jacobian()
        
        def tensor_gradient_func(query_points: np.ndarray) -> np.ndarray:
            """
            Evaluate tensor gradient at query points.
            
            Args:
                query_points: Points of shape (n_points, n_input_dims)
            
            Returns:
                Tensor gradients of shape (n_points, n_outputs, n_inputs)
            """
            return jacobian_func(query_points)
        
        if eval_points is not None:
            return tensor_gradient_func(eval_points)
        return tensor_gradient_func
