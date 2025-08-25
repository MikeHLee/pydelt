"""
Stochastic derivatives framework for pydelt interpolation methods.

This module provides stochastic link functions and derivative transforms
using Itô's lemma and Stratonovich integral approaches.
"""

import numpy as np
from scipy import stats
from typing import Callable, Optional, Union, Dict, Any
from abc import ABC, abstractmethod


class StochasticLinkFunction(ABC):
    """Abstract base class for stochastic link functions."""
    
    @abstractmethod
    def transform(self, x: np.ndarray) -> np.ndarray:
        """Transform input through the link function."""
        pass
    
    @abstractmethod
    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        """Inverse transform from link space back to original space."""
        pass
    
    @abstractmethod
    def derivative_transform(self, x: np.ndarray, dx: np.ndarray) -> np.ndarray:
        """Transform derivatives through the link function."""
        pass
    
    @abstractmethod
    def ito_correction(self, x: np.ndarray, d2x: np.ndarray) -> np.ndarray:
        """Apply Itô's lemma correction term."""
        pass
    
    @abstractmethod
    def stratonovich_correction(self, x: np.ndarray, dx: np.ndarray) -> np.ndarray:
        """Apply Stratonovich integral correction."""
        pass


class NormalLink(StochasticLinkFunction):
    """Normal (Gaussian) distribution link function."""
    
    def __init__(self, mu: float = 0.0, sigma: float = 1.0):
        self.mu = mu
        self.sigma = sigma
    
    def transform(self, x: np.ndarray) -> np.ndarray:
        """Identity transform for normal distribution."""
        return x
    
    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        """Identity inverse transform."""
        return y
    
    def derivative_transform(self, x: np.ndarray, dx: np.ndarray) -> np.ndarray:
        """Identity derivative transform."""
        return dx
    
    def ito_correction(self, x: np.ndarray, d2x: np.ndarray) -> np.ndarray:
        """No correction needed for normal distribution."""
        return np.zeros_like(x)
    
    def stratonovich_correction(self, x: np.ndarray, dx: np.ndarray) -> np.ndarray:
        """No correction needed for normal distribution."""
        return np.zeros_like(x)


class LogNormalLink(StochasticLinkFunction):
    """Log-normal distribution link function."""
    
    def __init__(self, mu: float = 0.0, sigma: float = 1.0):
        self.mu = mu
        self.sigma = sigma
    
    def transform(self, x: np.ndarray) -> np.ndarray:
        """Log transform."""
        return np.log(np.maximum(x, 1e-10))  # Avoid log(0)
    
    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        """Exponential inverse transform."""
        return np.exp(y)
    
    def derivative_transform(self, x: np.ndarray, dx: np.ndarray) -> np.ndarray:
        """Transform derivatives: d(log(x))/dt = (1/x) * dx/dt."""
        return dx / np.maximum(x, 1e-10)
    
    def ito_correction(self, x: np.ndarray, d2x: np.ndarray) -> np.ndarray:
        """Itô correction: -0.5 * sigma^2 for log-normal."""
        return -0.5 * self.sigma**2 * np.ones_like(x)
    
    def stratonovich_correction(self, x: np.ndarray, dx: np.ndarray) -> np.ndarray:
        """Stratonovich correction for log-normal."""
        return 0.5 * self.sigma**2 * np.ones_like(x)


class GammaLink(StochasticLinkFunction):
    """Gamma distribution link function (log link)."""
    
    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        self.alpha = alpha
        self.beta = beta
    
    def transform(self, x: np.ndarray) -> np.ndarray:
        """Log transform for gamma distribution."""
        return np.log(np.maximum(x, 1e-10))
    
    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        """Exponential inverse transform."""
        return np.exp(y)
    
    def derivative_transform(self, x: np.ndarray, dx: np.ndarray) -> np.ndarray:
        """Transform derivatives through log link."""
        return dx / np.maximum(x, 1e-10)
    
    def ito_correction(self, x: np.ndarray, d2x: np.ndarray) -> np.ndarray:
        """Itô correction for gamma distribution."""
        variance = self.alpha / (self.beta**2)
        return -0.5 * variance / np.maximum(x**2, 1e-10)
    
    def stratonovich_correction(self, x: np.ndarray, dx: np.ndarray) -> np.ndarray:
        """Stratonovich correction for gamma distribution."""
        variance = self.alpha / (self.beta**2)
        return 0.5 * variance / np.maximum(x**2, 1e-10)


class BetaLink(StochasticLinkFunction):
    """Beta distribution link function (logit link)."""
    
    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        self.alpha = alpha
        self.beta = beta
    
    def transform(self, x: np.ndarray) -> np.ndarray:
        """Logit transform: log(x/(1-x))."""
        x_clipped = np.clip(x, 1e-10, 1 - 1e-10)
        return np.log(x_clipped / (1 - x_clipped))
    
    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        """Inverse logit transform: exp(y)/(1+exp(y))."""
        return 1 / (1 + np.exp(-y))
    
    def derivative_transform(self, x: np.ndarray, dx: np.ndarray) -> np.ndarray:
        """Transform derivatives through logit link."""
        x_clipped = np.clip(x, 1e-10, 1 - 1e-10)
        return dx / (x_clipped * (1 - x_clipped))
    
    def ito_correction(self, x: np.ndarray, d2x: np.ndarray) -> np.ndarray:
        """Itô correction for beta distribution."""
        x_clipped = np.clip(x, 1e-10, 1 - 1e-10)
        variance = (self.alpha * self.beta) / ((self.alpha + self.beta)**2 * (self.alpha + self.beta + 1))
        return -0.5 * variance / (x_clipped**2 * (1 - x_clipped)**2)
    
    def stratonovich_correction(self, x: np.ndarray, dx: np.ndarray) -> np.ndarray:
        """Stratonovich correction for beta distribution."""
        x_clipped = np.clip(x, 1e-10, 1 - 1e-10)
        variance = (self.alpha * self.beta) / ((self.alpha + self.beta)**2 * (self.alpha + self.beta + 1))
        return 0.5 * variance / (x_clipped**2 * (1 - x_clipped)**2)


class ExponentialLink(StochasticLinkFunction):
    """Exponential distribution link function (log link)."""
    
    def __init__(self, rate: float = 1.0):
        self.rate = rate
    
    def transform(self, x: np.ndarray) -> np.ndarray:
        """Log transform for exponential distribution."""
        return np.log(np.maximum(x, 1e-10))
    
    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        """Exponential inverse transform."""
        return np.exp(y)
    
    def derivative_transform(self, x: np.ndarray, dx: np.ndarray) -> np.ndarray:
        """Transform derivatives through log link."""
        return dx / np.maximum(x, 1e-10)
    
    def ito_correction(self, x: np.ndarray, d2x: np.ndarray) -> np.ndarray:
        """Itô correction for exponential distribution."""
        variance = 1 / (self.rate**2)
        return -0.5 * variance / np.maximum(x**2, 1e-10)
    
    def stratonovich_correction(self, x: np.ndarray, dx: np.ndarray) -> np.ndarray:
        """Stratonovich correction for exponential distribution."""
        variance = 1 / (self.rate**2)
        return 0.5 * variance / np.maximum(x**2, 1e-10)


class PoissonLink(StochasticLinkFunction):
    """Poisson distribution link function (log link)."""
    
    def __init__(self, lam: float = 1.0):
        self.lam = lam
    
    def transform(self, x: np.ndarray) -> np.ndarray:
        """Log transform for Poisson distribution."""
        return np.log(np.maximum(x, 1e-10))
    
    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        """Exponential inverse transform."""
        return np.exp(y)
    
    def derivative_transform(self, x: np.ndarray, dx: np.ndarray) -> np.ndarray:
        """Transform derivatives through log link."""
        return dx / np.maximum(x, 1e-10)
    
    def ito_correction(self, x: np.ndarray, d2x: np.ndarray) -> np.ndarray:
        """Itô correction for Poisson distribution."""
        # Variance equals mean for Poisson
        return -0.5 * self.lam / np.maximum(x**2, 1e-10)
    
    def stratonovich_correction(self, x: np.ndarray, dx: np.ndarray) -> np.ndarray:
        """Stratonovich correction for Poisson distribution."""
        return 0.5 * self.lam / np.maximum(x**2, 1e-10)


class StochasticDerivativeTransform:
    """
    Transforms derivatives using stochastic calculus principles.
    
    Supports both Itô's lemma and Stratonovich integral approaches.
    """
    
    def __init__(self, link_function: StochasticLinkFunction, method: str = "ito"):
        """
        Initialize stochastic derivative transform.
        
        Args:
            link_function: The stochastic link function to use
            method: Either "ito" or "stratonovich" for the integration method
        """
        self.link_function = link_function
        if method not in ["ito", "stratonovich"]:
            raise ValueError("Method must be 'ito' or 'stratonovich'")
        self.method = method
    
    def transform_derivatives(self, x: np.ndarray, derivatives: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        """
        Transform derivatives using stochastic calculus.
        
        Args:
            x: Input values where derivatives are evaluated
            derivatives: Dictionary mapping derivative order to derivative values
        
        Returns:
            Dictionary of transformed derivatives
        """
        transformed = {}
        
        # First derivative transformation
        if 1 in derivatives:
            dx = derivatives[1]
            transformed_dx = self.link_function.derivative_transform(x, dx)
            
            # Apply stochastic correction
            if self.method == "ito" and 2 in derivatives:
                correction = self.link_function.ito_correction(x, derivatives[2])
                transformed_dx += correction
            elif self.method == "stratonovich":
                correction = self.link_function.stratonovich_correction(x, dx)
                transformed_dx += correction
            
            transformed[1] = transformed_dx
        
        # Higher-order derivatives (simplified approach)
        for order in derivatives:
            if order > 1:
                # For higher orders, apply the derivative transform recursively
                # This is a simplified approach - full stochastic calculus would be more complex
                transformed[order] = self.link_function.derivative_transform(x, derivatives[order])
        
        return transformed


# Registry of available link functions
LINK_FUNCTIONS = {
    'normal': NormalLink,
    'lognormal': LogNormalLink,
    'gamma': GammaLink,
    'beta': BetaLink,
    'exponential': ExponentialLink,
    'poisson': PoissonLink,
}


def create_link_function(name: str, **kwargs) -> StochasticLinkFunction:
    """
    Create a stochastic link function by name.
    
    Args:
        name: Name of the link function ('normal', 'lognormal', etc.)
        **kwargs: Parameters for the specific link function
    
    Returns:
        Initialized link function
    """
    if name not in LINK_FUNCTIONS:
        available = list(LINK_FUNCTIONS.keys())
        raise ValueError(f"Unknown link function '{name}'. Available: {available}")
    
    return LINK_FUNCTIONS[name](**kwargs)
