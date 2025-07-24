"""
Tests for automatic differentiation methods in PyDelt.
"""

import numpy as np
import pytest
from typing import List, Tuple, Dict, Union, Optional, Callable, Any
import warnings

# Skip tests if dependencies are not available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# Import functions to test
from pydelt.autodiff import (
    neural_network_derivative
)

# --- Additional edge/failure-case tests ---

def generate_sine_data(n_points: int = 50):
    """Generate sine wave data for testing."""
    time = np.linspace(0, 2 * np.pi, n_points)
    signal = np.sin(time)
    return time, signal

def test_neural_network_derivative_noisy_data():
    """Test neural network derivative with noisy data."""
    np.random.seed(42)
    time, signal = generate_sine_data(10000)
    noisy_signal = signal + np.random.normal(0, 0.2, size=signal.shape)
    deriv_func = neural_network_derivative(time, noisy_signal, framework='pytorch', hidden_layers=[64, 32, 16], dropout=0.025, epochs=300)
    query_time = np.linspace(0.1, 2*np.pi-0.1, 10)
    expected = np.cos(query_time)
    predicted = deriv_func(query_time)
    assert predicted.shape == expected.shape
    assert np.all(np.isfinite(predicted))

def test_neural_network_derivative_missing_values():
    """Test neural network derivative with missing values (NaNs)."""
    time, signal = generate_sine_data(10000)
    signal_missing = signal.copy()
    signal_missing[::100] = np.nan  # Introduce NaNs every 100th point
    with pytest.raises(ValueError, match="must not contain NaN values"):
        neural_network_derivative(time, signal_missing, framework='pytorch', hidden_layers=[64, 32, 16], dropout=0.025, epochs=300)

def test_neural_network_derivative_out_of_domain():
    """Test neural network derivative with out-of-domain query."""
    time, signal = generate_sine_data(10000)
    deriv_func = neural_network_derivative(time, signal, framework='pytorch', hidden_layers=[64, 32, 16], dropout=0.025, epochs=300)
    # Query far outside training domain
    query_time = np.array([-100, 100, 1e6])
    predicted = deriv_func(query_time)
    # Should return finite numbers (may not be accurate, but should not crash)
    assert predicted.shape == query_time.shape
    assert np.all(np.isfinite(predicted))

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
def test_neural_network_derivative_pytorch():
    """Test derivative calculation using PyTorch neural network."""
    time, signal = generate_sine_data(10000)
    
    # Create derivative function with small network and few epochs for testing
    deriv_func = neural_network_derivative(
        time, signal, framework='pytorch', 
        hidden_layers=[64, 32, 16], dropout=0.025, epochs=300
    )
    
    # Test at intermediate points
    query_time = np.linspace(0.1, 2*np.pi-0.1, 10)
    expected = np.cos(query_time)  # Derivative of sin(x) is cos(x)
    predicted = deriv_func(query_time)
    assert predicted.shape == expected.shape
    assert np.all(np.isfinite(predicted))

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
def test_neural_network_derivative_pytorch_second_order():
    """Test second-order derivative calculation using PyTorch neural network."""
    time, signal = generate_sine_data(10000)
    
    # Create derivative function for second-order derivative
    deriv_func = neural_network_derivative(
        time, signal, framework='pytorch', 
        hidden_layers=[64, 32, 16], dropout=0.025, epochs=300,
        order=2
    )
    
    # Test at intermediate points
    query_time = np.linspace(0.1, 2*np.pi-0.1, 10)
    expected = -np.sin(query_time)  # Second derivative of sin(x) is -sin(x)
    predicted = deriv_func(query_time)
    assert predicted.shape == expected.shape
    assert np.all(np.isfinite(predicted))

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
def test_neural_network_derivative_pytorch_with_model():
    """Test derivative calculation using PyTorch neural network with model return."""
    time, signal = generate_sine_data(10000)
    
    # Create derivative function with model return
    deriv_func, model = neural_network_derivative(
        time, signal, framework='pytorch',
        hidden_layers=[64, 32, 16], dropout=0.1, epochs=50,
        return_model=True
    )
    
    # Check that model is returned
    assert model is not None
    
    # Test function
    query_time = np.linspace(0.1, 2*np.pi-0.1, 10)
    predicted = deriv_func(query_time)
    assert predicted.shape == query_time.shape

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
def test_neural_network_derivative_pytorch_with_holdout():
    """Test derivative calculation using PyTorch neural network with holdout."""
    # Use 10,000 points for realistic deep learning test
    time, signal = generate_sine_data(10000)
    
    # Create derivative function with holdout
    deriv_func = neural_network_derivative(
        time, signal, framework='pytorch',
        hidden_layers=[64, 32, 16], dropout=0.1, epochs=50,
        holdout_fraction=0.2
    )
    
    # Test at intermediate points
    query_time = np.linspace(0.1, 2*np.pi-0.1, 10)
    expected = np.cos(query_time)
    predicted = deriv_func(query_time)
    assert predicted.shape == expected.shape
    assert np.all(np.isfinite(predicted))

@pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not installed")
def test_neural_network_derivative_tensorflow():
    """Test derivative calculation using TensorFlow neural network."""
    time, signal = generate_sine_data(10000)
    
    # Create derivative function with small network and few epochs for testing
    deriv_func = neural_network_derivative(
        time, signal, framework='tensorflow', 
        hidden_layers=[64, 32, 16], dropout=0.1, epochs=50
    )
    
    # Test at intermediate points
    query_time = np.linspace(0.1, 2*np.pi-0.1, 10)
    expected = np.cos(query_time)  # Derivative of sin(x) is cos(x)
    predicted = deriv_func(query_time)
    assert predicted.shape == expected.shape
    assert np.all(np.isfinite(predicted))

@pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not installed")
def test_neural_network_derivative_tensorflow_second_order():
    """Test second-order derivative calculation using TensorFlow neural network."""
    time, signal = generate_sine_data(10000)
    
    # Create derivative function for second-order derivative
    deriv_func = neural_network_derivative(
        time, signal, framework='tensorflow', 
        hidden_layers=[64, 32, 16], dropout=0.1, epochs=50,
        order=2
    )
    
    # Test at intermediate points
    query_time = np.linspace(0.1, 2*np.pi-0.1, 10)
    expected = -np.sin(query_time)  # Second derivative of sin(x) is -sin(x)
    predicted = deriv_func(query_time)
    assert predicted.shape == expected.shape
    assert np.all(np.isfinite(predicted))
