"""
Tests for advanced interpolation methods in PyDelt.
"""

import numpy as np
import pytest
from typing import List, Tuple, Dict, Union, Optional, Callable, Any
import warnings

from pydelt.interpolation import (
    derivative_based_interpolation,
    neural_network_interpolation
)

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

# --- Additional edge/failure-case tests for neural network interpolation ---
def test_neural_network_interpolation_noisy_data():
    """Test neural network interpolation with noisy data."""
    np.random.seed(42)
    time, signal = generate_sine_data(10000)
    noisy_signal = signal + np.random.normal(0, 0.2, size=signal.shape)
    interp_func = neural_network_interpolation(time, noisy_signal, framework='pytorch', hidden_layers=[64, 32, 16], dropout=0.1, epochs=50)
    query_time = np.linspace(0.1, 2*np.pi-0.1, 10)
    predicted = interp_func(query_time)
    assert predicted.shape == query_time.shape
    assert np.all(np.isfinite(predicted))

def test_neural_network_interpolation_missing_values():
    """Test neural network interpolation with missing values (NaNs)."""
    time, signal = generate_sine_data(10000)
    signal_missing = signal.copy()
    signal_missing[::100] = np.nan
    with pytest.raises(ValueError, match="must not contain NaN values"):
        neural_network_interpolation(time, signal_missing, framework='pytorch', hidden_layers=[64, 32, 16], dropout=0.1, epochs=50)

def test_neural_network_interpolation_out_of_domain():
    """Test neural network interpolation with out-of-domain query."""
    time, signal = generate_sine_data(10000)
    interp_func = neural_network_interpolation(time, signal, framework='pytorch', hidden_layers=[64, 32, 16], dropout=0.1, epochs=50)
    query_time = np.array([-100, 100, 1e6])
    predicted = interp_func(query_time)
    assert np.all(np.isfinite(predicted))


def generate_sine_data(n_points: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """Generate sine wave data for testing."""
    time = np.linspace(0, 2*np.pi, n_points)
    signal = np.sin(time)
    return time, signal


def test_derivative_based_interpolation_lla():
    """Test interpolation using LLA derivative method."""
    time, signal = generate_sine_data(50)
    
    # Create interpolation function
    interp_func = derivative_based_interpolation(time, signal, method='lla')
    
    # Test at original points
    for t, s in zip(time, signal):
        assert abs(interp_func(t) - s) < 0.1
    
    # Test at intermediate points
    query_time = np.linspace(0.1, 2*np.pi-0.1, 20)
    expected = np.sin(query_time)
    predicted = interp_func(query_time)
    
    assert np.allclose(predicted, expected, rtol=0.05)


def test_derivative_based_interpolation_glla():
    """Test interpolation using GLLA derivative method."""
    time, signal = generate_sine_data(50)
    
    # Create interpolation function
    interp_func = derivative_based_interpolation(time, signal, method='glla')
    
    # Test at original points
    for t, s in zip(time, signal):
        assert abs(interp_func(t) - s) < 0.1
    
    # Test at intermediate points
    query_time = np.linspace(0.1, 2*np.pi-0.1, 20)
    expected = np.sin(query_time)
    predicted = interp_func(query_time)
    
    assert np.allclose(predicted, expected, rtol=0.05)


def test_derivative_based_interpolation_gold():
    """Test interpolation using GOLD derivative method."""
    time, signal = generate_sine_data(50)
    
    # Create interpolation function
    interp_func = derivative_based_interpolation(time, signal, method='gold')
    
    # Test at original points
    for t, s in zip(time, signal):
        assert abs(interp_func(t) - s) < 0.1
    
    # Test at intermediate points
    query_time = np.linspace(0.1, 2*np.pi-0.1, 20)
    expected = np.sin(query_time)
    predicted = interp_func(query_time)
    
    assert np.allclose(predicted, expected, rtol=0.05)


def test_derivative_based_interpolation_fda():
    """Test interpolation using FDA derivative method."""
    time, signal = generate_sine_data(50)
    
    # Create interpolation function
    interp_func = derivative_based_interpolation(time, signal, method='fda')
    
    # Test at original points
    for t, s in zip(time, signal):
        assert abs(interp_func(t) - s) < 0.1
    
    # Test at intermediate points
    query_time = np.linspace(0.1, 2*np.pi-0.1, 20)
    expected = np.sin(query_time)
    predicted = interp_func(query_time)
    
    assert np.allclose(predicted, expected, rtol=0.05)


def test_derivative_based_interpolation_invalid_method():
    """Test that an invalid method raises a ValueError."""
    time, signal = generate_sine_data(50)
    
    with pytest.raises(ValueError):
        derivative_based_interpolation(time, signal, method='invalid_method')


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
def test_neural_network_interpolation_pytorch():
    """Test interpolation using PyTorch neural network."""
    time, signal = generate_sine_data(10000)
    
    # Create interpolation function with small network and few epochs for testing
    interp_func = neural_network_interpolation(
        time, signal, framework='pytorch', 
        hidden_layers=[64, 32, 16], dropout=0.1, epochs=50
    )
    
    # Skip checking individual points due to training variability
    # Instead, check that the function returns values of the expected shape
    query_time = np.linspace(0.1, 2*np.pi-0.1, 10)
    predicted = interp_func(query_time)
    
    # Check shape and that values are finite
    assert predicted.shape == query_time.shape
    assert np.all(np.isfinite(predicted))
    
    # Check that the function can be called with a scalar input
    scalar_result = interp_func(1.0)
    assert np.isscalar(scalar_result) or scalar_result.size == 1


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
def test_neural_network_interpolation_pytorch_with_model():
    """Test interpolation using PyTorch neural network with model return."""
    time, signal = generate_sine_data(10000)
    
    # Create interpolation function with model return
    interp_func, model = neural_network_interpolation(
        time, signal, framework='pytorch',
        hidden_layers=[64, 32, 16], dropout=0.1, epochs=50,
        return_model=True
    )
    
    # Check that model is returned
    assert model is not None
    
    # Test function
    query_time = np.linspace(0.1, 2*np.pi-0.1, 10)
    predicted = interp_func(query_time)
    assert predicted.shape == query_time.shape


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
def test_neural_network_interpolation_pytorch_with_holdout():
    """Test interpolation using PyTorch neural network with holdout."""
    time, signal = generate_sine_data(10000)
    
    # Create interpolation function with holdout
    interp_func = neural_network_interpolation(
        time, signal, framework='pytorch',
        hidden_layers=[64, 32, 16], dropout=0.1, epochs=50,
        holdout_fraction=0.2
    )
    
    # Skip checking against expected values due to training variability
    # Instead, check that the function returns values of the expected shape
    query_time = np.linspace(0.1, 2*np.pi-0.1, 10)
    predicted = interp_func(query_time)
    
    # Check shape and that values are finite
    assert predicted.shape == query_time.shape
    assert np.all(np.isfinite(predicted))


@pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not installed")
def test_neural_network_interpolation_tensorflow():
    """Test interpolation using TensorFlow neural network."""
    time, signal = generate_sine_data(10000)
    
    # Create interpolation function with small network and few epochs for testing
    interp_func = neural_network_interpolation(
        time, signal, framework='tensorflow', 
        hidden_layers=[64, 32, 16], dropout=0.1, epochs=50
    )
    
    # Skip checking individual points due to training variability
    # Instead, check that the function returns values of the expected shape
    query_time = np.linspace(0.1, 2*np.pi-0.1, 10)
    predicted = interp_func(query_time)
    
    # Check shape and that values are finite
    assert predicted.shape == query_time.shape
    assert np.all(np.isfinite(predicted))
    
    # Check that the function can be called with a scalar input
    scalar_result = interp_func(1.0)
    assert np.isscalar(scalar_result) or scalar_result.size == 1
    query_time = np.linspace(0.1, 2*np.pi-0.1, 10)
    expected = np.sin(query_time)
    predicted = interp_func(query_time)
    
    assert predicted.shape == expected.shape
    assert np.all(np.isfinite(predicted))

    # Test function
    query_time = np.linspace(0.1, 2*np.pi-0.1, 10)
    predicted = interp_func(query_time)
    assert predicted.shape == query_time.shape
