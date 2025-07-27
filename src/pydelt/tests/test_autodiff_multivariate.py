"""
Multivariate (vector-valued) tests for automatic differentiation in PyDelt.
"""
import numpy as np
import pytest
from pydelt.autodiff import neural_network_derivative

def generate_multivariate_data(n_points=100, n_out=2):
    time = np.linspace(0, 2 * np.pi, n_points)
    # Example: stack sin and cos as two outputs
    signal = np.stack([np.sin(time), np.cos(time)], axis=-1)
    return time, signal

def test_neural_network_derivative_multivariate_pytorch():
    pytest.importorskip('torch')
    time, signal = generate_multivariate_data(200, 2)
    deriv_func = neural_network_derivative(time, signal, framework='pytorch', hidden_layers=[32, 16], dropout=0.01, epochs=100)
    query_time = np.linspace(0.1, 2*np.pi-0.1, 10)
    jacobians = deriv_func(query_time)  # Should be (10, 2, 1) or (10, 2, n_in)
    assert jacobians.shape[:2] == (10, 2)
    # For 1D input, Jacobian last dim should be 1
    assert jacobians.shape[2] == 1
    # Check numerical correctness for first output (sin -> cos)
    assert np.allclose(jacobians[:, 0, 0], np.cos(query_time), rtol=0.15, atol=0.15)
    # Second output (cos -> -sin)
    assert np.allclose(jacobians[:, 1, 0], -np.sin(query_time), rtol=0.15, atol=0.15)

def test_neural_network_derivative_multivariate_tensorflow():
    pytest.importorskip('tensorflow')
    time, signal = generate_multivariate_data(200, 2)
    deriv_func = neural_network_derivative(time, signal, framework='tensorflow', hidden_layers=[32, 16], dropout=0.01, epochs=100)
    query_time = np.linspace(0.1, 2*np.pi-0.1, 10)
    jacobians = deriv_func(query_time)
    assert jacobians.shape[:2] == (10, 2)
    assert jacobians.shape[2] == 1
    assert np.allclose(jacobians[:, 0, 0], np.cos(query_time), rtol=0.2, atol=0.2)
    assert np.allclose(jacobians[:, 1, 0], -np.sin(query_time), rtol=0.2, atol=0.2)
