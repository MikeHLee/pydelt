"""
Multivariate (vector-valued) tests for integrals in PyDelt.
"""
import numpy as np
import pytest
from pydelt.integrals import integrate_derivative, integrate_derivative_with_error

def generate_multivariate_derivative(n_points=100, n_out=2):
    time = np.linspace(0, 2 * np.pi, n_points)
    # Derivative of sin is cos, derivative of cos is -sin
    derivative = np.stack([np.cos(time), -np.sin(time)], axis=-1)
    initial_value = np.array([0.0, 1.0])  # sin(0)=0, cos(0)=1
    true_signal = np.stack([np.sin(time), np.cos(time)], axis=-1)
    return time, derivative, initial_value, true_signal

def test_integrate_derivative_multivariate():
    time, derivative, initial_value, true_signal = generate_multivariate_derivative()
    integral = integrate_derivative(time, derivative, initial_value=initial_value)
    assert integral.shape == (len(time), 2)
    np.testing.assert_allclose(integral, true_signal, rtol=1e-2, atol=1e-2)

def test_integrate_derivative_with_error_multivariate():
    time, derivative, initial_value, true_signal = generate_multivariate_derivative()
    integral, error = integrate_derivative_with_error(time, derivative, initial_value=initial_value)
    assert integral.shape == (len(time), 2)
    assert error.shape == (len(time), 2)
    assert np.all(error >= 0)
    np.testing.assert_allclose(integral, true_signal, rtol=1e-2, atol=1e-2)
