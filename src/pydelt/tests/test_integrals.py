"""
Tests for the integrals module.
"""

import numpy as np
import pytest
from pydelt.integrals import integrate_derivative, integrate_derivative_with_error
from pydelt.derivatives import lla

def test_integrate_constant_derivative():
    """Test integration of a constant derivative (should give linear function)."""
    time = np.linspace(0, 10, 100)
    derivative = np.ones_like(time)  # constant derivative = 1
    integral = integrate_derivative(time, derivative)
    
    # Should approximate y = x
    expected = time
    np.testing.assert_allclose(integral, expected, rtol=1e-2)

def test_integrate_sine():
    """Test integration of cosine (derivative of sine) should give sine."""
    time = np.linspace(0, 10, 500)
    derivative = np.cos(time)
    integral = integrate_derivative(time, derivative)
    
    # Should approximate sine
    expected = np.sin(time)
    np.testing.assert_allclose(integral, expected, rtol=1e-2)

def test_integrate_with_initial_value():
    """Test integration with non-zero initial value."""
    time = np.linspace(0, 10, 100)
    derivative = np.ones_like(time)
    initial_value = 5.0
    integral = integrate_derivative(time, derivative, initial_value=initial_value)
    
    # Should approximate y = x + 5
    expected = time + initial_value
    np.testing.assert_allclose(integral, expected, rtol=1e-2)

def test_integrate_with_error():
    """Test error estimation in integration."""
    time = np.linspace(0, 10, 500)
    signal = np.sin(time)
    derivative, _ = lla(time.tolist(), signal.tolist(), window_size=5)
    
    reconstructed, error = integrate_derivative_with_error(time, derivative, initial_value=signal[0])
    
    # Error should be non-negative
    assert np.all(error >= 0)
    
    # Reconstructed signal should be close to original
    np.testing.assert_allclose(reconstructed, signal, rtol=1e-1)

def test_input_types():
    """Test that functions accept both lists and numpy arrays."""
    time = [0, 1, 2, 3, 4]
    derivative = [1, 1, 1, 1, 1]
    
    # Test with lists
    result_list = integrate_derivative(time, derivative)
    assert isinstance(result_list, np.ndarray)
    
    # Test with numpy arrays
    result_array = integrate_derivative(np.array(time), np.array(derivative))
    assert isinstance(result_array, np.ndarray)
    
    # Results should be identical
    np.testing.assert_array_equal(result_list, result_array)
