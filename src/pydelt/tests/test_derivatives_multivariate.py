"""
Multivariate (vector-valued) tests for derivatives in PyDelt.
"""
import numpy as np
import pytest
from pydelt.derivatives import lla, fda

def generate_multivariate_data(n_points=100, n_out=2):
    time = np.linspace(0, 2 * np.pi, n_points)
    signal = np.stack([np.sin(time), np.cos(time)], axis=-1)
    return time, signal

def test_lla_multivariate():
    time, signal = generate_multivariate_data()
    deriv, steps = lla(time, signal, window_size=7)
    assert deriv.shape == signal.shape
    # Check numerical accuracy for first output (sin -> cos)
    assert np.allclose(deriv[:, 0], np.cos(time), rtol=0.15, atol=0.15)
    # Second output (cos -> -sin)
    assert np.allclose(deriv[:, 1], -np.sin(time), rtol=0.15, atol=0.15)

def test_fda_multivariate():
    time, signal = generate_multivariate_data()
    result = fda(signal, time)
    dsignal = result['dsignal']
    # dsignal shape: (N, n_out, 3)
    assert dsignal.shape[:2] == signal.shape
    # Test first derivative
    assert np.allclose(dsignal[:, 0, 1], np.cos(time), rtol=0.2, atol=0.2)
    assert np.allclose(dsignal[:, 1, 1], -np.sin(time), rtol=0.2, atol=0.2)
