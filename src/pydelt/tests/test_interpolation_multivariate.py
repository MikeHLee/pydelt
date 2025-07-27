"""
Multivariate (vector-valued) tests for interpolation in PyDelt.
"""
import numpy as np
import pytest
from pydelt.interpolation import SplineInterpolator, LowessInterpolator, LoessInterpolator

def generate_multivariate_data(n_points=40, n_out=2):
    time = np.linspace(0, 2 * np.pi, n_points)
    signal = np.stack([np.sin(time), np.cos(time)], axis=-1)
    return time, signal

def test_spline_interpolator_multivariate():
    time, signal = generate_multivariate_data()
    interp = SplineInterpolator().fit(time, signal)
    query_time = np.linspace(0.1, 2*np.pi-0.1, 10)
    preds = interp.predict(query_time)
    assert preds.shape == (10, 2)
    assert np.allclose(preds[:, 0], np.sin(query_time), rtol=0.1, atol=0.1)
    assert np.allclose(preds[:, 1], np.cos(query_time), rtol=0.1, atol=0.1)

def test_lowess_interpolator_multivariate():
    time, signal = generate_multivariate_data()
    interp = LowessInterpolator().fit(time, signal)
    query_time = np.linspace(0.1, 2*np.pi-0.1, 10)
    preds = interp.predict(query_time)
    assert preds.shape == (10, 2)
    assert np.all(np.isfinite(preds))

def test_loess_interpolator_multivariate():
    time, signal = generate_multivariate_data()
    interp = LoessInterpolator().fit(time, signal)
    query_time = np.linspace(0.1, 2*np.pi-0.1, 10)
    preds = interp.predict(query_time)
    assert preds.shape == (10, 2)
    assert np.all(np.isfinite(preds))
