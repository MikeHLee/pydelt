import numpy as np
import pytest
from pydelt.derivatives import lla, gold, glla, fda

def test_lla_sine():
    time = np.linspace(0, 10, 100)
    signal = np.sin(time)
    derivative, steps = lla(time.tolist(), signal.tolist(), window_size=5)
    
    # Test against known derivative of sine (cosine)
    expected = np.cos(time)
    # Allow some error due to numerical approximation
    assert np.allclose(derivative, expected, rtol=0.1, atol=0.1)

def test_lla_input_validation():
    with pytest.raises(ValueError):
        lla([1, 2, 3], [1, 2], window_size=3)

def test_gold_sine():
    time = np.linspace(0, 10, 100)
    signal = np.sin(time)
    result = gold(signal, time, embedding=5, n=2)
    
    # Test first derivative against cosine
    expected = np.cos(time[2:-2])  # Account for boundary effects
    assert np.allclose(result['dsignal'][:, 1], expected[:result['dsignal'].shape[0]], rtol=0.1, atol=0.1)

def test_gold_input_validation():
    with pytest.raises(ValueError):
        gold(np.array([1, 2, 3]), np.array([1, 2]), embedding=3, n=2)
    with pytest.raises(ValueError):
        gold(np.array([1, 2]), np.array([1, 2]), embedding=3, n=2)
    with pytest.raises(ValueError):
        gold(np.array([1, 2, 3]), np.array([1, 2, 3]), embedding=2, n=2)

def test_glla_sine():
    time = np.linspace(0, 10, 100)
    signal = np.sin(time)
    result = glla(signal, time, embedding=5, n=2)
    
    # Test first derivative against cosine
    expected = np.cos(time[2:-2])  # Account for boundary effects
    assert np.allclose(result['dsignal'][:, 1], expected[:result['dsignal'].shape[0]], rtol=0.1, atol=0.1)

def test_glla_input_validation():
    with pytest.raises(ValueError):
        glla(np.array([1, 2, 3]), np.array([1, 2]), embedding=3, n=2)
    with pytest.raises(ValueError):
        glla(np.array([1, 2]), np.array([1, 2]), embedding=3, n=2)
    with pytest.raises(ValueError):
        glla(np.array([1, 2, 3]), np.array([1, 2, 3]), embedding=2, n=2)

def test_fda_sine():
    time = np.linspace(0, 10, 100)
    signal = np.sin(time)
    result = fda(signal, time)
    
    # Test first derivative against cosine
    expected = np.cos(time)
    assert np.allclose(result['dsignal'][:, 1], expected, rtol=0.2, atol=0.2)  # Increased tolerance for spline approximation
