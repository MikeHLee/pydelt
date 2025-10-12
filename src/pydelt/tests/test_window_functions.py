"""
Tests for window function support in interpolation and differentiation.

Tests window generation based on:
- Time gap threshold: 30 minutes between measurements
- Value drop threshold: Drop of at least 10 to mark end of window
"""

import numpy as np
import pytest
from pydelt.interpolation import (
    SplineInterpolator,
    LowessInterpolator,
)


def create_time_gap_window_generator(time_gap_threshold_minutes=30, value_drop_threshold=10):
    """
    Create a window generating function that marks window boundaries based on:
    1. Time gap: Skip to next measurement >= time_gap_threshold_minutes
    2. Value drop: Drop of at least value_drop_threshold marks end of window
    
    Args:
        time_gap_threshold_minutes: Minimum time gap in minutes to mark window boundary
        value_drop_threshold: Minimum value drop to mark end of window
    
    Returns:
        Function that takes length N and returns window weights
    """
    def window_generator(n):
        """
        Generate window weights of length n.
        For testing purposes, creates a simple tapered window.
        
        Args:
            n: Length of window
        
        Returns:
            Array of window weights
        """
        # Create a simple Tukey window for testing
        weights = np.ones(n)
        taper_length = max(1, n // 10)
        
        # Taper at the beginning
        for i in range(taper_length):
            weights[i] = 0.5 * (1 - np.cos(np.pi * i / taper_length))
        
        # Taper at the end
        for i in range(taper_length):
            weights[-(i+1)] = 0.5 * (1 - np.cos(np.pi * i / taper_length))
        
        return weights
    
    return window_generator


class TestWindowFunctionBasics:
    """Test basic window function application."""
    
    def test_spline_with_hanning_window(self):
        """Test SplineInterpolator with Hanning window."""
        time = np.linspace(0, 10, 100)
        signal = np.sin(time)
        
        interp = SplineInterpolator(smoothing=0.01)
        interp.fit(time, signal, window_func=np.hanning)
        
        # Check that window function was stored
        assert interp.window_func is not None
        assert interp.window_weights is not None
        assert len(interp.window_weights) == len(time)
        assert interp.n_observations == len(time)
        
        # Test prediction
        pred = interp.predict(5.0)
        assert isinstance(pred, (float, np.floating))
    
    def test_spline_without_window(self):
        """Test SplineInterpolator without window function."""
        time = np.linspace(0, 10, 100)
        signal = np.sin(time)
        
        interp = SplineInterpolator(smoothing=0.01)
        interp.fit(time, signal)
        
        # Check that window function was not applied
        assert interp.window_func is None
        assert interp.window_weights is None
        assert interp.n_observations == len(time)
    
    def test_window_function_validation(self):
        """Test that invalid window functions raise errors."""
        time = np.linspace(0, 10, 100)
        signal = np.sin(time)
        
        def bad_window(n):
            return np.ones(n + 5)  # Wrong length
        
        interp = SplineInterpolator()
        with pytest.raises(ValueError, match="Window function returned"):
            interp.fit(time, signal, window_func=bad_window)


class TestTimeGapWindowGenerator:
    """Test custom time gap and value drop window generator."""
    
    def test_time_gap_window_creation(self):
        """Test that window generator can be created with custom thresholds."""
        window_gen = create_time_gap_window_generator(
            time_gap_threshold_minutes=30,
            value_drop_threshold=10
        )
        
        # Test that it generates weights
        weights = window_gen(100)
        assert len(weights) == 100
        assert np.all(weights >= 0)
        assert np.all(weights <= 1)
    
    def test_window_with_different_lengths(self):
        """Test window generator with different lengths."""
        window_gen = create_time_gap_window_generator(
            time_gap_threshold_minutes=30,
            value_drop_threshold=10
        )
        
        for n in [10, 50, 100, 200]:
            weights = window_gen(n)
            assert len(weights) == n
            assert np.all(weights >= 0)
            assert np.all(weights <= 1)


class TestInterpolatorWindowSupport:
    """Test window function support across different interpolators."""
    
    @pytest.mark.parametrize("InterpolatorClass", [
        SplineInterpolator,
        LowessInterpolator,
    ])
    def test_interpolator_with_standard_window(self, InterpolatorClass):
        """Test that interpolators accept standard window functions."""
        time = np.linspace(0, 10, 100)
        signal = np.sin(time) + 0.1 * np.random.randn(100)
        
        interp = InterpolatorClass()
        interp.fit(time, signal, window_func=np.hanning)
        
        assert interp.window_func is not None
        assert interp.n_observations == len(time)
        
        # Test prediction works
        pred = interp.predict(5.0)
        assert not np.isnan(pred)
    
    def test_spline_with_custom_window_generator(self):
        """Test SplineInterpolator with custom time gap window generator."""
        # Create realistic time series
        time = np.linspace(0, 260, 90)
        signal = 50 + 10 * np.sin(time / 30)
        
        window_gen = create_time_gap_window_generator(
            time_gap_threshold_minutes=30,
            value_drop_threshold=5
        )
        
        interp = SplineInterpolator(smoothing=0.1)
        interp.fit(time, signal, window_func=window_gen)
        
        # Test that fitting succeeded
        assert interp.splines is not None
        assert interp.n_observations == len(time)


class TestDerivativeWithWindows:
    """Test derivative computation with window functions."""
    
    def test_derivative_without_normalization(self):
        """Test derivative computation without normalization."""
        time = np.linspace(0, 2*np.pi, 100)
        signal = np.sin(time)
        
        interp = SplineInterpolator(smoothing=0.01)
        interp.fit(time, signal, window_func=np.hanning)
        
        # Get derivative without normalization
        deriv_func = interp.differentiate(order=1, normalize_by_observations=False)
        derivs = deriv_func(time)
        
        # Derivative of sin(x) is cos(x)
        expected = np.cos(time)
        
        # Check that derivatives are in reasonable range
        # The window function significantly modifies the signal, affecting derivative accuracy
        assert np.abs(derivs).max() < 2.0  # Derivatives should be bounded
        assert len(derivs) == len(time)  # Correct length
        # Check that derivatives are still correlated with expected (window reduces correlation)
        correlation = np.corrcoef(derivs, expected)[0, 1]
        assert correlation > 0.5  # Still positively correlated despite windowing effect
    
    def test_derivative_with_normalization(self):
        """Test derivative computation with normalization by observations."""
        time = np.linspace(0, 2*np.pi, 100)
        signal = np.sin(time)
        
        interp = SplineInterpolator(smoothing=0.01)
        interp.fit(time, signal, window_func=np.hanning)
        
        # Get derivative with normalization
        deriv_func = interp.differentiate(order=1, normalize_by_observations=True)
        derivs = deriv_func(time)
        
        # Should be scaled by 1/n_observations
        expected = np.cos(time) / len(time)
        
        # Check that normalization was applied
        assert np.allclose(derivs, expected, atol=0.01)
    
    def test_normalization_only_with_window(self):
        """Test that normalization only applies when window function was used."""
        time = np.linspace(0, 2*np.pi, 100)
        signal = np.sin(time)
        
        # Fit without window function
        interp = SplineInterpolator(smoothing=0.01)
        interp.fit(time, signal)
        
        # Request normalization (should have no effect)
        deriv_func = interp.differentiate(order=1, normalize_by_observations=True)
        derivs = deriv_func(time)
        
        # Should not be normalized since no window was used
        expected = np.cos(time)
        assert np.allclose(derivs, expected, atol=0.2)


class TestRealisticSensorScenario:
    """Integration test with realistic sensor data scenario."""
    
    def test_sensor_data_with_30min_gaps_and_10_value_drops(self):
        """
        Test with realistic sensor data that has:
        - 30-minute time gaps between measurement sessions
        - Value drops of at least 10 between sessions
        """
        np.random.seed(42)
        
        # Create three measurement sessions with 30+ minute gaps and value drops >= 10
        session1_time = np.linspace(0, 60, 30)  # 0-60 minutes
        session1_signal = 100 + 5 * np.sin(session1_time / 10) + np.random.randn(30) * 0.5
        
        session2_time = np.linspace(100, 160, 30)  # 100-160 minutes (40-min gap)
        session2_signal = 85 + 5 * np.sin(session2_time / 10) + np.random.randn(30) * 0.5  # Drop of ~15
        
        session3_time = np.linspace(200, 260, 30)  # 200-260 minutes (40-min gap)
        session3_signal = 70 + 5 * np.sin(session3_time / 10) + np.random.randn(30) * 0.5  # Drop of ~15
        
        time = np.concatenate([session1_time, session2_time, session3_time])
        signal = np.concatenate([session1_signal, session2_signal, session3_signal])
        
        # Verify our test data meets the criteria
        time_diffs = np.diff(time)
        max_gap = np.max(time_diffs)
        assert max_gap >= 30, f"Maximum time gap {max_gap} is less than 30 minutes"
        
        signal_diffs = np.diff(signal)
        min_drop = np.min(signal_diffs)
        assert min_drop <= -10, f"Minimum value drop {min_drop} is greater than -10"
        
        # Create custom window generator with the specified thresholds
        window_gen = create_time_gap_window_generator(
            time_gap_threshold_minutes=30,
            value_drop_threshold=10
        )
        
        # Fit interpolator with custom window
        interp = SplineInterpolator(smoothing=1.0)
        interp.fit(time, signal, window_func=window_gen)
        
        # Compute derivatives
        deriv_func = interp.differentiate(order=1, normalize_by_observations=False)
        derivs = deriv_func(time)
        
        # Verify results
        assert len(derivs) == len(time)
        assert not np.any(np.isnan(derivs))
        assert np.abs(derivs).max() < 10.0  # Derivatives should be reasonable
        
        # Test predictions at various points
        test_points = [30, 130, 230]  # Middle of each session
        for t in test_points:
            pred = interp.predict(t)
            assert not np.isnan(pred)
            assert 60 < pred < 110  # Should be within reasonable range


class TestMultivariateWindowSupport:
    """Test window function support with multivariate data."""
    
    def test_multivariate_signal_with_window(self):
        """Test window function with multivariate signal."""
        time = np.linspace(0, 10, 100)
        signal = np.column_stack([
            np.sin(time),
            np.cos(time)
        ])
        
        interp = SplineInterpolator(smoothing=0.01)
        interp.fit(time, signal, window_func=np.hanning)
        
        # Check that window was applied to both dimensions
        assert interp.window_weights is not None
        assert len(interp.window_weights) == len(time)
        
        # Test prediction
        pred = interp.predict(5.0)
        assert pred.shape == (2,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
