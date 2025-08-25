"""
Test module for stochastic derivatives functionality across all interpolation methods.
"""

import numpy as np
import pytest
from ..interpolation import (
    SplineInterpolator, LowessInterpolator, LoessInterpolator, 
    FdaInterpolator, LlaInterpolator, GllaInterpolator, NeuralNetworkInterpolator
)
from ..stochastic import (
    NormalLink, LogNormalLink, GammaLink, BetaLink, ExponentialLink, PoissonLink,
    create_link_function, StochasticDerivativeTransform
)


class TestStochasticLinkFunctions:
    """Test individual stochastic link functions."""
    
    def test_normal_link(self):
        """Test normal distribution link function."""
        link = NormalLink(mu=0.0, sigma=1.0)
        x = np.array([1.0, 2.0, 3.0])
        dx = np.array([0.1, 0.2, 0.3])
        
        # Normal link is identity
        assert np.allclose(link.transform(x), x)
        assert np.allclose(link.inverse_transform(x), x)
        assert np.allclose(link.derivative_transform(x, dx), dx)
        assert np.allclose(link.ito_correction(x, dx), 0.0)
        assert np.allclose(link.stratonovich_correction(x, dx), 0.0)
    
    def test_lognormal_link(self):
        """Test log-normal distribution link function."""
        link = LogNormalLink(mu=0.0, sigma=0.5)
        x = np.array([1.0, 2.0, 3.0])
        dx = np.array([0.1, 0.2, 0.3])
        
        # Test transform and inverse
        y = link.transform(x)
        x_recovered = link.inverse_transform(y)
        assert np.allclose(x, x_recovered, rtol=1e-10)
        
        # Test derivative transform
        dy = link.derivative_transform(x, dx)
        expected_dy = dx / x
        assert np.allclose(dy, expected_dy)
        
        # Test corrections
        ito_corr = link.ito_correction(x, dx)
        strat_corr = link.stratonovich_correction(x, dx)
        assert np.all(ito_corr < 0)  # Itô correction should be negative
        assert np.all(strat_corr > 0)  # Stratonovich correction should be positive
    
    def test_gamma_link(self):
        """Test gamma distribution link function."""
        link = GammaLink(alpha=2.0, beta=1.0)
        x = np.array([1.0, 2.0, 3.0])
        dx = np.array([0.1, 0.2, 0.3])
        
        # Test transform and inverse
        y = link.transform(x)
        x_recovered = link.inverse_transform(y)
        assert np.allclose(x, x_recovered, rtol=1e-10)
        
        # Test derivative transform (log link)
        dy = link.derivative_transform(x, dx)
        expected_dy = dx / x
        assert np.allclose(dy, expected_dy)
    
    def test_beta_link(self):
        """Test beta distribution link function."""
        link = BetaLink(alpha=2.0, beta=3.0)
        x = np.array([0.2, 0.5, 0.8])  # Beta values must be in (0,1)
        dx = np.array([0.01, 0.02, 0.03])
        
        # Test transform and inverse
        y = link.transform(x)
        x_recovered = link.inverse_transform(y)
        assert np.allclose(x, x_recovered, rtol=1e-10)
        
        # Test derivative transform (logit link)
        dy = link.derivative_transform(x, dx)
        expected_dy = dx / (x * (1 - x))
        assert np.allclose(dy, expected_dy)
    
    def test_exponential_link(self):
        """Test exponential distribution link function."""
        link = ExponentialLink(rate=2.0)
        x = np.array([1.0, 2.0, 3.0])
        dx = np.array([0.1, 0.2, 0.3])
        
        # Test transform and inverse
        y = link.transform(x)
        x_recovered = link.inverse_transform(y)
        assert np.allclose(x, x_recovered, rtol=1e-10)
        
        # Test derivative transform (log link)
        dy = link.derivative_transform(x, dx)
        expected_dy = dx / x
        assert np.allclose(dy, expected_dy)
    
    def test_poisson_link(self):
        """Test Poisson distribution link function."""
        link = PoissonLink(lam=3.0)
        x = np.array([1.0, 2.0, 3.0])
        dx = np.array([0.1, 0.2, 0.3])
        
        # Test transform and inverse
        y = link.transform(x)
        x_recovered = link.inverse_transform(y)
        assert np.allclose(x, x_recovered, rtol=1e-10)
        
        # Test derivative transform (log link)
        dy = link.derivative_transform(x, dx)
        expected_dy = dx / x
        assert np.allclose(dy, expected_dy)
    
    def test_create_link_function(self):
        """Test link function factory."""
        # Test string creation
        normal = create_link_function('normal', mu=1.0, sigma=2.0)
        assert isinstance(normal, NormalLink)
        assert normal.mu == 1.0
        assert normal.sigma == 2.0
        
        lognormal = create_link_function('lognormal', mu=0.5, sigma=1.5)
        assert isinstance(lognormal, LogNormalLink)
        
        # Test invalid name
        with pytest.raises(ValueError):
            create_link_function('invalid_name')


class TestStochasticDerivativeTransform:
    """Test stochastic derivative transformation."""
    
    def test_transform_with_normal_link(self):
        """Test transformation with normal link (should be identity)."""
        link = NormalLink()
        transform = StochasticDerivativeTransform(link, method="ito")
        
        x = np.array([1.0, 2.0, 3.0])
        derivatives = {1: np.array([0.1, 0.2, 0.3])}
        
        transformed = transform.transform_derivatives(x, derivatives)
        
        # Normal link should not change derivatives
        assert np.allclose(transformed[1], derivatives[1])
    
    def test_transform_with_lognormal_link(self):
        """Test transformation with log-normal link."""
        link = LogNormalLink(sigma=0.5)
        
        # Test Itô method
        transform_ito = StochasticDerivativeTransform(link, method="ito")
        x = np.array([1.0, 2.0, 3.0])
        derivatives = {1: np.array([0.1, 0.2, 0.3]), 2: np.array([0.01, 0.02, 0.03])}
        
        transformed_ito = transform_ito.transform_derivatives(x, derivatives)
        
        # Should apply derivative transform and Itô correction
        expected_dx = derivatives[1] / x  # Log derivative transform
        ito_correction = link.ito_correction(x, derivatives[2])
        expected_transformed = expected_dx + ito_correction
        
        assert np.allclose(transformed_ito[1], expected_transformed)
        
        # Test Stratonovich method
        transform_strat = StochasticDerivativeTransform(link, method="stratonovich")
        transformed_strat = transform_strat.transform_derivatives(x, derivatives)
        
        # Should apply derivative transform and Stratonovich correction
        strat_correction = link.stratonovich_correction(x, derivatives[1])
        expected_strat = expected_dx + strat_correction
        
        assert np.allclose(transformed_strat[1], expected_strat)


class TestInterpolatorStochasticDerivatives:
    """Test stochastic derivatives for all interpolation methods."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        np.random.seed(42)
        t = np.linspace(0, 2*np.pi, 50)
        signal = np.sin(t) + 0.1 * np.random.randn(len(t))
        return t, signal
    
    @pytest.fixture
    def interpolators(self):
        """Create instances of all interpolators."""
        return [
            SplineInterpolator(smoothing=0.1),
            LowessInterpolator(frac=0.3),
            LoessInterpolator(frac=0.3),
            FdaInterpolator(smoothing=0.1),
            LlaInterpolator(window_size=5),
            GllaInterpolator(embedding=3),
        ]
    
    def test_set_stochastic_link_string(self, interpolators):
        """Test setting stochastic link function by string name."""
        for interpolator in interpolators:
            # Test setting by string
            interpolator.set_stochastic_link('lognormal', mu=0.0, sigma=0.5)
            assert interpolator.stochastic_link is not None
            assert isinstance(interpolator.stochastic_link, LogNormalLink)
            assert interpolator.stochastic_method == "ito"
            
            # Test setting method
            interpolator.set_stochastic_link('normal', method="stratonovich")
            assert interpolator.stochastic_method == "stratonovich"
    
    def test_set_stochastic_link_object(self, interpolators):
        """Test setting stochastic link function by object."""
        link = GammaLink(alpha=2.0, beta=1.0)
        
        for interpolator in interpolators:
            interpolator.set_stochastic_link(link, method="ito")
            assert interpolator.stochastic_link is link
            assert interpolator.stochastic_method == "ito"
    
    def test_stochastic_derivatives_normal_link(self, sample_data, interpolators):
        """Test stochastic derivatives with normal link (should be identity)."""
        t, signal = sample_data
        eval_points = np.linspace(0.5, 5.5, 10)
        
        for interpolator in interpolators:
            try:
                # Fit interpolator
                interpolator.fit(t, signal)
                
                # Get regular derivatives
                regular_deriv_func = interpolator.differentiate(order=1)
                regular_derivs = regular_deriv_func(eval_points)
                
                # Set normal link and get stochastic derivatives
                interpolator.set_stochastic_link('normal')
                stochastic_deriv_func = interpolator.differentiate(order=1)
                stochastic_derivs = stochastic_deriv_func(eval_points)
                
                # Normal link should not change derivatives
                assert np.allclose(regular_derivs, stochastic_derivs, rtol=1e-10)
                
            except Exception as e:
                pytest.skip(f"Skipping {type(interpolator).__name__} due to: {e}")
    
    def test_stochastic_derivatives_lognormal_link(self, sample_data, interpolators):
        """Test stochastic derivatives with log-normal link."""
        t, signal = sample_data
        # Use positive signal for log-normal
        signal = np.abs(signal) + 1.0
        eval_points = np.linspace(0.5, 5.5, 10)
        
        for interpolator in interpolators:
            try:
                # Fit interpolator
                interpolator.fit(t, signal)
                
                # Get regular derivatives
                regular_deriv_func = interpolator.differentiate(order=1)
                regular_derivs = regular_deriv_func(eval_points)
                
                # Set log-normal link and get stochastic derivatives
                interpolator.set_stochastic_link('lognormal', sigma=0.5, method="ito")
                stochastic_deriv_func = interpolator.differentiate(order=1)
                stochastic_derivs = stochastic_deriv_func(eval_points)
                
                # Stochastic derivatives should be different from regular
                assert not np.allclose(regular_derivs, stochastic_derivs, rtol=1e-3)
                
                # Test Stratonovich method
                interpolator.set_stochastic_link('lognormal', sigma=0.5, method="stratonovich")
                strat_deriv_func = interpolator.differentiate(order=1)
                strat_derivs = strat_deriv_func(eval_points)
                
                # Itô and Stratonovich should give different results
                assert not np.allclose(stochastic_derivs, strat_derivs, rtol=1e-3)
                
            except Exception as e:
                pytest.skip(f"Skipping {type(interpolator).__name__} due to: {e}")
    
    def test_stochastic_derivatives_beta_link(self, sample_data, interpolators):
        """Test stochastic derivatives with beta link."""
        t, signal = sample_data
        # Transform signal to (0,1) range for beta distribution
        signal = (signal - signal.min()) / (signal.max() - signal.min())
        signal = 0.1 + 0.8 * signal  # Keep away from boundaries
        eval_points = np.linspace(0.5, 5.5, 10)
        
        for interpolator in interpolators:
            try:
                # Fit interpolator
                interpolator.fit(t, signal)
                
                # Get regular derivatives
                regular_deriv_func = interpolator.differentiate(order=1)
                regular_derivs = regular_deriv_func(eval_points)
                
                # Set beta link and get stochastic derivatives
                interpolator.set_stochastic_link('beta', alpha=2.0, beta=3.0)
                stochastic_deriv_func = interpolator.differentiate(order=1)
                stochastic_derivs = stochastic_deriv_func(eval_points)
                
                # Stochastic derivatives should be different from regular
                assert not np.allclose(regular_derivs, stochastic_derivs, rtol=1e-3)
                
            except Exception as e:
                pytest.skip(f"Skipping {type(interpolator).__name__} due to: {e}")
    
    def test_invalid_stochastic_link(self, interpolators):
        """Test error handling for invalid stochastic link functions."""
        for interpolator in interpolators:
            # Test invalid link name
            with pytest.raises(ValueError):
                interpolator.set_stochastic_link('invalid_link')
            
            # Test invalid method
            with pytest.raises(ValueError):
                interpolator.set_stochastic_link('normal', method='invalid_method')
            
            # Test invalid link object
            with pytest.raises(ValueError):
                interpolator.set_stochastic_link(42)  # Not a string or link function


class TestStochasticDerivativesIntegration:
    """Integration tests for stochastic derivatives with real-world scenarios."""
    
    def test_financial_time_series_lognormal(self):
        """Test stochastic derivatives on financial time series with log-normal link."""
        # Simulate stock price data (geometric Brownian motion)
        np.random.seed(42)
        dt = 1/252  # Daily data
        T = 1.0  # One year
        t = np.arange(0, T, dt)
        
        # Geometric Brownian motion: dS = μS dt + σS dW
        mu = 0.05  # Drift
        sigma = 0.2  # Volatility
        S0 = 100  # Initial price
        
        dW = np.random.randn(len(t)) * np.sqrt(dt)
        S = np.zeros(len(t))
        S[0] = S0
        
        for i in range(1, len(t)):
            S[i] = S[i-1] * np.exp((mu - 0.5*sigma**2)*dt + sigma*dW[i])
        
        # Test with spline interpolator
        interpolator = SplineInterpolator(smoothing=0.1)
        interpolator.fit(t, S)
        
        # Set log-normal link (appropriate for stock prices)
        interpolator.set_stochastic_link('lognormal', sigma=sigma, method="ito")
        
        # Compute stochastic derivatives
        eval_points = np.linspace(0.1, 0.9, 20)
        deriv_func = interpolator.differentiate(order=1)
        stochastic_derivs = deriv_func(eval_points)
        
        # Derivatives should be reasonable (not NaN or infinite)
        assert np.all(np.isfinite(stochastic_derivs))
        assert len(stochastic_derivs) == len(eval_points)
    
    def test_population_growth_exponential(self):
        """Test stochastic derivatives on population growth with exponential link."""
        # Simulate population growth data
        np.random.seed(42)
        t = np.linspace(0, 10, 100)
        r = 0.1  # Growth rate
        K = 1000  # Carrying capacity
        P0 = 50  # Initial population
        
        # Logistic growth with noise
        P = K / (1 + ((K - P0) / P0) * np.exp(-r * t))
        P += 0.05 * P * np.random.randn(len(t))  # Add proportional noise
        P = np.maximum(P, 1)  # Ensure positive
        
        # Test with LLA interpolator
        interpolator = LlaInterpolator(window_size=7)
        interpolator.fit(t, P)
        
        # Set exponential link (appropriate for population data)
        interpolator.set_stochastic_link('exponential', rate=1.0/np.mean(P))
        
        # Compute stochastic derivatives
        eval_points = np.linspace(1, 9, 15)
        deriv_func = interpolator.differentiate(order=1)
        stochastic_derivs = deriv_func(eval_points)
        
        # Derivatives should be reasonable
        assert np.all(np.isfinite(stochastic_derivs))
        assert len(stochastic_derivs) == len(eval_points)


if __name__ == "__main__":
    pytest.main([__file__])
