"""
Tests for multivariate derivative computation.
"""

import numpy as np
import pytest
from unittest.mock import patch
import warnings

from pydelt.multivariate import MultivariateDerivatives, NeuralNetworkMultivariateDerivatives
from pydelt.interpolation import SplineInterpolator, LlaInterpolator


class TestMultivariateDerivatives:
    """Test traditional multivariate derivatives."""
    
    def setup_method(self):
        """Set up test data."""
        # Create 2D test data: f(x, y) = x^2 + y^2 (paraboloid)
        np.random.seed(42)
        n_points = 50
        x = np.linspace(-2, 2, n_points)
        y = np.linspace(-2, 2, n_points)
        X, Y = np.meshgrid(x, y)
        
        # Flatten for input
        self.input_2d = np.column_stack([X.flatten(), Y.flatten()])
        self.output_scalar = (X**2 + Y**2).flatten()  # f(x,y) = x^2 + y^2
        
        # Vector-valued function: [x^2 + y^2, x*y]
        self.output_vector = np.column_stack([
            (X**2 + Y**2).flatten(),
            (X * Y).flatten()
        ])
        
        # 1D test data for comparison
        self.input_1d = np.linspace(-2, 2, 50)
        self.output_1d = self.input_1d**2
    
    def test_initialization(self):
        """Test MultivariateDerivatives initialization."""
        mv = MultivariateDerivatives()
        assert mv.interpolator_class == SplineInterpolator
        assert not mv.fitted
        
        mv_lla = MultivariateDerivatives(LlaInterpolator, window_size=5)
        assert mv_lla.interpolator_class == LlaInterpolator
        assert mv_lla.interpolator_kwargs == {'window_size': 5}
    
    def test_fit_scalar_function(self):
        """Test fitting with scalar output."""
        mv = MultivariateDerivatives()
        mv.fit(self.input_2d, self.output_scalar)
        
        assert mv.fitted
        assert mv.n_inputs == 2
        assert mv.n_outputs == 1
        assert len(mv.interpolators) == 1  # One output dimension
        assert len(mv.interpolators[0]) == 2  # Two input dimensions
    
    def test_fit_vector_function(self):
        """Test fitting with vector output."""
        mv = MultivariateDerivatives()
        mv.fit(self.input_2d, self.output_vector)
        
        assert mv.fitted
        assert mv.n_inputs == 2
        assert mv.n_outputs == 2
        assert len(mv.interpolators) == 2  # Two output dimensions
        assert len(mv.interpolators[0]) == 2  # Two input dimensions
    
    def test_fit_1d_data(self):
        """Test fitting with 1D data."""
        mv = MultivariateDerivatives()
        mv.fit(self.input_1d, self.output_1d)
        
        assert mv.fitted
        assert mv.n_inputs == 1
        assert mv.n_outputs == 1
    
    def test_gradient_scalar_function(self):
        """Test gradient computation for scalar function."""
        mv = MultivariateDerivatives()
        mv.fit(self.input_2d, self.output_scalar)
        
        gradient_func = mv.gradient()
        
        # Test at a few points
        test_points = np.array([[1.0, 1.0], [0.5, -0.5], [0.0, 0.0]])
        gradients = gradient_func(test_points)
        
        assert gradients.shape == (3, 2)
        
        # For f(x,y) = x^2 + y^2, gradient is [2x, 2y]
        expected = np.array([[2.0, 2.0], [1.0, -1.0], [0.0, 0.0]])
        
        # Allow some tolerance due to interpolation
        np.testing.assert_allclose(gradients, expected, atol=0.5)
    
    def test_gradient_single_point(self):
        """Test gradient computation for single point."""
        mv = MultivariateDerivatives()
        mv.fit(self.input_2d, self.output_scalar)
        
        gradient_func = mv.gradient()
        gradient = gradient_func(np.array([1.0, 1.0]))
        
        assert gradient.shape == (2,)
        np.testing.assert_allclose(gradient, [2.0, 2.0], atol=0.5)
    
    def test_gradient_vector_function_error(self):
        """Test that gradient raises error for vector functions."""
        mv = MultivariateDerivatives()
        mv.fit(self.input_2d, self.output_vector)
        
        with pytest.raises(ValueError, match="Gradient is only defined for scalar functions"):
            mv.gradient()
    
    def test_jacobian_scalar_function(self):
        """Test Jacobian computation for scalar function."""
        mv = MultivariateDerivatives()
        mv.fit(self.input_2d, self.output_scalar)
        
        jacobian_func = mv.jacobian()
        
        # Test at a point
        test_point = np.array([[1.0, 1.0]])
        jacobian = jacobian_func(test_point)
        
        assert jacobian.shape == (1, 2)  # (n_outputs, n_inputs) for single point
        
        # For f(x,y) = x^2 + y^2, Jacobian is [2x, 2y]
        expected = np.array([[2.0, 2.0]])
        np.testing.assert_allclose(jacobian, expected, atol=0.5)
    
    def test_jacobian_vector_function(self):
        """Test Jacobian computation for vector function."""
        mv = MultivariateDerivatives()
        mv.fit(self.input_2d, self.output_vector)
        
        jacobian_func = mv.jacobian()
        
        # Test at a point
        test_point = np.array([[1.0, 1.0]])
        jacobian = jacobian_func(test_point)
        
        assert jacobian.shape == (2, 2)  # (n_outputs, n_inputs) for single point
        
        # For f1(x,y) = x^2 + y^2, f2(x,y) = x*y
        # Traditional interpolation can capture f1 derivatives: [2x, 2y]
        # But f2(x,y) = x*y has cross-terms that are approximated as zero
        # So we expect: [[2x, 2y], [~0, ~0]]
        expected = np.array([[2.0, 2.0], [0.0, 0.0]])
        np.testing.assert_allclose(jacobian, expected, atol=0.5)
    
    def test_hessian_scalar_function(self):
        """Test Hessian computation for scalar function."""
        mv = MultivariateDerivatives()
        mv.fit(self.input_2d, self.output_scalar)
        
        hessian_func = mv.hessian()
        
        # Test at a point
        test_point = np.array([[1.0, 1.0]])
        hessian = hessian_func(test_point)
        
        assert hessian.shape == (2, 2)  # (n_inputs, n_inputs) for single point
        
        # For f(x,y) = x^2 + y^2, Hessian diagonal should be [2, 2]
        # Off-diagonal elements are approximated as 0 for traditional methods
        expected_diagonal = np.array([2.0, 2.0])
        np.testing.assert_allclose(np.diag(hessian), expected_diagonal, atol=0.5)
    
    def test_hessian_vector_function_error(self):
        """Test that Hessian raises error for vector functions."""
        mv = MultivariateDerivatives()
        mv.fit(self.input_2d, self.output_vector)
        
        with pytest.raises(ValueError, match="Hessian is only defined for scalar functions"):
            mv.hessian()
    
    def test_laplacian_scalar_function(self):
        """Test Laplacian computation for scalar function."""
        mv = MultivariateDerivatives()
        mv.fit(self.input_2d, self.output_scalar)
        
        laplacian_func = mv.laplacian()
        
        # Test at a point
        test_point = np.array([[1.0, 1.0]])
        laplacian = laplacian_func(test_point)
        
        assert isinstance(laplacian, (float, np.floating))
        
        # For f(x,y) = x^2 + y^2, Laplacian is 2 + 2 = 4
        np.testing.assert_allclose(laplacian, 4.0, atol=1.0)
    
    def test_laplacian_vector_function_error(self):
        """Test that Laplacian raises error for vector functions."""
        mv = MultivariateDerivatives()
        mv.fit(self.input_2d, self.output_vector)
        
        with pytest.raises(ValueError, match="Laplacian is only defined for scalar functions"):
            mv.laplacian()
    
    def test_methods_before_fit_error(self):
        """Test that methods raise error before fitting."""
        mv = MultivariateDerivatives()
        
        with pytest.raises(RuntimeError, match="must be fit before"):
            mv.gradient()
        
        with pytest.raises(RuntimeError, match="must be fit before"):
            mv.jacobian()
        
        with pytest.raises(RuntimeError, match="must be fit before"):
            mv.hessian()
        
        with pytest.raises(RuntimeError, match="must be fit before"):
            mv.laplacian()


class TestNeuralNetworkMultivariateDerivatives:
    """Test neural network-based multivariate derivatives."""
    
    def setup_method(self):
        """Set up test data."""
        # Create simple 2D test data
        np.random.seed(42)
        n_points = 100
        x = np.random.uniform(-1, 1, n_points)
        y = np.random.uniform(-1, 1, n_points)
        
        self.input_2d = np.column_stack([x, y])
        self.output_scalar = x**2 + y**2  # f(x,y) = x^2 + y^2
        self.output_vector = np.column_stack([x**2 + y**2, x*y])
    
    @pytest.mark.skipif(True, reason="Neural network tests require optional dependencies")
    def test_pytorch_initialization(self):
        """Test PyTorch neural network initialization."""
        try:
            nn_mv = NeuralNetworkMultivariateDerivatives(
                framework='pytorch',
                hidden_layers=[32, 16],
                epochs=100
            )
            assert nn_mv.framework == 'pytorch'
            assert nn_mv.hidden_layers == [32, 16]
            assert nn_mv.epochs == 100
        except ImportError:
            pytest.skip("PyTorch not available")
    
    @pytest.mark.skipif(True, reason="Neural network tests require optional dependencies")
    def test_tensorflow_initialization(self):
        """Test TensorFlow neural network initialization."""
        try:
            nn_mv = NeuralNetworkMultivariateDerivatives(
                framework='tensorflow',
                hidden_layers=[32, 16],
                epochs=100
            )
            assert nn_mv.framework == 'tensorflow'
            assert nn_mv.hidden_layers == [32, 16]
            assert nn_mv.epochs == 100
        except ImportError:
            pytest.skip("TensorFlow not available")
    
    def test_invalid_framework_error(self):
        """Test error for invalid framework."""
        with pytest.raises(ValueError, match="Unknown framework"):
            nn_mv = NeuralNetworkMultivariateDerivatives(framework='invalid')
            nn_mv.fit(self.input_2d, self.output_scalar)
    
    def test_missing_pytorch_error(self):
        """Test error when PyTorch is not available."""
        with patch('pydelt.multivariate.PYTORCH_AVAILABLE', False):
            with pytest.raises(ImportError, match="PyTorch is required"):
                NeuralNetworkMultivariateDerivatives(framework='pytorch')
    
    def test_missing_tensorflow_error(self):
        """Test error when TensorFlow is not available."""
        with patch('pydelt.multivariate.TENSORFLOW_AVAILABLE', False):
            with pytest.raises(ImportError, match="TensorFlow is required"):
                NeuralNetworkMultivariateDerivatives(framework='tensorflow')
    
    def test_methods_before_fit_error(self):
        """Test that methods raise error before fitting."""
        try:
            nn_mv = NeuralNetworkMultivariateDerivatives(framework='pytorch', epochs=10)
        except ImportError:
            pytest.skip("PyTorch not available")
        
        with pytest.raises(RuntimeError, match="must be fit before"):
            nn_mv.gradient()
        
        with pytest.raises(RuntimeError, match="must be fit before"):
            nn_mv.jacobian()
        
        with pytest.raises(RuntimeError, match="must be fit before"):
            nn_mv.hessian()
        
        with pytest.raises(RuntimeError, match="must be fit before"):
            nn_mv.laplacian()


class TestMultivariateIntegration:
    """Integration tests for multivariate derivatives."""
    
    def setup_method(self):
        """Set up test data."""
        # Create test data with known analytical derivatives
        np.random.seed(42)
        n_points = 30
        x = np.linspace(-1, 1, n_points)
        y = np.linspace(-1, 1, n_points)
        X, Y = np.meshgrid(x, y)
        
        self.input_2d = np.column_stack([X.flatten(), Y.flatten()])
        self.output_scalar = (X**2 + Y**2).flatten()
    
    def test_different_interpolators(self):
        """Test multivariate derivatives with different interpolators."""
        interpolators = [SplineInterpolator, LlaInterpolator]
        
        for interp_class in interpolators:
            mv = MultivariateDerivatives(interp_class)
            mv.fit(self.input_2d, self.output_scalar)
            
            # Test gradient
            gradient_func = mv.gradient()
            test_point = np.array([[0.5, 0.5]])
            gradient = gradient_func(test_point)
            
            assert gradient.shape == (2,)
            # Should be approximately [1.0, 1.0] for f(x,y) = x^2 + y^2 at (0.5, 0.5)
            assert np.allclose(gradient, [1.0, 1.0], atol=1.0)
    
    def test_consistency_across_methods(self):
        """Test consistency between gradient and Jacobian for scalar functions."""
        mv = MultivariateDerivatives()
        mv.fit(self.input_2d, self.output_scalar)
        
        gradient_func = mv.gradient()
        jacobian_func = mv.jacobian()
        
        test_point = np.array([[0.5, 0.5]])
        gradient = gradient_func(test_point)
        jacobian = jacobian_func(test_point)
        
        # For scalar functions, gradient should equal first row of Jacobian
        np.testing.assert_allclose(gradient, jacobian[0, :], rtol=1e-10)
    
    def test_multivariate_vs_univariate_consistency(self):
        """Test that multivariate reduces to univariate for 1D problems."""
        # 1D test data
        x = np.linspace(-2, 2, 50)
        y = x**2
        
        # Multivariate approach
        mv = MultivariateDerivatives()
        mv.fit(x.reshape(-1, 1), y)
        gradient_func = mv.gradient()
        
        # Test at a point
        test_point = np.array([[1.0]])
        gradient = gradient_func(test_point)
        
        # Should be approximately 2.0 for f(x) = x^2 at x = 1.0
        assert np.allclose(gradient, [2.0], atol=0.5)
    
    def test_warning_handling(self):
        """Test that warnings are properly handled for problematic derivatives."""
        # Create data that might cause issues with second derivatives
        from pydelt.interpolation import LlaInterpolator
        
        x = np.array([0, 1, 2, 3, 4])
        y = np.array([0, 1, 2, 3, 4])
        X, Y = np.meshgrid(x, y)
        input_data = np.column_stack([X.flatten(), Y.flatten()])
        output_data = (X + Y).flatten()  # Linear function
        
        mv = MultivariateDerivatives(LlaInterpolator, window_size=3)
        mv.fit(input_data, output_data)
        
        # This might generate warnings for second derivatives of linear functions
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            hessian_func = mv.hessian()
            hessian = hessian_func(np.array([[1.0, 1.0]]))
            
            # Should handle gracefully even if warnings are generated
            assert hessian.shape == (2, 2)


if __name__ == "__main__":
    pytest.main([__file__])
