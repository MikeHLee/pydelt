#!/usr/bin/env python3
"""
Comprehensive demonstration of stochastic derivatives functionality in pydelt.

This script demonstrates the new stochastic derivatives feature that applies
probability distribution link functions to transform derivative outputs using
both Itô's lemma and Stratonovich integral approaches.
"""

import numpy as np
import matplotlib.pyplot as plt
from src.pydelt.interpolation import (
    SplineInterpolator, LoessInterpolator, FdaInterpolator, 
    LlaInterpolator, GllaInterpolator
)
from src.pydelt.stochastic import create_link_function

def generate_sample_data():
    """Generate sample data for demonstration."""
    np.random.seed(42)
    t = np.linspace(0, 4*np.pi, 50)
    
    # Create different types of signals for different link functions
    signals = {
        'normal': np.sin(t) + 0.2 * np.random.randn(len(t)),
        'lognormal': np.abs(np.sin(t)) + 1.0 + 0.1 * np.random.randn(len(t)),
        'beta': 0.3 + 0.4 * (np.sin(t) + 1) / 2 + 0.05 * np.random.randn(len(t)),
        'exponential': np.abs(np.cos(t)) + 0.5 + 0.1 * np.random.randn(len(t))
    }
    
    return t, signals

def demonstrate_link_functions():
    """Demonstrate different stochastic link functions."""
    print("=" * 80)
    print("STOCHASTIC LINK FUNCTIONS DEMONSTRATION")
    print("=" * 80)
    
    # Test different link functions
    link_functions = [
        ('Normal', 'normal', {}),
        ('Log-Normal', 'lognormal', {'sigma': 0.3}),
        ('Gamma', 'gamma', {'alpha': 2.0, 'beta': 1.0}),
        ('Beta', 'beta', {'alpha': 2.0, 'beta': 3.0}),
        ('Exponential', 'exponential', {'rate': 1.0}),
        ('Poisson', 'poisson', {'lam': 2.0})
    ]
    
    x = np.array([1.0, 2.0, 3.0])
    dx = np.array([0.1, 0.2, 0.3])
    
    print(f"Input values: {x}")
    print(f"Input derivatives: {dx}")
    print("\nLink Function Transformations:")
    print("-" * 50)
    
    for name, link_name, params in link_functions:
        try:
            link = create_link_function(link_name, **params)
            
            # Transform and inverse transform
            y = link.transform(x)
            x_recovered = link.inverse_transform(y)
            
            # Derivative transform
            dy = link.derivative_transform(x, dx)
            
            # Corrections
            ito_corr = link.ito_correction(x, dx)
            strat_corr = link.stratonovich_correction(x, dx)
            
            print(f"{name:12} | Transform: {np.mean(y):.3f} | Deriv: {np.mean(dy):.3f} | "
                  f"Itô: {np.mean(ito_corr):.3f} | Strat: {np.mean(strat_corr):.3f}")
            
        except Exception as e:
            print(f"{name:12} | Error: {e}")

def demonstrate_interpolator_stochastic_derivatives():
    """Demonstrate stochastic derivatives across all interpolation methods."""
    print("\n" + "=" * 80)
    print("INTERPOLATOR STOCHASTIC DERIVATIVES DEMONSTRATION")
    print("=" * 80)
    
    t, signals = generate_sample_data()
    eval_points = np.linspace(2, 10, 8)
    
    # Test interpolators
    interpolators = [
        ('Spline', SplineInterpolator(smoothing=0.1)),
        ('LOESS', LoessInterpolator(frac=0.3)),
        ('FDA', FdaInterpolator(smoothing=0.1)),
        ('LLA', LlaInterpolator(window_size=7)),
        ('GLLA', GllaInterpolator(embedding=3))
    ]
    
    # Test with log-normal signal (most interesting for stochastic derivatives)
    signal = signals['lognormal']
    
    print(f"Using log-normal signal with {len(t)} data points")
    print(f"Evaluating derivatives at {len(eval_points)} points")
    print("\nResults:")
    print("-" * 80)
    
    for name, interpolator in interpolators:
        try:
            # Fit interpolator
            interpolator.fit(t, signal)
            
            # Regular derivatives
            regular_deriv = interpolator.differentiate(order=1)
            regular_vals = regular_deriv(eval_points)
            
            # Log-normal Itô derivatives
            interpolator.set_stochastic_link('lognormal', sigma=0.4, method='ito')
            ito_deriv = interpolator.differentiate(order=1)
            ito_vals = ito_deriv(eval_points)
            
            # Log-normal Stratonovich derivatives
            interpolator.set_stochastic_link('lognormal', sigma=0.4, method='stratonovich')
            strat_deriv = interpolator.differentiate(order=1)
            strat_vals = strat_deriv(eval_points)
            
            # Calculate statistics
            reg_mean, reg_std = np.mean(regular_vals), np.std(regular_vals)
            ito_mean, ito_std = np.mean(ito_vals), np.std(ito_vals)
            strat_mean, strat_std = np.mean(strat_vals), np.std(strat_vals)
            
            # Check if transformations are active
            ito_different = not np.allclose(regular_vals, ito_vals, rtol=1e-3)
            strat_different = not np.allclose(regular_vals, strat_vals, rtol=1e-3)
            methods_different = not np.allclose(ito_vals, strat_vals, rtol=1e-3)
            
            status = "✓" if ito_different and strat_different else "⚠"
            
            print(f"{name:8} {status} | Regular: {reg_mean:+7.3f}±{reg_std:.3f} | "
                  f"Itô: {ito_mean:+7.3f}±{ito_std:.3f} | "
                  f"Strat: {strat_mean:+7.3f}±{strat_std:.3f} | "
                  f"Active: {ito_different}")
            
        except Exception as e:
            print(f"{name:8} ✗ | Error: {e}")

def demonstrate_financial_application():
    """Demonstrate stochastic derivatives in financial modeling context."""
    print("\n" + "=" * 80)
    print("FINANCIAL APPLICATION: GEOMETRIC BROWNIAN MOTION")
    print("=" * 80)
    
    # Simulate stock price data (geometric Brownian motion)
    np.random.seed(123)
    dt = 1/252  # Daily data
    T = 1.0     # One year
    t = np.arange(0, T, dt)
    
    # Parameters for geometric Brownian motion: dS = μS dt + σS dW
    mu = 0.08    # Drift (8% annual return)
    sigma = 0.25 # Volatility (25% annual)
    S0 = 100     # Initial price
    
    # Generate price path
    dW = np.random.randn(len(t)) * np.sqrt(dt)
    S = np.zeros(len(t))
    S[0] = S0
    
    for i in range(1, len(t)):
        S[i] = S[i-1] * np.exp((mu - 0.5*sigma**2)*dt + sigma*dW[i])
    
    print(f"Generated {len(t)} daily stock prices")
    print(f"Initial price: ${S0:.2f}")
    print(f"Final price: ${S[-1]:.2f}")
    print(f"Total return: {(S[-1]/S0 - 1)*100:.1f}%")
    
    # Fit spline interpolator with log-normal link
    interpolator = SplineInterpolator(smoothing=0.01)
    interpolator.fit(t, S)
    
    # Set log-normal link (appropriate for stock prices)
    interpolator.set_stochastic_link('lognormal', sigma=sigma, method='ito')
    
    # Compute derivatives (price velocities)
    eval_times = np.linspace(0.1, 0.9, 20)
    deriv_func = interpolator.differentiate(order=1)
    price_velocities = deriv_func(eval_times)
    
    print(f"\nStochastic price velocities (dS/dt):")
    print(f"Mean velocity: ${np.mean(price_velocities):.2f}/year")
    print(f"Velocity std: ${np.std(price_velocities):.2f}/year")
    print(f"Max velocity: ${np.max(price_velocities):.2f}/year")
    
    # Compare with regular derivatives
    interpolator.set_stochastic_link('normal')  # Remove stochastic link
    regular_deriv_func = interpolator.differentiate(order=1)
    regular_velocities = regular_deriv_func(eval_times)
    
    print(f"\nRegular derivatives (no stochastic correction):")
    print(f"Mean velocity: ${np.mean(regular_velocities):.2f}/year")
    print(f"Difference from stochastic: ${np.mean(price_velocities - regular_velocities):.2f}/year")

def demonstrate_multiple_link_functions():
    """Demonstrate different link functions on the same data."""
    print("\n" + "=" * 80)
    print("MULTIPLE LINK FUNCTIONS COMPARISON")
    print("=" * 80)
    
    # Generate test data
    np.random.seed(42)
    t = np.linspace(0, 2*np.pi, 30)
    signal = 2 + np.sin(t) + 0.1 * np.random.randn(len(t))  # Positive signal
    eval_points = np.linspace(1, 5, 10)
    
    # Use spline interpolator
    interpolator = SplineInterpolator(smoothing=0.1)
    interpolator.fit(t, signal)
    
    # Test different link functions
    link_configs = [
        ('Normal', 'normal', {}),
        ('Log-Normal σ=0.3', 'lognormal', {'sigma': 0.3}),
        ('Log-Normal σ=0.6', 'lognormal', {'sigma': 0.6}),
        ('Gamma α=2,β=1', 'gamma', {'alpha': 2.0, 'beta': 1.0}),
        ('Exponential λ=0.5', 'exponential', {'rate': 0.5})
    ]
    
    print("Link Function Derivative Comparison (Itô method):")
    print("-" * 60)
    
    results = {}
    for name, link_name, params in link_configs:
        try:
            interpolator.set_stochastic_link(link_name, method='ito', **params)
            deriv_func = interpolator.differentiate(order=1)
            derivatives = deriv_func(eval_points)
            
            mean_deriv = np.mean(derivatives)
            std_deriv = np.std(derivatives)
            results[name] = derivatives
            
            print(f"{name:20} | Mean: {mean_deriv:+7.3f} | Std: {std_deriv:6.3f}")
            
        except Exception as e:
            print(f"{name:20} | Error: {e}")
    
    # Show differences from normal link
    if 'Normal' in results:
        print("\nDifferences from Normal link:")
        print("-" * 40)
        normal_derivs = results['Normal']
        for name, derivs in results.items():
            if name != 'Normal':
                diff = np.mean(np.abs(derivs - normal_derivs))
                print(f"{name:20} | Mean |Δ|: {diff:.4f}")

def main():
    """Run all demonstrations."""
    print("PYDELT STOCHASTIC DERIVATIVES COMPREHENSIVE DEMONSTRATION")
    print("=" * 80)
    print("This demonstration showcases the new stochastic derivatives feature")
    print("that applies probability distribution link functions to transform")
    print("derivative outputs using Itô's lemma and Stratonovich approaches.")
    
    try:
        demonstrate_link_functions()
        demonstrate_interpolator_stochastic_derivatives()
        demonstrate_financial_application()
        demonstrate_multiple_link_functions()
        
        print("\n" + "=" * 80)
        print("🎉 STOCHASTIC DERIVATIVES DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("\nKey Features Demonstrated:")
        print("• 6 stochastic link functions (Normal, Log-Normal, Gamma, Beta, Exponential, Poisson)")
        print("• Both Itô's lemma and Stratonovich integral approaches")
        print("• Integration with all pydelt interpolation methods")
        print("• Real-world financial modeling application")
        print("• Comparative analysis across different link functions")
        
    except Exception as e:
        print(f"\n❌ Demonstration failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
