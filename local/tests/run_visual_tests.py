"""
Run all visual tests for the pydelt package.
This script will generate HTML plots in the local/output directory.
"""

import os
import sys
import importlib.util
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Ensure output directory exists
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def import_module_from_file(module_name, file_path):
    """Import a module from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def main():
    """Run all visual tests."""
    print("=" * 80)
    print("Running visual tests for pydelt package")
    print("=" * 80)
    
    # Import and run the visual test modules
    tests_dir = Path(__file__).parent
    
    # Run derivatives tests
    print("\nRunning derivatives visual tests...")
    derivatives_tests = import_module_from_file(
        "visual_test_derivatives", 
        tests_dir / "visual_test_derivatives.py"
    )
    derivatives_tests.visual_test_lla_sine()
    derivatives_tests.visual_test_gold_sine()
    derivatives_tests.visual_test_glla_sine()
    derivatives_tests.visual_test_fda_sine()
    derivatives_tests.visual_test_algorithm_comparison()
    
    # Run noise-related derivatives tests
    print("\nRunning noise-related derivatives tests...")
    derivatives_tests.visual_test_noise_comparison()
    derivatives_tests.visual_test_window_size_comparison()
    
    # Run integrals tests
    print("\nRunning integrals visual tests...")
    integrals_tests = import_module_from_file(
        "visual_test_integrals", 
        tests_dir / "visual_test_integrals.py"
    )
    integrals_tests.visual_test_integrate_constant_derivative()
    integrals_tests.visual_test_integrate_sine()
    integrals_tests.visual_test_integrate_with_initial_value()
    integrals_tests.visual_test_integrate_with_error()

    # Run advanced interpolation visual tests
    print("\nRunning advanced interpolation visual tests...")
    interpolation_tests = import_module_from_file(
        "test_visual_interpolation",
        tests_dir / "test_visual_interpolation.py"
    )
    interpolation_tests.test_derivative_based_interpolation_visual()
    interpolation_tests.test_neural_network_interpolation_visual()
    interpolation_tests.test_classical_interpolation_visual()
    interpolation_tests.test_neural_network_derivative_visual()
    if hasattr(interpolation_tests, "test_regular_interpolation_visual"):
        interpolation_tests.test_regular_interpolation_visual()
    if hasattr(interpolation_tests, "test_combined_methods_comparison"):
        interpolation_tests.test_combined_methods_comparison()
    if hasattr(interpolation_tests, "visual_test_input_types"):
        interpolation_tests.visual_test_input_types()

    # Run noise-related integrals tests
    print("\nRunning noise-related integrals tests...")
    integrals_tests.visual_test_noise_effect_on_integration()
    integrals_tests.visual_test_derivative_reconstruction_with_noise()

    # Run new visual interpolation tests
    print("\nRunning visual interpolation tests (test_visual_interpolation.py)...")
    interp_tests = import_module_from_file(
        "test_visual_interpolation",
        tests_dir / "test_visual_interpolation.py"
    )
    interp_tests.test_derivative_based_interpolation_visual()
    interp_tests.test_neural_network_interpolation_visual()
    interp_tests.test_classical_interpolation_visual()
    interp_tests.test_neural_network_derivative_visual()

    # Run new visual autodiff/interpolation tests
    print("\nRunning visual autodiff/interpolation tests (test_visual_autodiff.py)...")
    autodiff_tests = import_module_from_file(
        "test_visual_autodiff",
        tests_dir / "test_visual_autodiff.py"
    )
    # Only call if the function exists (for robustness)
    if hasattr(autodiff_tests, "test_regular_interpolation_visual"):
        autodiff_tests.test_regular_interpolation_visual()
    if hasattr(autodiff_tests, "test_neural_network_derivative_visual"):
        autodiff_tests.test_neural_network_derivative_visual()
    if hasattr(autodiff_tests, "test_combined_methods_comparison"):
        autodiff_tests.test_combined_methods_comparison()

    print("\n" + "=" * 80)
    print(f"All visual tests completed! HTML files saved to {OUTPUT_DIR}")
    print("=" * 80)

if __name__ == "__main__":
    main()
