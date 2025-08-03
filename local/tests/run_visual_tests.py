"""
Run all visual tests for the pydelt package.
This script will generate HTML plots in the local/output directory.
"""

import os
import sys
import importlib.util
import traceback
from pathlib import Path
from typing import List, Tuple, Callable

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Ensure output directory exists
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def import_module_from_file(module_name: str, file_path: Path):
    """Import a module from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def run_test_safely(test_name: str, test_func: Callable) -> bool:
    """Run a test function safely with error handling."""
    try:
        print(f"  Running {test_name}...")
        test_func()
        print(f"  ✅ {test_name} completed successfully")
        return True
    except Exception as e:
        print(f"  ❌ {test_name} failed: {str(e)}")
        print(f"     Error details: {traceback.format_exc().splitlines()[-1]}")
        return False

def run_test_group(group_name: str, tests: List[Tuple[str, Callable]]) -> Tuple[int, int]:
    """Run a group of tests and return (passed, total) counts."""
    print(f"\n{'-' * 60}")
    print(f"Running {group_name}")
    print(f"{'-' * 60}")
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        if run_test_safely(test_name, test_func):
            passed += 1
    
    print(f"\n{group_name} Results: {passed}/{total} tests passed")
    return passed, total

def main():
    """Run all visual tests."""
    print("=" * 80)
    print("PYDELT VISUAL TEST SUITE")
    print("=" * 80)
    print(f"Output directory: {OUTPUT_DIR}")
    
    tests_dir = Path(__file__).parent
    total_passed = 0
    total_tests = 0
    
    # Test group 1: Core Derivatives
    try:
        derivatives_tests = import_module_from_file(
            "visual_test_derivatives", 
            tests_dir / "visual_test_derivatives.py"
        )
        
        derivative_test_list = [
            ("LLA Sine Test", derivatives_tests.visual_test_lla_sine),
            ("GOLD Sine Test", derivatives_tests.visual_test_gold_sine),
            ("GLLA Sine Test", derivatives_tests.visual_test_glla_sine),
            ("FDA Sine Test", derivatives_tests.visual_test_fda_sine),
            ("Algorithm Comparison", derivatives_tests.visual_test_algorithm_comparison),
            ("Noise Comparison", derivatives_tests.visual_test_noise_comparison),
            ("Window Size Comparison", derivatives_tests.visual_test_window_size_comparison),
        ]
        
        passed, total = run_test_group("Core Derivatives Tests", derivative_test_list)
        total_passed += passed
        total_tests += total
        
    except Exception as e:
        print(f"❌ Failed to load derivatives tests: {e}")
    
    # Test group 2: Integrals
    try:
        integrals_tests = import_module_from_file(
            "visual_test_integrals", 
            tests_dir / "visual_test_integrals.py"
        )
        
        integral_test_list = [
            ("Integrate Constant Derivative", integrals_tests.visual_test_integrate_constant_derivative),
            ("Integrate Sine", integrals_tests.visual_test_integrate_sine),
            ("Integrate with Initial Value", integrals_tests.visual_test_integrate_with_initial_value),
            ("Integrate with Error", integrals_tests.visual_test_integrate_with_error),
            ("Noise Effect on Integration", integrals_tests.visual_test_noise_effect_on_integration),
            ("Derivative Reconstruction with Noise", integrals_tests.visual_test_derivative_reconstruction_with_noise),
        ]
        
        passed, total = run_test_group("Integration Tests", integral_test_list)
        total_passed += passed
        total_tests += total
        
    except Exception as e:
        print(f"❌ Failed to load integrals tests: {e}")
    
    # Test group 3: Interpolation
    try:
        interpolation_tests = import_module_from_file(
            "test_visual_interpolation",
            tests_dir / "test_visual_interpolation.py"
        )
        
        interpolation_test_list = [
            ("Derivative-based Interpolation", interpolation_tests.test_derivative_based_interpolation_visual),
            ("Neural Network Interpolation", interpolation_tests.test_neural_network_interpolation_visual),
            ("Classical Interpolation", interpolation_tests.test_classical_interpolation_visual),
            ("Neural Network Derivative", interpolation_tests.test_neural_network_derivative_visual),
        ]
        
        # Add optional tests if they exist
        optional_tests = [
            ("Regular Interpolation", "test_regular_interpolation_visual"),
            ("Combined Methods Comparison", "test_combined_methods_comparison"),
            ("Input Types Test", "visual_test_input_types"),
        ]
        
        for test_name, test_attr in optional_tests:
            if hasattr(interpolation_tests, test_attr):
                interpolation_test_list.append((test_name, getattr(interpolation_tests, test_attr)))
        
        passed, total = run_test_group("Interpolation Tests", interpolation_test_list)
        total_passed += passed
        total_tests += total
        
    except Exception as e:
        print(f"❌ Failed to load interpolation tests: {e}")
    
    # Test group 4: Autodiff
    try:
        autodiff_tests = import_module_from_file(
            "test_visual_autodiff",
            tests_dir / "test_visual_autodiff.py"
        )
        
        autodiff_test_list = []
        
        # Add optional autodiff tests if they exist
        optional_autodiff_tests = [
            ("Regular Interpolation (Autodiff)", "test_regular_interpolation_visual"),
            ("Neural Network Derivative (Autodiff)", "test_neural_network_derivative_visual"),
            ("Combined Methods Comparison (Autodiff)", "test_combined_methods_comparison"),
        ]
        
        for test_name, test_attr in optional_autodiff_tests:
            if hasattr(autodiff_tests, test_attr):
                autodiff_test_list.append((test_name, getattr(autodiff_tests, test_attr)))
        
        if autodiff_test_list:
            passed, total = run_test_group("Autodiff Tests", autodiff_test_list)
            total_passed += passed
            total_tests += total
        else:
            print("\n❌ No autodiff tests found")
        
    except Exception as e:
        print(f"❌ Failed to load autodiff tests: {e}")
    
    # Test group 5: Universal Differentiation
    try:
        universal_tests = import_module_from_file(
            "test_universal_differentiation_visual",
            tests_dir / "test_universal_differentiation_visual.py"
        )
        
        universal_test_list = []
        
        # Find all test functions in the universal differentiation module
        for attr_name in dir(universal_tests):
            if attr_name.startswith('test_') and callable(getattr(universal_tests, attr_name)):
                test_func = getattr(universal_tests, attr_name)
                # Convert function name to readable test name
                readable_name = attr_name.replace('test_', '').replace('_', ' ').title()
                universal_test_list.append((readable_name, test_func))
        
        if universal_test_list:
            passed, total = run_test_group("Universal Differentiation Tests", universal_test_list)
            total_passed += passed
            total_tests += total
        else:
            print("\n❌ No universal differentiation tests found")
        
    except Exception as e:
        print(f"❌ Failed to load universal differentiation tests: {e}")
    
    # Test group 6: Multivariate Derivatives
    try:
        multivariate_tests = import_module_from_file(
            "test_multivariate_visual",
            tests_dir / "test_multivariate_visual.py"
        )
        
        multivariate_test_list = [
            ("2D Scalar Gradient Surface", multivariate_tests.test_2d_scalar_gradient_surface),
            ("2D Vector Field Jacobian", multivariate_tests.test_2d_vector_field_jacobian),
            ("3D Scalar Derivatives", multivariate_tests.test_3d_scalar_derivatives),
            ("Interpolator Comparison", multivariate_tests.test_interpolator_comparison),
        ]
        
        passed, total = run_test_group("Multivariate Derivatives Tests", multivariate_test_list)
        total_passed += passed
        total_tests += total
        
    except Exception as e:
        print(f"❌ Failed to load multivariate tests: {e}")
    
    # Final summary
    print("\n" + "=" * 80)
    print("VISUAL TEST SUITE SUMMARY")
    print("=" * 80)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {total_passed}")
    print(f"Failed: {total_tests - total_passed}")
    print(f"Success Rate: {(total_passed/total_tests*100):.1f}%" if total_tests > 0 else "No tests run")
    print(f"\nHTML files saved to: {OUTPUT_DIR}")
    print("=" * 80)

if __name__ == "__main__":
    main()
