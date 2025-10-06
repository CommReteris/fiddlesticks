#!/usr/bin/env python3
"""
Script to generate a table of test failures by Kornia module from pytest results.
"""

import subprocess
import sys
from collections import defaultdict

def run_pytest():
    """Run pytest and capture the output."""
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/unit/test_kornia_integration.py",
        "--tb=no", "-q"
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd="/home/rengo/fiddlesticks")
        return result.stdout, result.stderr, result.returncode
    except Exception as e:
        print(f"Error running pytest: {e}")
        return "", str(e), 1

def parse_pytest_output(output):
    """Parse pytest output to count failures by module."""
    lines = output.split('\n')
    module_failures = defaultdict(lambda: {"module": 0, "function": 0, "equivalence": 0})

    for line in lines:
        if line.startswith('FAILED'):
            # Extract test name
            parts = line.split()
            if len(parts) >= 2:
                test_name = parts[1]

                if "[" in test_name and "]" in test_name:
                    param = test_name.split("[")[1].split("]")[0]

                    if ":" in param:
                        # Format: module:function
                        module = param.split(":")[0]
                    else:
                        # Format: module (for module coverage tests)
                        module = param

                    # Determine test type
                    if "test_kornia_module_wrapper_coverage" in test_name:
                        module_failures[module]["module"] += 1
                    elif "test_kornia_module_function_covers_actual_kornia_modules" in test_name:
                        module_failures[module]["function"] += 1
                    elif "test_kornia_operation_equivalence" in test_name:
                        module_failures[module]["equivalence"] += 1

    return dict(module_failures)

def print_table(failures):
    """Print the failure table."""
    if not failures:
        print("No failure data found.")
        return

    # Header
    print("| Module | Module Coverage | Function Coverage | Equivalence | Total |")
    print("|--------|----------------|-------------------|-------------|-------|")

    total_module = 0
    total_function = 0
    total_equivalence = 0
    total_overall = 0

    for module in sorted(failures.keys()):
        data = failures[module]
        module_count = data["module"]
        function_count = data["function"]
        equivalence_count = data["equivalence"]
        total = module_count + function_count + equivalence_count

        print(f"| {module} | {module_count} | {function_count} | {equivalence_count} | {total} |")

        total_module += module_count
        total_function += function_count
        total_equivalence += equivalence_count
        total_overall += total

    # Total row
    print(f"| **Total** | **{total_module}** | **{total_function}** | **{total_equivalence}** | **{total_overall}** |")

def main():
    print("Running pytest to collect test results...")
    stdout, stderr, returncode = run_pytest()

    if returncode not in (0, 1):  # 0 = all pass, 1 = some fail
        print(f"Pytest failed with return code {returncode}")
        print("STDERR:", stderr)
        return

    print("Parsing results...")
    failures = parse_pytest_output(stdout)

    print("\nFailure counts by Kornia module:")
    print_table(failures)

if __name__ == "__main__":
    main()