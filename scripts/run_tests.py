#!/usr/bin/env python3
"""Test runner script for Area Occupancy Detection integration."""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], description: str) -> int:
    """Run a command and return its exit code."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    return result.returncode


def main() -> int:
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="Run tests for Area Occupancy Detection")
    parser.add_argument(
        "--coverage-only", 
        action="store_true", 
        help="Only run coverage analysis without tests"
    )
    parser.add_argument(
        "--unit-only", 
        action="store_true", 
        help="Only run unit tests (skip integration tests)"
    )
    parser.add_argument(
        "--integration-only", 
        action="store_true", 
        help="Only run integration tests"
    )
    parser.add_argument(
        "--module", 
        type=str, 
        help="Run tests for specific module (e.g., utils, decay)"
    )
    parser.add_argument(
        "--fast", 
        action="store_true", 
        help="Run fast tests only (exclude slow marker)"
    )
    parser.add_argument(
        "--verbose", 
        "-v", 
        action="store_true", 
        help="Verbose output"
    )
    parser.add_argument(
        "--debug", 
        action="store_true", 
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    # Base pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Add coverage options if not coverage-only
    if not args.coverage_only:
        cmd.extend([
            "--cov=custom_components/area_occupancy",
            "--cov-report=term-missing:skip-covered",
            "--cov-report=xml:coverage.xml",
            "--cov-report=html:htmlcov",
            "--cov-fail-under=85"
        ])
    
    # Add verbosity
    if args.verbose:
        cmd.append("-vv")
    else:
        cmd.append("-v")
    
    # Add logging
    if args.debug:
        cmd.extend(["--log-cli", "--log-cli-level=DEBUG"])
    
    # Filter by test type
    if args.unit_only:
        cmd.extend(["-m", "unit"])
    elif args.integration_only:
        cmd.extend(["-m", "integration"])
    
    # Filter by speed
    if args.fast:
        cmd.extend(["-m", "not slow"])
    
    # Filter by module
    if args.module:
        cmd.append(f"tests/test_{args.module}.py")
        if args.module.startswith("data_"):
            cmd[-1] = f"tests/test_{args.module}.py"
    
    # Add test directory if no specific module
    if not args.module:
        cmd.append("tests/")
    
    # Run the tests
    exit_code = run_command(cmd, "Area Occupancy Detection Tests")
    
    if exit_code == 0:
        print(f"\n{'='*60}")
        print("✅ All tests passed!")
        print("📊 Coverage report generated:")
        print("   - Terminal: shown above")
        print("   - XML: coverage.xml")
        print("   - HTML: htmlcov/index.html")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print("❌ Some tests failed!")
        print(f"Exit code: {exit_code}")
        print(f"{'='*60}")
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main()) 