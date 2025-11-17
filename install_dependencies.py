#!/usr/bin/env python3
"""
Install solar_regatta package before running the Space app.
This ensures all modules are importable in the Hugging Face Space.
"""

import subprocess
import sys
from pathlib import Path

def install_package():
    """Install the solar_regatta package from the current directory."""
    current_dir = Path.cwd()

    # Check if pyproject.toml or setup.py exists
    if (current_dir / "pyproject.toml").exists() or (current_dir / "setup.py").exists():
        print("Installing solar_regatta package...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", "."],
            cwd=current_dir
        )

        if result.returncode == 0:
            print("✓ Package installed successfully")
            return True
        else:
            print("✗ Failed to install package")
            return False
    else:
        print("No pyproject.toml or setup.py found, skipping package installation")
        return True

if __name__ == "__main__":
    success = install_package()
    sys.exit(0 if success else 1)
