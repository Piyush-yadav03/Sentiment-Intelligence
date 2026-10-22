"""Pytest configuration and shared fixtures"""

import sys
from pathlib import Path

# Add src to Python path for all tests
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
