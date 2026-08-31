# conftest.py
# -----------
# Adds the project root to sys.path so that pytest can import `src`
# regardless of the working directory from which pytest is invoked.

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
