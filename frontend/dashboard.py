"""
frontend/dashboard.py — Standalone Frontend Entry Point
Purpose : Thin launcher that re-exports src/app.py so the dashboard can be
          started from EITHER location:
          From project root:
              streamlit run frontend/dashboard.py
          OR:
              streamlit run src/app.py
          Both commands produce an identical result.
"""

import os
import sys
import runpy

# Resolve project root (one level up from this file)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Make src/ importable
src_path = os.path.join(ROOT, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Change working directory to project root so all relative paths (models/, data/) work
os.chdir(ROOT)

# Run the main Streamlit application
app_module = os.path.join(src_path, "app.py")
runpy.run_path(app_module, run_name="__main__")
