"""
FastAPI Application Entrypoint for Vercel
This file serves as the entrypoint that Vercel can discover.
It imports the actual FastAPI app from backend/app/main.py
"""

import sys
from pathlib import Path

# Add the parent directory to Python path to enable imports
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

# Import the FastAPI app instance from the actual application
from backend.app.main import app

# Export the app for Vercel
__all__ = ["app"]
