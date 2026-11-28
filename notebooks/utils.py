"""Shared setup helpers for project notebooks."""

from pathlib import Path
import sys


def setup_notebook_imports():
    """Add the project and notebook folders to Python's import path."""
    notebook_folder = Path(__file__).resolve().parent
    project_root = notebook_folder.parent
    for folder in (notebook_folder, project_root):
        if str(folder) not in sys.path:
            sys.path.insert(0, str(folder))
    return project_root
