"""Save and resume grid-search logs, scores, and cross-validation checkpoints."""

import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np


def make_run_directory(output_dir="results", resume_from=None):
    """Create a timestamped result directory or reuse a requested run directory."""
    if resume_from is None:
        run_dir = Path(output_dir) / datetime.now().strftime("grid_search_%Y%m%d_%H%M%S_%f")
    else:
        run_dir = Path(resume_from)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "figures").mkdir(exist_ok=True)
    return run_dir


def json_default(value):
    """Convert common NumPy and path values into JSON-compatible values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def make_signature(payload):
    """Create a stable signature for the settings used in a search run."""
    serialized = json.dumps(payload, sort_keys=True, default=json_default)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def load_json(path):
    """Read a JSON file when it exists."""
    path = Path(path)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def save_json(path, value):
    """Atomically write a JSON file."""
    path = Path(path)
    temporary_path = path.with_name(path.name + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as output_file:
        json.dump(value, output_file, indent=2, default=json_default)
    temporary_path.replace(path)


def save_rows_csv(path, rows):
    """Atomically write a list of flat result dictionaries as CSV."""
    if not rows:
        return
    path = Path(path)
    temporary_path = path.with_name(path.name + ".tmp")
    fieldnames = list(rows[0])
    with temporary_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(value, default=json_default)
                if isinstance(value, (dict, list, tuple))
                else value
                for key, value in row.items()
            })
    temporary_path.replace(path)


def save_npz(path, **arrays):
    """Atomically write NumPy arrays to a compressed archive."""
    path = Path(path)
    temporary_path = path.with_name(path.name + ".tmp")
    with temporary_path.open("wb") as output_file:
        np.savez_compressed(output_file, **arrays)
    temporary_path.replace(path)


def load_npz(path):
    """Load all arrays from a NumPy archive into memory."""
    with np.load(path, allow_pickle=False) as saved_arrays:
        return {name: saved_arrays[name] for name in saved_arrays.files}


def log_progress(message, log_path):
    """Print a progress message and append it to the run log."""
    print(message, flush=True)
    with Path(log_path).open("a", encoding="utf-8") as log_file:
        log_file.write(f"{datetime.now().isoformat(timespec='seconds')}  {message}\n")
