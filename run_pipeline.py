"""
run_pipeline.py — Launcher for the Career Drift Trajectory Pipeline
====================================================================

This is a lightweight entry-point wrapper.  It exists so that the
project can be started with a single, obvious command:

    python run_pipeline.py

without having to remember CLI flags.  All configurable options are
defined below in the CONFIG block and map 1-to-1 to the argparse flags
accepted by career_drift_pipeline.py.

Typical usage
-------------
  Default run (uses cache when available):
      python run_pipeline.py

  Force full recomputation (e.g. after new data arrives):
      python run_pipeline.py --no-cache

  Skip plot generation (CI / headless servers):
      python run_pipeline.py --no-plots

  Custom data path:
      python run_pipeline.py --data-dir /path/to/my/data

All arguments are forwarded transparently to the main pipeline, so
every flag documented in career_drift_pipeline.py is also valid here.

Output layout (auto-created by the pipeline):
    results/
    ├── figures/          ← all matplotlib / seaborn plots (.png)
    ├── tables/           ← all CSV outputs
    └── logs/
        └── pipeline.log  ← full run log (mirrors terminal)
    cache/                ← parquet checkpoints, .npy embeddings
    models/               ← LightGBM .joblib checkpoints
    data/                 ← input parquet files (train/validation/test)
"""

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIG — edit these defaults rather than touching the core pipeline.
# ---------------------------------------------------------------------------
CONFIG = {
    "--data-dir":   "data",     # directory with train/validation/test .parquet
    "--output-dir": "results",  # root for figures/, tables/, logs/
    "--cache-dir":  "cache",    # embedding and checkpoint cache
    "--model-dir":  "models",   # saved LightGBM .joblib files
    "--seed":       "42",       # global random seed for reproducibility
}

# Optional flags (set to True to activate)
NO_CACHE = False   # True → ignore cached embeddings / checkpoints
NO_PLOTS = False   # True → skip all matplotlib visualisation steps
# ---------------------------------------------------------------------------


def _build_argv() -> list:
    """Convert CONFIG dict + flag booleans into sys.argv-style argument list."""
    argv = []
    for flag, value in CONFIG.items():
        argv.extend([flag, str(value)])
    if NO_CACHE:
        argv.append("--no-cache")
    if NO_PLOTS:
        argv.append("--no-plots")
    return argv


def main() -> None:
    # Ensure the project root is on sys.path regardless of working directory
    project_root = Path(__file__).resolve().parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Inject our configuration as if it were passed on the command line,
    # but allow the user to override anything by appending their own flags.
    base_argv = _build_argv()
    user_argv = sys.argv[1:]   # anything after 'python run_pipeline.py'

    # User flags override CONFIG (user flags come last — argparse last-wins)
    sys.argv = [sys.argv[0]] + base_argv + user_argv

    # Import and run the pipeline
    from career_drift_pipeline_v7 import main as pipeline_main
    pipeline_main()


if __name__ == "__main__":
    main()
