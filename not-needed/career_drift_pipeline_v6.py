"""
Career Drift Trajectory Analysis — Production Pipeline  v5
===========================================================
Phase 1: ISCO-08 L3 Occupation Representation Upgrade
Architecture : matched_code → ISCO L3 extraction → SBERT + LightGBM → Full Ensemble

Phase 1 upgrade rationale
--------------------------
The previous pipeline used 27 hand-crafted regex sectors (map_sectors / broad_map)
ported from an R implementation.  This compressed semantically distinct occupations
(e.g. Backend Engineer, ML Engineer, DevOps) into a single coarse bucket, destroying
transition granularity and degrading downstream prediction quality.

Phase 1 eliminates the regex weak-supervision system entirely.  The dataset's
matched_code column already contains authoritative ESCO occupation codes that map
1-to-1 with matched_label (confirmed by parquet inspection: 2,960 unique labels,
2,960 unique codes, zero multi-code labels).  The ISCO-08 Level-3 group (3-digit
prefix of the ESCO base code) is extracted algebraically:

    occupation_group = matched_code.split(".")[0][:3]

This gives 125 interpretable ISCO L3 groups with an 80.6% Markov matrix fill —
the optimal granularity level for this dataset (L2=42 is too coarse; L4=426 has
36.8% fill which is too sparse for reliable Markov estimation).

Label assignment strategy:
  * 83.8 % of rows: occupation_group assigned directly from matched_code — no model
  * 16.2 % of rows: matched_code is null or "unknown" — SBERT + LightGBM fallback
                     classifier trained on the 83.8 % ground-truth labels

The NLP classifier (SBERT + LightGBM) is now FALLBACK-ONLY.  It never touches
rows with a valid matched_code.  Training on ground-truth ISCO labels (not regex
pseudo-labels) is expected to improve NLP F1 by 8–15 pp.

All downstream stages (transition matrices, ensemble, PCA, drift, clustering,
association rules, visualisations) are unchanged — they consume esco_sector
generically and auto-scale to 125 groups.

v5 additions:
  1.  Phase 1 ISCO L3 migration (this file)
        - regex weak-supervision system fully removed
        - occupation_group assigned from matched_code (algebraic, no heuristics)
        - NLP is fallback-only for 16.2% of rows with missing/unknown codes
        - 125×125 transition matrices replace the old 27×27
        - ground-truth training signal for SBERT + LightGBM
  2.  All v4 engineering features preserved unchanged
        - unified output directory structure
        - dataset fingerprinting + cache validation
        - environment diagnostics
        - deterministic reproducibility
        - cache metadata tracking
        - graceful recovery
        - final summary export

Usage
-----
    python career_drift_pipeline.py [options]

    --data-dir          DIR   directory with train/validation/test parquet
    --output-dir        DIR   root output directory      (default: outputs)
    --cache-dir         DIR   .npy embedding caches      (default: cache)
    --model-dir         DIR   model checkpoints          (default: models)
    --seed              INT   global random seed         (default: 42)
    --embed-batch-size  INT   SBERT batch size           (default: 128)
    --embed-chunk-size  INT   rows per embedding chunk   (default: 50000)
    --confidence-thr    FLT   NLP confidence threshold for fallback rows
                              (default: 0.35 — lower than v4 because 125 classes
                               reduce max softmax scores; tune down to 0.30 if
                               >20% of fallback rows are dropped)
    --no-cache                force re-computation of all stages
    --no-plots                skip all matplotlib output

Output tree
-----------
    outputs/
    |-- figures/          all matplotlib / seaborn plots
    |-- tables/           CSV exports (rules, risk scores, distributions)
    |-- metrics/          JSON metric files + classification report
    |-- checkpoints/      intermediate parquet checkpoints
    |   |-- cleaned_data.parquet
    |   |-- labeled_data.parquet   (NEW — replaces regex_bootstrap.parquet)
    |   |-- nlp_predicted.parquet
    |   `-- transitions.parquet
    |-- logs/
    |   |-- pipeline.log
    |   `-- environment_info.json
    `-- summaries/
        `-- final_pipeline_summary.txt

Cache invalidation before first run
------------------------------------
    rm -rf cache/emb_*.npy cache/*.metadata.json
    rm -rf models/lgbm_*.joblib models/*.metadata.json
    rm -rf outputs/checkpoints/isco_l3_assigned.parquet
    rm -rf outputs/checkpoints/nlp_predicted.parquet
    # cleaned_data.parquet is still valid and can be reused
"""

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import argparse
import gc
import hashlib
import itertools
import json
import logging
import os
import platform
import random
# import re  # removed — regex occupation assignment eliminated in Phase 1
import sys
import time
import warnings
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Third-party
# ---------------------------------------------------------------------------
import joblib
import matplotlib
matplotlib.use("Agg")          # non-interactive — safe for VS Code / servers
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from lightgbm import LGBMClassifier
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def _setup_logging(log_path: Optional[Path] = None) -> logging.Logger:
    fmt      = "%(asctime)s  %(levelname)-8s  %(message)s"
    datefmt  = "%H:%M:%S"
    handlers: list = [logging.StreamHandler()]
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, mode="a", encoding="utf-8"))
    logging.basicConfig(level=logging.INFO, format=fmt, datefmt=datefmt,
                        handlers=handlers, force=True)
    return logging.getLogger("career_drift")


log = logging.getLogger("career_drift")   # re-bound in main()


# ---------------------------------------------------------------------------
# Timing context-manager
# ---------------------------------------------------------------------------
_STAGE_TIMES: Dict[str, float] = {}


@contextmanager
def timed(stage: str):
    t0 = time.perf_counter()
    log.info("▶  %s ...", stage)
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        _STAGE_TIMES[stage] = elapsed
        log.info("✓  %s  completed in %s", stage, _fmt_time(elapsed))


def _fmt_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s"


def print_timing_summary() -> None:
    log.info("")
    log.info("=" * 60)
    log.info("  STAGE TIMING SUMMARY")
    log.info("=" * 60)
    total = sum(_STAGE_TIMES.values())
    for stage, elapsed in _STAGE_TIMES.items():
        log.info("  %-40s  %s", stage, _fmt_time(elapsed))
    log.info("  %s", "-" * 55)
    log.info("  %-40s  %s", "TOTAL", _fmt_time(total))
    log.info("=" * 60)


# ===========================================================================
# PIPELINE VERSION
# ===========================================================================
PIPELINE_VERSION = "5.0"                          # Phase 1: ISCO-08 L3 upgrade
SBERT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ===========================================================================
# ISCO-08 LEVEL-3 NAME LOOKUP  (display / logging only)
# ===========================================================================
# Maps each 3-digit ISCO L3 code to a human-readable occupation group name.
# Used for plot labels and log messages; has no effect on pipeline logic.
# Source: ILO ISCO-08 Volume I.
# ---------------------------------------------------------------------------

ISCO_L3_NAMES: Dict[str, str] = {
    # Major Group 1 — Managers
    "111": "Legislators & Senior Officials",
    "112": "Managing Directors & Chief Executives",
    "121": "Business Services & Administration Managers",
    "122": "Sales, Marketing & Development Managers",
    "131": "Production & Specialised Services Managers",
    "132": "Supply Chain & Logistics Managers",
    "133": "ICT & Professional Services Managers",
    "134": "Hospitality, Retail & Other Services Managers",
    "141": "Hotel & Restaurant Managers",
    "142": "Retail & Wholesale Trade Managers",
    "143": "Other Services Managers",
    # Major Group 2 — Professionals
    "211": "Physical & Earth Science Professionals",
    "212": "Mathematicians, Actuaries & Statisticians",
    "213": "Life Science Professionals",
    "214": "Engineering Professionals (excl. Electrotechnology)",
    "215": "Electrotechnology Engineers",
    "216": "Architects, Planners, Surveyors & Designers",
    "221": "Medical Doctors",
    "222": "Nursing & Midwifery Professionals",
    "223": "Traditional & Complementary Medicine",
    "224": "Paramedical Practitioners",
    "225": "Veterinarians",
    "226": "Other Health Professionals",
    "231": "University & Higher Education Teachers",
    "232": "Vocational Education Teachers",
    "233": "Secondary Education Teachers",
    "234": "Primary School & Early Childhood Teachers",
    "235": "Other Teaching Professionals",
    "241": "Finance Professionals",
    "242": "Administration Professionals",
    "243": "Sales, Marketing & Public Relations Professionals",
    "251": "Software & Applications Developers",
    "252": "Database & Network Professionals",
    "261": "Legal Professionals",
    "262": "Librarians, Archivists & Curators",
    "263": "Social & Religious Professionals",
    "264": "Authors, Journalists & Linguists",
    "265": "Creative & Performing Arts Professionals",
    # Major Group 3 — Technicians & Associate Professionals
    "311": "Physical & Engineering Science Technicians",
    "312": "Mining, Manufacturing & Construction Supervisors",
    "313": "Process Control Technicians",
    "314": "Life Science Technicians",
    "315": "Ship & Aircraft Controllers & Technicians",
    "321": "Medical Imaging & Therapeutic Equipment Technicians",
    "322": "Medical & Pharmaceutical Technicians",
    "323": "Veterinary Technicians & Assistants",
    "324": "Opticians & Prosthetics Technicians",
    "325": "Other Health Associate Professionals",
    "331": "Financial & Mathematical Associate Professionals",
    "332": "Sales & Purchasing Agents & Brokers",
    "333": "Business Services Agents",
    "334": "Administrative & Executive Secretaries",
    "335": "Government Regulatory Associate Professionals",
    "341": "Legal, Social & Religious Associate Professionals",
    "342": "Sports & Fitness Workers",
    "343": "Artistic, Cultural & Culinary Associate Professionals",
    "351": "ICT Operations & User Support Technicians",
    "352": "Telecommunications & Broadcasting Technicians",
    # Major Group 4 — Clerical Support Workers
    "411": "General Office Clerks",
    "412": "Secretaries (General)",
    "413": "Keyboard Operators",
    "421": "Tellers, Collectors & Related Clerks",
    "422": "Client Information Workers",
    "431": "Numerical Clerks",
    "432": "Material-Recording & Transport Clerks",
    "441": "Other Clerical Support Workers",
    # Major Group 5 — Service & Sales Workers
    "511": "Travel Attendants & Travel Stewards",
    "512": "Cooks",
    "513": "Waiters & Bartenders",
    "514": "Hairdressers, Beauticians & Related Workers",
    "515": "Building & Housekeeping Supervisors",
    "516": "Other Personal Services Workers",
    "521": "Street & Market Salespersons",
    "522": "Shop Salespersons",
    "523": "Cashiers & Ticket Clerks",
    "524": "Other Sales Workers",
    "531": "Child Care Workers & Teachers' Aides",
    "532": "Personal Care Workers in Health Services",
    "541": "Protective Services Workers",
    # Major Group 6 — Skilled Agricultural, Forestry & Fishery
    "611": "Market Gardeners & Crop Growers",
    "612": "Animal Producers",
    "613": "Mixed Crop & Animal Producers",
    "621": "Forestry & Related Workers",
    "622": "Fishery Workers, Hunters & Trappers",
    "631": "Subsistence Farmers",
    # Major Group 7 — Craft & Related Trades
    "711": "Building Frame & Related Trades Workers",
    "712": "Building Finishers & Related Trades Workers",
    "713": "Painters & Building Structure Cleaners",
    "721": "Sheet & Structural Metal Workers",
    "722": "Blacksmiths, Toolmakers & Related Trades",
    "723": "Machinery Mechanics & Repairers",
    "731": "Handicraft Workers",
    "732": "Printing Trades Workers",
    "733": "Garment & Related Trades Workers",
    "741": "Electrical Equipment Installers & Repairers",
    "742": "Electronics & Telecom Installers & Repairers",
    "751": "Food Processing & Related Trades Workers",
    "752": "Wood Treaters & Cabinet-Makers",
    "753": "Garment Pattern-Makers & Cutters",
    "754": "Other Craft & Related Workers",
    # Major Group 8 — Plant & Machine Operators
    "811": "Mining & Mineral Processing Plant Operators",
    "812": "Metal Processing & Finishing Plant Operators",
    "813": "Chemical & Photographic Products Plant Operators",
    "814": "Rubber, Plastic & Paper Products Machine Operators",
    "815": "Textile, Fur & Leather Products Machine Operators",
    "816": "Food & Related Products Machine Operators",
    "817": "Wood Processing & Papermaking Plant Operators",
    "818": "Other Stationary Plant & Machine Operators",
    "821": "Assemblers",
    "831": "Locomotive Engine Drivers & Related Workers",
    "832": "Car, Van & Motorcycle Drivers",
    "833": "Heavy Truck & Bus Drivers",
    "834": "Mobile Plant Operators",
    "835": "Ships' Deck Crews & Related Workers",
    # Major Group 9 — Elementary Occupations
    "911": "Domestic, Hotel & Office Cleaners & Helpers",
    "912": "Vehicle, Window, Laundry & Other Hand Cleaners",
    "921": "Agricultural, Forestry & Fishery Labourers",
    "931": "Mining & Construction Labourers",
    "932": "Manufacturing Labourers",
    "933": "Transport & Storage Labourers",
    "941": "Food Preparation Assistants",
    "951": "Street & Related Service Workers",
    "952": "Refuse Workers",
    "961": "Other Elementary Workers",
    # Major Group 0 — Armed Forces
    "011": "Commissioned Armed Forces Officers",
    "021": "Non-Commissioned Armed Forces Officers",
    "031": "Armed Forces Occupations (Other)",
}


def _l3_to_group(l3_code: str) -> str:
    """
    Map an ISCO L3 code (3-digit string) to its ISCO major-group name.
    Used for PCA plot colouring only — 10 stable colour categories.
    Falls back to 'Other' for any unrecognised code.
    """
    l1 = str(l3_code)[:1]
    return {
        "1": "Managers",
        "2": "Professionals",
        "3": "Technicians & Associate Professionals",
        "4": "Clerical Support",
        "5": "Service & Sales",
        "6": "Skilled Agriculture",
        "7": "Craft & Related Trades",
        "8": "Plant & Machine Operators",
        "9": "Elementary Occupations",
        "0": "Armed Forces",
    }.get(l1, "Other")





# ===========================================================================
# 0.  CONFIGURATION & DIRECTORY SETUP
# ===========================================================================


# ===========================================================================
# 0.  CONFIGURATION & DIRECTORY SETUP
# ===========================================================================

def parse_args() -> dict:
    p = argparse.ArgumentParser(
        description="Career Drift Trajectory Analysis -- Production Pipeline v5 (Phase 1)"
    )
    p.add_argument("--data-dir",         default="data")
    p.add_argument("--output-dir",       default="outputs")
    p.add_argument("--cache-dir",        default="cache")
    p.add_argument("--model-dir",        default="models")
    p.add_argument("--seed",             type=int,   default=42)
    p.add_argument("--embed-batch-size", type=int,   default=128,
                   help="SBERT encode() batch size (reduce if GPU OOM)")
    p.add_argument("--embed-chunk-size", type=int,   default=50_000,
                   help="Rows per embedding chunk for large datasets")
    p.add_argument("--confidence-thr",   type=float, default=0.35,
                   help="Min NLP confidence for fallback rows (default 0.35 for 125 classes)")
    p.add_argument("--no-cache",   action="store_true")
    p.add_argument("--no-plots",   action="store_true")
    args = p.parse_args()

    od  = Path(args.output_dir)
    cfg: Dict[str, Any] = {
        "data_dir":         args.data_dir,
        "output_dir":       str(args.output_dir),
        "cache_dir":        args.cache_dir,
        "model_dir":        args.model_dir,
        "seed":             args.seed,
        "embed_batch_size": args.embed_batch_size,
        "embed_chunk_size": args.embed_chunk_size,
        "confidence_thr":   args.confidence_thr,
        "use_cache":        not args.no_cache,
        "plots":            not args.no_plots,
        # Unified sub-directories
        "figures_dir":      str(od / "figures"),
        "tables_dir":       str(od / "tables"),
        "metrics_dir":      str(od / "metrics"),
        "ckpt_dir":         str(od / "checkpoints"),
        "logs_dir":         str(od / "logs"),
        "summaries_dir":    str(od / "summaries"),
    }
    # Legacy alias so callers of cfg["plots_dir"] still work
    cfg["plots_dir"] = cfg["figures_dir"]
    return cfg


def make_dirs(cfg: dict) -> None:
    """Create the full unified output directory tree."""
    for key in (
        "output_dir", "cache_dir", "model_dir",
        "figures_dir", "tables_dir", "metrics_dir",
        "ckpt_dir", "logs_dir", "summaries_dir",
    ):
        Path(cfg[key]).mkdir(parents=True, exist_ok=True)


def save_run_config(cfg: dict, device: str) -> None:
    """Save full run configuration snapshot for reproducibility."""
    snapshot = {
        "pipeline_version":  PIPELINE_VERSION,
        "sbert_model":       SBERT_MODEL_NAME,
        "label_space":       "ISCO_L3",
        "occupation_source": "matched_code (algebraic extraction, NLP fallback for 16.2%)",
        "timestamp_utc":     datetime.utcnow().isoformat(),
        "device":            device,
        "seed":              cfg["seed"],
        "embed_batch_size":  cfg["embed_batch_size"],
        "embed_chunk_size":  cfg["embed_chunk_size"],
        "confidence_thr":    cfg["confidence_thr"],
        "use_cache":         cfg["use_cache"],
        "data_dir":          cfg["data_dir"],
        "output_dir":        cfg["output_dir"],
        "cache_dir":         cfg["cache_dir"],
        "model_dir":         cfg["model_dir"],
    }
    out = Path(cfg["metrics_dir"]) / "run_config.json"
    with open(out, "w") as fh:
        json.dump(snapshot, fh, indent=2)
    log.info("  Run config saved -> %s", out)


def seed_everything(seed: int) -> None:
    """Set all random seeds for full deterministic reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False
        log.info("  torch seeds set (manual_seed + cuda.manual_seed_all, cudnn.deterministic=True)")
    except ImportError:
        pass
    log.info("Global seed : %d", seed)


# ===========================================================================
# 0b.  DEVICE DETECTION + ENVIRONMENT DIAGNOSTICS
# ===========================================================================

def _pkg_version(pkg: str) -> str:
    try:
        import importlib.metadata
        return importlib.metadata.version(pkg)
    except Exception:
        return "unknown"


def detect_device_and_log_env(cfg: dict) -> str:
    """
    Detect compute device (CUDA / CPU) and log full environment diagnostics.

    Logs:
      - Python version, OS / platform
      - torch, CUDA, sentence-transformers, LightGBM versions
      - GPU name and VRAM when available
      - Likely causes when CUDA is unavailable

    Saves: outputs/logs/environment_info.json

    Returns 'cuda' or 'cpu'.
    """
    env: Dict[str, Any] = {
        "pipeline_version":         PIPELINE_VERSION,
        "timestamp_utc":            datetime.utcnow().isoformat(),
        "python_version":           sys.version,
        "platform":                 platform.platform(),
        "os":                       platform.system(),
        "cpu_count":                os.cpu_count(),
        "sentence_transformers":    _pkg_version("sentence-transformers"),
        "lightgbm":                 _pkg_version("lightgbm"),
        "scikit_learn":             _pkg_version("scikit-learn"),
        "numpy":                    _pkg_version("numpy"),
        "pandas":                   _pkg_version("pandas"),
    }

    log.info("=" * 60)
    log.info("  ENVIRONMENT DIAGNOSTICS")
    log.info("=" * 60)
    log.info("  Python             : %s", sys.version.split()[0])
    log.info("  Platform           : %s", platform.platform())
    log.info("  sentence-transformers : %s", env["sentence_transformers"])
    log.info("  lightgbm           : %s", env["lightgbm"])

    try:
        import torch
        torch_ver  = torch.__version__
        cuda_built = torch.version.cuda

        env["torch_version"]  = torch_ver
        env["torch_cuda_build"] = cuda_built
        log.info("  torch              : %s  (CUDA build: %s)", torch_ver, cuda_built)

        if torch.cuda.is_available():
            device   = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            cuda_ver = torch.version.cuda
            vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
            env.update({
                "device":      "cuda",
                "gpu_name":    gpu_name,
                "cuda_version": cuda_ver,
                "vram_gb":     round(vram_gb, 2),
            })
            log.info("  Device             : cuda")
            log.info("  GPU                : %s", gpu_name)
            log.info("  CUDA               : %s", cuda_ver)
            log.info("  VRAM               : %.1f GB", vram_gb)
        else:
            device = "cpu"
            env["device"] = "cpu"
            env["cuda_unavailable_reasons"] = [
                "torch.cuda.is_available() returned False",
                "Possible causes: CPU-only torch build, CUDA driver mismatch, "
                "unsupported CUDA version, or no NVIDIA GPU present.",
            ]
            log.info("  Device             : cpu")
            log.info("  CUDA unavailable -- possible causes:")
            log.info("    * CPU-only torch build (pip install torch without CUDA extras)")
            log.info("    * CUDA driver version incompatible with installed torch")
            log.info("    * No NVIDIA GPU detected in this environment")
    except ImportError:
        device = "cpu"
        env["device"] = "cpu"
        env["torch_version"] = "not installed"
        log.info("  torch              : NOT INSTALLED -- using CPU")

    log.info("=" * 60)

    out = Path(cfg["logs_dir"]) / "environment_info.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(env, fh, indent=2)
    log.info("  Environment info saved -> %s", out)
    return device



# ===========================================================================
# DATASET FINGERPRINTING + CACHE METADATA
# ===========================================================================

def compute_dataset_hash(df: pd.DataFrame) -> str:
    """
    Compute a deterministic SHA-256 fingerprint of the dataset.

    Uses the combined_text column if present (post-regex stage),
    otherwise falls back to matched_label + start_date.
    This hash gates every downstream cache so a dataset change
    automatically invalidates stale embeddings and checkpoints.
    """
    if "combined_text" in df.columns:
        raw = df["combined_text"].astype(str)
    else:
        raw = df["matched_label"].astype(str) + "|" + df["start_date"].astype(str)
    digest = hashlib.sha256(
        "|".join(raw.sort_values().values).encode("utf-8", errors="replace")
    ).hexdigest()[:16]
    return digest


def _meta_path(cache_path: Path) -> Path:
    """Return the .metadata.json sidecar path for a cache file."""
    return cache_path.with_suffix(cache_path.suffix + ".metadata.json")


def write_cache_metadata(
    cache_path: Path,
    dataset_hash: str,
    cfg: dict,
    device: str,
    extra: Optional[Dict] = None,
) -> None:
    """Write a sidecar .metadata.json next to any cache/checkpoint file."""
    meta: Dict[str, Any] = {
        "pipeline_version": PIPELINE_VERSION,
        "created_utc":      datetime.utcnow().isoformat(),
        "dataset_hash":     dataset_hash,
        "sbert_model":      SBERT_MODEL_NAME,
        "embed_batch_size": cfg.get("embed_batch_size"),
        "embed_chunk_size": cfg.get("embed_chunk_size"),
        "device":           device,
        "seed":             cfg.get("seed"),
    }
    if extra:
        meta.update(extra)
    mp = _meta_path(cache_path)
    with open(mp, "w") as fh:
        json.dump(meta, fh, indent=2)


def cache_is_valid(cache_path: Path, dataset_hash: str) -> bool:
    """
    Return True only when:
      1. cache_path exists
      2. its sidecar metadata exists
      3. the stored dataset_hash matches the current hash

    Any failure (missing file, missing metadata, hash mismatch,
    JSON parse error) returns False so the stage is safely recomputed.
    """
    if not cache_path.exists():
        return False
    mp = _meta_path(cache_path)
    if not mp.exists():
        log.info("    Cache metadata missing for %s -- will recompute", cache_path.name)
        return False
    try:
        with open(mp) as fh:
            meta = json.load(fh)
        stored_hash = meta.get("dataset_hash", "")
        if stored_hash != dataset_hash:
            log.warning(
                "    Dataset hash mismatch for %s "
                "(stored=%s current=%s) -- invalidating stale cache",
                cache_path.name, stored_hash, dataset_hash,
            )
            return False
        return True
    except Exception as exc:
        log.warning("    Metadata read error for %s: %s -- recomputing", cache_path.name, exc)
        return False



def load_data(data_dir: str) -> pd.DataFrame:
    dp = Path(data_dir)
    log.info("  Loading parquet files from %s", dp)
    frames = []
    for split in ("train", "validation", "test"):
        path = dp / f"{split}.parquet"
        if not path.exists():
            log.warning("  %s not found -- skipping", path)
            continue
        f = pd.read_parquet(path)
        frames.append(f)
        log.info("    %-14s  %s rows", split, f"{len(f):,}")
    if not frames:
        raise FileNotFoundError(f"No parquet files found in {dp}")
    df = pd.concat(frames, ignore_index=True)
    log.info("  Combined   : %s rows  x  %d columns", f"{len(df):,}", df.shape[1])
    log.info("  Columns    : %s", ", ".join(df.columns))
    return df


# ===========================================================================
# 2.  DATA CLEANING
# ===========================================================================

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    n0 = len(df)
    df = df[df["matched_label"].notna()]
    df = df[df["matched_label"].str.strip() != ""]
    df = df[df["matched_label"].str.lower() != "unknown"]
    log.info("  Dropped unknown/blank labels : %s", f"{n0 - len(df):,}")

    n1 = len(df)
    df = df.drop_duplicates(subset=["person_id", "matched_label", "start_date"])
    log.info("  Dropped duplicates           : %s", f"{n1 - len(df):,}")

    df["year"] = (
        df["start_date"].astype(str).str.extract(r"(\d{4})")[0].astype(float)
    )
    n2 = len(df)
    df = df[(df["year"] >= 1950) & (df["year"] <= 2024)].copy()
    df["year"] = df["year"].astype(int)
    log.info("  Dropped out-of-range years   : %s", f"{n2 - len(df):,}")

    df = df.sort_values(["person_id", "year"]).reset_index(drop=True)
    log.info("  Clean rows : %s", f"{len(df):,}")
    log.info("  Users      : %s", f"{df['person_id'].nunique():,}")
    log.info("  Unique jobs: %s", f"{df['matched_label'].nunique():,}")
    log.info("  Years      : %d-%d", df["year"].min(), df["year"].max())
    return df



# ===========================================================================
# 3.  ISCO-08 L3 OCCUPATION LABEL ASSIGNMENT  (matched_code → algebraic)
# ===========================================================================
#
# Phase 1 architecture — NO regex, NO heuristics, NO weak supervision.
#
# The matched_code column in the dataset is an authoritative ESCO occupation
# code that maps 1-to-1 with matched_label (2,960 unique labels, 2,960 unique
# codes, zero multi-code labels — confirmed by parquet inspection).
#
# The ISCO L3 group is extracted purely algebraically:
#
#     occupation_group = matched_code.split(".")[0][:3]
#
# Examples:
#   "3343.1"  → base "3343" → L3 "334" (Administrative & Executive Secretaries)
#   "2511.17" → base "2511" → L3 "251" (Software & Applications Developers)
#   "9333.8"  → base "9333" → L3 "933" (Transport & Storage Labourers)
#
# Rows with null or "unknown" codes (16.2% of the dataset) are flagged as
# needs_nlp=True and handled by the SBERT + LightGBM fallback classifier
# in predict_nlp_sectors().  No rows are dropped at this stage.
# ---------------------------------------------------------------------------


def _combine_text(row) -> str:
    """Combine matched_label and matched_description for SBERT input."""
    title = str(row.get("matched_label", ""))
    desc  = str(row.get("matched_description", ""))
    if desc in ("nan", "None", ""):
        desc = ""
    return f"{title} [SEP] {desc}".strip()


def assign_occupation_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign ISCO-08 Level-3 occupation groups from matched_code.

    Strategy (per the Phase 1 engineering plan):
      1. For rows with a valid ESCO code:
           occupation_group = matched_code.split(".")[0][:3]
           needs_nlp        = False
      2. For rows with null or 'unknown' code:
           occupation_group = None
           needs_nlp        = True
         These rows are handled by predict_nlp_sectors() using the
         SBERT + LightGBM classifier trained on the valid-code rows.

    Columns added:
      combined_text    -- SBERT input (title [SEP] description)
      occupation_group -- 3-digit ISCO L3 code string, or None for NLP rows
      needs_nlp        -- bool; True for the 16.2% without valid codes

    No rows are dropped here.  All rows proceed to downstream stages;
    the NLP fallback fills the 16.2% gap before transition modeling.
    """
    log.info("--- OCCUPATION LABEL ASSIGNMENT (ISCO L3 from matched_code) ---")
    df = df.copy()

    # Build combined_text for SBERT (used by NLP fallback for unknown-code rows)
    df["combined_text"] = df.apply(_combine_text, axis=1)

    # Identify rows with a valid ESCO code
    # Guard: matched_code may be NaN (missing) or the literal string "unknown"
    has_code = (
        df["matched_code"].notna() &
        (df["matched_code"].str.lower() != "unknown")
    )

    # Algebraic ISCO L3 extraction:
    #   "3343.1"  → split(".")[0] = "3343" → [:3] = "334"
    #   "2511.17" → split(".")[0] = "2511" → [:3] = "251"
    df["occupation_group"] = None
    df.loc[has_code, "occupation_group"] = (
        df.loc[has_code, "matched_code"]
        .str.split(".", n=1).str[0]   # take the ISCO base (numeric part before first dot)
        .str[:3]                       # first 3 digits = L3 group
    )

    # Flag rows that require NLP fallback
    df["needs_nlp"] = ~has_code

    n_direct  = int(has_code.sum())
    n_missing = int((~has_code).sum())
    n_total   = len(df)

    log.info("  Direct L3 from matched_code : %s (%.1f%%)",
             f"{n_direct:,}", n_direct / n_total * 100)
    log.info("  Need NLP fallback (no code) : %s (%.1f%%)",
             f"{n_missing:,}", n_missing / n_total * 100)
    log.info("  Distinct L3 groups assigned : %d",
             df["occupation_group"].nunique())

    # Log top 20 L3 groups by row count
    dist = (
        df["occupation_group"].dropna()
        .map(lambda c: f"{c} — {ISCO_L3_NAMES.get(c, 'Unknown')}")
        .value_counts()
        .head(20)
    )
    log.info("  Top 20 L3 groups by row count:\n%s", dist.to_string())

    # Sort for trajectory ordering
    df = df.sort_values(["person_id", "year"]).reset_index(drop=True)
    return df


# ===========================================================================
# 4.  NLP SECTOR CLASSIFIER  (SBERT + LightGBM — FALLBACK-ONLY)
# ===========================================================================

def _embed_chunked(
    texts: List[str],
    model: SentenceTransformer,
    cache_path: Optional[Path],
    use_cache: bool,
    label: str,
    batch_size: int,
    chunk_size: int,
    device: str,
    dataset_hash: str = "",
    cfg: Optional[dict] = None,
) -> np.ndarray:
    """
    Generate or load L2-normalised SBERT embeddings in memory-safe chunks.

    Cache contract (with dataset fingerprinting):
      cache exists + metadata hash matches + use_cache=True -> load from .npy
      otherwise -> encode in chunks, save with metadata sidecar, return

    Graceful recovery: any load failure is caught; recomputation follows.
    """
    if use_cache and cache_path:
        if cache_is_valid(cache_path, dataset_hash):
            try:
                arr = np.load(cache_path)
                log.info("    Cache hit  [%s]: %s  (%.1f MB)",
                         label, cache_path.name, arr.nbytes / 1e6)
                return arr
            except Exception as exc:
                log.warning("    Cache load failed [%s]: %s -- recomputing", label, exc)

    n        = len(texts)
    n_chunks = max(1, (n + chunk_size - 1) // chunk_size)
    log.info("    Encoding [%s] -- %s texts, %d chunk(s), batch=%d ...",
             label, f"{n:,}", n_chunks, batch_size)

    chunks: List[np.ndarray] = []
    for c_idx in range(n_chunks):
        start       = c_idx * chunk_size
        end         = min(start + chunk_size, n)
        chunk_texts = texts[start:end]
        log.info("      chunk %d/%d  rows %s-%s ...",
                 c_idx + 1, n_chunks, f"{start:,}", f"{end - 1:,}")
        emb = model.encode(
            chunk_texts,
            batch_size=batch_size,
            show_progress_bar=(n_chunks == 1),
            convert_to_numpy=True,
            normalize_embeddings=True,
            device=device,
        )
        chunks.append(emb)
        if device == "cuda":
            try:
                import torch
                torch.cuda.empty_cache()
            except ImportError:
                pass

    result = np.vstack(chunks)
    del chunks
    gc.collect()

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, result)
        write_cache_metadata(cache_path, dataset_hash, cfg or {}, device)
        log.info("    Cached  -> %s  (%.1f MB)", cache_path.name, result.nbytes / 1e6)
    return result


def train_nlp_classifier(
    df: pd.DataFrame,
    cfg: dict,
    device: str,
    dataset_hash: str,
) -> Tuple[LGBMClassifier, SentenceTransformer, dict]:
    """
    Train SBERT + LightGBM ISCO L3 classifier on ground-truth labels.

    Training data:  rows where matched_code was valid (needs_nlp == False).
                    These rows have authoritative ISCO L3 codes extracted
                    algebraically from matched_code — NOT from any regex system.
                    This covers 83.8% of the dataset (~1.26M rows).

    Training target: occupation_group — 3-digit ISCO L3 code string.
                     This is a real, consistent ground-truth label.

    The trained classifier is used ONLY as a fallback for the 16.2% of rows
    that have null/unknown matched_code (needs_nlp == True).

    LightGBM hyperparameters scaled for ~125 classes:
      num_leaves=127  (2^7-1, finer decision boundaries)
      n_estimators=600 (more rounds for larger label space)
      min_child_samples=20 (prevents overfitting on rare ISCO groups)
      class_weight='balanced' (handles unequal L3 group sizes)

    Cache policy: LightGBM checkpoint is validated against the dataset hash.
    If the hash mismatches or the file is corrupted, retraining occurs automatically.

    Returns
    -------
    classifier      : fitted LGBMClassifier (predicts ISCO L3 codes)
    embedding_model : loaded SentenceTransformer (kept for fallback prediction)
    nlp_metrics     : dict with accuracy / precision / recall / f1
    """
    seed      = cfg["seed"]
    cache_dir = Path(cfg["cache_dir"])
    model_dir = Path(cfg["model_dir"])
    use_cache = cfg["use_cache"]
    bs        = cfg["embed_batch_size"]
    cs        = cfg["embed_chunk_size"]
    ckpt_path = model_dir / "lgbm_isco_l3_classifier.joblib"

    # Use ONLY rows with ground-truth ISCO codes (needs_nlp == False)
    # These have authoritative labels from matched_code — NOT regex pseudo-labels
    labeled_mask = df["needs_nlp"] == False
    labeled_df   = df[labeled_mask].copy()
    log.info("  Training on %s ground-truth labeled rows (%.1f%% of dataset)",
             f"{len(labeled_df):,}", len(labeled_df) / len(df) * 100)
    log.info("  Distinct ISCO L3 classes in labeled data: %d",
             labeled_df["occupation_group"].nunique())

    # NLP train/val/test split — stratified on ISCO L3 codes
    # NOTE: independent of the ensemble prediction split in Section 7
    train_t, temp_t, train_l, temp_l = train_test_split(
        labeled_df["combined_text"].tolist(),
        labeled_df["occupation_group"].tolist(),   # ground-truth 3-digit ISCO L3 codes
        test_size=0.30,
        stratify=labeled_df["occupation_group"],
        random_state=seed,
    )
    val_t, test_t, val_l, test_l = train_test_split(
        temp_t, temp_l,
        test_size=0.50,
        stratify=temp_l,
        random_state=seed,
    )
    log.info("  NLP split (ground-truth ISCO L3) -- train: %s | val: %s | test: %s",
             f"{len(train_t):,}", f"{len(val_t):,}", f"{len(test_t):,}")
    log.info("  ISCO L3 classes in training set: %d", len(set(train_l)))

    log.info("  Loading SBERT: %s  (device=%s) ...", SBERT_MODEL_NAME, device)
    embedding_model = SentenceTransformer(SBERT_MODEL_NAME, device=device)

    X_train = _embed_chunked(train_t, embedding_model, cache_dir / "emb_train.npy",
                              use_cache, "train", bs, cs, device, dataset_hash, cfg)
    X_val   = _embed_chunked(val_t,   embedding_model, cache_dir / "emb_val.npy",
                              use_cache, "val",   bs, cs, device, dataset_hash, cfg)
    X_test  = _embed_chunked(test_t,  embedding_model, cache_dir / "emb_test.npy",
                              use_cache, "test",  bs, cs, device, dataset_hash, cfg)

    # LightGBM checkpoint -- validated against dataset hash, with graceful recovery
    lgbm_loaded = False
    if use_cache and cache_is_valid(ckpt_path, dataset_hash):
        try:
            classifier = joblib.load(ckpt_path)
            log.info("  LightGBM ISCO L3 checkpoint loaded: %s", ckpt_path)
            lgbm_loaded = True
        except Exception as exc:
            log.warning("  LightGBM checkpoint load failed: %s -- retraining", exc)

    if not lgbm_loaded:
        n_classes = len(set(train_l))
        log.info("  Training LightGBM on %s ground-truth ISCO L3 labels (%d classes) ...",
                 f"{len(train_t):,}", n_classes)
        # Hyperparameters scaled for ~125 classes:
        #   num_leaves=127      (2^7-1, supports finer decision boundaries)
        #   n_estimators=600    (more rounds for larger label space vs v4's 400)
        #   min_child_samples=20 (prevents overfitting on rare ISCO groups)
        #   class_weight=balanced (handles unequal L3 group sizes)
        classifier = LGBMClassifier(
            objective="multiclass",
            n_estimators=600,
            learning_rate=0.05,
            max_depth=12,
            num_leaves=127,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_samples=20,
            class_weight="balanced",
            random_state=seed,
            verbose=-1,
        )
        classifier.fit(X_train, train_l, eval_set=[(X_val, val_l)])
        joblib.dump(classifier, ckpt_path)
        write_cache_metadata(ckpt_path, dataset_hash, cfg, device,
                             extra={"n_classes":    n_classes,
                                    "n_estimators": 500,
                                    "label_space":  "ISCO_L3"})
        log.info("  Checkpoint saved -> %s", ckpt_path)

    # Evaluation
    preds    = classifier.predict(X_test)
    accuracy = accuracy_score(test_l, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_l, preds, average="weighted"
    )
    log.info("  NLP ISCO L3 classifier performance (vs ground-truth labels):")
    log.info("    Accuracy  : %.4f", accuracy)
    log.info("    Precision : %.4f", precision)
    log.info("    Recall    : %.4f", recall)
    log.info("    F1 Score  : %.4f", f1)
    report_str = classification_report(test_l, preds)
    print(report_str)

    # Export NLP metrics -> metrics/
    metrics_dir = Path(cfg["metrics_dir"])
    tables_dir  = Path(cfg["tables_dir"])
    nlp_metrics = dict(accuracy=accuracy, precision=precision, recall=recall, f1=f1)

    with open(metrics_dir / "nlp_metrics.json", "w") as fh:
        json.dump({**nlp_metrics,
                   "label_space":  "ISCO_L3",
                   "n_classes":    int(len(classifier.classes_)),
                   "dataset_hash": dataset_hash,
                   "timestamp":    datetime.utcnow().isoformat()}, fh, indent=2)

    with open(metrics_dir / "nlp_classification_report.txt", "w") as fh:
        fh.write(report_str)

    cm = confusion_matrix(test_l, preds, labels=classifier.classes_)
    pd.DataFrame(cm, index=classifier.classes_,
                 columns=classifier.classes_).to_csv(
        tables_dir / "nlp_confusion_matrix.csv"
    )
    log.info("  NLP metrics saved -> %s", metrics_dir)

    # Confusion matrix plot -- with many ISCO classes, suppress tick labels
    # to keep the figure readable; full data is in the CSV.
    if cfg.get("plots", True):
        n_cls = len(classifier.classes_)
        fs    = max(4, 7 - max(0, (n_cls - 27) // 10))  # shrink font as classes grow
        fig, ax = plt.subplots(figsize=(max(16, n_cls // 3), max(14, n_cls // 3)))
        sns.heatmap(cm, cmap="Blues", ax=ax,
                    xticklabels=(classifier.classes_ if n_cls <= 50 else False),
                    yticklabels=(classifier.classes_ if n_cls <= 50 else False))
        ax.set_title(
            f"NLP ISCO-08 L3 Classification -- Confusion Matrix\n"
            f"({n_cls} classes, vs ISCO L3 assignment labels)",
            fontsize=12, fontweight="bold",
        )
        if n_cls <= 50:
            plt.xticks(rotation=60, ha="right", fontsize=fs)
            plt.yticks(fontsize=fs)
        plt.tight_layout()
        out = Path(cfg["figures_dir"]) / "nlp_confusion_matrix.png"
        plt.savefig(out, dpi=150)
        plt.close()
        log.info("  Saved: %s", out)

    del X_train, X_val, X_test
    gc.collect()
    return classifier, embedding_model, nlp_metrics


def predict_nlp_sectors(
    df: pd.DataFrame,
    classifier: LGBMClassifier,
    embedding_model: SentenceTransformer,
    cfg: dict,
    device: str,
    dataset_hash: str,
) -> pd.DataFrame:
    """
    Apply NLP ISCO L3 classifier ONLY to rows with missing/unknown matched_code.

    Phase 1 architecture — NLP is FALLBACK-ONLY:
      * Rows with valid matched_code (needs_nlp == False, 83.8%):
          esco_sector = occupation_group (set directly from matched_code)
          — NLP is never applied to these rows
      * Rows with null/unknown matched_code (needs_nlp == True, 16.2%):
          SBERT + LightGBM inference is run
          esco_sector = NLP prediction if confidence >= conf_thr, else dropped

    This is a fundamental architectural change from v5:
      v5 (wrong) : embedded and classified ALL rows
      v6 (correct): embeds and classifies ONLY the 16.2% fallback rows

    After merging, esco_sector holds authoritative ISCO L3 codes for all rows.
    All downstream code (transitions, Markov, PCA, ensemble) is unchanged.

    Saves:
      figures/nlp_confidence_distribution.png  (fallback rows only)
      tables/sector_distribution.csv
    """
    cache_dir = Path(cfg["cache_dir"])
    bs        = cfg["embed_batch_size"]
    cs        = cfg["embed_chunk_size"]
    use_cache = cfg["use_cache"]
    conf_thr  = cfg["confidence_thr"]

    # Split into direct-code rows and fallback rows
    df_coded   = df[df["needs_nlp"] == False].copy()   # 83.8% — no NLP needed
    df_fallback = df[df["needs_nlp"] == True].copy()   # 16.2% — NLP fallback

    log.info("  Direct ISCO L3 (from matched_code) : %s rows (%.1f%%)",
             f"{len(df_coded):,}", len(df_coded) / len(df) * 100)
    log.info("  NLP fallback (unknown/null code)   : %s rows (%.1f%%)",
             f"{len(df_fallback):,}", len(df_fallback) / len(df) * 100)

    # Assign esco_sector directly for rows with valid codes — no NLP needed
    df_coded["esco_sector"] = df_coded["occupation_group"]

    if len(df_fallback) == 0:
        log.info("  No NLP inference needed — all rows have valid matched_code.")
        df_merged = df_coded
    else:
        # Run SBERT + LightGBM ONLY on the fallback rows
        texts = df_fallback["combined_text"].tolist()
        log.info("  Embedding %s fallback rows for NLP inference ...", f"{len(texts):,}")
        emb = _embed_chunked(
            texts, embedding_model,
            cache_dir / "emb_nlp_fallback.npy",
            use_cache, "nlp_fallback",
            bs, cs, device, dataset_hash, cfg,
        )

        log.info("  Running LightGBM ISCO L3 predict_proba on %s fallback rows ...",
                 f"{len(texts):,}")
        probs    = classifier.predict_proba(emb)
        preds    = classifier.classes_[np.argmax(probs, axis=1)]
        max_prob = probs.max(axis=1)
        del emb; gc.collect()

        # Confidence distribution plot — helps calibrate conf_thr for 125 classes
        if cfg.get("plots", True):
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.hist(max_prob, bins=60, color="steelblue", alpha=0.85, edgecolor="white")
            ax.axvline(conf_thr, color="red", linestyle="--", linewidth=1.5,
                       label=f"threshold = {conf_thr}")
            n_kept = (max_prob >= conf_thr).sum()
            n_drop = (max_prob <  conf_thr).sum()
            ax.set_title(
                f"NLP Confidence Distribution — Fallback Rows Only\n"
                f"Kept: {n_kept:,} ({n_kept/len(max_prob)*100:.1f}%)  "
                f"Dropped: {n_drop:,} ({n_drop/len(max_prob)*100:.1f}%)",
                fontsize=11, fontweight="bold",
            )
            ax.set_xlabel("Max softmax probability"); ax.set_ylabel("Row count")
            ax.legend(fontsize=9); plt.tight_layout()
            out = Path(cfg["figures_dir"]) / "nlp_confidence_distribution.png"
            plt.savefig(out, dpi=150); plt.close()
            log.info("  Saved: %s", out)

        # Apply confidence threshold — drop fallback rows below threshold
        df_fallback = df_fallback.copy()
        df_fallback["esco_sector"] = [
            s if p >= conf_thr else None
            for s, p in zip(preds, max_prob)
        ]
        before_drop = len(df_fallback)
        df_fallback = df_fallback[df_fallback["esco_sector"].notna()].copy()
        log.info("  Fallback rows removed (low confidence) : %s",
                 f"{before_drop - len(df_fallback):,}")
        log.info("  Fallback rows kept                     : %s",
                 f"{len(df_fallback):,}")

        del probs; gc.collect()

        # Merge direct-code rows + accepted fallback rows
        df_merged = pd.concat([df_coded, df_fallback], ignore_index=True)

    total_before = len(df)
    log.info("  Total rows before NLP stage  : %s", f"{total_before:,}")
    log.info("  Total rows after NLP stage   : %s", f"{len(df_merged):,}")
    log.info("  Rows removed (low confidence): %s",
             f"{total_before - len(df_merged):,}")
    log.info("  Distinct ISCO L3 groups      : %d", df_merged["esco_sector"].nunique())

    # Validate: for direct-code rows, esco_sector must match occupation_group
    coded_check = df_merged[df_merged["needs_nlp"] == False]
    if len(coded_check) > 0:
        mismatch = (coded_check["esco_sector"] != coded_check["occupation_group"]).sum()
        if mismatch > 0:
            log.warning("  VALIDATION WARNING: %d direct-code rows have esco_sector "
                        "≠ occupation_group (should be 0)", mismatch)
        else:
            log.info("  Validation OK: all direct-code rows have esco_sector == occupation_group")

    # ISCO L3 distribution -> tables/
    sc = df_merged["esco_sector"].value_counts().reset_index()
    sc.columns = ["esco_sector", "N"]
    sc["pct"]  = (sc["N"] / len(df_merged) * 100).round(2)
    sc["l3_name"] = sc["esco_sector"].map(ISCO_L3_NAMES).fillna("Unknown")
    sc.to_csv(Path(cfg["tables_dir"]) / "sector_distribution.csv", index=False)
    log.info("  Sector distribution saved -> tables/sector_distribution.csv")

    df_merged = df_merged.sort_values(["person_id", "year"]).reset_index(drop=True)
    return df_merged


# ===========================================================================
# 5.  TRANSITION MODELING
# ===========================================================================

SMOOTHING = 0.01


def build_transitions(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Create next_sector / prev_sector columns and university flag.
    Returns (df_with_flags, transitions_df, person_univ_dict).
    """
    df = df.sort_values(["person_id", "year"]).reset_index(drop=True)
    df["next_sector"] = df.groupby("person_id")["esco_sector"].shift(-1)
    df["prev_sector"] = df.groupby("person_id")["esco_sector"].shift(1)
    transitions = df[df["next_sector"].notna()].copy().reset_index(drop=True)

    seq_stats = df.groupby("person_id").size().reset_index(name="n_jobs")
    log.info("  Total transitions    : %s", f"{len(transitions):,}")
    log.info("  Users >= 2 jobs      : %s", f"{(seq_stats['n_jobs'] >= 2).sum():,}")
    log.info("  Median seq length    : %.1f", seq_stats["n_jobs"].median())
    log.info("  Unique sector pairs  : %s",
             f"{transitions.groupby(['esco_sector','next_sector']).ngroups:,}")

    if "university_studies" in df.columns:
        univ_agg = (
            df.groupby("person_id")["university_studies"]
            .apply(lambda x: (x == True).sum() / max(x.notna().sum(), 1) >= 0.5)
            .reset_index()
        )
        univ_agg.columns = ["person_id", "univ_flag"]
        person_univ = dict(zip(univ_agg["person_id"], univ_agg["univ_flag"]))
        pct = np.mean(list(person_univ.values())) * 100
        log.info("  University flag: %.1f%% uni | %.1f%% no-uni", pct, 100 - pct)
    else:
        person_univ = {pid: False for pid in df["person_id"].unique()}
        log.info("  university_studies absent -- flag set False for all")

    transitions["univ_flag"] = transitions["person_id"].map(person_univ).fillna(False)
    df["univ_flag"]           = df["person_id"].map(person_univ).fillna(False)
    return df, transitions, person_univ


def build_trans_mat(
    data: pd.DataFrame,
    sector_idx: Dict[str, int],
    n_sec: int,
    sloop_mult: Dict[str, float],
    weights: Optional[str] = None,
) -> np.ndarray:
    """Build a smoothed, stickiness-adjusted, row-normalised transition matrix."""
    mat = np.zeros((n_sec, n_sec))

    if weights is None:
        ct = data.groupby(["esco_sector", "next_sector"]).size().reset_index(name="val")
    else:
        ct = data.groupby(["esco_sector", "next_sector"])[weights].sum().reset_index()
        ct.columns = ["esco_sector", "next_sector", "val"]

    ci = ct["esco_sector"].map(sector_idx).values
    ni = ct["next_sector"].map(sector_idx).values
    ok = ~pd.isna(ci) & ~pd.isna(ni)
    np.add.at(mat, (ci[ok].astype(int), ni[ok].astype(int)), ct["val"].values[ok])

    for s, si in sector_idx.items():
        mat[si, si] *= sloop_mult.get(s, 1.0)

    row_sums = mat.sum(axis=1)
    for i in range(n_sec):
        sv = 0.50 if row_sums[i] < 10 else (0.10 if row_sums[i] < 100 else SMOOTHING)
        mat[i] += sv

    mat /= mat.sum(axis=1, keepdims=True)
    return mat


# ===========================================================================
# 6.  PCA ON TRANSITION PROFILES
# ===========================================================================

def run_pca(transitions: pd.DataFrame, cfg: dict) -> Tuple[np.ndarray, List[str]]:
    """PCA on ISCO-08 L3 sector transition profiles with major-group colouring."""
    all_pca = sorted(set(transitions["esco_sector"]) | set(transitions["next_sector"]))
    n_pca   = len(all_pca)
    si_pca  = {s: i for i, s in enumerate(all_pca)}

    ct  = transitions.groupby(["esco_sector", "next_sector"]).size().reset_index(name="N")
    mat = np.zeros((n_pca, n_pca))
    ci  = ct["esco_sector"].map(si_pca).values
    ni  = ct["next_sector"].map(si_pca).values
    ok  = ~pd.isna(ci) & ~pd.isna(ni)
    np.add.at(mat, (ci[ok].astype(int), ni[ok].astype(int)), ct["N"].values[ok])
    mat += 0.01
    mat /= mat.sum(axis=1, keepdims=True)

    # Guard: PCA needs n_components <= min(n_samples, n_features)
    n_components = min(5, n_pca - 1)
    if n_components < 2:
        log.warning("  PCA skipped -- too few distinct ISCO L3 groups (%d)", n_pca)
        del mat; gc.collect()
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0]), all_pca

    pca_model  = PCA(n_components=n_components)
    pca_result = pca_model.fit_transform(mat)
    pca_var    = np.round(pca_model.explained_variance_ratio_ * 100, 1)
    # Pad to length 5 so callers expecting pca_var[0..4] never raise IndexError
    pca_var_padded = np.pad(pca_var, (0, max(0, 5 - len(pca_var))))
    log.info("  PCA variance -- PC1:%.1f%%  PC2:%.1f%%  Cum(1+2):%.1f%%",
             pca_var_padded[0], pca_var_padded[1], pca_var_padded[0] + pca_var_padded[1])

    # Use _l3_to_group() for colouring — maps ISCO L3 codes to 10 major groups
    # (based on the leading digit of the 3-digit code, stable regardless of name)
    ISCO_GROUP_COLORS = {
        "Managers":                              "#185FA5",
        "Professionals":                         "#1D9E75",
        "Technicians & Associate Professionals": "#E24B4A",
        "Clerical Support":                      "#A32D2D",
        "Service & Sales":                       "#993556",
        "Skilled Agriculture":                   "#2D7A2D",
        "Craft & Related Trades":                "#8B4513",
        "Plant & Machine Operators":             "#CC6600",
        "Elementary Occupations":                "#999999",
        "Armed Forces":                          "#534AB7",
        "Other":                                 "#CCCCCC",
    }

    pca_df          = pd.DataFrame({
        "PC1":    pca_result[:, 0],
        "PC2":    pca_result[:, 1],
        "sector": all_pca,
    })
    # _l3_to_group() reads the first digit of the L3 code — works for numeric codes
    pca_df["group"] = pca_df["sector"].apply(_l3_to_group)

    if cfg.get("plots", True):
        n_labels = len(all_pca)
        annotate = n_labels <= 60   # suppress text labels when very many groups
        fig, ax  = plt.subplots(figsize=(14, 10))
        for grp, color in ISCO_GROUP_COLORS.items():
            sub = pca_df[pca_df["group"] == grp]
            if sub.empty:
                continue
            ax.scatter(sub["PC1"], sub["PC2"], label=grp, color=color,
                       s=55, alpha=0.85, zorder=3)
            if annotate:
                for _, row in sub.iterrows():
                    ax.annotate(row["sector"][:30], (row["PC1"], row["PC2"]),
                                fontsize=5, alpha=0.75)
        ax.set_title(
            f"PCA of ISCO-08 L3 Career-Transition Profiles  ({n_labels} groups)\n"
            f"PC1:{pca_var_padded[0]}%  |  PC2:{pca_var_padded[1]}% variance",
            fontsize=13, fontweight="bold",
        )
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        ax.legend(fontsize=7, loc="best", ncol=2)
        plt.tight_layout()
        out = Path(cfg["plots_dir"]) / "pca_sector_profiles.png"
        plt.savefig(out, dpi=150); plt.close()
        log.info("  Saved: %s", out)

    del mat, pca_result; gc.collect()
    return pca_var_padded, all_pca


# ===========================================================================
# 7.  FULL ENSEMBLE PREDICTION SYSTEM
# ===========================================================================

def stratified_user_split(
    df: pd.DataFrame,
    transitions: pd.DataFrame,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """64/16/20 stratified split by dominant sector -- mirrors R exactly."""
    user_dom = (
        df.groupby("person_id")["esco_sector"]
        .agg(lambda x: x.value_counts().index[0])
        .reset_index()
    )
    user_dom.columns = ["person_id", "dom_sector"]

    np.random.seed(seed)
    split_rows = []
    for _, grp in user_dom.groupby("dom_sector"):
        n    = len(grp)
        ord_ = np.random.permutation(n)
        sp   = np.where(ord_ < int(n * 0.64), "train",
               np.where(ord_ < int(n * 0.80), "val", "test"))
        split_rows.append(pd.DataFrame({"person_id": grp["person_id"].values, "split": sp}))

    split_df = pd.concat(split_rows, ignore_index=True)
    train_u  = set(split_df.loc[split_df["split"] == "train", "person_id"])
    val_u    = set(split_df.loc[split_df["split"] == "val",   "person_id"])
    test_u   = set(split_df.loc[split_df["split"] == "test",  "person_id"])

    tr = transitions[transitions["person_id"].isin(train_u)].copy().reset_index(drop=True)
    va = transitions[transitions["person_id"].isin(val_u)].copy().reset_index(drop=True)
    te = transitions[transitions["person_id"].isin(test_u)].copy().reset_index(drop=True)
    log.info("  Stratified split -- Train: %s | Val: %s | Test: %s",
             f"{len(tr):,}", f"{len(va):,}", f"{len(te):,}")
    return tr, va, te, split_df


def build_ensemble_components(
    train_data: pd.DataFrame,
    all_sectors: List[str],
    sector_to_idx: Dict[str, int],
    self_rates: Dict[str, float],
    self_loop_mult: Dict[str, float],
    person_univ: Dict,
    n_sec: int,
    seed: int,
) -> dict:
    """
    Build all ensemble model components from training data:
      global_matrix, recency_matrix, per-decade matrices, market_drift,
      second_order_probs, user_hist_lookup, univ_boost vectors,
      archetype clustering, per-archetype matrices, user_cluster_map.
    All research logic preserved exactly from R implementation.
    """
    def _bmat(data, weights=None):
        return build_trans_mat(data, sector_to_idx, n_sec, self_loop_mult, weights)

    # 7.2  Global matrix
    global_matrix = _bmat(train_data)
    log.info("  Global matrix: %s", str(global_matrix.shape))

    # 7.3  Recency-weighted matrix (decay = 0.12)
    DECAY_RATE = 0.12
    max_yr     = int(train_data["year"].max())
    td         = train_data.copy()
    td["recency_weight"] = np.exp(DECAY_RATE * (td["year"] - max_yr))
    recency_matrix = _bmat(td, weights="recency_weight")
    del td; gc.collect()
    log.info("  Recency matrix built (decay=%.2f)", DECAY_RATE)

    # 7.4  Per-decade transition matrices
    years_sorted   = sorted(train_data["year"].unique())
    WINDOW_SIZE    = 10
    decade_windows = []
    i = 0
    while i < len(years_sorted):
        decade_windows.append((
            years_sorted[i],
            years_sorted[min(i + WINDOW_SIZE - 1, len(years_sorted) - 1)],
        ))
        i += WINDOW_SIZE

    transition_matrices: Dict[str, dict] = {}
    log.info("  %-15s | %10s | Status", "Years", "Sparsity")
    for start, end in decade_windows:
        lbl = f"{start}-{end}"
        wd  = train_data[(train_data["year"] >= start) & (train_data["year"] <= end)]
        if len(wd) == 0:
            continue
        raw = np.zeros((n_sec, n_sec))
        ct  = wd.groupby(["esco_sector", "next_sector"]).size().reset_index(name="N")
        ci  = ct["esco_sector"].map(sector_to_idx).values
        ni  = ct["next_sector"].map(sector_to_idx).values
        ok  = ~pd.isna(ci) & ~pd.isna(ni)
        np.add.at(raw, (ci[ok].astype(int), ni[ok].astype(int)), ct["N"].values[ok])
        for s, si in sector_to_idx.items():
            raw[si, si] *= self_loop_mult.get(s, 1.0)
        sparsity = (raw == 0).sum() / n_sec ** 2 * 100
        sv = 0.5 if sparsity > 70 else (0.1 if sparsity > 30 else SMOOTHING)
        raw += sv
        mat = raw / raw.sum(axis=1, keepdims=True)
        transition_matrices[lbl] = {"mat": mat, "start": start, "end": end}
        log.info("  %-15s | %9.2f%% | Built", lbl, sparsity)
        del raw, mat; gc.collect()

    # 7.5  Market drift
    market_drift: Dict[str, float] = {}
    dkeys = list(transition_matrices.keys())
    for j in range(1, len(dkeys)):
        diff = (transition_matrices[dkeys[j]]["mat"]
                - transition_matrices[dkeys[j - 1]]["mat"])
        market_drift[dkeys[j]] = float(np.sqrt((diff ** 2).sum()))
    log.info("  Market drift (Frobenius):")
    for nm, val in market_drift.items():
        log.info("    %-15s : %.4f", nm, val)

    # 7.6  2nd-order Markov with back-off (MIN_BIGRAM = 5)
    MIN_BIGRAM = 5
    so_data = train_data[
        train_data["prev_sector"].notna() &
        train_data["prev_sector"].isin(all_sectors)
    ].copy()
    bpc = so_data.groupby(["prev_sector", "esco_sector"]).size().reset_index(name="pair_count")
    btr = so_data.groupby(["prev_sector", "esco_sector", "next_sector"]).size().reset_index(name="N")

    second_order_probs: Dict[str, np.ndarray] = {}
    n_backoff = 0
    for _, brow in bpc.iterrows():
        if brow["pair_count"] < MIN_BIGRAM:
            n_backoff += 1
            continue
        key = f"{brow['prev_sector']}|{brow['esco_sector']}"
        sub = btr[(btr["prev_sector"] == brow["prev_sector"]) &
                  (btr["esco_sector"] == brow["esco_sector"])]
        vec = np.full(n_sec, SMOOTHING)
        ni_ = sub["next_sector"].map(sector_to_idx).values
        ok_ = ~pd.isna(ni_)
        vec[ni_[ok_].astype(int)] = sub["N"].values[ok_] + SMOOTHING
        cs = brow["esco_sector"]
        if cs in sector_to_idx:
            vec[sector_to_idx[cs]] *= self_loop_mult.get(cs, 1.0)
        second_order_probs[key] = vec / vec.sum()
    log.info("  2nd-order keys: %s | Back-off: %s",
             f"{len(second_order_probs):,}", f"{n_backoff:,}")
    del so_data, bpc, btr; gc.collect()

    # 7.7  Per-user history priors (HIST_DECAY = 0.15)
    HIST_DECAY = 0.15
    hr = train_data[["person_id", "esco_sector", "year"]].copy()
    hr["rw"] = np.exp(HIST_DECAY * (hr["year"] - max_yr))
    uw = hr.groupby(["person_id", "esco_sector"])["rw"].sum().reset_index()
    uw.columns = ["person_id", "sector", "wt_count"]
    ut = uw.groupby("person_id")["wt_count"].sum().reset_index()
    ut.columns = ["person_id", "total"]
    multi_u = ut.loc[ut["total"] >= 2, "person_id"].values

    user_hist_lookup: Dict[str, np.ndarray] = {}
    for uid in multi_u:
        rows = uw[uw["person_id"] == uid]
        vec  = np.full(n_sec, 0.01)
        si_  = rows["sector"].map(sector_to_idx).values
        ok_  = ~pd.isna(si_)
        np.add.at(vec, si_[ok_].astype(int), rows["wt_count"].values[ok_])
        user_hist_lookup[str(int(uid))] = vec / vec.sum()
    log.info("  User history priors: %s users", f"{len(user_hist_lookup):,}")
    del hr, uw, ut; gc.collect()

    # 7.8  University boost vectors
    univ_boost_TRUE  = np.ones(n_sec)
    univ_boost_FALSE = np.ones(n_sec)
    if train_data["univ_flag"].any():
        tru  = train_data[train_data["univ_flag"] == True]
        tnu  = train_data[train_data["univ_flag"] == False]
        dall = train_data.groupby("next_sector").size().reset_index(name="N_all")
        du   = tru.groupby("next_sector").size().reset_index(name="N_u")
        dn   = tnu.groupby("next_sector").size().reset_index(name="N_n")
        ta, tu_, tn_ = len(train_data), len(tru), len(tnu)
        for s, si in sector_to_idx.items():
            pa  = (dall.loc[dall["next_sector"] == s, "N_all"].sum() + SMOOTHING) / (ta  + n_sec * SMOOTHING)
            pu2 = (du.loc[du["next_sector"]     == s, "N_u"].sum()   + SMOOTHING) / (tu_ + n_sec * SMOOTHING)
            pn2 = (dn.loc[dn["next_sector"]     == s, "N_n"].sum()   + SMOOTHING) / (tn_ + n_sec * SMOOTHING)
            if pa > 0:
                univ_boost_TRUE[si]  = np.clip(pu2 / pa, 0.5, 2.0)
                univ_boost_FALSE[si] = np.clip(pn2 / pa, 0.5, 2.0)
        log.info("  University boost vectors built.")
        top5 = sorted(zip(all_sectors, univ_boost_TRUE), key=lambda x: -x[1])[:5]
        for nm, val in top5:
            log.info("    %-45s : %.3f", nm, val)
    else:
        log.info("  No university flag variation -- boost vectors uniform (1.0)")

    # 7.9  Archetype clustering on training data (entropy-based drift)
    ci_tr = train_data["esco_sector"].map(sector_to_idx).values
    ni_tr = train_data["next_sector"].map(sector_to_idx).values
    ok_tr = ~pd.isna(ci_tr) & ~pd.isna(ni_tr)
    lp_tr = np.full(len(train_data), np.log(1e-9))
    lp_tr[ok_tr] = np.log(np.maximum(
        global_matrix[ci_tr[ok_tr].astype(int), ni_tr[ok_tr].astype(int)], 1e-9
    ))
    td2           = train_data.copy()
    td2["log_prob"] = lp_tr
    td2["is_self"]  = td2["esco_sector"] == td2["next_sector"]
    del lp_tr; gc.collect()

    risk_tr = (
        td2.groupby("person_id").agg(
            drift_score   =("log_prob",    lambda x: -x.mean()),
            seq_length    =("esco_sector", "count"),
            self_loop_rate=("is_self",     "mean"),
        ).reset_index()
    )
    risk_tr = risk_tr[risk_tr["seq_length"] >= 2].copy()

    feat_arch   = StandardScaler().fit_transform(
        risk_tr[["drift_score", "seq_length", "self_loop_rate"]].dropna()
    )
    valid_idx_a = risk_tr[["drift_score", "seq_length", "self_loop_rate"]].dropna().index
    np.random.seed(seed)
    km_arch = KMeans(n_clusters=5, n_init=25, max_iter=100, random_state=seed)
    risk_tr.loc[valid_idx_a, "cluster"] = km_arch.fit_predict(feat_arch)

    cs_arch = (
        risk_tr[risk_tr["cluster"].notna()]
        .groupby("cluster").agg(
            mean_drift=("drift_score",    "mean"),
            mean_len  =("seq_length",     "mean"),
            mean_loop =("self_loop_rate", "mean"),
        ).reset_index()
    )
    q80d = cs_arch["mean_drift"].quantile(0.8)
    q20d = cs_arch["mean_drift"].quantile(0.2)
    q50d = cs_arch["mean_drift"].quantile(0.5)
    q75l = cs_arch["mean_len"].quantile(0.75)

    def _arch(row):
        if row["mean_drift"] > q80d and row["mean_loop"] < 0.3: return "Career Switcher"
        if row["mean_drift"] < q20d and row["mean_loop"] > 0.5: return "Sector Loyalist"
        if row["mean_len"]   > q75l:                             return "Career Veteran"
        if row["mean_drift"] > q50d:                             return "Gradual Mover"
        return "Stable Specialist"

    cs_arch["archetype"] = cs_arch.apply(_arch, axis=1)
    log.info("  Archetype labels:\n%s",
             cs_arch[["cluster", "archetype", "mean_drift", "mean_loop"]].round(4).to_string(index=False))

    user_cluster_map = (
        risk_tr[["person_id", "cluster"]]
        .merge(cs_arch[["cluster", "archetype"]], on="cluster")
    )
    td2 = td2.merge(user_cluster_map[["person_id", "archetype"]], on="person_id", how="left")
    td2["archetype"] = td2["archetype"].fillna("Stable Specialist")
    td2["univ_flag"] = td2["person_id"].map(person_univ).fillna(False)

    # 7.10  Per-archetype matrices
    archetype_matrices: Dict[str, np.ndarray] = {}
    log.info("  Per-archetype matrices:")
    for arch in cs_arch["archetype"].unique():
        sub = td2[td2["archetype"] == arch]
        if len(sub) >= 50:
            archetype_matrices[arch] = _bmat(sub)
            log.info("    %-25s : %s transitions", arch, f"{len(sub):,}")
        else:
            archetype_matrices[arch] = global_matrix
            log.info("    %-25s : too few (%d) -- using global", arch, len(sub))

    del td2; gc.collect()

    return {
        "global_matrix":       global_matrix,
        "recency_matrix":      recency_matrix,
        "transition_matrices": transition_matrices,
        "market_drift":        market_drift,
        "second_order_probs":  second_order_probs,
        "user_hist_lookup":    user_hist_lookup,
        "univ_boost_TRUE":     univ_boost_TRUE,
        "univ_boost_FALSE":    univ_boost_FALSE,
        "archetype_matrices":  archetype_matrices,
        "user_cluster_map":    user_cluster_map,
        "cs_arch":             cs_arch,
        "max_yr":              max_yr,
    }


def get_win_key(yr: int, transition_matrices: dict) -> str:
    for key, w in transition_matrices.items():
        if w["start"] <= yr <= w["end"]:
            return key
    return "global"


def compute_ensemble_prob(
    ci: int,
    wk: str,
    arch: str,
    univ: bool,
    uid,
    prev_s,
    curr_s: str,
    wh: float,
    ws: float,
    wa: float,
    components: dict,
    sector_to_idx: Dict[str, int],
    all_sectors: List[str],
    n_sec: int,
) -> np.ndarray:
    """
    Core ensemble probability vector.
    Mirrors R ensemble_topk() inner logic exactly:
        base  = 0.60 * rec_p + 0.40 * dyn_p
        ens_p = w_rest * base + wa * arch_p + ws * so_p + wh * hist_p
        ens_p = ens_p * boost / sum(ens_p)
    """
    gm  = components["global_matrix"]
    rm  = components["recency_matrix"]
    tm  = components["transition_matrices"]
    so  = components["second_order_probs"]
    uh  = components["user_hist_lookup"]
    am  = components["archetype_matrices"]

    w_rest = max(0.05, 1.0 - wh - ws - wa)
    dyn_p  = tm[wk]["mat"][ci] if (wk != "global" and wk in tm) else gm[ci]
    rec_p  = rm[ci]
    arch_p = am.get(arch, gm)[ci]

    so_key = (
        f"{prev_s}|{curr_s}"
        if (prev_s is not None and not pd.isna(prev_s) and prev_s in sector_to_idx)
        else None
    )
    so_p   = so[so_key] if (so_key and so_key in so) else gm[ci]
    hist_p = uh.get(str(int(uid)), gm[ci])

    base  = 0.60 * rec_p + 0.40 * dyn_p
    ens_p = w_rest * base + wa * arch_p + ws * so_p + wh * hist_p

    boost  = components["univ_boost_TRUE"] if univ else components["univ_boost_FALSE"]
    ens_p  = ens_p * boost
    ens_p /= ens_p.sum()
    return ens_p


def tune_sector_weights(
    val_data: pd.DataFrame,
    all_sectors: List[str],
    sector_to_idx: Dict[str, int],
    self_rates: Dict[str, float],
    components: dict,
    n_sec: int,
    seed: int,
) -> Dict[str, dict]:
    """Per-sector ensemble weight tuning on validation set."""
    log.info("  Tuning per-sector ensemble weights on validation data ...")

    w_hist_g = [0.10, 0.20, 0.30, 0.40, 0.50]
    w_so_g   = [0.00, 0.10, 0.20, 0.30]
    w_arch_g = [0.05, 0.10, 0.20]

    sector_best_w: Dict[str, dict] = {}
    for s in all_sectors:
        sr_s = self_rates.get(s, 0.15)
        if sr_s > 0.25:
            sector_best_w[s] = {"w_hist": 0.30, "w_so": 0.10, "w_arch": 0.10}
        elif sr_s < 0.10:
            sector_best_w[s] = {"w_hist": 0.20, "w_so": 0.30, "w_arch": 0.10}
        else:
            sector_best_w[s] = {"w_hist": 0.20, "w_so": 0.20, "w_arch": 0.10}

    tm = components["transition_matrices"]

    for curr_s in all_sectors:
        sub_val = val_data[val_data["esco_sector"] == curr_s]
        if len(sub_val) < 15:
            continue
        if len(sub_val) > 2000:
            sub_val = sub_val.sample(2000, random_state=seed)
        ci = sector_to_idx.get(curr_s)
        if ci is None:
            continue

        best_acc = -1.0
        best_w   = sector_best_w[curr_s]
        for wh, ws, wa in itertools.product(w_hist_g, w_so_g, w_arch_g):
            if 1.0 - wh - ws - wa < 0.05:
                continue
            hits = 0
            for _, row in sub_val.iterrows():
                wk    = get_win_key(row["year"], tm)
                ens_p = compute_ensemble_prob(
                    ci, wk, row["archetype"], row["univ_flag"],
                    row["person_id"], row.get("prev_sector"), curr_s,
                    wh, ws, wa, components, sector_to_idx, all_sectors, n_sec,
                )
                if all_sectors[int(np.argmax(ens_p))] == row["next_sector"]:
                    hits += 1
            acc = hits / len(sub_val)
            if acc > best_acc:
                best_acc = acc
                best_w   = {"w_hist": wh, "w_so": ws, "w_arch": wa}
        sector_best_w[curr_s] = best_w

    log.info("  Weight tuning complete.")
    return sector_best_w


def evaluate_ensemble(
    test_data: pd.DataFrame,
    all_sectors: List[str],
    sector_to_idx: Dict[str, int],
    sector_best_w: Dict[str, dict],
    components: dict,
    n_sec: int,
) -> dict:
    """Evaluate ensemble on test set at Top-1 / Top-3 / Top-5."""
    log.info("  Evaluating ensemble on %s test transitions ...", f"{len(test_data):,}")
    TOP_K = [1, 3, 5]
    max_k = max(TOP_K)
    tm    = components["transition_matrices"]

    valid = test_data[test_data["esco_sector"].isin(all_sectors)].copy()
    preds = []
    for _, row in valid.iterrows():
        curr = row["esco_sector"]
        ci   = sector_to_idx.get(curr)
        if ci is None:
            preds.append(all_sectors[:max_k])
            continue
        w  = sector_best_w.get(curr, {"w_hist": 0.20, "w_so": 0.20, "w_arch": 0.10})
        wk = get_win_key(row["year"], tm)
        ens_p = compute_ensemble_prob(
            ci, wk, row["archetype"], row["univ_flag"],
            row["person_id"], row.get("prev_sector"), curr,
            w["w_hist"], w["w_so"], w["w_arch"],
            components, sector_to_idx, all_sectors, n_sec,
        )
        top_idx = np.argsort(ens_p)[::-1][:max_k]
        preds.append([all_sectors[i] for i in top_idx])

    persistence_acc = (valid["esco_sector"] == valid["next_sector"]).mean()
    results = {"persistence_top1": float(persistence_acc)}
    for k in TOP_K:
        hits = [true in pred[:k] for true, pred in zip(valid["next_sector"], preds)]
        results[f"top_{k}"] = float(np.mean(hits))

    log.info("  %-22s | %7s | %7s | %7s", "Model", "Top-1", "Top-3", "Top-5")
    log.info("  %s", "-" * 55)
    log.info("  %-22s | %6.2f%% | %7s | %7s",
             "Persistence Baseline", persistence_acc * 100, "N/A", "N/A")
    log.info("  %-22s | %6.2f%% | %6.2f%% | %6.2f%%",
             "Ensemble", results["top_1"]*100, results["top_3"]*100, results["top_5"]*100)
    log.info("  Ensemble vs Baseline: %+.2f%%",
             (results["top_1"] - persistence_acc) * 100)
    return results


# ===========================================================================
# 8.  DRIFT ANALYSIS & CLUSTERING
# ===========================================================================

def compute_drift_and_clusters(
    df: pd.DataFrame,
    transitions: pd.DataFrame,
    global_matrix: np.ndarray,
    sector_to_idx: Dict[str, int],
    all_sectors: List[str],
    seed: int,
    cfg: dict,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Entropy-based drift scores, KMeans archetype clustering, archetype labelling.
    Returns (risk_df, cs2, dom_sec_df).
    """
    ci_all = transitions["esco_sector"].map(sector_to_idx).values
    ni_all = transitions["next_sector"].map(sector_to_idx).values
    ok_all = ~pd.isna(ci_all) & ~pd.isna(ni_all)
    lp_all = np.full(len(transitions), np.log(1e-9))
    lp_all[ok_all] = np.log(np.maximum(
        global_matrix[ci_all[ok_all].astype(int), ni_all[ok_all].astype(int)], 1e-9
    ))
    tr2           = transitions.copy()
    tr2["log_prob"] = lp_all
    tr2["is_self"]  = tr2["esco_sector"] == tr2["next_sector"]

    risk_df = (
        tr2.groupby("person_id").agg(
            drift_score       =("log_prob",    lambda x: -x.mean()),
            total_transitions =("esco_sector", "count"),
            self_loop_rate    =("is_self",     "mean"),
        ).reset_index()
    )
    risk_df = risk_df[risk_df["total_transitions"] >= 2].copy()
    risk_df = risk_df.sort_values("drift_score", ascending=False)

    seq_len_df = df.groupby("person_id").size().reset_index(name="seq_length")
    dom_sec_df = (
        df.groupby("person_id")["esco_sector"]
        .agg(lambda x: x.value_counts().index[0])
        .reset_index()
    )
    dom_sec_df.columns = ["person_id", "dominant_sector"]

    risk_df = risk_df.merge(seq_len_df, on="person_id", how="left")
    risk_df = risk_df.merge(dom_sec_df, on="person_id", how="left")

    log.info("  Users in risk_df   : %s", f"{len(risk_df):,}")
    log.info("  Median drift score : %.4f", risk_df["drift_score"].median())

    feat  = risk_df[["drift_score", "seq_length", "self_loop_rate"]].dropna()
    X_c   = StandardScaler().fit_transform(feat)
    vi    = feat.index
    np.random.seed(seed)
    km2   = KMeans(n_clusters=5, n_init=25, max_iter=100, random_state=seed)
    risk_df.loc[vi, "cluster"] = km2.fit_predict(X_c)

    cs2 = (
        risk_df[risk_df["cluster"].notna()]
        .groupby("cluster").agg(
            n_users   =("person_id",      "count"),
            mean_drift=("drift_score",    "mean"),
            mean_len  =("seq_length",     "mean"),
            mean_loop =("self_loop_rate", "mean"),
        ).reset_index()
    )
    q80c = cs2["mean_drift"].quantile(0.8)
    q20c = cs2["mean_drift"].quantile(0.2)
    q50c = cs2["mean_drift"].quantile(0.5)
    q75c = cs2["mean_len"].quantile(0.75)

    def _arch2(row):
        if row["mean_drift"] > q80c and row["mean_loop"] < 0.3: return "Career Switcher"
        if row["mean_drift"] < q20c and row["mean_loop"] > 0.5: return "Sector Loyalist"
        if row["mean_len"]   > q75c:                             return "Career Veteran"
        if row["mean_drift"] > q50c:                             return "Gradual Mover"
        return "Stable Specialist"

    cs2["archetype"] = cs2.apply(_arch2, axis=1)
    risk_df = risk_df.merge(cs2[["cluster", "archetype"]], on="cluster", how="left")
    log.info("  Cluster summary:\n%s",
             cs2[["cluster", "archetype", "n_users", "mean_drift", "mean_loop"]]
             .round(4).to_string(index=False))

    out = Path(cfg["tables_dir"]) / "drift_risk_scores.csv"
    risk_df.to_csv(out, index=False)
    log.info("  Saved: %s", out)
    log.info("  TOP 10 DRIFT RISK:\n%s",
             risk_df[["person_id", "drift_score", "total_transitions"]].head(10).to_string(index=False))

    del tr2, lp_all; gc.collect()
    return risk_df, cs2, dom_sec_df


# ===========================================================================
# 9.  ASSOCIATION RULES
# ===========================================================================

def compute_association_rules(transitions: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Direct support / confidence / lift -- mirrors R exactly."""
    tot = len(transitions)
    pc  = transitions.groupby(["esco_sector", "next_sector"]).size().reset_index(name="pair_count")
    ac  = transitions.groupby("esco_sector").size().reset_index(name="ant_count")
    cc  = transitions.groupby("next_sector").size().reset_index(name="con_count")

    rules = (
        pc.merge(ac, on="esco_sector")
          .merge(cc, on="next_sector")
          .rename(columns={"esco_sector": "from", "next_sector": "to"})
    )
    rules["support"]    = (rules["pair_count"] / tot).round(4)
    rules["confidence"] = (rules["pair_count"] / rules["ant_count"]).round(4)
    rules["lift"]       = (rules["confidence"] / (rules["con_count"] / tot)).round(4)
    rules = (
        rules[["from", "to", "support", "confidence", "lift", "pair_count"]]
        .sort_values("lift", ascending=False).reset_index(drop=True)
    )
    log.info("  Total rules: %s", f"{len(rules):,}")
    log.info("  TOP 15 BY LIFT:\n%s", rules.head(15).to_string(index=False))

    out = Path(cfg["tables_dir"]) / "association_rules.csv"
    rules.to_csv(out, index=False)
    log.info("  Saved: %s", out)
    return rules


# ===========================================================================
# 10.  VISUALISATIONS
# ===========================================================================

ARCHETYPE_COLORS = {
    "Career Switcher":   "#E24B4A",
    "Sector Loyalist":   "#1D9E75",
    "Career Veteran":    "#378ADD",
    "Gradual Mover":     "#BA7517",
    "Stable Specialist": "#534AB7",
}


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("  Saved: %s", path)


def plot_sector_distribution(df: pd.DataFrame, plots_dir: str) -> None:
    sc = df["esco_sector"].value_counts().reset_index()
    sc.columns = ["esco_sector", "N"]
    sc["pct"]  = (sc["N"] / sc["N"].sum() * 100).round(1)
    sc = sc.sort_values("N")
    fig, ax = plt.subplots(figsize=(13, 9))
    ax.barh(sc["esco_sector"], sc["N"], color="steelblue", alpha=0.85)
    for _, row in sc.iterrows():
        ax.text(row["N"] * 1.005, row.name, f"{row['pct']}%", va="center", fontsize=8)
    ax.set_title(f"Job Distribution Across {df['esco_sector'].nunique()} ISCO-08 L3 Groups (NLP-classified)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Records")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.tight_layout()
    _save(fig, Path(plots_dir) / "sector_distribution.png")


def plot_sequence_length(df: pd.DataFrame, plots_dir: str) -> None:
    sq = df.groupby("person_id").size().reset_index(name="n_jobs")
    sq["nc"] = sq["n_jobs"].clip(upper=15)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(sq["nc"], bins=range(1, 17), color="steelblue", alpha=0.85, edgecolor="white")
    ax.set_xticks(range(1, 16))
    ax.set_xticklabels([str(i) if i < 15 else "15+" for i in range(1, 16)])
    ax.set_title("Career Sequence Length Distribution", fontsize=12, fontweight="bold")
    ax.set_xlabel("Number of classified jobs"); ax.set_ylabel("Number of users")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.tight_layout()
    _save(fig, Path(plots_dir) / "sequence_length_distribution.png")


def plot_drift_distribution(risk_df: pd.DataFrame, plots_dir: str) -> None:
    mv = float(risk_df["drift_score"].median())
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(risk_df["drift_score"], bins=50, color="gray", alpha=0.85, edgecolor="white")
    ax.axvline(mv, color="red", linestyle="--", linewidth=1.1)
    ax.text(mv + 0.02, ax.get_ylim()[1] * 0.92, f"Median: {mv:.2f}", color="red", fontsize=10)
    ax.set_title("Career Drift Distribution (entropy-based)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Drift Score  (higher = more volatile)"); ax.set_ylabel("Number of Users")
    plt.tight_layout()
    _save(fig, Path(plots_dir) / "career_drift_distribution.png")


def plot_transition_heatmap(transitions: pd.DataFrame, df: pd.DataFrame, plots_dir: str) -> None:
    hd          = transitions.groupby(["esco_sector", "next_sector"]).size().reset_index(name="N")
    hd["from_s"] = hd["esco_sector"].str[:18]
    hd["to_s"]   = hd["next_sector"].str[:18]
    pivot        = hd.pivot_table(index="from_s", columns="to_s", values="N", fill_value=0)
    log_pivot    = np.log1p(pivot)
    fig, ax      = plt.subplots(figsize=(14, 12))
    sns.heatmap(log_pivot, cmap="Blues", ax=ax, linewidths=0.3, linecolor="white",
                cbar_kws={"label": "log(count+1)"})
    ax.set_title(f"Career Transition Heatmap ({df['esco_sector'].nunique()} ISCO-08 L3 Groups)",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Next Sector"); ax.set_ylabel("Current Sector")
    plt.xticks(rotation=60, ha="right", fontsize=7); plt.yticks(fontsize=7)
    plt.tight_layout()
    _save(fig, Path(plots_dir) / "transition_heatmap.png")


def plot_top_transitions(transitions: pd.DataFrame, plots_dir: str) -> None:
    top = (
        transitions.groupby(["esco_sector", "next_sector"]).size()
        .reset_index(name="count").sort_values("count", ascending=False).head(20)
    )
    top["transition"] = top["esco_sector"].str[:25] + " ->\n" + top["next_sector"].str[:25]
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.barh(top["transition"].iloc[::-1].values, top["count"].iloc[::-1].values,
            color="steelblue", alpha=0.85)
    ax.set_title("Top 20 Most Common Career Transitions", fontsize=12, fontweight="bold")
    ax.set_xlabel("Count")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.tight_layout()
    _save(fig, Path(plots_dir) / "top_transitions.png")


def plot_market_drift(market_drift: dict, plots_dir: str) -> None:
    if not market_drift:
        log.info("  market_drift empty -- skipping plot")
        return
    ddf = pd.DataFrame({"window": list(market_drift.keys()), "drift": list(market_drift.values())})
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ddf["window"], ddf["drift"], color="steelblue", linewidth=1.2, marker="o", markersize=5)
    ax.set_title(
        "Market Drift Over Time (Drifting Markov Model)\n"
        "Frobenius norm between consecutive decade matrices",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlabel("Decade window"); ax.set_ylabel("Drift (Frobenius norm)")
    plt.xticks(rotation=30, ha="right"); plt.tight_layout()
    _save(fig, Path(plots_dir) / "market_drift.png")


def plot_user_clusters(risk_df: pd.DataFrame, cs2: pd.DataFrame, plots_dir: str) -> None:
    pdf2 = risk_df[risk_df["cluster"].notna()].copy()
    fig, ax = plt.subplots(figsize=(12, 7))
    for arch, color in ARCHETYPE_COLORS.items():
        sub = pdf2[pdf2["archetype"] == arch]
        ax.scatter(sub["drift_score"], sub["seq_length"],
                   alpha=0.35, s=6, color=color, label=arch)
    for _, crow in cs2.iterrows():
        color = ARCHETYPE_COLORS.get(crow["archetype"], "black")
        ax.scatter(crow["mean_drift"], crow["mean_len"], color=color, s=120, marker="D", zorder=5)
    ax.set_title("User Career Clusters (k-means, k=5)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Drift Score (entropy-based)"); ax.set_ylabel("Sequence Length")
    ax.legend(fontsize=9, loc="upper right"); plt.tight_layout()
    _save(fig, Path(plots_dir) / "user_clusters.png")


def plot_cluster_sector_breakdown(risk_df: pd.DataFrame, cs2: pd.DataFrame,
                                   dom_sec_df: pd.DataFrame, plots_dir: str) -> None:
    # Ensure dom_sec_df has exactly the expected column name before merging.
    # The column may arrive as "dominant_sector" or, if risk_df already contains
    # it from a prior merge, we must avoid the _x / _y duplication that causes
    # KeyError: 'dominant_sector'.
    dom_clean = dom_sec_df[["person_id", "dominant_sector"]].drop_duplicates("person_id")

    base = risk_df[risk_df["cluster"].notna()][["person_id", "cluster"]].copy()
    base = base.merge(dom_clean, on="person_id", how="left")

    tsp = (
        base
        .groupby(["cluster", "dominant_sector"])
        .size().reset_index(name="n")
        .sort_values(["cluster", "n"], ascending=[True, False])
        .groupby("cluster").head(3)
        .merge(cs2[["cluster", "archetype"]], on="cluster", how="left")
    )
    tsp["dominant_sector"] = tsp["dominant_sector"].fillna("Unknown").astype(str)

    unique_archs = tsp["archetype"].dropna().unique()
    if len(unique_archs) == 0:
        log.warning("  plot_cluster_sector_breakdown: no archetype data -- skipping")
        return

    fig, axes = plt.subplots(1, max(len(unique_archs), 1), figsize=(14, 6), sharey=False)
    if len(unique_archs) == 1:
        axes = [axes]
    for ax, arch in zip(axes, unique_archs):
        sub = tsp[tsp["archetype"] == arch].sort_values("n")
        ax.barh(sub["dominant_sector"].str[:22], sub["n"],
                color=ARCHETYPE_COLORS.get(arch, "steelblue"), alpha=0.85)
        ax.set_title(arch, fontsize=9, fontweight="bold")
        ax.set_xlabel("Users"); ax.tick_params(labelsize=7)
    plt.suptitle("Top Sectors per Career Archetype Cluster", fontsize=12, fontweight="bold")
    plt.tight_layout()
    _save(fig, Path(plots_dir) / "cluster_sector_breakdown.png")


def plot_accuracy_comparison(ensemble_results: dict, plots_dir: str) -> None:
    persistence_acc = ensemble_results["persistence_top1"]
    labels = ["Top-1", "Top-3", "Top-5"]
    values = [ensemble_results["top_1"]*100,
               ensemble_results["top_3"]*100,
               ensemble_results["top_5"]*100]
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, values, color="#185FA5", alpha=0.9, width=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", fontsize=12, fontweight="bold", color="#185FA5")
    ax.axhline(persistence_acc * 100, color="red", linestyle="--", linewidth=1.2)
    ax.text(0.05, persistence_acc * 100 + 1.5,
            f"Persistence Baseline: {persistence_acc * 100:.2f}%", color="red", fontsize=10)
    ax.text(0, ensemble_results["top_1"] * 100 / 2,
            f"{(ensemble_results['top_1'] - persistence_acc) * 100:+.2f}%\nabove\nbaseline",
            ha="center", color="white", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.set_title("Career Prediction Accuracy -- Ensemble Model (ISCO-08 L3 NLP labels)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Prediction Task"); ax.set_ylabel("Accuracy (%)")
    plt.tight_layout()
    _save(fig, Path(plots_dir) / "accuracy_comparison.png")


def plot_association_rules_lift(assoc_rules: pd.DataFrame, plots_dir: str) -> None:
    top20 = assoc_rules[assoc_rules["from"] != assoc_rules["to"]].head(20).copy()
    top20["rule"] = top20["from"].str[:22] + " ->\n" + top20["to"].str[:22]
    cmap_vals = np.linspace(0.3, 0.85, len(top20))
    fig, ax   = plt.subplots(figsize=(13, 9))
    for i, (_, row) in enumerate(top20.iloc[::-1].iterrows()):
        ax.barh(row["rule"], row["lift"], color=plt.cm.Blues(cmap_vals[i]), alpha=0.9)
        ax.text(row["lift"] + 0.01, i, f"lift={row['lift']:.2f}", va="center", fontsize=8)
    ax.set_title("Top 20 Cross-Sector Transitions by Lift\nLift > 1 = more likely than chance",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Lift"); plt.tight_layout()
    _save(fig, Path(plots_dir) / "association_rules_lift.png")


def run_all_plots(
    df: pd.DataFrame,
    transitions: pd.DataFrame,
    risk_df: pd.DataFrame,
    cs2: pd.DataFrame,
    dom_sec_df: pd.DataFrame,
    assoc_rules: pd.DataFrame,
    market_drift: dict,
    ensemble_results: dict,
    plots_dir: str,
) -> None:
    plot_sector_distribution(df, plots_dir)
    plot_sequence_length(df, plots_dir)
    plot_drift_distribution(risk_df, plots_dir)
    plot_transition_heatmap(transitions, df, plots_dir)
    plot_top_transitions(transitions, plots_dir)
    plot_market_drift(market_drift, plots_dir)
    plot_user_clusters(risk_df, cs2, plots_dir)
    plot_cluster_sector_breakdown(risk_df, cs2, dom_sec_df, plots_dir)
    plot_accuracy_comparison(ensemble_results, plots_dir)
    plot_association_rules_lift(assoc_rules, plots_dir)


# ===========================================================================
# 11.  FINAL METRICS EXPORT + SUMMARY
# ===========================================================================

def export_final_metrics(
    nlp_metrics: dict,
    ensemble_results: dict,
    pca_var: np.ndarray,
    cfg: dict,
    dataset_hash: str = "",
    df_stats: Optional[dict] = None,
) -> None:
    """Export all final metrics to JSON in metrics/ for publication reproducibility."""
    metrics_dir = Path(cfg["metrics_dir"])
    payload: Dict[str, Any] = {
        "timestamp_utc":      datetime.utcnow().isoformat(),
        "pipeline_version":   PIPELINE_VERSION,
        "sbert_model":        SBERT_MODEL_NAME,
        "dataset_hash":       dataset_hash,
        "seed":               cfg["seed"],
        "confidence_threshold": cfg["confidence_thr"],
        "embed_batch_size":   cfg["embed_batch_size"],
        "embed_chunk_size":   cfg["embed_chunk_size"],
        "nlp_classification": {
            "accuracy":  round(nlp_metrics.get("accuracy",  0.0), 6),
            "precision": round(nlp_metrics.get("precision", 0.0), 6),
            "recall":    round(nlp_metrics.get("recall",    0.0), 6),
            "f1":        round(nlp_metrics.get("f1",        0.0), 6),
        },
        "career_prediction_ensemble": {
            "persistence_baseline_top1": round(ensemble_results["persistence_top1"] * 100, 4),
            "top_1_pct": round(ensemble_results["top_1"] * 100, 4),
            "top_3_pct": round(ensemble_results["top_3"] * 100, 4),
            "top_5_pct": round(ensemble_results["top_5"] * 100, 4),
            "improvement_vs_baseline_pct": round(
                (ensemble_results["top_1"] - ensemble_results["persistence_top1"]) * 100, 4
            ),
        },
        "pca_explained_variance": {
            "PC1_pct":                float(pca_var[0]),
            "PC2_pct":                float(pca_var[1]),
            "cumulative_PC1_PC2_pct": float(pca_var[0] + pca_var[1]),
        },
        "stage_timing_seconds": {k: round(v, 2) for k, v in _STAGE_TIMES.items()},
    }
    if df_stats:
        payload["dataset_statistics"] = df_stats

    out = metrics_dir / "final_metrics.json"
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=2)
    log.info("  Final metrics -> %s", out)


def write_pipeline_summary(
    nlp_metrics: dict,
    ensemble_results: dict,
    pca_var: np.ndarray,
    cfg: dict,
    df_stats: dict,
    risk_df: pd.DataFrame,
    cs2: pd.DataFrame,
    n_transitions: int,
    device: str,
    dataset_hash: str,
) -> None:
    """
    Write a human-readable final pipeline summary to
    outputs/summaries/final_pipeline_summary.txt
    """
    lines: List[str] = []
    sep  = "=" * 70
    dash = "-" * 70

    def h(title):
        lines.append(sep)
        lines.append(f"  {title}")
        lines.append(sep)

    def s(k, v):
        lines.append(f"  {k:<40} {v}")

    lines.append(sep)
    lines.append(f"  CAREER DRIFT TRAJECTORY ANALYSIS -- PIPELINE SUMMARY")
    lines.append(f"  Generated : {datetime.utcnow().isoformat()} UTC")
    lines.append(f"  Pipeline  : v{PIPELINE_VERSION}  (Phase 1 -- ISCO-08 L3 upgrade)")
    lines.append(sep)
    lines.append("")

    h("1. DATASET STATISTICS")
    s("Dataset hash (SHA-256[:16])", dataset_hash)
    for k, v in df_stats.items():
        s(k, str(v))
    s("Total transitions", f"{n_transitions:,}")
    lines.append("")

    h("2. RUNTIME")
    s("Device", device)
    total_sec = sum(_STAGE_TIMES.values())
    for stage, elapsed in _STAGE_TIMES.items():
        s(stage, _fmt_time(elapsed))
    lines.append(dash)
    s("TOTAL", _fmt_time(total_sec))
    lines.append("")

    h("3. NLP CLASSIFICATION  (vs ground-truth ISCO L3 labels from matched_code)")
    s("Accuracy",  f"{nlp_metrics.get('accuracy',  0.0):.4f}")
    s("Precision", f"{nlp_metrics.get('precision', 0.0):.4f}")
    s("Recall",    f"{nlp_metrics.get('recall',    0.0):.4f}")
    s("F1 Score",  f"{nlp_metrics.get('f1',        0.0):.4f}")
    lines.append("")

    h("4. CAREER PREDICTION  (Ensemble)")
    s("Persistence baseline Top-1", f"{ensemble_results['persistence_top1']*100:.2f}%")
    s("Ensemble Top-1",             f"{ensemble_results['top_1']*100:.2f}%")
    s("Ensemble Top-3",             f"{ensemble_results['top_3']*100:.2f}%")
    s("Ensemble Top-5",             f"{ensemble_results['top_5']*100:.2f}%")
    s("Improvement vs baseline",
      f"{(ensemble_results['top_1']-ensemble_results['persistence_top1'])*100:+.2f}%")
    lines.append("")

    h("5. PCA EXPLAINED VARIANCE")
    s("PC1", f"{pca_var[0]:.1f}%")
    s("PC2", f"{pca_var[1]:.1f}%")
    s("Cumulative PC1+PC2", f"{pca_var[0]+pca_var[1]:.1f}%")
    lines.append("")

    h("6. CLUSTERING SUMMARY")
    for _, row in cs2.iterrows():
        s(f"  Cluster {int(row['cluster'])} -- {row.get('archetype','?')}",
          f"n={int(row['n_users'])}  drift={row['mean_drift']:.3f}  "
          f"loop={row['mean_loop']:.3f}")
    lines.append("")

    h("7. DRIFT SUMMARY")
    s("Users with drift scores", f"{len(risk_df):,}")
    s("Median drift score",      f"{risk_df['drift_score'].median():.4f}")
    s("Max drift score",         f"{risk_df['drift_score'].max():.4f}")
    s("Min drift score",         f"{risk_df['drift_score'].min():.4f}")
    lines.append("")

    h("8. OUTPUT LOCATIONS")
    paths = {
        "Figures":      cfg["figures_dir"],
        "Tables":       cfg["tables_dir"],
        "Metrics":      cfg["metrics_dir"],
        "Checkpoints":  cfg["ckpt_dir"],
        "Logs":         cfg["logs_dir"],
        "Summaries":    cfg["summaries_dir"],
        "Model":        cfg["model_dir"],
        "Embeddings":   cfg["cache_dir"],
    }
    for name, path in paths.items():
        s(name, path)
    lines.append(sep)

    txt = "\n".join(lines)
    out = Path(cfg["summaries_dir"]) / "final_pipeline_summary.txt"
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(txt)
    log.info("  Summary saved -> %s", out)
    print(txt)


def print_summary(
    nlp_metrics: dict,
    ensemble_results: dict,
    pca_var: np.ndarray,
    cfg: dict,
    dataset_hash: str = "",
    df_stats: Optional[dict] = None,
    risk_df: Optional[pd.DataFrame] = None,
    cs2: Optional[pd.DataFrame] = None,
    n_transitions: int = 0,
    device: str = "cpu",
) -> None:
    sep = "=" * 65
    log.info("\n%s", sep)
    log.info("  CAREER DRIFT PIPELINE v5 -- FINAL SUMMARY  (Phase 1)")
    log.info("%s", sep)
    log.info("  Architecture")
    log.info("    Occupation labels : matched_code -> ISCO L3 extraction (algebraic, no regex)")
    log.info("    NLP fallback      : SBERT all-MiniLM-L6-v2 + LightGBM (16.2%% of rows)")
    log.info("    Training signal   : ground-truth ISCO codes from matched_code")
    log.info("    Transition matrix : 125x125 ISCO L3 groups (was 27x27)")
    log.info("  %s", "-" * 60)
    log.info("  NLP ISCO L3 Classification  (vs ground-truth ISCO labels)")
    log.info("    Accuracy  : %.4f", nlp_metrics.get("accuracy",  0.0))
    log.info("    Precision : %.4f", nlp_metrics.get("precision", 0.0))
    log.info("    Recall    : %.4f", nlp_metrics.get("recall",    0.0))
    log.info("    F1 Score  : %.4f", nlp_metrics.get("f1",        0.0))
    log.info("  %s", "-" * 60)
    log.info("  Career Prediction  (Ensemble)")
    log.info("    Persistence Baseline Top-1 : %.2f%%",
             ensemble_results["persistence_top1"] * 100)
    log.info("    Ensemble Top-1             : %.2f%%", ensemble_results["top_1"]*100)
    log.info("    Ensemble Top-3             : %.2f%%", ensemble_results["top_3"]*100)
    log.info("    Ensemble Top-5             : %.2f%%", ensemble_results["top_5"]*100)
    log.info("    Ensemble vs Baseline       : %+.2f%%",
             (ensemble_results["top_1"] - ensemble_results["persistence_top1"]) * 100)
    log.info("  %s", "-" * 60)
    log.info("  PCA Explained Variance")
    log.info("    PC1: %.1f%%   PC2: %.1f%%   Cum(1+2): %.1f%%",
             pca_var[0], pca_var[1], pca_var[0] + pca_var[1])
    log.info("%s", sep)

    ALL_OUTPUTS = {
        "figures":     ["nlp_confusion_matrix.png", "nlp_confidence_distribution.png",
                        "sector_distribution.png", "sequence_length_distribution.png",
                        "pca_sector_profiles.png", "career_drift_distribution.png",
                        "transition_heatmap.png", "top_transitions.png",
                        "market_drift.png", "user_clusters.png",
                        "cluster_sector_breakdown.png", "accuracy_comparison.png",
                        "association_rules_lift.png"],
        "tables":      ["sector_distribution.csv", "nlp_confusion_matrix.csv",
                        "drift_risk_scores.csv", "association_rules.csv"],
        "metrics":     ["nlp_metrics.json", "nlp_classification_report.txt",
                        "final_metrics.json", "run_config.json"],
        "checkpoints": ["cleaned_data.parquet", "labeled_data.parquet",
                        "nlp_predicted.parquet", "transitions.parquet"],
        "logs":        ["pipeline.log", "environment_info.json"],
        "summaries":   ["final_pipeline_summary.txt"],
    }
    log.info("Output file manifest:")
    dir_map = {
        "figures":     cfg["figures_dir"],
        "tables":      cfg["tables_dir"],
        "metrics":     cfg["metrics_dir"],
        "checkpoints": cfg["ckpt_dir"],
        "logs":        cfg["logs_dir"],
        "summaries":   cfg["summaries_dir"],
    }
    for sub, files in ALL_OUTPUTS.items():
        root = Path(dir_map[sub])
        for f in files:
            exists = (root / f).exists()
            log.info("  %s  %s/%s", "v" if exists else "x", sub, f)

    export_final_metrics(nlp_metrics, ensemble_results, pca_var, cfg,
                         dataset_hash=dataset_hash, df_stats=df_stats)

    if risk_df is not None and cs2 is not None and df_stats is not None:
        write_pipeline_summary(
            nlp_metrics, ensemble_results, pca_var, cfg,
            df_stats, risk_df, cs2, n_transitions, device, dataset_hash,
        )

    print_timing_summary()


# ===========================================================================
# HELPERS
# ===========================================================================

def _annotate_split(
    split_df: pd.DataFrame,
    user_cluster_map: pd.DataFrame,
    person_univ: Dict,
    transition_matrices: dict,
) -> pd.DataFrame:
    """Merge archetype + univ_flag + win_key into a split DataFrame."""
    split_df = (
        split_df
        .drop(columns=["archetype"], errors="ignore")
        .merge(user_cluster_map[["person_id", "archetype"]], on="person_id", how="left")
    )
    split_df["archetype"] = split_df["archetype"].fillna("Stable Specialist")
    split_df["univ_flag"] = split_df["person_id"].map(person_univ).fillna(False)
    split_df["win_key"]   = split_df["year"].apply(
        lambda yr: get_win_key(yr, transition_matrices)
    )
    return split_df


# ===========================================================================
# MAIN
# ===========================================================================

def main() -> None:
    cfg = parse_args()
    make_dirs(cfg)

    global log
    log = _setup_logging(Path(cfg["logs_dir"]) / "pipeline.log")

    seed_everything(cfg["seed"])
    device = detect_device_and_log_env(cfg)
    save_run_config(cfg, device)

    log.info("Pipeline configuration:")
    for k, v in cfg.items():
        log.info("  %-20s : %s", k, v)

    # ------------------------------------------------------------------ 1
    ckpt_clean = Path(cfg["ckpt_dir"]) / "cleaned_data.parquet"
    with timed("1. Data loading & cleaning"):
        if cfg["use_cache"] and ckpt_clean.exists():
            try:
                log.info("  Resuming from checkpoint: %s", ckpt_clean)
                df = pd.read_parquet(ckpt_clean)
            except Exception as exc:
                log.warning("  Checkpoint load failed (%s) -- recomputing", exc)
                df = clean_data(load_data(cfg["data_dir"]))
                df.to_parquet(ckpt_clean, index=False)
        else:
            df = clean_data(load_data(cfg["data_dir"]))
            df.to_parquet(ckpt_clean, index=False)
            log.info("  Checkpoint saved -> %s", ckpt_clean)

    # ------------------------------------------------------------------ 2
    # Phase 1: ISCO L3 occupation label assignment from matched_code.
    # No regex. No heuristics. Labels extracted algebraically.
    # NLP fallback rows are flagged (needs_nlp=True) for Stage 3.
    ckpt_labeled = Path(cfg["ckpt_dir"]) / "labeled_data.parquet"
    with timed("2. Occupation label assignment (ISCO L3 from matched_code)"):
        if cfg["use_cache"] and ckpt_labeled.exists():
            try:
                log.info("  Resuming from checkpoint: %s", ckpt_labeled)
                df = pd.read_parquet(ckpt_labeled)
            except Exception as exc:
                log.warning("  Checkpoint load failed (%s) -- recomputing", exc)
                df = assign_occupation_labels(df)
                df.to_parquet(ckpt_labeled, index=False)
        else:
            df = assign_occupation_labels(df)
            df.to_parquet(ckpt_labeled, index=False)
            log.info("  Checkpoint saved -> %s", ckpt_labeled)

    # Compute dataset fingerprint AFTER ISCO pipeline (combined_text is now present)
    dataset_hash = compute_dataset_hash(df)
    log.info("  Dataset fingerprint (SHA-256[:16]) : %s", dataset_hash)

    # ------------------------------------------------------------------ 3
    ckpt_nlp  = Path(cfg["ckpt_dir"]) / "nlp_predicted.parquet"
    ckpt_lgbm = Path(cfg["model_dir"]) / "lgbm_isco_l3_classifier.joblib"  # updated name
    with timed("3. NLP ISCO L3 classification (SBERT + LightGBM)"):
        nlp_loaded = False
        if cfg["use_cache"] and cache_is_valid(ckpt_nlp, dataset_hash):
            try:
                log.info("  Resuming from NLP checkpoint: %s", ckpt_nlp)
                df = pd.read_parquet(ckpt_nlp)
                classifier = joblib.load(ckpt_lgbm) if ckpt_lgbm.exists() else None
                nlp_metrics = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
                log.info("  NLP ISCO L3 metrics not re-evaluated from cache "
                         "(use --no-cache to recompute)")
                embedding_model = None
                nlp_loaded = True
            except Exception as exc:
                log.warning("  NLP checkpoint load failed (%s) -- retraining", exc)

        if not nlp_loaded:
            classifier, embedding_model, nlp_metrics = train_nlp_classifier(
                df, cfg, device, dataset_hash
            )
            df = predict_nlp_sectors(df, classifier, embedding_model, cfg, device, dataset_hash)
            df.to_parquet(ckpt_nlp, index=False)
            write_cache_metadata(ckpt_nlp, dataset_hash, cfg, device)
            log.info("  Checkpoint saved -> %s", ckpt_nlp)

    if embedding_model is not None:
        del embedding_model
    gc.collect()

    # ------------------------------------------------------------------ 4
    ckpt_tr = Path(cfg["ckpt_dir"]) / "transitions.parquet"
    with timed("4. Transition modeling"):
        if cfg["use_cache"] and ckpt_tr.exists():
            try:
                log.info("  Resuming from checkpoint: %s", ckpt_tr)
                transitions = pd.read_parquet(ckpt_tr)
                df, transitions, person_univ = build_transitions(df)
            except Exception as exc:
                log.warning("  Transitions checkpoint load failed (%s) -- recomputing", exc)
                df, transitions, person_univ = build_transitions(df)
                transitions.to_parquet(ckpt_tr, index=False)
        else:
            df, transitions, person_univ = build_transitions(df)
            transitions.to_parquet(ckpt_tr, index=False)
            log.info("  Checkpoint saved -> %s", ckpt_tr)

    # Collect dataset statistics for summary
    df_stats = {
        "total_rows":         len(df),
        "unique_users":       df["person_id"].nunique(),
        "unique_l3_groups":   df["esco_sector"].nunique(),
        "year_range":         f"{df['year'].min()}-{df['year'].max()}",
        "unique_job_titles":  df["matched_label"].nunique(),
        "direct_code_pct":    f"{(df.get('needs_nlp', pd.Series([False]*len(df))) == False).mean()*100:.1f}%",
        "nlp_fallback_pct":   f"{(df.get('needs_nlp', pd.Series([False]*len(df))) == True).mean()*100:.1f}%",
    }

    # ------------------------------------------------------------------ 5
    with timed("5. PCA on transition profiles"):
        pca_var, _ = run_pca(transitions, cfg)

    # ------------------------------------------------------------------ 6
    with timed("6. Ensemble prediction system"):
        all_sectors_tr = sorted(
            set(transitions["esco_sector"]) | set(transitions["next_sector"])
        )
        n_sec_tr    = len(all_sectors_tr)
        sect_idx_tr = {s: i for i, s in enumerate(all_sectors_tr)}

        sr_tr = (
            transitions
            .assign(is_self=lambda x: x["esco_sector"] == x["next_sector"])
            .groupby("esco_sector")["is_self"].mean()
        )
        self_rates_tr = {s: float(sr_tr.get(s, 0.2)) for s in all_sectors_tr}
        sloop_mult_tr = {s: 1.0 + 2.0 * self_rates_tr[s] for s in all_sectors_tr}

        train_data, val_data, test_data, _ = stratified_user_split(
            df, transitions, cfg["seed"]
        )
        components = build_ensemble_components(
            train_data, all_sectors_tr, sect_idx_tr,
            self_rates_tr, sloop_mult_tr, person_univ, n_sec_tr, cfg["seed"],
        )
        user_cluster_map = components["user_cluster_map"]
        tm               = components["transition_matrices"]

        val_data  = _annotate_split(val_data,  user_cluster_map, person_univ, tm)
        test_data = _annotate_split(test_data, user_cluster_map, person_univ, tm)

        sector_best_w = tune_sector_weights(
            val_data, all_sectors_tr, sect_idx_tr,
            self_rates_tr, components, n_sec_tr, cfg["seed"],
        )
        ensemble_results = evaluate_ensemble(
            test_data, all_sectors_tr, sect_idx_tr,
            sector_best_w, components, n_sec_tr,
        )

    del train_data, val_data, test_data; gc.collect()

    # ------------------------------------------------------------------ 7
    all_sectors   = sorted(set(transitions["esco_sector"]) | set(transitions["next_sector"]))
    sector_to_idx = {s: i for i, s in enumerate(all_sectors)}

    with timed("7. Drift analysis & clustering"):
        risk_df, cs2, dom_sec_df = compute_drift_and_clusters(
            df, transitions, components["global_matrix"],
            sector_to_idx, all_sectors, cfg["seed"], cfg,
        )

    # ------------------------------------------------------------------ 8
    with timed("8. Association rules"):
        assoc_rules = compute_association_rules(transitions, cfg)

    # ------------------------------------------------------------------ 9
    if cfg.get("plots", True):
        with timed("9. Visualisations"):
            run_all_plots(
                df, transitions, risk_df, cs2, dom_sec_df,
                assoc_rules, components["market_drift"],
                ensemble_results, cfg["figures_dir"],
            )
    else:
        log.info("Visualisations skipped (--no-plots)")

    # ----------------------------------------------------------------- 10
    print_summary(
        nlp_metrics, ensemble_results, pca_var, cfg,
        dataset_hash=dataset_hash,
        df_stats=df_stats,
        risk_df=risk_df,
        cs2=cs2,
        n_transitions=len(transitions),
        device=device,
    )
    log.info("Pipeline complete.")


# ===========================================================================
if __name__ == "__main__":
    main()