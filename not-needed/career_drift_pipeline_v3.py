"""
Career Drift Trajectory Analysis — Production Pipeline  v3
===========================================================
Architecture : Weak Supervision → SBERT + LightGBM → Full Ensemble

Weak-supervision rationale
---------------------------
The original R implementation (career_pred12.R) uses hand-crafted expert
regex rules (map_sectors / broad_map) to assign esco_sector labels.
We port those rules exactly into Python, run them to produce bootstrap
pseudo-labels, then train an SBERT + LightGBM classifier on those labels.
The NLP model is the FINAL sector classifier; the regex system is used
only internally as a weak supervisor.  This gives us:
  • expert domain knowledge (27 ESCO-aligned sectors) as a starting point
  • semantic generalisation to unseen / variant job titles
  • better handling of synonyms, abbreviations, informal phrasing
  • production-grade reproducibility and measurable improvement over regex

References: Ratner et al. (Snorkel, 2017); Dawid & Skene (1979).

v2 improvements (all methodology-preserving):
  1.  Explicit GPU/CPU detection with torch.cuda + named logging
  2.  Chunked embedding generation — safe for 1 M+ rows
  3.  Intermediate parquet checkpoints at every major stage
  4.  Full resume capability — every stage is skip-if-cached
  5.  Per-stage wall-clock timing logs + final timing summary
  6.  Aggressive gc.collect() + del after every large allocation
  7.  Configurable --embed-batch-size and --embed-chunk-size (CLI)
  8.  Organised output tree: outputs/{plots,metrics,checkpoints}/
  9.  Final metrics export to JSON + CSV (NLP report, confusion, sector dist)
  10. All research semantics fully preserved

Usage
-----
    python career_drift_pipeline.py [options]

    --data-dir          DIR  directory with train/validation/test parquet
    --output-dir        DIR  root output directory     (default: outputs)
    --cache-dir         DIR  .npy embedding caches     (default: cache)
    --model-dir         DIR  model checkpoints         (default: models)
    --seed              INT  global random seed        (default: 42)
    --embed-batch-size  INT  SBERT batch size          (default: 128)
    --embed-chunk-size  INT  rows per embedding chunk  (default: 50000)
    --confidence-thr    FLT  NLP confidence threshold  (default: 0.45)
    --no-cache               force re-computation of all stages
    --no-plots               skip all matplotlib output
"""

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import argparse
import gc
import itertools
import json
import logging
import os
import random
import re
import time
import warnings
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
# 0.  CONFIGURATION & DIRECTORY SETUP
# ===========================================================================

def parse_args() -> dict:
    p = argparse.ArgumentParser(
        description="Career Drift Trajectory Analysis — Production Pipeline v2"
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
    p.add_argument("--confidence-thr",   type=float, default=0.45,
                   help="Min NLP confidence; rows below are dropped")
    p.add_argument("--no-cache",   action="store_true")
    p.add_argument("--no-plots",   action="store_true")
    args = p.parse_args()

    od  = Path(args.output_dir)
    cfg = {
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
        # derived sub-directories
        "plots_dir":        str(od / "plots"),
        "metrics_dir":      str(od / "metrics"),
        "ckpt_dir":         str(od / "checkpoints"),
    }
    return cfg


def make_dirs(cfg: dict) -> None:
    for key in ("output_dir", "cache_dir", "model_dir",
                "plots_dir", "metrics_dir", "ckpt_dir"):
        Path(cfg[key]).mkdir(parents=True, exist_ok=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    log.info("Global seed : %d", seed)


# ===========================================================================
# 0b.  DEVICE DETECTION
# ===========================================================================

def detect_device() -> str:
    """
    Detect CUDA availability using torch.cuda.
    Logs device type, GPU name, CUDA version, and VRAM if available.
    Returns 'cuda' or 'cpu' for SentenceTransformer device argument.
    """
    try:
        import torch
        if torch.cuda.is_available():
            device   = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            cuda_ver = torch.version.cuda
            vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
            log.info("Device     : cuda")
            log.info("GPU        : %s", gpu_name)
            log.info("CUDA       : %s", cuda_ver)
            log.info("VRAM       : %.1f GB", vram_gb)
        else:
            device = "cpu"
            log.info("CUDA unavailable -- using CPU.")
    except ImportError:
        device = "cpu"
        log.info("torch not installed -- using CPU.")
    return device


# ===========================================================================
# 1.  DATA LOADING
# ===========================================================================

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
# 3.  WEAK SUPERVISION -- REGEX BOOTSTRAP LABELLING
# ===========================================================================
#
#  The regex rules below are a COMPLETE Python port of R's map_sectors() and
#  broad_map() functions.  They are the SOLE source of esco_sector_regex
#  bootstrap labels used to train the NLP classifier.
#  They are NOT the final sector assignment -- that role belongs to the NLP model.
# ---------------------------------------------------------------------------

_MAP_SECTORS_RULES: List[Tuple[re.Pattern, str]] = [
    (re.compile(
        r"software engineer|software developer|software architect|web developer|"
        r"web designer|front.?end|back.?end|full.?stack|mobile developer|"
        r"android developer|ios developer|app developer|devops engineer|"
        r"devops specialist|site reliability|cloud engineer|platform engineer|"
        r"qa engineer|quality assurance engineer|test engineer|automation engineer|"
        r"data engineer|machine learning engineer|ai engineer|nlp engineer|"
        r"embedded engineer|firmware engineer|game developer|systems programmer"
    ), "Software & IT Development"),
    (re.compile(
        r"teacher|lecturer|professor|tutor|instructor|educator|trainer|"
        r"teaching assistant|learning support|head teacher|principal|\bschool\b|"
        r"university lecturer|\bacademic\b|curriculum|\bfaculty\b|\bdean\b|"
        r"nursery teacher|primary teacher|secondary teacher|special education|"
        r"eyfs|sen teacher|class teacher|form teacher|maths teacher|"
        r"science teacher|pe teacher|esl teacher|tefl teacher|"
        r"english language teacher"
    ), "Education - Teaching"),
    (re.compile(
        r"registered nurse|staff nurse|charge nurse|ward nurse|theatre nurse|"
        r"mental health nurse|community nurse|neonatal nurse|paediatric nurse|"
        r"nursing manager|nursing director|nursing assistant|healthcare assistant|"
        r"\bhca\b|midwife|midwifery|paramedic|ambulance technician|"
        r"physiotherapist|occupational therapist|speech therapist|radiographer|"
        r"sonographer|dietitian|care worker|care assistant|support worker|"
        r"health visitor|practice nurse|nurse practitioner|pharmacy technician|"
        r"biomedical scientist"
    ), "Healthcare - Nursing & Allied"),
    (re.compile(
        r"\bdoctor\b|general practitioner|\bgp\b|physician|surgeon|"
        r"specialist physician|anaesthetist|radiologist|pathologist|psychiatrist|"
        r"paediatrician|cardiologist|neurologist|oncologist|dermatologist|"
        r"\bdentist\b|dental surgeon|ophthalmologist|optometrist|\bpharmacist\b|"
        r"medical officer|medical director|clinical specialist|clinical director|"
        r"clinical lead|gp registrar|foundation doctor|junior doctor|house officer"
    ), "Healthcare - Clinical"),
    (re.compile(
        r"mechanical engineer|civil engineer|structural engineer|electrical engineer|"
        r"electronic engineer|manufacturing engineer|process engineer|"
        r"chemical engineer|industrial engineer|quality engineer|design engineer|"
        r"aerospace engineer|automotive engineer|maintenance engineer|plant engineer|"
        r"commissioning engineer|controls engineer|instrumentation engineer|"
        r"production engineer|machine operator|plant operator|toolmaker|machinist|"
        r"\bfitter\b|fabricator|assembler|welder|production supervisor|"
        r"engineering manager|chief engineer|lead engineer|principal engineer"
    ), "Engineering & Manufacturing"),
    (re.compile(
        r"\baccountant\b|\bauditor\b|finance manager|financial analyst|"
        r"management accountant|tax advisor|tax manager|\bpayroll\b|bookkeeper|"
        r"credit controller|accounts payable|accounts receivable|"
        r"financial controller|finance director|\bcfo\b|chief financial|"
        r"\btreasury\b|budget analyst|cost accountant|chartered accountant|"
        r"fp.a|financial reporting|accounts manager|finance officer|"
        r"finance analyst|finance coordinator"
    ), "Finance & Accounting"),
    (re.compile(
        r"sales manager|sales director|account manager|account executive|"
        r"business development|sales executive|sales representative|"
        r"sales consultant|sales engineer|pre.?sales|inside sales|regional sales|"
        r"national sales|area sales|territory manager|commercial manager|"
        r"commercial director|revenue manager|\bbdr\b|\bsdr\b"
    ), "Sales & Business Development"),
    (re.compile(
        r"marketing manager|marketing director|brand manager|digital marketing|"
        r"marketing executive|marketing analyst|\bseo\b|social media manager|"
        r"content manager|communications manager|pr manager|public relations|"
        r"communications director|campaign manager|growth manager|email marketing|"
        r"performance marketing|media manager|marketing coordinator"
    ), "Marketing & Communications"),
    (re.compile(
        r"project manager|programme manager|project director|project coordinator|"
        r"project lead|delivery manager|\bpmo\b|project management office|"
        r"scrum master|agile coach|product manager|product owner|release manager|"
        r"change manager|implementation manager"
    ), "Project & Programme Management"),
    (re.compile(
        r"operations manager|operations director|general manager|business manager|"
        r"site manager|branch manager|centre manager|operations coordinator|"
        r"operations analyst|business analyst|process improvement|\blean\b|"
        r"six sigma|continuous improvement|facilities manager|office manager|"
        r"administration manager"
    ), "Operations & General Management"),
    (re.compile(
        r"hr manager|human resources manager|hr director|recruitment consultant|"
        r"\brecruiter\b|talent acquisition|hr business partner|hr advisor|"
        r"hr officer|hr coordinator|learning and development|l.d manager|"
        r"training manager|organisational development|people manager|"
        r"workforce planning|compensation|benefits manager|employee relations|"
        r"\bhrbp\b"
    ), "Human Resources & Recruitment"),
    (re.compile(
        r"it manager|it director|head of it|it support|systems administrator|"
        r"network engineer|network administrator|infrastructure engineer|"
        r"systems engineer|it analyst|\bhelpdesk\b|service desk|it technician|"
        r"desktop support|it operations"
    ), "IT Infrastructure & Networks"),
    (re.compile(
        r"ict manager|ict director|technology manager|technology director|"
        r"digital transformation|solutions architect|enterprise architect|"
        r"it consultant|technology consultant|\bcio\b|\bcto\b|chief information|"
        r"chief technology|digital director|head of technology|technical director|"
        r"chief digital"
    ), "ICT Management & Consulting"),
    (re.compile(
        r"data scientist|data analyst|business intelligence|bi developer|bi analyst|"
        r"data manager|analytics manager|data architect|statistician|"
        r"quantitative analyst|reporting analyst|data consultant|insights analyst|"
        r"research analyst|data lead"
    ), "Data Science & Analytics"),
    (re.compile(
        r"\bcyber\b|security analyst|security engineer|security architect|"
        r"information security|penetration tester|\bpentest\b|soc analyst|"
        r"vulnerability|threat intelligence|\bforensic\b|incident response|"
        r"\bgdpr\b|data protection officer|information assurance|security manager|"
        r"security consultant"
    ), "Cybersecurity & Compliance"),
    (re.compile(
        r"solicitor|barrister|\blawyer\b|legal advisor|legal counsel|legal manager|"
        r"paralegal|legal executive|compliance manager|compliance officer|"
        r"legal director|general counsel|regulatory affairs|regulatory manager|"
        r"legal officer|governance manager|contract manager|legal specialist|"
        r"conveyancer"
    ), "Legal & Compliance"),
    (re.compile(
        r"investment banker|investment analyst|fund manager|portfolio manager|"
        r"wealth manager|financial advisor|financial planner|stockbroker|"
        r"equity trader|risk analyst|risk manager|underwriter|actuary|"
        r"mortgage advisor|mortgage broker|loan officer|asset manager|"
        r"private equity|hedge fund|credit analyst|capital markets|banking analyst|"
        r"insurance analyst"
    ), "Banking, Finance & Insurance"),
    (re.compile(
        r"social worker|community worker|probation officer|youth worker|case manager|"
        r"welfare officer|housing officer|counsellor|psychotherapist|"
        r"mental health support worker|disability support worker|"
        r"rehabilitation specialist|child protection officer|safeguarding officer|"
        r"outreach worker|family support|foster care|addiction counsellor|"
        r"drug and alcohol worker|advocacy worker"
    ), "Social Work & Community Services"),
    (re.compile(
        r"construction manager|site engineer|quantity surveyor|building manager|"
        r"construction director|\barchitect\b|urban planner|interior designer|"
        r"electrician|\bplumber\b|carpenter|bricklayer|hvac engineer|"
        r"building surveyor|building inspector|planning engineer|planning manager|"
        r"estimator|scaffolder|construction worker|fit.?out manager|property manager|"
        r"structural technician|architectural technician"
    ), "Construction & Architecture"),
    (re.compile(
        r"\bceo\b|chief executive|managing director|\bcoo\b|chief operating|"
        r"vice president|\bsvp\b|\bevp\b|group director|global director|"
        r"executive director|country manager|general director|chief officer|"
        r"\bpresident\b|board member|chairman|non.?executive director"
    ), "Senior Management & C-Suite"),
    (re.compile(
        r"supply chain manager|supply chain director|logistics manager|"
        r"procurement manager|purchasing manager|warehouse manager|"
        r"inventory manager|distribution manager|transport manager|freight manager|"
        r"shipping manager|materials manager|demand planner|supply planner|"
        r"fulfilment manager|\bbuyer\b"
    ), "Supply Chain & Logistics"),
    (re.compile(
        r"retail manager|store manager|shop manager|hotel manager|"
        r"hospitality manager|restaurant manager|catering manager|events manager|"
        r"venue manager|tourism manager|leisure manager|food and beverage|"
        r"f.b manager|front of house|bar manager|floor manager"
    ), "Retail, Hospitality & Events"),
    (re.compile(
        r"customer service manager|customer success|customer experience|"
        r"call centre manager|contact centre manager|customer relations|"
        r"client services manager|customer support manager|client manager|"
        r"customer care|customer advisor|client advisor"
    ), "Customer Service & Support"),
    (re.compile(
        r"policy officer|civil servant|\bgovernment\b|\bcouncil\b|local authority|"
        r"public sector|regulatory officer|administration officer|\bclerk\b|"
        r"executive officer|personal assistant|executive assistant|"
        r"document control|records manager|public administrator|policy analyst|"
        r"parliamentary"
    ), "Public Sector & Administration"),
    (re.compile(
        r"environment|sustainability|health and safety|hse manager|ehs manager|"
        r"she manager|esg manager|renewable energy|energy manager|climate change|"
        r"carbon manager|waste manager|ecology|conservation|environmental manager|"
        r"safety manager|environmental consultant|sustainability manager"
    ), "Environment, Safety & Sustainability"),
    (re.compile(
        r"\bresearcher\b|research officer|\bscientist\b|laboratory manager|"
        r"postdoctoral|phd researcher|principal investigator|research director|"
        r"research manager|research fellow|biologist|biochemist|chemist|physicist|"
        r"epidemiologist|clinical researcher|r.d manager|research scientist|"
        r"research associate"
    ), "Research & Academia"),
    (re.compile(
        r"graphic designer|creative director|art director|media manager|"
        r"content creator|\bjournalist\b|\beditor\b|photographer|videographer|"
        r"animator|illustrator|broadcast|\bfilm\b|copywriter|ux designer|"
        r"ui designer|visual designer|interaction designer"
    ), "Creative, Media & Design"),
]


def map_sectors(txt: str) -> str:
    """Port of R map_sectors(). First match wins (replicates R fcase semantics)."""
    for pattern, label in _MAP_SECTORS_RULES:
        if pattern.search(txt):
            return label
    return "Other & Unclassified"


_BROAD_SIMPLE: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"teach|school|educat|tutor|lectur|instruct|learn|curriculum|classroom|pupil|student"),
     "Education - Teaching"),
    (re.compile(r"nurs|care assistant|healthcare assistant|\bhca\b|midwif|paramedic|physiother|radiograph|dietit|pharmacist"),
     "Healthcare - Nursing & Allied"),
    (re.compile(r"\bdoctor\b|physician|surgeon|\bgp\b|psychiatr|dentist|optometr|consultant physician|medical officer"),
     "Healthcare - Clinical"),
    (re.compile(r"engineer|manufactur|machine|fitter|welder|assembl|production|quality inspector|commissioning"),
     "Engineering & Manufacturing"),
    (re.compile(r"accountant|auditor|payroll|bookkeep|treasury|\bcfo\b|financial controller|fp.a|accounts payable"),
     "Finance & Accounting"),
    (re.compile(r"market|brand|digital marketing|seo|social media|campaign|content|advertising|public relat"),
     "Marketing & Communications"),
    (re.compile(r"project|programme|scrum|agile|\bpmo\b|product owner|delivery|sprint|waterfall|kanban"),
     "Project & Programme Management"),
    (re.compile(r"operat|general manag|business manag|facilities|office manag|process|lean|six sigma"),
     "Operations & General Management"),
    (re.compile(r"supply chain|logistic|procurement|purchas|warehouse|inventor|distribut|transport|freight|shipping|\bbuyer\b"),
     "Supply Chain & Logistics"),
    (re.compile(r"retail|store|shop|hotel|hospitality|restaurant|catering|events|venue|tourism|leisure|food.beverage|bar manager"),
     "Retail, Hospitality & Events"),
    (re.compile(r"customer serv|call cent|contact cent|helpdesk|support desk|client serv|customer success"),
     "Customer Service & Support"),
    (re.compile(r"software|developer|programmer|coder|web dev|app dev|devops|cloud|frontend|backend|fullstack"),
     "Software & IT Development"),
    (re.compile(r"hr |human resourc|recruit|talent|l.d|learning.develop|training manager|people manager|workforce"),
     "Human Resources & Recruitment"),
    (re.compile(r"it support|network|infrastructure|sysadmin|server|desktop support|systems admin|helpdesk"),
     "IT Infrastructure & Networks"),
    (re.compile(r"data scientist|data analyst|analytics|intelligence|insight|statistician|sql|python|tableau|reporting"),
     "Data Science & Analytics"),
    (re.compile(r"construct|architect|build|civil|structural|survey|planning|estimat|site manag|scaff"),
     "Construction & Architecture"),
    (re.compile(r"legal|law|solicitor|compli|regulat|gdpr|contract|barrister|paralegal|governance"),
     "Legal & Compliance"),
    (re.compile(r"bank|invest|fund|wealth|insur|actuar|mortgage|underwrite|trading|broker|financial advis"),
     "Banking, Finance & Insurance"),
    (re.compile(r"social work|communit|youth|welfare|outreach|counsell|psychother|safeguard|advocacy|family support"),
     "Social Work & Community Services"),
    (re.compile(r"\bceo\b|managing director|chief executive|vice president|\bsvp\b|\bcoo\b|non.?exec|chairman|president"),
     "Senior Management & C-Suite"),
    (re.compile(r"cyber|security|infosec|soc|pentest|vulnerab|threat|forensic|incident|identity access"),
     "Cybersecurity & Compliance"),
    (re.compile(r"public sector|civil serv|government|council|local authority|policy|regulatory officer|clerk|parliament"),
     "Public Sector & Administration"),
    (re.compile(r"environment|sustainab|health.safety|\bhse\b|\behs\b|renewable|energy manag|climate|carbon|waste|ecology"),
     "Environment, Safety & Sustainability"),
    (re.compile(r"research|scientist|laborator|postdoc|phd|investigat|biolog|biochem|chemist|physicist|epidemiolog"),
     "Research & Academia"),
    (re.compile(r"ict|information technolog|it manager|it director|it consultant|technology manager|digital transform|solutions architect|\bcto\b|\bcio\b"),
     "ICT Management & Consulting"),
    # catchall -- always last
    (re.compile(r"manager|officer|analyst|specialist|advisor|coordinator|director|consultant|assistant|lead|supervisor"),
     "Operations & General Management"),
]

_BROAD_SALES_KW   = re.compile(r"sales|account executive|business development|\bbdr\b|\bsdr\b|territory|revenue|commercial")
_BROAD_SALES_ROLE = re.compile(r"manager|executive|director|officer|lead|head|represent|consult")
_BROAD_CREAT_KW   = re.compile(r"design|graphic|creative|media|content|writer|journalist|editor|photog|video|animat|illustrat|film|broadcast|ux|ui")
_BROAD_CREAT_ROLE = re.compile(r"manager|director|specialist|lead|producer|designer")


def broad_map(txt: str) -> str:
    """Port of R broad_map(). Order exactly matches R fcase() evaluation sequence."""
    for pat, label in _BROAD_SIMPLE[:5]:
        if pat.search(txt):
            return label
    if _BROAD_SALES_KW.search(txt) and _BROAD_SALES_ROLE.search(txt):
        return "Sales & Business Development"
    for pat, label in _BROAD_SIMPLE[5:-1]:
        if pat.search(txt):
            return label
    if _BROAD_CREAT_KW.search(txt) and _BROAD_CREAT_ROLE.search(txt):
        return "Creative, Media & Design"
    if _BROAD_SIMPLE[-1][0].search(txt):
        return "Operations & General Management"
    return "Other & Unclassified"


def _combine_text(row) -> str:
    title = str(row.get("matched_label", ""))
    desc  = str(row.get("matched_description", ""))
    if desc in ("nan", "None", ""):
        desc = ""
    return f"{title} [SEP] {desc}".strip()


def apply_regex_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    4-pass weak-supervision label generation (mirrors R passes 1 / 2 / 2b / 3).

    Pass 1  : map_sectors() on matched_label
    Pass 2  : map_sectors() on matched_description  (rows still 'Other')
    Pass 2b : broad_map() on label then description (rows still 'Other')
    Pass 3  : drop rows still 'Other & Unclassified'

    Adds esco_sector_regex and combined_text to df.
    Returns a NEW DataFrame -- does not modify in place.
    """
    OTHER    = "Other & Unclassified"
    df       = df.copy()
    has_desc = "matched_description" in df.columns

    log.info("  Pass 1 -- map_sectors() on job title ...")
    df["esco_sector_regex"] = df["matched_label"].astype(str).str.lower().apply(map_sectors)
    n_other = (df["esco_sector_regex"] == OTHER).sum()
    log.info("    'Other' after pass 1 : %s (%.1f%%)", f"{n_other:,}", n_other / len(df) * 100)

    if has_desc:
        idx2 = df.index[df["esco_sector_regex"] == OTHER]
        log.info("  Pass 2 -- map_sectors() on description for %s rows ...", f"{len(idx2):,}")
        new_labels = df.loc[idx2, "matched_description"].astype(str).str.lower().apply(map_sectors)
        rescued    = (new_labels != OTHER).sum()
        df.loc[idx2, "esco_sector_regex"] = new_labels.values
        n_other = (df["esco_sector_regex"] == OTHER).sum()
        log.info("    Rescued %s | 'Other' remaining: %s (%.1f%%)",
                 f"{rescued:,}", f"{n_other:,}", n_other / len(df) * 100)
    else:
        log.info("  Pass 2 -- skipped (no matched_description column)")

    idx2b = df.index[df["esco_sector_regex"] == OTHER]
    log.info("  Pass 2b -- broad_map() rescue for %s rows ...", f"{len(idx2b):,}")
    if len(idx2b) > 0:
        new_labels_2b = df.loc[idx2b, "matched_label"].astype(str).str.lower().apply(broad_map)
        if has_desc:
            still_other = new_labels_2b.index[new_labels_2b == OTHER]
            if len(still_other) > 0:
                new_labels_2b.loc[still_other] = (
                    df.loc[still_other, "matched_description"]
                    .astype(str).str.lower().apply(broad_map).values
                )
        df.loc[idx2b, "esco_sector_regex"] = new_labels_2b.values
        rescued_2b = (new_labels_2b != OTHER).sum()
        n_other    = (df["esco_sector_regex"] == OTHER).sum()
        log.info("    Rescued %s | 'Other' after 2b: %s (%.1f%%)",
                 f"{rescued_2b:,}", f"{n_other:,}", n_other / len(df) * 100)

    before = len(df)
    n_drop  = (df["esco_sector_regex"] == OTHER).sum()
    df = df[df["esco_sector_regex"] != OTHER].copy()
    log.info("  Pass 3 -- removed %s (%.1f%%) | kept %s",
             f"{n_drop:,}", n_drop / before * 100, f"{len(df):,}")

    df["combined_text"] = df.apply(_combine_text, axis=1)
    df = df.sort_values(["person_id", "year"]).reset_index(drop=True)

    sc = df["esco_sector_regex"].value_counts().reset_index()
    sc.columns = ["sector", "N"]
    sc["pct"]  = (sc["N"] / len(df) * 100).round(1)
    log.info("Bootstrap sector distribution:\n%s", sc.to_string(index=False))
    return df


# ===========================================================================
# 4.  NLP SECTOR CLASSIFIER  (SBERT + LightGBM)
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
) -> np.ndarray:
    """
    Generate or load L2-normalised SBERT embeddings in memory-safe chunks.

    For large datasets the encoding runs in chunks of `chunk_size` rows to
    prevent RAM / VRAM spikes.  Each chunk is encoded independently; chunks
    are vstack-assembled only once all chunks are complete.

    Cache contract:
      cache exists + use_cache=True  -> load from .npy (O(1) I/O, no GPU used)
      otherwise                      -> encode in chunks, save, return
    """
    if use_cache and cache_path and cache_path.exists():
        log.info("    Loading cached embeddings [%s]: %s", label, cache_path)
        return np.load(cache_path)

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
        log.info("    Cached -> %s  (%.1f MB)", cache_path, result.nbytes / 1e6)
    return result


def train_nlp_classifier(
    df: pd.DataFrame,
    cfg: dict,
    device: str,
) -> Tuple[LGBMClassifier, SentenceTransformer, dict]:
    """
    Train SBERT + LightGBM sector classifier using weak-supervision bootstrap labels.

    Returns
    -------
    classifier      : fitted LGBMClassifier
    embedding_model : loaded SentenceTransformer (kept for full-dataset prediction)
    nlp_metrics     : dict with accuracy / precision / recall / f1
    """
    seed      = cfg["seed"]
    cache_dir = Path(cfg["cache_dir"])
    model_dir = Path(cfg["model_dir"])
    use_cache = cfg["use_cache"]
    bs        = cfg["embed_batch_size"]
    cs        = cfg["embed_chunk_size"]
    ckpt_path = model_dir / "lgbm_sector_classifier.joblib"

    # NLP train/val/test split -- stratified on bootstrap labels
    # NOTE: this split is independent of the ensemble prediction split (Section 7)
    train_t, temp_t, train_l, temp_l = train_test_split(
        df["combined_text"].tolist(),
        df["esco_sector_regex"].tolist(),
        test_size=0.30,
        stratify=df["esco_sector_regex"],
        random_state=seed,
    )
    val_t, test_t, val_l, test_l = train_test_split(
        temp_t, temp_l,
        test_size=0.50,
        stratify=temp_l,
        random_state=seed,
    )
    log.info("  NLP split -- train: %s | val: %s | test: %s",
             f"{len(train_t):,}", f"{len(val_t):,}", f"{len(test_t):,}")

    log.info("  Loading SBERT: all-MiniLM-L6-v2  (device=%s) ...", device)
    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2",
                                          device=device)

    X_train = _embed_chunked(train_t, embedding_model, cache_dir / "emb_train.npy",
                              use_cache, "train", bs, cs, device)
    X_val   = _embed_chunked(val_t,   embedding_model, cache_dir / "emb_val.npy",
                              use_cache, "val",   bs, cs, device)
    X_test  = _embed_chunked(test_t,  embedding_model, cache_dir / "emb_test.npy",
                              use_cache, "test",  bs, cs, device)

    if use_cache and ckpt_path.exists():
        log.info("  Loading LightGBM checkpoint: %s", ckpt_path)
        classifier = joblib.load(ckpt_path)
    else:
        log.info("  Training LightGBM on %s bootstrap labels ...", f"{len(train_t):,}")
        classifier = LGBMClassifier(
            objective="multiclass",
            n_estimators=400,
            learning_rate=0.05,
            max_depth=10,
            num_leaves=63,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=seed,
            verbose=-1,
        )
        classifier.fit(X_train, train_l, eval_set=[(X_val, val_l)])
        joblib.dump(classifier, ckpt_path)
        log.info("  Checkpoint saved -> %s", ckpt_path)

    # Evaluation
    preds    = classifier.predict(X_test)
    accuracy = accuracy_score(test_l, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_l, preds, average="weighted"
    )
    log.info("  NLP performance (vs bootstrap labels):")
    log.info("    Accuracy  : %.4f", accuracy)
    log.info("    Precision : %.4f", precision)
    log.info("    Recall    : %.4f", recall)
    log.info("    F1 Score  : %.4f", f1)
    report_str = classification_report(test_l, preds)
    print(report_str)

    # Export NLP metrics to metrics/
    metrics_dir = Path(cfg["metrics_dir"])
    nlp_metrics = dict(accuracy=accuracy, precision=precision, recall=recall, f1=f1)

    with open(metrics_dir / "nlp_metrics.json", "w") as fh:
        json.dump({**nlp_metrics, "timestamp": datetime.utcnow().isoformat()}, fh, indent=2)

    with open(metrics_dir / "nlp_classification_report.txt", "w") as fh:
        fh.write(report_str)

    cm = confusion_matrix(test_l, preds, labels=classifier.classes_)
    pd.DataFrame(cm, index=classifier.classes_,
                 columns=classifier.classes_).to_csv(
        metrics_dir / "nlp_confusion_matrix.csv"
    )
    log.info("  NLP metrics saved -> %s", metrics_dir)

    if cfg.get("plots", True):
        fig, ax = plt.subplots(figsize=(16, 14))
        sns.heatmap(cm, cmap="Blues", ax=ax,
                    xticklabels=classifier.classes_,
                    yticklabels=classifier.classes_)
        ax.set_title(
            "NLP Sector Classification -- Confusion Matrix\n(vs bootstrap regex labels)",
            fontsize=12, fontweight="bold",
        )
        plt.xticks(rotation=60, ha="right", fontsize=7)
        plt.yticks(fontsize=7)
        plt.tight_layout()
        out = Path(cfg["plots_dir"]) / "nlp_confusion_matrix.png"
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
) -> pd.DataFrame:
    """
    Apply NLP classifier to the full dataset; assign final esco_sector.
    Chunked embedding prevents memory spikes on large datasets.
    Rows below confidence_thr are dropped (mirrors R Pass 3 logic).
    """
    cache_dir = Path(cfg["cache_dir"])
    bs        = cfg["embed_batch_size"]
    cs        = cfg["embed_chunk_size"]
    use_cache = cfg["use_cache"]
    conf_thr  = cfg["confidence_thr"]

    all_emb = _embed_chunked(
        df["combined_text"].tolist(), embedding_model,
        cache_dir / "emb_full.npy", use_cache, "full dataset", bs, cs, device,
    )

    log.info("  Running LightGBM predict_proba on %s rows ...", f"{len(df):,}")
    probs    = classifier.predict_proba(all_emb)
    preds    = classifier.classes_[np.argmax(probs, axis=1)]
    max_prob = probs.max(axis=1)
    del all_emb; gc.collect()

    df = df.copy()
    df["esco_sector"] = [
        s if p >= conf_thr else "Other & Unclassified"
        for s, p in zip(preds, max_prob)
    ]
    before = len(df)
    df     = df[df["esco_sector"] != "Other & Unclassified"].copy()
    log.info("  Removed low-confidence : %s", f"{before - len(df):,}")
    log.info("  Kept                   : %s", f"{len(df):,}")
    log.info("  Distinct NLP sectors   : %d", df["esco_sector"].nunique())

    agree  = (df["esco_sector_regex"] == df["esco_sector"]).mean()
    n_diff = (df["esco_sector_regex"] != df["esco_sector"]).sum()
    log.info("  Regex vs NLP agreement : %.2f%%  (%s cases differ)",
             agree * 100, f"{n_diff:,}")
    if n_diff > 0:
        pairs = (
            df[df["esco_sector_regex"] != df["esco_sector"]]
            .groupby(["esco_sector_regex", "esco_sector"])
            .size().reset_index(name="count")
            .sort_values("count", ascending=False).head(10)
        )
        log.info("  Top disagreement pairs:\n%s", pairs.to_string(index=False))

    # Save sector distribution to metrics/
    sc = df["esco_sector"].value_counts().reset_index()
    sc.columns = ["esco_sector", "N"]
    sc["pct"]  = (sc["N"] / len(df) * 100).round(2)
    sc.to_csv(Path(cfg["metrics_dir"]) / "sector_distribution.csv", index=False)

    del probs; gc.collect()
    df = df.sort_values(["person_id", "year"]).reset_index(drop=True)
    return df


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
    """PCA on sector transition profiles with sector-group colouring."""
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

    pca_model  = PCA(n_components=5)
    pca_result = pca_model.fit_transform(mat)
    pca_var    = np.round(pca_model.explained_variance_ratio_ * 100, 1)
    log.info("  PCA variance -- PC1:%.1f%%  PC2:%.1f%%  Cum(1+2):%.1f%%",
             pca_var[0], pca_var[1], pca_var[0] + pca_var[1])

    GROUP_MAP = {
        "Software & IT Development": "Technology",
        "Data Science & Analytics": "Technology",
        "Cybersecurity & Compliance": "Technology",
        "IT Infrastructure & Networks": "Technology",
        "ICT Management & Consulting": "Technology",
        "Healthcare - Clinical": "Health & Science",
        "Healthcare - Nursing & Allied": "Health & Science",
        "Social Work & Community Services": "Health & Science",
        "Research & Academia": "Health & Science",
        "Finance & Accounting": "Finance & Legal",
        "Banking, Finance & Insurance": "Finance & Legal",
        "Legal & Compliance": "Finance & Legal",
        "Education - Teaching": "Public & Education",
        "Public Sector & Administration": "Public & Education",
        "Environment, Safety & Sustainability": "Public & Education",
        "Sales & Business Development": "Business & Management",
        "Marketing & Communications": "Business & Management",
        "Customer Service & Support": "Business & Management",
        "Senior Management & C-Suite": "Business & Management",
        "Operations & General Management": "Business & Management",
        "Human Resources & Recruitment": "Business & Management",
        "Project & Programme Management": "Business & Management",
    }
    GROUP_COLORS = {
        "Technology": "#185FA5",
        "Health & Science": "#1D9E75",
        "Finance & Legal": "#BA7517",
        "Public & Education": "#993556",
        "Business & Management": "#534AB7",
        "Industry & Trades": "#A32D2D",
    }

    pca_df          = pd.DataFrame({"PC1": pca_result[:, 0], "PC2": pca_result[:, 1],
                                     "sector": all_pca})
    pca_df["group"] = pca_df["sector"].map(GROUP_MAP).fillna("Industry & Trades")

    if cfg.get("plots", True):
        fig, ax = plt.subplots(figsize=(13, 9))
        for grp, color in GROUP_COLORS.items():
            sub = pca_df[pca_df["group"] == grp]
            ax.scatter(sub["PC1"], sub["PC2"], label=grp, color=color, s=60, alpha=0.85)
            for _, row in sub.iterrows():
                ax.annotate(row["sector"], (row["PC1"], row["PC2"]), fontsize=7, alpha=0.8)
        ax.set_title(
            f"PCA of Sector Career-Transition Profiles\n"
            f"PC1:{pca_var[0]}%  |  PC2:{pca_var[1]}% variance",
            fontsize=13, fontweight="bold",
        )
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        ax.legend(fontsize=9, loc="best")
        plt.tight_layout()
        out = Path(cfg["plots_dir"]) / "pca_sector_profiles.png"
        plt.savefig(out, dpi=150); plt.close()
        log.info("  Saved: %s", out)

    del mat, pca_result; gc.collect()
    return pca_var, all_pca


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

    out = Path(cfg["metrics_dir"]) / "drift_risk_scores.csv"
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

    out = Path(cfg["metrics_dir"]) / "association_rules.csv"
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
    ax.set_title(f"Job Distribution Across {df['esco_sector'].nunique()} Sectors (NLP-classified)",
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
    ax.set_title(f"Career Transition Heatmap ({df['esco_sector'].nunique()} Sectors)",
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
    tsp = (
        risk_df[risk_df["cluster"].notna()]
        .merge(dom_sec_df, on="person_id", how="left")
        .groupby(["cluster", "dominant_sector"]).size().reset_index(name="n")
        .sort_values(["cluster", "n"], ascending=[True, False])
        .groupby("cluster").head(3)
        .merge(cs2[["cluster", "archetype"]], on="cluster")
    )
    unique_archs = tsp["archetype"].unique()
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
    ax.set_title("Career Prediction Accuracy -- Ensemble Model (NLP labels)",
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
) -> None:
    """Export all final metrics to JSON in metrics/ for publication reproducibility."""
    metrics_dir = Path(cfg["metrics_dir"])
    payload = {
        "timestamp":          datetime.utcnow().isoformat(),
        "pipeline_version":   "2.0",
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
    out = metrics_dir / "final_metrics.json"
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=2)
    log.info("  Final metrics -> %s", out)


def print_summary(
    nlp_metrics: dict,
    ensemble_results: dict,
    pca_var: np.ndarray,
    cfg: dict,
) -> None:
    sep = "=" * 65
    log.info("\n%s", sep)
    log.info("  CAREER DRIFT PIPELINE v2 -- FINAL SUMMARY")
    log.info("%s", sep)
    log.info("  Architecture")
    log.info("    Bootstrap labels : regex map_sectors() + broad_map() (R port)")
    log.info("    NLP classifier   : SBERT all-MiniLM-L6-v2 + LightGBM")
    log.info("    Weak supervision : regex -> bootstrap labels -> NLP training")
    log.info("  %s", "-" * 60)
    log.info("  NLP Classification  (vs bootstrap labels)")
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
        "plots":       ["nlp_confusion_matrix.png", "sector_distribution.png",
                        "sequence_length_distribution.png", "pca_sector_profiles.png",
                        "career_drift_distribution.png", "transition_heatmap.png",
                        "top_transitions.png", "market_drift.png", "user_clusters.png",
                        "cluster_sector_breakdown.png", "accuracy_comparison.png",
                        "association_rules_lift.png"],
        "metrics":     ["nlp_metrics.json", "nlp_classification_report.txt",
                        "nlp_confusion_matrix.csv", "sector_distribution.csv",
                        "drift_risk_scores.csv", "association_rules.csv",
                        "final_metrics.json"],
        "checkpoints": ["cleaned_data.parquet", "regex_bootstrap.parquet",
                        "nlp_predicted.parquet", "transitions.parquet"],
    }
    log.info("Output file manifest:")
    for sub, files in ALL_OUTPUTS.items():
        root = Path(cfg["plots_dir"] if sub == "plots" else
                    cfg["metrics_dir"] if sub == "metrics" else cfg["ckpt_dir"])
        for f in files:
            exists = (root / f).exists()
            log.info("  %s  %s/%s", "v" if exists else "x", sub, f)

    export_final_metrics(nlp_metrics, ensemble_results, pca_var, cfg)
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
    log = _setup_logging(Path(cfg["output_dir"]) / "pipeline.log")

    seed_everything(cfg["seed"])
    device = detect_device()

    log.info("Pipeline configuration:")
    for k, v in cfg.items():
        log.info("  %-20s : %s", k, v)

    # ---------------------------------------------------------------------- 1
    ckpt_clean = Path(cfg["ckpt_dir"]) / "cleaned_data.parquet"
    with timed("1. Data loading & cleaning"):
        if cfg["use_cache"] and ckpt_clean.exists():
            log.info("  Resuming from checkpoint: %s", ckpt_clean)
            df = pd.read_parquet(ckpt_clean)
        else:
            df = load_data(cfg["data_dir"])
            df = clean_data(df)
            df.to_parquet(ckpt_clean, index=False)
            log.info("  Checkpoint saved -> %s", ckpt_clean)

    # ---------------------------------------------------------------------- 2
    ckpt_reg = Path(cfg["ckpt_dir"]) / "regex_bootstrap.parquet"
    with timed("2. Weak supervision (regex bootstrap)"):
        if cfg["use_cache"] and ckpt_reg.exists():
            log.info("  Resuming from checkpoint: %s", ckpt_reg)
            df = pd.read_parquet(ckpt_reg)
        else:
            df = apply_regex_pipeline(df)
            df.to_parquet(ckpt_reg, index=False)
            log.info("  Checkpoint saved -> %s", ckpt_reg)

    # ---------------------------------------------------------------------- 3
    ckpt_nlp  = Path(cfg["ckpt_dir"]) / "nlp_predicted.parquet"
    ckpt_lgbm = Path(cfg["model_dir"]) / "lgbm_sector_classifier.joblib"
    with timed("3. NLP sector classification (SBERT + LightGBM)"):
        if cfg["use_cache"] and ckpt_nlp.exists():
            log.info("  Resuming from checkpoint: %s", ckpt_nlp)
            df = pd.read_parquet(ckpt_nlp)
            classifier = joblib.load(ckpt_lgbm) if ckpt_lgbm.exists() else None
            nlp_metrics = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
            log.info("  NLP metrics not re-evaluated (use --no-cache to recompute)")
            embedding_model = None
        else:
            classifier, embedding_model, nlp_metrics = train_nlp_classifier(df, cfg, device)
            df = predict_nlp_sectors(df, classifier, embedding_model, cfg, device)
            df.to_parquet(ckpt_nlp, index=False)
            log.info("  Checkpoint saved -> %s", ckpt_nlp)

    if embedding_model is not None:
        del embedding_model
    gc.collect()

    # ---------------------------------------------------------------------- 4
    ckpt_tr = Path(cfg["ckpt_dir"]) / "transitions.parquet"
    with timed("4. Transition modeling"):
        if cfg["use_cache"] and ckpt_tr.exists():
            log.info("  Resuming from checkpoint: %s", ckpt_tr)
            transitions = pd.read_parquet(ckpt_tr)
            df, transitions, person_univ = build_transitions(df)
        else:
            df, transitions, person_univ = build_transitions(df)
            transitions.to_parquet(ckpt_tr, index=False)
            log.info("  Checkpoint saved -> %s", ckpt_tr)

    # ---------------------------------------------------------------------- 5
    with timed("5. PCA on transition profiles"):
        pca_var, _ = run_pca(transitions, cfg)

    # ---------------------------------------------------------------------- 6
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

    # ---------------------------------------------------------------------- 7
    all_sectors   = sorted(set(transitions["esco_sector"]) | set(transitions["next_sector"]))
    sector_to_idx = {s: i for i, s in enumerate(all_sectors)}

    with timed("7. Drift analysis & clustering"):
        risk_df, cs2, dom_sec_df = compute_drift_and_clusters(
            df, transitions, components["global_matrix"],
            sector_to_idx, all_sectors, cfg["seed"], cfg,
        )

    # ---------------------------------------------------------------------- 8
    with timed("8. Association rules"):
        assoc_rules = compute_association_rules(transitions, cfg)

    # ---------------------------------------------------------------------- 9
    if cfg.get("plots", True):
        with timed("9. Visualisations"):
            run_all_plots(
                df, transitions, risk_df, cs2, dom_sec_df,
                assoc_rules, components["market_drift"],
                ensemble_results, cfg["plots_dir"],
            )
    else:
        log.info("Visualisations skipped (--no-plots)")

    # --------------------------------------------------------------------- 10
    print_summary(nlp_metrics, ensemble_results, pca_var, cfg)
    log.info("Pipeline complete.")


# ===========================================================================
if __name__ == "__main__":
    main()