"""
Career Drift Trajectory Analysis — Production Pipeline  v4
===========================================================
Architecture : Weak Supervision -> SBERT + LightGBM -> Full Ensemble

Weak-supervision rationale
---------------------------
The original R implementation (career_pred12.R) uses hand-crafted expert
regex rules (map_sectors / broad_map) to assign esco_sector labels.
We port those rules exactly into Python, run them to produce bootstrap
pseudo-labels, then train an SBERT + LightGBM classifier on those labels.
The NLP model is the FINAL sector classifier; the regex system is used
only internally as a weak supervisor.  This gives us:
  * expert domain knowledge (27 ESCO-aligned sectors) as a starting point
  * semantic generalisation to unseen / variant job titles
  * better handling of synonyms, abbreviations, informal phrasing
  * production-grade reproducibility and measurable improvement over regex

References: Ratner et al. (Snorkel, 2017); Dawid & Skene (1979).

v3 additions (all methodology-preserving, engineering only):
  1.  Unified output directory structure:
        outputs/figures/ tables/ metrics/ checkpoints/ logs/ summaries/
  2.  Dataset fingerprinting + cache validation
        - SHA-256 of combined_text column gates every cache/checkpoint
        - stale caches from a different dataset are auto-invalidated
        - per-cache .metadata.json tracks hash, model, device, versions
  3.  Full environment diagnostics at startup
        - Python, torch, CUDA, sentence-transformers, LightGBM versions
        - OS / platform info
        - saved to outputs/logs/environment_info.json
  4.  Strong deterministic reproducibility
        - torch.manual_seed / cuda.manual_seed_all / cudnn flags
        - full run config snapshot saved to outputs/metrics/run_config.json
  5.  Cache metadata tracking (per .npy / .joblib / .parquet)
  6.  Graceful recovery logic
        - corrupted or mismatched caches are caught and safely recomputed
        - pipeline never crashes from stale or partial caches
  7.  Final summary export
        - outputs/summaries/final_pipeline_summary.txt (human-readable)
  8.  Fixed KeyError: dominant_sector in plot_cluster_sector_breakdown()

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
    --confidence-thr    FLT   NLP confidence threshold   (default: 0.45)
    --no-cache                force re-computation of all stages
    --no-plots                skip all matplotlib output

Output tree
-----------
    results/
    |-- figures/          all matplotlib / seaborn plots
    |-- tables/           CSV exports (rules, risk scores, distributions)
    |-- metrics/          JSON metric files + classification report
    |-- checkpoints/      intermediate parquet checkpoints
    |-- logs/
    |   |-- pipeline.log
    |   `-- environment_info.json
    `-- summaries/
        `-- final_pipeline_summary.txt
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
import re
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
PIPELINE_VERSION = "4.0"                          # Phase 1: ISCO-08 L3 upgrade
SBERT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# ISCO-08 Level-3 occupation groups (~125 groups).
# Each entry: (isco_l3_code, label, compiled regex pattern).
# Rules are evaluated in order; first match wins (same semantics as previous
# map_sectors / broad_map regime).
#
# Source: ILO ISCO-08 Volume I structure and group definitions.
# Only groups that are realistically observable from job-title text are
# included; remaining rows fall through to occupation_group = "Other".
# ---------------------------------------------------------------------------
_ISCO_L3_RULES: List[Tuple[str, str, re.Pattern]] = [
    # ── Major Group 1 – Managers ────────────────────────────────────────────
    ("111", "Chief Executives, Senior Officials & Legislators",
     re.compile(r"\bceo\b|chief executive|managing director|\bcoo\b|chief operating|"
                r"vice president|\bsvp\b|\bevp\b|group director|global director|"
                r"executive director|country manager|general director|chief officer|"
                r"\bpresident\b|board member|chairman|non.?executive director|"
                r"secretary of state|minister\b|head of government")),

    ("121", "Business Services & Administration Managers",
     re.compile(r"operations manager|operations director|general manager|business manager|"
                r"site manager|branch manager|centre manager|office manager|"
                r"administration manager|business administrator")),

    ("122", "Sales, Marketing & Development Managers",
     re.compile(r"sales manager|sales director|marketing manager|marketing director|"
                r"commercial manager|commercial director|revenue manager|"
                r"business development manager|growth manager")),

    ("123", "Research & Development Managers",
     re.compile(r"r.d manager|research director|research manager|head of research|"
                r"principal investigator|chief scientist|vp research")),

    ("124", "Advertising & Public Relations Managers",
     re.compile(r"pr manager|public relations manager|communications director|"
                r"communications manager|brand manager|media manager|"
                r"marketing communications|content director")),

    ("131", "Production Managers – Agriculture, Forestry & Fisheries",
     re.compile(r"farm manager|agricultural manager|forestry manager|fishery manager|"
                r"crop manager|livestock manager")),

    ("132", "Manufacturing, Mining & Construction Managers",
     re.compile(r"plant manager|production manager|manufacturing manager|"
                r"construction director|construction manager|site director|"
                r"operations manager.*manufactur")),

    ("133", "ICT Service Managers",
     re.compile(r"it manager|it director|head of it|head of technology|"
                r"technology manager|technology director|digital director|"
                r"chief digital|chief information|\bcio\b|\bcto\b|ict manager")),

    ("134", "Professional Services Managers",
     re.compile(r"project manager|programme manager|project director|delivery manager|"
                r"\bpmo\b|project management office|scrum master|agile coach|"
                r"change manager|implementation manager|portfolio manager.*project")),

    ("141", "Hotel & Restaurant Managers",
     re.compile(r"hotel manager|restaurant manager|catering manager|food and beverage|"
                r"f.b manager|front of house|bar manager|floor manager|"
                r"hospitality manager|venue manager")),

    ("142", "Retail & Wholesale Trade Managers",
     re.compile(r"retail manager|store manager|shop manager|retail director|"
                r"merchandising manager|buying manager|\bbuyer\b")),

    ("143", "Other Services Managers",
     re.compile(r"events manager|leisure manager|tourism manager|spa manager|"
                r"recreation manager|club manager|sports manager")),

    # ── Major Group 2 – Professionals ───────────────────────────────────────
    ("211", "Physical & Earth Science Professionals",
     re.compile(r"physicist|geologist|geophysicist|meteorologist|astronomer|"
                r"chemist\b|material scientist|oceanographer")),

    ("212", "Mathematicians, Actuaries & Statisticians",
     re.compile(r"mathematician|actuary|statistician|quantitative analyst|"
                r"data scientist|operations research|econometrician")),

    ("213", "Life Science Professionals",
     re.compile(r"biologist|biochemist|microbiologist|botanist|zoologist|"
                r"ecologist|geneticist|epidemiologist|toxicologist|"
                r"life scientist|cell biologist")),

    ("214", "Engineering Professionals (Excl. Electrotechnology)",
     re.compile(r"civil engineer|structural engineer|mechanical engineer|"
                r"chemical engineer|aerospace engineer|automotive engineer|"
                r"industrial engineer|manufacturing engineer|process engineer|"
                r"design engineer|materials engineer|nuclear engineer")),

    ("215", "Electrotechnology Engineers",
     re.compile(r"electrical engineer|electronic engineer|electronics engineer|"
                r"power engineer|telecoms engineer|rf engineer|embedded engineer|"
                r"firmware engineer|control engineer|instrumentation engineer")),

    ("216", "Architects, Planners, Surveyors & Designers",
     re.compile(r"\barchitect\b|urban planner|town planner|quantity surveyor|"
                r"building surveyor|interior designer|landscape architect|"
                r"ux designer|ui designer|visual designer|interaction designer|"
                r"structural technician|architectural technician")),

    ("221", "Medical Doctors",
     re.compile(r"\bdoctor\b|general practitioner|\bgp\b|physician|surgeon|"
                r"anaesthetist|radiologist|pathologist|psychiatrist|"
                r"paediatrician|cardiologist|neurologist|oncologist|"
                r"dermatologist|ophthalmologist|gp registrar|"
                r"foundation doctor|junior doctor|house officer|medical officer")),

    ("222", "Nursing & Midwifery Professionals",
     re.compile(r"registered nurse|staff nurse|charge nurse|ward nurse|"
                r"theatre nurse|mental health nurse|community nurse|"
                r"neonatal nurse|paediatric nurse|nursing manager|"
                r"nurse practitioner|practice nurse|health visitor|midwife|midwifery")),

    ("223", "Traditional & Complementary Medicine Professionals",
     re.compile(r"acupuncturist|homeopath|naturopath|osteopath|chiropractor|"
                r"herbalist|ayurvedic")),

    ("224", "Paramedical Practitioners",
     re.compile(r"paramedic|ambulance technician|emergency medical")),

    ("225", "Veterinarians",
     re.compile(r"veterinarian|veterinary surgeon|\bvet\b|animal doctor")),

    ("226", "Other Health Professionals",
     re.compile(r"physiotherapist|occupational therapist|speech therapist|"
                r"radiographer|sonographer|dietitian|\bpharmacist\b|"
                r"pharmacy technician|biomedical scientist|optometrist|\bdentist\b|"
                r"dental surgeon")),

    ("231", "University & Higher Education Teachers",
     re.compile(r"professor|university lecturer|lecturer\b|\bacademic\b|\bfaculty\b|"
                r"\bdean\b|postdoctoral|phd researcher|research fellow|tutor\b")),

    ("232", "Vocational Education Teachers",
     re.compile(r"vocational trainer|skills trainer|further education teacher|"
                r"nvq trainer|apprenticeship trainer")),

    ("233", "Secondary Education Teachers",
     re.compile(r"secondary teacher|high school teacher|form teacher|maths teacher|"
                r"science teacher|pe teacher|secondary school teacher")),

    ("234", "Primary School Teachers",
     re.compile(r"primary teacher|class teacher|nursery teacher|reception teacher|"
                r"eyfs|ks1 teacher|ks2 teacher|infant teacher|junior teacher")),

    ("235", "Other Teaching Professionals",
     re.compile(r"teacher\b|educator|instructor\b|esl teacher|tefl teacher|"
                r"english language teacher|special education|sen teacher|"
                r"learning support|teaching assistant|head teacher|principal\b")),

    ("241", "Finance Professionals",
     re.compile(r"\baccountant\b|\bauditor\b|management accountant|"
                r"financial analyst|financial controller|finance director|\bcfo\b|"
                r"chief financial|chartered accountant|fp.a|financial reporting|"
                r"finance manager|finance analyst|cost accountant|budget analyst")),

    ("242", "Policy & Administration Professionals",
     re.compile(r"policy officer|policy analyst|civil servant|\bgovernment\b|"
                r"\bcouncil\b|local authority|regulatory officer|parliamentary|"
                r"public administrator|administration officer|\bclerk\b")),

    ("243", "Administrative & Specialist Managers",
     re.compile(r"hr manager|human resources manager|hr director|\bhrbp\b|"
                r"hr business partner|talent acquisition|recruitment consultant|"
                r"\brecruiter\b")),

    ("251", "Software & Applications Developers & Analysts",
     re.compile(r"software engineer|software developer|software architect|"
                r"web developer|web designer|front.?end|back.?end|full.?stack|"
                r"mobile developer|android developer|ios developer|app developer|"
                r"game developer|systems programmer|devops engineer|"
                r"cloud engineer|platform engineer")),

    ("252", "Database & Network Professionals",
     re.compile(r"database administrator|dba|network engineer|network administrator|"
                r"systems administrator|infrastructure engineer|site reliability|"
                r"devops specialist|cloud architect|storage engineer")),

    ("261", "Legal Professionals",
     re.compile(r"solicitor|barrister|\blawyer\b|legal advisor|legal counsel|"
                r"paralegal|legal executive|legal director|general counsel|"
                r"contract manager|conveyancer|legal specialist|legal officer")),

    ("262", "Archivists, Librarians & Records Managers",
     re.compile(r"archivist|librarian|records manager|document control|"
                r"information officer|knowledge manager")),

    ("263", "Social & Religious Professionals",
     re.compile(r"social worker|community worker|probation officer|youth worker|"
                r"welfare officer|housing officer|counsellor|psychotherapist|"
                r"outreach worker|family support|advocacy worker|"
                r"rehabilitation specialist|safeguarding officer|"
                r"child protection officer|addiction counsellor")),

    ("264", "Authors, Journalists & Linguists",
     re.compile(r"\bjournalist\b|\beditor\b|copywriter|content creator|author\b|"
                r"technical writer|translator|interpreter|linguist|"
                r"proofreader|scriptwriter")),

    ("265", "Creative & Performing Arts Professionals",
     re.compile(r"graphic designer|creative director|art director|animator|"
                r"illustrator|photographer|videographer|broadcast|film director|"
                r"art director")),

    # ── Major Group 3 – Technicians & Associate Professionals ───────────────
    ("311", "Physical & Engineering Science Technicians",
     re.compile(r"lab technician|laboratory technician|engineering technician|"
                r"quality technician|test technician|process technician|"
                r"maintenance technician|mechanical technician")),

    ("312", "Construction & Manufacturing Supervisors",
     re.compile(r"construction supervisor|production supervisor|site supervisor|"
                r"manufacturing supervisor|assembly supervisor")),

    ("313", "Process Control Technicians",
     re.compile(r"plant operator|machine operator|process operator|"
                r"production operator|control room operator")),

    ("314", "Life Science Technicians",
     re.compile(r"lab analyst|laboratory analyst|clinical lab|"
                r"biological technician|research technician")),

    ("315", "Ship & Aircraft Controllers & Technicians",
     re.compile(r"air traffic controller|flight dispatcher|marine officer|"
                r"ship captain|pilot\b|aviation technician")),

    ("321", "Medical Imaging & Therapeutic Equipment Technicians",
     re.compile(r"radiographer|sonographer|mri technician|ct technician|"
                r"medical imaging|radiation therapist")),

    ("322", "Medical & Pharmaceutical Technicians",
     re.compile(r"pharmacy technician|dispensary technician|pharmaceutical technician|"
                r"biomedical technician|pathology technician")),

    ("323", "Veterinary Technicians",
     re.compile(r"veterinary nurse|vet nurse|veterinary technician|vet technician")),

    ("325", "Dental Technicians",
     re.compile(r"dental technician|dental nurse|dental therapist|dental hygienist")),

    ("331", "Financial & Accounting Associate Professionals",
     re.compile(r"tax advisor|tax manager|\bpayroll\b|bookkeeper|"
                r"credit controller|accounts payable|accounts receivable|"
                r"accounts manager|finance officer|finance coordinator")),

    ("332", "Sales & Purchasing Agents & Brokers",
     re.compile(r"sales executive|sales representative|sales consultant|"
                r"account executive|account manager|pre.?sales|inside sales|"
                r"territory manager|area sales|regional sales|national sales|"
                r"stockbroker|financial advisor|mortgage advisor|mortgage broker|"
                r"loan officer|insurance advisor|\bbdr\b|\bsdr\b")),

    ("333", "Business Services Agents",
     re.compile(r"business analyst|operations analyst|management consultant|"
                r"business consultant|it consultant|technology consultant|"
                r"solutions architect|enterprise architect|digital transformation")),

    ("334", "Administrative & Executive Secretaries",
     re.compile(r"executive assistant|personal assistant|executive secretary|"
                r"administrative assistant|office coordinator")),

    ("335", "Regulatory Government Associate Professionals",
     re.compile(r"compliance manager|compliance officer|regulatory affairs|"
                r"regulatory manager|governance manager|gdpr|data protection officer|"
                r"information assurance")),

    ("341", "Legal Associate Professionals",
     re.compile(r"legal manager|legal officer|paralegal\b")),

    ("342", "Sports, Fitness & Recreation Associate Professionals",
     re.compile(r"personal trainer|fitness instructor|sports coach|"
                r"fitness coach|gym instructor|sports analyst")),

    ("343", "Artistic, Cultural & Culinary Associate Professionals",
     re.compile(r"chef\b|sous chef|head chef|pastry chef|cook\b|baker\b|"
                r"catering assistant")),

    # ── Major Group 3 continued – ICT ───────────────────────────────────────
    ("351", "ICT Operations & User Support Technicians",
     re.compile(r"it support|it technician|desktop support|service desk|"
                r"\bhelpdesk\b|technical support|systems support|it operations")),

    ("352", "Telecommunications & Broadcasting Technicians",
     re.compile(r"telecom technician|broadcast technician|network technician|"
                r"rf technician|communications technician")),

    # ── Major Group 4 – Clerical Support Workers ────────────────────────────
    ("411", "General Office Clerks",
     re.compile(r"\bclerk\b|office clerk|general admin|administrative clerk|"
                r"data entry|filing clerk")),

    ("412", "Secretaries",
     re.compile(r"secretary\b|receptionist|front desk|administrative secretary")),

    ("421", "Tellers & Counter Clerks",
     re.compile(r"bank teller|counter clerk|post office clerk|cashier\b")),

    ("431", "Numerical Clerks",
     re.compile(r"accounts clerk|billing clerk|payroll clerk|invoice clerk")),

    ("441", "Other Clerical Support Workers",
     re.compile(r"customer service advisor|customer service representative|"
                r"call centre agent|contact centre agent|customer advisor|"
                r"customer care agent|client advisor")),

    # ── Major Group 5 – Services & Sales Workers ────────────────────────────
    ("511", "Travel Attendants & Travel Stewards",
     re.compile(r"flight attendant|cabin crew|travel agent|tour guide|"
                r"holiday rep|tourist guide")),

    ("512", "Cooks",
     re.compile(r"line cook|commis chef|kitchen assistant|cook\b")),

    ("513", "Waiters & Related Workers",
     re.compile(r"waiter|waitress|barista|bartender|bar staff|table service")),

    ("521", "Retail Salespersons",
     re.compile(r"retail assistant|shop assistant|sales assistant|"
                r"store associate|retail associate|shop worker")),

    ("531", "Child Care Workers",
     re.compile(r"nursery nurse|childminder|nanny|au pair|childcare worker|"
                r"early years worker")),

    ("532", "Personal Care Workers in Health Services",
     re.compile(r"care worker|care assistant|support worker|\bhca\b|"
                r"healthcare assistant|nursing assistant|home carer|"
                r"domiciliary carer|mental health support worker|"
                r"disability support worker")),

    ("541", "Protective Services Workers",
     re.compile(r"police officer|security officer|security guard|firefighter|"
                r"prison officer|border officer|customs officer|warden\b")),

    # ── Major Group 6 – Agricultural, Forestry & Fishery Workers ────────────
    ("611", "Market Gardeners & Crop Growers",
     re.compile(r"market gardener|crop grower|horticulturist|farm worker|"
                r"agricultural worker|grower\b")),

    ("621", "Livestock Farmers",
     re.compile(r"livestock farmer|dairy farmer|cattle farmer|shepherd\b|"
                r"poultry farmer|pig farmer")),

    # ── Major Group 7 – Craft & Related Trades Workers ──────────────────────
    ("711", "Building Frame & Related Trades Workers",
     re.compile(r"carpenter|bricklayer|scaffolder|roofer|plasterer|"
                r"construction worker|concretor|steel fixer")),

    ("712", "Building Finishers & Related Trades Workers",
     re.compile(r"painter\b|decorator\b|floor layer|glazier|tiler\b")),

    ("713", "Painters, Building Structure Cleaners",
     re.compile(r"painter and decorator|industrial painter|sandblaster")),

    ("721", "Sheet & Structural Metal Workers",
     re.compile(r"fabricator|sheet metal worker|structural steelwork|"
                r"metal fabricator|boilermaker")),

    ("722", "Blacksmiths, Toolmakers & Related Trades",
     re.compile(r"toolmaker|machinist\b|\bfitter\b|tool and die maker|"
                r"die caster|precision engineer")),

    ("723", "Machinery Mechanics & Repair Workers",
     re.compile(r"maintenance engineer|plant engineer|commissioning engineer|"
                r"hvac engineer|refrigeration engineer|service engineer|"
                r"machinery mechanic|equipment mechanic")),

    ("731", "Handicraft Workers",
     re.compile(r"jeweller|watchmaker|glassblower|potter|ceramicist|craftsman")),

    ("741", "Electrical Equipment Installers & Repairers",
     re.compile(r"electrician\b|electrical installer|electrical technician|"
                r"electrical fitter|instrumentation technician")),

    ("742", "Electronics & Telecommunications Installers",
     re.compile(r"electronics technician|telecom installer|broadband engineer|"
                r"network installer|avionics technician")),

    ("751", "Food Processing & Related Trades Workers",
     re.compile(r"food technologist|food scientist|food production|"
                r"quality assurance.*food|food safety")),

    # ── Major Group 8 – Plant & Machine Operators ───────────────────────────
    ("811", "Mining & Mineral Processing Plant Operators",
     re.compile(r"mining operator|mineral processing|drill operator|rig operator")),

    ("821", "Metal Processing & Finishing Operators",
     re.compile(r"welder\b|assembler\b|press operator|cnc operator|"
                r"lathe operator|grinding operator")),

    ("831", "Locomotive & Related Workers",
     re.compile(r"train driver|locomotive driver|rail driver|tram driver")),

    ("832", "Car, Van & Motorcycle Drivers",
     re.compile(r"delivery driver|van driver|courier\b|taxi driver|"
                r"uber driver|rideshare driver")),

    ("833", "Heavy Truck & Bus Drivers",
     re.compile(r"hgv driver|lgv driver|truck driver|bus driver|"
                r"coach driver|lorry driver|heavy goods")),

    ("834", "Mobile Plant Operators",
     re.compile(r"forklift operator|crane operator|excavator operator|"
                r"plant operator")),

    # ── Major Group 9 – Elementary Occupations ──────────────────────────────
    ("911", "Domestic Cleaners",
     re.compile(r"cleaner\b|domestic cleaner|housekeeper\b|housekeeping\b|"
                r"cleaning operative")),

    ("962", "Refuse Workers & Other Elementary Workers",
     re.compile(r"refuse collector|waste operative|recycling operative|"
                r"refuse worker")),

    # ── Specialised groups not fitting cleanly above ─────────────────────────
    ("243b", "Human Resources & Recruitment Professionals",
     re.compile(r"hr advisor|hr officer|hr coordinator|hr business partner|"
                r"learning and development|l.d manager|training manager|"
                r"organisational development|people manager|workforce planning|"
                r"compensation|benefits manager|employee relations")),

    ("252b", "Data Science & Analytics Professionals",
     re.compile(r"data analyst|business intelligence|bi developer|bi analyst|"
                r"data manager|analytics manager|data architect|"
                r"reporting analyst|data consultant|insights analyst|data lead|"
                r"machine learning engineer|ai engineer|nlp engineer|data engineer")),

    ("335b", "Cybersecurity Professionals",
     re.compile(r"\bcyber\b|security analyst|security engineer|security architect|"
                r"information security|penetration tester|\bpentest\b|soc analyst|"
                r"vulnerability|threat intelligence|\bforensic\b|incident response|"
                r"security manager|security consultant|identity access")),

    ("133b", "Supply Chain & Logistics Managers",
     re.compile(r"supply chain manager|supply chain director|logistics manager|"
                r"procurement manager|purchasing manager|warehouse manager|"
                r"inventory manager|distribution manager|transport manager|"
                r"freight manager|shipping manager|materials manager|"
                r"demand planner|supply planner|fulfilment manager")),

    ("244", "Environment, Health & Safety Professionals",
     re.compile(r"environment|sustainability|health and safety|hse manager|"
                r"ehs manager|she manager|esg manager|renewable energy|"
                r"energy manager|climate change|carbon manager|waste manager|"
                r"ecology|conservation|environmental manager|safety manager|"
                r"environmental consultant|sustainability manager")),

    ("211b", "Research & Academic Professionals",
     re.compile(r"\bresearcher\b|research officer|\bscientist\b|laboratory manager|"
                r"postdoctoral|principal investigator|research fellow|biologist|"
                r"biochemist|epidemiologist|clinical researcher|"
                r"research scientist|research associate")),

    ("335c", "Banking, Finance & Insurance Professionals",
     re.compile(r"investment banker|investment analyst|fund manager|"
                r"portfolio manager|wealth manager|financial planner|"
                r"equity trader|risk analyst|risk manager|underwriter|"
                r"private equity|hedge fund|credit analyst|capital markets|"
                r"banking analyst|insurance analyst|\btreasury\b")),
]

# ISCO L3 group that receives all unmatched rows (mirrors "Other & Unclassified")
_ISCO_UNMATCHED = "Other & Unclassified (ISCO)"

# Mapping from ISCO L3 label to a broader grouping for PCA colouring.
# Groups intentionally mirror ISCO Major Groups so plots remain interpretable.
ISCO_GROUP_MAP: Dict[str, str] = {
    # Major Group 1 - Managers
    "Chief Executives, Senior Officials & Legislators":         "Managers",
    "Business Services & Administration Managers":             "Managers",
    "Sales, Marketing & Development Managers":                  "Managers",
    "Research & Development Managers":                          "Managers",
    "Advertising & Public Relations Managers":                  "Managers",
    "Production Managers – Agriculture, Forestry & Fisheries":  "Managers",
    "Manufacturing, Mining & Construction Managers":            "Managers",
    "ICT Service Managers":                                     "Managers",
    "Professional Services Managers":                           "Managers",
    "Hotel & Restaurant Managers":                              "Managers",
    "Retail & Wholesale Trade Managers":                        "Managers",
    "Other Services Managers":                                  "Managers",
    # Major Group 2 - Professionals
    "Physical & Earth Science Professionals":                   "Science & Engineering Professionals",
    "Mathematicians, Actuaries & Statisticians":                "Science & Engineering Professionals",
    "Life Science Professionals":                               "Science & Engineering Professionals",
    "Engineering Professionals (Excl. Electrotechnology)":      "Science & Engineering Professionals",
    "Electrotechnology Engineers":                              "Science & Engineering Professionals",
    "Architects, Planners, Surveyors & Designers":              "Science & Engineering Professionals",
    "Research & Academic Professionals":                        "Science & Engineering Professionals",
    "Medical Doctors":                                          "Health Professionals",
    "Nursing & Midwifery Professionals":                        "Health Professionals",
    "Traditional & Complementary Medicine Professionals":       "Health Professionals",
    "Paramedical Practitioners":                                "Health Professionals",
    "Veterinarians":                                            "Health Professionals",
    "Other Health Professionals":                               "Health Professionals",
    "University & Higher Education Teachers":                   "Education Professionals",
    "Vocational Education Teachers":                            "Education Professionals",
    "Secondary Education Teachers":                             "Education Professionals",
    "Primary School Teachers":                                  "Education Professionals",
    "Other Teaching Professionals":                             "Education Professionals",
    "Finance Professionals":                                    "Business & Administrative Professionals",
    "Policy & Administration Professionals":                    "Business & Administrative Professionals",
    "Administrative & Specialist Managers":                     "Business & Administrative Professionals",
    "Legal Professionals":                                      "Business & Administrative Professionals",
    "Social & Religious Professionals":                         "Business & Administrative Professionals",
    "Authors, Journalists & Linguists":                         "Creative & ICT Professionals",
    "Creative & Performing Arts Professionals":                 "Creative & ICT Professionals",
    "Software & Applications Developers & Analysts":            "Creative & ICT Professionals",
    "Database & Network Professionals":                         "Creative & ICT Professionals",
    "Data Science & Analytics Professionals":                   "Creative & ICT Professionals",
    "Cybersecurity Professionals":                              "Creative & ICT Professionals",
    "Human Resources & Recruitment Professionals":              "Business & Administrative Professionals",
    "Banking, Finance & Insurance Professionals":               "Business & Administrative Professionals",
    "Environment, Health & Safety Professionals":               "Business & Administrative Professionals",
    "Archivists, Librarians & Records Managers":               "Business & Administrative Professionals",
    "Supply Chain & Logistics Managers":                        "Managers",
    # Major Group 3 - Technicians & Associates
    "Physical & Engineering Science Technicians":               "Technicians & Associate Professionals",
    "Construction & Manufacturing Supervisors":                 "Technicians & Associate Professionals",
    "Process Control Technicians":                              "Technicians & Associate Professionals",
    "Life Science Technicians":                                 "Technicians & Associate Professionals",
    "Ship & Aircraft Controllers & Technicians":                "Technicians & Associate Professionals",
    "Medical Imaging & Therapeutic Equipment Technicians":      "Technicians & Associate Professionals",
    "Medical & Pharmaceutical Technicians":                     "Technicians & Associate Professionals",
    "Veterinary Technicians":                                   "Technicians & Associate Professionals",
    "Dental Technicians":                                       "Technicians & Associate Professionals",
    "Financial & Accounting Associate Professionals":           "Technicians & Associate Professionals",
    "Sales & Purchasing Agents & Brokers":                      "Technicians & Associate Professionals",
    "Business Services Agents":                                 "Technicians & Associate Professionals",
    "Administrative & Executive Secretaries":                   "Clerical Support",
    "Regulatory Government Associate Professionals":            "Technicians & Associate Professionals",
    "Legal Associate Professionals":                            "Technicians & Associate Professionals",
    "Sports, Fitness & Recreation Associate Professionals":     "Services & Sales",
    "Artistic, Cultural & Culinary Associate Professionals":    "Services & Sales",
    "ICT Operations & User Support Technicians":                "Technicians & Associate Professionals",
    "Telecommunications & Broadcasting Technicians":            "Technicians & Associate Professionals",
    # Major Group 4 - Clerical
    "General Office Clerks":                                    "Clerical Support",
    "Secretaries":                                              "Clerical Support",
    "Tellers & Counter Clerks":                                 "Clerical Support",
    "Numerical Clerks":                                         "Clerical Support",
    "Other Clerical Support Workers":                           "Clerical Support",
    # Major Group 5 - Services & Sales
    "Travel Attendants & Travel Stewards":                      "Services & Sales",
    "Cooks":                                                    "Services & Sales",
    "Waiters & Related Workers":                                "Services & Sales",
    "Retail Salespersons":                                      "Services & Sales",
    "Child Care Workers":                                       "Services & Sales",
    "Personal Care Workers in Health Services":                 "Services & Sales",
    "Protective Services Workers":                              "Services & Sales",
    # Major Group 6 - Agricultural
    "Market Gardeners & Crop Growers":                          "Agricultural & Related",
    "Livestock Farmers":                                        "Agricultural & Related",
    # Major Group 7 - Craft & Trade
    "Building Frame & Related Trades Workers":                  "Craft & Trades",
    "Building Finishers & Related Trades Workers":              "Craft & Trades",
    "Painters, Building Structure Cleaners":                    "Craft & Trades",
    "Sheet & Structural Metal Workers":                         "Craft & Trades",
    "Blacksmiths, Toolmakers & Related Trades":                "Craft & Trades",
    "Machinery Mechanics & Repair Workers":                     "Craft & Trades",
    "Handicraft Workers":                                       "Craft & Trades",
    "Electrical Equipment Installers & Repairers":              "Craft & Trades",
    "Electronics & Telecommunications Installers":              "Craft & Trades",
    "Food Processing & Related Trades Workers":                 "Craft & Trades",
    # Major Group 8 - Machine Operators
    "Mining & Mineral Processing Plant Operators":              "Plant & Machine Operators",
    "Metal Processing & Finishing Operators":                   "Plant & Machine Operators",
    "Locomotive & Related Workers":                             "Plant & Machine Operators",
    "Car, Van & Motorcycle Drivers":                            "Plant & Machine Operators",
    "Heavy Truck & Bus Drivers":                                "Plant & Machine Operators",
    "Mobile Plant Operators":                                   "Plant & Machine Operators",
    # Major Group 9 - Elementary
    "Domestic Cleaners":                                        "Elementary Occupations",
    "Refuse Workers & Other Elementary Workers":                "Elementary Occupations",
}

# ===========================================================================
# 0.  CONFIGURATION & DIRECTORY SETUP
# ===========================================================================

def parse_args() -> dict:
    p = argparse.ArgumentParser(
        description="Career Drift Trajectory Analysis -- Production Pipeline v3"
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
# 3.  ISCO-08 L3 OCCUPATION ASSIGNMENT
# ===========================================================================
#
# Phase 1 upgrade: replace the previous 27-sector regex weak-supervision
# system with ISCO-08 Level-3 occupation groups (~125 groups).
#
# Architecture unchanged:
#   Step A  assign_isco_l3()   -> occupation_group column (weak supervisor)
#   Step B  SBERT embeddings   (unchanged)
#   Step C  LightGBM classifier trained on occupation_group labels
#   Step D  NLP predictions replace occupation_group labels -> esco_sector
#
# The esco_sector column now holds ISCO L3 labels (not 27 custom sectors).
# All downstream code (transitions, Markov, PCA, ensemble) is unaffected
# because it consumes esco_sector generically; only the number of classes
# and the group-colouring in PCA change.
# ---------------------------------------------------------------------------

def assign_isco_l3(title: str, description: str = "") -> str:
    """
    Assign an ISCO-08 Level-3 occupation group label to a job title
    (and optionally its description) by sequential regex matching.

    Evaluation order:
      Pass 1: match title (lowercased) against _ISCO_L3_RULES in order
      Pass 2: if unmatched, match description (lowercased) against rules
      Returns _ISCO_UNMATCHED if neither pass matches.

    The function is intentionally pure (no side effects) so it can be
    applied with df.apply() or df.map() safely.
    """
    title_l = title.lower().strip()
    for _code, label, pattern in _ISCO_L3_RULES:
        if pattern.search(title_l):
            return label

    if description:
        desc_l = description.lower().strip()
        for _code, label, pattern in _ISCO_L3_RULES:
            if pattern.search(desc_l):
                return label

    return _ISCO_UNMATCHED


def _combine_text(row) -> str:
    """Combine matched_label and matched_description for SBERT input."""
    title = str(row.get("matched_label", ""))
    desc  = str(row.get("matched_description", ""))
    if desc in ("nan", "None", ""):
        desc = ""
    return f"{title} [SEP] {desc}".strip()


def apply_isco_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    ISCO-08 L3 occupation assignment pipeline.

    Pass 1  : assign_isco_l3() on matched_label (+ matched_description)
    Pass 2  : drop rows still _ISCO_UNMATCHED (mirrors previous Pass 3)

    Adds:
      occupation_group  -- ISCO L3 label used as NLP training target
      combined_text     -- SBERT input (title [SEP] description)

    Returns a NEW DataFrame. Does not modify in place.

    NOTE: The column 'occupation_group' replaces the previous
    'esco_sector_regex'. Downstream code reads 'esco_sector' (the NLP
    prediction), so no other function needs changing.
    """
    df       = df.copy()
    has_desc = "matched_description" in df.columns

    log.info("  ISCO-08 L3 assignment -- Pass 1: title matching ...")
    if has_desc:
        df["occupation_group"] = [
            assign_isco_l3(
                str(row["matched_label"]),
                str(row.get("matched_description", "")) if has_desc else "",
            )
            for _, row in df.iterrows()
        ]
    else:
        df["occupation_group"] = df["matched_label"].astype(str).apply(
            lambda t: assign_isco_l3(t)
        )

    n_unmatched = (df["occupation_group"] == _ISCO_UNMATCHED).sum()
    n_matched   = len(df) - n_unmatched
    log.info("    Matched   : %s (%.1f%%)", f"{n_matched:,}",  n_matched  / len(df) * 100)
    log.info("    Unmatched : %s (%.1f%%)", f"{n_unmatched:,}", n_unmatched / len(df) * 100)

    # Pass 2: drop unmatched rows
    before = len(df)
    df = df[df["occupation_group"] != _ISCO_UNMATCHED].copy()
    log.info("  Pass 2 -- removed %s (%.1f%%) | kept %s",
             f"{before - len(df):,}", (before - len(df)) / before * 100, f"{len(df):,}")

    # Combined text for SBERT embedding
    df["combined_text"] = df.apply(_combine_text, axis=1)
    df = df.sort_values(["person_id", "year"]).reset_index(drop=True)

    sc = df["occupation_group"].value_counts().reset_index()
    sc.columns = ["occupation_group", "N"]
    sc["pct"]  = (sc["N"] / len(df) * 100).round(1)
    log.info("ISCO-08 L3 distribution (%d groups):\n%s",
             df["occupation_group"].nunique(), sc.to_string(index=False))
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
    Train SBERT + LightGBM ISCO-08 L3 classifier.

    Training target: occupation_group (ISCO L3 labels assigned by assign_isco_l3).
    Architecture: unchanged from v4 -- SBERT embeddings + LightGBM multiclass.
    LightGBM hyperparameters tuned for ~125 classes (more num_leaves, estimators).

    Cache policy: LightGBM checkpoint is validated against the dataset hash.
    If the hash mismatches or the file is corrupted, retraining occurs automatically.

    Returns
    -------
    classifier      : fitted LGBMClassifier (predicts ISCO L3 labels)
    embedding_model : loaded SentenceTransformer (kept for full-dataset prediction)
    nlp_metrics     : dict with accuracy / precision / recall / f1
    """
    seed      = cfg["seed"]
    cache_dir = Path(cfg["cache_dir"])
    model_dir = Path(cfg["model_dir"])
    use_cache = cfg["use_cache"]
    bs        = cfg["embed_batch_size"]
    cs        = cfg["embed_chunk_size"]
    ckpt_path = model_dir / "lgbm_isco_l3_classifier.joblib"   # new name avoids stale v4 cache

    # NLP train/val/test split -- stratified on ISCO L3 labels
    # NOTE: this split is independent of the ensemble prediction split (Section 7)
    train_t, temp_t, train_l, temp_l = train_test_split(
        df["combined_text"].tolist(),
        df["occupation_group"].tolist(),        # CHANGED: was esco_sector_regex
        test_size=0.30,
        stratify=df["occupation_group"],        # CHANGED
        random_state=seed,
    )
    val_t, test_t, val_l, test_l = train_test_split(
        temp_t, temp_l,
        test_size=0.50,
        stratify=temp_l,
        random_state=seed,
    )
    log.info("  NLP split (ISCO L3) -- train: %s | val: %s | test: %s",
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
        log.info("  Training LightGBM on %s ISCO L3 labels (%d classes) ...",
                 f"{len(train_t):,}", n_classes)
        # Hyperparameters scaled for ~125 classes:
        #   num_leaves=127  (2^7-1, supports finer decision boundaries)
        #   n_estimators=500 (more rounds for larger label space)
        #   min_child_samples=10 (prevents overfitting on rare ISCO groups)
        classifier = LGBMClassifier(
            objective="multiclass",
            n_estimators=500,
            learning_rate=0.05,
            max_depth=12,
            num_leaves=127,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_samples=10,
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
    log.info("  NLP ISCO L3 classifier performance:")
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
    Apply NLP ISCO L3 classifier to the full dataset; assign final esco_sector.

    esco_sector now holds ISCO-08 Level-3 labels (not 27 custom sectors).
    All downstream code is unchanged because it consumes esco_sector generically.
    Chunked embedding prevents memory spikes on large datasets.
    Rows below confidence_thr are dropped (same logic as v4).
    Cache is validated against the dataset fingerprint.
    """
    cache_dir = Path(cfg["cache_dir"])
    bs        = cfg["embed_batch_size"]
    cs        = cfg["embed_chunk_size"]
    use_cache = cfg["use_cache"]
    conf_thr  = cfg["confidence_thr"]

    all_emb = _embed_chunked(
        df["combined_text"].tolist(), embedding_model,
        cache_dir / "emb_full.npy", use_cache, "full dataset",
        bs, cs, device, dataset_hash, cfg,
    )

    log.info("  Running LightGBM ISCO L3 predict_proba on %s rows ...", f"{len(df):,}")
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
    log.info("  Removed low-confidence rows  : %s", f"{before - len(df):,}")
    log.info("  Kept                         : %s", f"{len(df):,}")
    log.info("  Distinct ISCO L3 groups      : %d", df["esco_sector"].nunique())

    # Agreement between ISCO L3 assignment and NLP prediction
    # (occupation_group is the training supervisor; esco_sector is the NLP output)
    if "occupation_group" in df.columns:
        agree  = (df["occupation_group"] == df["esco_sector"]).mean()
        n_diff = (df["occupation_group"] != df["esco_sector"]).sum()
        log.info("  ISCO assignment vs NLP agreement : %.2f%%  (%s cases differ)",
                 agree * 100, f"{n_diff:,}")
        if n_diff > 0:
            pairs = (
                df[df["occupation_group"] != df["esco_sector"]]
                .groupby(["occupation_group", "esco_sector"])
                .size().reset_index(name="count")
                .sort_values("count", ascending=False).head(10)
            )
            log.info("  Top disagreement pairs:\n%s", pairs.to_string(index=False))

    # ISCO L3 distribution -> tables/
    sc = df["esco_sector"].value_counts().reset_index()
    sc.columns = ["esco_sector", "N"]
    sc["pct"]  = (sc["N"] / len(df) * 100).round(2)
    sc.to_csv(Path(cfg["tables_dir"]) / "sector_distribution.csv", index=False)
    log.info("  Sector distribution saved -> tables/sector_distribution.csv")

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

    # Use ISCO_GROUP_MAP for colouring; unmapped labels fall back to "Other"
    ISCO_GROUP_COLORS = {
        "Managers":                             "#185FA5",
        "Science & Engineering Professionals":  "#1D9E75",
        "Health Professionals":                 "#E24B4A",
        "Education Professionals":              "#993556",
        "Business & Administrative Professionals": "#534AB7",
        "Creative & ICT Professionals":         "#BA7517",
        "Technicians & Associate Professionals":"#378ADD",
        "Clerical Support":                     "#A32D2D",
        "Services & Sales":                     "#6B6B6B",
        "Agricultural & Related":               "#2D7A2D",
        "Craft & Trades":                       "#8B4513",
        "Plant & Machine Operators":            "#CC6600",
        "Elementary Occupations":               "#999999",
        "Other":                                "#CCCCCC",
    }

    pca_df          = pd.DataFrame({
        "PC1":    pca_result[:, 0],
        "PC2":    pca_result[:, 1],
        "sector": all_pca,
    })
    pca_df["group"] = pca_df["sector"].map(ISCO_GROUP_MAP).fillna("Other")

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

    h("3. NLP CLASSIFICATION  (vs bootstrap regex labels)")
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
    log.info("  CAREER DRIFT PIPELINE v4 (ISCO-08 L3) -- FINAL SUMMARY")
    log.info("%s", sep)
    log.info("  Architecture")
    log.info("    ISCO L3 assignment : assign_isco_l3() -- ISCO-08 Level-3 (~125 groups)")
    log.info("    NLP classifier     : SBERT all-MiniLM-L6-v2 + LightGBM (multiclass)")
    log.info("    Label space        : ISCO-08 L3 occupation groups (replaces 27 sectors)")
    log.info("  %s", "-" * 60)
    log.info("  NLP ISCO L3 Classification  (vs ISCO L3 assignment labels)")
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
        "figures":     ["nlp_confusion_matrix.png", "sector_distribution.png",
                        "sequence_length_distribution.png", "pca_sector_profiles.png",
                        "career_drift_distribution.png", "transition_heatmap.png",
                        "top_transitions.png", "market_drift.png", "user_clusters.png",
                        "cluster_sector_breakdown.png", "accuracy_comparison.png",
                        "association_rules_lift.png"],
        "tables":      ["sector_distribution.csv", "nlp_confusion_matrix.csv",
                        "drift_risk_scores.csv", "association_rules.csv"],
        "metrics":     ["nlp_metrics.json", "nlp_classification_report.txt",
                        "final_metrics.json", "run_config.json"],
        "checkpoints": ["cleaned_data.parquet", "isco_l3_assigned.parquet",
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
    ckpt_reg = Path(cfg["ckpt_dir"]) / "isco_l3_assigned.parquet"    # renamed from regex_bootstrap
    with timed("2. ISCO-08 L3 occupation assignment"):
        if cfg["use_cache"] and ckpt_reg.exists():
            try:
                log.info("  Resuming from checkpoint: %s", ckpt_reg)
                df = pd.read_parquet(ckpt_reg)
            except Exception as exc:
                log.warning("  Checkpoint load failed (%s) -- recomputing", exc)
                df = apply_isco_pipeline(df)
                df.to_parquet(ckpt_reg, index=False)
        else:
            df = apply_isco_pipeline(df)
            df.to_parquet(ckpt_reg, index=False)
            log.info("  Checkpoint saved -> %s", ckpt_reg)

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
        "total_rows":       len(df),
        "unique_users":     df["person_id"].nunique(),
        "unique_sectors":   df["esco_sector"].nunique(),
        "year_range":       f"{df['year'].min()}-{df['year'].max()}",
        "unique_job_titles": df["matched_label"].nunique(),
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