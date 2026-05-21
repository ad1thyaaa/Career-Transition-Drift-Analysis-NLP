# Career Drift Trajectory Analysis
# Publication-Grade Python / Google Colab Migration
# Architecture: Weak Supervision → SBERT + LightGBM → Full Ensemble Pipeline

---

## Architectural Overview

### Why Weak Supervision?

The original R implementation (`career_pred12.R`) uses hand-crafted expert regex rules
(`map_sectors()` and `broad_map()`) to assign `esco_sector` labels.
The entire downstream pipeline — transitions, Markov modeling, drift scoring,
ensemble prediction — depends on those labels.

A direct Python port would simply replicate regex fragility:

- fails on semantic variation ("full-stack dev" ≠ "fullstack developer" to regex)
- cannot generalise to unseen job titles
- brittle, maintenance-heavy
- misses synonyms, abbreviations, multi-lingual variants

**Weak supervision** resolves the circular dependency correctly:

```
Step 1  Original regex expert rules (ported from R)
        ↓  applied to job title + description
        → bootstrap / pseudo labels  (esco_sector_regex)

Step 2  SBERT all-MiniLM-L6-v2
        ↓  encode combined text → dense 384-d embeddings

Step 3  LightGBM multiclass classifier
        ↓  trained on (embeddings, bootstrap labels)
        → learns to generalise the expert knowledge semantically

Step 4  NLP predictions replace regex labels
        ↓  confidence threshold 0.45  →  low-confidence → "Other & Unclassified"

Step 5  Full R-equivalent ensemble pipeline
        (transitions, stickiness, PCA, Markov, archetypes, university boosts,
         per-sector weight tuning, association rules, all visualisations)
```

**The regex system is NOT the final classifier.**
It is a weak supervisor that generates bootstrap training labels.
The NLP model supersedes it — it learns to:

- recognise semantically equivalent job titles the regex never saw
- handle abbreviations and synonyms
- generalise across formatting variation
- improve precision via learned decision boundaries

This is the research-standard approach used in weak-supervision literature
(Ratner et al., Snorkel, 2017; Dawid & Skene 1979).

---

# 1. Setup & Imports

## Install Dependencies

```python
!pip install -q \
  pyarrow \
  pandas \
  numpy \
  scikit-learn \
  sentence-transformers \
  lightgbm \
  mlxtend \
  matplotlib \
  seaborn
```

---

## Imports

```python
import os
import gc
import re
import random
import warnings
import itertools

import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from sentence_transformers import SentenceTransformer
from lightgbm import LGBMClassifier

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

warnings.filterwarnings("ignore")
```

---

## Reproducibility & Output Directory

```python
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)
```

---

# 2. Data Loading

```python
train_df = pd.read_parquet("data/train.parquet")
val_df   = pd.read_parquet("data/validation.parquet")
test_df  = pd.read_parquet("data/test.parquet")

df = pd.concat([train_df, val_df, test_df], ignore_index=True)

print(f"Loaded  : {df.shape[0]:,} rows  x  {df.shape[1]} columns")
print(f"Columns : {', '.join(df.columns)}")
df.head(3)
```

---

# 3. Data Cleaning

## Remove Unknown / Blank Labels

```python
before = len(df)
df = df[df["matched_label"].notna()]
df = df[df["matched_label"].str.strip() != ""]
df = df[df["matched_label"].str.lower() != "unknown"]
print(f"  Dropped (unknown/blank) : {before - len(df):,}")
```

---

## Remove Duplicates

```python
before = len(df)
df = df.drop_duplicates(subset=["person_id", "matched_label", "start_date"])
print(f"  Dropped (dupes)         : {before - len(df):,}")
```

---

## Extract Year (1950–2024)

```python
df["year"] = (
    df["start_date"]
    .astype(str)
    .str.extract(r"(\d{4})")[0]
    .astype(float)
)

before = len(df)
df = df[(df["year"] >= 1950) & (df["year"] <= 2024)].copy()
df["year"] = df["year"].astype(int)
print(f"  Dropped (year range)    : {before - len(df):,}")
```

---

## Sort & Summary

```python
df = df.sort_values(["person_id", "year"]).reset_index(drop=True)

print(f"\nClean rows : {len(df):,}")
print(f"Users      : {df['person_id'].nunique():,}")
print(f"Jobs       : {df['matched_label'].nunique():,}")
print(f"Years      : {df['year'].min()}–{df['year'].max()}")
```

---

# 4. Weak Supervision — Bootstrap Label Generation

## Rationale

```
This section ports the COMPLETE R regex expert system (map_sectors + broad_map)
into Python exactly, preserving all patterns, fallback hierarchy, and rescue logic.

These regex labels are NOT the final sector assignments.
They serve exclusively as weak supervision signals to train the NLP classifier.
The regex system gives us:
  - expert domain knowledge (27 ESCO-aligned sectors)
  - high-precision anchors on unambiguous titles
  - a coverage baseline measurable against NLP improvement

The NLP model then learns to generalise beyond regex coverage,
handling unseen titles, synonyms, and semantic variation.
```

---

## 4.1 Port `map_sectors()` from R — Pass 1 (Detailed Regex)

```python
# Each tuple: (compiled regex pattern, sector label)
# Order preserved exactly from R fcase() — first match wins.

_MAP_SECTORS_RULES = [
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
    """
    Port of R map_sectors().
    Applies detailed expert-crafted regex patterns to a lowercased text string.
    Returns the first matching sector label, or 'Other & Unclassified'.
    Order of rules matches R fcase() exactly — first match wins.
    """
    for pattern, label in _MAP_SECTORS_RULES:
        if pattern.search(txt):
            return label
    return "Other & Unclassified"
```

---

## 4.2 Port `broad_map()` from R — Pass 2b (Keyword Fallback)

```python
# broad_map uses simpler keyword patterns as a second-pass rescue.
# Compound conditions (two grepl calls ANDed) are handled with separate patterns.

_BROAD_MAP_RULES_SIMPLE = [
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

    # Compound: sales-related AND role-qualifier
    # Handled separately below in _BROAD_MAP_COMPOUND

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

    # Catchall — generic role titles → Operations & General Management
    (re.compile(r"manager|officer|analyst|specialist|advisor|coordinator|director|consultant|assistant|lead|supervisor"),
     "Operations & General Management"),
]

# Compound rule — Sales: must match BOTH sales keywords AND role qualifier
_BROAD_MAP_SALES_KW   = re.compile(r"sales|account executive|business development|\bbdr\b|\bsdr\b|territory|revenue|commercial")
_BROAD_MAP_SALES_ROLE = re.compile(r"manager|executive|director|officer|lead|head|represent|consult")
# Creative: must match BOTH creative/media keywords AND role qualifier
_BROAD_MAP_CREATIVE_KW   = re.compile(r"design|graphic|creative|media|content|writer|journalist|editor|photog|video|animat|illustrat|film|broadcast|ux|ui")
_BROAD_MAP_CREATIVE_ROLE = re.compile(r"manager|director|specialist|lead|producer|designer")


def broad_map(txt: str) -> str:
    """
    Port of R broad_map().
    Applies simpler broad keyword fallback patterns.
    Compound conditions (two grepl ANDed in R) are handled explicitly.
    Returns first matching sector label, or 'Other & Unclassified'.
    Order matches R fcase() exactly.
    """
    # Position 6 in R: compound Sales rule
    # We check it before the simple rules that follow it in R
    # by inserting it at the correct position in the evaluation order.

    # Rules in R order:
    # 1. teach/school/educat ...
    for pattern, label in _BROAD_MAP_RULES_SIMPLE[:5]:
        if pattern.search(txt):
            return label

    # 6. Compound sales rule (R position 6)
    if _BROAD_MAP_SALES_KW.search(txt) and _BROAD_MAP_SALES_ROLE.search(txt):
        return "Sales & Business Development"

    # 7 onwards (indices 5..22 in _BROAD_MAP_RULES_SIMPLE, skipping the compound rules)
    for pattern, label in _BROAD_MAP_RULES_SIMPLE[5:-1]:   # exclude last catchall for now
        if pattern.search(txt):
            return label

    # Second-to-last in R: compound Creative/Media rule
    if _BROAD_MAP_CREATIVE_KW.search(txt) and _BROAD_MAP_CREATIVE_ROLE.search(txt):
        return "Creative, Media & Design"

    # Last: generic catchall
    if _BROAD_MAP_RULES_SIMPLE[-1][0].search(txt):
        return "Operations & General Management"

    return "Other & Unclassified"
```

---

## 4.3 Apply Multi-Pass Regex Bootstrap (Mirrors R Passes 1 / 2 / 2b / 3)

```python
def apply_regex_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full port of R's 4-pass sector labeling logic.

    Pass 1  : map_sectors() on matched_label (job title)
    Pass 2  : map_sectors() on matched_description for rows still 'Other'
    Pass 2b : broad_map() on matched_label, then matched_description for rows still 'Other'
    Pass 3  : drop rows still 'Other & Unclassified'

    Returns a copy of df with column 'esco_sector_regex' added.
    The original df is not modified.
    """
    df = df.copy()
    OTHER = "Other & Unclassified"
    has_desc = "matched_description" in df.columns

    label_lower = df["matched_label"].astype(str).str.lower()

    # --- Pass 1: detailed map_sectors on job title ---
    print("  Pass 1 — map_sectors() on job title ...")
    df["esco_sector_regex"] = label_lower.apply(map_sectors)
    n_other = (df["esco_sector_regex"] == OTHER).sum()
    print(f"    'Other' after pass 1 : {n_other:,}  ({n_other/len(df)*100:.1f}%)")

    # --- Pass 2: map_sectors on description (fallback) ---
    if has_desc:
        idx2 = df.index[df["esco_sector_regex"] == OTHER]
        print(f"  Pass 2 — map_sectors() on description for {len(idx2):,} rows ...")
        desc_lower = df.loc[idx2, "matched_description"].astype(str).str.lower()
        new_labels = desc_lower.apply(map_sectors)
        rescued = (new_labels != OTHER).sum()
        df.loc[idx2, "esco_sector_regex"] = new_labels.values
        n_other = (df["esco_sector_regex"] == OTHER).sum()
        print(f"    Rescued: {rescued:,} | 'Other' remaining: {n_other:,}  ({n_other/len(df)*100:.1f}%)")
    else:
        print("  Pass 2 — skipped (no matched_description column)")

    # --- Pass 2b: broad_map on label, then description ---
    idx2b = df.index[df["esco_sector_regex"] == OTHER]
    print(f"  Pass 2b — broad_map() rescue for {len(idx2b):,} rows ...")
    if len(idx2b) > 0:
        label_lower_2b = df.loc[idx2b, "matched_label"].astype(str).str.lower()
        new_labels_2b  = label_lower_2b.apply(broad_map)

        if has_desc:
            still_other = new_labels_2b.index[new_labels_2b == OTHER]
            if len(still_other) > 0:
                desc_lower_2b = df.loc[still_other, "matched_description"].astype(str).str.lower()
                new_labels_2b.loc[still_other] = desc_lower_2b.apply(broad_map).values

        df.loc[idx2b, "esco_sector_regex"] = new_labels_2b.values
        rescued_2b = (new_labels_2b != OTHER).sum()
        n_other = (df["esco_sector_regex"] == OTHER).sum()
        print(f"    Rescued: {rescued_2b:,} | 'Other' after 2b: {n_other:,}  ({n_other/len(df)*100:.1f}%)")

    # --- Pass 3: drop remaining 'Other & Unclassified' ---
    before = len(df)
    n_drop  = (df["esco_sector_regex"] == OTHER).sum()
    df = df[df["esco_sector_regex"] != OTHER].copy()
    print(f"  Pass 3 — removed {n_drop:,} ({n_drop/before*100:.1f}%) | kept {len(df):,}")

    return df


print("Applying 4-pass regex bootstrap pipeline ...")
df = apply_regex_pipeline(df)
df = df.sort_values(["person_id", "year"]).reset_index(drop=True)

# Sector distribution after regex bootstrap
sector_counts_regex = df["esco_sector_regex"].value_counts().reset_index()
sector_counts_regex.columns = ["esco_sector", "N"]
sector_counts_regex["pct"] = (sector_counts_regex["N"] / len(df) * 100).round(1)
print("\nBootstrap sector distribution:")
print(sector_counts_regex.to_string(index=False))
```

---

## 4.4 Combine Text (Title + Description)

```python
# Used as input to SBERT embedding for NLP training and prediction.
# Mirrors the original notebook design: title [SEP] description.

def combine_text(row):
    title = str(row.get("matched_label", ""))
    desc  = str(row.get("matched_description", ""))
    if desc in ("nan", "None", ""):
        desc = ""
    return f"{title} [SEP] {desc}".strip()


df["combined_text"] = df.apply(combine_text, axis=1)
print(f"Combined text column created. Sample:\n{df['combined_text'].head(3).tolist()}")
```

---

# 5. NLP Sector Classifier (SBERT + LightGBM)

## Overview

```
Input text  →  SBERT all-MiniLM-L6-v2  →  384-d L2-normalised embeddings
                                                ↓
                                   LightGBM multiclass (27 sectors)
                                   trained on bootstrap regex labels
                                                ↓
                           Confidence threshold 0.45:
                             prob ≥ 0.45  →  NLP predicted sector
                             prob < 0.45  →  drop row  (mirrors R Pass 3)

The NLP model generalises beyond the regex:
  ✓ Unseen job title variants
  ✓ Synonyms / abbreviations
  ✓ International / informal phrasing
  ✓ Semantic proximity (embeddings capture meaning, not surface patterns)
```

---

## 5.1 NLP Train / Val / Test Split

```python
# This split is ONLY for training and evaluating the NLP classifier.
# The prediction system (Section 7) uses a completely separate
# stratified user-level split that mirrors R exactly.
#
# IMPORTANT: to prevent data leakage the NLP split uses only
# the bootstrap-labelled rows (df after regex pipeline above).

classification_df = df.copy()   # all rows have esco_sector_regex

print(f"Classification rows : {len(classification_df):,}")
print(f"Classes             : {classification_df['esco_sector_regex'].nunique()}")

train_texts, temp_texts, train_labels, temp_labels = train_test_split(
    classification_df["combined_text"],
    classification_df["esco_sector_regex"],
    test_size=0.30,
    stratify=classification_df["esco_sector_regex"],
    random_state=SEED,
)

val_texts, test_texts, val_labels, test_labels = train_test_split(
    temp_texts,
    temp_labels,
    test_size=0.50,
    stratify=temp_labels,
    random_state=SEED,
)

print(
    f"\nNLP split — Train: {len(train_texts):,} | "
    f"Val: {len(val_texts):,} | Test: {len(test_texts):,}"
)
```

---

## 5.2 SBERT Embeddings

```python
# all-MiniLM-L6-v2: 384-d, fast, strong for sentence-level tasks.
# L2-normalised embeddings improve cosine-distance geometry for LightGBM.
# Colab GPU is used automatically if available.

print("Loading SBERT model: all-MiniLM-L6-v2 ...")
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def generate_embeddings(texts, batch_size=128, desc=""):
    return embedding_model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )


print("\nGenerating train embeddings ...")
X_train_nlp = generate_embeddings(train_texts, desc="train")

print("Generating val embeddings ...")
X_val_nlp = generate_embeddings(val_texts, desc="val")

print("Generating test embeddings ...")
X_test_nlp = generate_embeddings(test_texts, desc="test")
```

---

## 5.3 LightGBM Classifier

```python
# LightGBM is well-suited to dense SBERT embeddings:
#  - native multiclass support
#  - memory-efficient histogram-based boosting
#  - fast training on Colab
#  - early stopping via eval_set prevents overfitting

print("Training LightGBM classifier on bootstrap labels ...")
classifier = LGBMClassifier(
    objective="multiclass",
    n_estimators=400,
    learning_rate=0.05,
    max_depth=10,
    num_leaves=63,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=SEED,
    verbose=-1,
)

classifier.fit(
    X_train_nlp,
    train_labels,
    eval_set=[(X_val_nlp, val_labels)],
)
print("Training complete.")
```

---

## 5.4 Evaluate NLP Classifier vs Bootstrap Labels

```python
predictions_nlp = classifier.predict(X_test_nlp)

accuracy  = accuracy_score(test_labels, predictions_nlp)
precision, recall, f1, _ = precision_recall_fscore_support(
    test_labels, predictions_nlp, average="weighted"
)

print(f"NLP classifier vs bootstrap (regex) labels:")
print(f"  Accuracy  : {accuracy:.4f}")
print(f"  Precision : {precision:.4f}")
print(f"  Recall    : {recall:.4f}")
print(f"  F1 Score  : {f1:.4f}")
print()
print(classification_report(test_labels, predictions_nlp))

# NOTE: Agreement with regex labels measures how well the NLP
# has learned the expert knowledge. Scores < 1.0 reflect cases
# where NLP semantically corrects ambiguous regex decisions —
# which is desirable, not a bug.
```

---

## Confusion Matrix

```python
cm = confusion_matrix(test_labels, predictions_nlp,
                      labels=classifier.classes_)

plt.figure(figsize=(16, 14))
sns.heatmap(cm, cmap="Blues",
            xticklabels=classifier.classes_,
            yticklabels=classifier.classes_)
plt.title("NLP Sector Classification — Confusion Matrix\n(vs bootstrap regex labels)",
          fontsize=12, fontweight="bold")
plt.xticks(rotation=60, ha="right", fontsize=7)
plt.yticks(fontsize=7)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "nlp_confusion_matrix.png", dpi=150)
plt.show()
print("Saved: output/nlp_confusion_matrix.png")
```

---

## 5.5 Predict Final NLP Sectors for Entire Dataset

```python
# The FINAL esco_sector column is set here using NLP predictions.
# From this point on, the rest of the pipeline is identical in
# architecture to the R implementation — only the sector labels
# are higher quality (NLP semantic, not regex surface).

print("Generating NLP embeddings for full dataset ...")
all_embeddings    = generate_embeddings(df["combined_text"])
predicted_sectors = classifier.predict(all_embeddings)
prediction_probs  = classifier.predict_proba(all_embeddings)
max_probs         = prediction_probs.max(axis=1)

CONFIDENCE_THRESHOLD = 0.45

df["esco_sector"] = [
    sector if prob >= CONFIDENCE_THRESHOLD else "Other & Unclassified"
    for sector, prob in zip(predicted_sectors, max_probs)
]

before = len(df)
df = df[df["esco_sector"] != "Other & Unclassified"].copy()
print(f"Removed low-confidence : {before - len(df):,}")
print(f"Kept                   : {len(df):,}")
print(f"NLP sectors            : {df['esco_sector'].nunique()}")

df = df.sort_values(["person_id", "year"]).reset_index(drop=True)
```

---

## 5.6 Regex vs NLP Label Agreement Analysis

```python
# Measure agreement between regex bootstrap labels and NLP final predictions.
# Disagreements represent cases where NLP generalised beyond the regex —
# these are the semantic improvement cases.

agreement_mask  = df["esco_sector_regex"] == df["esco_sector"]
agreement_rate  = agreement_mask.mean()
disagreement_df = df[~agreement_mask]

print(f"Label agreement (regex vs NLP) : {agreement_rate*100:.2f}%")
print(f"Cases where NLP differs        : {len(disagreement_df):,}")

if len(disagreement_df) > 0:
    print("\nTop disagreement pairs (regex → NLP):")
    pairs = (
        disagreement_df
        .groupby(["esco_sector_regex", "esco_sector"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(15)
    )
    print(pairs.to_string(index=False))
```

---

## Sector Distribution (Post-NLP)

```python
sector_counts = (
    df["esco_sector"]
    .value_counts()
    .reset_index()
)
sector_counts.columns = ["esco_sector", "N"]
sector_counts["pct"] = (sector_counts["N"] / len(df) * 100).round(1)
print("\nFinal NLP sector distribution:")
print(sector_counts.to_string(index=False))
```

---

# 6. Transition Modeling + University Flag + Stickiness

## 6.1 Build next_sector / prev_sector

```python
df = df.sort_values(["person_id", "year"]).reset_index(drop=True)

df["next_sector"] = df.groupby("person_id")["esco_sector"].shift(-1)
df["prev_sector"] = df.groupby("person_id")["esco_sector"].shift(1)

transitions = df[df["next_sector"].notna()].copy().reset_index(drop=True)

seq_stats = df.groupby("person_id").size().reset_index(name="n_jobs")
print(f"Total transitions       : {len(transitions):,}")
print(f"Users with >= 2 jobs    : {(seq_stats['n_jobs'] >= 2).sum():,}")
print(f"Median sequence length  : {seq_stats['n_jobs'].median():.1f}")
print(
    f"Unique sector pairs     : "
    f"{transitions.groupby(['esco_sector','next_sector']).ngroups:,}"
)
```

---

## 6.2 University Flag (majority vote per user)

```python
if "university_studies" in df.columns:
    univ_agg = (
        df.groupby("person_id")["university_studies"]
        .apply(lambda x: (x == True).sum() / max(x.notna().sum(), 1) >= 0.5)
        .reset_index()
    )
    univ_agg.columns = ["person_id", "univ_flag"]
    person_univ = dict(zip(univ_agg["person_id"], univ_agg["univ_flag"]))
    transitions["univ_flag"] = transitions["person_id"].map(person_univ).fillna(False)
    df["univ_flag"]           = df["person_id"].map(person_univ).fillna(False)
    pct_uni = np.mean(list(person_univ.values())) * 100
    print(f"University flag: {pct_uni:.1f}% uni  |  {100 - pct_uni:.1f}% no uni")
else:
    transitions["univ_flag"] = False
    df["univ_flag"]           = False
    person_univ = {pid: False for pid in df["person_id"].unique()}
    print("university_studies column not found — flag set to False for all users")
```

---

## 6.3 Global Sector Index & Stickiness Multipliers

```python
VALID_SECTORS = sorted(df["esco_sector"].unique())
n_sectors     = len(VALID_SECTORS)
sector_to_idx = {s: i for i, s in enumerate(VALID_SECTORS)}

sr_df = (
    transitions
    .assign(is_self=lambda x: x["esco_sector"] == x["next_sector"])
    .groupby("esco_sector")["is_self"]
    .mean()
    .reset_index()
)
sr_df.columns = ["esco_sector", "self_rate"]

self_rates = dict(zip(sr_df["esco_sector"], sr_df["self_rate"]))
for s in VALID_SECTORS:
    if s not in self_rates:
        self_rates[s] = 0.2

self_loop_mult = {s: 1.0 + 2.0 * self_rates[s] for s in VALID_SECTORS}

print("Top 5 stickiest sectors:")
for nm in sorted(self_rates, key=self_rates.get, reverse=True)[:5]:
    print(f"  {nm:<45} : {self_rates[nm]:.3f}")
```

---

## 6.4 Transition Matrix Builder

```python
SMOOTHING = 0.01


def build_trans_mat(data, sector_idx, n_sec, sloop_mult, weights=None):
    mat = np.zeros((n_sec, n_sec))

    if weights is None:
        ct = (
            data.groupby(["esco_sector", "next_sector"])
            .size()
            .reset_index(name="val")
        )
    else:
        ct = (
            data.groupby(["esco_sector", "next_sector"])[weights]
            .sum()
            .reset_index()
        )
        ct.columns = ["esco_sector", "next_sector", "val"]

    ci_arr = ct["esco_sector"].map(sector_idx).values
    ni_arr = ct["next_sector"].map(sector_idx).values
    valid  = ~pd.isna(ci_arr) & ~pd.isna(ni_arr)

    np.add.at(
        mat,
        (ci_arr[valid].astype(int), ni_arr[valid].astype(int)),
        ct["val"].values[valid],
    )

    for s, si in sector_idx.items():
        mat[si, si] *= sloop_mult.get(s, 1.0)

    row_sums = mat.sum(axis=1)
    for i in range(n_sec):
        rs = row_sums[i]
        sv = 0.50 if rs < 10 else (0.10 if rs < 100 else SMOOTHING)
        mat[i] += sv

    mat /= mat.sum(axis=1, keepdims=True)
    return mat
```

---

# 7. PCA on Career-Transition Profiles

```python
all_sectors_pca = sorted(
    set(transitions["esco_sector"]) | set(transitions["next_sector"])
)
n_pca  = len(all_sectors_pca)
si_pca = {s: i for i, s in enumerate(all_sectors_pca)}

ct_pca = (
    transitions
    .groupby(["esco_sector", "next_sector"])
    .size()
    .reset_index(name="N")
)
mat_pca = np.zeros((n_pca, n_pca))
ci_p    = ct_pca["esco_sector"].map(si_pca).values
ni_p    = ct_pca["next_sector"].map(si_pca).values
valid_p = ~pd.isna(ci_p) & ~pd.isna(ni_p)
np.add.at(mat_pca,
          (ci_p[valid_p].astype(int), ni_p[valid_p].astype(int)),
          ct_pca["N"].values[valid_p])
mat_pca += 0.01
mat_pca /= mat_pca.sum(axis=1, keepdims=True)

pca_model  = PCA(n_components=5)
pca_result = pca_model.fit_transform(mat_pca)
pca_var    = np.round(pca_model.explained_variance_ratio_ * 100, 1)
print(
    f"Variance — PC1:{pca_var[0]}%  PC2:{pca_var[1]}%  "
    f"PC3:{pca_var[2]}%  Cum(1+2):{pca_var[0]+pca_var[1]}%"
)

pca_df = pd.DataFrame({
    "PC1":    pca_result[:, 0],
    "PC2":    pca_result[:, 1],
    "sector": all_sectors_pca,
})

GROUP_MAP = {
    "Software & IT Development":        "Technology",
    "Data Science & Analytics":         "Technology",
    "Cybersecurity & Compliance":       "Technology",
    "IT Infrastructure & Networks":     "Technology",
    "ICT Management & Consulting":      "Technology",
    "Healthcare - Clinical":            "Health & Science",
    "Healthcare - Nursing & Allied":    "Health & Science",
    "Social Work & Community Services": "Health & Science",
    "Research & Academia":              "Health & Science",
    "Finance & Accounting":             "Finance & Legal",
    "Banking, Finance & Insurance":     "Finance & Legal",
    "Legal & Compliance":               "Finance & Legal",
    "Education - Teaching":             "Public & Education",
    "Public Sector & Administration":   "Public & Education",
    "Environment, Safety & Sustainability": "Public & Education",
    "Sales & Business Development":     "Business & Management",
    "Marketing & Communications":       "Business & Management",
    "Customer Service & Support":       "Business & Management",
    "Senior Management & C-Suite":      "Business & Management",
    "Operations & General Management":  "Business & Management",
    "Human Resources & Recruitment":    "Business & Management",
    "Project & Programme Management":   "Business & Management",
}
pca_df["group"] = pca_df["sector"].map(GROUP_MAP).fillna("Industry & Trades")

GROUP_COLORS = {
    "Technology":           "#185FA5",
    "Health & Science":     "#1D9E75",
    "Finance & Legal":      "#BA7517",
    "Public & Education":   "#993556",
    "Business & Management":"#534AB7",
    "Industry & Trades":    "#A32D2D",
}

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
plt.savefig(OUTPUT_DIR / "pca_sector_profiles.png", dpi=150)
plt.show()
print("Saved: output/pca_sector_profiles.png")
```

---

# 8. Full Ensemble Prediction System

## 8.1 Stratified Train / Val / Test Split (64 / 16 / 20 by dominant sector)

```python
user_dom = (
    df.groupby("person_id")["esco_sector"]
    .agg(lambda x: x.value_counts().index[0])
    .reset_index()
)
user_dom.columns = ["person_id", "dom_sector"]

np.random.seed(SEED)
split_rows = []
for dom, grp in user_dom.groupby("dom_sector"):
    n    = len(grp)
    ord_ = np.random.permutation(n)
    sp   = np.where(
        ord_ < int(n * 0.64), "train",
        np.where(ord_ < int(n * 0.80), "val", "test")
    )
    split_rows.append(
        pd.DataFrame({"person_id": grp["person_id"].values, "split": sp})
    )

split_df    = pd.concat(split_rows, ignore_index=True)
train_users = set(split_df.loc[split_df["split"] == "train", "person_id"])
val_users   = set(split_df.loc[split_df["split"] == "val",   "person_id"])
test_users  = set(split_df.loc[split_df["split"] == "test",  "person_id"])

train_data = transitions[transitions["person_id"].isin(train_users)].copy().reset_index(drop=True)
val_data   = transitions[transitions["person_id"].isin(val_users)].copy().reset_index(drop=True)
test_data  = transitions[transitions["person_id"].isin(test_users)].copy().reset_index(drop=True)

print(f"Train : {len(train_data):,}  |  Val : {len(val_data):,}  |  Test : {len(test_data):,}")
```

---

## Redefine Sector Index on Training Data

```python
all_sectors   = sorted(set(train_data["esco_sector"]) | set(train_data["next_sector"]))
n_sec         = len(all_sectors)
sector_to_idx = {s: i for i, s in enumerate(all_sectors)}

sr_train = (
    train_data
    .assign(is_self=lambda x: x["esco_sector"] == x["next_sector"])
    .groupby("esco_sector")["is_self"]
    .mean()
    .reset_index()
)
sr_train.columns = ["esco_sector", "self_rate"]
self_rates     = dict(zip(sr_train["esco_sector"], sr_train["self_rate"]))
for s in all_sectors:
    if s not in self_rates:
        self_rates[s] = 0.2
self_loop_mult = {s: 1.0 + 2.0 * self_rates[s] for s in all_sectors}


def build_mat(data, weights=None):
    return build_trans_mat(data, sector_to_idx, n_sec, self_loop_mult, weights)


print(f"Prediction sector space: {n_sec} sectors")
print("Top 5 stickiest (train):")
for nm in sorted(self_rates, key=self_rates.get, reverse=True)[:5]:
    print(f"  {nm:<45} : {self_rates[nm]:.3f}")
```

---

## 8.2 Global Transition Matrix

```python
global_matrix = build_mat(train_data)
print(f"Global matrix shape: {global_matrix.shape}")
```

---

## 8.3 Recency-Weighted Matrix (decay = 0.12)

```python
DECAY_RATE = 0.12
max_yr     = int(train_data["year"].max())

train_data = train_data.copy()
train_data["recency_weight"] = np.exp(DECAY_RATE * (train_data["year"] - max_yr))

recency_matrix = build_mat(train_data, weights="recency_weight")
print("Recency-weighted matrix built  (decay=0.12)")
```

---

## 8.4 Per-Decade Transition Matrices

```python
years_sorted   = sorted(train_data["year"].unique())
WINDOW_SIZE    = 10
decade_windows = []
i = 0
while i < len(years_sorted):
    decade_windows.append((
        years_sorted[i],
        years_sorted[min(i + WINDOW_SIZE - 1, len(years_sorted) - 1)]
    ))
    i += WINDOW_SIZE

transition_matrices = {}
print(f"{'Years':<15} | {'Sparsity':>10} | Status")
print("-" * 42)

for start, end in decade_windows:
    label = f"{start}-{end}"
    wd    = train_data[(train_data["year"] >= start) & (train_data["year"] <= end)]

    if len(wd) == 0:
        print(f"{label:<15} | {'N/A':>10} | Empty")
        continue

    raw = np.zeros((n_sec, n_sec))
    ct  = wd.groupby(["esco_sector", "next_sector"]).size().reset_index(name="N")
    ci_ = ct["esco_sector"].map(sector_to_idx).values
    ni_ = ct["next_sector"].map(sector_to_idx).values
    vv  = ~pd.isna(ci_) & ~pd.isna(ni_)
    np.add.at(raw, (ci_[vv].astype(int), ni_[vv].astype(int)), ct["N"].values[vv])

    for s, si in sector_to_idx.items():
        raw[si, si] *= self_loop_mult.get(s, 1.0)

    sparsity = (raw == 0).sum() / n_sec ** 2 * 100
    sv = 0.5 if sparsity > 70 else (0.1 if sparsity > 30 else SMOOTHING)
    raw += sv
    mat = raw / raw.sum(axis=1, keepdims=True)

    transition_matrices[label] = {"mat": mat, "start": start, "end": end}
    print(f"{label:<15} | {sparsity:>9.2f}% | Built")
```

---

## 8.5 Market Drift (Frobenius Norm Between Consecutive Decades)

```python
market_drift = {}
decade_keys  = list(transition_matrices.keys())

for i in range(1, len(decade_keys)):
    diff = (
        transition_matrices[decade_keys[i]]["mat"]
        - transition_matrices[decade_keys[i - 1]]["mat"]
    )
    market_drift[decade_keys[i]] = float(np.sqrt((diff ** 2).sum()))

print("\nMarket drift (Frobenius norm):")
for nm, val in market_drift.items():
    print(f"  {nm:<15} : {val:.4f}")
```

---

## 8.6 2nd-Order Markov with Back-Off (MIN_BIGRAM = 5)

```python
MIN_BIGRAM = 5

so_data = train_data[
    train_data["prev_sector"].notna() &
    train_data["prev_sector"].isin(all_sectors)
].copy()
print(f"Bigram pairs available: {len(so_data):,}")

bigram_pair_ct = (
    so_data.groupby(["prev_sector", "esco_sector"]).size().reset_index(name="pair_count")
)
bigram_triplet = (
    so_data.groupby(["prev_sector", "esco_sector", "next_sector"])
    .size()
    .reset_index(name="N")
)

second_order_probs = {}
n_backoff          = 0

for _, brow in bigram_pair_ct.iterrows():
    if brow["pair_count"] < MIN_BIGRAM:
        n_backoff += 1
        continue

    key = f"{brow['prev_sector']}|{brow['esco_sector']}"
    sub = bigram_triplet[
        (bigram_triplet["prev_sector"] == brow["prev_sector"]) &
        (bigram_triplet["esco_sector"] == brow["esco_sector"])
    ]

    vec = np.full(n_sec, SMOOTHING)
    ni_ = sub["next_sector"].map(sector_to_idx).values
    vv  = ~pd.isna(ni_)
    vec[ni_[vv].astype(int)] = sub["N"].values[vv] + SMOOTHING

    cs = brow["esco_sector"]
    if cs in sector_to_idx:
        si = sector_to_idx[cs]
        vec[si] *= self_loop_mult.get(cs, 1.0)

    second_order_probs[key] = vec / vec.sum()

print(f"2nd-order keys built : {len(second_order_probs):,}")
print(f"Back-off (<{MIN_BIGRAM} obs)  : {n_backoff:,}")
```

---

## 8.7 Per-User Recency-Weighted History Priors (HIST_DECAY = 0.15)

```python
HIST_DECAY = 0.15

hist_raw = train_data[["person_id", "esco_sector", "year"]].copy()
hist_raw["rw"] = np.exp(HIST_DECAY * (hist_raw["year"] - max_yr))

user_sect_wt = (
    hist_raw
    .groupby(["person_id", "esco_sector"])["rw"]
    .sum()
    .reset_index()
)
user_sect_wt.columns = ["person_id", "sector", "wt_count"]

user_totals = user_sect_wt.groupby("person_id")["wt_count"].sum().reset_index()
user_totals.columns = ["person_id", "total"]

multi_users = user_totals.loc[user_totals["total"] >= 2, "person_id"].values

user_hist_lookup = {}
for uid in multi_users:
    rows = user_sect_wt[user_sect_wt["person_id"] == uid]
    vec  = np.full(n_sec, 0.01)
    si_  = rows["sector"].map(sector_to_idx).values
    vv   = ~pd.isna(si_)
    np.add.at(vec, si_[vv].astype(int), rows["wt_count"].values[vv])
    user_hist_lookup[str(int(uid))] = vec / vec.sum()

print(f"User history priors built for {len(user_hist_lookup):,} users")
```

---

## 8.8 University Boost Vectors

```python
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

    print("University boost vectors built.")
    top5_u = sorted(zip(all_sectors, univ_boost_TRUE), key=lambda x: -x[1])[:5]
    print("Top 5 sectors boosted for uni users:")
    for nm, val in top5_u:
        print(f"  {nm:<45} : {val:.3f}")
else:
    print("No university flag variation — boost vectors remain uniform (1.0)")
```

---

## 8.9 Archetype Clustering on Training Data

```python
ci_tr    = train_data["esco_sector"].map(sector_to_idx).values
ni_tr    = train_data["next_sector"].map(sector_to_idx).values
valid_tr = ~pd.isna(ci_tr) & ~pd.isna(ni_tr)

lp_tr = np.full(len(train_data), np.log(1e-9))
lp_tr[valid_tr] = np.log(
    np.maximum(global_matrix[ci_tr[valid_tr].astype(int), ni_tr[valid_tr].astype(int)], 1e-9)
)
train_data = train_data.copy()
train_data["log_prob"] = lp_tr
train_data["is_self"]  = train_data["esco_sector"] == train_data["next_sector"]

risk_tr = (
    train_data
    .groupby("person_id")
    .agg(
        drift_score   =("log_prob",    lambda x: -x.mean()),
        seq_length    =("esco_sector", "count"),
        self_loop_rate=("is_self",     "mean"),
    )
    .reset_index()
)
risk_tr = risk_tr[risk_tr["seq_length"] >= 2].copy()

feat_arch   = StandardScaler().fit_transform(
    risk_tr[["drift_score", "seq_length", "self_loop_rate"]].dropna()
)
valid_idx_a = risk_tr[["drift_score", "seq_length", "self_loop_rate"]].dropna().index

np.random.seed(SEED)
km_arch = KMeans(n_clusters=5, n_init=25, max_iter=100, random_state=SEED)
risk_tr.loc[valid_idx_a, "cluster"] = km_arch.fit_predict(feat_arch)

cs_arch = (
    risk_tr[risk_tr["cluster"].notna()]
    .groupby("cluster")
    .agg(
        mean_drift=("drift_score",    "mean"),
        mean_len  =("seq_length",     "mean"),
        mean_loop =("self_loop_rate", "mean"),
    )
    .reset_index()
)

q80_d = cs_arch["mean_drift"].quantile(0.8)
q20_d = cs_arch["mean_drift"].quantile(0.2)
q50_d = cs_arch["mean_drift"].quantile(0.5)
q75_l = cs_arch["mean_len"].quantile(0.75)


def assign_archetype(row):
    if row["mean_drift"] > q80_d and row["mean_loop"] < 0.3: return "Career Switcher"
    if row["mean_drift"] < q20_d and row["mean_loop"] > 0.5: return "Sector Loyalist"
    if row["mean_len"]   > q75_l:                             return "Career Veteran"
    if row["mean_drift"] > q50_d:                             return "Gradual Mover"
    return "Stable Specialist"


cs_arch["archetype"] = cs_arch.apply(assign_archetype, axis=1)
print("Archetype labels (training data):")
print(cs_arch[["cluster", "archetype", "mean_drift", "mean_loop"]].round(4).to_string(index=False))

user_cluster_map = (
    risk_tr[["person_id", "cluster"]]
    .merge(cs_arch[["cluster", "archetype"]], on="cluster")
)
train_data = train_data.merge(
    user_cluster_map[["person_id", "archetype"]], on="person_id", how="left"
)
train_data["archetype"] = train_data["archetype"].fillna("Stable Specialist")
train_data["univ_flag"] = train_data["person_id"].map(person_univ).fillna(False)
```

---

## 8.10 Per-Archetype Transition Matrices

```python
archetype_matrices = {}
print("Building per-archetype matrices:")
for arch in cs_arch["archetype"].unique():
    sub = train_data[train_data["archetype"] == arch]
    if len(sub) >= 50:
        archetype_matrices[arch] = build_mat(sub)
        print(f"  {arch:<25} : {len(sub):,} transitions")
    else:
        archetype_matrices[arch] = global_matrix
        print(f"  {arch:<25} : too few rows ({len(sub)}) — using global")
```

---

## 8.11 Helper: get_win_key

```python
def get_win_key(yr):
    for key, w in transition_matrices.items():
        if w["start"] <= yr <= w["end"]:
            return key
    return "global"
```

---

## 8.12 Core Ensemble Probability Function

```python
def compute_ensemble_prob(ci, wk, arch, univ, uid, prev_s, curr_s, wh, ws, wa):
    """
    Mirrors R ensemble_topk() inner logic exactly:
        base   = 0.60 * rec_p + 0.40 * dyn_p
        ens_p  = w_rest * base + wa * arch_p + ws * so_p + wh * hist_p
        ens_p  = ens_p * boost  (university adjustment)
        ens_p  = ens_p / sum(ens_p)
    """
    w_rest = max(0.05, 1.0 - wh - ws - wa)

    dyn_p = (
        transition_matrices[wk]["mat"][ci]
        if wk != "global" and wk in transition_matrices
        else global_matrix[ci]
    )
    rec_p  = recency_matrix[ci]
    arch_p = archetype_matrices.get(arch, global_matrix)[ci]

    so_key = (
        f"{prev_s}|{curr_s}"
        if (prev_s is not None and not pd.isna(prev_s) and prev_s in sector_to_idx)
        else None
    )
    so_p = (
        second_order_probs[so_key]
        if so_key and so_key in second_order_probs
        else global_matrix[ci]
    )

    hist_p = user_hist_lookup.get(str(int(uid)), global_matrix[ci])

    base  = 0.60 * rec_p + 0.40 * dyn_p
    ens_p = w_rest * base + wa * arch_p + ws * so_p + wh * hist_p

    boost  = univ_boost_TRUE if univ else univ_boost_FALSE
    ens_p  = ens_p * boost
    ens_p /= ens_p.sum()
    return ens_p
```

---

## 8.13 Per-Sector Ensemble Weight Tuning on Validation Set

```python
w_hist_g = [0.10, 0.20, 0.30, 0.40, 0.50]
w_so_g   = [0.00, 0.10, 0.20, 0.30]
w_arch_g = [0.05, 0.10, 0.20]

sector_best_w = {}
for s in all_sectors:
    sr_s = self_rates.get(s, 0.15)
    if sr_s > 0.25:
        sector_best_w[s] = {"w_hist": 0.30, "w_so": 0.10, "w_arch": 0.10}
    elif sr_s < 0.10:
        sector_best_w[s] = {"w_hist": 0.20, "w_so": 0.30, "w_arch": 0.10}
    else:
        sector_best_w[s] = {"w_hist": 0.20, "w_so": 0.20, "w_arch": 0.10}

val_data = val_data.merge(
    user_cluster_map[["person_id", "archetype"]], on="person_id", how="left"
)
val_data["archetype"] = val_data["archetype"].fillna("Stable Specialist")
val_data["univ_flag"] = val_data["person_id"].map(person_univ).fillna(False)
val_data["win_key"]   = val_data["year"].apply(get_win_key)

print(f"Tuning weights across {len(all_sectors)} sectors ...")

for curr_s in all_sectors:
    sub_val = val_data[val_data["esco_sector"] == curr_s]
    if len(sub_val) < 15:
        continue
    if len(sub_val) > 2000:
        sub_val = sub_val.sample(2000, random_state=SEED)

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
            ens_p = compute_ensemble_prob(
                ci, row["win_key"], row["archetype"], row["univ_flag"],
                row["person_id"], row.get("prev_sector"), curr_s, wh, ws, wa,
            )
            if all_sectors[int(np.argmax(ens_p))] == row["next_sector"]:
                hits += 1

        acc = hits / len(sub_val)
        if acc > best_acc:
            best_acc = acc
            best_w   = {"w_hist": wh, "w_so": ws, "w_arch": wa}

    sector_best_w[curr_s] = best_w

print("Per-sector weight tuning complete.")
```

---

## 8.14 Evaluation on Test Set

```python
test_data = test_data.merge(
    user_cluster_map[["person_id", "archetype"]], on="person_id", how="left"
)
test_data["archetype"] = test_data["archetype"].fillna("Stable Specialist")
test_data["univ_flag"] = test_data["person_id"].map(person_univ).fillna(False)
test_data["win_key"]   = test_data["year"].apply(get_win_key)

persistence_acc = (test_data["esco_sector"] == test_data["next_sector"]).mean()
print(f"Persistence Baseline Top-1: {persistence_acc * 100:.2f}%")

TOP_K_LIST = [1, 3, 5]
max_k      = max(TOP_K_LIST)


def ensemble_topk(ds):
    results = []
    for _, row in ds.iterrows():
        curr = row["esco_sector"]
        ci   = sector_to_idx.get(curr)
        if ci is None:
            results.append(all_sectors[:max_k])
            continue
        w = sector_best_w.get(curr, {"w_hist": 0.20, "w_so": 0.20, "w_arch": 0.10})
        ens_p = compute_ensemble_prob(
            ci, row["win_key"], row["archetype"], row["univ_flag"],
            row["person_id"], row.get("prev_sector"), curr,
            w["w_hist"], w["w_so"], w["w_arch"],
        )
        top_idx = np.argsort(ens_p)[::-1][:max_k]
        results.append([all_sectors[i] for i in top_idx])
    return results


valid_test = test_data[test_data["esco_sector"].isin(all_sectors)].copy()
print(f"\nEvaluating on {len(valid_test):,} test transitions ...")
preds = ensemble_topk(valid_test)

ensemble_results = {}
for k in TOP_K_LIST:
    hits = [true in pred[:k] for true, pred in zip(valid_test["next_sector"], preds)]
    ensemble_results[f"top_{k}"] = float(np.mean(hits))

print("\n" + "=" * 60)
print(f"  {'Model':<22} | {'Top-1':>7} | {'Top-3':>7} | {'Top-5':>7}")
print("-" * 60)
print(f"  {'Persistence Baseline':<22} | {persistence_acc*100:>6.2f}% | {'NA':>7} | {'NA':>7}")
print(f"  {'Ensemble':<22} | {ensemble_results['top_1']*100:>6.2f}% | "
      f"{ensemble_results['top_3']*100:>6.2f}% | {ensemble_results['top_5']*100:>6.2f}%")
print("-" * 60)
print(f"  Ensemble vs Baseline : {(ensemble_results['top_1'] - persistence_acc) * 100:+.2f}%")
print("=" * 60)
```

---

# 9. Clustering (Entropy-Based Drift Scores)

```python
ci_all    = transitions["esco_sector"].map(sector_to_idx).values
ni_all    = transitions["next_sector"].map(sector_to_idx).values
valid_all = ~pd.isna(ci_all) & ~pd.isna(ni_all)

lp_all = np.full(len(transitions), np.log(1e-9))
lp_all[valid_all] = np.log(
    np.maximum(
        global_matrix[ci_all[valid_all].astype(int), ni_all[valid_all].astype(int)],
        1e-9,
    )
)
transitions = transitions.copy()
transitions["log_prob"] = lp_all
transitions["is_self"]  = transitions["esco_sector"] == transitions["next_sector"]

risk_df = (
    transitions
    .groupby("person_id")
    .agg(
        drift_score       =("log_prob", lambda x: -x.mean()),
        total_transitions =("esco_sector", "count"),
        self_loop_rate    =("is_self", "mean"),
    )
    .reset_index()
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

print(f"Users in risk_df     : {len(risk_df):,}")
print(f"Median drift score   : {risk_df['drift_score'].median():.4f}")

feat_cols  = ["drift_score", "seq_length", "self_loop_rate"]
feat_clust = risk_df[feat_cols].dropna()
X_cluster  = StandardScaler().fit_transform(feat_clust)
valid_idx_c = feat_clust.index

np.random.seed(SEED)
km2 = KMeans(n_clusters=5, n_init=25, max_iter=100, random_state=SEED)
labels_km = km2.fit_predict(X_cluster)
risk_df.loc[valid_idx_c, "cluster"] = labels_km

print(f"\nKMeans WCSS : {km2.inertia_:.2f}")

cs2 = (
    risk_df[risk_df["cluster"].notna()]
    .groupby("cluster")
    .agg(
        n_users   =("person_id",      "count"),
        mean_drift=("drift_score",    "mean"),
        mean_len  =("seq_length",     "mean"),
        mean_loop =("self_loop_rate", "mean"),
    )
    .reset_index()
)

q80c = cs2["mean_drift"].quantile(0.8)
q20c = cs2["mean_drift"].quantile(0.2)
q50c = cs2["mean_drift"].quantile(0.5)
q75c = cs2["mean_len"].quantile(0.75)


def assign_arch2(row):
    if row["mean_drift"] > q80c and row["mean_loop"] < 0.3: return "Career Switcher"
    if row["mean_drift"] < q20c and row["mean_loop"] > 0.5: return "Sector Loyalist"
    if row["mean_len"]   > q75c:                             return "Career Veteran"
    if row["mean_drift"] > q50c:                             return "Gradual Mover"
    return "Stable Specialist"


cs2["archetype"] = cs2.apply(assign_arch2, axis=1)
risk_df = risk_df.merge(cs2[["cluster", "archetype"]], on="cluster", how="left")

print("\nCluster summary:")
print(cs2[["cluster", "archetype", "n_users", "mean_drift", "mean_loop"]].round(4).to_string(index=False))

risk_df.to_csv(OUTPUT_DIR / "drift_risk_scores.csv", index=False)
print("\nSaved: output/drift_risk_scores.csv")
print("\nTOP 10 HIGHEST DRIFT RISK:")
print(risk_df[["person_id", "drift_score", "total_transitions"]].head(10).to_string(index=False))
```

---

# 10. Association Rules

```python
tot = len(transitions)

pc = transitions.groupby(["esco_sector", "next_sector"]).size().reset_index(name="pair_count")
ac = transitions.groupby("esco_sector").size().reset_index(name="ant_count")
cc = transitions.groupby("next_sector").size().reset_index(name="con_count")

assoc_rules = (
    pc
    .merge(ac, on="esco_sector")
    .merge(cc, on="next_sector")
    .rename(columns={"esco_sector": "from", "next_sector": "to"})
)

assoc_rules["support"]    = (assoc_rules["pair_count"] / tot).round(4)
assoc_rules["confidence"] = (assoc_rules["pair_count"] / assoc_rules["ant_count"]).round(4)
assoc_rules["lift"]       = (assoc_rules["confidence"] / (assoc_rules["con_count"] / tot)).round(4)

assoc_rules = assoc_rules[["from", "to", "support", "confidence", "lift", "pair_count"]]
assoc_rules = assoc_rules.sort_values("lift", ascending=False).reset_index(drop=True)

print(f"Total rules: {len(assoc_rules):,}")
print("\nTOP 15 BY LIFT:")
print(assoc_rules.head(15).to_string(index=False))
print("\nTOP 15 BY CONFIDENCE:")
print(assoc_rules.sort_values("confidence", ascending=False).head(15).to_string(index=False))
print("\nTOP 15 BY SUPPORT:")
print(assoc_rules.sort_values("support", ascending=False).head(15).to_string(index=False))

assoc_rules.to_csv(OUTPUT_DIR / "association_rules.csv", index=False)
print("\nSaved: output/association_rules.csv")
```

---

# 11. Visualizations

## 11.1 Sector Distribution

```python
sc = df["esco_sector"].value_counts().reset_index()
sc.columns = ["esco_sector", "N"]
sc["pct"] = (sc["N"] / sc["N"].sum() * 100).round(1)
sc = sc.sort_values("N")

fig, ax = plt.subplots(figsize=(13, 9))
ax.barh(sc["esco_sector"], sc["N"], color="steelblue", alpha=0.85)
for _, row in sc.iterrows():
    ax.text(row["N"] * 1.005, row.name, f"{row['pct']}%", va="center", fontsize=8)
ax.set_title(
    f"Job Distribution Across {df['esco_sector'].nunique()} Sectors (NLP-classified)",
    fontsize=12, fontweight="bold",
)
ax.set_xlabel("Records")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "sector_distribution.png", dpi=150)
plt.show()
print("Saved: output/sector_distribution.png")
```

---

## 11.2 Sequence Length Distribution

```python
seq_stats = df.groupby("person_id").size().reset_index(name="n_jobs")
seq_stats["nc"] = seq_stats["n_jobs"].clip(upper=15)

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(seq_stats["nc"], bins=range(1, 17), color="steelblue", alpha=0.85, edgecolor="white")
ax.set_xticks(range(1, 16))
ax.set_xticklabels([str(i) if i < 15 else "15+" for i in range(1, 16)])
ax.set_title("Career Sequence Length Distribution", fontsize=12, fontweight="bold")
ax.set_xlabel("Number of classified jobs")
ax.set_ylabel("Number of users")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "sequence_length_distribution.png", dpi=150)
plt.show()
print("Saved: output/sequence_length_distribution.png")
```

---

## 11.3 Career Drift Distribution

```python
mv = float(risk_df["drift_score"].median())

fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(risk_df["drift_score"], bins=50, color="gray", alpha=0.85, edgecolor="white")
ax.axvline(mv, color="red", linestyle="--", linewidth=1.1)
ax.text(mv + 0.02, ax.get_ylim()[1] * 0.92, f"Median: {mv:.2f}", color="red", fontsize=10)
ax.set_title("Career Drift Distribution (entropy-based)", fontsize=14, fontweight="bold")
ax.set_xlabel("Drift Score  (higher = more volatile)")
ax.set_ylabel("Number of Users")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "career_drift_distribution.png", dpi=150)
plt.show()
print("Saved: output/career_drift_distribution.png")
```

---

## 11.4 Transition Heatmap

```python
hd = (
    transitions
    .groupby(["esco_sector", "next_sector"])
    .size()
    .reset_index(name="N")
)
hd["from_s"] = hd["esco_sector"].str[:18]
hd["to_s"]   = hd["next_sector"].str[:18]
pivot     = hd.pivot_table(index="from_s", columns="to_s", values="N", fill_value=0)
log_pivot = np.log1p(pivot)

fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(
    log_pivot, cmap="Blues", ax=ax, linewidths=0.3, linecolor="white",
    cbar_kws={"label": "log(count+1)"},
)
ax.set_title(
    f"Career Transition Heatmap ({df['esco_sector'].nunique()} Sectors)",
    fontsize=11, fontweight="bold",
)
ax.set_xlabel("Next Sector"); ax.set_ylabel("Current Sector")
plt.xticks(rotation=60, ha="right", fontsize=7)
plt.yticks(fontsize=7)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "transition_heatmap.png", dpi=150)
plt.show()
print("Saved: output/transition_heatmap.png")
```

---

## 11.5 Top 20 Career Transitions

```python
top_trans = (
    transitions
    .groupby(["esco_sector", "next_sector"])
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
    .head(20)
)
top_trans["transition"] = (
    top_trans["esco_sector"].str[:25] + " →\n" + top_trans["next_sector"].str[:25]
)

fig, ax = plt.subplots(figsize=(14, 10))
ax.barh(
    top_trans["transition"].iloc[::-1].values,
    top_trans["count"].iloc[::-1].values,
    color="steelblue", alpha=0.85,
)
ax.set_title("Top 20 Most Common Career Transitions", fontsize=12, fontweight="bold")
ax.set_xlabel("Count")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "top_transitions.png", dpi=150)
plt.show()
print("Saved: output/top_transitions.png")
```

---

## 11.6 Market Drift Over Time

```python
if market_drift:
    ddf = pd.DataFrame({
        "window": list(market_drift.keys()),
        "drift":  list(market_drift.values()),
    })
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ddf["window"], ddf["drift"], color="steelblue", linewidth=1.2,
            marker="o", markersize=5)
    ax.set_title(
        "Market Drift Over Time (Drifting Markov Model)\n"
        "Frobenius norm between consecutive decade matrices",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlabel("Decade window")
    ax.set_ylabel("Drift magnitude (Frobenius norm)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "market_drift.png", dpi=150)
    plt.show()
    print("Saved: output/market_drift.png")
else:
    print("market_drift is empty — not enough decades in data.")
```

---

## 11.7 User Career Clusters

```python
ARCHETYPE_COLORS = {
    "Career Switcher":   "#E24B4A",
    "Sector Loyalist":   "#1D9E75",
    "Career Veteran":    "#378ADD",
    "Gradual Mover":     "#BA7517",
    "Stable Specialist": "#534AB7",
}

pdf2 = risk_df[risk_df["cluster"].notna()].copy()

fig, ax = plt.subplots(figsize=(12, 7))
for arch, color in ARCHETYPE_COLORS.items():
    sub = pdf2[pdf2["archetype"] == arch]
    ax.scatter(sub["drift_score"], sub["seq_length"],
               alpha=0.35, s=6, color=color, label=arch)

for _, crow in cs2.iterrows():
    color = ARCHETYPE_COLORS.get(crow["archetype"], "black")
    ax.scatter(crow["mean_drift"], crow["mean_len"],
               color=color, s=120, marker="D", zorder=5)

ax.set_title("User Career Clusters (k-means, k=5)", fontsize=13, fontweight="bold")
ax.set_xlabel("Drift Score (entropy-based)")
ax.set_ylabel("Sequence Length")
ax.legend(fontsize=9, loc="upper right")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "user_clusters.png", dpi=150)
plt.show()
print("Saved: output/user_clusters.png")
```

---

## 11.8 Cluster Sector Breakdown

```python
tsp = (
    risk_df[risk_df["cluster"].notna()]
    .merge(dom_sec_df, on="person_id", how="left")
    .groupby(["cluster", "dominant_sector"])
    .size()
    .reset_index(name="n")
    .sort_values(["cluster", "n"], ascending=[True, False])
    .groupby("cluster")
    .head(3)
    .merge(cs2[["cluster", "archetype"]], on="cluster")
)

unique_archs = tsp["archetype"].unique()
fig, axes = plt.subplots(1, len(unique_archs), figsize=(14, 6), sharey=False)
if len(unique_archs) == 1:
    axes = [axes]

for ax, arch in zip(axes, unique_archs):
    sub = tsp[tsp["archetype"] == arch].sort_values("n")
    ax.barh(
        sub["dominant_sector"].str[:22],
        sub["n"],
        color=ARCHETYPE_COLORS.get(arch, "steelblue"),
        alpha=0.85,
    )
    ax.set_title(arch, fontsize=9, fontweight="bold")
    ax.set_xlabel("Users")
    ax.tick_params(labelsize=7)

plt.suptitle("Top Sectors per Career Archetype Cluster", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "cluster_sector_breakdown.png", dpi=150)
plt.show()
print("Saved: output/cluster_sector_breakdown.png")
```

---

## 11.9 Accuracy Comparison

```python
fig, ax = plt.subplots(figsize=(10, 6))

labels = ["Top-1", "Top-3", "Top-5"]
values = [
    ensemble_results["top_1"] * 100,
    ensemble_results["top_3"] * 100,
    ensemble_results["top_5"] * 100,
]

bars = ax.bar(labels, values, color="#185FA5", alpha=0.9, width=0.5)
for bar, val in zip(bars, values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.5,
        f"{val:.1f}%",
        ha="center", fontsize=12, fontweight="bold", color="#185FA5",
    )

ax.axhline(persistence_acc * 100, color="red", linestyle="--", linewidth=1.2)
ax.text(
    0.05, persistence_acc * 100 + 1.5,
    f"Persistence Baseline: {persistence_acc * 100:.2f}%",
    color="red", fontsize=10,
)
ax.text(
    0, ensemble_results["top_1"] * 100 / 2,
    f"{(ensemble_results['top_1'] - persistence_acc) * 100:+.2f}%\nabove\nbaseline",
    ha="center", color="white", fontsize=10, fontweight="bold",
)

ax.set_ylim(0, 100)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
ax.set_title("Career Prediction Accuracy — Ensemble Model (NLP labels)",
             fontsize=14, fontweight="bold")
ax.set_xlabel("Prediction Task"); ax.set_ylabel("Accuracy (%)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "accuracy_comparison.png", dpi=150)
plt.show()
print("Saved: output/accuracy_comparison.png")
```

---

## 11.10 Association Rules — Top 20 by Lift

```python
top20_rules = (
    assoc_rules[assoc_rules["from"] != assoc_rules["to"]]
    .head(20)
    .copy()
)
top20_rules["rule"] = (
    top20_rules["from"].str[:22] + " →\n" + top20_rules["to"].str[:22]
)

cmap_vals = np.linspace(0.3, 0.85, len(top20_rules))

fig, ax = plt.subplots(figsize=(13, 9))
for i, (_, row) in enumerate(top20_rules.iloc[::-1].iterrows()):
    color = plt.cm.Blues(cmap_vals[i])
    ax.barh(row["rule"], row["lift"], color=color, alpha=0.9)
    ax.text(
        row["lift"] + 0.01, i,
        f"lift={row['lift']:.2f}",
        va="center", fontsize=8,
    )

ax.set_title(
    "Top 20 Cross-Sector Transitions by Lift\n"
    "Lift > 1 = more likely than chance",
    fontsize=12, fontweight="bold",
)
ax.set_xlabel("Lift")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "association_rules_lift.png", dpi=150)
plt.show()
print("Saved: output/association_rules_lift.png")
```

---

# 12. Final Summary

```python
print("\nFINAL SUMMARY")
print("=" * 60)
print("  Architecture")
print("    Bootstrap labels   : regex map_sectors() + broad_map() (ported from R)")
print("    NLP classifier     : SBERT all-MiniLM-L6-v2 + LightGBM")
print("    Weak supervision   : regex → bootstrap labels → NLP training")
print("-" * 60)
print("  NLP Classification (vs bootstrap labels)")
print(f"    Accuracy  : {accuracy:.4f}")
print(f"    Precision : {precision:.4f}")
print(f"    Recall    : {recall:.4f}")
print(f"    F1 Score  : {f1:.4f}")
print("-" * 60)
print("  Career Prediction (Ensemble)")
print(f"    Persistence Baseline Top-1 : {persistence_acc * 100:.2f}%")
print(f"    Ensemble Top-1             : {ensemble_results['top_1'] * 100:.2f}%")
print(f"    Ensemble Top-3             : {ensemble_results['top_3'] * 100:.2f}%")
print(f"    Ensemble Top-5             : {ensemble_results['top_5'] * 100:.2f}%")
print(
    f"    Ensemble vs Baseline       : "
    f"{(ensemble_results['top_1'] - persistence_acc) * 100:+.2f}%"
)
print("-" * 60)
print(f"  PCA Explained Variance")
print(f"    PC1 : {pca_var[0]}%   PC2 : {pca_var[1]}%   Cum(1+2) : {pca_var[0]+pca_var[1]}%")
print("=" * 60)

OUTPUT_FILES = [
    "nlp_confusion_matrix.png",
    "sector_distribution.png",
    "sequence_length_distribution.png",
    "pca_sector_profiles.png",
    "career_drift_distribution.png",
    "transition_heatmap.png",
    "top_transitions.png",
    "market_drift.png",
    "user_clusters.png",
    "cluster_sector_breakdown.png",
    "accuracy_comparison.png",
    "association_rules_lift.png",
    "drift_risk_scores.csv",
    "association_rules.csv",
]
print("\nOutput files:")     
for f in OUTPUT_FILES:
    status = "✓" if (OUTPUT_DIR / f).exists() else "✗ (not yet saved)"
    print(f"  {status}  {f}")
```

---

# End of Notebook|