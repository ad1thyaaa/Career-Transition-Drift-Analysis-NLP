# Career Drift Trajectory Analysis — NLP Production Pipeline

Production-grade NLP pipeline for career trajectory and drift analysis using:

* Weak Supervision (Regex Bootstrap Labels)
* SBERT Sentence Embeddings
* LightGBM Multiclass Classification
* Transition Modeling
* Drift Scoring
* PCA Analysis
* Archetype Clustering
* Ensemble Prediction System

The pipeline preserves equivalence with the original R research architecture while adding:

* GPU acceleration
* resumable checkpoints
* dataset fingerprinting
* cache validation
* deterministic reproducibility
* environment diagnostics
* production logging
* publication-ready results

---

# Project Structure

```text
project/
│
├── data/
│   ├── train.parquet
│   ├── validation.parquet
│   └── test.parquet
│
├── results/
│   ├── figures/
│   ├── tables/
│   ├── metrics/
│   ├── checkpoints/
│   ├── logs/
│   └── summaries/
│
├── cache/
├── models/
│
├── career_drift_pipeline.py
├── run_pipeline.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

# Requirements

## Recommended Python Version

Python 3.11.9

Python 3.14 is NOT recommended because several ML libraries still have CUDA compatibility issues.

---

# GPU Setup (CUDA)

This project supports GPU acceleration using NVIDIA CUDA through PyTorch.

## Install CUDA-enabled PyTorch

Inside the activated virtual environment:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

---

# Create Virtual Environment

## Windows PowerShell

```powershell
py -3.11 -m venv .venv
```

Activate:

```powershell
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate again.

---

# Install Dependencies

```bash
pip install -r requirements.txt
```

Requirements include:

* pandas
* numpy
* scikit-learn
* lightgbm
* sentence-transformers
* matplotlib
* seaborn
* pyarrow
* joblib

---

# Input Data

Place parquet datasets inside:

```text
data/
```

Required files:

```text
train.parquet
validation.parquet
test.parquet
```

---

# Run Pipeline

Default run:

```bash
python run_pipeline.py
```

Force full recomputation:

```bash
python run_pipeline.py --no-cache
```

Skip visualisations:

```bash
python run_pipeline.py --no-plots
```

---

# GPU Verification

The pipeline automatically detects CUDA.

Expected startup logs:

```text
Using device: cuda
GPU: NVIDIA GeForce GTX 950M
CUDA available: True
```

If CUDA is unavailable:

```text
CUDA unavailable — using CPU
```

---

# Output Structure

All results are automatically saved.

## Figures

```text
results/figures/
```

Contains:

* transition heatmaps
* drift distributions
* PCA plots
* clustering visualisations
* sector distributions

## Tables

```text
results/tables/
```

Contains:

* transition tables
* sector distributions
* confusion matrices
* clustering summaries

## Metrics

```text
results/metrics/
```

Contains:

* NLP metrics
* classification reports
* run configuration
* reproducibility configs

## Checkpoints

```text
results/checkpoints/
```

Contains:

* cleaned_data.parquet
* regex_bootstrap.parquet
* nlp_predicted.parquet
* transitions.parquet

## Logs

```text
results/logs/
```

Contains:

* pipeline.log
* environment_info.json

## Summaries

```text
results/summaries/
```

Contains:

* final_pipeline_summary.txt

---

# Cache & Resume System

The pipeline supports intelligent resume functionality.

If cached files already exist:

* embeddings are reused
* model checkpoints are reused
* intermediate parquet checkpoints are reused

Dataset fingerprinting automatically invalidates stale caches if the dataset changes.

---

# Research Architecture

IMPORTANT:

The following methodology remains unchanged from the original research pipeline:

* Regex weak supervision
* SBERT embeddings
* LightGBM classifier
* confidence thresholding
* transition modeling
* drift scoring
* PCA
* clustering
* ensemble prediction system

Engineering improvements were added WITHOUT altering research semantics.

---

# Team Setup

After cloning:

1. Install Python 3.11.9
2. Create virtual environment
3. Install CUDA PyTorch
4. Install requirements
5. Add parquet files into data/
6. Run:

```bash
python run_pipeline.py
```

---

# Notes

* Large embedding caches are intentionally ignored via `.gitignore`
* `.venv/` should NEVER be pushed to GitHub
* results can optionally be committed for reproducibility
* GPU acceleration mainly speeds up SBERT embedding generation

---

# Pipeline Version

Production Pipeline v4
