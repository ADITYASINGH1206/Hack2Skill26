# Redrob Hackathon: Role 2 ML Pipeline

This repository contains the Machine Learning & Ranking Engine (Role 2) for the Redrob AI "Intelligent Candidate Discovery & Ranking Challenge."

Our engine is designed to strictly adhere to the hackathon's constraints: **16GB RAM limit, CPU-only, under 5 minutes inference, and no external LLM network calls.**

## Setup & Installation

1. Make sure you are using Python 3.10+ (tested on Python 3.13).
2. Install the exact required dependencies (which include CPU-only optimized versions of FAISS and PyTorch):

```bash
pip install -r requirements.txt
```

## How to Run

The pipeline is split into two phases to maximize efficiency during the 5-minute scoring window:

### Phase 1: Offline Pre-computation (Indexer)
This script runs the heavy sentence-transformers embedding model (`BAAI/bge-base-en-v1.5`) and builds our FAISS and BM25 indexes. You only need to run this **once** when the candidate data changes. 

```bash
python -m src.ml.indexer
```
*Note: The first run will download the ~430MB model weights. It will generate 4 lightweight artifacts (`.faiss`, `.pkl`, `.npy`) totaling under 1 MB for the sample data.*

### Phase 2: Online Inference Test (Scorer)
This script loads the pre-computed artifacts and runs our ultra-fast 5-layer heuristic scoring engine (Honeypot detection, Trajectory, Title, Experience, Location).

To run a deep-dive test and see the Top 10 Ranked candidates with a full breakdown of *why* they were scored the way they were:

```bash
python tests/test_pipeline.py
```

## Architecture

- **`src/ml/config.py`**: The single source of truth for all constants, including our "Secret Weapon" dual-JD queries (Semantic natural language vs BM25 keyword specific).
- **`src/ml/indexer.py`**: The offline artifact generator.
- **`src/ml/scorer.py`**: The online ranking algorithm.
- **`tests/test_pipeline.py`**: An end-to-end wrapper to visualize the engine's output.

*(Note: The final `rank.py` script and automated reasoning required for the Stage 3 Hackathon Submission will be integrated by Role 3).*
