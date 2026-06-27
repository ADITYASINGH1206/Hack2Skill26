# Intelligent Candidate Discovery & Ranking Engine - Redrob AI Challenge

This repository contains a production-grade, resource-constrained candidate retrieval and ranking pipeline engineered for the Redrob AI Challenge. The system is designed to parse a raw dataset of 100,000 candidate profiles and identify the top 100 individuals best suited for a highly specific technical Job Description (JD). 

To achieve maximum precision while strictly adhering to hardware and execution time constraints, the pipeline implements a hybrid search architecture. It combines dense semantic embeddings (FAISS) with exact lexical token matching (BM25) and is fortified with rigorous data engineering filters to eliminate synthetic anomalies, honeypot records, and unqualified profiles.

---

## 1. Reproduction Instructions

The evaluation environment for this challenge consists of an offline, resource-restricted container. The reproduction process is split into two phases based on the evaluation round:

### Evaluation: Runtime Inference
During the initial sandbox evaluation, the system executes the 5-minute ranking step using the pre-computed artifact matrices included in this repository. 

**Execute Inference Engine**
Run the ranking script from the root directory. This script strictly utilizes pre-computed local matrices to comply with the timeout constraint:
```bash
pip install -r requirements.txt
python rank.py --candidates clean_pool.jsonl --out submission_final.csv
```

Note: Full team details, AI disclosures, and sandbox environment configurations are documented in the accompanying `submission_metadata.yaml` file.

## 2. End-to-End System Architecture

To meet the strict evaluation timeout constraints without sacrificing retrieval accuracy, the architecture is decoupled into two primary phases: an offline pre-computation pipeline and an ultra-lightweight runtime inference engine.

### Phase 1: Preprocessing, Data Engineering, and Indexing (pipeline/)
The foundational data engineering layer was executed offline on the raw 100,000 JSONL candidate pool. This module is responsible for cleaning the data, applying heuristics, and generating the optimized index structures.
- **Text Normalization & Tokenization:** Candidate fields (experience, skills, current role) are concatenated into a unified text representation, normalized for casing, and tokenized to handle edge-case formatting.
- **Dense Vector Generation:** We utilize sentence-transformers to map candidate profiles into a high-dimensional continuous vector space, capturing deep semantic relationships between skills (e.g., understanding that "PyTorch" and "Deep Learning" are contextually linked).
- **Sparse Vector Generation:** We implement a BM25 lexical model to ensure exact keyword matches are preserved, preventing the loss of highly specific technical acronyms required by the JD.
- **Artifact Export:** The surviving, valid profiles are exported into optimized `faiss_index_v2.bin` and `bm25_model_v2.pkl` local structures to be consumed by the runtime ranker.

### Phase 2: Runtime Inference Engine (rank.py)
This is the primary script executed by the grading container. Because the pipeline's output is saved locally as pre-computed matrices, this module skips the computationally expensive embedding generation phase entirely.
- **Artifact Ingestion:** Instantly loads the FAISS binary index and BM25 pickle model into memory.
- **Query Processing:** The target Job Description is embedded and tokenized locally.
- **Similarity Matrix Evaluation:** The engine queries both the dense and sparse indices simultaneously, retrieving mathematical distance metrics for the candidate pool.
- **Context-Anchored Reasoning:** Dynamically compiles concise, 1-2 sentence reasoning strings directly anchored to specific JSON profile attributes. This deterministic text generation eliminates the risk of LLM hallucination penalties.

## 3. Scoring Algorithm & Tie-Breaking Mechanism

Our final ranking score is a composite metric designed to maximize the NDCG@10 evaluation criteria. The raw embedding distance metrics and lexical scores are normalized into a bounded scale (0.0 to 1.0).

The final composite score is calculated using a weighted hybrid approach:
`S_final = (alpha * S_dense) + (beta * S_sparse) + H_bonus - P_penalty`
*(Where `alpha` and `beta` are empirically tuned weights, `H_bonus` represents heuristic bonuses for exact location matches, and `P_penalty` represents penalties for logistical mismatches).*

**Deterministic Sorting:**
To prevent evaluation plateau loops and guarantee a strictly non-increasing monotonic series, the system resolves exact numerical score ties deterministically using an alphanumeric fallback sort on the `candidate_id` field.

## 4. Data Engineering Guardrails & Traps

The dataset contains engineered anomalies designed to test system robustness. Our Phase 1 pipeline implements deep architectural rules to identify and discard these records:
- **Honeypot and Anomaly Isolation:** Identifies structural contradictions. For example, the system flags and drops candidate profiles claiming "Expert" proficiency in advanced deep learning frameworks while concurrently listing 0 months of total professional experience.
- **Title Trap Safeguards:** Utilizes tokenized regular expression boundaries to filter out non-technical corporate personas (e.g., Marketing Managers, Recruiters, and HR Generalists) attempting to manipulate the ranker via keyword stuffing.
- **Logistical Pruning:** Enforces strict operational cutoffs extracted directly from the JD. The system penalizes notice periods exceeding 90 days and rigorously drops candidates who fall entirely outside the target 5-9 year experience range.

## 5. Repository Structure

The project has been fully flattened into a root execution tier to eliminate nested module path conflicts during automated orchestration.
- `rank.py`: The core CPU inference execution script.
- `requirements.txt`: Lightweight manifest specifying explicit runtime dependencies (faiss-cpu, pandas, rank_bm25, sentence-transformers, PyYAML).
- `submission_metadata.yaml`: System and team metadata declaration file.
- `submission_final.csv`: The validated output file containing the top 100 sequential candidate mappings.
- `faiss_index_v2.bin` & `bm25_model_v2.pkl`: Pre-computed search artifact matrices.
- `pipeline/`: Isolated historical development folder containing offline data engineering logic (`ingest_and_filter.py`) and programmatic structural validation checks.
