"""
config.py — Single source of truth for all Role 2 constants.
Imported by both indexer.py and scorer.py to prevent drift.
"""

import os

# =============================================================================
# JD QUERIES — Dual strategy: semantic (natural language) + BM25 (keywords)
# =============================================================================

SEMANTIC_JD_QUERY = """
Senior AI Engineer with 6 to 8 years of applied machine learning experience at product companies.
Built and shipped end-to-end ranking systems, search engines, and recommendation systems to real users at scale.
Production experience with embeddings-based retrieval using sentence-transformers, BGE, or E5 models.
Hands-on with vector databases and hybrid search infrastructure including FAISS, Pinecone, Weaviate, Qdrant, Milvus, or Elasticsearch.
Designed evaluation frameworks for ranking quality using NDCG, MRR, MAP, precision, and A/B testing.
Strong Python engineer who writes production-quality code, not just prototypes.
Experience with LLM integration, fine-tuning with LoRA and PEFT, and retrieval-augmented generation.
Worked at startups or product-focused technology companies, not IT consulting or services firms.
Understands the full lifecycle of building ML systems from offline training to online serving and monitoring.
"""

BM25_JD_QUERY = """
AI engineer machine learning ranking retrieval recommendation search
embeddings sentence-transformers FAISS Pinecone Weaviate Qdrant Milvus Elasticsearch
vector database hybrid search semantic search information retrieval
NDCG MRR MAP precision recall evaluation A/B testing
Python PyTorch TensorFlow scikit-learn production deployment
LLM fine-tuning LoRA QLoRA PEFT RAG retrieval-augmented generation
NLP natural language processing text classification named entity recognition
product company startup SaaS platform scale
data pipeline feature engineering model serving inference optimization
"""

# =============================================================================
# EMBEDDING MODEL
# =============================================================================

EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
EMBEDDING_DIMENSION = 768

# =============================================================================
# SCORING WEIGHTS — Hybrid fusion
# =============================================================================

SEMANTIC_WEIGHT = 0.65
BM25_WEIGHT = 0.35

# =============================================================================
# CONSULTING FIRMS — Career trajectory penalty
# =============================================================================

CONSULTING_FIRMS = {
    'tcs', 'tata consultancy', 'infosys', 'wipro', 'accenture',
    'cognizant', 'capgemini', 'hcl', 'tech mahindra', 'mindtree',
    'mphasis', 'l&t infotech', 'lti', 'ltimindtree', 'hexaware',
    'cyient', 'persistent systems', 'zensar', 'niit technologies',
    'birlasoft', 'coforge', 'sonata software', 'mastek',
}

# =============================================================================
# TITLE RELEVANCE TIERS — Keyword stuffer detection
# =============================================================================

TITLE_TIER_PERFECT = {
    'ai engineer', 'ml engineer', 'machine learning engineer',
    'senior ai engineer', 'senior ml engineer', 'senior machine learning engineer',
    'lead ai engineer', 'lead ml engineer', 'staff ai engineer',
    'data scientist', 'senior data scientist', 'lead data scientist',
    'nlp engineer', 'senior nlp engineer',
    'recommendation systems engineer', 'search engineer',
    'ranking engineer', 'applied scientist', 'research engineer',
    'junior ai engineer', 'junior ml engineer',
    'senior machine learning engineer',
}

TITLE_TIER_STRONG_ADJACENT = {
    'software engineer', 'senior software engineer', 'staff software engineer',
    'backend engineer', 'senior backend engineer',
    'full stack developer', 'senior full stack developer',
    'data engineer', 'senior data engineer',
    'platform engineer', 'infrastructure engineer',
    'devops engineer', 'senior devops engineer',
    '.net developer', 'java developer', 'python developer',
}

TITLE_TIER_WEAK_ADJACENT = {
    'project manager', 'senior project manager',
    'business analyst', 'senior business analyst',
    'product manager', 'senior product manager',
    'qa engineer', 'senior qa engineer', 'test engineer',
    'cloud engineer', 'senior cloud engineer',
    'frontend engineer', 'senior frontend engineer',
    'mobile developer', 'senior mobile developer',
    'technical lead', 'engineering manager',
}

# Anything not in the above tiers is treated as NON_TECH_TRAP with 0.05 multiplier.
# This includes: HR Manager, Marketing Manager, Sales Executive, Accountant,
# Content Writer, Graphic Designer, Customer Support, Civil Engineer,
# Mechanical Engineer, Operations Manager, etc.

TITLE_SCORES = {
    'perfect': 1.0,
    'strong_adjacent': 0.85,
    'weak_adjacent': 0.5,
    'non_tech_trap': 0.05,
}

# =============================================================================
# EXPERIENCE RANGE — JD says 5-9 years
# =============================================================================

EXPERIENCE_SCORES = [
    # (min_years, max_years, multiplier)
    (5, 9, 1.0),      # Sweet spot
    (4, 5, 0.85),     # Close — slightly under
    (9, 12, 0.85),    # Close — slightly over
    (3, 4, 0.6),      # Stretch — junior
    (12, 15, 0.6),    # Stretch — too senior
    (0, 3, 0.3),      # Poor fit — too junior
    (15, 100, 0.3),   # Poor fit — too senior
]

# =============================================================================
# LOCATION SCORING — JD prefers Pune/Noida/India
# =============================================================================

PREFERRED_CITIES_TIER1 = {
    'pune', 'noida', 'delhi', 'new delhi', 'gurgaon', 'gurugram',
    'faridabad', 'ghaziabad', 'greater noida',
}

PREFERRED_CITIES_TIER2 = {
    'hyderabad', 'mumbai', 'bangalore', 'bengaluru', 'chennai',
}

LOCATION_SCORES = {
    'india_tier1': 1.3,
    'india_tier2': 1.15,
    'india_other_relocate': 1.0,
    'india_other_no_relocate': 0.7,
    'outside_india_relocate': 0.5,
    'outside_india_no_relocate': 0.2,
}

# =============================================================================
# HONEYPOT THRESHOLDS
# =============================================================================

HONEYPOT_DATE_MISMATCH_TOLERANCE = 0.5  # 50% discrepancy between date-calculated and stated duration
HONEYPOT_CAREER_OVERLAP_DAYS = 60       # Max acceptable days of overlap between roles
HONEYPOT_EXPERT_COUNT_THRESHOLD = 8     # Flag if 8+ expert skills
HONEYPOT_EXPERT_AVG_DURATION_MIN = 6    # Flag if avg duration of expert skills < 6 months
HONEYPOT_EXPERIENCE_RATIO = 1.5         # Flag if total career months > years_of_experience * 12 * this

# =============================================================================
# ARTIFACT FILE PATHS
# =============================================================================

ARTIFACTS_DIR = "artifacts"
DENSE_INDEX_PATH = os.path.join(ARTIFACTS_DIR, "dense_index.faiss")
BM25_INDEX_PATH = os.path.join(ARTIFACTS_DIR, "bm25_index.pkl")
CANDIDATES_META_PATH = os.path.join(ARTIFACTS_DIR, "candidates_meta.pkl")
JD_VECTOR_PATH = os.path.join(ARTIFACTS_DIR, "jd_vector.npy")
