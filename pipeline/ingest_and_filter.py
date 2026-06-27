import json
import sys
import time
from datetime import datetime


# HARD DROP RULES (Binary — candidate is permanently removed)

# Non-technical titles that should never appear in an AI Engineering search.
# These are the "keyword stuffer" traps described in the hackathon rules.
NON_TECH_TITLES = {
    "marketing manager", "hr manager", "human resources",
    "sales executive", "sales manager", "sales representative",
    "recruiter", "talent acquisition",
    "customer support", "customer service",
    "accountant", "accounting manager",
    "content writer", "copywriter",
    "graphic designer",
    "civil engineer", "mechanical engineer", "electrical engineer",
    "chemical engineer",
    "operations manager", "office manager", "admin",
    "teacher", "professor", "lecturer",
    "lawyer", "legal counsel",
    "nurse", "doctor", "pharmacist",
    "chef", "cook",
    "driver", "delivery",
    "receptionist", "secretary",
    "interior designer", "fashion designer",
    "photographer", "videographer",
    "real estate", "property manager",
    "financial analyst", "investment banker",
    "insurance agent",
    "journalist", "editor",
    "social media manager",
    "event manager", "event coordinator",
    "supply chain manager", "logistics manager",
    "warehouse manager",
    "retail manager", "store manager",
}


# HONEYPOT DETECTION (Hard Drop)

def is_honeypot(candidate: dict) -> bool:
    """
    Detect candidates with mathematically impossible profiles.
    These are the ~80 honeypot candidates planted in the dataset.
    
    Returns True if any impossibility is detected.
    """
    # ── Check 1: Expert skills with 0 months duration ────────────────────
    expert_zeros = [
        s for s in candidate.get("skills", [])
        if s.get("proficiency", "").lower() in ("expert", "advanced") and s.get("duration_months", 1) == 0
    ]
    if len(expert_zeros) >= 2:
        return True

    # ── Check 2: Mass expert claims with suspiciously low durations ──────
    expert_skills = [
        s for s in candidate.get("skills", [])
        if s.get("proficiency") == "expert"
    ]
    if len(expert_skills) >= 8:
        durations = [s.get("duration_months", 0) for s in expert_skills]
        avg_duration = sum(durations) / len(durations) if durations else 0
        if avg_duration < 6:
            return True

    # ── Check 3: Career duration impossibly exceeds stated experience ────
    history = candidate.get("career_history", [])
    total_career_months = sum(h.get("duration_months", 0) for h in history)
    stated_years = candidate.get("profile", {}).get("years_of_experience", 0)
    stated_months = stated_years * 12
    if stated_months > 0 and total_career_months > stated_months * 1.5:
        return True

    # ── Check 4: Date-math mismatch in individual roles ──────────────────
    for role in history:
        start_str = role.get("start_date")
        end_str = role.get("end_date")
        stated_duration = role.get("duration_months", 0)

        if start_str and end_str and stated_duration > 0:
            try:
                start = datetime.strptime(start_str, "%Y-%m-%d")
                end = datetime.strptime(end_str, "%Y-%m-%d")
                calculated_months = max(0, (end.year - start.year) * 12 + (end.month - start.month))
                if calculated_months > 0:
                    discrepancy = abs(calculated_months - stated_duration) / calculated_months
                    if discrepancy > 0.5:  # 50% mismatch
                        return True
            except (ValueError, TypeError):
                pass

    # ── Check 5: Overlapping career timelines ────────────────────────────
    dated_roles = []
    for role in history:
        start_str = role.get("start_date")
        end_str = role.get("end_date")
        if start_str:
            try:
                start = datetime.strptime(start_str, "%Y-%m-%d")
                end = (
                    datetime.strptime(end_str, "%Y-%m-%d")
                    if end_str
                    else datetime.now()
                )
                dated_roles.append((start, end))
            except (ValueError, TypeError):
                pass

    dated_roles.sort(key=lambda x: x[0])
    for i in range(len(dated_roles) - 1):
        _, end_i = dated_roles[i]
        start_next, _ = dated_roles[i + 1]
        overlap_days = (end_i - start_next).days
        if overlap_days > 60:  # > 60 days overlap = impossible
            return True

    return False


# NON-TECHNICAL TITLE CHECK (Hard Drop)

def is_non_technical_title(candidate: dict) -> bool:
    """
    Check if the candidate's current title is completely irrelevant
    to the AI Engineering role. These are keyword-stuffer traps.
    """
    title = candidate.get("profile", {}).get("current_title", "").lower().strip()

    # Direct match against known non-tech titles
    if title in NON_TECH_TITLES:
        return True

    # Partial match for common non-tech patterns
    non_tech_patterns = [
        "marketing", "hr ", "human resource", "sales",
        "customer support", "customer service",
        "accountant", "accounting",
        "civil engineer", "mechanical engineer",
        "electrical engineer", "chemical engineer",
        "nurse", "doctor", "pharmacist",
        "teacher", "professor", "lecturer",
        "lawyer", "legal",
        "chef", "cook", "driver",
        "receptionist", "secretary",
        "interior design", "fashion design",
        "real estate", "insurance",
        "journalist",
        "warehouse", "retail manager", "store manager",
    ]

    for pattern in non_tech_patterns:
        if pattern in title:
            return True

    return False


# WRONG DOMAIN CHECK (Hard Drop)

def is_wrong_domain(candidate: dict) -> bool:
    """
    If a candidate has 3+ skills in Robotics/Computer Vision/Self-Driving,
    but ZERO skills in NLP/Retrieval/Search, drop them.
    This prevents irrelevant AI researchers from clogging the ranks.
    """
    skills = candidate.get("skills", [])
    if not skills:
        return False
        
    skill_names = {s.get("name", "").lower() for s in skills}
    
    cv_skills = {'computer vision', 'image classification', 'object detection',
                 'image segmentation', 'speech recognition', 'tts', 'text-to-speech',
                 'asr', 'robotics', 'slam', 'point cloud', 'pose estimation', 'lidar'}
                 
    nlp_skills = {'nlp', 'retrieval', 'ranking', 'recommendation', 'search',
                  'embeddings', 'faiss', 'pinecone', 'llm', 'rag', 'language model'}
                  
    cv_hits = sum(1 for w in cv_skills if any(w in x for x in skill_names))
    nlp_hits = sum(1 for w in nlp_skills if any(w in x for x in skill_names))
    
    return (cv_hits >= 3 and nlp_hits == 0)


# MAIN PIPELINE

def clean_candidates(input_path: str, output_path: str):
    """
    Stream candidates.jsonl, apply hard filters, write clean_pool.jsonl.
    O(1) memory — processes one candidate at a time.
    """
    print("=" * 60)
    print("  PHASE 1: DATA INGESTION & HARD TRAP FILTER")
    print("=" * 60)

    start = time.time()

    total = 0
    honeypots_dropped = 0
    title_dropped = 0
    kept = 0

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue

            total += 1
            candidate = json.loads(line)

            # Hard Drop 1: Honeypot detection
            if is_honeypot(candidate):
                honeypots_dropped += 1
                continue

            # Hard Drop 2: Non-technical title
            if is_non_technical_title(candidate):
                title_dropped += 1
                continue

            # Hard Drop 3: Wrong Domain (e.g. CV/Robotics with no NLP)
            if is_wrong_domain(candidate):
                title_dropped += 1  # count it under same category for simplicity
                continue

            # Candidate survives — write to clean pool
            fout.write(json.dumps(candidate) + "\n")
            kept += 1

            # Progress indicator
            if total % 10000 == 0:
                print(f"  Processed {total:,} candidates ...")

    elapsed = time.time() - start

    # ── Print stats ──────────────────────────────────────────────────────
    dropped = honeypots_dropped + title_dropped
    print(f"\n{'=' * 60}")
    print(f"  RESULTS")
    print(f"{'=' * 60}")
    print(f"  Total candidates processed:  {total:,}")
    print(f"  Honeypots dropped:           {honeypots_dropped:,}")
    print(f"  Non-tech titles dropped:     {title_dropped:,}")
    print(f"  Total dropped:               {dropped:,} ({dropped/total*100:.1f}%)")
    print(f"  Clean pool size:             {kept:,} ({kept/total*100:.1f}%)")
    print(f"  Output file:                 {output_path}")
    print(f"  Time elapsed:                {elapsed:.1f}s")
    print(f"{'=' * 60}")


# CLI ENTRY POINT

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python -m src.data.cleaner <input.jsonl> <output.jsonl>")
        print("Example: python -m src.data.cleaner candidates.jsonl clean_pool.jsonl")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    clean_candidates(input_file, output_file)
