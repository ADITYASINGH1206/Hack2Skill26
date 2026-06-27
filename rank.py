"""
rank.py -- Final Ranking Script (Role 3)

Produces the submission CSV from pre-computed artifacts.
This is the script judges will run inside the 5-minute sandbox.

Usage:
    python rank.py --candidates candidates.jsonl --out submission.csv
"""

import os
import sys
import csv
import time
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.ml.scorer import rank_candidates


# =============================================================================
# REASONING ENGINE
# =============================================================================

# JD core requirements for connecting reasoning to the job description
JD_CORE_SKILLS = {
    "faiss", "pinecone", "weaviate", "qdrant", "milvus", "elasticsearch",
    "sentence-transformers", "bge", "e5", "embeddings",
    "ranking", "retrieval", "recommendation", "search",
    "ndcg", "mrr", "map", "a/b testing", "evaluation",
    "lora", "qlora", "peft", "rag", "fine-tuning",
    "pytorch", "tensorflow", "scikit-learn",
    "nlp", "natural language processing",
    "python", "machine learning", "deep learning", "neural network",
}


def generate_reasoning(candidate: dict, rank: int) -> str:
    """
    Generate a specific, honest, varied reasoning string for each candidate.

    Rules we follow (from hackathon Stage 4 manual review):
    - Reference specific facts from the candidate's profile
    - Connect to JD requirements
    - Acknowledge concerns honestly
    - Never hallucinate (only cite skills that exist)
    - Vary the reasoning across candidates
    - Match tone to rank position
    """
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    skills = candidate.get("skills", [])
    career = candidate.get("career_history", []) if "career_history" in candidate else []

    title = profile.get("current_title", "Unknown")
    yoe = profile.get("years_of_experience", 0)
    location = profile.get("location", "Unknown")
    country = profile.get("country", "Unknown")
    company = profile.get("current_company", "Unknown")
    industry = profile.get("current_industry", "")

    # Extract actual skill names from the candidate's profile (no hallucination)
    skill_names = [s.get("name", "").lower() for s in skills]
    skill_names_display = [s.get("name", "") for s in skills]

    # Find JD-relevant skills this candidate actually has
    matched_jd_skills = []
    for s in skill_names_display:
        if s.lower() in JD_CORE_SKILLS:
            matched_jd_skills.append(s)
    # Also check partial matches (e.g., "Machine Learning" contains "machine learning")
    for s in skill_names_display:
        sl = s.lower()
        for jd_skill in JD_CORE_SKILLS:
            if jd_skill in sl or sl in jd_skill:
                if s not in matched_jd_skills:
                    matched_jd_skills.append(s)
                    break

    # Behavioral signals
    response_rate = signals.get("recruiter_response_rate", -1)
    notice_period = signals.get("notice_period_days", -1)
    open_to_work = signals.get("open_to_work_flag", False)
    last_active = signals.get("last_active_date", "")
    interview_rate = signals.get("interview_completion_rate", -1)
    github_score = signals.get("github_activity_score", -1)

    # Career info
    recent_roles = []
    for role in career[:2]:
        r_title = role.get("title", "")
        r_company = role.get("company", "")
        r_duration = role.get("duration_months", 0)
        if r_title and r_company:
            recent_roles.append(f"{r_title} at {r_company} ({r_duration}mo)")

    # ── Build reasoning parts ────────────────────────────────────────────

    parts = []

    # Part 1: Title + Experience + Company (always included)
    if rank <= 10:
        parts.append(f"{title} with {yoe:.1f} years of experience currently at {company}")
    elif rank <= 50:
        parts.append(f"{title} with {yoe:.1f} years based in {location}")
    else:
        parts.append(f"{title} ({yoe:.1f} yrs) in {location} {country}")

    # Part 2: JD-relevant skills (specific facts, no hallucination)
    if matched_jd_skills:
        if rank <= 20:
            skill_str = " | ".join(matched_jd_skills[:4])
            parts.append(f"directly relevant skills include {skill_str}")
        elif rank <= 50:
            skill_str = " | ".join(matched_jd_skills[:3])
            parts.append(f"relevant skills: {skill_str}")
        else:
            skill_str = " | ".join(matched_jd_skills[:2])
            parts.append(f"some relevant skills ({skill_str})")
    else:
        if rank <= 50:
            parts.append("limited direct overlap with core JD skill requirements (ranking, retrieval, embeddings)")
        else:
            parts.append("minimal JD skill alignment")

    # Part 3: Career trajectory (for top candidates)
    if recent_roles and rank <= 30:
        parts.append(f"recent experience: {recent_roles[0]}")

    # Part 4: Strengths (tone matches rank)
    strengths = []
    concerns = []

    # Location
    if country.lower() == "india":
        city_lower = location.lower()
        if any(c in city_lower for c in ["pune", "noida", "delhi", "gurgaon", "gurugram"]):
            strengths.append("based in preferred JD location")
        elif any(c in city_lower for c in ["bangalore", "bengaluru", "hyderabad", "mumbai", "chennai"]):
            strengths.append(f"India-based ({location})")
    else:
        concerns.append(f"located outside India ({country})")

    # Experience fit
    if 5 <= yoe <= 9:
        strengths.append("experience within JD's 5-9 year sweet spot")
    elif yoe < 5:
        concerns.append(f"under the JD's 5-year minimum ({yoe:.1f} yrs)")
    elif yoe > 12:
        concerns.append(f"significantly over JD's 9-year upper range ({yoe:.1f} yrs)")
    elif yoe > 9:
        concerns.append(f"slightly over JD's preferred range ({yoe:.1f} yrs)")

    # Behavioral signals
    if response_rate >= 0:
        if response_rate >= 0.6:
            strengths.append(f"strong recruiter engagement ({response_rate:.0%} response rate)")
        elif response_rate < 0.15:
            concerns.append(f"very low recruiter response rate ({response_rate:.0%})")
        elif response_rate < 0.3:
            concerns.append(f"below-average recruiter response rate ({response_rate:.0%})")

    if notice_period >= 0:
        if notice_period <= 30:
            strengths.append(f"short notice period ({notice_period}d)")
        elif notice_period >= 90:
            concerns.append(f"long notice period ({notice_period} days)")

    if open_to_work:
        strengths.append("actively open to new opportunities")

    if github_score >= 0:
        if github_score >= 60:
            strengths.append(f"strong GitHub presence (score: {github_score})")
        elif github_score <= 10 and github_score >= 0:
            concerns.append("minimal GitHub activity")

    if interview_rate >= 0 and interview_rate < 0.5:
        concerns.append(f"low interview completion rate ({interview_rate:.0%})")

    # Part 5: Honeypot flag
    if candidate.get("is_honeypot", False):
        concerns.append("profile contains timeline inconsistencies suggesting data integrity issues")

    # ── Assemble final reasoning ─────────────────────────────────────────

    # Add strengths
    if strengths and rank <= 60:
        top_strengths = strengths[:2] if rank <= 30 else strengths[:1]
        parts.append("; ".join(top_strengths))

    # Add concerns (honest acknowledgment)
    if concerns:
        if rank <= 10:
            # Top candidates: minor concern framing
            if len(concerns) == 1:
                parts.append(f"minor consideration: {concerns[0]}")
        elif rank <= 50:
            # Mid-range: balanced concern
            parts.append(f"however, {concerns[0]}")
        else:
            # Bottom half: concerns are the main story
            concern_str = "; ".join(concerns[:2])
            parts.append(f"concerns: {concern_str}")

    reasoning = "; ".join(parts) + "."

    # Capitalize first letter
    reasoning = reasoning[0].upper() + reasoning[1:]

    return reasoning


# =============================================================================
# SUBMISSION CSV WRITER
# =============================================================================

def write_submission_csv(scored_candidates: list, output_path: str):
    """
    Write the final submission CSV with exactly 100 rows.
    Enforces all hackathon format rules.
    """
    top_100 = scored_candidates[:100]

    # Validate we have exactly 100
    if len(top_100) < 100:
        print(f"[WARN] Only {len(top_100)} candidates available (need 100)")

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])

        prev_score = float("inf")
        for rank, candidate in enumerate(top_100, 1):
            cid = candidate["candidate_id"]
            score = round(candidate["final_score"], 4)

            # Enforce non-increasing scores
            if score > prev_score:
                score = prev_score
            prev_score = score

            # Remove any accidental commas to prevent CSV quotes
            reasoning = generate_reasoning(candidate, rank).replace(",", ";")

            # Format score to exactly 4 decimal places
            score_str = f"{score:.4f}"

            writer.writerow([cid, rank, score_str, reasoning])

    print(f"\n[OK] Submission written to: {output_path}")
    print(f"     Rows: {len(top_100)} + 1 header = {len(top_100) + 1} total")


# =============================================================================
# VALIDATION
# =============================================================================

def validate_submission(output_path: str):
    """Run local validation checks matching the hackathon auto-validator."""
    print("\n-- Submission Validation --")
    errors = []

    with open(output_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Check 1: Exactly 100 rows
    if len(rows) != 100:
        errors.append(f"Expected 100 rows, got {len(rows)}")

    # Check 2: Required columns
    expected_cols = {"candidate_id", "rank", "score", "reasoning"}
    if rows:
        actual_cols = set(rows[0].keys())
        missing = expected_cols - actual_cols
        if missing:
            errors.append(f"Missing columns: {missing}")

    # Check 3: Ranks 1-100, each exactly once
    ranks = [int(r["rank"]) for r in rows]
    if sorted(ranks) != list(range(1, 101)):
        errors.append("Ranks must be exactly 1-100, each used once")

    # Check 4: Unique candidate_ids
    cids = [r["candidate_id"] for r in rows]
    if len(set(cids)) != len(cids):
        errors.append("Duplicate candidate_ids found")

    # Check 5: candidate_id format
    import re
    for cid in cids:
        if not re.match(r"^CAND_\d{7}$", cid):
            errors.append(f"Invalid candidate_id format: {cid}")
            break

    # Check 6: Non-increasing scores
    scores = [float(r["score"]) for r in rows]
    for i in range(1, len(scores)):
        if scores[i] > scores[i - 1]:
            errors.append(f"Score increases at rank {i+1}: {scores[i]} > {scores[i-1]}")
            break

    # Check 7: No empty reasoning
    empty_count = sum(1 for r in rows if not r.get("reasoning", "").strip())
    if empty_count > 0:
        errors.append(f"{empty_count} rows have empty reasoning")

    # Check 8: Reasoning variation (no identical strings)
    reasonings = [r.get("reasoning", "") for r in rows]
    unique_reasonings = len(set(reasonings))
    if unique_reasonings < 90:  # Allow a tiny bit of collision
        errors.append(f"Only {unique_reasonings}/100 unique reasonings (too templated)")

    if errors:
        print("  [FAIL] Validation errors:")
        for e in errors:
            print(f"    - {e}")
    else:
        print("  [PASS] All 8 checks passed!")
        print(f"  Score range: {scores[0]:.4f} (rank 1) to {scores[-1]:.4f} (rank 100)")
        print(f"  Unique reasonings: {unique_reasonings}/100")

        # Sample 3 reasonings to eyeball
        print("\n  -- Sample Reasonings --")
        for sample_rank in [1, 50, 100]:
            r = rows[sample_rank - 1]
            print(f"  Rank {sample_rank}: {r['reasoning'][:120]}...")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Redrob Hackathon - Produce submission CSV")
    parser.add_argument("--candidates", default="candidates.jsonl",
                        help="Path to candidates.jsonl (not used if artifacts exist)")
    parser.add_argument("--out", default="submission.csv",
                        help="Output CSV filename")
    parser.add_argument("--top-k", type=int, default=100,
                        help="Number of candidates to rank (default: 100)")
    args = parser.parse_args()

    print("=" * 60)
    print("  REDROB HACKATHON - SUBMISSION GENERATOR")
    print("=" * 60)

    start = time.time()

    # Step 1: Run the scorer to get ranked candidates
    print("\n[Step 1] Running scoring engine ...")
    scored = rank_candidates(top_k=args.top_k)

    # Step 2: Generate reasoning and write CSV
    print("\n[Step 2] Generating reasonings and writing CSV ...")
    write_submission_csv(scored, args.out)

    # Step 3: Validate
    print("\n[Step 3] Validating submission ...")
    validate_submission(args.out)

    elapsed = time.time() - start
    print(f"\n{'=' * 60}")
    print(f"  TOTAL TIME: {elapsed:.1f} seconds")
    if elapsed > 300:
        print("  [WARN] Exceeded 5-minute limit!")
    else:
        print(f"  [OK] Within 5-minute limit ({elapsed/300*100:.1f}% used)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
