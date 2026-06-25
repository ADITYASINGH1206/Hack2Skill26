import os
import sys

# Add project root to sys.path so we can import src
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.ml import scorer
from src.ml.config import DENSE_INDEX_PATH

def main():
    print("==================================================")
    print("  RUNNING ROLE 2 ML PIPELINE END-TO-END TEST")
    print("==================================================\n")
    
    # 1. Check if artifacts exist
    if not os.path.exists(DENSE_INDEX_PATH):
        print("[ERROR] Artifacts not found!")
        print(f"  Expected: {DENSE_INDEX_PATH}")
        print("  Please run the Colab notebook first and place the 4 artifact files in the artifacts/ folder.")
        return
    
    print("[OK] Artifacts found in artifacts/ folder.\n")
        
    # 2. Run the scorer
    print("Running Online Scorer...\n")
    top_candidates = scorer.rank_candidates(top_k=10)
    
    print("\n==================================================")
    print("  TOP CANDIDATES PROFILES (Deep Dive)")
    print("==================================================")
    
    for i, c in enumerate(top_candidates, 1):
        profile = c.get('profile', {})
        signals = c.get('redrob_signals', {})
        skills = c.get('skills', [])
        
        print(f"\n{i}. [Rank {i}] ID: {c['candidate_id']} | Final Score: {c['final_score']:.4f}")
        print("-" * 50)
        print(f"   Title:    {profile.get('current_title', 'N/A')}")
        print(f"   Exp:      {profile.get('years_of_experience', 'N/A')} years")
        print(f"   Location: {profile.get('location', 'N/A')}, {profile.get('country', 'N/A')}")
        print(f"   Relocate: {signals.get('willing_to_relocate', False)}")
        
        # Format skills nicely
        skill_str = ", ".join([s['name'] for s in skills[:5]])
        if len(skills) > 5:
            skill_str += f" (+{len(skills)-5} more)"
        print(f"   Skills:   {skill_str}")
        
        print("\n   [Scoring Breakdown]")
        print(f"   Semantic Match: {c['semantic_score']:.3f}")
        print(f"   Keyword Match:  {c['bm25_score']:.3f}")
        print(f"   Title Mult:     {c['title_score']}x")
        print(f"   Traj Mult:      {c['trajectory_score']}x")
        print(f"   Exp Mult:       {c['experience_score']}x")
        print(f"   Loc Mult:       {c['location_score']}x")
        print(f"   Honeypot:       {'YES!' if c['is_honeypot'] else 'No'}")

if __name__ == "__main__":
    main()
