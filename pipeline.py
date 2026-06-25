import gzip
import json
import os

INPUT_FILE_GZ = 'DATASET/candidates.jsonl.gz'
INPUT_FILE_UNCOMPRESSED = 'DATASET/candidates.jsonl'
OUTPUT_FILE = 'DATASET/clean_pool.jsonl'

def process_pipeline():
    # Determine which input file exists
    if os.path.exists(INPUT_FILE_GZ):
        input_path = INPUT_FILE_GZ
        print(f"Reading compressed dataset: {input_path}")
        open_func = lambda p: gzip.open(p, 'rt', encoding='utf-8')
    elif os.path.exists(INPUT_FILE_UNCOMPRESSED):
        input_path = INPUT_FILE_UNCOMPRESSED
        print(f"Reading uncompressed dataset: {input_path}")
        open_func = lambda p: open(p, 'r', encoding='utf-8')
    else:
        print(f"Error: Neither {INPUT_FILE_GZ} nor {INPUT_FILE_UNCOMPRESSED} found.")
        return

    total_processed = 0
    total_dropped = 0
    total_saved = 0

    # Consulting firms explicitly blacklisted for pure-history candidates
    consulting_firms = ['tcs', 'infosys', 'wipro', 'accenture', 'cognizant', 'capgemini']

    with open_func(input_path) as infile, open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
        for line in infile:
            total_processed += 1
            if total_processed % 10000 == 0:
                print(f"Processed {total_processed} rows...")

            line = line.strip()
            if not line:
                continue

            try:
                candidate = json.loads(line)
            except json.JSONDecodeError:
                total_dropped += 1
                continue

            # --- Rule 1: The Honeypot Trap ---
            # Spec: "expert proficiency in 10 skills with 0 years used"
            # We flag if they have >= 5 expert skills with 0 duration
            suspicious_expert_skills = 0
            skills = candidate.get('skills', [])
            if skills:
                for skill in skills:
                    proficiency = skill.get('proficiency', '')
                    if proficiency is None:
                        continue
                    proficiency = str(proficiency).lower()
                    duration = skill.get('duration_months')
                    if proficiency == 'expert' and duration is not None and isinstance(duration, (int, float)) and duration == 0:
                        suspicious_expert_skills += 1
            
            if suspicious_expert_skills >= 5:
                total_dropped += 1
                continue

            # --- Rule 1.5: The Consulting Firm Trap ---
            career_history = candidate.get('career_history', [])
            if career_history:
                all_consulting = True
                for role in career_history:
                    if role is None:
                        continue
                    company_name = str(role.get('company_name', '')).lower()
                    if not any(firm in company_name for firm in consulting_firms):
                        all_consulting = False
                        break
                if all_consulting:
                    total_dropped += 1
                    continue

            # --- Rule 2: The Logistical Filter ---
            redrob_signals = candidate.get('redrob_signals', {})
            if redrob_signals is None:
                redrob_signals = {}
            
            # Notice period check removed: JD says < 30 days is ideal, but submission spec shows 120-day notice period candidate ranked #3.
            # So we pass it through.
            
            willing_to_relocate = redrob_signals.get('willing_to_relocate')
            if willing_to_relocate is False:
                profile = candidate.get('profile', {})
                if profile is None:
                    profile = {}
                location = profile.get('location', '')
                if location is None:
                    location = ''
                location = str(location).lower()
                allowed_locations = ['pune', 'noida', 'hyderabad', 'mumbai', 'delhi','ncr']
                if not any(city in location for city in allowed_locations):
                    total_dropped += 1
                    continue

            # --- Rule 3: Feature Extraction ---
            profile = candidate.get('profile', {})
            if profile is None:
                profile = {}
            
            dense_parts = []
            headline = profile.get('headline')
            if headline and headline is not None:
                dense_parts.append(str(headline).strip())
                
            summary = profile.get('summary')
            if summary and summary is not None:
                dense_parts.append(str(summary).strip())
                
            if career_history:
                for role in career_history:
                    if role is None:
                        continue
                    description = role.get('description')
                    if description and description is not None:
                        dense_parts.append(str(description).strip())
            
            if skills:
                for skill in skills:
                    skill_name = skill.get('name')
                    if skill_name:
                        dense_parts.append(str(skill_name).strip())
                    
            dense_text = " ".join(dense_parts)

            # Output with all behavioral signals for runtime weighting
            output_obj = {
                "candidate_id": candidate.get("candidate_id"),
                "dense_text": dense_text,
                "redrob_signals": redrob_signals
            }
            
            outfile.write(json.dumps(output_obj) + '\n')
            total_saved += 1

    print("\n--- Final Summary ---")
    print(f"Total Processed: {total_processed}")
    print(f"Total Dropped: {total_dropped}")
    print(f"Total Saved: {total_saved}")

if __name__ == '__main__':
    process_pipeline()