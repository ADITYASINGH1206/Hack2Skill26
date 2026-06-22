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
            # If ANY skill has "proficiency": "expert" AND "duration_months" strictly less than 3, drop.
            dropped_rule1 = False
            skills = candidate.get('skills', [])
            if skills:
                for skill in skills:
                    proficiency = skill.get('proficiency', '')
                    if proficiency is None:
                        continue
                    proficiency = str(proficiency).lower()
                    duration = skill.get('duration_months')
                    if proficiency == 'expert' and duration is not None and isinstance(duration, (int, float)) and duration < 3:
                        dropped_rule1 = True
                        break
            
            if dropped_rule1:
                total_dropped += 1
                continue

            # --- Rule 2: The Logistical Filter ---
            # redrob_signals.notice_period_days > 90 -> drop
            # redrob_signals.willing_to_relocate == False -> check profile.location -> if not contains Pune or Noida -> drop
            redrob_signals = candidate.get('redrob_signals', {})
            if redrob_signals is None:
                redrob_signals = {}
            
            notice_period = redrob_signals.get('notice_period_days')
            if notice_period is not None and isinstance(notice_period, (int, float)) and notice_period > 90:
                total_dropped += 1
                continue
            
            willing_to_relocate = redrob_signals.get('willing_to_relocate')
            if willing_to_relocate is False:
                profile = candidate.get('profile', {})
                if profile is None:
                    profile = {}
                location = profile.get('location', '')
                if location is None:
                    location = ''
                location = str(location).lower()
                if 'pune' not in location and 'noida' not in location:
                    total_dropped += 1
                    continue

            # --- Rule 3: Feature Extraction ---
            # Extract and concatenate into dense_text:
            # profile.headline
            # profile.summary
            # description from first two items in career_history
            profile = candidate.get('profile', {})
            if profile is None:
                profile = {}
            
            dense_parts = []
            
            headline = profile.get('headline')
            if headline:
                dense_parts.append(str(headline).strip())
                
            summary = profile.get('summary')
            if summary:
                dense_parts.append(str(summary).strip())
                
            career_history = candidate.get('career_history', [])
            if career_history:
                for role in career_history[:2]:
                    if role is None:
                        continue
                    description = role.get('description')
                    if description:
                        dense_parts.append(str(description).strip())
                    
            dense_text = " ".join(dense_parts)

            # Output
            output_obj = {
                "candidate_id": candidate.get("candidate_id"),
                "dense_text": dense_text,
                "redrob_signals": {
                    "recruiter_response_rate": redrob_signals.get("recruiter_response_rate")
                }
            }
            
            outfile.write(json.dumps(output_obj) + '\n')
            total_saved += 1

    print("\n--- Final Summary ---")
    print(f"Total Processed: {total_processed}")
    print(f"Total Dropped: {total_dropped}")
    print(f"Total Saved: {total_saved}")

if __name__ == '__main__':
    process_pipeline()
