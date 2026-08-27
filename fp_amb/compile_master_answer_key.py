#!/usr/bin/env python3
"""
Master Answer Key Compiler for FP-AMB
-------------------------------------
Rebuilds data/master_ground_truth_answer_key.json from the current
data/fp_amb_cross_session_questions.json, merging in any dynamic answer keys
from data/dynamic_answer_keys.json (harvested advice, fact corrections, tool
learning). This reads the current question set as the source of truth and
never overwrites it -- it is the only safe way to rebuild the master key.

Run standalone: python3 -m fp_amb.compile_master_answer_key
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS_FILE = ROOT / "data" / "fp_amb_cross_session_questions.json"
DYNAMIC_KEYS_FILE = ROOT / "data" / "dynamic_answer_keys.json"
MASTER_KEY_OUTPUT = ROOT / "data" / "master_ground_truth_answer_key.json"


def compile_master_key():
    questions = json.load(open(QUESTIONS_FILE))

    dynamic_keys = {}
    if DYNAMIC_KEYS_FILE.exists():
        raw_dyn = json.load(open(DYNAMIC_KEYS_FILE))
        for section, extractor in [
            ("harvested_output_advice_keys", "captured_advice"),
            ("fact_correction_keys", "correct_answer"),
            ("tool_learning_keys", "correct_answer"),
        ]:
            for v in raw_dyn.get(section, {}).values():
                q_text = v.get("evaluation_question", "").strip().lower()
                if q_text and extractor in v:
                    dynamic_keys[q_text] = v[extractor]

    master_key = {}
    dynamic_count = 0
    for item in questions:
        q_id = item["id"]
        q = item["question"]
        expected = item["expected_answer"]
        accepted_list = item.get("accepted_answers", [expected])
        q_lower = q.strip().lower()

        is_dyn = q_lower in dynamic_keys
        if is_dyn:
            expected = dynamic_keys[q_lower]
            dynamic_count += 1
            if expected not in accepted_list:
                accepted_list = [expected] + accepted_list

        master_key[q_id] = {
            "id": q_id,
            "category": item["category"],
            "question": q,
            "ground_truth_answer": expected,
            "accepted_answers": accepted_list,
            "is_dynamic_key": is_dyn,
            "description": item.get("description", ""),
            "grading_mode": item.get("grading_mode", "exact"),
            "list_items": item.get("list_items", []),
            "distractor_type": item.get("distractor_type", ""),
        }

    with open(MASTER_KEY_OUTPUT, "w") as f:
        json.dump(master_key, f, indent=2)

    print(f"Compiled {len(master_key)} items into master key ({dynamic_count} dynamic-key merges).")
    return master_key


if __name__ == "__main__":
    compile_master_key()
