#!/usr/bin/env python3
"""
FP-AMB Dataset & Master Ground-Truth Loader
--------------------------------------------
Loads the ~512k token corpus and 281-item Master Ground-Truth Answer Key.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_CORPUS_FILE = ROOT / "data" / "fp_amb_500k_cross_session.jsonl"
MASTER_KEY_FILE = ROOT / "data" / "master_ground_truth_answer_key.json"

def load_corpus():
    """Loads the 677 conversation turns from the ~512k token corpus."""
    turns = []
    with open(RAW_CORPUS_FILE, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                entry = json.loads(line)
                if entry.get("type") != "SESSION_DELIMITER":
                    turns.append(entry)
    return turns

def load_master_answer_key():
    """Loads the 281 items across 9 categories from the Master Ground-Truth Answer Key."""
    with open(MASTER_KEY_FILE, 'r') as f:
        return json.load(f)
