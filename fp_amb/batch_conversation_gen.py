#!/usr/bin/env python3
"""
FP-AMB Batch Conversation Generator
--------------------------------------
Runs conversation_gen.generate_exchange() across a spread of existing sessions,
rotating personas for variety, appending everything to the corpus as it goes
(so partial progress is never lost if interrupted).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fp_amb.conversation_gen import generate_exchange, append_to_corpus, HUMAN_PERSONAS

CORPUS_FILE = Path(__file__).resolve().parent.parent / "data" / "fp_amb_500k_cross_session.jsonl"


def load_sessions():
    rows = [json.loads(l) for l in open(CORPUS_FILE)]
    sessions = {}
    for r in rows:
        if r.get("type") == "SESSION_DELIMITER":
            sessions[r["session_id"]] = (r["day"], r["timestamp"])
    return sessions


def main(n_sessions: int, turns_per_session: int):
    sessions = load_sessions()
    ordered = sorted(sessions.items(), key=lambda kv: int(kv[0].split("_")[1]))
    # spread evenly across the 60 sessions
    step = max(1, len(ordered) // n_sessions)
    picked = ordered[::step][:n_sessions]

    print(f"Generating {len(picked)} session exchanges, {turns_per_session} turns each...", flush=True)

    for i, (session_id, (day, ts)) in enumerate(picked, 1):
        persona = HUMAN_PERSONAS[i % len(HUMAN_PERSONAS)]
        # offset the start time from the session's canonical start so it doesn't collide
        from fp_amb.conversation_gen import _bump_timestamp
        start_ts = _bump_timestamp(ts, minutes=45)

        print(f"\n[{i}/{len(picked)}] {session_id} (day {day}) -- persona: {persona}", flush=True)
        try:
            result = generate_exchange(persona, session_id, day, start_ts, turns_per_session)
            append_to_corpus(result)
            for t in result:
                print(f"  {t['text'][:130]}", flush=True)
            print(f"  -> appended {len(result)} turns", flush=True)
        except Exception as e:
            print(f"  ERROR on {session_id}: {e}", flush=True)
            continue

    print(f"\nDone. Generated exchanges for {len(picked)} sessions.", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", type=int, default=25)
    parser.add_argument("--turns", type=int, default=8)
    args = parser.parse_args()
    main(args.sessions, args.turns)
