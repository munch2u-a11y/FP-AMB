#!/usr/bin/env python3
"""
FP-AMB Multi-Persona Conversation Generator
-----------------------------------------------
Has a human persona and the AI Agent persona converse back and forth, live, via a
local Ollama model, to generate genuinely fresh first-person dialogue -- not
recombined from a fixed fact pool. Produces natural filler, tangents, and (for
human personas) occasional light typos, alongside real established facts and
genuinely fresh, un-scripted content.
"""

import json
import random
import re
import requests
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PERSONAS_DIR = ROOT / "personas"
CORPUS_FILE = ROOT / "data" / "fp_amb_500k_cross_session.jsonl"
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "granite4.1:8b"

HUMAN_PERSONAS = ["sarah", "alex", "mark", "dave", "elena"]


def load_persona(name: str) -> str:
    return (PERSONAS_DIR / f"{name}.md").read_text()


def query(system: str, history: str, speaker_label: str) -> str:
    prompt = (
        f"{system}\n\n"
        f"--- Conversation so far ---\n{history}\n\n"
        f"Write ONLY the next message from {speaker_label}. One message, natural length "
        f"(1-4 sentences typically). Do not include a speaker label or timestamp, just "
        f"the message text itself."
    )
    resp = requests.post(OLLAMA_URL, json={
        "model": MODEL, "prompt": prompt, "stream": False,
        "options": {"temperature": 0.85, "num_predict": 180},
    }, timeout=120)
    text = resp.json().get("response", "").strip().strip('"')
    # strip a self-applied "Speaker:" prefix the model sometimes adds despite instructions
    label_prefix = re.match(rf"^{re.escape(speaker_label)}:\s*", text, re.IGNORECASE)
    if label_prefix:
        text = text[label_prefix.end():]
    return text.strip()


def generate_exchange(human_name: str, session_id: str, day: int, start_ts: str, turns: int = 6) -> list:
    human_persona = load_persona(human_name)
    agent_persona = load_persona("ai_agent")
    human_label = human_name.capitalize()

    history = ""
    results = []
    ts = start_ts
    for i in range(turns):
        if i % 2 == 0:
            msg = query(human_persona, history or "(start of conversation)", human_label)
            speaker = human_label
        else:
            msg = query(agent_persona, history, "the AI Agent")
            speaker = "Assistant"

        history += f"{speaker}: {msg}\n"
        results.append({
            "session_id": session_id, "day": day, "timestamp": ts,
            "speaker": speaker, "text": f"[{ts}] {speaker}: {msg}",
            "is_needle": False,
        })
        ts = _bump_timestamp(ts, minutes=random.randint(2, 6))

    return results


def _bump_timestamp(ts: str, minutes: int) -> str:
    from datetime import datetime, timedelta
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    dt += timedelta(minutes=minutes)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def append_to_corpus(turns: list):
    with open(CORPUS_FILE, "a") as f:
        for t in turns:
            f.write(json.dumps(t) + "\n")


if __name__ == "__main__":
    import argparse
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--day", type=int, required=True)
    parser.add_argument("--start-ts", required=True)
    parser.add_argument("--persona", required=True, choices=HUMAN_PERSONAS)
    parser.add_argument("--turns", type=int, default=6)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    result = generate_exchange(args.persona, args.session_id, args.day, args.start_ts, args.turns)
    for t in result:
        print(t["text"], flush=True)

    if not args.dry_run:
        append_to_corpus(result)
        print(f"\nAppended {len(result)} turns to corpus.", flush=True)
