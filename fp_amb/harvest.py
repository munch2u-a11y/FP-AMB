#!/usr/bin/env python3
"""
FP-AMB Live Output-Harvesting Pipeline
-----------------------------------------
The missing self-referential loop: send a prescripted prompt to a LIVE model,
capture what it ACTUALLY says (no scripting, no canned text), then:
  1. ingest that real captured response into the corpus as a new Assistant turn
  2. save it into dynamic_answer_keys.json in the existing schema
  3. rebuild the question set + master key so recall of it can be tested later

Run standalone: python3 -m fp_amb.harvest
"""

import json
import time
import requests
from pathlib import Path
from dataclasses import dataclass

import os

ROOT = Path(__file__).resolve().parent.parent
OLLAMA_URL = os.getenv("FP_AMB_OLLAMA_URL", "http://localhost:11434/api/generate")
DEFAULT_MODEL = os.getenv("FP_AMB_MODEL", "llama3")

CORPUS_FILE = ROOT / "data" / "fp_amb_500k_cross_session.jsonl"
DYNAMIC_KEYS_FILE = ROOT / "data" / "dynamic_answer_keys.json"
QUESTIONS_FILE = ROOT / "data" / "fp_amb_cross_session_questions.json"


@dataclass
class Probe:
    probe_id: str
    captured_prompt: str      # sent live to the model -- its response is NOT scripted
    evaluation_question: str  # asked later to test recall of the model's own answer
    session_id: str
    day: int
    timestamp: str
    speaker: str = "Assistant"


PROBES = [
    Probe(
        probe_id="priya_job_offer_advice",
        captured_prompt=(
            "Hey, quick question -- my coworker Priya is deciding between two job offers: "
            "a stable role at a large enterprise company, or an early-stage startup with equity "
            "but real risk. What factors should she weigh in making that call? Keep it to 2-3 "
            "concrete points."
        ),
        evaluation_question="What factors did you tell me Priya should weigh when choosing between the enterprise job and the startup offer?",
        session_id="Session_36", day=71, timestamp="2026-11-14T09:20:00Z",
    ),
    Probe(
        probe_id="ray_todo_app_stack_advice",
        captured_prompt=(
            "My neighbor's kid Ray just started learning to code and wants to build a simple "
            "to-do list app as his first project. What's a good, simple tech stack for a total "
            "beginner? Keep it to 2-3 concrete recommendations."
        ),
        evaluation_question="What tech stack did you recommend for Ray's first to-do list app?",
        session_id="Session_38", day=75, timestamp="2026-11-18T15:20:00Z",
    ),
]


def _query_llm(prompt: str, model: str, ollama_url: str) -> str:
    payload = {"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0.4}}
    resp = requests.post(ollama_url, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


def harvest(probes: list, model: str = DEFAULT_MODEL, ollama_url: str = OLLAMA_URL) -> list:
    """Runs each probe against a LIVE model. Returns list of (probe, captured_text)."""
    results = []
    for p in probes:
        print(f"[harvest] sending probe '{p.probe_id}' to {model}...", flush=True)
        t0 = time.time()
        captured = _query_llm(p.captured_prompt, model, ollama_url)
        print(f"[harvest] captured {len(captured)} chars in {time.time()-t0:.1f}s", flush=True)
        print(f"[harvest] response: {captured}\n", flush=True)
        results.append((p, captured))
    return results


def persist(results: list, model: str) -> None:
    """Writes captured outputs into the corpus, dynamic_answer_keys.json, and the question set."""
    # 1. Ingest as real corpus turns
    new_turns = []
    for p, captured in results:
        text = f"[{p.timestamp}] {p.speaker}: {captured}"
        new_turns.append({
            "session_id": p.session_id, "day": p.day, "timestamp": p.timestamp,
            "speaker": p.speaker, "text": text, "is_needle": True,
        })
    with open(CORPUS_FILE, "a") as f:
        for t in new_turns:
            f.write(json.dumps(t) + "\n")
    print(f"[persist] appended {len(new_turns)} live-captured turns to corpus.", flush=True)

    # 2. Save into dynamic_answer_keys.json
    dyn = json.load(open(DYNAMIC_KEYS_FILE)) if DYNAMIC_KEYS_FILE.exists() else {}
    dyn.setdefault("harvested_output_advice_keys", {})
    for p, captured in results:
        dyn["harvested_output_advice_keys"][p.probe_id] = {
            "probe_id": f"probe_{p.probe_id}",
            "captured_prompt": p.captured_prompt,
            "captured_advice": captured,
            "evaluation_question": p.evaluation_question,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": model,
        }
    with open(DYNAMIC_KEYS_FILE, "w") as f:
        json.dump(dyn, f, indent=2)
    print(f"[persist] saved {len(results)} entries to {DYNAMIC_KEYS_FILE}.", flush=True)

    # 3. Add corresponding questions to the source question set
    questions = json.load(open(QUESTIONS_FILE))
    existing_ids = {q["id"] for q in questions}
    next_num = 1
    while f"CAT5_PM_H{next_num:03d}" in existing_ids:
        next_num += 1
    for p, captured in results:
        qid = f"CAT5_PM_H{next_num:03d}"
        # keyword accepted-answers extracted from the real captured text (first few significant words)
        words = [w.strip(".,()") for w in captured.split() if len(w.strip(".,()")) > 4]
        keywords = list(dict.fromkeys(words))[:5]
        questions.append({
            "id": qid,
            "category": "Self-Referential & Procedural Tool Memory",
            "question": p.evaluation_question,
            "expected_answer": captured,
            "accepted_answers": [captured] + keywords,
            "description": f"Live-harvested self-referential recall probe ({p.probe_id})",
        })
        next_num += 1
    with open(QUESTIONS_FILE, "w") as f:
        json.dump(questions, f, indent=2)
    print(f"[persist] added {len(results)} new questions to {QUESTIONS_FILE}.", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="FP-AMB live output-harvesting")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ollama-url", default=OLLAMA_URL)
    args = parser.parse_args()

    results = harvest(PROBES, args.model, args.ollama_url)
    persist(results, args.model)
    print("\nDone. Re-run the master-key compile step to pick these up in evaluation.")
