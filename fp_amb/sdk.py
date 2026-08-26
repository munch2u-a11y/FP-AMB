#!/usr/bin/env python3
"""
FP-AMB Framework SDK & Plug-and-Play Evaluator
----------------------------------------------
Allows ANY external memory system (Mem0, MemPalace, mRAG, Zep, LangChain, LlamaIndex, Custom)
to evaluate against the ~512k token FP-AMB Cross-Session Evaluation Battery in 5 lines of code!
"""

import json
import time
import requests
from abc import ABC, abstractmethod
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_CORPUS_FILE = ROOT / "data" / "fp_amb_500k_cross_session.jsonl"
QUESTIONS_FILE = ROOT / "data" / "fp_amb_cross_session_questions.json"

class BaseMemoryProvider(ABC):
    """Abstract Base Class for Plug-and-Play Memory Systems in FP-AMB"""
    
    @abstractmethod
    def ingest_turn(self, session_id: str, timestamp: str, speaker: str, text: str):
        """Called for every conversation turn during the ~512k token ingestion phase."""
        pass

    @abstractmethod
    def retrieve_context(self, query: str, top_k: int = 5) -> str:
        """Called during evaluation to retrieve context string for a user query."""
        pass


import os

class FPAMBEvaluator:
    def __init__(self, provider: BaseMemoryProvider, model_name: str = os.getenv("FP_AMB_MODEL", "llama3"), ollama_url: str = os.getenv("FP_AMB_OLLAMA_URL", "http://localhost:11434/api/generate")):
        self.provider = provider
        self.model_name = model_name
        self.ollama_url = ollama_url

    def run_ingestion(self):
        print(f"FP-AMB Evaluator: Starting Ingestion Phase over ~512k token corpus from '{RAW_CORPUS_FILE}'...", flush=True)
        turn_count = 0
        with open(RAW_CORPUS_FILE, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    if entry.get("type") != "SESSION_DELIMITER":
                        turn_count += 1
                        self.provider.ingest_turn(
                            session_id=entry.get("session_id", "Session"),
                            timestamp=entry.get("timestamp", ""),
                            speaker=entry.get("speaker", "User"),
                            text=entry.get("text", "")
                        )
        print(f"FP-AMB Evaluator: Successfully ingested {turn_count} conversation turns.", flush=True)

    def query_llm(self, prompt: str) -> str:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 80}
        }
        for _ in range(3):
            try:
                resp = requests.post(self.ollama_url, json=payload, timeout=90)
                if resp.status_code == 200:
                    return resp.json().get("response", "").strip()
            except Exception:
                time.sleep(1)
        return "[Connection Error]"

    def evaluate(self, zero_llm_smoke_test: bool = False) -> dict:
        self.run_ingestion()
        
        with open(QUESTIONS_FILE, 'r') as f:
            questions = json.load(f)

        print(f"\nFP-AMB Evaluator: Evaluating {len(questions)} items ({'Zero-LLM Smoke Test' if zero_llm_smoke_test else 'Full LLM Accuracy'})...\n", flush=True)

        correct = 0
        total = len(questions)
        start_time = time.time()
        category_stats = {}

        for idx, item in enumerate(questions, 1):
            q = item["question"]
            expected = item["expected_answer"]
            cat = item["category"]

            if cat not in category_stats:
                category_stats[cat] = {"pass": 0, "total": 0}

            context = self.provider.retrieve_context(q)
            
            if zero_llm_smoke_test:
                exp_clean = expected.lower().split("(")[0].strip()
                key_words = [w for w in exp_clean.replace(".", "").split() if len(w) > 3]
                match = any(w in context.lower() for w in key_words)
            else:
                prompt = f"[RETRIEVED MEMORY CONTEXT]\n{context}\n\n[QUESTION]\n{q}\n\nAnswer accurately based on context above:"
                resp = self.query_llm(prompt)
                exp_clean = expected.lower().split("(")[0].strip()
                key_words = [w for w in exp_clean.replace(".", "").split() if len(w) > 3]
                match = any(w in resp.lower() for w in key_words) and "unknown" not in resp.lower()

            category_stats[cat]["total"] += 1
            if match:
                category_stats[cat]["pass"] += 1
                correct += 1

            if idx % 25 == 0 or idx == 1:
                print(f"[{idx:03d}/{total}] Item #{idx:03d} -> {'[PASS]' if match else '[FAIL]'}", flush=True)

        elapsed = time.time() - start_time
        score = (correct / total) * 100

        print(f"\n=========================================================================")
        print(f"       FP-AMB EVALUATION COMPLETE: {score:.1f}% ACCURACY ({correct}/{total})")
        print(f"=========================================================================\n")

        return {
            "overall_accuracy_pct": round(score, 1),
            "total_items": total,
            "correct_items": correct,
            "elapsed_seconds": round(elapsed, 2),
            "category_breakdown": category_stats
        }
