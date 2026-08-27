#!/usr/bin/env python3
"""
Real Fractal Memory Provider Adapter for FP-AMB
------------------------------------------------
Wraps the actual Fractal Memory engine (frequency-driven vault crystallization
graph memory: topics start as lightweight routing nodes and promote to full
vault nodes with dedicated micro-databases once they accumulate enough hits)
behind the FP-AMB BaseMemoryProvider interface. No mocking -- uses the real
FractalLTM.store_memory()/retrieve_memory() pipeline exactly as designed.

Runs against a vendored snapshot pinned to a specific reviewed commit rather
than the live project (which has uncommitted, in-progress work on top of it)
-- see examples/_vendor/fractal_memory_task695/PROVENANCE.md for exactly which
commit/subfolder this is and why.

Uses the project's own LLMClient for embeddings (Ollama qwen3-embedding:0.6b,
real vectors, not a dummy fallback), matching how the engine is meant to be used.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "_vendor"))

from fractal_memory_task695 import FractalLTM
from fractal_memory_task695.llm_client import LLMClient

from fp_amb import BaseMemoryProvider


class RealFractalMemoryProvider(BaseMemoryProvider):
    def __init__(self):
        self.llm = LLMClient()
        self.ltm = FractalLTM()

    def ingest_turn(self, session_id: str, timestamp: str, speaker: str, text: str):
        embedding = self.llm.get_embedding(text)
        self.ltm.store_memory(text, embedding=embedding, metadata={"session_id": session_id, "timestamp": timestamp})

    def retrieve_context(self, query: str, top_k: int = 5) -> str:
        query_embedding = self.llm.get_embedding(query)
        result = self.ltm.retrieve_memory(query, query_embedding=query_embedding)
        return result.get("system_prompt_injection", "")


if __name__ == "__main__":
    import time

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import json

    sample = json.load(open("/tmp/fpamb_sample.json"))
    provider = RealFractalMemoryProvider()

    t0 = time.time()
    for t in sample:
        provider.ingest_turn(t["session_id"], t["timestamp"], t["speaker"], t["text"])
    print(f"ingested {len(sample)} turns in {time.time()-t0:.2f}s")

    t0 = time.time()
    ctx = provider.retrieve_context("Who is Alex's brother?")
    print(f"retrieve_context took {time.time()-t0:.2f}s")
    print(ctx)
