#!/usr/bin/env python3
"""
Sample Memory Provider Implementation for FP-AMB Exam
------------------------------------------------------
Demonstrates how to implement BaseMemoryProvider in pure Python.
This template can be used as a reference to integrate your own long-term memory system.
"""

from fp_amb import BaseMemoryProvider, FPAMBEvaluator


class SampleMemoryProvider(BaseMemoryProvider):
    """
    A simple baseline memory provider using standard keyword matching.
    Replace the logic in ingest_turn and retrieve_context with your memory system!
    """

    def __init__(self):
        self.turns = []

    def ingest_turn(self, session_id: str, timestamp: str, speaker: str, text: str):
        """Ingest conversation turn into your memory index or graph store."""
        self.turns.append(f"[{session_id} | {timestamp}] {speaker}: {text}")

    def retrieve_context(self, query: str, top_k: int = 5) -> str:
        """Retrieve relevant context payload for the evaluation query."""
        keywords = [w.lower() for w in query.split() if len(w) > 3]
        matched = []
        for turn in self.turns:
            if any(kw in turn.lower() for kw in keywords):
                matched.append(turn)
        # Return top_k most recent matches
        return "\n".join(matched[-top_k:])


if __name__ == "__main__":
    # Execute benchmark evaluation
    evaluator = FPAMBEvaluator(
        provider=SampleMemoryProvider(),
        provider_name="SampleMemoryProvider"
    )
    evaluator.evaluate(zero_llm_smoke_test=True)
