#!/usr/bin/env python3
"""
Sample Memory Provider Implementation for FP-AMB Exam
------------------------------------------------------
Demonstrates how to implement BaseMemoryProvider in 15 lines of code.
"""

from fp_amb import BaseMemoryProvider
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class SampleTFIDFMemoryProvider(BaseMemoryProvider):
    def __init__(self):
        self.corpus_turns = []
        self.vectorizer = None
        self.corpus_matrix = None

    def ingest_turn(self, session_id: str, timestamp: str, speaker: str, text: str):
        self.corpus_turns.append(f"[{timestamp}] {speaker}: {text}")

    def retrieve_context(self, query: str, top_k: int = 5) -> str:
        if self.corpus_matrix is None:
            self.vectorizer = TfidfVectorizer(sublinear_tf=True, stop_words='english')
            self.corpus_matrix = self.vectorizer.fit_transform(self.corpus_turns)

        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.corpus_matrix).flatten()
        top_indices = sims.argsort()[::-1][:top_k]
        selected = [self.corpus_turns[i] for i in top_indices if sims[i] > 0]
        return "\n".join(selected)
