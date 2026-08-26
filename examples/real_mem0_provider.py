#!/usr/bin/env python3
"""
Real Mem0 Provider Adapter for FP-AMB
----------------------------------------
Wraps the actual mem0ai library (github.com/mem0ai/mem0), configured to run
fully locally against Ollama (LLM: granite4.1:8b, embeddings: qwen3-embedding:0.6b)
and a local Chroma vector store -- no cloud API, no API key required.
"""

from mem0 import Memory
from fp_amb import BaseMemoryProvider

CONFIG = {
    "llm": {
        "provider": "ollama",
        "config": {
            "model": "granite4.1:8b",
            "ollama_base_url": "http://localhost:11434",
        },
    },
    "embedder": {
        "provider": "ollama",
        "config": {
            "model": "qwen3-embedding:0.6b",
            "ollama_base_url": "http://localhost:11434",
        },
    },
    "vector_store": {
        "provider": "chroma",
        "config": {
            "collection_name": "fp_amb_real_mem0",
            "path": "/tmp/fp_amb_mem0_chroma",
        },
    },
}


class RealMem0Provider(BaseMemoryProvider):
    def __init__(self):
        self.mem = Memory.from_config(CONFIG)
        self.user_id = "fp_amb_corpus"

    def ingest_turn(self, session_id: str, timestamp: str, speaker: str, text: str):
        self.mem.add(
            f"[{timestamp}] {speaker}: {text}",
            user_id=self.user_id,
            metadata={"session_id": session_id, "timestamp": timestamp, "speaker": speaker},
        )

    def retrieve_context(self, query: str, top_k: int = 5) -> str:
        result = self.mem.search(query, filters={"user_id": self.user_id}, limit=top_k)
        items = result.get("results", result) if isinstance(result, dict) else result
        lines = [r.get("memory", str(r)) for r in items]
        return "\n".join(lines)
