#!/usr/bin/env python3
"""
Real mRAG (Micro-RAG) Memory Provider Adapter for FP-AMB
------------------------------------------------------------
Wraps the user's actual mRAG package (github.com/munch2u-a11y/mRAG)
behind the FP-AMB BaseMemoryProvider interface. No mocking -- uses the real
MemoryIngestor (Layer 1, zero LLM calls) and PreGenerativeInjector (multi-head
retrieval) exactly as designed.
"""

import sys
import tempfile
from pathlib import Path

# Dynamically resolve local package path if present
local_mrag_dir = str(Path.home() / "Local-mRag")
if local_mrag_dir not in sys.path:
    sys.path.insert(0, local_mrag_dir)

from mrag import BeliefStore, create_vector_store, PreGenerativeInjector
from mrag.core.memory_ingestor import MemoryIngestor

from fp_amb import BaseMemoryProvider


class RealMRAGProvider(BaseMemoryProvider):
    def __init__(self):
        self._tmpdir = tempfile.mkdtemp(prefix="fpamb_mrag_")
        self.belief_store = BeliefStore(data_dir=self._tmpdir)
        self.vector_store = create_vector_store("chromadb", persist_dir=self._tmpdir + "/chroma")
        self.ingestor = MemoryIngestor(self.belief_store)
        self.injector = PreGenerativeInjector(
            belief_store=self.belief_store, vector_store=self.vector_store
        )

    def ingest_turn(self, session_id: str, timestamp: str, speaker: str, text: str):
        self.ingestor.add_event(
            text=text, source="user_input", timestamp=timestamp,
            session_id=session_id, speaker=speaker,
        )

    def retrieve_context(self, query: str, top_k: int = 5) -> str:
        return self.injector.inject(trigger_text=query)
