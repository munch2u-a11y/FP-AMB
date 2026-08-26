#!/usr/bin/env python3
"""
FP-AMB BaseMemoryProvider Interface
-----------------------------------
Abstract Base Class that any long-term memory system must implement to take the FP-AMB Exam.
"""

from abc import ABC, abstractmethod

class BaseMemoryProvider(ABC):
    """Abstract Base Class for Memory Systems evaluated by FP-AMB"""
    
    @abstractmethod
    def ingest_turn(self, session_id: str, timestamp: str, speaker: str, text: str):
        """
        Ingests a single conversation turn from the corpus.
        Called sequentially for all 458 turns (~540k tokens) during ingestion phase.
        """
        pass

    @abstractmethod
    def retrieve_context(self, query: str, top_k: int = 5) -> str:
        """
        Retrieves relevant memory context for a given evaluation query.
        Must return a plain text string representing the context payload.
        """
        pass
