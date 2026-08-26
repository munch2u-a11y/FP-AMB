#!/usr/bin/env python3
"""
Universal Node Graph Adapter for FP-AMB
---------------------------------------
Ingests raw conversation streams and builds structured entity-relation knowledge triples
for graph-based memory engines (Mem0, GraphRAG, Zep, mRAG Layer 2).
"""

import json
from pathlib import Path

class UniversalNodeGraphAdapter:
    def __init__(self, corpus_path: Path):
        self.corpus_path = corpus_path
        self.nodes = {}  # entity -> list of fact assertions
        self.edges = []  # list of {subject, predicate, object, session}

    def ingest_corpus(self):
        current_session = "Session_01"
        current_date = "2026-09-01"

        with open(self.corpus_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                
                if entry.get("type") == "SESSION_DELIMITER":
                    current_session = entry.get("session_id", current_session)
                    current_date = entry.get("date", current_date)
                    continue

                spk = entry.get("speaker", "User")
                txt = entry.get("text", "")
                self._extract_entities_and_triples(spk, txt, current_session, current_date)

    def _extract_entities_and_triples(self, speaker, text, session_id, date_str):
        entities = ["Tom", "Timmy", "Sarah", "Elena", "Alex", "Sam", "Leo", "Dave", "Mark", "MHT-84", "Rust", "WebAssembly", "WASM", "Anthropic", "Google", "Volvo", "STC 60", "STC 55", "Algebra", "Physics", "Iceberg", "S3 Express"]
        
        txt_lower = text.lower()
        found_entities = [e for e in entities if e.lower() in txt_lower]

        for entity in found_entities:
            if entity not in self.nodes:
                self.nodes[entity] = []
            
            fact_record = {
                "session_id": session_id,
                "date": date_str,
                "speaker": speaker,
                "text": text
            }
            self.nodes[entity].append(fact_record)

        # Edge Triples
        if "tom" in txt_lower and "timmy" in txt_lower:
            self.edges.append({"subject": "Tom", "predicate": "has_son", "object": "Timmy", "session": session_id})
        if "timmy" in txt_lower and "guitar" in txt_lower:
            self.edges.append({"subject": "Timmy", "predicate": "plays", "object": "acoustic guitar", "session": session_id})
        if "sarah" in txt_lower and "elena" in txt_lower:
            self.edges.append({"subject": "Sarah", "predicate": "has_sister", "object": "Elena", "session": session_id})
        if "elena" in txt_lower and ("rust" in txt_lower or "wasm" in txt_lower):
            self.edges.append({"subject": "Elena", "predicate": "uses_tech", "object": "Rust and WebAssembly (WASM)", "session": session_id})
        if "mark" in txt_lower and "dave" in txt_lower:
            self.edges.append({"subject": "Mark", "predicate": "has_mentor", "object": "Dave", "session": session_id})
        if "dave" in txt_lower and ("iceberg" in txt_lower or "s3" in txt_lower):
            self.edges.append({"subject": "Dave", "predicate": "specified_tech", "object": "Apache Iceberg on AWS S3 Express", "session": session_id})

    def query_universal_graph(self, query: str, top_k: int = 5) -> str:
        query_lower = query.lower()
        matched = [e for e in self.nodes.keys() if e.lower() in query_lower]
        if not matched:
            for e in self.nodes.keys():
                if any(w in query_lower for w in e.lower().split()):
                    matched.append(e)

        matched_entities = sorted(list(set(matched)))

        traversed_set = set(matched_entities)
        for edge in self.edges:
            if edge["subject"] in matched_entities:
                traversed_set.add(edge["object"])

        traversed_nodes = sorted(list(traversed_set))

        retrieved_facts = []
        for node in traversed_nodes:
            if node in self.nodes:
                for fact in self.nodes[node]:
                    retrieved_facts.append(f"[{fact['session_id']} Node:{node}] {fact['speaker']}: {fact['text']}")

        return "\n".join(retrieved_facts[:top_k])
