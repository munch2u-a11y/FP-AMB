#!/usr/bin/env python3
"""
Real MemPalace Memory Provider Adapter for FP-AMB
------------------------------------------------------
Wraps the actual MemPalace CLI (ChromaDB + BM25 hybrid search, not a mock)
behind the FP-AMB BaseMemoryProvider interface. Buffers ingested turns,
writes them as a real conversation JSONL, mines them into a fresh palace via
`mempalace mine --mode convos`, then shells out to `mempalace search` per query.
"""

import json
import subprocess
import sys
import tempfile
import shutil
from pathlib import Path

from fp_amb import BaseMemoryProvider

# Dynamically resolve mempalace package and executable paths
home = Path.home()
mempalace_dir = str(home / "mempalace")
if mempalace_dir not in sys.path:
    sys.path.insert(0, mempalace_dir)

MEMPALACE_BIN = shutil.which("mempalace") or str(home / "mempalace" / ".venv" / "bin" / "mempalace")


class RealMemPalaceProvider(BaseMemoryProvider):
    def __init__(self):
        self.workdir = Path(tempfile.mkdtemp(prefix="fpamb_mempalace_"))
        self.convo_dir = self.workdir / "convos"
        self.convo_dir.mkdir()
        self.palace_dir = self.workdir / "palace"
        self.turns = []
        self._mined = False

    def ingest_turn(self, session_id: str, timestamp: str, speaker: str, text: str):
        role = "assistant" if speaker == "Assistant" else "user"
        self.turns.append({
            "session_id": session_id,
            "type": role,
            "message": {"role": role, "content": text},
            "timestamp": timestamp,
        })

    def _mine(self):
        by_session = {}
        for t in self.turns:
            by_session.setdefault(t["session_id"], []).append(t)
        for session_id, turns in by_session.items():
            path = self.convo_dir / f"{session_id}.jsonl"
            with open(path, "w") as f:
                for t in turns:
                    f.write(json.dumps({
                        "type": t["type"],
                        "message": t["message"],
                        "timestamp": t["timestamp"],
                    }) + "\n")

        subprocess.run(
            [MEMPALACE_BIN, "--palace", str(self.palace_dir), "init",
             str(self.convo_dir), "--yes", "--no-llm"],
            capture_output=True, text=True, timeout=120,
        )
        subprocess.run(
            [MEMPALACE_BIN, "--palace", str(self.palace_dir), "mine",
             str(self.convo_dir), "--mode", "convos"],
            capture_output=True, text=True, timeout=600,
        )
        self._mined = True

    def retrieve_context(self, query: str, top_k: int = 5) -> str:
        if not self._mined:
            self._mine()
        # Use the real programmatic search API (same hybrid BM25+cosine
        # ranking the CLI uses internally) instead of the CLI's `search`
        # subcommand, whose terminal display truncates each result's text
        # mid-sentence -- that truncation was silently cutting facts out of
        # the scored context before the benchmark ever saw them.
        from mempalace.searcher import search_memories

        result = search_memories(query=query, palace_path=str(self.palace_dir), n_results=top_k)
        hits = result.get("results", [])
        return "\n\n".join(h.get("text", "") for h in hits)

    def __del__(self):
        shutil.rmtree(self.workdir, ignore_errors=True)


if __name__ == "__main__":
    import time

    sample = json.load(open("/tmp/fpamb_sample.json"))
    provider = RealMemPalaceProvider()

    t0 = time.time()
    for t in sample:
        provider.ingest_turn(t["session_id"], t["timestamp"], t["speaker"], t["text"])
    print(f"buffered {len(sample)} turns in {time.time()-t0:.2f}s")

    t0 = time.time()
    ctx = provider.retrieve_context("Who is Alex's brother?")
    print(f"first retrieve_context (includes mining) took {time.time()-t0:.2f}s")
    print(ctx)

    t0 = time.time()
    ctx2 = provider.retrieve_context("What did Sarah's sister decide about Rust?")
    print(f"second retrieve_context took {time.time()-t0:.2f}s")
    print(ctx2)
