import atexit
import hashlib
import json
import os
import time
import urllib.request
from typing import Any, Dict, List, Optional

# Was hardcoded to another tool's scratch directory, which coupled this project to
# ~/.gemini/antigravity/... and left the repo's own embedding_cache.json dead.
# The default still points there so the existing 3k-entry cache keeps working;
# override with FRACTAL_EMBED_CACHE to move it.
CACHE_FILE = os.environ.get(
    "FRACTAL_EMBED_CACHE",
    os.path.expanduser("~/.gemini/antigravity/scratch/fractal_memory/embedding_cache.json"),
)
# New embeddings are appended here a line at a time; the big JSON is only rewritten
# on compaction. Rewriting the whole cache per miss cost ~1.1s and ~74MB of I/O
# *per new embedding* once the cache had grown.
CACHE_APPEND = os.path.splitext(CACHE_FILE)[0] + ".jsonl"


class EmbeddingError(RuntimeError):
    """Raised when an embedding genuinely could not be produced.

    Deliberately fatal. The previous behaviour returned a 16-dimension pseudo-vector
    while real embeddings are 1024-dimension; cosine_similarity() returns 0.0 on a
    length mismatch, so a transient Ollama hiccup silently made those memories
    unretrievable for the rest of the run, with nothing logged.
    """


def get_text_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


class LLMClient:
    """LLM & embedding client backed by a local Ollama instance."""

    def __init__(self, host: str = "http://localhost:11434",
                 llm_model: str = "granite4.1:8b",
                 embed_model: str = "qwen3-embedding:0.6b",
                 retries: int = 3):
        self.host = host.rstrip("/")
        self.llm_model = llm_model
        self.embed_model = embed_model
        self.retries = retries
        self.cache: Dict[str, List[float]] = self._load_cache()
        self._dim: Optional[int] = None
        self._pending = 0
        for v in self.cache.values():
            if v:
                self._dim = len(v)
                break
        atexit.register(self.compact)

    # ------------------------------------------------------------------- cache

    def _load_cache(self) -> Dict[str, List[float]]:
        cache: Dict[str, List[float]] = {}
        if os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, "r") as f:
                    cache = json.load(f)
            except Exception:
                cache = {}
        if os.path.exists(CACHE_APPEND):
            try:
                with open(CACHE_APPEND, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            k, v = json.loads(line)
                            cache[k] = v
                        except Exception:
                            continue
            except Exception:
                pass
        return cache

    def _append(self, key: str, vec: List[float]):
        try:
            os.makedirs(os.path.dirname(CACHE_APPEND), exist_ok=True)
            with open(CACHE_APPEND, "a") as f:
                f.write(json.dumps([key, vec]) + "\n")
            self._pending += 1
        except Exception:
            pass

    def compact(self):
        """Fold the append log back into the main file. Cheap to call; a no-op when
        nothing new has been written."""
        if not self._pending:
            return
        try:
            os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
            tmp = CACHE_FILE + ".tmp"
            with open(tmp, "w") as f:
                json.dump(self.cache, f)
            os.replace(tmp, CACHE_FILE)
            if os.path.exists(CACHE_APPEND):
                os.remove(CACHE_APPEND)
            self._pending = 0
        except Exception:
            pass

    # -------------------------------------------------------------- embeddings

    def get_embedding(self, text: str) -> List[float]:
        h = get_text_hash(text)
        cached = self.cache.get(h)
        if cached:
            return cached

        url = f"{self.host}/api/embeddings"
        data = json.dumps({"model": self.embed_model, "prompt": text}).encode("utf-8")
        last = None
        for attempt in range(self.retries):
            req = urllib.request.Request(url, data=data,
                                         headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    emb = json.loads(resp.read().decode("utf-8")).get("embedding", [])
                if emb:
                    if self._dim is None:
                        self._dim = len(emb)
                    elif len(emb) != self._dim:
                        raise EmbeddingError(
                            f"embedder returned {len(emb)} dims, expected {self._dim}; "
                            f"model changed under an existing cache?")
                    self.cache[h] = emb
                    self._append(h, emb)
                    return emb
                last = "empty embedding in response"
            except EmbeddingError:
                raise
            except Exception as exc:
                last = exc
            if attempt + 1 < self.retries:
                time.sleep(0.5 * (attempt + 1))

        raise EmbeddingError(
            f"could not embed text after {self.retries} attempts "
            f"({self.embed_model} at {self.host}): {last}")

    # ---------------------------------------------------------------- generate

    def generate(self, prompt: str, system: Optional[str] = None,
                 temperature: float = 0.0, max_tokens: int = 150) -> str:
        url = f"{self.host}/api/generate"
        payload: Dict[str, Any] = {
            "model": self.llm_model, "prompt": prompt, "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if system:
            payload["system"] = system
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data,
                                     headers={"Content-Type": "application/json"})
        # Raises rather than returning "[LLM Answer Error: ...]": that string used to
        # be written straight into memory as if it were a knowledge statement.
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8")).get("response", "").strip()
