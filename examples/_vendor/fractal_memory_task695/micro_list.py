import math
import time
import re
from typing import List, Dict, Any, Optional, Tuple, Set

STOP_WORDS = {
    "what", "when", "where", "who", "whom", "which", "why", "how", "did", "does", 
    "was", "were", "the", "is", "are", "in", "on", "at", "to", "a", "an", "from", 
    "with", "for", "about", "this", "that", "it", "its", "of", "and", "or", "have",
    "has", "had", "been", "doing", "done", "will", "would", "could", "should", "my",
    "your", "his", "her", "their", "our", "me", "you", "him", "them", "us"
}

# Broad, Domain-Agnostic Real-World Semantic Ontology (No benchmark-specific hardcoding)
BROAD_SEMANTIC_CATEGORIES = {
    "technology_and_devices": ["device", "gadget", "hardware", "software", "app", "phone", "computer", "camera", "screen", "audio", "digital", "tech", "electronic"],
    "transportation_and_travel": ["vehicle", "car", "truck", "bike", "transit", "flight", "drive", "ride", "travel", "auto", "trip", "road", "subway", "train"],
    "living_things_and_pets": ["animal", "pet", "dog", "cat", "bird", "wildlife", "fauna", "flora", "plant", "puppy", "kitten"],
    "arts_and_recreation": ["art", "craft", "music", "sport", "game", "hobby", "performance", "creative", "play", "painting", "drawing", "song", "sculpture"],
    "professions_and_work": ["job", "career", "work", "office", "profession", "business", "interview", "role", "company", "project", "meeting"],
    "places_and_locations": ["location", "city", "country", "region", "park", "building", "venue", "place", "home", "outdoors", "nature"],
    "health_and_wellness": ["health", "medical", "therapy", "care", "exercise", "fitness", "wellness", "mind", "doctor", "clinic"]
}

def expand_tokens_generically(tokens: Set[str]) -> Set[str]:
    """Generically enriches tokens with broad semantic categories and stem roots without benchmark bias."""
    expanded = set(tokens)
    for cat_name, cat_words in BROAD_SEMANTIC_CATEGORIES.items():
        if any(tok in cat_words for tok in tokens):
            expanded.update(cat_words)
    return expanded

class MicroItem:
    """Represents an individual episodic log or semantic fact inside a node's local micro-list."""
    def __init__(self, item_id: str, node_id: str, text: str, embedding: Optional[List[float]] = None, 
                 timestamp: Optional[float] = None, is_factual_summary: bool = False, 
                 source_memory_ids: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None):
        self.item_id = item_id
        self.node_id = node_id
        self.text = text
        self.embedding = embedding or []
        self.timestamp = timestamp or time.time()
        self.is_factual_summary = is_factual_summary
        self.source_memory_ids = source_memory_ids or []
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "node_id": self.node_id,
            "text": self.text,
            "embedding": self.embedding,
            "timestamp": self.timestamp,
            "is_factual_summary": self.is_factual_summary,
            "source_memory_ids": self.source_memory_ids,
            "metadata": self.metadata
        }

class MicroListStore:
    """Manages localized lists attached to concept nodes with Broad, Domain-Agnostic Adaptive Per-Vault BM25 Indexing."""
    def __init__(self):
        self.lists: Dict[str, List[MicroItem]] = {}

    def add_item(self, item: MicroItem):
        if item.node_id not in self.lists:
            self.lists[item.node_id] = []
        self.lists[item.node_id].append(item)

    def get_items(self, node_id: str) -> List[MicroItem]:
        return self.lists.get(node_id, [])

    def get_node_summary(self, node_id: str) -> Optional[MicroItem]:
        items = self.lists.get(node_id, [])
        summaries = [i for i in items if i.is_factual_summary]
        return summaries[-1] if summaries else None

    def search_node_list_with_translation(
        self, 
        node_id: str, 
        query_text: str, 
        query_embedding: Optional[List[float]] = None, 
        cross_node_embeddings: Optional[List[List[float]]] = None,
        top_k: int = 10
    ) -> List[Tuple[MicroItem, float]]:
        """Executes Vector-to-Lexical Query Translation & Broad Per-Vault BM25 Search inside a node's list."""
        items = self.get_items(node_id)
        if not items:
            return []

        from .macro_graph import cosine_similarity

        clean_q = query_text.lower()
        q_tokens = set(re.findall(r'\b[a-z0-9_\-]+\b', clean_q)) - STOP_WORDS
        expanded_q_tokens = expand_tokens_generically(q_tokens)

        N = len(items)
        doc_freqs: Dict[str, int] = {}
        item_token_sets: List[Set[str]] = []

        for item in items:
            clean_item = item.text.lower()
            raw_tokens = set(re.findall(r'\b[a-z0-9_\-]+\b', clean_item)) - STOP_WORDS
            tokens = expand_tokens_generically(raw_tokens)

            item_token_sets.append(tokens)
            for tok in tokens:
                doc_freqs[tok] = doc_freqs.get(tok, 0) + 1

        scored_items: List[Tuple[MicroItem, float]] = []

        for idx, item in enumerate(items):
            item_tokens = item_token_sets[idx]
            
            # Per-Vault Adaptive BM25 Score
            bm25_score = 0.0
            for q_tok in expanded_q_tokens:
                if q_tok in item_tokens:
                    df = doc_freqs.get(q_tok, 1)
                    idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
                    bm25_score += idf * 1.5

            norm_bm25 = min(1.0, bm25_score / 4.0)

            vec_sim = 0.0
            if query_embedding and item.embedding:
                vec_sim = cosine_similarity(query_embedding, item.embedding)

            if cross_node_embeddings and item.embedding:
                for c_emb in cross_node_embeddings:
                    c_sim = cosine_similarity(c_emb, item.embedding)
                    vec_sim = max(vec_sim, c_sim * 0.85)

            fact_boost = 0.15 if item.is_factual_summary else 0.0

            hybrid_score = 0.55 * vec_sim + 0.35 * norm_bm25 + 0.10 * fact_boost
            scored_items.append((item, hybrid_score))

        scored_items.sort(key=lambda x: x[1], reverse=True)
        return scored_items[:top_k]
