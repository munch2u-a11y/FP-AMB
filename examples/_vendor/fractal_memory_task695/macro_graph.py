import math
import time
from typing import List, Dict, Any, Optional, Tuple

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Computes cosine similarity between two vector lists."""
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return dot / (norm1 * norm2)

class ConceptNode:
    """Represents a macro-level concept node supporting Frequency-Driven Vault Crystallization (Habit Formation).
    Nodes start as lightweight Routing Nodes (is_vault_node=False). Upon accumulating direct hits,
    they promote/crystallize into Vault Nodes (is_vault_node=True) with dedicated micro-databases.
    """
    def __init__(self, node_id: str, label: str, embedding: Optional[List[float]] = None, is_vault_node: bool = False, promotion_threshold: int = 4, metadata: Optional[Dict[str, Any]] = None):
        self.node_id = node_id
        self.label = label
        self.embedding = embedding or []
        self.is_vault_node = is_vault_node
        self.activation_count = 1 if is_vault_node else 0
        self.promotion_threshold = promotion_threshold
        self.metadata = metadata or {}
        self.created_at = time.time()
        self.last_accessed = time.time()

    def record_hit(self, boost: int = 1) -> bool:
        """Increments activation frequency count. Promotes node to a Vault Node if threshold is reached."""
        self.activation_count += boost
        self.last_accessed = time.time()
        if not self.is_vault_node and self.activation_count >= self.promotion_threshold:
            self.is_vault_node = True
            return True  # Newly promoted to Vault Node!
        return False

    def update_embedding(self, new_embedding: List[float]):
        self.embedding = new_embedding

    def update_centroid(self, embeddings: List[List[float]]):
        valid_embs = [e for e in embeddings if e]
        if not valid_embs:
            return
        dim = len(valid_embs[0])
        self.embedding = [sum(e[d] for e in valid_embs) / len(valid_embs) for d in range(dim)]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "label": self.label,
            "embedding": self.embedding,
            "is_vault_node": self.is_vault_node,
            "activation_count": self.activation_count,
            "promotion_threshold": self.promotion_threshold,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed
        }

class Edge:
    """Represents a weighted relation between two concept nodes in the graph."""
    def __init__(self, source_id: str, target_id: str, relation: str, weight: float = 1.0, decay_rate: float = 0.01):
        self.source_id = source_id
        self.target_id = target_id
        self.relation = relation
        self.weight = weight
        self.decay_rate = decay_rate
        self.last_traversed = time.time()

    def touch(self, boost: float = 0.1):
        """Reinforces edge weight upon traversal (Spaced Repetition / Reinforcement)."""
        now = time.time()
        time_elapsed = now - self.last_traversed
        self.weight = max(0.1, self.weight * math.exp(-self.decay_rate * (time_elapsed / 3600.0)))
        self.weight += boost
        self.last_traversed = now

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation": self.relation,
            "weight": self.weight,
            "decay_rate": self.decay_rate,
            "last_traversed": self.last_traversed
        }

class MacroGraph:
    """Vectorized Macro Graph supporting Frequency-Driven Vault Crystallization (Habit Formation)."""
    def __init__(self):
        self.nodes: Dict[str, ConceptNode] = {}
        self.edges: List[Edge] = []

    def add_node(self, node_id: str, label: str, embedding: Optional[List[float]] = None, is_vault_node: bool = False, promotion_threshold: int = 4, metadata: Optional[Dict[str, Any]] = None) -> ConceptNode:
        if node_id in self.nodes:
            node = self.nodes[node_id]
            node.label = label
            if embedding:
                node.embedding = embedding
            if metadata:
                node.metadata.update(metadata)
            node.record_hit()
            return node
        node = ConceptNode(node_id, label, embedding, is_vault_node=is_vault_node, promotion_threshold=promotion_threshold, metadata=metadata)
        self.nodes[node_id] = node
        return node

    def add_edge(self, source_id: str, target_id: str, relation: str, weight: float = 1.0) -> Edge:
        for edge in self.edges:
            if edge.source_id == source_id and edge.target_id == target_id and edge.relation == relation:
                edge.touch(boost=0.2)
                return edge
        edge = Edge(source_id, target_id, relation, weight)
        self.edges.append(edge)
        return edge

    def find_entry_nodes(self, query_text: str, query_embedding: Optional[List[float]] = None, top_k: int = 6, vault_only: bool = False) -> List[Tuple[ConceptNode, float]]:
        """Finds candidate entry nodes matching query vector or label.
        If vault_only=True, restricts entry node search strictly to Vault Nodes.
        """
        results: List[Tuple[ConceptNode, float]] = []

        for node_id, node in self.nodes.items():
            if vault_only and not node.is_vault_node:
                continue

            score = 0.0
            if query_embedding and node.embedding:
                score = cosine_similarity(query_embedding, node.embedding)
            
            clean_q = query_text.lower()
            clean_label = node.label.lower()
            if clean_label in clean_q or any(w in clean_q for w in clean_label.split() if len(w) > 3):
                score = max(score, 0.75)

            if score > 0.0:
                results.append((node, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def traverse_neighborhood(self, start_node_ids: List[str], max_hops: int = 2) -> Tuple[List[ConceptNode], List[Edge]]:
        """Traverses Cypher graph neighborhood across all nodes, incrementing frequency hits,
        and returning activated Vault Nodes for micro-database search.
        """
        visited_all: Dict[str, ConceptNode] = {}
        traversed_edges: List[Edge] = []
        queue: List[Tuple[str, int]] = [(nid, 0) for nid in start_node_ids if nid in self.nodes]

        for nid, _ in queue:
            node = self.nodes[nid]
            node.record_hit()
            visited_all[nid] = node

        while queue:
            curr_id, current_hop = queue.pop(0)
            if current_hop >= max_hops:
                continue

            for edge in self.edges:
                neighbor_id = None
                if edge.source_id == curr_id:
                    neighbor_id = edge.target_id
                elif edge.target_id == curr_id:
                    neighbor_id = edge.source_id

                if neighbor_id and neighbor_id in self.nodes:
                    edge.touch(boost=0.1)
                    traversed_edges.append(edge)
                    if neighbor_id not in visited_all:
                        neighbor_node = self.nodes[neighbor_id]
                        neighbor_node.record_hit()
                        visited_all[neighbor_id] = neighbor_node
                        queue.append((neighbor_id, current_hop + 1))

        vault_nodes = [node for node in visited_all.values() if node.is_vault_node]
        if not vault_nodes:
            vault_nodes = list(visited_all.values())

        return vault_nodes, traversed_edges
