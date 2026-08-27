from typing import List, Dict, Any, Optional, Tuple, Set
import math
from .macro_graph import MacroGraph, ConceptNode, cosine_similarity
from .micro_list import MicroListStore, MicroItem
from .consolidation_engine import ConsolidationEngine

class FractalLTM:
    """Unified Fractal Graph Long-Term Memory Engine with Vault-Prioritized Entry Search,
    Cypher Graph Web Traversal, Order-of-Operations RRF Reranking, Meta-Tag Deduplication,
    and Clean Whole-Item Injection Assembly (free of noisy node label brackets).
    """
    def __init__(self, threshold_count: int = 4):
        self.macro_graph = MacroGraph()
        self.micro_store = MicroListStore()
        self.consolidation_engine = ConsolidationEngine(
            macro_graph=self.macro_graph,
            micro_store=self.micro_store,
            threshold_count=threshold_count
        )

    def store_memory(self, text: str, embedding: Optional[List[float]] = None, metadata: Optional[Dict[str, Any]] = None) -> MicroItem:
        return self.consolidation_engine.ingest_log(text, embedding=embedding, metadata=metadata)

    def retrieve_memory(
        self, 
        query_text: str, 
        query_embedding: Optional[List[float]] = None, 
        global_target_k: int = 30,
        max_hops: int = 2,
        max_injection_chars: int = 4500
    ) -> Dict[str, Any]:
        """Executes Order-of-Operations Optimized Retrieval Pipeline:
        1. Macro-Graph Entry Search (Targeting Vault Nodes strictly)
        2. Cypher Graph Web Neighborhood Traversal (Traverses Routing + Vault Nodes up to 2 hops away)
        3. Un-truncated Micro-List Multi-Head Search across activated Vault Nodes
        4. Unified Global Multi-Head Reranking & Cosine Fusion
        5. Rank-Based Meta-Tag Deduplication & Clean Intact Injection Assembly
        """
        entry_nodes = self.macro_graph.find_entry_nodes(query_text, query_embedding=query_embedding, top_k=8, vault_only=True)
        
        if not entry_nodes:
            entry_nodes = self.macro_graph.find_entry_nodes(query_text, query_embedding=query_embedding, top_k=8, vault_only=False)

        if not entry_nodes or entry_nodes[0][1] <= 0.0:
            return {
                "system_prompt_injection": "No matching long-term memories found.",
                "activated_nodes": [],
                "retrieved_memories": []
            }

        start_ids = [node.node_id for node, score in entry_nodes if score > 0.0]
        
        activated_vault_nodes, traversed_edges = self.macro_graph.traverse_neighborhood(start_ids, max_hops=max_hops)

        activated_node_embeddings = [node.embedding for node in activated_vault_nodes if node.embedding]

        candidate_items: List[Tuple[MicroItem, float]] = []

        for idx, node in enumerate(activated_vault_nodes):
            other_node_embeddings = [emb for i, emb in enumerate(activated_node_embeddings) if i != idx]
            
            node_hits = self.micro_store.search_node_list_with_translation(
                node_id=node.node_id,
                query_text=query_text,
                query_embedding=query_embedding,
                cross_node_embeddings=other_node_embeddings,
                top_k=50
            )
            candidate_items.extend(node_hits)

        if not candidate_items:
            return {
                "system_prompt_injection": "No matching long-term memories found.",
                "activated_nodes": [node.label for node in activated_vault_nodes],
                "retrieved_memories": []
            }

        rescored_candidates: List[Tuple[MicroItem, float]] = []
        for item, base_score in candidate_items:
            vec_sim = 0.0
            if query_embedding and item.embedding:
                vec_sim = cosine_similarity(query_embedding, item.embedding)
            
            final_rescore = 0.6 * vec_sim + 0.4 * base_score
            rescored_candidates.append((item, final_rescore))

        rescored_candidates.sort(key=lambda x: x[1], reverse=True)

        final_selected: List[Tuple[MicroItem, float]] = []
        selected_factual_source_ids: Set[str] = set()
        seen_texts: Set[str] = set()

        for item, score in rescored_candidates:
            if len(final_selected) >= global_target_k:
                break

            clean_t = item.text.strip().lower()
            if clean_t in seen_texts:
                continue

            if not item.is_factual_summary and item.item_id in selected_factual_source_ids:
                continue

            if item.is_factual_summary and item.source_memory_ids:
                selected_factual_source_ids.update(item.source_memory_ids)

            seen_texts.add(clean_t)
            final_selected.append((item, score))

        node_summaries: List[str] = []
        retrieved_items_meta: List[Dict[str, Any]] = []

        for item, score in final_selected:
            node_label = self.macro_graph.nodes[item.node_id].label if item.node_id in self.macro_graph.nodes else "Concept"
            metadata_dict = item.metadata or {}
            retrieved_items_meta.append({
                "item_id": item.item_id,
                "node_id": item.node_id,
                "node_label": node_label,
                "text": item.text,
                "score": round(score, 4),
                "is_factual_summary": item.is_factual_summary,
                "metadata": metadata_dict
            })
            
            if item.is_factual_summary:
                clean_summary = item.text.replace("[Factual Summary]: ", "")
                clean_node_name = node_label.replace("Vault: ", "")
                node_summaries.append(f"- **{clean_node_name}**: {clean_summary}")

        injection_lines = ["[SYSTEM LONG-TERM MEMORY INJECTION]", "Activated Concept Knowledge:"]
        
        if node_summaries:
            for s_line in node_summaries[:4]:
                if len("\n".join(injection_lines)) + len(s_line) + 2 <= max_injection_chars:
                    injection_lines.append(s_line)
                else:
                    break
        else:
            injection_lines.append("- (Direct episodic memory matches retrieved)")

        injection_lines.append("\nRelevant Specific Memories & Facts:")
        
        included_items_meta = []
        for res in retrieved_items_meta:
            # Clean format: - [FACT STATEMENT]: "text" OR - "text" (No noisy node label brackets!)
            if res["is_factual_summary"]:
                item_line = f'- [FACT STATEMENT]: "{res["text"]}"'
            else:
                item_line = f'- "{res["text"]}"'
            
            if len("\n".join(injection_lines)) + len(item_line) + 2 <= max_injection_chars:
                injection_lines.append(item_line)
                included_items_meta.append(res)
            else:
                break

        system_prompt_injection = "\n".join(injection_lines)

        return {
            "system_prompt_injection": system_prompt_injection,
            "activated_nodes": [node.label for node in activated_vault_nodes],
            "retrieved_memories": included_items_meta
        }
