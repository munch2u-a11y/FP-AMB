import time
import re
from typing import List, Dict, Any, Optional, Callable
from .macro_graph import MacroGraph, ConceptNode, cosine_similarity
from .micro_list import MicroListStore, MicroItem
from .cluster_splitter import ClusterSplitter
from .entity_extractor import extract_topical_entities

class ConsolidationEngine:
    """Asynchronous Consolidation Engine managing Frequency-Driven Vault Crystallization (Habit Formation).
    Topics start as Routing Nodes (is_vault_node=False). Upon accumulating direct hits/frequency across turns,
    they promote/crystallize into Vault Nodes (is_vault_node=True) with dedicated micro-databases.
    """
    def __init__(self, macro_graph: MacroGraph, micro_store: MicroListStore, 
                 threshold_count: int = 4,
                 entry_threshold: float = 0.45,
                 promotion_threshold: int = 3,
                 summary_llm_fn: Optional[Callable[[List[str]], str]] = None,
                 label_llm_fn: Optional[Callable[[List[str]], str]] = None):
        self.macro_graph = macro_graph
        self.micro_store = micro_store
        self.threshold_count = threshold_count
        self.entry_threshold = entry_threshold
        self.promotion_threshold = promotion_threshold
        self.cluster_splitter = ClusterSplitter(min_cluster_size=2, base_threshold=0.45, max_threshold=0.68)
        self.summary_llm_fn = summary_llm_fn or self._default_summarizer
        self.label_llm_fn = label_llm_fn or self._default_labeler

        self.node_deltas: Dict[str, int] = {}

    def ingest_log(self, log_text: str, embedding: Optional[List[float]] = None, metadata: Optional[Dict[str, Any]] = None) -> MicroItem:
        speaker, topics = extract_topical_entities(log_text)
        primary_topic = topics[0] if topics else "General"
        
        target_nodes: List[ConceptNode] = []

        # 1. Process Topic Nodes and record direct frequency hits
        for topic in topics[:3]:
            node_id = f"node_{topic.lower()}"
            concept_node = self.macro_graph.add_node(
                node_id, 
                topic, 
                embedding=embedding, 
                is_vault_node=False, 
                promotion_threshold=self.promotion_threshold
            )
            # Record hit to build activation frequency towards crystallization!
            concept_node.record_hit()
            target_nodes.append(concept_node)

        # 2. Add Speaker Routing Node
        if speaker and speaker.lower() != "user":
            speaker_node_id = f"routing_{speaker.lower()}"
            speaker_node = self.macro_graph.add_node(speaker_node_id, speaker, embedding=embedding, is_vault_node=False)
            for c_node in target_nodes:
                self.macro_graph.add_edge(c_node.node_id, speaker_node.node_id, relation="speaker_factor", weight=1.0)

        # 3. Frequency-Driven Vault Promotion / Storage Allocation
        target_vaults = [n for n in target_nodes if n.is_vault_node]
        
        # If no topic node has crystallized into a Vault Node yet, promote the primary topic node!
        if not target_vaults:
            primary_node = target_nodes[0]
            primary_node.is_vault_node = True
            target_vaults.append(primary_node)

        item_id = f"item_{int(time.time()*1000)}"
        primary_item = None

        # Store memory turn into all active/crystallized Vault Nodes
        for v_node in set(target_vaults):
            item_meta = dict(metadata or {})
            item_meta["speaker"] = speaker
            
            micro_item = MicroItem(
                item_id=item_id,
                node_id=v_node.node_id,
                text=log_text,
                embedding=embedding,
                is_factual_summary=False,
                metadata=item_meta
            )
            self.micro_store.add_item(micro_item)
            if not primary_item:
                primary_item = micro_item

            node_items = self.micro_store.get_items(v_node.node_id)
            item_embeddings = [i.embedding for i in node_items if i.embedding]
            v_node.update_centroid(item_embeddings)

            self.node_deltas[v_node.node_id] = self.node_deltas.get(v_node.node_id, 0) + 1

            if self.node_deltas[v_node.node_id] >= self.threshold_count:
                self.run_node_consolidation(v_node.node_id)

        return primary_item or MicroItem(item_id=item_id, node_id="vault_general", text=log_text)

    def _create_and_link_web_node(self, node_id: str, label: str, embedding: Optional[List[float]] = None, is_vault_node: bool = False) -> ConceptNode:
        new_node = self.macro_graph.add_node(node_id, label, embedding=embedding, is_vault_node=is_vault_node)
        
        if embedding and len(self.macro_graph.nodes) > 1:
            neighbors = []
            for other_id, other_node in self.macro_graph.nodes.items():
                if other_id != node_id and other_node.embedding:
                    sim = cosine_similarity(embedding, other_node.embedding)
                    if sim > 0.25:
                        neighbors.append((other_id, sim))
            neighbors.sort(key=lambda x: x[1], reverse=True)
            for neighbor_id, sim in neighbors[:3]:
                self.macro_graph.add_edge(node_id, neighbor_id, relation="semantic_neighbor", weight=round(sim, 4))

        return new_node

    def run_node_consolidation(self, node_id: str):
        items = self.micro_store.get_items(node_id)
        if not items:
            return

        self.node_deltas[node_id] = 0

        self._consolidate_overlapping_facts(node_id, items)

        total_nodes = len([n for n in self.macro_graph.nodes.values() if n.is_vault_node])
        clusters = self.cluster_splitter.find_clusters(items, total_graph_nodes=total_nodes)

        if len(clusters) > 1:
            self._split_node(node_id, clusters)
        else:
            raw_items = [i for i in items if not i.is_factual_summary]
            if not raw_items:
                return

            anchor_item = raw_items[-1]
            subset: List[MicroItem] = [anchor_item]

            if anchor_item.embedding:
                scored = []
                for item in raw_items[:-1]:
                    if item.embedding:
                        sim = cosine_similarity(anchor_item.embedding, item.embedding)
                        if sim >= 0.45:
                            scored.append((item, sim))
                scored.sort(key=lambda x: x[1], reverse=True)
                subset.extend([i[0] for i in scored[:4]])
            else:
                subset.extend(raw_items[-4:-1])

            self._extract_factual_summary(node_id, subset)

    def _consolidate_overlapping_facts(self, node_id: str, items: List[MicroItem]):
        facts = [i for i in items if i.is_factual_summary and i.embedding]
        if len(facts) < 2:
            return

        merged_any = False
        to_remove = set()

        for i in range(len(facts)):
            if facts[i].item_id in to_remove:
                continue
            for j in range(i + 1, len(facts)):
                if facts[j].item_id in to_remove:
                    continue

                sim = cosine_similarity(facts[i].embedding, facts[j].embedding)
                if sim >= 0.50:
                    fact_a = facts[i]
                    fact_b = facts[j]

                    combined_texts = [fact_a.text, fact_b.text]
                    merged_text = self.summary_llm_fn(combined_texts)

                    combined_sources = list(set(fact_a.source_memory_ids + fact_b.source_memory_ids))
                    
                    merged_emb = None
                    if fact_a.embedding and fact_b.embedding:
                        dim = len(fact_a.embedding)
                        merged_emb = [(a + b) / 2.0 for a, b in zip(fact_a.embedding, fact_b.embedding)]

                    merged_item = MicroItem(
                        item_id=f"fact_merged_{int(time.time()*1000)}",
                        node_id=node_id,
                        text=merged_text,
                        embedding=merged_emb,
                        is_factual_summary=True,
                        source_memory_ids=combined_sources
                    )

                    to_remove.add(fact_a.item_id)
                    to_remove.add(fact_b.item_id)
                    self.micro_store.add_item(merged_item)
                    merged_any = True
                    break

        if to_remove:
            self.micro_store.lists[node_id] = [i for i in self.micro_store.lists[node_id] if i.item_id not in to_remove]

    def _extract_factual_summary(self, node_id: str, items: List[MicroItem]):
        texts = [i.text for i in items]
        source_ids = [i.item_id for i in items]
        
        summary_text = self.summary_llm_fn(texts)
        summary_item = MicroItem(
            item_id=f"fact_{int(time.time()*1000)}",
            node_id=node_id,
            text=summary_text,
            is_factual_summary=True,
            source_memory_ids=source_ids
        )
        self.micro_store.add_item(summary_item)

    def _split_node(self, parent_node_id: str, clusters: List[List[MicroItem]]):
        parent_node = self.macro_graph.nodes.get(parent_node_id)
        parent_label = parent_node.label if parent_node else "Concept"

        for idx, cluster in enumerate(clusters):
            texts = [item.text for item in cluster]
            spk, topics = extract_topical_entities(" ".join(texts))
            sub_topic = topics[0] if topics else f"Part {idx+1}"
            cluster_label = f"{parent_label} - {sub_topic}"
            child_node_id = f"{parent_node_id}_child_{idx+1}_{int(time.time()*1000)}"

            embeddings = [i.embedding for i in cluster if i.embedding]
            centroid = None
            if embeddings:
                dim = len(embeddings[0])
                centroid = [sum(e[d] for e in embeddings) / len(embeddings) for d in range(dim)]

            self._create_and_link_web_node(child_node_id, cluster_label, embedding=centroid, is_vault_node=True)
            self.macro_graph.add_edge(parent_node_id, child_node_id, relation="sub_concept", weight=1.0)

            for item in cluster:
                item.node_id = child_node_id
                self.micro_store.add_item(item)

        self.micro_store.lists[parent_node_id] = []

    def _extract_quick_label(self, text: str) -> str:
        spk, topics = extract_topical_entities(text)
        return topics[0] if topics else "General"

    def _default_summarizer(self, texts: List[str]) -> str:
        return f"[Factual Summary]: " + "; ".join(texts[:3])

    def _default_labeler(self, texts: List[str]) -> str:
        spk, topics = extract_topical_entities(" ".join(texts))
        return topics[0] if topics else "Sub-topic"
