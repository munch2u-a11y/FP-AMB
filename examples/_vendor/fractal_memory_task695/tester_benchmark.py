import os
import json
import time
import re
import argparse
from collections import defaultdict
from typing import List, Dict, Any, Optional

from .fractal_ltm import FractalLTM
from .llm_client import LLMClient
from .vector_adapters import ChromaVectorAdapter

LOCOMO_PATH = "/home/nemo/locomo/data/locomo10.json"
LONGMEMEVAL_PATH = "/home/nemo/Local-mRag/longmemeval_data/data/longmemeval_s_cleaned.json"

CATEGORY_NAMES = {
    1: "Multi-Hop Reasoning",
    2: "Temporal / Time-based",
    3: "Fact Recall / Single-hop",
    4: "User Preference / Persona",
    "single-hop": "Single-Hop Fact Recall",
    "multi-hop": "Multi-Hop Reasoning",
    "temporal-reasoning": "Temporal Reasoning",
    "user-persona": "User Persona / Preference",
    "knowledge-update": "Knowledge Update"
}

def clean_text(text: str) -> str:
    return re.sub(r'\s+', ' ', text.lower().strip())

class FractalTesterModel:
    """Benchmark & Accuracy Tester Model for Fractal LTM Architecture with ChromaDB Integration."""
    def __init__(self, threshold_count: int = 4, collection_name: str = "fractal_bench"):
        self.client = LLMClient()
        self.ltm = FractalLTM(threshold_count=threshold_count)

    def reset(self):
        self.ltm = FractalLTM(threshold_count=self.ltm.consolidation_engine.threshold_count)

    def ingest_conversation_turns(self, turns: List[Dict[str, Any]], session_idx: int = 1, date_str: str = ""):
        for idx, turn in enumerate(turns):
            speaker = turn.get("speaker", turn.get("role", "User")).title()
            text = turn.get("text", turn.get("content", ""))
            dia_id = turn.get("dia_id", f"D{session_idx}:{idx+1}")
            if not text:
                continue

            clean_turn_text = f"{speaker}: {text}"
            emb = self.client.get_embedding(clean_turn_text)
            self.ltm.store_memory(
                text=clean_turn_text, 
                embedding=emb, 
                metadata={"speaker": speaker, "date": date_str, "dia_id": dia_id}
            )

    def evaluate_retrieval_accuracy(self, question: str, expected_answer: str, evidence_keys: Optional[List[str]] = None, global_target_k: int = 30) -> Dict[str, Any]:
        """Evaluates pure factual/evidence retrieval accuracy using clean text embeddings."""
        q_emb = self.client.get_embedding(question)

        t0 = time.time()
        retrieval_res = self.ltm.retrieve_memory(query_text=question, query_embedding=q_emb, global_target_k=global_target_k)
        retrieval_time = time.time() - t0

        retrieved_memories = retrieval_res["retrieved_memories"]
        recalled_texts = [m["text"] for m in retrieved_memories]
        recalled_dia_ids = [m.get("metadata", {}).get("dia_id") for m in retrieved_memories if m.get("metadata")]

        context_injection = retrieval_res["system_prompt_injection"]
        expected_clean = clean_text(str(expected_answer))
        recalled_clean = clean_text(context_injection)

        found_ev = []
        if evidence_keys:
            for ev in evidence_keys:
                ev_str = str(ev).strip()
                if ev_str in recalled_dia_ids or clean_text(ev_str) in recalled_clean:
                    found_ev.append(ev_str)

        evidence_hit = len(found_ev) > 0 if evidence_keys else False
        answer_in_context = len(expected_clean) > 1 and expected_clean in recalled_clean
        context_hit = evidence_hit or answer_in_context

        return {
            "question": question,
            "expected_answer": expected_answer,
            "context_hit": 1 if context_hit else 0,
            "evidence_recall": len(found_ev) / max(1, len(evidence_keys)) if evidence_keys else (1.0 if answer_in_context else 0.0),
            "activated_nodes": retrieval_res["activated_nodes"],
            "retrieved_count": len(retrieved_memories),
            "retrieval_time": round(retrieval_time, 4)
        }

def run_large_5conv_benchmark(conv_indices: List[int] = [0, 1, 2, 3, 4], max_q_per_conv: int = 100, global_target_k: int = 30, threshold_count: int = 4):
    """Runs 5-conversation benchmark with 100 questions per conversation across a full mix of categories."""
    if not os.path.exists(LOCOMO_PATH):
        print(f"Error: LoCoMo dataset not found at {LOCOMO_PATH}")
        return

    with open(LOCOMO_PATH, "r") as f:
        data = json.load(f)

    tester = FractalTesterModel(threshold_count=threshold_count)

    print("==================================================================")
    print(f"STARTING 5-CONVERSATION BENCHMARK: 100 QUESTIONS PER CONV (Top-K={global_target_k})")
    print("==================================================================")

    cat_stats = defaultdict(lambda: {"total": 0, "context_hits": 0, "evidence_recall_sum": 0.0})
    conv_stats = {}
    total_q = 0
    total_context_hits = 0
    total_evidence_recall = 0.0

    for c_idx in conv_indices:
        if c_idx >= len(data):
            continue

        tester.reset()
        conv_data = data[c_idx]
        qa_list = [q for q in conv_data.get("qa", []) if q.get("category") != 5]

        # Stratified category sampling up to max_q_per_conv
        cat_buckets = defaultdict(list)
        for q in qa_list:
            cat_buckets[q.get("category", 1)].append(q)

        sampled_qa = []
        per_cat_quota = max(1, max_q_per_conv // max(1, len(cat_buckets)))
        for cat, items in cat_buckets.items():
            sampled_qa.extend(items[:per_cat_quota])
        
        if len(sampled_qa) < max_q_per_conv:
            remaining = [q for q in qa_list if q not in sampled_qa]
            sampled_qa.extend(remaining[:max_q_per_conv - len(sampled_qa)])

        conv_dict = conv_data.get("conversation", conv_data)
        session_num = 1
        t_ingest_start = time.time()
        print(f"\nIngesting Conversation {c_idx} ({len(sampled_qa)} sampled QA pairs) & Forming Graph...")

        while f"session_{session_num}" in conv_dict:
            turns = conv_dict.get(f"session_{session_num}", [])
            date_str = conv_dict.get(f"session_{session_num}_date_time", f"Session {session_num}")
            tester.ingest_conversation_turns(turns, session_idx=session_num, date_str=date_str)
            session_num += 1

        t_ingest = time.time() - t_ingest_start
        node_count = len(tester.ltm.macro_graph.nodes)
        edge_count = len(tester.ltm.macro_graph.edges)
        print(f"Node Graph Formed ({t_ingest:.1f}s): {node_count} Macro Nodes | {edge_count} Edges | Evaluating {len(sampled_qa)} Questions...")

        c_hits = 0
        c_ev_sum = 0.0

        for q_idx, q_item in enumerate(sampled_qa):
            cat = q_item.get("category", "General")
            cat_name = CATEGORY_NAMES.get(cat, f"Category {cat}")
            question = q_item.get("question", "")
            expected = str(q_item.get("answer", ""))
            evidence_keys = q_item.get("evidence", [])
            if isinstance(evidence_keys, str):
                evidence_keys = [evidence_keys]

            res = tester.evaluate_retrieval_accuracy(question, expected, evidence_keys, global_target_k=global_target_k)

            cat_stats[cat_name]["total"] += 1
            cat_stats[cat_name]["context_hits"] += res["context_hit"]
            cat_stats[cat_name]["evidence_recall_sum"] += res["evidence_recall"]

            c_hits += res["context_hit"]
            c_ev_sum += res["evidence_recall"]

            total_q += 1
            total_context_hits += res["context_hit"]
            total_evidence_recall += res["evidence_recall"]

            if (q_idx + 1) % 25 == 0 or (q_idx + 1) == len(sampled_qa):
                print(f"  Progress [Conv {c_idx}]: Processed {q_idx+1}/{len(sampled_qa)} Questions | Running Recall: {(c_hits/(q_idx+1))*100:.1f}%")

        c_tot = max(1, len(sampled_qa))
        conv_stats[c_idx] = {"hits": c_hits, "total": c_tot, "rate": (c_hits/c_tot)*100, "ev_recall": (c_ev_sum/c_tot)*100}
        print(f"--- Conv {c_idx} Summary ---")
        print(f"Context Recall Hit Rate: {(c_hits/c_tot)*100:.1f}% ({c_hits}/{c_tot}) | Evidence Recall: {(c_ev_sum/c_tot)*100:.1f}%\n")

    print("==================================================================")
    print("5-CONVERSATION RETRIEVAL ACCURACY BREAKDOWN BY CONVERSATION")
    print("==================================================================")
    for c_idx, stats in conv_stats.items():
        print(f"Conversation {c_idx:<2} | Context Hit: {stats['rate']:5.1f}% ({stats['hits']}/{stats['total']}) | Evidence Recall: {stats['ev_recall']:5.1f}%")

    print("\n==================================================================")
    print("5-CONVERSATION RETRIEVAL ACCURACY BREAKDOWN BY QUESTION CATEGORY")
    print("==================================================================")
    for cat_name, stats in cat_stats.items():
        tot = max(1, stats["total"])
        c_acc = (stats["context_hits"] / tot) * 100
        ev_rec = (stats["evidence_recall_sum"] / tot) * 100
        print(f"{cat_name:<30} | Context Hit: {c_acc:5.1f}% ({stats['context_hits']}/{tot}) | Evidence Recall: {ev_rec:5.1f}%")

    print("------------------------------------------------------------------")
    print(f"TOTAL QUESTIONS EVALUATED:        {total_q}")
    print(f"OVERALL CONTEXT RECALL HIT RATE:  {(total_context_hits/max(1, total_q))*100:.1f}% ({total_context_hits}/{total_q})")
    print(f"AVERAGE EVIDENCE RECALL RATE:     {(total_evidence_recall/max(1, total_q))*100:.1f}%")
    print("==================================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="5-Conversation 100-QA Benchmark Runner")
    parser.add_argument("--convs", type=str, default="0,1,2,3,4", help="Comma-separated conversation indices")
    parser.add_argument("--max_q", type=int, default=100, help="Max questions per conversation")
    parser.add_argument("--top_k", type=int, default=30, help="Global target Top-K retrieval depth")
    parser.add_argument("--threshold", type=int, default=4, help="Node consolidation threshold item count")
    args = parser.parse_args()

    indices = [int(x) for x in args.convs.split(",")]
    run_large_5conv_benchmark(conv_indices=indices, max_q_per_conv=args.max_q, global_target_k=args.top_k, threshold_count=args.threshold)
