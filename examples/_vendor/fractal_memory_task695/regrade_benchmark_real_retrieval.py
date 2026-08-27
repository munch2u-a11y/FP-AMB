import os
import json
import time
import re
from collections import defaultdict
from typing import List, Dict, Any, Optional

from fractal_memory.fractal_ltm import FractalLTM
from fractal_memory.llm_client import LLMClient

LOCOMO_PATH = "/home/nemo/locomo/data/locomo10.json"

CATEGORY_NAMES = {
    1: "Fact Recall / Single-hop",
    2: "Temporal / Time-based",
    3: "Multi-Hop / Hypothetical",
    4: "User Preference / Persona"
}

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def run_true_retrieval_regrade(conv_indices: List[int] = [0, 1, 2, 3, 4], max_q_per_conv: int = 100, global_target_k: int = 30):
    """Evaluates REAL evidence retrieval hit rate across all 481 questions, eliminating false negatives
    caused by hypothetical answer strings (e.g. 'Likely no').
    """
    if not os.path.exists(LOCOMO_PATH):
        print(f"Error: LoCoMo dataset not found at {LOCOMO_PATH}")
        return

    with open(LOCOMO_PATH, "r") as f:
        data = json.load(f)

    client = LLMClient()
    
    print("==================================================================")
    print(f"STARTING REAL EVIDENCE RETRIEVAL RE-GRADE BENCHMARK (Top-K={global_target_k})")
    print("==================================================================")

    cat_stats = defaultdict(lambda: {"total": 0, "true_hits": 0, "string_hits": 0})
    total_q = 0
    total_true_hits = 0
    total_string_hits = 0

    for c_idx in conv_indices:
        if c_idx >= len(data):
            continue

        ltm = FractalLTM(threshold_count=4)
        ltm.consolidation_engine.promotion_threshold = 3

        conv_data = data[c_idx]
        qa_list = [q for q in conv_data.get("qa", []) if q.get("category") != 5]

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
        print(f"\nIngesting Conversation {c_idx} ({len(sampled_qa)} sampled QA pairs)...")

        while f"session_{session_num}" in conv_dict:
            turns = conv_dict.get(f"session_{session_num}", [])
            date_str = conv_dict.get(f"session_{session_num}_date_time", f"Session {session_num}")
            
            for turn in turns:
                speaker = turn.get("speaker", "User")
                text = turn.get("text", "")
                dia_id = turn.get("dia_id") or turn.get("id") or turn.get("turn_id")
                clean_turn_text = f"{speaker}: {text}"
                emb = client.get_embedding(clean_turn_text)
                ltm.store_memory(
                    text=clean_turn_text,
                    embedding=emb,
                    metadata={"speaker": speaker, "date": date_str, "dia_id": dia_id}
                )
            session_num += 1

        c_true_hits = 0
        c_string_hits = 0

        for q_idx, q_item in enumerate(sampled_qa):
            cat = q_item.get("category", 1)
            cat_name = CATEGORY_NAMES.get(cat, f"Category {cat}")
            question = q_item.get("question", "")
            expected = str(q_item.get("answer", ""))
            evidence_keys = q_item.get("evidence", [])
            if isinstance(evidence_keys, str):
                evidence_keys = [evidence_keys]

            q_emb = client.get_embedding(question)
            retrieval_res = ltm.retrieve_memory(query_text=question, query_embedding=q_emb, global_target_k=global_target_k)

            retrieved_memories = retrieval_res["retrieved_memories"]
            recalled_dia_ids = [str(m.get("metadata", {}).get("dia_id")).strip() for m in retrieved_memories if m.get("metadata")]
            recalled_clean = clean_text(retrieval_res["system_prompt_injection"])
            expected_clean = clean_text(expected)

            # 1. Evidence Key Match (True Retrieval Hit)
            evidence_hit = False
            if evidence_keys:
                for ev in evidence_keys:
                    ev_str = str(ev).strip()
                    if ev_str in recalled_dia_ids or clean_text(ev_str) in recalled_clean:
                        evidence_hit = True
                        break

            # 2. String Match
            string_hit = len(expected_clean) > 1 and expected_clean in recalled_clean

            # True Retrieval Hit = Evidence Turn Was Successfully Injected in Context
            true_hit = evidence_hit or string_hit

            cat_stats[cat_name]["total"] += 1
            if true_hit:
                cat_stats[cat_name]["true_hits"] += 1
                c_true_hits += 1
                total_true_hits += 1
            if string_hit:
                cat_stats[cat_name]["string_hits"] += 1
                c_string_hits += 1
                total_string_hits += 1

            total_q += 1

            if (q_idx + 1) % 25 == 0 or (q_idx + 1) == len(sampled_qa):
                print(f"  Progress [Conv {c_idx}]: Processed {q_idx+1}/{len(sampled_qa)} Questions | True Retrieval Recall: {(c_true_hits/(q_idx+1))*100:.1f}%")

        print(f"--- Conv {c_idx} Summary ---")
        print(f"REAL True Evidence Retrieval Rate: {(c_true_hits / max(1, len(sampled_qa))) * 100:.1f}% ({c_true_hits}/{len(sampled_qa)})")

    print("\n==================================================================")
    print("REAL EVIDENCE RETRIEVAL ACCURACY BREAKDOWN BY QUESTION CATEGORY")
    print("==================================================================")
    for cat_name, stats in cat_stats.items():
        t = max(1, stats["total"])
        true_pct = (stats["true_hits"] / t) * 100.0
        str_pct = (stats["string_hits"] / t) * 100.0
        print(f"{cat_name:<30} | REAL True Evidence Hit: {true_pct:5.1f}% ({stats['true_hits']}/{stats['total']}) | String Match: {str_pct:5.1f}%")

    print("------------------------------------------------------------------")
    print(f"TOTAL QUESTIONS EVALUATED:        {total_q}")
    print(f"OVERALL REAL TRUE RETRIEVAL HIT:  {(total_true_hits / max(1, total_q))*100:.1f}% ({total_true_hits}/{total_q})")
    print(f"OLD RIGID STRING MATCH RATE:      {(total_string_hits / max(1, total_q))*100:.1f}% ({total_string_hits}/{total_q})")
    print("==================================================================")

if __name__ == "__main__":
    run_true_retrieval_regrade()
