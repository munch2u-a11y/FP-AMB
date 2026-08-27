import os
import json
import time
from typing import List, Dict, Any

from .fractal_ltm import FractalLTM
from .llm_client import LLMClient

LOCOMO_PATH = "/home/nemo/locomo/data/locomo10.json"

def run_ingestion_cluster_scan_test(conv_indices: List[int] = [0, 1, 2], threshold_count: int = 4):
    """Tests turn-by-turn ingestion, dynamic floor threshold scaling, and cluster scanning across 300+ turns."""
    if not os.path.exists(LOCOMO_PATH):
        print(f"Error: LoCoMo dataset not found at {LOCOMO_PATH}")
        return

    with open(LOCOMO_PATH, "r") as f:
        data = json.load(f)

    client = LLMClient()
    ltm = FractalLTM(threshold_count=threshold_count)

    print("==================================================================")
    print("STARTING DYNAMIC FLOOR THRESHOLD INGESTION & CLUSTER SCAN TEST")
    print("==================================================================")

    total_ingested_turns = 0
    node_creation_history = []

    for c_idx in conv_indices:
        if c_idx >= len(data):
            continue

        conv_dict = data[c_idx].get("conversation", data[c_idx])
        session_num = 1
        conv_turns = 0

        print(f"\n--- Ingesting Conversation {c_idx} ---")

        while f"session_{session_num}" in conv_dict:
            turns = conv_dict.get(f"session_{session_num}", [])
            date_str = conv_dict.get(f"session_{session_num}_date_time", f"Session {session_num}")

            for idx, turn in enumerate(turns):
                speaker = turn.get("speaker", "User").title()
                text = turn.get("text", "")
                dia_id = turn.get("dia_id", f"D{session_num}:{idx+1}")
                if not text:
                    continue

                clean_turn = f"{speaker}: {text}"
                emb = client.get_embedding(clean_turn)
                
                # Ingest turn
                ltm.store_memory(clean_turn, embedding=emb, metadata={"speaker": speaker, "date": date_str, "dia_id": dia_id})
                
                total_ingested_turns += 1
                conv_turns += 1

                # Log progression every 30 turns
                if total_ingested_turns % 30 == 0:
                    current_node_count = len(ltm.macro_graph.nodes)
                    current_dynamic_thresh = ltm.consolidation_engine.cluster_splitter.get_dynamic_threshold(current_node_count)
                    node_creation_history.append((total_ingested_turns, current_node_count, current_dynamic_thresh))
                    
                    print(f"  Ingested {total_ingested_turns:3d} turns | Graph Nodes: {current_node_count:2d} | Dynamic Floor Threshold: {current_dynamic_thresh:.4f}")

            session_num += 1

    print("\n==================================================================")
    print("DYNAMIC FLOOR THRESHOLD SCALING PROGRESSION REPORT")
    print("==================================================================")
    print(f"{'Total Turns Ingested':<22} | {'Macro Node Count':<18} | {'Dynamic Floor Threshold':<25}")
    print("-" * 72)
    for turns_count, node_cnt, thresh in node_creation_history:
        print(f"{turns_count:<22} | {node_cnt:<18} | {thresh:<25.4f}")

    print("\n==================================================================")
    print("FINAL CLUSTER SCANNING & GRAPH TOPOLOGY SUMMARY")
    print("==================================================================")
    print(f"Total Ingested Dialogue Turns:     {total_ingested_turns}")
    print(f"Final Macro Node Count:            {len(ltm.macro_graph.nodes)}")
    print(f"Final Dynamic Floor Threshold:     {ltm.consolidation_engine.cluster_splitter.get_dynamic_threshold(len(ltm.macro_graph.nodes)):.4f}")
    print(f"Macro Graph Relation Edges:        {len(ltm.macro_graph.edges)}")

    # Print formed Macro Node Labels
    print("\nMacro Node Concept Topics Formed:")
    for n_id, node in list(ltm.macro_graph.nodes.items())[:10]:
        item_cnt = len(ltm.micro_store.get_items(n_id))
        print(f"  • Node '{node.label}' ({item_cnt} stored items)")
    if len(ltm.macro_graph.nodes) > 10:
        print(f"  ... (+ {len(ltm.macro_graph.nodes) - 10} more nodes)")

    print("==================================================================")

if __name__ == "__main__":
    import sys
    convs = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [0, 1, 2]
    run_ingestion_cluster_scan_test(conv_indices=convs, threshold_count=4)
