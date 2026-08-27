import os
import json
import time
from typing import List, Dict, Any

from .fractal_ltm import FractalLTM
from .macro_graph import cosine_similarity
from .llm_client import LLMClient
from .vector_adapters import ChromaVectorAdapter

LOCOMO_PATH = "/home/nemo/locomo/data/locomo10.json"

def run_actual_system_consolidation_test(conv_idx: int = 0, threshold_count: int = 4):
    """Tests the ACTUAL production consolidation pipeline using real ChromaDB vector database,
    real Ollama embedding model (qwen3-embedding:0.6b), real Ollama LLM (granite4.1:8b),
    and scikit-learn clustering algorithms.
    """
    if not os.path.exists(LOCOMO_PATH):
        print(f"Error: LoCoMo dataset not found at {LOCOMO_PATH}")
        return

    with open(LOCOMO_PATH, "r") as f:
        data = json.load(f)

    if conv_idx >= len(data):
        print(f"Error: Conversation index {conv_idx} out of range.")
        return

    print("==================================================================")
    print("INITIALIZING ACTUAL PRODUCTION SYSTEM CONSOLIDATION TEST")
    print("==================================================================")
    print("• Vector Store:  ChromaDB (Persistent)")
    print("• Embedder:      Ollama qwen3-embedding:0.6b")
    print("• LLM Engine:    Ollama granite4.1:8b")
    print("• Dataset:       LoCoMo Conversation 0")
    print("==================================================================\n")

    # Initialize real ChromaDB Vector Adapter
    chroma_dir = f"/tmp/chroma_fractal_test_conv_{conv_idx}"
    if os.path.exists(chroma_dir):
        import shutil
        shutil.rmtree(chroma_dir)

    chroma_adapter = ChromaVectorAdapter(collection_name=f"fractal_conv_{conv_idx}", persist_directory=chroma_dir)
    client = LLMClient()
    ltm = FractalLTM(threshold_count=threshold_count)

    # Real LLM Consolidation Function calling granite4.1:8b
    def real_ollama_summarizer(texts: List[str]) -> str:
        prompt = (
            "You are an AI Memory Consolidation System. "
            "Synthesize the following 4-5 highly related conversation turns into 1 or 2 clear, concise, timeless factual knowledge statements about the user or participants. "
            "Do NOT include conversational filler like 'User said' or 'Hello'. Output ONLY the factual statements.\n\n"
            "Raw Conversation Turns:\n" + "\n".join([f"- {t}" for t in texts]) + "\n\nFactual Knowledge Statements:"
        )
        return client.generate(prompt=prompt, max_tokens=120)

    ltm.consolidation_engine.summary_llm_fn = real_ollama_summarizer

    conv_dict = data[conv_idx].get("conversation", data[conv_idx])
    session_num = 1
    total_turns = 0

    print(f"Ingesting Conversation {conv_idx} turn-by-turn into ChromaDB & Macro Graph...\n")

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
            
            # Store in ChromaDB vector adapter as well as Micro Store
            chroma_adapter.upsert(vector_id=f"item_{dia_id}", embedding=emb, metadata={"speaker": speaker, "date": date_str, "text": clean_turn})
            
            # Ingest through Consolidation Engine (triggers threshold scans & node splits)
            ltm.store_memory(clean_turn, embedding=emb, metadata={"speaker": speaker, "date": date_str, "dia_id": dia_id})
            total_turns += 1

        session_num += 1

    print(f"\nIngestion Complete! Ingested {total_turns} clean turns into ChromaDB.")
    print(f"Macro Graph Formed: {len(ltm.macro_graph.nodes)} Macro Concept Nodes | {len(ltm.macro_graph.edges)} Edges\n")

    # Inspect Factual Summaries generated across all nodes
    consolidation_events = []
    for node_id, node in ltm.macro_graph.nodes.items():
        items = ltm.micro_store.get_items(node_id)
        facts = [i for i in items if i.is_factual_summary]
        for fact in facts:
            sources = [i for i in items if i.item_id in fact.source_memory_ids]
            source_embs = [s.embedding for s in sources if s.embedding]
            
            pairwise_sims = []
            if len(source_embs) > 1:
                for i in range(len(source_embs)):
                    for j in range(i+1, len(source_embs)):
                        pairwise_sims.append(cosine_similarity(source_embs[i], source_embs[j]))
            avg_cohesion = sum(pairwise_sims) / max(1, len(pairwise_sims))

            consolidation_events.append({
                "node_label": node.label,
                "fact_text": fact.text,
                "source_count": len(sources),
                "sources": [s.text for s in sources],
                "source_ids": fact.source_memory_ids,
                "semantic_cohesion": round(avg_cohesion, 4)
            })

    print("==================================================================")
    print(f"ACTUAL SYSTEM CONSOLIDATION REPORT ({len(consolidation_events)} Factual Summaries Generated)")
    print("==================================================================")

    for idx, event in enumerate(consolidation_events, 1):
        print(f"\n[ACTUAL CONSOLIDATION EVENT #{idx}] — Node: '{event['node_label']}'")
        print(f"  Semantic Cohesion Score: {event['semantic_cohesion']:.4f} (Avg Pairwise Vector Similarity)")
        print(f"  Micro-Cluster Sources Consolidated ({event['source_count']} items):")
        for src in event['sources']:
            print(f"    • {src[:95]}")
        print(f"  👉 Generated Factual Knowledge Statement (by granite4.1:8b):")
        print(f"     \"{event['fact_text']}\"")
        print(f"  Meta-Tag Source IDs: {event['source_ids'][:3]}...")
        print("-" * 66)

    print("\n==================================================================")
    print("ACTUAL PRODUCTION SYSTEM CONSOLIDATION SUMMARY")
    print("==================================================================")
    print(f"Vector Store Engine:               ChromaDB ({chroma_dir})")
    print(f"Total Macro Nodes Created:         {len(ltm.macro_graph.nodes)}")
    print(f"Total Consolidation Triggers:      {len(consolidation_events)}")
    if consolidation_events:
        avg_cohesion = sum(e['semantic_cohesion'] for e in consolidation_events) / len(consolidation_events)
        print(f"Average Micro-Cluster Cohesion:    {avg_cohesion:.4f}")
    print("==================================================================")

if __name__ == "__main__":
    import sys
    c_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    run_actual_system_consolidation_test(conv_idx=c_id, threshold_count=4)
