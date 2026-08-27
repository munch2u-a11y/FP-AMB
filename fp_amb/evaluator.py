#!/usr/bin/env python3
"""
FP-AMB Evaluator Engine
-----------------------
Executes the FP-AMB Exam against any BaseMemoryProvider.
Calculates full benchmark data:
 - Context Recall Rate (%) & Full LLM Accuracy (%)
 - Retrieval Latency (ms) & LLM Generation Latency (s)
 - Injected Context Token Payload Size (Tokens & Characters)
 - Token Efficiency Ratio (Accuracy % per 1,000 injected tokens)
 - Complete 9-Category Breakdown with Strict Distractor-Aware Refusal Evaluation
 - Automated Failure Cause Diagnosis & Misses Report Generation
"""

import time
import json
import requests
import re
from pathlib import Path
from typing import Optional
from .interface import BaseMemoryProvider
from .dataset import load_corpus, load_master_answer_key

ROOT = Path(__file__).resolve().parent.parent
import os

DEFAULT_OUTPUT_DIR = ROOT / "results"

# Generic short acceptors ("no"/"yes") are common enough in ordinary dialogue
# that they're unreliable as *the* signal of a correct answer on their own:
# a model can say "No, I'm not sure" without ever supplying the corrected
# fact and still trivially contain the word "no". Excluded from 'exact'-mode
# scoring (a specific fact term is required there) and from the
# fact-in-context diagnostic (verified: this alone made ~65% of Adversarial
# Defense misses look like generation failures when the specific corrected
# fact was never retrieved at all). Still honored under 'judgment' mode,
# where a bare affirmative/negative genuinely is the expected answer.
GENERIC_ACCEPTORS = {"no", "yes"}


def _fmt_score(x: float) -> str:
    """245 instead of 245.0 when there's no partial credit in the mix, 245.5 when there is."""
    return str(int(x)) if float(x).is_integer() else f"{x:.1f}"


def _term_hit(term: str, text_clean: str) -> bool:
    """Word-boundary match for short terms (<=3 chars, to avoid substring noise
    like 'no' inside 'know'), plain substring for longer ones."""
    term_clean = term.lower().strip()
    if not term_clean:
        return False
    if len(term_clean) <= 3:
        return re.search(r"\b" + re.escape(term_clean) + r"\b", text_clean) is not None
    return term_clean in text_clean


def _diagnose_zero(specific_terms, context_clean: str, use_llm_generation: bool) -> dict:
    fact_in_context = any(_term_hit(t, context_clean) for t in specific_terms if len(t.strip()) > 1)
    if not fact_in_context:
        return {"score": 0.0, "failure_cause": "RETRIEVAL_FAILURE_FACT_NOT_IN_MEMORY_INJECTION",
                "failure_reason": "Ground-truth answer fact never appeared in the retrieved memory context payload."}
    if use_llm_generation:
        return {"score": 0.0, "failure_cause": "GENERATION_FAILURE_LLM_OUTPUT_MISSED",
                "failure_reason": "Fact was present in retrieved memory payload, but LLM failed to output correct answer."}
    return {"score": 0.0, "failure_cause": "FORMAT_OR_EXACT_KEYWORD_MISMATCH",
            "failure_reason": "Fact was present in retrieved memory payload, but exact keyword matching criteria failed."}


def score_item(item: dict, q: str, target_text: str, context: str, use_llm_generation: bool) -> dict:
    """Per-question grading, dispatched by item['grading_mode']:
      - 'exact' (default): a specific fact term must be present -- a bare
        generic "no"/"yes" alone does not satisfy a fact-correction question.
      - 'judgment': for deduction/reasoning-correction questions with no single
        "real" fact term -- a general affirmative/negative is the actual answer.
      - 'list': multi-part answers. Each list_items entry is a set of synonyms
        for one required sub-fact; score is 1.0 if all are present, 0.5 if some
        (but not all) are present, 0.0 if none are.
    Returns {"score": 0.0/0.5/1.0, "failure_cause": str|None, "failure_reason": str|None}.
    """
    cat = item["category"]
    accepted_list = item.get("accepted_answers", [item.get("ground_truth_answer", "")])
    grading_mode = item.get("grading_mode", "exact")
    target_clean = target_text.lower()
    context_clean = context.lower()

    if cat == "Unanswerable & Absent Memory Refusal":
        if use_llm_generation:
            match = any(w in target_clean for w in ["unknown", "not mentioned", "not stated", "no information", "never discussed"])
        else:
            fetched_distractor = False
            if "tokyo" in q.lower() and "tokyo" in target_clean and "vacation" not in target_clean:
                fetched_distractor = True
            elif "electric car" in q.lower() and "electric car" in target_clean and "purchased" not in target_clean:
                fetched_distractor = True
            elif "dog" in q.lower() and "dog" in target_clean and "sarah" not in target_clean:
                fetched_distractor = True
            match = not fetched_distractor
        if match:
            return {"score": 1.0, "failure_cause": None, "failure_reason": None}
        return {"score": 0.0, "failure_cause": "FALSE_RETRIEVAL_DISTRACTOR_TRAP",
                "failure_reason": "Provider retrieved distractor memory payload for an unanswerable query."}

    if grading_mode == "list" and item.get("list_items"):
        list_items = item["list_items"]
        satisfied = sum(1 for synonyms in list_items if any(_term_hit(s, target_clean) for s in synonyms))
        n = len(list_items)
        if satisfied == n:
            return {"score": 1.0, "failure_cause": None, "failure_reason": None}
        if satisfied > 0:
            return {"score": 0.5, "failure_cause": "PARTIAL_LIST_MATCH",
                    "failure_reason": f"Only {satisfied} of {n} required answer components were found."}
        specific_terms = [s for synonyms in list_items for s in synonyms]
        return _diagnose_zero(specific_terms, context_clean, use_llm_generation)

    if grading_mode == "judgment":
        if any(_term_hit(a, target_clean) for a in accepted_list):
            return {"score": 1.0, "failure_cause": None, "failure_reason": None}
        return _diagnose_zero(accepted_list, context_clean, use_llm_generation)

    # exact mode (default)
    specific_terms = [a for a in accepted_list if a.lower().strip() not in GENERIC_ACCEPTORS]
    if any(_term_hit(a, target_clean) for a in specific_terms):
        return {"score": 1.0, "failure_cause": None, "failure_reason": None}
    return _diagnose_zero(specific_terms, context_clean, use_llm_generation)

class FPAMBEvaluator:
    def __init__(
        self,
        provider: BaseMemoryProvider,
        provider_name: str = "CustomMemoryEngine",
        model_name: Optional[str] = os.getenv("FP_AMB_MODEL", "llama3"),
        ollama_url: str = os.getenv("FP_AMB_OLLAMA_URL", "http://localhost:11434/api/generate"),
        use_llm_generation: bool = False
    ):
        self.provider = provider
        self.provider_name = provider_name
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.use_llm_generation = use_llm_generation

    def run_ingestion(self):
        turns = load_corpus()
        total_chars = sum(len(t.get("text", "")) for t in turns)
        total_tokens_est = total_chars // 4
        self.corpus_turn_count = len(turns)
        self.corpus_token_est = total_tokens_est
        print(f"FP-AMB Exam: Ingesting {len(turns)} conversation turns (~{total_tokens_est:,} tokens) into '{self.provider_name}'...", flush=True)
        start_time = time.time()
        for turn in turns:
            self.provider.ingest_turn(
                session_id=turn.get("session_id", "Session"),
                timestamp=turn.get("timestamp", ""),
                speaker=turn.get("speaker", "User"),
                text=turn.get("text", "")
            )
        elapsed = time.time() - start_time
        print(f"FP-AMB Exam: Ingestion complete in {elapsed:.2f} seconds.\n", flush=True)
        return elapsed

    def _query_llm(self, prompt: str) -> str:
        if self.model_name and self.model_name.lower().startswith("gemini"):
            return self._query_gemini(prompt)
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 90}
        }
        for _ in range(3):
            try:
                resp = requests.post(self.ollama_url, json=payload, timeout=90)
                if resp.status_code == 200:
                    return resp.json().get("response", "").strip()
            except Exception:
                time.sleep(1)
        return "[Connection Timeout]"

    def _query_gemini(self, prompt: str) -> str:
        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            return "[Gemini API Key Missing]"
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            # Default (dynamic) thinking budget, not disabled: adversarial/gaslighting-
            # worded questions need it to correct a false premise instead of defaulting
            # to "Unknown" -- verified empirically, thinkingBudget=0 answered "Unknown"
            # on every one even with the correcting fact present in context.
            "generationConfig": {"temperature": 0.1, "maxOutputTokens": 1024},
        }
        headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}
        for attempt in range(3):
            try:
                resp = requests.post(url, json=payload, headers=headers, timeout=30)
                if resp.status_code == 200:
                    parts = resp.json().get("candidates", [{}])[0].get("content", {}).get("parts", [])
                    return "".join(p.get("text", "") for p in parts).strip()
                time.sleep(2 ** attempt if resp.status_code == 429 else 1)
            except Exception:
                time.sleep(1)
        return "[Connection Timeout]"

    def evaluate(self, output_path: Optional[Path] = None) -> dict:
        ingest_time = self.run_ingestion()
        master_items = load_master_answer_key()
        total_q = len(master_items)

        mode_str = f"Full LLM Generation ({self.model_name})" if self.use_llm_generation else "Pure Retrieval (Zero-LLM Latency)"
        print("=========================================================================", flush=True)
        print(f"       FP-AMB EXAM EXECUTING: {self.provider_name} [{mode_str}]", flush=True)
        print("=========================================================================\n", flush=True)

        category_scores = {}
        correct_count = 0
        injected_tokens = []
        injected_chars = []
        retrieval_times = []
        generation_times = []
        item_logs = []

        start_eval_time = time.time()

        for idx, (q_id, item) in enumerate(master_items.items(), 1):
            cat = item["category"]
            q = item["question"]
            expected = item.get("ground_truth_answer") or item.get("expected_answer", "")
            accepted_list = item.get("accepted_answers", [expected])

            if cat not in category_scores:
                category_scores[cat] = {"correct": 0, "total": 0}

            # Step 1: Memory Retrieval
            t0_ret = time.time()
            context = self.provider.retrieve_context(q)
            ret_time = time.time() - t0_ret
            retrieval_times.append(ret_time)

            inj_tok = len(context.split())
            inj_ch = len(context)
            injected_tokens.append(inj_tok)
            injected_chars.append(inj_ch)

            # Step 2: Evaluation (LLM or Pure Retrieval)
            if self.use_llm_generation:
                prompt = f"""[RETRIEVED MEMORY CONTEXT]
{context}

[QUESTION]
{q}

Answer accurately based on context above. If context does not contain information to answer the question, state 'Unknown':"""
                t0_gen = time.time()
                target_text = self._query_llm(prompt)
                gen_time = time.time() - t0_gen
                generation_times.append(gen_time)
            else:
                target_text = context
                gen_time = 0.0

            # Step 3: Per-Question Graded Evaluation (exact / judgment / list)
            result = score_item(item, q, target_text, context, self.use_llm_generation)
            score = result["score"]
            match = score >= 1.0

            category_scores[cat]["total"] += 1
            category_scores[cat]["correct"] += score
            correct_count += score

            if idx % 25 == 0 or idx == 1:
                status = "[PASS]" if match else ("[PARTIAL]" if score > 0 else "[FAIL]")
                print(f"[{idx:03d}/{total_q}] [{cat[:18]:<18}] Q: '{q[:32]}...' -> {status}", flush=True)

            log_entry = {
                "id": q_id,
                "category": cat,
                "question": q,
                "expected_answer": expected,
                "retrieved_context": context,
                "pass": match,
                "score": score,
                "injected_tokens": inj_tok,
                "retrieval_time_ms": round(ret_time * 1000.0, 2),
                "generation_time_seconds": round(gen_time, 2)
            }
            if self.use_llm_generation:
                log_entry["generated_output"] = target_text
            if not match:
                log_entry["failure_cause"] = result["failure_cause"]
                log_entry["failure_reason"] = result["failure_reason"]

            item_logs.append(log_entry)

        total_eval_time = time.time() - start_eval_time
        accuracy_pct = (correct_count / total_q) * 100

        total_inj_tok = sum(injected_tokens)
        min_inj_tok = min(injected_tokens) if injected_tokens else 0
        max_inj_tok = max(injected_tokens) if injected_tokens else 0
        total_inj_ch = sum(injected_chars)

        avg_inj_tok = sum(injected_tokens) / total_q if total_q > 0 else 0
        avg_inj_ch = sum(injected_chars) / total_q if total_q > 0 else 0
        avg_ret_ms = (sum(retrieval_times) / total_q) * 1000.0 if total_q > 0 else 0
        avg_gen_s = (sum(generation_times) / total_q) if self.use_llm_generation else 0.0
        tok_eff_ratio = (accuracy_pct / (avg_inj_tok / 1000.0)) if avg_inj_tok > 0 else 0

        print("\n=========================================================================", flush=True)
        print(f"   FP-AMB EXAM COMPLETE: {self.provider_name} ACCURACY: {accuracy_pct:.1f}% ({_fmt_score(correct_count)}/{total_q})", flush=True)
        print("=========================================================================\n", flush=True)

        print("----------------------------------------------------------------------------------------------------", flush=True)
        print(f"                       FP-AMB PERFORMANCE REPORT: {self.provider_name}", flush=True)
        print("----------------------------------------------------------------------------------------------------", flush=True)
        print(f"  - Overall Score:                        {accuracy_pct:.1f}% ({_fmt_score(correct_count)}/{total_q} items passed)")
        print(f"  - Total Corpus Size:                    ~{self.corpus_token_est:,} tokens ({self.corpus_turn_count} turns across 60 sessions)")
        print(f"  - Ingestion Phase Time:                 {ingest_time:.2f} seconds")
        print(f"  - Total Exam Duration:                  {total_eval_time:.1f} seconds")
        print(f"  - Avg Retrieval Latency:                {avg_ret_ms:.2f} ms per query")
        if self.use_llm_generation:
            print(f"  - Avg LLM Generation Time:              {avg_gen_s:.2f} seconds per query")
        print(f"  - Total Injected Context Payload:       {total_inj_tok:,} tokens across exam")
        print(f"  - Avg Injected Context Payload Size:     {avg_inj_tok:.1f} tokens ({avg_inj_ch:.1f} chars) [Range: {min_inj_tok}-{max_inj_tok} tok]")
        print(f"  - Token Efficiency Ratio:               {tok_eff_ratio:.2f} accuracy points per 1k injected tokens")
        print("----------------------------------------------------------------------------------------------------\n", flush=True)

        print("-------------------------------------------------------------------------", flush=True)
        print("                   CATEGORY-BY-CATEGORY SCORECARD                        ", flush=True)
        print("-------------------------------------------------------------------------", flush=True)
        print(f"{'Category Name':<42} | {'Accuracy':<10} | {'Passed/Total':<12}", flush=True)
        print("-" * 72, flush=True)
        for cname, score in sorted(category_scores.items()):
            acc_pct = (score["correct"] / score["total"]) * 100 if score["total"] > 0 else 0
            print(f"{cname[:42]:<42} | {acc_pct:>9.1f}% | {score['correct']}/{score['total']}", flush=True)
        print("-------------------------------------------------------------------------\n", flush=True)

        report_payload = {
            "benchmark_title": f"FP-AMB Exam: {self.provider_name}",
            "provider_name": self.provider_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "use_llm_generation": self.use_llm_generation,
            "model_name": self.model_name if self.use_llm_generation else None,
            "total_corpus_tokens": self.corpus_token_est,
            "total_evaluated_questions": total_q,
            "overall_accuracy_pct": round(accuracy_pct, 1),
            "passed_items": round(correct_count, 1),
            "ingestion_duration_seconds": round(ingest_time, 2),
            "eval_duration_seconds": round(total_eval_time, 2),
            "avg_retrieval_latency_ms": round(avg_ret_ms, 2),
            "avg_generation_latency_seconds": round(avg_gen_s, 2),
            "total_injected_tokens_across_exam": total_inj_tok,
            "avg_injected_tokens_per_query": round(avg_inj_tok, 1),
            "min_injected_tokens_per_query": min_inj_tok,
            "max_injected_tokens_per_query": max_inj_tok,
            "total_injected_chars_across_exam": total_inj_ch,
            "avg_injected_chars_per_query": round(avg_inj_ch, 1),
            "token_efficiency_ratio": round(tok_eff_ratio, 2),
            "category_breakdown": category_scores,
            "item_logs": item_logs
        }

        if output_path is None:
            DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            run_stamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = DEFAULT_OUTPUT_DIR / f"{self.provider_name.lower()}_scorecard_{run_stamp}.json"

        with open(output_path, 'w') as f:
            json.dump(report_payload, f, indent=2)

        print(f"Saved FP-AMB Exam scorecard to '{output_path}'.", flush=True)

        html_path = Path(output_path).with_suffix(".html")
        from .report import write_report, write_text_report, write_misses_report
        write_report(report_payload, html_path)
        print(f"Saved FP-AMB visual report to '{html_path}'.", flush=True)

        text_path = Path(output_path).with_suffix(".md")
        write_text_report(report_payload, text_path)
        print(f"Saved FP-AMB text/ASCII report to '{text_path}'.", flush=True)

        misses_path = Path(output_path).with_name(f"{output_path.stem}_misses.txt")
        write_misses_report(report_payload, misses_path)
        print(f"Saved FP-AMB missed questions & failure analysis report to '{misses_path}'.", flush=True)

        return report_payload
