# FP-AMB: First-Person Agent Memory Benchmark (v6.0)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

**FP-AMB (First-Person Agent Memory Benchmark)** is an industry-standard, framework-agnostic evaluation suite engineered for long-term AI agent memory systems operating over realistic, multi-session conversational streams.

It evaluates context recall, multi-hop graph reasoning, temporal date math, adaptability/fact overwrites, speaker attribution traps (incorporating **LoCoMO** and **BEAM** features), refusal to hallucinate absent memories (incorporating **LongMemEval** features), source credibility resolution, and agentic tool-use execution across **10 core categories plus dynamic answer key bindings** over **~540,878 tokens** and 458 turns spanning 60 distinct conversational sessions.

---

## 📌 Table of Contents
- [✨ Key Features](#-key-features)
- [⚡ Quick Start & Installation](#-quick-start--installation)
- [🔌 Implementing Custom Memory Providers](#-implementing-custom-memory-providers)
- [💻 CLI Usage](#-cli-usage)
- [📊 Evaluation Categories (10 Categories + Dynamic Key Portion)](#-evaluation-categories-10-categories--dynamic-key-portion)
- [🏆 Benchmark Evaluation & Scorecards](#-benchmark-evaluation--scorecards)
- [🖼️ Sample Evaluation Reports](#️-sample-evaluation-reports)
- [📁 Repository Structure](#-repository-structure)
- [🤝 Contributing](#-contributing)
- [📜 License](#-license)

---

## ✨ Key Features

- **Realistic Multi-Session Corpus**: 60 multi-turn user/assistant conversational sessions comprising **458 turns and ~540,878 tokens**.
- **Framework Agnostic**: Simple 2-method SDK interface (`ingest_turn` and `retrieve_context`) compatible with any RAG, Graph, or Agent memory system (Mem0, Zep, MemGPT, LangChain, LlamaIndex, Pinecone, etc.).
- **10 Core Evaluation Categories**: Evaluates recall, multi-hop links, temporal reasoning, fact overwrites, speaker traps, refusal, credibility, and agentic tool usage.
- **Dynamic Answer Key Compilation Portion**: Compiles dynamic key bindings (`harvested_output_advice_keys`, `fact_correction_keys`, `tool_learning_keys`) into the master ground-truth answer key.
- **Category 10 Agentic Tool-Use Suite**: Evaluates whether retrieved memory enables an agent to choose sanctioned tools, invoke them in exact order, and use tool returns (`fp_amb/agentic_eval.py`).
- **Dual Evaluation Modes**: Supports ultra-fast pure retrieval evaluation (zero-LLM latency, <60s) as well as full LLM generation accuracy checks.
- **Rich Visual Scorecards**: Generates detailed JSON, interactive visual HTML dashboards, and terminal-ready Markdown/ASCII reports stored in `results/`.

---

## ⚡ Quick Start & Installation

### 1. Clone & Install Dependencies
```bash
# Clone repository
git clone https://github.com/fp-amb/fp-amb-benchmark.git
cd fp-amb-benchmark

# Install package in editable mode
pip install -e .
```

### 2. Run Pure Retrieval Evaluation (< 60s)
Evaluate your memory provider using the FP-AMB CLI:

```bash
# Evaluate Sample Baseline Memory Provider
python -m fp_amb evaluate --provider examples/sample_memory_provider.py
```

### 3. Run Category 10 Agentic Tool-Use Evaluation
```bash
# Evaluate using any model (e.g. llama3, qwen2.5, mistral, llama3.1, or custom local model)
python -m fp_amb.agentic_eval --provider examples/sample_memory_provider.py --model llama3
```

---

## 🔌 Implementing Custom Memory Providers

To benchmark your custom memory engine, subclass `BaseMemoryProvider` and implement `ingest_turn` and `retrieve_context`:

```python
from fp_amb import BaseMemoryProvider, FPAMBEvaluator

class MyCustomMemoryEngine(BaseMemoryProvider):
    def ingest_turn(self, session_id: str, timestamp: str, speaker: str, text: str):
        """Ingest a conversation turn into your memory index or graph store."""
        my_memory_store.add(session_id=session_id, timestamp=timestamp, speaker=speaker, content=text)

    def retrieve_context(self, query: str, top_k: int = 5) -> str:
        """Retrieve relevant context string for the benchmark query."""
        results = my_memory_store.search(query, top_k=top_k)
        return "\n".join(results)

if __name__ == "__main__":
    # Execute benchmark exam
    evaluator = FPAMBEvaluator(
        provider=MyCustomMemoryEngine(),
        provider_name="MyCustomMemoryEngine"
    )
    evaluator.evaluate()
```

---

## 💻 CLI Usage

The FP-AMB CLI allows you to evaluate any provider script from the command line:

```bash
fp-amb evaluate --help
```

**Options:**
- `--provider PATH`: Path to Python script implementing `BaseMemoryProvider` *(required)*.
- `--llm`: Enable full LLM generation evaluation instead of pure retrieval.
- `--model NAME`: Model name for evaluation (e.g. `llama3`, `qwen2.5`, `mistral`, `llama3.1`; default: `llama3` or `FP_AMB_MODEL` env var).
- `--ollama-url URL`: Local/remote LLM API endpoint for Ollama, llama.cpp, vLLM, LM Studio, etc. (default: `http://localhost:11434/api/generate` or `FP_AMB_OLLAMA_URL` env var).
- `--output PATH`: Save scorecard to custom JSON filepath.

---

## 📊 Evaluation Categories (10 Categories + Dynamic Key Portion)

1. **Category 1: Single-Hop Fact Recall** *(35 items)*: Single-session direct fact retrieval.
2. **Category 2: Cross-Session Multi-Hop Reasoning** *(45 items)*: Linking multi-session entities across graph edges.
3. **Category 3: Temporal Reasoning & Session Math** *(35 items)*: Timestamp deltas, timeline math, and date resolution.
4. **Category 4: Adaptability & Fact Correction Overwrites** *(35 items)*: Tracking dynamic preference updates and fact revisions over time.
5. **Category 5: Self-Referential & Procedural Tool Memory** *(37 items)*: Recalling procedural tool rules and interaction instructions.
6. **Category 6: Adversarial Defense & Gaslighting Robustness** *(41 items)*: Resisting false premises and user gaslighting traps.
7. **Category 7: Speaker Attribution Traps** *(15 items)*: Disambiguating User vs. Assistant assertions (*LoCoMO & BEAM feature*).
8. **Category 8: Unanswerable & Absent Memory Refusal** *(35 items)*: Correctly refusing queries for non-existent memories (*LongMemEval feature*).
9. **Category 9: Source Credibility & Conflict Resolution** *(3 items)*: Disambiguating conflicting multi-source assertions and credibility weighting.
10. **Category 10: Agentic Tool-Use & Execution Order Evaluation**: Multi-turn tool execution loop checking tool selection, invocation order, and data payload utilization (`fp_amb/agentic_eval.py`).

### 🔑 Dynamic Answer Key Portion (`data/dynamic_answer_keys.json`)
Compiles dynamic key sections into the master answer key mapping:
* **Assistant Advice Keys**: Ingested turns containing dynamic assistant guidance and workflow advice.
* **Fact Correction Keys**: Dynamic state updates and preference overwrites compiled at runtime.
* **Tool Learning Keys**: Procedural tool usage rules and dynamic parameter bindings.

---

## 🏆 Benchmark Evaluation & Scorecards

> [!NOTE]
> Benchmark evaluation scorecards (JSON data, visual HTML dashboards, and Markdown reports) are generated dynamically for each provider run and output into the `results/` directory.

To run the benchmark battery and generate fresh scorecards for any memory provider:

```bash
# Benchmark a memory provider script
python -m fp_amb evaluate --provider examples/sample_memory_provider.py
```

---

## 🖼️ Sample Evaluation Reports & Automated Failure Analysis

FP-AMB automatically exports interactive HTML dashboards, terminal-ready Markdown scorecards, and a dedicated **Misses & Failure Taxonomy Text Report** (`*_misses.txt`) after every evaluation run:

### 1. Interactive Visual HTML Dashboard
![FP-AMB Visual HTML Exam Report](assets/html_report_sample.png)

### 2. Markdown & Terminal Scorecard Report
![FP-AMB Markdown Scorecard Report](assets/markdown_report_sample.png)

### 3. Automated Misses & Root-Cause Failure Report (`*_misses.txt`)
Saved automatically alongside each scorecard run to analyze and debug provider performance. It categorizes failed evaluation items first by **Question Category Type**, then by **Root Cause of Miss**:
- **`RETRIEVAL_FAILURE_FACT_NOT_IN_MEMORY_INJECTION`**: The ground-truth fact never appeared in the retrieved memory context payload.
- **`GENERATION_FAILURE_LLM_OUTPUT_MISSED`**: The ground-truth fact was present in retrieved context, but the LLM generation output failed to produce it.
- **`FALSE_RETRIEVAL_DISTRACTOR_TRAP`**: The provider retrieved distractor memory context for an unanswerable refusal query.
- **`FORMAT_OR_EXACT_KEYWORD_MISMATCH`**: Ground-truth fact was in context, but exact keyword matching rules failed.

---

## 📈 Comprehensive Token & Payload Metrics

FP-AMB tracks detailed context payload efficiency to prevent memory providers from "cheating" by injecting excessive context:
- **Total Corpus Size**: Total tokens (~540k) across 458 conversation turns.
- **Total Injected Context Payload**: Cumulative tokens injected across all queries in the battery.
- **Avg & Range Payload Size**: Average tokens/chars per query payload, along with minimum and maximum payload boundaries.
- **Token Efficiency Ratio**: $\text{Accuracy \%} / (\text{Avg Payload Tokens} / 1000)$, measuring accuracy points achieved per 1,000 injected tokens.

---

## 📁 Repository Structure

```
fp-amb-benchmark/
├── assets/                               # Sample report screenshots for documentation
│   ├── html_report_sample.png
│   └── markdown_report_sample.png
├── data/                                 # Corpus datasets & compiled ground-truth answer keys
│   ├── fp_amb_500k_cross_session.jsonl   # 60 sessions, 458 turns (~540k tokens)
│   ├── fp_amb_cross_session_questions.json # 281 static evaluation questions
│   ├── dynamic_answer_keys.json          # Dynamic keys (advice, fact corrections, tool learning)
│   └── master_ground_truth_answer_key.json # Master answer key (281 items + dynamic bindings)
├── examples/                             # Provider template & baseline examples
│   └── sample_memory_provider.py         # Pure Python baseline provider template
├── fp_amb/                               # Core FP-AMB Benchmark Engine
│   ├── __init__.py                       # Package exports (BaseMemoryProvider, FPAMBEvaluator)
│   ├── __main__.py                       # python -m fp_amb entry point
│   ├── agentic_eval.py                   # Category 10 Agentic tool-use evaluation engine
│   ├── cli.py                            # CLI command interface
│   ├── compile_master_answer_key.py      # Master answer key compiler utility
│   ├── dataset.py                        # Corpus & ground-truth data loader
│   ├── evaluator.py                      # Main evaluation runner (Categories 1-9)
│   ├── graph_adapter.py                  # Graph memory adapter utilities
│   ├── harvest.py                        # Metric harvesting & dynamic key compiler
│   ├── interface.py                      # BaseMemoryProvider abstract interface
│   ├── report.py                         # HTML & Markdown report generators
│   └── sdk.py                            # Framework SDK & plug-and-play evaluator
├── .github/                              # GitHub Issue and PR templates
│   ├── ISSUE_TEMPLATE/
│   └── PULL_REQUEST_TEMPLATE.md
├── results/                              # Output scorecards (JSON, HTML, MD) — gitignored
├── .gitignore                            # Standard Python & results exclusion rules
├── pyproject.toml                        # Package configuration (PEP 621)
├── CONTRIBUTING.md                       # Contribution guidelines
├── LICENSE                               # MIT License
└── README.md                             # Project documentation
```

---

## 🤝 Contributing

Contributions are welcome! Please check out [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to add memory providers, report bugs, or submit pull requests.

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).
