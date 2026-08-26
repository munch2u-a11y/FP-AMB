# Contributing to FP-AMB

Thank you for your interest in contributing to the **First-Person Agent Memory Benchmark (FP-AMB)**! We welcome contributions from the AI agent memory and RAG research communities, including new provider integrations, benchmark evaluation items, bug fixes, and documentation improvements.

---

## 🛠️ How to Contribute

### 1. Adding a New Memory Provider Integration
Evaluating your memory system (e.g. custom RAG, graph database, or agent memory framework) is straightforward:

1. Create a new provider module in `examples/your_provider_name.py`.
2. Inherit from `BaseMemoryProvider` (`from fp_amb import BaseMemoryProvider`).
3. Implement the required `ingest_turn` and `retrieve_context` methods:
   ```python
   from fp_amb import BaseMemoryProvider, FPAMBEvaluator

   class YourCustomMemoryProvider(BaseMemoryProvider):
       def ingest_turn(self, session_id: str, timestamp: str, speaker: str, text: str):
           # Store conversation turn in your memory store
           pass

       def retrieve_context(self, query: str, top_k: int = 5) -> str:
           # Query your memory store and return retrieved text context
           return ""
   ```
4. Verify your integration by running `FPAMBEvaluator(provider=YourCustomMemoryProvider()).evaluate()`.
5. Submit a Pull Request containing your provider class under `examples/`.

---

## 📋 Pull Request Guidelines

1. **Fork and Branch**: Create a feature branch off `master` (e.g., `git checkout -b feat/add-zep-provider`).
2. **Code Style**: Ensure Python code complies with standard PEP 8 conventions.
3. **No Generated Binary Files**: Do not commit `.pyc`, `__pycache__`, or local temporary cache directories.
4. **Descriptive Commit Messages**: Provide clear, standard commit titles and descriptions.

---

## 🐛 Reporting Bugs & Requesting Features

* **Bug Reports**: Open an issue describing the step-by-step reproduction, observed behavior, expected behavior, and environment details.
* **Feature Requests**: Describe the proposed addition, motivation, and potential integration approach.

---

## 📜 License
By contributing to FP-AMB, you agree that your contributions will be licensed under the project's [MIT License](LICENSE).
