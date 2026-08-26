#!/usr/bin/env python3
"""
FP-AMB CLI Command-Line Interface
---------------------------------
Allows evaluating ANY external memory provider file against the FP-AMB Exam:
  python3 -m fp_amb evaluate --provider path/to/provider_script.py [--llm] [--model llama3]
"""

import argparse
import sys
import importlib.util
from pathlib import Path
from .interface import BaseMemoryProvider
from .evaluator import FPAMBEvaluator

def load_provider_from_file(file_path: Path) -> BaseMemoryProvider:
    if not file_path.exists():
        print(f"Error: Provider script file '{file_path}' does not exist.")
        sys.exit(1)

    module_name = file_path.stem
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        print(f"Error: Could not load python module from '{file_path}'.")
        sys.exit(1)

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Find subclass of BaseMemoryProvider inside the file
    provider_class = None
    for name, obj in module.__dict__.items():
        if isinstance(obj, type) and issubclass(obj, BaseMemoryProvider) and obj is not BaseMemoryProvider:
            provider_class = obj
            break

    if provider_class is None:
        print(f"Error: No class inheriting from BaseMemoryProvider was found in '{file_path}'.")
        sys.exit(1)

    print(f"Loaded memory provider class '{provider_class.__name__}' from '{file_path}'.")
    return provider_class()

import os

def main():
    parser = argparse.ArgumentParser(description="FP-AMB First-Person Agent Memory Benchmark Exam CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    eval_parser = subparsers.add_parser("evaluate", help="Run FP-AMB Exam against a memory provider script")
    eval_parser.add_argument("--provider", required=True, type=Path, help="Path to Python script implementing BaseMemoryProvider")
    eval_parser.add_argument("--llm", action="store_true", help="Run full LLM generation instead of pure retrieval")
    eval_parser.add_argument("--model", default=os.getenv("FP_AMB_MODEL", "llama3"), help="Model name for generation evaluation (default: llama3 or FP_AMB_MODEL env var)")
    eval_parser.add_argument("--ollama-url", default=os.getenv("FP_AMB_OLLAMA_URL", "http://localhost:11434/api/generate"), help="LLM API endpoint URL (Ollama, llama.cpp, vLLM, LM Studio, etc.)")
    eval_parser.add_argument("--output", type=Path, help="Output scorecard JSON file path")

    args = parser.parse_args()

    if args.command == "evaluate":
        provider_instance = load_provider_from_file(args.provider)
        evaluator = FPAMBEvaluator(
            provider=provider_instance,
            provider_name=provider_instance.__class__.__name__,
            model_name=args.model,
            ollama_url=args.ollama_url,
            use_llm_generation=args.llm
        )
        evaluator.evaluate(output_path=args.output)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
