# Provenance

Vendored copy of the "Fractal Memory" project (`/home/nemo/fractal_memory`) at its
task-695 baseline, for reproducible FP-AMB benchmarking against a pinned, reviewed
version rather than that project's live (uncommitted, in-flux) working tree.

- Source commit: `ac8c461e41c266a8bf8e88419a313b7efeda1024`
  ("Snapshot: task-695 baseline plus unreviewed v2 work")
- Extracted from: `revert_task695/` inside that commit (a self-contained snapshot of
  the reviewed pre-"v2" state), NOT the commit's root-level files, which its own
  commit message and `handover_summary.md` flag as "unreviewed v2 work... nothing
  here has been reviewed or accepted."
- `__init__.py`, `llm_client.py`, `vector_adapters.py` were unchanged between the
  task-695 baseline and the v2 rework, so those three came from the commit root
  rather than `revert_task695/` (which doesn't include them).
- There is also a `revert_candidates/` folder in the same commit with different,
  simpler (non-crystallization) versions of the core files -- not used here. It's a
  list of individual revert targets, not a complete self-contained snapshot, and its
  architecture doesn't match the "frequency-driven vault crystallization" design
  `handover_summary.md` describes as the system's actual intended design.

Extraction commands (run from `/home/nemo/fractal_memory`):
```
for f in DESIGN_BLUEPRINT_AND_GOALS.md benchmark_task468_results.json cluster_splitter.py \
         consolidation_engine.py entity_extractor.py fractal_ltm.py macro_graph.py \
         micro_list.py regrade_benchmark_real_retrieval.py test_consolidation_branch.py \
         test_ingestion_cluster_scan.py tester_benchmark.py; do
  git show ac8c461:revert_task695/$f > $f
done
for f in __init__.py llm_client.py vector_adapters.py requirements.txt; do
  git show ac8c461:$f > $f
done
```

No lines were edited after extraction.
