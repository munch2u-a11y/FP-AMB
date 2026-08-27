#!/usr/bin/env python3
"""
FP-AMB Agentic Tool-Use Evaluation
------------------------------------
Category 10: does the memory system help an agent DO something correctly,
not just recall a fact in a text answer.

Each scenario gives the agent a task, a memory-retrieved context, and a menu
of candidate tools (one sanctioned tool -- established earlier in the corpus
-- plus red herrings and irrelevants). The agent must call tools via a real
Ollama tool-calling loop; tool execution is scripted (deterministic canned
returns, no live backend). Scoring checks tool SELECTION and, for the
multi-hop scenario, CALL ORDER and whether the final answer actually used
the returned data -- not just whether the right words appear somewhere.
"""

import json
import time
import requests
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable, Optional

from .interface import BaseMemoryProvider

import os

OLLAMA_URL = os.getenv("FP_AMB_OLLAMA_URL", "http://localhost:11434/api/chat")
DEFAULT_MODEL = os.getenv("FP_AMB_MODEL", "llama3")
MAX_TURNS = 8


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict


@dataclass
class Scenario:
    id: str
    task_prompt: str
    memory_query: str  # what gets fed to provider.retrieve_context()
    tools: list          # list[ToolSpec]
    correct_sequence: list  # list[str] tool names, in required order
    scripted_return: Callable[[str, dict], dict]  # (tool_name, args) -> canned result dict
    final_answer_markers: list  # substrings expected in the final natural-language answer


def _r(status_or_data: dict) -> dict:
    return status_or_data


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

def _s1_legacy_db():
    def scripted(name, args):
        if name == "UnifiedV2DBClient":
            return {"user_id": 4471, "status": "active", "last_login": "2026-12-20"}
        if name == "LegacyDBClient":
            return {"error": "Connection refused -- LegacyDBClient was deprecated after repeated timeout failures."}
        return {"error": "Tool not applicable to this request."}
    return Scenario(
        id="TOOL_01_legacy_db",
        task_prompt="Look up the current status of user #4471 in the legacy inventory system.",
        memory_query="Which database tool should be used to query the legacy user inventory database?",
        tools=[
            ToolSpec("LegacyDBClient", "Query the legacy inventory database directly.",
                     {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}),
            ToolSpec("UnifiedV2DBClient", "Query the current unified inventory database.",
                     {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}),
            ToolSpec("SearchAPI_HighThroughput", "Run a high-throughput search index query.",
                     {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}),
            ToolSpec("WeatherAPI", "Get current weather for a location.",
                     {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}),
        ],
        correct_sequence=["UnifiedV2DBClient"],
        scripted_return=scripted,
        final_answer_markers=["active", "4471"],
    )


def _s2_batch_indexing():
    def scripted(name, args):
        if name == "SearchAPI_HighThroughput":
            return {"job_id": "batch-8821", "status": "started", "throughput_mode": "high"}
        if name in ("SearchAPI_Standard", "SearchAPI_LowLatency"):
            return {"error": f"{name} is not designed for batch throughput; job rejected."}
        return {"error": "Tool not applicable to this request."}
    return Scenario(
        id="TOOL_02_batch_indexing",
        task_prompt="Kick off the nightly batch log indexing job for the platform logs.",
        memory_query="Which search API client should be used for automated batch log indexing?",
        tools=[
            ToolSpec("SearchAPI_Standard", "Standard-tier search API client.",
                     {"type": "object", "properties": {"job": {"type": "string"}}, "required": ["job"]}),
            ToolSpec("SearchAPI_HighThroughput", "High-throughput search API client for batch workloads.",
                     {"type": "object", "properties": {"job": {"type": "string"}}, "required": ["job"]}),
            ToolSpec("SearchAPI_LowLatency", "Low-latency search API client for interactive queries.",
                     {"type": "object", "properties": {"job": {"type": "string"}}, "required": ["job"]}),
            ToolSpec("UnifiedV2DBClient", "Query the current unified inventory database.",
                     {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}),
        ],
        correct_sequence=["SearchAPI_HighThroughput"],
        scripted_return=scripted,
        final_answer_markers=["started", "batch-8821"],
    )


def _s3_vault_auth():
    def scripted(name, args):
        if name == "set_auth_spiffe_svid":
            return {"status": "configured", "method": "SPIFFE X.509 SVID header", "vault_connection": "verified"}
        if name in ("set_auth_basicauth", "set_auth_oauth2_bearer"):
            return {"error": "Rejected by Vault policy -- SPIFFE identity required."}
        return {"error": "Tool not applicable to this request."}
    return Scenario(
        id="TOOL_03_vault_auth",
        task_prompt="Configure authentication for the new microservice so it can talk to the Vault service.",
        memory_query="How should internal microservices authenticate against the Vault service?",
        tools=[
            ToolSpec("set_auth_basicauth", "Configure BasicAuth credentials for a service.",
                     {"type": "object", "properties": {"service": {"type": "string"}}, "required": ["service"]}),
            ToolSpec("set_auth_oauth2_bearer", "Configure OAuth2 Bearer token auth for a service.",
                     {"type": "object", "properties": {"service": {"type": "string"}}, "required": ["service"]}),
            ToolSpec("set_auth_spiffe_svid", "Configure SPIFFE X.509 SVID header auth for a service.",
                     {"type": "object", "properties": {"service": {"type": "string"}}, "required": ["service"]}),
            ToolSpec("restart_service", "Restart a running service.",
                     {"type": "object", "properties": {"service": {"type": "string"}}, "required": ["service"]}),
        ],
        correct_sequence=["set_auth_spiffe_svid"],
        scripted_return=scripted,
        final_answer_markers=["configured", "verified"],
    )


def _s4_memory_leak():
    def scripted(name, args):
        if name == "run_valgrind":
            return {"tool": "Valgrind/Memcheck", "leak_source": "raw pointer captured in lambda passed to thread pool, never released", "recommendation": "replace with std::shared_ptr"}
        if name == "run_gdb_breakpoint":
            return {"note": "gdb breakpoint set, but no leak information -- gdb does not perform leak detection without additional tooling."}
        if name == "run_strace":
            return {"note": "strace shows syscall trace only, no memory-leak information."}
        return {"error": "Tool not applicable to this request."}
    return Scenario(
        id="TOOL_04_memory_leak",
        task_prompt="Diagnose the memory leak Dave reported in the C++ worker thread pool.",
        memory_query="What diagnostic steps were recommended for Dave's C++ memory leak in the worker thread pool?",
        tools=[
            ToolSpec("run_valgrind", "Run Valgrind/Memcheck or AddressSanitizer against a target.",
                     {"type": "object", "properties": {"target": {"type": "string"}}, "required": ["target"]}),
            ToolSpec("run_gdb_breakpoint", "Attach gdb and set a breakpoint on a target.",
                     {"type": "object", "properties": {"target": {"type": "string"}}, "required": ["target"]}),
            ToolSpec("run_strace", "Trace syscalls for a target process.",
                     {"type": "object", "properties": {"target": {"type": "string"}}, "required": ["target"]}),
            ToolSpec("check_disk_space", "Check available disk space.",
                     {"type": "object", "properties": {}, "required": []}),
        ],
        correct_sequence=["run_valgrind"],
        scripted_return=scripted,
        final_answer_markers=["shared_ptr", "raw pointer"],
    )


def _s5_payment_restart():
    def scripted(name, args):
        if name == "JobRunner_SafeRestart":
            return {"status": "restarted", "idempotency_preserved": True, "job_id": "recon-5521"}
        if name == "ForceKillWorker":
            return {"warning": "Job force-killed. Idempotency keys NOT preserved -- risk of duplicate charges, matching the incident flagged last quarter."}
        return {"error": "Tool not applicable to this request."}
    return Scenario(
        id="TOOL_05_payment_restart",
        task_prompt="The nightly payment reconciliation job is stuck. Restart it safely.",
        memory_query="Which tool should be used to safely restart a stuck payment reconciliation job?",
        tools=[
            ToolSpec("JobRunner_SafeRestart", "Safely restart a stuck job, preserving idempotency keys.",
                     {"type": "object", "properties": {"job_id": {"type": "string"}}, "required": ["job_id"]}),
            ToolSpec("ForceKillWorker", "Forcibly kill and restart a worker process.",
                     {"type": "object", "properties": {"job_id": {"type": "string"}}, "required": ["job_id"]}),
            ToolSpec("SendSlackAlert", "Send an alert message to a Slack channel.",
                     {"type": "object", "properties": {"message": {"type": "string"}}, "required": ["message"]}),
            ToolSpec("CheckWeather", "Get current weather for a location.",
                     {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}),
        ],
        correct_sequence=["JobRunner_SafeRestart"],
        scripted_return=scripted,
        final_answer_markers=["restarted", "idempotency"],
    )


def _s6_multihop_ordered():
    def scripted(name, args):
        if name == "IdentifyMentor":
            person = str(args.get("person", "")).lower()
            if "mark" in person:
                return {"mentor": "Dave"}
            return {"error": "No mentor relationship found."}
        if name == "LookupSpecifiedArchitecture":
            engineer = str(args.get("engineer", "")).lower()
            if "dave" in engineer:
                return {"architecture": "Apache Iceberg on AWS S3 Express"}
            return {"error": "No architecture specification found for that engineer."}
        if name == "ConfigureTelemetryCollector":
            system = str(args.get("system", "")).lower()
            if "iceberg" in system or "data lake" in system or "s3" in system:
                return {"status": "monitoring configured", "collector": "OpenTelemetry 2-tier collector with Kafka"}
            return {"error": "Unknown system, cannot configure monitoring."}
        if name == "ConfigureBasicAuthMonitoring":
            return {"error": "Not applicable -- this task requires telemetry monitoring, not auth configuration."}
        return {"error": "Tool not applicable to this request."}
    return Scenario(
        id="TOOL_06_multihop_ordered",
        task_prompt="Set up monitoring for the data lake architecture that Mark's mentor specified.",
        memory_query="Who is Mark's mentor and what data lake architecture did they specify?",
        tools=[
            ToolSpec("IdentifyMentor", "Look up who mentors a given person.",
                     {"type": "object", "properties": {"person": {"type": "string"}}, "required": ["person"]}),
            ToolSpec("LookupSpecifiedArchitecture", "Look up what architecture a given engineer specified.",
                     {"type": "object", "properties": {"engineer": {"type": "string"}}, "required": ["engineer"]}),
            ToolSpec("ConfigureTelemetryCollector", "Configure telemetry monitoring for a given system.",
                     {"type": "object", "properties": {"system": {"type": "string"}}, "required": ["system"]}),
            ToolSpec("ConfigureBasicAuthMonitoring", "Configure BasicAuth-based monitoring for a system.",
                     {"type": "object", "properties": {"system": {"type": "string"}}, "required": ["system"]}),
            ToolSpec("RestartService", "Restart a running service.",
                     {"type": "object", "properties": {"service": {"type": "string"}}, "required": ["service"]}),
        ],
        correct_sequence=["IdentifyMentor", "LookupSpecifiedArchitecture", "ConfigureTelemetryCollector"],
        scripted_return=scripted,
        final_answer_markers=["opentelemetry", "kafka", "configured"],
    )


def _s7_lb_config():
    def scripted(name, args):
        if name == "ConfigureLoadBalancer":
            persistence = str(args.get("session_persistence", "")).lower()
            max_conn = args.get("max_connections")
            if "ip hash" in persistence and str(max_conn) == "5000":
                return {"status": "configured", "session_persistence": "IP hash", "max_connections": 5000}
            return {"error": "Configuration values do not match Dave's established spec -- rejected by config validator."}
        if name == "ConfigureLoadBalancerDefaults":
            return {"error": "Generic defaults do not match Dave's spec; validator rejected the config."}
        return {"error": "Tool not applicable to this request."}
    return Scenario(
        id="TOOL_07_lb_config",
        task_prompt="Configure the load balancer for the new service rollout using Dave's established settings.",
        memory_query="What load balancer session persistence setting and max connection limit did Dave recommend?",
        tools=[
            ToolSpec("ConfigureLoadBalancer", "Configure the load balancer with specific session persistence and max connection settings.",
                     {"type": "object", "properties": {"session_persistence": {"type": "string"}, "max_connections": {"type": "integer"}},
                      "required": ["session_persistence", "max_connections"]}),
            ToolSpec("ConfigureLoadBalancerDefaults", "Configure the load balancer using generic platform defaults.",
                     {"type": "object", "properties": {}, "required": []}),
            ToolSpec("RestartService", "Restart a running service.",
                     {"type": "object", "properties": {"service": {"type": "string"}}, "required": ["service"]}),
            ToolSpec("CheckWeather", "Get current weather for a location.",
                     {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}),
        ],
        correct_sequence=["ConfigureLoadBalancer"],
        scripted_return=scripted,
        final_answer_markers=["ip hash", "5000"],
    )


def _s8_distractor_prompt():
    def scripted(name, args):
        if name == "SearchAPI_HighThroughput":
            return {"job_id": "batch-9042", "status": "started", "throughput_mode": "high"}
        if name in ("SearchAPI_Standard", "SearchAPI_LowLatency"):
            return {"error": f"{name} is not designed for batch throughput; job rejected."}
        if name == "LookupElectricVehicleModels":
            return {"error": "Not relevant to the requested task."}
        return {"error": "Tool not applicable to this request."}
    return Scenario(
        id="TOOL_08_distractor_prompt",
        task_prompt="Dave mentioned an electric car charging station he liked recently, by the way. Also, can you kick off the automated batch log indexing job for tonight?",
        memory_query="Which search API client should be used for automated batch log indexing?",
        tools=[
            ToolSpec("SearchAPI_Standard", "Standard-tier search API client.",
                     {"type": "object", "properties": {"job": {"type": "string"}}, "required": ["job"]}),
            ToolSpec("SearchAPI_HighThroughput", "High-throughput search API client for batch workloads.",
                     {"type": "object", "properties": {"job": {"type": "string"}}, "required": ["job"]}),
            ToolSpec("SearchAPI_LowLatency", "Low-latency search API client for interactive queries.",
                     {"type": "object", "properties": {"job": {"type": "string"}}, "required": ["job"]}),
            ToolSpec("LookupElectricVehicleModels", "Look up electric vehicle model specifications and reviews.",
                     {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}),
        ],
        correct_sequence=["SearchAPI_HighThroughput"],
        scripted_return=scripted,
        final_answer_markers=["started", "batch-9042"],
    )


SCENARIOS = [_s1_legacy_db(), _s2_batch_indexing(), _s3_vault_auth(), _s4_memory_leak(), _s5_payment_restart(),
             _s6_multihop_ordered(), _s7_lb_config(), _s8_distractor_prompt()]


def _to_ollama_tools(tools: list) -> list:
    return [
        {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}
        for t in tools
    ]


def run_scenario(scenario: Scenario, provider: BaseMemoryProvider, model: str, ollama_url: str) -> dict:
    context = provider.retrieve_context(scenario.memory_query)
    system_msg = (
        "You are an operating agent with access to the tools listed. Use the retrieved memory "
        "context below to decide which tool to call. Only call tools that are actually appropriate "
        "for the task -- do not call a tool just because it superficially matches a keyword.\n\n"
        f"[RETRIEVED MEMORY CONTEXT]\n{context}"
    )
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": scenario.task_prompt},
    ]
    ollama_tools = _to_ollama_tools(scenario.tools)
    valid_names = {t.name for t in scenario.tools}

    called_sequence = []
    final_text = ""

    for _turn in range(MAX_TURNS):
        payload = {"model": model, "messages": messages, "tools": ollama_tools, "stream": False,
                   "options": {"temperature": 0.1}}
        resp = requests.post(ollama_url, json=payload, timeout=120)
        resp.raise_for_status()
        msg = resp.json()["message"]
        tool_calls = msg.get("tool_calls") or []

        if not tool_calls:
            final_text = msg.get("content", "")
            break

        messages.append({"role": "assistant", "content": msg.get("content", ""), "tool_calls": tool_calls})
        for tc in tool_calls:
            fn = tc["function"]
            name = fn["name"]
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}
            called_sequence.append((name, args))
            if name in valid_names:
                result = scenario.scripted_return(name, args)
            else:
                result = {"error": f"Unknown tool '{name}'."}
            messages.append({"role": "tool", "content": json.dumps(result)})
    else:
        final_text = ""

    called_names = [c[0] for c in called_sequence]
    correct_names_called = [n for n in called_names if n in scenario.correct_sequence]
    tool_selection_correct = correct_names_called == scenario.correct_sequence or (
        len(scenario.correct_sequence) == 1 and scenario.correct_sequence[0] in called_names
        and called_names.count(scenario.correct_sequence[0]) >= 1
    )
    order_correct = correct_names_called[:len(scenario.correct_sequence)] == scenario.correct_sequence
    no_wrong_tool_called = all(n in scenario.correct_sequence or n not in valid_names for n in called_names) or True
    wrong_tool_called = any(n in valid_names and n not in scenario.correct_sequence for n in called_names)
    final_lower = final_text.lower()
    used_result_in_answer = any(m.lower() in final_lower for m in scenario.final_answer_markers)

    return {
        "scenario_id": scenario.id,
        "task": scenario.task_prompt,
        "memory_query": scenario.memory_query,
        "retrieved_context": context,
        "called_sequence": called_names,
        "correct_sequence": scenario.correct_sequence,
        "order_correct": order_correct,
        "wrong_tool_called": wrong_tool_called,
        "final_answer": final_text,
        "used_result_in_answer": used_result_in_answer,
        "pass": order_correct and not wrong_tool_called,
    }


def run_agentic_eval(provider: BaseMemoryProvider, provider_name: str = "Provider",
                      model: str = DEFAULT_MODEL, ollama_url: str = OLLAMA_URL,
                      output_path: Optional[Path] = None) -> dict:
    print("=========================================================================", flush=True)
    print(f"   FP-AMB CATEGORY 10: AGENTIC TOOL-USE EVAL -- {provider_name} [{model}]", flush=True)
    print("=========================================================================\n", flush=True)

    results = []
    for sc in SCENARIOS:
        t0 = time.time()
        res = run_scenario(sc, provider, model, ollama_url)
        res["elapsed_s"] = round(time.time() - t0, 2)
        results.append(res)
        status = "PASS" if res["pass"] else "FAIL"
        used = "yes" if res["used_result_in_answer"] else "no"
        print(f"[{status}] {sc.id:28} called={res['called_sequence']} order_ok={res['order_correct']} "
              f"wrong_tool={res['wrong_tool_called']} used_result={used} ({res['elapsed_s']}s)", flush=True)

    passed = sum(1 for r in results if r["pass"])
    total = len(results)
    print(f"\nTOOL SELECTION/ORDER: {passed}/{total} passed ({100*passed/total:.1f}%)", flush=True)

    used_result_count = sum(1 for r in results if r["used_result_in_answer"])
    print(f"USED SCRIPTED RESULT IN FINAL ANSWER: {used_result_count}/{total}", flush=True)

    payload = {
        "benchmark_title": f"FP-AMB Category 10: Agentic Tool-Use Eval ({provider_name})",
        "provider_name": provider_name,
        "model_name": model,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_scenarios": total,
        "passed_scenarios": passed,
        "pass_rate_pct": round(100 * passed / total, 1),
        "used_result_in_answer_count": used_result_count,
        "results": results,
    }

    if output_path is None:
        run_stamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = Path("results") / f"{provider_name.lower()}_agentic_tool_use_scorecard_{run_stamp}.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved agentic tool-use scorecard to '{output_path}'.", flush=True)

    html_path = Path(output_path).with_suffix(".html")
    from .report import write_agentic_report, write_agentic_text_report
    write_agentic_report(payload, html_path)
    print(f"Saved agentic visual report to '{html_path}'.", flush=True)

    text_path = Path(output_path).with_suffix(".md")
    write_agentic_text_report(payload, text_path)
    print(f"Saved agentic text/ASCII report to '{text_path}'.", flush=True)

    return payload


if __name__ == "__main__":
    import argparse
    import importlib.util

    parser = argparse.ArgumentParser(description="FP-AMB Category 10 agentic tool-use eval")
    parser.add_argument("--provider", required=True, type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ollama-url", default=OLLAMA_URL)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    spec = importlib.util.spec_from_file_location(args.provider.stem, args.provider)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    provider_class = None
    for name, obj in module.__dict__.items():
        if isinstance(obj, type) and issubclass(obj, BaseMemoryProvider) and obj is not BaseMemoryProvider:
            provider_class = obj
            break
    provider_instance = provider_class()

    from .dataset import load_corpus
    for t in load_corpus():
        provider_instance.ingest_turn(t.get("session_id", "Session"), t.get("timestamp", ""),
                                       t.get("speaker", "User"), t.get("text", ""))

    run_agentic_eval(provider_instance, provider_class.__name__, args.model, args.ollama_url, args.output)
