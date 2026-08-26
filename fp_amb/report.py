#!/usr/bin/env python3
"""
FP-AMB Visual Exam Report Generator
------------------------------------
Renders a self-contained HTML scorecard (hero stat, KPI row, category
breakdown chart) from an evaluator scorecard dict. No external assets —
opens offline from disk, works in light and dark mode.
"""

import html
import json
from pathlib import Path

# Reference palette (dataviz skill default) — unchanged values.
BLUE_LIGHT = "#2a78d6"
BLUE_DARK = "#3987e5"
STATUS_GOOD = "#0ca30c"
STATUS_WARNING = "#fab219"
STATUS_CRITICAL = "#d03b3b"

CATEGORY_SHORT = {
    "Single-Hop Fact Recall": "Single-Hop Fact Recall",
    "Cross-Session Multi-Hop Reasoning": "Cross-Session Multi-Hop",
    "Temporal Reasoning & Session Math": "Temporal Reasoning",
    "Adaptability & Fact Correction Overwrites": "Adaptability & Overwrites",
    "Self-Referential & Procedural Tool Memory": "Self-Referential Memory",
    "Adversarial Defense & Gaslighting Robustness": "Adversarial Defense",
    "Speaker Attribution Traps": "Speaker Attribution",
    "Unanswerable & Absent Memory Refusal": "Unanswerable Refusal",
}


_BASE_STYLE = """
  :root {
    color-scheme: light;
    --surface-1:      #fcfcfb;
    --page-plane:     #f9f9f7;
    --text-primary:   #0b0b0b;
    --text-secondary: #52514e;
    --muted:          #898781;
    --gridline:       #e1e0d9;
    --border:         rgba(11,11,11,0.10);
    --track:          #eeede8;
  }
  @media (prefers-color-scheme: dark) {
    :root:where(:not([data-theme="light"])) {
      color-scheme: dark;
      --surface-1:      #1a1a19;
      --page-plane:     #0d0d0d;
      --text-primary:   #ffffff;
      --text-secondary: #c3c2b7;
      --muted:          #898781;
      --gridline:       #2c2c2a;
      --border:         rgba(255,255,255,0.10);
      --track:          #26261f;
    }
  }
  :root[data-theme="dark"] {
    color-scheme: dark;
    --surface-1:      #1a1a19;
    --page-plane:     #0d0d0d;
    --text-primary:   #ffffff;
    --text-secondary: #c3c2b7;
    --muted:          #898781;
    --gridline:       #2c2c2a;
    --border:         rgba(255,255,255,0.10);
    --track:          #26261f;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    padding: 32px 20px 60px;
    background: var(--page-plane);
    color: var(--text-primary);
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  }
  .wrap { max-width: 880px; margin: 0 auto; }
  header { margin-bottom: 28px; }
  h1 { font-size: 22px; margin: 0 0 4px; }
  .meta { color: var(--text-secondary); font-size: 13px; }
  .card {
    background: var(--surface-1);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 20px;
  }
  .hero { display: flex; align-items: baseline; gap: 16px; flex-wrap: wrap; }
  .hero-value { font-size: 56px; font-weight: 700; line-height: 1; }
  .hero-sub { color: var(--text-secondary); font-size: 15px; }
  .tiles {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 12px;
  }
  .tile {
    background: var(--surface-1);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 16px;
  }
  .tile-label { font-size: 12px; color: var(--muted); margin-bottom: 6px; }
  .tile-value { font-size: 20px; font-weight: 600; font-variant-numeric: tabular-nums; }
  .tile-sub { font-size: 11px; color: var(--muted); margin-top: 2px; }
  h2 { font-size: 15px; margin: 0 0 16px; color: var(--text-secondary); font-weight: 600; }
  .cat-label { fill: var(--text-secondary); font-size: 13px; }
  .track { fill: var(--track); }
  .val-label { fill: var(--text-primary); font-size: 13px; font-variant-numeric: tabular-nums; }
  .val-sub { fill: var(--muted); }
  footer { color: var(--muted); font-size: 12px; margin-top: 24px; }
"""


def _status_color(pct: float) -> str:
    if pct >= 80:
        return STATUS_GOOD
    if pct >= 50:
        return STATUS_WARNING
    return STATUS_CRITICAL


def _bar_chart_svg(category_breakdown: dict) -> str:
    rows = []
    for cat, s in category_breakdown.items():
        pct = (s["correct"] / s["total"] * 100) if s["total"] else 0.0
        rows.append((cat, s["correct"], s["total"], pct))
    rows.sort(key=lambda r: r[3])  # worst first

    row_h = 34
    gap = 10
    chart_h = len(rows) * (row_h + gap)
    label_w = 210
    track_w = 480
    svg_w = label_w + track_w + 70
    svg_h = chart_h + 10

    parts = [
        f'<svg viewBox="0 0 {svg_w} {svg_h}" width="100%" height="{svg_h}" '
        f'role="img" aria-label="Category accuracy breakdown">'
    ]

    for i, (cat, correct, total, pct) in enumerate(rows):
        y = i * (row_h + gap)
        bar_y = y + 4
        bar_h = row_h - 8
        max_track = track_w - 8
        bar_w = max(4, (pct / 100.0) * max_track)
        color = _status_color(pct)
        label = CATEGORY_SHORT.get(cat, cat)

        parts.append(f'<g>')
        parts.append(
            f'<text x="{label_w - 12}" y="{bar_y + bar_h/2 + 4}" '
            f'text-anchor="end" class="cat-label">{html.escape(label)}</text>'
        )
        # track (recessive baseline)
        parts.append(
            f'<rect x="{label_w}" y="{bar_y}" width="{max_track}" height="{bar_h}" '
            f'rx="4" class="track"/>'
        )
        # bar
        parts.append(
            f'<rect x="{label_w}" y="{bar_y}" width="{bar_w:.1f}" height="{bar_h}" '
            f'rx="4" fill="{color}">'
            f'<title>{html.escape(label)}: {correct}/{total} ({pct:.1f}%)</title>'
            f'</rect>'
        )
        # value label
        parts.append(
            f'<text x="{label_w + track_w + 10}" y="{bar_y + bar_h/2 + 4}" '
            f'class="val-label">{pct:.1f}% <tspan class="val-sub">({correct}/{total})</tspan></text>'
        )
        parts.append('</g>')

    parts.append('</svg>')
    return "".join(parts)


def _stat_tile(label: str, value: str, sub: str = "") -> str:
    sub_html = f'<div class="tile-sub">{html.escape(sub)}</div>' if sub else ""
    return (
        '<div class="tile">'
        f'<div class="tile-label">{html.escape(label)}</div>'
        f'<div class="tile-value">{html.escape(value)}</div>'
        f'{sub_html}'
        '</div>'
    )


def generate_html_report(scorecard: dict) -> str:
    provider = scorecard.get("provider_name", "Memory Provider")
    overall = scorecard.get("overall_accuracy_pct", 0.0)
    passed = scorecard.get("passed_items", 0)
    total = scorecard.get("total_evaluated_questions", 0)
    mode = "Full LLM Generation" if scorecard.get("use_llm_generation") else "Pure Retrieval (Zero-LLM)"
    model = scorecard.get("model_name")
    mode_label = f"{mode} — {model}" if model else mode
    timestamp = scorecard.get("timestamp", "")
    cat_breakdown = scorecard.get("category_breakdown", {})

    hero_color = _status_color(overall)
    bar_svg = _bar_chart_svg(cat_breakdown) if cat_breakdown else ""

    tiles = "".join([
        _stat_tile("Avg Retrieval Latency", f'{scorecard.get("avg_retrieval_latency_ms", 0):.2f} ms'),
        _stat_tile("Avg Injected Context", f'{scorecard.get("avg_injected_tokens_per_query", 0):.0f} tok'),
        _stat_tile("Token Efficiency", f'{scorecard.get("token_efficiency_ratio", 0):.2f}', "accuracy pts / 1k tok"),
        _stat_tile("Ingestion Time", f'{scorecard.get("ingestion_duration_seconds", 0):.2f} s'),
        _stat_tile("Exam Duration", f'{scorecard.get("eval_duration_seconds", 0):.1f} s'),
    ])

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>FP-AMB Report — {html.escape(provider)}</title>
<style>
{_BASE_STYLE}
  .hero-value {{ color: {hero_color}; }}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>FP-AMB Exam Report</h1>
    <div class="meta">{html.escape(provider)} &middot; {html.escape(mode_label)} &middot; {html.escape(timestamp)}</div>
  </header>

  <div class="card hero">
    <div class="hero-value">{overall:.1f}%</div>
    <div class="hero-sub">{passed} / {total} items passed</div>
  </div>

  <div class="card">
    <h2>Performance</h2>
    <div class="tiles">{tiles}</div>
  </div>

  <div class="card">
    <h2>Category Breakdown</h2>
    {bar_svg}
  </div>

  <footer>Generated by fp_amb.report &middot; {html.escape(timestamp)}</footer>
</div>
</body>
</html>"""


def write_report(scorecard: dict, output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.write_text(generate_html_report(scorecard))
    return output_path


def _scenario_rows_table(results: list) -> str:
    rows = []
    for r in results:
        status = "PASS" if r.get("pass") else "FAIL"
        color = STATUS_GOOD if r.get("pass") else STATUS_CRITICAL
        called = " → ".join(r.get("called_sequence", [])) or "(none)"
        wanted = " → ".join(r.get("correct_sequence", [])) or "(none)"
        order_ok = "✓" if r.get("order_correct") else "✗"
        wrong = "yes" if r.get("wrong_tool_called") else "no"
        used = "✓" if r.get("used_result_in_answer") else "✗"
        rows.append(
            "<tr>"
            f'<td><span class="status-dot" style="background:{color}"></span>{html.escape(status)}</td>'
            f'<td>{html.escape(r.get("scenario_id", ""))}</td>'
            f'<td class="mono">{html.escape(called)}</td>'
            f'<td class="mono">{html.escape(wanted)}</td>'
            f'<td class="center">{order_ok}</td>'
            f'<td class="center">{html.escape(wrong)}</td>'
            f'<td class="center">{used}</td>'
            f'<td class="center">{r.get("elapsed_s", 0):.1f}s</td>'
            "</tr>"
        )
    return "".join(rows)


def generate_agentic_html_report(scorecard: dict) -> str:
    provider = scorecard.get("provider_name", "Memory Provider")
    model = scorecard.get("model_name", "")
    passed = scorecard.get("passed_scenarios", 0)
    total = scorecard.get("total_scenarios", 0)
    pct = scorecard.get("pass_rate_pct", 0.0)
    used_count = scorecard.get("used_result_in_answer_count", 0)
    timestamp = scorecard.get("timestamp", "")
    results = scorecard.get("results", [])
    hero_color = _status_color(pct)

    tiles = "".join([
        _stat_tile("Scenarios Passed", f"{passed} / {total}"),
        _stat_tile("Used Scripted Result", f"{used_count} / {total}", "final answer referenced tool output"),
        _stat_tile("Model", model or "—"),
    ])

    table_rows = _scenario_rows_table(results)

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>FP-AMB Agentic Report — {html.escape(provider)}</title>
<style>
{_BASE_STYLE}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th {{ text-align: left; color: var(--muted); font-weight: 600; font-size: 11px;
        text-transform: uppercase; letter-spacing: 0.02em; padding: 8px 10px;
        border-bottom: 1px solid var(--gridline); }}
  td {{ padding: 10px; border-bottom: 1px solid var(--gridline); vertical-align: top; }}
  td.mono {{ font-family: ui-monospace, monospace; font-size: 12px; color: var(--text-secondary); }}
  td.center {{ text-align: center; }}
  .status-dot {{ display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; }}
  .table-wrap {{ overflow-x: auto; }}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>FP-AMB Agentic Tool-Use Report</h1>
    <div class="meta">{html.escape(provider)} &middot; {html.escape(model)} &middot; {html.escape(timestamp)}</div>
  </header>

  <div class="card hero">
    <div class="hero-value" style="color:{hero_color}">{pct:.1f}%</div>
    <div class="hero-sub">{passed} / {total} scenarios passed (correct tool, correct order)</div>
  </div>

  <div class="card">
    <h2>Summary</h2>
    <div class="tiles">{tiles}</div>
  </div>

  <div class="card">
    <h2>Scenario Detail</h2>
    <div class="table-wrap">
    <table>
      <thead><tr>
        <th>Status</th><th>Scenario</th><th>Tools Called</th><th>Tools Expected</th>
        <th>Order OK</th><th>Wrong Tool</th><th>Used Result</th><th>Time</th>
      </tr></thead>
      <tbody>{table_rows}</tbody>
    </table>
    </div>
  </div>

  <footer>Generated by fp_amb.report &middot; {html.escape(timestamp)}</footer>
</div>
</body>
</html>"""


def write_agentic_report(scorecard: dict, output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.write_text(generate_agentic_html_report(scorecard))
    return output_path


# ---------------------------------------------------------------------------
# Plain-text / Markdown + ASCII + Mermaid reports (no browser, no Artifact)
# ---------------------------------------------------------------------------

def _ascii_bar(pct: float, width: int = 28) -> str:
    filled = max(0, min(width, round(pct / 100 * width)))
    return "█" * filled + "░" * (width - filled)


def _status_tag(pct: float) -> str:
    if pct >= 80:
        return "GOOD"
    if pct >= 50:
        return "WARN"
    return "CRIT"


def generate_text_report(scorecard: dict) -> str:
    provider = scorecard.get("provider_name", "Memory Provider")
    overall = scorecard.get("overall_accuracy_pct", 0.0)
    passed = scorecard.get("passed_items", 0)
    total = scorecard.get("total_evaluated_questions", 0)
    mode = "Full LLM Generation" if scorecard.get("use_llm_generation") else "Pure Retrieval (Zero-LLM)"
    model = scorecard.get("model_name")
    mode_label = f"{mode} — {model}" if model else mode
    timestamp = scorecard.get("timestamp", "")
    cat_breakdown = scorecard.get("category_breakdown", {})

    rows = []
    for cat, s in cat_breakdown.items():
        pct = (s["correct"] / s["total"] * 100) if s["total"] else 0.0
        rows.append((cat, s["correct"], s["total"], pct))
    rows.sort(key=lambda r: r[3])
    name_w = max((len(r[0]) for r in rows), default=20)

    lines = []
    lines.append(f"# FP-AMB Exam Report — {provider}")
    lines.append(f"_{mode_label} · {timestamp}_")
    lines.append("")
    lines.append(f"## {overall:.1f}%  ({passed} / {total} items passed)")
    lines.append("")
    lines.append("## Performance")
    lines.append(f"- Avg Retrieval Latency: {scorecard.get('avg_retrieval_latency_ms', 0):.2f} ms")
    lines.append(f"- Avg Injected Context: {scorecard.get('avg_injected_tokens_per_query', 0):.0f} tok")
    lines.append(f"- Token Efficiency: {scorecard.get('token_efficiency_ratio', 0):.2f} pts / 1k tok")
    lines.append(f"- Ingestion Time: {scorecard.get('ingestion_duration_seconds', 0):.2f} s")
    lines.append(f"- Exam Duration: {scorecard.get('eval_duration_seconds', 0):.1f} s")
    lines.append("")
    lines.append("## Category Breakdown (worst first)")
    lines.append("```")
    for cat, correct, total_c, pct in rows:
        tag = _status_tag(pct)
        bar = _ascii_bar(pct)
        lines.append(f"{tag:4}  {cat:<{name_w}}  [{bar}]  {pct:5.1f}%  ({correct}/{total_c})")
    lines.append("```")
    lines.append("")
    lines.append("```mermaid")
    lines.append("pie showData")
    lines.append(f'    title Category accuracy — {provider}')
    for cat, correct, total_c, pct in rows:
        short = CATEGORY_SHORT.get(cat, cat)
        lines.append(f'    "{short}" : {pct:.1f}')
    lines.append("```")
    lines.append("")
    lines.append(f"_Generated by fp_amb.report · {timestamp}_")
    return "\n".join(lines)


def write_text_report(scorecard: dict, output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.write_text(generate_text_report(scorecard))
    return output_path


def generate_agentic_text_report(scorecard: dict) -> str:
    provider = scorecard.get("provider_name", "Memory Provider")
    model = scorecard.get("model_name", "")
    passed = scorecard.get("passed_scenarios", 0)
    total = scorecard.get("total_scenarios", 0)
    pct = scorecard.get("pass_rate_pct", 0.0)
    used_count = scorecard.get("used_result_in_answer_count", 0)
    timestamp = scorecard.get("timestamp", "")
    results = scorecard.get("results", [])

    lines = []
    lines.append(f"# FP-AMB Agentic Tool-Use Report — {provider}")
    lines.append(f"_{model} · {timestamp}_")
    lines.append("")
    lines.append(f"## {pct:.1f}%  ({passed} / {total} scenarios passed)")
    lines.append(f"Used scripted result in final answer: {used_count} / {total}")
    lines.append("")
    lines.append("## Scenario Detail")
    lines.append("")
    lines.append("| Status | Scenario | Tools Called | Tools Expected | Order OK | Wrong Tool | Used Result | Time |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in results:
        status = "PASS" if r.get("pass") else "FAIL"
        called = " → ".join(r.get("called_sequence", [])) or "(none)"
        wanted = " → ".join(r.get("correct_sequence", [])) or "(none)"
        order_ok = "yes" if r.get("order_correct") else "no"
        wrong = "yes" if r.get("wrong_tool_called") else "no"
        used = "yes" if r.get("used_result_in_answer") else "no"
        lines.append(
            f"| {status} | {r.get('scenario_id','')} | `{called}` | `{wanted}` | "
            f"{order_ok} | {wrong} | {used} | {r.get('elapsed_s', 0):.1f}s |"
        )
    lines.append("")
    lines.append("```mermaid")
    lines.append("graph LR")
    for i, r in enumerate(results):
        node = f"S{i}"
        cls = "pass" if r.get("pass") else "fail"
        label = r.get("scenario_id", "").replace('"', "'")
        lines.append(f'    {node}["{label}"]:::{cls}')
    lines.append("    classDef pass fill:#0ca30c,color:#fff,stroke:none")
    lines.append("    classDef fail fill:#d03b3b,color:#fff,stroke:none")
    lines.append("```")
    lines.append("")
    lines.append(f"_Generated by fp_amb.report · {timestamp}_")
    return "\n".join(lines)


def write_agentic_text_report(scorecard: dict, output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.write_text(generate_agentic_text_report(scorecard))
    return output_path


def generate_misses_report(scorecard: dict) -> str:
    provider = scorecard.get("provider_name", "Memory Provider")
    overall = scorecard.get("overall_accuracy_pct", 0.0)
    passed = scorecard.get("passed_items", 0)
    total = scorecard.get("total_evaluated_questions", 0)
    total_misses = total - passed
    mode = "Full LLM Generation" if scorecard.get("use_llm_generation") else "Pure Retrieval (Zero-LLM)"
    model = scorecard.get("model_name")
    mode_label = f"{mode} — {model}" if model else mode
    timestamp = scorecard.get("timestamp", "")
    item_logs = scorecard.get("item_logs", [])

    misses = [item for item in item_logs if not item.get("pass")]

    cause_counts = {}
    for m in misses:
        cause = m.get("failure_cause", "UNKNOWN_FAILURE_CAUSE")
        cause_counts[cause] = cause_counts.get(cause, 0) + 1

    by_category = {}
    for m in misses:
        cat = m.get("category", "General")
        cause = m.get("failure_cause", "UNKNOWN_FAILURE_CAUSE")
        if cat not in by_category:
            by_category[cat] = {}
        if cause not in by_category[cat]:
            by_category[cat][cause] = []
        by_category[cat][cause].append(m)

    lines = []
    lines.append("================================================================================")
    lines.append("               FP-AMB MISSED QUESTIONS & FAILURE TAXONOMY REPORT")
    lines.append(f"Provider: {provider} | Mode: {mode_label} | Date: {timestamp}")
    lines.append(f"Overall Pass Rate: {overall:.1f}% ({passed}/{total} passed) | Total Misses: {total_misses}")
    lines.append("================================================================================\n")

    lines.append("TOKEN & PAYLOAD METRICS SUMMARY")
    lines.append("--------------------------------------------------------------------------------")
    lines.append(f"  • Total Corpus Size:                     ~{scorecard.get('total_corpus_tokens', 0):,} tokens")
    lines.append(f"  • Total Injected Payload (Across Exam):  {scorecard.get('total_injected_tokens_across_exam', 0):,} tokens")
    lines.append(f"  • Avg Injected Context Payload Size:     {scorecard.get('avg_injected_tokens_per_query', 0):.1f} tokens ({scorecard.get('avg_injected_chars_per_query', 0):.1f} chars)")
    lines.append(f"  • Injected Payload Size Range:           {scorecard.get('min_injected_tokens_per_query', 0)} min - {scorecard.get('max_injected_tokens_per_query', 0)} max tokens")
    lines.append(f"  • Token Efficiency Ratio:                {scorecard.get('token_efficiency_ratio', 0):.2f} accuracy pts per 1k injected tokens")
    lines.append("--------------------------------------------------------------------------------\n")

    lines.append("SUMMARY OF MISSES BY FAILURE CAUSE")
    lines.append("--------------------------------------------------------------------------------")
    if cause_counts:
        for cause, count in sorted(cause_counts.items(), key=lambda x: x[1], reverse=True):
            pct = (count / total_misses * 100) if total_misses > 0 else 0
            lines.append(f"  [{cause}]")
            lines.append(f"    Count: {count} misses ({pct:.1f}% of total misses)")
    else:
        lines.append("  (No misses recorded - 100% pass rate!)")
    lines.append("--------------------------------------------------------------------------------\n")

    lines.append("DETAILED BREAKDOWN BY QUESTION CATEGORY & ROOT CAUSE OF MISS")
    lines.append("================================================================================\n")

    for cat_name, causes_dict in sorted(by_category.items()):
        cat_miss_count = sum(len(items) for items in causes_dict.values())
        lines.append(f"================================================================================")
        lines.append(f"CATEGORY: {cat_name} ({cat_miss_count} misses)")
        lines.append(f"================================================================================")

        for cause_name, items in sorted(causes_dict.items()):
            lines.append(f"\n  ► FAILURE CAUSE: {cause_name} ({len(items)} items)")
            lines.append("  " + "-" * 76)
            for idx, item in enumerate(items, 1):
                q_id = item.get("id", f"Q_{idx}")
                q_text = item.get("question", "")
                expected = item.get("expected_answer", "")
                reason = item.get("failure_reason", "")
                context = item.get("retrieved_context", "")
                gen_out = item.get("generated_output", "")
                inj_tok = item.get("injected_tokens", len(context.split()))

                context_snippet = context.strip().replace("\n", " ")
                if len(context_snippet) > 250:
                    context_snippet = context_snippet[:250] + " ... [TRUNCATED]"

                lines.append(f"  [{idx:02d}] Question ID: {q_id}")
                lines.append(f"       Question:         \"{q_text}\"")
                lines.append(f"       Expected Answer:  \"{expected}\"")
                lines.append(f"       Diagnosed Cause:  {cause_name}")
                lines.append(f"       Diagnostic Note:  {reason}")
                lines.append(f"       Injected Context ({inj_tok} tokens): \"{context_snippet}\"")
                if gen_out:
                    lines.append(f"       Model Generation: \"{gen_out}\"")
                lines.append("  " + "-" * 76)
        lines.append("\n")

    lines.append(f"Report generated by FP-AMB Evaluator Engine · {timestamp}")
    return "\n".join(lines)


def write_misses_report(scorecard: dict, output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.write_text(generate_misses_report(scorecard))
    return output_path


if __name__ == "__main__":
    import sys
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_suffix(".html")
    scorecard = json.loads(src.read_text())
    write_report(scorecard, dst)
    print(f"Wrote report to {dst}")
