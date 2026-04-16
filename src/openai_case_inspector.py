"""LLM-based case inspection CLI for OpenAI and Gemini."""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from src.models import Case, ExecutionLog

load_dotenv()

DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
OPENAI_PROVIDER = "openai"
GEMINI_PROVIDER = "gemini"
SUPPORTED_PROVIDERS = (OPENAI_PROVIDER, GEMINI_PROVIDER)
DEFAULT_INSTRUCTIONS = (
    "You are a strict benchmark reviewer. Be concrete, concise, and point out "
    "contradictions, ambiguity, and evaluator gaps."
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Inspect a benchmark case and optional execution log with an LLM."
    )
    parser.add_argument(
        "--case",
        dest="case_path",
        type=Path,
        help="Path to the case JSON file.",
    )
    parser.add_argument(
        "--execution-log",
        dest="execution_log_path",
        type=Path,
        help="Optional execution log JSON path.",
    )
    parser.add_argument(
        "--provider",
        choices=SUPPORTED_PROVIDERS,
        default=OPENAI_PROVIDER,
        help="LLM provider used for inspection.",
    )
    parser.add_argument(
        "--model",
        help="Override the model name for the selected provider.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON output path for the inspection result.",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Inspect all matched cases under the scenarios root.",
    )
    parser.add_argument(
        "--scenarios-dir",
        type=Path,
        default=Path("scenarios"),
        help="Root directory that contains scenario JSON files.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Root directory that contains execution log JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory used for batch-mode per-case JSON outputs.",
    )
    parser.add_argument(
        "--domain",
        action="append",
        default=[],
        help="Filter batch mode by scenario domain directory. Can be passed multiple times.",
    )
    parser.add_argument(
        "--task-id",
        action="append",
        default=[],
        help="Filter batch mode by task_id. Can be passed multiple times.",
    )
    parser.add_argument(
        "--include-templates",
        action="store_true",
        help="Include *_template.json scenarios in batch mode.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the request payload preview without calling the provider API.",
    )
    return parser.parse_args()


def build_inspection_prompt(case: Case, execution_log: ExecutionLog | None) -> str:
    """Build the prompt sent to the provider."""

    sections = [
        "You are reviewing a benchmark case.",
        "Inspect the definition carefully and highlight concrete issues.",
        "1. Summarize the case setup.",
        "2. Check rubric and goal_condition alignment.",
        "3. Check events against initial_state and required_artifacts.",
        "4. Identify evaluator blind spots or contradictions.",
        "5. If an execution log is provided, check whether it satisfies the case.",
        "",
        "Output format:",
        "- summary: 2-4 bullets",
        "- findings: bullet list",
        "- suggested_fixes: bullet list",
        "",
        "Case JSON:",
        json.dumps(case.model_dump(mode="json"), ensure_ascii=False, indent=2),
    ]

    if execution_log is not None:
        sections.extend(
            [
                "",
                "ExecutionLog JSON:",
                json.dumps(execution_log.model_dump(mode="json"), ensure_ascii=False, indent=2),
            ]
        )

    return "\n".join(sections)


def resolve_model(provider: str, requested_model: str | None) -> str:
    if requested_model:
        return requested_model
    if provider == GEMINI_PROVIDER:
        return os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
    return os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)


def build_output_path(
    provider: str,
    case_path: Path,
    output_path: Path | None,
) -> Path:
    if output_path is not None:
        return output_path
    return Path("results") / f"{provider}_inspection_{case_path.stem}.json"


def build_batch_output_path(
    provider: str,
    case_path: Path,
    output_dir: Path | None,
) -> Path:
    base_dir = output_dir or (Path("results") / f"{provider}_inspections")
    return base_dir / case_path.parent.name / f"{case_path.stem}.json"


def discover_cases(
    scenarios_dir: Path,
    include_templates: bool,
    domain_filters: set[str],
    task_id_filters: set[str],
) -> list[Path]:
    case_paths: list[Path] = []
    for case_path in sorted(scenarios_dir.rglob("*.json")):
        if not include_templates and case_path.stem.endswith("_template"):
            continue
        case = Case.from_path(case_path)
        if domain_filters and case_path.parent.name not in domain_filters:
            continue
        if task_id_filters and case.task_id not in task_id_filters:
            continue
        case_paths.append(case_path)
    return case_paths


def discover_execution_logs(results_dir: Path) -> dict[str, Path]:
    log_index: dict[str, Path] = {}
    for log_path in sorted(results_dir.rglob("*.json")):
        try:
            execution_log = ExecutionLog.from_path(log_path)
        except Exception:
            continue
        if execution_log.task_id not in log_index:
            log_index[execution_log.task_id] = log_path
    return log_index


def inspect_case_payload(
    provider: str,
    model: str,
    case_path: Path,
    execution_log_path: Path | None,
    dry_run: bool,
) -> dict[str, Any]:
    case = Case.from_path(case_path)
    execution_log = ExecutionLog.from_path(execution_log_path) if execution_log_path else None
    prompt = build_inspection_prompt(case=case, execution_log=execution_log)
    overview = build_case_overview(case=case, execution_log=execution_log)

    if dry_run:
        return {
            "provider": provider,
            "model": model,
            "case_path": str(case_path),
            "execution_log_path": str(execution_log_path) if execution_log_path else None,
            "case_overview": overview,
            "prompt_preview": prompt[:2000],
        }

    if provider == GEMINI_PROVIDER:
        inspection_text, raw_response = call_gemini(model=model, prompt=prompt)
    else:
        inspection_text, raw_response = call_openai(model=model, prompt=prompt)

    return {
        "provider": provider,
        "model": model,
        "case_path": str(case_path),
        "execution_log_path": str(execution_log_path) if execution_log_path else None,
        "case_overview": overview,
        "inspection_text": inspection_text,
        "raw_response": raw_response,
    }


def build_case_overview(case: Case, execution_log: ExecutionLog | None) -> dict[str, Any]:
    overview = {
        "task_id": case.task_id,
        "domain": case.domain,
        "difficulty": case.difficulty,
        "required_artifacts": [
            artifact.artifact_id for artifact in case.initial_state.required_artifacts
        ],
        "event_count": len(case.events),
        "events": [
            {
                "turn": event.turn,
                "type": event.type,
                "expected_tasks": event.expected_tasks,
                "expected_artifact_updates": event.expected_artifact_updates,
            }
            for event in case.events
        ],
    }
    if execution_log is not None:
        overview["execution_log_summary"] = {
            "action_count": len(execution_log.actions),
            "question_count": len(execution_log.questions_asked),
            "completed_tasks": execution_log.completed_tasks,
            "finalized": any(action.action_type == "finalize" for action in execution_log.actions),
            "timeline": [
                {
                    "turn": action.turn,
                    "action_type": action.action_type,
                    "acknowledged_event_turns": action.acknowledged_event_turns,
                    "artifact_updates": action.artifact_updates,
                    "notes": action.notes,
                }
                for action in execution_log.actions
            ],
        }
    return overview


def run_batch(args: argparse.Namespace) -> None:
    model = resolve_model(args.provider, args.model)
    case_paths = discover_cases(
        scenarios_dir=args.scenarios_dir,
        include_templates=args.include_templates,
        domain_filters=set(args.domain),
        task_id_filters=set(args.task_id),
    )
    log_index = discover_execution_logs(args.results_dir)

    manifest_cases: list[dict[str, Any]] = []
    for case_path in case_paths:
        case = Case.from_path(case_path)
        execution_log_path = log_index.get(case.task_id)
        output_path = build_batch_output_path(args.provider, case_path, args.output_dir)
        payload = inspect_case_payload(
            provider=args.provider,
            model=model,
            case_path=case_path,
            execution_log_path=execution_log_path,
            dry_run=args.dry_run,
        )
        save_payload(output_path, payload)
        manifest_cases.append(
            {
                "task_id": case.task_id,
                "domain": case.domain,
                "case_path": str(case_path),
                "execution_log_path": str(execution_log_path) if execution_log_path else None,
                "output_path": str(output_path),
            }
        )

    manifest_path = (args.output_dir or (Path("results") / f"{args.provider}_inspections")) / (
        "index.json"
    )
    manifest_payload = {
        "provider": args.provider,
        "model": model,
        "case_count": len(manifest_cases),
        "dry_run": args.dry_run,
        "cases": manifest_cases,
    }
    save_payload(manifest_path, manifest_payload)
    print(json.dumps(manifest_payload, ensure_ascii=False, indent=2))


def call_openai(model: str, prompt: str) -> tuple[str, dict[str, Any]]:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")

    client = OpenAI()
    response = client.responses.create(
        model=model,
        instructions=DEFAULT_INSTRUCTIONS,
        input=prompt,
    )
    return response.output_text, response.model_dump(mode="json")


def call_gemini(model: str, prompt: str) -> tuple[str, dict[str, Any]]:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")

    model_name = model if model.startswith("models/") else f"models/{model}"
    endpoint = (
        f"https://generativelanguage.googleapis.com/v1beta/"
        f"{model_name}:generateContent?key={urllib.parse.quote(api_key)}"
    )
    payload = {
        "systemInstruction": {
            "parts": [
                {
                    "text": DEFAULT_INSTRUCTIONS,
                }
            ]
        },
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt,
                    }
                ]
            }
        ],
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Gemini API request failed: {exc.code} {error_body}") from exc

    return extract_gemini_text(response_payload), response_payload


def extract_gemini_text(response_payload: dict[str, Any]) -> str:
    texts: list[str] = []
    for candidate in response_payload.get("candidates", []):
        content = candidate.get("content", {})
        for part in content.get("parts", []):
            text = part.get("text")
            if text:
                texts.append(text)
    return "\n".join(texts).strip()


def save_payload(output_path: Path, payload: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    """CLI entrypoint."""

    args = parse_args()
    if args.batch:
        run_batch(args)
        return

    if args.case_path is None:
        raise RuntimeError("--case is required unless --batch is used.")

    model = resolve_model(args.provider, args.model)
    output_path = build_output_path(args.provider, args.case_path, args.output)
    result_payload = inspect_case_payload(
        provider=args.provider,
        model=model,
        case_path=args.case_path,
        execution_log_path=args.execution_log_path,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        result_payload["output_path"] = str(output_path)
    save_payload(output_path, result_payload)
    print(json.dumps(result_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
