"""Minimal benchmark runner for Dynamic Agent Correctness Benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCENARIOS_DIR = ROOT / "scenarios"

REQUIRED_TOP_LEVEL_FIELDS = [
    "task_id",
    "domain",
    "difficulty",
    "initial_request",
    "initial_state",
    "allowed_actions",
    "events",
    "goal_condition",
    "rubric",
]


def list_case_files(domain: str | None = None) -> list[Path]:
    if domain:
      domain_dir = SCENARIOS_DIR / domain
      return sorted(domain_dir.glob("*.json")) if domain_dir.exists() else []
    return sorted(SCENARIOS_DIR.glob("*/*.json"))


def load_case(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_case_structure(case: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_TOP_LEVEL_FIELDS:
        if field not in case:
            errors.append(f"missing top-level field: {field}")
    if "events" in case and not isinstance(case["events"], list):
        errors.append("events must be a list")
    if "allowed_actions" in case and not isinstance(case["allowed_actions"], list):
        errors.append("allowed_actions must be a list")
    if "initial_state" in case:
        state = case["initial_state"]
        for field in ["timezone", "deadline", "participants", "budget", "required_artifacts", "constraints"]:
            if field not in state:
                errors.append(f"missing initial_state field: {field}")
    if "rubric" in case:
        rubric = case["rubric"]
        for field in ["outcome", "process", "recovery"]:
            if field not in rubric:
                errors.append(f"missing rubric field: {field}")
    return errors


def summarize_case(case: dict[str, Any], path: Path) -> str:
    participants = len(case.get("initial_state", {}).get("participants", []))
    artifacts = len(case.get("initial_state", {}).get("required_artifacts", []))
    events = len(case.get("events", []))
    return (
        f"[OK] {case.get('task_id', 'unknown')} "
        f"domain={case.get('domain')} difficulty={case.get('difficulty')} "
        f"participants={participants} artifacts={artifacts} events={events} "
        f"path={path.relative_to(ROOT)}"
    )


def run(domain: str | None = None, case_path: str | None = None) -> int:
    if case_path:
        files = [ROOT / case_path]
    else:
        files = list_case_files(domain)

    if not files:
        print("No scenario files found.")
        return 1

    total_errors = 0
    for path in files:
        case = load_case(path)
        errors = validate_case_structure(case)
        if errors:
            total_errors += len(errors)
            print(f"[ERROR] {path.relative_to(ROOT)}")
            for error in errors:
                print(f"  - {error}")
            continue
        print(summarize_case(case, path))

    if total_errors:
        print(f"\nValidation finished with {total_errors} error(s).")
        return 1

    print(f"\nValidation finished successfully for {len(files)} case(s).")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run minimal validation for benchmark cases.")
    parser.add_argument("--domain", help="Scenario domain directory under scenarios/, e.g. meeting")
    parser.add_argument("--case", help="Single scenario path relative to repository root")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(run(domain=args.domain, case_path=args.case))
