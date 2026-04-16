"""OpenAI Responses API を使って science ドメインの benchmark case を量産する CLI。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from src.models import Case, dump_json
from src.science.case_builder import build_case
from src.science.prompts import build_manual_prompt, build_seed_prompt, select_blueprint
from src.science.type import (
    build_exact_generation_classes,
    choose_count_profile,
)

load_dotenv()

DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_OUTPUT_DIR = Path("scenarios/science")


def parse_args() -> argparse.Namespace:
    """CLI 引数を解釈する。"""

    parser = argparse.ArgumentParser(
        description="OpenAI Responses API で science benchmark case を生成する"
    )
    parser.add_argument("--count", type=int, default=3, help="生成する case 数")
    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="task_id の開始番号。例: 1 -> science_001",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"生成した case JSON の出力先。既定値: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL", DEFAULT_MODEL),
        help=f"Responses API で使うモデル名。既定値: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="1 case あたりの再試行回数",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="既存の JSON があっても上書きする",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="API は呼ばず、生成予定の task_id とテーマだけを表示する",
    )
    return parser.parse_args()


def difficulty_for_index(index: int) -> str:
    """easy / normal / hard を緩くローテーションする。"""

    cycle = ("easy", "normal", "hard")
    return cycle[(index - 1) % len(cycle)]


def build_task_id(index: int) -> str:
    """science_001 形式の task_id を返す。"""

    return f"science_{index:03d}"


def generate_case(
    *,
    client: OpenAI,
    task_id: str,
    difficulty: str,
    model: str,
    max_attempts: int,
) -> Case:
    """OpenAI から 1 件の science case を生成し、Case として検証する。"""

    blueprint = select_blueprint(int(task_id.split("_")[-1]))
    count_profile = choose_count_profile(difficulty, key=f"{task_id}:{difficulty}")
    seed_class, bundle_class, draft_class = build_exact_generation_classes(
        difficulty,
        count_profile.hazard_count,
        count_profile.selected_constraint_count,
    )
    previous_error: str | None = None
    for _attempt in range(1, max_attempts + 1):
        try:
            seed_response = client.responses.parse(
                model=model,
                instructions=(
                    "You create benchmark case seeds for evaluating executive functions in "
                    "science hazard-response operations. Write all field values in English. "
                    "Return only valid JSON that matches the requested schema. "
                    "Do not generate hazard manuals in this phase."
                ),
                input=build_seed_prompt(
                    task_id=task_id,
                    difficulty=difficulty,
                    count_profile=count_profile,
                    blueprint=blueprint,
                    previous_error=previous_error,
                ),
                text_format=seed_class,
                temperature=0.8,
                max_output_tokens=20000,
            )
            seed = seed_response.output_parsed
            if seed is None:
                previous_error = "JSON が取得できませんでした。必ず JSON だけを返してください。"
                continue
            bundle_response = client.responses.parse(
                model=model,
                instructions=(
                    "You create hazard manuals, question targets, and hidden runtime facts for "
                    "a pre-generated science hazard-response benchmark seed. "
                    "Write all field values in English. Return only valid JSON that matches the "
                    "requested schema."
                ),
                input=build_manual_prompt(
                    task_id=task_id,
                    difficulty=difficulty,
                    count_profile=count_profile,
                    blueprint=blueprint,
                    seed=seed,
                    previous_error=previous_error,
                ),
                text_format=bundle_class,
                temperature=0.3,
                max_output_tokens=30000,
            )
            bundle = bundle_response.output_parsed
            if bundle is None:
                previous_error = "manual bundle JSON が取得できませんでした。"
                continue
            draft = draft_class.model_validate(
                {
                    "initial_request": seed.initial_request,
                    "reference_data": {
                        **seed.reference_data.model_dump(mode="python"),
                        "hazard_manuals": bundle.hazard_manuals,
                        "hazard_question_targets": bundle.hazard_question_targets,
                        "hidden_hazard_facts": bundle.hidden_hazard_facts,
                    },
                    "events": seed.events,
                },
                context={"difficulty": difficulty, "count_profile": count_profile},
            )
        except Exception as exc:  # noqa: BLE001
            previous_error = f"{type(exc).__name__}: {exc}"
            continue

        try:
            return build_case(task_id=task_id, difficulty=difficulty, draft=draft)
        except Exception as exc:  # noqa: BLE001
            previous_error = f"{type(exc).__name__}: {exc}"

    raise RuntimeError(f"{task_id} の生成に失敗しました。last_error={previous_error}")


def write_case(path: Path, case: Case) -> None:
    """Case を安定した JSON として保存する。"""

    path.write_text(f"{dump_json(case.model_dump(mode='json'))}\n", encoding="utf-8")


def validate_paths(output_dir: Path, count: int, start_index: int, overwrite: bool) -> list[Path]:
    """出力先の事前衝突チェックを行う。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for index in range(start_index, start_index + count):
        path = output_dir / f"{build_task_id(index)}.json"
        if path.exists() and not overwrite:
            raise FileExistsError(
                f"{path} already exists. Use --overwrite or change --start-index."
            )
        paths.append(path)
    return paths


def preview_plan(count: int, start_index: int, output_dir: Path) -> list[dict[str, Any]]:
    """dry-run 用の生成予定一覧を返す。"""

    preview: list[dict[str, Any]] = []
    for index in range(start_index, start_index + count):
        blueprint = select_blueprint(index)
        task_id = build_task_id(index)
        difficulty = difficulty_for_index(index)
        count_profile = choose_count_profile(difficulty, key=f"{task_id}:{difficulty}")
        preview.append(
            {
                "task_id": task_id,
                "difficulty": difficulty,
                "blueprint": blueprint.slug,
                "theme": blueprint.label,
                "hazard_count": count_profile.hazard_count,
                "selected_constraint_count": count_profile.selected_constraint_count,
                "question_target_count": count_profile.question_target_count,
                "output_path": str(output_dir / f"{task_id}.json"),
            }
        )
    return preview


def main() -> None:
    """CLI エントリポイント。"""

    args = parse_args()
    if args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        print(
            json.dumps(
                preview_plan(args.count, args.start_index, args.output_dir),
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    validate_paths(
        output_dir=args.output_dir,
        count=args.count,
        start_index=args.start_index,
        overwrite=args.overwrite,
    )

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY が未設定です。")

    client = OpenAI()
    for index in range(args.start_index, args.start_index + args.count):
        task_id = build_task_id(index)
        difficulty = difficulty_for_index(index)
        blueprint = select_blueprint(index)
        case = generate_case(
            client=client,
            task_id=task_id,
            difficulty=difficulty,
            model=args.model,
            max_attempts=args.max_attempts,
        )
        output_path = args.output_dir / f"{task_id}.json"
        write_case(output_path, case)
        print(
            json.dumps(
                {
                    "task_id": task_id,
                    "difficulty": difficulty,
                    "blueprint": blueprint.slug,
                    "output_path": str(output_path),
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
