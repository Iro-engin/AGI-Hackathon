from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

from ...models.meeting import (
    Attendance,
    MeetingCase,
    MeetingDo,
    MeetingEvent,
    MeetingInfo,
    MeetingState,
    PeopleInfo,
)

load_dotenv()  # .env ファイルから環境変数を読み込む

Difficulty = Literal["easy", "medium", "hard"]
EventType = Literal["schedule_change", "schedule_pause", "info_update", "context_note"]

DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"

EVENT_COUNT_BY_DIFFICULTY: dict[Difficulty, int] = {
    "easy": 2,
    "medium": 4,
    "hard": 4,
}

INITIAL_REQUIRE_COUNT_BY_DIFFICULTY: dict[Difficulty, int] = {
    "easy": 2,
    "medium": 4,
    "hard": 6,
}

PARTICIPANT_COUNT_BY_DIFFICULTY: dict[Difficulty, int] = {
    "easy": 3,
    "medium": 4,
    "hard": 5,
}

NAME_POOL = [
    "Alice",
    "Bob",
    "Carol",
    "David",
    "Eve",
    "Frank",
    "Grace",
    "Henry",
    "Iris",
    "Jack",
]

PLACE_POOL = [
    "online",
    "meeting_room_a",
    "meeting_room_b",
    "meeting_room_c",
    "conference_room_3f",
    "conference_room_4f",
]

MEETING_CONSTRAINTS_TEMPLATE = [
    "All participants must share the same candidate slot",
    "Only slots with {need_duration} consecutive minutes available are valid",
    "Prefer the earliest available time slot",
]

OPENAI_INSTRUCTIONS = """
You are a meeting case generation assistant.
Return only valid JSON. Do not include Markdown or explanatory text.
""".strip()


@dataclass(frozen=True)
class Slot:
    """会議候補1件を内部で扱うための時刻情報。"""

    slot_id: str
    where: str
    when_from: datetime
    when_to: datetime

    def to_attendance(self) -> Attendance:
        """Attendance モデルへ変換する。"""
        return Attendance(where=self.where, when_from=self.when_from, when_to=self.when_to)

    @property
    def key(self) -> tuple[str, datetime, datetime]:
        """会議候補を一意に表すキーを返す。"""
        return (self.where, self.when_from, self.when_to)


class MeetingEventPlan(BaseModel):
    """OpenAI が返す event 設計。"""

    target_person: str
    event_type: EventType
    message: str
    exp_do_slot_id: str
    constraint_hint: str | None = None


class MeetingPlan(BaseModel):
    """参加者、会議時間、event 方針の下書き。"""

    participant_names: list[str] = Field(default_factory=list)
    need_duration: int
    events: list[MeetingEventPlan] = Field(default_factory=list)


class PersonScheduleDraft(BaseModel):
    """OpenAI が返す1人分のスケジュール下書き。"""

    name: str
    constraints: list[str] = Field(default_factory=list)
    slot_ids: list[str] = Field(default_factory=list)


class ScheduleSnapshotDraft(BaseModel):
    """OpenAI が返す1時点分の全参加者スケジュール。"""

    people: list[PersonScheduleDraft] = Field(default_factory=list)


class MeetingSchedulePlan(BaseModel):
    """初期状態と各 event 時点のスケジュール集合。"""

    initial_people: list[PersonScheduleDraft] = Field(default_factory=list)
    event_people: list[ScheduleSnapshotDraft] = Field(default_factory=list)


def generate_meeting_case(
    task_id: str,
    difficulty: Difficulty = "medium",
    seed: int | None = None,
    model: str = DEFAULT_OPENAI_MODEL,
) -> MeetingCase:
    """OpenAI を使って 1 件分の meeting ケースを生成する。"""
    rng = random.Random(seed)
    event_count = EVENT_COUNT_BY_DIFFICULTY[difficulty]
    initial_require_count = INITIAL_REQUIRE_COUNT_BY_DIFFICULTY[difficulty]
    participant_count = PARTICIPANT_COUNT_BY_DIFFICULTY[difficulty]

    start_day = _next_weekday(date.today())
    slot_catalog = _build_slot_catalog(
        start_day=start_day,
        count=12,
    )

    meeting_plan = _generate_meeting_plan(
        model=model,
        difficulty=difficulty,
        seed=seed,
        participant_count=participant_count,
        event_count=event_count,
        slot_catalog=slot_catalog,
    )
    meeting_plan = _normalize_meeting_plan(
        meeting_plan=meeting_plan,
        difficulty=difficulty,
        event_count=event_count,
        slot_catalog=slot_catalog,
    )

    schedule_plan = _generate_schedule_plan(
        model=model,
        difficulty=difficulty,
        seed=seed,
        meeting_plan=meeting_plan,
        slot_catalog=slot_catalog,
    )
    schedule_plan = _normalize_schedule_plan(
        schedule_plan=schedule_plan,
        event_count=event_count,
    )

    snapshots = _build_people_snapshots(
        rng=rng,
        schedule_plan=schedule_plan,
        meeting_plan=meeting_plan,
        slot_catalog=slot_catalog,
    )

    constraint = _build_constraints(meeting_plan.need_duration)
    initial_people = snapshots[0]
    hidden_people_count = max(1, initial_require_count // 2)
    initial_hidden_names = {
        person.name for person in initial_people[:hidden_people_count]
    }
    initial_state = _build_partial_state(
        people=initial_people,
        need_duration=meeting_plan.need_duration,
        constraint=constraint,
        hidden_names=initial_hidden_names,
        hidden_slots_per_person=2,
    )
    initial_requires = _build_requires_from_state(initial_state)[:initial_require_count]
    initial_request = _build_initial_request(
        people=initial_people,
        visible_state=initial_state.got_info,
        need_duration=meeting_plan.need_duration,
    )

    initial_best = _find_best_slot(initial_people)
    if initial_best is None:
        raise ValueError("meeting openai generation failed: no common slot in initial snapshot")
    initial_do = MeetingDo(
        where=initial_best.where,
        when_from=initial_best.when_from,
        when_to=initial_best.when_to,
    )

    events: list[MeetingEvent] = []
    for index, event_plan in enumerate(meeting_plan.events):
        state_people = snapshots[index + 1]
        best_slot = _find_best_slot(state_people)
        if best_slot is None:
            raise ValueError("meeting openai generation failed: no common slot in event snapshot")

        hidden_names = {event_plan.target_person}
        if event_plan.event_type == "context_note":
            hidden_names = {state_people[index % len(state_people)].name}

        state_before = _build_partial_state(
            people=state_people,
            need_duration=meeting_plan.need_duration,
            constraint=constraint,
            hidden_names=hidden_names,
            hidden_slots_per_person=2,
        )
        events.append(
            MeetingEvent(
                message=event_plan.message,
                state_before=state_before,
                exp_do=MeetingDo(
                    where=best_slot.where,
                    when_from=best_slot.when_from,
                    when_to=best_slot.when_to,
                ),
                should_require=_build_requires_from_state(state_before)[
                    : _question_count(difficulty)
                ],
            )
        )

    return MeetingCase(
        task_id=task_id,
        difficulty=difficulty,
        initial_request=initial_request,
        initial_state=initial_state,
        initial_do=initial_do,
        events=events,
        initial_requires=initial_requires,
    )


def generate_meeting_cases(
    count: int,
    difficulty: Difficulty = "medium",
    seed: int | None = None,
    task_id_prefix: str = "meeting_openai",
    model: str = DEFAULT_OPENAI_MODEL,
) -> list[MeetingCase]:
    """OpenAI を使って meeting ケースを複数生成する。"""
    base_rng = random.Random(seed)
    return [
        generate_meeting_case(
            task_id=f"{task_id_prefix}_{index + 1:03d}",
            difficulty=difficulty,
            seed=base_rng.randint(0, 10**9),
            model=model,
        )
        for index in range(count)
    ]


def write_meeting_cases(
    output_dir: str | Path,
    count: int,
    difficulty: Difficulty = "medium",
    seed: int | None = None,
    task_id_prefix: str = "meeting_openai",
    model: str = DEFAULT_OPENAI_MODEL,
) -> list[Path]:
    """OpenAI で生成した meeting ケース群を JSON ファイルとして保存する。"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    case_paths: list[Path] = []
    for case in generate_meeting_cases(
        count=count,
        difficulty=difficulty,
        seed=seed,
        task_id_prefix=task_id_prefix,
        model=model,
    ):
        path = output_path / f"{case.task_id}.json"
        path.write_text(case.model_dump_json(indent=2), encoding="utf-8")
        case_paths.append(path)
    return case_paths


def parse_args() -> argparse.Namespace:
    """OpenAI 版 meeting ケース生成 CLI 用の引数を解釈する。"""
    parser = argparse.ArgumentParser(description="Generate meeting cases with OpenAI.")
    parser.add_argument("--count", type=int, default=5, help="Number of cases to generate.")
    parser.add_argument(
        "--difficulty",
        choices=("easy", "medium", "hard"),
        default="medium",
        help="Difficulty used for generated cases.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed.")
    parser.add_argument(
        "--model",
        default=DEFAULT_OPENAI_MODEL,
        help="OpenAI model name used for staged generation.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/generated/inaba/meeting_openai"),
        help="Directory where generated case JSON files are written.",
    )
    parser.add_argument(
        "--task-id-prefix",
        default="meeting_openai",
        help="Prefix used for generated task_id values.",
    )
    parser.add_argument(
        "--all-difficulties",
        action="store_true",
        help="Generate easy / medium / hard in sequence with difficulty suffix in task_id.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI から OpenAI 版 meeting ケース生成を実行する。"""
    args = parse_args()
    difficulties: tuple[Difficulty, ...] = (
        ("easy", "medium", "hard") if args.all_difficulties else (args.difficulty,)
    )
    all_paths: list[Path] = []
    for difficulty in difficulties:
        prefix = (
            f"{args.task_id_prefix}_{difficulty}" if args.all_difficulties else args.task_id_prefix
        )
        paths = write_meeting_cases(
            output_dir=args.output_dir,
            count=args.count,
            difficulty=difficulty,
            seed=args.seed,
            task_id_prefix=prefix,
            model=args.model,
        )
        all_paths.extend(paths)
    print(f"generated {len(all_paths)} meeting cases into {args.output_dir}")
    for path in all_paths:
        print(path)


def _call_openai_json(model: str, prompt: str) -> dict[str, Any]:
    """OpenAI Responses API を呼び出し、JSON オブジェクトとして解釈する。"""
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")

    client = OpenAI()
    response = client.responses.create(
        model=model,
        instructions=OPENAI_INSTRUCTIONS,
        input=prompt,
    )
    return _parse_json_text(response.output_text)


def _parse_json_text(text: str) -> dict[str, Any]:
    """JSON 文字列だけを安全側に切り出して辞書へ変換する。"""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return json.loads(stripped)


def _generate_meeting_plan(
    model: str,
    difficulty: Difficulty,
    seed: int | None,
    participant_count: int,
    event_count: int,
    slot_catalog: list[Slot],
) -> MeetingPlan:
    """Step1: 参加者、event 種別、do_exp、event 文面を OpenAI に生成させる。"""
    prompt = f"""
We are creating a difficulty={difficulty} meeting scheduling case. Reference seed: {seed}.
Number of participants: {participant_count}, number of events: {event_count}.
Available slot catalog:
{json.dumps(_slot_catalog_to_prompt(slot_catalog), ensure_ascii=False)}

Return JSON satisfying the following:
- participant_names: {participant_count} unique English first names (no duplicates)
- need_duration: one of 30, 45, or 60
- events: exactly {event_count} items
- each event has: target_person, event_type, message, exp_do_slot_id, constraint_hint
- event_type: one of schedule_change, schedule_pause, info_update, context_note
- exp_do_slot_id: chosen from the slot catalog above
- message: natural English sentence
- mix events with actual changes ("schedule changed", "Thursday is now free") and
  context-only events where no schedule actually changes
- include at least one schedule_change or schedule_pause
- include at least one info_update or context_note
- constraint_hint: English hint only for info_update events with useful context; null otherwise

Return format:
{{
  "participant_names": ["Alice", "Bob", "Carol"],
  "need_duration": 45,
  "events": [
    {{
      "target_person": "Alice",
      "event_type": "schedule_change",
      "message": "Alice's schedule has changed. Morning slots may need to be reconsidered.",
      "exp_do_slot_id": "S03",
      "constraint_hint": null
    }}
  ]
}}
""".strip()
    return MeetingPlan.model_validate(_call_openai_json(model=model, prompt=prompt))


def _generate_schedule_plan(
    model: str,
    difficulty: Difficulty,
    seed: int | None,
    meeting_plan: MeetingPlan,
    slot_catalog: list[Slot],
) -> MeetingSchedulePlan:
    """Step2: 初期状態と各 event 時点の全員スケジュールを OpenAI に組み直させる。"""
    prompt = f"""
We are creating a difficulty={difficulty} meeting scheduling case. Reference seed: {seed}.
Participants: {json.dumps(meeting_plan.participant_names, ensure_ascii=False)}.
Meeting duration: {meeting_plan.need_duration} minutes.
Event plan: {json.dumps(meeting_plan.model_dump(), ensure_ascii=False)}.
Available slots: {json.dumps(_slot_catalog_to_prompt(slot_catalog), ensure_ascii=False)}.

Return JSON satisfying the following:
- return initial_people and event_people
- initial_people: schedule for all participants
- event_people: same length as number of events
- each person has: name, constraints, slot_ids
- slot_ids must only use IDs from the slot list above
- at every time point (initial and each event), all participants share at least one common slot
- for each event_people[i], the event's exp_do_slot_id must be in all participants' common slots
- exp_do_slot_id must be the earliest common slot at that time point
- for schedule_change / schedule_pause, change target_person's slot_ids from the previous step
- for info_update, you may add a new constraint to target_person's constraints
- for context_note, schedules may remain unchanged
- each person must have between 3 and 6 slot_ids

Return format:
{{
  "initial_people": [
    {{
      "name": "Alice",
      "constraints": ["Mornings are generally flexible"],
      "slot_ids": ["S01", "S03", "S06"]
    }}
  ],
  "event_people": [
    {{
      "people": [
        {{
          "name": "Alice",
          "constraints": ["Thursday afternoons are relatively open"],
          "slot_ids": ["S03", "S06", "S08"]
        }}
      ]
    }}
  ]
}}
""".strip()
    return MeetingSchedulePlan.model_validate(_call_openai_json(model=model, prompt=prompt))


def _normalize_meeting_plan(
    meeting_plan: MeetingPlan,
    difficulty: Difficulty,
    event_count: int,
    slot_catalog: list[Slot],
) -> MeetingPlan:
    """OpenAI が返した meeting 設計を件数・型の面で整える。"""
    participant_count = PARTICIPANT_COUNT_BY_DIFFICULTY[difficulty]
    participant_names: list[str] = []
    for name in meeting_plan.participant_names:
        cleaned = str(name).strip()
        if cleaned and cleaned not in participant_names:
            participant_names.append(cleaned)

    for name in NAME_POOL:
        if len(participant_names) >= participant_count:
            break
        if name not in participant_names:
            participant_names.append(name)
    participant_names = participant_names[:participant_count]

    need_duration = meeting_plan.need_duration if meeting_plan.need_duration in {30, 45, 60} else 45
    valid_slot_catalog = [
        slot for slot in slot_catalog if _slot_duration_minutes(slot) == need_duration
    ]
    if not valid_slot_catalog:
        raise ValueError("meeting openai generation failed: no valid slots for need_duration")
    slot_ids = {slot.slot_id for slot in valid_slot_catalog}
    events = list(meeting_plan.events[:event_count])
    fallback_types: list[EventType] = [
        "schedule_change",
        "info_update",
        "schedule_pause",
        "context_note",
    ]
    while len(events) < event_count:
        index = len(events)
        events.append(
            MeetingEventPlan(
                target_person=participant_names[index % len(participant_names)],
                event_type=fallback_types[index % len(fallback_types)],
                message=(
                    f"There is an update regarding "
                    f"{participant_names[index % len(participant_names)]}'s situation."
                ),
                exp_do_slot_id=valid_slot_catalog[
                    min(index, len(valid_slot_catalog) - 1)
                ].slot_id,
                constraint_hint=None,
            )
        )

    normalized_events: list[MeetingEventPlan] = []
    for index, event in enumerate(events):
        target_person = (
            event.target_person
            if event.target_person in participant_names
            else participant_names[index % len(participant_names)]
        )
        exp_do_slot_id = (
            event.exp_do_slot_id
            if event.exp_do_slot_id in slot_ids
            else valid_slot_catalog[min(index, len(valid_slot_catalog) - 1)].slot_id
        )
        message = event.message.strip() or f"There has been a schedule update for {target_person}."
        normalized_events.append(
            MeetingEventPlan(
                target_person=target_person,
                event_type=event.event_type,
                message=message,
                exp_do_slot_id=exp_do_slot_id,
                constraint_hint=event.constraint_hint,
            )
        )

    has_schedule_change = any(
        event.event_type in {"schedule_change", "schedule_pause"}
        for event in normalized_events
    )
    if not has_schedule_change:
        first = normalized_events[0]
        normalized_events[0] = MeetingEventPlan(
            target_person=first.target_person,
            event_type="schedule_change",
            message=f"{first.target_person}'s schedule has changed. Please review the candidates.",
            exp_do_slot_id=first.exp_do_slot_id,
            constraint_hint=None,
        )
    has_context_event = any(
        event.event_type in {"info_update", "context_note"}
        for event in normalized_events
    )
    if not has_context_event:
        last = normalized_events[-1]
        normalized_events[-1] = MeetingEventPlan(
            target_person=last.target_person,
            event_type="info_update",
            message=(
                f"It turns out {last.target_person} is relatively flexible on Thursdays."
            ),
            exp_do_slot_id=last.exp_do_slot_id,
            constraint_hint="Thursday is relatively flexible",
        )

    return MeetingPlan(
        participant_names=participant_names,
        need_duration=need_duration,
        events=normalized_events,
    )


def _normalize_schedule_plan(
    schedule_plan: MeetingSchedulePlan,
    event_count: int,
) -> MeetingSchedulePlan:
    """OpenAI が返した schedule plan の event 件数をそろえる。"""
    event_people = list(schedule_plan.event_people[:event_count])
    while len(event_people) < event_count:
        event_people.append(ScheduleSnapshotDraft(people=[]))
    return MeetingSchedulePlan(
        initial_people=schedule_plan.initial_people,
        event_people=event_people,
    )


def _build_people_snapshots(
    rng: random.Random,
    schedule_plan: MeetingSchedulePlan,
    meeting_plan: MeetingPlan,
    slot_catalog: list[Slot],
) -> list[list[PeopleInfo]]:
    """OpenAI のスケジュール下書きから初期状態と event ごとの状態を組み立てる。"""
    slot_by_id = {
        slot.slot_id: slot
        for slot in slot_catalog
        if _slot_duration_minutes(slot) == meeting_plan.need_duration
    }
    if not slot_by_id:
        raise ValueError("meeting openai generation failed: no slots matching need_duration")

    valid_slot_ids = list(slot_by_id.keys())
    initial_people = _draft_people_to_people(
        drafts=schedule_plan.initial_people,
        participant_names=meeting_plan.participant_names,
        slot_by_id=slot_by_id,
        valid_slot_ids=valid_slot_ids,
    )
    initial_people = _ensure_shared_slot_snapshot(
        people=initial_people,
        slot_by_id=slot_by_id,
        preferred_slot_id=valid_slot_ids[0],
    )

    snapshots: list[list[PeopleInfo]] = [initial_people]
    previous_people = initial_people
    for event_plan, snapshot_draft in zip(
        meeting_plan.events,
        schedule_plan.event_people,
        strict=True,
    ):
        current_people = _draft_people_to_people(
            drafts=snapshot_draft.people,
            participant_names=meeting_plan.participant_names,
            slot_by_id=slot_by_id,
            valid_slot_ids=valid_slot_ids,
        )
        current_people = _reflect_event_change_if_needed(
            rng=rng,
            previous_people=previous_people,
            current_people=current_people,
            event_plan=event_plan,
            slot_by_id=slot_by_id,
        )
        current_people = _ensure_shared_slot_snapshot(
            people=current_people,
            slot_by_id=slot_by_id,
            preferred_slot_id=event_plan.exp_do_slot_id,
        )
        snapshots.append(current_people)
        previous_people = current_people
    return snapshots


def _draft_people_to_people(
    drafts: list[PersonScheduleDraft],
    participant_names: list[str],
    slot_by_id: dict[str, Slot],
    valid_slot_ids: list[str],
) -> list[PeopleInfo]:
    """OpenAI の1時点分の下書きを PeopleInfo 群へ変換する。"""
    draft_map = {draft.name: draft for draft in drafts}
    people: list[PeopleInfo] = []

    for index, name in enumerate(participant_names):
        draft = draft_map.get(name)
        slot_ids = _normalize_slot_ids(
            raw_slot_ids=draft.slot_ids if draft else [],
            valid_slot_ids=valid_slot_ids,
            fallback_slot_ids=valid_slot_ids[index:index + 3] or valid_slot_ids[:3],
        )
        attendances = sorted(
            [slot_by_id[slot_id].to_attendance() for slot_id in slot_ids if slot_id in slot_by_id],
            key=lambda slot: slot.when_from or datetime.max,
        )
        constraints = list(draft.constraints) if draft else []
        people.append(
            PeopleInfo(
                name=name,
                can_attend=attendances,
                constraints=constraints,
                person_index=index,
            )
        )
    return people


def _normalize_slot_ids(
    raw_slot_ids: list[str],
    valid_slot_ids: list[str],
    fallback_slot_ids: list[str],
) -> list[str]:
    """slot_ids を重複なく有効IDだけにそろえる。"""
    normalized: list[str] = []
    valid = set(valid_slot_ids)
    for slot_id in raw_slot_ids:
        if slot_id in valid and slot_id not in normalized:
            normalized.append(slot_id)

    for slot_id in fallback_slot_ids:
        if len(normalized) >= 3:
            break
        if slot_id in valid and slot_id not in normalized:
            normalized.append(slot_id)

    return normalized[:6]


def _reflect_event_change_if_needed(
    rng: random.Random,
    previous_people: list[PeopleInfo],
    current_people: list[PeopleInfo],
    event_plan: MeetingEventPlan,
    slot_by_id: dict[str, Slot],
) -> list[PeopleInfo]:
    """schedule_change / pause / info_update が state_before に反映されるよう補正する。"""
    previous_map = {person.name: person for person in previous_people}
    current_map = {person.name: person for person in current_people}
    target = current_map[event_plan.target_person]
    previous_target = previous_map[event_plan.target_person]

    previous_keys = {
        (slot.where, slot.when_from, slot.when_to) for slot in previous_target.can_attend
    }
    current_keys = {
        (slot.where, slot.when_from, slot.when_to) for slot in target.can_attend
    }

    needs_schedule_reflection = event_plan.event_type in {"schedule_change", "schedule_pause"}
    if needs_schedule_reflection and previous_keys == current_keys:
        if len(target.can_attend) > 3:
            target.can_attend = target.can_attend[1:]
        else:
            extra_slot = slot_by_id[rng.choice(list(slot_by_id.keys()))]
            if extra_slot.key not in current_keys:
                target.can_attend.append(extra_slot.to_attendance())
        target.can_attend = sorted(
            target.can_attend,
            key=lambda slot: slot.when_from or datetime.max,
        )

    if event_plan.event_type == "schedule_pause":
        exp_slot_key = slot_by_id[event_plan.exp_do_slot_id].key
        target.can_attend = [
            slot
            for slot in target.can_attend
            if (slot.where, slot.when_from, slot.when_to) != exp_slot_key
        ]
        target.can_attend = target.can_attend[: max(2, len(target.can_attend) - 1)]
        target.can_attend.append(slot_by_id[event_plan.exp_do_slot_id].to_attendance())
        target.can_attend = sorted(
            target.can_attend,
            key=lambda slot: slot.when_from or datetime.max,
        )

    if event_plan.event_type == "info_update" and event_plan.constraint_hint:
        if event_plan.constraint_hint not in target.constraints:
            target.constraints.append(event_plan.constraint_hint)

    return [current_map[person.name] for person in current_people]


def _ensure_shared_slot_snapshot(
    people: list[PeopleInfo],
    slot_by_id: dict[str, Slot],
    preferred_slot_id: str,
) -> list[PeopleInfo]:
    """指定スロットが共通候補になり、かつ最早になるように各人の候補を微調整する。"""
    preferred_slot = slot_by_id[preferred_slot_id]
    for person in people:
        person_keys = {
            (slot.where, slot.when_from, slot.when_to) for slot in person.can_attend
        }
        if preferred_slot.key not in person_keys:
            person.can_attend.append(preferred_slot.to_attendance())

    while True:
        best_slot = _find_best_slot(people)
        if best_slot is None or best_slot.key == preferred_slot.key:
            break
        first_person = people[0]
        first_person.can_attend = [
            slot
            for slot in first_person.can_attend
            if (slot.where, slot.when_from, slot.when_to) != best_slot.key
        ]

    for person in people:
        unique_slots: dict[tuple[str | None, datetime | None, datetime | None], Attendance] = {}
        for slot in person.can_attend:
            unique_slots[(slot.where, slot.when_from, slot.when_to)] = slot
        person.can_attend = sorted(
            unique_slots.values(),
            key=lambda slot: slot.when_from or datetime.max,
        )[:6]
    return people


def _slot_catalog_to_prompt(slot_catalog: list[Slot]) -> list[dict[str, str]]:
    """OpenAI プロンプトに渡しやすいスロット一覧へ変換する。"""
    return [
        {
            "slot_id": slot.slot_id,
            "where": slot.where,
            "when_from": slot.when_from.isoformat(),
            "when_to": slot.when_to.isoformat(),
        }
        for slot in slot_catalog
    ]


def _slot_duration_minutes(slot: Slot) -> int:
    """スロット長を分単位で返す。"""
    return int((slot.when_to - slot.when_from).total_seconds() // 60)


def _next_weekday(today: date) -> date:
    """指定日より後で最初に来る平日を返す。"""
    candidate = today + timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate += timedelta(days=1)
    return candidate


def _build_slot_catalog(
    start_day: date,
    count: int,
) -> list[Slot]:
    """OpenAI に選ばせる会議候補スロット一覧を作る。"""
    hour_candidates = [9, 10, 11, 13, 14, 15, 16]
    duration_candidates = [30, 45, 60]
    slots: list[Slot] = []
    offset = 0

    while len(slots) < count:
        meeting_day = start_day + timedelta(days=offset)
        if meeting_day.weekday() >= 5:
            offset += 1
            continue
        hour = hour_candidates[len(slots) % len(hour_candidates)]
        duration = duration_candidates[len(slots) % len(duration_candidates)]
        minute = 30 if duration == 45 and hour in {10, 14} else 0
        when_from = datetime.combine(meeting_day, time(hour=hour, minute=minute))
        when_to = when_from + timedelta(minutes=duration)
        slots.append(
            Slot(
                slot_id=f"S{len(slots) + 1:02d}",
                where=PLACE_POOL[len(slots) % len(PLACE_POOL)],
                when_from=when_from,
                when_to=when_to,
            )
        )
        if len(slots) % 3 == 0:
            offset += 1
    return sorted(slots, key=lambda slot: slot.when_from)


def _build_constraints(need_duration: int) -> list[str]:
    """meeting ケース共通の制約文を組み立てる。"""
    return [
        template.format(need_duration=need_duration)
        for template in MEETING_CONSTRAINTS_TEMPLATE
    ]


def _build_partial_state(
    people: list[PeopleInfo],
    need_duration: int,
    constraint: list[str],
    hidden_names: set[str],
    hidden_slots_per_person: int = 1,
) -> MeetingState:
    """完全情報から、見えている情報と隠れている情報を分離した State を作る。"""
    visible_people: list[PeopleInfo] = []
    hidden_people: list[PeopleInfo] = []

    for index, person in enumerate(people):
        hidden_indexes = (
            list(range(min(hidden_slots_per_person, len(person.can_attend))))
            if person.name in hidden_names
            else []
        )
        visible_slots: list[Attendance] = []
        hidden_slots: list[Attendance] = []

        for attendance_index, slot in enumerate(person.can_attend):
            should_hide_where = attendance_index in hidden_indexes
            visible_slots.append(
                Attendance(
                    where=None if should_hide_where else slot.where,
                    when_from=slot.when_from,
                    when_to=slot.when_to,
                )
            )
            if should_hide_where:
                hidden_slots.append(
                    Attendance(
                        where=slot.where,
                        when_from=None,
                        when_to=None,
                    )
                )

        visible_people.append(
            PeopleInfo(
                name=person.name,
                can_attend=visible_slots,
                constraints=person.constraints,
                person_index=index,
                hidden_attendance_indexes=hidden_indexes,
            )
        )
        if hidden_indexes:
            hidden_people.append(
                PeopleInfo(
                    name=person.name,
                    can_attend=hidden_slots,
                    constraints=person.constraints,
                    person_index=index,
                    hidden_attendance_indexes=hidden_indexes,
                )
            )

    full_info = MeetingInfo(
        need_duration=need_duration,
        state_people=[
            PeopleInfo(
                name=person.name,
                can_attend=[
                    Attendance(
                        where=slot.where,
                        when_from=slot.when_from,
                        when_to=slot.when_to,
                    )
                    for slot in person.can_attend
                ],
                constraints=person.constraints,
                person_index=person.person_index,
            )
            for person in people
        ],
    )
    return MeetingState(
        constraint=list(constraint),
        hidden_info=MeetingInfo(
            need_duration=need_duration,
            state_people=hidden_people,
        ),
        info=full_info,
        got_info=MeetingInfo(
            need_duration=need_duration,
            state_people=visible_people,
        ),
    )


def _build_requires_from_people(people: list[PeopleInfo]) -> list[str]:
    """隠された参加候補に対して、追加で確認すべき項目名を作る。"""
    requires: list[str] = []
    for person in people:
        for hidden_index in person.hidden_attendance_indexes:
            requires.append(f"{person.name}'s attendance slot {hidden_index + 1} location")
    return requires


def _build_requires_from_state(state: MeetingState) -> list[str]:
    """meeting の hidden_info から追加質問項目を取り出す。"""
    return _build_requires_from_people(state.hidden_info.state_people)


def _build_initial_request(
    people: list[PeopleInfo],
    visible_state: MeetingInfo,
    need_duration: int,
) -> str:
    """ユーザーに見せる初期依頼文を自然文で組み立てる。"""
    participant_names = ", ".join(person.name for person in people)
    hidden_names = ", ".join(
        person.name
        for person in visible_state.state_people
        if person.hidden_attendance_indexes
    )
    return (
        f"We want to schedule a {need_duration}-minute meeting with {participant_names}."
        " Attendance candidates are partially visible,"
        f" but some slots for {hidden_names} are missing location information."
        " Select the earliest available slot and ask for missing information if needed."
    )


def _find_best_slot(people: list[PeopleInfo]) -> Slot | None:
    """全参加者に共通する候補のうち、最も早いものを返す。"""
    if not people:
        return None
    shared_keys: set[tuple[str, datetime, datetime]] | None = None
    for person in people:
        person_keys = {
            (slot.where, slot.when_from, slot.when_to)
            for slot in person.can_attend
            if slot.where and slot.when_from and slot.when_to
        }
        if shared_keys is None:
            shared_keys = person_keys
        else:
            shared_keys &= person_keys
    if not shared_keys:
        return None

    best_key = min(shared_keys, key=lambda key: key[1])
    return Slot(
        slot_id="shared",
        where=best_key[0],
        when_from=best_key[1],
        when_to=best_key[2],
    )


def _question_count(difficulty: Difficulty) -> int:
    """難易度ごとの追加確認項目数を返す。"""
    return INITIAL_REQUIRE_COUNT_BY_DIFFICULTY[difficulty]


if __name__ == "__main__":
    main()
