from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Literal

from ...models.meeting import (
    Attendance,
    MeetingCase,
    MeetingDo,
    MeetingEvent,
    MeetingInfo,
    MeetingState,
    PeopleInfo,
)

Difficulty = Literal["easy", "medium", "hard"]

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

NAME_POOL = [
    "田中",
    "佐藤",
    "鈴木",
    "高橋",
    "伊藤",
    "渡辺",
    "中村",
    "小林",
    "加藤",
    "吉田",
]

PLACE_POOL = [
    "online",
    "meeting_room_a",
    "meeting_room_b",
    "meeting_room_c",
    "conference_room_3f",
    "conference_room_4f",
]

PERSON_CONSTRAINT_POOL = [
    "移動があるため会議室(conference_room)は避けたい",
    "大体木曜日は、午後は空いていることが多い",
]


@dataclass(frozen=True)
class Slot:
    where: str
    when_from: datetime
    when_to: datetime

    def to_attendance(self) -> Attendance:
        """内部の時間枠表現を Attendance モデルへ変換する。"""
        return Attendance(where=self.where, when_from=self.when_from, when_to=self.when_to)

    @property
    def key(self) -> tuple[str, datetime, datetime]:
        """時間枠を一意に比較するためのキーを返す。"""
        return (self.where, self.when_from, self.when_to)


def generate_meeting_case(
    task_id: str,
    difficulty: Difficulty = "medium",
    seed: int | None = None,
) -> MeetingCase:
    """1件分の meeting ケースを生成する。"""
    rng = random.Random(seed)
    event_count = EVENT_COUNT_BY_DIFFICULTY[difficulty]
    initial_require_count = INITIAL_REQUIRE_COUNT_BY_DIFFICULTY[difficulty]

    participant_count = {"easy": 3, "medium": 4, "hard": 5}[difficulty]
    need_duration = rng.choice([30, 45, 60])
    start_day = _next_weekday(date.today())
    common_slots = _build_common_slots(
        rng=rng,
        start_day=start_day,
        need_duration=need_duration,
        count=event_count + 3,
    )

    people_full = _build_people(
        rng=rng,
        participant_count=participant_count,
        common_slots=common_slots,
        need_duration=need_duration,
    )
    constraint = _build_constraints(need_duration)

    hidden_people_count = max(1, initial_require_count // 2)
    hidden_names = {person.name for person in people_full[:hidden_people_count]}
    initial_state = _build_partial_state(
        people=people_full,
        need_duration=need_duration,
        constraint=constraint,
        hidden_names=hidden_names,
        hidden_slots_per_person=2,
    )

    initial_requires = _build_requires_from_state(initial_state)[:initial_require_count]
    initial_request = _build_initial_request(
        people=people_full,
        visible_state=initial_state.got_info,
        need_duration=need_duration,
    )

    events: list[MeetingEvent] = []
    current_people = people_full
    for index in range(event_count):
        current_people, event = _build_event(
            index=index,
            people=current_people,
            need_duration=need_duration,
            constraint=constraint,
        )
        events.append(event)

    return MeetingCase(
        task_id=task_id,
        difficulty=difficulty,
        initial_request=initial_request,
        initial_state=initial_state,
        events=events,
        initial_requires=initial_requires,
    )


def generate_meeting_cases(
    count: int,
    difficulty: Difficulty = "medium",
    seed: int | None = None,
    task_id_prefix: str = "meeting",
) -> list[MeetingCase]:
    """複数の meeting ケースをまとめて生成する。"""
    base_rng = random.Random(seed)
    return [
        generate_meeting_case(
            task_id=f"{task_id_prefix}_{index + 1:03d}",
            difficulty=difficulty,
            seed=base_rng.randint(0, 10**9),
        )
        for index in range(count)
    ]


def write_meeting_cases(
    output_dir: str | Path,
    count: int,
    difficulty: Difficulty = "medium",
    seed: int | None = None,
    task_id_prefix: str = "meeting",
) -> list[Path]:
    """生成した meeting ケース群を JSON ファイルとして保存する。"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    case_paths: list[Path] = []
    for case in generate_meeting_cases(
        count=count,
        difficulty=difficulty,
        seed=seed,
        task_id_prefix=task_id_prefix,
    ):
        path = output_path / f"{case.task_id}.json"
        path.write_text(case.model_dump_json(indent=2), encoding="utf-8")
        case_paths.append(path)
    return case_paths


def parse_args() -> argparse.Namespace:
    """meeting ケース生成 CLI 用の引数を解釈する。"""
    parser = argparse.ArgumentParser(description="Generate meeting cases for the inaba models.")
    parser.add_argument("--count", type=int, default=5, help="Number of cases to generate.")
    parser.add_argument(
        "--difficulty",
        choices=("easy", "medium", "hard"),
        default="medium",
        help="Difficulty used for generated cases.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/generated/inaba/meeting"),
        help="Directory where generated case JSON files are written.",
    )
    parser.add_argument(
        "--task-id-prefix",
        default="meeting",
        help="Prefix used for generated task_id values.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI から meeting ケース生成を実行する。"""
    args = parse_args()
    paths = write_meeting_cases(
        output_dir=args.output_dir,
        count=args.count,
        difficulty=args.difficulty,
        seed=args.seed,
        task_id_prefix=args.task_id_prefix,
    )
    print(f"generated {len(paths)} meeting cases into {args.output_dir}")
    for path in paths:
        print(path)


def _next_weekday(today: date) -> date:
    """指定日より後で最初に来る平日を返す。"""
    candidate = today + timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate += timedelta(days=1)
    return candidate


def _build_common_slots(
    rng: random.Random,
    start_day: date,
    need_duration: int,
    count: int,
) -> list[Slot]:
    """参加者全員に共通で候補になりうる会議枠を作る。"""
    hour_candidates = [9, 10, 11, 13, 14, 15, 16]
    slots: list[Slot] = []
    day_offset = 0

    while len(slots) < count:
        meeting_day = start_day + timedelta(days=day_offset)
        if meeting_day.weekday() >= 5:
            day_offset += 1
            continue
        hour = hour_candidates[len(slots) % len(hour_candidates)]
        minute = 30 if need_duration == 45 and hour in {10, 14} else 0
        start_at = datetime.combine(meeting_day, time(hour=hour, minute=minute))
        end_at = start_at + timedelta(minutes=need_duration)
        slots.append(
            Slot(
                where=PLACE_POOL[len(slots) % len(PLACE_POOL)],
                when_from=start_at,
                when_to=end_at,
            )
        )
        day_offset += 1 if len(slots) % 2 == 0 else 0
    rng.shuffle(slots[2:])
    slots[2:] = sorted(slots[2:], key=lambda slot: slot.when_from)
    return sorted(slots, key=lambda slot: slot.when_from)


def _build_people(
    rng: random.Random,
    participant_count: int,
    common_slots: list[Slot],
    need_duration: int,
) -> list[PeopleInfo]:
    """共通候補と個別候補を持つ参加者情報を生成する。"""
    names = rng.sample(NAME_POOL, k=participant_count)
    people: list[PeopleInfo] = []
    for index, name in enumerate(names):
        extra_slots = _build_extra_slots(
            rng=rng,
            base_slots=common_slots,
            need_duration=need_duration,
            count=2,
        )
        all_slots = sorted(
            [slot.to_attendance() for slot in common_slots + extra_slots],
            key=lambda attendance: attendance.when_from or datetime.max,
        )
        people.append(
            PeopleInfo(
                name=name,
                can_attend=all_slots,
                constraints=[rng.choice(PERSON_CONSTRAINT_POOL)],
                person_index=index,
            )
        )
    return people


def _build_extra_slots(
    rng: random.Random,
    base_slots: list[Slot],
    need_duration: int,
    count: int,
) -> list[Slot]:
    """各参加者だけが持つ追加候補枠を生成する。"""
    extra: list[Slot] = []
    base_day = base_slots[-1].when_from.date()
    for offset in range(count):
        meeting_day = base_day + timedelta(days=offset + 1)
        if meeting_day.weekday() >= 5:
            meeting_day += timedelta(days=7 - meeting_day.weekday())
        start_at = datetime.combine(
            meeting_day,
            time(hour=rng.choice([9, 11, 14, 16]), minute=0),
        )
        extra.append(
            Slot(
                where=rng.choice(PLACE_POOL),
                when_from=start_at,
                when_to=start_at + timedelta(minutes=need_duration),
            )
        )
    return extra


def _build_constraints(need_duration: int) -> list[str]:
    """meeting ケース共通の制約文を組み立てる。"""
    return [
        "参加者全員が同じ候補で参加できること",
        f"{need_duration} 分連続で確保できる候補のみ有効とする",
        "できるだけ早い時間帯の候補を優先する",
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

        if person.name in hidden_names:
            visible_people.append(
                PeopleInfo(
                    name=person.name,
                    can_attend=visible_slots,
                    constraints=person.constraints,
                    person_index=index,
                    hidden_attendance_indexes=hidden_indexes,
                )
            )
            hidden_people.append(
                PeopleInfo(
                    name=person.name,
                    can_attend=hidden_slots,
                    constraints=person.constraints,
                    person_index=index,
                    hidden_attendance_indexes=hidden_indexes,
                )
            )
        else:
            visible_people.append(
                PeopleInfo(
                    name=person.name,
                    can_attend=visible_slots,
                    constraints=person.constraints,
                    person_index=index,
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
            requires.append(f"{person.name}の参加候補{hidden_index + 1}の場所")
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
    participant_names = "、".join(person.name for person in people)
    hidden_names = "、".join(
        person.name
        for person in visible_state.state_people
        if person.hidden_attendance_indexes
    )
    return (
        f"{participant_names}で {need_duration} 分の会議を設定したいです。"
        " 参加可能な時間帯はかなり見えていますが、"
        f" {hidden_names} の一部候補は場所情報がまだ欠けています。"
        " できるだけ早い時間で、必要なら不足情報を確認しつつ場所と開始時刻を決めてください。"
    )


def _build_event(
    index: int,
    people: list[PeopleInfo],
    need_duration: int,
    constraint: list[str],
) -> tuple[list[PeopleInfo], MeetingEvent]:
    """途中で候補が崩れる event と、その後の最適状態を生成する。"""
    current_best = _find_best_slot(people)
    if current_best is None:
        raise ValueError("meeting event generation failed: no common slot before mutation")

    target_index = index % len(people)
    target_person = people[target_index]
    updated_people = _clone_people(people)
    updated_target = updated_people[target_index]
    updated_target.can_attend = [
        slot
        for slot in updated_target.can_attend
        if (slot.where, slot.when_from, slot.when_to) != current_best.key
    ]

    next_best = _find_best_slot(updated_people)
    if next_best is None:
        raise ValueError("meeting event generation failed: no common slot after mutation")

    state_before = _build_partial_state(
        people=updated_people,
        need_duration=need_duration,
        constraint=constraint,
        hidden_names={target_person.name},
        hidden_slots_per_person=2,
    )
    should_require = _build_requires_from_state(state_before)
    message = (
        f"{target_person.name}さんの予定に変更が入りました。"
        " これまで見ていた最速候補がそのまま使えるとは限らず、"
        " 一部候補の場所情報も未確認です。"
        " 最新候補を確認して、会議設定を見直してください。"
    )

    return updated_people, MeetingEvent(
        message=message,
        state_before=state_before,
        exp_do=MeetingDo(
            where=next_best.where,
            when_from=next_best.when_from,
            when_to=next_best.when_to,
        ),
        should_require=should_require,
    )


def _clone_people(people: list[PeopleInfo]) -> list[PeopleInfo]:
    """参加者情報を event 用に安全に複製する。"""
    return [
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
            constraints=list(person.constraints),
            person_index=person.person_index,
            hidden_attendance_indexes=list(person.hidden_attendance_indexes),
        )
        for person in people
    ]


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
    return Slot(where=best_key[0], when_from=best_key[1], when_to=best_key[2])


if __name__ == "__main__":
    main()
