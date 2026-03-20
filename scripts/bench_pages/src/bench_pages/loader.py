from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from bench_pages.errors import ValidationError
from bench_pages.models import ArtifactRecord, SweepCase, SweepData, SweepRun


REQUIRED_TOP_LEVEL_FILES = (
    "comparison.json",
    "matrix.json",
    "matrix_expanded.json",
    "schedule.json",
    "verdict.json",
)
REQUIRED_CASE_FILES = (
    "provenance.json",
    "requested_case.json",
    "resolved_case.json",
    "summary.json",
    "verdict.json",
)


def load_sweep(root: Path) -> SweepData:
    sweep_root = root.resolve()
    if not sweep_root.is_dir():
        raise ValidationError(f"sweep root does not exist: {sweep_root}")

    sweep_artifacts = _load_required_json_files(sweep_root, REQUIRED_TOP_LEVEL_FILES)
    matrix = sweep_artifacts["matrix"]
    matrix_expanded = sweep_artifacts["matrix_expanded"]
    schedule = sweep_artifacts["schedule"]
    comparison = sweep_artifacts["comparison"]
    verdict = sweep_artifacts["verdict"]
    schema_version = _read_text_if_present(sweep_root / "schema_version.txt")

    artifact_inventory = _walk_artifacts(sweep_root)
    top_level_artifacts = [
        artifact
        for artifact in artifact_inventory
        if not artifact.relative_path.startswith("cases/")
    ]

    expanded_by_id = {
        case_entry.get("case_id"): case_entry
        for case_entry in matrix_expanded
        if isinstance(case_entry, dict) and case_entry.get("case_id")
    }
    comparison_by_id = {
        case_entry.get("case_id"): case_entry
        for case_entry in comparison.get("cases", [])
        if isinstance(case_entry, dict) and case_entry.get("case_id")
    }

    cases_root = sweep_root / "cases"
    cases: list[SweepCase] = []
    if cases_root.exists():
        for case_root in _sorted_child_dirs(cases_root):
            cases.append(
                _load_case(
                    sweep_root=sweep_root,
                    case_root=case_root,
                    expanded_case=expanded_by_id.get(case_root.name),
                    comparison_case=comparison_by_id.get(case_root.name),
                )
            )

    sweep_execution_mode = _normalize_execution_mode(
        [
            _normalize_execution_signal(matrix.get("base", {}).get("mode"), "matrix.base.mode"),
            *[
                _normalize_execution_signal(case.execution_mode, f"case {case.case_id}")
                for case in cases
            ],
        ]
    )

    return SweepData(
        root=sweep_root,
        execution_mode=sweep_execution_mode,
        schema_version=schema_version,
        matrix=matrix,
        matrix_expanded=matrix_expanded,
        schedule=schedule,
        comparison=comparison,
        verdict=verdict,
        cases=cases,
        artifact_inventory=artifact_inventory,
        top_level_artifacts=top_level_artifacts,
    )


def _load_case(
    sweep_root: Path,
    case_root: Path,
    expanded_case: dict[str, Any] | None,
    comparison_case: dict[str, Any] | None,
) -> SweepCase:
    case_artifacts = _load_required_json_files(case_root, REQUIRED_CASE_FILES)
    requested_case = case_artifacts["requested_case"]
    resolved_case = case_artifacts["resolved_case"]
    summary = case_artifacts["summary"]
    verdict = case_artifacts["verdict"]
    provenance = case_artifacts["provenance"]
    cpu_profile = _load_optional_json(case_root / "cpu_profile.json")
    validation = _load_optional_json(case_root / "validation.json")

    case_execution_mode = _normalize_execution_mode(
        [
            _normalize_execution_signal(
                requested_case.get("execution"),
                f"{case_root.name}/requested_case.json:execution",
            ),
            _normalize_execution_signal(
                provenance.get("backend"),
                f"{case_root.name}/provenance.json:backend",
            ),
        ]
    )

    runs = _load_runs(sweep_root, case_root / "runs")

    return SweepCase(
        case_id=case_root.name,
        root=case_root,
        execution_mode=case_execution_mode,
        axis_assignments=(expanded_case or {}).get("axis_assignments", {}),
        requested_case=requested_case,
        resolved_case=resolved_case,
        summary=summary,
        verdict=verdict,
        provenance=provenance,
        cpu_profile=cpu_profile,
        comparison_case=comparison_case,
        validation=validation,
        runs=runs,
        artifacts=_artifacts_under(sweep_root, case_root),
    )


def _validate_required_files(root: Path, required_files: tuple[str, ...]) -> None:
    missing = [name for name in required_files if not (root / name).is_file()]
    if missing:
        raise ValidationError(
            f"missing required files under {root}: {', '.join(sorted(missing))}"
        )


def _load_required_json_files(root: Path, filenames: tuple[str, ...]) -> dict[str, Any]:
    _validate_required_files(root, filenames)
    return {
        Path(filename).stem: _load_json(root / filename)
        for filename in filenames
    }


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise ValidationError(f"missing JSON artifact: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValidationError(f"invalid JSON artifact {path}: {exc}") from exc


def _read_text_if_present(path: Path) -> str | None:
    if not path.exists():
        return None
    return path.read_text().strip()


def _load_optional_json(path: Path) -> Any:
    if not path.exists():
        return None
    return _load_json(path)


def _load_runs(sweep_root: Path, runs_root: Path) -> list[SweepRun]:
    if not runs_root.exists():
        return []

    runs: list[SweepRun] = []
    for run_root in _sorted_child_dirs(runs_root):
        runs.append(
            SweepRun(
                run_id=run_root.name,
                root=run_root,
                result=_load_optional_json(run_root / "result.json"),
                artifacts=_artifacts_under(sweep_root, run_root),
            )
        )
    return runs


def _sorted_child_dirs(root: Path) -> list[Path]:
    return sorted(path for path in root.iterdir() if path.is_dir())


def _walk_artifacts(root: Path) -> list[ArtifactRecord]:
    return _artifact_records(root, relative_to=root)


def _artifacts_under(sweep_root: Path, sub_root: Path) -> list[ArtifactRecord]:
    return _artifact_records(sub_root, relative_to=sweep_root)


def _artifact_records(root: Path, relative_to: Path) -> list[ArtifactRecord]:
    return [
        ArtifactRecord(
            relative_path=str(path.relative_to(relative_to)),
            size_bytes=path.stat().st_size,
        )
        for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file())
    ]


def _normalize_execution_signal(raw_value: Any, source: str) -> str | None:
    if raw_value is None:
        return None
    if raw_value in ("Native", "native"):
        return "native"
    if isinstance(raw_value, str):
        if raw_value.lower() == "docker":
            return "docker"
        raise ValidationError(f"unsupported execution value at {source}: {raw_value!r}")
    if isinstance(raw_value, dict):
        if len(raw_value) != 1:
            raise ValidationError(f"ambiguous execution value at {source}: {raw_value!r}")
        tag = next(iter(raw_value))
        normalized = tag.lower()
        if normalized in {"native", "docker"}:
            return normalized
    raise ValidationError(f"unsupported execution value at {source}: {raw_value!r}")


def _normalize_execution_mode(signals: list[str | None]) -> str:
    modes = {signal for signal in signals if signal is not None}
    if not modes:
        raise ValidationError("unable to normalize execution mode from sweep artifacts")
    if len(modes) != 1:
        raise ValidationError(f"conflicting execution modes in sweep artifacts: {sorted(modes)}")
    return next(iter(modes))
