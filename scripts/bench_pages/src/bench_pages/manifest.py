from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from bench_pages.docker_metadata import case_docker_image_metadata
from bench_pages.models import DockerImageRecord, SweepCase, SweepData, SweepRun


def build_sweep_id(sweep: SweepData) -> str:
    axis_part = _slug("-".join(_axis_names(sweep)) or "no-axes")
    commit_part = _short_commit(_git_commit(sweep) or "unknown")
    fixture_part = _slug(_fixture_identity(sweep) or "unknown-fixture")
    matrix_hash = _matrix_hash(sweep.matrix)
    return f"{sweep.execution_mode}-{axis_part}-{fixture_part}-{commit_part}-{matrix_hash}"


def build_manifest(
    sweep: SweepData,
    docker_images: list[DockerImageRecord] | None = None,
) -> dict[str, Any]:
    sweep_id = build_sweep_id(sweep)
    docker_image_records = docker_images or _collect_docker_images(sweep)

    manifest = {
        "sweep": {
            "id": sweep_id,
            "source_sweep_path": str(sweep.root),
            "execution_mode": sweep.execution_mode,
            "fixture_identity": _fixture_identity(sweep),
            "git_commit": _git_commit(sweep),
            "build_profile": _build_profile(sweep),
            "axis_names": _axis_names(sweep),
            "verdict": sweep.verdict,
            "schema_version": sweep.schema_version,
        },
        "source_artifacts": {
            "matrix": sweep.matrix,
            "matrix_expanded": sweep.matrix_expanded,
            "schedule": sweep.schedule,
            "comparison": sweep.comparison,
            "verdict": sweep.verdict,
        },
        "top_level_artifacts": [
            _artifact_dict(record, sweep_id=sweep_id)
            for record in sweep.top_level_artifacts
        ],
        "artifact_bundle": _artifact_bundle_dict(sweep_id),
        "cases": [_case_manifest(case, sweep_id=sweep_id) for case in sweep.cases],
        "docker_images": [asdict(record) for record in docker_image_records],
        "artifact_inventory": [
            _artifact_dict(record, sweep_id=sweep_id)
            for record in sweep.artifact_inventory
        ],
    }
    return manifest


def _case_manifest(case: SweepCase, sweep_id: str) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "execution_mode": case.execution_mode,
        "axis_assignments": case.axis_assignments,
        "requested_case": case.requested_case,
        "resolved_case": case.resolved_case,
        "summary": case.summary,
        "verdict": case.verdict,
        "provenance": case.provenance,
        "cpu_profile": _cpu_profile_manifest(case, sweep_id=sweep_id),
        "validation": case.validation,
        "comparison_case": case.comparison_case,
        "artifacts": [_artifact_dict(record, sweep_id=sweep_id) for record in case.artifacts],
        "runs": [_run_manifest(run, sweep_id=sweep_id) for run in case.runs],
    }


def _run_manifest(run: SweepRun, sweep_id: str) -> dict[str, Any]:
    return {
        "run_id": run.run_id,
        "result": run.result,
        "artifacts": [_artifact_dict(record, sweep_id=sweep_id) for record in run.artifacts],
    }


def _artifact_dict(record: Any, sweep_id: str) -> dict[str, Any]:
    return {
        "relative_path": record.relative_path,
        "size_bytes": record.size_bytes,
        "href": _artifact_href(sweep_id, record.relative_path),
    }


def _artifact_bundle_dict(sweep_id: str) -> dict[str, Any]:
    filename = f"{sweep_id}-artifacts.tar.gz"
    return {
        "filename": filename,
        "href": f"sweeps/{sweep_id}/{filename}",
        "size_bytes": None,
    }


def _cpu_profile_manifest(case: SweepCase, sweep_id: str) -> dict[str, Any] | None:
    if not case.cpu_profile:
        return None

    output_relative_path = str(case.cpu_profile["output_relative_path"])
    symbol_dir_relative_path = str(case.cpu_profile["symbol_dir_relative_path"])
    symbol_binary_relative_path = str(case.cpu_profile["symbol_binary_relative_path"])
    published_profile_path = _case_relative_path(case.case_id, output_relative_path)
    published_symbol_dir = _case_relative_path(case.case_id, symbol_dir_relative_path)
    published_symbol_binary = _case_relative_path(case.case_id, symbol_binary_relative_path)

    return {
        "profiler_kind": case.cpu_profile.get("profiler_kind"),
        "sample_rate_hz": case.cpu_profile.get("sample_rate_hz"),
        "execution_kind": case.cpu_profile.get("execution_kind"),
        "profile_artifact": {
            "relative_path": published_profile_path,
            "href": _artifact_href(sweep_id, published_profile_path),
        },
        "symbol_dir": {
            "relative_path": published_symbol_dir,
        },
        "symbol_binary": {
            "relative_path": published_symbol_binary,
            "href": _artifact_href(sweep_id, published_symbol_binary),
        },
        "load_command": (
            "samply load --symbol-dir "
            f"artifacts/{published_symbol_dir} "
            f"artifacts/{published_profile_path}"
        ),
    }


def _case_relative_path(case_id: str, relative_path: str) -> str:
    return str(Path("cases") / case_id / relative_path)


def _collect_docker_images(sweep: SweepData) -> list[DockerImageRecord]:
    records: list[DockerImageRecord] = []
    seen: set[str] = set()
    for case in sweep.cases:
        if case.execution_mode != "docker":
            continue
        digest, local_ref, identity = case_docker_image_metadata(case)
        if identity in seen:
            continue
        seen.add(identity)
        records.append(
            DockerImageRecord(
                canonical_identity=identity,
                local_image_ref=local_ref,
                provenance_image_digest=digest,
            )
        )
    return records


def _axis_names(sweep: SweepData) -> list[str]:
    comparison_axis_names = sweep.comparison.get("axis_names")
    if isinstance(comparison_axis_names, list):
        return [str(name) for name in comparison_axis_names]
    axes = sweep.matrix.get("axes", {})
    if isinstance(axes, dict):
        return [str(name) for name in axes.keys()]
    return []


def _fixture_identity(sweep: SweepData) -> str | None:
    first_case = _first_case(sweep)
    if first_case is None:
        return None

    for candidate in (
        first_case.resolved_case.get("fixture_sha256_hex"),
        first_case.provenance.get("fixture_sha256_hex"),
        first_case.requested_case.get("fixture_path"),
    ):
        if candidate:
            if isinstance(candidate, str) and "/" in candidate:
                return Path(candidate).name
            return str(candidate)
    return None


def _git_commit(sweep: SweepData) -> str | None:
    first_case = _first_case(sweep)
    if first_case is None:
        return None

    git = first_case.provenance.get("git", {})
    commit = git.get("commit")
    return str(commit) if commit else None


def _build_profile(sweep: SweepData) -> str | None:
    first_case = _first_case(sweep)
    if first_case is None:
        return None

    binary = first_case.provenance.get("binary", {})
    profile = binary.get("build_profile")
    return str(profile) if profile else None


def _first_case(sweep: SweepData) -> SweepCase | None:
    if not sweep.cases:
        return None
    return sweep.cases[0]


def _artifact_href(sweep_id: str, relative_path: str) -> str:
    return f"sweeps/{sweep_id}/artifacts/{relative_path}"


def _short_commit(commit: str) -> str:
    return _slug(commit[:7] if len(commit) >= 7 else commit)


def _matrix_hash(matrix: dict[str, Any]) -> str:
    encoded = json.dumps(matrix, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:8]


def _slug(value: str) -> str:
    slug = []
    previous_dash = False
    for char in value.lower():
        if char.isalnum():
            slug.append(char)
            previous_dash = False
            continue
        if previous_dash:
            continue
        slug.append("-")
        previous_dash = True
    return "".join(slug).strip("-") or "unknown"
