from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from typing import Any

from bench_pages.errors import ExternalCommandError, ValidationError
from bench_pages.models import DockerImageRecord, SweepData


Runner = Callable[[list[str]], subprocess.CompletedProcess[str]]


def derive_ghcr_tag(provenance_digest: str) -> str:
    digest = provenance_digest.strip()
    if not digest:
        raise ValidationError("cannot derive GHCR tag without a provenance digest")
    if ":" in digest:
        algorithm, value = digest.split(":", 1)
        return f"{algorithm}-{value}"
    return digest.replace("/", "-")


def publish_docker_images(
    sweep: SweepData,
    owner: str,
    ghcr_package: str,
    publish: bool = True,
    runner: Runner | None = None,
) -> list[DockerImageRecord]:
    run = runner or _run_command
    records = _extract_docker_images(sweep)

    for record in records:
        if not record.provenance_image_digest:
            _enrich_from_local_inspect(record, run)
        if not record.ghcr_tag:
            if not record.provenance_image_digest:
                raise ValidationError("docker publication requires a provenance image digest")
            record.ghcr_tag = derive_ghcr_tag(record.provenance_image_digest)

        if not publish:
            record.publish_status = "planned"
            continue

        remote_ref = _ghcr_ref(owner, ghcr_package, record.ghcr_tag)
        if _remote_tag_exists(remote_ref, run):
            record.ghcr_digest = record.provenance_image_digest
            record.publish_status = "already-present"
            continue

        if not record.local_image_ref:
            raise ValidationError(
                f"cannot publish GHCR image {remote_ref} without a local Docker image reference"
            )

        _run_checked(run, ["docker", "tag", record.local_image_ref, remote_ref])
        _run_checked(run, ["docker", "push", remote_ref])
        record.ghcr_digest = record.provenance_image_digest
        record.publish_status = "pushed"

    return records


def _extract_docker_images(sweep: SweepData) -> list[DockerImageRecord]:
    records: list[DockerImageRecord] = []
    seen: set[str] = set()
    for case in sweep.cases:
        if case.execution_mode != "docker":
            continue
        requested = _docker_dict(case.requested_case.get("execution"))
        provenance = _docker_dict(case.provenance.get("backend"))
        resolved = case.resolved_case.get("docker", {})

        digest = _string_or_none(provenance.get("image_digest"))
        local_ref = (
            _string_or_none(provenance.get("image_tag"))
            or _string_or_none(requested.get("image_tag"))
            or _string_or_none(resolved.get("image_tag"))
        )
        canonical_identity = digest or local_ref or case.case_id
        if canonical_identity in seen:
            continue
        seen.add(canonical_identity)
        records.append(
            DockerImageRecord(
                canonical_identity=canonical_identity,
                local_image_ref=local_ref,
                provenance_image_digest=digest,
                ghcr_tag=derive_ghcr_tag(digest) if digest else None,
            )
        )
    return records


def _enrich_from_local_inspect(record: DockerImageRecord, runner: Runner) -> None:
    if not record.local_image_ref:
        raise ValidationError("cannot inspect local Docker metadata without a local image reference")
    inspect = _run_checked(runner, ["docker", "image", "inspect", record.local_image_ref])
    payload = json.loads(inspect.stdout or "[]")
    if not payload:
        raise ValidationError(f"docker image inspect returned no data for {record.local_image_ref}")
    entry = payload[0]
    record.local_image_id = _string_or_none(entry.get("Id"))
    record.local_image_size_bytes = entry.get("Size")
    record.canonical_identity = record.canonical_identity or record.local_image_id


def _remote_tag_exists(remote_ref: str, runner: Runner) -> bool:
    result = runner(["docker", "manifest", "inspect", remote_ref])
    return result.returncode == 0


def _ghcr_ref(owner: str, ghcr_package: str, tag: str) -> str:
    return f"ghcr.io/{owner}/{ghcr_package}:{tag}"


def _docker_dict(raw_value: Any) -> dict[str, Any]:
    if isinstance(raw_value, dict):
        if "Docker" in raw_value and isinstance(raw_value["Docker"], dict):
            return raw_value["Docker"]
        if "docker" in raw_value and isinstance(raw_value["docker"], dict):
            return raw_value["docker"]
    return {}


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=False, capture_output=True, text=True)


def _run_checked(
    runner: Runner,
    command: list[str],
) -> subprocess.CompletedProcess[str]:
    result = runner(command)
    if result.returncode != 0:
        raise ExternalCommandError(
            f"command failed ({result.returncode}): {' '.join(command)}\n{result.stderr}"
        )
    return result
