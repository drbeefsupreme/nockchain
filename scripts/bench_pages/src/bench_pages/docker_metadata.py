from __future__ import annotations

from typing import Any

from bench_pages.models import SweepCase


def docker_payload(raw_value: Any) -> dict[str, Any]:
    if not isinstance(raw_value, dict):
        return {}

    for key in ("Docker", "docker"):
        candidate = raw_value.get(key)
        if isinstance(candidate, dict):
            return candidate
    return {}


def string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def case_docker_image_metadata(case: SweepCase) -> tuple[str | None, str | None, str]:
    provenance = docker_payload(case.provenance.get("backend"))
    requested = docker_payload(case.requested_case.get("execution"))
    resolved = case.resolved_case.get("docker", {})

    digest = string_or_none(provenance.get("image_digest"))
    local_ref = (
        string_or_none(provenance.get("image_tag"))
        or string_or_none(requested.get("image_tag"))
        or string_or_none(resolved.get("image_tag"))
    )
    canonical_identity = digest or local_ref or case.case_id
    return digest, local_ref, canonical_identity
