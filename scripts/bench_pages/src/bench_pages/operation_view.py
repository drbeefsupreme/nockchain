from __future__ import annotations

from collections.abc import Callable
from typing import Any

from bench_pages.value_stats import is_number, stats_scalar, stats_max


MetricFormatter = Callable[[int | float, str], str]
OptionalFormatter = Callable[[Any, str], str]
NestedMappingFinder = Callable[[Any, str], dict[str, Any] | None]


def missing_peek_status_reasons(case: dict[str, Any]) -> list[str]:
    reasons = []
    for step_type in ("peek_height", "peek_height_cold"):
        metrics = ((case.get("summary") or {}).get("by_step_type") or {}).get(step_type)
        if not isinstance(metrics, dict):
            continue
        missing_count = metrics.get("missing_count")
        missing = stats_scalar(missing_count)
        if not missing:
            continue
        max_missing = stats_max(missing_count)
        if max_missing is None:
            reasons.append(f"Missing peeks: {step_type} median {missing:.0f} per measured run.")
        else:
            reasons.append(
                f"Missing peeks: {step_type} median {missing:.0f}, max {max_missing:.0f} per measured run."
            )
    return reasons


def step_missing_count(case: dict[str, Any], step_type: str) -> float:
    by_step_type = (case.get("summary") or {}).get("by_step_type")
    if not isinstance(by_step_type, dict):
        return 0.0
    metrics = by_step_type.get(step_type)
    if not isinstance(metrics, dict):
        return 0.0
    return float(stats_scalar(metrics.get("missing_count")) or 0.0)


def summarize_plan_operations(
    steps: list[Any], summary: dict[str, Any]
) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    heights: dict[str, list[int]] = {}
    for step in steps:
        if not isinstance(step, dict):
            continue
        step_type = step.get("type")
        if not step_type:
            continue
        step_type = str(step_type)
        counts[step_type] = counts.get(step_type, 0) + 1
        height = step.get("height")
        if isinstance(height, int):
            heights.setdefault(step_type, []).append(height)

    by_step_type = summary.get("by_step_type")
    if isinstance(by_step_type, dict):
        for step_type, metrics in by_step_type.items():
            if not isinstance(metrics, dict) or step_type in counts:
                continue
            count = metrics.get("count_per_run")
            if is_number(count):
                counts[str(step_type)] = int(count)

    rows = []
    for step_type, count in sorted(counts.items()):
        step_heights = heights.get(step_type) or []
        if step_heights:
            height_range = (
                str(step_heights[0])
                if min(step_heights) == max(step_heights)
                else f"{min(step_heights)}-{max(step_heights)}"
            )
        else:
            height_range = "n/a"
        rows.append({"type": step_type, "count_raw": count, "range": height_range})
    return rows


def build_operation_health_rows(
    case_sections: list[dict[str, Any]],
    *,
    format_metric: MetricFormatter,
) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for section in case_sections:
        case = section["case"]
        by_step_type = (case.get("summary") or {}).get("by_step_type")
        if isinstance(by_step_type, dict):
            for step_type, metrics in by_step_type.items():
                if isinstance(metrics, dict):
                    grouped.setdefault(str(step_type), []).append(metrics)
        else:
            for row in section["operation_rows"]:
                grouped.setdefault(row["step_type"], []).append({"count_per_run": row["count"]})
    rows = []
    for step_type, metrics in sorted(grouped.items()):
        count_values = [stats_scalar(item.get("count_per_run")) for item in metrics]
        success_values = [stats_scalar(item.get("success_count")) for item in metrics]
        missing_values = [stats_scalar(item.get("missing_count")) for item in metrics]
        error_values = [stats_scalar(item.get("error_count")) for item in metrics]
        cold_values = [stats_scalar(item.get("cold_verified_count")) for item in metrics]
        duration_values = [stats_scalar(item.get("duration_ms")) for item in metrics]
        throughput_values = [
            stats_scalar(item.get("throughput_per_second")) for item in metrics
        ]
        missing_total = sum(value or 0 for value in missing_values)
        error_total = sum(value or 0 for value in error_values)
        health_class = "ok"
        if error_total > 0:
            health_class = "error"
        elif missing_total > 0:
            health_class = "warn"
        notes = []
        if step_type == "peek_height" and missing_total > 0:
            notes.append("peek throughput suppressed when misses are present")
        if step_type == "peek_height_cold":
            cold_range = format_number_range(
                [v for v in cold_values if v is not None],
                format_metric=format_metric,
            )
            if cold_range != "n/a":
                notes.append(f"cold verified {cold_range}")
        if not notes:
            notes.append("healthy")
        rows.append(
            {
                "step_type": step_type,
                "case_count": str(len(metrics)),
                "planned": format_number_range(
                    [v for v in count_values if v is not None],
                    format_metric=format_metric,
                ),
                "outcome": format_operation_outcome(
                    [v for v in success_values if v is not None],
                    [v for v in missing_values if v is not None],
                    [v for v in error_values if v is not None],
                    format_metric=format_metric,
                ),
                "duration": format_number_range(
                    [v for v in duration_values if v is not None],
                    "duration_ms",
                    format_metric=format_metric,
                ),
                "throughput": format_number_range(
                    [v for v in throughput_values if v is not None],
                    format_metric=format_metric,
                ),
                "notes": "; ".join(notes),
                "health_class": health_class,
            }
        )
    return rows


def format_operation_outcome(
    success_values: list[float | int],
    missing_values: list[float | int],
    error_values: list[float | int],
    *,
    format_metric: MetricFormatter,
) -> str:
    parts = []
    success = format_number_range(success_values, format_metric=format_metric)
    missing = format_number_range(missing_values, format_metric=format_metric)
    errors = format_number_range(error_values, format_metric=format_metric)
    if success != "n/a":
        parts.append(f"OK {success}")
    if missing != "n/a" and any(value > 0 for value in missing_values):
        parts.append(f"Missing {missing}")
    if errors != "n/a" and any(value > 0 for value in error_values):
        parts.append(f"Errors {errors}")
    return " · ".join(parts) if parts else "n/a"


def format_number_range(
    values: list[float | int],
    key: str = "count",
    *,
    format_metric: MetricFormatter,
) -> str:
    values = [value for value in values if is_number(value)]
    if not values:
        return "n/a"
    low = min(values)
    high = max(values)
    if low == high:
        return format_metric(low, key)
    return f"{format_metric(low, key)}-{format_metric(high, key)}"


def build_case_operation_rows(
    case: dict[str, Any],
    *,
    find_nested_mapping: NestedMappingFinder,
    format_optional: OptionalFormatter,
) -> list[dict[str, str]]:
    by_step_type = (case.get("summary") or {}).get("by_step_type")
    if not isinstance(by_step_type, dict):
        return build_planned_operation_rows(case, find_nested_mapping=find_nested_mapping)
    rows = []
    for step_type, metrics in sorted(by_step_type.items()):
        if not isinstance(metrics, dict):
            continue
        missing_count = stats_scalar(metrics.get("missing_count")) or 0.0
        throughput = metrics.get("throughput_per_second")
        if step_type in {"peek_height", "peek_height_cold"} and missing_count > 0:
            throughput = None
        rows.append(
            {
                "step_type": str(step_type),
                "count": format_optional(metrics.get("count_per_run"), ""),
                "duration": format_optional(metrics.get("duration_ms"), "duration_ms"),
                "throughput": format_optional(throughput, ""),
                "success": format_optional(metrics.get("success_count"), ""),
                "missing": format_optional(metrics.get("missing_count"), ""),
                "errors": format_optional(metrics.get("error_count"), ""),
                "cold": format_optional(metrics.get("cold_verified_count"), ""),
            }
        )
    return rows


def build_planned_operation_rows(
    case: dict[str, Any],
    *,
    find_nested_mapping: NestedMappingFinder,
) -> list[dict[str, str]]:
    plan = find_nested_mapping(case, "trusted_plan")
    steps = plan.get("steps") if plan else None
    if not isinstance(steps, list):
        return []
    counts: dict[str, int] = {}
    for step in steps:
        if not isinstance(step, dict):
            continue
        step_type = step.get("type")
        if not step_type:
            continue
        counts[str(step_type)] = counts.get(str(step_type), 0) + 1
    return [
        {
            "step_type": step_type,
            "count": str(count),
            "duration": "planned",
            "throughput": "n/a",
            "success": "n/a",
            "missing": "n/a",
            "errors": "n/a",
            "cold": "planned" if "cold" in step_type else "n/a",
        }
        for step_type, count in sorted(counts.items())
    ]
