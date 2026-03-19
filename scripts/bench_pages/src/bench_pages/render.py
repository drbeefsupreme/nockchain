from __future__ import annotations

import html
import json
import shutil
from importlib.resources import files
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape
from markupsafe import Markup


# -- Metric classification for UI tiers --

# Ordered list of summary metrics for the primary comparison table.
_COMPARISON_METRICS = [
    "throughput_blocks_per_second",
    "total_replay_time_secs",
    "init_time_secs",
    "average_block_time_ms",
    "peak_process_rss_bytes",
    "minor_faults_total",
    "major_faults_total",
    "measured_runs_requested",
    "measured_runs_succeeded",
]

# Ordered list of run-result keys for per-case run tables.
_RUN_KEY_ORDER = [
    "success",
    "throughput_blocks_per_second",
    "init_time_secs",
    "total_replay_time_secs",
    "average_block_time_ms",
    "peak_process_rss_bytes",
    "minor_faults_total",
    "major_faults_total",
    "blocks_poked",
    "failed_pokes",
    "checkpoint_count",
    "checkpoint_total_time_secs",
    "average_checkpoint_time_secs",
]

# Keys excluded from run tables (structural, not metric).
_RUN_EXCLUDE_KEYS = {"run_id", "error"}

# Human-readable labels for known metric keys.
_METRIC_LABELS: dict[str, str] = {
    "throughput_blocks_per_second": "Throughput",
    "total_replay_time_secs": "Replay",
    "init_time_secs": "Init",
    "average_block_time_ms": "Avg Block",
    "peak_process_rss_bytes": "Peak RSS",
    "minor_faults_total": "Minor Faults",
    "major_faults_total": "Major Faults",
    "measured_runs_requested": "Runs Req",
    "measured_runs_succeeded": "Runs OK",
    "failed_pokes": "Failed Pokes",
    "checkpoint_count": "Checkpoints",
    "average_checkpoint_time_secs": "Avg Ckpt",
    "checkpoint_total_time_secs": "Ckpt Total",
    "blocks_poked": "Blocks",
    "success": "OK",
}


# -- Public API --

def render_sweep_page(manifest: dict[str, Any]) -> str:
    template = _environment().get_template("sweep.html.j2")
    cases = manifest["cases"]
    comparison = _build_comparison_table(cases)
    case_sections = [_case_section(case) for case in cases]
    return template.render(
        manifest=manifest,
        sweep=manifest["sweep"],
        source_artifacts=manifest["source_artifacts"],
        top_level_artifacts=manifest.get("top_level_artifacts", []),
        comparison=comparison,
        case_sections=case_sections,
        docker_images=manifest["docker_images"],
        artifact_inventory=manifest["artifact_inventory"],
        render_value=_render_value_markup,
        pretty_json=_pretty_json,
    )


def render_index_page(entries: list[dict[str, Any]]) -> str:
    template = _environment().get_template("index.html.j2")
    return template.render(entries=entries)


def write_index_json(entries: list[dict[str, Any]], output_path: Path) -> None:
    output_path.write_text(json.dumps(entries, indent=2, sort_keys=True) + "\n")


def copy_assets(output_dir: Path) -> Path:
    target = output_dir / "assets"
    source = assets_dir()
    target.mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        destination = target / path.name
        if path.is_dir():
            shutil.copytree(path, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(path, destination)
    return target


def assets_dir() -> Path:
    return Path(str(files("bench_pages").joinpath("assets")))


# -- Template environment --

def _environment() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(files("bench_pages").joinpath("templates"))),
        autoescape=select_autoescape(("html", "xml")),
        trim_blocks=True,
        lstrip_blocks=True,
    )


# -- Primary comparison table --

def _build_comparison_table(cases: list[dict[str, Any]]) -> dict[str, Any]:
    all_keys: set[str] = set()
    for case in cases:
        all_keys.update(case["summary"].keys())

    columns: list[dict[str, str]] = []
    seen: set[str] = set()
    for key in _COMPARISON_METRICS:
        if key not in all_keys:
            continue
        if not any(_has_displayable_value(case["summary"].get(key)) for case in cases):
            continue
        columns.append({"key": key, "label": _METRIC_LABELS.get(key, key)})
        seen.add(key)

    for key in sorted(all_keys - seen - {"failed_runs"}):
        if not any(_has_displayable_value(case["summary"].get(key)) for case in cases):
            continue
        columns.append({"key": key, "label": _METRIC_LABELS.get(key, key)})

    rows = []
    for case in cases:
        cells = []
        for col in columns:
            value = case["summary"].get(col["key"])
            cells.append({"markup": _render_value_compact(value, col["key"])})

        verdict = case.get("verdict", {})
        verdict_label = (
            verdict.get("validity", "Unknown")
            if isinstance(verdict, dict)
            else str(verdict)
        )

        axis_parts = [
            f"{k}={v}" for k, v in case.get("axis_assignments", {}).items()
        ]
        axis_summary = ", ".join(axis_parts) if axis_parts else "\u2014"

        failed_runs = case["summary"].get("failed_runs", [])
        failed_count = len(failed_runs) if isinstance(failed_runs, list) else 0

        rows.append(
            {
                "case_id": case["case_id"],
                "axis_summary": axis_summary,
                "verdict_label": verdict_label,
                "failed_count": failed_count,
                "cells": cells,
            }
        )

    return {"columns": columns, "rows": rows}


def _has_displayable_value(value: Any) -> bool:
    if value is None:
        return False
    if _is_value_stats(value):
        return value.get("median") is not None
    return _is_number(value) or isinstance(value, str)


# -- Per-case sections --

def _case_section(case: dict[str, Any]) -> dict[str, Any]:
    run_tables = _build_run_tables(case["runs"])
    verdict = case.get("verdict", {})
    verdict_label = (
        verdict.get("validity", "Unknown")
        if isinstance(verdict, dict)
        else str(verdict)
    )
    return {
        "case": case,
        "verdict_label": verdict_label,
        "run_tables": run_tables,
        "summary_markup": _render_object_table(case["summary"]),
        "provenance_markup": _render_object_table(case["provenance"]),
        "requested_case_markup": _render_object_table(case["requested_case"]),
        "resolved_case_markup": _render_object_table(case["resolved_case"]),
        "verdict_markup": _render_object_table(case["verdict"]),
        "validation_markup": (
            _render_object_table(case["validation"])
            if case["validation"]
            else None
        ),
    }


def _build_run_tables(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not runs:
        return []

    all_keys: set[str] = set()
    for run in runs:
        all_keys.update((run.get("result") or {}).keys())
    all_keys -= _RUN_EXCLUDE_KEYS

    ordered: list[str] = []
    seen: set[str] = set()
    for key in _RUN_KEY_ORDER:
        if key in all_keys:
            ordered.append(key)
            seen.add(key)
    for key in sorted(all_keys - seen):
        ordered.append(key)

    columns = [
        {"key": key, "label": _METRIC_LABELS.get(key, key)} for key in ordered
    ]

    rows = []
    for run in runs:
        result = run.get("result") or {}
        cells = []
        for key in ordered:
            value = result.get(key)
            cells.append({"markup": _render_value_compact(value, key)})
        rows.append(
            {
                "run_id": run["run_id"],
                "cells": cells,
                "artifacts": run["artifacts"],
            }
        )

    return [{"columns": columns, "rows": rows}]


# -- Value rendering --

def _render_value_compact(value: Any, key: str = "") -> Markup:
    """Compact rendering for table cells.

    ValueStats: median as primary line, min/max/cv as secondary.
    """
    if value is None:
        return Markup('<span class="na">n/a</span>')
    if _is_value_stats(value):
        median = value.get("median")
        if median is None:
            return Markup('<span class="na">n/a</span>')
        primary = _format_metric(median, key)
        parts = []
        if value.get("min") is not None:
            parts.append(f"min {_format_metric(value['min'], key)}")
        if value.get("max") is not None:
            parts.append(f"max {_format_metric(value['max'], key)}")
        if value.get("cv") is not None:
            parts.append(f"cv {_format_number(value['cv'])}")
        secondary = " \u00b7 ".join(parts)
        return Markup(
            '<span class="vs-primary">{primary}</span>'
            '<span class="vs-detail">{secondary}</span>'.format(
                primary=html.escape(primary),
                secondary=html.escape(secondary),
            )
        )
    if isinstance(value, bool):
        css = "val-ok" if value else "val-fail"
        label = "true" if value else "false"
        return Markup(f'<span class="{css}">{html.escape(label)}</span>')
    if _is_number(value):
        return Markup(html.escape(_format_metric(value, key)))
    if isinstance(value, list):
        return Markup(html.escape(str(len(value))))
    return Markup(html.escape(str(value)))


def _render_object_table(value: Any) -> Markup:
    if not isinstance(value, dict):
        return _render_value_markup(value)
    rows = []
    for key, item in value.items():
        rows.append(
            "<tr><th>{key}</th><td>{value}</td></tr>".format(
                key=html.escape(str(key)),
                value=_render_value_markup(item),
            )
        )
    return Markup('<table class="kv-table">{rows}</table>'.format(rows="".join(rows)))


def _render_value_markup(value: Any) -> Markup:
    """Full-fidelity rendering for evidence drawers and detail views."""
    if value is None:
        return Markup('<span class="na">n/a</span>')
    if _is_value_stats(value):
        labels = ("median", "min", "max", "mad", "stddev", "cv", "values")
        rows = []
        for label in labels:
            rows.append(
                "<tr><th>{label}</th><td>{value}</td></tr>".format(
                    label=html.escape(label),
                    value=_render_value_markup(value.get(label)),
                )
            )
        return Markup(
            '<table class="valuestats">{rows}</table>'.format(rows="".join(rows))
        )
    if isinstance(value, dict):
        return _render_object_table(value)
    if isinstance(value, list):
        items = "".join(
            f"<li>{_render_value_markup(item)}</li>" for item in value
        )
        return Markup(f'<ul class="json-list">{items}</ul>')
    if isinstance(value, bool):
        return Markup(html.escape("true" if value else "false"))
    if _is_number(value):
        return Markup(html.escape(_format_number(value)))
    return Markup(html.escape(str(value)))


# -- Formatting helpers --

def _format_metric(value: int | float, key: str = "") -> str:
    if key.endswith("_bytes") and isinstance(value, (int, float)) and abs(value) >= 1024:
        return _humanize_bytes(value)
    return _format_number(value)


def _humanize_bytes(value: int | float) -> str:
    v = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(v) < 1024:
            if v == int(v):
                return f"{int(v)} {unit}"
            return f"{v:.1f} {unit}"
        v /= 1024
    return f"{v:.1f} PiB"


def _pretty_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True)


def _is_value_stats(value: Any) -> bool:
    return isinstance(value, dict) and {
        "median", "min", "max", "mad", "stddev", "cv", "values"
    }.issubset(value.keys())


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _format_number(value: int | float) -> str:
    if isinstance(value, int):
        return str(value)
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _infer_unit(key: str) -> str:
    if key.endswith("_bytes"):
        return "bytes"
    if key.endswith("_secs"):
        return "seconds"
    if key.endswith("_ms"):
        return "milliseconds"
    if key.endswith("_per_second"):
        return "per second"
    return ""
