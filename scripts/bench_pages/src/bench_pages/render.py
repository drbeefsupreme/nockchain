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

# Keys that always appear in tables even if absent or null in data.
_ALWAYS_SHOW_KEYS = {"minor_faults_total", "major_faults_total"}

# Metrics that get strip charts showing per-case run spread.
_STRIP_CHART_METRICS = [
    ("throughput_blocks_per_second", "Throughput (blk/s)"),
    ("total_replay_time_secs", "Replay Time (s)"),
    ("minor_faults_total", "Minor Faults"),
]

# Human-readable labels with units for known metric keys.
_METRIC_LABELS: dict[str, str] = {
    "throughput_blocks_per_second": "Throughput (blk/s)",
    "total_replay_time_secs": "Replay (s)",
    "init_time_secs": "Init (s)",
    "average_block_time_ms": "Avg Block (ms)",
    "peak_process_rss_bytes": "Peak RSS",
    "minor_faults_total": "Minor Fault",
    "major_faults_total": "Major Fault",
    "measured_runs_requested": "Runs Req",
    "measured_runs_succeeded": "Runs OK",
    "failed_pokes": "Fld Pokes",
    "checkpoint_count": "Ckpts",
    "average_checkpoint_time_secs": "Avg Ckpt (s)",
    "checkpoint_total_time_secs": "Ckpt Tot (s)",
    "blocks_poked": "Blocks",
    "success": "OK",
}

# Hover tooltip descriptions for metric fields.
_FIELD_TOOLTIPS: dict[str, str] = {
    # Summary / comparison metrics
    "throughput_blocks_per_second": (
        "Blocks replayed per second. Higher is better."
    ),
    "total_replay_time_secs": (
        "Total wall-clock time for the block replay phase (seconds). Lower is better."
    ),
    "init_time_secs": (
        "Time to initialize the replay environment (seconds)."
    ),
    "average_block_time_ms": (
        "Average wall-clock time to replay one block (milliseconds). Lower is better."
    ),
    "peak_process_rss_bytes": (
        "Peak resident set size (physical memory) of the benchmark process."
    ),
    "minor_faults_total": (
        "Minor (soft) page faults: resolved from page cache without disk I/O."
    ),
    "major_faults_total": (
        "Major (hard) page faults: required disk I/O to resolve."
    ),
    "measured_runs_requested": (
        "Number of measured benchmark runs requested by the sweep matrix."
    ),
    "measured_runs_succeeded": (
        "Number of measured runs that completed successfully."
    ),
    "failed_runs": "List of run identifiers that failed.",
    # Run-level fields
    "success": "Whether this individual run completed successfully.",
    "blocks_poked": "Total number of blocks replayed in this run.",
    "failed_pokes": "Block replay operations that failed within a run.",
    "checkpoint_count": "State checkpoints created during replay.",
    "checkpoint_total_time_secs": "Total time spent creating checkpoints (seconds).",
    "average_checkpoint_time_secs": "Average time per checkpoint (seconds).",
    # Provenance / evidence fields
    "validity": (
        "Validity assessment. Valid = all runs completed within acceptable parameters."
    ),
    "fixture_sha256_hex": "SHA-256 hash of the test fixture file.",
    "capture_timestamp_ms": "Unix timestamp (ms) when provenance was captured.",
    "schema_version": "Artifact schema version.",
    "build_profile": "Cargo build profile (e.g. release, debug).",
    "realized_memory_max": "Maximum memory limit for the container (bytes).",
    "realized_memory_current": "Current cgroup memory usage of the container (bytes).",
    "total_memory_bytes": "Total physical memory on the host system.",
    "realized_cpuset": "CPUs available to the container.",
    "realized_cpu_max": "CPU bandwidth limit (max period).",
    "allocation_request_bytes": "Memory allocation requested for the benchmark.",
    "memory_limit_matches": "Whether the realized memory limit matches the requested limit.",
}

_VERDICT_TOOLTIPS: dict[str, str] = {
    "Valid": "All measured runs completed within acceptable parameters.",
    "Invalid": "One or more runs failed or produced out-of-range results.",
    "Unknown": "Validity could not be determined.",
}


# -- Public API --

def render_sweep_page(manifest: dict[str, Any]) -> str:
    template = _environment().get_template("sweep.html.j2")
    cases = manifest["cases"]
    comparison = _build_comparison_table(cases)
    case_sections = [_case_section(case) for case in cases]
    strip_charts = _build_strip_charts(cases)
    return template.render(
        manifest=manifest,
        sweep=manifest["sweep"],
        source_artifacts=manifest["source_artifacts"],
        top_level_artifacts=manifest.get("top_level_artifacts", []),
        comparison=comparison,
        case_sections=case_sections,
        strip_charts=strip_charts,
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
    # Always include minor/major faults even if absent from data.
    all_keys.update(_ALWAYS_SHOW_KEYS)

    columns: list[dict[str, str]] = []
    seen: set[str] = set()
    for key in _COMPARISON_METRICS:
        if key not in all_keys:
            continue
        columns.append({
            "key": key,
            "label": _METRIC_LABELS.get(key, key),
            "tooltip": _FIELD_TOOLTIPS.get(key, ""),
        })
        seen.add(key)

    for key in sorted(all_keys - seen - {"failed_runs"}):
        columns.append({
            "key": key,
            "label": _METRIC_LABELS.get(key, key),
            "tooltip": _FIELD_TOOLTIPS.get(key, ""),
        })

    rows = []
    for case in cases:
        cells = []
        for col in columns:
            value = case["summary"].get(col["key"])
            cells.append({
                "markup": _render_value_compact(value, col["key"]),
                "tooltip": _cell_tooltip(value, col["key"]),
            })

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
                "verdict_tooltip": _VERDICT_TOOLTIPS.get(verdict_label, ""),
                "failed_count": failed_count,
                "cells": cells,
            }
        )

    return {"columns": columns, "rows": rows}


def _is_trivial_value(value: Any) -> bool:
    """True if value is null, zero, or a ValueStats where all samples are zero."""
    if value is None:
        return True
    if _is_value_stats(value):
        return value.get("min") == 0 and value.get("max") == 0
    if _is_number(value):
        return value == 0
    if isinstance(value, list):
        return len(value) == 0
    return False


# -- Strip charts --

def _build_strip_charts(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build SVG strip charts for key metrics showing per-case run spread."""
    charts = []
    for key, label in _STRIP_CHART_METRICS:
        svg = _render_strip_chart_svg(cases, key, label)
        if svg is not None:
            charts.append({"label": label, "svg": svg})
    return charts


def _render_strip_chart_svg(
    cases: list[dict[str, Any]], key: str, label: str,
) -> Markup | None:
    """Generate an inline SVG strip chart for one metric across all cases."""
    # Collect per-case run values and medians.
    rows: list[dict[str, Any]] = []
    all_values: list[float] = []
    for case in cases:
        summary_val = case["summary"].get(key)
        if not _is_value_stats(summary_val):
            continue
        values = [v for v in summary_val.get("values", []) if v is not None]
        median = summary_val.get("median")
        if not values or median is None:
            continue
        rows.append({
            "case_id": case["case_id"],
            "median": median,
            "values": values,
        })
        all_values.extend(values)

    if not rows or not all_values:
        return None

    # Scale with padding so dots aren't at the very edge.
    vmin = min(all_values)
    vmax = max(all_values)
    pad = (vmax - vmin) * 0.08
    if pad == 0:
        pad = max(abs(vmin) * 0.05, 0.5)
    scale_lo = vmin - pad
    scale_hi = vmax + pad
    scale_range = scale_hi - scale_lo

    # Layout constants.
    label_w = 180
    plot_l = label_w + 10
    chart_w = 700
    plot_r = chart_w - 15
    plot_w = plot_r - plot_l
    row_h = 28
    top_m = 6
    bot_m = 22
    chart_h = top_m + len(rows) * row_h + bot_m

    def xp(v: float) -> float:
        return plot_l + (v - scale_lo) / scale_range * plot_w

    parts: list[str] = []
    parts.append(
        f'<svg class="strip-chart" viewBox="0 0 {chart_w} {chart_h}" '
        f'xmlns="http://www.w3.org/2000/svg">'
    )

    # Axis gridlines and tick labels.
    n_ticks = 5
    plot_bot = chart_h - bot_m
    for i in range(n_ticks):
        t = scale_lo + scale_range * i / (n_ticks - 1)
        tx = xp(t)
        parts.append(
            f'<line class="strip-axis-line" '
            f'x1="{tx:.1f}" y1="{top_m}" x2="{tx:.1f}" y2="{plot_bot}"/>'
        )
        tick_label = _format_metric(t, key)
        parts.append(
            f'<text class="strip-tick-label" '
            f'x="{tx:.1f}" y="{chart_h - 5}">{html.escape(tick_label)}</text>'
        )

    # One row per case.
    for idx, row in enumerate(rows):
        cy = top_m + idx * row_h + row_h // 2

        # Case label.
        clabel = row["case_id"]
        if len(clabel) > 28:
            clabel = clabel[:26] + "\u2026"
        parts.append(
            f'<text class="strip-case-label" '
            f'x="{label_w}" y="{cy + 3.5}">{html.escape(clabel)}</text>'
        )

        # Spread line (min to max of run values).
        vals = row["values"]
        x1 = xp(min(vals))
        x2 = xp(max(vals))
        parts.append(
            f'<line class="strip-spread" '
            f'x1="{x1:.1f}" y1="{cy}" x2="{x2:.1f}" y2="{cy}"/>'
        )

        # Individual run dots.
        for j, v in enumerate(vals):
            dx = xp(v)
            tip = f"run-{j}: {_format_metric(v, key)}"
            parts.append(
                f'<circle class="strip-dot" cx="{dx:.1f}" cy="{cy}" r="3">'
                f'<title>{html.escape(tip)}</title></circle>'
            )

        # Median marker (on top).
        mx = xp(row["median"])
        tip = f"median: {_format_metric(row['median'], key)}"
        parts.append(
            f'<circle class="strip-median" cx="{mx:.1f}" cy="{cy}" r="4.5">'
            f'<title>{html.escape(tip)}</title></circle>'
        )

    parts.append("</svg>")
    return Markup("\n".join(parts))


# -- Per-case sections --

def _case_section(case: dict[str, Any]) -> dict[str, Any]:
    run_tables = _build_run_tables(case["runs"])
    verdict = case.get("verdict", {})
    verdict_label = (
        verdict.get("validity", "Unknown")
        if isinstance(verdict, dict)
        else str(verdict)
    )
    # Find samply CPU profile artifact if present.
    samply_profile = None
    for artifact in case.get("artifacts", []):
        if "samply-profile" in artifact.get("relative_path", ""):
            samply_profile = artifact
            break
    return {
        "case": case,
        "verdict_label": verdict_label,
        "run_tables": run_tables,
        "samply_profile": samply_profile,
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
    # Always include minor/major faults even if absent from data.
    all_keys.update(_ALWAYS_SHOW_KEYS)

    ordered: list[str] = []
    seen: set[str] = set()
    for key in _RUN_KEY_ORDER:
        if key in all_keys:
            ordered.append(key)
            seen.add(key)
    for key in sorted(all_keys - seen):
        ordered.append(key)

    columns = [
        {
            "key": key,
            "label": _METRIC_LABELS.get(key, key),
            "tooltip": _FIELD_TOOLTIPS.get(key, ""),
        }
        for key in ordered
    ]

    rows = []
    for run in runs:
        result = run.get("result") or {}
        cells = []
        for key in ordered:
            value = result.get(key)
            cells.append({
                "markup": _render_value_compact(value, key),
                "tooltip": _cell_tooltip(value, key),
            })
        rows.append(
            {
                "run_id": run["run_id"],
                "cells": cells,
                "artifacts": run["artifacts"],
            }
        )

    return [{"columns": columns, "rows": rows}]


# -- Tooltips --

def _cell_tooltip(value: Any, key: str = "") -> str:
    """Generate a hover tooltip for a table cell."""
    if value is None:
        return "No data available for this metric."
    if _is_value_stats(value):
        return _valuestats_tooltip(value)
    if isinstance(value, bool):
        return "Run completed successfully." if value else "Run failed."
    return _FIELD_TOOLTIPS.get(key, "")


def _valuestats_tooltip(value: dict[str, Any]) -> str:
    """Tooltip showing full ValueStats breakdown."""
    parts = []
    for field in ("median", "min", "max", "stddev", "mad"):
        v = value.get(field)
        if v is not None:
            parts.append(f"{field}: {_format_number(v)}")
    cv = value.get("cv")
    if cv is not None:
        parts.append(f"cv=stddev/mean: {cv:.4f} (lower = more consistent)")
    n = len(value.get("values", []))
    parts.append(f"samples: {n}")
    return " | ".join(parts)


# -- Value rendering --

def _render_value_compact(value: Any, key: str = "") -> Markup:
    """Compact rendering for table cells.

    ValueStats: median as primary line, min-max range + cv as secondary.
    """
    if value is None:
        return Markup('<span class="na">n/a</span>')
    if _is_value_stats(value):
        median = value.get("median")
        if median is None:
            return Markup('<span class="na">n/a</span>')
        primary = _format_metric(median, key)
        parts = []
        vmin = value.get("min")
        vmax = value.get("max")
        if vmin is not None and vmax is not None:
            parts.append(
                f"{_format_metric(vmin, key)}\u2013{_format_metric(vmax, key)}"
            )
        cv = value.get("cv")
        if cv is not None:
            parts.append(f"cv {cv:.3f}")
        secondary = " ".join(parts)
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
    """Render a dict as a key-value table with tooltips and byte humanization."""
    if not isinstance(value, dict):
        return _render_value_markup(value)
    rows = []
    for key, item in value.items():
        tooltip = _FIELD_TOOLTIPS.get(key, "")
        title_attr = f' title="{html.escape(tooltip)}"' if tooltip else ""
        rows.append(
            "<tr><th{title}>{key}</th><td>{value}</td></tr>".format(
                title=title_attr,
                key=html.escape(str(key)),
                value=_render_value_for_key(key, item),
            )
        )
    return Markup('<table class="kv-table">{rows}</table>'.format(rows="".join(rows)))


def _render_value_for_key(key: str, value: Any) -> Markup:
    """Key-aware rendering: humanizes byte values, falls back to full fidelity."""
    if (
        _is_number(value)
        and not isinstance(value, bool)
        and _key_suggests_bytes(key)
        and abs(value) >= 1024
    ):
        human = _humanize_bytes(value)
        raw = _format_number(value)
        return Markup(
            '{human} <span class="raw-bytes">({raw})</span>'.format(
                human=html.escape(human),
                raw=html.escape(raw),
            )
        )
    return _render_value_markup(value)


def _key_suggests_bytes(key: str) -> bool:
    """Heuristic: does this field key suggest the value is in bytes?"""
    if "_bytes" in key:
        return True
    if key.startswith("realized_memory_") or key.startswith("total_memory"):
        return True
    return False


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
    return _format_compact(value)


def _format_compact(value: int | float) -> str:
    """Compact number formatting for table cells.

    Adapts decimal places to magnitude for readability without excess precision.
    """
    if isinstance(value, int):
        return str(value)
    if value == 0:
        return "0"
    # Float that is exactly an integer value
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    av = abs(value)
    if av >= 1000:
        return f"{value:.0f}"
    if av >= 100:
        return f"{value:.1f}"
    if av >= 1:
        return f"{value:.2f}"
    return f"{value:.3g}"


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
    """Full-precision formatting for evidence/detail views."""
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
