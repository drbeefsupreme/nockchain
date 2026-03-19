from __future__ import annotations

import html
import json
import shutil
from importlib.resources import files
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape
from markupsafe import Markup


def render_sweep_page(manifest: dict[str, Any]) -> str:
    template = _environment().get_template("sweep.html.j2")
    cases = manifest["cases"]
    case_ids = [case["case_id"] for case in cases]
    case_comparison_rows = _case_comparison_rows(cases)
    case_sections = [_case_section(case) for case in cases]
    chart_payloads = _collect_chart_payloads(manifest)
    return template.render(
        manifest=manifest,
        sweep=manifest["sweep"],
        source_artifacts=manifest["source_artifacts"],
        case_ids=case_ids,
        case_comparison_rows=case_comparison_rows,
        case_sections=case_sections,
        docker_images=manifest["docker_images"],
        artifact_inventory=manifest["artifact_inventory"],
        chart_payloads=chart_payloads,
        chart_payloads_json=json.dumps(chart_payloads, separators=(",", ":")),
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


def _environment() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(files("bench_pages").joinpath("templates"))),
        autoescape=select_autoescape(("html", "xml")),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def _case_comparison_rows(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary_keys = sorted({key for case in cases for key in case["summary"].keys()})
    rows = []
    for key in summary_keys:
        rows.append(
            {
                "key": key,
                "values": [
                    {
                        "case_id": case["case_id"],
                        "markup": _render_value_markup(case["summary"].get(key)),
                    }
                    for case in cases
                ],
            }
        )
    return rows


def _case_section(case: dict[str, Any]) -> dict[str, Any]:
    run_rows = _run_rows(case["runs"])
    return {
        "case": case,
        "summary_markup": _render_object_table(case["summary"]),
        "provenance_markup": _render_object_table(case["provenance"]),
        "requested_case_markup": _render_object_table(case["requested_case"]),
        "resolved_case_markup": _render_object_table(case["resolved_case"]),
        "verdict_markup": _render_object_table(case["verdict"]),
        "validation_markup": _render_object_table(case["validation"]) if case["validation"] else None,
        "run_rows": run_rows,
    }


def _run_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = sorted(
        {
            key
            for run in runs
            for key in (run.get("result") or {}).keys()
        }
    )
    rows = []
    for run in runs:
        rows.append(
            {
                "run_id": run["run_id"],
                "cells": [{"key": key, "markup": _render_value_markup((run.get("result") or {}).get(key))} for key in keys],
                "artifacts": run["artifacts"],
            }
        )
    return [{"keys": keys, "rows": rows}] if rows else []


def _collect_chart_payloads(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    cases = manifest["cases"]

    summary_keys = sorted({key for case in cases for key in case["summary"].keys()})
    for key in summary_keys:
        labels = []
        values = []
        for case in cases:
            value = case["summary"].get(key)
            numeric_value = None
            stat = "value"
            if _is_value_stats(value):
                numeric_value = value.get("median")
                stat = "median"
            elif _is_number(value):
                numeric_value = value
            labels.append(case["case_id"])
            values.append(numeric_value)
        if any(value is not None for value in values):
            payloads.append(
                {
                    "id": f"case-summary-{key}",
                    "title": f"{key} across cases",
                    "kind": "case-summary",
                    "stat": stat,
                    "key": key,
                    "labels": labels,
                    "values": values,
                    "unit": _infer_unit(key),
                }
            )

    for case in cases:
        for key, value in case["summary"].items():
            if not _is_value_stats(value):
                continue
            payloads.append(
                {
                    "id": f"case-values-{case['case_id']}-{key}",
                    "title": f"{case['case_id']} {key} values",
                    "kind": "case-values",
                    "stat": "values",
                    "key": key,
                    "labels": [f"sample-{index + 1}" for index in range(len(value.get('values', [])))],
                    "values": value.get("values", []),
                    "unit": _infer_unit(key),
                }
            )

        run_keys = sorted(
            {
                key
                for run in case["runs"]
                for key, run_value in (run.get("result") or {}).items()
                if _is_number(run_value)
            }
        )
        for key in run_keys:
            labels = [run["run_id"] for run in case["runs"]]
            values = [(run.get("result") or {}).get(key) for run in case["runs"]]
            payloads.append(
                {
                    "id": f"run-results-{case['case_id']}-{key}",
                    "title": f"{case['case_id']} {key} by run",
                    "kind": "run-results",
                    "stat": "value",
                    "key": key,
                    "labels": labels,
                    "values": values,
                    "unit": _infer_unit(key),
                }
            )

    return payloads


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
        return Markup('<table class="valuestats">{rows}</table>'.format(rows="".join(rows)))
    if isinstance(value, dict):
        return _render_object_table(value)
    if isinstance(value, list):
        items = "".join(f"<li>{_render_value_markup(item)}</li>" for item in value)
        return Markup(f'<ul class="json-list">{items}</ul>')
    if isinstance(value, bool):
        return Markup(html.escape("true" if value else "false"))
    if _is_number(value):
        return Markup(html.escape(_format_number(value)))
    return Markup(html.escape(str(value)))


def _pretty_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True)


def _is_value_stats(value: Any) -> bool:
    return isinstance(value, dict) and {"median", "min", "max", "mad", "stddev", "cv", "values"}.issubset(value.keys())


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
