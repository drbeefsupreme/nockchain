#!/usr/bin/env python3
"""Publish a SOL matrix run (with perf + Tracy artifacts) into the GH Pages archive."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import re
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path


def to_float(v: str, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def to_int(v: str, default: int = 0) -> int:
    try:
        return int(float(v))
    except Exception:
        return default


def parse_run_id(run_root: Path) -> str:
    m = re.match(r"^(\d{8}_\d{6})", run_root.name)
    if m:
        return m.group(1)
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def load_rows(tsv: Path) -> list[dict]:
    with tsv.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def pick_pass_rows(rows: list[dict]) -> tuple[int, list[dict]]:
    passes = sorted({to_int(r.get("pass", "1"), 1) for r in rows})
    p = passes[-1] if passes else 1
    sel = [r for r in rows if to_int(r.get("pass", "1"), 1) == p]
    return p, sel


def compute_summary(rows: list[dict]) -> dict:
    best_native = max(
        (r for r in rows if r["env"] == "native"),
        key=lambda r: to_float(r.get("throughput_blocks_s", "0")),
    )
    best_docker = max(
        (r for r in rows if r["env"] == "docker"),
        key=lambda r: to_float(r.get("throughput_blocks_s", "0")),
    )

    branches = ["master", "bump PMA", "btree"]
    avg = {"native": {}, "docker": {}}
    for env in ("native", "docker"):
        for b in branches:
            vals = [
                to_float(r.get("throughput_blocks_s", "0"))
                for r in rows
                if r["env"] == env and r["branch"] == b
            ]
            avg[env][b] = round(sum(vals) / len(vals), 2) if vals else 0.0

    return {
        "best_native": {
            "branch": best_native["branch"],
            "fixture": best_native["fixture"],
            "throughput_bps": round(to_float(best_native["throughput_blocks_s"]), 2),
        },
        "best_docker": {
            "branch": best_docker["branch"],
            "fixture": best_docker["fixture"],
            "throughput_bps": round(to_float(best_docker["throughput_blocks_s"]), 2),
        },
        "avg_throughput_bps": avg,
    }


def html_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def rel_copy(src: Path, dst_root: Path, rel: Path) -> str:
    dst = dst_root / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return str(rel).replace("\\", "/")


CAUSAL_METRICS = [
    {"key": "throughput_blocks_s", "label": "Throughput (blocks/s)", "higher_is_better": True},
    {"key": "total_poke_time_s", "label": "Total Poke Time (s)", "higher_is_better": False},
    {"key": "init_time_s", "label": "Init Time (s)", "higher_is_better": False},
    {"key": "peak_rss_mib", "label": "Peak RSS (MiB)", "higher_is_better": False},
]


def slugify(value: str) -> str:
    return value.lower().replace(" ", "-").replace("/", "-").replace("_", "-")


def tuple_key(env: str, branch: str, fixture: str) -> str:
    return f"{env}::{branch}::{fixture}"


def median(values: list[float]) -> float | None:
    vals = sorted(v for v in values if math.isfinite(v))
    n = len(vals)
    if n == 0:
        return None
    mid = n // 2
    if n % 2 == 1:
        return vals[mid]
    return (vals[mid - 1] + vals[mid]) / 2.0


def mad(values: list[float], center: float | None = None) -> float | None:
    vals = [v for v in values if math.isfinite(v)]
    if not vals:
        return None
    c = center if center is not None else median(vals)
    if c is None:
        return None
    deviations = [abs(v - c) for v in vals]
    return median(deviations)


def finite_or_none(value: float | None, ndigits: int = 6) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return round(float(value), ndigits)


def canonical_branch(value: str) -> str:
    v = value.strip()
    if v.lower() == "current":
        return "master"
    return v


def canonical_fixture(value: str) -> str:
    v = value.strip()
    m = re.match(r"^(v[0-9]+)_first[0-9]+$", v)
    if m:
        return m.group(1)
    return v


def normalize_summary_row(row: dict) -> dict:
    env = (row.get("env") or row.get("runtime") or "").strip()
    branch = canonical_branch((row.get("branch") or "").strip())
    fixture = canonical_fixture((row.get("fixture") or "").strip())
    return {
        "pass": to_int(row.get("pass", "1"), 1),
        "env": env,
        "branch": branch,
        "fixture": fixture,
        "throughput_blocks_s": to_float(row.get("throughput_blocks_s", "nan"), float("nan")),
        "init_time_s": to_float(row.get("init_time_s", "nan"), float("nan")),
        "total_poke_time_s": to_float(row.get("total_poke_time_s", "nan"), float("nan")),
        "peak_rss_mib": to_float(row.get("peak_rss_mib", "nan"), float("nan")),
        "failed_pokes": to_float(row.get("failed_pokes", "0"), 0.0),
        "exit_status": to_int(row.get("exit_status", "0"), 0),
    }


def pick_latest_tuple_rows(rows: list[dict]) -> list[dict]:
    latest: dict[str, dict] = {}
    for row in rows:
        env = row.get("env", "")
        branch = row.get("branch", "")
        fixture = row.get("fixture", "")
        if not env or not branch or not fixture:
            continue
        key = tuple_key(env, branch, fixture)
        prev = latest.get(key)
        if prev is None or to_int(row.get("pass", 1), 1) >= to_int(prev.get("pass", 1), 1):
            latest[key] = row
    return list(latest.values())


def load_history_samples(
    docs_root: Path, runs: list[dict], max_runs: int = 24
) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    candidates = [r for r in runs if r.get("run_id")]
    # Use oldest->newest ordering for stable baseline time series.
    candidates = sorted(candidates, key=lambda r: str(r.get("run_id")))
    if max_runs > 0 and len(candidates) > max_runs:
        candidates = candidates[-max_runs:]
    for run in candidates:
        run_id = str(run.get("run_id", ""))
        if not run_id:
            continue
        run_dir = docs_root / "sol-runs" / "runs" / run_id
        tsv = run_dir / "combined_summary.tsv"
        if not tsv.is_file():
            continue
        try:
            parsed = [normalize_summary_row(r) for r in load_rows(tsv)]
        except Exception:
            continue
        rows = pick_latest_tuple_rows(parsed)
        for row in rows:
            if row["exit_status"] != 0 or row["failed_pokes"] > 0:
                continue
            key = tuple_key(row["env"], row["branch"], row["fixture"])
            perf_summary = (
                run_dir
                / "trace"
                / slugify(row["env"])
                / slugify(row["branch"])
                / row["fixture"]
                / "perf-summary.json"
            )
            out[key].append(
                {
                    "run_id": run_id,
                    "date": str(run.get("date", "")),
                    "throughput_blocks_s": row["throughput_blocks_s"],
                    "init_time_s": row["init_time_s"],
                    "total_poke_time_s": row["total_poke_time_s"],
                    "peak_rss_mib": row["peak_rss_mib"],
                    "perf_summary": str(perf_summary) if perf_summary.is_file() else None,
                }
            )
    return out


def eval_metric(candidate: float, baseline_values: list[float], higher_is_better: bool) -> dict:
    values = [v for v in baseline_values if math.isfinite(v)]
    med = median(values)
    d = mad(values, med)
    sigma = (d * 1.4826) if (d is not None and math.isfinite(d)) else None
    if sigma is None or sigma < 1e-9:
        fallback = max(abs(med or 0.0) * 0.02, 1e-6)
        sigma = fallback
    delta_pct = None
    if med is not None and abs(med) > 1e-12:
        delta_pct = ((candidate - med) / abs(med)) * 100.0
    z_score = None
    if med is not None and sigma and sigma > 0:
        z_score = (candidate - med) / sigma

    band_low = med - 2.0 * sigma if med is not None else None
    band_high = med + 2.0 * sigma if med is not None else None
    direction = "higher_is_better" if higher_is_better else "lower_is_better"
    return {
        "candidate": finite_or_none(candidate),
        "baseline_median": finite_or_none(med),
        "baseline_mad": finite_or_none(d),
        "delta_pct": finite_or_none(delta_pct),
        "z_score": finite_or_none(z_score),
        "expected_low": finite_or_none(band_low),
        "expected_high": finite_or_none(band_high),
        "direction": direction,
    }


def parse_perf_leaf_pcts(perf_summary_path: Path) -> dict[str, float]:
    if not perf_summary_path.is_file():
        return {}
    try:
        payload = json.loads(perf_summary_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    out: dict[str, float] = {}
    for row in payload.get("top_leaves") or []:
        symbol = str(row.get("symbol", "")).strip()
        pct = to_float(str(row.get("pct", "0")), 0.0)
        if not symbol or pct <= 0:
            continue
        out[symbol] = pct
    return out


def average_symbol_maps(maps: list[dict[str, float]]) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for m in maps:
        for symbol, pct in m.items():
            totals[symbol] += pct
            counts[symbol] += 1
    out: dict[str, float] = {}
    for symbol, total in totals.items():
        c = counts.get(symbol, 0)
        if c > 0:
            out[symbol] = total / c
    return out


def top_symbol_shifts(
    candidate: dict[str, float], baseline: dict[str, float], limit: int = 8
) -> list[dict]:
    # Rank by absolute pct-point change while preserving direction.
    candidates = sorted(candidate.items(), key=lambda kv: kv[1], reverse=True)[:24]
    baselines = sorted(baseline.items(), key=lambda kv: kv[1], reverse=True)[:24]
    symbols = {k for k, _ in candidates} | {k for k, _ in baselines}
    shifts: list[dict] = []
    for symbol in symbols:
        c = candidate.get(symbol, 0.0)
        b = baseline.get(symbol, 0.0)
        delta = c - b
        shifts.append(
            {
                "symbol": symbol,
                "candidate_pct": finite_or_none(c),
                "baseline_pct": finite_or_none(b),
                "delta_pct_points": finite_or_none(delta),
                "abs_delta": abs(delta),
            }
        )
    shifts.sort(key=lambda item: item["abs_delta"], reverse=True)
    return [{k: v for k, v in item.items() if k != "abs_delta"} for item in shifts[:limit]]


def build_rust_file_index(repo_root: Path) -> list[dict]:
    crates = repo_root / "crates"
    if not crates.is_dir():
        return []
    out = []
    for path in crates.rglob("*.rs"):
        rel = path.relative_to(repo_root).as_posix()
        out.append({"rel": rel, "lower": rel.lower()})
    return out


def symbol_tokens(symbol: str) -> list[str]:
    cleaned = re.sub(r"[^A-Za-z0-9_:]+", "", symbol).strip(":")
    cleaned = cleaned.lstrip("<")
    parts = [p for p in cleaned.split("::") if p and len(p) >= 2 and not p.startswith("0x")]
    return parts[:6]


def resolve_symbol_files(symbol: str, rust_index: list[dict]) -> list[str]:
    tokens = symbol_tokens(symbol)
    if not tokens:
        return []
    crate = tokens[0].lower()
    module_tokens = [t.lower() for t in tokens[1:4] if len(t) >= 3]
    scored: list[tuple[int, str]] = []
    for entry in rust_index:
        p = entry["lower"]
        score = 0
        if f"/{crate}/" in p:
            score += 6
        for tok in module_tokens:
            if tok in p:
                score += 2
        if score > 0:
            scored.append((score, entry["rel"]))
    scored.sort(key=lambda it: (-it[0], it[1]))
    return [path for _, path in scored[:3]]


def recent_commit_hint(repo_root: Path, rel_path: str) -> str | None:
    try:
        out = subprocess.check_output(
            [
                "git",
                "-C",
                str(repo_root),
                "log",
                "-n",
                "1",
                "--pretty=format:%h %ad %s",
                "--date=short",
                "--",
                rel_path,
            ],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None
    return out or None


def classify_tuple(metric_eval: dict, baseline_samples: int) -> str:
    if baseline_samples < 3:
        return "insufficient_baseline"
    delta = metric_eval.get("delta_pct")
    z_score = metric_eval.get("z_score")
    low = metric_eval.get("expected_low")
    candidate = metric_eval.get("candidate")
    if (
        candidate is not None
        and low is not None
        and candidate < low
        and (
            (delta is not None and delta <= -4.0)
            or (z_score is not None and z_score <= -1.0)
        )
    ):
        return "regression"
    high = metric_eval.get("expected_high")
    if (
        candidate is not None
        and high is not None
        and candidate > high
        and (
            (delta is not None and delta >= 4.0)
            or (z_score is not None and z_score >= 1.0)
        )
    ):
        return "improvement"
    return "stable"


def build_causal_records(
    rows: list[dict],
    trace_records: list[dict],
    history: dict[str, list[dict]],
    run_dst: Path,
    repo_root: Path | None,
) -> list[dict]:
    trace_by_key = {
        tuple_key(tr["env"], tr["branch"], tr["fixture"]): tr for tr in trace_records
    }
    rust_index = build_rust_file_index(repo_root) if (repo_root and repo_root.is_dir()) else []
    commit_cache: dict[str, str | None] = {}

    records: list[dict] = []
    for row in sorted(rows, key=lambda x: (x["env"], x["branch"], x["fixture"])):
        key = tuple_key(row["env"], row["branch"], row["fixture"])
        baseline = history.get(key, [])
        metric_map: dict[str, dict] = {}
        for spec in CAUSAL_METRICS:
            metric_key = spec["key"]
            candidate_val = to_float(row.get(metric_key, "nan"), float("nan"))
            baseline_vals = [
                to_float(str(sample.get(metric_key, "nan")), float("nan"))
                for sample in baseline
            ]
            metric_map[metric_key] = eval_metric(
                candidate=candidate_val,
                baseline_values=baseline_vals,
                higher_is_better=bool(spec["higher_is_better"]),
            )

        throughput_eval = metric_map["throughput_blocks_s"]
        classification = classify_tuple(throughput_eval, len(baseline))

        tr = trace_by_key.get(key)
        candidate_perf: dict[str, float] = {}
        if tr and tr.get("summary_json"):
            candidate_perf = parse_perf_leaf_pcts(run_dst / str(tr["summary_json"]))
        baseline_maps = []
        for sample in baseline:
            perf_summary = sample.get("perf_summary")
            if perf_summary:
                baseline_maps.append(parse_perf_leaf_pcts(Path(str(perf_summary))))
        baseline_perf = average_symbol_maps(baseline_maps)
        shifts = top_symbol_shifts(candidate_perf, baseline_perf, limit=8)

        file_scores: dict[str, float] = defaultdict(float)
        for shift in shifts[:6]:
            symbol = str(shift.get("symbol", ""))
            delta = abs(to_float(str(shift.get("delta_pct_points", "0")), 0.0))
            if delta <= 0.0:
                continue
            for f in resolve_symbol_files(symbol, rust_index):
                file_scores[f] += delta
        top_files = sorted(file_scores.items(), key=lambda it: (-it[1], it[0]))[:5]
        likely_files = []
        for rel_path, score in top_files:
            commit_hint = None
            if repo_root and repo_root.is_dir():
                if rel_path not in commit_cache:
                    commit_cache[rel_path] = recent_commit_hint(repo_root, rel_path)
                commit_hint = commit_cache[rel_path]
            likely_files.append(
                {
                    "path": rel_path,
                    "score": finite_or_none(score, ndigits=4),
                    "recent_commit": commit_hint,
                }
            )

        z_abs = abs(to_float(str(throughput_eval.get("z_score")), 0.0))
        symbol_signal = sum(
            abs(to_float(str(s.get("delta_pct_points", "0")), 0.0)) for s in shifts[:3]
        )
        confidence = min(
            1.0,
            0.35 * min(len(baseline) / 8.0, 1.0)
            + 0.45 * min(z_abs / 3.0, 1.0)
            + 0.20 * min(symbol_signal / 8.0, 1.0),
        )

        records.append(
            {
                "env": row["env"],
                "branch": row["branch"],
                "fixture": row["fixture"],
                "classification": classification,
                "confidence": finite_or_none(confidence, ndigits=4),
                "baseline_samples": len(baseline),
                "metrics": metric_map,
                "top_symbol_shifts": shifts,
                "likely_files": likely_files,
                "baseline_runs": [
                    {
                        "run_id": s.get("run_id"),
                        "date": s.get("date"),
                        "throughput_blocks_s": finite_or_none(
                            to_float(str(s.get("throughput_blocks_s", "nan")), float("nan"))
                        ),
                    }
                    for s in baseline[-12:]
                ],
            }
        )
    return records


def build_trace_records(run_rows: list[dict], run_dst: Path) -> list[dict]:
    out: list[dict] = []
    for r in sorted(
        run_rows, key=lambda x: (x["env"], x["branch"], x["fixture"])
    ):
        perf_svg = Path(r.get("perf_flamegraph_svg", ""))
        perf_sum = Path(r.get("perf_summary_json", ""))
        perf_rep = Path(r.get("perf_report_txt", ""))
        perf_folded = Path(r.get("perf_folded", ""))
        perf_data = Path(r.get("perf_data", ""))
        perf_script = perf_data.with_suffix(".script")
        tracy_trace = Path(r.get("tracy_trace", ""))
        tracy_capture_log = Path(r.get("tracy_capture_log", ""))
        tracy_size_bytes = to_int(r.get("tracy_size_bytes", "0"))
        rec = {
            "env": r["env"],
            "branch": r["branch"],
            "fixture": r["fixture"],
            "throughput_bps": to_float(r.get("throughput_blocks_s", "0")),
            "perf_samples": to_int(r.get("perf_samples", "0")),
            "perf_unique_stacks": to_int(r.get("perf_unique_stacks", "0")),
            "tracy_size_bytes": tracy_size_bytes,
            "tracy_size_mib": (tracy_size_bytes / (1024 * 1024)) if tracy_size_bytes else 0.0,
            "tracy_frames": to_int(r.get("tracy_frames", "0")),
            "tracy_time_span_s": to_float(r.get("tracy_time_span_s", "0")),
            "tracy_zones": to_int(r.get("tracy_zones", "0")),
            "flamegraph": None,
            "summary_json": None,
            "report_txt": None,
            "raw_folded": None,
            "raw_perf_data": None,
            "raw_perf_script": None,
            "raw_tracy_trace": None,
            "raw_tracy_log": None,
        }
        env_slug = r["env"].lower().replace(" ", "-").replace("/", "-").replace("_", "-")
        branch_slug = (
            r["branch"].lower().replace(" ", "-").replace("/", "-").replace("_", "-")
        )
        fixture_slug = r["fixture"]
        if perf_svg.is_file():
            rec["flamegraph"] = rel_copy(
                perf_svg,
                run_dst,
                Path("trace") / env_slug / branch_slug / fixture_slug / "perf-flamegraph.svg",
            )
        if perf_sum.is_file():
            rec["summary_json"] = rel_copy(
                perf_sum,
                run_dst,
                Path("trace") / env_slug / branch_slug / fixture_slug / "perf-summary.json",
            )
        if perf_rep.is_file():
            rec["report_txt"] = rel_copy(
                perf_rep,
                run_dst,
                Path("trace") / env_slug / branch_slug / fixture_slug / "perf-report.txt",
            )
        if perf_folded.is_file():
            rec["raw_folded"] = rel_copy(
                perf_folded,
                run_dst,
                Path("trace") / env_slug / branch_slug / fixture_slug / "perf.folded",
            )
        if perf_data.is_file():
            rec["raw_perf_data"] = rel_copy(
                perf_data,
                run_dst,
                Path("trace") / env_slug / branch_slug / fixture_slug / "perf.data",
            )
        if perf_script.is_file():
            rec["raw_perf_script"] = rel_copy(
                perf_script,
                run_dst,
                Path("trace") / env_slug / branch_slug / fixture_slug / "perf.script",
            )
        if tracy_trace.is_file():
            rec["raw_tracy_trace"] = rel_copy(
                tracy_trace,
                run_dst,
                Path("trace") / env_slug / branch_slug / fixture_slug / "trace.tracy",
            )
        if tracy_capture_log.is_file():
            rec["raw_tracy_log"] = rel_copy(
                tracy_capture_log,
                run_dst,
                Path("trace") / env_slug / branch_slug / fixture_slug / "trace-capture.log",
            )
        out.append(rec)
    return out


def build_guard_records(run_root: Path, run_dst: Path) -> list[dict]:
    out: list[dict] = []
    for guard_json in sorted(run_root.glob("guard-*.json")):
        try:
            payload = json.loads(guard_json.read_text(encoding="utf-8"))
        except Exception:
            continue
        context = payload.get("context") or {}
        metrics = payload.get("metrics") or []
        failed_rules = sum(1 for m in metrics if not bool(m.get("passed", False)))
        rec = {
            "env": context.get("env", ""),
            "branch": context.get("branch", ""),
            "fixture": context.get("fixture", ""),
            "verdict": str(payload.get("verdict", "unknown")),
            "baseline_samples": to_int(payload.get("baseline_samples", 0)),
            "failed_rules": failed_rules,
            "json": rel_copy(guard_json, run_dst, Path("guard") / guard_json.name),
            "md": None,
        }
        guard_md = guard_json.with_suffix(".md")
        if guard_md.is_file():
            rec["md"] = rel_copy(guard_md, run_dst, Path("guard") / guard_md.name)
        out.append(rec)
    return out


def render_html(
    run_id: str,
    generated_on: str,
    pass_no: int,
    rows: list[dict],
    trace_records: list[dict],
    guard_records: list[dict],
    causal_records: list[dict],
    summary: dict,
    base_prefix: str,
    fixture_label: str,
) -> str:
    cards = {
        "run_rows": len(rows),
        "best_native": summary["best_native"],
        "best_docker": summary["best_docker"],
        "native_master": summary["avg_throughput_bps"]["native"]["master"],
        "native_bump": summary["avg_throughput_bps"]["native"]["bump PMA"],
        "native_btree": summary["avg_throughput_bps"]["native"]["btree"],
        "docker_master": summary["avg_throughput_bps"]["docker"]["master"],
        "docker_bump": summary["avg_throughput_bps"]["docker"]["bump PMA"],
        "docker_btree": summary["avg_throughput_bps"]["docker"]["btree"],
    }

    score_rows = []
    for r in rows:
        score_rows.append(
            {
                "env": r["env"],
                "branch": r["branch"],
                "fixture": r["fixture"],
                "blocks_poked": to_int(r.get("blocks_poked", "0")),
                "init_time_s": round(to_float(r.get("init_time_s", "0")), 4),
                "total_poke_time_s": round(to_float(r.get("total_poke_time_s", "0")), 4),
                "throughput_blocks_s": round(to_float(r.get("throughput_blocks_s", "0")), 3),
                "failed_pokes": to_int(r.get("failed_pokes", "0")),
                "exit_status": to_int(r.get("exit_status", "0")),
                "wall_clock": r.get("wall_clock", "0:00.00"),
            }
        )
    score_rows_json = json.dumps(score_rows)

    trace_rows = []
    options = []
    for i, tr in enumerate(trace_records):
        flame = (
            f'<a href="{base_prefix}{tr["flamegraph"]}" target="_blank">flamegraph</a>'
            if tr["flamegraph"]
            else '<span class="muted">n/a</span>'
        )
        summ = (
            f'<a href="{base_prefix}{tr["summary_json"]}" target="_blank">summary</a>'
            if tr["summary_json"]
            else '<span class="muted">n/a</span>'
        )
        rep = (
            f'<a href="{base_prefix}{tr["report_txt"]}" target="_blank">report</a>'
            if tr["report_txt"]
            else '<span class="muted">n/a</span>'
        )
        raw_folded = (
            f'<a href="{base_prefix}{tr["raw_folded"]}" target="_blank">folded</a>'
            if tr["raw_folded"]
            else '<span class="muted">n/a</span>'
        )
        raw_data = (
            f'<a href="{base_prefix}{tr["raw_perf_data"]}" target="_blank">perf.data</a>'
            if tr["raw_perf_data"]
            else '<span class="muted">n/a</span>'
        )
        raw_script = (
            f'<a href="{base_prefix}{tr["raw_perf_script"]}" target="_blank">perf.script</a>'
            if tr["raw_perf_script"]
            else '<span class="muted">n/a</span>'
        )
        raw_tracy = (
            f'<a href="{base_prefix}{tr["raw_tracy_trace"]}" target="_blank">trace.tracy</a>'
            if tr["raw_tracy_trace"]
            else '<span class="muted">n/a</span>'
        )
        raw_tracy_log = (
            f'<a href="{base_prefix}{tr["raw_tracy_log"]}" target="_blank">trace.log</a>'
            if tr["raw_tracy_log"]
            else '<span class="muted">n/a</span>'
        )
        trace_rows.append(
            "<tr>"
            f"<td>{html_escape(tr['env'])}</td>"
            f"<td>{html_escape(tr['branch'])}</td>"
            f"<td>{html_escape(tr['fixture'])}</td>"
            f"<td>{tr['throughput_bps']:.3f}</td>"
            f"<td>{tr['perf_samples']}</td>"
            f"<td>{tr['perf_unique_stacks']}</td>"
            f"<td>{tr['tracy_size_mib']:.2f}</td>"
            f"<td>{tr['tracy_zones']}</td>"
            f"<td>{raw_tracy} {raw_tracy_log} {flame} {summ} {rep} {raw_folded} {raw_script} {raw_data}</td>"
            "</tr>"
        )
        if tr["flamegraph"]:
            options.append(
                f'<option value="{html_escape(base_prefix + tr["flamegraph"])}"'
                + (" selected" if i == 0 else "")
                + f'>{html_escape(tr["env"])} / {html_escape(tr["branch"])} / {html_escape(tr["fixture"])}</option>'
            )

    default_flame = options[0] if options else ""
    default_url = ""
    if default_flame:
        m = re.search(r'value="([^"]+)"', default_flame)
        default_url = m.group(1) if m else ""

    guard_rows = []
    for gr in sorted(guard_records, key=lambda x: (x["env"], x["branch"], x["fixture"])):
        report_link = (
            f'<a href="{base_prefix}{gr["md"]}" target="_blank">md</a>'
            if gr["md"]
            else '<span class="muted">md n/a</span>'
        )
        guard_rows.append(
            "<tr>"
            f"<td>{html_escape(gr['env'])}</td>"
            f"<td>{html_escape(gr['branch'])}</td>"
            f"<td>{html_escape(gr['fixture'])}</td>"
            f"<td>{html_escape(gr['verdict'])}</td>"
            f"<td>{gr['baseline_samples']}</td>"
            f"<td>{gr['failed_rules']}</td>"
            f"<td><a href=\"{base_prefix}{gr['json']}\" target=\"_blank\">json</a> {report_link}</td>"
            "</tr>"
        )
    guard_rows_html = "".join(guard_rows) if guard_rows else "<tr><td colspan=\"7\" class=\"muted\">No guard reports found for this run.</td></tr>"
    causal_records_json = json.dumps(causal_records)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>SOL Benchmark Trace Report ({run_id})</title>
  <style>
    :root {{
      --bg:#f3f6fb; --card:#fff; --line:#d5dbe5; --text:#1f2937; --muted:#5d6777;
      --accent:#0e6a4e; --accent2:#0d5fd3;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin:0; background:radial-gradient(circle at 20% 0%, #e6f2ea 0%, var(--bg) 42%); color:var(--text); font-family:"IBM Plex Sans","Segoe UI",Arial,sans-serif; }}
    main {{ max-width:1280px; margin:24px auto; padding:0 16px 36px; display:grid; gap:14px; }}
    .card {{ background:var(--card); border:1px solid var(--line); border-radius:12px; box-shadow:0 10px 24px rgba(20,36,60,.06); padding:14px; }}
    h1, h2 {{ margin:0 0 10px; }}
    .muted {{ color:var(--muted); }}
    .grid4 {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; }}
    .grid5 {{ display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:10px; }}
    .stat {{ border:1px solid var(--line); border-radius:10px; padding:10px; background:#fafcfe; }}
    .k {{ font-size:.78rem; color:var(--muted); margin-bottom:4px; }}
    .v {{ font-size:1.03rem; font-weight:700; }}
    table {{ width:100%; border-collapse: collapse; font-size:.92rem; }}
    th, td {{ text-align:left; border-bottom:1px solid var(--line); padding:8px; vertical-align:top; }}
    th {{ font-size:.78rem; text-transform:uppercase; letter-spacing:.04em; color:#49576a; background:#f2f5f8; }}
    .wrap {{ overflow:auto; border:1px solid var(--line); border-radius:10px; }}
    .links a {{ margin-right:8px; text-decoration:none; color:var(--accent2); }}
    select {{ font:inherit; padding:7px 8px; border:1px solid var(--line); border-radius:8px; }}
    .controls {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; margin:10px 0; }}
    .control label {{ display:block; font-size:.78rem; color:var(--muted); margin-bottom:4px; }}
    .control select {{ width:100%; }}
    .stack-list, .file-list {{ margin:0; padding-left:18px; }}
    .stack-list li, .file-list li {{ margin:4px 0; }}
    .mini {{ font-size:.82rem; }}
    .th-sort {{ cursor:pointer; user-select:none; }}
    .th-sort::after {{ content:" ↕"; color:#7b8798; font-size:.74rem; }}
    .th-sort.active.asc::after {{ content:" ↑"; color:var(--accent2); }}
    .th-sort.active.desc::after {{ content:" ↓"; color:var(--accent2); }}
    object {{ width:100%; height:620px; border:1px solid var(--line); border-radius:8px; background:#fff; }}
    code {{ background:#f1f4f8; border:1px solid #dde3eb; border-radius:5px; padding:1px 5px; }}
    @media (max-width: 980px) {{ .grid4 {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} .grid5 {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} .controls {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} object {{ height:480px; }} }}
    @media (max-width: 640px) {{ .grid4 {{ grid-template-columns:1fr; }} .grid5 {{ grid-template-columns:1fr; }} .controls {{ grid-template-columns:1fr; }} object {{ height:360px; }} }}
  </style>
</head>
<body>
  <main>
    <section class="card">
      <h1>SOL Trace + Sampling Report</h1>
      <p class="muted">Run <code>{run_id}</code>, pass <code>{pass_no}</code> scoreboard ({html_escape(fixture_label)}, native+docker, checkpointing off, memory profiling off). Generated {generated_on}.</p>
      <div class="grid4">
        <div class="stat"><div class="k">Rows In Report</div><div class="v">{cards["run_rows"]}</div></div>
        <div class="stat"><div class="k">Best Native</div><div class="v">{cards["best_native"]["branch"]} / {cards["best_native"]["fixture"]} / {cards["best_native"]["throughput_bps"]:.2f} bps</div></div>
        <div class="stat"><div class="k">Best Docker</div><div class="v">{cards["best_docker"]["branch"]} / {cards["best_docker"]["fixture"]} / {cards["best_docker"]["throughput_bps"]:.2f} bps</div></div>
        <div class="stat"><div class="k">Tracy Status</div><div class="v">captured (native)</div></div>
      </div>
      <p class="muted">Average throughput native: master {cards["native_master"]:.2f}, bump PMA {cards["native_bump"]:.2f}, btree {cards["native_btree"]:.2f} bps. Docker: master {cards["docker_master"]:.2f}, bump PMA {cards["docker_bump"]:.2f}, btree {cards["docker_btree"]:.2f} bps.</p>
    </section>

    <section class="card">
      <h2>Runtime Matrix (Pass {pass_no})</h2>
      <p class="muted">Filter by fixture and environment, then click any column header to sort ascending/descending.</p>
      <div class="controls">
        <div class="control">
          <label for="scoreEnv">Environment</label>
          <select id="scoreEnv">
            <option value="all">all</option>
          </select>
        </div>
        <div class="control">
          <label for="scoreFixture">Fixture</label>
          <select id="scoreFixture">
            <option value="all">all</option>
          </select>
        </div>
        <div class="control">
          <label for="scoreSortState">Sort State</label>
          <div id="scoreSortState" class="muted">throughput_blocks_s desc</div>
        </div>
        <div class="control">
          <label for="scoreStats">Rows</label>
          <div id="scoreStats" class="muted">0</div>
        </div>
      </div>
      <div class="wrap">
        <table id="scoreTable">
          <thead><tr><th class="th-sort" data-key="env">Env</th><th class="th-sort" data-key="branch">Branch</th><th class="th-sort" data-key="fixture">Fixture</th><th class="th-sort" data-key="blocks_poked">Blocks</th><th class="th-sort" data-key="init_time_s">Init (s)</th><th class="th-sort" data-key="total_poke_time_s">Poke (s)</th><th class="th-sort active desc" data-key="throughput_blocks_s">BPS</th><th class="th-sort" data-key="failed_pokes">Failed</th><th class="th-sort" data-key="exit_status">Exit</th><th class="th-sort" data-key="wall_clock">Wall</th></tr></thead>
          <tbody id="scoreBody"></tbody>
        </table>
      </div>
    </section>

    <section class="card">
      <h2>Tracing + Sampling Artifacts (Native + Docker)</h2>
      <p class="muted">Runs include Tracy captures and perf stack sampling when available. Perf flamegraphs are generated from folded stacks.</p>
      <p class="muted">Raw downloads per run include: <code>trace.tracy</code>, <code>trace-capture.log</code>, <code>perf.folded</code>, <code>perf.script</code>, and <code>perf.data</code>.</p>
      <div class="wrap">
        <table>
          <thead><tr><th>Env</th><th>Branch</th><th>Fixture</th><th>BPS</th><th>Perf Samples</th><th>Stacks</th><th>Tracy (MiB)</th><th>Tracy Zones</th><th>Artifacts</th></tr></thead>
          <tbody>
            {''.join(trace_rows)}
          </tbody>
        </table>
      </div>
      <div style="margin-top:12px;">
        <label for="fgSel" class="muted">Flamegraph preview:</label>
        <select id="fgSel">{''.join(options)}</select>
      </div>
      <div style="margin-top:8px;">
        <object id="fgObj" data="{html_escape(default_url)}" type="image/svg+xml"></object>
      </div>
    </section>

    <section class="card">
      <h2>Guard Verdicts</h2>
      <p class="muted">Contract verdicts from optional <code>sol guard</code> post-run checks.</p>
      <div class="wrap">
        <table>
          <thead><tr><th>Env</th><th>Branch</th><th>Fixture</th><th>Verdict</th><th>Baseline Samples</th><th>Failed Rules</th><th>Reports</th></tr></thead>
          <tbody>
            {guard_rows_html}
          </tbody>
        </table>
      </div>
    </section>

    <section class="card">
      <h2>Why Did It Change? (Causal Attribution)</h2>
      <p class="muted">Experimental attribution model combining historical baseline statistics, symbol shifts, and source-file hints. Select a tuple to inspect likely causes.</p>
      <div class="controls">
        <div class="control">
          <label for="causalEnv">Environment</label>
          <select id="causalEnv"></select>
        </div>
        <div class="control">
          <label for="causalBranch">Branch</label>
          <select id="causalBranch"></select>
        </div>
        <div class="control">
          <label for="causalFixture">Fixture</label>
          <select id="causalFixture"></select>
        </div>
        <div class="control">
          <label for="causalClass">Classification</label>
          <div id="causalClass" class="muted">-</div>
        </div>
      </div>
      <div class="grid5">
        <div class="stat"><div class="k">Confidence</div><div id="causalConfidence" class="v">-</div></div>
        <div class="stat"><div class="k">Baseline Samples</div><div id="causalSamples" class="v">-</div></div>
        <div class="stat"><div class="k">Throughput Candidate</div><div id="causalThroughputCandidate" class="v">-</div></div>
        <div class="stat"><div class="k">Throughput Baseline</div><div id="causalThroughputBaseline" class="v">-</div></div>
        <div class="stat"><div class="k">Throughput Delta</div><div id="causalThroughputDelta" class="v">-</div></div>
      </div>
      <div style="margin-top:12px;" class="wrap">
        <table>
          <thead><tr><th>Metric</th><th>Candidate</th><th>Baseline Median</th><th>Expected Range</th><th>Delta (%)</th><th>Z Score</th></tr></thead>
          <tbody id="causalMetricBody"></tbody>
        </table>
      </div>
      <div style="margin-top:12px;" class="grid4">
        <div class="stat">
          <div class="k">Top Symbol Shifts (pct points)</div>
          <ul id="causalSymbolList" class="stack-list mini"></ul>
        </div>
        <div class="stat">
          <div class="k">Likely Source Files</div>
          <ul id="causalFileList" class="file-list mini"></ul>
        </div>
        <div class="stat">
          <div class="k">Baseline Throughput History</div>
          <ul id="causalBaselineList" class="file-list mini"></ul>
        </div>
        <div class="stat">
          <div class="k">Model Notes</div>
          <div id="causalNotes" class="mini muted">Uses robust baseline bands (median ± 2σ, with σ from MAD) and perf leaf shift deltas.</div>
        </div>
      </div>
    </section>

    <section class="card links">
      <h2>Artifacts</h2>
      <a href="{base_prefix}combined_summary.tsv" target="_blank">combined_summary.tsv</a>
      <a href="{base_prefix}sol-benchmark-transplant-report.md" target="_blank">report.md</a>
      <a href="{base_prefix}sol-benchmark-transplant-memory-profiles.json" target="_blank">memory-profiles.json</a>
      <a href="{base_prefix}causal-attribution.json" target="_blank">causal-attribution.json</a>
    </section>
  </main>
  <script>
    const SCORE_ROWS = {score_rows_json};
    const CAUSAL_RECORDS = {causal_records_json};
    const scoreState = {{
      env: 'all',
      fixture: 'all',
      sortKey: 'throughput_blocks_s',
      sortDir: 'desc',
    }};
    const numericKeys = new Set([
      'blocks_poked',
      'init_time_s',
      'total_poke_time_s',
      'throughput_blocks_s',
      'failed_pokes',
      'exit_status',
    ]);

    function sortValues(list, preferred) {{
      return list.slice().sort((a, b) => {{
        if (preferred) {{
          const pa = preferred[a] ?? 999;
          const pb = preferred[b] ?? 999;
          if (pa !== pb) return pa - pb;
        }}
        return String(a).localeCompare(String(b), undefined, {{ numeric: true }});
      }});
    }}

    function uniqueValues(rows, key, preferred = null) {{
      const values = [...new Set(rows.map((row) => row[key]))];
      return sortValues(values, preferred);
    }}

    function fillSelect(select, values) {{
      if (!select) return;
      const keep = select.value || 'all';
      select.innerHTML = '<option value="all">all</option>' + values.map((v) => `<option value="${{v}}">${{v}}</option>`).join('');
      if (values.includes(keep)) {{
        select.value = keep;
      }}
    }}

    function compareRows(a, b, key, dir) {{
      let cmp = 0;
      if (numericKeys.has(key)) {{
        cmp = (Number(a[key]) || 0) - (Number(b[key]) || 0);
      }} else {{
        cmp = String(a[key] ?? '').localeCompare(String(b[key] ?? ''), undefined, {{ numeric: true, sensitivity: 'base' }});
      }}
      return dir === 'asc' ? cmp : -cmp;
    }}

    function formatScoreValue(row, key) {{
      if (key === 'init_time_s' || key === 'total_poke_time_s') return Number(row[key] || 0).toFixed(4);
      if (key === 'throughput_blocks_s') return Number(row[key] || 0).toFixed(3);
      return row[key];
    }}

    function setSortVisuals() {{
      const headers = document.querySelectorAll('#scoreTable thead th.th-sort');
      headers.forEach((th) => {{
        th.classList.remove('active', 'asc', 'desc');
        if (th.dataset.key === scoreState.sortKey) {{
          th.classList.add('active', scoreState.sortDir);
        }}
      }});
      const sortState = document.getElementById('scoreSortState');
      if (sortState) sortState.textContent = `${{scoreState.sortKey}} ${{scoreState.sortDir}}`;
    }}

    function renderScoreTable() {{
      const body = document.getElementById('scoreBody');
      const stats = document.getElementById('scoreStats');
      if (!body) return;

      const filtered = SCORE_ROWS.filter((row) => {{
        if (scoreState.env !== 'all' && row.env !== scoreState.env) return false;
        if (scoreState.fixture !== 'all' && row.fixture !== scoreState.fixture) return false;
        return true;
      }}).sort((a, b) => compareRows(a, b, scoreState.sortKey, scoreState.sortDir));

      if (!filtered.length) {{
        body.innerHTML = '<tr><td colspan="10" class="muted">No runs match current filter.</td></tr>';
        if (stats) stats.textContent = `0 of ${{SCORE_ROWS.length}}`;
        setSortVisuals();
        return;
      }}

      const columns = ['env', 'branch', 'fixture', 'blocks_poked', 'init_time_s', 'total_poke_time_s', 'throughput_blocks_s', 'failed_pokes', 'exit_status', 'wall_clock'];
      body.innerHTML = filtered.map((row) => {{
        const tds = columns.map((key) => `<td>${{formatScoreValue(row, key)}}</td>`).join('');
        return `<tr>${{tds}}</tr>`;
      }}).join('');
      if (stats) stats.textContent = `${{filtered.length}} of ${{SCORE_ROWS.length}}`;
      setSortVisuals();
    }}

    function formatMaybe(value, digits = 3) {{
      if (value === null || value === undefined || Number.isNaN(Number(value))) return 'n/a';
      return Number(value).toFixed(digits);
    }}

    function fillExactSelect(select, values, preferred = null) {{
      if (!select) return;
      const opts = values.map((v) => `<option value="${{v}}">${{v}}</option>`).join('');
      select.innerHTML = opts;
      if (!values.length) return;
      if (preferred && values.includes(preferred)) {{
        select.value = preferred;
      }} else {{
        select.value = values[0];
      }}
    }}

    function uniqueCausal(key, preferred = null) {{
      const vals = [...new Set(CAUSAL_RECORDS.map((r) => r[key]))];
      return sortValues(vals, preferred);
    }}

    function lookupCausalRecord(env, branch, fixture) {{
      return CAUSAL_RECORDS.find((r) => r.env === env && r.branch === branch && r.fixture === fixture) || null;
    }}

    function renderCausal() {{
      const envSel = document.getElementById('causalEnv');
      const branchSel = document.getElementById('causalBranch');
      const fixtureSel = document.getElementById('causalFixture');
      const classEl = document.getElementById('causalClass');
      const confEl = document.getElementById('causalConfidence');
      const samplesEl = document.getElementById('causalSamples');
      const tCandEl = document.getElementById('causalThroughputCandidate');
      const tBaseEl = document.getElementById('causalThroughputBaseline');
      const tDeltaEl = document.getElementById('causalThroughputDelta');
      const metricBody = document.getElementById('causalMetricBody');
      const symbolList = document.getElementById('causalSymbolList');
      const fileList = document.getElementById('causalFileList');
      const baselineList = document.getElementById('causalBaselineList');
      if (!envSel || !branchSel || !fixtureSel || !metricBody || !symbolList || !fileList || !baselineList) return;

      const rec = lookupCausalRecord(envSel.value, branchSel.value, fixtureSel.value);
      if (!rec) {{
        classEl.textContent = 'n/a';
        confEl.textContent = '-';
        samplesEl.textContent = '-';
        tCandEl.textContent = '-';
        tBaseEl.textContent = '-';
        tDeltaEl.textContent = '-';
        metricBody.innerHTML = '<tr><td colspan="6" class="muted">No causal record for selected tuple.</td></tr>';
        symbolList.innerHTML = '<li class="muted">No symbol shift data.</li>';
        fileList.innerHTML = '<li class="muted">No file hints available.</li>';
        baselineList.innerHTML = '<li class="muted">No baseline history.</li>';
        return;
      }}

      const thr = rec.metrics?.throughput_blocks_s || {{}};
      classEl.textContent = rec.classification || 'unknown';
      confEl.textContent = rec.confidence !== null && rec.confidence !== undefined ? `${{Math.round(Number(rec.confidence) * 100)}}%` : 'n/a';
      samplesEl.textContent = String(rec.baseline_samples ?? 0);
      tCandEl.textContent = `${{formatMaybe(thr.candidate, 3)}} bps`;
      tBaseEl.textContent = `${{formatMaybe(thr.baseline_median, 3)}} bps`;
      tDeltaEl.textContent = thr.delta_pct === null || thr.delta_pct === undefined ? 'n/a' : `${{formatMaybe(thr.delta_pct, 2)}}%`;

      const metricOrder = [
        ['throughput_blocks_s', 'Throughput (blocks/s)'],
        ['total_poke_time_s', 'Total Poke Time (s)'],
        ['init_time_s', 'Init Time (s)'],
        ['peak_rss_mib', 'Peak RSS (MiB)'],
      ];
      metricBody.innerHTML = metricOrder.map(([key, label]) => {{
        const m = rec.metrics?.[key];
        if (!m) return '';
        const range = (m.expected_low !== null && m.expected_high !== null)
          ? `${{formatMaybe(m.expected_low, 3)}} .. ${{formatMaybe(m.expected_high, 3)}}`
          : 'n/a';
        return `<tr>
          <td>${{label}}</td>
          <td>${{formatMaybe(m.candidate, 3)}}</td>
          <td>${{formatMaybe(m.baseline_median, 3)}}</td>
          <td>${{range}}</td>
          <td>${{m.delta_pct === null || m.delta_pct === undefined ? 'n/a' : (formatMaybe(m.delta_pct, 2) + '%')}}</td>
          <td>${{formatMaybe(m.z_score, 3)}}</td>
        </tr>`;
      }}).join('') || '<tr><td colspan="6" class="muted">No metric attribution data.</td></tr>';

      const shifts = rec.top_symbol_shifts || [];
      symbolList.innerHTML = shifts.length
        ? shifts.map((s) => `<li><code>${{String(s.symbol || '').slice(0, 90)}}</code> <span class="muted">Δ ${{formatMaybe(s.delta_pct_points, 3)}}pp (cand ${{formatMaybe(s.candidate_pct, 3)}} / base ${{formatMaybe(s.baseline_pct, 3)}})</span></li>`).join('')
        : '<li class="muted">No symbol shift data.</li>';

      const files = rec.likely_files || [];
      fileList.innerHTML = files.length
        ? files.map((f) => `<li><code>${{f.path}}</code> <span class="muted">(score ${{formatMaybe(f.score, 3)}}${{f.recent_commit ? ('; ' + f.recent_commit) : ''}})</span></li>`).join('')
        : '<li class="muted">No file hints available.</li>';

      const baseline = rec.baseline_runs || [];
      baselineList.innerHTML = baseline.length
        ? baseline.map((b) => `<li><code>${{b.run_id}}</code> <span class="muted">${{formatMaybe(b.throughput_blocks_s, 3)}} bps</span></li>`).join('')
        : '<li class="muted">No baseline history.</li>';
    }}

    function initCausal() {{
      const envSel = document.getElementById('causalEnv');
      const branchSel = document.getElementById('causalBranch');
      const fixtureSel = document.getElementById('causalFixture');
      if (!envSel || !branchSel || !fixtureSel) return;
      if (!CAUSAL_RECORDS.length) {{
        const metricBody = document.getElementById('causalMetricBody');
        if (metricBody) metricBody.innerHTML = '<tr><td colspan="6" class="muted">No causal records generated for this run.</td></tr>';
        return;
      }}

      fillExactSelect(envSel, uniqueCausal('env', {{ native: 0, docker: 1 }}));
      fillExactSelect(branchSel, uniqueCausal('branch', {{ master: 0, 'bump PMA': 1, btree: 2 }}));
      fillExactSelect(fixtureSel, uniqueCausal('fixture', {{ v0: 0, v1: 1, v2: 2 }}));
      envSel.addEventListener('change', renderCausal);
      branchSel.addEventListener('change', renderCausal);
      fixtureSel.addEventListener('change', renderCausal);
      renderCausal();
    }}

    const scoreEnv = document.getElementById('scoreEnv');
    const scoreFixture = document.getElementById('scoreFixture');
    if (scoreEnv && scoreFixture) {{
      fillSelect(scoreEnv, uniqueValues(SCORE_ROWS, 'env', {{ native: 0, docker: 1 }}));
      fillSelect(scoreFixture, uniqueValues(SCORE_ROWS, 'fixture', {{ v0: 0, v1: 1, v2: 2 }}));
      scoreEnv.addEventListener('change', () => {{
        scoreState.env = scoreEnv.value;
        renderScoreTable();
      }});
      scoreFixture.addEventListener('change', () => {{
        scoreState.fixture = scoreFixture.value;
        renderScoreTable();
      }});
    }}

    document.querySelectorAll('#scoreTable thead th.th-sort').forEach((th) => {{
      th.addEventListener('click', () => {{
        const key = th.dataset.key;
        if (!key) return;
        if (scoreState.sortKey === key) {{
          scoreState.sortDir = scoreState.sortDir === 'asc' ? 'desc' : 'asc';
        }} else {{
          scoreState.sortKey = key;
          scoreState.sortDir = key === 'throughput_blocks_s' ? 'desc' : 'asc';
        }}
        renderScoreTable();
      }});
    }});

    renderScoreTable();
    initCausal();

    const sel = document.getElementById('fgSel');
    const obj = document.getElementById('fgObj');
    if (sel && obj) {{
      sel.addEventListener('change', () => {{
        obj.setAttribute('data', sel.value);
      }});
    }}
  </script>
</body>
</html>
"""


def render_md(
    run_id: str,
    generated_on: str,
    pass_no: int,
    trace_records: list[dict],
    guard_records: list[dict],
    causal_records: list[dict],
    summary: dict,
    fixture_label: str,
) -> str:
    lines = []
    lines.append("# SOL Trace + Sampling Report")
    lines.append("")
    lines.append(f"- Date: `{generated_on}`")
    lines.append(f"- Run ID: `{run_id}`")
    lines.append(f"- Report pass: `{pass_no}`")
    lines.append(f"- Matrix: `3 branches x 3 fixtures x native+docker ({fixture_label})`")
    lines.append("- Checkpointing: `off`")
    lines.append("- Memory profiling: `off`")
    lines.append("- Tracing: `Tracy capture (.tracy) collected for native and docker runs`")
    lines.append(
        "- Stack sampling: `perf record -g` (native and docker runs); flamegraphs generated from folded stacks"
    )
    lines.append("")
    lines.append("## Throughput Summary")
    lines.append("")
    lines.append(
        f"- Best native: `{summary['best_native']['branch']} / {summary['best_native']['fixture']} / {summary['best_native']['throughput_bps']:.2f} bps`"
    )
    lines.append(
        f"- Best docker: `{summary['best_docker']['branch']} / {summary['best_docker']['fixture']} / {summary['best_docker']['throughput_bps']:.2f} bps`"
    )
    lines.append("")
    lines.append("## Tracing + Sampling Artifacts")
    lines.append("")
    lines.append(
        "Raw downloads are provided per run: `trace.tracy`, `trace-capture.log`, `perf.folded`, `perf.script`, and `perf.data`."
    )
    lines.append("")
    lines.append("| env | branch | fixture | bps | perf samples | unique stacks | tracy MiB | tracy zones | artifacts |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---|")
    for tr in trace_records:
        links = []
        if tr["raw_tracy_trace"]:
            links.append(f"[trace.tracy]({tr['raw_tracy_trace']})")
        if tr["raw_tracy_log"]:
            links.append(f"[trace.log]({tr['raw_tracy_log']})")
        if tr["flamegraph"]:
            links.append(f"[flamegraph]({tr['flamegraph']})")
        if tr["summary_json"]:
            links.append(f"[summary]({tr['summary_json']})")
        if tr["report_txt"]:
            links.append(f"[report]({tr['report_txt']})")
        if tr["raw_folded"]:
            links.append(f"[folded]({tr['raw_folded']})")
        if tr["raw_perf_script"]:
            links.append(f"[perf.script]({tr['raw_perf_script']})")
        if tr["raw_perf_data"]:
            links.append(f"[perf.data]({tr['raw_perf_data']})")
        lines.append(
            f"| {tr['env']} | {tr['branch']} | {tr['fixture']} | {tr['throughput_bps']:.3f} | {tr['perf_samples']} | {tr['perf_unique_stacks']} | {tr['tracy_size_mib']:.2f} | {tr['tracy_zones']} | {' '.join(links) if links else 'n/a'} |"
        )
    lines.append("")
    lines.append("## Guard Verdicts")
    lines.append("")
    lines.append("| env | branch | fixture | verdict | baseline samples | failed rules | reports |")
    lines.append("|---|---|---|---|---:|---:|---|")
    if guard_records:
        for gr in sorted(guard_records, key=lambda x: (x["env"], x["branch"], x["fixture"])):
            reports = [f"[json]({gr['json']})"]
            if gr["md"]:
                reports.append(f"[md]({gr['md']})")
            lines.append(
                f"| {gr['env']} | {gr['branch']} | {gr['fixture']} | {gr['verdict']} | {gr['baseline_samples']} | {gr['failed_rules']} | {' '.join(reports)} |"
            )
    else:
        lines.append("| n/a | n/a | n/a | n/a | 0 | 0 | n/a |")
    lines.append("")
    lines.append("## Why Did It Change? (Causal Attribution)")
    lines.append("")
    lines.append("| env | branch | fixture | classification | confidence | baseline samples | throughput delta (%) | z-score |")
    lines.append("|---|---|---|---|---:|---:|---:|---:|")
    if causal_records:
        for rec in sorted(causal_records, key=lambda x: (x["env"], x["branch"], x["fixture"])):
            thr = (rec.get("metrics") or {}).get("throughput_blocks_s") or {}
            delta = thr.get("delta_pct")
            z = thr.get("z_score")
            conf = rec.get("confidence")
            conf_s = f"{(conf * 100.0):.1f}%" if conf is not None else "n/a"
            lines.append(
                f"| {rec.get('env', '')} | {rec.get('branch', '')} | {rec.get('fixture', '')} | {rec.get('classification', '')} | "
                f"{conf_s} | {rec.get('baseline_samples', 0)} | "
                f"{delta if delta is not None else 'n/a'} | {z if z is not None else 'n/a'} |"
            )
    else:
        lines.append("| n/a | n/a | n/a | n/a | n/a | 0 | n/a | n/a |")
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append("- `combined_summary.tsv`")
    lines.append("- `sol-benchmark-transplant-report.html`")
    lines.append("- `sol-benchmark-transplant-report.md`")
    lines.append("- `sol-benchmark-transplant-memory-profiles.json`")
    lines.append("- `causal-attribution.json`")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True, type=Path)
    ap.add_argument(
        "--pages-root",
        type=Path,
        default=Path("/shared/Dropbox/zorp/agents/nockchain-bench-opt"),
    )
    ap.add_argument(
        "--title",
        default="100-block fixtures ext4 matrix x2 (Tracy + perf sampling, memory profiling off)",
    )
    ap.add_argument(
        "--scope",
        default="master + bump PMA + btree, native + docker(16g,memory-swap=16g), 2 passes, checkpointing off, memory profiling off, tracy on, perf native sampling on",
    )
    ap.add_argument(
        "--fixture-label",
        default="100-block fixtures",
    )
    ap.add_argument(
        "--repo-root",
        type=Path,
        default=Path("/shared/nockchain"),
        help="Repo root used for source file + recent commit hints in causal attribution.",
    )
    args = ap.parse_args()

    run_root = args.run_root.resolve()
    combined = run_root / "combined_summary.tsv"
    if not combined.is_file():
        raise FileNotFoundError(f"Missing combined summary: {combined}")

    pages_root = args.pages_root.resolve()
    docs_root = pages_root / "docs" / "nockchain-bench"
    runs_root = docs_root / "sol-runs" / "runs"
    manifest_path = docs_root / "sol-runs" / "runs-manifest.json"

    run_id = parse_run_id(run_root)
    generated_on = dt.date.today().isoformat()
    rows_all = load_rows(combined)
    pass_no, rows = pick_pass_rows(rows_all)

    envs = sorted({r["env"] for r in rows})
    branches = sorted({r["branch"] for r in rows})
    fixtures = sorted({r["fixture"] for r in rows})
    combos = {(r["env"], r["branch"], r["fixture"]) for r in rows}
    expected_rows = len(envs) * len(branches) * len(fixtures)
    if len(rows) != len(combos):
        raise RuntimeError(
            f"Duplicate env/branch/fixture rows detected for selected pass: "
            f"{len(rows)} rows but {len(combos)} unique combinations"
        )
    if len(rows) != expected_rows:
        raise RuntimeError(
            f"Incomplete matrix for selected pass: expected {expected_rows} rows from "
            f"{len(envs)} envs x {len(branches)} branches x {len(fixtures)} fixtures, "
            f"found {len(rows)}"
        )

    summary = compute_summary(rows)

    manifest = {"updated_on": generated_on, "latest_run_id": run_id, "runs": []}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    runs = manifest.get("runs", [])
    runs = [r for r in runs if r.get("run_id") != run_id]
    history = load_history_samples(docs_root, runs, max_runs=24)

    run_dst = runs_root / run_id
    if run_dst.exists():
        shutil.rmtree(run_dst)
    run_dst.mkdir(parents=True, exist_ok=True)

    shutil.copy2(combined, run_dst / "combined_summary.tsv")

    trace_records = build_trace_records(rows, run_dst)
    guard_records = build_guard_records(run_root, run_dst)
    repo_root = args.repo_root.resolve() if args.repo_root else None
    causal_records = build_causal_records(
        rows=rows,
        trace_records=trace_records,
        history=history,
        run_dst=run_dst,
        repo_root=repo_root,
    )
    causal_payload = {
        "generated_on": generated_on,
        "run_id": run_id,
        "records": causal_records,
    }
    (run_dst / "causal-attribution.json").write_text(
        json.dumps(causal_payload, indent=2), encoding="utf-8"
    )

    mem_payload = {
        "generated_on": generated_on,
        "run_id": run_id,
        "source": str(run_root),
        "run_count": 0,
        "runs": [],
    }
    (run_dst / "sol-benchmark-transplant-memory-profiles.json").write_text(
        json.dumps(mem_payload, indent=2), encoding="utf-8"
    )

    report_md = render_md(
        run_id=run_id,
        generated_on=generated_on,
        pass_no=pass_no,
        trace_records=trace_records,
        guard_records=guard_records,
        causal_records=causal_records,
        summary=summary,
        fixture_label=args.fixture_label,
    )
    (run_dst / "sol-benchmark-transplant-report.md").write_text(
        report_md, encoding="utf-8"
    )

    report_html_run = render_html(
        run_id=run_id,
        generated_on=generated_on,
        pass_no=pass_no,
        rows=rows,
        trace_records=trace_records,
        guard_records=guard_records,
        causal_records=causal_records,
        summary=summary,
        base_prefix="./",
        fixture_label=args.fixture_label,
    )
    (run_dst / "sol-benchmark-transplant-report.html").write_text(
        report_html_run, encoding="utf-8"
    )

    # Update moving latest files.
    latest_html = render_html(
        run_id=run_id,
        generated_on=generated_on,
        pass_no=pass_no,
        rows=rows,
        trace_records=trace_records,
        guard_records=guard_records,
        causal_records=causal_records,
        summary=summary,
        base_prefix=f"./sol-runs/runs/{run_id}/",
        fixture_label=args.fixture_label,
    )
    (docs_root / "sol-benchmark-transplant-report.html").write_text(
        latest_html, encoding="utf-8"
    )
    (docs_root / "sol-benchmark-transplant-report.md").write_text(
        report_md, encoding="utf-8"
    )
    (docs_root / "sol-benchmark-transplant-memory-profiles.json").write_text(
        json.dumps(mem_payload, indent=2), encoding="utf-8"
    )
    (docs_root / "sol-benchmark-transplant-causal-attribution.json").write_text(
        json.dumps(causal_payload, indent=2), encoding="utf-8"
    )

    new_entry = {
        "run_id": run_id,
        "date": generated_on,
        "title": args.title,
        "scope": args.scope,
        "summary": summary,
        "links": {
            "report_html": f"./runs/{run_id}/sol-benchmark-transplant-report.html",
            "report_md": f"./runs/{run_id}/sol-benchmark-transplant-report.md",
            "memory_json": f"./runs/{run_id}/sol-benchmark-transplant-memory-profiles.json",
            "causal_json": f"./runs/{run_id}/causal-attribution.json",
            "combined_summary_tsv": f"./runs/{run_id}/combined_summary.tsv",
        },
    }
    runs.insert(0, new_entry)
    manifest["runs"] = runs
    manifest["updated_on"] = generated_on
    manifest["latest_run_id"] = run_id
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Published run_id: {run_id}")
    print(f"Run page: {run_dst / 'sol-benchmark-transplant-report.html'}")
    print(f"Latest page: {docs_root / 'sol-benchmark-transplant-report.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
