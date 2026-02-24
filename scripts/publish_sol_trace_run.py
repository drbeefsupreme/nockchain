#!/usr/bin/env python3
"""Publish a SOL matrix run (with perf + Tracy artifacts) into the GH Pages archive."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
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

CALIBRATION_FUTURE_WINDOW = 3
CALIBRATION_MIN_TRAINING_SAMPLES = 24
CALIBRATION_BINS = 10


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


def clamp(value: float, low: float, high: float) -> float:
    if value < low:
        return low
    if value > high:
        return high
    return value


def clamp_prob(value: float) -> float:
    return clamp(value, 1e-6, 1.0 - 1e-6)


def normal_cdf(value: float) -> float:
    # Φ(x) from erf; stdlib-only for deterministic static publishing environments.
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def two_sided_tail_prob_from_z(z_score: float | None) -> float:
    if z_score is None or not math.isfinite(z_score):
        return 1.0
    z = abs(float(z_score))
    # Two-sided tail: p = 2 * (1 - Φ(|z|)).
    return clamp(2.0 * (1.0 - normal_cdf(z)), 0.0, 1.0)


def raw_change_probability_from_eval(metric_eval: dict, baseline_samples: int) -> float:
    if baseline_samples < 3:
        return 0.5
    z_score = metric_eval.get("z_score")
    z = to_float(str(z_score), float("nan"))
    if not math.isfinite(z):
        return 0.5
    # Evidence-style score under a normal residual assumption.
    p_change = 1.0 - two_sided_tail_prob_from_z(z)
    return clamp_prob(p_change)


def brier_score(points: list[tuple[float, int]]) -> float | None:
    if not points:
        return None
    total = 0.0
    for p, y in points:
        pp = clamp_prob(float(p))
        yy = 1.0 if int(y) else 0.0
        total += (pp - yy) ** 2
    return total / float(len(points))


def build_reliability_bins(
    points: list[tuple[float, int]], bins: int = CALIBRATION_BINS
) -> list[dict]:
    if bins <= 0:
        bins = 10
    counts = [0 for _ in range(bins)]
    prob_sums = [0.0 for _ in range(bins)]
    label_sums = [0.0 for _ in range(bins)]
    for p, y in points:
        pp = clamp_prob(float(p))
        idx = min(int(pp * bins), bins - 1)
        counts[idx] += 1
        prob_sums[idx] += pp
        label_sums[idx] += 1.0 if int(y) else 0.0
    out = []
    for i in range(bins):
        count = counts[i]
        prob_avg = (prob_sums[i] / count) if count else None
        label_avg = (label_sums[i] / count) if count else None
        out.append(
            {
                "bin_index": i,
                "start": i / bins,
                "end": (i + 1) / bins,
                "count": count,
                "avg_predicted": finite_or_none(prob_avg),
                "empirical_rate": finite_or_none(label_avg),
                "abs_gap": finite_or_none(
                    abs(prob_avg - label_avg) if (prob_avg is not None and label_avg is not None) else None
                ),
            }
        )
    return out


def expected_calibration_error(reliability_bins: list[dict]) -> float | None:
    total = sum(int(b.get("count", 0)) for b in reliability_bins)
    if total <= 0:
        return None
    err = 0.0
    for b in reliability_bins:
        count = int(b.get("count", 0))
        avg_pred = b.get("avg_predicted")
        empirical = b.get("empirical_rate")
        if count <= 0 or avg_pred is None or empirical is None:
            continue
        err += (count / total) * abs(float(avg_pred) - float(empirical))
    return err


def fit_isotonic_pav(points: list[tuple[float, int]]) -> list[dict]:
    """Pool-adjacent-violators isotonic regression on probabilities."""
    if not points:
        return []
    sorted_points = sorted((clamp_prob(float(p)), float(int(y))) for p, y in points)
    blocks: list[dict] = []
    for p, y in sorted_points:
        blocks.append(
            {
                "sum_y": y,
                "weight": 1.0,
                "min_p": p,
                "max_p": p,
                "value": y,
            }
        )
        while len(blocks) >= 2 and blocks[-2]["value"] > blocks[-1]["value"]:
            b2 = blocks.pop()
            b1 = blocks.pop()
            sum_y = b1["sum_y"] + b2["sum_y"]
            weight = b1["weight"] + b2["weight"]
            blocks.append(
                {
                    "sum_y": sum_y,
                    "weight": weight,
                    "min_p": b1["min_p"],
                    "max_p": b2["max_p"],
                    "value": (sum_y / weight) if weight > 0 else 0.5,
                }
            )
    return [
        {
            "min_p": finite_or_none(b["min_p"], ndigits=8),
            "max_p": finite_or_none(b["max_p"], ndigits=8),
            "value": finite_or_none(clamp(float(b["value"]), 0.0, 1.0), ndigits=8),
            "weight": int(b["weight"]),
        }
        for b in blocks
    ]


def apply_isotonic_pav(probability: float, model: list[dict]) -> float:
    if not model:
        return clamp_prob(float(probability))
    p = clamp_prob(float(probability))
    for segment in model:
        max_p = float(segment.get("max_p", 1.0))
        if p <= max_p:
            return clamp_prob(float(segment.get("value", p)))
    return clamp_prob(float(model[-1].get("value", p)))


def calibration_metric_summary(points: list[tuple[float, int]]) -> dict:
    rel = build_reliability_bins(points, bins=CALIBRATION_BINS)
    return {
        "count": len(points),
        "positive_rate": finite_or_none(
            (sum(int(y) for _, y in points) / len(points)) if points else None
        ),
        "brier": finite_or_none(brier_score(points)),
        "ece": finite_or_none(expected_calibration_error(rel)),
        "reliability_bins": rel,
    }


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

        raw_change_prob = raw_change_probability_from_eval(
            throughput_eval, baseline_samples=len(baseline)
        )
        confidence = (1.0 - raw_change_prob) if classification == "stable" else raw_change_prob

        records.append(
            {
                "env": row["env"],
                "branch": row["branch"],
                "fixture": row["fixture"],
                "classification": classification,
                "confidence": finite_or_none(confidence, ndigits=4),
                "confidence_model": "raw_z_tail",
                "raw_change_probability": finite_or_none(raw_change_prob, ndigits=6),
                "calibrated_change_probability": finite_or_none(raw_change_prob, ndigits=6),
                "calibration_status": "raw_only",
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


def load_causal_records_for_run(docs_root: Path, run_id: str) -> list[dict]:
    payload_path = docs_root / "sol-runs" / "runs" / run_id / "causal-attribution.json"
    if not payload_path.is_file():
        return []
    try:
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    records = payload.get("records")
    if isinstance(records, list):
        return records
    return []


def record_tuple_key(rec: dict) -> str:
    return tuple_key(str(rec.get("env", "")), str(rec.get("branch", "")), str(rec.get("fixture", "")))


def build_prediction_row(run_id: str, run_date: str, rec: dict) -> dict:
    thr = (rec.get("metrics") or {}).get("throughput_blocks_s") or {}
    baseline_samples = to_int(str(rec.get("baseline_samples", 0)), 0)
    raw = rec.get("raw_change_probability")
    raw_p = to_float(str(raw), float("nan"))
    if not math.isfinite(raw_p):
        raw_p = raw_change_probability_from_eval(thr, baseline_samples)
    else:
        raw_p = clamp_prob(raw_p)
    return {
        "run_id": run_id,
        "date": run_date,
        "env": str(rec.get("env", "")),
        "branch": str(rec.get("branch", "")),
        "fixture": str(rec.get("fixture", "")),
        "tuple_key": record_tuple_key(rec),
        "classification": str(rec.get("classification", "stable")),
        "baseline_samples": baseline_samples,
        "candidate_throughput": to_float(str(thr.get("candidate", "nan")), float("nan")),
        "baseline_median": to_float(str(thr.get("baseline_median", "nan")), float("nan")),
        "expected_low": to_float(str(thr.get("expected_low", "nan")), float("nan")),
        "expected_high": to_float(str(thr.get("expected_high", "nan")), float("nan")),
        "z_score": to_float(str(thr.get("z_score", "nan")), float("nan")),
        "raw_change_probability": raw_p,
        "calibrated_change_probability": raw_p,
        "confidence": raw_p,
        "confidence_model": "raw_z_tail",
        "calibration_status": "raw_only",
        "calibration_training_samples": 0,
        "label": None,
        "label_status": "pending",
        "label_support": None,
        "label_window": CALIBRATION_FUTURE_WINDOW,
        "label_resolved_at_run_id": None,
        "label_future_run_ids": [],
        "label_future_values": [],
    }


def build_calibration_predictions(
    run_entries: list[dict], causal_by_run_id: dict[str, list[dict]]
) -> list[dict]:
    rows: list[dict] = []
    ordered = sorted(run_entries, key=lambda r: str(r.get("run_id", "")))
    for run in ordered:
        run_id = str(run.get("run_id", ""))
        if not run_id:
            continue
        run_date = str(run.get("date", ""))
        for rec in causal_by_run_id.get(run_id, []):
            if not (rec.get("env") and rec.get("branch") and rec.get("fixture")):
                continue
            rows.append(build_prediction_row(run_id, run_date, rec))
    return rows


def resolve_prediction_labels(
    predictions: list[dict], future_window: int = CALIBRATION_FUTURE_WINDOW
) -> None:
    if future_window <= 0:
        future_window = 1
    series: dict[str, list[dict]] = defaultdict(list)
    for p in predictions:
        series[p["tuple_key"]].append(
            {
                "run_id": p["run_id"],
                "value": (
                    float(p["candidate_throughput"])
                    if math.isfinite(float(p["candidate_throughput"]))
                    else float("nan")
                ),
            }
        )
    for key in series:
        series[key].sort(key=lambda row: str(row["run_id"]))

    idx_map: dict[tuple[str, str], int] = {}
    for key, rows in series.items():
        for i, row in enumerate(rows):
            idx_map[(key, str(row["run_id"]))] = i

    for p in predictions:
        p["label"] = None
        p["label_status"] = "pending"
        p["label_support"] = None
        p["label_resolved_at_run_id"] = None
        p["label_future_run_ids"] = []
        p["label_future_values"] = []

        key = p["tuple_key"]
        run_id = p["run_id"]
        rows = series.get(key, [])
        start_idx = idx_map.get((key, run_id))
        if start_idx is None:
            p["label_status"] = "pending_missing_tuple_history"
            continue
        future_rows = [
            row for row in rows[start_idx + 1 :] if math.isfinite(float(row["value"]))
        ][:future_window]
        if len(future_rows) < future_window:
            p["label_status"] = "pending_future_window"
            continue

        candidate = float(p.get("candidate_throughput", float("nan")))
        low = float(p.get("expected_low", float("nan")))
        high = float(p.get("expected_high", float("nan")))
        if not (math.isfinite(candidate) and math.isfinite(low) and math.isfinite(high)):
            p["label_status"] = "pending_missing_band"
            continue

        future_values = [float(row["value"]) for row in future_rows]
        if candidate < low:
            support = sum(1 for v in future_values if v < low)
        elif candidate > high:
            support = sum(1 for v in future_values if v > high)
        else:
            support = sum(1 for v in future_values if (v < low or v > high))
        quorum = (future_window // 2) + 1
        label = 1 if support >= quorum else 0

        p["label"] = label
        p["label_status"] = "resolved"
        p["label_support"] = support
        p["label_resolved_at_run_id"] = str(future_rows[-1]["run_id"])
        p["label_future_run_ids"] = [str(row["run_id"]) for row in future_rows]
        p["label_future_values"] = [finite_or_none(v) for v in future_values]


def apply_online_isotonic_calibration(predictions: list[dict], run_order: list[str]) -> dict:
    by_run: dict[str, list[dict]] = defaultdict(list)
    for p in predictions:
        by_run[str(p["run_id"])].append(p)

    run_summaries = []
    for run_id in sorted(run_order):
        run_preds = by_run.get(run_id, [])
        train_points = [
            (float(p["raw_change_probability"]), int(p["label"]))
            for p in predictions
            if p.get("label") is not None
            and p.get("label_resolved_at_run_id")
            and str(p["label_resolved_at_run_id"]) < run_id
        ]
        has_pos = any(int(y) == 1 for _, y in train_points)
        has_neg = any(int(y) == 0 for _, y in train_points)
        use_isotonic = (
            len(train_points) >= CALIBRATION_MIN_TRAINING_SAMPLES and has_pos and has_neg
        )
        model = fit_isotonic_pav(train_points) if use_isotonic else []
        model_name = "isotonic_online" if use_isotonic else "raw_z_tail"

        for p in run_preds:
            raw_p = clamp_prob(float(p["raw_change_probability"]))
            cal_p = apply_isotonic_pav(raw_p, model) if use_isotonic else raw_p
            cls = str(p.get("classification", "stable"))
            class_conf = (1.0 - cal_p) if cls == "stable" else cal_p
            p["raw_change_probability"] = raw_p
            p["calibrated_change_probability"] = cal_p
            p["confidence"] = clamp_prob(class_conf)
            p["confidence_model"] = model_name
            p["calibration_status"] = model_name
            p["calibration_training_samples"] = len(train_points)

        resolved = [p for p in run_preds if p.get("label") is not None]
        raw_points = [
            (float(p["raw_change_probability"]), int(p["label"])) for p in resolved
        ]
        cal_points = [
            (float(p["calibrated_change_probability"]), int(p["label"])) for p in resolved
        ]
        run_summaries.append(
            {
                "run_id": run_id,
                "tuple_count": len(run_preds),
                "resolved_count": len(resolved),
                "pending_count": max(len(run_preds) - len(resolved), 0),
                "model": model_name,
                "training_samples": len(train_points),
                "brier_raw": finite_or_none(brier_score(raw_points)),
                "brier_calibrated": finite_or_none(brier_score(cal_points)),
                "ece_raw": finite_or_none(
                    expected_calibration_error(
                        build_reliability_bins(raw_points, bins=CALIBRATION_BINS)
                    )
                ),
                "ece_calibrated": finite_or_none(
                    expected_calibration_error(
                        build_reliability_bins(cal_points, bins=CALIBRATION_BINS)
                    )
                ),
                "mean_confidence": finite_or_none(
                    (
                        sum(float(p["confidence"]) for p in run_preds) / len(run_preds)
                        if run_preds
                        else None
                    )
                ),
                "model_curve": model,
            }
        )

    resolved_all = [p for p in predictions if p.get("label") is not None]
    raw_points_all = [
        (float(p["raw_change_probability"]), int(p["label"])) for p in resolved_all
    ]
    cal_points_all = [
        (float(p["calibrated_change_probability"]), int(p["label"]))
        for p in resolved_all
    ]
    return {
        "run_summaries": run_summaries,
        "global_raw": calibration_metric_summary(raw_points_all),
        "global_calibrated": calibration_metric_summary(cal_points_all),
    }


def attach_calibrated_fields_to_causal_records(causal_records: list[dict], predictions: list[dict]) -> None:
    pred_map = {p["tuple_key"]: p for p in predictions}
    for rec in causal_records:
        key = record_tuple_key(rec)
        p = pred_map.get(key)
        if not p:
            continue
        rec["confidence"] = finite_or_none(float(p["confidence"]), ndigits=4)
        rec["confidence_model"] = p.get("confidence_model")
        rec["raw_change_probability"] = finite_or_none(
            float(p["raw_change_probability"]), ndigits=6
        )
        rec["calibrated_change_probability"] = finite_or_none(
            float(p["calibrated_change_probability"]), ndigits=6
        )
        rec["calibration_status"] = p.get("calibration_status")
        rec["calibration_training_samples"] = to_int(
            str(p.get("calibration_training_samples", 0)), 0
        )
        rec["label"] = p.get("label")
        rec["label_status"] = p.get("label_status")
        rec["label_support"] = p.get("label_support")
        rec["label_window"] = p.get("label_window")
        rec["label_resolved_at_run_id"] = p.get("label_resolved_at_run_id")


def write_calibration_feed_tsv(predictions: list[dict], dst_path: Path) -> None:
    columns = [
        "run_id",
        "date",
        "env",
        "branch",
        "fixture",
        "classification",
        "baseline_samples",
        "candidate_throughput",
        "baseline_median",
        "expected_low",
        "expected_high",
        "z_score",
        "raw_change_probability",
        "calibrated_change_probability",
        "confidence",
        "confidence_model",
        "calibration_status",
        "calibration_training_samples",
        "label",
        "label_status",
        "label_support",
        "label_window",
        "label_resolved_at_run_id",
    ]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=columns, delimiter="\t")
    writer.writeheader()
    for p in sorted(
        predictions,
        key=lambda row: (str(row.get("run_id", "")), str(row.get("env", "")), str(row.get("branch", "")), str(row.get("fixture", ""))),
    ):
        writer.writerow(
            {
                "run_id": p.get("run_id", ""),
                "date": p.get("date", ""),
                "env": p.get("env", ""),
                "branch": p.get("branch", ""),
                "fixture": p.get("fixture", ""),
                "classification": p.get("classification", ""),
                "baseline_samples": p.get("baseline_samples", 0),
                "candidate_throughput": finite_or_none(p.get("candidate_throughput")),
                "baseline_median": finite_or_none(p.get("baseline_median")),
                "expected_low": finite_or_none(p.get("expected_low")),
                "expected_high": finite_or_none(p.get("expected_high")),
                "z_score": finite_or_none(p.get("z_score")),
                "raw_change_probability": finite_or_none(p.get("raw_change_probability"), ndigits=8),
                "calibrated_change_probability": finite_or_none(
                    p.get("calibrated_change_probability"), ndigits=8
                ),
                "confidence": finite_or_none(p.get("confidence"), ndigits=8),
                "confidence_model": p.get("confidence_model", ""),
                "calibration_status": p.get("calibration_status", ""),
                "calibration_training_samples": p.get("calibration_training_samples", 0),
                "label": p.get("label"),
                "label_status": p.get("label_status", ""),
                "label_support": p.get("label_support"),
                "label_window": p.get("label_window"),
                "label_resolved_at_run_id": p.get("label_resolved_at_run_id"),
            }
        )
    dst_path.write_text(buf.getvalue(), encoding="utf-8")


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
      <p class="muted">Causal attribution with calibrated confidence. Change probability starts from robust throughput z-score evidence and is calibrated online from prior resolved tuples.</p>
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
        <div class="stat"><div class="k">Confidence (calibrated)</div><div id="causalConfidence" class="v">-</div></div>
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
          <div id="causalNotes" class="mini muted">Confidence is class probability: stable uses 1-p(change), non-stable uses p(change). Calibration may be raw-only when resolved sample history is still small.</div>
        </div>
      </div>
    </section>

    <section class="card links">
      <h2>Artifacts</h2>
      <a href="{base_prefix}combined_summary.tsv" target="_blank">combined_summary.tsv</a>
      <a href="{base_prefix}sol-benchmark-transplant-report.md" target="_blank">report.md</a>
      <a href="{base_prefix}sol-benchmark-transplant-memory-profiles.json" target="_blank">memory-profiles.json</a>
      <a href="{base_prefix}causal-attribution.json" target="_blank">causal-attribution.json</a>
      <a href="{base_prefix}calibration-eval.json" target="_blank">calibration-eval.json</a>
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
      const notesEl = document.getElementById('causalNotes');
      if (!envSel || !branchSel || !fixtureSel || !metricBody || !symbolList || !fileList || !baselineList || !notesEl) return;

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
        notesEl.textContent = 'No calibration details available.';
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

      const rawP = rec.raw_change_probability;
      const calP = rec.calibrated_change_probability;
      const model = rec.confidence_model || 'raw_z_tail';
      const trainN = rec.calibration_training_samples ?? 0;
      notesEl.textContent = `raw p(change) ${{rawP === null || rawP === undefined ? 'n/a' : (formatMaybe(rawP, 3))}}, calibrated p(change) ${{calP === null || calP === undefined ? 'n/a' : (formatMaybe(calP, 3))}}, model ${{model}}, training samples ${{trainN}}.`;
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
    lines.append(
        "| env | branch | fixture | classification | confidence (class) | p(change) raw | p(change) calibrated | model | baseline samples | throughput delta (%) | z-score |"
    )
    lines.append("|---|---|---|---|---:|---:|---:|---|---:|---:|---:|")
    if causal_records:
        for rec in sorted(causal_records, key=lambda x: (x["env"], x["branch"], x["fixture"])):
            thr = (rec.get("metrics") or {}).get("throughput_blocks_s") or {}
            delta = thr.get("delta_pct")
            z = thr.get("z_score")
            conf = rec.get("confidence")
            raw_p = rec.get("raw_change_probability")
            cal_p = rec.get("calibrated_change_probability")
            model = rec.get("confidence_model", "")
            conf_s = f"{(conf * 100.0):.1f}%" if conf is not None else "n/a"
            lines.append(
                f"| {rec.get('env', '')} | {rec.get('branch', '')} | {rec.get('fixture', '')} | {rec.get('classification', '')} | "
                f"{conf_s} | {raw_p if raw_p is not None else 'n/a'} | {cal_p if cal_p is not None else 'n/a'} | {model} | {rec.get('baseline_samples', 0)} | "
                f"{delta if delta is not None else 'n/a'} | {z if z is not None else 'n/a'} |"
            )
    else:
        lines.append("| n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0 | n/a | n/a |")
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append("- `combined_summary.tsv`")
    lines.append("- `sol-benchmark-transplant-report.html`")
    lines.append("- `sol-benchmark-transplant-report.md`")
    lines.append("- `sol-benchmark-transplant-memory-profiles.json`")
    lines.append("- `causal-attribution.json`")
    lines.append("- `calibration-eval.json`")
    return "\n".join(lines) + "\n"


def render_sol_runs_calibration_index(
    updated_on: str,
    calibration_start_run_id: str,
    calibration_runs: list[dict],
    archive_runs: list[dict],
    calibration_feed: dict,
) -> str:
    template = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>SOL Calibration Index</title>
  <style>
    :root {
      --bg: #f2f6fa;
      --card: #ffffff;
      --line: #d4deea;
      --text: #1f2937;
      --muted: #5a687c;
      --accent: #0b6c4f;
      --accent2: #0b5fd7;
      --alert: #8a3b00;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: radial-gradient(circle at 8% -10%, #dbf1e4 0%, #e9f1fb 35%, var(--bg) 68%);
      color: var(--text);
      font-family: "IBM Plex Sans", "Segoe UI", Arial, sans-serif;
    }
    main { max-width: 1260px; margin: 30px auto; padding: 0 16px 42px; display: grid; gap: 14px; }
    .card { background: var(--card); border: 1px solid var(--line); border-radius: 14px; box-shadow: 0 10px 24px rgba(17, 32, 55, .06); padding: 14px; }
    h1, h2, h3 { margin: 0 0 8px; }
    p { margin: 0; }
    .muted { color: var(--muted); }
    .hero { display: grid; gap: 10px; }
    .row4 { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; }
    .row3 { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; }
    .stat { border: 1px solid var(--line); border-radius: 10px; padding: 10px; background: #f9fcff; }
    .k { font-size: .78rem; color: var(--muted); margin-bottom: 4px; }
    .v { font-size: 1.06rem; font-weight: 700; }
    .toolbar { display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 10px; margin-top: 8px; }
    .control label { display: block; font-size: .78rem; color: var(--muted); margin-bottom: 4px; }
    .control select, .control input {
      width: 100%;
      font: inherit;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px 10px;
      background: #fff;
    }
    .chart-wrap { border: 1px solid var(--line); border-radius: 12px; background: linear-gradient(180deg,#fbfdff 0%, #f5f9ff 100%); padding: 8px; }
    .chart-wrap canvas { width: 100%; height: 320px; display: block; }
    .summary { margin-top: 8px; color: var(--muted); font-size: .9rem; }
    .split { display: grid; grid-template-columns: 2fr 1fr; gap: 12px; }
    .table-wrap { border: 1px solid var(--line); border-radius: 12px; overflow: auto; }
    table { width: 100%; border-collapse: collapse; min-width: 960px; font-size: .92rem; }
    th, td { text-align: left; padding: 9px; border-bottom: 1px solid var(--line); vertical-align: top; }
    th { font-size: .78rem; text-transform: uppercase; letter-spacing: .04em; color: #4a5a71; background: #f0f5fb; }
    tr:last-child td { border-bottom: 0; }
    .pill { display: inline-block; padding: 2px 8px; border-radius: 999px; border: 1px solid #cfe0f1; color: #43556d; background: #f6fbff; font-size: .74rem; }
    .links a { display: inline-block; margin-right: 6px; margin-bottom: 6px; padding: 4px 8px; border: 1px solid #c9d9ea; border-radius: 6px; text-decoration: none; color: var(--accent2); font-size: .8rem; background: #f7fbff; }
    .links a:hover { background: #edf5ff; }
    pre { margin: 0; overflow: auto; border: 1px solid var(--line); border-radius: 10px; padding: 10px; background: #0f172a; color: #dbeafe; font-size: .78rem; }
    .warn { color: var(--alert); font-weight: 600; }
    @media (max-width: 980px) {
      .row4 { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .row3 { grid-template-columns: 1fr; }
      .toolbar { grid-template-columns: repeat(3, minmax(0, 1fr)); }
      .split { grid-template-columns: 1fr; }
    }
    @media (max-width: 640px) {
      .row4, .toolbar { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <main>
    <section class="card hero">
      <h1>SOL Regression Calibration Index</h1>
      <p class="muted">This page only tracks calibration-era runs (starting at <code>__CAL_START__</code>). Legacy runs remain available in the archive page.</p>
      <div class="row4">
        <div class="stat"><div class="k">Calibration-Era Runs</div><div class="v" id="statCalRuns">-</div></div>
        <div class="stat"><div class="k">Legacy Archived Runs</div><div class="v" id="statArchiveRuns">-</div></div>
        <div class="stat"><div class="k">Resolved Labels</div><div class="v" id="statResolved">-</div></div>
        <div class="stat"><div class="k">Pending Labels</div><div class="v" id="statPending">-</div></div>
      </div>
      <div class="row3">
        <div class="stat"><div class="k">Calibrated Brier</div><div class="v" id="statBrier">-</div></div>
        <div class="stat"><div class="k">Calibrated ECE</div><div class="v" id="statEce">-</div></div>
        <div class="stat"><div class="k">Updated</div><div class="v" id="statUpdated">-</div></div>
      </div>
      <div class="links">
        <a href="./archive.html">Open Legacy Archive</a>
        <a href="../sol-benchmark-transplant-report.html">Open Moving Latest Report</a>
        <a href="./calibration-feed.json">Calibration Feed JSON</a>
        <a href="./calibration-feed.tsv">Calibration Feed TSV</a>
      </div>
    </section>

    <section class="card split">
      <div>
        <h2>Reliability Curve</h2>
        <div class="chart-wrap"><canvas id="reliabilityChart"></canvas></div>
        <div id="reliabilitySummary" class="summary">Loading reliability bins...</div>
      </div>
      <div>
        <h3>LLM Feed Notes</h3>
        <p class="muted">Use <code>calibration-feed.json</code> for model-aware automation. It includes tuple-level probabilities, labels, label status, and per-run/global calibration metrics.</p>
        <p class="muted" style="margin-top:8px;">Suggested LLM ranking signals:</p>
        <p class="muted">1) high calibrated change probability and high confidence</p>
        <p class="muted">2) persistent negative throughput deltas with resolved labels</p>
        <p class="muted">3) repeated symbols/files in associated per-run causal reports</p>
        <pre id="schemaSnippet"></pre>
      </div>
    </section>

    <section class="card">
      <h2>Tuple Explorer</h2>
      <p class="muted">Mode A: select one tuple and see trajectory across calibration-era runs. Mode B: select one run and compare fixtures for a branch. Here a tuple means one benchmark cell: <code>(environment, branch, fixture)</code> for a specific run.</p>
      <div class="toolbar">
        <div class="control"><label for="modeSel">View</label><select id="modeSel"><option value="fixture_history">Fixture Across Runs</option><option value="branch_snapshot">Branch Across Fixtures</option></select></div>
        <div class="control"><label for="metricSel">Metric</label><select id="metricSel"></select></div>
        <div class="control"><label for="envSel">Environment</label><select id="envSel"></select></div>
        <div class="control"><label for="branchSel">Branch</label><select id="branchSel"></select></div>
        <div class="control" id="fixtureWrap"><label for="fixtureSel">Fixture</label><select id="fixtureSel"></select></div>
        <div class="control" id="runWrap" style="display:none;"><label for="runSel">Run</label><select id="runSel"></select></div>
      </div>
      <div class="chart-wrap"><canvas id="trendChart"></canvas></div>
      <div class="summary" id="trendSummary">Loading explorer data...</div>
    </section>

    <section class="card">
      <h2>Calibration-Era Runs</h2>
      <div class="control" style="margin-bottom:10px;"><label for="searchInput">Filter runs</label><input id="searchInput" type="search" placeholder="run id, title, scope, branch..." /></div>
      <div id="runCount" class="summary">-</div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr><th>Run</th><th>Scope</th><th>Resolved/Total</th><th>Brier (cal)</th><th>ECE (cal)</th><th>Links</th></tr>
          </thead>
          <tbody id="runRows"></tbody>
        </table>
      </div>
    </section>

    <section class="card">
      <h2>How To Read These Numbers (Pedagogical Explainer)</h2>
      <p class="muted">First, what is a <strong>tuple</strong> on this page? It is one benchmark cell identified by <code>(environment, branch, fixture)</code> in one run. Example: <code>(docker, btree, v1)</code>.</p>
      <p class="muted" style="margin-top:8px;">For each tuple, the model estimates <strong>p(change)</strong>: the probability that this tuple's throughput has a real persistent shift (not just normal run-to-run noise). The starting signal is a throughput <strong>z-score</strong>, which says how far today's throughput is from baseline in units of baseline spread. Roughly: <code>z=0</code> is typical, larger <code>|z|</code> means more unusual.</p>
      <p class="muted" style="margin-top:8px;">Raw probabilities are then <strong>calibrated</strong> using prior resolved tuples so reported percentages match observed frequencies better. This is done with isotonic calibration (a monotonic remapping learned from historical prediction/outcome pairs).</p>
      <p class="muted" style="margin-top:8px;"><strong>Confidence</strong> is the probability that the displayed class is correct. For classes <code>regression</code> or <code>improvement</code>, confidence is <code>p(change)</code>. For class <code>stable</code>, confidence is <code>1 - p(change)</code>.</p>
      <p class="muted" style="margin-top:8px;"><strong>Brier score</strong> measures probability error using squared loss: for each resolved tuple, error is <code>(p - y)^2</code>, where <code>p</code> is predicted probability and <code>y</code> is the actual outcome (<code>1</code> means persistent change happened, <code>0</code> means it did not). The page reports the mean of this error. Example: predicting <code>0.80</code> when <code>y=1</code> gives <code>0.04</code>; predicting <code>0.80</code> when <code>y=0</code> gives <code>0.64</code>.</p>
      <p class="muted" style="margin-top:8px;"><strong>Reliability curve</strong>: we group predicted probabilities into ranges called bins. A <code>0.1</code>-wide bin is a 10-point probability interval, like <code>[0.2, 0.3)</code> or <code>[0.7, 0.8)</code>. For each bin, we compare (1) the average predicted probability vs (2) the observed positive rate (how often change actually happened). Perfect calibration is the diagonal line <code>y=x</code>.</p>
      <p class="muted" style="margin-top:8px;"><strong>ECE (Expected Calibration Error)</strong> compresses that curve into one number: in each bin, compute the absolute gap between predicted and observed rate, weight by that bin's sample share, and sum: <code>ECE = sum_k (n_k / N) * |acc_k - conf_k|</code>. Lower is better; <code>0</code> means perfect calibration on the evaluated data.</p>
      <p class="muted" style="margin-top:8px;">Labeling rule for calibration: each tuple is resolved only after a fixed future window of runs (currently <code>3</code>). If later runs support a persistent out-of-band shift for that tuple, label is positive. Before enough future runs exist, labels are pending, so calibration statistics are intentionally sparse right after rollout.</p>
    </section>
  </main>
  <script>
    const CALIBRATION_RUNS = __CAL_RUNS_JSON__;
    const ARCHIVE_RUNS = __ARCHIVE_RUNS_JSON__;
    const CALIBRATION_FEED = __CAL_FEED_JSON__;
    const UPDATED_ON = "__UPDATED_ON__";

    const metricDefs = {
      candidate_throughput: { label: "Throughput (blocks/s)", digits: 3 },
      calibrated_change_probability: { label: "Calibrated Change Probability", digits: 3 },
      raw_change_probability: { label: "Raw Change Probability", digits: 3 },
      confidence: { label: "Class Confidence", digits: 3 }
    };

    const el = {
      statCalRuns: document.getElementById("statCalRuns"),
      statArchiveRuns: document.getElementById("statArchiveRuns"),
      statResolved: document.getElementById("statResolved"),
      statPending: document.getElementById("statPending"),
      statBrier: document.getElementById("statBrier"),
      statEce: document.getElementById("statEce"),
      statUpdated: document.getElementById("statUpdated"),
      reliabilityChart: document.getElementById("reliabilityChart"),
      reliabilitySummary: document.getElementById("reliabilitySummary"),
      schemaSnippet: document.getElementById("schemaSnippet"),
      modeSel: document.getElementById("modeSel"),
      metricSel: document.getElementById("metricSel"),
      envSel: document.getElementById("envSel"),
      branchSel: document.getElementById("branchSel"),
      fixtureSel: document.getElementById("fixtureSel"),
      runSel: document.getElementById("runSel"),
      fixtureWrap: document.getElementById("fixtureWrap"),
      runWrap: document.getElementById("runWrap"),
      trendChart: document.getElementById("trendChart"),
      trendSummary: document.getElementById("trendSummary"),
      searchInput: document.getElementById("searchInput"),
      runCount: document.getElementById("runCount"),
      runRows: document.getElementById("runRows"),
    };

    const runSummaryMap = Object.fromEntries((CALIBRATION_FEED.run_summaries || []).map((r) => [r.run_id, r]));
    const predictions = CALIBRATION_FEED.predictions || [];
    const runIdsDesc = CALIBRATION_RUNS.map((r) => r.run_id);
    const runIdsAsc = runIdsDesc.slice().sort((a, b) => String(a).localeCompare(String(b), undefined, { numeric: true }));
    const trendState = { mode: "fixture_history", metric: "candidate_throughput", env: "native", branch: "master", fixture: "v0", runId: runIdsDesc[0] || "" };

    function fmt(value, digits = 3) {
      const n = Number(value);
      if (!Number.isFinite(n)) return "n/a";
      return n.toFixed(digits);
    }

    function fillSelect(sel, values, desired = "") {
      sel.innerHTML = values.map((v) => `<option value="${v}">${v}</option>`).join("");
      if (!values.length) return;
      if (desired && values.includes(desired)) sel.value = desired;
      else sel.value = values[0];
    }

    function uniq(values) {
      return Array.from(new Set(values)).sort((a, b) => String(a).localeCompare(String(b), undefined, { numeric: true, sensitivity: "base" }));
    }

    function setupCanvas(canvas) {
      const ratio = window.devicePixelRatio || 1;
      const width = Math.max(canvas.clientWidth || 320, 320);
      const height = Math.max(canvas.clientHeight || 260, 240);
      canvas.width = Math.floor(width * ratio);
      canvas.height = Math.floor(height * ratio);
      const ctx = canvas.getContext("2d");
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      ctx.clearRect(0, 0, width, height);
      return { ctx, width, height };
    }

    function drawEmpty(canvas, message) {
      const { ctx, width, height } = setupCanvas(canvas);
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, width, height);
      ctx.fillStyle = "#5a687c";
      ctx.font = '14px "IBM Plex Sans", sans-serif';
      ctx.textAlign = "center";
      ctx.fillText(message, width / 2, height / 2);
    }

    function drawReliability() {
      const bins = CALIBRATION_FEED?.global_metrics?.calibrated?.reliability_bins || [];
      const used = bins.filter((b) => Number(b.count) > 0 && Number.isFinite(Number(b.avg_predicted)) && Number.isFinite(Number(b.empirical_rate)));
      if (!used.length) {
        drawEmpty(el.reliabilityChart, "No resolved calibration bins yet.");
        el.reliabilitySummary.textContent = "Reliability curve will appear after enough future windows resolve labels.";
        return;
      }

      const { ctx, width, height } = setupCanvas(el.reliabilityChart);
      const margin = { top: 18, right: 20, bottom: 42, left: 54 };
      const innerW = width - margin.left - margin.right;
      const innerH = height - margin.top - margin.bottom;
      const toX = (v) => margin.left + v * innerW;
      const toY = (v) => margin.top + (1 - v) * innerH;

      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, width, height);

      ctx.strokeStyle = "#e1e8f0";
      for (let i = 0; i <= 5; i++) {
        const t = i / 5;
        const x = toX(t);
        const y = toY(t);
        ctx.beginPath(); ctx.moveTo(toX(0), y); ctx.lineTo(toX(1), y); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(x, toY(0)); ctx.lineTo(x, toY(1)); ctx.stroke();
      }

      ctx.strokeStyle = "#9ca8ba";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(toX(0), toY(0));
      ctx.lineTo(toX(1), toY(1));
      ctx.stroke();

      ctx.strokeStyle = "#0b5fd7";
      ctx.lineWidth = 2;
      ctx.beginPath();
      used.forEach((b, i) => {
        const x = toX(Number(b.avg_predicted));
        const y = toY(Number(b.empirical_rate));
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.stroke();

      used.forEach((b) => {
        const x = toX(Number(b.avg_predicted));
        const y = toY(Number(b.empirical_rate));
        const r = 2 + Math.min(8, Math.sqrt(Number(b.count || 1)));
        ctx.fillStyle = "#0b6c4f";
        ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fill();
      });

      ctx.fillStyle = "#334155";
      ctx.font = '12px "IBM Plex Sans", sans-serif';
      ctx.textAlign = "left";
      ctx.fillText("Predicted probability", margin.left, height - 10);
      ctx.save();
      ctx.translate(12, margin.top + innerH / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText("Empirical positive rate", 0, 0);
      ctx.restore();

      const ece = fmt(CALIBRATION_FEED?.global_metrics?.calibrated?.ece, 4);
      const brier = fmt(CALIBRATION_FEED?.global_metrics?.calibrated?.brier, 4);
      el.reliabilitySummary.textContent = `Calibrated reliability over ${used.length} populated bins. Brier ${brier}, ECE ${ece}.`;
    }

    function seriesFixtureHistory() {
      const out = [];
      for (const runId of runIdsAsc) {
        const row = predictions.find((p) => p.run_id === runId && p.env === trendState.env && p.branch === trendState.branch && p.fixture === trendState.fixture);
        if (!row) continue;
        const v = Number(row[trendState.metric]);
        if (!Number.isFinite(v)) continue;
        out.push({ label: runId, value: v });
      }
      return out;
    }

    function seriesBranchSnapshot() {
      return predictions
        .filter((p) => p.run_id === trendState.runId && p.env === trendState.env && p.branch === trendState.branch)
        .map((p) => ({ label: p.fixture, value: Number(p[trendState.metric]) }))
        .filter((x) => Number.isFinite(x.value))
        .sort((a, b) => String(a.label).localeCompare(String(b.label), undefined, { numeric: true }));
    }

    function drawTrendSeries(items, isBar) {
      if (!items.length) {
        drawEmpty(el.trendChart, "No rows for this selection.");
        return;
      }
      const { ctx, width, height } = setupCanvas(el.trendChart);
      const margin = { top: 18, right: 20, bottom: 58, left: 66 };
      const innerW = width - margin.left - margin.right;
      const innerH = height - margin.top - margin.bottom;

      let minV = Math.min(...items.map((i) => i.value));
      let maxV = Math.max(...items.map((i) => i.value));
      if (minV === maxV) { const pad = Math.max(0.1, Math.abs(minV) * 0.1); minV -= pad; maxV += pad; }
      const toY = (v) => margin.top + ((maxV - v) / (maxV - minV)) * innerH;

      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, width, height);
      ctx.strokeStyle = "#e3ebf4";
      ctx.fillStyle = "#5a687c";
      ctx.font = '11px "IBM Plex Sans", sans-serif';
      ctx.textAlign = "right";
      for (let i = 0; i <= 5; i++) {
        const y = margin.top + (innerH * i / 5);
        const v = maxV - ((maxV - minV) * i / 5);
        ctx.beginPath(); ctx.moveTo(margin.left, y); ctx.lineTo(width - margin.right, y); ctx.stroke();
        ctx.fillText(fmt(v, metricDefs[trendState.metric]?.digits ?? 3), margin.left - 8, y + 3);
      }

      if (!isBar) {
        const toX = (idx) => items.length === 1 ? margin.left + innerW / 2 : margin.left + (innerW * idx / (items.length - 1));
        ctx.strokeStyle = "#0b5fd7";
        ctx.lineWidth = 2;
        ctx.beginPath();
        items.forEach((it, i) => { const x = toX(i); const y = toY(it.value); if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); });
        ctx.stroke();
        ctx.fillStyle = "#0b6c4f";
        items.forEach((it, i) => { const x = toX(i); const y = toY(it.value); ctx.beginPath(); ctx.arc(x, y, 3, 0, Math.PI * 2); ctx.fill(); });
        const step = Math.max(1, Math.ceil(items.length / 7));
        ctx.fillStyle = "#5a687c";
        ctx.textAlign = "center";
        items.forEach((it, i) => {
          if (i % step !== 0 && i !== items.length - 1) return;
          const x = toX(i);
          ctx.save();
          ctx.translate(x, height - margin.bottom + 18);
          ctx.rotate(-0.52);
          ctx.fillText(String(it.label).slice(2), 0, 0);
          ctx.restore();
        });
      } else {
        const slot = innerW / items.length;
        const bw = Math.max(12, slot * 0.62);
        items.forEach((it, i) => {
          const x = margin.left + i * slot + (slot - bw) / 2;
          const y = toY(it.value);
          const bh = margin.top + innerH - y;
          ctx.fillStyle = "#0b6c4f";
          ctx.fillRect(x, y, bw, bh);
          ctx.fillStyle = "#334155";
          ctx.font = '11px "IBM Plex Sans", sans-serif';
          ctx.textAlign = "center";
          ctx.fillText(String(it.label), x + bw / 2, height - margin.bottom + 16);
        });
      }
      ctx.fillStyle = "#334155";
      ctx.textAlign = "left";
      ctx.font = '12px "IBM Plex Sans", sans-serif';
      ctx.fillText(metricDefs[trendState.metric]?.label || trendState.metric, margin.left, 14);
    }

    function renderTrend() {
      const fixtureMode = trendState.mode === "fixture_history";
      el.fixtureWrap.style.display = fixtureMode ? "" : "none";
      el.runWrap.style.display = fixtureMode ? "none" : "";
      const points = fixtureMode ? seriesFixtureHistory() : seriesBranchSnapshot();
      drawTrendSeries(points, !fixtureMode);
      if (!points.length) {
        el.trendSummary.textContent = "No records for this explorer selection.";
        return;
      }
      const values = points.map((p) => p.value);
      el.trendSummary.textContent = `${metricDefs[trendState.metric]?.label || trendState.metric}: range ${fmt(Math.min(...values), 3)} .. ${fmt(Math.max(...values), 3)} over ${points.length} point(s).`;
    }

    function renderRunTable() {
      const q = (el.searchInput.value || "").trim().toLowerCase();
      const rows = CALIBRATION_RUNS.filter((run) => {
        if (!q) return true;
        const text = [run.run_id, run.date, run.title, run.scope, run.summary?.best_native?.branch, run.summary?.best_docker?.branch].join(" ").toLowerCase();
        return text.includes(q);
      });
      el.runCount.textContent = `${rows.length} run(s) shown`;
      if (!rows.length) {
        el.runRows.innerHTML = '<tr><td colspan="6" class="warn">No calibration runs matched the filter.</td></tr>';
        return;
      }
      el.runRows.innerHTML = rows.map((run) => {
        const rs = runSummaryMap[run.run_id] || {};
        const resolved = Number(rs.resolved_count || 0);
        const total = Number(rs.tuple_count || 0);
        const links = run.links || {};
        return `<tr>
          <td><strong>${run.run_id}</strong><div class="muted">${run.date || ""}</div><div><span class="pill">${run.title || "run"}</span></div></td>
          <td>${run.scope || '<span class="muted">n/a</span>'}</td>
          <td>${resolved}/${total}</td>
          <td>${fmt(rs.brier_calibrated, 4)}</td>
          <td>${fmt(rs.ece_calibrated, 4)}</td>
          <td class="links">
            <a href="${links.report_html || "#"}">HTML</a>
            <a href="${links.report_md || "#"}">Markdown</a>
            <a href="${links.combined_summary_tsv || "#"}">Summary TSV</a>
            ${links.causal_json ? `<a href="${links.causal_json}">Causal JSON</a>` : ""}
            ${links.calibration_eval_json ? `<a href="${links.calibration_eval_json}">Calibration Eval</a>` : ""}
          </td>
        </tr>`;
      }).join("");
    }

    function init() {
      el.statCalRuns.textContent = String(CALIBRATION_RUNS.length);
      el.statArchiveRuns.textContent = String(ARCHIVE_RUNS.length);
      const globalCal = CALIBRATION_FEED?.global_metrics?.calibrated || {};
      const resolved = Number(globalCal.count || 0);
      const pending = Number((CALIBRATION_FEED?.prediction_count || 0) - resolved);
      el.statResolved.textContent = String(resolved);
      el.statPending.textContent = String(Math.max(0, pending));
      el.statBrier.textContent = fmt(globalCal.brier, 4);
      el.statEce.textContent = fmt(globalCal.ece, 4);
      el.statUpdated.textContent = UPDATED_ON;

      el.schemaSnippet.textContent = JSON.stringify(
        {
          run_id: "YYYYMMDD_HHMMSS",
          env: "native|docker",
          branch: "master|bump PMA|btree",
          fixture: "v0|v1|v2",
          classification: "stable|regression|improvement",
          raw_change_probability: 0.0,
          calibrated_change_probability: 0.0,
          confidence: 0.0,
          label: 0,
          label_status: "resolved|pending_future_window"
        },
        null,
        2
      );

      const envs = uniq(predictions.map((p) => p.env));
      const branches = uniq(predictions.map((p) => p.branch));
      const fixtures = uniq(predictions.map((p) => p.fixture));
      fillSelect(el.metricSel, Object.keys(metricDefs), trendState.metric);
      fillSelect(el.envSel, envs, trendState.env);
      fillSelect(el.branchSel, branches, trendState.branch);
      fillSelect(el.fixtureSel, fixtures, trendState.fixture);
      fillSelect(el.runSel, runIdsDesc, trendState.runId);
      trendState.env = el.envSel.value;
      trendState.branch = el.branchSel.value;
      trendState.fixture = el.fixtureSel.value;
      trendState.runId = el.runSel.value;

      el.modeSel.addEventListener("change", () => { trendState.mode = el.modeSel.value; renderTrend(); });
      el.metricSel.addEventListener("change", () => { trendState.metric = el.metricSel.value; renderTrend(); });
      el.envSel.addEventListener("change", () => { trendState.env = el.envSel.value; renderTrend(); });
      el.branchSel.addEventListener("change", () => { trendState.branch = el.branchSel.value; renderTrend(); });
      el.fixtureSel.addEventListener("change", () => { trendState.fixture = el.fixtureSel.value; renderTrend(); });
      el.runSel.addEventListener("change", () => { trendState.runId = el.runSel.value; renderTrend(); });
      el.searchInput.addEventListener("input", renderRunTable);
      window.addEventListener("resize", () => { drawReliability(); renderTrend(); });

      drawReliability();
      renderTrend();
      renderRunTable();
    }

    init();
  </script>
</body>
</html>
"""
    return (
        template.replace("__CAL_RUNS_JSON__", json.dumps(calibration_runs))
        .replace("__ARCHIVE_RUNS_JSON__", json.dumps(archive_runs))
        .replace("__CAL_FEED_JSON__", json.dumps(calibration_feed))
        .replace("__UPDATED_ON__", html_escape(updated_on))
        .replace("__CAL_START__", html_escape(calibration_start_run_id))
    )


def render_sol_runs_archive_index(
    updated_on: str,
    calibration_start_run_id: str,
    archive_runs: list[dict],
    calibration_runs: list[dict],
) -> str:
    template = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>SOL Legacy Run Archive</title>
  <style>
    :root {
      --bg: #f5f7fb;
      --card: #ffffff;
      --line: #d7deea;
      --text: #1f2937;
      --muted: #5b6778;
      --accent: #0b5fd7;
    }
    * { box-sizing: border-box; }
    body { margin: 0; background: radial-gradient(circle at 15% -10%, #e8efff 0%, var(--bg) 62%); color: var(--text); font-family: "IBM Plex Sans", "Segoe UI", Arial, sans-serif; }
    main { max-width: 1220px; margin: 32px auto; padding: 0 16px 40px; display: grid; gap: 14px; }
    .card { background: var(--card); border: 1px solid var(--line); border-radius: 14px; box-shadow: 0 10px 24px rgba(16, 31, 54, .06); padding: 14px; }
    .muted { color: var(--muted); }
    .row { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; margin-top: 10px; }
    .stat { border: 1px solid var(--line); border-radius: 10px; padding: 10px; background: #fafcff; }
    .k { font-size: .78rem; color: var(--muted); margin-bottom: 4px; }
    .v { font-size: 1.02rem; font-weight: 700; }
    .links a { display: inline-block; margin-right: 8px; margin-bottom: 6px; text-decoration: none; border: 1px solid #c9d7ea; border-radius: 6px; padding: 4px 8px; color: var(--accent); background: #f7fbff; font-size: .82rem; }
    .search { margin-top: 10px; width: 100%; border: 1px solid var(--line); border-radius: 8px; padding: 8px 10px; font: inherit; }
    .table-wrap { border: 1px solid var(--line); border-radius: 12px; overflow: auto; }
    table { width: 100%; border-collapse: collapse; min-width: 940px; font-size: .92rem; }
    th, td { text-align: left; border-bottom: 1px solid var(--line); padding: 9px; vertical-align: top; }
    th { font-size: .78rem; text-transform: uppercase; letter-spacing: .04em; color: #4a596f; background: #f0f4fb; }
    tr:last-child td { border-bottom: 0; }
    @media (max-width: 780px) { .row { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <main>
    <section class="card">
      <h1>Legacy SOL Run Archive</h1>
      <p class="muted">Runs before calibration era (<code>__CAL_START__</code>). These snapshots are preserved as historical context and are not retrofitted with new calibration metrics.</p>
      <div class="row">
        <div class="stat"><div class="k">Legacy Runs</div><div class="v" id="statLegacy">-</div></div>
        <div class="stat"><div class="k">Calibration-Era Runs</div><div class="v" id="statCal">-</div></div>
        <div class="stat"><div class="k">Updated</div><div class="v" id="statUpdated">-</div></div>
      </div>
      <div class="links" style="margin-top:10px;">
        <a href="./index.html">Open Calibration Index</a>
        <a href="../sol-benchmark-transplant-report.html">Open Moving Latest Report</a>
      </div>
    </section>
    <section class="card">
      <h2>Runs</h2>
      <input id="searchInput" class="search" type="search" placeholder="Filter by run id, title, scope..." />
      <p id="count" class="muted" style="margin:8px 0;">-</p>
      <div class="table-wrap">
        <table>
          <thead><tr><th>Run</th><th>Scope</th><th>Best Native</th><th>Best Docker</th><th>Links</th></tr></thead>
          <tbody id="rows"></tbody>
        </table>
      </div>
    </section>
  </main>
  <script>
    const ARCHIVE_RUNS = __ARCHIVE_RUNS_JSON__;
    const CALIBRATION_RUNS = __CAL_RUNS_JSON__;
    const UPDATED_ON = "__UPDATED_ON__";
    const el = {
      statLegacy: document.getElementById("statLegacy"),
      statCal: document.getElementById("statCal"),
      statUpdated: document.getElementById("statUpdated"),
      searchInput: document.getElementById("searchInput"),
      count: document.getElementById("count"),
      rows: document.getElementById("rows"),
    };

    function fmtBest(best) {
      if (!best) return "n/a";
      return `${best.branch} / ${best.fixture} / ${Number(best.throughput_bps).toFixed(2)} bps`;
    }

    function renderRows() {
      const q = (el.searchInput.value || "").trim().toLowerCase();
      const rows = ARCHIVE_RUNS.filter((run) => {
        if (!q) return true;
        const txt = [run.run_id, run.date, run.title, run.scope].join(" ").toLowerCase();
        return txt.includes(q);
      });
      el.count.textContent = `${rows.length} legacy run(s) shown`;
      if (!rows.length) {
        el.rows.innerHTML = '<tr><td colspan="5" class="muted">No runs matched the filter.</td></tr>';
        return;
      }
      el.rows.innerHTML = rows.map((run) => {
        const links = run.links || {};
        return `<tr>
          <td><strong>${run.run_id}</strong><div class="muted">${run.date || ""}</div><div class="muted">${run.title || ""}</div></td>
          <td>${run.scope || '<span class="muted">n/a</span>'}</td>
          <td>${fmtBest(run.summary?.best_native)}</td>
          <td>${fmtBest(run.summary?.best_docker)}</td>
          <td class="links">
            <a href="${links.report_html || "#"}">HTML</a>
            <a href="${links.report_md || "#"}">Markdown</a>
            <a href="${links.combined_summary_tsv || "#"}">Summary TSV</a>
            ${links.causal_json ? `<a href="${links.causal_json}">Causal JSON</a>` : ""}
          </td>
        </tr>`;
      }).join("");
    }

    el.statLegacy.textContent = String(ARCHIVE_RUNS.length);
    el.statCal.textContent = String(CALIBRATION_RUNS.length);
    el.statUpdated.textContent = UPDATED_ON;
    el.searchInput.addEventListener("input", renderRows);
    renderRows();
  </script>
</body>
</html>
"""
    return (
        template.replace("__ARCHIVE_RUNS_JSON__", json.dumps(archive_runs))
        .replace("__CAL_RUNS_JSON__", json.dumps(calibration_runs))
        .replace("__UPDATED_ON__", html_escape(updated_on))
        .replace("__CAL_START__", html_escape(calibration_start_run_id))
    )


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
    calibration_start_run_id = str(manifest.get("calibration_start_run_id", "")).strip()
    if not calibration_start_run_id:
        # First publish after rollout: start calibration era now and keep previous runs as legacy archive.
        calibration_start_run_id = run_id

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

    # Build calibration-era dataset from existing run artifacts plus current run.
    calibration_runs_existing = [
        r for r in runs if str(r.get("run_id", "")) >= calibration_start_run_id
    ]
    causal_by_run_id: dict[str, list[dict]] = {}
    for run in calibration_runs_existing:
        rid = str(run.get("run_id", ""))
        if not rid:
            continue
        causal_by_run_id[rid] = load_causal_records_for_run(docs_root, rid)
    causal_by_run_id[run_id] = causal_records
    calibration_run_entries = calibration_runs_existing + [
        {
            "run_id": run_id,
            "date": generated_on,
            "title": args.title,
            "scope": args.scope,
            "summary": summary,
        }
    ]
    prediction_rows = build_calibration_predictions(
        run_entries=calibration_run_entries,
        causal_by_run_id=causal_by_run_id,
    )
    resolve_prediction_labels(prediction_rows, future_window=CALIBRATION_FUTURE_WINDOW)
    run_order = sorted({str(r.get("run_id", "")) for r in calibration_run_entries if r.get("run_id")})
    calibration_eval = apply_online_isotonic_calibration(prediction_rows, run_order)

    current_prediction_rows = [p for p in prediction_rows if p.get("run_id") == run_id]
    attach_calibrated_fields_to_causal_records(causal_records, current_prediction_rows)

    causal_payload = {
        "generated_on": generated_on,
        "run_id": run_id,
        "confidence_semantics": "Confidence is class probability. For stable class confidence=1-p(change), for regression/improvement confidence=p(change).",
        "records": causal_records,
    }
    (run_dst / "causal-attribution.json").write_text(
        json.dumps(causal_payload, indent=2), encoding="utf-8"
    )

    current_run_summary = next(
        (
            r
            for r in calibration_eval.get("run_summaries", [])
            if str(r.get("run_id", "")) == run_id
        ),
        {},
    )

    current_run_eval_payload = {
        "generated_on": generated_on,
        "run_id": run_id,
        "calibration_start_run_id": calibration_start_run_id,
        "future_window": CALIBRATION_FUTURE_WINDOW,
        "min_training_samples": CALIBRATION_MIN_TRAINING_SAMPLES,
        "run_summary": current_run_summary,
        "global_metrics": {
            "raw": calibration_eval.get("global_raw", {}),
            "calibrated": calibration_eval.get("global_calibrated", {}),
        },
        "tuples": current_prediction_rows,
    }
    (run_dst / "calibration-eval.json").write_text(
        json.dumps(current_run_eval_payload, indent=2), encoding="utf-8"
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
    (docs_root / "sol-benchmark-transplant-calibration-eval.json").write_text(
        json.dumps(current_run_eval_payload, indent=2), encoding="utf-8"
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
            "calibration_eval_json": f"./runs/{run_id}/calibration-eval.json",
            "combined_summary_tsv": f"./runs/{run_id}/combined_summary.tsv",
        },
        "calibration": {
            "resolved_count": current_run_summary.get("resolved_count"),
            "tuple_count": current_run_summary.get("tuple_count"),
            "brier_calibrated": current_run_summary.get("brier_calibrated"),
            "ece_calibrated": current_run_summary.get("ece_calibrated"),
            "model": current_run_summary.get("model"),
        },
    }
    runs.insert(0, new_entry)

    calibration_runs = [
        r for r in runs if str(r.get("run_id", "")) >= calibration_start_run_id
    ]
    archive_runs = [
        r for r in runs if str(r.get("run_id", "")) < calibration_start_run_id
    ]

    def safe_prediction_row(p: dict) -> dict:
        return {
            "run_id": p.get("run_id"),
            "date": p.get("date"),
            "env": p.get("env"),
            "branch": p.get("branch"),
            "fixture": p.get("fixture"),
            "classification": p.get("classification"),
            "baseline_samples": to_int(str(p.get("baseline_samples", 0)), 0),
            "candidate_throughput": finite_or_none(
                p.get("candidate_throughput"), ndigits=6
            ),
            "baseline_median": finite_or_none(p.get("baseline_median"), ndigits=6),
            "expected_low": finite_or_none(p.get("expected_low"), ndigits=6),
            "expected_high": finite_or_none(p.get("expected_high"), ndigits=6),
            "z_score": finite_or_none(p.get("z_score"), ndigits=6),
            "raw_change_probability": finite_or_none(
                p.get("raw_change_probability"), ndigits=8
            ),
            "calibrated_change_probability": finite_or_none(
                p.get("calibrated_change_probability"), ndigits=8
            ),
            "confidence": finite_or_none(p.get("confidence"), ndigits=8),
            "confidence_model": p.get("confidence_model"),
            "calibration_status": p.get("calibration_status"),
            "calibration_training_samples": to_int(
                str(p.get("calibration_training_samples", 0)), 0
            ),
            "label": p.get("label"),
            "label_status": p.get("label_status"),
            "label_support": p.get("label_support"),
            "label_window": p.get("label_window"),
            "label_resolved_at_run_id": p.get("label_resolved_at_run_id"),
            "label_future_run_ids": p.get("label_future_run_ids") or [],
            "label_future_values": p.get("label_future_values") or [],
        }

    feed_predictions = [safe_prediction_row(p) for p in prediction_rows]
    calibration_feed = {
        "schema_version": 1,
        "generated_on": generated_on,
        "calibration_start_run_id": calibration_start_run_id,
        "future_window": CALIBRATION_FUTURE_WINDOW,
        "min_training_samples": CALIBRATION_MIN_TRAINING_SAMPLES,
        "prediction_count": len(feed_predictions),
        "global_metrics": {
            "raw": calibration_eval.get("global_raw", {}),
            "calibrated": calibration_eval.get("global_calibrated", {}),
        },
        "run_summaries": calibration_eval.get("run_summaries", []),
        "predictions": feed_predictions,
    }
    sol_runs_root = docs_root / "sol-runs"
    (sol_runs_root / "calibration-feed.json").write_text(
        json.dumps(calibration_feed, indent=2), encoding="utf-8"
    )
    write_calibration_feed_tsv(feed_predictions, sol_runs_root / "calibration-feed.tsv")

    (sol_runs_root / "index.html").write_text(
        render_sol_runs_calibration_index(
            updated_on=generated_on,
            calibration_start_run_id=calibration_start_run_id,
            calibration_runs=calibration_runs,
            archive_runs=archive_runs,
            calibration_feed=calibration_feed,
        ),
        encoding="utf-8",
    )
    (sol_runs_root / "archive.html").write_text(
        render_sol_runs_archive_index(
            updated_on=generated_on,
            calibration_start_run_id=calibration_start_run_id,
            archive_runs=archive_runs,
            calibration_runs=calibration_runs,
        ),
        encoding="utf-8",
    )

    manifest["runs"] = runs
    manifest["updated_on"] = generated_on
    manifest["latest_run_id"] = run_id
    manifest["calibration_start_run_id"] = calibration_start_run_id
    manifest["calibration_run_count"] = len(calibration_runs)
    manifest["archive_run_count"] = len(archive_runs)
    manifest["latest_calibration_run_id"] = (
        calibration_runs[0].get("run_id") if calibration_runs else None
    )
    manifest["calibration_metrics"] = {
        "raw": calibration_eval.get("global_raw", {}),
        "calibrated": calibration_eval.get("global_calibrated", {}),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Published run_id: {run_id}")
    print(f"Run page: {run_dst / 'sol-benchmark-transplant-report.html'}")
    print(f"Latest page: {docs_root / 'sol-benchmark-transplant-report.html'}")
    print(f"Calibration index: {sol_runs_root / 'index.html'}")
    print(f"Legacy archive page: {sol_runs_root / 'archive.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
