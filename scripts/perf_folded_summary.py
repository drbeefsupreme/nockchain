#!/usr/bin/env python3
"""Summarize folded perf stacks into compact JSON."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def parse_folded(path: Path) -> tuple[list[tuple[list[str], int]], int]:
    rows: list[tuple[list[str], int]] = []
    total = 0
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            stack_txt, count_txt = line.rsplit(" ", 1)
            count = int(count_txt)
        except ValueError:
            continue
        frames = [p for p in stack_txt.split(";") if p]
        if not frames:
            continue
        rows.append((frames, count))
        total += count
    return rows, total


def top_stacks(rows: list[tuple[list[str], int]], total: int, n: int = 12) -> list[dict]:
    rows_sorted = sorted(rows, key=lambda x: x[1], reverse=True)[:n]
    out: list[dict] = []
    for frames, count in rows_sorted:
        pct = (count / total * 100.0) if total else 0.0
        out.append(
            {
                "stack": ";".join(frames),
                "leaf": frames[-1],
                "count": count,
                "pct": round(pct, 4),
            }
        )
    return out


def top_leaves(rows: list[tuple[list[str], int]], total: int, n: int = 16) -> list[dict]:
    agg: Counter[str] = Counter()
    for frames, count in rows:
        agg[frames[-1]] += count
    out: list[dict] = []
    for name, count in agg.most_common(n):
        pct = (count / total * 100.0) if total else 0.0
        out.append({"symbol": name, "count": count, "pct": round(pct, 4)})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--top-stacks", type=int, default=12)
    ap.add_argument("--top-leaves", type=int, default=16)
    args = ap.parse_args()

    rows, total = parse_folded(args.input)
    payload = {
        "sample_count": total,
        "unique_stacks": len(rows),
        "top_stacks": top_stacks(rows, total, args.top_stacks),
        "top_leaves": top_leaves(rows, total, args.top_leaves),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

