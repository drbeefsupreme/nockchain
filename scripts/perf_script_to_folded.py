#!/usr/bin/env python3
"""Convert `perf script` output to folded stacks for flamegraph generation."""

from __future__ import annotations

import argparse
import collections
import re
from pathlib import Path


HEX_RE = re.compile(r"^[0-9a-fA-F]+$")
OFFSET_RE = re.compile(r"\+0x[0-9a-fA-F]+$")


def sanitize(part: str) -> str:
    return (
        part.strip()
        .replace("[unknown]", "anon")
        .replace(";", ":")
        .replace(" ", "_")
        .replace("\t", "_")
        .replace("/", "|")
    )


def parse_header_command(line: str) -> str:
    tok = line.strip().split()
    if not tok:
        return "unknown"
    return sanitize(tok[0])


def parse_frame(line: str) -> str | None:
    s = line.strip()
    if not s or s.startswith("#") or s.startswith("|"):
        return None
    parts = s.split()
    if not parts:
        return None
    addr = None
    if HEX_RE.match(parts[0]):
        addr = parts[0].lower()
    if addr is not None and len(parts) > 1:
        parts = parts[1:]
    if not parts:
        return None
    sym = parts[0]
    sym = OFFSET_RE.sub("", sym)
    mod = ""
    if len(parts) > 1:
        mod = parts[1].strip("()")
    if sym == "[unknown]":
        mod_short = mod.split("/")[-1] if mod else "anon"
        if mod_short in ("[unknown]", "unknown"):
            mod_short = "anon"
        if addr is not None:
            sym = f"0x{addr}@{mod_short}"
        else:
            sym = f"{sym}@{mod_short}"
    return sanitize(sym)


def to_folded(
    lines: list[str], exclude_comm_re: re.Pattern[str] | None
) -> collections.Counter[str]:
    out: collections.Counter[str] = collections.Counter()
    cmd = "unknown"
    frames: list[str] = []
    include = True

    def flush() -> None:
        nonlocal frames
        if not include or not frames:
            return
        # Collapse consecutive identical frames to avoid giant "same address" towers.
        compact: list[str] = []
        for fr in frames:
            if not compact or compact[-1] != fr:
                compact.append(fr)
        stack = ";".join([cmd] + list(reversed(compact)))
        out[stack] += 1
        frames = []

    for raw in lines:
        if not raw.strip():
            flush()
            continue
        if raw[0].isspace():
            frame = parse_frame(raw)
            if frame:
                frames.append(frame)
            continue
        flush()
        cmd = parse_header_command(raw)
        include = True
        if exclude_comm_re and exclude_comm_re.search(cmd):
            include = False

    flush()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument(
        "--exclude-comm-regex",
        default=r"^perf",
        help="Regex for command names to exclude from folded stacks.",
    )
    args = ap.parse_args()

    lines = args.input.read_text(encoding="utf-8", errors="replace").splitlines()
    comm_re = None
    if args.exclude_comm_regex:
        comm_re = re.compile(args.exclude_comm_regex)
    folded = to_folded(lines, comm_re)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for stack, count in folded.most_common():
            f.write(f"{stack} {count}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
