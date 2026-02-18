#!/usr/bin/env python3
"""Render folded stacks into a static flamegraph SVG."""

from __future__ import annotations

import argparse
import hashlib
import html
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Node:
    name: str
    count: int = 0
    children: dict[str, "Node"] = field(default_factory=dict)
    x: float = 0.0
    w: float = 0.0
    depth: int = 0
    path: str = ""


def parse_folded(path: Path) -> Node:
    root = Node("root")
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
        root.count += count
        cur = root
        for frame in frames:
            child = cur.children.get(frame)
            if child is None:
                child = Node(frame)
                cur.children[frame] = child
            child.count += count
            cur = child
    return root


def max_depth(node: Node, depth: int = 0) -> int:
    if not node.children:
        return depth
    return max(max_depth(ch, depth + 1) for ch in node.children.values())


def layout(node: Node, x: float, width: float, depth: int, path: str) -> None:
    node.x = x
    node.w = width
    node.depth = depth
    node.path = path
    if not node.children or node.count <= 0:
        return
    children = sorted(
        node.children.values(), key=lambda n: (-n.count, n.name.lower())
    )
    cx = x
    for ch in children:
        cw = width * (ch.count / node.count)
        layout(ch, cx, cw, depth + 1, f"{path};{ch.name}" if path else ch.name)
        cx += cw


def color_for(name: str) -> str:
    h = hashlib.sha1(name.encode("utf-8")).digest()
    r = 200 + (h[0] % 40)
    g = 80 + (h[1] % 130)
    b = 60 + (h[2] % 80)
    return f"rgb({r},{g},{b})"


def render_svg(root: Node, title: str, width: int, frame_h: int) -> str:
    depth = max_depth(root)
    pad_x = 12
    pad_top = 38
    pad_bottom = 24
    graph_w = max(100, width - (2 * pad_x))
    graph_h = max(80, (depth + 1) * frame_h)
    total_h = pad_top + graph_h + pad_bottom
    px_per_char = 7.0
    min_label_w = 28.0

    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{total_h}" viewBox="0 0 {width} {total_h}" role="img" aria-label="{html.escape(title)}">'
    )
    parts.append(
        '<style>'
        "text{font-family:'IBM Plex Sans','Segoe UI',sans-serif;font-size:12px;fill:#1d2430}"
        ".sub{font-size:11px;fill:#5a6677}"
        ".frame:hover{stroke:#111;stroke-width:0.7px}"
        "</style>"
    )
    parts.append(
        f'<rect x="0" y="0" width="{width}" height="{total_h}" fill="#f8fafc" />'
    )
    parts.append(
        f'<text x="{pad_x}" y="20" font-weight="700">{html.escape(title)}</text>'
    )
    parts.append(
        f'<text class="sub" x="{pad_x}" y="33">Samples: {root.count:,} | Depth: {depth}</text>'
    )

    def walk(node: Node) -> None:
        for ch in sorted(node.children.values(), key=lambda n: (-n.count, n.name.lower())):
            y = pad_top + (depth - ch.depth) * frame_h
            x = pad_x + ch.x
            w = max(0.2, ch.w)
            h = frame_h - 1
            fill = color_for(ch.name)
            pct = (ch.count / root.count * 100.0) if root.count else 0.0
            parts.append(
                f'<g class="frame"><title>{html.escape(ch.path)} ({ch.count:,} samples, {pct:.3f}%)</title>'
                f'<rect x="{x:.3f}" y="{y:.3f}" width="{w:.3f}" height="{h}" fill="{fill}" rx="1" ry="1"/></g>'
            )
            if w >= min_label_w:
                max_chars = int((w - 6) / px_per_char)
                if max_chars > 2:
                    label = ch.name if len(ch.name) <= max_chars else f"{ch.name[:max_chars-1]}…"
                    parts.append(
                        f'<text x="{x + 3:.3f}" y="{y + frame_h - 4:.3f}">{html.escape(label)}</text>'
                    )
            walk(ch)

    # Skip rendering synthetic root frame.
    walk(root)
    parts.append("</svg>")
    return "\n".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--title", required=True)
    ap.add_argument("--width", type=int, default=1800)
    ap.add_argument("--frame-height", type=int, default=18)
    args = ap.parse_args()

    root = parse_folded(args.input)
    layout(root, 0.0, float(max(100, args.width - 24)), 0, "")
    svg = render_svg(root, args.title, args.width, args.frame_height)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(svg, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

