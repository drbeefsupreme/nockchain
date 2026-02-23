#!/usr/bin/env python3
"""Publish a SOL matrix run (with perf + Tracy artifacts) into the GH Pages archive."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
import shutil
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
    .th-sort {{ cursor:pointer; user-select:none; }}
    .th-sort::after {{ content:" ↕"; color:#7b8798; font-size:.74rem; }}
    .th-sort.active.asc::after {{ content:" ↑"; color:var(--accent2); }}
    .th-sort.active.desc::after {{ content:" ↓"; color:var(--accent2); }}
    object {{ width:100%; height:620px; border:1px solid var(--line); border-radius:8px; background:#fff; }}
    code {{ background:#f1f4f8; border:1px solid #dde3eb; border-radius:5px; padding:1px 5px; }}
    @media (max-width: 980px) {{ .grid4 {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} .controls {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} object {{ height:480px; }} }}
    @media (max-width: 640px) {{ .grid4 {{ grid-template-columns:1fr; }} .controls {{ grid-template-columns:1fr; }} object {{ height:360px; }} }}
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

    <section class="card links">
      <h2>Artifacts</h2>
      <a href="{base_prefix}combined_summary.tsv" target="_blank">combined_summary.tsv</a>
      <a href="{base_prefix}sol-benchmark-transplant-report.md" target="_blank">report.md</a>
      <a href="{base_prefix}sol-benchmark-transplant-memory-profiles.json" target="_blank">memory-profiles.json</a>
    </section>
  </main>
  <script>
    const SCORE_ROWS = {score_rows_json};
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
    lines.append("## Files")
    lines.append("")
    lines.append("- `combined_summary.tsv`")
    lines.append("- `sol-benchmark-transplant-report.html`")
    lines.append("- `sol-benchmark-transplant-report.md`")
    lines.append("- `sol-benchmark-transplant-memory-profiles.json`")
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

    run_dst = runs_root / run_id
    if run_dst.exists():
        shutil.rmtree(run_dst)
    run_dst.mkdir(parents=True, exist_ok=True)

    shutil.copy2(combined, run_dst / "combined_summary.tsv")

    trace_records = build_trace_records(rows, run_dst)
    guard_records = build_guard_records(run_root, run_dst)

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
        run_id, generated_on, pass_no, trace_records, guard_records, summary, args.fixture_label
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

    manifest = {"updated_on": generated_on, "latest_run_id": run_id, "runs": []}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    runs = manifest.get("runs", [])
    runs = [r for r in runs if r.get("run_id") != run_id]
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
