from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from bench_pages.errors import ExternalCommandError, ValidationError
from bench_pages.file_ops import copy_directory_contents


Runner = Callable[[list[str]], subprocess.CompletedProcess[str]]


def bootstrap_pages_checkout(
    repo_root: Path,
    pages_root: Path,
    branch: str = "gh-pages",
    runner: Runner | None = None,
) -> Path:
    run = runner or _run_command

    branch_exists = run(
        ["git", "-C", str(repo_root), "show-ref", "--verify", f"refs/heads/{branch}"]
    ).returncode == 0
    _run_checked(run, ["git", "-C", str(repo_root), "worktree", "add", "--detach", str(pages_root)])
    if branch_exists:
        _run_checked(run, ["git", "-C", str(pages_root), "checkout", branch])
        _validate_existing_pages_layout(pages_root)
    else:
        _run_checked(run, ["git", "-C", str(pages_root), "checkout", "--orphan", branch])

    ensure_pages_layout(pages_root)
    return pages_root


def ensure_pages_layout(pages_root: Path) -> None:
    pages_root.mkdir(parents=True, exist_ok=True)
    (pages_root / ".nojekyll").write_text("")
    if not (pages_root / "index.json").exists():
        (pages_root / "index.json").write_text("[]\n")
    if not (pages_root / "index.html").exists():
        (pages_root / "index.html").write_text("<!doctype html><title>Bench Pages</title>\n")


def publish_sweep_to_pages(
    pages_root: Path,
    sweep_root: Path,
    manifest: dict[str, Any],
    sweep_html: str,
    index_html: str,
    assets_dir: Path,
    replace: bool = False,
    entries: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    ensure_pages_layout(pages_root)
    copy_directory_contents(assets_dir, pages_root / "assets")

    sweep_id = manifest["sweep"]["id"]
    sweep_dir = pages_root / "sweeps" / sweep_id
    if replace and sweep_dir.exists():
        shutil.rmtree(sweep_dir)
    (sweep_dir / "artifacts").mkdir(parents=True, exist_ok=True)

    shutil.copytree(sweep_root, sweep_dir / "artifacts", dirs_exist_ok=True)
    _write_json(sweep_dir / "manifest.json", manifest)
    (sweep_dir / "index.html").write_text(sweep_html)

    if entries is None:
        entries = prepare_index_entries(pages_root, manifest, replace=replace)
    _write_json(pages_root / "index.json", entries)
    (pages_root / "index.html").write_text(index_html)
    return entries


def prepare_index_entries(
    pages_root: Path,
    manifest: dict[str, Any],
    replace: bool = False,
) -> list[dict[str, Any]]:
    entries = _load_index_entries(pages_root / "index.json")
    return _upsert_index_entry(entries, _index_entry_from_manifest(manifest), replace=replace)


def commit_pages_changes(
    pages_root: Path,
    message: str,
    branch: str = "gh-pages",
    push: bool = False,
    runner: Runner | None = None,
) -> None:
    run = runner or _run_command
    _run_checked(run, ["git", "-C", str(pages_root), "add", "."])
    _run_checked(run, ["git", "-C", str(pages_root), "commit", "-m", message])
    if push:
        result = run(["git", "-C", str(pages_root), "push", "origin", branch])
        if result.returncode != 0:
            raise ExternalCommandError(
                f"git push failed for {branch}; concurrent publish may require retry\n{result.stderr}"
            )


def _validate_existing_pages_layout(pages_root: Path) -> None:
    index_path = pages_root / "index.json"
    if not index_path.exists():
        raise ValidationError(
            "existing gh-pages branch does not contain the publisher index.json layout; "
            "delete or replace the legacy branch before publishing"
        )


def _load_index_entries(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return json.loads(path.read_text())


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _upsert_index_entry(
    entries: list[dict[str, Any]],
    new_entry: dict[str, Any],
    replace: bool,
) -> list[dict[str, Any]]:
    updated: list[dict[str, Any]] = []
    inserted = False
    for entry in entries:
        if entry.get("id") == new_entry["id"]:
            if replace or not inserted:
                updated.append(new_entry)
                inserted = True
            continue
        updated.append(entry)
    if not inserted:
        updated.append(new_entry)
    return sorted(updated, key=lambda entry: entry["id"])


def _index_entry_from_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    sweep = manifest["sweep"]
    verdict = sweep.get("verdict", {})
    validity = verdict.get("validity")
    if isinstance(validity, dict):
        validity_value = next(iter(validity.keys()), "unknown")
    else:
        validity_value = validity or "unknown"
    return {
        "id": sweep["id"],
        "path": f"sweeps/{sweep['id']}/index.html",
        "execution_mode": sweep.get("execution_mode"),
        "fixture_identity": sweep.get("fixture_identity"),
        "git_commit": sweep.get("git_commit"),
        "build_profile": sweep.get("build_profile"),
        "axis_names": sweep.get("axis_names", []),
        "verdict": validity_value,
    }


def _run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=False, capture_output=True, text=True)


def _run_checked(
    runner: Runner,
    command: list[str],
) -> subprocess.CompletedProcess[str]:
    result = runner(command)
    if result.returncode != 0:
        raise ExternalCommandError(
            f"command failed ({result.returncode}): {' '.join(command)}\n{result.stderr}"
        )
    return result
