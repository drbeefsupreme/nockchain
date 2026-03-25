"""Sync crates/nockchain-bench into an existing PMA checkout."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import TextIO
from typing import Callable


RunCommand = Callable[[list[str], Path], subprocess.CompletedProcess[str]]
BENCH_CRATE_MEMBER = "crates/nockchain-bench"


@dataclass(frozen=True)
class SourceState:
    source_root: Path
    source_crate_dir: Path
    is_dirty: bool


@dataclass(frozen=True)
class TargetState:
    target_dir: Path
    manifest_path: Path
    target_crate_dir: Path


@dataclass(frozen=True)
class TargetOverwritePlan:
    target_crate_exists: bool
    should_delete_target_crate: bool


@dataclass(frozen=True)
class SyncResult:
    source: SourceState
    target: TargetState
    manifest_patch_needed: bool
    overwrite_plan: TargetOverwritePlan
    dry_run: bool
    planned_actions: list[str]
    build_ran: bool
    source_commit: str | None
    stamp_path: Path | None
    binary_path: Path
    manifest_patched: bool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-dir", required=True, type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-build", action="store_true")
    parser.add_argument("--allow-dirty-source", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def default_source_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_source_root(source_root: Path | None) -> Path:
    return (source_root or default_source_root()).resolve()


def run_command(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )


def validate_source(
    source_root: Path,
    *,
    allow_dirty_source: bool,
    run_command: RunCommand = run_command,
) -> SourceState:
    resolved_root = source_root.resolve()
    source_crate_dir = resolved_root / "crates" / "nockchain-bench"
    if not source_crate_dir.is_dir():
        raise ValueError(
            f"Source root {resolved_root} is missing crates/nockchain-bench."
        )

    inside_work_tree = run_command(
        ["git", "rev-parse", "--is-inside-work-tree"],
        resolved_root,
    )
    if inside_work_tree.returncode != 0 or inside_work_tree.stdout.strip() != "true":
        raise ValueError(f"Source root {resolved_root} must be a git checkout.")

    dirty_status = run_command(["git", "status", "--porcelain"], resolved_root)
    if dirty_status.returncode != 0:
        stderr = dirty_status.stderr.strip() or "git status --porcelain failed."
        raise ValueError(
            f"Failed to inspect source checkout {resolved_root}: {stderr}"
        )

    is_dirty = bool(dirty_status.stdout.strip())
    if is_dirty and not allow_dirty_source:
        raise ValueError(
            f"Source checkout {resolved_root} is dirty; rerun with --allow-dirty-source to proceed."
        )

    return SourceState(
        source_root=resolved_root,
        source_crate_dir=source_crate_dir,
        is_dirty=is_dirty,
    )


def validate_target(target_dir: Path) -> TargetState:
    resolved_target = target_dir.resolve()
    if not resolved_target.is_dir():
        raise ValueError(f"Target directory {resolved_target} does not exist.")

    manifest_path = resolved_target / "Cargo.toml"
    if not manifest_path.is_file():
        raise ValueError(f"Target directory {resolved_target} is missing Cargo.toml.")

    return TargetState(
        target_dir=resolved_target,
        manifest_path=manifest_path,
        target_crate_dir=resolved_target / "crates" / "nockchain-bench",
    )


def evaluate_target_crate_overwrite(
    target_crate_dir: Path,
    *,
    force: bool,
) -> TargetOverwritePlan:
    target_crate_exists = target_crate_dir.exists()
    if target_crate_exists and not force:
        raise ValueError(
            f"Target crate {target_crate_dir} already exists; rerun with --force to replace it."
        )
    return TargetOverwritePlan(
        target_crate_exists=target_crate_exists,
        should_delete_target_crate=target_crate_exists and force,
    )


def load_tomlkit():
    try:
        import tomlkit
    except ModuleNotFoundError as exc:
        raise ValueError(
            "tomlkit is required for manifest patching; refresh the uv environment for scripts/bench_sync."
        ) from exc
    return tomlkit


def load_workspace_members(manifest_path: Path):
    try:
        tomlkit = load_tomlkit()
    except ModuleNotFoundError as exc:
        raise ValueError(
            "tomlkit is required for manifest patching; refresh the uv environment for scripts/bench_sync."
        ) from exc
    document = tomlkit.parse(manifest_path.read_text(encoding="utf-8"))
    workspace = document.get("workspace")
    if workspace is None:
        raise ValueError(f"Target manifest {manifest_path} is missing [workspace].")
    members = workspace.get("members")
    if members is None:
        raise ValueError(
            f"Target manifest {manifest_path} is missing workspace.members."
        )
    return tomlkit, document, members


def compute_manifest_patch_need(manifest_path: Path) -> bool:
    _, _, members = load_workspace_members(manifest_path)
    return BENCH_CRATE_MEMBER not in list(members)


def delete_target_crate(target_crate_dir: Path) -> None:
    shutil.rmtree(target_crate_dir)


def copy_source_crate(source_crate_dir: Path, target_crate_dir: Path) -> None:
    shutil.copytree(source_crate_dir, target_crate_dir)


def patch_target_manifest(manifest_path: Path) -> bool:
    tomlkit, document, members = load_workspace_members(manifest_path)
    if BENCH_CRATE_MEMBER in list(members):
        return False
    members.append(BENCH_CRATE_MEMBER)
    manifest_path.write_text(tomlkit.dumps(document), encoding="utf-8")
    return True


def expected_binary_path(target_dir: Path) -> Path:
    return target_dir / "target" / "release" / "nockchain-bench"


def default_stamp_path(target_crate_dir: Path) -> Path:
    return target_crate_dir / ".pma-bench-sync-stamp"


def get_source_commit(
    source_root: Path,
    *,
    run_command: RunCommand = run_command,
) -> str:
    result = run_command(["git", "rev-parse", "HEAD"], source_root)
    if result.returncode != 0:
        stderr = result.stderr.strip() or "git rev-parse HEAD failed."
        raise ValueError(f"Failed to resolve source commit for {source_root}: {stderr}")
    return result.stdout.strip()


def current_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_sync_stamp(
    target_crate_dir: Path,
    *,
    source_root: Path,
    source_commit: str,
    synced_at: str,
) -> Path:
    stamp_path = default_stamp_path(target_crate_dir)
    stamp_text = (
        f"source_root={source_root}\n"
        f"source_commit={source_commit}\n"
        f"synced_at={synced_at}\n"
    )
    stamp_path.write_text(stamp_text, encoding="utf-8")
    return stamp_path


def build_release_binary(
    target_dir: Path,
    *,
    run_command: RunCommand = run_command,
) -> Path:
    result = run_command(
        [
            "cargo",
            "build",
            "-p",
            "nockchain-bench",
            "--release",
            "--features",
            "pma-runtime-compat",
        ],
        target_dir,
    )
    if result.returncode != 0:
        details = "\n".join(
            line
            for line in [result.stdout.strip(), result.stderr.strip()]
            if line
        )
        raise ValueError(f"cargo build failed in {target_dir}.\n{details}".rstrip())
    return expected_binary_path(target_dir)


def planned_actions_for_sync(
    *,
    manifest_patch_needed: bool,
    overwrite_plan: TargetOverwritePlan,
    no_build: bool,
) -> list[str]:
    planned_actions: list[str] = []
    if overwrite_plan.should_delete_target_crate:
        planned_actions.append("delete existing target crate")
    planned_actions.append("copy source crate")
    if manifest_patch_needed:
        planned_actions.append("patch target Cargo.toml")
    planned_actions.append("write sync stamp")
    if not no_build:
        planned_actions.append("build release binary")
    return planned_actions


def execute_sync(
    args: argparse.Namespace,
    *,
    validate_source_fn: Callable[..., SourceState] = validate_source,
    validate_target_fn: Callable[[Path], TargetState] = validate_target,
    compute_manifest_patch_need_fn: Callable[[Path], bool] = compute_manifest_patch_need,
    evaluate_target_crate_overwrite_fn: Callable[..., TargetOverwritePlan] = evaluate_target_crate_overwrite,
    delete_target_crate_fn: Callable[[Path], None] = delete_target_crate,
    copy_source_crate_fn: Callable[[Path, Path], None] = copy_source_crate,
    patch_manifest_fn: Callable[[Path], None] = patch_target_manifest,
    get_source_commit_fn: Callable[..., str] = get_source_commit,
    write_stamp_fn: Callable[..., Path | None] = write_sync_stamp,
    synced_at_fn: Callable[[], str] = current_timestamp,
    build_binary_fn: Callable[..., Path | None] = build_release_binary,
) -> SyncResult:
    source = validate_source_fn(
        resolve_source_root(args.source_root),
        allow_dirty_source=args.allow_dirty_source,
    )
    target = validate_target_fn(args.target_dir)
    manifest_patch_needed = compute_manifest_patch_need_fn(target.manifest_path)
    overwrite_plan = evaluate_target_crate_overwrite_fn(
        target.target_crate_dir,
        force=args.force,
    )
    planned_actions = planned_actions_for_sync(
        manifest_patch_needed=manifest_patch_needed,
        overwrite_plan=overwrite_plan,
        no_build=args.no_build,
    )

    if args.dry_run:
        return SyncResult(
            source=source,
            target=target,
            manifest_patch_needed=manifest_patch_needed,
            overwrite_plan=overwrite_plan,
            dry_run=True,
            planned_actions=planned_actions,
            build_ran=False,
            source_commit=None,
            stamp_path=default_stamp_path(target.target_crate_dir),
            binary_path=expected_binary_path(target.target_dir),
            manifest_patched=False,
        )

    if overwrite_plan.should_delete_target_crate:
        delete_target_crate_fn(target.target_crate_dir)
    copy_source_crate_fn(source.source_crate_dir, target.target_crate_dir)
    manifest_patched = False
    if manifest_patch_needed:
        patch_manifest_fn(target.manifest_path)
        manifest_patched = True
    source_commit = get_source_commit_fn(source.source_root)
    stamp_path = write_stamp_fn(
        target.target_crate_dir,
        source_root=source.source_root,
        source_commit=source_commit,
        synced_at=synced_at_fn(),
    ) or default_stamp_path(target.target_crate_dir)
    build_ran = not args.no_build
    binary_path = expected_binary_path(target.target_dir)
    if build_ran:
        binary_path = build_binary_fn(target.target_dir) or binary_path

    return SyncResult(
        source=source,
        target=target,
        manifest_patch_needed=manifest_patch_needed,
        overwrite_plan=overwrite_plan,
        dry_run=False,
        planned_actions=planned_actions,
        build_ran=build_ran,
        source_commit=source_commit,
        stamp_path=stamp_path,
        binary_path=binary_path,
        manifest_patched=manifest_patched,
    )


def render_success_summary(result: SyncResult) -> str:
    manifest_patched = "yes" if result.manifest_patched else "no"
    build_ran = "yes" if result.build_ran else "no"
    lines = [
        f"Source root: {result.source.source_root}",
        f"Target dir: {result.target.target_dir}",
        f"Source commit: {result.source_commit}",
        f"Manifest patched: {manifest_patched}",
        f"Stamp path: {result.stamp_path}",
        f"Build ran: {build_ran}",
        f"Binary path: {result.binary_path}",
        "",
        "Placeholder quick-bench command:",
        f"{result.binary_path} sol quick-bench \\",
        "  --fixture /path/to/your/fixture.soltest \\",
        "  --blocks 10 \\",
        "  --checkpoint-every-blocks 0",
    ]
    return "\n".join(lines)


def render_dry_run_summary(result: SyncResult) -> str:
    lines = [
        f"Dry run for target dir: {result.target.target_dir}",
        f"Source root: {result.source.source_root}",
        "Planned actions:",
    ]
    lines.extend(f"- {action}" for action in result.planned_actions)
    lines.append(f"Expected binary path: {result.binary_path}")
    return "\n".join(lines)


def main(
    argv: list[str] | None = None,
    *,
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
    validate_source_fn: Callable[..., SourceState] = validate_source,
    validate_target_fn: Callable[[Path], TargetState] = validate_target,
    compute_manifest_patch_need_fn: Callable[[Path], bool] = compute_manifest_patch_need,
    evaluate_target_crate_overwrite_fn: Callable[..., TargetOverwritePlan] = evaluate_target_crate_overwrite,
    delete_target_crate_fn: Callable[[Path], None] = delete_target_crate,
    copy_source_crate_fn: Callable[[Path, Path], None] = copy_source_crate,
    patch_manifest_fn: Callable[[Path], None] = patch_target_manifest,
    get_source_commit_fn: Callable[..., str] = get_source_commit,
    write_stamp_fn: Callable[..., Path | None] = write_sync_stamp,
    synced_at_fn: Callable[[], str] = current_timestamp,
    build_binary_fn: Callable[..., Path | None] = build_release_binary,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = execute_sync(
            args,
            validate_source_fn=validate_source_fn,
            validate_target_fn=validate_target_fn,
            compute_manifest_patch_need_fn=compute_manifest_patch_need_fn,
            evaluate_target_crate_overwrite_fn=evaluate_target_crate_overwrite_fn,
            delete_target_crate_fn=delete_target_crate_fn,
            copy_source_crate_fn=copy_source_crate_fn,
            patch_manifest_fn=patch_manifest_fn,
            get_source_commit_fn=get_source_commit_fn,
            write_stamp_fn=write_stamp_fn,
            synced_at_fn=synced_at_fn,
            build_binary_fn=build_binary_fn,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=stderr)
        return 1

    if result.dry_run:
        print(render_dry_run_summary(result), file=stdout)
    else:
        print(render_success_summary(result), file=stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
