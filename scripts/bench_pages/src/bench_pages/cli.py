from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

from bench_pages.gh_pages import (
    bootstrap_pages_checkout,
    commit_pages_changes,
    prepare_index_entries,
    publish_sweep_to_pages,
)
from bench_pages.ghcr import publish_docker_images
from bench_pages.loader import load_sweep
from bench_pages.manifest import build_manifest
from bench_pages.render import assets_dir, render_index_page, render_sweep_page


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    sweep_root = Path(args.sweep_root).resolve()
    sweep = load_sweep(sweep_root)

    docker_images = []
    if _should_plan_ghcr_publish(args, sweep.execution_mode):
        docker_images = publish_docker_images(
            sweep=sweep,
            owner=args.owner,
            ghcr_package=args.ghcr_package,
            publish=_should_push_outputs(args),
        )

    manifest = build_manifest(sweep, docker_images=docker_images)
    sweep_html = render_sweep_page(manifest)

    if args.output_dir is not None:
        pages_root = Path(args.output_dir).resolve()
        pages_root.mkdir(parents=True, exist_ok=True)
        _publish_site_tree(pages_root, sweep_root, manifest, sweep_html, replace=args.replace)
        _print_summary(manifest, pages_root, docker_images, dry_run=True)
        return 0

    if args.dry_run:
        with tempfile.TemporaryDirectory(prefix="bench-pages-dry-run-") as temp_dir:
            pages_root = Path(temp_dir)
            _publish_site_tree(pages_root, sweep_root, manifest, sweep_html, replace=args.replace)
            _print_summary(manifest, pages_root, docker_images, dry_run=True)
        return 0

    repo_root = Path.cwd()
    tmp_root = repo_root / ".tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    pages_root = Path(tempfile.mkdtemp(prefix="bench-pages-", dir=tmp_root))
    try:
        bootstrap_pages_checkout(
            repo_root=repo_root,
            pages_root=pages_root,
            branch=args.pages_branch,
        )
        _publish_site_tree(pages_root, sweep_root, manifest, sweep_html, replace=args.replace)
        commit_pages_changes(
            pages_root=pages_root,
            message=f"Publish sweep {manifest['sweep']['id']}",
            branch=args.pages_branch,
            push=args.push,
        )
        _print_summary(manifest, pages_root, docker_images, dry_run=not args.push)
    finally:
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(pages_root)],
            check=False,
            capture_output=True,
            text=True,
        )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Publish SOL sweep reports to GitHub Pages and GHCR.")
    parser.add_argument("--sweep-root", required=True)
    parser.add_argument("--pages-branch", default="gh-pages")
    parser.add_argument("--owner", default="drbeefsupreme")
    parser.add_argument("--repo", default="nockchain")
    parser.add_argument("--ghcr-package", default="nockchain-bench")
    parser.add_argument("--push", action="store_true", help="Push gh-pages and GHCR updates.")
    parser.add_argument("--replace", action="store_true", help="Replace an existing published sweep id in place.")
    parser.add_argument("--dry-run", action="store_true", help="Materialize output without pushing changes.")
    parser.add_argument("--output-dir", help="Write a local site tree here instead of touching gh-pages.")
    parser.set_defaults(publish_ghcr=True)
    parser.add_argument("--publish-ghcr", dest="publish_ghcr", action="store_true")
    parser.add_argument("--no-publish-ghcr", dest="publish_ghcr", action="store_false")
    return parser


def _publish_site_tree(
    pages_root: Path,
    sweep_root: Path,
    manifest: dict,
    sweep_html: str,
    replace: bool,
) -> None:
    entries = prepare_index_entries(pages_root, manifest, replace=replace)
    publish_sweep_to_pages(
        pages_root=pages_root,
        sweep_root=sweep_root,
        manifest=manifest,
        sweep_html=sweep_html,
        index_html=render_index_page(entries),
        assets_dir=assets_dir(),
        replace=replace,
        entries=entries,
    )


def _should_plan_ghcr_publish(args: argparse.Namespace, execution_mode: str) -> bool:
    return execution_mode == "docker" and args.publish_ghcr


def _should_push_outputs(args: argparse.Namespace) -> bool:
    return args.push and not args.dry_run and args.output_dir is None


def _print_summary(
    manifest: dict,
    output_root: Path,
    docker_images: list,
    dry_run: bool,
) -> None:
    print(f"Sweep ID: {manifest['sweep']['id']}")
    print(f"Pages output: {output_root}")
    print(f"Mode: {manifest['sweep']['execution_mode']}")
    print(f"Action: {'dry-run' if dry_run else 'published'}")
    if docker_images:
        print("Docker image plan:")
        for record in docker_images:
            print(
                f"  - {record.local_image_ref or 'n/a'} -> ghcr.io/...:{record.ghcr_tag} "
                f"({record.publish_status or 'planned'})"
            )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
