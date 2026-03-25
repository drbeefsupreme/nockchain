from __future__ import annotations

import argparse
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import tomlkit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pma_bench_sync


class BuildParserTests(unittest.TestCase):
    def test_target_dir_is_required(self) -> None:
        parser = pma_bench_sync.build_parser()

        with self.assertRaises(SystemExit):
            parser.parse_args([])

    def test_source_root_defaults_to_repo_containing_script(self) -> None:
        parser = pma_bench_sync.build_parser()

        args = parser.parse_args(["--target-dir", "/tmp/target"])

        self.assertEqual(args.source_root, None)
        self.assertEqual(
            pma_bench_sync.default_source_root(),
            Path(pma_bench_sync.__file__).resolve().parents[2],
        )

    def test_force_defaults_to_false(self) -> None:
        parser = pma_bench_sync.build_parser()

        args = parser.parse_args(["--target-dir", "/tmp/target"])

        self.assertFalse(args.force)

    def test_no_build_defaults_to_false(self) -> None:
        parser = pma_bench_sync.build_parser()

        args = parser.parse_args(["--target-dir", "/tmp/target"])

        self.assertFalse(args.no_build)

    def test_allow_dirty_source_defaults_to_false(self) -> None:
        parser = pma_bench_sync.build_parser()

        args = parser.parse_args(["--target-dir", "/tmp/target"])

        self.assertFalse(args.allow_dirty_source)

    def test_dry_run_defaults_to_false(self) -> None:
        parser = pma_bench_sync.build_parser()

        args = parser.parse_args(["--target-dir", "/tmp/target"])

        self.assertFalse(args.dry_run)


class ValidateSourceTests(unittest.TestCase):
    def test_validate_source_requires_bench_crate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            source_root = Path(tmp_dir)

            with self.assertRaisesRegex(ValueError, "crates/nockchain-bench"):
                pma_bench_sync.validate_source(source_root, allow_dirty_source=False)

    def test_validate_source_rejects_non_git_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            source_root = Path(tmp_dir)
            (source_root / "crates" / "nockchain-bench").mkdir(parents=True)

            run = mock.Mock(
                side_effect=[
                    mock.Mock(returncode=1, stdout="", stderr="not a git repo"),
                ]
            )

            with self.assertRaisesRegex(ValueError, "git checkout"):
                pma_bench_sync.validate_source(
                    source_root,
                    allow_dirty_source=False,
                    run_command=run,
                )

    def test_validate_source_rejects_dirty_checkout_without_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            source_root = Path(tmp_dir)
            (source_root / "crates" / "nockchain-bench").mkdir(parents=True)

            run = mock.Mock(
                side_effect=[
                    mock.Mock(returncode=0, stdout="true\n", stderr=""),
                    mock.Mock(returncode=0, stdout=" M crates/nockchain-bench/Cargo.toml\n", stderr=""),
                ]
            )

            with self.assertRaisesRegex(ValueError, "dirty"):
                pma_bench_sync.validate_source(
                    source_root,
                    allow_dirty_source=False,
                    run_command=run,
                )

    def test_validate_source_allows_dirty_with_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            source_root = Path(tmp_dir)
            (source_root / "crates" / "nockchain-bench").mkdir(parents=True)

            run = mock.Mock(
                side_effect=[
                    mock.Mock(returncode=0, stdout="true\n", stderr=""),
                    mock.Mock(returncode=0, stdout=" M crates/nockchain-bench/Cargo.toml\n", stderr=""),
                ]
            )

            source = pma_bench_sync.validate_source(
                source_root,
                allow_dirty_source=True,
                run_command=run,
            )

            self.assertEqual(source.source_root, source_root)
            self.assertEqual(source.source_crate_dir, source_root / "crates" / "nockchain-bench")
            self.assertTrue(source.is_dirty)


class ValidateTargetTests(unittest.TestCase):
    def test_validate_target_requires_root_cargo_toml(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            target_dir = Path(tmp_dir)

            with self.assertRaisesRegex(ValueError, "Cargo.toml"):
                pma_bench_sync.validate_target(target_dir)


class SyncOrchestrationTests(unittest.TestCase):
    def make_args(self, **overrides: object) -> argparse.Namespace:
        values: dict[str, object] = {
            "target_dir": Path("/tmp/target"),
            "source_root": None,
            "force": False,
            "no_build": False,
            "allow_dirty_source": False,
            "dry_run": False,
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    def make_source_state(self) -> pma_bench_sync.SourceState:
        return pma_bench_sync.SourceState(
            source_root=Path("/tmp/source"),
            source_crate_dir=Path("/tmp/source/crates/nockchain-bench"),
            is_dirty=False,
        )

    def make_target_state(self) -> pma_bench_sync.TargetState:
        return pma_bench_sync.TargetState(
            target_dir=Path("/tmp/target"),
            manifest_path=Path("/tmp/target/Cargo.toml"),
            target_crate_dir=Path("/tmp/target/crates/nockchain-bench"),
        )

    def test_existing_target_crate_fails_without_force(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            target_crate_dir = Path(tmp_dir) / "crates" / "nockchain-bench"
            target_crate_dir.mkdir(parents=True)

            with self.assertRaisesRegex(ValueError, "--force"):
                pma_bench_sync.evaluate_target_crate_overwrite(
                    target_crate_dir,
                    force=False,
                )

    def test_existing_target_crate_is_deleted_with_force(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            target_crate_dir = Path(tmp_dir) / "crates" / "nockchain-bench"
            target_crate_dir.mkdir(parents=True)

            overwrite = pma_bench_sync.evaluate_target_crate_overwrite(
                target_crate_dir,
                force=True,
            )

            self.assertTrue(overwrite.target_crate_exists)
            self.assertTrue(overwrite.should_delete_target_crate)

    def test_dry_run_performs_prechecks_and_reports_actions_without_mutation(self) -> None:
        events: list[str] = []
        source = self.make_source_state()
        target = self.make_target_state()
        delete_target_crate = mock.Mock(side_effect=lambda _: events.append("delete"))
        copy_source_crate = mock.Mock(side_effect=lambda *_: events.append("copy"))
        patch_manifest = mock.Mock(side_effect=lambda *_: events.append("patch"))
        write_stamp = mock.Mock(side_effect=lambda *_: events.append("stamp"))
        build_binary = mock.Mock(side_effect=lambda *_: events.append("build"))

        result = pma_bench_sync.execute_sync(
            self.make_args(force=True, dry_run=True),
            validate_source_fn=lambda *_args, **_kwargs: events.append("validate_source") or source,
            validate_target_fn=lambda *_args, **_kwargs: events.append("validate_target") or target,
            compute_manifest_patch_need_fn=lambda *_args, **_kwargs: events.append("patch_need") or True,
            evaluate_target_crate_overwrite_fn=lambda *_args, **_kwargs: events.append("overwrite_need")
            or pma_bench_sync.TargetOverwritePlan(True, True),
            delete_target_crate_fn=delete_target_crate,
            copy_source_crate_fn=copy_source_crate,
            patch_manifest_fn=patch_manifest,
            write_stamp_fn=write_stamp,
            build_binary_fn=build_binary,
        )

        self.assertEqual(
            events,
            ["validate_source", "validate_target", "patch_need", "overwrite_need"],
        )
        self.assertTrue(result.dry_run)
        self.assertIn("delete existing target crate", result.planned_actions)
        self.assertIn("copy source crate", result.planned_actions)
        self.assertIn("patch target Cargo.toml", result.planned_actions)
        self.assertIn("write sync stamp", result.planned_actions)
        self.assertIn("build release binary", result.planned_actions)
        delete_target_crate.assert_not_called()
        copy_source_crate.assert_not_called()
        patch_manifest.assert_not_called()
        write_stamp.assert_not_called()
        build_binary.assert_not_called()

    def test_execute_sync_waits_until_prechecks_complete_before_mutation(self) -> None:
        events: list[str] = []
        source = self.make_source_state()
        target = self.make_target_state()

        result = pma_bench_sync.execute_sync(
            self.make_args(force=True),
            validate_source_fn=lambda *_args, **_kwargs: events.append("validate_source") or source,
            validate_target_fn=lambda *_args, **_kwargs: events.append("validate_target") or target,
            compute_manifest_patch_need_fn=lambda *_args, **_kwargs: events.append("patch_need") or True,
            evaluate_target_crate_overwrite_fn=lambda *_args, **_kwargs: events.append("overwrite_need")
            or pma_bench_sync.TargetOverwritePlan(True, True),
            delete_target_crate_fn=lambda *_args, **_kwargs: events.append("delete"),
            copy_source_crate_fn=lambda *_args, **_kwargs: events.append("copy"),
            patch_manifest_fn=lambda *_args, **_kwargs: events.append("patch"),
            get_source_commit_fn=lambda *_args, **_kwargs: events.append("commit") or "abc123",
            write_stamp_fn=lambda *_args, **_kwargs: events.append("stamp"),
            synced_at_fn=lambda: "2026-03-25T18:22:00+00:00",
            build_binary_fn=lambda *_args, **_kwargs: events.append("build"),
        )

        self.assertFalse(result.dry_run)
        self.assertEqual(
            events,
            [
                "validate_source",
                "validate_target",
                "patch_need",
                "overwrite_need",
                "delete",
                "copy",
                "patch",
                "commit",
                "stamp",
                "build",
            ],
        )


class ManifestPatchTests(unittest.TestCase):
    def test_patch_target_manifest_is_noop_when_member_already_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest_path = Path(tmp_dir) / "Cargo.toml"
            manifest_path.write_text(
                '[workspace]\nmembers = [\n  "crates/foo",\n  "crates/nockchain-bench",\n]\n',
                encoding="utf-8",
            )

            self.assertFalse(pma_bench_sync.compute_manifest_patch_need(manifest_path))
            self.assertFalse(pma_bench_sync.patch_target_manifest(manifest_path))

    def test_patch_target_manifest_adds_missing_member(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest_path = Path(tmp_dir) / "Cargo.toml"
            manifest_path.write_text(
                '[workspace]\nmembers = [\n  "crates/foo",\n  "crates/bar",\n]\n',
                encoding="utf-8",
            )

            self.assertTrue(pma_bench_sync.compute_manifest_patch_need(manifest_path))
            self.assertTrue(pma_bench_sync.patch_target_manifest(manifest_path))

            members = tomlkit.parse(manifest_path.read_text(encoding="utf-8"))["workspace"]["members"]
            self.assertEqual(
                list(members),
                ["crates/foo", "crates/bar", "crates/nockchain-bench"],
            )

    def test_manifest_patch_requires_tomlkit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest_path = Path(tmp_dir) / "Cargo.toml"
            manifest_path.write_text('[workspace]\nmembers = ["crates/foo"]\n', encoding="utf-8")

            with mock.patch.object(
                pma_bench_sync,
                "load_tomlkit",
                side_effect=ModuleNotFoundError("tomlkit"),
            ):
                with self.assertRaisesRegex(ValueError, "uv"):
                    pma_bench_sync.compute_manifest_patch_need(manifest_path)


class OutputAndBuildTests(unittest.TestCase):
    def make_source_state(self, source_root: Path | None = None) -> pma_bench_sync.SourceState:
        base = source_root or Path("/tmp/source")
        return pma_bench_sync.SourceState(
            source_root=base,
            source_crate_dir=base / "crates" / "nockchain-bench",
            is_dirty=False,
        )

    def make_target_state(self, target_dir: Path | None = None) -> pma_bench_sync.TargetState:
        base = target_dir or Path("/tmp/target")
        return pma_bench_sync.TargetState(
            target_dir=base,
            manifest_path=base / "Cargo.toml",
            target_crate_dir=base / "crates" / "nockchain-bench",
        )

    def test_get_source_commit_reads_head_commit(self) -> None:
        run = mock.Mock(return_value=mock.Mock(returncode=0, stdout="abc123\n", stderr=""))

        self.assertEqual(
            pma_bench_sync.get_source_commit(Path("/tmp/source"), run_command=run),
            "abc123",
        )

    def test_write_sync_stamp_writes_expected_format(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            target_crate_dir = Path(tmp_dir) / "crates" / "nockchain-bench"
            target_crate_dir.mkdir(parents=True)

            stamp_path = pma_bench_sync.write_sync_stamp(
                target_crate_dir,
                source_root=Path("/tmp/source"),
                source_commit="abc123",
                synced_at="2026-03-25T18:22:00+00:00",
            )

            self.assertEqual(stamp_path, target_crate_dir / ".pma-bench-sync-stamp")
            self.assertEqual(
                stamp_path.read_text(encoding="utf-8"),
                "source_root=/tmp/source\n"
                "source_commit=abc123\n"
                "synced_at=2026-03-25T18:22:00+00:00\n",
            )

    def test_build_release_binary_runs_expected_command(self) -> None:
        run = mock.Mock(return_value=mock.Mock(returncode=0, stdout="", stderr=""))
        target_dir = Path("/tmp/target")

        binary_path = pma_bench_sync.build_release_binary(target_dir, run_command=run)

        self.assertEqual(
            binary_path,
            target_dir / "target" / "release" / "nockchain-bench",
        )
        run.assert_called_once_with(
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

    def test_execute_sync_skips_build_when_no_build_is_set(self) -> None:
        build_binary = mock.Mock()

        result = pma_bench_sync.execute_sync(
            argparse.Namespace(
                target_dir=Path("/tmp/target"),
                source_root=None,
                force=False,
                no_build=True,
                allow_dirty_source=False,
                dry_run=False,
            ),
            validate_source_fn=lambda *_args, **_kwargs: self.make_source_state(),
            validate_target_fn=lambda *_args, **_kwargs: self.make_target_state(),
            compute_manifest_patch_need_fn=lambda *_args, **_kwargs: False,
            evaluate_target_crate_overwrite_fn=lambda *_args, **_kwargs: pma_bench_sync.TargetOverwritePlan(
                False,
                False,
            ),
            delete_target_crate_fn=lambda *_args, **_kwargs: None,
            copy_source_crate_fn=lambda *_args, **_kwargs: None,
            patch_manifest_fn=lambda *_args, **_kwargs: None,
            get_source_commit_fn=lambda *_args, **_kwargs: "abc123",
            write_stamp_fn=lambda *_args, **_kwargs: Path("/tmp/target/crates/nockchain-bench/.pma-bench-sync-stamp"),
            synced_at_fn=lambda: "2026-03-25T18:22:00+00:00",
            build_binary_fn=build_binary,
        )

        self.assertFalse(result.build_ran)
        build_binary.assert_not_called()

    def test_build_release_binary_surfaces_failure(self) -> None:
        run = mock.Mock(
            return_value=mock.Mock(
                returncode=1,
                stdout="building...\n",
                stderr="linker error\n",
            )
        )

        with self.assertRaisesRegex(ValueError, "cargo build"):
            pma_bench_sync.build_release_binary(Path("/tmp/target"), run_command=run)

    def test_render_success_summary_includes_operator_fields(self) -> None:
        target = self.make_target_state()
        result = pma_bench_sync.SyncResult(
            source=self.make_source_state(),
            target=target,
            manifest_patch_needed=True,
            overwrite_plan=pma_bench_sync.TargetOverwritePlan(True, True),
            dry_run=False,
            planned_actions=[],
            build_ran=True,
            source_commit="abc123",
            stamp_path=target.target_crate_dir / ".pma-bench-sync-stamp",
            binary_path=target.target_dir / "target" / "release" / "nockchain-bench",
            manifest_patched=True,
        )

        summary = pma_bench_sync.render_success_summary(result)

        self.assertIn("Source root: /tmp/source", summary)
        self.assertIn("Target dir: /tmp/target", summary)
        self.assertIn("Source commit: abc123", summary)
        self.assertIn("Manifest patched: yes", summary)
        self.assertIn("Stamp path: /tmp/target/crates/nockchain-bench/.pma-bench-sync-stamp", summary)
        self.assertIn("Build ran: yes", summary)
        self.assertIn("Binary path: /tmp/target/target/release/nockchain-bench", summary)
        self.assertIn("/path/to/your/fixture.soltest", summary)
        self.assertIn("sol quick-bench", summary)

    def test_main_returns_nonzero_on_manifest_patch_failure(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()

        exit_code = pma_bench_sync.main(
            ["--target-dir", "/tmp/target"],
            stdout=stdout,
            stderr=stderr,
            validate_source_fn=lambda *_args, **_kwargs: self.make_source_state(),
            validate_target_fn=lambda *_args, **_kwargs: self.make_target_state(),
            compute_manifest_patch_need_fn=lambda *_args, **_kwargs: True,
            evaluate_target_crate_overwrite_fn=lambda *_args, **_kwargs: pma_bench_sync.TargetOverwritePlan(
                False,
                False,
            ),
            delete_target_crate_fn=lambda *_args, **_kwargs: None,
            copy_source_crate_fn=lambda *_args, **_kwargs: None,
            patch_manifest_fn=mock.Mock(side_effect=ValueError("manifest patch failure")),
            get_source_commit_fn=lambda *_args, **_kwargs: "abc123",
            write_stamp_fn=lambda *_args, **_kwargs: Path("/tmp/stamp"),
            synced_at_fn=lambda: "2026-03-25T18:22:00+00:00",
            build_binary_fn=lambda *_args, **_kwargs: Path("/tmp/binary"),
        )

        self.assertEqual(exit_code, 1)
        self.assertIn("manifest patch failure", stderr.getvalue())

    def test_main_returns_nonzero_on_stamp_write_failure(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()

        exit_code = pma_bench_sync.main(
            ["--target-dir", "/tmp/target"],
            stdout=stdout,
            stderr=stderr,
            validate_source_fn=lambda *_args, **_kwargs: self.make_source_state(),
            validate_target_fn=lambda *_args, **_kwargs: self.make_target_state(),
            compute_manifest_patch_need_fn=lambda *_args, **_kwargs: False,
            evaluate_target_crate_overwrite_fn=lambda *_args, **_kwargs: pma_bench_sync.TargetOverwritePlan(
                False,
                False,
            ),
            delete_target_crate_fn=lambda *_args, **_kwargs: None,
            copy_source_crate_fn=lambda *_args, **_kwargs: None,
            patch_manifest_fn=lambda *_args, **_kwargs: None,
            get_source_commit_fn=lambda *_args, **_kwargs: "abc123",
            write_stamp_fn=mock.Mock(side_effect=ValueError("stamp write failure")),
            synced_at_fn=lambda: "2026-03-25T18:22:00+00:00",
            build_binary_fn=lambda *_args, **_kwargs: Path("/tmp/binary"),
        )

        self.assertEqual(exit_code, 1)
        self.assertIn("stamp write failure", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
