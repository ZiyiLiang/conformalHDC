"""Completion depends only on the expected output files being present."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments_ood import common, diagnose_ood as diagnostic


class OutputPresenceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="chdc-output-presence-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.group = 3
        self.config = {"seed_group": self.group}

    def workflows(self):
        return (("original", common.is_complete,
                 (f"seed{self.group}_ood.csv", f"seed{self.group}_operating.csv")),
                ("diagnostic", diagnostic.is_complete,
                 tuple(diagnostic.file_names(self.group).values())))

    def outputs(self, workflow, names):
        output = self.root / workflow
        output.mkdir(parents=True, exist_ok=True)
        for name in names:
            (output / name).touch()
        return output / f"seed{self.group}_complete.json"

    def test_empty_output_files_suffice_without_marker(self):
        for workflow, checker, names in self.workflows():
            with self.subTest(workflow=workflow):
                marker = self.outputs(workflow, names)
                self.assertFalse(marker.exists())
                self.assertTrue(checker(marker, self.config))

    def test_each_missing_output_requires_a_run(self):
        for workflow, checker, names in self.workflows():
            marker = self.outputs(workflow, names)
            marker.write_text("{}", encoding="utf-8")
            for name in names:
                with self.subTest(workflow=workflow, missing=name):
                    path = marker.parent / name
                    path.unlink()
                    self.assertFalse(checker(marker, self.config))
                    path.touch()

    def test_source_settings_and_marker_contents_are_ignored(self):
        source = self.root / "experiment.py"
        source.write_text("VALUE = 1\n", encoding="utf-8")
        for workflow, checker, names in self.workflows():
            with self.subTest(workflow=workflow):
                marker = self.outputs(workflow, names)
                marker.write_text("not valid JSON", encoding="utf-8")
                changed = {"seed_group": self.group, "dimension": 99, "temperature": 7.0,
                           "source_hashes": {"experiment.py": "changed"},
                           "random_states": [100], "package_versions": {"fixture": "different"}}
                source.write_text("VALUE = 2\n", encoding="utf-8")
                self.assertTrue(checker(marker, changed))
                marker.write_text('{"files": {}, "run_config": null}', encoding="utf-8")
                self.assertTrue(checker(marker, changed))

    def test_other_seed_group_requires_its_own_outputs(self):
        for workflow, checker, names in self.workflows():
            with self.subTest(workflow=workflow):
                marker = self.outputs(workflow, names)
                self.assertFalse(checker(marker, {"seed_group": self.group + 1}))

    def test_directory_does_not_count_as_an_output_file(self):
        for workflow, checker, names in self.workflows():
            marker = self.outputs(workflow, names)
            for name in names:
                with self.subTest(workflow=workflow, directory=name):
                    path = marker.parent / name
                    path.unlink()
                    path.mkdir()
                    self.assertFalse(checker(marker, self.config))
                    path.rmdir()
                    path.touch()

    def entrypoints(self):
        return ((common.main, ("mnist",), self.root / "ood_mnist", ()),
                (diagnostic.main, (), self.root / "ood_diagnostics" / "ood_mnist",
                 ("--dataset", "mnist")))

    def test_both_main_entrypoints_skip_before_loading_data(self):
        for entrypoint, arguments, output, extra in self.entrypoints():
            with self.subTest(entrypoint=entrypoint.__module__):
                self.outputs(str(output), diagnostic.file_names(self.group).values())
                argv = ["runner", str(self.group), "--results-root", str(self.root),
                        "--device", "cpu", "--threads", "1", *extra]
                stdout = io.StringIO()
                with patch("sys.argv", argv), redirect_stdout(stdout), \
                        patch.object(common, "load_inputs", side_effect=AssertionError("Dataset was loaded")) as load:
                    entrypoint(*arguments)
                load.assert_not_called()
                self.assertIn("Already complete", stdout.getvalue())

    def test_check_complete_cli_uses_the_same_file_presence_rule(self):
        for entrypoint, arguments, output, extra in self.entrypoints():
            with self.subTest(entrypoint=entrypoint.__module__):
                self.outputs(str(output), diagnostic.file_names(self.group).values())
                argv = ["runner", str(self.group), "--results-root", str(self.root),
                        "--device", "cpu", "--threads", "1", "--check-complete", *extra]
                with patch("sys.argv", argv), \
                        patch.object(common, "load_inputs", side_effect=AssertionError("Dataset was loaded")) as load:
                    with self.assertRaises(SystemExit) as result:
                        entrypoint(*arguments)
                load.assert_not_called()
                self.assertEqual(result.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
