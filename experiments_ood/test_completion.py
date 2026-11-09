"""Completion checks tolerate comments while retaining provenance and export validation."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from experiments_ood import common, completion, diagnose_ood as diagnostic


SOURCE = b'''"""Experiment fixture."""
import math
# Original experiment comment.
VALUE = 3
TEXT = "score #1"
def score(x):
    if x:
        return x + VALUE
    return VALUE
'''
COMMENTED = b"# Added explanation.\n" + SOURCE


def digest(data):
    return hashlib.sha256(data).hexdigest()


class GitFixture(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="chdc-completion-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "experiment.py"
        self.source.write_bytes(SOURCE)
        self.git("init", "-q")
        self.git("add", "experiment.py")
        self.git("-c", "user.name=Completion tests", "-c", "user.email=test@example.invalid",
                 "-c", "commit.gpgsign=false", "-c", f"core.hooksPath={self.root / 'no-hooks'}",
                 "commit", "-q", "-m", "Fixture source")
        self.revision = self.git("rev-parse", "HEAD").strip()
        self.saved = {
            "schema_version": 1, "dataset": "mnist", "seed_group": 1,
            "repetitions": 1, "random_states": [1], "methods": list(common.METHODS),
            "alpha_grid": [0.5], "nominal_alpha": 0.05, "temperature": 1.0,
            "dimension": 8, "device": "cpu", "threads": 1,
            "data_root": str(self.root / "data"), "package_versions": {"fixture": "1"},
            "source_hashes": {"experiment.py": digest(SOURCE)},
        }
        self.requested = copy.deepcopy(self.saved)
        self.set_current(COMMENTED)
        root_patch = patch.object(completion, "PROJECT_ROOT", self.root)
        root_patch.start()
        self.addCleanup(root_patch.stop)

    def git(self, *arguments):
        return subprocess.check_output(["git", *arguments], cwd=self.root,
                                       text=True, stderr=subprocess.PIPE)

    def set_current(self, source):
        self.source.write_bytes(source)
        self.requested["source_hashes"]["experiment.py"] = digest(source)

    def matches(self, saved=None, requested=None, revision=None):
        return completion.run_configs_match(
            self.saved if saved is None else saved,
            self.requested if requested is None else requested,
            self.revision if revision is None else revision,
        )


class SourceCompatibilityTests(GitFixture):
    def test_added_removed_comments_and_line_endings_match(self):
        variants = [COMMENTED, SOURCE.replace(b"# Original experiment comment.\n", b""),
                    COMMENTED.replace(b"\n", b"\r\n")]
        saved_before = copy.deepcopy(self.saved)
        for source in variants:
            with self.subTest(source=source[:30]):
                self.set_current(source)
                self.assertTrue(self.matches())
                self.assertEqual(completion.source_fingerprint(SOURCE, "experiment.py"),
                                 completion.source_fingerprint(source, "experiment.py"))
        self.assertEqual(self.saved, saved_before)

    def test_historical_crlf_hash_is_verified_against_git_blob(self):
        saved = copy.deepcopy(self.saved)
        saved["source_hashes"]["experiment.py"] = digest(SOURCE.replace(b"\n", b"\r\n"))
        self.assertTrue(self.matches(saved=saved))

    def test_executable_constants_strings_docstrings_imports_and_indentation_fail(self):
        changes = {
            "constant": (b"VALUE = 3", b"VALUE = 4"),
            "hash_in_string": (b"score #1", b"score #2"),
            "docstring": (b"Experiment fixture.", b"Changed experiment."),
            "import": (b"import math", b"import cmath"),
            "indentation": (b"\n    return VALUE\n", b"\n        return VALUE\n"),
        }
        for label, (old, new) in changes.items():
            with self.subTest(change=label):
                source = COMMENTED.replace(old, new)
                self.set_current(source)
                self.assertFalse(self.matches())
                self.assertNotEqual(completion.source_fingerprint(SOURCE, "experiment.py"),
                                    completion.source_fingerprint(source, "experiment.py"))

    def test_non_source_settings_fail(self):
        for key, value in {"dimension": 9, "temperature": 2.0, "random_states": [2],
                           "alpha_grid": [0.25], "data_root": "elsewhere"}.items():
            with self.subTest(setting=key):
                requested = copy.deepcopy(self.requested)
                requested[key] = value
                self.assertFalse(self.matches(requested=requested))

    def test_missing_history_wrong_raw_hash_and_unverified_current_bytes_fail(self):
        self.assertFalse(self.matches(revision="f" * 40))
        self.assertFalse(completion.run_configs_match(self.saved, self.requested, None))
        saved = copy.deepcopy(self.saved)
        saved["source_hashes"]["experiment.py"] = "0" * 64
        self.assertFalse(self.matches(saved=saved))
        requested = copy.deepcopy(self.requested)
        requested["source_hashes"]["experiment.py"] = "0" * 64
        self.assertFalse(self.matches(requested=requested))

    def test_equal_malformed_configs_are_rejected(self):
        for config in [{}, {"source_hashes": {}},
                       {"source_hashes": {"experiment.py": "not-a-sha256"}},
                       {"source_hashes": {"../experiment.py": "a" * 64}}]:
            with self.subTest(config=config):
                self.assertFalse(completion.run_configs_match(config, copy.deepcopy(config), self.revision))

    def test_changed_source_path_set_fails(self):
        requested = copy.deepcopy(self.requested)
        requested["source_hashes"]["another.py"] = digest(SOURCE)
        self.assertFalse(self.matches(requested=requested))


class CheckerAdapterTests(unittest.TestCase):
    def test_only_known_checker_migration_is_normalized(self):
        cases = [
            ("experiments_ood/common.py", "is_complete(marker_path, config)",
             'marker["run_config"] != config',
             'not run_configs_match(marker["run_config"], config, marker.get("git_revision"))',
             "from experiments_ood.completion import run_configs_match\n"),
            ("experiments_ood/diagnose_ood.py", "validate_export(marker, output, config)",
             'marker.get("run_config") != config or any(marker.get(key) != value '
             'for key, value in config.items())',
             'not common.run_configs_match(marker.get("run_config"), config, marker.get("git_revision")) '
             'or any(marker.get(key) != value for key, value in marker["run_config"].items())', ""),
        ]
        for path, signature, old_test, new_test, new_import in cases:
            with self.subTest(path=path):
                old = (f"def {signature}:\n    if {old_test}:\n        return False\n"
                       '    if marker["bad_csv"]:\n        return False\n    return True\n')
                new = new_import + old.replace(old_test, new_test)
                fingerprint = completion.source_fingerprint(old.encode(), path)
                self.assertEqual(fingerprint, completion.source_fingerprint(new.encode(), path))
                for change in [new.replace("return True", "return False"),
                               new.replace('marker["bad_csv"]', "False")]:
                    self.assertNotEqual(fingerprint, completion.source_fingerprint(change.encode(), path))
                self.assertNotEqual(completion.source_fingerprint(old.encode(), "unrelated.py"),
                                    completion.source_fingerprint(new.encode(), "unrelated.py"))


class CompletionIntegrationTests(GitFixture):
    def setUp(self):
        super().setUp()
        self.output = self.root / "exports"
        self.output.mkdir()
        self.frames, self.metadata = self.export_frames()
        self.files = diagnostic.file_names(1)
        for kind, name in self.files.items():
            compression = {"method": "gzip", "mtime": 0} if name.endswith(".gz") else None
            self.frames[kind].to_csv(self.output / name, index=False, compression=compression)
        self.marker = {
            **copy.deepcopy(self.saved), "run_config": copy.deepcopy(self.saved),
            "files": self.files, "sha256": {
                kind: common.sha256_file(self.output / name) for kind, name in self.files.items()},
            "repetition_metadata": [self.metadata], "input_array_sha256": "a" * 64,
            "git_revision": self.revision, "roc_positive_class": "ood",
            "threshold_rule": "id_score < threshold",
        }
        self.marker_path = self.output / "seed1_complete.json"
        self.marker_path.write_text(json.dumps(self.marker), encoding="utf-8")

    def export_frames(self):
        parts = [("calibration", [0.6, 0.8], [0, 0], [2, 3]),
                 ("test", [0.7, 0.9], [0, 0], [4, 5]),
                 ("ood", [0.1, 0.2, 0.3, 0.4], [6, 7, 8, 9], [6, 7, 8, 9])]
        score_rows = []
        for partition, values, labels, indices in parts:
            for value, label, index in zip(values, labels, indices):
                score_rows.append({"dataset": "mnist", "random_state": 1,
                                   "partition": partition, "sample_index": index, "label": label,
                                   **dict.fromkeys(common.METHODS, value)})
        sizes = {"n_train": 2, "n_cal": 2, "n_test": 2, "n_ood": 4}
        metadata = {"random_state": 1, **sizes,
                    "class_counts": {"train": {"0": 2}, "calibration": {"0": 2},
                                     "test": {"0": 2}, "ood": {str(i): 1 for i in range(6, 10)}}}
        truth = np.array([1, 1, 0, 0, 0, 0])
        values = np.array([0.7, 0.9, 0.1, 0.2, 0.3, 0.4])
        auroc_rows, operating_rows, curves = [], [], []
        for method in common.METHODS:
            shared = {"dataset": "mnist", "random_state": 1, "method": method,
                      "nominal_alpha": 0.05, "temperature": 1.0, "dimension": 8, **sizes}
            auroc_rows.append({**shared, "ood_auroc": common.roc_auc_score(truth, values)})
            fpr, tpr, thresholds = diagnostic.roc_curve(1 - truth, -values, drop_intermediate=False)
            curves.append(pd.DataFrame({"dataset": "mnist", "random_state": 1, "method": method,
                                        "id_false_rejection": fpr, "ood_rejection": tpr,
                                        "detection_threshold": thresholds}))
            operating_rows.append({**shared, "alpha": 0.5,
                                   "threshold": common.lower_tail_threshold(np.array([0.6, 0.8]), 0.5),
                                   "id_rejected": 0, "ood_rejected": 4,
                                   "id_false_rejection": 0.0, "ood_rejection": 1.0})
        return {"scores": pd.DataFrame(score_rows, columns=diagnostic.SCORE_COLUMNS),
                "roc": pd.concat(curves, ignore_index=True), "auroc": pd.DataFrame(auroc_rows),
                "operating": pd.DataFrame(operating_rows)}, metadata

    def assert_rejected_by_both(self, requested):
        self.assertFalse(common.is_complete(self.marker_path, requested))
        with self.assertRaises(ValueError):
            diagnostic.validate_export(self.marker, self.output, requested)
        self.assertFalse(diagnostic.is_complete(self.marker_path, requested))

    def test_both_checkers_skip_after_comments_without_rewriting_marker(self):
        before = self.marker_path.read_bytes()
        self.assertTrue(common.is_complete(self.marker_path, self.requested))
        self.assertTrue(diagnostic.validate_export(self.marker, self.output, self.requested))
        self.assertTrue(diagnostic.is_complete(self.marker_path, self.requested))
        self.assertEqual(self.marker_path.read_bytes(), before)

    def test_both_checkers_reject_executable_and_configuration_changes(self):
        self.set_current(COMMENTED.replace(b"VALUE = 3", b"VALUE = 4"))
        self.assert_rejected_by_both(self.requested)
        self.set_current(COMMENTED)
        requested = copy.deepcopy(self.requested)
        requested["dimension"] = 9
        self.assert_rejected_by_both(requested)

    def test_both_checkers_reject_corrupted_csv(self):
        (self.output / self.files["auroc"]).write_bytes(b"corrupted export\n")
        self.assert_rejected_by_both(self.requested)

    def test_diagnostic_rejects_top_level_config_mismatch(self):
        self.marker["dimension"] = 9
        with self.assertRaises(ValueError):
            diagnostic.validate_export(self.marker, self.output, self.requested)

    def test_diagnostic_keeps_cross_file_numerical_validation(self):
        frame = self.frames["auroc"].copy()
        frame.loc[0, "ood_auroc"] = 0.5
        path = self.output / self.files["auroc"]
        frame.to_csv(path, index=False)
        self.marker["sha256"]["auroc"] = common.sha256_file(path)
        with self.assertRaises(ValueError):
            diagnostic.validate_export(self.marker, self.output, self.requested)


if __name__ == "__main__":
    unittest.main()
