"""Tests for submission packaging and local admission pre-validation."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arena_clients import package


def _pilot_admission():
    for parent in Path(__file__).resolve().parents:
        path = parent / ".scratch" / "senthil-sandbox" / "runner" / "admission.py"
        if path.is_file():
            spec = importlib.util.spec_from_file_location("pilot_admission", path)
            if spec is None or spec.loader is None:
                return None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    return None


class ManifestValidationTests(unittest.TestCase):
    def _manifest(self, **overrides: object) -> dict:
        value = {
            "schemaVersion": 1,
            "kind": "GauntletAgentSubmission",
            "metadata": {"agentId": "sample-agent", "agentName": "Sample Agent"},
            "runtime": {
                "language": "python",
                "pythonVersion": "3.12",
                "workingDirectory": ".",
                "entrypoint": ["python", "agent.py"],
            },
            "dependencies": {"lockfile": "requirements.lock"},
            "artifact": {"files": ["agent.py", "requirements.lock"]},
        }
        value.update(overrides)
        return value

    def test_valid_manifest_is_accepted(self) -> None:
        package.validate_manifest(self._manifest())

    def test_uppercase_agent_id_is_rejected(self) -> None:
        manifest = self._manifest()
        manifest["metadata"] = {"agentId": "Sample-Agent", "agentName": "Sample"}
        with self.assertRaisesRegex(package.PackagingError, "agentId"):
            package.validate_manifest(manifest)

    def test_shell_entrypoint_is_rejected(self) -> None:
        manifest = self._manifest()
        manifest["runtime"] = {
            "language": "python",
            "pythonVersion": "3.12",
            "workingDirectory": ".",
            "entrypoint": ["bash", "agent.py"],
        }
        with self.assertRaisesRegex(package.PackagingError, "python"):
            package.validate_manifest(manifest)

    def test_undeclared_entrypoint_is_rejected(self) -> None:
        manifest = self._manifest()
        manifest["runtime"]["entrypoint"] = ["python", "missing.py"]
        with self.assertRaisesRegex(package.PackagingError, "entrypoint script"):
            package.validate_manifest(manifest)

    def test_python_version_must_be_3_12(self) -> None:
        manifest = self._manifest()
        manifest["runtime"]["pythonVersion"] = "3.11"
        with self.assertRaisesRegex(package.PackagingError, "3.12"):
            package.validate_manifest(manifest)


class DirectoryAdmissionTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self._temporary.name) / "gauntlet-submission"
        self.directory.mkdir()
        (self.directory / "agent.py").write_text("print('ok')\n", encoding="utf-8")
        (self.directory / "requirements.lock").write_text("demo==1.0\n", encoding="utf-8")
        (self.directory / "submission.json").write_text(
            json.dumps(
                {
                    "schemaVersion": 1,
                    "kind": "GauntletAgentSubmission",
                    "metadata": {"agentId": "sample-agent", "agentName": "Sample"},
                    "runtime": {
                        "language": "python",
                        "pythonVersion": "3.12",
                        "workingDirectory": ".",
                        "entrypoint": ["python", "agent.py"],
                    },
                    "dependencies": {"lockfile": "requirements.lock"},
                    "artifact": {"files": ["agent.py", "requirements.lock"]},
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def test_valid_directory_passes(self) -> None:
        manifest = package.validate_submission_directory(self.directory)
        self.assertEqual(manifest["metadata"]["agentId"], "sample-agent")

    def test_undeclared_file_is_rejected(self) -> None:
        (self.directory / "extra.txt").write_text("nope", encoding="utf-8")
        with self.assertRaisesRegex(package.PackagingError, "undeclared files"):
            package.validate_submission_directory(self.directory)

    def test_missing_declared_file_is_rejected(self) -> None:
        (self.directory / "agent.py").unlink()
        with self.assertRaisesRegex(package.PackagingError, "declared files are missing"):
            package.validate_submission_directory(self.directory)

    def test_env_file_is_rejected_even_when_declared(self) -> None:
        manifest = json.loads((self.directory / "submission.json").read_text())
        manifest["artifact"]["files"].append(".env")
        (self.directory / "submission.json").write_text(json.dumps(manifest))
        (self.directory / ".env").write_text("ARENA_API_KEY=not-a-real-key\n")
        with self.assertRaisesRegex(package.PackagingError, "credential-shaped filename"):
            package.validate_submission_directory(self.directory)

    def test_symlink_is_rejected(self) -> None:
        manifest = json.loads((self.directory / "submission.json").read_text())
        manifest["artifact"]["files"].append("linked.py")
        (self.directory / "submission.json").write_text(json.dumps(manifest))
        (self.directory / "linked.py").symlink_to("agent.py")
        with self.assertRaisesRegex(package.PackagingError, "symlinks"):
            package.validate_submission_directory(self.directory)

    def test_raw_api_key_assignment_is_rejected(self) -> None:
        (self.directory / "agent.py").write_text(
            "ARENA_API_KEY=not-a-real-key\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(package.PackagingError, "raw API key assignment"):
            package.validate_submission_directory(self.directory)

    def test_api_key_name_in_prose_is_allowed(self) -> None:
        (self.directory / "agent.py").write_text(
            '"""Uses the ARENA_API_KEY environment variable.\n\nvalue = elsewhere\n"""\n',
            encoding="utf-8",
        )
        manifest = package.validate_submission_directory(self.directory)
        self.assertEqual(manifest["metadata"]["agentId"], "sample-agent")

    def test_oversized_file_is_rejected(self) -> None:
        (self.directory / "agent.py").write_bytes(b"x" * (package.MAX_FILE_BYTES + 1))
        with self.assertRaisesRegex(package.PackagingError, "file limit"):
            package.validate_submission_directory(self.directory)


class PackagingCliTests(unittest.TestCase):
    def test_packages_python_simple_and_writes_checksum(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            result = package.package_submission(
                ROOT,
                output,
                agent_id="python-simple-reference",
                agent_name="Python Simple Reference",
            )
            directory = result["directory"]
            archive = result["archive"]
            checksum = result["checksum"]
            assert isinstance(directory, Path)
            assert isinstance(archive, Path)
            assert isinstance(checksum, Path)
            manifest = json.loads((directory / "submission.json").read_text())
            self.assertEqual(manifest["kind"], "GauntletAgentSubmission")
            self.assertEqual(manifest["runtime"]["pythonVersion"], "3.12")
            self.assertEqual(
                manifest["runtime"]["entrypoint"],
                ["python", "examples/python_simple/agent.py"],
            )
            self.assertIn("requirements.lock", manifest["artifact"]["files"])
            self.assertIn("arena_clients/package.py", manifest["artifact"]["files"])
            self.assertIn("my_strategy.py", manifest["artifact"]["files"])
            self.assertTrue(archive.is_file())
            digest = hashlib.sha256(archive.read_bytes()).hexdigest()
            self.assertTrue(checksum.read_text(encoding="utf-8").startswith(digest))
            with tarfile.open(archive, "r:gz") as handle:
                names = handle.getnames()
            self.assertTrue(all(name == "gauntlet-submission" or name.startswith("gauntlet-submission/") for name in names))
            self.assertIn("gauntlet-submission/submission.json", names)

    def test_cli_requires_identity(self) -> None:
        code = package.main([])
        self.assertEqual(code, 2)

    def test_cli_check_accepts_valid_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            result = package.package_submission(
                ROOT,
                output,
                agent_id="python-simple-reference",
                agent_name="Python Simple Reference",
            )
            directory = result["directory"]
            assert isinstance(directory, Path)
            self.assertEqual(package.main(["--check", str(directory)]), 0)

    def test_pilot_admission_accepts_generated_package_when_reachable(self) -> None:
        admission = _pilot_admission()
        if admission is None:
            self.skipTest("pilot runner/admission.py is not on this machine")
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            result = package.package_submission(
                ROOT,
                output,
                agent_id="python-simple-reference",
                agent_name="Python Simple Reference",
            )
            directory = result["directory"]
            assert isinstance(directory, Path)
            record = admission.admit(directory)
            self.assertEqual(record["agentId"], "python-simple-reference")
            self.assertTrue(record["artifactDigest"].startswith("sha256:"))


class ReferenceSubmissionTests(unittest.TestCase):
    def test_committed_reference_matches_generated_python_simple_manifest(self) -> None:
        reference = (
            ROOT
            / "examples"
            / "python_simple"
            / "reference_submission"
            / "submission.json"
        )
        self.assertTrue(reference.is_file(), "reference submission.json must ship")
        with tempfile.TemporaryDirectory() as temporary:
            result = package.package_submission(
                ROOT,
                Path(temporary),
                agent_id="python-simple-reference",
                agent_name="Python Simple Reference",
            )
            directory = result["directory"]
            assert isinstance(directory, Path)
            generated = json.loads((directory / "submission.json").read_text())
        expected = json.loads(reference.read_text(encoding="utf-8"))
        self.assertEqual(generated, expected)


if __name__ == "__main__":
    unittest.main()
