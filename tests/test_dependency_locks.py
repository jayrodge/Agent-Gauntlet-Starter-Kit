from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class DependencyLockTests(unittest.TestCase):
    def test_install_entrypoints_resolve_to_hash_locks(self) -> None:
        expected = {
            ROOT / "requirements.txt": "-r requirements/requirements.lock",
            ROOT / "requirements-all.txt": "-r requirements/requirements-all.lock",
            ROOT
            / "examples"
            / "python_simple"
            / "requirements.txt": "-r ../../requirements/requirements.lock",
            ROOT
            / "examples"
            / "python_reference"
            / "requirements.txt": "-r ../../requirements/requirements.lock",
            ROOT
            / "examples"
            / "langgraph"
            / "requirements.txt": "-r ../../requirements/requirements.lock",
            ROOT
            / "examples"
            / "crewai"
            / "requirements.txt": "-r ../../requirements/requirements-all.lock",
        }

        for path, include in expected.items():
            with self.subTest(path=path):
                self.assertIn(include, path.read_text(encoding="utf-8"))

    def test_generated_locks_pin_and_hash_every_package(self) -> None:
        for filename in ("requirements.lock", "requirements-all.lock"):
            path = ROOT / "requirements" / filename
            text = path.read_text(encoding="utf-8")
            package_lines = [
                line
                for line in text.splitlines()
                if line and not line[0].isspace() and not line.startswith("#")
            ]

            with self.subTest(path=path):
                self.assertGreater(len(package_lines), 10)
                self.assertTrue(all("==" in line for line in package_lines))
                self.assertGreaterEqual(text.count("--hash=sha256:"), len(package_lines))


if __name__ == "__main__":
    unittest.main()
