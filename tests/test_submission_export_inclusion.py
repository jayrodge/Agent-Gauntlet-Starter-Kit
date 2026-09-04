"""Guard the submission path against a silent export omit.

The packaging CLI, guide, and reference example must ship to competitors.
These assertions fail if a future exclude-list edit drops that surface.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path, PurePosixPath
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arena_clients.package import PARTICIPANT_SUBMISSION_PATHS


def _load_export_module():
    path = ROOT / "scripts" / "export_participant_package.py"
    if not path.is_file():
        return None
    spec = importlib.util.spec_from_file_location("export_participant_package", path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SubmissionExportInclusionTests(unittest.TestCase):
    def test_submission_surface_files_exist(self) -> None:
        for relative in PARTICIPANT_SUBMISSION_PATHS:
            path = ROOT / relative
            self.assertTrue(path.is_file(), f"missing {relative}")

    def test_submission_surface_names_are_not_forbidden_export_prefixes(self) -> None:
        forbidden_name_prefixes = ("qualification_",)
        forbidden_path_prefixes = ("simulation/", "sample_agents/", "local_teams/")
        forbidden_names = {
            "production-qualification.md",
            "model_selector_heuristics.py",
        }
        for relative in PARTICIPANT_SUBMISSION_PATHS:
            name = Path(relative).name
            self.assertFalse(
                any(name.startswith(prefix) for prefix in forbidden_name_prefixes),
                f"{relative} trips FORBIDDEN_NAME_PREFIXES",
            )
            self.assertFalse(
                any(
                    relative == prefix.rstrip("/") or relative.startswith(prefix)
                    for prefix in forbidden_path_prefixes
                ),
                f"{relative} trips FORBIDDEN_EXPORT_PREFIXES",
            )
            self.assertNotIn(name, forbidden_names)
            self.assertFalse(name.startswith("brev_"))
            self.assertFalse(name.startswith("setup_qualification_"))
            self.assertFalse(name.startswith("run_qualification_"))

    def test_exclude_list_does_not_omit_submission_surface(self) -> None:
        export = _load_export_module()
        if export is None:
            self.skipTest("export script is organizer-only and is not in this tree")
        self.assertEqual(export.REQUIRED_EXPORT_PATHS, PARTICIPANT_SUBMISSION_PATHS)
        patterns = export._load_exclude_patterns()
        omitted = [
            relative
            for relative in PARTICIPANT_SUBMISSION_PATHS
            if export.should_omit(PurePosixPath(relative), patterns)
        ]
        self.assertEqual(
            omitted,
            [],
            "participant_export_exclude.txt would drop the submission path",
        )
        self.assertTrue(
            export.should_omit(
                PurePosixPath("local_teams/team-01/my_strategy.py"),
                patterns,
            ),
            "local_teams/** must stay out of the participant export",
        )


if __name__ == "__main__":
    unittest.main()
