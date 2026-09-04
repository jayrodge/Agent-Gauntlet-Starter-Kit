"""Pin the vendored submission schema to the organizer admission contract.

The competitor packager and the organizer runner both treat
``blueprint/submission.schema.json`` as the shared field contract. This test
fails if the copy in ``arena_clients/`` is reformatted or rewritten so the two
sides can drift.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
VENDORED_SCHEMA = ROOT / "arena_clients" / "submission.schema.json"

# SHA-256 of the pilot blueprint/submission.schema.json that was vendored.
# Recompute from that file when the shared contract is intentionally revised.
PILOT_SCHEMA_SHA256 = (
    "13ce8eed8dea5bc782041f19b13cf55bb5cb14f7d44f0844d83d326fb56cb871"
)


def _discover_pilot_schema() -> Path | None:
    env_file = os.environ.get("GAUNTLET_PILOT_SCHEMA")
    if env_file:
        path = Path(env_file)
        if path.is_file():
            return path
    env_root = os.environ.get("GAUNTLET_PILOT_ROOT")
    if env_root:
        path = Path(env_root) / "blueprint" / "submission.schema.json"
        if path.is_file():
            return path
    for parent in Path(__file__).resolve().parents:
        candidate = parent / ".scratch" / "senthil-sandbox" / "blueprint" / "submission.schema.json"
        if candidate.is_file():
            return candidate
    return None


class SubmissionSchemaContractTests(unittest.TestCase):
    def test_vendored_schema_matches_pinned_digest(self) -> None:
        raw = VENDORED_SCHEMA.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        self.assertEqual(
            digest,
            PILOT_SCHEMA_SHA256,
            "vendored submission.schema.json no longer matches the pinned "
            "pilot digest; update the pin only when the organizer contract "
            "actually changes",
        )

    def test_vendored_schema_encodes_required_admission_fields(self) -> None:
        schema = json.loads(VENDORED_SCHEMA.read_text(encoding="utf-8"))
        self.assertEqual(schema["required"], [
            "schemaVersion",
            "kind",
            "metadata",
            "runtime",
            "dependencies",
            "artifact",
        ])
        self.assertEqual(schema["properties"]["schemaVersion"]["const"], 1)
        self.assertEqual(
            schema["properties"]["kind"]["const"],
            "GauntletAgentSubmission",
        )
        self.assertEqual(
            schema["properties"]["runtime"]["properties"]["pythonVersion"]["const"],
            "3.12",
        )
        self.assertEqual(
            schema["properties"]["runtime"]["properties"]["language"]["const"],
            "python",
        )
        self.assertEqual(
            schema["properties"]["metadata"]["properties"]["agentId"]["pattern"],
            "^[a-z][a-z0-9-]{0,62}$",
        )

    def test_vendored_schema_matches_pilot_copy_when_reachable(self) -> None:
        pilot = _discover_pilot_schema()
        if pilot is None:
            self.skipTest(
                "pilot blueprint/submission.schema.json is not on this machine"
            )
        vendored = json.loads(VENDORED_SCHEMA.read_text(encoding="utf-8"))
        remote = json.loads(pilot.read_text(encoding="utf-8"))
        self.assertEqual(
            vendored,
            remote,
            f"vendored schema drifted from pilot file {pilot}",
        )
        self.assertEqual(
            VENDORED_SCHEMA.read_bytes(),
            pilot.read_bytes(),
            f"vendored schema bytes drifted from pilot file {pilot}",
        )


if __name__ == "__main__":
    unittest.main()
