"""Package a competitor agent into an organizer-admissible submission.

The field contract lives in ``submission.schema.json`` (vendored from the
organizer blueprint). Extra rules here match organizer admission: the
entrypoint must launch Python directly, declared files must be complete,
symlinks and credential-shaped files are rejected, and the same size and
secret heuristics apply. Failures must surface on the laptop, before upload.

Usage:
    python -m arena_clients.package --agent-id my-team --agent-name "My Team"
    python -m arena_clients.package --check dist/gauntlet-submission
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


MANIFEST_NAME = "submission.json"
SCHEMA_NAME = "submission.schema.json"
SUBMISSION_DIR_NAME = "gauntlet-submission"
DEFAULT_ENTRYPOINT = "examples/python_simple/agent.py"
DEFAULT_LOCKFILE_SOURCE = "requirements/requirements.lock"
DEFAULT_LOCKFILE_DEST = "requirements.lock"
DEFAULT_RUNTIME_FILES = (
    "base_strategy.py",
    "model_selector.py",
    "my_strategy.py",
)

# Files that must reach competitors with this CLI. The organizer export gate
# asserts the same paths still exist after applying the omit list.
PARTICIPANT_SUBMISSION_PATHS = (
    "arena_clients/package.py",
    "arena_clients/submission.schema.json",
    "docs/submitting.md",
    "examples/python_simple/reference_submission/submission.json",
)

# Constants copied from organizer admission so a locally valid package is not
# later rejected for a different size, name, or secret rule.
AGENT_ID = re.compile(r"^[a-z][a-z0-9-]{0,62}$")
PYTHON_VERSIONS = {"3.12"}
PYTHON_COMMANDS = {"python", "python3"}
MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_TOTAL_BYTES = 50 * 1024 * 1024
DENIED_NAMES = {
    ".env",
    "credentials",
    "credentials.json",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "id_rsa",
}
DENIED_SUFFIXES = {".key", ".p12", ".pem", ".pfx"}
SECRET_PATTERNS = (
    ("private key", re.compile(rb"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----")),
    (
        "raw API key assignment",
        re.compile(
            rb"(?m)^[ \t]*(?:ARENA|NVIDIA|OPENAI|GOOGLE)_API_KEY[ \t]*=[ \t]*(?!openshell:resolve:env:)[^\s#]+"
        ),
    ),
)
SKIP_COPY_NAMES = {"__pycache__", ".DS_Store"}
SKIP_COPY_SUFFIXES = {".pyc", ".pyo"}


class PackagingError(RuntimeError):
    """The submission cannot be assembled or does not pass admission."""


def schema_path() -> Path:
    return Path(__file__).resolve().parent / SCHEMA_NAME


def load_schema() -> dict[str, Any]:
    raw = schema_path().read_text(encoding="utf-8")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise PackagingError(f"{SCHEMA_NAME} must be a JSON object")
    return value


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PackagingError(f"{label} must be an object")
    return value


def _text(value: Any, label: str, *, maximum: int = 512) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise PackagingError(
            f"{label} must be a non-empty string up to {maximum} characters"
        )
    return value.strip()


def _keys(value: dict[str, Any], label: str, expected: set[str]) -> None:
    extra = set(value) - expected
    if extra:
        raise PackagingError(
            f"{label} contains unsupported fields: {', '.join(sorted(extra))}"
        )


def _relative_path(value: Any, label: str) -> str:
    text = _text(value, label, maximum=256)
    path = PurePosixPath(text)
    if path.is_absolute() or text != path.as_posix() or text in {".", ".."} or ".." in path.parts:
        raise PackagingError(f"{label} must be a normalized repository-relative path")
    return text


def validate_manifest(value: dict[str, Any]) -> dict[str, Any]:
    """Apply the same manifest rules organizer admission uses."""
    schema = load_schema()
    required = set(schema.get("required") or [])
    missing = required - set(value)
    if missing:
        raise PackagingError(
            f"{MANIFEST_NAME} is missing required fields: {', '.join(sorted(missing))}"
        )
    _keys(
        value,
        MANIFEST_NAME,
        {"schemaVersion", "kind", "metadata", "runtime", "dependencies", "artifact"},
    )
    if value.get("schemaVersion") != schema["properties"]["schemaVersion"]["const"]:
        raise PackagingError("schemaVersion must equal 1")
    if value.get("kind") != schema["properties"]["kind"]["const"]:
        raise PackagingError("kind must equal GauntletAgentSubmission")

    metadata = _object(value.get("metadata"), "metadata")
    _keys(metadata, "metadata", {"agentId", "agentName"})
    agent_id = _text(metadata.get("agentId"), "metadata.agentId", maximum=63)
    if not AGENT_ID.fullmatch(agent_id):
        raise PackagingError(
            "metadata.agentId must use lowercase letters, digits, and hyphens"
        )
    _text(metadata.get("agentName"), "metadata.agentName", maximum=80)

    runtime = _object(value.get("runtime"), "runtime")
    _keys(runtime, "runtime", {"language", "pythonVersion", "workingDirectory", "entrypoint"})
    if runtime.get("language") != "python":
        raise PackagingError("the initial profile accepts only language=python")
    if runtime.get("pythonVersion") not in PYTHON_VERSIONS:
        raise PackagingError(
            "the initial executable profile requires runtime.pythonVersion=3.12"
        )
    if runtime.get("workingDirectory") != ".":
        raise PackagingError("runtime.workingDirectory must equal '.'")
    entrypoint = runtime.get("entrypoint")
    if not isinstance(entrypoint, list) or not 2 <= len(entrypoint) <= 32:
        raise PackagingError("runtime.entrypoint must contain 2 through 32 arguments")
    if any(not isinstance(item, str) or not item or len(item) > 512 for item in entrypoint):
        raise PackagingError("runtime.entrypoint contains an invalid argument")
    if entrypoint[0] not in PYTHON_COMMANDS:
        raise PackagingError("runtime.entrypoint must launch python or python3 directly")

    dependencies = _object(value.get("dependencies"), "dependencies")
    _keys(dependencies, "dependencies", {"lockfile"})
    lockfile = _relative_path(dependencies.get("lockfile"), "dependencies.lockfile")

    artifact = _object(value.get("artifact"), "artifact")
    _keys(artifact, "artifact", {"files"})
    files = artifact.get("files")
    if not isinstance(files, list) or not 1 <= len(files) <= 2048:
        raise PackagingError("artifact.files must contain 1 through 2048 paths")
    declared = [_relative_path(item, "artifact.files[]") for item in files]
    if len(set(declared)) != len(declared):
        raise PackagingError("artifact.files contains duplicate paths")
    if lockfile not in declared:
        raise PackagingError("dependencies.lockfile must be declared in artifact.files")
    entry_script = _relative_path(entrypoint[1], "runtime.entrypoint[1]")
    if entry_script not in declared:
        raise PackagingError("runtime entrypoint script must be declared in artifact.files")
    return value


def _forbidden_path(path: str) -> str | None:
    name = PurePosixPath(path).name.lower()
    if name == ".env" or name.startswith(".env.") or name in DENIED_NAMES:
        return "credential-shaped filename"
    if PurePosixPath(name).suffix in DENIED_SUFFIXES:
        return "credential-shaped file suffix"
    return None


def validate_submission_directory(directory: Path) -> dict[str, Any]:
    """Validate a packed directory with the organizer admission rules."""
    root = directory.resolve()
    if not root.is_dir():
        raise PackagingError(f"submission directory does not exist: {directory}")
    path = root / MANIFEST_NAME
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise PackagingError(f"could not read {MANIFEST_NAME}: {exc}") from exc
    manifest = validate_manifest(_object(value, MANIFEST_NAME))
    declared = set(manifest["artifact"]["files"])

    actual: set[str] = set()
    for child in root.rglob("*"):
        relative = child.relative_to(root).as_posix()
        if child.is_symlink():
            raise PackagingError(f"symlinks are not accepted: {relative}")
        if child.is_dir():
            continue
        if not child.is_file():
            raise PackagingError(f"unsupported filesystem object: {relative}")
        if relative != MANIFEST_NAME:
            actual.add(relative)
    missing = sorted(declared - actual)
    extra = sorted(actual - declared)
    if missing:
        raise PackagingError(f"declared files are missing: {', '.join(missing)}")
    if extra:
        raise PackagingError(f"undeclared files are present: {', '.join(extra)}")

    total = 0
    for relative in sorted(declared):
        reason = _forbidden_path(relative)
        if reason:
            raise PackagingError(f"{relative}: {reason}")
        file_path = root / relative
        try:
            data = file_path.read_bytes()
        except OSError as exc:
            raise PackagingError(f"could not read {relative}: {exc}") from exc
        if len(data) > MAX_FILE_BYTES:
            raise PackagingError(f"{relative} exceeds the {MAX_FILE_BYTES}-byte file limit")
        total += len(data)
        if total > MAX_TOTAL_BYTES:
            raise PackagingError(
                f"submission exceeds the {MAX_TOTAL_BYTES}-byte total limit"
            )
        for label, pattern in SECRET_PATTERNS:
            if pattern.search(data):
                raise PackagingError(f"{relative}: detected {label}")
    return manifest


def kit_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _skip_copy(path: Path) -> bool:
    if path.name in SKIP_COPY_NAMES or path.suffix in SKIP_COPY_SUFFIXES:
        return True
    return any(part in SKIP_COPY_NAMES for part in path.parts)


def _posix_relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def collect_runtime_files(
    source: Path,
    *,
    entrypoint: str,
    extra_files: Iterable[str] = (),
) -> list[str]:
    """Enumerate runtime files the default starter-kit entrypoint needs."""
    source = source.resolve()
    collected: list[str] = []

    clients = source / "arena_clients"
    if not clients.is_dir():
        raise PackagingError(f"missing runtime package: {clients}")
    for child in sorted(clients.rglob("*")):
        if _skip_copy(child) or not child.is_file():
            continue
        if child.is_symlink():
            raise PackagingError(
                f"symlinks are not accepted: {_posix_relative(child, source)}"
            )
        collected.append(_posix_relative(child, source))

    for relative in DEFAULT_RUNTIME_FILES:
        path = source / relative
        if not path.is_file():
            raise PackagingError(f"missing runtime file: {relative}")
        if path.is_symlink():
            raise PackagingError(f"symlinks are not accepted: {relative}")
        collected.append(relative)

    entry = _relative_path(entrypoint, "entrypoint")
    entry_path = source / entry
    if not entry_path.is_file():
        raise PackagingError(f"entrypoint file does not exist: {entry}")
    if entry_path.is_symlink():
        raise PackagingError(f"symlinks are not accepted: {entry}")
    collected.append(entry)

    for relative in extra_files:
        extra = _relative_path(relative, "--include")
        extra_path = source / extra
        if not extra_path.is_file():
            raise PackagingError(f"included file does not exist: {extra}")
        if extra_path.is_symlink():
            raise PackagingError(f"symlinks are not accepted: {extra}")
        collected.append(extra)

    unique: list[str] = []
    seen: set[str] = set()
    for item in collected:
        if item == MANIFEST_NAME:
            continue
        if item in seen:
            continue
        seen.add(item)
        unique.append(item)
    return unique


def build_manifest(
    *,
    agent_id: str,
    agent_name: str,
    entrypoint: str,
    files: list[str],
    lockfile: str = DEFAULT_LOCKFILE_DEST,
    extra_entrypoint_args: Iterable[str] = (),
) -> dict[str, Any]:
    command = ["python", entrypoint, *extra_entrypoint_args]
    manifest = {
        "schemaVersion": 1,
        "kind": "GauntletAgentSubmission",
        "metadata": {
            "agentId": agent_id,
            "agentName": agent_name,
        },
        "runtime": {
            "language": "python",
            "pythonVersion": "3.12",
            "workingDirectory": ".",
            "entrypoint": command,
        },
        "dependencies": {"lockfile": lockfile},
        "artifact": {"files": list(files)},
    }
    return validate_manifest(manifest)


def _copy_file(source: Path, destination: Path) -> None:
    if source.is_symlink():
        raise PackagingError(
            f"symlinks are not accepted: {source.name}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination, follow_symlinks=False)


def assemble_submission(
    source: Path,
    destination: Path,
    *,
    agent_id: str,
    agent_name: str,
    entrypoint: str = DEFAULT_ENTRYPOINT,
    lockfile_source: str = DEFAULT_LOCKFILE_SOURCE,
    lockfile_dest: str = DEFAULT_LOCKFILE_DEST,
    extra_files: Iterable[str] = (),
    extra_entrypoint_args: Iterable[str] = (),
    force: bool = False,
) -> dict[str, Any]:
    source = source.resolve()
    destination = destination.resolve()
    if destination.exists():
        if not force:
            raise PackagingError(
                f"output directory already exists: {destination} (pass --force)"
            )
        if destination.is_dir():
            shutil.rmtree(destination)
        else:
            destination.unlink()
    destination.mkdir(parents=True)

    files = collect_runtime_files(
        source,
        entrypoint=entrypoint,
        extra_files=extra_files,
    )
    lock_src = source / lockfile_source
    if not lock_src.is_file():
        raise PackagingError(f"lockfile does not exist: {lockfile_source}")
    if lock_src.is_symlink():
        raise PackagingError(f"symlinks are not accepted: {lockfile_source}")
    _copy_file(lock_src, destination / lockfile_dest)
    if lockfile_dest not in files:
        files.append(lockfile_dest)

    for relative in files:
        if relative == lockfile_dest:
            continue
        _copy_file(source / relative, destination / relative)

    manifest = build_manifest(
        agent_id=agent_id,
        agent_name=agent_name,
        entrypoint=entrypoint,
        files=files,
        lockfile=lockfile_dest,
        extra_entrypoint_args=extra_entrypoint_args,
    )
    (destination / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    validate_submission_directory(destination)
    return manifest


def write_archive(submission_dir: Path, archive_path: Path) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_path.exists():
        archive_path.unlink()
    with tarfile.open(archive_path, "w:gz", format=tarfile.USTAR_FORMAT) as archive:
        archive.add(submission_dir, arcname=SUBMISSION_DIR_NAME)


def write_checksum(archive_path: Path) -> Path:
    digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    checksum_path = archive_path.with_name(archive_path.name + ".sha256")
    checksum_path.write_text(f"{digest}  {archive_path.name}\n", encoding="utf-8")
    return checksum_path


def package_submission(
    source: Path,
    output: Path,
    *,
    agent_id: str,
    agent_name: str,
    entrypoint: str = DEFAULT_ENTRYPOINT,
    lockfile_source: str = DEFAULT_LOCKFILE_SOURCE,
    lockfile_dest: str = DEFAULT_LOCKFILE_DEST,
    extra_files: Iterable[str] = (),
    extra_entrypoint_args: Iterable[str] = (),
    force: bool = False,
) -> dict[str, Path | dict[str, Any]]:
    submission_dir = output / SUBMISSION_DIR_NAME
    manifest = assemble_submission(
        source,
        submission_dir,
        agent_id=agent_id,
        agent_name=agent_name,
        entrypoint=entrypoint,
        lockfile_source=lockfile_source,
        lockfile_dest=lockfile_dest,
        extra_files=extra_files,
        extra_entrypoint_args=extra_entrypoint_args,
        force=force,
    )
    archive_path = output / f"{agent_id}-submission.tar.gz"
    write_archive(submission_dir, archive_path)
    checksum_path = write_checksum(archive_path)
    return {
        "directory": submission_dir,
        "archive": archive_path,
        "checksum": checksum_path,
        "manifest": manifest,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m arena_clients.package",
        description="Assemble and self-check an Agent Gauntlet submission.",
    )
    parser.add_argument(
        "--check",
        type=Path,
        help="Validate an existing submission directory and exit.",
    )
    parser.add_argument(
        "--agent-id",
        help="Approved agent id (lowercase letters, digits, hyphens).",
    )
    parser.add_argument(
        "--agent-name",
        help="Leaderboard display name.",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Starter-kit root to package (default: this kit).",
    )
    parser.add_argument(
        "--entrypoint",
        default=DEFAULT_ENTRYPOINT,
        help=f"Python file launched by the organizer (default: {DEFAULT_ENTRYPOINT}).",
    )
    parser.add_argument(
        "--entrypoint-arg",
        action="append",
        default=[],
        dest="entrypoint_args",
        help="Extra argument appended to runtime.entrypoint. Repeatable.",
    )
    parser.add_argument(
        "--lockfile",
        default=DEFAULT_LOCKFILE_SOURCE,
        help=f"Hash-locked requirements file to copy (default: {DEFAULT_LOCKFILE_SOURCE}).",
    )
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        dest="include",
        help="Additional source-relative file to copy. Repeatable.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Directory for gauntlet-submission/ plus the tarball (default: <source>/dist).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing gauntlet-submission directory.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.check is not None:
            manifest = validate_submission_directory(args.check)
            print(
                f"PASS: {manifest['metadata']['agentId']} "
                f"({len(manifest['artifact']['files'])} files)"
            )
            return 0

        agent_id = (args.agent_id or "").strip()
        agent_name = (args.agent_name or "").strip()
        if not agent_id or not agent_name:
            raise PackagingError(
                "pass --agent-id and --agent-name; do not invent an identity"
            )
        source = (args.source or kit_root()).resolve()
        output = (args.output or (source / "dist")).resolve()
        result = package_submission(
            source,
            output,
            agent_id=agent_id,
            agent_name=agent_name,
            entrypoint=args.entrypoint,
            lockfile_source=args.lockfile,
            extra_files=args.include,
            extra_entrypoint_args=args.entrypoint_args,
            force=args.force,
        )
        archive = result["archive"]
        checksum = result["checksum"]
        manifest = result["manifest"]
        assert isinstance(archive, Path)
        assert isinstance(checksum, Path)
        assert isinstance(manifest, dict)
        print(f"PASS: wrote {result['directory']}")
        print(f"Archive: {archive}")
        print(f"SHA-256: {checksum.read_text(encoding='utf-8').split()[0]}")
        print(f"Checksum file: {checksum}")
        print(
            "Publish these two files to a new public GitHub repository "
            "(see docs/submitting.md)."
        )
        print(f"Entrypoint: {' '.join(manifest['runtime']['entrypoint'])}")
        print(f"Files: {len(manifest['artifact']['files'])}")
        return 0
    except PackagingError as exc:
        print(f"REJECTED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
