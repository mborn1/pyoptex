"""Sync the version and release date in CITATION.cff with src/pyoptex/__init__.py.

date-released is the release date. Once the tag v<version> exists, its date is authoritative (the tag date of
an annotated tag, the commit date of a lightweight tag), and a --date that contradicts it is an error.
Before the tag exists (the release is being prepared), date-released is --date YYYY-MM-DD if given,
otherwise today's date.

Without git the tag cannot be looked up: update mode then requires --date, and --check checks only the
version. --check only reports mismatches, with the same date as update mode would write.
"""

import argparse
import re
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import NoReturn

ROOT = Path(__file__).resolve().parent.parent
INIT = ROOT / "src" / "pyoptex" / "__init__.py"
CFF = ROOT / "CITATION.cff"


def read_version() -> str:
    text = INIT.read_text(encoding="utf-8")
    match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', text)
    if not match:
        print(f"ERROR: could not find __version__ in {INIT}", file=sys.stderr)
        sys.exit(1)
    return match.group(1)


def read_field(text: str, key: str) -> str | None:
    match = re.search(rf'^{key}:\s*"([^"]+)"', text, re.MULTILINE)
    return match.group(1) if match else None


class GitUnavailableError(Exception):
    """git is not installed, or cannot read the repository, so the tags cannot be looked up."""


def error(message: str) -> NoReturn:
    print(f"ERROR: {message}", file=sys.stderr)
    sys.exit(1)


def tag_date(version: str) -> str | None:
    """Date (YYYY-MM-DD) of the tag v<version>, or None if there is no such tag."""
    try:
        result = subprocess.run(
            ["git", "for-each-ref", "--format=%(creatordate:short)", f"refs/tags/v{version}"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as e:
        raise GitUnavailableError("git not found") from e
    if result.returncode != 0:
        raise GitUnavailableError(f"git failed: {result.stderr.strip()}")
    lines = result.stdout.split()
    return lines[0] if lines else None


def update_citation(text: str, version: str, release_date: str) -> None:
    new_text = re.sub(r'^version:\s*".*"', f'version: "{version}"', text, flags=re.MULTILINE)
    new_text = re.sub(r'^date-released:\s*".*"', f'date-released: "{release_date}"', new_text, flags=re.MULTILINE)
    CFF.write_text(new_text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync the version and release date in CITATION.cff.")
    parser.add_argument("--check", action="store_true", help="only report mismatches (exit code 1 if any)")
    parser.add_argument(
        "--date",
        type=date.fromisoformat,
        help="release date YYYY-MM-DD, for a release without a tag v<version> yet (default: today)",
    )
    args = parser.parse_args()

    version = read_version()
    given_date = args.date.isoformat() if args.date is not None else None
    git_problem = None
    try:
        tagged_date = tag_date(version)
    except GitUnavailableError as e:
        tagged_date, git_problem = None, str(e)
    if tagged_date is not None and given_date is not None and given_date != tagged_date:
        error(f"tag v{version} exists with date {tagged_date}; --date {given_date} contradicts it")
    release_date = tagged_date or given_date
    source = f"tag v{version}" if tagged_date is not None else "--date"

    text = CFF.read_text(encoding="utf-8")
    cff_version = read_field(text, "version")
    cff_date = read_field(text, "date-released")

    if args.check:
        errors = []
        if cff_version != version:
            errors.append(f"CITATION.cff version ({cff_version}) != __init__.py ({version})")
        if release_date is not None and cff_date != release_date:
            errors.append(f"CITATION.cff date-released ({cff_date}) != date of {source} ({release_date})")
        if errors:
            print("\n".join(errors))
            sys.exit(1)
        if release_date is not None:
            print(f"CITATION.cff is in sync (version {version}, date-released {release_date})")
        elif git_problem is not None:
            print(f"CITATION.cff version is in sync ({version}); {git_problem}, so date-released not checked")
        else:
            print(f"CITATION.cff version is in sync ({version}); no tag v{version} yet, date-released not checked")
        return

    if release_date is None:
        if git_problem is not None:
            error(f"{git_problem}; pass --date YYYY-MM-DD")
        release_date = date.today().isoformat()
        print(f"Note: no tag v{version} yet, so date-released is today ({release_date}); pass --date to set it")

    if cff_version == version and cff_date == release_date:
        print(f"CITATION.cff already at version {version}, date-released {release_date}")
        return

    update_citation(text, version, release_date)
    print(f"Updated CITATION.cff: version {cff_version} -> {version}, date-released {cff_date} -> {release_date}")


if __name__ == "__main__":
    main()
