"""Sync the version and release date in CITATION.cff with src/pyoptex/__init__.py.

date-released is the release date: the date of the git tag v<version> if it exists, otherwise today's date
(the release is being prepared). --date YYYY-MM-DD overrides both. --check only reports mismatches.
"""

import argparse
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

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


def tag_date(version: str) -> str | None:
    """Date (YYYY-MM-DD) of the commit tagged v<version>, or None if there is no such tag."""
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%ad", "--date=short", f"refs/tags/v{version}", "--"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:  # git is not installed
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


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
        help="release date YYYY-MM-DD, instead of the date of the tag v<version> or today",
    )
    args = parser.parse_args()

    version = read_version()
    release_date = args.date.isoformat() if args.date is not None else tag_date(version)
    source = "--date" if args.date is not None else f"tag v{version}"

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
        if release_date is None:
            print(f"CITATION.cff version is in sync ({version}); no tag v{version} yet, date-released not checked")
        else:
            print(f"CITATION.cff is in sync (version {version}, date-released {release_date})")
        return

    if release_date is None:
        release_date = date.today().isoformat()
        print(f"Note: no tag v{version} yet, so date-released is today ({release_date}); pass --date to set it")

    if cff_version == version and cff_date == release_date:
        print(f"CITATION.cff already at version {version}, date-released {release_date}")
        return

    update_citation(text, version, release_date)
    print(f"Updated CITATION.cff: version {cff_version} -> {version}, date-released {cff_date} -> {release_date}")


if __name__ == "__main__":
    main()
