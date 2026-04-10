"""Sync the version in CITATION.cff with src/pyoptex/__init__.py."""

import re
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


def update_citation(version: str) -> bool:
    text = CFF.read_text(encoding="utf-8")
    today = date.today().isoformat()

    new_text = re.sub(r'^version:\s*".*"', f'version: "{version}"', text, flags=re.MULTILINE)
    new_text = re.sub(r'^date-released:\s*".*"', f'date-released: "{today}"', new_text, flags=re.MULTILINE)

    if new_text == text:
        return False

    CFF.write_text(new_text, encoding="utf-8")
    return True


def main() -> None:
    check_only = "--check" in sys.argv
    version = read_version()

    cff_text = CFF.read_text(encoding="utf-8")
    cff_version_match = re.search(r'^version:\s*"([^"]+)"', cff_text, re.MULTILINE)
    cff_version = cff_version_match.group(1) if cff_version_match else None

    if check_only:
        if cff_version != version:
            print(f"CITATION.cff version ({cff_version}) != __init__.py ({version})")
            sys.exit(1)
        print(f"CITATION.cff version is in sync ({version})")
        return

    if cff_version == version:
        print(f"CITATION.cff already at version {version}")
        return

    update_citation(version)
    print(f"Updated CITATION.cff: {cff_version} -> {version}")


if __name__ == "__main__":
    main()
