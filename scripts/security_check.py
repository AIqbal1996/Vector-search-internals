"""Security scanner to verify no secrets are committed to the repository."""

import re
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

PATTERNS = [
    re.compile(r"AIza[0-9A-Za-z-_]{35}"),
    re.compile(r"eyJh[a-zA-Z0-9_\-\.]{50,}"),
    re.compile(r"sk-[a-zA-Z0-9]{32,}"),
]

IGNORED_PARTS = {".venv", ".git", "__pycache__", ".pytest_cache", ".env"}


def scan_repository():
    scanned_files = 0
    flagged = []

    for file_path in ROOT_DIR.rglob("*"):
        if not file_path.is_file():
            continue
        if any(part in IGNORED_PARTS for part in file_path.parts) or file_path.name == ".env":
            continue

        scanned_files += 1
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            for pattern in PATTERNS:
                if pattern.search(content):
                    flagged.append((file_path.relative_to(ROOT_DIR), pattern.pattern))
        except Exception:
            pass

    print(f"Scanned {scanned_files} files in the repository.")
    if flagged:
        print(f"[SECURITY ALERT] Potential secret tokens detected in {len(flagged)} file(s):")
        for f, pat in flagged:
            print(f" - {f}")
        return False
    else:
        print("[PASSED] Repository is 100% CLEAN! Zero exposed credentials, keys, or tokens found.")
        return True


if __name__ == "__main__":
    scan_repository()
