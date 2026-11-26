#!/usr/bin/env python3
"""Fail if a tracked file looks like it holds a credential.

Run from anywhere inside the repository (CI runs it on every push):

    python3 scripts/check_secrets.py

It scans every file git tracks for the key formats this project's providers
issue, private-key blocks and committed environment files. Test fixtures use
deliberately fake keys that contain the word SENTINEL; those are allowed. No
dependency, no network: a scanner that needs installing is a scanner that
gets skipped.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

PATTERNS: dict[str, re.Pattern[str]] = {
    "Google / Gemini API key": re.compile(r"AIza[0-9A-Za-z_\-]{35}"),
    "NVIDIA API key": re.compile(r"nvapi-[0-9A-Za-z_\-]{20,}"),
    "Anthropic API key": re.compile(r"sk-ant-[0-9A-Za-z_\-]{20,}"),
    "OpenAI-style secret key": re.compile(r"\bsk-(?:proj-)?[0-9A-Za-z]{32,}"),
    "AWS access key id": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "GitHub token": re.compile(r"\bgh[pousr]_[0-9A-Za-z]{36,}\b"),
    "private key block": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
}

#: Fake credentials in tests say so in the value itself.
ALLOWED_MARKER = "SENTINEL"

#: Files that must never be tracked, whatever they contain.
FORBIDDEN_NAMES = re.compile(r"(^|/)\.env(\.[^/]*)?$|\.zip$|(^|/)\.mcp\.json$")
ALLOWED_NAMES = re.compile(r"\.example$")


def tracked_files(root: Path) -> list[str]:
    output = subprocess.run(
        ["git", "ls-files", "-z"], cwd=root, check=True, capture_output=True
    ).stdout
    return [name for name in output.decode().split("\0") if name]


def main() -> int:
    root = Path(
        subprocess.run(
            ["git", "rev-parse", "--show-toplevel"], check=True, capture_output=True, text=True
        ).stdout.strip()
    )
    problems: list[str] = []
    for name in tracked_files(root):
        if FORBIDDEN_NAMES.search(name) and not ALLOWED_NAMES.search(name):
            problems.append(f"{name}: this kind of file must not be committed")
            continue
        path = root / name
        if not path.is_file():
            continue
        data = path.read_bytes()
        if b"\0" in data[:8192]:
            continue  # binary
        text = data.decode("utf-8", errors="ignore")
        for label, pattern in PATTERNS.items():
            for match in pattern.finditer(text):
                if ALLOWED_MARKER in match.group(0):
                    continue
                line = text.count("\n", 0, match.start()) + 1
                # The value itself is never printed.
                problems.append(f"{name}:{line}: looks like a {label}")
    if problems:
        print("Possible secrets in tracked files:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        print(
            "Remove the value, rotate the credential if it was real, and keep "
            "secrets in environment variables.",
            file=sys.stderr,
        )
        return 1
    print("No secrets found in tracked files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
