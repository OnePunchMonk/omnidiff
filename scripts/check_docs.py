#!/usr/bin/env python3
"""Check that the documentation only names things that exist.

Scans README.md, packages/*/README.md, docs/reference/*.md and docs/guides/*.md for:

* repo paths (under packages/, configs/, docs/, scripts/, examples/) that must exist;
* relative markdown links, which must resolve from the file's directory;
* test references ``packages/.../test_x.py::test_name``, where the file and the function must exist;
* ``uv run diffusion-omni <cmd> [--flag ...]`` lines: the subcommand must be registered, and every long
  flag named on the line must appear in ``uv run diffusion-omni <cmd> --help``.

Paths under ``runs/`` are outputs of commands and are not checked. Exit status is 1 when anything fails.

    uv run python scripts/check_docs.py
"""

from __future__ import annotations

import re
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PATH_ROOTS = ("packages", "configs", "docs", "scripts", "examples")

PATH_RE = re.compile(r"(?<![\w./-])((?:%s)/[A-Za-z0-9_./*-]+)" % "|".join(PATH_ROOTS))
LINK_RE = re.compile(r"\]\(([^)\s]+)\)")
TEST_RE = re.compile(r"((?:packages/|tests/)[\w./-]+?\.py)::([A-Za-z_]\w*)")
CMD_RE = re.compile(r"uv run diffusion-omni\s+([a-z][a-z-]*)([^`\n]*)")
BARE_TEST_RE = re.compile(r"`(test_\w+)`")
FLAG_RE = re.compile(r"(?<![\w-])(--[a-z][a-z0-9-]*)")


@lru_cache(maxsize=None)
def test_names() -> frozenset[str]:
    names = set()
    for f in (ROOT / "packages").glob("*/tests/**/*.py"):
        names.update(re.findall(r"^\s*def (test_\w+)", f.read_text(), re.M))
    return frozenset(names)


def doc_files() -> list[Path]:
    files = [ROOT / "README.md"]
    files += sorted((ROOT / "packages").glob("*/README.md"))
    files += sorted((ROOT / "docs" / "reference").glob("*.md"))
    files += sorted((ROOT / "docs" / "guides").glob("*.md"))
    return [f for f in files if f.exists()]


def logical_lines(text: str) -> list[str]:
    """Join shell line continuations so a command split over lines is checked as one."""
    out, buf = [], ""
    for line in text.splitlines():
        if line.rstrip().endswith("\\"):
            buf += line.rstrip()[:-1] + " "
        else:
            out.append(buf + line)
            buf = ""
    if buf:
        out.append(buf)
    return out


@lru_cache(maxsize=None)
def diffusion_omni_help(args: tuple[str, ...]) -> str | None:
    """Run ``uv run diffusion-omni <args> --help``. None when it fails."""
    proc = subprocess.run(["uv", "run", "diffusion-omni", *args, "--help"], cwd=ROOT, capture_output=True, text=True)
    return proc.stdout + proc.stderr if proc.returncode == 0 else None


def registered_commands() -> set[str]:
    help_text = diffusion_omni_help(())
    if help_text is None:
        return set()
    m = re.search(r"\{([a-z,\-]+)\}", help_text)
    return set(m.group(1).split(",")) if m else set()


def check_file(path: Path, commands: set[str], failures: list[str], counts: dict[str, int]) -> None:
    rel = path.relative_to(ROOT)
    text = path.read_text()

    for m in PATH_RE.finditer(text):
        target = m.group(1).rstrip(".,;:)")
        if "*" in target or target.startswith("runs/"):
            continue
        counts["paths"] += 1
        if not (ROOT / target).exists():
            failures.append(f"{rel}: path does not exist: {target}")

    for m in LINK_RE.finditer(text):
        link = m.group(1)
        if link.startswith(("http://", "https://", "mailto:", "#")):
            continue
        target = link.split("#", 1)[0]
        if not target:
            continue
        counts["links"] += 1
        if not (path.parent / target).resolve().exists():
            failures.append(f"{rel}: link target does not exist: {link}")

    for m in TEST_RE.finditer(text):
        file_part, name = m.group(1), m.group(2)
        counts["tests"] += 1
        test_file = next((c for c in (ROOT / file_part, path.parent / file_part) if c.exists()), None)
        if test_file is None:
            failures.append(f"{rel}: test file does not exist: {file_part}")
        elif not re.search(rf"^\s*def {re.escape(name)}\b", test_file.read_text(), re.M):
            failures.append(f"{rel}: no test function {name} in {file_part}")

    # Bare test names in backticks (e.g. in a table): must be defined in some test file.
    for m in BARE_TEST_RE.finditer(text):
        name = m.group(1)
        counts["tests"] += 1
        if name not in test_names():
            failures.append(f"{rel}: no test function named {name} under packages/*/tests")

    for line in logical_lines(text):
        for m in CMD_RE.finditer(line):
            cmd, rest = m.group(1), m.group(2)
            counts["commands"] += 1
            if cmd not in commands:
                failures.append(f"{rel}: unknown diffusion-omni command '{cmd}': {line.strip()[:120]}")
                continue
            help_text = diffusion_omni_help((cmd,)) or ""
            for flag in FLAG_RE.findall(rest):
                if flag not in help_text:
                    failures.append(f"{rel}: 'diffusion-omni {cmd}' has no flag {flag}: {line.strip()[:120]}")


def main() -> int:
    failures: list[str] = []
    counts = {"paths": 0, "links": 0, "tests": 0, "commands": 0}
    commands = registered_commands()
    if not commands:
        print("could not read the diffusion-omni command list from `uv run diffusion-omni --help`")
        return 1
    files = doc_files()
    for path in files:
        check_file(path, commands, failures, counts)
    print(f"checked {len(files)} files: {counts['paths']} paths, {counts['links']} links, "
          f"{counts['tests']} test refs, {counts['commands']} diffusion-omni commands")
    for f in failures:
        print("FAIL", f)
    if failures:
        print(f"{len(failures)} problem(s)")
        return 1
    print("ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
