"""claude-do: give Claude a file, and it does what the file says."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .core import MODEL, Workspace, run_task

__version__ = "0.1.0"
__all__ = ["main", "run_task", "Workspace"]


def _ask(command: str) -> bool:
    return input(f"run `{command}`? [y/N] ").strip().lower() in {"y", "yes"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="claude-do", description=__doc__)
    parser.add_argument("file", type=Path, help="task file (any text or markdown)")
    parser.add_argument("--dir", type=Path, default=Path("."), help="working directory Claude may touch")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("-y", "--yes", action="store_true", help="run shell commands without asking")
    args = parser.parse_args(argv)

    if not args.file.is_file():
        parser.error(f"{args.file} not found")

    workspace = Workspace(args.dir, approve=(lambda _: True) if args.yes else _ask)
    summary = run_task(args.file.read_text(), workspace, model=args.model)
    print(f"\n{summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
