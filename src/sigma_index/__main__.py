"""CLI entry point for sigma-index.

Usage:
    python -m sigma_index index ./src
    python -m sigma_index search "def compress"
    python -m sigma_index search --mode exact "class"
"""

from __future__ import annotations

import argparse
import sys

from sigma_index import SigmaIndex

_INDEX_FILE = ".sigma_index.pkl"


def cmd_index(args: argparse.Namespace) -> None:
    idx = SigmaIndex()
    count = idx.index_directory(args.path, glob_pattern=args.glob)
    idx.save(_INDEX_FILE)
    print(f"Indexed {count} files -> {_INDEX_FILE}")


def cmd_search(args: argparse.Namespace) -> None:
    idx = SigmaIndex()
    try:
        idx.load(_INDEX_FILE)
    except FileNotFoundError:
        print(f"No index found at {_INDEX_FILE}. Run: python -m sigma_index index <path>",
              file=sys.stderr)
        sys.exit(1)

    results = idx.search(args.query, mode=args.mode)
    if not results:
        print("No results found.")
        return
    for i, r in enumerate(results[:10], 1):
        print(f"{i:2}. [{r.mode}] {r.file}:{r.line}  score={r.score:.3f}")
        if r.snippet:
            for line in r.snippet.splitlines()[:3]:
                print(f"      {line}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="sigma_index", description="Sigma code search engine")
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="Index a directory")
    p_index.add_argument("path", help="Directory to index")
    p_index.add_argument("--glob", default="**/*.py", help="File glob pattern")

    p_search = sub.add_parser("search", help="Search the index")
    p_search.add_argument("query", help="Search query")
    p_search.add_argument("--mode", default="hybrid",
                          choices=["exact", "semantic", "hybrid"])

    args = parser.parse_args()
    if args.command == "index":
        cmd_index(args)
    elif args.command == "search":
        cmd_search(args)


if __name__ == "__main__":
    main()
