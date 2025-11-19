#!/usr/bin/env python3
"""Utility to record and verify tokenizer artifact checksums."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Optional


def compute_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def dump_metadata(path: Path, sha256: str, output: Optional[Path]) -> None:
    if not output:
        return
    payload = {
        "tokenizer_path": str(path),
        "sha256": sha256,
    }
    output.write_text(json.dumps(payload, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        required=True,
        help="Path to the SentencePiece tokenizer model (.model).",
    )
    parser.add_argument(
        "--expected-sha256",
        type=str,
        default=None,
        help="Optional expected checksum; if provided and mismatch occurs, exits with non-zero status.",
    )
    parser.add_argument(
        "--metadata-json",
        type=Path,
        default=None,
        help="Optional path to write checksum metadata as JSON.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only emit errors; suppress the default stdout line.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer_path = args.tokenizer_path
    if not tokenizer_path.exists():
        raise SystemExit(f"Tokenizer file not found: {tokenizer_path}")

    sha256 = compute_sha256(tokenizer_path)
    if not args.quiet:
        print(f"{sha256}  {tokenizer_path}")

    dump_metadata(tokenizer_path, sha256, args.metadata_json)

    expected = args.expected_sha256
    if expected and expected.lower() != sha256.lower():
        raise SystemExit(
            f"Checksum mismatch for {tokenizer_path} "
            f"(expected {expected}, got {sha256})"
        )


if __name__ == "__main__":
    main()
