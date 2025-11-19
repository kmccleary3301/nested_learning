#!/usr/bin/env python
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Optional

import typer


app = typer.Typer(add_completion=False, help="Validate mixture manifests and shard inventories.")


def _dir_stats(path: Path, sample_limit: int = 2000) -> tuple[dict[str, float], set[str]]:
    total_bytes = 0
    file_count = 0
    sampled_names: set[str] = set()
    for entry in sorted(path.rglob("*.npy")):
        total_bytes += entry.stat().st_size
        file_count += 1
        if len(sampled_names) < sample_limit:
            sampled_names.add(f"{path.name}/{entry.relative_to(path).as_posix()}")
    return {"files": file_count, "bytes": total_bytes}, sampled_names


@app.command()
def main(
    manifest: Path = typer.Option(..., help="Path to data/manifest/*.json file."),
    output: Optional[Path] = typer.Option(None, help="Optional JSON output path for the report."),
    overlap_threshold: float = typer.Option(0.05, help="Warn when filename overlap exceeds this Jaccard."),
) -> None:
    spec = json.loads(manifest.read_text())
    report = {"manifest": spec.get("name"), "sources": []}
    sampled_sets: dict[str, set[str]] = {}
    for entry in spec.get("sources", []):
        shards_dir = Path(entry["shards_dir"])
        source_report = dict(entry)
        source_report["exists"] = shards_dir.exists()
        if shards_dir.exists():
            stats, sampled = _dir_stats(shards_dir)
            source_report.update(stats)
            sampled_sets[entry["name"]] = sampled
        stats_file = entry.get("stats_file")
        if stats_file and Path(stats_file).exists():
            try:
                stats_payload = json.loads(Path(stats_file).read_text())
                source_report["stats_snapshot"] = stats_payload.get(entry["name"])
            except json.JSONDecodeError:
                source_report["stats_snapshot"] = "unreadable"
        report["sources"].append(source_report)
    overlaps = []
    for a, b in combinations(sampled_sets.keys(), 2):
        set_a = sampled_sets[a]
        set_b = sampled_sets[b]
        if not set_a or not set_b:
            continue
        jaccard = len(set_a & set_b) / len(set_a | set_b)
        entry = {"pair": [a, b], "jaccard": jaccard}
        if jaccard >= overlap_threshold:
            entry["warning"] = True
        overlaps.append(entry)
    report["filename_overlap"] = overlaps
    summary = json.dumps(report, indent=2)
    typer.echo(summary)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(summary)


if __name__ == "__main__":
    app()
