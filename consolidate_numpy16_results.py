#!/usr/bin/env python3
"""Consolidate the NumPy-only 16x16 grid-search results."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path


BASE = Path(__file__).resolve().parent
OUT = BASE / "grid_search_results_numpy16"
SOURCES = ["grid_outputs_numpy16_30", "grid_outputs_numpy16_highlr_30"]


def read_summary(source: Path) -> dict:
    return json.loads((source / "grid_summary.json").read_text(encoding="utf-8"))


def key(row: dict[str, str]) -> tuple[int, float, float, float]:
    return (
        int(row["hidden_dim"]),
        float(row["learning_rate"]),
        float(row["lr_decay"]),
        float(row["weight_decay"]),
    )


def load_rows() -> list[dict[str, str]]:
    rows = []
    seen = set()
    for source_name in SOURCES:
        source = BASE / source_name
        with (source / "grid_results.csv").open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                row_key = key(row)
                if row_key in seen:
                    continue
                seen.add(row_key)
                rows.append(
                    {
                        "run_id": "0",
                        "source_dir": source_name,
                        "source_run_id": row["run_id"],
                        "hidden_dim": row["hidden_dim"],
                        "learning_rate": row["learning_rate"],
                        "lr_decay": row["lr_decay"],
                        "weight_decay": row["weight_decay"],
                        "best_epoch": row["best_epoch"],
                        "best_val_accuracy": row["best_val_accuracy"],
                        "final_train_accuracy": row["final_train_accuracy"],
                        "final_val_accuracy": row["final_val_accuracy"],
                        "elapsed_seconds": row["elapsed_seconds"],
                    }
                )
    rows.sort(key=lambda r: (int(r["hidden_dim"]), float(r["learning_rate"]), float(r["lr_decay"]), float(r["weight_decay"])))
    for idx, row in enumerate(rows, start=1):
        row["run_id"] = str(idx)
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fields = [
        "run_id",
        "source_dir",
        "source_run_id",
        "hidden_dim",
        "learning_rate",
        "lr_decay",
        "weight_decay",
        "best_epoch",
        "best_val_accuracy",
        "final_train_accuracy",
        "final_val_accuracy",
        "elapsed_seconds",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    best = max(rows, key=lambda r: float(r["best_val_accuracy"]))
    best_source = BASE / best["source_dir"]
    best_summary = read_summary(best_source)

    write_csv(OUT / "grid_results.csv", rows)
    write_csv(OUT / "best_row.csv", [best])

    for name in [
        "best_model.npz",
        "confusion_matrix.csv",
        "confusion_matrix.png",
        "error_examples.png",
        "learning_curves.png",
        "report.md",
        "report.pdf",
    ]:
        shutil.copy2(best_source / name, OUT / name)

    summary = {
        "image_size": 16,
        "grid_size": len(rows),
        "source_dirs": SOURCES,
        "hidden_dims": sorted({int(row["hidden_dim"]) for row in rows}),
        "learning_rates": sorted({float(row["learning_rate"]) for row in rows}),
        "lr_decays": sorted({float(row["lr_decay"]) for row in rows}),
        "weight_decays": sorted({float(row["weight_decay"]) for row in rows}),
        "best_row": best,
        "best_config": {
            key: value
            for key, value in best_summary["best_config"].items()
            if key != "backend" and not key.endswith("_id")
        },
        "best_epoch": best_summary["best_epoch"],
        "best_val_accuracy": best_summary["best_val_accuracy"],
        "test_loss": best_summary["test_loss"],
        "test_accuracy": best_summary["test_accuracy"],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    readme = f"""# NumPy-only 16x16 Grid Search Results

This directory contains the final strict-NumPy result.  Only `image_size=16`
is used; no 32x32 or 64x64 experiments are included in the final report.

## Grid

- image_size: 16
- hidden_dim: {summary['hidden_dims']}
- learning_rate: {summary['learning_rates']}
- lr_decay: {summary['lr_decays']}
- weight_decay: {summary['weight_decays']}
- epochs: 30
- batch_size: 256

## Best Result

- hidden_dim: {best['hidden_dim']}
- learning_rate: {best['learning_rate']}
- lr_decay: {best['lr_decay']}
- weight_decay: {best['weight_decay']}
- best_epoch: {best['best_epoch']}
- best_val_accuracy: {float(best['best_val_accuracy']):.4f}
- test_accuracy: {summary['test_accuracy']:.4f}
- test_loss: {summary['test_loss']:.4f}
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    print(f"Wrote {len(rows)} rows to {OUT}")


if __name__ == "__main__":
    main()
