from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CLASSES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="grid_search_results/best_image_size_16/best_model.npz")
    parser.add_argument("--output-dir", default="grid_search_results/best_image_size_16")
    return parser.parse_args()


def infer_image_size(w1: np.ndarray) -> int:
    pixels = w1.shape[0] // 3
    image_size = int(round(math.sqrt(pixels)))
    if image_size * image_size * 3 != w1.shape[0]:
        raise ValueError(f"Cannot infer square RGB image size from W1 shape {w1.shape}")
    return image_size


def normalize_filter(weights: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(weights, [1, 99])
    if hi <= lo:
        return np.full_like(weights, 0.5, dtype=np.float32)
    return np.clip((weights - lo) / (hi - lo), 0, 1).astype(np.float32)


def reshape_filter(vector: np.ndarray, image_size: int) -> np.ndarray:
    return vector.reshape(image_size, image_size, 3)


def plot_filter_grid(w1: np.ndarray, image_size: int, output_dir: Path) -> None:
    norms = np.linalg.norm(w1, axis=0)
    unit_ids = np.argsort(norms)[-36:][::-1]
    fig, axes = plt.subplots(6, 6, figsize=(8.5, 8.5))
    for ax, unit_id in zip(axes.flat, unit_ids):
        image = normalize_filter(reshape_filter(w1[:, unit_id], image_size))
        ax.imshow(image)
        ax.set_title(f"h{unit_id}", fontsize=7)
        ax.axis("off")
    fig.suptitle("First-layer hidden-unit weights with largest L2 norms", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_dir / "first_layer_weight_grid.png", dpi=220)
    plt.close(fig)


def plot_class_related_filters(w1: np.ndarray, w2: np.ndarray, image_size: int, output_dir: Path) -> None:
    chosen = ["Forest", "River"]
    fig, axes = plt.subplots(len(chosen), 8, figsize=(12, 3.2))
    for row, class_name in enumerate(chosen):
        class_idx = CLASSES.index(class_name)
        unit_ids = np.argsort(w2[:, class_idx])[-8:][::-1]
        for col, unit_id in enumerate(unit_ids):
            image = normalize_filter(reshape_filter(w1[:, unit_id], image_size))
            ax = axes[row, col]
            ax.imshow(image)
            ax.set_title(f"{class_name}\nh{unit_id}", fontsize=7)
            ax.axis("off")
    fig.suptitle("Hidden units with strongest positive outgoing weights for Forest and River", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_dir / "forest_river_related_weights.png", dpi=220)
    plt.close(fig)


def plot_all_class_related_filters(w1: np.ndarray, w2: np.ndarray, image_size: int, output_dir: Path) -> None:
    fig, axes = plt.subplots(len(CLASSES), 6, figsize=(9, 14))
    for row, class_name in enumerate(CLASSES):
        class_idx = CLASSES.index(class_name)
        unit_ids = np.argsort(w2[:, class_idx])[-6:][::-1]
        for col, unit_id in enumerate(unit_ids):
            image = normalize_filter(reshape_filter(w1[:, unit_id], image_size))
            ax = axes[row, col]
            ax.imshow(image)
            ax.set_title(f"{class_name}\nh{unit_id}", fontsize=6)
            ax.axis("off")
    fig.suptitle("Top first-layer patterns connected to each class output", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_dir / "all_class_related_weights.png", dpi=220)
    plt.close(fig)


def plot_class_templates(w1: np.ndarray, w2: np.ndarray, image_size: int, output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for class_idx, class_name in enumerate(CLASSES):
        # This is a linearized class template: it ignores the ReLU gate, but is
        # useful for seeing which colors/spatial regions push the class logit up.
        template = w1 @ w2[:, class_idx]
        image = normalize_filter(reshape_filter(template, image_size))
        ax = axes.flat[class_idx]
        ax.imshow(image)
        ax.set_title(class_name, fontsize=8)
        ax.axis("off")
    fig.suptitle("Linearized class templates from W1 @ W2[:, class]", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_dir / "linearized_class_templates.png", dpi=220)
    plt.close(fig)


def write_weight_pattern_stats(w1: np.ndarray, w2: np.ndarray, image_size: int, output_dir: Path) -> None:
    channel_names = ["R", "G", "B"]
    rows = []
    for class_idx, class_name in enumerate(CLASSES):
        template = reshape_filter(w1 @ w2[:, class_idx], image_size)
        channel_mean = template.mean(axis=(0, 1))
        abs_map = np.abs(template).mean(axis=2)
        row_strength = abs_map.mean(axis=1)
        col_strength = abs_map.mean(axis=0)
        top_units = np.argsort(w2[:, class_idx])[-6:][::-1]

        rows.append(
            {
                "class": class_name,
                "mean_R": channel_mean[0],
                "mean_G": channel_mean[1],
                "mean_B": channel_mean[2],
                "dominant_channel": channel_names[int(np.argmax(channel_mean))],
                "abs_strength": abs_map.mean(),
                "row_peak": int(np.argmax(row_strength)),
                "col_peak": int(np.argmax(col_strength)),
                "top_positive_hidden_units": " ".join(str(int(i)) for i in top_units),
            }
        )

    with (output_dir / "weight_pattern_stats.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = np.load(model_path)
    w1 = model["w1"]
    w2 = model["w2"]
    image_size = infer_image_size(w1)

    plot_filter_grid(w1, image_size, output_dir)
    plot_class_related_filters(w1, w2, image_size, output_dir)
    plot_all_class_related_filters(w1, w2, image_size, output_dir)
    plot_class_templates(w1, w2, image_size, output_dir)
    write_weight_pattern_stats(w1, w2, image_size, output_dir)
    print(f"Wrote weight visualizations to {output_dir}")


if __name__ == "__main__":
    main()
