from __future__ import annotations

import argparse
import csv
import itertools
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from solution import (
    CLASSES,
    Config,
    MLP,
    confusion_matrix,
    load_dataset,
    loss_and_accuracy,
    normalize_from_train,
    plot_confusion,
    plot_error_examples,
    plot_history,
    split_indices,
    train_epoch,
    write_pdf_report,
    write_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="EuroSAT_RGB")
    parser.add_argument("--output-dir", default="grid_outputs_numpy16")
    parser.add_argument("--image-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 512])
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[0.005, 0.01, 0.02])
    parser.add_argument("--lr-decays", type=float, nargs="+", default=[0.95, 0.97])
    parser.add_argument("--weight-decays", type=float, nargs="+", default=[0.0, 1e-4])
    parser.add_argument("--activation", default="relu", choices=["relu", "tanh", "sigmoid"])
    return parser.parse_args()


def make_base_config(args: argparse.Namespace) -> Config:
    return Config(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        image_size=args.image_size,
        hidden_dim=args.hidden_dims[0],
        activation=args.activation,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rates[0],
        lr_decay=args.lr_decays[0],
        weight_decay=args.weight_decays[0],
        seed=args.seed,
    )


def run_one_config(
    cfg: Config,
    x_train,
    y_train,
    x_val,
    y_val,
    rng_seed: int,
) -> tuple[dict[str, list[float]], dict[str, np.ndarray], float, int]:
    model = MLP(x_train.shape[1], cfg.hidden_dim, len(CLASSES), cfg.activation, cfg.seed)
    rng = np.random.default_rng(rng_seed)
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "lr": []}
    best_state = model.state_dict()
    best_val_acc = -1.0
    best_epoch = 0

    for epoch in range(1, cfg.epochs + 1):
        lr = cfg.learning_rate * (cfg.lr_decay ** (epoch - 1))
        train_epoch(model, x_train, y_train, cfg, lr, rng)
        train_loss, train_acc = loss_and_accuracy(model, x_train, y_train, cfg.weight_decay)
        val_loss, val_acc = loss_and_accuracy(model, x_val, y_val, cfg.weight_decay)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(lr)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()
            best_epoch = epoch

    return history, best_state, best_val_acc, best_epoch


def main() -> None:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = base_dir / data_dir
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = base_dir / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = make_base_config(args)
    start_time = time.time()

    print(f"Loading {args.image_size}x{args.image_size} images from {data_dir} ...", flush=True)
    x, y, paths = load_dataset(data_dir, args.image_size)
    train_idx, val_idx, test_idx = split_indices(y, cfg)
    x_train, x_val, x_test, mean, std = normalize_from_train(x[train_idx], x[val_idx], x[test_idx])
    y_train_np, y_val_np, y_test_np = y[train_idx], y[val_idx], y[test_idx]
    y_train, y_val, y_test = y_train_np, y_val_np, y_test_np

    grid = list(
        itertools.product(args.hidden_dims, args.learning_rates, args.lr_decays, args.weight_decays)
    )
    print(f"Running {len(grid)} configs with NumPy ...", flush=True)

    rows: list[dict[str, float | int | str]] = []
    best: dict[str, object] | None = None

    for run_id, (hidden_dim, learning_rate, lr_decay, weight_decay) in enumerate(grid, start=1):
        run_cfg = replace(
            cfg,
            hidden_dim=hidden_dim,
            learning_rate=learning_rate,
            lr_decay=lr_decay,
            weight_decay=weight_decay,
        )
        run_start = time.time()
        history, state, best_val_acc, best_epoch = run_one_config(
            run_cfg, x_train, y_train, x_val, y_val, cfg.seed + run_id
        )
        elapsed = time.time() - run_start
        row = {
            "run_id": run_id,
            "hidden_dim": hidden_dim,
            "learning_rate": learning_rate,
            "lr_decay": lr_decay,
            "weight_decay": weight_decay,
            "best_epoch": best_epoch,
            "best_val_accuracy": best_val_acc,
            "final_train_accuracy": history["train_acc"][-1],
            "final_val_accuracy": history["val_acc"][-1],
            "elapsed_seconds": elapsed,
        }
        rows.append(row)
        print(
            f"[{run_id:02d}/{len(grid)}] hidden={hidden_dim} lr={learning_rate} "
            f"decay={lr_decay} wd={weight_decay} best_val={best_val_acc:.4f} "
            f"epoch={best_epoch} time={elapsed:.1f}s",
            flush=True,
        )
        if best is None or best_val_acc > best["best_val_accuracy"]:
            best = {
                "config": run_cfg,
                "history": history,
                "state": state,
                "best_val_accuracy": best_val_acc,
                "best_epoch": best_epoch,
                "row": row,
            }

    assert best is not None
    results_csv = output_dir / "grid_results.csv"
    with results_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    best_cfg: Config = best["config"]  # type: ignore[assignment]
    best_model = MLP(x_train.shape[1], best_cfg.hidden_dim, len(CLASSES), best_cfg.activation, best_cfg.seed)
    best_model.load_state_dict(best["state"])  # type: ignore[arg-type]
    test_loss, test_acc = loss_and_accuracy(best_model, x_test, y_test, best_cfg.weight_decay)
    y_pred = best_model.predict(x_test)
    matrix = confusion_matrix(y_test_np, y_pred, len(CLASSES))

    np.savez(output_dir / "best_model.npz", **best["state"], mean=mean, std=std, classes=np.array(CLASSES))
    np.savetxt(output_dir / "confusion_matrix.csv", matrix, fmt="%d", delimiter=",")
    plot_history(best["history"], output_dir)  # type: ignore[arg-type]
    plot_confusion(matrix, output_dir)
    plot_error_examples(paths, y_test_np, y_pred, test_idx, output_dir)
    write_report(best_cfg, output_dir, best["history"], test_loss, test_acc, matrix)  # type: ignore[arg-type]
    write_pdf_report(best_cfg, output_dir, best["history"], test_loss, test_acc, matrix)  # type: ignore[arg-type]

    summary = {
        "grid_size": len(grid),
        "total_elapsed_seconds": time.time() - start_time,
        "best_config": asdict(best_cfg),
        "best_epoch": best["best_epoch"],
        "best_val_accuracy": best["best_val_accuracy"],
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "results_csv": str(results_csv),
    }
    (output_dir / "grid_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
