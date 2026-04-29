from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from PIL import Image


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


@dataclass
class Config:
    data_dir: str = "EuroSAT_RGB"
    output_dir: str = "outputs"
    image_size: int = 16
    hidden_dim: int = 512
    activation: str = "relu"
    epochs: int = 30
    batch_size: int = 256
    learning_rate: float = 0.16
    lr_decay: float = 0.90
    weight_decay: float = 5e-4
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    seed: int = 42


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    for field, value in asdict(Config()).items():
        arg_type = type(value)
        parser.add_argument(f"--{field.replace('_', '-')}", default=value, type=arg_type)
    args = parser.parse_args()
    return Config(**vars(args))


def one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    encoded = np.zeros((y.shape[0], num_classes), dtype=np.float32)
    encoded[np.arange(y.shape[0]), y] = 1.0
    return encoded


def load_dataset(data_dir: Path, image_size: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    xs: list[np.ndarray] = []
    ys: list[int] = []
    paths: list[str] = []

    for label, class_name in enumerate(CLASSES):
        class_dir = data_dir / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")
        for image_path in sorted(class_dir.glob("*.jpg")):
            image = Image.open(image_path).convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
            xs.append(np.asarray(image, dtype=np.float32).reshape(-1) / 255.0)
            ys.append(label)
            paths.append(str(image_path))

    return np.vstack(xs).astype(np.float32), np.asarray(ys, dtype=np.int64), paths


def split_indices(y: np.ndarray, cfg: Config) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []

    for label in np.unique(y):
        idx = np.flatnonzero(y == label)
        rng.shuffle(idx)
        n_train = int(len(idx) * cfg.train_ratio)
        n_val = int(len(idx) * cfg.val_ratio)
        train_parts.append(idx[:n_train])
        val_parts.append(idx[n_train : n_train + n_val])
        test_parts.append(idx[n_train + n_val :])

    train_idx = np.concatenate(train_parts)
    val_idx = np.concatenate(val_parts)
    test_idx = np.concatenate(test_parts)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


def normalize_from_train(
    x_train: np.ndarray, x_val: np.ndarray, x_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True) + 1e-6
    return (x_train - mean) / std, (x_val - mean) / std, (x_test - mean) / std, mean, std


class MLP:
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        activation: str,
        seed: int,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.activation = activation
        self.w1 = (rng.standard_normal((input_dim, hidden_dim)) * math.sqrt(2.0 / input_dim)).astype(np.float32)
        self.b1 = np.zeros((1, hidden_dim), dtype=np.float32)
        self.w2 = (rng.standard_normal((hidden_dim, output_dim)) * math.sqrt(2.0 / hidden_dim)).astype(np.float32)
        self.b2 = np.zeros((1, output_dim), dtype=np.float32)

    def _activate(self, z):
        if self.activation == "relu":
            return np.maximum(z, 0.0)
        if self.activation == "tanh":
            return np.tanh(z)
        if self.activation == "sigmoid":
            return 1.0 / (1.0 + np.exp(-z))
        raise ValueError(f"Unsupported activation: {self.activation}")

    def _activation_grad(self, z, a):
        if self.activation == "relu":
            return (z > 0).astype(np.float32)
        if self.activation == "tanh":
            return 1.0 - a * a
        if self.activation == "sigmoid":
            return a * (1.0 - a)
        raise ValueError(f"Unsupported activation: {self.activation}")

    def forward(self, x):
        z1 = x @ self.w1 + self.b1
        a1 = self._activate(z1)
        logits = a1 @ self.w2 + self.b2
        return z1, a1, logits

    def predict(self, x: np.ndarray, batch_size: int = 2048) -> np.ndarray:
        preds = []
        for start in range(0, len(x), batch_size):
            _, _, logits = self.forward(x[start : start + batch_size])
            preds.append(np.argmax(logits, axis=1))
        return np.concatenate(preds)

    def state_dict(self) -> dict[str, np.ndarray]:
        return {"w1": self.w1.copy(), "b1": self.b1.copy(), "w2": self.w2.copy(), "b2": self.b2.copy()}

    def load_state_dict(self, state: dict[str, np.ndarray]) -> None:
        self.w1 = state["w1"].copy()
        self.b1 = state["b1"].copy()
        self.w2 = state["w2"].copy()
        self.b2 = state["b2"].copy()


def loss_and_accuracy(model: MLP, x: np.ndarray, y: np.ndarray, weight_decay: float, batch_size: int = 2048) -> tuple[float, float]:
    total_loss = 0.0
    total_correct = 0
    for start in range(0, len(x), batch_size):
        xb = x[start : start + batch_size]
        yb = y[start : start + batch_size]
        _, _, logits = model.forward(xb)
        logits = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        total_loss += float(-np.log(probs[np.arange(len(yb)), yb] + 1e-12).sum())
        total_correct += int((np.argmax(probs, axis=1) == yb).sum())
    l2 = 0.5 * weight_decay * (np.sum(model.w1 * model.w1) + np.sum(model.w2 * model.w2))
    return total_loss / len(x) + float(l2), total_correct / len(x)


def train_epoch(model: MLP, x: np.ndarray, y: np.ndarray, cfg: Config, lr: float, rng: np.random.Generator) -> None:
    indices = rng.permutation(len(x))
    for start in range(0, len(indices), cfg.batch_size):
        batch_idx = indices[start : start + cfg.batch_size]
        xb = x[batch_idx]
        yb = y[batch_idx]
        z1, a1, logits = model.forward(xb)
        logits -= logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        probs[np.arange(len(yb)), yb] -= 1.0
        probs /= len(yb)

        dw2 = a1.T @ probs + cfg.weight_decay * model.w2
        db2 = probs.sum(axis=0, keepdims=True)
        da1 = probs @ model.w2.T
        dz1 = da1 * model._activation_grad(z1, a1)
        dw1 = xb.T @ dz1 + cfg.weight_decay * model.w1
        db1 = dz1.sum(axis=0, keepdims=True)

        model.w1 -= lr * dw1
        model.b1 -= lr * db1
        model.w2 -= lr * dw2
        model.b2 -= lr * db2


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        matrix[t, p] += 1
    return matrix


def plot_history(history: dict[str, list[float]], output_dir: Path) -> None:
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"], label="Validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(epochs, history["train_acc"], label="Train")
    axes[1].plot(epochs, history["val_acc"], label="Validation")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_dir / "learning_curves.png", dpi=180)
    plt.close(fig)


def plot_confusion(matrix: np.ndarray, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(np.arange(len(CLASSES)), labels=CLASSES, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(CLASSES)), labels=CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=180)
    plt.close(fig)


def plot_error_examples(paths: list[str], y_true: np.ndarray, y_pred: np.ndarray, test_idx: np.ndarray, output_dir: Path) -> None:
    wrong = np.flatnonzero(y_true != y_pred)[:16]
    if len(wrong) == 0:
        return
    cols = 4
    rows = math.ceil(len(wrong) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(10, 2.7 * rows))
    axes = np.asarray(axes).reshape(-1)
    for ax, local_idx in zip(axes, wrong):
        image_path = paths[test_idx[local_idx]]
        ax.imshow(Image.open(image_path).convert("RGB"))
        ax.set_title(f"T: {CLASSES[y_true[local_idx]]}\nP: {CLASSES[y_pred[local_idx]]}", fontsize=8)
        ax.axis("off")
    for ax in axes[len(wrong) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "error_examples.png", dpi=180)
    plt.close(fig)


def write_report(
    cfg: Config,
    output_dir: Path,
    history: dict[str, list[float]],
    test_loss: float,
    test_acc: float,
    matrix: np.ndarray,
) -> None:
    per_class = matrix.diagonal() / np.maximum(matrix.sum(axis=1), 1)
    hardest = np.argsort(per_class)[:3]
    best_epoch = int(np.argmax(history["val_acc"]) + 1)

    lines = [
        "# HW1 EuroSAT RGB MLP Report",
        "",
        "## Experimental Setup",
        "",
        f"- Model: one-hidden-layer MLP, hidden dimension = {cfg.hidden_dim}, activation = {cfg.activation}.",
        f"- Input: RGB images resized from 64x64 to {cfg.image_size}x{cfg.image_size}, flattened and standardized by the training split.",
        f"- Split: stratified {cfg.train_ratio:.0%}/{cfg.val_ratio:.0%}/{1 - cfg.train_ratio - cfg.val_ratio:.0%} train/validation/test.",
        f"- Optimizer: mini-batch SGD, initial learning rate = {cfg.learning_rate}, decay = {cfg.lr_decay} per epoch.",
        f"- Loss: cross-entropy with L2 weight decay = {cfg.weight_decay}.",
        "- Array backend: NumPy.",
        f"- Best validation epoch: {best_epoch}.",
        "",
        "## Results",
        "",
        f"- Test loss: {test_loss:.4f}",
        f"- Test accuracy: {test_acc:.4f}",
        "",
        "![Learning curves](learning_curves.png)",
        "",
        "![Confusion matrix](confusion_matrix.png)",
        "",
        "## Error Analysis",
        "",
        "The weakest classes by per-class test accuracy are "
        + ", ".join(f"{CLASSES[i]} ({per_class[i]:.3f})" for i in hardest)
        + ". These errors are plausible because several EuroSAT categories share visual texture and color cues at low resolution. "
        "For example, PermanentCrop, AnnualCrop, and HerbaceousVegetation can all contain repetitive green field patterns, while River and Highway may both appear as long narrow structures crossing mixed backgrounds.",
        "",
        "Representative mistakes are shown below when the model produces any test errors.",
        "",
        "![Error examples](error_examples.png)",
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def write_pdf_report(
    cfg: Config,
    output_dir: Path,
    history: dict[str, list[float]],
    test_loss: float,
    test_acc: float,
    matrix: np.ndarray,
) -> None:
    per_class = matrix.diagonal() / np.maximum(matrix.sum(axis=1), 1)
    hardest = np.argsort(per_class)[:3]
    best_epoch = int(np.argmax(history["val_acc"]) + 1)
    summary = [
        "HW1 EuroSAT RGB MLP Report",
        "",
        "Experimental Setup",
        f"Model: one-hidden-layer MLP, hidden_dim={cfg.hidden_dim}, activation={cfg.activation}.",
        f"Input: RGB images resized to {cfg.image_size}x{cfg.image_size}, flattened and standardized.",
        f"Split: stratified {cfg.train_ratio:.0%}/{cfg.val_ratio:.0%}/{1 - cfg.train_ratio - cfg.val_ratio:.0%}.",
        f"Optimizer: mini-batch SGD, lr={cfg.learning_rate}, lr_decay={cfg.lr_decay}, batch_size={cfg.batch_size}.",
        f"Loss: cross-entropy + L2 weight decay ({cfg.weight_decay}).",
        "Array backend: NumPy.",
        "",
        "Results",
        f"Best validation epoch: {best_epoch}",
        f"Best validation accuracy: {max(history['val_acc']):.4f}",
        f"Test loss: {test_loss:.4f}",
        f"Test accuracy: {test_acc:.4f}",
        "",
        "Error Analysis",
        "Weakest classes: "
        + ", ".join(f"{CLASSES[i]} ({per_class[i]:.3f})" for i in hardest)
        + ".",
        "Likely causes include shared texture and color cues among crop/vegetation classes,",
        "and narrow structures such as River/Highway becoming ambiguous after downsampling.",
    ]

    with PdfPages(output_dir / "report.pdf") as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.08, 0.94, "\n".join(summary), va="top", ha="left", fontsize=11, linespacing=1.55)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        for image_name in ["learning_curves.png", "confusion_matrix.png", "error_examples.png"]:
            image_path = output_dir / image_name
            if not image_path.exists():
                continue
            image = plt.imread(image_path)
            fig, ax = plt.subplots(figsize=(8.27, 11.69))
            ax.imshow(image)
            ax.axis("off")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def main() -> None:
    cfg = parse_args()
    base_dir = Path(__file__).resolve().parent
    data_dir = Path(cfg.data_dir)
    if not data_dir.is_absolute():
        data_dir = base_dir / data_dir
    output_dir = Path(cfg.output_dir)
    if not output_dir.is_absolute():
        output_dir = base_dir / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    print(f"Loading images from {data_dir} ...")
    x, y, paths = load_dataset(data_dir, cfg.image_size)
    train_idx, val_idx, test_idx = split_indices(y, cfg)
    x_train, x_val, x_test, mean, std = normalize_from_train(x[train_idx], x[val_idx], x[test_idx])
    y_train_np, y_val_np, y_test_np = y[train_idx], y[val_idx], y[test_idx]
    y_train, y_val, y_test = y_train_np, y_val_np, y_test_np

    print("Using NumPy backend.")
    model = MLP(x_train.shape[1], cfg.hidden_dim, len(CLASSES), cfg.activation, cfg.seed)
    rng = np.random.default_rng(cfg.seed + 1)
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "lr": []}
    best_state = model.state_dict()
    best_val_acc = -1.0

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
            np.savez(output_dir / "best_model.npz", **best_state, mean=mean, std=std, classes=np.array(CLASSES))
        print(
            f"epoch {epoch:02d}/{cfg.epochs} lr={lr:.5f} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

    model.load_state_dict(best_state)
    test_loss, test_acc = loss_and_accuracy(model, x_test, y_test, cfg.weight_decay)
    y_pred = model.predict(x_test)
    matrix = confusion_matrix(y_test_np, y_pred, len(CLASSES))

    plot_history(history, output_dir)
    plot_confusion(matrix, output_dir)
    plot_error_examples(paths, y_test_np, y_pred, test_idx, output_dir)
    write_report(cfg, output_dir, history, test_loss, test_acc, matrix)
    write_pdf_report(cfg, output_dir, history, test_loss, test_acc, matrix)
    np.savetxt(output_dir / "confusion_matrix.csv", matrix, fmt="%d", delimiter=",")
    (output_dir / "metrics.json").write_text(
        json.dumps(
            {
                "config": asdict(cfg),
                "best_val_accuracy": best_val_acc,
                "test_loss": test_loss,
                "test_accuracy": test_acc,
                "elapsed_seconds": time.time() - start_time,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
