import json

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from config import CLASS_NAMES, RESULTS_DIR


def plot_confusion_matrix(cm, path):
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    ax.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix (test)")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return fig


def plot_training_history(path):
    history_path = RESULTS_DIR / "training_history.json"
    if not history_path.exists():
        return None
    with open(history_path, encoding="utf-8") as f:
        history = json.load(f)

    epochs = [r["epoch"] for r in history]
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 4))
    ax_loss.plot(epochs, [r["train_loss"] for r in history], label="train")
    ax_loss.plot(epochs, [r["val_loss"] for r in history], label="val")
    ax_loss.set_title("Loss")
    ax_acc.plot(epochs, [r["train_accuracy"] for r in history], label="train")
    ax_acc.plot(epochs, [r["val_accuracy"] for r in history], label="val")
    ax_acc.set_title("Accuracy")
    for ax in (ax_loss, ax_acc):
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return fig


def plot_misclassified(samples, y_true, y_pred, sick_probs, path, max_images=12):
    """samples: list of (image_path, label, original_class) in test-loader order."""
    wrong = [i for i, (t, p) in enumerate(zip(y_true, y_pred)) if t != p]
    if not wrong:
        return None

    shown = wrong[:max_images]
    cols = min(4, len(shown))
    rows = (len(shown) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.4 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, i in zip(axes.flat, shown):
        with Image.open(samples[i][0]) as img:
            ax.imshow(img.convert("RGB"))
        ax.set_title(
            f"actual={CLASS_NAMES[y_true[i]]}\npred={CLASS_NAMES[y_pred[i]]} "
            f"(p_sick={sick_probs[i]:.2f})",
            fontsize=9,
        )
    fig.suptitle(f"Misclassified test images ({len(shown)} of {len(wrong)})")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return fig
