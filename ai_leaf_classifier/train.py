import copy
import json
import time

import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

from config import (
    BEST_MODEL_PATH,
    CHECKPOINT_PATH,
    RESULTS_DIR,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    EARLY_STOPPING_PATIENCE,
    CLASS_NAMES,
)
from dataset import create_dataloaders
from model import build_model


def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_class_weights(train_dataset, device):
    targets = torch.tensor(train_dataset.targets)
    counts = torch.bincount(targets, minlength=len(CLASS_NAMES)).float()
    weights = len(targets) / (len(CLASS_NAMES) * counts)

    print(f"[INFO] Binary train counts: healthy={int(counts[0])}, sick={int(counts[1])}")
    print(f"[INFO] Loss weights       : healthy={weights[0]:.4f}, sick={weights[1]:.4f}")

    return weights.to(device)


def run_epoch(model, loader, criterion, device, optimizer=None, desc=""):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    progress = tqdm(loader, desc=desc, leave=False, unit="batch")
    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)

            if training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(1) == labels).sum().item()
        total_samples += labels.size(0)

        progress.set_postfix(
            loss=f"{total_loss / total_samples:.4f}",
            acc=f"{total_correct / total_samples:.4f}",
        )

    return total_loss / total_samples, total_correct / total_samples


def main():
    device = choose_device()
    print(f"[INFO] Device: {device}")

    train_loader, val_loader, _ = create_dataloaders()
    print(
        f"[INFO] Samples: train={len(train_loader.dataset)}, "
        f"val={len(val_loader.dataset)}"
    )

    model = build_model(pretrained=True).to(device)
    class_weights = make_class_weights(train_loader.dataset, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    start_epoch = 1
    best_val_acc = 0.0
    wait = 0
    history = []

    if CHECKPOINT_PATH.exists():
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_acc = checkpoint["best_val_acc"]
        wait = checkpoint["wait"]
        history = checkpoint["history"]
        print(
            f"[RESUME] Melanjutkan dari epoch {start_epoch} "
            f"(best_val_acc={best_val_acc:.4f}, wait={wait})"
        )

    for epoch in range(start_epoch, EPOCHS + 1):
        start = time.time()

        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, device, optimizer,
            desc=f"Epoch {epoch:02d}/{EPOCHS} [train]",
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, device,
            desc=f"Epoch {epoch:02d}/{EPOCHS} [val]",
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
        }
        history.append(row)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"{time.time() - start:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            wait = 0
            torch.save(
                {
                    "model_state_dict": copy.deepcopy(model.state_dict()),
                    "class_names": CLASS_NAMES,
                    "val_accuracy": best_val_acc,
                },
                BEST_MODEL_PATH,
            )
            print(f"[SAVE] {BEST_MODEL_PATH}")
        else:
            wait += 1

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_acc": best_val_acc,
                "wait": wait,
                "history": history,
            },
            CHECKPOINT_PATH,
        )

        with open(RESULTS_DIR / "training_history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        if wait >= EARLY_STOPPING_PATIENCE:
            print("[INFO] Early stopping.")
            break

    print(f"[DONE] Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
