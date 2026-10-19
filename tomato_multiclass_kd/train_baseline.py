import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch.optim import AdamW
from tqdm import tqdm

from config import (
    EPOCHS,
    LEARNING_RATE,
    OUTPUT_DIR,
    PATIENCE,
    SEED,
    WEIGHT_DECAY,
)
from data import make_loaders
from models import MODEL_REGISTRY, build_model, count_parameters


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, preds = [], []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += criterion(logits, y).item() * y.size(0)
        ys.extend(y.cpu().tolist())
        preds.extend(logits.argmax(1).cpu().tolist())

    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": accuracy_score(ys, preds),
        "balanced_accuracy": balanced_accuracy_score(ys, preds),
        "macro_f1": f1_score(ys, preds, average="macro"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=MODEL_REGISTRY.keys())
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()

    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders, class_names = make_loaders(args.data_root)

    model = build_model(args.model, num_classes=len(class_names), pretrained=True).to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    out_dir = OUTPUT_DIR / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best.pt"

    best_f1 = -1.0
    bad_epochs = 0

    print(f"Device: {device}")
    print(f"Classes: {class_names}")
    print(f"Parameters: {count_parameters(model):,}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0

        for x, y in tqdm(loaders["train"], desc=f"Epoch {epoch}"):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running += loss.item() * y.size(0)

        val = evaluate(model, loaders["val"], device)
        train_loss = running / len(loaders["train"].dataset)

        print(
            f"epoch={epoch} train_loss={train_loss:.4f} "
            f"val_loss={val['loss']:.4f} val_acc={val['accuracy']:.4f} "
            f"val_macro_f1={val['macro_f1']:.4f}"
        )

        if val["macro_f1"] > best_f1:
            best_f1 = val["macro_f1"]
            bad_epochs = 0
            torch.save({
                "model": model.state_dict(),
                "model_name": args.model,
                "classes": class_names,
                "epoch": epoch,
                "val": val,
            }, best_path)
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print("Early stopping.")
                break

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    test = evaluate(model, loaders["test"], device)

    result = {
        "model": args.model,
        "parameters": count_parameters(model),
        "best_val_macro_f1": best_f1,
        "test": test,
        "classes": class_names,
    }

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("Test:", test)
    print("Saved:", best_path)


if __name__ == "__main__":
    main()
