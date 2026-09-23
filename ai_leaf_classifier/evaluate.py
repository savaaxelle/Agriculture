import json
import time

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from config import BEST_MODEL_PATH, CLASS_NAMES, RESULTS_DIR
from dataset import create_dataloaders
from model import build_model


def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    device = choose_device()
    _, _, test_loader = create_dataloaders()

    checkpoint = torch.load(BEST_MODEL_PATH, map_location=device)
    model = build_model(pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    y_true = []
    y_pred = []
    sick_probs = []
    latencies = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)

            start = time.perf_counter()
            logits = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - start) * 1000

            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(1).cpu().numpy()

            latencies.append(elapsed_ms / images.size(0))
            y_true.extend(labels.numpy().tolist())
            y_pred.extend(preds.tolist())
            sick_probs.extend(probs[:, 1].cpu().numpy().tolist())

    accuracy = accuracy_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    roc_auc = roc_auc_score(y_true, sick_probs)
    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1),
        "roc_auc_sick": float(roc_auc),
        "average_latency_ms_per_image": float(np.mean(latencies)),
        "confusion_matrix": cm.tolist(),
        "test_samples": len(y_true),
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        zero_division=0,
    )
    with open(RESULTS_DIR / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    print("\n=== TEST RESULTS ===")
    print(json.dumps(metrics, indent=2))
    print("\nClassification report:")
    print(report)


if __name__ == "__main__":
    main()
