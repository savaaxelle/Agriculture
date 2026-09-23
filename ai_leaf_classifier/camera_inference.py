import argparse
import time

import cv2
from PIL import Image
import torch

from config import BEST_MODEL_PATH
from dataset import get_transforms
from model import build_model


def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class TomatoLeafClassifier:
    def __init__(self, model_path):
        self.device = choose_device()
        checkpoint = torch.load(model_path, map_location=self.device)
        self.class_names = checkpoint.get("class_names", ["healthy", "sick"])

        self.model = build_model(pretrained=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device).eval()

        _, self.transform = get_transforms()

    def predict(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        start = time.perf_counter()

        with torch.no_grad():
            probs = torch.softmax(self.model(tensor), dim=1)[0]

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        latency_ms = (time.perf_counter() - start) * 1000
        idx = int(probs.argmax().item())

        return self.class_names[idx], float(probs[idx]), latency_ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--model", default=str(BEST_MODEL_PATH))
    args = parser.parse_args()

    classifier = TomatoLeafClassifier(args.model)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {args.camera}")

    print("Press Q to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        label, confidence, latency_ms = classifier.predict(frame)

        text = f"{label.upper()} | {confidence:.1%} | {latency_ms:.1f} ms"

        cv2.putText(
            frame,
            text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        cv2.imshow("Tomato Leaf Health - EfficientNet-B2", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
