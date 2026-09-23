import argparse
import time

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to tomato leaf image")
    parser.add_argument("--model", default=str(BEST_MODEL_PATH))
    args = parser.parse_args()

    device = choose_device()
    checkpoint = torch.load(args.model, map_location=device)

    class_names = checkpoint.get("class_names", ["healthy", "sick"])

    model = build_model(pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    _, transform = get_transforms()
    image = Image.open(args.image).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    start = time.perf_counter()
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]

    if device.type == "cuda":
        torch.cuda.synchronize()

    latency_ms = (time.perf_counter() - start) * 1000
    idx = int(probs.argmax().item())

    print(f"Prediction : {class_names[idx]}")
    print(f"Confidence : {probs[idx].item():.2%}")

    for name, prob in zip(class_names, probs.cpu().tolist()):
        print(f"{name:8s} : {prob:.2%}")

    print(f"Latency    : {latency_ms:.2f} ms")


if __name__ == "__main__":
    main()
