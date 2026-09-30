import torch
import torch.nn as nn

from config import NUM_CLASSES, PRETRAINED_PATH
from pestconformer import convvit_base_patch16

# Loss weights from the official Pest-ConFormer fine-tuning recipe (engine_finetune.py).
LAMBDA_LAYER = 0.5
LAMBDA_COMB = 1.0
LAMBDA_DROP = 5.0


class PestConFormer(nn.Module):
    """Pest-ConFormer (ConvViT + FPN/APN + weakly selector + GCN combiner).

    The backbone returns a dict of logits in training (needed for the multi-part loss)
    and only the final `comb_outs` logits in eval, so predict/evaluate/camera code can
    keep calling `model(x)` and taking a softmax.
    """

    def __init__(self, drop_path_rate: float = 0.1):
        super().__init__()
        self.net = convvit_base_patch16(num_classes=NUM_CLASSES, drop_path_rate=drop_path_rate)

    def forward(self, x):
        outputs = self.net(x)
        return outputs if self.training else outputs["comb_outs"]


def load_pretrained_weights(model: PestConFormer, path=PRETRAINED_PATH):
    """Load the ConvMAE-Base checkpoint released with Pest-ConFormer into the backbone."""
    if not path.exists():
        print(f"[WARN] Pretrained checkpoint not found: {path}. Training from random init.")
        return

    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint.get("model", checkpoint)
    own = model.net.state_dict()
    # The checkpoint's pos_embed includes a cls-token slot (1 + 196); the model has none.
    pe = state.get("pos_embed")
    if pe is not None and pe.shape[1] == own["pos_embed"].shape[1] + 1:
        state["pos_embed"] = pe[:, 1:]
    state = {k: v for k, v in state.items() if k in own and v.shape == own[k].shape}
    msg = model.net.load_state_dict(state, strict=False)
    print(f"[INFO] Loaded pretrained weights: {path.name} ({len(state)} tensors)")
    print(f"[INFO] Not initialised from checkpoint: {len(msg.missing_keys)} tensors")


def build_model(pretrained: bool = True):
    model = PestConFormer()
    if pretrained:
        load_pretrained_weights(model)
    return model


def compute_loss(outputs, labels, class_weights=None):
    """Multi-branch loss: per-level CE + drop-region MSE + combined-classifier CE."""
    ce = nn.CrossEntropyLoss(weight=class_weights)
    loss = 0.0
    for name, out in outputs.items():
        if name.startswith("drop_"):
            # Unselected (background) regions are pushed towards a uniform -1 tanh target.
            target = torch.full_like(out, -1.0)
            loss = loss + LAMBDA_DROP * nn.functional.mse_loss(torch.tanh(out), target)
        elif name.startswith("layer"):
            loss = loss + LAMBDA_LAYER * ce(out.mean(1), labels)
        elif name == "comb_outs":
            loss = loss + LAMBDA_COMB * ce(out, labels)
    return loss
