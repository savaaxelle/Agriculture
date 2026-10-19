from pathlib import Path

import torch
import torch.nn as nn

from ai_leaf_classifier.pestconformer import convvit_base_patch16
from config import NUM_CLASSES

LAMBDA_LAYER = 0.5
LAMBDA_COMB = 1.0
LAMBDA_DROP = 5.0


class PestConFormer10(nn.Module):
    """10-class Pest-ConFormer teacher.

    In training mode, the original multi-branch dictionary is returned.
    In eval mode, only final combiner logits are returned.
    """

    def __init__(self, num_classes=NUM_CLASSES, drop_path_rate=0.1):
        super().__init__()
        self.net = convvit_base_patch16(
            num_classes=num_classes,
            drop_path_rate=drop_path_rate,
        )

    def forward(self, x):
        outputs = self.net(x)
        return outputs if self.training else outputs["comb_outs"]


def load_convmae_pretrained(model, checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("model", checkpoint)
    own = model.net.state_dict()

    pe = state.get("pos_embed")
    if pe is not None and "pos_embed" in own:
        if pe.shape[1] == own["pos_embed"].shape[1] + 1:
            state["pos_embed"] = pe[:, 1:]

    compatible = {
        k: v for k, v in state.items()
        if k in own and v.shape == own[k].shape
    }
    msg = model.net.load_state_dict(compatible, strict=False)

    print(f"Loaded {len(compatible)} pretrained tensors.")
    print(f"Missing tensors: {len(msg.missing_keys)}")


def teacher_loss(outputs, labels, class_weights=None):
    ce = nn.CrossEntropyLoss(weight=class_weights)
    loss = 0.0

    for name, out in outputs.items():
        if name.startswith("drop_"):
            target = torch.full_like(out, -1.0)
            loss = loss + LAMBDA_DROP * nn.functional.mse_loss(torch.tanh(out), target)
        elif name.startswith("layer"):
            loss = loss + LAMBDA_LAYER * ce(out.mean(1), labels)
        elif name == "comb_outs":
            loss = loss + LAMBDA_COMB * ce(out, labels)

    return loss
