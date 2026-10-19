import timm

from config import NUM_CLASSES

# One common interface for the comparison models requested for the paper.
MODEL_REGISTRY = {
    "mobilenetv3": "mobilenetv3_large_100.ra_in1k",
    "efficientnet_b0": "efficientnet_b0.ra_in1k",
    "efficientnet_b2": "efficientnet_b2.ra_in1k",
    "convnext_tiny": "convnext_tiny.fb_in22k_ft_in1k",
    "deit_small": "deit_small_patch16_224.fb_in1k",
    "vit_small": "vit_small_patch16_224.augreg_in1k",
    "mobilevit_s": "mobilevit_s.cvnets_in1k",
    "swin_tiny": "swin_tiny_patch4_window7_224.ms_in1k",
    "dinov2_small": "vit_small_patch14_dinov2.lvd142m",
}


def build_model(name: str, num_classes: int = NUM_CLASSES, pretrained: bool = True):
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Available: {', '.join(MODEL_REGISTRY)}"
        )

    timm_name = MODEL_REGISTRY[name]
    kwargs = dict(pretrained=pretrained, num_classes=num_classes)

    # DINOv2 was pretrained at a larger native resolution, but timm can
    # resample its positional embedding when the model is created at 224.
    if name == "dinov2_small":
        kwargs["img_size"] = 224

    return timm.create_model(timm_name, **kwargs)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    print("Prepared comparison models:")
    for alias, timm_name in MODEL_REGISTRY.items():
        print(f"- {alias:18s} -> {timm_name}")
