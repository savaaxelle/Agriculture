# Tomato Multiclass + Knowledge Distillation

Starter branch for the next classification experiment.

## Goal

Move from binary `healthy/sick` classification to the original 10 tomato-leaf classes, then compare several pretrained baselines and apply knowledge distillation from a large Pest-ConFormer teacher to a lightweight student.

### 10 classes

1. Tomato___Bacterial_spot
2. Tomato___Early_blight
3. Tomato___Late_blight
4. Tomato___Leaf_Mold
5. Tomato___Septoria_leaf_spot
6. Tomato___Spider_mites Two-spotted_spider_mite
7. Tomato___Target_Spot
8. Tomato___Tomato_Yellow_Leaf_Curl_Virus
9. Tomato___Tomato_mosaic_virus
10. Tomato___healthy

## Baseline models prepared

- MobileNetV3-Large
- EfficientNet-B0
- EfficientNet-B2
- ConvNeXt-Tiny
- DeiT-Small
- ViT-Small
- MobileViT-S
- Swin-Tiny
- DINOv2-Small

All baseline wrappers are in `models.py` and use pretrained weights through `timm`.

Pest-ConFormer is kept as a separate 10-class teacher in `pestconformer_teacher.py` because its training loss is multi-branch.

## Clone in VS Code

```bash
git clone -b multiclass-kd --single-branch https://github.com/savaaxelle/Agriculture.git
cd Agriculture/tomato_multiclass_kd
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Expected dataset layout

```text
dataset/
├── train/
│   ├── Tomato___Bacterial_spot/
│   ├── Tomato___Early_blight/
│   └── ...
├── val/
│   └── ...
└── test/
    └── ...
```

## Check available models

```bash
python models.py
```

## Train one baseline

Example:

```bash
python train_baseline.py --model efficientnet_b2 --data-root dataset
```

Other examples:

```bash
python train_baseline.py --model mobilenetv3 --data-root dataset
python train_baseline.py --model efficientnet_b0 --data-root dataset
python train_baseline.py --model convnext_tiny --data-root dataset
python train_baseline.py --model deit_small --data-root dataset
python train_baseline.py --model vit_small --data-root dataset
python train_baseline.py --model mobilevit_s --data-root dataset
python train_baseline.py --model swin_tiny --data-root dataset
python train_baseline.py --model dinov2_small --data-root dataset
```

## Knowledge distillation plan

The intended experiment is:

```text
Pest-ConFormer (teacher)
        ↓ soft logits
MobileNetV3 / EfficientNet-B0 (student)
        ↓
10-class tomato disease classifier
```

Compare at minimum:

1. student trained normally,
2. the same student trained with knowledge distillation,
3. Pest-ConFormer teacher.

The core measurements should later include multiclass accuracy, macro-F1, per-class recall, confusion matrix, parameter count, model size, and Android inference latency/memory.

## Important

This branch is a clean starter for the multiclass/KD experiment. It does not overwrite the existing binary experiment on `master`.
