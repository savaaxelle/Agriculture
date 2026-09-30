from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

ARCHIVE_PATH = PROJECT_DIR / "archive.zip"
RAW_DATA_DIR = PROJECT_DIR / "raw_dataset" / "tomato"
MODEL_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

BEST_MODEL_PATH = MODEL_DIR / "pestconformer_tomato_health.pt"
CHECKPOINT_PATH = MODEL_DIR / "checkpoint_last.pt"

# ConvMAE-Base ImageNet-pretrained weights released with Pest-ConFormer
# (https://drive.google.com/file/d/1AEPivXw0A0b_m5EwEi6fg2pOAoDr8C31). Put the file here.
PRETRAINED_PATH = MODEL_DIR / "base.pth"

# Pest-ConFormer is hard-wired to 224 x 224 input (patch grid 56/28/14).
IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_WORKERS = 4

CLASS_NAMES = ["healthy", "sick"]
NUM_CLASSES = len(CLASS_NAMES)

# Original archive has 10,000 training images (1,000 per original class).
# We use 80% of each original class for training and 20% for validation.
TRAIN_RATIO = 0.80

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10
RANDOM_SEED = 42
