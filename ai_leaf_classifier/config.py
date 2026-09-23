from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

ARCHIVE_PATH = PROJECT_DIR / "archive.zip"
RAW_DATA_DIR = PROJECT_DIR / "raw_dataset" / "tomato"
MODEL_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

BEST_MODEL_PATH = MODEL_DIR / "efficientnet_b2_tomato_health.pt"

# torchvision EfficientNet-B2 ImageNet weights use 288 x 288 crop size.
IMAGE_SIZE = 288
BATCH_SIZE = 16
NUM_WORKERS = 2

CLASS_NAMES = ["healthy", "sick"]
NUM_CLASSES = len(CLASS_NAMES)

# Original archive has 10,000 training images (1,000 per original class).
# We use 80% of each original class for training and 20% for validation.
TRAIN_RATIO = 0.80

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 20
EARLY_STOPPING_PATIENCE = 5
RANDOM_SEED = 42
