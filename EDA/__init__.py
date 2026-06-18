from pathlib import Path
import sys
import os

ROOT = Path(__file__).parent.parent.resolve()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TUMOR_TRAIN_PATH = ROOT / "brain-Tumor-1/train/Tumor"
TUMOR_TEST_PATH = ROOT / "brain-Tumor-1/test/Tumor"
TUMOR_VAL_PATH = ROOT / "brain-Tumor-1/valid/Tumor"
NOTUMOR_TRAIN_PATH = ROOT / "brain-Tumor-1/train/NoTumor"
NOTUMOR_TEST_PATH = ROOT / "brain-Tumor-1/test/NoTumor"
NOTUMOR_VAL_PATH = ROOT / "brain-Tumor-1/val/NoTumor"
