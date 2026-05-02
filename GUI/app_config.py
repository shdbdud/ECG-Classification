from pathlib import Path
import json
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["Normal", "Abnormal"]

BASE_DIR = Path(r"C:\Users\LYG Y9000x\OneDrive\Desktop\proj")
DATA_ROOT = BASE_DIR / "ecg_dataset" / "development"
WEIGHTS_DIR = BASE_DIR / "weights_from_images_binary"
OUTPUTS_DIR = BASE_DIR / "outputs_binary"
RAG_BASE_DIR = OUTPUTS_DIR / "rag_explainability"

MODEL_NAME_LIST = [
    "Enhanced_DSR_SE",
    "Original_DSR",
    "Original_DSR_Attn_Proto",
    "Pure_CNN"
]

CKPTS = {
    "Original_DSR":    WEIGHTS_DIR / "Original_DSR_binary_best.pth",
    "Enhanced_DSR_SE": WEIGHTS_DIR / "Enhanced_DSR_SE_binary_best.pth",
    "Original_DSR_Attn_Proto": WEIGHTS_DIR / "Original_DSR_Attn_Proto_best.pth",
    "Pure_CNN": WEIGHTS_DIR / "Pure_CNN_binary_best.pth"
}

RAG_DIRS = {name: RAG_BASE_DIR / name for name in MODEL_NAME_LIST}

THR_JSON_PATH = WEIGHTS_DIR / "THR_MAP.json"
if THR_JSON_PATH.exists():
    THR_MAP = json.loads(THR_JSON_PATH.read_text(encoding="utf-8"))
else:
    THR_MAP = {
        "Enhanced_DSR_SE": 0.34,
        "Original_DSR": 0.105,
        "Pure_CNN": 0.50,
        "CNN_LSTM": 0.50,
        "Pure_LSTM": 0.50,
    }

MODEL_RUNTIME_CONFIG = {
    "Enhanced_DSR_SE": {
        "use_cwt_norm": False,
        "prediction_mode": "threshold",
        "model_loader": "enhanced_dsr_se",
    },

    "Original_DSR": {
        "use_cwt_norm": False,
        "prediction_mode": "threshold",
        "model_loader": "dsr_from_images",
    },

    "Pure_CNN": {
        "use_cwt_norm": False,
        "prediction_mode": "threshold",
        "model_loader": "pure_cnn",
    },

    "Original_DSR_Attn_Proto": {
        "use_cwt_norm": True,
        "prediction_mode": "argmax",
        "model_loader": "original_dsr_ablation_attn",
    },
}