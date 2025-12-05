"""
Seg-Mind Configuration File
Contains all system-wide settings, paths, and hyperparameters
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ==================== BASE PATHS ====================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
UPLOADS_DIR = DATA_DIR / "uploads"
MODELS_DIR = BASE_DIR / "ml_models"
LOGS_DIR = BASE_DIR / "logs"

# BraTS Dataset Path
BRATS_DATASET_PATH = Path("D:/BraTs 2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData")

# ==================== API SETTINGS ====================
API_TITLE = "Seg-Mind API"
API_VERSION = "1.0.0"
API_DESCRIPTION = "AI-powered Brain Tumor Segmentation and Diagnostic System"
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))
DEBUG_MODE = os.getenv("DEBUG", "True").lower() == "true"

# ==================== FIREBASE SETTINGS ====================
FIREBASE_CREDENTIALS_PATH = os.getenv("FIREBASE_CREDENTIALS_PATH", "firebase-credentials.json")
FIREBASE_STORAGE_BUCKET = os.getenv("FIREBASE_STORAGE_BUCKET", "seg-mind.appspot.com")

# ==================== SECURITY ====================
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 120  # 2 hours as per SRS

# ==================== FILE UPLOAD SETTINGS ====================
MAX_UPLOAD_SIZE = 5 * 1024 * 1024 * 1024  # 5GB as per SRS
ALLOWED_DICOM_EXTENSIONS = [".dcm", ".dicom"]
ALLOWED_NIFTI_EXTENSIONS = [".nii", ".nii.gz"]
ALLOWED_EXTENSIONS = ALLOWED_DICOM_EXTENSIONS + ALLOWED_NIFTI_EXTENSIONS

# ==================== MRI MODALITIES ====================
MRI_MODALITIES = ["t1n", "t1c", "t2w", "t2f"]  # T1-native, T1-contrast, T2-weighted, T2-FLAIR
REQUIRED_MODALITIES = ["t1c", "t2w", "t2f"]  # Minimum required for segmentation

# ==================== SEGMENTATION MODEL SETTINGS ====================
SEGMENTATION_MODEL = {
    "name": "3D U-Net",
    "framework": "MONAI",
    "input_shape": (4, 128, 128, 128),  # 4 modalities, 128x128x128 volume
    "num_classes": 4,  # Background, Enhancing Tumor, Tumor Core, Edema
    "in_channels": 4,
    "out_channels": 4,
    "channels": (16, 32, 64, 128, 256),
    "strides": (2, 2, 2, 2),
    "dropout": 0.2,
}

# Training hyperparameters
SEGMENTATION_TRAINING = {
    "batch_size": 2,  # Small due to 3D volume size
    "learning_rate": 1e-4,
    "num_epochs": 300,
    "validation_split": 0.2,
    "early_stopping_patience": 30,
    "lr_scheduler": "ReduceLROnPlateau",
    "optimizer": "Adam",
    "loss_function": "DiceCELoss",  # Dice + Cross Entropy
}

# ==================== CLASSIFICATION MODEL SETTINGS ====================
CLASSIFICATION_MODEL = {
    "name": "ResNet50",
    "framework": "PyTorch",
    "num_classes": 4,  # Glioblastoma, Glioma, Meningioma, Other
    "input_size": (224, 224),
    "pretrained": True,
}

# Training hyperparameters
CLASSIFICATION_TRAINING = {
    "batch_size": 16,
    "learning_rate": 1e-4,
    "num_epochs": 100,
    "validation_split": 0.2,
    "early_stopping_patience": 15,
    "optimizer": "Adam",
    "loss_function": "CrossEntropyLoss",
}

# ==================== TUMOR CLASSES ====================
TUMOR_TYPES = {
    0: "Glioblastoma (GBM)",
    1: "Low-Grade Glioma (LGG)",
    2: "Meningioma",
    3: "Other"
}

WHO_GRADES = {
    1: "Grade I (Low malignancy)",
    2: "Grade II (Low malignancy)",
    3: "Grade III (High malignancy)",
    4: "Grade IV (High malignancy)"
}

# Tumor regions for segmentation
TUMOR_REGIONS = {
    0: "Background",
    1: "Necrotic/Non-enhancing tumor core (NCR/NET)",
    2: "Peritumoral edema (ED)",
    3: "GD-enhancing tumor (ET)"
}

# Combined tumor regions (as per BraTS)
REGION_LABELS = {
    "ET": 3,  # Enhancing Tumor
    "TC": [1, 3],  # Tumor Core (NET + ET)
    "WT": [1, 2, 3]  # Whole Tumor (NET + ED + ET)
}

# ==================== PREPROCESSING SETTINGS ====================
PREPROCESSING = {
    "normalize_method": "z_score",  # or "min_max"
    "clip_percentiles": (1, 99),  # Clip intensity outliers
    "target_spacing": (1.0, 1.0, 1.0),  # mm
    "target_size": (128, 128, 128),  # Resize to this for training
    "noise_reduction": True,
    "bias_correction": True,
}

# ==================== VALIDATION METRICS ====================
METRICS = {
    "segmentation": ["dice_score", "hausdorff_distance", "sensitivity", "specificity"],
    "classification": ["accuracy", "precision", "recall", "f1_score", "auc_roc"]
}

# Confidence thresholds as per SRS
CONFIDENCE_THRESHOLDS = {
    "high": 0.90,  # >90% high confidence
    "medium": 0.70,  # 70-90% medium confidence
    "low": 0.70  # <70% low confidence (trigger review)
}

# ==================== EXPLAINABLE AI SETTINGS ====================
XAI_CONFIG = {
    "grad_cam_layer": "layer4",  # For ResNet
    "shap_samples": 100,
    "heatmap_colormap": "jet",
    "heatmap_opacity": 0.5,
}

# ==================== LSTM GROWTH PREDICTION ====================
LSTM_CONFIG = {
    "input_features": 10,  # Tumor volume, region stats, etc.
    "hidden_size": 64,
    "num_layers": 2,
    "dropout": 0.3,
    "sequence_length": 3,  # Require at least 3 previous scans
    "prediction_horizons": [1, 3, 6],  # months
}

# ==================== LOGGING ====================
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": LOGS_DIR / "seg_mind.log",
    "max_bytes": 10 * 1024 * 1024,  # 10MB
    "backup_count": 5,
}

# ==================== DATABASE ====================
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
DATABASE_NAME = "seg_mind_db"

# ==================== CORS SETTINGS ====================
CORS_ORIGINS = [
    "http://localhost:3000",  # React development server
    "http://localhost:8000",
    "https://seg-mind.web.app",  # Firebase hosting
]

# ==================== GPU/DEVICE SETTINGS ====================
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4  # For data loaders
PIN_MEMORY = True if torch.cuda.is_available() else False

# ==================== REPORT GENERATION ====================
REPORT_SETTINGS = {
    "logo_path": BASE_DIR / "assets" / "logo.png",
    "font_family": "Helvetica",
    "page_size": "A4",
    "include_xai": True,
    "include_visualizations": True,
}

# ==================== DEVELOPMENT FLAGS ====================
ENABLE_PROFILING = False
ENABLE_DETAILED_LOGGING = DEBUG_MODE
CACHE_PREPROCESSED_DATA = True

# Only print config on main process (not worker processes)
if __name__ != "__mp_main__":
    print(f"✓ Configuration loaded successfully")
    print(f"  - Device: {DEVICE}")
    print(f"  - BraTS Dataset: {BRATS_DATASET_PATH}")
    print(f"  - Debug Mode: {DEBUG_MODE}")
