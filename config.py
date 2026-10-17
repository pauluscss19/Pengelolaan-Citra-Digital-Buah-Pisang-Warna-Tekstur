# Banana Ripeness Classification Project
# Konfigurasi project

import os

# --- Path Configuration ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Default dataset path - bisa diganti via UI Streamlit
DEFAULT_DATASET_PATH = os.path.join(PROJECT_ROOT, "dataset")

# Artifacts
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
MODELS_DIR = os.path.join(ARTIFACTS_DIR, "models")
FEATURE_CACHE_DIR = os.path.join(ARTIFACTS_DIR, "feature_cache")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
CONFUSION_MATRICES_DIR = os.path.join(RESULTS_DIR, "confusion_matrices")
CLASSIFICATION_REPORTS_DIR = os.path.join(RESULTS_DIR, "classification_reports")

# --- Preprocessing Configuration ---
TARGET_SIZE = (128, 128)  # Ukuran gambar setelah resize
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

# --- Feature Extraction Configuration ---
HISTOGRAM_BINS = 8  # Bin per kanal histogram warna

# --- GLCM Configuration ---
GLCM_GRAY_LEVELS = 32  # Kuantisasi grayscale
GLCM_DISTANCES = [1, 3]  # Jarak piksel
GLCM_ANGLES_DEG = [0, 45, 90, 135]  # Sudut dalam derajat
GLCM_PROPERTIES = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
GLCM_RESIZE = (64, 64)  # Resolusi pemrosesan tekstur (lebih kecil untuk kecepatan)

# --- SVM Configuration ---
SVM_KERNEL = 'rbf'
SVM_C = 1.0
SVM_GAMMA = 'scale'
SVM_CLASS_WEIGHT = 'balanced'
SVM_RANDOM_STATE = 42

# --- Experiment Configuration ---
QUICK_MODE_SAMPLES_PER_CLASS = 200  # Batas sampel per kelas di mode cepat
RANDOM_SEED = 42

# --- Label Mapping (Mode Kelompok Kematangan) ---
MATURITY_GROUP_MAPPING = {
    'unripe': 'Mentah',
    'freshunripe': 'Mentah',
    'ripe': 'Matang',
    'freshripe': 'Matang',
    'overripe': 'Terlalu Matang',
    'rotten': 'Busuk',
}

# --- Experiment Names ---
EXPERIMENT_NAMES = {
    1: "RGB + SVM",
    2: "HSV + SVM",
    3: "CIELAB + SVM",
    4: "RGB + GLCM + SVM",
    5: "HSV + GLCM + SVM",
    6: "CIELAB + GLCM + SVM",
}

EXPERIMENT_COLOR_SPACES = {
    1: "RGB",
    2: "HSV",
    3: "CIELAB",
    4: "RGB",
    5: "HSV",
    6: "CIELAB",
}

EXPERIMENT_USE_GLCM = {
    1: False,
    2: False,
    3: False,
    4: True,
    5: True,
    6: True,
}

# Variasi nama folder validasi yang didukung
VALID_FOLDER_NAMES = ['valid', 'val', 'validation']


def ensure_dirs():
    """Buat semua direktori yang diperlukan."""
    for d in [ARTIFACTS_DIR, MODELS_DIR, FEATURE_CACHE_DIR,
              RESULTS_DIR, PLOTS_DIR, CONFUSION_MATRICES_DIR,
              CLASSIFICATION_REPORTS_DIR]:
        os.makedirs(d, exist_ok=True)


# Otomatis buat direktori saat modul diimpor
ensure_dirs()
