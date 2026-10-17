"""
model_training.py
Modul pelatihan model SVM untuk enam eksperimen klasifikasi kematangan pisang.
Menggunakan sklearn Pipeline (StandardScaler + SVC).
Menyediakan cache fitur dan penyimpanan model.
"""

import os
import time
import json
import hashlib
import numpy as np
import pandas as pd
import joblib
from typing import Dict, List, Optional, Tuple, Callable
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import (
    SVM_KERNEL, SVM_C, SVM_GAMMA, SVM_CLASS_WEIGHT, SVM_RANDOM_STATE,
    MODELS_DIR, FEATURE_CACHE_DIR, RANDOM_SEED,
    EXPERIMENT_NAMES, EXPERIMENT_COLOR_SPACES, EXPERIMENT_USE_GLCM,
    QUICK_MODE_SAMPLES_PER_CLASS,
)
from src.preprocessing import preprocess_image
from src.feature_extraction import extract_all_features, get_feature_dimension


def _get_cache_key(
    dataset_path: str,
    split: str,
    color_space: str,
    use_glcm: bool,
    use_segmentation: bool,
    label_mode: str,
    max_per_class: Optional[int],
) -> str:
    """Generate cache key berdasarkan parameter."""
    key_str = f"{dataset_path}|{split}|{color_space}|{use_glcm}|{use_segmentation}|{label_mode}|{max_per_class}"
    return hashlib.md5(key_str.encode()).hexdigest()


def extract_features_for_split(
    image_paths: List[str],
    labels: List[str],
    color_space: str,
    use_glcm: bool,
    use_segmentation: bool = False,
    cache_key: Optional[str] = None,
    progress_callback: Optional[Callable] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Ekstrak fitur untuk seluruh gambar dalam satu split.
    
    Args:
        image_paths: list path gambar
        labels: list label
        color_space: 'RGB', 'HSV', atau 'CIELAB'
        use_glcm: gunakan GLCM
        use_segmentation: gunakan GrabCut
        cache_key: key untuk caching
        progress_callback: callback(current, total, message)
    
    Returns:
        (X, y, feature_names, skipped_files)
    """
    # Cek cache
    if cache_key:
        cache_path = os.path.join(FEATURE_CACHE_DIR, f"{cache_key}.npz")
        meta_path = os.path.join(FEATURE_CACHE_DIR, f"{cache_key}_meta.json")
        if os.path.exists(cache_path) and os.path.exists(meta_path):
            try:
                data = np.load(cache_path, allow_pickle=True)
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                return (
                    data['X'],
                    data['y'],
                    meta['feature_names'],
                    meta.get('skipped_files', []),
                )
            except Exception:
                pass  # Cache rusak, lanjut ekstraksi
    
    all_features = []
    valid_labels = []
    feature_names = None
    skipped = []
    total = len(image_paths)
    
    for i, (path, label) in enumerate(zip(image_paths, labels)):
        if progress_callback:
            progress_callback(i + 1, total, f"Memproses {os.path.basename(path)}")
        
        # Preprocessing
        result = preprocess_image(path, use_segmentation=use_segmentation)
        if result is None:
            skipped.append(path)
            continue
        
        # Ekstraksi fitur
        try:
            features, names = extract_all_features(
                result, color_space, use_glcm
            )
            
            if feature_names is None:
                feature_names = names
            
            # Validasi dimensi
            expected_dim = get_feature_dimension(color_space, use_glcm)
            if len(features) != expected_dim:
                skipped.append(path)
                continue
            
            all_features.append(features)
            valid_labels.append(label)
            
        except Exception:
            skipped.append(path)
            continue
    
    if not all_features:
        raise ValueError("Tidak ada gambar yang berhasil diproses.")
    
    X = np.array(all_features, dtype=np.float64)
    y = np.array(valid_labels)
    
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    
    # Simpan cache
    if cache_key:
        cache_path = os.path.join(FEATURE_CACHE_DIR, f"{cache_key}.npz")
        meta_path = os.path.join(FEATURE_CACHE_DIR, f"{cache_key}_meta.json")
        try:
            np.savez_compressed(cache_path, X=X, y=y)
            with open(meta_path, 'w') as f:
                json.dump({
                    'feature_names': feature_names,
                    'skipped_files': skipped,
                    'color_space': color_space,
                    'use_glcm': use_glcm,
                    'n_samples': len(valid_labels),
                    'n_features': X.shape[1],
                }, f, indent=2)
        except Exception:
            pass
    
    return X, y, feature_names, skipped


def create_svm_pipeline(
    kernel: str = SVM_KERNEL,
    C: float = SVM_C,
    gamma=SVM_GAMMA,
    class_weight=SVM_CLASS_WEIGHT,
    random_state: int = SVM_RANDOM_STATE,
) -> Pipeline:
    """
    Buat pipeline scikit-learn: StandardScaler + SVC.
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            class_weight=class_weight,
            random_state=random_state,
            decision_function_shape='ovr',
        )),
    ])
    return pipeline


def train_experiment(
    experiment_id: int,
    train_paths: List[str],
    train_labels: List[str],
    dataset_path: str,
    use_segmentation: bool = False,
    label_mode: str = 'original',
    max_per_class: Optional[int] = None,
    svm_params: Optional[Dict] = None,
    progress_callback: Optional[Callable] = None,
) -> Dict:
    """
    Jalankan satu eksperimen: ekstraksi fitur + pelatihan SVM.
    
    Returns:
        Dict berisi model, metadata, waktu, dan hasil.
    """
    color_space = EXPERIMENT_COLOR_SPACES[experiment_id]
    use_glcm = EXPERIMENT_USE_GLCM[experiment_id]
    exp_name = EXPERIMENT_NAMES[experiment_id]
    
    result = {
        'experiment_id': experiment_id,
        'experiment_name': exp_name,
        'color_space': color_space,
        'use_glcm': use_glcm,
        'use_segmentation': use_segmentation,
        'label_mode': label_mode,
        'status': 'pending',
    }
    
    start_time = time.time()
    
    # Cache key
    cache_key = _get_cache_key(
        dataset_path, 'train', color_space, use_glcm,
        use_segmentation, label_mode, max_per_class,
    )
    
    # Ekstrak fitur
    if progress_callback:
        progress_callback(0, len(train_paths), f"[{exp_name}] Mengekstrak fitur...")
    
    try:
        X_train, y_train, feature_names, skipped = extract_features_for_split(
            train_paths, train_labels, color_space, use_glcm,
            use_segmentation, cache_key, progress_callback,
        )
    except Exception as e:
        result['status'] = 'error'
        result['error'] = str(e)
        return result
    
    extract_time = time.time() - start_time
    
    # Buat dan latih model
    if svm_params:
        pipeline = create_svm_pipeline(**svm_params)
    else:
        pipeline = create_svm_pipeline()
    
    train_start = time.time()
    try:
        pipeline.fit(X_train, y_train)
    except Exception as e:
        result['status'] = 'error'
        result['error'] = f"Gagal melatih model: {str(e)}"
        return result
    
    train_time = time.time() - train_start
    total_time = time.time() - start_time
    
    # Simpan model
    model_info = {
        'pipeline': pipeline,
        'feature_names': feature_names,
        'classes': list(pipeline.classes_),
        'color_space': color_space,
        'use_glcm': use_glcm,
        'use_segmentation': use_segmentation,
        'label_mode': label_mode,
        'experiment_id': experiment_id,
        'experiment_name': exp_name,
        'n_train_samples': len(y_train),
        'n_features': X_train.shape[1],
        'svm_params': {
            'kernel': pipeline.named_steps['svm'].kernel,
            'C': pipeline.named_steps['svm'].C,
            'gamma': pipeline.named_steps['svm'].gamma,
            'class_weight': str(pipeline.named_steps['svm'].class_weight),
        },
    }
    
    model_path = os.path.join(MODELS_DIR, f"experiment_{experiment_id}.joblib")
    joblib.dump(model_info, model_path)
    
    result.update({
        'status': 'success',
        'model_path': model_path,
        'feature_names': feature_names,
        'n_train_samples': len(y_train),
        'n_features': X_train.shape[1],
        'n_skipped': len(skipped),
        'extract_time_s': round(extract_time, 2),
        'train_time_s': round(train_time, 2),
        'total_time_s': round(total_time, 2),
        'classes': list(pipeline.classes_),
    })
    
    return result


def load_trained_model(experiment_id: int) -> Optional[Dict]:
    """
    Muat model yang sudah dilatih.
    
    Returns:
        Dict model_info atau None jika tidak ditemukan.
    """
    model_path = os.path.join(MODELS_DIR, f"experiment_{experiment_id}.joblib")
    if not os.path.exists(model_path):
        return None
    try:
        return joblib.load(model_path)
    except Exception:
        return None


def get_available_models() -> Dict[int, str]:
    """
    Cek model mana saja yang sudah dilatih.
    
    Returns:
        Dict experiment_id -> model_path
    """
    available = {}
    for exp_id in EXPERIMENT_NAMES:
        path = os.path.join(MODELS_DIR, f"experiment_{exp_id}.joblib")
        if os.path.exists(path):
            available[exp_id] = path
    return available
