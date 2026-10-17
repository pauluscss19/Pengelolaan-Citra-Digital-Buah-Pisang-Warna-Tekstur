"""
prediction.py
Modul prediksi gambar baru menggunakan model yang sudah dilatih.
"""

import os
import numpy as np
from typing import Dict, List, Optional, Tuple

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import EXPERIMENT_NAMES
from src.preprocessing import preprocess_image
from src.feature_extraction import extract_all_features
from src.model_training import load_trained_model


def predict_image(
    image_path: str,
    experiment_id: int,
    use_segmentation: bool = False,
) -> Optional[Dict]:
    """
    Prediksi tingkat kematangan dari satu gambar menggunakan model terlatih.
    
    Args:
        image_path: path gambar input
        experiment_id: ID eksperimen (1-6)
        use_segmentation: gunakan GrabCut
    
    Returns:
        Dict berisi prediksi, fitur, dan informasi preprocessing.
        None jika model belum dilatih atau gambar gagal diproses.
    """
    # Muat model
    model_info = load_trained_model(experiment_id)
    if model_info is None:
        return {
            'error': True,
            'message': (
                f"Model eksperimen {experiment_id} ({EXPERIMENT_NAMES.get(experiment_id, '')}) "
                f"belum dilatih. Silakan jalankan pelatihan terlebih dahulu "
                f"di halaman 'Training & Eksperimen'."
            ),
        }
    
    pipeline = model_info['pipeline']
    color_space = model_info['color_space']
    use_glcm = model_info['use_glcm']
    feature_names = model_info['feature_names']
    model_seg = model_info.get('use_segmentation', False)
    
    # Gunakan setting segmentasi yang sama dengan saat training
    effective_seg = model_seg
    
    # Preprocessing
    preprocessed = preprocess_image(image_path, use_segmentation=effective_seg)
    if preprocessed is None:
        return {
            'error': True,
            'message': "Gagal membaca atau memproses gambar. Pastikan format file didukung (JPG, PNG).",
        }
    
    # Ekstraksi fitur
    try:
        features, names = extract_all_features(
            preprocessed, color_space, use_glcm
        )
    except Exception as e:
        return {
            'error': True,
            'message': f"Gagal mengekstrak fitur: {str(e)}",
        }
    
    # Validasi dimensi fitur
    expected_dim = len(feature_names)
    if len(features) != expected_dim:
        return {
            'error': True,
            'message': (
                f"Dimensi fitur tidak sesuai. "
                f"Diharapkan {expected_dim}, didapat {len(features)}. "
                f"Pastikan konfigurasi preprocessing konsisten."
            ),
        }
    
    # Prediksi
    X = features.reshape(1, -1)
    prediction = pipeline.predict(X)[0]
    
    # Decision function (bukan probabilitas)
    try:
        decision = pipeline.decision_function(X)
        if decision.ndim == 1:
            decision_values = decision.tolist()
        else:
            decision_values = decision[0].tolist()
    except Exception:
        decision_values = None
    
    return {
        'error': False,
        'prediction': prediction,
        'experiment_id': experiment_id,
        'experiment_name': EXPERIMENT_NAMES.get(experiment_id, ''),
        'color_space': color_space,
        'use_glcm': use_glcm,
        'feature_names': feature_names,
        'feature_values': features.tolist(),
        'decision_values': decision_values,
        'classes': model_info['classes'],
        'preprocessed': preprocessed,
        'use_segmentation': effective_seg,
        'n_features': len(features),
    }


def predict_all_models(
    image_path: str,
    use_segmentation: bool = False,
) -> Dict[int, Dict]:
    """
    Prediksi menggunakan semua model yang tersedia.
    
    Returns:
        Dict experiment_id -> prediction_result
    """
    results = {}
    for exp_id in EXPERIMENT_NAMES:
        result = predict_image(image_path, exp_id, use_segmentation)
        if result is not None:
            results[exp_id] = result
    return results
