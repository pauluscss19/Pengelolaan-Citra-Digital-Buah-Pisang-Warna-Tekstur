"""
feature_extraction.py
Modul ekstraksi fitur warna (RGB, HSV, CIELAB) dan tekstur (GLCM).
Setiap fungsi menghasilkan vektor fitur berdimensi tetap.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from skimage.feature import graycomatrix, graycoprops

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import (
    HISTOGRAM_BINS,
    GLCM_GRAY_LEVELS,
    GLCM_DISTANCES,
    GLCM_ANGLES_DEG,
    GLCM_PROPERTIES,
    GLCM_RESIZE,
)
from src.preprocessing import apply_mask


# ==================== FITUR WARNA ====================

def _channel_stats(pixels: np.ndarray) -> List[float]:
    """
    Hitung statistik satu kanal: mean, std, median.
    
    Args:
        pixels: array 1D piksel satu kanal.
    
    Returns:
        [mean, std, median]
    """
    if len(pixels) == 0:
        return [0.0, 0.0, 0.0]
    return [
        float(np.mean(pixels)),
        float(np.std(pixels)),
        float(np.median(pixels)),
    ]


def _channel_histogram(pixels: np.ndarray, bins: int, value_range: Tuple[int, int]) -> List[float]:
    """
    Hitung histogram ternormalisasi satu kanal.
    
    Args:
        pixels: array 1D piksel satu kanal.
        bins: jumlah bin.
        value_range: (min, max) range nilai.
    
    Returns:
        List histogram ternormalisasi (jumlah elemen = bins).
    """
    if len(pixels) == 0:
        return [0.0] * bins
    
    hist, _ = np.histogram(pixels, bins=bins, range=value_range)
    total = hist.sum()
    if total > 0:
        hist = hist.astype(np.float64) / total
    return hist.tolist()


def extract_rgb_features(
    img_rgb: np.ndarray, 
    mask: np.ndarray,
    bins: int = HISTOGRAM_BINS,
) -> Tuple[np.ndarray, List[str]]:
    """
    Ekstrak fitur warna RGB.
    
    Returns:
        (feature_vector, feature_names)
    """
    pixels = apply_mask(img_rgb, mask)  # (N, 3)
    
    features = []
    names = []
    
    channel_names = ['R', 'G', 'B']
    for i, ch_name in enumerate(channel_names):
        ch_pixels = pixels[:, i] if len(pixels) > 0 else np.array([])
        
        # Statistik
        stats = _channel_stats(ch_pixels)
        features.extend(stats)
        names.extend([f'RGB_{ch_name}_mean', f'RGB_{ch_name}_std', f'RGB_{ch_name}_median'])
        
        # Histogram
        hist = _channel_histogram(ch_pixels, bins, (0, 256))
        features.extend(hist)
        names.extend([f'RGB_{ch_name}_hist_{j}' for j in range(bins)])
    
    return np.array(features, dtype=np.float64), names


def extract_hsv_features(
    img_hsv: np.ndarray,
    mask: np.ndarray,
    bins: int = HISTOGRAM_BINS,
) -> Tuple[np.ndarray, List[str]]:
    """
    Ekstrak fitur warna HSV.
    Catatan: Kanal H pada OpenCV memiliki range 0-179.
    
    Returns:
        (feature_vector, feature_names)
    """
    pixels = apply_mask(img_hsv, mask)  # (N, 3)
    
    features = []
    names = []
    
    channel_names = ['H', 'S', 'V']
    # Range tiap kanal pada OpenCV: H=[0,180), S=[0,256), V=[0,256)
    channel_ranges = [(0, 180), (0, 256), (0, 256)]
    
    for i, (ch_name, ch_range) in enumerate(zip(channel_names, channel_ranges)):
        ch_pixels = pixels[:, i] if len(pixels) > 0 else np.array([])
        
        # Statistik
        stats = _channel_stats(ch_pixels)
        features.extend(stats)
        names.extend([f'HSV_{ch_name}_mean', f'HSV_{ch_name}_std', f'HSV_{ch_name}_median'])
        
        # Histogram
        hist = _channel_histogram(ch_pixels, bins, ch_range)
        features.extend(hist)
        names.extend([f'HSV_{ch_name}_hist_{j}' for j in range(bins)])
    
    return np.array(features, dtype=np.float64), names


def extract_lab_features(
    img_lab: np.ndarray,
    mask: np.ndarray,
    bins: int = HISTOGRAM_BINS,
) -> Tuple[np.ndarray, List[str]]:
    """
    Ekstrak fitur warna CIELAB.
    Catatan: pada OpenCV, L=[0,255], a=[0,255], b=[0,255] (skala 8-bit).
    
    Returns:
        (feature_vector, feature_names)
    """
    pixels = apply_mask(img_lab, mask)  # (N, 3)
    
    features = []
    names = []
    
    channel_names = ['L', 'a', 'b']
    
    for i, ch_name in enumerate(channel_names):
        ch_pixels = pixels[:, i] if len(pixels) > 0 else np.array([])
        
        # Statistik
        stats = _channel_stats(ch_pixels)
        features.extend(stats)
        names.extend([f'LAB_{ch_name}_mean', f'LAB_{ch_name}_std', f'LAB_{ch_name}_median'])
        
        # Histogram
        hist = _channel_histogram(ch_pixels, bins, (0, 256))
        features.extend(hist)
        names.extend([f'LAB_{ch_name}_hist_{j}' for j in range(bins)])
    
    return np.array(features, dtype=np.float64), names


def extract_color_features(
    color_space: str,
    img_rgb: np.ndarray,
    img_hsv: np.ndarray, 
    img_lab: np.ndarray,
    mask: np.ndarray,
    bins: int = HISTOGRAM_BINS,
) -> Tuple[np.ndarray, List[str]]:
    """
    Ekstrak fitur warna berdasarkan ruang warna yang dipilih.
    
    Args:
        color_space: 'RGB', 'HSV', atau 'CIELAB'
    
    Returns:
        (feature_vector, feature_names)
    """
    if color_space == 'RGB':
        return extract_rgb_features(img_rgb, mask, bins)
    elif color_space == 'HSV':
        return extract_hsv_features(img_hsv, mask, bins)
    elif color_space == 'CIELAB':
        return extract_lab_features(img_lab, mask, bins)
    else:
        raise ValueError(f"Ruang warna tidak dikenal: {color_space}")


# ==================== FITUR TEKSTUR GLCM ====================

def extract_glcm_features(
    img_gray: np.ndarray,
    mask: Optional[np.ndarray] = None,
    gray_levels: int = GLCM_GRAY_LEVELS,
    distances: List[int] = None,
    angles_deg: List[int] = None,
    properties: List[str] = None,
    glcm_resize: Tuple[int, int] = GLCM_RESIZE,
) -> Tuple[np.ndarray, List[str]]:
    """
    Ekstrak fitur tekstur GLCM dari gambar grayscale.
    
    Langkah:
    1. Resize ke resolusi GLCM (untuk kecepatan).
    2. Kuantisasi ke gray_levels tingkat.
    3. Hitung GLCM untuk beberapa jarak dan sudut.
    4. Ekstrak properti dan agregasi (mean per properti).
    
    Returns:
        (feature_vector, feature_names)
    """
    if distances is None:
        distances = GLCM_DISTANCES
    if angles_deg is None:
        angles_deg = GLCM_ANGLES_DEG
    if properties is None:
        properties = GLCM_PROPERTIES
    
    # Konversi sudut ke radian
    angles_rad = [np.deg2rad(a) for a in angles_deg]
    
    # Resize untuk kecepatan
    if glcm_resize is not None:
        img_proc = cv2.resize(img_gray, glcm_resize, interpolation=cv2.INTER_AREA)
        # Resize mask jika ada
        if mask is not None:
            mask_proc = cv2.resize(
                mask.astype(np.uint8), glcm_resize, 
                interpolation=cv2.INTER_NEAREST
            ).astype(np.bool_)
        else:
            mask_proc = None
    else:
        img_proc = img_gray.copy()
        mask_proc = mask
    
    # Terapkan mask: set piksel background ke 0
    if mask_proc is not None:
        img_proc = img_proc.copy()
        img_proc[~mask_proc] = 0
    
    # Kuantisasi
    if img_proc.max() > 0:
        img_quant = (img_proc.astype(np.float64) / img_proc.max() * (gray_levels - 1)).astype(np.uint8)
    else:
        img_quant = np.zeros_like(img_proc, dtype=np.uint8)
    
    # Hitung GLCM
    try:
        glcm = graycomatrix(
            img_quant,
            distances=distances,
            angles=angles_rad,
            levels=gray_levels,
            symmetric=True,
            normed=True,
        )
    except Exception:
        # Fallback: return zeros
        n_features = len(properties) * len(distances)
        feature_names = []
        for prop in properties:
            for d in distances:
                feature_names.append(f'GLCM_{prop}_d{d}')
        return np.zeros(n_features, dtype=np.float64), feature_names
    
    # Ekstrak properti
    features = []
    feature_names = []
    
    for prop in properties:
        try:
            prop_values = graycoprops(glcm, prop)  # shape: (n_distances, n_angles)
        except Exception:
            prop_values = np.zeros((len(distances), len(angles_rad)))
        
        # Agregasi: rata-rata semua sudut per jarak
        for d_idx, d in enumerate(distances):
            val = np.mean(prop_values[d_idx, :])
            
            # Handle NaN/Inf
            if not np.isfinite(val):
                val = 0.0
            
            features.append(float(val))
            feature_names.append(f'GLCM_{prop}_d{d}')
    
    return np.array(features, dtype=np.float64), feature_names


# ==================== GABUNGAN ====================

def extract_all_features(
    preprocessed: Dict,
    color_space: str,
    use_glcm: bool = False,
    bins: int = HISTOGRAM_BINS,
) -> Tuple[np.ndarray, List[str]]:
    """
    Ekstrak semua fitur dari satu gambar yang sudah di-preprocessing.
    
    Args:
        preprocessed: output dari preprocess_image()
        color_space: 'RGB', 'HSV', atau 'CIELAB'
        use_glcm: apakah menambahkan fitur GLCM
    
    Returns:
        (feature_vector, feature_names)
    """
    mask = preprocessed['mask']
    
    # Fitur warna
    color_features, color_names = extract_color_features(
        color_space,
        preprocessed['img_rgb'],
        preprocessed['img_hsv'],
        preprocessed['img_lab'],
        mask,
        bins,
    )
    
    if use_glcm:
        # Fitur tekstur
        glcm_features, glcm_names = extract_glcm_features(
            preprocessed['img_gray'],
            mask,
        )
        
        # Gabung
        features = np.concatenate([color_features, glcm_features])
        names = color_names + glcm_names
    else:
        features = color_features
        names = color_names
    
    # Validasi: pastikan tidak ada NaN atau Inf
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    return features, names


def get_feature_dimension(color_space: str, use_glcm: bool = False, bins: int = HISTOGRAM_BINS) -> int:
    """Hitung dimensi fitur yang dihasilkan."""
    # Warna: 3 kanal × (3 stats + bins histogram)
    color_dim = 3 * (3 + bins)
    
    if use_glcm:
        glcm_dim = len(GLCM_PROPERTIES) * len(GLCM_DISTANCES)
        return color_dim + glcm_dim
    
    return color_dim
