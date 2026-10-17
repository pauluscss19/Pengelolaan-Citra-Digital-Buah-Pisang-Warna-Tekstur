"""
preprocessing.py
Modul preprocessing citra: resize, konversi ruang warna, dan segmentasi foreground.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, Dict

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import TARGET_SIZE


def read_image(path: str) -> Optional[np.ndarray]:
    """
    Baca gambar dari file menggunakan OpenCV.
    
    Returns:
        Gambar BGR atau None jika gagal.
    """
    try:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            return None
        if img.size == 0:
            return None
        return img
    except Exception:
        return None


def resize_image(img: np.ndarray, target_size: Tuple[int, int] = TARGET_SIZE) -> np.ndarray:
    """
    Resize gambar ke target_size dengan menjaga aspek rasio menggunakan padding.
    """
    h, w = img.shape[:2]
    target_w, target_h = target_size
    
    # Hitung rasio skala
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Buat canvas dengan padding
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    
    return canvas


def convert_color_space(img_bgr: np.ndarray, color_space: str) -> np.ndarray:
    """
    Konversi gambar BGR ke ruang warna yang ditentukan.
    
    Args:
        img_bgr: gambar BGR
        color_space: 'RGB', 'HSV', atau 'CIELAB'
    
    Returns:
        Gambar dalam ruang warna target.
    """
    if color_space == 'RGB':
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    elif color_space == 'HSV':
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    elif color_space == 'CIELAB':
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    else:
        raise ValueError(f"Ruang warna tidak dikenal: {color_space}")


def segment_foreground(
    img_bgr: np.ndarray,
    iterations: int = 3,
) -> Tuple[np.ndarray, bool]:
    """
    Segmentasi foreground menggunakan GrabCut.
    
    Returns:
        (mask, success): mask boolean dan apakah segmentasi berhasil.
    """
    h, w = img_bgr.shape[:2]
    
    # Inisialisasi mask
    mask = np.zeros((h, w), np.uint8)
    
    # Margin untuk bounding box
    margin = max(5, int(min(h, w) * 0.05))
    rect = (margin, margin, w - 2 * margin, h - 2 * margin)
    
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    
    try:
        cv2.grabCut(img_bgr, mask, rect, bgd_model, fgd_model,
                     iterations, cv2.GC_INIT_WITH_RECT)
        
        # Mask foreground: GC_FGD=1 dan GC_PR_FGD=3
        fg_mask = np.where(
            (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 
            True, False
        ).astype(np.bool_)
        
        # Periksa apakah segmentasi menghasilkan area yang cukup
        fg_ratio = np.sum(fg_mask) / fg_mask.size
        if fg_ratio < 0.05 or fg_ratio > 0.98:
            # Segmentasi gagal: terlalu sedikit atau hampir seluruh gambar
            return np.ones((h, w), dtype=np.bool_), False
        
        return fg_mask, True
        
    except Exception:
        # Fallback: gunakan seluruh gambar
        return np.ones((h, w), dtype=np.bool_), False


def preprocess_image(
    path: str,
    target_size: Tuple[int, int] = TARGET_SIZE,
    use_segmentation: bool = False,
) -> Optional[Dict]:
    """
    Pipeline preprocessing lengkap untuk satu gambar.
    
    Returns:
        Dict berisi gambar BGR, RGB, mask, dan status preprocessing.
        None jika gambar tidak bisa dibaca.
    """
    img = read_image(path)
    if img is None:
        return None
    
    # Resize
    img_resized = resize_image(img, target_size)
    
    # Segmentasi
    if use_segmentation:
        mask, seg_success = segment_foreground(img_resized)
    else:
        mask = np.ones(img_resized.shape[:2], dtype=np.bool_)
        seg_success = True
    
    # Konversi warna
    img_rgb = convert_color_space(img_resized, 'RGB')
    img_hsv = convert_color_space(img_resized, 'HSV')
    img_lab = convert_color_space(img_resized, 'CIELAB')
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    return {
        'path': path,
        'img_bgr': img_resized,
        'img_rgb': img_rgb,
        'img_hsv': img_hsv,
        'img_lab': img_lab,
        'img_gray': img_gray,
        'mask': mask,
        'segmentation_success': seg_success,
        'target_size': target_size,
        'use_segmentation': use_segmentation,
    }


def apply_mask(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Terapkan mask pada gambar, menghasilkan piksel-piksel foreground saja."""
    if len(img.shape) == 3:
        pixels = img[mask]
    else:
        pixels = img[mask]
    return pixels
