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
    iterations: int = 5,
) -> Tuple[np.ndarray, bool]:
    """
    Segmentasi foreground menggunakan GrabCut.
    
    Returns:
        (mask, success): mask boolean dan apakah segmentasi berhasil.
    """
    h, w = img_bgr.shape[:2]
    
    # Inisialisasi mask
    mask = np.zeros((h, w), np.uint8)
    
    # Margin untuk bounding box proporsional terhadap dimensi citra
    margin_x = max(2, int(w * 0.03))
    margin_y = max(2, int(h * 0.03))
    rect = (margin_x, margin_y, max(1, w - 2 * margin_x), max(1, h - 2 * margin_y))
    
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
        if fg_ratio < 0.03 or fg_ratio > 0.98:
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
    
    h, w = img.shape[:2]
    target_w, target_h = target_size
    
    # Hitung skala aspek rasio
    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    
    # Resize menjaga rasio aspek (sebelum diberi padding hitam)
    img_scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Segmentasi dijalankan pada citra asli/scaled SEBELUM padding buatan
    # agar tepi latar belakang asli foto tidak terganggu oleh warna padding hitam
    if use_segmentation:
        mask_scaled, seg_success = segment_foreground(img_scaled)
    else:
        mask_scaled = np.ones((new_h, new_w), dtype=np.bool_)
        seg_success = True
    
    # Buat canvas target_size dengan padding
    canvas_bgr = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    canvas_mask = np.zeros((target_h, target_w), dtype=np.bool_)
    
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    
    canvas_bgr[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = img_scaled
    canvas_mask[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = mask_scaled
    
    # Konversi warna
    img_rgb = convert_color_space(canvas_bgr, 'RGB')
    img_hsv = convert_color_space(canvas_bgr, 'HSV')
    img_lab = convert_color_space(canvas_bgr, 'CIELAB')
    img_gray = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2GRAY)
    
    return {
        'path': path,
        'img_bgr': canvas_bgr,
        'img_rgb': img_rgb,
        'img_hsv': img_hsv,
        'img_lab': img_lab,
        'img_gray': img_gray,
        'mask': canvas_mask,
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
