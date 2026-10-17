"""
dataset_loader.py
Modul untuk memuat dan memeriksa dataset klasifikasi kematangan pisang.
Mendukung struktur folder: train/valid(val/validation)/test dengan subfolder per kelas.
"""

import os
import hashlib
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import (
    DEFAULT_DATASET_PATH,
    SUPPORTED_EXTENSIONS,
    VALID_FOLDER_NAMES,
    MATURITY_GROUP_MAPPING,
    QUICK_MODE_SAMPLES_PER_CLASS,
    RANDOM_SEED,
)


def find_split_folders(dataset_path: str) -> Dict[str, str]:
    """
    Mencari folder train, valid/val/validation, dan test dalam dataset_path.
    
    Returns:
        Dict mapping split name ('train', 'valid', 'test') ke path absolut.
    """
    splits = {}
    
    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(
            f"Direktori dataset tidak ditemukan: {dataset_path}"
        )
    
    contents = os.listdir(dataset_path)
    
    # Cari train
    if 'train' in contents:
        splits['train'] = os.path.join(dataset_path, 'train')
    
    # Cari valid/val/validation
    for name in VALID_FOLDER_NAMES:
        if name in contents:
            splits['valid'] = os.path.join(dataset_path, name)
            break
    
    # Cari test
    if 'test' in contents:
        splits['test'] = os.path.join(dataset_path, 'test')
    
    return splits


def get_classes_from_split(split_path: str) -> List[str]:
    """
    Membaca nama kelas dari subfolder di dalam split.
    
    Returns:
        List nama kelas yang terurut secara alfabet.
    """
    if not os.path.isdir(split_path):
        return []
    
    classes = sorted([
        d for d in os.listdir(split_path)
        if os.path.isdir(os.path.join(split_path, d))
    ])
    return classes


def count_images_in_folder(folder_path: str) -> int:
    """Hitung jumlah file gambar di folder (tanpa rekursi)."""
    if not os.path.isdir(folder_path):
        return 0
    
    count = 0
    for f in os.listdir(folder_path):
        ext = os.path.splitext(f)[1].lower()
        if ext in SUPPORTED_EXTENSIONS:
            count += 1
    return count


def get_image_paths(folder_path: str) -> List[str]:
    """Dapatkan semua path gambar di folder."""
    if not os.path.isdir(folder_path):
        return []
    
    paths = []
    for f in sorted(os.listdir(folder_path)):
        ext = os.path.splitext(f)[1].lower()
        if ext in SUPPORTED_EXTENSIONS:
            paths.append(os.path.join(folder_path, f))
    return paths


def load_dataset_info(dataset_path: str) -> Dict:
    """
    Memuat informasi lengkap dataset.
    
    Returns:
        Dict berisi:
        - splits: dict split name -> path
        - classes: list nama kelas
        - split_counts: dict split -> dict class -> count
        - total_images: int
        - warnings: list peringatan
    """
    info = {
        'dataset_path': dataset_path,
        'splits': {},
        'classes': [],
        'split_counts': {},
        'total_images': 0,
        'warnings': [],
        'is_valid': False,
    }
    
    try:
        splits = find_split_folders(dataset_path)
    except FileNotFoundError as e:
        info['warnings'].append(str(e))
        return info
    
    info['splits'] = splits
    
    if 'train' not in splits:
        info['warnings'].append("Folder 'train' tidak ditemukan dalam dataset.")
        return info
    
    # Baca kelas dari train sebagai referensi
    train_classes = get_classes_from_split(splits['train'])
    if not train_classes:
        info['warnings'].append("Tidak ada subfolder kelas dalam folder train.")
        return info
    
    info['classes'] = train_classes
    
    # Hitung gambar per split per kelas
    total = 0
    for split_name, split_path in splits.items():
        split_classes = get_classes_from_split(split_path)
        counts = OrderedDict()
        
        for cls in train_classes:
            cls_path = os.path.join(split_path, cls)
            if os.path.isdir(cls_path):
                c = count_images_in_folder(cls_path)
                counts[cls] = c
                total += c
            else:
                counts[cls] = 0
                info['warnings'].append(
                    f"Kelas '{cls}' tidak ditemukan di split '{split_name}'."
                )
        
        # Cek kelas tambahan yang ada di split ini tapi tidak di train
        extra_classes = set(split_classes) - set(train_classes)
        if extra_classes:
            info['warnings'].append(
                f"Split '{split_name}' memiliki kelas tambahan: {extra_classes}"
            )
        
        info['split_counts'][split_name] = counts
    
    info['total_images'] = total
    info['is_valid'] = total > 0
    
    if 'valid' not in splits:
        info['warnings'].append(
            "Folder validasi tidak ditemukan. "
            "Evaluasi validasi tidak akan tersedia."
        )
    if 'test' not in splits:
        info['warnings'].append(
            "Folder test tidak ditemukan. "
            "Evaluasi akhir tidak akan tersedia."
        )
    
    return info


def load_split_data(
    dataset_path: str,
    split: str,
    classes: List[str],
    label_mode: str = 'original',
    max_per_class: Optional[int] = None,
    seed: int = RANDOM_SEED,
) -> Tuple[List[str], List[str]]:
    """
    Memuat path gambar dan label untuk satu split.
    
    Args:
        dataset_path: root dataset
        split: 'train', 'valid', atau 'test'
        classes: daftar nama kelas
        label_mode: 'original' atau 'grouped'
        max_per_class: batas sampel per kelas (untuk mode cepat)
        seed: random seed untuk sampling
    
    Returns:
        (image_paths, labels)
    """
    splits = find_split_folders(dataset_path)
    
    if split not in splits:
        raise ValueError(f"Split '{split}' tidak ditemukan di dataset.")
    
    split_path = splits[split]
    all_paths = []
    all_labels = []
    
    rng = np.random.RandomState(seed)
    
    for cls in classes:
        cls_path = os.path.join(split_path, cls)
        paths = get_image_paths(cls_path)
        
        if max_per_class is not None and len(paths) > max_per_class:
            indices = rng.choice(len(paths), max_per_class, replace=False)
            indices.sort()
            paths = [paths[i] for i in indices]
        
        if label_mode == 'grouped':
            label = MATURITY_GROUP_MAPPING.get(cls, cls)
        else:
            label = cls
        
        all_paths.extend(paths)
        all_labels.extend([label] * len(paths))
    
    return all_paths, all_labels


def check_duplicate_filenames(dataset_path: str) -> List[str]:
    """
    Periksa duplikasi nama file antar split.
    
    Returns:
        List peringatan jika ada nama file yang sama di split berbeda.
    """
    splits = find_split_folders(dataset_path)
    warnings = []
    
    split_files = {}
    for split_name, split_path in splits.items():
        files = set()
        classes = get_classes_from_split(split_path)
        for cls in classes:
            cls_path = os.path.join(split_path, cls)
            for f in os.listdir(cls_path):
                ext = os.path.splitext(f)[1].lower()
                if ext in SUPPORTED_EXTENSIONS:
                    files.add(f)
        split_files[split_name] = files
    
    split_names = list(split_files.keys())
    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            s1, s2 = split_names[i], split_names[j]
            overlap = split_files[s1] & split_files[s2]
            if overlap:
                warnings.append(
                    f"⚠️ Ditemukan {len(overlap)} nama file yang sama "
                    f"antara '{s1}' dan '{s2}'. "
                    f"Potensi kebocoran data (data leakage)."
                )
    
    return warnings


def get_label_mapping_info(classes: List[str], label_mode: str) -> Dict:
    """
    Dapatkan informasi pemetaan label yang digunakan.
    
    Returns:
        Dict berisi mapping dan kelas output.
    """
    if label_mode == 'grouped':
        mapping = {}
        for cls in classes:
            mapped = MATURITY_GROUP_MAPPING.get(cls, cls)
            mapping[cls] = mapped
        output_classes = sorted(set(mapping.values()))
        return {
            'mode': 'grouped',
            'mapping': mapping,
            'output_classes': output_classes,
            'description': 'Mode kelompok kematangan (Mentah/Matang/Terlalu Matang/Busuk)'
        }
    else:
        return {
            'mode': 'original',
            'mapping': {cls: cls for cls in classes},
            'output_classes': classes,
            'description': 'Mode kelas asli (sesuai nama folder dataset)'
        }
