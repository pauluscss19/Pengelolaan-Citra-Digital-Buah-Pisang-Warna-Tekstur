"""
utils.py
Utilitas umum untuk project.
"""

import os
import hashlib
from typing import List


def file_hash(path: str) -> str:
    """Hitung MD5 hash dari file."""
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def safe_filename(name: str) -> str:
    """Konversi string menjadi nama file yang aman."""
    return "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in name)


def format_duration(seconds: float) -> str:
    """Format durasi ke string yang mudah dibaca."""
    if seconds < 60:
        return f"{seconds:.1f} detik"
    elif seconds < 3600:
        m = int(seconds // 60)
        s = seconds % 60
        return f"{m} menit {s:.0f} detik"
    else:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h} jam {m} menit"
