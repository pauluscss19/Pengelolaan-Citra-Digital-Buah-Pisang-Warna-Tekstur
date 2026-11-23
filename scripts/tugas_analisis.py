"""
scripts/tugas_analisis.py
Script pengujian ekstraksi fitur GLCM dan berbagai operator deteksi tepi
pada dataset klasifikasi kematangan buah pisang.
Menghasilkan data kuantitatif dan visualisasi untuk tugas pengganti PCD.
"""

import os
import glob
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops

# Konfigurasi path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "tugas_ekstraksi_dan_tepi")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATASET_TEST_DIR = os.path.join(PROJECT_ROOT, "dataset", "test")
CLASSES = ["unripe", "ripe", "overripe", "rotten"]
CLASS_LABELS = {
    "unripe": "Mentah (Unripe)",
    "ripe": "Matang (Ripe)",
    "overripe": "Lewat Matang (Overripe)",
    "rotten": "Busuk (Rotten)"
}

# 1. Ekstrak fitur GLCM pada sejumlah sampel untuk analisis statistik
print("--- [1] Menghitung Statistik Fitur Tekstur GLCM ---")
glcm_records = []

# Ambil hingga 30 sampel per kelas untuk statistik representatif
SAMPLES_PER_CLASS = 30
np.random.seed(42)

sample_images = {}

for cls in CLASSES:
    cls_folder = os.path.join(DATASET_TEST_DIR, cls)
    image_paths = sorted(glob.glob(os.path.join(cls_folder, "*.jpg")))
    if not image_paths:
        continue
    
    # Simpan satu sampel representatif untuk visualisasi
    sample_images[cls] = image_paths[0]
    
    chosen_paths = image_paths[:SAMPLES_PER_CLASS]
    for p in chosen_paths:
        img_bgr = cv2.imread(p)
        if img_bgr is None:
            continue
        
        # Resize standar
        img_resized = cv2.resize(img_bgr, (128, 128))
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        
        # Kuantisasi 32 level
        gray_quant = (gray // 8).astype(np.uint8)
        
        # Hitung GLCM (jarak d=1, 3; sudut 0, 45, 90, 135 deg)
        distances = [1, 3]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        glcm = graycomatrix(gray_quant, distances=distances, angles=angles, levels=32, symmetric=True, normed=True)
        
        # Haralick properties
        contrast = float(np.mean(graycoprops(glcm, 'contrast')))
        dissimilarity = float(np.mean(graycoprops(glcm, 'dissimilarity')))
        homogeneity = float(np.mean(graycoprops(glcm, 'homogeneity')))
        energy = float(np.mean(graycoprops(glcm, 'energy')))
        correlation = float(np.mean(graycoprops(glcm, 'correlation')))
        
        # Hitung Edge Density menggunakan Canny (50, 150)
        edges_canny = cv2.Canny(gray, 50, 150)
        edge_density_canny = float(np.count_nonzero(edges_canny)) / edges_canny.size
        
        # Hitung Sobel Magnitude
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_mag = np.hypot(sobelx, sobely)
        sobel_mean = float(np.mean(sobel_mag))
        
        glcm_records.append({
            "class": cls,
            "class_label": CLASS_LABELS[cls],
            "contrast": contrast,
            "dissimilarity": dissimilarity,
            "homogeneity": homogeneity,
            "energy": energy,
            "correlation": correlation,
            "edge_density_canny": edge_density_canny,
            "sobel_mean_magnitude": sobel_mean
        })

df = pd.DataFrame(glcm_records)
csv_path = os.path.join(OUTPUT_DIR, "quantitative_analysis.csv")
df.to_csv(csv_path, index=False)
print(f"Data kuantitatif disimpan di: {csv_path}")

# Tampilkan ringkasan rata-rata
summary = df.groupby("class_label")[["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "edge_density_canny", "sobel_mean_magnitude"]].mean().reset_index()
print("\n--- Ringkasan Rata-rata Fitur per Kelas ---")
print(summary.to_string(index=False))

# Simpan tabel summary
summary.to_csv(os.path.join(OUTPUT_DIR, "summary_per_class.csv"), index=False)

# 2. Visualisasi Plot Perbandingan Fitur GLCM
print("\n--- [2] Membuat Visualisasi Plot Fitur GLCM ---")
plt.figure(figsize=(15, 8))
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

metrics = [
    ('contrast', 'Contrast (Variasi Lokal)', 'Blues'),
    ('dissimilarity', 'Dissimilarity (Ketidaksamaan)', 'Purples'),
    ('homogeneity', 'Homogeneity (Keseragaman)', 'Greens'),
    ('energy', 'Energy / ASM (Keteraturan)', 'Oranges'),
    ('edge_density_canny', 'Canny Edge Density (Kerapatan Tepi)', 'Reds'),
    ('sobel_mean_magnitude', 'Sobel Mean Magnitude (Kekuatan Tepi)', 'Greys')
]

for idx, (col, title, cmap) in enumerate(metrics, 1):
    plt.subplot(2, 3, idx)
    means = df.groupby('class_label')[col].mean()
    # Urutkan berdasarkan tahapan kematangan
    order = [CLASS_LABELS['unripe'], CLASS_LABELS['ripe'], CLASS_LABELS['overripe'], CLASS_LABELS['rotten']]
    means = means.reindex(order)
    
    bars = plt.bar(range(len(order)), means.values, color=['#2ca02c', '#bcbd22', '#ff7f0e', '#8c564b'], alpha=0.85, edgecolor='black', linewidth=1)
    plt.xticks(range(len(order)), ['Unripe', 'Ripe', 'Overripe', 'Rotten'], fontsize=9, fontweight='bold')
    plt.title(title, fontsize=11, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Label nilai di atas bar
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., h + 0.01*max(means.values), f'{h:.3f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plot_glcm_path = os.path.join(OUTPUT_DIR, "glcm_and_edge_metrics_comparison.png")
plt.savefig(plot_glcm_path, dpi=300)
plt.close()
print(f"Plot metrik disimpan di: {plot_glcm_path}")

# 3. Visualisasi Deteksi Tepi pada Sampel Riil 4 Kelas
print("\n--- [3] Menghasilkan Visualisasi Deteksi Tepi (Sobel, Canny, Laplacian) ---")
fig, axes = plt.subplots(nrows=4, ncols=6, figsize=(18, 12))

col_titles = [
    "1. Asli (RGB)", 
    "2. Grayscale", 
    "3. Sobel X (|Gx|)", 
    "4. Sobel Y (|Gy|)", 
    "5. Sobel Magnitude", 
    "6. Canny (50, 150)"
]

for row_idx, cls in enumerate(CLASSES):
    img_path = sample_images[cls]
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (160, 160))
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    
    # Sobel
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    abs_sobelx = np.uint8(np.absolute(sobelx))
    abs_sobely = np.uint8(np.absolute(sobely))
    sobel_mag = np.uint8(np.clip(np.hypot(sobelx, sobely), 0, 255))
    
    # Canny
    canny = cv2.Canny(gray, 50, 150)
    
    images_to_show = [img_rgb, gray, abs_sobelx, abs_sobely, sobel_mag, canny]
    
    for col_idx, im in enumerate(images_to_show):
        ax = axes[row_idx, col_idx]
        if col_idx == 0:
            ax.imshow(im)
            ax.set_ylabel(CLASS_LABELS[cls], fontsize=11, fontweight='bold')
        else:
            ax.imshow(im, cmap='gray')
            
        if row_idx == 0:
            ax.set_title(col_titles[col_idx], fontsize=11, fontweight='bold')
            
        ax.set_xticks([])
        ax.set_yticks([])

plt.suptitle("Perbandingan Hasil Operator Deteksi Tepi pada Citra Buah Pisang (4 Tingkat Kematangan)", fontsize=14, fontweight='bold', y=0.99)
plt.tight_layout()
plot_edges_path = os.path.join(OUTPUT_DIR, "edge_detection_comparison.png")
plt.savefig(plot_edges_path, dpi=300)
plt.close()
print(f"Plot deteksi tepi disimpan di: {plot_edges_path}")

print("--- Semua proses analisis berhasil dijalankan! ---")
