# 🍌 Klasifikasi Tingkat Kematangan Buah Pisang

## Analisis Pengaruh Ruang Warna dan Fusi Fitur Warna–Tekstur Menggunakan GLCM terhadap Klasifikasi Tingkat Kematangan Buah Pisang dengan Support Vector Machine (SVM)

**Mata Kuliah:** Pengelolaan Citra Digital

---

## 📋 Deskripsi Penelitian

Penelitian ini bertujuan untuk:
1. Mengidentifikasi tingkat kematangan pisang berdasarkan fitur citra.
2. Membandingkan ruang warna **RGB**, **HSV**, dan **CIELAB**.
3. Menguji apakah penambahan fitur tekstur **GLCM** meningkatkan performa klasifikasi.
4. Menggunakan algoritma **SVM** yang sama untuk seluruh eksperimen agar perbandingan adil.
5. Mengevaluasi hasil menggunakan accuracy, precision, recall, macro F1-score, dan confusion matrix.

## 📂 Sumber Dataset

**Dataset:** Banana Ripeness Classification v6  
**Sumber:** [Roboflow Universe](https://universe.roboflow.com/roboflow-universe-projects/banana-ripeness-classification)  
**Lisensi:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) (Creative Commons Attribution 4.0 International)  
**Atribusi:** Dataset disediakan oleh kontributor Roboflow Universe.

### Struktur Dataset
```
dataset/
├── train/
│   ├── overripe/
│   ├── ripe/
│   ├── rotten/
│   └── unripe/
├── valid/
│   ├── overripe/
│   ├── ripe/
│   ├── rotten/
│   └── unripe/
└── test/
    ├── overripe/
    ├── ripe/
    ├── rotten/
    └── unripe/
```

Distribusi (dari dataset v6):
- **Train:** 11.793 gambar
- **Valid:** 1.123 gambar
- **Test:** 562 gambar
- **Total:** 13.478 gambar

## 🏗️ Arsitektur Sistem

```
📷 Input Gambar
    ↓
🔄 Preprocessing (Resize, Konversi Warna, Segmentasi Opsional)
    ↓
🎨 Ekstraksi Fitur Warna (RGB / HSV / CIELAB)
    ↓                           ↓
📐 Fitur Warna Saja        ➕ Fusi dengan Fitur GLCM
    ↓                           ↓
🤖 Klasifikasi SVM (StandardScaler → SVC RBF)
    ↓
📊 Evaluasi (Accuracy, Precision, Recall, F1, Confusion Matrix)
```

## 🧪 Enam Eksperimen

| No | Eksperimen | Ruang Warna | Fitur Warna | GLCM (Tekstur) | Total Dimensi |
|---|---|---|---|---|---|
| 1 | RGB + SVM | RGB | 33 fitur | ❌ (0) | 33 fitur |
| 2 | HSV + SVM | HSV | 33 fitur | ❌ (0) | 33 fitur |
| 3 | CIELAB + SVM | CIELAB | 33 fitur | ❌ (0) | 33 fitur |
| 4 | RGB + GLCM + SVM | RGB | 33 fitur | ✅ (20) | 53 fitur |
| 5 | HSV + GLCM + SVM | HSV | 33 fitur | ✅ (20) | 53 fitur |
| 6 | CIELAB + GLCM + SVM | CIELAB | 33 fitur | ✅ (20) | 53 fitur |

---

## 🔬 Rincian dan Fungsi Fitur Ekstraksi Citra

Ekstraksi fitur dirancang secara deterministik tanpa deep learning, berfokus pada teknik pengolahan citra digital klasik (*handcrafted features*):

### A. Fitur Warna (33 Dimensi per Ruang Warna)
Fitur warna diekstrak untuk mengidentifikasi tingkat kematangan melalui degradasi klorofil (warna hijau) menjadi karotenoid/lutein (warna kuning) hingga timbulnya bintik nekrosis melanin (warna cokelat/hitam).

Setiap ruang warna memiliki 3 kanal, di mana setiap kanal diekstrak **11 nilai**:
- **3 Nilai Statistik Kanal:**
  - **Mean ($\mu$):** Nilai rata-rata intensitas kanal, mencerminkan kecenderungan warna dominan buah.
  - **Standard Deviation ($\sigma$):** Mengukur variasi sebaran nilai piksel pada kanal tersebut.
  - **Median:** Nilai tengah distribusi intensitas kanal, tahan terhadap noise/outlier kilau cahaya (specular reflection).
- **8 Nilai Histogram Ternormalisasi:**
  - Mengelompokkan sebaran intensitas menjadi 8 rentang bin yang dinormalisasi ($\sum = 1.0$), menangkap sebaran spektrum warna tanpa terpengaruh ukuran piksel objek.

#### Ruang Warna yang Dianalisis:
1. **RGB (Red, Green, Blue):**
   - Kanal R, G, dan B (range 0–255).
   - Sensitif terhadap intensitas cahaya dan bayangan.
2. **HSV (Hue, Saturation, Value):**
   - **Hue ($H$, range 0–179 di OpenCV):** Merepresentasikan warna murni (sudut warna). Sangat efektif membedakan hijau (mentah) dan kuning (matang) terlepas dari tingkat pencahayaan.
   - **Saturation ($S$, range 0–255):** Kejenuhan/kemurnian warna kulit pisang.
   - **Value ($V$, range 0–255):** Kecerahan relatif warna.
3. **CIELAB ($L^*, a^*, b^*$):**
   - Didesain mendekati persepsi penglihatan manusia (*perceptually uniform*).
   - **$L^*$ (Lightness, range 0–255):** Tingkat kecerahan (gelap ke terang).
   - **$a^*$ (range 0–255):** Sumbu gradien Hijau (nilai rendah) ke Merah (nilai tinggi). Sangat sensitif terhadap hilangnya klorofil hijau saat pisang matang.
   - **$b^*$ (range 0–255):** Sumbu gradien Biru (nilai rendah) ke Kuning (nilai tinggi). Sangat sensitif terhadap peningkatan warna kuning saat pembentukan pigmen matang.

---

### B. Fitur Tekstur GLCM (20 Dimensi)
Gray-Level Co-occurrence Matrix (GLCM) merepresentasikan hubungan spasial pasangan piksel bertetangga dengan orientasi dan jarak tertentu pada citra grayscale kulit pisang.

#### Parameter GLCM:
- **Kuantisasi Grayscale:** Dikuantisasi ke 32 level abu-abu untuk mempercepat komputasi dan mengurangi matriks renggang (*sparse matrix*).
- **Jarak Piksel ($d$):** Menggunakan 2 jarak: $d = 1$ dan $d = 3$ piksel (menangkap tekstur mikro dan makro).
- **Orientasi Sudut ($\theta$):** 4 arah: $0^\circ, 45^\circ, 90^\circ, 135^\circ$.
- **Invarian Arah (Directional Invariance):** Nilai dari ke-4 sudut orientasi dirata-ratakan (*mean across angles*) untuk tiap jarak $d$.

#### 5 Properti Tekstur Haralick yang Dihitung:
1. **Contrast (Kontras):**
   $$\sum_{i,j} |i - j|^2 \cdot P(i,j)$$
   *Fungsi:* Mengukur variasi intensitas lokal antar piksel bertetangga. Nilai kontras tinggi mengindikasikan adanya bintik-bintik cokelat/hitam (senescence spot) pada pisang *overripe* atau *rotten*.
2. **Dissimilarity (Ketidaksamaan):**
   $$\sum_{i,j} |i - j| \cdot P(i,j)$$
   *Fungsi:* Mirip dengan kontras namun berbobot linier, mengukur perbedaan keabuan lokal pada permukaan kulit pisang.
3. **Homogeneity (Homogenitas / IDM):**
   $$\sum_{i,j} \frac{P(i,j)}{1 + (i - j)^2}$$
   *Fungsi:* Mengukur keseragaman permukaan kulit. Pisang mentah (*unripe*) memiliki homogenitas tinggi karena kulitnya halus dan berwarna hijau rata tanpa bercak.
4. **Energy (Energi / Angular Second Moment):**
   $$\sum_{i,j} P(i,j)^2$$
   *Fungsi:* Mengukur keteraturan tekstur. Bernilai tinggi saat distribusi GLCM terpusat pada nilai piksel seragam.
5. **Correlation (Korelasi):**
   $$\sum_{i,j} \frac{(i - \mu_i)(j - \mu_j) P(i,j)}{\sigma_i \sigma_j}$$
   *Fungsi:* Mengukur keterikatan linier nilai derajat abu-abu antar piksel. Mengindikasikan pola struktur serat kulit buah pisang.

**Perhitungan Dimensi GLCM:**
$$5 \text{ properti} \times 2 \text{ jarak (d=1, d=3)} = 20 \text{ fitur}$$

---

### C. Fusi Fitur Warna–Tekstur (Early Fusion)
Pada eksperimen 4, 5, dan 6, vektor fitur warna digabungkan secara langsung dengan vektor fitur tekstur GLCM:
$$\mathbf{x}_{\text{fusi}} = [\mathbf{x}_{\text{warna}} \, (33) \ \Vert \ \mathbf{x}_{\text{GLCM}} \, (20)] \in \mathbb{R}^{53}$$
Seluruh fitur distandarisasi menggunakan `StandardScaler` ($\mu=0, \sigma=1$) sebelum masuk ke kernel SVM RBF agar fitur dengan skala besar tidak mendominasi fitur bernilai kecil.

---

## 💻 Fitur-Fitur Fungsional Aplikasi Streamlit

Aplikasi ini dilengkapi antarmuka web interaktif berbasis Streamlit yang mencakup 7 modul utama:

1. **🏠 Halaman Beranda:**
   - Ringkasan tujuan penelitian, arsitektur sistem pengolahan citra, status dataset, dan ketersediaan model hasil pelatihan.
2. **📁 Manajemen Dataset:**
   - Verifikasi otomatis struktur folder (`train/`, `valid/`, `test/`).
   - Pilihan mode label:
     - *Mode Kelas Asli* (menggunakan kelas bawaan dataset, misal: freshripe, freshunripe, overripe, ripe, rotten, unripe).
     - *Mode Kelompok Kematangan* (pemetaan akademis ke 4 kelas: Mentah, Matang, Terlalu Matang, Busuk).
   - Diagram batang distribusi jumlah gambar per kelas dan galeri pratinjau gambar acak.
3. **🔬 Preprocessing & Ekstraksi Fitur Interaktif:**
   - Pratinjau hasil resize proporsional dan segmentasi latar belakang menggunakan OpenCV GrabCut.
   - Inspeksi perbandingan citra asli pada ruang warna RGB, HSV, dan CIELAB.
   - Tampilan visual plot histogram warna tiap kanal dan nilai numerik 5 properti GLCM secara langsung.
4. **⚙️ Pelatihan Model & Eksperimen:**
   - **Mode Cepat (Uji Coba):** Memilih subset stratified terkontrol (misal 50 sampel/kelas) untuk smoke testing 6 model dalam waktu singkat.
   - **Mode Lengkap:** Pelatihan menyeluruh pada seluruh data train untuk hasil eksperimen final.
   - Sistem **Feature Caching (`.npz`)**: Fitur yang sudah diekstrak disimpan lokal sehingga pelatihan ulang tidak perlu memproses ulang ribuan citra dari awal.
   - Pelatihan otomatis 6 pipeline SVM secara konsisten dan pencatatan waktu komputasi.
5. **📊 Evaluasi Komprehensif:**
   - Tabel perbandingan metrik 6 eksperimen: Accuracy, Macro Precision, Macro Recall, dan Macro F1-Score.
   - Peringkat metode terbaik berdasarkan Macro F1.
   - Grafik batang visualisasi performa model (tersimpan otomatis di `results/plots/`).
   - Visualisasi Confusion Matrix per model dan Classification Report per kelas.
   - Pembuatan otomatis dokumen laporan penelitian lengkap berformat Markdown (`results/laporan_hasil.md`).
6. **🔮 Prediksi Gambar Baru:**
   - Unggah gambar baru format `.jpg`, `.jpeg`, atau `.png`.
   - Pratinjau gambar asli, hasil segmentasi, dan ekstraksi fitur warna serta GLCM.
   - Prediksi kematangan pisang disertai skor estimasi keyakinan (*confidence probability score*).
   - Perbandingan prediksi silang dari seluruh model yang telah dilatih.
7. **📚 Metodologi & Edukasi:**
   - Penjelasan teori ruang warna (RGB, HSV, CIELAB), matematika GLCM Haralick, dan algoritma Support Vector Machine (SVM).
   - Kutipan atribusi dataset sesuai lisensi CC BY 4.0.


## 📁 Struktur Project

```
Pisang/
├── app.py                  # Aplikasi Streamlit utama
├── config.py               # Konfigurasi project
├── requirements.txt        # Dependensi Python
├── README.md               # Dokumentasi ini
├── .gitignore              # Aturan Git
├── src/
│   ├── __init__.py
│   ├── dataset_loader.py   # Pemuat dan pemeriksaan dataset
│   ├── preprocessing.py    # Preprocessing citra
│   ├── feature_extraction.py # Ekstraksi fitur warna & GLCM
│   ├── model_training.py   # Pelatihan model SVM
│   ├── evaluation.py       # Evaluasi dan visualisasi
│   ├── prediction.py       # Prediksi gambar baru
│   └── utils.py            # Utilitas umum
├── tests/
│   └── test_modules.py     # Pengujian unit
├── dataset/                # Dataset gambar (tidak masuk Git)
├── artifacts/              # Model dan cache (otomatis dibuat)
│   ├── models/
│   └── feature_cache/
└── results/                # Hasil eksperimen (otomatis dibuat)
    ├── metrics_summary.csv
    ├── laporan_hasil.md
    ├── experiment_metadata.json
    ├── plots/
    ├── confusion_matrices/
    └── classification_reports/
```

## 🛠️ Instalasi (Windows PowerShell)

### 1. Buka PowerShell dan masuk ke direktori project

```powershell
cd C:\Users\ACER\Pisang
```

### 2. Buat virtual environment

```powershell
py -m venv .venv
```

### 3. Aktifkan virtual environment

```powershell
.\.venv\Scripts\Activate.ps1
```

> Jika muncul error tentang execution policy, jalankan:
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
> ```

### 4. Upgrade pip dan install dependensi

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 5. (Opsional) Install pytest untuk pengujian

```powershell
pip install pytest
```

## ▶️ Cara Menjalankan Aplikasi

```powershell
streamlit run app.py
```

Aplikasi akan terbuka di browser pada `http://localhost:8501`.

## 📖 Cara Menggunakan

### 1. Memeriksa Dataset
- Buka halaman **📂 Dataset**
- Pastikan lokasi dataset benar (default: `dataset/`)
- Periksa distribusi kelas dan jumlah gambar

### 2. Melihat Preprocessing
- Buka halaman **🔬 Preprocessing & Fitur**
- Pilih gambar untuk preview
- Periksa hasil konversi warna dan fitur yang diekstrak

### 3. Melatih Model
- Buka halaman **⚙️ Training & Eksperimen**
- Pilih **Mode Cepat** untuk pengujian (lebih cepat) atau **Mode Lengkap** untuk eksperimen akhir
- Klik **Latih Semua 6 Eksperimen**
- Tunggu hingga selesai

### 4. Melihat Evaluasi
- Buka halaman **📊 Evaluasi**
- Klik **Evaluasi Semua Model**
- Lihat tabel perbandingan, grafik, dan confusion matrix
- Hasil otomatis disimpan ke folder `results/`

### 5. Prediksi Gambar Baru
- Buka halaman **🔮 Prediksi**
- Upload gambar pisang (JPG/PNG)
- Pilih model dan lihat hasil prediksi

### 6. Membaca Hasil
- **`results/metrics_summary.csv`** - Tabel ringkasan metrik
- **`results/laporan_hasil.md`** - Laporan eksperimen otomatis
- **`results/experiment_metadata.json`** - Metadata lengkap
- **`results/plots/`** - Grafik perbandingan
- **`results/confusion_matrices/`** - Confusion matrix per eksperimen
- **`results/classification_reports/`** - Classification report per eksperimen

## 🧪 Menjalankan Test

```powershell
python -m pytest tests/ -v --tb=short
```

## ⚙️ Mengatur Lokasi Dataset

### Melalui Antarmuka
Ubah path dataset di sidebar kiri aplikasi Streamlit.

### Melalui Konfigurasi
Edit `config.py`, ubah variabel `DEFAULT_DATASET_PATH`:

```python
DEFAULT_DATASET_PATH = r"C:\path\ke\dataset\anda"
```

## ⚠️ Keterbatasan

1. **Handcrafted features** — Fitur dirancang manual, bukan deep learning.
2. **Parameter SVM tetap** — Tidak dilakukan tuning individual per eksperimen.
3. **Segmentasi sederhana** — GrabCut mungkin tidak optimal untuk semua kondisi.
4. **Resolusi** — Gambar di-resize ke 128×128 yang dapat mengurangi detail.
5. **Tidak ada augmentasi** — Tidak ditambahkan augmentasi saat training.

## 📄 Lisensi Dataset

Dataset menggunakan lisensi **CC BY 4.0**.  
Sumber: https://universe.roboflow.com/roboflow-universe-projects/banana-ripeness-classification

---

*Project ini dibuat untuk keperluan mata kuliah Pengelolaan Citra Digital.*
