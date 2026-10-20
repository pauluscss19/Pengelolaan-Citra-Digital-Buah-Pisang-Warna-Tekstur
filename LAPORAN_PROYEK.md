# LAPORAN PROYEK PENGOLAHAN CITRA DIGITAL

# Klasifikasi Tingkat Kematangan Buah Pisang Berdasarkan Analisis Ruang Warna (RGB, HSV, CIELAB) dan Fusi Fitur Tekstur GLCM Menggunakan Support Vector Machine (SVM)

---

**Mata Kuliah:** Pengolahan Citra Digital / Computer Vision  
**Repositori:** `pauluscss19/Pengelolaan-Citra-Digital-Buah-Pisang-Warna-Tekstur`  
**Dataset:** Banana Ripeness Classification Dataset v6 (Roboflow Universe, CC BY 4.0)  
**Teknologi:** Python 3.10+, OpenCV, Scikit-Image, Scikit-Learn, Streamlit  
**Dokumen:** Laporan Akhir Proyek & Analisis Eksperimen

---

## DAFTAR ISI

1. [Abstrak](#abstrak)
2. [BAB I: Pendahuluan](#bab-i-pendahuluan)
   - 1.1 Latar Belakang
   - 1.2 Rumusan Masalah
   - 1.3 Tujuan Proyek
   - 1.4 Manfaat Penelitian
   - 1.5 Batasan Masalah
3. [BAB II: Landasan Teori](#bab-ii-landasan-teori)
   - 2.1 Tingkat Kematangan Buah Pisang
   - 2.2 Ruang Warna Citra Digital (RGB, HSV, CIELAB)
   - 2.3 Ekstraksi Fitur Warna (Statistik Kanal & Histogram)
   - 2.4 Ekstraksi Fitur Tekstur GLCM (*Gray-Level Co-occurrence Matrix*)
   - 2.5 Fusi Fitur (*Feature Fusion / Early Concatenation*)
   - 2.6 Support Vector Machine (SVM) dengan Kernel RBF
   - 2.7 Metrik Evaluasi Kinerja
4. [BAB III: Metodologi dan Perancangan Sistem](#bab-iii-metodologi-dan-perancangan-sistem)
   - 3.1 Diagram Alur Sistem (*System Pipeline*)
   - 3.2 Karakteristik dan Distribusi Dataset
   - 3.3 Tahap *Preprocessing* Citra
   - 3.4 Desain Ekstraksi Fitur dan Dimensi Vektor
   - 3.5 Desain Skenario 6 Eksperimen
5. [BAB IV: Implementasi Perangkat Lunak](#bab-iv-implementasi-perangkat-lunak)
   - 4.1 Struktur Arsitektur Kode Sumber
   - 4.2 Sistem *Feature Caching* Berbasis Hashing
   - 4.3 Antarmuka Web Interaktif Berbasis Streamlit
6. [BAB V: Hasil Eksperimen dan Pembahasan](#bab-v-hasil-eksperimen-dan-pembahasan)
   - 5.1 Perbandingan Kinerja Kuantitatif 6 Model
   - 5.2 Analisis Pengaruh Ruang Warna (RGB vs HSV vs CIELAB)
   - 5.3 Analisis Pengaruh Penambahan Fitur Tekstur GLCM
   - 5.4 Evaluasi Rinci per Kelas Kematangan
   - 5.5 Pembahasan *Confusion Matrix* dan Pola Misklasifikasi
   - 5.6 Analisis Waktu Komputasi dan Efisiensi
7. [BAB VI: Kesimpulan dan Saran](#bab-vi-kesimpulan-dan-saran)
   - 6.1 Kesimpulan
   - 6.2 Saran dan Pengembangan Lanjutan
8. [Daftar Pustaka](#daftar-pustaka)
9. [Lampiran](#lampiran)

---

## ABSTRAK

Tingkat kematangan buah pisang merupakan indikator vital dalam industri agrikultur dan pascapanen untuk menentukan waktu distribusi, kelayakan konsumsi, serta penentuan harga komoditas. Metode konvensional berbasis inspeksi visual manual manusia rentan terhadap subjektivitas, kelelahan, dan inkonsistensi. Penelitian dan proyek ini bertujuan mengembangkan sistem klasifikasi otomatis tingkat kematangan buah pisang (*unripe*, *ripe*, *overripe*, dan *rotten*) secara objektif dan non-destruktif berbasis pengolahan citra digital klasik (*handcrafted features*).

Fokus investigasi berpusat pada dua aspek utama: (1) evaluasi efektivitas representasi ruang warna berbeda, yaitu RGB (*Red-Green-Blue*), HSV (*Hue-Saturation-Value*), dan CIELAB ($L^*a^*b^*$), serta (2) pengujian signifikansi fusi fitur tekstur *Gray-Level Co-occurrence Matrix* (GLCM) terhadap peningkatan akurasi pengenalan pola kematangan. Ekstraksi fitur warna mengekstrak 33 dimensi (3 parameter statistik: *mean*, *standard deviation*, *median*; serta 8-*bin normalized histogram* per kanal), sedangkan ekstraksi tekstur GLCM menghasilkan 20 dimensi (5 properti Haralick: *contrast*, *dissimilarity*, *homogeneity*, *energy*, dan *correlation* pada jarak piksel $d \in \{1, 3\}$ dengan perataan 4 arah orientasi $\theta \in \{0^\circ, 45^\circ, 90^\circ, 135^\circ\}$). Vektor gabungan (*early fusion*, 53 dimensi) distandarisasi menggunakan `StandardScaler` dan diklasifikasikan menggunakan *Support Vector Machine* (SVM) berkernal *Radial Basis Function* (RBF) dengan *balanced class weights*.

Pengujian dilakukan secara objektif pada 562 citra uji terpisah (*test set*). Hasil eksperimen menunjukkan bahwa penambahan fitur tekstur GLCM secara konsisten meningkatkan performa klasifikasi pada seluruh ruang warna, dengan peningkatan rata-rata *Macro F1-Score* sebesar **+0.0353** (dari 0.8432 menjadi 0.8786). Metode terbaik dicapai oleh konfigurasi **CIELAB + GLCM + SVM** dengan akurasi **88.08%**, *Macro Precision* **88.01%**, *Macro Recall* **89.27%**, dan *Macro F1-Score* **0.8838**. Fitur tekstur GLCM terbukti secara empiris krusial dalam mendiskriminasi kelas *overripe* dan *rotten* yang memiliki spektrum warna serupa namun memiliki densitas bintik nekrosis melanin (*senescence spotting*) yang berbeda signifikan.

**Kata Kunci:** *Pengolahan Citra Digital, Kematangan Pisang, Ruang Warna, CIELAB, GLCM, Haralick Texture, Fusi Fitur, Support Vector Machine (SVM).*

---

## BAB I: PENDAHULUAN

### 1.1 Latar Belakang

Buah pisang (*Musa paradisiaca*) adalah salah satu komoditas hortikultura buah tropis terpenting di dunia dengan volume konsumsi dan nilai perdagangan yang sangat tinggi. Karakteristik biologis pisang tergolong sebagai buah klimakterik (*climacteric fruit*), yakni buah yang mengalami lonjakan laju respirasi dan peningkatan biosintesis gas etilen yang pesat setelah fase pemetikan. Proses klimakterik ini memicu perombakan fisiologis dan biokimia drastis pada kulit buah, meliputi:
1. Degradasi pigmen klorofil yang menyebabkan perubahan warna dari hijau pekat (*unripe*) menjadi kuning cerah (*ripe*).
2. Perubahan pati menjadi gula sederhana yang melunakkan dinding sel buah.
3. Terjadinya reaksi enzimatik polifenol oksidase yang memicu nekrosis sel, menghasilkan bintik-bintik cokelat atau kehitaman (*senescence spots*) pada fase lewat matang (*overripe*).
4. Kerusakan jaringan seluler secara menyeluruh akibat pembusukan mikrobiologis yang menghitamkan kulit buah secara luas (*rotten*).

Dalam rantai pasok industri modern, sortasi dan *grading* kematangan pisang umumnya masih dilakukan oleh tenaga manusia (*manual visual inspection*). Cara manual ini memiliki berbagai kelemahan fundamental, antara lain:
- Subjektivitas persepsi operator manusia yang berbeda-beda terhadap intensitas warna.
- Penurunan konsentrasi dan akurasi akibat kelelahan visual (*eye fatigue*) dalam shift kerja panjang.
- Biaya operasional tinggi dan ketidakmampuan beroperasi secara berkesinambungan dengan kecepatan tinggi.

Oleh karena itu, otomasi berbasis *Pengolahan Citra Digital* (*Digital Image Processing*) dan *Machine Learning* menawarkan solusi non-destruktif, cepat, objektif, dan dapat direproduksi secara konsisten. Namun demikian, penentuan kematangan buah pisang murni berbasis warna sering mengalami kegagalan (*ambiguity*) ketika membedakan pisang *ripe* bertotol sedikit dengan *overripe*, atau pisang *overripe* dengan *rotten*, karena kedua fase tersebut sama-sama didominasi oleh percampuran warna kuning, kecokelatan, dan hitam. Di sinilah integrasi fitur tekstur spasial permukaan kulit menjadi sangat esensial.

### 1.2 Rumusan Masalah

Berdasarkan latar belakang di atas, rumusan masalah dalam proyek ini dirumuskan sebagai berikut:
1. Bagaimana merancang arsitektur sistem pengolahan citra digital klasik (*handcrafted pipeline*) yang mampu mengklasifikasikan tingkat kematangan buah pisang ke dalam empat kelas (*unripe*, *ripe*, *overripe*, dan *rotten*) secara akurat?
2. Bagaimana perbandingan performa klasifikasi antara representasi ruang warna RGB, HSV, dan CIELAB dalam merepresentasikan degradasi klorofil pada kulit pisang?
3. Apakah penambahan (*fusion*) deskriptor tekstur spasial *Gray-Level Co-occurrence Matrix* (GLCM) dapat meningkatkan akurasi dan *Macro F1-Score* klasifikasi secara signifikan dibandingkan penggunaan fitur warna tunggal?
4. Kombinasi ruang warna dan fitur apa yang memberikan kinerja klasifikasi tertinggi ketika diuji menggunakan model *Support Vector Machine* (SVM) dengan parameter tetap yang adil?

### 1.3 Tujuan Proyek

Tujuan yang ingin dicapai melalui pengerjaan proyek ini adalah:
1. Mengembangkan pipeline terstruktur dari *preprocessing*, ekstraksi fitur warna (33 dimensi), ekstraksi tekstur GLCM (20 dimensi), standardisasi, hingga klasifikasi SVM.
2. Membangun dan menguji 6 skenario eksperimen terkontrol:
   - Eksperimen 1: RGB + SVM (33 fitur)
   - Eksperimen 2: HSV + SVM (33 fitur)
   - Eksperimen 3: CIELAB + SVM (33 fitur)
   - Eksperimen 4: RGB + GLCM + SVM (53 fitur)
   - Eksperimen 5: HSV + GLCM + SVM (53 fitur)
   - Eksperimen 6: CIELAB + GLCM + SVM (53 fitur)
3. Menganalisis pengaruh pemilihan ruang warna dan kontribusi penambahan matriks ko-okurensi spasial GLCM terhadap performa presisi, *recall*, dan matriks konfusi per kelas.
4. Membangun antarmuka aplikasi interaktif (*Graphical User Interface*) berbasis web menggunakan Streamlit yang mendukung pengujian dataset, visualisasi fitur, inferensi gambar baru, dan inspeksi metrik.

### 1.4 Manfaat Penelitian

1. **Bagi Akademisi & Mahasiswa:** Memberikan pemahaman komprehensif mengenai aplikasi riil konsep dasar pengolahan citra (konversi ruang warna, histogram, GLCM Haralick) dan *machine learning* (SVM RBF) tanpa ketergantungan pada model *black-box deep learning*.
2. **Bagi Industri Pertanian & Retail:** Menyediakan acuan desain algoritma sistem sortasi otomatis yang berbobot komputasi ringan sehingga dapat diimplementasikan pada perangkat keras *edge computing* (seperti Raspberry Pi atau industri kamera pintar).

### 1.5 Batasan Masalah

1. Metode yang digunakan berfokus pada pendekatan klasik (*handcrafted feature extraction*), bukan *deep learning* (CNN/ViT).
2. Model klasifikasi menggunakan algoritma *Support Vector Machine* (SVM) berkernal RBF dengan hyperparameter seragam ($C=1.0$, $\gamma=\text{scale}$, *class weight* = *balanced*) untuk menjamin perbandingan fitur berlangsung secara adil (*fair comparison*).
3. Resolusi citra diseragamkan (*resizing*) ke dimensi $128 \times 128$ piksel dengan *aspect-ratio padding* untuk menjaga konsistensi komputasi.
4. Dataset yang digunakan adalah *Banana Ripeness Classification v6* dari Roboflow Universe yang mencakup 4 kelas kematangan (*unripe*, *ripe*, *overripe*, *rotten*).

---

## BAB II: LANDASAN TEORI

### 2.1 Tingkat Kematangan Buah Pisang

Perjalanan fisiologis kematangan pisang umumnya dipetakan ke dalam skala indeks warna komersial (standar Von Loesecke, Skala 1 hingga 7). Dalam penelitian berbasis penglihatan komputer ini, kelas disederhanakan menjadi 4 kelas representatif:
- **Mentah (*Unripe*):** Kulit pisang didominasi warna hijau murni ($>85\%$), tekstur permukaan halus tanpa bercak, menandakan konsentrasi pigmen klorofil tinggi dan kandungan pati belum terhidrolisis.
- **Matang (*Ripe*):** Kulit buah berwarna kuning cerah merata, ujung buah terkadang masih memiliki semburat hijau muda, tekstur permukaan relatif homogen tanpa bintik nekrosis berat.
- **Terlalu Matang (*Overripe*):** Warna dasar kuning mulai memudar ke arah cokelat muda, disertai munculnya bintik-bintik kecil nekrosis melanin (*sugar spots* / *senescence spots*) yang tersebar secara acak di permukaan kulit.
- **Busuk (*Rotten*):** Seluruh atau sebagian besar permukaan kulit menghitam ($>50\%$), tekstur kulit rusak, terjadi dekomposisi jaringan organik secara luas.

---

### 2.2 Ruang Warna Citra Digital

Warna adalah persepsi visual manusia terhadap distribusi spektral radiasi elektromagnetik cahaya. Representasi numerik warna dalam citra digital dilakukan melalui *ruang warna* (*color space*):

```
       [Citra Input BGR]
         /      |      \
        v       v       v
     [ RGB ] [ HSV ] [ CIELAB ]
```

#### A. Ruang Warna RGB (Red, Green, Blue)
Ruang warna RGB adalah model aditif primer berbasis perangkat keras penangkap citra (*hardware-oriented*). Nilai intensitas kanal berkisar antara 0 hingga 255 pada format 8-bit.
- **Kelemahan:** Kanal R, G, dan B memiliki korelasi tinggi terhadap intensitas pencahayaan (*luminance*). Perubahan kondisi cahaya lingkungan (bayangan atau pantulan kilau) akan mengubah nilai ketiga kanal secara simultan, sehingga sensitif terhadap noise variasi pencahayaan ruangan.

#### B. Ruang Warna HSV (Hue, Saturation, Value)
Ruang warna HSV memisahkan komponen kromatisitas dari intensitas pencahayaan:
- **Hue ($H$):** Merepresentasikan jenis panjang gelombang warna murni (sudut warna $0^\circ - 360^\circ$; pada OpenCV diskalakan ke rentang $[0, 179]$). Sudut $H$ membedakan warna hijau ($\approx 35-85$) dan kuning ($\approx 20-35$) secara independen dari intensitas cahaya.
- **Saturation ($S$):** Kejenuhan atau kemurnian warna ($[0, 255]$). Nilai tinggi menunjukkan warna pekat, sedangkan nilai rendah mengindikasikan warna memudar mendekati abu-abu.
- **Value ($V$):** Tingkat kecerahan relatif warna ($[0, 255]$).

#### C. Ruang Warna CIELAB ($L^*, a^*, b^*$)
Didefinisikan oleh *Commission Internationale de l'Éclairage* (CIE) pada tahun 1976, CIELAB dirancang agar seragam secara persepsi manusia (*perceptually uniform*):
- Jarak Euclidean $\Delta E$ antara dua titik warna di ruang CIELAB sebanding dengan perbedaan warna yang dirasakan oleh mata manusia.
- **Kanal $L^*$ (Lightness):** Menyatakan tingkat kecerahan dari hitam ($0$) hingga putih ($100$; pada OpenCV dipetakan ke $[0, 255]$).
- **Kanal $a^*$:** Sumbu kromatik gradien warna **Hijau (nilai rendah/negatif) ke Merah/Magenta (nilai tinggi/positif)**. Kanal ini sangat sensitif terhadap fenomena degradasi klorofil pada kulit pisang.
- **Kanal $b^*$:** Sumbu kromatik gradien warna **Biru (nilai rendah/negatif) ke Kuning (nilai tinggi/positif)**. Kanal ini menangkap sintesis pigmen karotenoid kuning saat pisang mengalami pematangan.

---

### 2.3 Ekstraksi Fitur Warna

Untuk mereduksi dimensi matriks citra tanpa menghilangkan informasi statistik distribusi warna, setiap kanal $c \in \{1, 2, 3\}$ dari ruang warna yang dipilih diekstraksi menjadi **11 fitur numerik**:

1. **Mean ($\mu_c$):** Rata-rata intensitas kanal:
   $$\mu_c = \frac{1}{N} \sum_{i=1}^{N} p_{c}(i)$$
2. **Standard Deviation ($\sigma_c$):** Dispersi variasi intensitas piksel kanal:
   $$\sigma_c = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (p_{c}(i) - \mu_c)^2}$$
3. **Median ($M_c$):** Nilai tengah data intensitas kanal, tangguh (*robust*) terhadap gangguan kilau pantulan cahaya eksternal (*specular highlights*).
4. **Normalized Color Histogram (8 Bins):** Histogram intensitas piksel dibagi menjadi 8 interval (*bins*) seragam dan dinormalisasi sehingga total jumlahan probabilitas bernilai 1.0:
   $$h_c(b) = \frac{\text{count}(p_c \in \text{bin}_b)}{\sum_{k=1}^{8} \text{count}(p_c \in \text{bin}_k)}, \quad b \in \{1, \dots, 8\}$$

Dengan 3 kanal per ruang warna, total dimensi fitur warna adalah:
$$\text{Dim}_{\text{warna}} = 3 \text{ kanal} \times (3 \text{ statistik} + 8 \text{ bin histogram}) = 33 \text{ dimensi}$$

---

### 2.4 Ekstraksi Fitur Tekstur GLCM (*Gray-Level Co-occurrence Matrix*)

Metode GLCM yang diperkenalkan oleh Robert M. Haralick (1973) adalah teknik analisis tekstur statistik orde kedua (*second-order statistical texture analysis*). GLCM menghitung probabilitas transisi spasial $P(i, j | d, \theta)$, yaitu frekuensi kemunculan pasangan piksel dengan nilai intensitas $i$ dan $j$ yang terpisah oleh jarak translasi spasial $d$ piksel pada sudut orientasi $\theta$.

```
Matriks Grayscale (32 level) ──> Hitung Pasangan Spasial (d=1,3; θ=0°,45°,90°,135°)
                              ──> Matriks GLCM Simetris & Ternormalisasi P(i,j)
                              ──> Hitung 5 Properti Haralick (Rata-rata 4 Arah)
```

#### Parameter Konfigurasi GLCM dalam Proyek:
- **Kuantisasi Keabuan (*Gray Levels*):** 32 level ($L=32$). Citra keabuan dikuantisasi ke rentang $[0, 31]$ guna mempercepat pembentukan matriks dan mencegah *sparse matrix* (matriks kosong).
- **Jarak Translasi Spasial ($d$):** Menggunakan 2 skala jarak: $d=1$ piksel (menangkap mikro-tekstur permukaan halus) dan $d=3$ piksel (menangkap makro-tekstur persebaran bintik).
- **Sudut Orientasi ($\theta$):** 4 arah mata angin utama: $0^\circ$ (horizontal), $45^\circ$ (diagonal kanan atas), $90^\circ$ (vertikal), dan $135^\circ$ (diagonal kiri atas).
- **Invarian Arah (*Directional Invariance*):** Nilai properti dari keempat sudut dirata-ratakan untuk menghasilkan deskriptor yang tahan rotasi orientasi buah.
- **Normalisasi dan Simetri:** Matriks GLCM diatur simetris dan dinormalisasi $\sum_{i,j} P(i, j) = 1$.

#### Lima Properti Haralick yang Dihitung:

1. **Kontras (*Contrast*):**
   Mengukur variasi intensitas lokal antar piksel bertetangga. Nilai kontras akan melonjak drastis apabila permukaan pisang memiliki kontras tinggi antara kulit kuning dengan bintik nekrosis hitam pekat.
   $$\text{Contrast} = \sum_{i=0}^{L-1} \sum_{j=0}^{L-1} |i - j|^2 P(i, j)$$

2. **Ketidaksamaan (*Dissimilarity*):**
   Mirip dengan kontras namun menggunakan pembobotan linier absolut. Mengukur perbedaan derajat keabuan lokal secara proporsional.
   $$\text{Dissimilarity} = \sum_{i=0}^{L-1} \sum_{j=0}^{L-1} |i - j| P(i, j)$$

3. **Homogenitas (*Homogeneity / Inverse Difference Moment*):**
   Mengukur keseragaman dan kehalusan tekstur permukaan kulit. Nilai homogenitas tinggi diasosiasikan dengan pisang mentah (*unripe*) atau matang sempurna (*ripe*) tanpa bintik.
   $$\text{Homogeneity} = \sum_{i=0}^{L-1} \sum_{j=0}^{L-1} \frac{P(i, j)}{1 + (i - j)^2}$$

4. **Energi (*Energy / Angular Second Moment*):**
   Mengukur keteraturan distribusi tekstur. Bernilai tinggi saat elemen matriks GLCM terkonsentrasi pada pola seragam.
   $$\text{Energy} = \sqrt{\sum_{i=0}^{L-1} \sum_{j=0}^{L-1} P(i, j)^2}$$

5. **Korelasi (*Correlation*):**
   Mengukur ketergantungan linier nilai derajat abu-abu antar pasangan piksel bertetangga, merefleksikan arah serat longitudinal buah pisang.
   $$\text{Correlation} = \sum_{i=0}^{L-1} \sum_{j=0}^{L-1} \frac{(i - \mu_i)(j - \mu_j) P(i, j)}{\sigma_i \sigma_j}$$
   di mana $\mu_i, \mu_j$ dan $\sigma_i, \sigma_j$ berturut-turut adalah rata-rata dan deviasi standar baris dan kolom matriks GLCM.

Dengan 5 properti tekstur dan 2 jarak translasi spasial ($d=1, 3$), total dimensi fitur tekstur GLCM adalah:
$$\text{Dim}_{\text{GLCM}} = 5 \text{ properti} \times 2 \text{ jarak} = 20 \text{ dimensi}$$

---

### 2.5 Fusi Fitur (*Feature Fusion / Early Concatenation*)

Fusi fitur tingkat awal (*early fusion / feature-level concatenation*) menggabungkan informasi fotometrik (warna) dan morfologi spasial (tekstur) ke dalam satu vektor representasi tunggal:
$$\mathbf{x}_{\text{fusi}} = \left[ \mathbf{x}_{\text{warna}}^{\top} \, (33\text{D}) \ \Big\Vert \ \mathbf{x}_{\text{GLCM}}^{\top} \, (20\text{D}) \right]^{\top} \in \mathbb{R}^{53}$$

Sebelum dimasukkan ke dalam algoritma klasifikasi, dilakukan penskalaan standar (*z-score standardization*):
$$z = \frac{x - \mu}{\sigma}$$
Standardisasi ini sangat fundamental agar fitur dengan skala magnitudo besar (misalnya *mean* warna rentang $0-255$) tidak menenggelamkan fitur bernilai pecahan desimal kecil (seperti energi GLCM rentang $0.001-0.1$).

---

### 2.6 Support Vector Machine (SVM) dengan Kernel RBF

Support Vector Machine (Cortes & Vapnik, 1995) adalah model pembelajaran terawasi (*supervised learning*) yang bekerja dengan cara mencari bidang pemisah (*hyperplane*) optimal yang memaksimalkan margin jarak antara kelas terdekat (*support vectors*).

Karena ruang fitur citra buah pisang bersifat tidak dapat dipisahkan secara linier (*non-linearly separable*), digunakan fungsi kernel *Radial Basis Function* (RBF):
$$K(\mathbf{x}_i, \mathbf{x}_j) = \exp\left(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2\right)$$

#### Konfigurasi Hyperparameter Tetap:
- **Kernel:** RBF (*Radial Basis Function*) untuk menangani batas keputusan non-linier yang kompleks.
- **Regularization Parameter ($C=1.0$):** Mengatur keseimbangan antara meminimalkan kesalahan pelatihan dan memaksimalkan margin generalisasi.
- **Gamma ($\gamma = \text{'scale'}$):** Parameter dispersi kernel yang dihitung otomatis berdasarkan variansi data fitur: $\gamma = \frac{1}{n_{\text{features}} \cdot \text{Var}(X)}$.
- **Class Weight:** `'balanced'`, yang secara otomatis menghitung bobot penalti terbalik proporsional terhadap frekuensi sampel kelas untuk mengatasi masalah ketidakseimbangan kelas (*class imbalance*).

---

### 2.7 Metrik Evaluasi Kinerja

Evaluasi model klasifikasi multi-kelas ($K=4$) diukur menggunakan:
1. **Akurasi (*Accuracy*):** Rasio total prediksi benar terhadap total seluruh sampel uji:
   $$\text{Accuracy} = \frac{\sum_{k=1}^{K} TP_k}{N}$$
2. **Macro Precision:** Rata-rata aritmatika dari presisi seluruh kelas:
   $$\text{Precision}_{\text{macro}} = \frac{1}{K} \sum_{k=1}^{K} \frac{TP_k}{TP_k + FP_k}$$
3. **Macro Recall:** Rata-rata aritmatika dari *recall* (sensitivitas) seluruh kelas:
   $$\text{Recall}_{\text{macro}} = \frac{1}{K} \sum_{k=1}^{K} \frac{TP_k}{TP_k + FN_k}$$
4. **Macro F1-Score:** Rata-rata harmonik antara presisi dan recall secara *macro-averaged*, menjadi metrik pembanding utama karena memberikan bobot evaluasi setara untuk setiap kelas kematangan:
   $$\text{F1}_{\text{macro}} = \frac{1}{K} \sum_{k=1}^{K} 2 \cdot \frac{\text{Precision}_k \cdot \text{Recall}_k}{\text{Precision}_k + \text{Recall}_k}$$
5. **Confusion Matrix:** Matriks kontingensi ukuran $4 \times 4$ yang memetakan jumlah sampel aktual versus hasil prediksi model.

---

## BAB III: METODOLOGI DAN PERANCANGAN SISTEM

### 3.1 Diagram Alur Sistem (*System Pipeline*)

Alur kerja sistem klasifikasi dirancang secara modular dan berurutan dari penanganan dataset hingga inferensi:

```
[ Dataset Citra Pisang ] 
         │
         ▼
[ Preprocessing Citra ] ──> Resize (128x128) dgn Aspect Ratio Padding
         │              ──> Konversi Ruang Warna (RGB / HSV / CIELAB / Gray)
         │              ──> Segmentasi Opsional (GrabCut Background Removal)
         ▼
[ Ekstraksi Fitur ]
  ├── Fitur Warna (33D) ──> 3 Statistik (Mean, Std, Median) + 8-Bin Hist per Kanal
  └── Fitur Tekstur (20D) ──> Kuantisasi Grayscale 32 Level -> GLCM (d=1,3; 4 Arah)
         │
         ▼
[ Feature Fusion & Caching ] ──> Gabung Vektor (Early Fusion 53D) & Simpan Cache (.npz)
         │
         ▼
[ Scikit-Learn Pipeline ]
  ├── StandardScaler (Mean=0, Var=1)
  └── SVC (Kernel='rbf', C=1.0, Gamma='scale', ClassWeight='balanced')
         │
         ▼
[ Evaluasi & Visualisasi ] ──> Accuracy, Macro F1, Confusion Matrix, Web UI
```

---

### 3.2 Karakteristik dan Distribusi Dataset

Dataset yang digunakan adalah **Banana Ripeness Classification v6** yang dipublikasikan melalui Roboflow Universe dengan lisensi Creative Commons Attribution 4.0 International (CC BY 4.0).

#### Distribusi Citra Dataset:
- **Set Pelatihan (*Train Set*):** 11.793 citra
- **Set Validasi (*Validation Set*):** 1.123 citra
- **Set Pengujian (*Test Set*):** 562 citra
- **Total Keseluruhan:** 13.478 citra

#### Distribusi Kelas pada Data Uji (*Test Set*, $N=562$):
| No | Label Kelas | Kondisi Kematangan | Jumlah Sampel Uji | Persentase |
|:--:|:-----------|:-------------------|:-----------------:|:----------:|
| 1 | `unripe` | Pisang Mentah (Hijau) | 110 citra | 19.57% |
| 2 | `ripe` | Pisang Matang (Kuning) | 154 citra | 27.40% |
| 3 | `overripe` | Terlalu Matang (Bercak cokelat) | 113 citra | 20.11% |
| 4 | `rotten` | Busuk (Hitam dominan) | 185 citra | 32.92% |
| **Total** | | | **562 citra** | **100.0%** |

---

### 3.3 Tahap Preprocessing Citra

1. **Pembacaan Citra:** Citra dibaca menggunakan `cv2.imread()` dalam format native OpenCV (BGR 8-bit). Dilakukan verifikasi integritas data (*null check*).
2. **Penskalaan Proporsional dengan Padding (*Aspect-Ratio Preserved Resizing*):**
   - Menghindari distorsi geometri pisang akibat penskalaan *stretching*.
   - Faktor skala dihitung $\min\left(\frac{128}{W}, \frac{128}{H}\right)$.
   - Citra hasil resize disematkan pada kanvas hitam berukuran tepat $128 \times 128$ piksel secara *centered* dengan padding simetris.
3. **Konversi Ruang Warna:**
   - RGB: `cv2.COLOR_BGR2RGB`
   - HSV: `cv2.COLOR_BGR2HSV`
   - CIELAB: `cv2.COLOR_BGR2LAB`
   - Grayscale: `cv2.COLOR_BGR2GRAY`
4. **Segmentasi Latar Belakang (Opsional):**
   - Menggunakan algoritma *GrabCut* berbasis *graph-cut energy minimization* dengan batas inisialisasi persegi (*bounding rectangle*). Menghasilkan mask biner untuk mengisolasi piksel buah dari latar belakang permukaan meja atau lantai.

---

### 3.4 Desain Ekstraksi Fitur dan Dimensi Vektor

Tabel berikut merangkum rincian vektor fitur yang dihasilkan oleh setiap skema ekstraksi:

| Kelompok Fitur | Sub-Komponen | Metode Komputasi | Dimensi |
|:---------------|:-------------|:-----------------|:-------:|
| **Fitur Warna** | Statistik Kanal | Mean, Standard Deviation, Median per kanal | $3 \times 3 = 9$ |
| | Histogram Warna | 8-bin normalized histogram per kanal | $3 \times 8 = 24$ |
| | **Subtotal Fitur Warna** | | **33 Dimensi** |
| **Fitur Tekstur** | GLCM Jarak $d=1$ | Rata-rata 4 arah $(\theta \in \{0^\circ, 45^\circ, 90^\circ, 135^\circ\})$ dari Contrast, Dissimilarity, Homogeneity, Energy, Correlation | 5 |
| | GLCM Jarak $d=3$ | Rata-rata 4 arah $(\theta \in \{0^\circ, 45^\circ, 90^\circ, 135^\circ\})$ dari Contrast, Dissimilarity, Homogeneity, Energy, Correlation | 5 |
| | **Subtotal Fitur GLCM** | (Dihitung pada kuantisasi 32 level, resolusi $64 \times 64$) | **20 Dimensi** |
| **Fusi Lengkap** | **Warna + GLCM** | **Early Concatenation $[\mathbf{x}_{\text{warna}} \Vert \mathbf{x}_{\text{GLCM}}]$** | **53 Dimensi** |

---

### 3.5 Desain Skenario 6 Eksperimen

Untuk menguji hipotesis secara komprehensif, dibentuk 6 konfigurasi eksperimen dengan hyperparameter SVM yang dipertahankan identik:

| ID | Nama Eksperimen | Ruang Warna | Fitur Warna | Fitur GLCM | Dimensi Total Fitur |
|:--:|:----------------|:-----------:|:-----------:|:----------:|:-------------------:|
| **1** | RGB + SVM | RGB | 33 fitur | ❌ (0) | 33 fitur |
| **2** | HSV + SVM | HSV | 33 fitur | ❌ (0) | 33 fitur |
| **3** | CIELAB + SVM | CIELAB | 33 fitur | ❌ (0) | 33 fitur |
| **4** | RGB + GLCM + SVM | RGB | 33 fitur | ✅ (20) | 53 fitur |
| **5** | HSV + GLCM + SVM | HSV | 33 fitur | ✅ (20) | 53 fitur |
| **6** | CIELAB + GLCM + SVM | CIELAB | 33 fitur | ✅ (20) | 53 fitur |

---

## BAB IV: IMPLEMENTASI PERANGKAT LUNAK

### 4.1 Struktur Arsitektur Kode Sumber

Proyek dirancang dengan pola arsitektur modular yang memisahkan logika pemrosesan data, pelatihan model, evaluasi, dan presentasi antarmuka pengguna:

```
Pisang/
├── app.py                      # Aplikasi Graphical User Interface Streamlit (7 Modul)
├── config.py                   # Konfigurasi path, parameter GLCM, SVM, dan konstanta
├── requirements.txt            # Daftar pustaka dependensi Python
├── README.md                   # Dokumentasi ringkas proyek
├── LAPORAN_PROYEK.md           # Dokumen laporan akademik komprehensif
├── src/
│   ├── dataset_loader.py       # Validasi integritas dataset, scanning split & label mapping
│   ├── preprocessing.py        # Resize proporsional, konversi ruang warna & GrabCut
│   ├── feature_extraction.py   # Ekstraksi statistik warna, histogram & GLCM Haralick
│   ├── model_training.py       # Pipeline Scikit-Learn, training SVM & feature caching
│   ├── evaluation.py           # Komputasi metrik evaluasi, plot chart & confusion matrix
│   ├── prediction.py           # Pipeline inferensi gambar baru dan estimasi probabilitas
│   └── utils.py                # Utilitas serialisasi model (joblib) & helper I/O
├── tests/
│   └── test_modules.py         # Unit test & smoke test otomatis untuk modul-modul sistem
├── artifacts/
│   ├── models/                 # Model tersimpan format .joblib (6 eksperimen)
│   └── feature_cache/          # Cache fitur terkompresi format .npz & metadata .json
└── results/
    ├── metrics_summary.csv     # Rekapitulasi metrik numerik seluruh eksperimen
    ├── laporan_hasil.md        # Laporan Markdown otomatis hasil eksperimen
    ├── experiment_metadata.json# Metadata rinci JSON per eksperimen
    ├── plots/                  # Visualisasi perbandingan metrik (perbandingan_metrik.png)
    ├── confusion_matrices/     # Plot matriks konfusi grafis untuk ke-6 model
    └── classification_reports/ # Laporan teks klasifikasi per kelas untuk ke-6 model
```

---

### 4.2 Sistem Feature Caching Berbasis Hashing

Ekstraksi fitur GLCM dan histogram pada belasan ribu citra membutuhkan beban komputasi yang tinggi. Untuk menghindari redundansi komputasi saat eksperimen diulang, diimplementasikan mekanisme *caching* cerdas:
- Kunci cache (*hash key*) dihitung menggunakan algoritma MD5 dari string parameter:
  $$\text{Key} = \text{MD5}(\text{dataset\_path} \parallel \text{split} \parallel \text{color\_space} \parallel \text{use\_glcm} \parallel \text{use\_segmentation} \parallel \text{label\_mode} \parallel \text{max\_samples})$$
- Hasil ekstraksi fitur disimpan ke disk dalam format biner NumPy terkompresi (`.npz`) beserta file metadata pendukung (`_meta.json`).
- Saat proses pelatihan dipanggil ulang dengan parameter yang sama, sistem langsung memuat matriks fitur dari cache disk hanya dalam hitungan detik.

---

### 4.3 Antarmuka Web Interaktif Berbasis Streamlit

Aplikasi GUI interaktif dibangun menggunakan Streamlit (`app.py`) yang mencakup 7 modul fungsional:
1. **🏠 Beranda:** Dashboard ringkasan proyek, metodologi, dan status ketersediaan model hasil pelatihan.
2. **📁 Dataset:** Modul inspeksi visual struktur direktori dataset, pilihan label (*original* vs *maturity group*), diagram distribusi kelas, dan galeri penampil sampel citra acak.
3. **🔬 Preprocessing & Fitur:** Halaman interaktif untuk memvisualisasikan hasil resize, konversi warna RGB/HSV/CIELAB, inspeksi grafik histogram 8-bin per kanal, serta nilai numerik 5 properti GLCM secara *real-time*.
4. **⚙️ Training & Eksperimen:** Panel eksekusi pelatihan 6 model eksperimen secara berurutan, dilengkapi pemilihan *Quick Mode* (sub-sampling stratified) atau *Full Mode*, serta pencatat waktu komputasi.
5. **📊 Evaluasi:** Panel komparasi performa model, menyajikan tabel metrik terintegrasi, visualisasi grafik batang komparasi akurasi & F1-score, galeri *confusion matrix*, dan penampil laporan klasifikasi per kelas.
6. **🔮 Prediksi Gambar Baru:** Fitur unggah citra (`JPG`, `PNG`), preview visualisasi fitur, inferensi tingkat kematangan, estimasi skor probabilitas keyakinan (*confidence score*), serta perbandingan silang prediksi antar ke-6 model.
7. **📚 Metodologi & Edukasi:** Penjelasan teoritis mendalam mengenai matematika warna, GLCM Haralick, prinsip kerja SVM, serta atribusi lisensi dataset Roboflow CC BY 4.0.

---

## BAB V: HASIL EKSPERIMEN DAN PEMBAHASAN

### 5.1 Perbandingan Kinerja Kuantitatif 6 Model

Seluruh model dilatih pada subset pelatihan dan dievaluasi secara independen terhadap **562 citra pada Test Set**. Hasil evaluasi kuantitatif disajikan pada tabel berikut:

| No | Eksperimen | Akurasi | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | Jumlah Sampel Uji |
|:--:|:-----------|:-------:|:-----------------:|:--------------:|:----------------:|:-----------------:|
| 1 | RGB + SVM | 0.8381 (83.81%) | 0.8426 | 0.8479 | 0.8437 | 562 |
| 2 | HSV + SVM | 0.8416 (84.16%) | 0.8406 | 0.8596 | 0.8467 | 562 |
| 3 | CIELAB + SVM | 0.8345 (83.45%) | 0.8326 | 0.8517 | 0.8394 | 562 |
| 4 | RGB + GLCM + SVM | 0.8737 (87.37%) | 0.8739 | 0.8893 | 0.8777 | 562 |
| 5 | HSV + GLCM + SVM | 0.8701 (87.01%) | 0.8686 | 0.8874 | 0.8742 | 562 |
| **6** | **CIELAB + GLCM + SVM** | **0.8808 (88.08%)** | **0.8801** | **0.8927** | **0.8838** | **562** |

---

### 5.2 Analisis Pengaruh Ruang Warna (RGB vs HSV vs CIELAB)

Ketika dievaluasi **hanya menggunakan fitur warna** (Eksperimen 1, 2, dan 3):
- **Ruang Warna HSV** menghasilkan kinerja tertinggi di antara model warna murni dengan akurasi **84.16%** dan F1-Score **0.8467**. Hal ini disebabkan oleh kanal *Hue* ($H$) yang secara eksplisit memisahkan warna murni pisang (hijau untuk *unripe* dan kuning untuk *ripe*) dari variasi intensitas cahaya (*Value*).
- **Ruang Warna RGB** menempati posisi kedua dengan akurasi **83.81%** dan F1-Score **0.8437**. Meskipun sederhana, kombinasi statistik dan histogram 8-bin pada kanal R, G, dan B terbukti mampu menangkap penurunan intensitas hijau dan kenaikan intensitas merah-kuning dengan cukup baik.
- **Ruang Warna CIELAB** mencatatkan kinerja terendah pada kategori warna tunggal dengan akurasi **83.45%** dan F1-Score **0.8394**. Pada model linier warna tanpa tekstur, rentang nilai koordinat $a^*$ dan $b^*$ pada kulit pisang yang mulai menua saling tumpang tindih (*overlap*) antara bercak cokelat dan degradasi klorofil, sehingga membutuhkan parameter spasial tambahan agar mampu mendiskriminasi pola dengan optimal.

---

### 5.3 Analisis Pengaruh Penambahan Fitur Tekstur GLCM

Penambahan 20 fitur tekstur GLCM (Eksperimen 4, 5, dan 6) memberikan **peningkatan kinerja yang sangat signifikan dan konsisten** di semua ruang warna:

```
Rata-rata Macro F1-Score Fitur Warna Saja   : 0.8432
Rata-rata Macro F1-Score Fusi Warna + GLCM : 0.8786
Delta Peningkatan Performa Rata-rata        : +0.0353 (+3.53%)
```

#### Rincian Peningkatan per Ruang Warna:
1. **Pada Ruang Warna RGB:**
   - F1-Score melonjak dari $0.8437 \rightarrow 0.8777$ (**+0.0340** / kenaikan akurasi **+3.56%**).
2. **Pada Ruang Warna HSV:**
   - F1-Score melonjak dari $0.8467 \rightarrow 0.8742$ (**+0.0275** / kenaikan akurasi **+2.85%**).
3. **Pada Ruang Warna CIELAB:**
   - F1-Score melonjak dari $0.8394 \rightarrow 0.8838$ (**+0.0444** / kenaikan akurasi **+4.63%**).

> **Temuan Kunci:** Ruang warna CIELAB mengalami lonjakan peningkatan terbesar (+4.63% akurasi dan +0.0444 F1-Score) saat digabungkan dengan tekstur GLCM, menjadikannya **konfigurasi terbaik secara keseluruhan di dalam penelitian ini (Akurasi 88.08%, Macro F1 0.8838)**. Hal ini membuktikan adanya sinergi komplementer yang kuat antara pemisahan kroma perseptual $L^*a^*b^*$ dengan deteksi variasi keabuan lokal bintik nekrosis oleh GLCM.

---

### 5.4 Evaluasi Rinci per Kelas Kematangan

Berdasarkan *Classification Report* pada metode terbaik (**CIELAB + GLCM + SVM**):

| Kelas Kematangan | Presisi | Recall | F1-Score | Support (Jumlah Sampel Uji) |
|:-----------------|:-------:|:------:|:--------:|:---------------------------:|
| `unripe` (Mentah) | 0.85 | **0.96** | **0.91** | 110 |
| `ripe` (Matang) | 0.86 | **0.94** | **0.90** | 154 |
| `overripe` (Lewat Matang) | 0.89 | 0.89 | 0.89 | 113 |
| `rotten` (Busuk) | **0.92** | 0.78 | 0.84 | 185 |
| **Rata-rata Rata (Macro Avg)** | **0.88** | **0.89** | **0.88** | **562** |
| **Rata-rata Berbobot (Weighted Avg)** | **0.88** | **0.88** | **0.88** | **562** |

#### Pembahasan Karakteristik Kelas:
1. **Kelas `unripe` (F1 = 0.91, Recall = 0.96):**
   Memiliki sensitivitas (*recall*) tertinggi (96%). Warna hijau dominan pada kanal $a^*$ yang bernilai negatif serta homogenitas GLCM yang tinggi membuat pisang mentah sangat mudah dipisahkan dari kelas lainnya tanpa ambiguitas berarti.
2. **Kelas `ripe` (F1 = 0.90, Recall = 0.94):**
   Memiliki performa klasifikasi yang sangat stabil. Nilai kanal $b^*$ yang tinggi (kuning cerah) dan ketiadaan bintik nekrosis (kontras GLCM rendah) menjadi fitur pembeda yang kuat terhadap kelas `unripe` dan `rotten`.
3. **Kelas `overripe` (F1 = 0.89, Presisi = 0.89, Recall = 0.89):**
   Mendapatkan manfaat paling besar dari integrasi GLCM. Kehadiran bintik-bintik nekrosis halus meningkatkan nilai properti *Contrast* dan *Dissimilarity* GLCM secara spesifik, sehingga model mampu membedakannya dari pisang `ripe` bersih.
4. **Kelas `rotten` (F1 = 0.84, Presisi = 0.92, Recall = 0.78):**
   Memiliki presisi tertinggi (92%), artinya jika model memprediksi busuk, probabilitas kebenarannya mencapai 92%. Namun, *recall*-nya relatif terendah (78%), di mana 41 sampel pisang busuk mengalami *false negative* (terprediksi sebagai *overripe* atau *ripe* akibat area pembusukan lokal yang belum merata ke seluruh permukaan citra).

---

### 5.5 Pembahasan Confusion Matrix dan Pola Misklasifikasi

Berikut adalah matriks konfusi numerik aktual dari model terbaik (**Eksperimen 6: CIELAB + GLCM + SVM**):

```
                     Prediksi Model
 Aktual      Overripe   Ripe   Rotten   Unripe  │ Total
────────────────────────────────────────────────┼──────
 Overripe       101       12       0        0   │  113
 Ripe             0      144      10        0   │  154
 Rotten          12       11     144       18   │  185
 Unripe           1        0       3      106   │  110
────────────────────────────────────────────────┼──────
 Total Pred     114      167     157      124   │  562
```

#### Pola Analisis Kesalahan (*Error Analysis*):
1. **Kebingungan Antara `overripe` dan `ripe` (12 sampel):**
   Sebanyak 12 pisang *overripe* diprediksi sebagai *ripe*. Hal ini terjadi pada pisang yang baru memasuki fase transisi awal (*early overripe*), di mana bintik nekrosis yang muncul masih sangat sedikit dan berukuran sangat kecil sehingga properti kontras GLCM belum mengalami lonjakan signifikan.
2. **Kebingungan Antara `rotten` dan Kelas Lain (41 sampel):**
   Sebanyak 12 sampel *rotten* terprediksi *overripe*, 11 terprediksi *ripe*, dan 18 terprediksi *unripe*. Investigasi pada citra yang salah menunjukkan bahwa pada beberapa sampel, kerusakan mikrobiologis pembusukan terjadi pada satu sisi pisang yang tersembunyi dari sudut pandang kamera, atau warna pisang yang membusuk cenderung pucat keabu-abuan bukan hitam legam.
3. **Ketiadaan Kebingungan Ekstrem:**
   Hampir tidak ada kesalahan fatal antar kelas ekstrem, seperti pisang `unripe` yang terprediksi `overripe` (hanya 1 sampel) atau terprediksi `ripe` (0 sampel). Hal ini membuktikan bahwa batas keputusan hyperplane SVM pada ruang fusi warna-tekstur telah terdefinisi secara teratur (*well-separated boundary*).

---

### 5.6 Analisis Waktu Komputasi dan Efisiensi

Pengujian efisiensi komputasi menunjukkan:
- **Waktu Ekstraksi Fitur Warna Saja (33D):** $\approx 0.008$ detik per citra ($128 \times 128$).
- **Waktu Ekstraksi Fitur GLCM (20D):** $\approx 0.025$ detik per citra ($64 \times 64$, kuantisasi 32 level).
- **Total Waktu Inferensi per Citra Baru:** $\approx 0.035$ detik (mampu mencapai throughput $\approx 28-30$ frame per detik / FPS).
- **Throughput ini membuktikan bahwa sistem berbasis handcrafted color-GLCM-SVM sangat efisien dan memenuhi kriteria aplikasi waktu-nyata (*real-time inspection*) pada lini sortir industri tanpa membutuhkan GPU berdaya tinggi.**

---

## BAB VI: KESIMPULAN DAN SARAN

### 6.1 Kesimpulan

Berdasarkan serangkaian perancangan, implementasi, dan pengujian empiris terhadap 562 citra uji pada 6 skenario eksperimen, dapat disimpulkan bahwa:

1. **Efektivitas Sistem:** Sistem klasifikasi tingkat kematangan buah pisang berbasis pengolahan citra digital klasik (*handcrafted features*) dan *Support Vector Machine* (SVM) berhasil dibangun dan mampu mengklasifikasikan 4 fase kematangan (*unripe*, *ripe*, *overripe*, *rotten*) dengan akurasi tinggi.
2. **Pengaruh Ruang Warna:** Pada model berbasis warna tunggal, ruang warna **HSV** memberikan performa terbaik (Akurasi 84.16%, F1 0.8467) karena kanal *Hue* memisahkan kromatisitas hijau-kuning dari fluktuasi intensitas pencahayaan.
3. **Signifikansi Fitur Tekstur GLCM:** Penambahan fitur tekstur GLCM terbukti secara empiris meningkatkan kinerja klasifikasi di seluruh ruang warna dengan kenaikan rata-rata *Macro F1-Score* sebesar **+0.0353**. GLCM menyediakan deskriptor spasial krusial dalam mendeteksi bercak nekrosis penuaan yang tidak dapat diakomodasi oleh histogram warna semata.
4. **Metode Terbaik:** Konfigurasi **CIELAB + GLCM + SVM** dinobatkan sebagai metode terbaik dengan akurasi **88.08%** dan *Macro F1-Score* **0.8838**, menghasilkan sinergi optimal antara keseragaman perseptual warna kroma ($L^*, a^*, b^*$) dan analisis tekstur orde kedua Haralick.

---

### 6.2 Saran dan Pengembangan Lanjutan

Untuk penyempurnaan sistem pada penelitian atau proyek selanjutnya, beberapa saran yang direkomendasikan adalah:

1. **Segmentasi Latar Belakang Adaptif:** Mengganti GrabCut statis dengan segmentasi berbasis *thresholding Otsu* adaptif pada kanal warna atau *U-Net lightweight* untuk mengatasi variasi latar belakang kompleks pada lingkungan nyata.
2. **Hyperparameter Tuning:** Melakukan optimasi parameter SVM ($C$, $\gamma$) dan pemilihan kernel (misalnya perbandingan dengan kernel Polynomial atau Linear) menggunakan metode *GridSearchCV* atau *Bayesian Optimization* untuk masing-masing model secara individual.
3. **Multi-Scale GLCM:** Mengeksplorasi jarak translasi piksel yang lebih bervariasi ($d \in \{1, 2, 4, 8\}$) guna menangkap spektrum tekstur bintik halus hingga bercak pembusukan skala makro.
4. **Studi Komparasi dengan Deep Learning:** Membandingkan kinerja metode klasik ini dengan arsitektur *Deep Learning* modern berbobot ringan (*MobileNetV3* atau *EfficientNet-B0*) untuk mengukur trade-off antara akurasi, kebutuhan data latih, dan latensi komputasi pada perangkat berdaya rendah (*edge devices*).

---

## DAFTAR PUSTAKA

1. **Haralick, R. M., Shanmugam, K., & Dinstein, I.** (1973). *Textural Features for Image Classification*. IEEE Transactions on Systems, Man, and Cybernetics, SMC-3(6), 610–621.
2. **Cortes, C., & Vapnik, V.** (1995). *Support-Vector Networks*. Machine Learning, 20(3), 273–297.
3. **Gonzalez, R. C., & Woods, R. E.** (2018). *Digital Image Processing* (4th ed.). Pearson Education.
4. **Pedregosa, F., et al.** (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research, 12, 2825–2830.
5. **Bradski, G.** (2000). *The OpenCV Library*. Dr. Dobb's Journal of Software Tools.
6. **Van der Walt, S., et al.** (2014). *scikit-image: Image processing in Python*. PeerJ, 2, e453.
7. **Roboflow Universe.** (2024). *Banana Ripeness Classification Computer Vision Dataset (v6)*. Dilisensikan di bawah CC BY 4.0. https://universe.roboflow.com/roboflow-universe-projects/banana-ripeness-classification

---

## LAMPIRAN

### Lampiran 1: Matriks Konfusi Numerik Lengkap 6 Model

#### 1. RGB + SVM (Akurasi: 83.81%, F1: 0.8437)
```
             Overripe   Ripe   Rotten   Unripe
Overripe        91       13        8        1
Ripe             0      137       17        0
Rotten          13       16      139       17
Unripe           0        0        6      104
```

#### 2. HSV + SVM (Akurasi: 84.16%, F1: 0.8467)
```
             Overripe   Ripe   Rotten   Unripe
Overripe       102       11        0        0
Ripe             4      135       15        0
Rotten          23       13      132       17
Unripe           0        1        5      104
```

#### 3. CIELAB + SVM (Akurasi: 83.45%, F1: 0.8394)
```
             Overripe   Ripe   Rotten   Unripe
Overripe        99       13        1        0
Ripe             3      132       19        0
Rotten          21        9      133       22
Unripe           0        0        5      105
```

#### 4. RGB + GLCM + SVM (Akurasi: 87.37%, F1: 0.8777)
```
             Overripe   Ripe   Rotten   Unripe
Overripe       103        9        0        1
Ripe             0      145        9        0
Rotten          13       16      137       19
Unripe           0        0        4      106
```

#### 5. HSV + GLCM + SVM (Akurasi: 87.01%, F1: 0.8742)
```
             Overripe   Ripe   Rotten   Unripe
Overripe       103       10        0        0
Ripe             0      141       12        1
Rotten          17       12      137       19
Unripe           0        0        2      108
```

#### 6. CIELAB + GLCM + SVM (Akurasi: 88.08%, F1: 0.8838) — *METODE TERBAIK*
```
             Overripe   Ripe   Rotten   Unripe
Overripe       101       12        0        0
Ripe             0      144       10        0
Rotten          12       11      144       18
Unripe           1        0        3      106
```

---

### Lampiran 2: Petunjuk Instalasi dan Menjalankan Aplikasi

#### 1. Persiapan Virtual Environment (Windows PowerShell)
```powershell
cd C:\Users\ACER\Pisang
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

#### 2. Instalasi Dependensi
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

#### 3. Menjalankan Aplikasi Web Streamlit
```powershell
streamlit run app.py
```
Aplikasi secara otomatis dapat diakses melalui browser pada alamat default `http://localhost:8501`.

#### 4. Menjalankan Unit Testing
```powershell
python -m pytest tests/ -v --tb=short
```

---
*Laporan akhir ini disusun secara sistematis berdasarkan data eksperimen nyata proyek Pengelolaan Citra Digital Klasifikasi Buah Pisang.*
