"""
app.py - Aplikasi Streamlit Klasifikasi Kematangan Pisang
Analisis Pengaruh Ruang Warna dan Fusi Fitur Warna-Tekstur
Menggunakan GLCM terhadap Klasifikasi dengan SVM.
"""

import streamlit as st
import os
import sys
import time
import json
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Tambahkan root project ke path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from config import (
    DEFAULT_DATASET_PATH, TARGET_SIZE, HISTOGRAM_BINS,
    GLCM_GRAY_LEVELS, GLCM_DISTANCES, GLCM_ANGLES_DEG, GLCM_PROPERTIES,
    SVM_KERNEL, SVM_C, SVM_GAMMA, SVM_CLASS_WEIGHT,
    QUICK_MODE_SAMPLES_PER_CLASS, RANDOM_SEED,
    EXPERIMENT_NAMES, EXPERIMENT_COLOR_SPACES, EXPERIMENT_USE_GLCM,
    MATURITY_GROUP_MAPPING, MODELS_DIR, RESULTS_DIR, PLOTS_DIR,
    CONFUSION_MATRICES_DIR, CLASSIFICATION_REPORTS_DIR,
    ensure_dirs,
)
from src.dataset_loader import (
    load_dataset_info, load_split_data, check_duplicate_filenames,
    get_label_mapping_info, get_image_paths, get_classes_from_split,
)
from src.preprocessing import (
    preprocess_image, read_image, resize_image, convert_color_space,
    segment_foreground,
)
from src.feature_extraction import (
    extract_all_features, extract_color_features,
    extract_glcm_features, get_feature_dimension,
)
from src.model_training import (
    train_experiment, load_trained_model, get_available_models,
    extract_features_for_split, _get_cache_key,
)
from src.evaluation import (
    evaluate_on_split, run_all_evaluations, create_metrics_summary,
    save_metrics_summary, plot_comparison_chart, plot_confusion_matrix,
    save_classification_report, save_experiment_metadata,
    generate_report_markdown,
)
from src.prediction import predict_image, predict_all_models
from src.utils import format_duration

ensure_dirs()

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Klasifikasi Kematangan Pisang",
    page_icon="🍌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #FFD54F 0%, #FFA726 50%, #FF7043 100%);
        padding: 2rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        color: #333;
        text-align: center;
    }
    .main-header h1 {
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        color: #333;
    }
    .main-header p {
        font-size: 1rem;
        opacity: 0.85;
        margin-top: 0.5rem;
        color: #555;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        border-left: 4px solid #FFA726;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .metric-card h3 {
        font-size: 1.8rem;
        font-weight: 700;
        color: #FF7043;
        margin: 0;
    }
    .metric-card p {
        font-size: 0.85rem;
        color: #666;
        margin: 0.3rem 0 0 0;
    }
    
    .status-ok { color: #4CAF50; font-weight: 600; }
    .status-warn { color: #FF9800; font-weight: 600; }
    .status-error { color: #F44336; font-weight: 600; }
    
    .info-box {
        background: #E3F2FD;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        border-left: 4px solid #2196F3;
        margin: 0.5rem 0;
    }
    .warn-box {
        background: #FFF3E0;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        border-left: 4px solid #FF9800;
        margin: 0.5rem 0;
    }
    
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #FFF8E1, #FFFFFF);
    }
    
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s;
    }
</style>
""", unsafe_allow_html=True)


# ==================== SESSION STATE ====================
def init_session_state():
    defaults = {
        'dataset_path': DEFAULT_DATASET_PATH,
        'dataset_info': None,
        'label_mode': 'original',
        'use_segmentation': False,
        'quick_mode': True,
        'training_results': {},
        'eval_results': {},
        'training_done': False,
        'eval_done': False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()


# ==================== SIDEBAR ====================
st.sidebar.markdown("## 🍌 Navigasi")
page = st.sidebar.radio(
    "Pilih Halaman:",
    [
        "🏠 Beranda",
        "📂 Dataset",
        "🔬 Preprocessing & Fitur",
        "⚙️ Training & Eksperimen",
        "📊 Evaluasi",
        "🔮 Prediksi",
        "📖 Tentang & Metodologi",
    ],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Pengaturan Dataset")
new_path = st.sidebar.text_input(
    "Lokasi Dataset:",
    value=st.session_state['dataset_path'],
    help="Path ke folder dataset (berisi subfolder train/valid/test)",
)
if new_path != st.session_state['dataset_path']:
    st.session_state['dataset_path'] = new_path
    st.session_state['dataset_info'] = None

# Mode label
st.sidebar.markdown("### 🏷️ Mode Label")
label_mode = st.sidebar.selectbox(
    "Pemetaan Label:",
    ['original', 'grouped'],
    format_func=lambda x: 'Kelas Asli' if x == 'original' else 'Kelompok Kematangan',
    index=0 if st.session_state['label_mode'] == 'original' else 1,
)
st.session_state['label_mode'] = label_mode

if label_mode == 'grouped':
    st.sidebar.markdown("""
    <div class="info-box">
    <b>Pemetaan Kelompok:</b><br>
    • unripe, freshunripe → Mentah<br>
    • ripe, freshripe → Matang<br>
    • overripe → Terlalu Matang<br>
    • rotten → Busuk
    </div>
    """, unsafe_allow_html=True)


# ==================== HELPER FUNCTIONS ====================
def load_dataset_cached():
    """Load dataset info with caching."""
    if st.session_state['dataset_info'] is None:
        st.session_state['dataset_info'] = load_dataset_info(
            st.session_state['dataset_path']
        )
    return st.session_state['dataset_info']


def show_header(title, subtitle=""):
    st.markdown(f"""
    <div class="main-header">
        <h1>🍌 {title}</h1>
        <p>{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


# ==================== HALAMAN BERANDA ====================
if page == "🏠 Beranda":
    show_header(
        "Klasifikasi Kematangan Pisang",
        "Analisis Pengaruh Ruang Warna dan Fusi Fitur Warna–Tekstur Menggunakan GLCM dengan SVM"
    )
    
    st.markdown("### 📋 Tentang Penelitian")
    st.markdown("""
    Penelitian ini bertujuan untuk mengidentifikasi tingkat kematangan buah pisang 
    berdasarkan citra menggunakan metode pengolahan citra digital klasik. 
    Penelitian membandingkan tiga ruang warna (**RGB**, **HSV**, **CIELAB**) 
    dan menguji pengaruh penambahan fitur tekstur **GLCM** terhadap performa 
    klasifikasi **SVM**.
    """)
    
    # Status ringkasan
    st.markdown("### 📊 Status Sistem")
    col1, col2, col3, col4 = st.columns(4)
    
    info = load_dataset_cached()
    
    with col1:
        if info['is_valid']:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{info['total_images']:,}</h3>
                <p>Total Gambar</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="metric-card">
                <h3>⚠️</h3>
                <p>Dataset Tidak Valid</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        n_classes = len(info['classes'])
        st.markdown(f"""
        <div class="metric-card">
            <h3>{n_classes}</h3>
            <p>Kelas</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        available = get_available_models()
        st.markdown(f"""
        <div class="metric-card">
            <h3>{len(available)}/6</h3>
            <p>Model Terlatih</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        n_splits = len(info.get('splits', {}))
        st.markdown(f"""
        <div class="metric-card">
            <h3>{n_splits}</h3>
            <p>Split Data</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Alur kerja
    st.markdown("### 🔄 Alur Kerja Pengolahan Citra")
    st.markdown("""
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
    """)
    
    # Enam eksperimen
    st.markdown("### 🧪 Enam Eksperimen")
    exp_df = pd.DataFrame([
        {"No": k, "Eksperimen": v, "Ruang Warna": EXPERIMENT_COLOR_SPACES[k],
         "GLCM": "✅" if EXPERIMENT_USE_GLCM[k] else "❌"}
        for k, v in EXPERIMENT_NAMES.items()
    ])
    st.dataframe(exp_df, use_container_width=True, hide_index=True)


# ==================== HALAMAN DATASET ====================
elif page == "📂 Dataset":
    show_header("Dataset", "Pemeriksaan dan Eksplorasi Dataset Kematangan Pisang")
    
    info = load_dataset_cached()
    
    st.markdown("### 📁 Lokasi Dataset")
    st.code(st.session_state['dataset_path'])
    
    if not info['is_valid']:
        st.error("❌ Dataset tidak valid atau tidak ditemukan.")
        for w in info['warnings']:
            st.warning(w)
        st.info(f"Pastikan dataset tersedia di: `{st.session_state['dataset_path']}`")
        st.stop()
    
    st.success("✅ Dataset berhasil dimuat!")
    
    # Info kelas
    st.markdown("### 🏷️ Kelas yang Ditemukan")
    st.write(", ".join([f"`{c}`" for c in info['classes']]))
    
    # Label mapping info
    mapping_info = get_label_mapping_info(info['classes'], st.session_state['label_mode'])
    st.markdown(f"**Mode label aktif:** {mapping_info['description']}")
    
    if st.session_state['label_mode'] == 'grouped':
        mapping_df = pd.DataFrame([
            {"Kelas Asli": k, "Label Kelompok": v}
            for k, v in mapping_info['mapping'].items()
        ])
        st.dataframe(mapping_df, use_container_width=True, hide_index=True)
    
    # Distribusi data
    st.markdown("### 📊 Distribusi Data")
    
    # Tabel distribusi
    dist_rows = []
    for split_name in ['train', 'valid', 'test']:
        if split_name in info['split_counts']:
            counts = info['split_counts'][split_name]
            row = {'Split': split_name.capitalize()}
            for cls, count in counts.items():
                row[cls] = count
            row['Total'] = sum(counts.values())
            dist_rows.append(row)
    
    if dist_rows:
        dist_df = pd.DataFrame(dist_rows)
        st.dataframe(dist_df, use_container_width=True, hide_index=True)
    
    # Grafik distribusi
    if dist_rows:
        fig, axes = plt.subplots(1, len(dist_rows), figsize=(5 * len(dist_rows), 4))
        if len(dist_rows) == 1:
            axes = [axes]
        
        colors = ['#FFD54F', '#FFA726', '#FF7043', '#E53935', '#8E24AA', '#1E88E5']
        
        for idx, (ax, row) in enumerate(zip(axes, dist_rows)):
            classes = [c for c in info['classes'] if c in row]
            values = [row[c] for c in classes]
            ax.bar(classes, values, color=colors[:len(classes)])
            ax.set_title(f"Split: {row['Split']}", fontweight='bold')
            ax.set_ylabel("Jumlah Gambar")
            for i, v in enumerate(values):
                ax.text(i, v + 5, str(v), ha='center', fontsize=9)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    # Peringatan
    if info['warnings']:
        st.markdown("### ⚠️ Peringatan")
        for w in info['warnings']:
            st.warning(w)
    
    # Cek duplikasi
    with st.expander("🔍 Periksa Potensi Duplikasi Antar Split"):
        if st.button("Jalankan Pemeriksaan Duplikasi"):
            with st.spinner("Memeriksa duplikasi nama file..."):
                dup_warnings = check_duplicate_filenames(st.session_state['dataset_path'])
                if dup_warnings:
                    for w in dup_warnings:
                        st.warning(w)
                else:
                    st.success("✅ Tidak ditemukan nama file yang sama antar split.")
    
    # Preview gambar
    st.markdown("### 🖼️ Preview Gambar")
    
    preview_split = st.selectbox("Pilih split:", list(info['splits'].keys()))
    preview_class = st.selectbox("Pilih kelas:", info['classes'])
    
    split_path = info['splits'][preview_split]
    cls_path = os.path.join(split_path, preview_class)
    
    if os.path.isdir(cls_path):
        images = get_image_paths(cls_path)[:8]
        if images:
            cols = st.columns(4)
            for i, img_path in enumerate(images):
                with cols[i % 4]:
                    img = read_image(img_path)
                    if img is not None:
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        st.image(img_rgb, caption=os.path.basename(img_path),
                                 use_container_width=True)
        else:
            st.info("Tidak ada gambar ditemukan.")


# ==================== HALAMAN PREPROCESSING & FITUR ====================
elif page == "🔬 Preprocessing & Fitur":
    show_header("Preprocessing & Ekstraksi Fitur",
                "Visualisasi Preprocessing dan Fitur Warna serta GLCM")
    
    info = load_dataset_cached()
    if not info['is_valid']:
        st.error("Dataset tidak valid. Periksa halaman Dataset.")
        st.stop()
    
    # Pilih gambar untuk preview
    st.markdown("### 🖼️ Pilih Gambar untuk Preview")
    
    col1, col2 = st.columns(2)
    with col1:
        preview_split = st.selectbox("Split:", list(info['splits'].keys()), key="pp_split")
    with col2:
        preview_class = st.selectbox("Kelas:", info['classes'], key="pp_class")
    
    split_path = info['splits'][preview_split]
    cls_path = os.path.join(split_path, preview_class)
    images = get_image_paths(cls_path)
    
    if not images:
        st.warning("Tidak ada gambar ditemukan.")
        st.stop()
    
    img_idx = st.slider("Pilih gambar:", 0, min(len(images) - 1, 19), 0)
    selected_path = images[img_idx]
    
    use_seg = st.checkbox("Gunakan Segmentasi Foreground (GrabCut)", value=False, key="pp_seg")
    
    # Preprocessing
    result = preprocess_image(selected_path, use_segmentation=use_seg)
    
    if result is None:
        st.error("Gagal memproses gambar.")
        st.stop()
    
    # Tampilkan hasil preprocessing
    st.markdown("### 🔄 Hasil Preprocessing")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("**Gambar Asli (RGB)**")
        st.image(result['img_rgb'], use_container_width=True)
    
    with col2:
        st.markdown("**Grayscale**")
        st.image(result['img_gray'], use_container_width=True)
    
    with col3:
        st.markdown("**Mask Segmentasi**")
        mask_display = (result['mask'].astype(np.uint8) * 255)
        st.image(mask_display, use_container_width=True)
    
    with col4:
        st.markdown("**Foreground**")
        fg = result['img_rgb'].copy()
        fg[~result['mask']] = 0
        st.image(fg, use_container_width=True)
    
    if use_seg and not result['segmentation_success']:
        st.warning("⚠️ Segmentasi GrabCut gagal. Menggunakan seluruh gambar sebagai fallback.")
    
    # Perbandingan ruang warna
    st.markdown("### 🎨 Perbandingan Ruang Warna")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**RGB**")
        st.image(result['img_rgb'], use_container_width=True)
    
    with col2:
        st.markdown("**HSV**")
        # Normalize HSV untuk tampilan
        hsv_display = result['img_hsv'].copy().astype(np.float32)
        hsv_display[:, :, 0] = hsv_display[:, :, 0] / 180.0
        hsv_display[:, :, 1] = hsv_display[:, :, 1] / 255.0
        hsv_display[:, :, 2] = hsv_display[:, :, 2] / 255.0
        st.image(hsv_display, use_container_width=True, clamp=True)
    
    with col3:
        st.markdown("**CIELAB**")
        lab_display = result['img_lab'].copy().astype(np.float32) / 255.0
        st.image(lab_display, use_container_width=True, clamp=True)
    
    # Ringkasan fitur
    st.markdown("### 📐 Ringkasan Fitur")
    
    for cs in ['RGB', 'HSV', 'CIELAB']:
        with st.expander(f"Fitur Warna {cs}"):
            features, names = extract_color_features(
                cs, result['img_rgb'], result['img_hsv'], 
                result['img_lab'], result['mask']
            )
            feat_df = pd.DataFrame({
                'Fitur': names,
                'Nilai': [f"{v:.4f}" for v in features],
            })
            st.dataframe(feat_df, use_container_width=True, hide_index=True)
    
    with st.expander("Fitur GLCM"):
        glcm_features, glcm_names = extract_glcm_features(
            result['img_gray'], result['mask']
        )
        glcm_df = pd.DataFrame({
            'Fitur': glcm_names,
            'Nilai': [f"{v:.6f}" for v in glcm_features],
        })
        st.dataframe(glcm_df, use_container_width=True, hide_index=True)
        
        st.markdown("""
        **Parameter GLCM yang digunakan:**
        - Kuantisasi: 32 tingkat abu-abu
        - Jarak piksel: 1, 3
        - Sudut: 0°, 45°, 90°, 135°
        - Properti: contrast, dissimilarity, homogeneity, energy, correlation
        - Agregasi: rata-rata semua sudut per jarak
        """)


# ==================== HALAMAN TRAINING ====================
elif page == "⚙️ Training & Eksperimen":
    show_header("Training & Eksperimen",
                "Latih Model SVM untuk Enam Eksperimen Klasifikasi")
    
    info = load_dataset_cached()
    if not info['is_valid']:
        st.error("Dataset tidak valid. Periksa halaman Dataset.")
        st.stop()
    
    # Pengaturan
    st.markdown("### ⚙️ Pengaturan Eksperimen")
    
    col1, col2 = st.columns(2)
    with col1:
        quick_mode = st.checkbox(
            "🚀 Mode Cepat (subset data untuk pengujian)",
            value=st.session_state.get('quick_mode', True),
            help=f"Gunakan maksimal {QUICK_MODE_SAMPLES_PER_CLASS} sampel per kelas untuk mempercepat pengujian."
        )
        st.session_state['quick_mode'] = quick_mode
    
    with col2:
        use_seg = st.checkbox(
            "✂️ Gunakan Segmentasi GrabCut",
            value=st.session_state.get('use_segmentation', False),
            help="Segmentasi foreground menggunakan GrabCut. Lebih lambat tapi bisa meningkatkan akurasi."
        )
        st.session_state['use_segmentation'] = use_seg
    
    # Info konfigurasi
    st.markdown("### 📋 Konfigurasi SVM")
    cfg_col1, cfg_col2, cfg_col3, cfg_col4 = st.columns(4)
    with cfg_col1:
        st.metric("Kernel", SVM_KERNEL.upper())
    with cfg_col2:
        st.metric("C", str(SVM_C))
    with cfg_col3:
        st.metric("Gamma", str(SVM_GAMMA))
    with cfg_col4:
        st.metric("Class Weight", str(SVM_CLASS_WEIGHT))
    
    # Info data
    max_per_class = QUICK_MODE_SAMPLES_PER_CLASS if quick_mode else None
    mode_text = f"Mode Cepat (max {QUICK_MODE_SAMPLES_PER_CLASS}/kelas)" if quick_mode else "Mode Lengkap"
    
    st.info(f"**Mode:** {mode_text} | **Label:** {st.session_state['label_mode']} | **Segmentasi:** {'Ya' if use_seg else 'Tidak'}")
    
    # Status model
    st.markdown("### 📦 Status Model")
    available = get_available_models()
    
    status_data = []
    for exp_id, exp_name in EXPERIMENT_NAMES.items():
        status_data.append({
            'No': exp_id,
            'Eksperimen': exp_name,
            'Status': '✅ Terlatih' if exp_id in available else '❌ Belum Dilatih',
        })
    st.dataframe(pd.DataFrame(status_data), use_container_width=True, hide_index=True)
    
    # Tombol training
    st.markdown("### 🚀 Mulai Training")
    
    if st.button("🎯 Latih Semua 6 Eksperimen", type="primary", use_container_width=True):
        classes = info['classes']
        label_mode = st.session_state['label_mode']
        dataset_path = st.session_state['dataset_path']
        
        # Load train data
        progress_bar = st.progress(0, text="Memuat data training...")
        
        try:
            train_paths, train_labels = load_split_data(
                dataset_path, 'train', classes, label_mode,
                max_per_class=max_per_class,
            )
        except Exception as e:
            st.error(f"Gagal memuat data training: {e}")
            st.stop()
        
        st.info(f"📊 Jumlah data training: **{len(train_paths)}** gambar")
        
        training_results = {}
        total_start = time.time()
        
        for exp_id in range(1, 7):
            exp_name = EXPERIMENT_NAMES[exp_id]
            st.markdown(f"---")
            st.markdown(f"#### Eksperimen {exp_id}: {exp_name}")
            
            exp_progress = st.progress(0, text=f"Memulai {exp_name}...")
            exp_status = st.empty()
            
            def progress_cb(current, total, msg):
                if total > 0:
                    exp_progress.progress(current / total, text=msg)
            
            exp_start = time.time()
            
            result = train_experiment(
                experiment_id=exp_id,
                train_paths=train_paths,
                train_labels=train_labels,
                dataset_path=dataset_path,
                use_segmentation=use_seg,
                label_mode=label_mode,
                max_per_class=max_per_class,
                progress_callback=progress_cb,
            )
            
            training_results[exp_id] = result
            
            if result['status'] == 'success':
                exp_progress.progress(1.0, text=f"✅ {exp_name} selesai!")
                exp_status.success(
                    f"✅ **{exp_name}**: {result['n_train_samples']} sampel, "
                    f"{result['n_features']} fitur, "
                    f"waktu: {format_duration(result['total_time_s'])}"
                )
            else:
                exp_progress.progress(1.0, text=f"❌ {exp_name} gagal!")
                exp_status.error(f"❌ **{exp_name}**: {result.get('error', 'Unknown error')}")
        
        total_time = time.time() - total_start
        progress_bar.progress(1.0, text="Training selesai!")
        
        st.session_state['training_results'] = training_results
        st.session_state['training_done'] = True
        
        st.success(f"🎉 Training selesai! Total waktu: {format_duration(total_time)}")
    
    # Tampilkan hasil training sebelumnya
    if st.session_state.get('training_done') and st.session_state.get('training_results'):
        st.markdown("### 📋 Hasil Training Terakhir")
        tr_data = []
        for exp_id, tr in st.session_state['training_results'].items():
            if tr['status'] == 'success':
                tr_data.append({
                    'Eksperimen': tr['experiment_name'],
                    'Sampel': tr['n_train_samples'],
                    'Fitur': tr['n_features'],
                    'Waktu Ekstraksi': f"{tr['extract_time_s']:.1f}s",
                    'Waktu Training': f"{tr['train_time_s']:.1f}s",
                    'Total': f"{tr['total_time_s']:.1f}s",
                })
        if tr_data:
            st.dataframe(pd.DataFrame(tr_data), use_container_width=True, hide_index=True)


# ==================== HALAMAN EVALUASI ====================
elif page == "📊 Evaluasi":
    show_header("Evaluasi Hasil", "Perbandingan Performa Enam Eksperimen")
    
    info = load_dataset_cached()
    if not info['is_valid']:
        st.error("Dataset tidak valid.")
        st.stop()
    
    available = get_available_models()
    if not available:
        st.warning("⚠️ Belum ada model yang dilatih. Jalankan training terlebih dahulu di halaman 'Training & Eksperimen'.")
        st.stop()
    
    # Pilih split evaluasi
    eval_split = st.selectbox(
        "Split Evaluasi:",
        [s for s in ['test', 'valid'] if s in info['splits']],
        index=0,
    )
    
    label_mode = st.session_state['label_mode']
    use_seg = st.session_state.get('use_segmentation', False)
    quick_mode = st.session_state.get('quick_mode', True)
    
    if st.button("📊 Evaluasi Semua Model", type="primary", use_container_width=True):
        classes = info['classes']
        dataset_path = st.session_state['dataset_path']
        
        # Load data evaluasi
        with st.spinner("Memuat data evaluasi..."):
            try:
                eval_paths, eval_labels = load_split_data(
                    dataset_path, eval_split, classes, label_mode,
                )
            except Exception as e:
                st.error(f"Gagal memuat data: {e}")
                st.stop()
        
        st.info(f"📊 Jumlah data {eval_split}: **{len(eval_paths)}** gambar")
        
        eval_results = {}
        eval_bar = st.progress(0, text="Memulai evaluasi...")
        
        for idx, exp_id in enumerate(sorted(available.keys())):
            exp_name = EXPERIMENT_NAMES[exp_id]
            eval_bar.progress(
                (idx) / len(available),
                text=f"Mengevaluasi {exp_name}..."
            )
            
            result = evaluate_on_split(
                exp_id, eval_paths, eval_labels,
                dataset_path, eval_split, use_seg, label_mode,
            )
            
            if result and 'error' not in result:
                eval_results[exp_id] = result
        
        eval_bar.progress(1.0, text="Evaluasi selesai!")
        
        if eval_results:
            st.session_state['eval_results'] = eval_results
            st.session_state['eval_done'] = True
            
            # Simpan hasil
            df_summary = create_metrics_summary(eval_results)
            save_metrics_summary(df_summary)
            
            # Simpan confusion matrix dan reports
            for exp_id, r in eval_results.items():
                plot_confusion_matrix(
                    r['confusion_matrix'], r['classes'], r['experiment_name']
                )
                save_classification_report(
                    r['classification_report_text'], r['experiment_name']
                )
            
            # Plot perbandingan
            plot_comparison_chart(eval_results)
            
            # Simpan metadata
            training_results = st.session_state.get('training_results', {})
            config = {
                'label_mode': label_mode,
                'quick_mode': quick_mode,
                'use_segmentation': use_seg,
                'svm_kernel': SVM_KERNEL,
                'svm_c': SVM_C,
                'svm_gamma': SVM_GAMMA,
                'class_weight': SVM_CLASS_WEIGHT,
                'eval_split': eval_split,
            }
            save_experiment_metadata(eval_results, training_results, config)
            
            # Generate laporan
            generate_report_markdown(eval_results, training_results, config)
            
            st.success("✅ Hasil evaluasi disimpan ke folder `results/`")
    
    # Tampilkan hasil evaluasi
    if st.session_state.get('eval_done') and st.session_state.get('eval_results'):
        eval_results = st.session_state['eval_results']
        
        # Tabel ringkasan
        st.markdown("### 📋 Tabel Perbandingan Enam Eksperimen")
        df_summary = create_metrics_summary(eval_results)
        
        # Highlight baris terbaik
        st.dataframe(
            df_summary.style.highlight_max(
                subset=['Accuracy', 'F1-Score (Macro)'],
                color='#C8E6C9',
            ),
            use_container_width=True,
            hide_index=True,
        )
        
        # Metode terbaik
        if not df_summary.empty:
            best_idx = df_summary['F1-Score (Macro)'].idxmax()
            best_row = df_summary.iloc[best_idx]
            st.success(
                f"🏆 **Metode terbaik:** {best_row['Eksperimen']} "
                f"(F1-Score: {best_row['F1-Score (Macro)']:.4f}, "
                f"Accuracy: {best_row['Accuracy']:.4f})"
            )
        
        # Grafik perbandingan
        st.markdown("### 📈 Grafik Perbandingan")
        chart_path = os.path.join(PLOTS_DIR, 'perbandingan_metrik.png')
        if os.path.exists(chart_path):
            st.image(chart_path, use_container_width=True)
        
        # Confusion matrices
        st.markdown("### 🔢 Confusion Matrix")
        cm_cols = st.columns(3)
        
        for idx, (exp_id, r) in enumerate(sorted(eval_results.items())):
            if 'error' in r:
                continue
            with cm_cols[idx % 3]:
                safe_name = r['experiment_name'].replace(' ', '_').replace('+', 'plus')
                cm_path = os.path.join(CONFUSION_MATRICES_DIR, f'cm_{safe_name}.png')
                if os.path.exists(cm_path):
                    st.image(cm_path, caption=r['experiment_name'],
                             use_container_width=True)
        
        # Classification reports
        st.markdown("### 📄 Classification Reports")
        for exp_id in sorted(eval_results.keys()):
            r = eval_results[exp_id]
            if 'error' in r:
                continue
            with st.expander(f"Report: {r['experiment_name']}"):
                st.text(r['classification_report_text'])


# ==================== HALAMAN PREDIKSI ====================
elif page == "🔮 Prediksi":
    show_header("Prediksi Gambar Baru",
                "Unggah Gambar Pisang dan Dapatkan Prediksi Tingkat Kematangan")
    
    available = get_available_models()
    
    if not available:
        st.warning(
            "⚠️ Belum ada model yang dilatih. "
            "Silakan jalankan pelatihan di halaman **Training & Eksperimen** terlebih dahulu."
        )
        st.stop()
    
    # Upload gambar
    st.markdown("### 📤 Unggah Gambar")
    uploaded = st.file_uploader(
        "Pilih gambar pisang (JPG, PNG):",
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=False,
    )
    
    if uploaded is not None:
        # Simpan file sementara
        temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'artifacts', 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, uploaded.name)
        
        with open(temp_path, 'wb') as f:
            f.write(uploaded.getbuffer())
        
        # Tampilkan gambar
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📷 Gambar Asli")
            pil_img = Image.open(uploaded)
            st.image(pil_img, use_container_width=True)
        
        with col2:
            st.markdown("#### 🔄 Hasil Preprocessing")
            result = preprocess_image(temp_path)
            if result:
                st.image(result['img_rgb'], use_container_width=True)
            else:
                st.error("Gagal memproses gambar.")
        
        # Pilih model
        st.markdown("### 🤖 Pilih Model")
        
        model_options = {
            exp_id: f"Eksperimen {exp_id}: {EXPERIMENT_NAMES[exp_id]}"
            for exp_id in sorted(available.keys())
        }
        
        selected_exp = st.selectbox(
            "Model untuk prediksi:",
            options=list(model_options.keys()),
            format_func=lambda x: model_options[x],
        )
        
        col_single, col_all = st.columns(2)
        
        with col_single:
            if st.button("🔮 Prediksi dengan Model Terpilih", type="primary", use_container_width=True):
                with st.spinner("Memproses prediksi..."):
                    pred = predict_image(temp_path, selected_exp)
                
                if pred and not pred.get('error', False):
                    st.markdown("---")
                    st.markdown("### 🎯 Hasil Prediksi")
                    
                    st.markdown(f"""
                    <div class="metric-card" style="border-left-color: #4CAF50;">
                        <h3>{pred['prediction']}</h3>
                        <p>{pred['experiment_name']} | {pred['color_space']} | GLCM: {'Ya' if pred['use_glcm'] else 'Tidak'}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Ringkasan fitur
                    with st.expander("📐 Detail Fitur"):
                        feat_df = pd.DataFrame({
                            'Fitur': pred['feature_names'],
                            'Nilai': [f"{v:.4f}" for v in pred['feature_values']],
                        })
                        st.dataframe(feat_df, use_container_width=True, hide_index=True)
                    
                    if pred.get('decision_values'):
                        with st.expander("📊 Decision Values (bukan probabilitas)"):
                            st.caption("⚠️ Nilai ini adalah decision function SVM, bukan probabilitas.")
                            if len(pred['classes']) > 2:
                                dv_df = pd.DataFrame({
                                    'Kelas': pred['classes'],
                                    'Decision Value': [f"{v:.4f}" for v in pred['decision_values']],
                                })
                            else:
                                dv_df = pd.DataFrame({
                                    'Decision Value': [f"{pred['decision_values'][0]:.4f}" if isinstance(pred['decision_values'], list) else f"{pred['decision_values']:.4f}"],
                                })
                            st.dataframe(dv_df, use_container_width=True, hide_index=True)
                elif pred:
                    st.error(pred.get('message', 'Terjadi kesalahan.'))
        
        with col_all:
            if st.button("🔮 Prediksi dengan Semua Model", use_container_width=True):
                with st.spinner("Memproses prediksi semua model..."):
                    all_preds = predict_all_models(temp_path)
                
                st.markdown("---")
                st.markdown("### 🎯 Perbandingan Prediksi Semua Model")
                
                pred_rows = []
                for exp_id in sorted(all_preds.keys()):
                    p = all_preds[exp_id]
                    if not p.get('error', False):
                        pred_rows.append({
                            'Eksperimen': p['experiment_name'],
                            'Prediksi': p['prediction'],
                            'Ruang Warna': p['color_space'],
                            'GLCM': '✅' if p['use_glcm'] else '❌',
                        })
                    else:
                        pred_rows.append({
                            'Eksperimen': EXPERIMENT_NAMES.get(exp_id, f'Exp {exp_id}'),
                            'Prediksi': '⚠️ ' + p.get('message', 'Error')[:50],
                            'Ruang Warna': '-',
                            'GLCM': '-',
                        })
                
                if pred_rows:
                    st.dataframe(pd.DataFrame(pred_rows), use_container_width=True, hide_index=True)
        
        # Cleanup temp
        try:
            os.remove(temp_path)
        except Exception:
            pass


# ==================== HALAMAN TENTANG ====================
elif page == "📖 Tentang & Metodologi":
    show_header("Tentang & Metodologi",
                "Penjelasan Metode, Konfigurasi, dan Sumber Dataset")
    
    # RGB
    with st.expander("🔴🟢🔵 Ruang Warna RGB", expanded=False):
        st.markdown("""
        **RGB (Red, Green, Blue)** adalah model warna aditif yang merepresentasikan 
        warna sebagai kombinasi tiga komponen: merah, hijau, dan biru.
        
        - Setiap kanal memiliki range 0-255.
        - Merupakan representasi default pada kebanyakan kamera dan layar digital.
        - Dalam OpenCV, gambar dibaca dalam format BGR, sehingga perlu konversi ke RGB.
        
        **Fitur yang diekstrak:** Mean, standar deviasi, median, dan histogram 
        ternormalisasi (8 bin) untuk setiap kanal R, G, dan B.
        """)
    
    # HSV
    with st.expander("🌈 Ruang Warna HSV", expanded=False):
        st.markdown("""
        **HSV (Hue, Saturation, Value)** memisahkan informasi warna (Hue) 
        dari kecerahan (Value) dan saturasi.
        
        - **Hue (H)**: Komponen warna, range 0-179 pada OpenCV.
        - **Saturation (S)**: Tingkat kejenuhan warna, range 0-255.
        - **Value (V)**: Kecerahan, range 0-255.
        
        HSV lebih tahan terhadap variasi pencahayaan dibanding RGB 
        karena informasi warna terpisah dari intensitas.
        
        **Fitur yang diekstrak:** Statistik dan histogram ternormalisasi 
        untuk setiap kanal H, S, dan V.
        """)
    
    # CIELAB
    with st.expander("🎨 Ruang Warna CIELAB", expanded=False):
        st.markdown("""
        **CIELAB (CIE L*a*b*)** dirancang untuk mendekati persepsi warna manusia.
        
        - **L* (Lightness)**: Kecerahan, 0 (hitam) hingga 100 (putih).
        - **a***: Posisi antara merah/magenta dan hijau.
        - **b***: Posisi antara kuning dan biru.
        
        Dalam OpenCV (8-bit), nilai di-skala ke range 0-255.
        Keunggulan: perceptually uniform, sehingga perbedaan numerik 
        lebih merepresentasikan perbedaan visual.
        
        **Fitur yang diekstrak:** Statistik dan histogram ternormalisasi 
        untuk setiap kanal L, a, dan b.
        """)
    
    # GLCM
    with st.expander("📐 Gray-Level Co-occurrence Matrix (GLCM)", expanded=False):
        st.markdown(f"""
        **GLCM** menangkap pola tekstur dengan menganalisis distribusi 
        pasangan piksel pada jarak dan arah tertentu.
        
        **Parameter yang digunakan:**
        - Kuantisasi grayscale: {GLCM_GRAY_LEVELS} tingkat
        - Jarak piksel: {GLCM_DISTANCES}
        - Sudut: {GLCM_ANGLES_DEG}° (0°, 45°, 90°, 135°)
        - Resolusi pemrosesan: {GLCM_RESIZE}
        
        **Properti yang dihitung:**
        | Properti | Deskripsi |
        |---|---|
        | **Contrast** | Mengukur perbedaan intensitas antar piksel berpasangan |
        | **Dissimilarity** | Mirip contrast, tetapi lebih linier |
        | **Homogeneity** | Mengukur kedekatan distribusi elemen GLCM ke diagonal |
        | **Energy** | Mengukur uniformitas (keseragaman tekstur) |
        | **Correlation** | Mengukur korelasi linier antar piksel berpasangan |
        
        **Agregasi:** Rata-rata semua sudut per jarak, menghasilkan 
        {len(GLCM_PROPERTIES)} × {len(GLCM_DISTANCES)} = {len(GLCM_PROPERTIES) * len(GLCM_DISTANCES)} fitur.
        """)
    
    # SVM
    with st.expander("🤖 Support Vector Machine (SVM)", expanded=False):
        st.markdown(f"""
        **SVM** adalah algoritma supervised learning yang mencari hyperplane 
        optimal untuk memisahkan kelas data.
        
        **Konfigurasi yang digunakan:**
        - **Kernel:** {SVM_KERNEL.upper()} (Radial Basis Function)
        - **C (Regularization):** {SVM_C}
        - **Gamma:** {SVM_GAMMA}
        - **Class Weight:** {SVM_CLASS_WEIGHT}
        - **Pipeline:** StandardScaler → SVC
        
        **Keputusan desain:**
        - Parameter SVM yang **sama** digunakan untuk seluruh 6 eksperimen 
          agar perbandingan adil.
        - StandardScaler diterapkan pada fitur agar setiap fitur memiliki 
          mean=0 dan std=1.
        - `class_weight='balanced'` untuk menangani ketidakseimbangan kelas.
        - Scaling dihitung **hanya** dari data training.
        """)
    
    # Preprocessing
    with st.expander("🔄 Konfigurasi Preprocessing", expanded=False):
        st.markdown(f"""
        **Pipeline preprocessing:**
        1. Baca gambar (OpenCV, BGR)
        2. Resize ke {TARGET_SIZE} dengan menjaga aspek rasio (padding hitam)
        3. Konversi ke RGB, HSV, dan CIELAB
        4. Konversi ke grayscale (untuk GLCM)
        5. Segmentasi foreground (opsional, GrabCut)
        
        **Catatan:**
        - Ukuran gambar diseragamkan untuk konsistensi fitur.
        - Segmentasi GrabCut memiliki fallback jika gagal.
        - Preprocessing yang sama diterapkan pada training dan prediksi.
        """)
    
    # Dataset
    with st.expander("📂 Sumber Dataset", expanded=False):
        st.markdown("""
        **Dataset:** Banana Ripeness Classification v6
        
        **Sumber:** [Roboflow Universe](https://universe.roboflow.com/roboflow-universe-projects/banana-ripeness-classification)
        
        **Lisensi:** CC BY 4.0 (Creative Commons Attribution 4.0 International)
        
        **Atribusi:** Dataset disediakan oleh kontributor Roboflow Universe.
        
        **Preprocessing dataset (dari sumber):**
        - Auto-orientation (EXIF stripping)
        - Resize ke 416×416 (stretch)
        - Augmentasi: flip, rotasi, crop, brightness, exposure, blur
        """)
    
    # Keterbatasan
    with st.expander("⚠️ Keterbatasan Penelitian", expanded=False):
        st.markdown("""
        1. **Handcrafted features:** Fitur dirancang manual, bukan dipelajari 
           secara otomatis seperti deep learning.
        2. **Parameter SVM tetap:** Tidak dilakukan hyperparameter tuning 
           individual per eksperimen untuk menjaga keadilan perbandingan.
        3. **Segmentasi sederhana:** GrabCut mungkin tidak optimal 
           untuk semua kondisi gambar.
        4. **Resolusi:** Gambar di-resize ke ukuran kecil, 
           yang dapat mengurangi detail tekstur.
        5. **Augmentasi:** Tidak ada augmentasi tambahan saat training.
        6. **Generalisasi:** Hasil mungkin berbeda pada dataset lain 
           atau kondisi pengambilan gambar yang berbeda.
        """)
    
    # Info teknis
    st.markdown("### 🔧 Informasi Teknis")
    st.markdown(f"""
    - **Histogram bins:** {HISTOGRAM_BINS} per kanal
    - **Dimensi fitur warna:** {3 * (3 + HISTOGRAM_BINS)} per ruang warna
    - **Dimensi fitur GLCM:** {len(GLCM_PROPERTIES) * len(GLCM_DISTANCES)}
    - **Total fitur (warna saja):** {3 * (3 + HISTOGRAM_BINS)}
    - **Total fitur (warna + GLCM):** {3 * (3 + HISTOGRAM_BINS) + len(GLCM_PROPERTIES) * len(GLCM_DISTANCES)}
    - **Random seed:** {RANDOM_SEED}
    """)
