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
    GLCM_GRAY_LEVELS, GLCM_DISTANCES, GLCM_ANGLES_DEG, GLCM_PROPERTIES, GLCM_RESIZE,
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
    segment_foreground, apply_mask,
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
# ==================== HALAMAN PREDIKSI ====================
elif page == "🔮 Prediksi":
    show_header("Prediksi & Pipeline Pemrosesan Citra Digital",
                "Visualisasi Lengkap Preprocessing (Segmentasi, Grayscale, Foreground), Ekstraksi Fitur Warna-Tekstur GLCM, dan Klasifikasi SVM")
    
    available = get_available_models()
    
    if not available:
        st.warning(
            "⚠️ Belum ada model yang dilatih. "
            "Silakan jalankan pelatihan di halaman **Training & Eksperimen** terlebih dahulu."
        )
        st.stop()
    
    # 1. Sumber citra: Upload atau Sampel Dataset
    st.markdown("### 📥 1. Pilih Citra Masukan")
    input_source = st.radio(
        "Pilih metode pemilihan citra:",
        ["📤 Unggah Gambar Baru", "📂 Gunakan Sampel dari Dataset"],
        horizontal=True,
    )
    
    selected_image_path = None
    input_filename = ""
    
    if input_source == "📤 Unggah Gambar Baru":
        uploaded = st.file_uploader(
            "Pilih gambar buah pisang (JPG, JPEG, PNG):",
            type=['jpg', 'jpeg', 'png'],
            accept_multiple_files=False,
        )
        if uploaded is not None:
            temp_dir = os.path.join(PROJECT_ROOT, 'artifacts', 'temp')
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = os.path.join(temp_dir, uploaded.name)
            with open(temp_path, 'wb') as f:
                f.write(uploaded.getbuffer())
            selected_image_path = temp_path
            input_filename = uploaded.name
    else:
        info = load_dataset_cached()
        if not info['is_valid']:
            st.error("Dataset tidak valid. Periksa konfigurasi di halaman Dataset.")
        else:
            col_s1, col_s2, col_s3 = st.columns([1, 1, 2])
            with col_s1:
                splits_list = list(info['splits'].keys())
                default_split_idx = splits_list.index('test') if 'test' in splits_list else 0
                ds_split = st.selectbox("Pilih Split Dataset:", splits_list, index=default_split_idx)
            with col_s2:
                ds_class = st.selectbox("Pilih Kelas Kematangan:", info['classes'])
            
            split_folder = info['splits'][ds_split]
            cls_folder = os.path.join(split_folder, ds_class)
            sample_imgs = get_image_paths(cls_folder)
            
            if sample_imgs:
                with col_s3:
                    s_idx = st.slider("Pilih Sampel Gambar:", 0, min(len(sample_imgs) - 1, 30), 0)
                    selected_image_path = sample_imgs[s_idx]
                    input_filename = os.path.basename(selected_image_path)
            else:
                st.warning("Tidak ada gambar ditemukan pada folder kelas ini.")
    
    if selected_image_path and os.path.exists(selected_image_path):
        raw_bgr = read_image(selected_image_path)
        if raw_bgr is None:
            st.error("Gagal membaca file citra. Pastikan format file didukung.")
            st.stop()
        
        orig_h, orig_w = raw_bgr.shape[:2]
        raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
        
        # Preprocessing: Jalankan segmentasi GrabCut & non-segmentasi
        with st.spinner("Memproses tahapan pemrosesan citra digital..."):
            prep_seg = preprocess_image(selected_image_path, use_segmentation=True)
            prep_unseg = preprocess_image(selected_image_path, use_segmentation=False)
        
        if prep_seg is None or prep_unseg is None:
            st.error("Gagal memproses tahapan preprocessing citra.")
            st.stop()
        
        # Komponen Citra Preprocessing
        resized_rgb = prep_seg['img_rgb']
        resized_gray = prep_seg['img_gray']
        seg_mask = prep_seg['mask']
        seg_success = prep_seg['segmentation_success']
        mask_vis = (seg_mask.astype(np.uint8) * 255)
        
        # Foreground RGB (Latar belakang hitam)
        fg_rgb = resized_rgb.copy()
        fg_rgb[~seg_mask] = 0
        
        # Kontur Segmentasi
        contour_img = resized_rgb.copy()
        contours, _ = cv2.findContours(seg_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
        
        # Statistik Luas Foreground
        fg_pixels = int(np.sum(seg_mask))
        total_pixels = int(seg_mask.size)
        fg_percentage = (fg_pixels / total_pixels) * 100.0
        
        # ================= TAHAPAN PIPELINE PCD =================
        st.markdown("---")
        st.markdown("### 🔄 2. Tahapan Pemrosesan Citra Digital (Pipeline)")
        st.caption("Alur visualisasi transformasi citra: Dari citra masukan asli, penyeragaman ukuran, konversi keabuan, segmentasi GrabCut, hingga ekstraksi objek foreground.")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.markdown("**1. Citra Asli**")
            st.image(raw_rgb, use_container_width=True)
            st.caption(f"📏 {orig_w}×{orig_h} px (RGB Asli)")
            
        with col2:
            st.markdown("**2. Resize & Padding**")
            st.image(resized_rgb, use_container_width=True)
            st.caption(f"📐 {TARGET_SIZE[0]}×{TARGET_SIZE[1]} px (Letterbox)")
            
        with col3:
            st.markdown("**3. Grayscale**")
            st.image(resized_gray, use_container_width=True)
            st.caption("⚪ Derajat Keabuan (GLCM)")
            
        with col4:
            st.markdown("**4. Masker Segmentasi**")
            st.image(mask_vis, use_container_width=True)
            st.caption(f"✂️ GrabCut Mask ({fg_percentage:.1f}% FG)")
            
        with col5:
            st.markdown("**5. Foreground Pisang**")
            st.image(fg_rgb, use_container_width=True)
            st.caption("🍌 Objek Tanpa Background")
        
        # ================= DETAIL TAB ANALISIS =================
        tab_pcd, tab_colors, tab_glcm = st.tabs([
            "🔍 Detail Segmentasi & Kontur Objek",
            "🎨 Dekomposisi Ruang Warna (RGB, HSV, CIELAB)",
            "📐 Analisis Tekstur GLCM (Haralick)",
        ])
        
        with tab_pcd:
            col_det1, col_det2, col_det3 = st.columns([1, 1, 1.2])
            with col_det1:
                st.markdown("**Deteksi Batas Fisik Objek (Kontur Hijau)**")
                st.image(contour_img, use_container_width=True)
                st.caption("Garis batas hijau menunjukkan kontur terluar buah pisang hasil segmentasi.")
            with col_det2:
                st.markdown("**Foreground Grayscale (Tekstur Buah)**")
                fg_gray = resized_gray.copy()
                fg_gray[~seg_mask] = 0
                st.image(fg_gray, use_container_width=True)
                st.caption("Hanya piksel pisang dalam derajat keabuan (menghilangkan pengaruh tekstur latar belakang).")
            with col_det3:
                st.markdown("**Statistik Pemrosesan Segmentasi**")
                status_text = "✅ Berhasil (GrabCut konvergen)" if seg_success else "⚠️ Fallback (Citra Penuh)"
                st.markdown(f"""
                <div style="background: rgba(76, 175, 80, 0.1); padding: 1.1rem; border-radius: 10px; border-left: 4px solid #4CAF50; border: 1px solid rgba(76, 175, 80, 0.25);">
                    <p style="margin: 0.35rem 0;"><b>Status Segmentasi:</b> {status_text}</p>
                    <p style="margin: 0.35rem 0;"><b>Ukuran Target:</b> {TARGET_SIZE[0]} × {TARGET_SIZE[1]} piksel</p>
                    <p style="margin: 0.35rem 0;"><b>Luas Foreground:</b> {fg_pixels:,} piksel ({fg_percentage:.1f}%)</p>
                    <p style="margin: 0.35rem 0;"><b>Luas Background:</b> {total_pixels - fg_pixels:,} piksel ({100 - fg_percentage:.1f}%)</p>
                    <p style="margin: 0.35rem 0;"><b>Algoritma:</b> GrabCut Iterative Graph-Cut & Gaussian Mixture Models (GMM)</p>
                </div>
                """, unsafe_allow_html=True)
        
        with tab_colors:
            st.markdown("#### Perbandingan 3 Ruang Warna & Kanal")
            st.caption("Pemisahan komponen warna sebelum ekstraksi momen statistik (Mean, Std, Median) dan Histogram 8-bin:")
            
            c_rgb, c_hsv, c_lab = st.columns(3)
            with c_rgb:
                st.markdown("**Ruang Warna RGB**")
                st.image(prep_seg['img_rgb'], use_container_width=True)
                st.caption("R (Red), G (Green), B (Blue)")
            with c_hsv:
                st.markdown("**Ruang Warna HSV**")
                hsv_disp = prep_seg['img_hsv'].copy().astype(np.float32)
                hsv_disp[:, :, 0] /= 180.0
                hsv_disp[:, :, 1] /= 255.0
                hsv_disp[:, :, 2] /= 255.0
                st.image(hsv_disp, use_container_width=True, clamp=True)
                st.caption("H (Hue: Warna), S (Saturation: Kejenuhan), V (Value: Kecerahan)")
            with c_lab:
                st.markdown("**Ruang Warna CIELAB**")
                lab_disp = prep_seg['img_lab'].copy().astype(np.float32) / 255.0
                st.image(lab_disp, use_container_width=True, clamp=True)
                st.caption("L* (Lightness), a* (Hijau-Merah), b* (Biru-Kuning)")
            
            # Plot Histogram Kanal
            st.markdown("##### Histogram Intensitas Warna Objek Pisang (Foreground)")
            hist_cs = st.selectbox("Pilih Ruang Warna untuk Histogram:", ["RGB", "HSV", "CIELAB"], key="pred_hist_cs")
            fig_hist, ax_hist = plt.subplots(figsize=(8, 2.5))
            
            target_img = prep_seg['img_rgb'] if hist_cs == 'RGB' else (prep_seg['img_hsv'] if hist_cs == 'HSV' else prep_seg['img_lab'])
            fg_pixels_masked = apply_mask(target_img, seg_mask)
            
            if len(fg_pixels_masked) > 0:
                ch_labels = {'RGB': ['Red', 'Green', 'Blue'], 'HSV': ['Hue', 'Saturation', 'Value'], 'CIELAB': ['L*', 'a*', 'b*']}[hist_cs]
                ch_colors = {'RGB': ['#E53935', '#43A047', '#1E88E5'], 'HSV': ['#8E24AA', '#FB8C00', '#546E7A'], 'CIELAB': ['#424242', '#00897B', '#FDD835']}[hist_cs]
                
                for ch_i, (lbl, col) in enumerate(zip(ch_labels, ch_colors)):
                    vals = fg_pixels_masked[:, ch_i]
                    ax_hist.hist(vals, bins=32, color=col, alpha=0.5, label=lbl, density=True)
                
                ax_hist.set_xlabel("Nilai Intensitas Kanal")
                ax_hist.set_ylabel("Densitas Frekuensi")
                ax_hist.set_title(f"Distribusi Kanal {hist_cs} pada Area Foreground Pisang", fontsize=10, fontweight='bold')
                ax_hist.legend(loc='upper right')
                ax_hist.grid(True, linestyle='--', alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig_hist)
                plt.close(fig_hist)
            else:
                st.info("Area foreground kosong untuk perhitungan histogram.")
        
        with tab_glcm:
            st.markdown("#### Matriks & Ekstraksi Fitur Tekstur GLCM")
            st.caption("Analisis pola spasial hubungan antar pasangan piksel pada derajat keabuan terkuantisasi (32 level):")
            
            # Hitung GLCM untuk visualisasi
            gray_res = cv2.resize(resized_gray, GLCM_RESIZE, interpolation=cv2.INTER_AREA)
            mask_res = cv2.resize(seg_mask.astype(np.uint8), GLCM_RESIZE, interpolation=cv2.INTER_NEAREST).astype(np.bool_)
            gray_res_quant = (gray_res / (256 // GLCM_GRAY_LEVELS)).astype(np.uint8)
            gray_res_quant[~mask_res] = 0
            
            from skimage.feature import graycomatrix, graycoprops
            glcm_mat = graycomatrix(
                gray_res_quant,
                distances=[1],
                angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                levels=GLCM_GRAY_LEVELS,
                symmetric=True,
                normed=True
            )
            
            col_glcm1, col_glcm2 = st.columns([1.2, 1])
            with col_glcm1:
                st.markdown("**Visualisasi Heatmap GLCM (Jarak d=1, Sudut 0°)**")
                fig_glcm, ax_glcm = plt.subplots(figsize=(4.5, 3.5))
                im = ax_glcm.imshow(glcm_mat[:, :, 0, 0], cmap='viridis', interpolation='nearest')
                ax_glcm.set_title(f"GLCM Matrix ({GLCM_GRAY_LEVELS}×{GLCM_GRAY_LEVELS})", fontsize=10, fontweight='bold')
                ax_glcm.set_xlabel("Tingkat Abu Piksel J (0-31)")
                ax_glcm.set_ylabel("Tingkat Abu Piksel I (0-31)")
                plt.colorbar(im, ax=ax_glcm, fraction=0.046, pad=0.04)
                plt.tight_layout()
                st.pyplot(fig_glcm)
                plt.close(fig_glcm)
            
            with col_glcm2:
                st.markdown("**Nilai 5 Properti Tekstur Haralick**")
                glcm_vals, glcm_n = extract_glcm_features(resized_gray, mask=seg_mask)
                glcm_summary = []
                for prop in GLCM_PROPERTIES:
                    matching = [v for k, v in zip(glcm_n, glcm_vals) if prop in k]
                    avg_val = np.mean(matching) if matching else 0.0
                    glcm_summary.append({
                        "Properti GLCM": prop.capitalize(),
                        "Rata-rata Nilai": f"{avg_val:.6f}",
                    })
                st.dataframe(pd.DataFrame(glcm_summary), use_container_width=True, hide_index=True)
                st.caption("Contrast: Kontras lokal | Dissimilarity: Ketidaksamaan linear | Homogeneity: Keseragaman | Energy: Uniformitas | Correlation: Korelasi spasial")
        
        # ================= PENGATURAN PREDIKSI =================
        st.markdown("---")
        st.markdown("### 🤖 3. Konfigurasi & Jalankan Prediksi")
        
        col_m1, col_m2 = st.columns([1.5, 1.5])
        
        with col_m1:
            model_options = {
                exp_id: f"Eksperimen {exp_id}: {EXPERIMENT_NAMES[exp_id]}"
                for exp_id in sorted(available.keys())
            }
            selected_exp = st.selectbox(
                "Pilih Model Klasifikasi SVM:",
                options=list(model_options.keys()),
                format_func=lambda x: model_options[x],
            )
            
            # Cek metadata model
            model_info_obj = load_trained_model(selected_exp)
            model_is_seg = model_info_obj.get('use_segmentation', False) if model_info_obj else False
            model_cs = model_info_obj.get('color_space', '-') if model_info_obj else '-'
            model_glcm = model_info_obj.get('use_glcm', False) if model_info_obj else False
            
            st.caption(f"ℹ️ Model ini saat training menggunakan: **Ruang Warna: {model_cs}** | **GLCM: {'Ya' if model_glcm else 'Tidak'}** | **Segmentasi: {'Aktif' if model_is_seg else 'Nonaktif (Citra Penuh)'}**")
        
        with col_m2:
            seg_choice = st.radio(
                "Opsi Masking Segmentasi untuk Prediksi:",
                [
                    "Otomatis (Sesuai Konfigurasi Model Saat Training)",
                    "Paksa Gunakan Segmentasi Foreground",
                    "Gunakan Citra Penuh (Tanpa Segmentasi)",
                ],
                index=0,
                help="Menentukan apakah fitur diekstrak dari piksel foreground saja atau seluruh citra.",
            )
            
            if seg_choice == "Otomatis (Sesuai Konfigurasi Model Saat Training)":
                override_seg = None
            elif seg_choice == "Paksa Gunakan Segmentasi Foreground":
                override_seg = True
            else:
                override_seg = False
        
        col_single, col_all = st.columns(2)
        
        # Mapping warna & label untuk hasil prediksi
        ripeness_color_map = {
            'unripe': {'color': '#2E7D32', 'badge': 'Mentah / Hijau', 'icon': '🍏'},
            'freshunripe': {'color': '#43A047', 'badge': 'Segar Belum Matang', 'icon': '🍏'},
            'ripe': {'color': '#F57F17', 'badge': 'Matang Sempurna', 'icon': '🍌'},
            'freshripe': {'color': '#FBC02D', 'badge': 'Segar Matang', 'icon': '🍌'},
            'overripe': {'color': '#E65100', 'badge': 'Terlalu Matang', 'icon': '🍂'},
            'rotten': {'color': '#4E342E', 'badge': 'Busuk / Rusak', 'icon': '🟤'},
            'Mentah': {'color': '#2E7D32', 'badge': 'Mentah', 'icon': '🍏'},
            'Matang': {'color': '#F57F17', 'badge': 'Matang', 'icon': '🍌'},
            'Terlalu Matang': {'color': '#E65100', 'badge': 'Terlalu Matang', 'icon': '🍂'},
            'Busuk': {'color': '#4E342E', 'badge': 'Busuk', 'icon': '🟤'},
        }
        
        with col_single:
            if st.button("🔮 Prediksi dengan Model Terpilih", type="primary", use_container_width=True):
                with st.spinner("Mengekstrak fitur dan menjalankan klasifikasi SVM..."):
                    pred = predict_image(selected_image_path, selected_exp, use_segmentation=override_seg)
                
                if pred and not pred.get('error', False):
                    st.markdown("---")
                    st.markdown("### 🎯 Hasil Prediksi Model")
                    
                    pred_class = pred['prediction']
                    style_info = ripeness_color_map.get(str(pred_class), {'color': '#4CAF50', 'badge': str(pred_class), 'icon': '🍌'})
                    
                    # Indikator Alur Prediksi
                    st.markdown(f"""
                    <div style="background: rgba(139, 195, 74, 0.12); border: 1px solid rgba(139, 195, 74, 0.35); padding: 0.8rem 1rem; border-radius: 8px; margin-bottom: 1rem; font-size: 0.88rem;">
                        <b>Alur Pemrosesan Aktif:</b><br>
                        📁 {input_filename} ➔ 🔄 Resize ({TARGET_SIZE[0]}×{TARGET_SIZE[1]}) ➔ ✂️ Segmentasi ({'Aktif' if pred['use_segmentation'] else 'Nonaktif'}) ➔ 🔬 Ekstraksi Fitur ({pred['color_space']} + {'GLCM' if pred['use_glcm'] else 'Tanpa GLCM'}) ➔ 🤖 SVM ({pred['n_features']} fitur) ➔ <b>{pred_class}</b>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Card Hasil Prediksi
                    st.markdown(f"""
                    <div class="metric-card" style="border-left-color: {style_info['color']}; background: rgba(255, 255, 255, 0.04); text-align: left; padding: 1.5rem; border: 1px solid rgba(128, 128, 128, 0.2);">
                        <span style="font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px; color: #9E9E9E; font-weight: 600;">HASIL KLASIFIKASI KEMATANGAN:</span>
                        <h2 style="margin: 0.2rem 0; color: {style_info['color']}; font-size: 2.2rem; font-weight: 700;">
                            {style_info['icon']} {pred_class}
                        </h2>
                        <p style="margin: 0.2rem 0; font-size: 1rem;">
                            <b>Model:</b> {pred['experiment_name']} &nbsp;|&nbsp; 
                            <b>Ruang Warna:</b> {pred['color_space']} &nbsp;|&nbsp; 
                            <b>Fitur Tekstur GLCM:</b> {'✅ Ya (Fusi Warna+Tekstur)' if pred['use_glcm'] else '❌ Tidak (Warna Saja)'} &nbsp;|&nbsp; 
                            <b>Segmentasi:</b> {'✂️ Foreground' if pred['use_segmentation'] else '🖼️ Citra Utuh'}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Decision Values Visualization
                    if pred.get('decision_values'):
                        st.markdown("#### 📊 Nilai Fungsi Keputusan SVM (Decision Values)")
                        st.caption("Decision Value mengukur jarak margin bertanda sampel terhadap bidang pemisah (hyperplane). Nilai tertinggi menentukan kelas pemenang.")
                        
                        classes = pred['classes']
                        dvals = pred['decision_values']
                        if not isinstance(dvals, list):
                            dvals = [dvals]
                        
                        if len(classes) == len(dvals):
                            fig_dv, ax_dv = plt.subplots(figsize=(7, max(2.5, len(classes) * 0.6)))
                            y_pos = np.arange(len(classes))
                            bar_colors = [style_info['color'] if c == pred_class else '#B0BEC5' for c in classes]
                            
                            bars = ax_dv.barh(y_pos, dvals, color=bar_colors, height=0.55)
                            ax_dv.set_yticks(y_pos)
                            ax_dv.set_yticklabels(classes, fontweight='bold', fontsize=10)
                            ax_dv.axvline(0, color='#78909C', linestyle='--', linewidth=0.8)
                            ax_dv.set_xlabel("Margin / Decision Value")
                            ax_dv.set_title("Margin Jarak Terhadap Hyperplane Tiap Kelas", fontsize=10, fontweight='bold')
                            
                            for bar, val in zip(bars, dvals):
                                offset = 0.05 if val >= 0 else -0.15
                                ax_dv.text(val + offset, bar.get_y() + bar.get_height()/2, f"{val:.3f}", 
                                           va='center', ha='left' if val >= 0 else 'right', fontsize=9, fontweight='bold')
                            
                            plt.tight_layout()
                            st.pyplot(fig_dv)
                            plt.close(fig_dv)
                    
                    # Detail Fitur yang Diekstrak
                    with st.expander(f"📐 Detail {pred['n_features']} Fitur yang Diekstrak untuk Prediksi"):
                        col_f1, col_f2 = st.columns(2)
                        color_feat_names = [f for f in pred['feature_names'] if 'GLCM' not in f]
                        color_feat_vals = [pred['feature_values'][i] for i, f in enumerate(pred['feature_names']) if 'GLCM' not in f]
                        
                        with col_f1:
                            st.markdown(f"**Fitur Warna ({pred['color_space']}) - {len(color_feat_names)} Nilai:**")
                            df_cf = pd.DataFrame({'Fitur': color_feat_names, 'Nilai': [f"{v:.4f}" for v in color_feat_vals]})
                            st.dataframe(df_cf, use_container_width=True, hide_index=True)
                            
                        with col_f2:
                            glcm_feat_names = [f for f in pred['feature_names'] if 'GLCM' in f]
                            if glcm_feat_names:
                                glcm_feat_vals = [pred['feature_values'][i] for i, f in enumerate(pred['feature_names']) if 'GLCM' in f]
                                st.markdown(f"**Fitur Tekstur (GLCM) - {len(glcm_feat_names)} Nilai:**")
                                df_gf = pd.DataFrame({'Fitur': glcm_feat_names, 'Nilai': [f"{v:.6f}" for v in glcm_feat_vals]})
                                st.dataframe(df_gf, use_container_width=True, hide_index=True)
                            else:
                                st.markdown("**Fitur Tekstur (GLCM):**")
                                st.info("Model ini hanya menggunakan fitur warna (tidak menyertakan GLCM).")
                                
                elif pred:
                    st.error(pred.get('message', 'Terjadi kesalahan dalam prediksi.'))
        
        with col_all:
            if st.button("🔮 Prediksi dengan Semua Model (1 - 6)", use_container_width=True):
                with st.spinner("Memproses prediksi pada seluruh 6 model eksperimen..."):
                    all_preds = predict_all_models(selected_image_path, use_segmentation=override_seg)
                
                st.markdown("---")
                st.markdown("### 🎯 Perbandingan Hasil Semua Model")
                
                pred_rows = []
                pred_counts = {}
                for exp_id in sorted(all_preds.keys()):
                    p = all_preds[exp_id]
                    if not p.get('error', False):
                        pred_val = p['prediction']
                        pred_counts[pred_val] = pred_counts.get(pred_val, 0) + 1
                        pred_rows.append({
                            'Exp': exp_id,
                            'Nama Model': p['experiment_name'],
                            'Ruang Warna': p['color_space'],
                            'GLCM': '✅ Ya' if p['use_glcm'] else '❌ Tidak',
                            'Prediksi Kematangan': pred_val,
                        })
                    else:
                        pred_rows.append({
                            'Exp': exp_id,
                            'Nama Model': EXPERIMENT_NAMES.get(exp_id, f'Exp {exp_id}'),
                            'Ruang Warna': '-',
                            'GLCM': '-',
                            'Prediksi Kematangan': '⚠️ ' + p.get('message', 'Error')[:40],
                        })
                
                # Konsensus / Voting
                if pred_counts:
                    majority_class = max(pred_counts, key=pred_counts.get)
                    majority_count = pred_counts[majority_class]
                    total_success = sum(pred_counts.values())
                    st.markdown(f"""
                    <div class="info-box" style="border-left-color: #2196F3;">
                        <b>Konsensus Mayoritas Model:</b> <span style="font-size: 1.15rem; font-weight: 700; color: #1565C0;">{majority_class}</span> 
                        ({majority_count} dari {total_success} model sepakat).
                    </div>
                    """, unsafe_allow_html=True)
                
                if pred_rows:
                    st.dataframe(pd.DataFrame(pred_rows), use_container_width=True, hide_index=True)
                    
                    # Mini bar chart voting
                    if pred_counts:
                        fig_vote, ax_vote = plt.subplots(figsize=(6, 2.4))
                        cls_names = list(pred_counts.keys())
                        cls_votes = [pred_counts[c] for c in cls_names]
                        v_colors = [ripeness_color_map.get(str(c), {}).get('color', '#FFA726') for c in cls_names]
                        ax_vote.bar(cls_names, cls_votes, color=v_colors, width=0.45)
                        ax_vote.set_ylabel("Jumlah Model")
                        ax_vote.set_title("Distribusi Keputusan Klasifikasi (6 Model)", fontsize=10, fontweight='bold')
                        ax_vote.set_yticks(range(0, max(cls_votes) + 2))
                        for i_v, v_val in enumerate(cls_votes):
                            ax_vote.text(i_v, v_val + 0.1, str(v_val), ha='center', fontweight='bold', fontsize=9)
                        ax_vote.grid(axis='y', linestyle='--', alpha=0.3)
                        plt.tight_layout()
                        st.pyplot(fig_vote)
                        plt.close(fig_vote)


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
