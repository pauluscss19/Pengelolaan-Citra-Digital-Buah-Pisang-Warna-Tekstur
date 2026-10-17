"""
evaluation.py
Modul evaluasi: metrik, classification report, confusion matrix, dan visualisasi.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Callable
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import (
    RESULTS_DIR, PLOTS_DIR, CONFUSION_MATRICES_DIR,
    CLASSIFICATION_REPORTS_DIR, EXPERIMENT_NAMES,
    EXPERIMENT_COLOR_SPACES, EXPERIMENT_USE_GLCM,
)
from src.model_training import (
    load_trained_model,
    extract_features_for_split,
    _get_cache_key,
)


def evaluate_on_split(
    experiment_id: int,
    image_paths: List[str],
    labels: List[str],
    dataset_path: str,
    split_name: str = 'test',
    use_segmentation: bool = False,
    label_mode: str = 'original',
    max_per_class: Optional[int] = None,
    progress_callback: Optional[Callable] = None,
) -> Optional[Dict]:
    """
    Evaluasi model pada satu split (valid atau test).
    
    Returns:
        Dict berisi semua metrik evaluasi, atau None jika gagal.
    """
    model_info = load_trained_model(experiment_id)
    if model_info is None:
        return None
    
    pipeline = model_info['pipeline']
    color_space = model_info['color_space']
    use_glcm = model_info['use_glcm']
    exp_name = model_info['experiment_name']
    
    # Cache key
    cache_key = _get_cache_key(
        dataset_path, split_name, color_space, use_glcm,
        use_segmentation, label_mode, max_per_class,
    )
    
    # Ekstrak fitur
    try:
        X, y_true, feature_names, skipped = extract_features_for_split(
            image_paths, labels, color_space, use_glcm,
            use_segmentation, cache_key, progress_callback,
        )
    except Exception as e:
        return {'error': str(e)}
    
    # Prediksi
    y_pred = pipeline.predict(X)
    
    # Hitung metrik
    classes = sorted(list(set(y_true) | set(y_pred)))
    
    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    report = classification_report(
        y_true, y_pred, 
        labels=classes,
        zero_division=0, 
        output_dict=True
    )
    report_text = classification_report(
        y_true, y_pred,
        labels=classes,
        zero_division=0,
    )
    
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    return {
        'experiment_id': experiment_id,
        'experiment_name': exp_name,
        'split': split_name,
        'accuracy': float(acc),
        'precision_macro': float(prec_macro),
        'recall_macro': float(rec_macro),
        'f1_macro': float(f1_macro),
        'classification_report': report,
        'classification_report_text': report_text,
        'confusion_matrix': cm.tolist(),
        'classes': classes,
        'n_samples': len(y_true),
        'n_skipped': len(skipped),
        'y_true': y_true.tolist() if isinstance(y_true, np.ndarray) else list(y_true),
        'y_pred': y_pred.tolist() if isinstance(y_pred, np.ndarray) else list(y_pred),
    }


def run_all_evaluations(
    experiment_ids: List[int],
    test_paths: List[str],
    test_labels: List[str],
    dataset_path: str,
    split_name: str = 'test',
    use_segmentation: bool = False,
    label_mode: str = 'original',
    max_per_class: Optional[int] = None,
    progress_callback: Optional[Callable] = None,
) -> Dict[int, Dict]:
    """
    Evaluasi semua eksperimen pada split yang sama.
    
    Returns:
        Dict experiment_id -> evaluation_result
    """
    results = {}
    for exp_id in experiment_ids:
        if progress_callback:
            progress_callback(
                exp_id, len(experiment_ids),
                f"Mengevaluasi {EXPERIMENT_NAMES.get(exp_id, f'Exp {exp_id}')}..."
            )
        
        result = evaluate_on_split(
            exp_id, test_paths, test_labels, dataset_path,
            split_name, use_segmentation, label_mode, max_per_class,
        )
        if result:
            results[exp_id] = result
    
    return results


def create_metrics_summary(eval_results: Dict[int, Dict]) -> pd.DataFrame:
    """
    Buat tabel ringkasan metrik dari semua eksperimen.
    
    Returns:
        DataFrame dengan kolom: Eksperimen, Accuracy, Precision, Recall, F1-Score
    """
    rows = []
    for exp_id in sorted(eval_results.keys()):
        r = eval_results[exp_id]
        if 'error' in r:
            continue
        rows.append({
            'ID': exp_id,
            'Eksperimen': r['experiment_name'],
            'Accuracy': round(r['accuracy'], 4),
            'Precision (Macro)': round(r['precision_macro'], 4),
            'Recall (Macro)': round(r['recall_macro'], 4),
            'F1-Score (Macro)': round(r['f1_macro'], 4),
            'Jumlah Sampel': r['n_samples'],
        })
    
    return pd.DataFrame(rows)


def save_metrics_summary(df: pd.DataFrame, path: Optional[str] = None):
    """Simpan tabel metrik ke CSV."""
    if path is None:
        path = os.path.join(RESULTS_DIR, 'metrics_summary.csv')
    df.to_csv(path, index=False)


def plot_comparison_chart(
    eval_results: Dict[int, Dict],
    save_path: Optional[str] = None,
) -> str:
    """
    Buat grafik perbandingan accuracy dan F1-score antar eksperimen.
    
    Returns:
        Path file grafik yang disimpan.
    """
    if save_path is None:
        save_path = os.path.join(PLOTS_DIR, 'perbandingan_metrik.png')
    
    exp_names = []
    accuracies = []
    f1_scores = []
    
    for exp_id in sorted(eval_results.keys()):
        r = eval_results[exp_id]
        if 'error' in r:
            continue
        exp_names.append(r['experiment_name'])
        accuracies.append(r['accuracy'])
        f1_scores.append(r['f1_macro'])
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(exp_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy',
                    color='#2196F3', alpha=0.85)
    bars2 = ax.bar(x + width/2, f1_scores, width, label='Macro F1-Score',
                    color='#FF9800', alpha=0.85)
    
    ax.set_xlabel('Eksperimen', fontsize=12)
    ax.set_ylabel('Skor', fontsize=12)
    ax.set_title('Perbandingan Accuracy dan Macro F1-Score', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(exp_names, rotation=30, ha='right', fontsize=10)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)
    
    # Tambahkan label nilai
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=8)
    
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return save_path


def plot_confusion_matrix(
    cm: list,
    classes: List[str],
    exp_name: str,
    save_path: Optional[str] = None,
) -> str:
    """
    Buat visualisasi confusion matrix.
    
    Returns:
        Path file yang disimpan.
    """
    if save_path is None:
        safe_name = exp_name.replace(' ', '_').replace('+', 'plus')
        save_path = os.path.join(CONFUSION_MATRICES_DIR, f'cm_{safe_name}.png')
    
    cm_array = np.array(cm)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm_array, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(
        xticks=np.arange(cm_array.shape[1]),
        yticks=np.arange(cm_array.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        title=f'Confusion Matrix - {exp_name}',
        ylabel='Label Sebenarnya',
        xlabel='Label Prediksi',
    )
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Tampilkan angka
    threshold = cm_array.max() / 2
    for i in range(cm_array.shape[0]):
        for j in range(cm_array.shape[1]):
            ax.text(j, i, format(cm_array[i, j], 'd'),
                     ha='center', va='center',
                     color='white' if cm_array[i, j] > threshold else 'black')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return save_path


def save_classification_report(
    report_text: str,
    exp_name: str,
    save_path: Optional[str] = None,
):
    """Simpan classification report ke file teks."""
    if save_path is None:
        safe_name = exp_name.replace(' ', '_').replace('+', 'plus')
        save_path = os.path.join(
            CLASSIFICATION_REPORTS_DIR, f'report_{safe_name}.txt'
        )
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(f"Classification Report - {exp_name}\n")
        f.write("=" * 60 + "\n\n")
        f.write(report_text)


def save_experiment_metadata(
    eval_results: Dict[int, Dict],
    training_results: Dict[int, Dict],
    config: Dict,
    save_path: Optional[str] = None,
):
    """Simpan metadata eksperimen ke JSON."""
    if save_path is None:
        save_path = os.path.join(RESULTS_DIR, 'experiment_metadata.json')
    
    metadata = {
        'config': config,
        'experiments': {},
    }
    
    for exp_id in sorted(set(list(eval_results.keys()) + list(training_results.keys()))):
        exp_meta = {}
        
        if exp_id in training_results:
            tr = training_results[exp_id]
            exp_meta['training'] = {
                k: v for k, v in tr.items() 
                if k not in ('model_path',) and not callable(v)
            }
        
        if exp_id in eval_results:
            ev = eval_results[exp_id]
            exp_meta['evaluation'] = {
                k: v for k, v in ev.items()
                if k not in ('y_true', 'y_pred') and not callable(v)
            }
        
        metadata['experiments'][str(exp_id)] = exp_meta
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)


def generate_report_markdown(
    eval_results: Dict[int, Dict],
    training_results: Dict[int, Dict],
    config: Dict,
    save_path: Optional[str] = None,
) -> str:
    """
    Generate laporan eksperimen otomatis dalam Markdown.
    
    Returns:
        String konten Markdown.
    """
    if save_path is None:
        save_path = os.path.join(RESULTS_DIR, 'laporan_hasil.md')
    
    lines = []
    lines.append("# Laporan Hasil Eksperimen")
    lines.append("")
    lines.append("## Analisis Pengaruh Ruang Warna dan Fusi Fitur Warna–Tekstur")
    lines.append("## Menggunakan GLCM terhadap Klasifikasi Tingkat Kematangan Buah Pisang dengan SVM")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Konfigurasi
    lines.append("## 1. Konfigurasi Eksperimen")
    lines.append("")
    lines.append(f"- **Mode Label**: {config.get('label_mode', 'original')}")
    lines.append(f"- **Mode**: {'Cepat' if config.get('quick_mode', False) else 'Lengkap'}")
    lines.append(f"- **Segmentasi**: {'Aktif (GrabCut)' if config.get('use_segmentation', False) else 'Nonaktif'}")
    lines.append(f"- **SVM Kernel**: {config.get('svm_kernel', 'rbf')}")
    lines.append(f"- **SVM C**: {config.get('svm_c', 1.0)}")
    lines.append(f"- **SVM Gamma**: {config.get('svm_gamma', 'scale')}")
    lines.append(f"- **Class Weight**: {config.get('class_weight', 'balanced')}")
    lines.append("")
    
    # Data
    if training_results:
        first_tr = next(iter(training_results.values()))
        lines.append("## 2. Data yang Digunakan")
        lines.append("")
        lines.append(f"- **Jumlah sampel training**: {first_tr.get('n_train_samples', 'N/A')}")
        lines.append(f"- **Kelas**: {first_tr.get('classes', 'N/A')}")
        lines.append("")
    
    # Tabel hasil
    lines.append("## 3. Hasil Evaluasi")
    lines.append("")
    lines.append("| No | Eksperimen | Accuracy | Precision | Recall | F1-Score | Sampel |")
    lines.append("|---|---|---|---|---|---|---|")
    
    best_f1 = -1
    best_exp = ""
    
    for exp_id in sorted(eval_results.keys()):
        r = eval_results[exp_id]
        if 'error' in r:
            lines.append(f"| {exp_id} | {r.get('experiment_name', '')} | ERROR | - | - | - | - |")
            continue
        
        lines.append(
            f"| {exp_id} | {r['experiment_name']} | "
            f"{r['accuracy']:.4f} | {r['precision_macro']:.4f} | "
            f"{r['recall_macro']:.4f} | {r['f1_macro']:.4f} | "
            f"{r['n_samples']} |"
        )
        
        if r['f1_macro'] > best_f1:
            best_f1 = r['f1_macro']
            best_exp = r['experiment_name']
    
    lines.append("")
    
    # Analisis
    lines.append("## 4. Analisis Hasil")
    lines.append("")
    
    if best_exp:
        lines.append(f"**Metode dengan performa terbaik (berdasarkan Macro F1-Score):** {best_exp} ({best_f1:.4f})")
        lines.append("")
    
    # Bandingkan warna saja vs fusi
    color_only = {k: v for k, v in eval_results.items() if k <= 3 and 'error' not in v}
    fusion = {k: v for k, v in eval_results.items() if k > 3 and 'error' not in v}
    
    if color_only and fusion:
        avg_f1_color = np.mean([v['f1_macro'] for v in color_only.values()])
        avg_f1_fusion = np.mean([v['f1_macro'] for v in fusion.values()])
        
        lines.append(f"- Rata-rata F1 fitur warna saja: {avg_f1_color:.4f}")
        lines.append(f"- Rata-rata F1 fusi warna+GLCM: {avg_f1_fusion:.4f}")
        
        diff = avg_f1_fusion - avg_f1_color
        if diff > 0.005:
            lines.append(f"- Penambahan fitur GLCM **meningkatkan** performa rata-rata sebesar {diff:.4f}.")
        elif diff < -0.005:
            lines.append(f"- Penambahan fitur GLCM **menurunkan** performa rata-rata sebesar {abs(diff):.4f}.")
        else:
            lines.append(f"- Penambahan fitur GLCM memberikan perubahan yang **sangat kecil** ({diff:.4f}).")
        lines.append("")
    
    # Perbandingan ruang warna
    lines.append("### Perbandingan Ruang Warna")
    for cs in ['RGB', 'HSV', 'CIELAB']:
        color_exp = [k for k, v in eval_results.items() if k <= 3 and v.get('experiment_name', '').startswith(cs)]
        if color_exp:
            exp_id = color_exp[0]
            r = eval_results[exp_id]
            if 'error' not in r:
                lines.append(f"- **{cs}**: Accuracy={r['accuracy']:.4f}, F1={r['f1_macro']:.4f}")
    lines.append("")
    
    # Waktu
    lines.append("## 5. Waktu Komputasi")
    lines.append("")
    lines.append("| Eksperimen | Ekstraksi Fitur (s) | Training (s) | Total (s) |")
    lines.append("|---|---|---|---|")
    for exp_id in sorted(training_results.keys()):
        tr = training_results[exp_id]
        lines.append(
            f"| {tr.get('experiment_name', '')} | "
            f"{tr.get('extract_time_s', '-')} | "
            f"{tr.get('train_time_s', '-')} | "
            f"{tr.get('total_time_s', '-')} |"
        )
    lines.append("")
    
    # Keterbatasan
    lines.append("## 6. Keterbatasan")
    lines.append("")
    lines.append("- Penelitian ini menggunakan fitur handcrafted, bukan deep learning.")
    lines.append("- Parameter SVM yang sama digunakan untuk semua eksperimen tanpa tuning individual.")
    lines.append("- Segmentasi GrabCut mungkin tidak optimal untuk semua gambar.")
    lines.append("- Resolusi gambar diseragamkan, yang dapat mengurangi detail tekstur.")
    lines.append("")
    
    # Kesimpulan
    lines.append("## 7. Kesimpulan")
    lines.append("")
    if best_exp:
        lines.append(f"Berdasarkan eksperimen yang dilakukan, metode **{best_exp}** memberikan "
                      f"performa terbaik dengan Macro F1-Score sebesar **{best_f1:.4f}**.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*Laporan ini dihasilkan secara otomatis oleh sistem.*")
    
    content = "\n".join(lines)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return content
