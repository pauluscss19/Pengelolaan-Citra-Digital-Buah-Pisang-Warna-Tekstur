"""
tests/test_modules.py
Pengujian unit untuk modul-modul project klasifikasi kematangan pisang.
Menggunakan dataset mini sintetis untuk menguji alur kode.
CATATAN: Dataset mini ini BUKAN data eksperimen. Hanya untuk smoke test.
"""

import os
import sys
import tempfile
import shutil
import numpy as np
import cv2
import pytest

# Tambahkan root project ke path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def create_synthetic_dataset(base_dir, classes=None, n_per_class=5, img_size=(64, 64)):
    """
    Buat dataset mini sintetis untuk pengujian.
    BUKAN data eksperimen - hanya untuk smoke test alur kode.
    """
    if classes is None:
        classes = ['unripe', 'ripe', 'overripe', 'rotten']
    
    for split in ['train', 'valid', 'test']:
        for cls in classes:
            cls_dir = os.path.join(base_dir, split, cls)
            os.makedirs(cls_dir, exist_ok=True)
            
            for i in range(n_per_class):
                # Buat gambar acak dengan warna berbeda per kelas
                img = np.random.randint(0, 255, (*img_size, 3), dtype=np.uint8)
                
                # Beri sedikit bias warna per kelas
                if cls == 'unripe':
                    img[:, :, 1] += 50  # Lebih hijau
                elif cls == 'ripe':
                    img[:, :, 0] += 50  # Lebih biru (BGR)
                elif cls == 'overripe':
                    img[:, :, 2] += 50  # Lebih merah
                elif cls == 'rotten':
                    img[:, :, :] = np.clip(img[:, :, :].astype(int) - 50, 0, 255).astype(np.uint8)
                
                img = np.clip(img, 0, 255).astype(np.uint8)
                path = os.path.join(cls_dir, f"test_{cls}_{i}.jpg")
                cv2.imwrite(path, img)
    
    return base_dir


class TestDatasetLoader:
    """Tes modul dataset_loader."""
    
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.dataset_dir = str(tmp_path / "test_dataset")
        create_synthetic_dataset(self.dataset_dir)
    
    def test_find_split_folders(self):
        from src.dataset_loader import find_split_folders
        splits = find_split_folders(self.dataset_dir)
        assert 'train' in splits
        assert 'valid' in splits
        assert 'test' in splits
    
    def test_get_classes(self):
        from src.dataset_loader import get_classes_from_split
        classes = get_classes_from_split(os.path.join(self.dataset_dir, 'train'))
        assert len(classes) == 4
        assert 'unripe' in classes
        assert 'ripe' in classes
    
    def test_load_dataset_info(self):
        from src.dataset_loader import load_dataset_info
        info = load_dataset_info(self.dataset_dir)
        assert info['is_valid']
        assert info['total_images'] == 60  # 4 classes × 5 images × 3 splits
        assert len(info['classes']) == 4
    
    def test_load_split_data(self):
        from src.dataset_loader import load_split_data
        paths, labels = load_split_data(
            self.dataset_dir, 'train', 
            ['unripe', 'ripe', 'overripe', 'rotten']
        )
        assert len(paths) == 20  # 4 × 5
        assert len(labels) == 20
    
    def test_load_split_data_grouped(self):
        from src.dataset_loader import load_split_data
        paths, labels = load_split_data(
            self.dataset_dir, 'train',
            ['unripe', 'ripe', 'overripe', 'rotten'],
            label_mode='grouped',
        )
        unique_labels = set(labels)
        assert 'Mentah' in unique_labels  # unripe → Mentah
        assert 'Matang' in unique_labels  # ripe → Matang
    
    def test_load_split_data_max_per_class(self):
        from src.dataset_loader import load_split_data
        paths, labels = load_split_data(
            self.dataset_dir, 'train',
            ['unripe', 'ripe', 'overripe', 'rotten'],
            max_per_class=2,
        )
        assert len(paths) == 8  # 4 × 2
    
    def test_invalid_dataset_path(self):
        from src.dataset_loader import load_dataset_info
        info = load_dataset_info("/nonexistent/path")
        assert not info['is_valid']
    
    def test_label_mapping_info(self):
        from src.dataset_loader import get_label_mapping_info
        classes = ['unripe', 'ripe', 'overripe', 'rotten']
        
        info_orig = get_label_mapping_info(classes, 'original')
        assert info_orig['mode'] == 'original'
        assert info_orig['output_classes'] == classes
        
        info_grouped = get_label_mapping_info(classes, 'grouped')
        assert info_grouped['mode'] == 'grouped'
        assert 'Mentah' in info_grouped['output_classes']


class TestPreprocessing:
    """Tes modul preprocessing."""
    
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        # Buat gambar tes
        self.img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        self.img_path = str(tmp_path / "test_img.jpg")
        cv2.imwrite(self.img_path, self.img)
    
    def test_read_image(self):
        from src.preprocessing import read_image
        img = read_image(self.img_path)
        assert img is not None
        assert img.shape[2] == 3
    
    def test_read_invalid_image(self):
        from src.preprocessing import read_image
        img = read_image("/nonexistent/image.jpg")
        assert img is None
    
    def test_resize_image(self):
        from src.preprocessing import resize_image
        resized = resize_image(self.img, (128, 128))
        assert resized.shape == (128, 128, 3)
    
    def test_convert_color_space(self):
        from src.preprocessing import convert_color_space
        rgb = convert_color_space(self.img, 'RGB')
        hsv = convert_color_space(self.img, 'HSV')
        lab = convert_color_space(self.img, 'CIELAB')
        assert rgb.shape == self.img.shape
        assert hsv.shape == self.img.shape
        assert lab.shape == self.img.shape
    
    def test_preprocess_image(self):
        from src.preprocessing import preprocess_image
        result = preprocess_image(self.img_path)
        assert result is not None
        assert 'img_rgb' in result
        assert 'img_hsv' in result
        assert 'img_lab' in result
        assert 'img_gray' in result
        assert 'mask' in result
    
    def test_preprocess_with_segmentation(self):
        from src.preprocessing import preprocess_image
        result = preprocess_image(self.img_path, use_segmentation=True)
        assert result is not None
        assert 'mask' in result


class TestFeatureExtraction:
    """Tes modul feature_extraction."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        self.img_rgb = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        self.img_hsv = cv2.cvtColor(
            cv2.cvtColor(self.img_rgb, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2HSV
        )
        self.img_lab = cv2.cvtColor(
            cv2.cvtColor(self.img_rgb, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2LAB
        )
        self.img_gray = cv2.cvtColor(self.img_rgb, cv2.COLOR_RGB2GRAY)
        self.mask = np.ones((128, 128), dtype=np.bool_)
    
    def test_rgb_features(self):
        from src.feature_extraction import extract_rgb_features
        features, names = extract_rgb_features(self.img_rgb, self.mask)
        assert len(features) == len(names)
        assert not np.any(np.isnan(features))
        assert not np.any(np.isinf(features))
    
    def test_hsv_features(self):
        from src.feature_extraction import extract_hsv_features
        features, names = extract_hsv_features(self.img_hsv, self.mask)
        assert len(features) == len(names)
        assert not np.any(np.isnan(features))
    
    def test_lab_features(self):
        from src.feature_extraction import extract_lab_features
        features, names = extract_lab_features(self.img_lab, self.mask)
        assert len(features) == len(names)
        assert not np.any(np.isnan(features))
    
    def test_glcm_features(self):
        from src.feature_extraction import extract_glcm_features
        features, names = extract_glcm_features(self.img_gray, self.mask)
        assert len(features) == len(names)
        assert not np.any(np.isnan(features))
        assert not np.any(np.isinf(features))
    
    def test_glcm_constant_image(self):
        """Tes GLCM pada gambar konstan (semua piksel sama)."""
        from src.feature_extraction import extract_glcm_features
        const_img = np.full((64, 64), 128, dtype=np.uint8)
        features, names = extract_glcm_features(const_img)
        assert not np.any(np.isnan(features))
        assert not np.any(np.isinf(features))
    
    def test_feature_dimension_consistency(self):
        from src.feature_extraction import get_feature_dimension, extract_all_features
        from src.preprocessing import preprocess_image
        
        for cs in ['RGB', 'HSV', 'CIELAB']:
            for use_glcm in [False, True]:
                expected = get_feature_dimension(cs, use_glcm)
                # Buat dummy preprocessed
                preprocessed = {
                    'img_rgb': self.img_rgb,
                    'img_hsv': self.img_hsv,
                    'img_lab': self.img_lab,
                    'img_gray': self.img_gray,
                    'mask': self.mask,
                }
                features, names = extract_all_features(preprocessed, cs, use_glcm)
                assert len(features) == expected, \
                    f"{cs} GLCM={use_glcm}: expected {expected}, got {len(features)}"
                assert len(names) == expected
    
    def test_no_nan_inf(self):
        """Pastikan tidak ada NaN atau Inf di semua kombinasi fitur."""
        from src.feature_extraction import extract_all_features
        
        preprocessed = {
            'img_rgb': self.img_rgb,
            'img_hsv': self.img_hsv,
            'img_lab': self.img_lab,
            'img_gray': self.img_gray,
            'mask': self.mask,
        }
        
        for cs in ['RGB', 'HSV', 'CIELAB']:
            for use_glcm in [False, True]:
                features, _ = extract_all_features(preprocessed, cs, use_glcm)
                assert not np.any(np.isnan(features)), f"NaN in {cs} GLCM={use_glcm}"
                assert not np.any(np.isinf(features)), f"Inf in {cs} GLCM={use_glcm}"


class TestModelTraining:
    """Tes modul model_training menggunakan dataset mini sintetis."""
    
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.dataset_dir = str(tmp_path / "test_dataset")
        create_synthetic_dataset(self.dataset_dir, n_per_class=5)
    
    def test_create_svm_pipeline(self):
        from src.model_training import create_svm_pipeline
        pipeline = create_svm_pipeline()
        assert 'scaler' in pipeline.named_steps
        assert 'svm' in pipeline.named_steps
    
    def test_train_experiment_smoke(self):
        """Smoke test: latih satu eksperimen dengan dataset mini."""
        from src.model_training import train_experiment
        from src.dataset_loader import load_split_data
        
        classes = ['unripe', 'ripe', 'overripe', 'rotten']
        paths, labels = load_split_data(
            self.dataset_dir, 'train', classes
        )
        
        result = train_experiment(
            experiment_id=1,
            train_paths=paths,
            train_labels=labels,
            dataset_path=self.dataset_dir,
        )
        
        assert result['status'] == 'success'
        assert result['n_train_samples'] > 0
        assert result['n_features'] > 0


class TestEvaluation:
    """Tes modul evaluasi."""
    
    def test_metrics_computation(self):
        from sklearn.metrics import accuracy_score, f1_score
        
        y_true = ['ripe', 'unripe', 'ripe', 'overripe', 'rotten']
        y_pred = ['ripe', 'unripe', 'overripe', 'overripe', 'rotten']
        
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        assert 0 <= acc <= 1
        assert 0 <= f1 <= 1
    
    def test_confusion_matrix_shape(self):
        from sklearn.metrics import confusion_matrix
        
        y_true = ['ripe', 'unripe', 'ripe', 'overripe']
        y_pred = ['ripe', 'unripe', 'overripe', 'overripe']
        classes = sorted(set(y_true) | set(y_pred))
        
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        assert cm.shape == (len(classes), len(classes))


class TestPrediction:
    """Tes modul prediction."""
    
    def test_predict_without_model(self, tmp_path):
        from src.prediction import predict_image
        
        # Buat gambar tes
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        img_path = str(tmp_path / "test.jpg")
        cv2.imwrite(img_path, img)
        
        result = predict_image(img_path, 99)  # Model 99 pasti tidak ada
        assert result is not None
        assert result.get('error', False) == True


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
