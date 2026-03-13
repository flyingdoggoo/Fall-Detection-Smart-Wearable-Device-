# -*- coding: utf-8 -*-
"""
Fall Detection – CNN + LSTM Pipeline (Google Colab)
=====================================================
Dự án PBL5 – ESP32-C3 + MPU6050 + MAX30102
Phiên bản: v3 – Hoàn chỉnh

HƯỚNG DẪN SỬ DỤNG
------------------
1. Upload thư mục `collected/` gồm `Fall/` và `Normal/` lên Google Drive
2. Mount Drive hoặc upload trực tiếp
3. Chạy từng section theo thứ tự

CẤU TRÚC DỮ LIỆU MỖI SESSION
-------------------------------
  session_folder/
    ├── accel.csv     (accel_time_list, accel_x_list, accel_y_list, accel_z_list)
    ├── gyro.csv      (gyro_time_list, gyro_x_list, gyro_y_list, gyro_z_list)
    ├── label.txt     ("0" hoặc "1")
    └── metadata.json (sample_rate, window_size, ...)
"""

# ================================================================
# SECTION 0: CÀI ĐẶT THƯ VIỆN
# ================================================================
# !pip install -q tensorflow scikit-learn matplotlib seaborn pandas numpy

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, regularizers
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, f1_score
)
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings('ignore')
print(f"TensorFlow: {tf.__version__}")
print(f"GPU: {tf.config.list_physical_devices('GPU')}")

# ================================================================
# SECTION 1: CẤU HÌNH
# ================================================================
# Đường dẫn dữ liệu – THAY ĐỔI PHÙ HỢP
# Nếu dùng Google Drive:
#   from google.colab import drive
#   drive.mount('/content/drive')
#   DATA_ROOT = Path('/content/drive/MyDrive/PBL5/data/collected')
# Nếu upload trực tiếp:
#   DATA_ROOT = Path('/content/collected')

DATA_ROOT = Path('/content/collected')  # <-- THAY ĐỔI NẾU CẦN

SAMPLE_RATE     = 50       # Hz
WINDOW_SIZE     = 100      # samples = 2 giây
OVERLAP_RATIO   = 0.5      # 50% overlap khi sliding window
NUM_CHANNELS    = 6        # ax, ay, az, gx, gy, gz
TEST_SIZE       = 0.2
VAL_SIZE        = 0.15     # Từ train set
RANDOM_SEED     = 42
EPOCHS          = 100
BATCH_SIZE      = 32

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# ================================================================
# SECTION 2: ĐỌC VÀ TIỀN XỬ LÝ DỮ LIỆU
# ================================================================
def load_session(session_dir):
    """Đọc 1 session → DataFrame 6 cột (ax,ay,az,gx,gy,gz) + label."""
    session_dir = Path(session_dir)
    accel_path  = session_dir / 'accel.csv'
    gyro_path   = session_dir / 'gyro.csv'
    label_path  = session_dir / 'label.txt'

    if not accel_path.exists() or not gyro_path.exists() or not label_path.exists():
        return None, None

    try:
        accel = pd.read_csv(accel_path)
        gyro  = pd.read_csv(gyro_path)
        label = int(label_path.read_text().strip())
    except Exception as e:
        print(f"  ⚠ Error in {session_dir.name}: {e}")
        return None, None

    if accel.empty or gyro.empty:
        return None, None

    # Chuẩn hóa tên cột
    accel.columns = ['time', 'ax', 'ay', 'az']
    gyro.columns  = ['time', 'gx', 'gy', 'gz']

    # Merge theo index (cùng số dòng)
    min_len = min(len(accel), len(gyro))
    df = pd.concat([
        accel[['ax', 'ay', 'az']].iloc[:min_len].reset_index(drop=True),
        gyro[['gx', 'gy', 'gz']].iloc[:min_len].reset_index(drop=True)
    ], axis=1)

    # Loại bỏ NaN / Inf
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    if len(df) < WINDOW_SIZE // 2:
        return None, None

    return df, label


def sliding_window(data, window_size, overlap_ratio):
    """Chia data thành các windows có overlap."""
    step = int(window_size * (1 - overlap_ratio))
    step = max(step, 1)
    windows = []
    for start in range(0, len(data) - window_size + 1, step):
        windows.append(data[start:start + window_size])
    return windows


def load_all_data(data_root):
    """Đọc toàn bộ sessions → X (windows), y (labels)."""
    X_windows = []
    y_labels  = []
    session_stats = {'Normal': 0, 'Fall': 0, 'errors': 0}

    for class_name in ['Normal', 'Fall']:
        class_dir = data_root / class_name
        if not class_dir.exists():
            print(f"⚠ Thư mục {class_dir} không tồn tại!")
            continue

        label = 0 if class_name == 'Normal' else 1
        sessions = sorted([d for d in class_dir.iterdir() if d.is_dir()])
        print(f"\n📂 {class_name}: {len(sessions)} sessions")

        for sess in sessions:
            df, lbl = load_session(sess)
            if df is None:
                session_stats['errors'] += 1
                continue

            # Override label from folder structure (ưu tiên thư mục cha)
            lbl = label
            session_stats[class_name] += 1

            # Sliding window
            data_array = df[['ax', 'ay', 'az', 'gx', 'gy', 'gz']].values
            windows = sliding_window(data_array, WINDOW_SIZE, OVERLAP_RATIO)

            for w in windows:
                X_windows.append(w)
                y_labels.append(lbl)

    X = np.array(X_windows, dtype=np.float32)
    y = np.array(y_labels, dtype=np.int32)

    print(f"\n{'='*50}")
    print(f"✅ Loaded: {session_stats}")
    print(f"   Windows: {len(X)} | Shape: {X.shape}")
    print(f"   Normal: {np.sum(y==0)} | Fall: {np.sum(y==1)}")
    print(f"   Ratio Normal:Fall = {np.sum(y==0)}:{np.sum(y==1)}")

    return X, y


print("Loading data...")
X_raw, y_raw = load_all_data(DATA_ROOT)

# ================================================================
# SECTION 3: EXPLORATORY DATA ANALYSIS  (Visualization)
# ================================================================
def plot_class_distribution(y, title='Class Distribution'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Bar chart
    classes, counts = np.unique(y, return_counts=True)
    colors = ['#10b981', '#ef4444']
    bars = ax1.bar(['Normal (0)', 'Fall (1)'], counts, color=colors, edgecolor='black')
    for bar, c in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                 str(c), ha='center', va='bottom', fontweight='bold', fontsize=14)
    ax1.set_ylabel('Number of Windows')
    ax1.set_title(title)

    # Pie chart
    ax2.pie(counts, labels=['Normal', 'Fall'], autopct='%1.1f%%',
            colors=colors, startangle=90, textprops={'fontsize': 12})
    ax2.set_title('Proportion')

    plt.tight_layout()
    plt.show()


def plot_sample_windows(X, y, n_samples=3):
    """Hiển thị vài cửa sổ mẫu cho mỗi class."""
    fig, axes = plt.subplots(2, n_samples, figsize=(18, 8))
    channel_names = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
    colors = ['#ef4444', '#10b981', '#3b82f6', '#f97316', '#8b5cf6', '#ec4899']

    for cls in [0, 1]:
        indices = np.where(y == cls)[0]
        chosen  = np.random.choice(indices, n_samples, replace=False)
        for j, idx in enumerate(chosen):
            ax = axes[cls, j]
            for ch in range(6):
                ax.plot(X[idx, :, ch], label=channel_names[ch],
                        color=colors[ch], alpha=0.8, linewidth=0.8)
            title = 'Normal' if cls == 0 else 'Fall'
            ax.set_title(f'{title} – sample {j+1}', fontsize=11)
            ax.set_xlabel('Sample (50Hz)')
            if j == 0:
                ax.set_ylabel('Value')
                ax.legend(fontsize=7, loc='upper right')

    plt.suptitle('Sample Windows', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_feature_distributions(X, y):
    """Phân bố magnitude cho 2 class."""
    mag_normal = np.sqrt(np.sum(X[y==0, :, :3]**2, axis=2)).flatten()
    mag_fall   = np.sqrt(np.sum(X[y==1, :, :3]**2, axis=2)).flatten()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

    ax1.hist(mag_normal, bins=100, alpha=0.7, color='#10b981', label='Normal', density=True)
    ax1.hist(mag_fall,   bins=100, alpha=0.7, color='#ef4444', label='Fall',   density=True)
    ax1.set_title('Accel Magnitude Distribution')
    ax1.set_xlabel('Magnitude (m/s²)')
    ax1.legend()

    # Gyro magnitude
    gmag_normal = np.sqrt(np.sum(X[y==0, :, 3:]**2, axis=2)).flatten()
    gmag_fall   = np.sqrt(np.sum(X[y==1, :, 3:]**2, axis=2)).flatten()

    ax2.hist(gmag_normal, bins=100, alpha=0.7, color='#10b981', label='Normal', density=True)
    ax2.hist(gmag_fall,   bins=100, alpha=0.7, color='#ef4444', label='Fall',   density=True)
    ax2.set_title('Gyro Magnitude Distribution')
    ax2.set_xlabel('Magnitude (rad/s)')
    ax2.legend()

    plt.tight_layout()
    plt.show()


# Hiển thị EDA
plot_class_distribution(y_raw)
plot_sample_windows(X_raw, y_raw)
plot_feature_distributions(X_raw, y_raw)

# ================================================================
# SECTION 4: CHIA DATASET  (Stratified)
# ================================================================
# Bước 1: Train+Val / Test  (80/20)
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X_raw, y_raw, test_size=TEST_SIZE,
    random_state=RANDOM_SEED, stratify=y_raw
)

# Bước 2: Train / Val  (85/15 của trainval)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=VAL_SIZE,
    random_state=RANDOM_SEED, stratify=y_trainval
)

print(f"Train: {X_train.shape[0]} windows")
print(f"Val:   {X_val.shape[0]} windows")
print(f"Test:  {X_test.shape[0]} windows")
print(f"\nTrain distribution – Normal:{np.sum(y_train==0)} Fall:{np.sum(y_train==1)}")
print(f"Val   distribution – Normal:{np.sum(y_val==0)}   Fall:{np.sum(y_val==1)}")
print(f"Test  distribution – Normal:{np.sum(y_test==0)}  Fall:{np.sum(y_test==1)}")

# ================================================================
# SECTION 5: CHUẨN HÓA (StandardScaler per channel)
# ================================================================
# Reshape → (N*T, C) → fit scaler on TRAIN only → transform all
n_train, n_timesteps, n_channels = X_train.shape

scaler = StandardScaler()
X_train_flat = X_train.reshape(-1, n_channels)
scaler.fit(X_train_flat)

def scale_data(X, scaler):
    n, t, c = X.shape
    return scaler.transform(X.reshape(-1, c)).reshape(n, t, c)

X_train_s = scale_data(X_train, scaler)
X_val_s   = scale_data(X_val,   scaler)
X_test_s  = scale_data(X_test,  scaler)

print(f"\nAfter scaling – Train mean: {X_train_s.mean():.4f}, std: {X_train_s.std():.4f}")

# ================================================================
# SECTION 6: CLASS WEIGHT  (Giảm mất cân bằng)
# ================================================================
class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
print(f"Class weights: {class_weight_dict}")

# ================================================================
# SECTION 7: XÂY DỰNG MÔ HÌNH CNN + LSTM
# ================================================================
def build_cnn_lstm_model(input_shape, dropout_rate=0.4, l2_reg=1e-4):
    """
    Kiến trúc:
      Conv1D (64) → BN → Conv1D (128) → BN → MaxPool
      → Conv1D (64) → BN → MaxPool
      → LSTM (64, return_sequences) → LSTM (32)
      → Dense (32) → Dropout → Dense (1, sigmoid)
    """
    reg = regularizers.l2(l2_reg)
    inputs = keras.Input(shape=input_shape, name='sensor_input')

    # CNN Block 1
    x = layers.Conv1D(64, 5, padding='same', activation='relu', kernel_regularizer=reg)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(128, 5, padding='same', activation='relu', kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(dropout_rate * 0.5)(x)

    # CNN Block 2
    x = layers.Conv1D(64, 3, padding='same', activation='relu', kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(dropout_rate * 0.5)(x)

    # LSTM Block
    x = layers.LSTM(64, return_sequences=True, dropout=dropout_rate * 0.5,
                    recurrent_dropout=0, kernel_regularizer=reg)(x)
    x = layers.LSTM(32, return_sequences=False, dropout=dropout_rate * 0.5,
                    recurrent_dropout=0, kernel_regularizer=reg)(x)

    # Dense Head
    x = layers.Dense(32, activation='relu', kernel_regularizer=reg)(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(1, activation='sigmoid', name='output')(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name='FallDetection_CNN_LSTM')
    return model


model = build_cnn_lstm_model(
    input_shape=(WINDOW_SIZE, NUM_CHANNELS),
    dropout_rate=0.4,
    l2_reg=1e-4
)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy',
             keras.metrics.Precision(name='precision'),
             keras.metrics.Recall(name='recall'),
             keras.metrics.AUC(name='auc')]
)

model.summary()

# ================================================================
# SECTION 8: CALLBACKS (Giảm overfitting)
# ================================================================
cb_list = [
    # Giảm learning rate khi val_loss không giảm
    callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=8,
        min_lr=1e-6, verbose=1
    ),
    # Dừng sớm nếu val_loss không cải thiện
    callbacks.EarlyStopping(
        monitor='val_loss', patience=15,
        restore_best_weights=True, verbose=1
    ),
    # Lưu model tốt nhất
    callbacks.ModelCheckpoint(
        'best_model.keras', monitor='val_auc',
        mode='max', save_best_only=True, verbose=1
    )
]

# ================================================================
# SECTION 9: TRAINING
# ================================================================
print("\n" + "="*60)
print("  TRAINING CNN + LSTM MODEL")
print("="*60)

history = model.fit(
    X_train_s, y_train,
    validation_data=(X_val_s, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weight_dict,
    callbacks=cb_list,
    verbose=1
)

# ================================================================
# SECTION 10: VISUALIZATION KẾT QUẢ TRAINING
# ================================================================
def plot_training_history(history):
    metrics = ['loss', 'accuracy', 'precision', 'recall', 'auc']
    available = [m for m in metrics if m in history.history]

    n = len(available)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 4))
    if n == 1: axes = [axes]

    for ax, metric in zip(axes, available):
        ax.plot(history.history[metric], label=f'Train {metric}', linewidth=1.5)
        val_key = f'val_{metric}'
        if val_key in history.history:
            ax.plot(history.history[val_key], label=f'Val {metric}',
                    linewidth=1.5, linestyle='--')
        ax.set_title(metric.upper(), fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle('Training History', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # Learning rate nếu ReduceLROnPlateau hoạt động
    if 'lr' in history.history:
        plt.figure(figsize=(8, 3))
        plt.plot(history.history['lr'], color='orange')
        plt.title('Learning Rate Schedule')
        plt.xlabel('Epoch')
        plt.ylabel('LR')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.show()


plot_training_history(history)

# ================================================================
# SECTION 11: EVALUATION TRÊN TEST SET
# ================================================================
print("\n" + "="*60)
print("  EVALUATION ON TEST SET")
print("="*60)

# Load best model
best_model = keras.models.load_model('best_model.keras')

# Predict
y_pred_proba = best_model.predict(X_test_s, verbose=0).flatten()
y_pred       = (y_pred_proba >= 0.5).astype(int)

# Classification report
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Fall'], digits=4))

# Test metrics
test_loss, test_acc, test_prec, test_rec, test_auc = best_model.evaluate(X_test_s, y_test, verbose=0)
print(f"\nTest Loss:      {test_loss:.4f}")
print(f"Test Accuracy:  {test_acc:.4f}")
print(f"Test Precision: {test_prec:.4f}")
print(f"Test Recall:    {test_rec:.4f}")
print(f"Test AUC:       {test_auc:.4f}")
print(f"Test F1-Score:  {f1_score(y_test, y_pred):.4f}")

# ================================================================
# SECTION 12: CONFUSION MATRIX
# ================================================================
def plot_confusion_matrix(y_true, y_pred, classes=['Normal', 'Fall']):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=classes, yticklabels=classes)
    ax1.set_title('Confusion Matrix (Counts)', fontweight='bold')
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')

    # Normalized
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Greens', ax=ax2,
                xticklabels=classes, yticklabels=classes)
    ax2.set_title('Confusion Matrix (Normalized)', fontweight='bold')
    ax2.set_ylabel('True Label')
    ax2.set_xlabel('Predicted Label')

    plt.tight_layout()
    plt.show()

    # In số liệu
    tn, fp, fn, tp = cm.ravel()
    print(f"TP={tp}  TN={tn}  FP={fp}  FN={fn}")
    print(f"Sensitivity (Recall): {tp/(tp+fn):.4f}")
    print(f"Specificity:          {tn/(tn+fp):.4f}")


plot_confusion_matrix(y_test, y_pred)

# ================================================================
# SECTION 13: ROC CURVE & PRECISION-RECALL CURVE
# ================================================================
def plot_roc_and_pr(y_true, y_proba):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    ax1.plot(fpr, tpr, color='#667eea', linewidth=2, label=f'ROC (AUC={roc_auc:.4f})')
    ax1.plot([0,1], [0,1], 'k--', alpha=0.3)
    ax1.fill_between(fpr, tpr, alpha=0.1, color='#667eea')
    ax1.set(title='ROC Curve', xlabel='False Positive Rate', ylabel='True Positive Rate')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)

    # Precision-Recall
    precision_arr, recall_arr, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall_arr, precision_arr)
    ax2.plot(recall_arr, precision_arr, color='#ef4444', linewidth=2,
             label=f'PR (AUC={pr_auc:.4f})')
    ax2.fill_between(recall_arr, precision_arr, alpha=0.1, color='#ef4444')
    ax2.set(title='Precision-Recall Curve', xlabel='Recall', ylabel='Precision')
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


plot_roc_and_pr(y_test, y_pred_proba)

# ================================================================
# SECTION 14: RANDOM PREDICTION TEST
# ================================================================
def random_prediction_test(model, X_test, y_test, scaler_obj, n=10):
    """Chọn ngẫu nhiên n mẫu, hiển thị dự đoán vs ground truth."""
    indices = np.random.choice(len(X_test), n, replace=False)
    X_sample = X_test[indices]

    # Scale
    X_scaled = scale_data(X_sample, scaler_obj)
    preds = model.predict(X_scaled, verbose=0).flatten()

    print(f"\n{'='*65}")
    print(f"  RANDOM PREDICTION TEST ({n} samples)")
    print(f"{'='*65}")
    print(f"{'#':<4} {'True':<10} {'Pred Prob':<12} {'Pred Label':<12} {'Match':<8}")
    print(f"{'-'*65}")

    correct = 0
    for i, idx in enumerate(indices):
        true_label = 'Fall' if y_test[idx] == 1 else 'Normal'
        pred_label = 'Fall' if preds[i] >= 0.5 else 'Normal'
        match = '✓' if true_label == pred_label else '✗'
        if true_label == pred_label: correct += 1
        print(f"{i+1:<4} {true_label:<10} {preds[i]:<12.4f} {pred_label:<12} {match}")

    print(f"\nAccuracy on random sample: {correct}/{n} = {correct/n*100:.1f}%")

    # Visualize 4 samples
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    ch_names = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
    ch_colors = ['#ef4444', '#10b981', '#3b82f6', '#f97316', '#8b5cf6', '#ec4899']

    for i, ax in enumerate(axes.flat):
        if i >= min(4, n): break
        idx = indices[i]
        for ch in range(6):
            ax.plot(X_test[idx, :, ch], label=ch_names[ch],
                    color=ch_colors[ch], alpha=0.8, linewidth=0.8)
        true_l = 'Fall' if y_test[idx] == 1 else 'Normal'
        pred_l = 'Fall' if preds[i] >= 0.5 else 'Normal'
        color  = '#10b981' if true_l == pred_l else '#ef4444'
        ax.set_title(f'True: {true_l} | Pred: {pred_l} ({preds[i]:.2f})',
                     color=color, fontweight='bold')
        ax.legend(fontsize=7, loc='upper right')
        ax.set_xlabel('Sample')

    plt.suptitle('Random Prediction Visualization', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


random_prediction_test(best_model, X_test, y_test, scaler)

# ================================================================
# SECTION 15: LƯU MODEL & ARTIFACTS
# ================================================================
import pickle

# Lưu model đầy đủ (Keras)
best_model.save('fall_detection_cnn_lstm.keras')
print("✅ Model saved: fall_detection_cnn_lstm.keras")

# Lưu scaler để dùng khi inference
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✅ Scaler saved: scaler.pkl")

# Lưu config
config = {
    'sample_rate': SAMPLE_RATE,
    'window_size': WINDOW_SIZE,
    'num_channels': NUM_CHANNELS,
    'channel_names': ['ax', 'ay', 'az', 'gx', 'gy', 'gz'],
    'overlap_ratio': OVERLAP_RATIO,
    'threshold': 0.5,
    'model_file': 'fall_detection_cnn_lstm.keras',
    'scaler_file': 'scaler.pkl',
    'test_accuracy': float(test_acc),
    'test_f1': float(f1_score(y_test, y_pred)),
    'test_auc': float(test_auc),
    'train_samples': int(len(X_train)),
    'val_samples': int(len(X_val)),
    'test_samples': int(len(X_test)),
}
with open('model_config.json', 'w') as f:
    json.dump(config, f, indent=2)
print("✅ Config saved: model_config.json")

# Lưu TFLite (optional – deploy ESP32 nâng cao hoặc mobile)
converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
with open('fall_detection.tflite', 'wb') as f:
    f.write(tflite_model)
print(f"✅ TFLite saved: fall_detection.tflite ({len(tflite_model)/1024:.1f} KB)")

# ================================================================
# SECTION 16: HƯỚNG DẪN TIẾP TỤC TRAINING
# ================================================================
print("""
╔══════════════════════════════════════════════════════════════╗
║             CÁCH TRAIN TIẾP (FINE-TUNE)                    ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  # Load model đã lưu:                                       ║
║  model = keras.models.load_model('fall_detection_cnn_lstm.keras')
║                                                              ║
║  # Load scaler:                                              ║
║  with open('scaler.pkl', 'rb') as f:                         ║
║      scaler = pickle.load(f)                                 ║
║                                                              ║
║  # Giảm learning rate để fine-tune:                          ║
║  model.compile(                                              ║
║      optimizer=keras.optimizers.Adam(lr=1e-4),               ║
║      loss='binary_crossentropy',                             ║
║      metrics=['accuracy']                                    ║
║  )                                                           ║
║                                                              ║
║  # Train thêm với data mới:                                  ║
║  model.fit(X_new_scaled, y_new, epochs=30, ...)              ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")

# ================================================================
# SECTION 17: TÓM TẮT KẾT QUẢ
# ================================================================
print(f"\n{'='*60}")
print(f"  📊 FINAL RESULTS SUMMARY")
print(f"{'='*60}")
print(f"  Model:      CNN + LSTM (Fall Detection)")
print(f"  Input:      {WINDOW_SIZE} timesteps × {NUM_CHANNELS} channels (2s @ 50Hz)")
print(f"  Dataset:    Train {len(X_train)} | Val {len(X_val)} | Test {len(X_test)}")
print(f"  Accuracy:   {test_acc:.4f}")
print(f"  Precision:  {test_prec:.4f}")
print(f"  Recall:     {test_rec:.4f}")
print(f"  F1-Score:   {f1_score(y_test, y_pred):.4f}")
print(f"  AUC:        {test_auc:.4f}")
print(f"{'='*60}")
