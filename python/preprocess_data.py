"""
DATA PREPROCESSING FOR FALL DETECTION MODEL
============================================
Script này sẽ:
1. Merge WEDA-FALL dataset với data thu thập
2. Chuẩn hóa format (resample, normalize, feature extraction)
3. Tạo training dataset cho ML/DL model
4. Export sang format sẵn sàng train

Requirements:
    pip install pandas numpy scikit-learn scipy

Author: Fall Detection Team
Date: 2026-01-26
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import signal, interpolate
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import json
import warnings
warnings.filterwarnings('ignore')


class FallDataPreprocessor:
    """
    Class để preprocess data cho fall detection model
    """
    
    def __init__(self, target_sample_rate=50, window_size=2.0, overlap=0.5):
        """
        Args:
            target_sample_rate: Target sampling rate (Hz)
            window_size: Sliding window size (seconds)
            overlap: Window overlap ratio (0-1)
        """
        self.target_rate = target_sample_rate
        self.window_size = window_size
        self.overlap = overlap
        self.window_samples = int(target_sample_rate * window_size)
        self.hop_samples = int(self.window_samples * (1 - overlap))
        
        self.scaler = StandardScaler()
        
    def resample_data(self, df, time_col, data_cols, target_rate=None):
        """
        Resample data về target sampling rate (xử lý timestamps không đều)
        
        Args:
            df: DataFrame chứa data
            time_col: Tên cột timestamp
            data_cols: List tên cột data cần resample
            target_rate: Target rate (Hz), mặc định dùng self.target_rate
        """
        if target_rate is None:
            target_rate = self.target_rate
            
        # Tạo timeline đều
        start_time = df[time_col].iloc[0]
        end_time = df[time_col].iloc[-1]
        duration = end_time - start_time
        n_samples = int(duration * target_rate)
        
        new_time = np.linspace(start_time, end_time, n_samples)
        
        # Interpolate từng cột data
        resampled_data = {time_col: new_time}
        
        for col in data_cols:
            f = interpolate.interp1d(df[time_col], df[col], kind='linear', fill_value='extrapolate')
            resampled_data[col] = f(new_time)
        
        return pd.DataFrame(resampled_data)
    
    def extract_features(self, accel_df, gyro_df):
        """
        Extract features từ raw data
        """
        features = {}
        
        # Time domain features
        for axis in ['x', 'y', 'z']:
            accel_col = f'accel_{axis}_list'
            gyro_col = f'gyro_{axis}_list'
            
            # Accelerometer features
            features[f'accel_{axis}_mean'] = accel_df[accel_col].mean()
            features[f'accel_{axis}_std'] = accel_df[accel_col].std()
            features[f'accel_{axis}_max'] = accel_df[accel_col].max()
            features[f'accel_{axis}_min'] = accel_df[accel_col].min()
            features[f'accel_{axis}_range'] = features[f'accel_{axis}_max'] - features[f'accel_{axis}_min']
            
            # Gyroscope features
            features[f'gyro_{axis}_mean'] = gyro_df[gyro_col].mean()
            features[f'gyro_{axis}_std'] = gyro_df[gyro_col].std()
            features[f'gyro_{axis}_max'] = gyro_df[gyro_col].max()
            features[f'gyro_{axis}_min'] = gyro_df[gyro_col].min()
        
        # Magnitude features
        accel_mag = np.sqrt(
            accel_df['accel_x_list']**2 + 
            accel_df['accel_y_list']**2 + 
            accel_df['accel_z_list']**2
        )
        features['accel_magnitude_mean'] = accel_mag.mean()
        features['accel_magnitude_std'] = accel_mag.std()
        features['accel_magnitude_max'] = accel_mag.max()
        features['accel_magnitude_min'] = accel_mag.min()
        
        gyro_mag = np.sqrt(
            gyro_df['gyro_x_list']**2 + 
            gyro_df['gyro_y_list']**2 + 
            gyro_df['gyro_z_list']**2
        )
        features['gyro_magnitude_mean'] = gyro_mag.mean()
        features['gyro_magnitude_std'] = gyro_mag.std()
        features['gyro_magnitude_max'] = gyro_mag.max()
        
        # SMA (Signal Magnitude Area)
        features['sma'] = (
            np.abs(accel_df['accel_x_list']).sum() + 
            np.abs(accel_df['accel_y_list']).sum() + 
            np.abs(accel_df['accel_z_list']).sum()
        ) / (3.0 * len(accel_df))
        
        # Jerk features (Đạo hàm gia tốc)
        jerk_x = np.diff(accel_df['accel_x_list'])
        jerk_y = np.diff(accel_df['accel_y_list'])
        jerk_z = np.diff(accel_df['accel_z_list'])
        
        features['jerk_mean'] = (np.abs(jerk_x).mean() + np.abs(jerk_y).mean() + np.abs(jerk_z).mean()) / 3.0
        features['jerk_std'] = (jerk_x.std() + jerk_y.std() + jerk_z.std()) / 3.0
        
        # Frequency domain features (FFT)
        fft_accel_x = np.fft.fft(accel_df['accel_x_list'])
        fft_accel_y = np.fft.fft(accel_df['accel_y_list'])
        fft_accel_z = np.fft.fft(accel_df['accel_z_list'])
        
        features['fft_accel_x_energy'] = np.sum(np.abs(fft_accel_x)**2)
        features['fft_accel_y_energy'] = np.sum(np.abs(fft_accel_y)**2)
        features['fft_accel_z_energy'] = np.sum(np.abs(fft_accel_z)**2)
        
        return features
    
    def sliding_window_split(self, accel_df, gyro_df, label, fall_windows=None):
        """
        Chia data thành sliding windows
        
        Args:
            fall_windows: List of (start_time, end_time) tuples marking fall events
        
        Returns:
            List of (features_dict, label) tuples
        """
        windows = []
        
        # Đảm bảo accel và gyro cùng length
        min_len = min(len(accel_df), len(gyro_df))
        accel_df = accel_df.iloc[:min_len].reset_index(drop=True)
        gyro_df = gyro_df.iloc[:min_len].reset_index(drop=True)
        
        # Sliding window
        for start_idx in range(0, len(accel_df) - self.window_samples, self.hop_samples):
            end_idx = start_idx + self.window_samples
            
            if end_idx > len(accel_df):
                break
            
            window_accel = accel_df.iloc[start_idx:end_idx]
            window_gyro = gyro_df.iloc[start_idx:end_idx]
            
            # Label theo timestamp nếu có fall_windows
            window_label = label  # Default
            if fall_windows:
                # Lấy thời gian giữa window
                mid_time = (window_accel['accel_time_list'].iloc[0] + 
                           window_accel['accel_time_list'].iloc[-1]) / 2
                
                # Check nếu nằm trong fall window
                for fall_start, fall_end in fall_windows:
                    if fall_start <= mid_time <= fall_end:
                        window_label = 1  # Fall
                        break
                else:
                    window_label = 0  # ADL (ngoài fall windows)
            
            features = self.extract_features(window_accel, window_gyro)
            windows.append((features, window_label))
        
        return windows
    
    def process_weda_fall(self, weda_path, output_path):
        """
        Process toàn bộ WEDA-FALL dataset
        
        Args:
            weda_path: Path to WEDA-FALL dataset (50Hz folder)
            output_path: Output folder
        """
        weda_path = Path(weda_path)
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        all_windows = []
        
        print("\n" + "="*80)
        print("📦 PROCESSING WEDA-FALL DATASET")
        print("="*80)
        
        # Fall activities (F01-F08)
        fall_activities = [f'F{i:02d}' for i in range(1, 9)]
        
        # ADL activities (D01-D11)
        adl_activities = [f'D{i:02d}' for i in range(1, 12)]
        
        # Users
        young_users = [f'U{i:02d}' for i in range(1, 15)]
        elder_users = [f'U{i:02d}' for i in range(21, 32)]
        all_users = young_users  # Elder không có fall data
        
        # Process Falls
        for activity in fall_activities:
            activity_path = weda_path / activity
            if not activity_path.exists():
                continue
                
            print(f"\n🔴 Processing {activity} (Fall)...")
            
            for user in all_users:
                for run in ['R01', 'R02', 'R03']:
                    accel_file = activity_path / f"{user}_{run}_accel.csv"
                    gyro_file = activity_path / f"{user}_{run}_gyro.csv"
                    
                    if not accel_file.exists() or not gyro_file.exists():
                        continue
                    
                    try:
                        accel_df = pd.read_csv(accel_file)
                        gyro_df = pd.read_csv(gyro_file)
                        
                        # Resample nếu cần (timestamps không đều)
                        accel_df = self.resample_data(
                            accel_df, 
                            'accel_time_list',
                            ['accel_x_list', 'accel_y_list', 'accel_z_list']
                        )
                        gyro_df = self.resample_data(
                            gyro_df,
                            'gyro_time_list',
                            ['gyro_x_list', 'gyro_y_list', 'gyro_z_list']
                        )
                        
                        # Extract windows
                        windows = self.sliding_window_split(accel_df, gyro_df, label=1)  # Fall = 1
                        all_windows.extend(windows)
                        
                        print(f"  ✓ {user}_{run}: {len(windows)} windows")
                        
                    except Exception as e:
                        print(f"  ✗ {user}_{run}: {e}")
        
        # Process ADLs
        for activity in adl_activities:
            activity_path = weda_path / activity
            if not activity_path.exists():
                continue
                
            print(f"\n🟢 Processing {activity} (ADL)...")
            
            # ADL có cả Young và Elder
            for user in young_users + elder_users:
                for run in ['R01', 'R02', 'R03']:
                    accel_file = activity_path / f"{user}_{run}_accel.csv"
                    gyro_file = activity_path / f"{user}_{run}_gyro.csv"
                    
                    if not accel_file.exists() or not gyro_file.exists():
                        continue
                    
                    try:
                        accel_df = pd.read_csv(accel_file)
                        gyro_df = pd.read_csv(gyro_file)
                        
                        accel_df = self.resample_data(
                            accel_df,
                            'accel_time_list',
                            ['accel_x_list', 'accel_y_list', 'accel_z_list']
                        )
                        gyro_df = self.resample_data(
                            gyro_df,
                            'gyro_time_list',
                            ['gyro_x_list', 'gyro_y_list', 'gyro_z_list']
                        )
                        
                        windows = self.sliding_window_split(accel_df, gyro_df, label=0)  # ADL = 0
                        all_windows.extend(windows)
                        
                        print(f"  ✓ {user}_{run}: {len(windows)} windows")
                        
                    except Exception as e:
                        print(f"  ✗ {user}_{run}: {e}")
        
        print(f"\n{'='*80}")
        print(f"✅ Total windows extracted: {len(all_windows)}")
        
        # Convert to DataFrame
        features_list = []
        labels = []
        
        for features, label in all_windows:
            features_list.append(features)
            labels.append(label)
        
        features_df = pd.DataFrame(features_list)
        features_df['label'] = labels
        
        # Save
        output_file = output_path / 'weda_fall_processed.csv'
        features_df.to_csv(output_file, index=False)
        print(f"💾 Saved: {output_file}")
        
        # Statistics
        fall_count = (features_df['label'] == 1).sum()
        adl_count = (features_df['label'] == 0).sum()
        print(f"\n📊 Dataset Statistics:")
        print(f"   Fall windows: {fall_count}")
        print(f"   ADL windows: {adl_count}")
        print(f"   Total: {len(features_df)}")
        print(f"   Balance ratio: {fall_count / adl_count:.2f}")
        
        return features_df
    
    def process_collected_data(self, collected_path, label=None, output_path=None):
        """
        Process data thu thập từ ESP32 - Support timestamp-based labeling
        
        Args:
            collected_path: Path to collected session folder
            label: Label cho data này (0=ADL, 1=Fall). Nếu None, tự động detect
            output_path: Output folder
        """
        if output_path is None:
            output_path = Path(__file__).parent / "processed_data"
        collected_path = Path(collected_path)
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "="*80)
        print(f"📦 PROCESSING COLLECTED DATA: {collected_path.name}")
        print("="*80)
        
        # Load data
        accel_file = collected_path / 'accel.csv'
        gyro_file = collected_path / 'gyro.csv'
        
        accel_df = pd.read_csv(accel_file)
        gyro_df = pd.read_csv(gyro_file)
        
        print(f"✓ Loaded: {len(accel_df)} accel samples, {len(gyro_df)} gyro samples")
        
        # Resample (timestamps có thể không đều)
        accel_df = self.resample_data(
            accel_df,
            'accel_time_list',
            ['accel_x_list', 'accel_y_list', 'accel_z_list']
        )
        gyro_df = self.resample_data(
            gyro_df,
            'gyro_time_list',
            ['gyro_x_list', 'gyro_y_list', 'gyro_z_list']
        )
        
        # Load label từ label.txt
        label_file = collected_path / 'label.txt'
        if label_file.exists():
            with open(label_file, 'r') as f:
                label = int(f.read().strip())
            print(f"✓ Label from file: {label} ({'FALL' if label == 1 else 'NORMAL'})")
        else:
            # Fallback: detect từ folder name
            folder_name = collected_path.name
            if folder_name.startswith('label1_'):
                label = 1
            elif folder_name.startswith('label0_'):
                label = 0
            else:
                label = 0
            print(f"⚠️  No label.txt, using folder name: {label}")
        
        # Load fall markers (bỏ qua, không dùng nữa)
        fall_windows = None
        
        # Extract windows (toàn bộ session cùng label)
        windows = self.sliding_window_split(accel_df, gyro_df, label=label, fall_windows=None)
        print(f"✓ Extracted: {len(windows)} windows (all labeled as {label})")
        
        # Convert to DataFrame
        features_list = []
        labels = []
        
        for features, lbl in windows:
            features_list.append(features)
            labels.append(lbl)
        
        features_df = pd.DataFrame(features_list)
        features_df['label'] = labels
        
        # Save
        output_file = output_path / f'{collected_path.name}_processed.csv'
        features_df.to_csv(output_file, index=False)
        print(f"💾 Saved: {output_file}")
        
        return features_df
    
    def merge_datasets(self, weda_df, collected_dfs, output_path):
        """
        Merge WEDA-FALL với collected data
        
        Args:
            weda_df: WEDA-FALL processed DataFrame
            collected_dfs: List of collected DataFrames
            output_path: Output folder
        """
        print("\n" + "="*80)
        print("🔗 MERGING DATASETS")
        print("="*80)
        
        # Merge all
        merged_df = pd.concat([weda_df] + collected_dfs, ignore_index=True)
        
        # Shuffle
        merged_df = merged_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Save
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        output_file = output_path / 'merged_dataset.csv'
        merged_df.to_csv(output_file, index=False)
        print(f"💾 Saved merged dataset: {output_file}")
        
        # Statistics
        fall_count = (merged_df['label'] == 1).sum()
        adl_count = (merged_df['label'] == 0).sum()
        print(f"\n📊 Merged Dataset Statistics:")
        print(f"   Fall windows: {fall_count}")
        print(f"   ADL windows: {adl_count}")
        print(f"   Total: {len(merged_df)}")
        print(f"   Features: {len(merged_df.columns) - 1}")  # -1 for label
        
        # Save feature names
        feature_names = [col for col in merged_df.columns if col != 'label']
        with open(output_path / 'feature_names.json', 'w') as f:
            json.dump(feature_names, f, indent=2)
        print(f"💾 Saved feature names: feature_names.json")
        
        return merged_df


def main():
    """
    Main function - Usage example
    """
    print("="*80)
    print("🎯 DATA PREPROCESSING FOR FALL DETECTION")
    print("="*80)
    
    # Paths
    base_path = Path(__file__).parent
    weda_path = base_path / "WEDA-FALL-main" / "dataset" / "50Hz"
    collected_path = base_path / "server" / "data" / "collected"
    output_path = base_path / "processed_data"
    
    # Initialize preprocessor
    preprocessor = FallDataPreprocessor(
        target_sample_rate=50,
        window_size=2.0,      # 2 seconds window
        overlap=0.5           # 50% overlap
    )
    
    # ========================================
    # STEP 1: Process WEDA-FALL dataset
    # ========================================
    print("\n🔹 STEP 1: Processing WEDA-FALL dataset...")
    
    try:
        weda_df = preprocessor.process_weda_fall(weda_path, output_path)
    except Exception as e:
        print(f"⚠️ Error processing WEDA-FALL: {e}")
        weda_df = None
    
    # ========================================
    # STEP 2: Process collected data
    # ========================================
    print("\n🔹 STEP 2: Processing collected data...")
    
    collected_dfs = []
    
    try:
        # Tìm tất cả sessions
        sessions = sorted(collected_path.glob('session_*'))
        
        if not sessions:
            print("⚠️ No collected data found!")
        else:
            for session_path in sessions:
                print(f"\nProcessing: {session_path.name}")
                
                # TODO: Bạn cần label data này!
                # - Label 0 (ADL) nếu đây là hoạt động bình thường
                # - Label 1 (Fall) nếu đây là té ngã
                label = 0  # Mặc định ADL, thay đổi nếu cần
                
                try:
                    collected_df = preprocessor.process_collected_data(
                        session_path, 
                        label=label,
                        output_path=output_path
                    )
                    collected_dfs.append(collected_df)
                except Exception as e:
                    print(f"  ✗ Error: {e}")
    
    except Exception as e:
        print(f"⚠️ Error processing collected data: {e}")
    
    # ========================================
    # STEP 3: Merge datasets
    # ========================================
    if weda_df is not None or collected_dfs:
        print("\n🔹 STEP 3: Merging datasets...")
        
        datasets_to_merge = []
        if weda_df is not None:
            datasets_to_merge.append(weda_df)
        datasets_to_merge.extend(collected_dfs)
        
        if len(datasets_to_merge) > 1:
            merged_df = preprocessor.merge_datasets(
                datasets_to_merge[0],
                datasets_to_merge[1:],
                output_path
            )
        else:
            print("⚠️ Only one dataset available, skipping merge")
    
    print("\n" + "="*80)
    print("✅ PREPROCESSING COMPLETE!")
    print("="*80)
    print(f"📁 Output folder: {output_path}")
    print("📄 Files created:")
    print("   - weda_fall_processed.csv (WEDA-FALL dataset)")
    print("   - session_*_processed.csv (Your collected data)")
    print("   - merged_dataset.csv (Combined dataset)")
    print("   - feature_names.json (Feature list)")
    print("\n💡 Next step: Train your fall detection model!")
    print("="*80)


if __name__ == "__main__":
    main()
