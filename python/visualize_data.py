"""Visualization utilities.

Folder layout (recommended):
- python/: scripts
- server/: Node server + collected data under server/data/collected/
- WEDA-FALL-main/: dataset
- Comparison/: generated PNG outputs

Run:
    python python/visualize_data.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def _repo_root() -> Path:
    # This file lives under <repo>/python/visualize_data.py
    return Path(__file__).resolve().parents[1]

try:
    from scipy import signal  # optional (for low-pass)
except Exception:  # pragma: no cover
    signal = None

try:
    # Re-use preprocessing logic (resample) from preprocess_data.py
    from preprocess_data import FallDataPreprocessor
except Exception:  # pragma: no cover
    FallDataPreprocessor = None


plt.style.use("seaborn-v0_8-darkgrid")

class FallDataVisualizer:
    """Load + preprocess + plot comparison."""

    def __init__(self, weda_path: Path, collected_path: Path, sample_rate_hz: float = 50.0, verbose: bool = False):
        self.weda_path = Path(weda_path)
        self.collected_path = Path(collected_path)
        self.sample_rate_hz = float(sample_rate_hz)
        self.verbose = bool(verbose)

        self._preprocessor = None
        if FallDataPreprocessor is not None:
            self._preprocessor = FallDataPreprocessor(target_sample_rate=int(self.sample_rate_hz))
        
    def load_weda_data(self, activity_type='F01', user='U01', run='R01'):
        """
        Load dữ liệu WEDA-FALL
        
        Args:
            activity_type: F01-F08 (Fall) hoặc D01-D11 (ADL)
            user: U01-U14 (Young) hoặc U21-U31 (Elder)
            run: R01, R02, R03 (Số lần thực hiện)
        """
        base_path = self.weda_path / activity_type
        
        # Load accelerometer
        accel_file = base_path / f"{user}_{run}_accel.csv"
        accel_df = pd.read_csv(accel_file)
        
        # Load gyroscope
        gyro_file = base_path / f"{user}_{run}_gyro.csv"
        gyro_df = pd.read_csv(gyro_file)
        
        if self.verbose:
            print(f"Loaded WEDA-FALL: {activity_type}/{user}_{run} (accel={len(accel_df)}, gyro={len(gyro_df)})")
        
        return accel_df, gyro_df
    
    def load_collected_data(self, session_id=None):
        """
        Load dữ liệu thu thập từ ESP32
        
        Args:
            session_id: Tên session (VD: "label0_2026-01-26T10-30-00" hoặc "label1_...")
                       Nếu None, sẽ lấy session mới nhất
        """
        def _candidate_roots() -> list[Path]:
            roots = [self.collected_path]
            normal_dir = self.collected_path / "Normal"
            fall_dir = self.collected_path / "Fall"
            if normal_dir.exists():
                roots.append(normal_dir)
            if fall_dir.exists():
                roots.append(fall_dir)
            return roots

        def _find_sessions_under(root: Path) -> list[Path]:
            sessions = list(root.glob("label*_*"))
            if not sessions:
                sessions = list(root.glob("session_*"))
            return [p for p in sessions if p.is_dir()]

        if session_id is None:
            sessions: list[Path] = []
            for root in _candidate_roots():
                sessions.extend(_find_sessions_under(root))
            if not sessions:
                raise FileNotFoundError("Không tìm thấy session nào!")
            # pick newest by modified time
            session_path = max(sessions, key=lambda p: p.stat().st_mtime)
        else:
            # Accept either a folder name (label0_...) or a direct folder path
            sid_path = Path(session_id)
            if sid_path.exists() and sid_path.is_dir():
                session_path = sid_path
            else:
                # Try direct under collected root
                direct = self.collected_path / str(session_id)
                if direct.exists() and direct.is_dir():
                    session_path = direct
                else:
                    # Search under Normal/Fall
                    found = None
                    for root in _candidate_roots():
                        cand = root / str(session_id)
                        if cand.exists() and cand.is_dir():
                            found = cand
                            break
                    if found is None:
                        raise FileNotFoundError(f"Session not found: {session_id}")
                    session_path = found
            
        # Load accelerometer
        accel_file = session_path / "accel.csv"
        accel_df = pd.read_csv(accel_file)
        
        # Load gyroscope
        gyro_file = session_path / "gyro.csv"
        gyro_df = pd.read_csv(gyro_file)
        
        # Load label
        label_file = session_path / "label.txt"
        if label_file.exists():
            with open(label_file, 'r') as f:
                label = int(f.read().strip())
            label_text = "🔴 FALL" if label == 1 else "🟢 NORMAL"
        else:
            # Fallback: detect from folder name
            if 'label1' in session_path.name:
                label = 1
                label_text = "🔴 FALL"
            elif 'label0' in session_path.name:
                label = 0
                label_text = "🟢 NORMAL"
            else:
                label = None
                label_text = "❓ UNKNOWN"
        
        if self.verbose:
            print(f"Loaded collected: {session_path.name} ({label_text}), accel={len(accel_df)}, gyro={len(gyro_df)}")
        
        return accel_df, gyro_df, session_path.name, label, label_text

    def _ensure_monotonic_time(self, df: pd.DataFrame, time_col: str) -> pd.DataFrame:
        if time_col not in df.columns:
            raise KeyError(f"Missing column '{time_col}'")
        out = df.copy()
        out = out.dropna(subset=[time_col])
        out = out.sort_values(time_col)
        out = out.drop_duplicates(subset=[time_col], keep="first")
        return out.reset_index(drop=True)

    def _resample_if_needed(self, df: pd.DataFrame, time_col: str, data_cols: list[str]) -> pd.DataFrame:
        df = self._ensure_monotonic_time(df, time_col)

        # If we cannot import preprocessor, just return as-is.
        if self._preprocessor is None:
            return df

        start_time = float(df[time_col].iloc[0])
        end_time = float(df[time_col].iloc[-1])
        duration = max(0.0, end_time - start_time)
        # Avoid 0/1 sample edge cases
        if duration <= 0:
            return df

        return self._preprocessor.resample_data(df, time_col=time_col, data_cols=data_cols, target_rate=int(self.sample_rate_hz))

    def _lowpass(self, x: np.ndarray, cutoff_hz: float) -> np.ndarray:
        if signal is None:
            return x
        nyq = 0.5 * self.sample_rate_hz
        norm = float(cutoff_hz) / nyq
        if not (0 < norm < 1):
            return x
        b, a = signal.butter(4, norm, btype="low")
        # filtfilt to avoid phase shift
        return signal.filtfilt(b, a, x)

    def fine_tune_weda_to_collected(
        self,
        weda_accel: pd.DataFrame,
        weda_gyro: pd.DataFrame,
        collected_accel: pd.DataFrame,
        collected_gyro: pd.DataFrame,
        *,
        extra_lowpass_hz: float | None = None,
        remove_dc: bool = True,
        match_std: bool = False,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Fine-tune WEDA signals to make visual comparison fairer.

        This does NOT claim to make the datasets identical; it just reduces
        obvious differences caused by device placement, filtering, and bias.

        Steps (optional):
        - extra_lowpass_hz: apply additional low-pass to WEDA only
        - remove_dc: subtract per-axis mean (helps remove gravity/bias offsets)
        - match_std: scale WEDA per-axis std to match collected per-axis std
        """
        wa = weda_accel.copy()
        wg = weda_gyro.copy()

        accel_cols = ["accel_x_list", "accel_y_list", "accel_z_list"]
        gyro_cols = ["gyro_x_list", "gyro_y_list", "gyro_z_list"]

        if extra_lowpass_hz is not None:
            for c in accel_cols:
                wa[c] = self._lowpass(wa[c].to_numpy(), cutoff_hz=float(extra_lowpass_hz))
            for c in gyro_cols:
                wg[c] = self._lowpass(wg[c].to_numpy(), cutoff_hz=float(extra_lowpass_hz))

        if remove_dc:
            for c in accel_cols:
                wa[c] = wa[c] - float(np.nanmean(wa[c].to_numpy()))
            for c in gyro_cols:
                wg[c] = wg[c] - float(np.nanmean(wg[c].to_numpy()))

        if match_std:
            # Match per-axis amplitude scale to collected (after preprocessing)
            for c in accel_cols:
                src = wa[c].to_numpy()
                ref = collected_accel[c].to_numpy()
                src_std = float(np.nanstd(src)) + 1e-9
                ref_std = float(np.nanstd(ref)) + 1e-9
                wa[c] = src * (ref_std / src_std)
            for c in gyro_cols:
                src = wg[c].to_numpy()
                ref = collected_gyro[c].to_numpy()
                src_std = float(np.nanstd(src)) + 1e-9
                ref_std = float(np.nanstd(ref)) + 1e-9
                wg[c] = src * (ref_std / src_std)

        if self.verbose:
            print(
                "WEDA fine-tune applied:",
                {
                    "extra_lowpass_hz": extra_lowpass_hz,
                    "remove_dc": remove_dc,
                    "match_std": match_std,
                },
            )

        return wa, wg

    def preprocess_pair(
        self,
        accel_df: pd.DataFrame,
        gyro_df: pd.DataFrame,
        *,
        resample: bool = True,
        accel_unit: str = "auto",
        gyro_unit: str = "rad/s",
        lowpass_cutoff_hz: float | None = 10.0,
        normalize: str | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Light preprocessing for visualization.

        accel_unit:
            - 'g' or 'm/s2' or 'auto'
        gyro_unit:
            - 'rad/s' or 'deg/s'
        normalize:
            - None | 'zscore'
        """
        accel = accel_df.copy()
        gyro = gyro_df.copy()

        # Optional resample (WEDA timestamps are often irregular)
        if resample:
            accel = self._resample_if_needed(accel, "accel_time_list", ["accel_x_list", "accel_y_list", "accel_z_list"])
            gyro = self._resample_if_needed(gyro, "gyro_time_list", ["gyro_x_list", "gyro_y_list", "gyro_z_list"])

        # Units conversion
        if accel_unit.lower() in {"g", "gs"}:
            g = 9.80665
            for c in ["accel_x_list", "accel_y_list", "accel_z_list"]:
                accel[c] = accel[c] * g
        elif accel_unit.lower() == "auto":
            # Heuristic: if |z| ~ 1.0 on average, likely in g
            z_mean = float(np.nanmean(np.abs(accel["accel_z_list"].to_numpy())))
            if 0.2 <= z_mean <= 2.5:
                g = 9.80665
                for c in ["accel_x_list", "accel_y_list", "accel_z_list"]:
                    accel[c] = accel[c] * g

        if gyro_unit.lower() in {"deg/s", "degps", "dps"}:
            for c in ["gyro_x_list", "gyro_y_list", "gyro_z_list"]:
                gyro[c] = np.deg2rad(gyro[c])

        # Optional low-pass filter
        if lowpass_cutoff_hz is not None:
            for c in ["accel_x_list", "accel_y_list", "accel_z_list"]:
                accel[c] = self._lowpass(accel[c].to_numpy(), cutoff_hz=float(lowpass_cutoff_hz))
            for c in ["gyro_x_list", "gyro_y_list", "gyro_z_list"]:
                gyro[c] = self._lowpass(gyro[c].to_numpy(), cutoff_hz=float(lowpass_cutoff_hz))

        # Optional normalization (for shape comparison)
        if normalize is not None:
            normalize = normalize.lower().strip()
        if normalize == "zscore":
            for c in ["accel_x_list", "accel_y_list", "accel_z_list"]:
                v = accel[c].to_numpy()
                accel[c] = (v - np.nanmean(v)) / (np.nanstd(v) + 1e-9)
            for c in ["gyro_x_list", "gyro_y_list", "gyro_z_list"]:
                v = gyro[c].to_numpy()
                gyro[c] = (v - np.nanmean(v)) / (np.nanstd(v) + 1e-9)

        return accel, gyro

    @staticmethod
    def remove_dc(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        out = df.copy()
        for c in cols:
            v = out[c].to_numpy()
            out[c] = v - float(np.nanmean(v))
        return out

    def print_quick_stats(self, name: str, accel: pd.DataFrame, gyro: pd.DataFrame) -> None:
        if not self.verbose:
            return
        accel_cols = ["accel_x_list", "accel_y_list", "accel_z_list"]
        gyro_cols = ["gyro_x_list", "gyro_y_list", "gyro_z_list"]

        def _stats(df: pd.DataFrame, cols: list[str]) -> dict:
            out = {}
            for c in cols:
                arr = df[c].to_numpy(dtype=float)
                out[c] = {
                    "min": float(np.nanmin(arr)),
                    "max": float(np.nanmax(arr)),
                    "mean": float(np.nanmean(arr)),
                    "std": float(np.nanstd(arr)),
                }
            return out

        print(f"\n[{name}] stats")
        print("  accel:", _stats(accel, accel_cols))
        print("  gyro:", _stats(gyro, gyro_cols))
    
    def plot_comparison(self, weda_accel, weda_gyro, collected_accel, collected_gyro, 
                       weda_label='WEDA-FALL', collected_label='Your Device', collected_label_type=None):
        """
        Vẽ biểu đồ so sánh 2 dataset - MỖI SUBPLOT CHỈ 1 AXIS
        
        Args:
            collected_label_type: Label (e.g., "🔴 FALL" or "🟢 NORMAL")
        """
        fig = plt.figure(figsize=(20, 16))
        
        # Create title with label
        title = f'📊 {weda_label} vs {collected_label}'
        if collected_label_type:
            title += f' [{collected_label_type}]'
        fig.suptitle(title, fontsize=18, fontweight='bold', y=0.985)
        
        # 3 rows x 3 columns: Accel X/Y/Z, Gyro X/Y/Z, Magnitudes
        gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.35, top=0.93, bottom=0.05, left=0.08, right=0.98)
        
        # Row 1: Accelerometer X, Y, Z
        for i, axis_name in enumerate(['x', 'y', 'z']):
            ax = fig.add_subplot(gs[0, i])
            
            # WEDA-FALL
            weda_col = f'accel_{axis_name}_list'
            ax.plot(
                weda_accel["accel_time_list"],
                weda_accel[weda_col],
                label=weda_label,
                alpha=0.9,
                linewidth=2,
                color="#2E86AB",
            )
            
            # Collected
            collected_col = f'accel_{axis_name}_list'
            ax.plot(
                collected_accel["accel_time_list"],
                collected_accel[collected_col],
                label=collected_label,
                alpha=0.9,
                linewidth=2,
                linestyle="-",  # solid line as requested
                color="#F18F01",
            )
            
            ax.set_title(f'Accel {axis_name.upper()}', fontweight='bold', fontsize=12)
            ax.set_xlabel('Time (s)', fontsize=10)
            ax.set_ylabel('m/s²', fontsize=10)
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, alpha=0.3)
        
        # Row 2: Gyroscope X, Y, Z
        for i, axis_name in enumerate(['x', 'y', 'z']):
            ax = fig.add_subplot(gs[1, i])
            
            # WEDA-FALL
            weda_col = f'gyro_{axis_name}_list'
            ax.plot(
                weda_gyro["gyro_time_list"],
                weda_gyro[weda_col],
                label=weda_label,
                alpha=0.9,
                linewidth=2,
                color="#2E86AB",
            )
            
            # Collected
            collected_col = f'gyro_{axis_name}_list'
            ax.plot(
                collected_gyro["gyro_time_list"],
                collected_gyro[collected_col],
                label=collected_label,
                alpha=0.9,
                linewidth=2,
                linestyle="-",  # solid line
                color="#F18F01",
            )
            
            ax.set_title(f'Gyro {axis_name.upper()}', fontweight='bold', fontsize=12)
            ax.set_xlabel('Time (s)', fontsize=10)
            ax.set_ylabel('rad/s', fontsize=10)
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, alpha=0.3)
        
        # Row 3: Magnitudes (Accel, Gyro, Combined)
        # Accel Magnitude
        ax = fig.add_subplot(gs[2, 0])
        weda_mag = np.sqrt(weda_accel['accel_x_list']**2 + 
                          weda_accel['accel_y_list']**2 + 
                          weda_accel['accel_z_list']**2)
        collected_mag = np.sqrt(collected_accel['accel_x_list']**2 + 
                               collected_accel['accel_y_list']**2 + 
                               collected_accel['accel_z_list']**2)
        
        ax.plot(
            weda_accel["accel_time_list"],
            weda_mag,
            label=weda_label,
            alpha=0.9,
            linewidth=2,
            color="#2E86AB",
        )
        ax.plot(
            collected_accel["accel_time_list"],
            collected_mag,
            label=collected_label,
            alpha=0.9,
            linewidth=2,
            linestyle="-",
            color="#F18F01",
        )
        
        ax.set_title('Accel Magnitude', fontweight='bold', fontsize=12)
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('m/s²', fontsize=10)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Gyro Magnitude
        ax = fig.add_subplot(gs[2, 1])
        weda_gyro_mag = np.sqrt(weda_gyro['gyro_x_list']**2 + 
                               weda_gyro['gyro_y_list']**2 + 
                               weda_gyro['gyro_z_list']**2)
        collected_gyro_mag = np.sqrt(collected_gyro['gyro_x_list']**2 + 
                                     collected_gyro['gyro_y_list']**2 + 
                                     collected_gyro['gyro_z_list']**2)
        
        ax.plot(
            weda_gyro["gyro_time_list"],
            weda_gyro_mag,
            label=weda_label,
            alpha=0.9,
            linewidth=2,
            color="#2E86AB",
        )
        ax.plot(
            collected_gyro["gyro_time_list"],
            collected_gyro_mag,
            label=collected_label,
            alpha=0.9,
            linewidth=2,
            linestyle="-",
            color="#F18F01",
        )
        
        ax.set_title('Gyro Magnitude', fontweight='bold', fontsize=12)
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('rad/s', fontsize=10)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Empty subplot for symmetry
        ax = fig.add_subplot(gs[2, 2])
        ax.text(0.5, 0.5, '✅ Comparison Complete', 
               ha='center', va='center', fontsize=14, fontweight='bold',
               transform=ax.transAxes)
        ax.axis('off')
        
        return fig


def main():
# | D01 | Walking |
# | D02 | Jogging |
    WEDA_ACTIVITY = "D01" 
    WEDA_USER = "U22"
    WEDA_RUN = "R01"

    COLLECTED_SESSION_DIR = r"D:\Workspace\Nam4\PBL5\server\data\collected\Normal\label0_2026-01-28T10-43-12"  
    SESSION_ID = None  # None = latest session under server/data/collected

    # Preprocess options (for visualization)
    RESAMPLE_WEDA = True
    RESAMPLE_COLLECTED = False  # collected already 50Hz and evenly sampled
    # LOWPASS_CUTOFF_HZ = 10.0  # set None to disable
    LOWPASS_CUTOFF_HZ = None  # set None to disable
    NORMALIZE = None  # None | 'zscore' (use zscore if you only care about shape)

    # Fine-tune WEDA before comparing to your device
    # NOTE: your collected data is already filtered (Kalman + MPU6050 DLPF),
    # so WEDA may look much more "spiky". Fine-tuning helps compare shapes.
    WEDA_EXTRA_LOWPASS_HZ = None  # e.g. 5.0 to make WEDA smoother than default
    WEDA_REMOVE_DC = True         # subtract mean per axis (reduce gravity/bias offsets)
    WEDA_MATCH_STD = False        # scale WEDA per-axis std to match your device

    # Your device (collected) also includes gravity split across axes depending on how you wear the sensor.
    # Removing DC makes walking/jogging comparisons much more meaningful.
    # COLLECTED_REMOVE_DC = True
    COLLECTED_REMOVE_DC = False

    COLLECTED_ACCEL_UNIT = "auto"  # 'auto' | 'g' | 'm/s2'
    COLLECTED_GYRO_UNIT = "rad/s"  # 'rad/s' | 'deg/s'

    VERBOSE = False

    repo_root = _repo_root()
    weda_path = repo_root / "WEDA-FALL-main" / "dataset" / "50Hz"
    collected_path = repo_root / "server" / "data" / "collected"
    output_dir = repo_root / "Comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    # If user provided an explicit session folder, derive collected_path + SESSION_ID
    if COLLECTED_SESSION_DIR is not None:
        session_dir = Path(COLLECTED_SESSION_DIR)
        if not session_dir.exists():
            raise FileNotFoundError(f"COLLECTED_SESSION_DIR not found: {session_dir}")
        if not session_dir.is_dir():
            raise NotADirectoryError(f"COLLECTED_SESSION_DIR is not a folder: {session_dir}")
        collected_path = session_dir.parent
        SESSION_ID = session_dir.name

    viz = FallDataVisualizer(weda_path, collected_path, sample_rate_hz=50.0, verbose=VERBOSE)

    collected_accel, collected_gyro, session_name, label_value, label_text = viz.load_collected_data(SESSION_ID)
    weda_accel, weda_gyro = viz.load_weda_data(WEDA_ACTIVITY, WEDA_USER, WEDA_RUN)

    # Preprocess
    weda_accel, weda_gyro = viz.preprocess_pair(
        weda_accel,
        weda_gyro,
        resample=RESAMPLE_WEDA,
        accel_unit="m/s2",
        gyro_unit="rad/s",
        lowpass_cutoff_hz=LOWPASS_CUTOFF_HZ,
        normalize=NORMALIZE,
    )

    collected_accel, collected_gyro = viz.preprocess_pair(
        collected_accel,
        collected_gyro,
        resample=RESAMPLE_COLLECTED,
        accel_unit=COLLECTED_ACCEL_UNIT,
        gyro_unit=COLLECTED_GYRO_UNIT,
        lowpass_cutoff_hz=LOWPASS_CUTOFF_HZ,
        normalize=NORMALIZE,
    )

    if COLLECTED_REMOVE_DC:
        collected_accel = viz.remove_dc(collected_accel, ["accel_x_list", "accel_y_list", "accel_z_list"])
        collected_gyro = viz.remove_dc(collected_gyro, ["gyro_x_list", "gyro_y_list", "gyro_z_list"])
        if not VERBOSE:
            print("Note: COLLECTED_REMOVE_DC=True -> accel/gyro axes are centered around 0 (this is expected).")

    # Fine-tune WEDA (optional)
    weda_accel, weda_gyro = viz.fine_tune_weda_to_collected(
        weda_accel,
        weda_gyro,
        collected_accel,
        collected_gyro,
        extra_lowpass_hz=WEDA_EXTRA_LOWPASS_HZ,
        remove_dc=WEDA_REMOVE_DC,
        match_std=WEDA_MATCH_STD,
    )

    viz.print_quick_stats("WEDA (after preprocess)", weda_accel, weda_gyro)
    viz.print_quick_stats("Collected (after preprocess)", collected_accel, collected_gyro)

    # Plot
    fig = viz.plot_comparison(
        weda_accel,
        weda_gyro,
        collected_accel,
        collected_gyro,
        weda_label=f"WEDA-FALL ({WEDA_ACTIVITY}/{WEDA_USER}_{WEDA_RUN})",
        collected_label=f"Your Device ({session_name})",
        collected_label_type=label_text,
    )

    out_file = output_dir / "comparison_weda_vs_collected.png"
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_file}")
    plt.show()


if __name__ == "__main__":
    main()
