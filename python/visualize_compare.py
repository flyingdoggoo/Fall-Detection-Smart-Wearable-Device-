"""So sánh WEDA-FALL với data thu thập (script gọn).

Mục tiêu:
- Dễ chỉnh bằng python/.env
- Ít comment, dễ đọc
- Có sanity-check cho case đặt yên trên bàn

Chạy từ repo root:
    python python/visualize_compare.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None

try:
    from scipy import signal  # optional
except Exception:  # pragma: no cover
    signal = None

import os


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def as_bool(v: str | None, default: bool = False) -> bool:
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def as_float(v: str | None) -> Optional[float]:
    if v is None:
        return None
    s = v.strip()
    if s == "":
        return None
    return float(s)


@dataclass(frozen=True)
class Config:
    weda_activity: str
    weda_user: str
    weda_run: str

    collected_session_dir: Optional[Path]
    collected_label: str  # 0 | 1 | any

    resample_weda: bool
    resample_collected: bool
    lowpass_cutoff_hz: Optional[float]

    collected_remove_dc: bool

    weda_remove_dc: bool
    weda_extra_lowpass_hz: Optional[float]
    weda_match_std: bool

    verbose: bool


def load_config() -> Config:
    env_path = Path(__file__).resolve().parent / ".env"
    if load_dotenv is not None:
        load_dotenv(env_path)

    weda_activity = os.getenv("WEDA_ACTIVITY", "D01")
    weda_user = os.getenv("WEDA_USER", "U01")
    weda_run = os.getenv("WEDA_RUN", "R01")

    csd = os.getenv("COLLECTED_SESSION_DIR")
    collected_session_dir = Path(csd) if csd and csd.strip() else None

    collected_label = os.getenv("COLLECTED_LABEL", "any").strip().lower()
    if collected_label not in {"0", "1", "any"}:
        collected_label = "any"

    resample_weda = as_bool(os.getenv("RESAMPLE_WEDA"), True)
    resample_collected = as_bool(os.getenv("RESAMPLE_COLLECTED"), False)
    lowpass_cutoff_hz = as_float(os.getenv("LOWPASS_CUTOFF_HZ"))

    collected_remove_dc = as_bool(os.getenv("COLLECTED_REMOVE_DC"), False)

    weda_remove_dc = as_bool(os.getenv("WEDA_REMOVE_DC"), True)
    weda_extra_lowpass_hz = as_float(os.getenv("WEDA_EXTRA_LOWPASS_HZ"))
    weda_match_std = as_bool(os.getenv("WEDA_MATCH_STD"), False)

    verbose = as_bool(os.getenv("VERBOSE"), False)

    return Config(
        weda_activity=weda_activity,
        weda_user=weda_user,
        weda_run=weda_run,
        collected_session_dir=collected_session_dir,
        collected_label=collected_label,
        resample_weda=resample_weda,
        resample_collected=resample_collected,
        lowpass_cutoff_hz=lowpass_cutoff_hz,
        collected_remove_dc=collected_remove_dc,
        weda_remove_dc=weda_remove_dc,
        weda_extra_lowpass_hz=weda_extra_lowpass_hz,
        weda_match_std=weda_match_std,
        verbose=verbose,
    )


def lowpass(x: np.ndarray, sample_rate_hz: float, cutoff_hz: float) -> np.ndarray:
    if signal is None:
        return x
    nyq = 0.5 * sample_rate_hz
    norm = float(cutoff_hz) / nyq
    if not (0 < norm < 1):
        return x
    b, a = signal.butter(4, norm, btype="low")
    return signal.filtfilt(b, a, x)


def resample_linear(df: pd.DataFrame, time_col: str, cols: list[str], target_hz: float) -> pd.DataFrame:
    d = df.dropna(subset=[time_col]).sort_values(time_col).drop_duplicates(subset=[time_col])
    t = d[time_col].to_numpy(dtype=float)
    if len(t) < 2:
        return d.reset_index(drop=True)

    t0, t1 = float(t[0]), float(t[-1])
    if t1 <= t0:
        return d.reset_index(drop=True)

    dt = 1.0 / float(target_hz)
    new_t = np.arange(t0, t1 + 1e-9, dt)

    out = {time_col: new_t}
    for c in cols:
        y = d[c].to_numpy(dtype=float)
        out[c] = np.interp(new_t, t, y)

    return pd.DataFrame(out)


def remove_dc(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        v = out[c].to_numpy(dtype=float)
        out[c] = v - float(np.nanmean(v))
    return out


def magnitude(df: pd.DataFrame, x: str, y: str, z: str) -> np.ndarray:
    return np.sqrt(df[x].to_numpy(dtype=float) ** 2 + df[y].to_numpy(dtype=float) ** 2 + df[z].to_numpy(dtype=float) ** 2)


def load_weda(weda_root_50hz: Path, activity: str, user: str, run: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = weda_root_50hz / activity
    accel = pd.read_csv(base / f"{user}_{run}_accel.csv")
    gyro = pd.read_csv(base / f"{user}_{run}_gyro.csv")
    return accel, gyro


def find_latest_session(collected_root: Path, label: str) -> Path:
    roots = [collected_root]
    normal = collected_root / "Normal"
    fall = collected_root / "Fall"

    if label == "0" and normal.exists():
        roots = [normal]
    elif label == "1" and fall.exists():
        roots = [fall]
    else:
        if normal.exists():
            roots.append(normal)
        if fall.exists():
            roots.append(fall)

    sessions: list[Path] = []
    for r in roots:
        sessions.extend([p for p in r.glob("label*_*") if p.is_dir()])
        sessions.extend([p for p in r.glob("session_*") if p.is_dir()])

    if not sessions:
        raise FileNotFoundError(f"Không tìm thấy session trong {collected_root}")

    return max(sessions, key=lambda p: p.stat().st_mtime)


def load_collected(session_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    accel = pd.read_csv(session_dir / "accel.csv")
    gyro = pd.read_csv(session_dir / "gyro.csv")
    return accel, gyro


def fine_tune_weda(
    weda_accel: pd.DataFrame,
    weda_gyro: pd.DataFrame,
    collected_accel: pd.DataFrame,
    collected_gyro: pd.DataFrame,
    *,
    sample_rate_hz: float,
    remove_dc_flag: bool,
    extra_lowpass_hz: Optional[float],
    match_std: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    wa = weda_accel.copy()
    wg = weda_gyro.copy()

    accel_cols = ["accel_x_list", "accel_y_list", "accel_z_list"]
    gyro_cols = ["gyro_x_list", "gyro_y_list", "gyro_z_list"]

    if extra_lowpass_hz is not None:
        for c in accel_cols:
            wa[c] = lowpass(wa[c].to_numpy(), sample_rate_hz, float(extra_lowpass_hz))
        for c in gyro_cols:
            wg[c] = lowpass(wg[c].to_numpy(), sample_rate_hz, float(extra_lowpass_hz))

    if remove_dc_flag:
        wa = remove_dc(wa, accel_cols)
        wg = remove_dc(wg, gyro_cols)

    if match_std:
        for c in accel_cols:
            src_std = float(np.nanstd(wa[c].to_numpy(dtype=float))) + 1e-9
            ref_std = float(np.nanstd(collected_accel[c].to_numpy(dtype=float))) + 1e-9
            wa[c] = wa[c] * (ref_std / src_std)
        for c in gyro_cols:
            src_std = float(np.nanstd(wg[c].to_numpy(dtype=float))) + 1e-9
            ref_std = float(np.nanstd(collected_gyro[c].to_numpy(dtype=float))) + 1e-9
            wg[c] = wg[c] * (ref_std / src_std)

    return wa, wg


def plot_compare(weda_accel: pd.DataFrame, weda_gyro: pd.DataFrame, col_accel: pd.DataFrame, col_gyro: pd.DataFrame, title: str) -> plt.Figure:
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(title, fontsize=18, fontweight="bold", y=0.985)

    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.35, top=0.93, bottom=0.05, left=0.08, right=0.98)

    weda_c = "#2E86AB"
    col_c = "#F18F01"

    for i, axis_name in enumerate(["x", "y", "z"]):
        ax = fig.add_subplot(gs[0, i])
        ax.plot(weda_accel["accel_time_list"], weda_accel[f"accel_{axis_name}_list"], label="WEDA", color=weda_c, linewidth=2)
        ax.plot(col_accel["accel_time_list"], col_accel[f"accel_{axis_name}_list"], label="Collected", color=col_c, linewidth=2)
        ax.set_title(f"Accel {axis_name.upper()}", fontweight="bold")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("m/s²")
        ax.legend(loc="upper right", fontsize=9)

    for i, axis_name in enumerate(["x", "y", "z"]):
        ax = fig.add_subplot(gs[1, i])
        ax.plot(weda_gyro["gyro_time_list"], weda_gyro[f"gyro_{axis_name}_list"], label="WEDA", color=weda_c, linewidth=2)
        ax.plot(col_gyro["gyro_time_list"], col_gyro[f"gyro_{axis_name}_list"], label="Collected", color=col_c, linewidth=2)
        ax.set_title(f"Gyro {axis_name.upper()}", fontweight="bold")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("(unit)")
        ax.legend(loc="upper right", fontsize=9)

    ax = fig.add_subplot(gs[2, 0])
    ax.plot(weda_accel["accel_time_list"], magnitude(weda_accel, "accel_x_list", "accel_y_list", "accel_z_list"), label="WEDA", color=weda_c, linewidth=2)
    ax.plot(col_accel["accel_time_list"], magnitude(col_accel, "accel_x_list", "accel_y_list", "accel_z_list"), label="Collected", color=col_c, linewidth=2)
    ax.set_title("Accel Magnitude", fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("m/s²")
    ax.legend(loc="upper right", fontsize=9)

    ax = fig.add_subplot(gs[2, 1])
    ax.plot(weda_gyro["gyro_time_list"], magnitude(weda_gyro, "gyro_x_list", "gyro_y_list", "gyro_z_list"), label="WEDA", color=weda_c, linewidth=2)
    ax.plot(col_gyro["gyro_time_list"], magnitude(col_gyro, "gyro_x_list", "gyro_y_list", "gyro_z_list"), label="Collected", color=col_c, linewidth=2)
    ax.set_title("Gyro Magnitude", fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("(unit)")
    ax.legend(loc="upper right", fontsize=9)

    ax = fig.add_subplot(gs[2, 2])
    ax.text(0.5, 0.5, "OK", ha="center", va="center", fontsize=14, fontweight="bold", transform=ax.transAxes)
    ax.axis("off")

    return fig


def quick_sanity_stationary(accel: pd.DataFrame) -> None:
    mag = magnitude(accel, "accel_x_list", "accel_y_list", "accel_z_list")
    print(f"Sanity accel magnitude: mean={float(np.nanmean(mag)):.3f} m/s² | std={float(np.nanstd(mag)):.3f}")


def main() -> None:
    cfg = load_config()

    root = repo_root()
    weda_root = root / "WEDA-FALL-main" / "dataset" / "50Hz"
    collected_root = root / "server" / "data" / "collected"
    out_dir = root / "Comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    weda_accel, weda_gyro = load_weda(weda_root, cfg.weda_activity, cfg.weda_user, cfg.weda_run)

    session_dir = cfg.collected_session_dir
    if session_dir is None:
        session_dir = find_latest_session(collected_root, cfg.collected_label)

    col_accel, col_gyro = load_collected(session_dir)

    # Preprocess (nhẹ)
    if cfg.resample_weda:
        weda_accel = resample_linear(weda_accel, "accel_time_list", ["accel_x_list", "accel_y_list", "accel_z_list"], 50.0)
        weda_gyro = resample_linear(weda_gyro, "gyro_time_list", ["gyro_x_list", "gyro_y_list", "gyro_z_list"], 50.0)

    if cfg.resample_collected:
        col_accel = resample_linear(col_accel, "accel_time_list", ["accel_x_list", "accel_y_list", "accel_z_list"], 50.0)
        col_gyro = resample_linear(col_gyro, "gyro_time_list", ["gyro_x_list", "gyro_y_list", "gyro_z_list"], 50.0)

    if cfg.lowpass_cutoff_hz is not None:
        for c in ["accel_x_list", "accel_y_list", "accel_z_list"]:
            col_accel[c] = lowpass(col_accel[c].to_numpy(), 50.0, float(cfg.lowpass_cutoff_hz))
        for c in ["gyro_x_list", "gyro_y_list", "gyro_z_list"]:
            col_gyro[c] = lowpass(col_gyro[c].to_numpy(), 50.0, float(cfg.lowpass_cutoff_hz))
        for c in ["accel_x_list", "accel_y_list", "accel_z_list"]:
            weda_accel[c] = lowpass(weda_accel[c].to_numpy(), 50.0, float(cfg.lowpass_cutoff_hz))
        for c in ["gyro_x_list", "gyro_y_list", "gyro_z_list"]:
            weda_gyro[c] = lowpass(weda_gyro[c].to_numpy(), 50.0, float(cfg.lowpass_cutoff_hz))

    if cfg.verbose:
        print(f"WEDA: {cfg.weda_activity}/{cfg.weda_user}_{cfg.weda_run}")
        print(f"Collected: {session_dir}")
        quick_sanity_stationary(col_accel)

    if cfg.collected_remove_dc:
        col_accel = remove_dc(col_accel, ["accel_x_list", "accel_y_list", "accel_z_list"])
        col_gyro = remove_dc(col_gyro, ["gyro_x_list", "gyro_y_list", "gyro_z_list"])

    weda_accel, weda_gyro = fine_tune_weda(
        weda_accel,
        weda_gyro,
        col_accel,
        col_gyro,
        sample_rate_hz=50.0,
        remove_dc_flag=cfg.weda_remove_dc,
        extra_lowpass_hz=cfg.weda_extra_lowpass_hz,
        match_std=cfg.weda_match_std,
    )

    title = f"WEDA {cfg.weda_activity}/{cfg.weda_user}_{cfg.weda_run}  vs  Collected {session_dir.name}"
    fig = plot_compare(weda_accel, weda_gyro, col_accel, col_gyro, title)

    out_file = out_dir / "comparison_simple.png"
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_file}")
    plt.show()


if __name__ == "__main__":
    main()
