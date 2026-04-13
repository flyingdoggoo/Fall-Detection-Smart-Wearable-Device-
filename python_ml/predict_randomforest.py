from __future__ import annotations

import json
import math
import os
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.signal import welch
from scipy.stats import skew as scipy_skew

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "config_feature_based.json"
DEFAULT_SCALER_PATH = Path(__file__).resolve().parent / "config" / "scaler_rf.pkl"
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "model" / "final_rf_model.joblib"
DEFAULT_SAMPLE_RATE = 50.0
DEFAULT_WINDOW_SIZE = 100
DEFAULT_NUM_CHANNELS = 6
DEFAULT_THRESHOLD = 0.5

_THRESHOLD_KEYS = (
    "deployment_threshold_rf",
    "threshold_rf",
    "deployment_threshold",
    "threshold",
)


def _pick_existing_path(candidates: list[Path]) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _clean_skew(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0

    # Avoid scipy skew warnings/instability for constant or near-constant signals.
    if values.size < 3:
        return 0.0
    if float(np.std(values)) < 1e-6:
        return 0.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        raw = float(scipy_skew(values, bias=False))
    if math.isnan(raw) or math.isinf(raw):
        return 0.0
    return raw


def extract_features(window: np.ndarray, sample_rate: float = DEFAULT_SAMPLE_RATE) -> np.ndarray:
    if window.ndim != 2 or window.shape[1] != DEFAULT_NUM_CHANNELS:
        raise ValueError(f"Expected window shape (N, {DEFAULT_NUM_CHANNELS}), got {window.shape}")

    dt = 1.0 / max(sample_rate, 1.0)
    acc = window[:, :3]
    smv = np.sqrt(np.sum(acc ** 2, axis=1))

    features: list[float] = []

    for ch_idx in range(DEFAULT_NUM_CHANNELS):
        ch = window[:, ch_idx]
        features.append(float(np.var(ch)))
        features.append(float(np.std(ch)))
        features.append(float(np.mean(ch)))
        features.append(float(np.median(ch)))
        features.append(float(np.max(ch)))
        features.append(float(np.min(ch)))
        features.append(float(np.max(ch) - np.min(ch)))

        _, psd = welch(ch, fs=sample_rate, nperseg=min(32, len(ch)))
        psd_norm = psd / (np.sum(psd) + 1e-8)
        features.append(float(np.mean(psd)))
        features.append(float(-np.sum(psd_norm * np.log2(psd_norm + 1e-8))))
        features.append(_clean_skew(ch))

    features.append(_clean_skew(smv))
    jerk = np.diff(smv) / dt if smv.size > 1 else np.asarray([0.0], dtype=np.float32)
    features.append(float(np.max(np.abs(jerk))))

    feat_arr = np.asarray(features, dtype=np.float32)
    return np.nan_to_num(feat_arr, nan=0.0, posinf=0.0, neginf=0.0)


class RandomForestFallPredictor:
    def __init__(self, model_path: str | Path | None = None, threshold: float | None = None) -> None:
        base_dir = Path(__file__).resolve().parent

        config_env = os.getenv("ML_RF_CONFIG_PATH") or os.getenv("ML_CONFIG_PATH")
        scaler_env = os.getenv("ML_RF_SCALER_PATH") or os.getenv("ML_SCALER_PATH")
        model_env = model_path or os.getenv("ML_RF_MODEL_PATH")

        self.config_path = (
            Path(config_env)
            if config_env
            else _pick_existing_path(
                [
                    base_dir / "config" / "config_feature.json",
                    base_dir / "config" / "config_feature_based.json",
                ]
            )
        )
        self.scaler_path = (
            Path(scaler_env)
            if scaler_env
            else _pick_existing_path(
                [
                    base_dir / "config" / "scaler_rf.pkl",
                    base_dir / "config" / "scaler_feature.pkl",
                ]
            )
        )
        self.model_path = (
            Path(model_env)
            if model_env
            else _pick_existing_path(
                [
                    base_dir / "model" / "model_rf.joblib",
                    base_dir / "model" / "final_rf_model.joblib",
                ]
            )
        )

        self.config = self._load_config()
        self.threshold, self.threshold_source = self._resolve_threshold(threshold)

        self.window_size = _safe_int(self.config.get("window_size"), DEFAULT_WINDOW_SIZE)
        self.sample_rate = _safe_float(self.config.get("sample_rate"), DEFAULT_SAMPLE_RATE)
        self.num_channels = _safe_int(self.config.get("num_channels"), DEFAULT_NUM_CHANNELS)
        self.model_version = str(self.config.get("version") or self.model_path.stem)
        self.backend = "random_forest"

        feature_names = self.config.get("feature_names")
        self.feature_names = feature_names if isinstance(feature_names, list) else []
        self.expected_n_features = _safe_int(
            self.config.get("n_features"),
            len(self.feature_names) if self.feature_names else 0,
        )

        self.scaler = self._load_scaler()
        self.model = self._load_model()
        self.n_features = self._resolve_feature_count()

    def _load_config(self) -> dict[str, Any]:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with self.config_path.open("r", encoding="utf-8") as config_file:
            loaded = json.load(config_file)

        if not isinstance(loaded, dict):
            raise ValueError(f"Config must be a JSON object: {self.config_path}")

        return loaded

    def _resolve_threshold(self, threshold: float | None) -> tuple[float, str]:
        if threshold is not None:
            return _clip01(_safe_float(threshold, DEFAULT_THRESHOLD)), "constructor_argument"

        env_threshold = os.getenv("ML_FALL_THRESHOLD")
        if env_threshold is not None and env_threshold.strip() != "":
            return _clip01(_safe_float(env_threshold, DEFAULT_THRESHOLD)), "env:ML_FALL_THRESHOLD"

        for key in _THRESHOLD_KEYS:
            if key in self.config:
                return _clip01(_safe_float(self.config.get(key), DEFAULT_THRESHOLD)), f"config:{key}"

        return DEFAULT_THRESHOLD, "default:0.5"

    def _load_scaler(self):
        if not self.scaler_path.exists():
            raise FileNotFoundError(f"Scaler file not found: {self.scaler_path}")

        scaler = joblib.load(self.scaler_path)
        if not hasattr(scaler, "transform"):
            raise TypeError("Loaded scaler does not provide transform()")

        return scaler

    def _load_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        model = joblib.load(self.model_path)
        if not hasattr(model, "predict"):
            raise TypeError("Loaded model does not provide predict()")

        if not hasattr(model, "predict_proba") and not hasattr(model, "decision_function"):
            raise TypeError("Loaded model must provide predict_proba() or decision_function()")

        return model

    def _resolve_feature_count(self) -> int:
        scaler_features = getattr(self.scaler, "n_features_in_", None)
        model_features = getattr(self.model, "n_features_in_", None)

        candidates = [
            self.expected_n_features if self.expected_n_features > 0 else None,
            int(scaler_features) if scaler_features is not None else None,
            int(model_features) if model_features is not None else None,
        ]
        resolved = next((value for value in candidates if value is not None), None)

        if resolved is None:
            raise ValueError("Unable to determine expected feature count from config/model/scaler")

        if scaler_features is not None and int(scaler_features) != resolved:
            raise ValueError(
                f"Scaler feature count mismatch: expected {resolved}, got {int(scaler_features)}"
            )

        if model_features is not None and int(model_features) != resolved:
            raise ValueError(
                f"Model feature count mismatch: expected {resolved}, got {int(model_features)}"
            )

        return int(resolved)

    def _build_window(self, raw_window: dict[str, Any]) -> np.ndarray:
        accel_data = raw_window.get("accel_data")
        gyro_data = raw_window.get("gyro_data")

        if not isinstance(accel_data, list) or not isinstance(gyro_data, list):
            raise ValueError("raw_window.accel_data and raw_window.gyro_data must both be arrays")

        if len(accel_data) != self.window_size or len(gyro_data) != self.window_size:
            raise ValueError(
                f"Expected exactly {self.window_size} accel and gyro samples, "
                f"received {len(accel_data)} and {len(gyro_data)}"
            )

        rows = []
        for idx in range(self.window_size):
            accel = accel_data[idx] or {}
            gyro = gyro_data[idx] or {}
            rows.append(
                [
                    _safe_float(accel.get("x")),
                    _safe_float(accel.get("y")),
                    _safe_float(accel.get("z")),
                    _safe_float(gyro.get("x")),
                    _safe_float(gyro.get("y")),
                    _safe_float(gyro.get("z")),
                ]
            )

        window = np.asarray(rows, dtype=np.float32)
        if window.shape != (self.window_size, self.num_channels):
            raise ValueError(
                f"Expected window shape ({self.window_size}, {self.num_channels}), received {window.shape}"
            )

        return window

    def _extract_confidence(self, scaled_features: np.ndarray) -> float:
        if hasattr(self.model, "predict_proba"):
            proba = np.asarray(self.model.predict_proba(scaled_features), dtype=np.float32)
            if proba.ndim == 2 and proba.shape[1] >= 2:
                return _clip01(float(proba[0, 1]))
            if proba.size > 0:
                return _clip01(float(proba.reshape(-1)[0]))

        if hasattr(self.model, "decision_function"):
            score = float(np.asarray(self.model.decision_function(scaled_features)).reshape(-1)[0])
            return _clip01(1.0 / (1.0 + math.exp(-score)))

        pred = float(np.asarray(self.model.predict(scaled_features)).reshape(-1)[0])
        return _clip01(pred)

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        raw_window = payload.get("raw_window") or {}
        window = self._build_window(raw_window)
        feature_vector = extract_features(window, sample_rate=self.sample_rate)

        if feature_vector.size != self.n_features:
            raise ValueError(
                f"Extracted feature size mismatch: expected {self.n_features}, got {feature_vector.size}"
            )

        scaled = self.scaler.transform(feature_vector.reshape(1, -1)).astype(np.float32)
        confidence = self._extract_confidence(scaled)

        return {
            "fall_detected": confidence >= self.threshold,
            "confidence": round(confidence, 4),
            "threshold": self.threshold,
            "model_version": self.model_version,
            "backend": self.backend,
            "model_loaded": True,
            "warnings": [],
        }

    def get_status(self) -> dict[str, Any]:
        return {
            "status": "ready",
            "model_loaded": True,
            "model_path": str(self.model_path),
            "model_version": self.model_version,
            "backend": self.backend,
            "threshold": self.threshold,
            "threshold_source": self.threshold_source,
            "error": None,
            "config_path": str(self.config_path),
            "scaler_path": str(self.scaler_path),
            "window_size": self.window_size,
            "sample_rate": self.sample_rate,
            "num_channels": self.num_channels,
            "n_features": self.n_features,
            "config_n_features": self.expected_n_features,
            "scaler_loaded": True,
            "approach": self.config.get("approach", "feature_based"),
            "class_names": self.config.get("class_names", ["Normal", "Fall"]),
        }
