from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np

try:
    import tensorflow as tf
except Exception as exc:
    tf = None
    TENSORFLOW_IMPORT_ERROR = str(exc)
else:
    TENSORFLOW_IMPORT_ERROR = None


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "model_config_colab.json"
DEFAULT_SCALER_PATH = Path(__file__).resolve().parent / "config" / "scaler_colab.pkl"
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "model" / "model_v4.keras"
DEFAULT_LEARNING_RATE = 1e-3


if tf is not None:

    @tf.keras.utils.register_keras_serializable(package="FallDetection")
    class SelfAttention(tf.keras.layers.Layer):
        def __init__(self, units: int, **kwargs):
            super().__init__(**kwargs)
            self.units = units
            self.W = tf.keras.layers.Dense(units, name="W")
            self.V = tf.keras.layers.Dense(1, name="V")

        def build(self, input_shape):
            self.W.build(input_shape)
            w_output_shape = tf.TensorShape(input_shape[:-1]).concatenate(self.units)
            self.V.build(w_output_shape)
            super().build(input_shape)

        def call(self, inputs):
            score = tf.nn.tanh(self.W(inputs))
            attention_weights = tf.nn.softmax(self.V(score), axis=1)
            attention_weights = tf.cast(attention_weights, inputs.dtype)
            context_vector = attention_weights * inputs
            return tf.reduce_sum(context_vector, axis=1)

        def compute_output_shape(self, input_shape):
            return tf.TensorShape((input_shape[0], input_shape[-1]))

        def get_config(self):
            config = super().get_config()
            config.update({"units": self.units})
            return config


def focal_loss_fn(y_true, y_pred, gamma: float = 2.0, alpha: float = 0.25):
    if tf is None:
        raise RuntimeError("TensorFlow is required for focal_loss_fn")

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
    cross_entropy = -y_true * tf.math.log(y_pred) - (1.0 - y_true) * tf.math.log(1.0 - y_pred)
    weight = alpha * y_true * tf.pow(1.0 - y_pred, gamma) + (1.0 - alpha) * (1.0 - y_true) * tf.pow(y_pred, gamma)
    return tf.reduce_mean(weight * cross_entropy)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


class FallPredictor:
    def __init__(self, model_path: str | Path | None = None, threshold: float | None = None) -> None:
        self.model_path = Path(model_path or DEFAULT_MODEL_PATH)
        self.config_path = Path(os.getenv("ML_CONFIG_PATH", DEFAULT_CONFIG_PATH))
        self.scaler_path = Path(os.getenv("ML_SCALER_PATH", DEFAULT_SCALER_PATH))
        self.config = self._load_config()

        configured_threshold = _safe_float(self.config.get("threshold"), 0.8)
        env_threshold = os.getenv("ML_FALL_THRESHOLD")
        resolved_threshold = threshold if threshold is not None else env_threshold
        if resolved_threshold is None:
            resolved_threshold = configured_threshold

        self.threshold = _clip01(_safe_float(resolved_threshold, configured_threshold))
        self.window_size = int(self.config.get("window_size", 100) or 100)
        self.num_channels = int(self.config.get("num_channels", 6) or 6)
        self.model_version = self.model_path.stem
        self.backend = "tensorflow"
        self.scaler = self._load_scaler()
        self.model = self._load_model()

    def _load_config(self) -> dict[str, Any]:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with self.config_path.open("r", encoding="utf-8") as config_file:
            loaded = json.load(config_file)

        if not isinstance(loaded, dict):
            raise ValueError(f"Config must be a JSON object: {self.config_path}")

        return loaded

    def _load_scaler(self):
        if not self.scaler_path.exists():
            raise FileNotFoundError(f"Scaler file not found: {self.scaler_path}")

        with self.scaler_path.open("rb") as scaler_file:
            scaler = pickle.load(scaler_file)

        if not hasattr(scaler, "transform"):
            raise TypeError("Loaded scaler does not provide transform()")

        return scaler

    def _load_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        if tf is None:
            raise RuntimeError(f"TensorFlow unavailable: {TENSORFLOW_IMPORT_ERROR}")

        model = tf.keras.models.load_model(
            self.model_path,
            custom_objects={"SelfAttention": SelfAttention},
            compile=False,
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=DEFAULT_LEARNING_RATE),
            loss=focal_loss_fn,
            metrics=[
                "accuracy",
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
                tf.keras.metrics.AUC(name="auc"),
                tf.keras.metrics.AUC(name="auc_pr", curve="PR"),
            ],
        )
        return model

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        raw_window = payload.get("raw_window") or {}
        scaled_window = self._build_scaled_window(raw_window)
        prediction = self.model.predict(scaled_window, verbose=0)
        confidence = self._extract_confidence(prediction)

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
            "error": None,
            "config_path": str(self.config_path),
            "scaler_path": str(self.scaler_path),
            "window_size": self.window_size,
            "num_channels": self.num_channels,
            "scaler_loaded": True,
            "scaler_error": None,
        }

    def _build_scaled_window(self, raw_window: dict[str, Any]) -> np.ndarray:
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
        for index in range(self.window_size):
            accel = accel_data[index] or {}
            gyro = gyro_data[index] or {}
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

        return self.scaler.transform(window).reshape(1, self.window_size, self.num_channels).astype(np.float32)

    def _extract_confidence(self, prediction: Any) -> float:
        flat = np.asarray(prediction, dtype=np.float32).reshape(-1)
        if flat.size == 0:
            raise ValueError("Model returned an empty prediction array")
        return _clip01(float(flat[0]))
