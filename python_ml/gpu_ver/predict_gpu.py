from __future__ import annotations

import os
from typing import Any

from predict import FallPredictor as BaseFallPredictor
from predict import TENSORFLOW_IMPORT_ERROR, tf


def configure_gpu_runtime() -> dict[str, Any]:
    status: dict[str, Any] = {
        "mode": "gpu",
        "requested_gpu_index": os.getenv("ML_GPU_INDEX", "0"),
        "gpu_available": False,
        "gpu_devices": [],
        "gpu_selected": None,
        "memory_growth_enabled": False,
        "warning": None,
        "error": None,
    }

    if tf is None:
        status["error"] = f"TensorFlow unavailable: {TENSORFLOW_IMPORT_ERROR}"
        return status

    try:
        gpus = tf.config.list_physical_devices("GPU")
    except Exception as exc:  # pragma: no cover - depends on local runtime
        status["error"] = f"Failed to list GPU devices: {exc}"
        return status

    status["gpu_devices"] = [gpu.name for gpu in gpus]
    status["gpu_available"] = len(gpus) > 0

    if not gpus:
        status["warning"] = (
            "No TensorFlow GPU device detected. On Windows, TensorFlow >= 2.11 "
            "needs WSL2 or another supported Linux environment for CUDA GPU use."
        )
        return status

    try:
        requested_index = int(status["requested_gpu_index"])
    except ValueError:
        requested_index = 0

    if requested_index < 0 or requested_index >= len(gpus):
        requested_index = 0

    selected_gpu = gpus[requested_index]

    try:
        tf.config.set_visible_devices(selected_gpu, "GPU")
        tf.config.experimental.set_memory_growth(selected_gpu, True)
        logical_gpus = tf.config.list_logical_devices("GPU")
        status["gpu_selected"] = selected_gpu.name
        status["memory_growth_enabled"] = True
        status["logical_gpu_devices"] = [gpu.name for gpu in logical_gpus]
    except Exception as exc:  # pragma: no cover - depends on local runtime
        status["error"] = f"Failed to configure GPU runtime: {exc}"

    return status


class FallPredictor(BaseFallPredictor):
    def __init__(self, *args, **kwargs) -> None:
        self.gpu_status = configure_gpu_runtime()
        super().__init__(*args, **kwargs)

    def get_status(self) -> dict[str, Any]:
        status = super().get_status()
        status["gpu"] = self.gpu_status
        return status
