from __future__ import annotations

import logging
import os

from flask import Flask, jsonify, request

from predict import FallPredictor


app = Flask(__name__)
predictor = FallPredictor()
app.logger.setLevel(logging.INFO)


@app.get("/")
def root():
    return jsonify(
        {
            "service": "fall-detection-ml",
            "status": "running",
            "health": "/health",
            "predict": "/predict",
        }
    )


@app.get("/health")
def health():
    return jsonify(predictor.get_status())


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        app.logger.warning("[/predict] invalid_payload")
        return jsonify({"error": "Expected a JSON object request body"}), 400

    app.logger.info("[/predict] request %s", summarize_predict_payload(payload))
    try:
        result = predictor.predict(payload)
    except ValueError as exc:
        app.logger.warning("[/predict] bad_request %s", exc)
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        app.logger.exception("[/predict] failed")
        return jsonify({"error": str(exc)}), 500
    app.logger.info("[/predict] response %s", summarize_predict_response(result))
    return jsonify(result)


def summarize_predict_payload(payload):
    raw_window = payload.get("raw_window") or {}
    accel_data = raw_window.get("accel_data") or []
    gyro_data = raw_window.get("gyro_data") or []
    features = payload.get("features") or {}
    return {
        "accel_samples": len(accel_data) if isinstance(accel_data, list) else 0,
        "gyro_samples": len(gyro_data) if isinstance(gyro_data, list) else 0,
        "features": {
            "magnitude_avg": features.get("magnitude_avg"),
            "sma": features.get("sma"),
            "max_accel": features.get("max_accel"),
            "max_gyro": features.get("max_gyro"),
            "std_accel": features.get("std_accel"),
            "jerk_peak": features.get("jerk_peak"),
        },
    }


def summarize_predict_response(result):
    return {
        "fall_detected": result.get("fall_detected"),
        "confidence": result.get("confidence"),
        "threshold": result.get("threshold"),
    }


if __name__ == "__main__":
    host = os.getenv("ML_HOST", "0.0.0.0")
    port = int(os.getenv("ML_PORT", "5000"))
    debug = os.getenv("ML_DEBUG", "false").lower() == "true"
    app.run(host=host, port=port, debug=debug)
