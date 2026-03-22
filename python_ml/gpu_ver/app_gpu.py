from __future__ import annotations

import logging
import os

from flask import Flask, jsonify, request

from python_ml.gpu_ver.predict_gpu import FallPredictor


app = Flask(__name__)
predictor = FallPredictor()
app.logger.setLevel(logging.INFO)


@app.get("/")
def root():
    return jsonify(
        {
            "service": "fall-detection-ml-gpu",
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

    try:
        result = predictor.predict(payload)
    except ValueError as exc:
        app.logger.warning("[/predict] bad_request %s", exc)
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        app.logger.exception("[/predict] failed")
        return jsonify({"error": str(exc)}), 500
    return jsonify(result)


if __name__ == "__main__":
    host = os.getenv("ML_HOST", "0.0.0.0")
    port = int(os.getenv("ML_PORT", "5000"))
    debug = os.getenv("ML_DEBUG", "false").lower() == "true"
    app.run(host=host, port=port, debug=debug)
