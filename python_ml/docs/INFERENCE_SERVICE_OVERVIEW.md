# Inference Service Overview

This document explains how the ML and server-side files work together in the current inference setup.

## Files Covered

- `python_ml/predict.py`
- `python_ml/gpu_ver/predict_gpu.py`
- `python_ml/app.py`
- `python_ml/gpu_ver/app_gpu.py`
- `python_ml/start-ml.bat`
- `python_ml/gpu_ver/start-ml-gpu.bat`
- `server/server.js`
- `start-all.bat`
- `python_ml/requirements.txt`

## 1. `python_ml/predict.py`

`predict.py` is the main inference engine for the default CPU path.

Its job is to:

- load the trained `.keras` model
- load the training config
- load the saved scaler
- convert incoming sensor windows into the exact format the model expects
- run prediction
- return a clean JSON-friendly result

### What it uses and why

- `tensorflow`
  Used to load and run the saved Keras model.
- `numpy`
  Used to build the `(1, 100, 6)` input tensor expected by the model.
- `pickle`
  Used to load `scaler_colab.pkl`.
- `json`
  Used to load `model_config_colab.json`, which contains the training threshold, window size, and channel count.
- custom `SelfAttention` and `focal_loss_fn`
  Required because the saved model references these custom objects from training.

### Why the custom layer and loss are implemented here

The training notebook built and saved the model with custom objects:

- `SelfAttention`
- `focal_loss_fn`

When Keras loads the saved model, those objects must exist again in runtime code. Without them, `load_model(...)` fails.

### Main flow

1. `FallPredictor` loads config from `python_ml/config/model_config_colab.json`.
2. It resolves the threshold, preferring:
   - explicit constructor argument
   - environment variable
   - saved config threshold
3. It loads the scaler from `python_ml/config/scaler_colab.pkl`.
4. It loads the model from `python_ml/model/model_v4.keras` with:
   - `custom_objects=...`
   - `compile=False`
5. It compiles the model with:
   - `Adam(1e-3)`
   - `focal_loss_fn`
   - standard metrics
6. On each prediction request, it reads:
   - `raw_window.accel_data`
   - `raw_window.gyro_data`
7. It checks that both arrays contain exactly `100` samples.
8. It builds a `100 x 6` matrix in this channel order:
   - `ax, ay, az, gx, gy, gz`
9. It applies the saved scaler the same way the training notebook did.
10. It reshapes to `(1, 100, 6)` and runs `model.predict(...)`.
11. It compares the output probability against the saved threshold from training.

## 2. `python_ml/gpu_ver/predict_gpu.py`

`predict_gpu.py` is a GPU-oriented wrapper around `predict.py`.

It reuses the same `FallPredictor` from `predict.py` and adds GPU runtime setup and GPU status reporting.

### What it uses and why

- `tf.config.list_physical_devices("GPU")`
  Used to detect whether TensorFlow can see a GPU.
- `tf.config.set_visible_devices(...)`
  Used to select one GPU if multiple are available.
- `tf.config.experimental.set_memory_growth(...)`
  Used to prevent TensorFlow from grabbing all GPU memory immediately.

### Main flow

1. Before the base predictor is initialized, it checks available GPUs.
2. It reads `ML_GPU_INDEX` if you want to select a specific GPU.
3. If a GPU exists, it enables memory growth and records the selected device.
4. It then runs the normal `FallPredictor` initialization from `predict.py`.
5. `/health` includes GPU status so you can verify whether the machine is using GPU.

## 3. `python_ml/app.py`

`app.py` is the default Flask HTTP wrapper around `predict.py`.

### Why it exists

The Node.js server needs a simple HTTP microservice to call. Keeping HTTP handling separate from the prediction logic makes the code easier to test and maintain.

### Main flow

1. Flask starts.
2. A single `FallPredictor` instance is created at process startup.
3. `GET /health` returns model, scaler, and config status.
4. `POST /predict` validates the request body and forwards it to `predictor.predict(...)`.
5. The prediction result is returned as JSON.

### Logging

The app logs:

- invalid payloads
- compact request summaries
- prediction responses
- request failures

## 4. `python_ml/gpu_ver/app_gpu.py`

`app_gpu.py` follows the same structure as `app.py`, but imports its predictor from `predict_gpu.py`.

### Why keep it separate

- switching between CPU and GPU machines is easier
- GPU runtime configuration stays isolated
- no file swapping is needed

## 5. `python_ml/start-ml.bat`

`start-ml.bat` starts the default ML service.

### Why it is written this way

It explicitly prefers a known Python 3.12 interpreter, because TensorFlow installation and runtime compatibility were validated there.

### Main flow

1. Change into the `python_ml` folder.
2. Prefer `C:\Users\Admin\AppData\Local\Programs\Python\Python312\python.exe`.
3. Fall back to `python`, then `py -3`.
4. Print which interpreter is being used.
5. Start `app.py`.

## 6. `python_ml/gpu_ver/start-ml-gpu.bat`

`start-ml-gpu.bat` is the GPU-oriented launcher.

### Why it exists

Different machines may need different Python, TensorFlow, or GPU configurations. Keeping this as a separate launcher makes machine-specific startup easier.

## 7. `server/server.js`

`server.js` is the Express + Socket.IO data server.

### What it does

- receives sensor windows from ESP32
- saves session data to CSV and metadata files
- calls the Python ML service on each sensor batch
- returns ML results to the caller
- emits live updates to dashboard clients
- stores fall history, device status, and token data
- sends Firebase notifications when configured

### Main flow

1. ESP32 sends a batch to `/api/sensor-batch`.
2. Server validates the payload.
3. CSV and session logic runs.
4. Dashboard broadcasting runs.
5. Server calls Python `/predict`.
6. If ML returns a fall above threshold, server logs the event, emits `fallDetected`, and attempts notifications.
7. Server responds with the ML decision.

### Replay testing

The file also contains a removable replay harness for `server/data/Self-Collected`.

When enabled in code, it can:

- read stored self-collected sessions
- split them into `100`-sample windows
- replay them through the same internal ML request path
- pace replay according to recorded timing
- alternate selected fall and normal session groups for testing

## 8. `start-all.bat`

`start-all.bat` is the local development launcher.

### Main flow

1. start the Python ML service
2. wait briefly
3. start the Node.js server
4. open the dashboard HTML client

### Why this order is used

The server may call the ML service as soon as sensor batches arrive, so starting the ML service first reduces startup race conditions.

## 9. `python_ml/requirements.txt`

This file reflects the current inference runtime requirements.

- `Flask`
  HTTP microservice
- `numpy`
  array shaping and tensor preparation
- `tensorflow`
  model loading and inference
- `scikit-learn==1.6.1`
  required because `scaler_colab.pkl` was saved from scikit-learn and is loaded at runtime

### Why `scikit-learn` is pinned

The saved scaler is a pickle artifact. Pickled scikit-learn objects can be sensitive to version mismatch, so pinning improves reproducibility.
