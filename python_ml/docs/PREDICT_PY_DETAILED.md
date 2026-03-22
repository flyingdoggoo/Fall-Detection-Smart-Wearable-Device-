# `predict.py` Detailed Explanation

This document explains `python_ml/predict.py` in detail: what each part is responsible for, why it exists, and how the inference flow works from input payload to final decision.

## Purpose

`predict.py` is the core inference module for the ML microservice.

It is responsible for:

- loading the trained model
- loading the training-time configuration
- loading the saved scaler
- reconstructing the same input shape used during training
- running TensorFlow inference
- returning a confidence score and fall/non-fall decision

## High-Level Design

The file is built around one main class:

- `FallPredictor`

This class is created once at service startup by `app.py`, so the model is loaded a single time and reused across requests.

## Imports and Why They Are Needed

- `json`
  Reads `model_config_colab.json`.
- `os`
  Reads environment variables for override behavior.
- `pickle`
  Loads `scaler_colab.pkl`.
- `Path`
  Builds cross-platform file paths.
- `typing.Any`
  Used for flexible type hints in request parsing and model outputs.
- `numpy`
  Builds and reshapes input arrays for inference.
- `tensorflow`
  Loads and runs the Keras model.

## TensorFlow Import Guard

The file wraps the TensorFlow import in a `try/except`.

### Why

If TensorFlow is not installed or fails to import, the service can keep a clear error message and raise that error when the predictor tries to load the model.

## Custom Objects From Training

The training notebook used:

- `SelfAttention`
- `focal_loss_fn`

The saved `.keras` model references these custom objects, so they must be recreated here.

### `SelfAttention`

This custom layer performs learned attention over the time dimension after the sequence model.

What it does:

1. applies a dense projection `W`
2. applies `tanh`
3. applies another dense layer `V` to produce attention scores
4. uses `softmax` across the timestep dimension
5. multiplies attention weights with the sequence
6. reduces across time to produce a single context vector

Why the extra methods exist:

- `build(...)`
  Helps Keras correctly construct the internal dense layers during model deserialization.
- `compute_output_shape(...)`
  Helps Keras understand the layer output shape during graph reconstruction.
- `get_config(...)`
  Preserves the custom layer configuration for serialization.

### `focal_loss_fn`

This is recreated because the model was compiled with it during training.

The runtime code compiles the loaded model with this loss and the same optimizer family used in the deployment demo.

## File Path Constants

The file defines default paths for:

- `DEFAULT_MODEL_PATH`
- `DEFAULT_CONFIG_PATH`
- `DEFAULT_SCALER_PATH`
- `DEFAULT_LEARNING_RATE`

### Why

These keep the file self-contained and allow the service to start without manual path wiring every time.

## Utility Helpers

### `_safe_float`

Converts arbitrary values to `float`.

Why:

- incoming JSON may contain numbers, strings, or missing values
- this prevents conversion errors from breaking array construction

### `_clip01`

Clamps a number into the `[0, 1]` range.

Why:

- probabilities and thresholds should stay within valid bounds

## `FallPredictor` Class

This is the main runtime object.

## `__init__`

This method initializes everything needed for inference.

It:

1. resolves the model path
2. resolves the config path
3. resolves the scaler path
4. loads config JSON
5. resolves threshold
6. reads expected `window_size`
7. reads expected `num_channels`
8. loads scaler
9. loads model

### Threshold precedence

Threshold is resolved in this order:

1. explicit constructor argument
2. environment variable `ML_FALL_THRESHOLD`
3. saved threshold from `model_config_colab.json`

This gives flexibility while still defaulting to the training artifact.

## `_load_config`

Loads `model_config_colab.json`.

Why:

- this keeps the inference service aligned with the training notebook
- values like `window_size`, `num_channels`, and `threshold` should come from training, not be re-guessed in code

It verifies that the loaded value is a JSON object before returning it.

## `_load_scaler`

Loads `scaler_colab.pkl`.

Why:

The notebook trained on standardized windows, not raw windows. So inference must use the same scaler or the model sees data in a different numerical distribution.

What it checks:

- file exists
- pickle can be loaded
- loaded object has `transform(...)`

## `_load_model`

Loads the `.keras` model using:

- `custom_objects`
- `compile=False`

and then compiles it with:

- `Adam(1e-3)`
- `focal_loss_fn`
- accuracy, precision, recall, AUC, and PR-AUC metrics

### Why `custom_objects`

Needed for:

- `SelfAttention`
- `focal_loss_fn`

### Why `compile=False`

The file loads the saved model first, then applies the runtime compile configuration explicitly afterward.

## `predict`

This is the main inference method used by the Flask app.

### Input shape it expects

The model expects raw sensor windows shaped as:

- `100` timesteps
- `6` channels

Channel order:

- accelerometer x
- accelerometer y
- accelerometer z
- gyroscope x
- gyroscope y
- gyroscope z

### Flow inside `predict`

1. Read `raw_window` from the payload.
2. Build the scaled tensor using `_build_scaled_window(...)`.
3. Run TensorFlow prediction with `model.predict(...)`.
4. Extract confidence from the model output.
5. Compare confidence against threshold.
6. Return:
   - `fall_detected`
   - `confidence`
   - `threshold`
   - `model_version`
   - `backend`
   - `model_loaded`
   - `warnings`

## `_build_scaled_window`

This method is one of the most important parts of the file.

It:

1. reads `accel_data` and `gyro_data`
2. checks that both are arrays
3. validates the sample count
4. constructs rows in channel order:
   - `ax, ay, az, gx, gy, gz`
5. builds a NumPy array of shape `(100, 6)`
6. applies the saved scaler
7. reshapes to `(1, 100, 6)` for TensorFlow

### Why strict validation is used

The model was trained on fixed windows of `100 x 6`. Feeding other shapes would create invalid or inconsistent inputs for inference.

### Why the scaler is applied here

The training notebook used:

- `StandardScaler`
- fitted on reshaped sensor channels across windows

So the correct inference flow is:

1. build raw `(100, 6)` sequence
2. call `scaler.transform(...)`
3. reshape to `(1, 100, 6)`
4. call `model.predict(...)`

## `_extract_confidence`

Converts model output into a single confidence float.

Why it exists:

- model outputs are still arrays
- the response format needs one clean scalar probability

The method flattens the prediction output, checks that it contains at least one value, and returns the first probability clipped to `[0, 1]`.

## `get_status`

Returns runtime diagnostics for `/health`.

It includes:

- whether the model loaded
- model path
- model version
- backend
- threshold
- config path
- scaler path
- window size
- number of channels
- whether the scaler loaded
- scaler error

### Why this is useful

This gives quick visibility into whether the service is running with the expected artifacts and settings.

## Summary

`predict.py` is the heart of the inference system. It bridges the gap between the training notebook and real-time HTTP inference by making sure:

- the saved model can be deserialized
- the saved scaler is applied
- the correct input shape is used
- the correct threshold is used
- the final prediction result is returned in a clean API-friendly format
