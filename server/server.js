require("dotenv").config();
const crypto = require("crypto");
const express = require("express");
const http = require("http");
const https = require("https");
const socketIO = require("socket.io");
const cors = require("cors");
const os = require("os");
const fs = require("fs");
const path = require("path");
const dgram = require("dgram");

const app = express();
const server = http.createServer(app);
const io = socketIO(server, {
    cors: {
        origin: "*",
        methods: ["GET", "POST"],
    },
    pingInterval: 25000, // ping mỗi 25s (mặc định)
    pingTimeout: 60000, // chờ pong 60s trước khi disconnect
    transports: ["websocket", "polling"],
    allowUpgrades: false, // không upgrade → tránh disconnect/reconnect loop
});
const coap = dgram.createSocket("udp4");

app.use(cors());
app.use(express.json({ limit: "10mb" }));

let connectedClients = 0;
let currentLabel = "1";
let currentSessionId = null;
let currentSessionLabel = null;
let sessionStartTime = null;
let fallMarkers = [];
let lastTriggeredFallAt = 0;
let pendingMlWindow = {
    accel_data: [],
    gyro_data: [],
    esp32_fall_detected: false,
};

const PORT = Number(process.env.PORT || 3000);
const COAP_PORT = Number(process.env.COAP_PORT || 5683);
const ML_SERVICE_URL = process.env.ML_SERVICE_URL || "http://127.0.0.1:5000/predict";
const ML_TIMEOUT_MS = Number(process.env.ML_TIMEOUT_MS || 4000);
const FALL_CONFIDENCE_THRESHOLD = Number(process.env.FALL_CONFIDENCE_THRESHOLD || 0.8);
const FALL_NOTIFICATION_COOLDOWN_MS = Number(process.env.FALL_NOTIFICATION_COOLDOWN_MS || 30000);
const DEVICE_ONLINE_WINDOW_MS = Number(process.env.DEVICE_ONLINE_WINDOW_MS || 15000);
const MAX_FALL_HISTORY = Number(process.env.MAX_FALL_HISTORY || 50);
const ML_WINDOW_SAMPLES = Number(process.env.ML_WINDOW_SAMPLES || 100);
const COAP_TYPE = { CON: 0, NON: 1, ACK: 2 };
const COAP_CODE = {
    POST: 2,
    CREATED: 65,
    CHANGED: 68,
    BAD_REQUEST: 128,
    NOT_FOUND: 132,
    METHOD_NOT_ALLOWED: 133,
    INTERNAL_SERVER_ERROR: 160,
};
const SENSOR_PACKET_VERSION = 2;
const LEGACY_SENSOR_PACKET_VERSION = 1;
const SENSOR_SAMPLE_BYTES = 28;

const DATA_ROOT = path.join(__dirname, "data");
const COLLECTED_DIR = path.join(DATA_ROOT, "collected");
const NORMAL_DIR = path.join(COLLECTED_DIR, "Normal");
const FALL_DIR = path.join(COLLECTED_DIR, "Fall");
const TOKENS_FILE = path.join(DATA_ROOT, "fcm_tokens.json");
const FALL_HISTORY_FILE = path.join(DATA_ROOT, "fall_history.json");
const DEVICE_STATUS_FILE = path.join(DATA_ROOT, "device_status.json");
const FIREBASE_SERVICE_ACCOUNT_FILE = path.join(__dirname, "firebase-service-account.json");

const deviceStatus = {
    connected: false,
    last_seen: null,
    last_batch_size: 0,
    last_bpm: null,
    last_fsm_state: null,
    last_features: null,
    last_prediction: null,
    last_fall_at: null,
    updated_at: null,
};

ensureDirectory(DATA_ROOT);
ensureDirectory(COLLECTED_DIR);
ensureDirectory(NORMAL_DIR);
ensureDirectory(FALL_DIR);
ensureJsonFile(TOKENS_FILE, []);
ensureJsonFile(FALL_HISTORY_FILE, []);
ensureJsonFile(DEVICE_STATUS_FILE, deviceStatus);

const firebaseState = initializeFirebase();

// Realtime dashboard connection lifecycle.
io.on("connection", (socket) => {
    connectedClients += 1;
    console.log(`Client connected. Total: ${connectedClients}`);

    socket.on("disconnect", () => {
        connectedClients = Math.max(0, connectedClients - 1);
        console.log(`Client disconnected. Total: ${connectedClients}`);
    });
});

// Receive and broadcast single sensor samples.
app.post("/api/sensor", (req, res) => {
    const sensorData = req.body || {};
    if (!sensorData.timestamp) {
        sensorData.timestamp = new Date().toISOString();
    }

    io.emit("sensorData", sensorData);
    res.json({
        success: true,
        message: "Data received and broadcasted",
        clients: connectedClients,
    });
});

// Shared processor used by HTTP and CoAP sensor-batch ingestion.
async function processSensorBatch(batchData, source = "http") {
    if (!Array.isArray(batchData.accel_data) || !Array.isArray(batchData.gyro_data)) {
        logJson(`[/${source}/sensor-batch] invalid_payload`, {
            accel_is_array: Array.isArray(batchData.accel_data),
            gyro_is_array: Array.isArray(batchData.gyro_data),
        });
        return {
            statusCode: 400,
            payload: {
                success: false,
                error: "Missing accel_data or gyro_data arrays",
            },
        };
    }

    if (batchData.accel_data.length !== batchData.gyro_data.length) {
        return {
            statusCode: 400,
            payload: {
                success: false,
                error: "accel_data and gyro_data must have the same length",
            },
        };
    }

    if (batchData.accel_data.length === 0) {
        return {
            statusCode: 400,
            payload: {
                success: false,
                error: "Empty sensor batch",
            },
        };
    }

    logJson(`[/${source}/sensor-batch] request`, summarizeBatchData(batchData));

    if (!currentSessionId) {
        startNewSession();
    }

    updateDeviceStatus(batchData);

    try {
        saveToCSV(batchData);
    } catch (error) {
        console.error("Error saving CSV:", error);
    }

    const esp32FallDetected = Boolean(batchData.fall_detected);
    if (esp32FallDetected) {
        const marker = recordFallMarker("esp32_fsm");
        io.emit("fallDetected", {
            session_id: currentSessionId,
            timestamp: marker.timestamp,
            fsm_state: batchData.fsm_state ?? null,
            features: batchData.features || {},
            source: "esp32_fsm",
            elapsed_seconds: marker.elapsed_seconds,
        });
    }

    const mlInputBatch = enqueueAndMaybeBuildMlBatch(batchData);
    if (!mlInputBatch) {
        const responsePayload = {
            success: true,
            message: "Batch received. Waiting for 100 samples before ML inference",
            session_id: currentSessionId,
            label: currentLabel,
            clients: connectedClients,
            ml_processed: false,
            ml_fall_detected: false,
            confidence: 0,
            model_version: null,
            ml_backend: null,
            fall_event_id: null,
            buffered_samples: pendingMlWindow.accel_data.length,
            required_samples: ML_WINDOW_SAMPLES,
        };

        io.emit("sensorBatch", {
            ...batchData,
            ml_result: {
                available: false,
                fall_detected: false,
                confidence: 0,
                reason: "waiting_for_full_window",
            },
            session_id: currentSessionId,
            source,
            buffered_samples: pendingMlWindow.accel_data.length,
        });

        logJson(`[/${source}/sensor-batch] response`, responsePayload);
        return { statusCode: 200, payload: responsePayload };
    }

    const mlResult = await predictFall(mlInputBatch);
    logJson(`[/${source}/sensor-batch] ml_result`, mlResult);
    deviceStatus.last_prediction = mlResult;
    writeJsonFile(DEVICE_STATUS_FILE, deviceStatus);

    try {
        saveFeatures(mlInputBatch, mlResult);
    } catch (error) {
        console.error("Error saving features:", error);
    }

    const triggeredFallEvent = await maybeHandleFallDetection(
        mlInputBatch,
        mlResult,
        Boolean(mlInputBatch.fall_detected),
    );

    const batchForBroadcast = {
        ...mlInputBatch,
        ml_result: mlResult,
        session_id: currentSessionId,
        source,
        buffered_samples: pendingMlWindow.accel_data.length,
    };
    io.emit("sensorBatch", batchForBroadcast);

    const responsePayload = {
        success: true,
        message: "Batch data received and processed",
        session_id: currentSessionId,
        label: currentLabel,
        clients: connectedClients,
        ml_fall_detected: Boolean(mlResult.fall_detected),
        confidence: Number(mlResult.confidence || 0),
        model_version: mlResult.model_version || null,
        ml_backend: mlResult.backend || null,
        fall_event_id: triggeredFallEvent ? triggeredFallEvent.id : null,
        ml_processed: true,
        buffered_samples: pendingMlWindow.accel_data.length,
        required_samples: ML_WINDOW_SAMPLES,
    };

    logJson(`[/${source}/sensor-batch] response`, responsePayload);
    return { statusCode: 200, payload: responsePayload };
}

function enqueueAndMaybeBuildMlBatch(batchData) {
    pendingMlWindow.accel_data.push(...batchData.accel_data);
    pendingMlWindow.gyro_data.push(...batchData.gyro_data);
    pendingMlWindow.esp32_fall_detected = pendingMlWindow.esp32_fall_detected || Boolean(batchData.fall_detected);

    if (pendingMlWindow.accel_data.length < ML_WINDOW_SAMPLES || pendingMlWindow.gyro_data.length < ML_WINDOW_SAMPLES) {
        return null;
    }

    const accelWindow = pendingMlWindow.accel_data.slice(0, ML_WINDOW_SAMPLES);
    const gyroWindow = pendingMlWindow.gyro_data.slice(0, ML_WINDOW_SAMPLES);

    pendingMlWindow.accel_data = pendingMlWindow.accel_data.slice(ML_WINDOW_SAMPLES);
    pendingMlWindow.gyro_data = pendingMlWindow.gyro_data.slice(ML_WINDOW_SAMPLES);

    const mlBatch = {
        ...batchData,
        accel_data: accelWindow,
        gyro_data: gyroWindow,
        window_size: ML_WINDOW_SAMPLES,
        fall_detected: pendingMlWindow.esp32_fall_detected,
    };

    pendingMlWindow.esp32_fall_detected = false;
    return mlBatch;
}

// Receive one full sensor window, run ML, save data, and emit updates.
app.post("/api/sensor-batch", async (req, res) => {
    const result = await processSensorBatch(req.body || {}, "http");
    return res.status(result.statusCode).json(result.payload);
});

// Force-start a new labeled recording session.
app.post("/api/session/start", (req, res) => {
    startNewSession(currentLabel);
    res.json({
        success: true,
        session_id: currentSessionId,
        start_time: sessionStartTime,
        label: currentLabel,
    });
});

// Read the currently active recording label.
app.get("/api/label", (req, res) => {
    res.json({ label: currentLabel, text: currentLabel === "1" ? "FALL" : "NORMAL" });
});

// Change the active recording label for future sessions.
app.post("/api/label", (req, res) => {
    const { label } = req.body || {};
    if (label === "0" || label === "1") {
        currentLabel = label;
        const text = label === "1" ? "FALL" : "NORMAL";
        io.emit("labelChanged", { label: currentLabel, text });
        return res.json({ success: true, label: currentLabel, text });
    }

    return res.status(400).json({ success: false, error: 'Invalid label. Use "0" or "1"' });
});

// Stop the current recording session and flush markers to disk.
app.post("/api/session/stop", (req, res) => {
    stopCurrentSession();
    res.json({ success: true });
});

// Close the current session and immediately start another one.
app.post("/api/session/new", (req, res) => {
    const previousSessionId = currentSessionId;
    stopCurrentSession();

    const requestedLabel = req.body && typeof req.body.label !== "undefined" ? String(req.body.label) : null;
    const labelToUse = requestedLabel === "0" || requestedLabel === "1" ? requestedLabel : currentLabel;
    startNewSession(labelToUse);

    res.json({
        success: true,
        previous_session_id: previousSessionId,
        session_id: currentSessionId,
        start_time: sessionStartTime,
        label: labelToUse,
    });
});

// Add a manual fall marker to the active session.
app.post("/api/mark-fall", (req, res) => {
    if (!currentSessionId) {
        return res.status(400).json({ success: false, error: "No active session" });
    }

    const marker = recordFallMarker("manual");
    return res.json({
        success: true,
        elapsed_seconds: marker.elapsed_seconds,
        marker_count: fallMarkers.length,
    });
});

// Return how many fall and normal sessions have been collected.
app.get("/api/sessions/stats", (req, res) => {
    const countDirs = (dir) => {
        if (!fs.existsSync(dir)) {
            return 0;
        }
        return fs.readdirSync(dir).filter((entry) => {
            try {
                return fs.statSync(path.join(dir, entry)).isDirectory();
            } catch {
                return false;
            }
        }).length;
    };

    res.json({ fall: countDirs(FALL_DIR), normal: countDirs(NORMAL_DIR) });
});

// Register or refresh a mobile device token for notifications.
app.post("/api/register-device", (req, res) => {
    const token = sanitizeToken(req.body?.fcm_token || req.body?.token);
    if (!token) {
        return res.status(400).json({ success: false, error: "Missing fcm_token" });
    }

    const tokens = readJsonFile(TOKENS_FILE, []);
    const now = new Date().toISOString();
    const existing = tokens.find((entry) => entry.token === token);

    if (existing) {
        existing.updated_at = now;
        existing.platform = req.body?.platform || existing.platform || "unknown";
        existing.role = req.body?.role || existing.role || "member";
        existing.device_name = req.body?.device_name || existing.device_name || null;
    } else {
        tokens.push({
            token,
            platform: req.body?.platform || "unknown",
            role: req.body?.role || "member",
            device_name: req.body?.device_name || null,
            registered_at: now,
            updated_at: now,
        });
    }

    writeJsonFile(TOKENS_FILE, tokens);

    return res.json({
        success: true,
        registered_tokens: tokens.length,
        firebase_enabled: Boolean(firebaseState.messaging),
    });
});

// Read recent fall events from persistent history.
app.get("/api/fall-history", (req, res) => {
    const limit = Math.max(1, Math.min(Number(req.query.limit || MAX_FALL_HISTORY), MAX_FALL_HISTORY));
    const fallHistory = readJsonFile(FALL_HISTORY_FILE, []);
    res.json({
        success: true,
        count: Math.min(limit, fallHistory.length),
        falls: fallHistory.slice(0, limit),
    });
});

// Return the latest computed device connectivity and telemetry status.
app.get("/api/device/status", (req, res) => {
    res.json({
        success: true,
        device: getComputedDeviceStatus(),
    });
});

// Update how a user responded to a fall alert.
app.post("/api/fall-response", (req, res) => {
    const action = normalizeFallAction(req.body?.action);
    if (!action) {
        return res.status(400).json({
            success: false,
            error: 'Invalid action. Use "dismiss", "confirm", "confirmed", or "escalated"',
        });
    }

    const fallHistory = readJsonFile(FALL_HISTORY_FILE, []);
    const targetId = req.body?.event_id || req.body?.fall_id || getLatestPendingFallId(fallHistory);
    if (!targetId) {
        return res.status(404).json({ success: false, error: "No fall event available to update" });
    }

    const event = fallHistory.find((entry) => entry.id === targetId);
    if (!event) {
        return res.status(404).json({ success: false, error: `Fall event not found: ${targetId}` });
    }

    event.status = action;
    event.responded_at = new Date().toISOString();
    event.response_note = req.body?.note || null;
    event.response_source = req.body?.source || "mobile_app";

    writeJsonFile(FALL_HISTORY_FILE, fallHistory);
    io.emit("fallResponseUpdated", event);

    return res.json({ success: true, event });
});

// Return a compact health snapshot for the full server stack.
app.get("/api/status", (req, res) => {
    res.json({
        status: "running",
        connectedClients,
        label: currentLabel,
        session_id: currentSessionId,
        timestamp: new Date().toISOString(),
        ml_service_url: ML_SERVICE_URL,
        fall_confidence_threshold: FALL_CONFIDENCE_THRESHOLD,
        firebase: {
            configured: firebaseState.configured,
            messaging_enabled: Boolean(firebaseState.messaging),
            error: firebaseState.error,
        },
        device_status: getComputedDeviceStatus(),
    });
});

// --- Session and storage helpers -------------------------------------------------

// Recover the label prefix from a saved session id.
function inferLabelFromSessionId(sessionId) {
    if (typeof sessionId !== "string") {
        return null;
    }
    const match = sessionId.match(/^label([01])_/);
    return match ? match[1] : null;
}

// Resolve the directory where a session's files should live.
function getSessionDir(sessionId = currentSessionId, label = currentSessionLabel) {
    const effectiveLabel = label ?? inferLabelFromSessionId(sessionId) ?? currentLabel;
    const baseDir = effectiveLabel === "1" ? FALL_DIR : NORMAL_DIR;
    return path.join(baseDir, sessionId);
}

// Create a fresh session folder and seed its metadata files.
function startNewSession(label = currentLabel) {
    const timestamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, -5);
    currentSessionId = `label${label}_${timestamp}`;
    currentSessionLabel = String(label);
    sessionStartTime = new Date().toISOString();
    fallMarkers = [];
    pendingMlWindow = {
        accel_data: [],
        gyro_data: [],
        esp32_fall_detected: false,
    };

    const sessionDir = getSessionDir(currentSessionId, currentSessionLabel);
    ensureDirectory(sessionDir);

    fs.writeFileSync(path.join(sessionDir, "accel.csv"), "accel_time_list,accel_x_list,accel_y_list,accel_z_list\n");
    fs.writeFileSync(path.join(sessionDir, "gyro.csv"), "gyro_time_list,gyro_x_list,gyro_y_list,gyro_z_list\n");
    fs.writeFileSync(path.join(sessionDir, "label.txt"), label);

    const metadata = {
        session_id: currentSessionId,
        start_time: sessionStartTime,
        device: "ESP32-C3 + MPU6050 + MAX30102",
        sample_rate: 50,
        window_size: 100,
        label: Number(label),
    };
    fs.writeFileSync(path.join(sessionDir, "metadata.json"), JSON.stringify(metadata, null, 2));

    console.log(`New session started: ${currentSessionId} [Label: ${label}]`);
}

// Close the active session and persist any pending fall markers.
function stopCurrentSession() {
    if (currentSessionId) {
        const sessionDir = getSessionDir();
        fs.writeFileSync(path.join(sessionDir, "fall_markers.json"), JSON.stringify(fallMarkers, null, 2));
        console.log(`Session ${currentSessionId} stopped. Saved ${fallMarkers.length} fall markers.`);
    }

    currentSessionId = null;
    currentSessionLabel = null;
    sessionStartTime = null;
    fallMarkers = [];
    pendingMlWindow = {
        accel_data: [],
        gyro_data: [],
        esp32_fall_detected: false,
    };
}

// Append raw accelerometer and gyroscope samples to the session CSV files.
function saveToCSV(batchData) {
    if (!currentSessionId) {
        return;
    }

    const sessionDir = getSessionDir();
    const accelPath = path.join(sessionDir, "accel.csv");
    const gyroPath = path.join(sessionDir, "gyro.csv");

    const accelLines = batchData.accel_data.map((sample) => `${sample.t},${sample.x},${sample.y},${sample.z}`).join("\n");
    if (accelLines) {
        fs.appendFileSync(accelPath, `${accelLines}\n`);
    }

    const gyroLines = batchData.gyro_data.map((sample) => `${sample.t},${sample.x},${sample.y},${sample.z}`).join("\n");
    if (gyroLines) {
        fs.appendFileSync(gyroPath, `${gyroLines}\n`);
    }
}

// Append one feature row enriched with ML output to the session features CSV.
function saveFeatures(batchData, mlResult) {
    if (!currentSessionId) {
        return;
    }

    const sessionDir = getSessionDir();
    const featuresPath = path.join(sessionDir, "features.csv");
    if (!fs.existsSync(featuresPath)) {
        fs.writeFileSync(
            featuresPath,
            "window_time,magnitude_avg,sma,max_accel,max_gyro,std_accel,jerk_peak,bpm,fsm_state,fall_detected,ml_fall_detected,ml_confidence,ml_backend\n",
        );
    }

    const features = batchData.features || {};
    const elapsed = sessionStartTime ? (Date.now() - new Date(sessionStartTime).getTime()) / 1000 : 0;
    const line = [
        elapsed.toFixed(2),
        safeNumber(features.magnitude_avg),
        safeNumber(features.sma),
        safeNumber(features.max_accel),
        safeNumber(features.max_gyro),
        safeNumber(features.std_accel),
        safeNumber(features.jerk_peak),
        safeNumber(batchData.bpm),
        safeNumber(batchData.fsm_state),
        batchData.fall_detected ? 1 : 0,
        mlResult.fall_detected ? 1 : 0,
        safeNumber(mlResult.confidence),
        mlResult.backend || "unknown",
    ].join(",");

    fs.appendFileSync(featuresPath, `${line}\n`);
}

// Add a timestamped fall marker to the in-memory session list.
function recordFallMarker(source) {
    const marker = {
        timestamp: new Date().toISOString(),
        elapsed_seconds: sessionStartTime ? (Date.now() - new Date(sessionStartTime).getTime()) / 1000 : 0,
        source,
    };
    fallMarkers.push(marker);
    return marker;
}

// Persist the latest telemetry snapshot for device health APIs.
function updateDeviceStatus(batchData) {
    const now = new Date().toISOString();
    deviceStatus.connected = true;
    deviceStatus.last_seen = now;
    deviceStatus.last_batch_size = batchData.window_size || batchData.accel_data.length || 0;
    deviceStatus.last_bpm = safeNullableNumber(batchData.bpm);
    deviceStatus.last_fsm_state = batchData.fsm_state ?? null;
    deviceStatus.last_features = batchData.features || null;
    deviceStatus.updated_at = now;
    writeJsonFile(DEVICE_STATUS_FILE, deviceStatus);
}

// Compute whether the device is still considered online.
function getComputedDeviceStatus() {
    const stored = readJsonFile(DEVICE_STATUS_FILE, deviceStatus);
    const lastSeen = stored.last_seen ? new Date(stored.last_seen).getTime() : 0;
    return {
        ...stored,
        connected: Boolean(lastSeen && Date.now() - lastSeen <= DEVICE_ONLINE_WINDOW_MS),
        online_window_ms: DEVICE_ONLINE_WINDOW_MS,
    };
}

// --- ML and fall-detection helpers ----------------------------------------------

// Send one batch to the Python ML service and normalize its response.
async function predictFall(batchData) {
    const payload = {
        features: batchData.features || {},
        raw_window: {
            accel_data: batchData.accel_data || [],
            gyro_data: batchData.gyro_data || [],
        },
    };

    try {
        const response = await postJson(ML_SERVICE_URL, payload, ML_TIMEOUT_MS);
        return {
            fall_detected: Boolean(response.fall_detected),
            confidence: safeNumber(response.confidence),
            model_version: response.model_version || "unknown",
            backend: response.backend || "unknown",
            threshold: safeNumber(response.threshold),
            model_loaded: Boolean(response.model_loaded),
            warnings: Array.isArray(response.warnings) ? response.warnings : [],
            available: true,
        };
    } catch (error) {
        console.warn(`ML service unavailable: ${error.message}`);
        return {
            fall_detected: false,
            confidence: 0,
            model_version: "unavailable",
            backend: "unavailable",
            threshold: FALL_CONFIDENCE_THRESHOLD,
            model_loaded: false,
            warnings: [error.message],
            available: false,
        };
    }
}

// Decide whether an ML prediction should become a real fall event.
async function maybeHandleFallDetection(batchData, mlResult, esp32FallDetected) {
    const qualifies = Boolean(mlResult.fall_detected) && safeNumber(mlResult.confidence) >= FALL_CONFIDENCE_THRESHOLD;
    if (!qualifies) {
        return null;
    }

    const now = Date.now();
    if (now - lastTriggeredFallAt < FALL_NOTIFICATION_COOLDOWN_MS) {
        return null;
    }

    lastTriggeredFallAt = now;
    deviceStatus.last_prediction = mlResult;
    deviceStatus.last_fall_at = new Date(now).toISOString();
    writeJsonFile(DEVICE_STATUS_FILE, deviceStatus);

    const fallEvent = {
        id: crypto.randomUUID(),
        timestamp: new Date(now).toISOString(),
        session_id: currentSessionId,
        confidence: safeNumber(mlResult.confidence),
        threshold: FALL_CONFIDENCE_THRESHOLD,
        status: "pending",
        source: mlResult.backend || "ml_service",
        model_version: mlResult.model_version || null,
        esp32_fall_detected: esp32FallDetected,
        features: batchData.features || {},
        fsm_state: batchData.fsm_state ?? null,
        bpm: safeNullableNumber(batchData.bpm),
        notification: {
            requested_at: new Date(now).toISOString(),
        },
    };

    const notificationResult = await sendFallNotifications(fallEvent);
    fallEvent.notification = {
        ...fallEvent.notification,
        ...notificationResult,
    };

    appendFallHistory(fallEvent);
    io.emit("fallDetected", fallEvent);
    return fallEvent;
}

// Send push notifications to all registered mobile devices.
async function sendFallNotifications(fallEvent) {
    const tokens = readJsonFile(TOKENS_FILE, []);
    const tokenValues = tokens.map((entry) => entry.token).filter(Boolean);

    if (!firebaseState.messaging) {
        return {
            sent: false,
            reason: firebaseState.error || "Firebase Admin SDK is not configured",
            token_count: tokenValues.length,
        };
    }

    if (tokenValues.length === 0) {
        return {
            sent: false,
            reason: "No registered FCM tokens",
            token_count: 0,
        };
    }

    const message = {
        tokens: tokenValues,
        notification: {
            title: "Fall detected",
            body: `Potential fall detected. Confidence ${Math.round(fallEvent.confidence * 100)}%.`,
        },
        data: {
            eventId: String(fallEvent.id),
            confidence: String(fallEvent.confidence),
            timestamp: String(fallEvent.timestamp),
            status: String(fallEvent.status),
        },
        android: {
            priority: "high",
            notification: {
                sound: "default",
                channelId: "fall_alerts",
            },
        },
        apns: {
            headers: {
                "apns-priority": "10",
            },
            payload: {
                aps: {
                    sound: "default",
                },
            },
        },
    };

    try {
        const result = await firebaseState.messaging.sendEachForMulticast(message);
        const invalidTokens = [];
        result.responses.forEach((response, index) => {
            const errorCode = response.error?.code || "";
            if (errorCode === "messaging/invalid-registration-token" || errorCode === "messaging/registration-token-not-registered") {
                invalidTokens.push(tokenValues[index]);
            }
        });

        if (invalidTokens.length > 0) {
            const filtered = tokens.filter((entry) => !invalidTokens.includes(entry.token));
            writeJsonFile(TOKENS_FILE, filtered);
        }

        return {
            sent: result.successCount > 0,
            success_count: result.successCount,
            failure_count: result.failureCount,
            token_count: tokenValues.length,
            removed_invalid_tokens: invalidTokens.length,
        };
    } catch (error) {
        return {
            sent: false,
            reason: error.message,
            token_count: tokenValues.length,
        };
    }
}

// Save a new fall event at the front of the fall history list.
function appendFallHistory(fallEvent) {
    const history = readJsonFile(FALL_HISTORY_FILE, []);
    history.unshift(fallEvent);
    writeJsonFile(FALL_HISTORY_FILE, history.slice(0, MAX_FALL_HISTORY));
}

// Normalize allowed fall-response actions to stored status values.
function normalizeFallAction(action) {
    if (typeof action !== "string") {
        return null;
    }

    const normalized = action.trim().toLowerCase();
    if (normalized === "dismiss") {
        return "dismissed";
    }
    if (normalized === "confirm" || normalized === "confirmed") {
        return "confirmed";
    }
    if (normalized === "escalated") {
        return "escalated";
    }
    return null;
}

// Pick the newest unresolved fall event when no explicit id is provided.
function getLatestPendingFallId(history) {
    const latestPending = history.find((entry) => entry.status === "pending");
    return latestPending ? latestPending.id : null;
}

// Initialize Firebase Admin if service-account credentials are available.
function initializeFirebase() {
    const state = {
        configured: false,
        messaging: null,
        error: null,
    };

    let firebaseAdmin;
    try {
        firebaseAdmin = require("firebase-admin");
    } catch (error) {
        state.error = "firebase-admin is not installed";
        return state;
    }

    if (!fs.existsSync(FIREBASE_SERVICE_ACCOUNT_FILE)) {
        state.error = `Missing Firebase credentials at ${FIREBASE_SERVICE_ACCOUNT_FILE}`;
        return state;
    }

    try {
        const serviceAccount = JSON.parse(fs.readFileSync(FIREBASE_SERVICE_ACCOUNT_FILE, "utf8"));
        firebaseAdmin.initializeApp({
            credential: firebaseAdmin.credential.cert(serviceAccount),
        });
        state.messaging = firebaseAdmin.messaging();
        state.configured = true;
        return state;
    } catch (error) {
        state.error = error.message;
        return state;
    }
}

// Perform a JSON HTTP POST with timeout handling.
function postJson(urlString, payload, timeoutMs) {
    return new Promise((resolve, reject) => {
        let url;
        try {
            url = new URL(urlString);
        } catch (error) {
            reject(error);
            return;
        }

        const transport = url.protocol === "https:" ? https : http;
        const body = JSON.stringify(payload);

        const req = transport.request(
            {
                hostname: url.hostname,
                port: url.port || (url.protocol === "https:" ? 443 : 80),
                path: `${url.pathname}${url.search}`,
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "Content-Length": Buffer.byteLength(body),
                },
                timeout: timeoutMs,
            },
            (response) => {
                let raw = "";
                response.setEncoding("utf8");
                response.on("data", (chunk) => {
                    raw += chunk;
                });
                response.on("end", () => {
                    if (response.statusCode < 200 || response.statusCode >= 300) {
                        return reject(new Error(`HTTP ${response.statusCode}: ${raw}`));
                    }

                    try {
                        resolve(raw ? JSON.parse(raw) : {});
                    } catch (error) {
                        reject(new Error(`Invalid JSON from ML service: ${error.message}`));
                    }
                });
            },
        );

        req.on("timeout", () => {
            req.destroy(new Error(`Request timed out after ${timeoutMs}ms`));
        });
        req.on("error", reject);
        req.write(body);
        req.end();
    });
}

// --- CoAP transport helpers -----------------------------------------------------

function readCoapExtendedValue(nibble, buffer, offset) {
    if (nibble < 13) {
        return { value: nibble, offset };
    }

    if (nibble === 13) {
        if (offset >= buffer.length) {
            throw new Error("Truncated CoAP option");
        }
        return { value: buffer[offset] + 13, offset: offset + 1 };
    }

    if (nibble === 14) {
        if (offset + 1 >= buffer.length) {
            throw new Error("Truncated CoAP option");
        }
        return { value: buffer.readUInt16BE(offset) + 269, offset: offset + 2 };
    }

    throw new Error("Unsupported CoAP option value");
}

function parseCoapRequest(message) {
    if (!Buffer.isBuffer(message) || message.length < 4) {
        throw new Error("Invalid CoAP packet");
    }

    const version = message[0] >> 6;
    const type = (message[0] >> 4) & 0x03;
    const tokenLength = message[0] & 0x0f;
    if (version !== 1) {
        throw new Error(`Unsupported CoAP version: ${version}`);
    }

    if (tokenLength > 8 || message.length < 4 + tokenLength) {
        throw new Error("Invalid CoAP token length");
    }

    const code = message[1];
    const messageId = message.readUInt16BE(2);
    const token = message.subarray(4, 4 + tokenLength);

    let offset = 4 + tokenLength;
    let currentOptionNumber = 0;
    const uriPath = [];

    while (offset < message.length) {
        if (message[offset] === 0xff) {
            offset += 1;
            break;
        }

        const optionHeader = message[offset++];
        const deltaInfo = readCoapExtendedValue(optionHeader >> 4, message, offset);
        const lengthInfo = readCoapExtendedValue(optionHeader & 0x0f, message, deltaInfo.offset);
        const optionNumber = currentOptionNumber + deltaInfo.value;

        if (lengthInfo.offset + lengthInfo.value > message.length) {
            throw new Error("Truncated CoAP option payload");
        }

        const optionValue = message.subarray(lengthInfo.offset, lengthInfo.offset + lengthInfo.value);
        offset = lengthInfo.offset + lengthInfo.value;
        currentOptionNumber = optionNumber;

        if (optionNumber === 11) {
            uriPath.push(optionValue.toString("utf8"));
        }
    }

    return {
        type,
        code,
        messageId,
        token,
        path: `/${uriPath.join("/")}`,
        payload: offset <= message.length ? message.subarray(offset) : Buffer.alloc(0),
    };
}

function buildCoapResponse(request, code, payloadBuffer = null) {
    const payload = payloadBuffer && payloadBuffer.length ? payloadBuffer : null;
    const responseType = request.type === COAP_TYPE.CON ? COAP_TYPE.ACK : COAP_TYPE.NON;
    const responseMessageId = responseType === COAP_TYPE.ACK ? request.messageId : (request.messageId + 1) & 0xffff;
    const tokenLength = request.token.length;
    const totalLength = 4 + tokenLength + (payload ? payload.length + 1 : 0);
    const response = Buffer.alloc(totalLength);

    response[0] = (1 << 6) | (responseType << 4) | tokenLength;
    response[1] = code;
    response.writeUInt16BE(responseMessageId, 2);
    request.token.copy(response, 4);

    if (payload) {
        const markerOffset = 4 + tokenLength;
        response[markerOffset] = 0xff;
        payload.copy(response, markerOffset + 1);
    }

    return response;
}

function sendCoapResponse(request, rinfo, code, payloadBuffer = null) {
    coap.send(buildCoapResponse(request, code, payloadBuffer), rinfo.port, rinfo.address);
}

function parseRequestedLabel(payload) {
    if (!payload || payload.length === 0) {
        return null;
    }

    const text = payload.toString("utf8").trim();
    if (text === "0" || text === "1") {
        return text;
    }

    try {
        const parsed = JSON.parse(text);
        const label = String(parsed.label);
        return label === "0" || label === "1" ? label : null;
    } catch {
        return null;
    }
}

function decodeSensorBatchPayload(payload) {
    if (!Buffer.isBuffer(payload) || payload.length < 36) {
        throw new Error("Sensor payload too short");
    }

    let offset = 0;
    const version = payload.readUInt8(offset++);

    if (version === LEGACY_SENSOR_PACKET_VERSION) {
        const flags = payload.readUInt8(offset++);
        const fsmState = payload.readUInt8(offset++);
        offset += 1;

        const windowSize = payload.readUInt16LE(offset);
        offset += 2;
        const sampleIntervalMs = payload.readUInt16LE(offset);
        offset += 2;
        const windowStartMs = payload.readUInt32LE(offset);
        offset += 4;
        const bpm = payload.readUInt16LE(offset);
        offset += 2;
        const irRaw = payload.readUInt32LE(offset);
        offset += 4;

        const features = {
            magnitude_avg: payload.readFloatLE(offset),
            sma: payload.readFloatLE(offset + 4),
            max_accel: payload.readFloatLE(offset + 8),
            max_gyro: payload.readFloatLE(offset + 12),
            std_accel: payload.readFloatLE(offset + 16),
            jerk_peak: payload.readFloatLE(offset + 20),
        };
        offset += 24;

        const expectedLength = offset + windowSize * 12;
        if (payload.length !== expectedLength) {
            throw new Error(`Unexpected legacy payload length: got ${payload.length}, expected ${expectedLength}`);
        }

        const accelData = [];
        const gyroData = [];
        for (let i = 0; i < windowSize; i += 1) {
            const ax = payload.readInt16LE(offset) / 400;
            offset += 2;
            const ay = payload.readInt16LE(offset) / 400;
            offset += 2;
            const az = payload.readInt16LE(offset) / 400;
            offset += 2;
            const gx = payload.readInt16LE(offset) / 1000;
            offset += 2;
            const gy = payload.readInt16LE(offset) / 1000;
            offset += 2;
            const gz = payload.readInt16LE(offset) / 1000;
            offset += 2;
            const timestamp = Math.round(windowStartMs + i * sampleIntervalMs) / 1000;
            accelData.push({ t: timestamp, x: ax, y: ay, z: az });
            gyroData.push({ t: timestamp, x: gx, y: gy, z: gz });
        }

        return {
            status: "active",
            bpm,
            ir_raw: irRaw,
            window_size: windowSize,
            sample_rate: Math.round(1000 / sampleIntervalMs),
            sample_interval_ms: sampleIntervalMs,
            window_start_ms: windowStartMs,
            fsm_state: fsmState,
            fall_detected: Boolean(flags & 0x01),
            chunk_index: 0,
            total_chunks: 1,
            features,
            accel_data: accelData,
            gyro_data: gyroData,
        };
    }

    if (version !== SENSOR_PACKET_VERSION) {
        throw new Error(`Unsupported sensor packet version: ${version}`);
    }

    const flags = payload.readUInt8(offset++);
    const fsmState = payload.readUInt8(offset++);
    offset += 1;

    const windowSize = payload.readUInt16LE(offset);
    offset += 2;
    const chunkIndex = payload.readUInt8(offset++);
    const totalChunks = payload.readUInt8(offset++);
    const chunkSamples = payload.readUInt8(offset++);
    offset += 1;
    const sampleIntervalMs = payload.readUInt16LE(offset);
    offset += 2;
    const windowStartMs = payload.readUInt32LE(offset);
    offset += 4;
    const bpm = payload.readUInt16LE(offset);
    offset += 2;
    const irRaw = payload.readUInt32LE(offset);
    offset += 4;

    const features = {
        magnitude_avg: payload.readFloatLE(offset),
        sma: payload.readFloatLE(offset + 4),
        max_accel: payload.readFloatLE(offset + 8),
        max_gyro: payload.readFloatLE(offset + 12),
        std_accel: payload.readFloatLE(offset + 16),
        jerk_peak: payload.readFloatLE(offset + 20),
    };
    offset += 24;

    if (windowSize <= 0 || windowSize > 200) {
        throw new Error(`Invalid window size: ${windowSize}`);
    }
    if (sampleIntervalMs <= 0 || sampleIntervalMs > 1000) {
        throw new Error(`Invalid sample interval: ${sampleIntervalMs}`);
    }
    if (totalChunks <= 0 || chunkIndex >= totalChunks) {
        throw new Error(`Invalid chunk info: ${chunkIndex}/${totalChunks}`);
    }
    if (chunkSamples <= 0 || chunkSamples > windowSize) {
        throw new Error(`Invalid chunk sample count: ${chunkSamples}`);
    }

    const expectedLength = offset + chunkSamples * SENSOR_SAMPLE_BYTES;
    if (payload.length !== expectedLength) {
        throw new Error(`Unexpected payload length: got ${payload.length}, expected ${expectedLength}`);
    }

    const accelData = [];
    const gyroData = [];
    for (let i = 0; i < chunkSamples; i += 1) {
        const timestamp = payload.readFloatLE(offset);
        offset += 4;
        const ax = payload.readFloatLE(offset);
        offset += 4;
        const ay = payload.readFloatLE(offset);
        offset += 4;
        const az = payload.readFloatLE(offset);
        offset += 4;
        const gx = payload.readFloatLE(offset);
        offset += 4;
        const gy = payload.readFloatLE(offset);
        offset += 4;
        const gz = payload.readFloatLE(offset);
        offset += 4;

        accelData.push({ t: timestamp, x: ax, y: ay, z: az });
        gyroData.push({ t: timestamp, x: gx, y: gy, z: gz });
    }

    return {
        status: "active",
        bpm,
        ir_raw: irRaw,
        window_size: windowSize,
        sample_rate: Math.round(1000 / sampleIntervalMs),
        sample_interval_ms: sampleIntervalMs,
        window_start_ms: windowStartMs,
        fsm_state: fsmState,
        fall_detected: Boolean(flags & 0x01),
        chunk_index: chunkIndex,
        total_chunks: totalChunks,
        features,
        accel_data: accelData,
        gyro_data: gyroData,
    };
}

// --- Generic filesystem and data helpers ----------------------------------------

// Ensure a directory exists before writing files into it.
function ensureDirectory(dirPath) {
    if (!fs.existsSync(dirPath)) {
        fs.mkdirSync(dirPath, { recursive: true });
    }
}

// Create a JSON file with fallback content if it does not exist yet.
function ensureJsonFile(filePath, fallback) {
    if (!fs.existsSync(filePath)) {
        fs.writeFileSync(filePath, JSON.stringify(fallback, null, 2));
    }
}

// Read a JSON file and fall back safely if parsing fails.
function readJsonFile(filePath, fallback) {
    try {
        return JSON.parse(fs.readFileSync(filePath, "utf8"));
    } catch {
        return fallback;
    }
}

// Write JSON to disk with pretty formatting.
function writeJsonFile(filePath, data) {
    fs.writeFileSync(filePath, JSON.stringify(data, null, 2));
}

// Validate that a device token looks usable before storing it.
function sanitizeToken(token) {
    if (typeof token !== "string") {
        return null;
    }
    const trimmed = token.trim();
    return trimmed.length >= 20 ? trimmed : null;
}

// Convert arbitrary values into finite numbers with a fallback.
function safeNumber(value, fallback = 0) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : fallback;
}

// Convert arbitrary values into finite numbers or null.
function safeNullableNumber(value) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
}

// Build a compact log summary for a sensor batch.
function summarizeBatchData(batchData) {
    const features = batchData.features || {};
    return {
        accel_samples: Array.isArray(batchData.accel_data) ? batchData.accel_data.length : 0,
        gyro_samples: Array.isArray(batchData.gyro_data) ? batchData.gyro_data.length : 0,
        window_size: batchData.window_size ?? null,
        bpm: safeNullableNumber(batchData.bpm),
        fsm_state: batchData.fsm_state ?? null,
        fall_detected: Boolean(batchData.fall_detected),
        features: {
            magnitude_avg: safeNullableNumber(features.magnitude_avg),
            sma: safeNullableNumber(features.sma),
            max_accel: safeNullableNumber(features.max_accel),
            max_gyro: safeNullableNumber(features.max_gyro),
            std_accel: safeNullableNumber(features.std_accel),
            jerk_peak: safeNullableNumber(features.jerk_peak),
        },
    };
}

// Print structured JSON logs without crashing on bad payloads.
function logJson(prefix, payload) {
    try {
        console.log(`${prefix} ${JSON.stringify(payload)}`);
    } catch (error) {
        console.log(`${prefix} <unserializable: ${error.message}>`);
    }
}

// Pick a LAN IP to display in startup logs.
function getLocalIP() {
    if (process.env.SERVER_IP) {
        return process.env.SERVER_IP;
    }

    const interfaces = os.networkInterfaces();
    const preferredPrefix = process.env.PREFERRED_IP_PREFIX || "192.168.1";
    let preferredIP = null;
    let fallbackIP = null;

    for (const name of Object.keys(interfaces)) {
        for (const iface of interfaces[name]) {
            if (iface.family === "IPv4" && !iface.internal) {
                if (iface.address.startsWith(preferredPrefix)) {
                    preferredIP = iface.address;
                }
                if (!fallbackIP) {
                    fallbackIP = iface.address;
                }
            }
        }
    }

    return preferredIP || fallbackIP || "localhost";
}

const localIP = getLocalIP();
// Start the HTTP server and print local/network entry points.
server.listen(PORT, "0.0.0.0", () => {
    console.log("");
    console.log("Fall detection server running");
    console.log(`Local:   http://localhost:${PORT}`);
    console.log(`Network: http://${localIP}:${PORT}`);
    console.log(`CoAP:    coap://${localIP}:${COAP_PORT}`);
    console.log(`ML:      ${ML_SERVICE_URL}`);
    console.log(`FCM:     ${firebaseState.messaging ? "enabled" : `disabled (${firebaseState.error})`}`);
    console.log("");
});

coap.on("message", async (message, rinfo) => {
    let request;
    try {
        request = parseCoapRequest(message);
    } catch (error) {
        console.error(`CoAP parse error from ${rinfo.address}:${rinfo.port}:`, error.message);
        return;
    }

    if (request.code !== COAP_CODE.POST) {
        sendCoapResponse(request, rinfo, COAP_CODE.METHOD_NOT_ALLOWED);
        return;
    }

    try {
        switch (request.path) {
            case "/api/session/start": {
                startNewSession(currentLabel);
                sendCoapResponse(request, rinfo, COAP_CODE.CREATED);
                break;
            }
            case "/api/session/new": {
                const previousSessionId = currentSessionId;
                stopCurrentSession();

                const requestedLabel = parseRequestedLabel(request.payload);
                const labelToUse = requestedLabel === "0" || requestedLabel === "1" ? requestedLabel : currentLabel;
                startNewSession(labelToUse);

                logJson("[/coap/session/new] started", {
                    from: `${rinfo.address}:${rinfo.port}`,
                    previous_session_id: previousSessionId,
                    session_id: currentSessionId,
                    label: labelToUse,
                });
                sendCoapResponse(request, rinfo, COAP_CODE.CREATED);
                break;
            }
            case "/api/session/stop": {
                stopCurrentSession();
                sendCoapResponse(request, rinfo, COAP_CODE.CHANGED);
                break;
            }
            case "/api/sensor-batch": {
                let batchData;
                try {
                    batchData = decodeSensorBatchPayload(request.payload);
                } catch (error) {
                    console.error(`CoAP batch decode error from ${rinfo.address}:${rinfo.port}:`, error.message);
                    sendCoapResponse(request, rinfo, COAP_CODE.BAD_REQUEST);
                    break;
                }

                const result = await processSensorBatch(batchData, "coap");
                if (result.statusCode >= 200 && result.statusCode < 300) {
                    sendCoapResponse(request, rinfo, COAP_CODE.CHANGED);
                } else if (result.statusCode === 400) {
                    sendCoapResponse(request, rinfo, COAP_CODE.BAD_REQUEST);
                } else {
                    sendCoapResponse(request, rinfo, COAP_CODE.INTERNAL_SERVER_ERROR);
                }
                break;
            }
            default:
                sendCoapResponse(request, rinfo, COAP_CODE.NOT_FOUND);
                break;
        }
    } catch (error) {
        console.error(`CoAP request error at ${request.path}:`, error);
        sendCoapResponse(request, rinfo, COAP_CODE.INTERNAL_SERVER_ERROR);
    }
});

coap.on("error", (error) => {
    console.error("CoAP server error:", error);
});

coap.bind(COAP_PORT, "0.0.0.0", () => {
    console.log(`CoAP UDP server listening on port ${COAP_PORT}`);
});
