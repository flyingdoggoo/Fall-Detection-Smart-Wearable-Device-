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

app.use(cors());
app.use(express.json({ limit: "10mb" }));

let connectedClients = 0;
let currentLabel = "1";
let currentSessionId = null;
let currentSessionLabel = null;
let sessionStartTime = null;
let fallMarkers = [];
let lastTriggeredFallAt = 0;

const PORT = Number(process.env.PORT || 3000);
const ML_SERVICE_URL = process.env.ML_SERVICE_URL || "http://127.0.0.1:5000/predict";
const ML_TIMEOUT_MS = Number(process.env.ML_TIMEOUT_MS || 4000);
const FALL_CONFIDENCE_THRESHOLD = Number(process.env.FALL_CONFIDENCE_THRESHOLD || 0.8);
const FALL_NOTIFICATION_COOLDOWN_MS = Number(process.env.FALL_NOTIFICATION_COOLDOWN_MS || 30000);
const DEVICE_ONLINE_WINDOW_MS = Number(process.env.DEVICE_ONLINE_WINDOW_MS || 15000);
const MAX_FALL_HISTORY = Number(process.env.MAX_FALL_HISTORY || 50);

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

// Receive one full sensor window, run ML, save data, and emit updates.
app.post("/api/sensor-batch", async (req, res) => {
    const batchData = req.body || {};
    if (!Array.isArray(batchData.accel_data) || !Array.isArray(batchData.gyro_data)) {
        logJson("[/api/sensor-batch] invalid_payload", {
            accel_is_array: Array.isArray(batchData.accel_data),
            gyro_is_array: Array.isArray(batchData.gyro_data),
        });
        return res.status(400).json({
            success: false,
            error: "Missing accel_data or gyro_data arrays",
        });
    }

    logJson("[/api/sensor-batch] request", summarizeBatchData(batchData));

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

    const mlResult = await predictFall(batchData);
    logJson("[/api/sensor-batch] ml_result", mlResult);
    deviceStatus.last_prediction = mlResult;
    writeJsonFile(DEVICE_STATUS_FILE, deviceStatus);

    try {
        saveFeatures(batchData, mlResult);
    } catch (error) {
        console.error("Error saving features:", error);
    }

    const triggeredFallEvent = await maybeHandleFallDetection(batchData, mlResult, esp32FallDetected);

    const batchForBroadcast = {
        ...batchData,
        ml_result: mlResult,
        session_id: currentSessionId,
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
    };

    logJson("[/api/sensor-batch] response", responsePayload);
    res.json(responsePayload);
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
    console.log(`ML:      ${ML_SERVICE_URL}`);
    console.log(`FCM:     ${firebaseState.messaging ? "enabled" : `disabled (${firebaseState.error})`}`);
    console.log("");
});
