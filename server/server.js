require('dotenv').config();
const express = require('express');
const http = require('http');
const socketIO = require('socket.io');
const cors = require('cors');
const os = require('os');
const fs = require('fs');
const path = require('path');

const app = express();
const server = http.createServer(app);
const io = socketIO(server, {
  cors: {
    origin: "*",
    methods: ["GET", "POST"]
  }
});

app.use(cors());
app.use(express.json({ limit: '10mb' })); 

let connectedClients = 0;

const CURRENT_LABEL = '0'; 

const DATA_DIR = path.join(__dirname, 'data', 'collected');
const NORMAL_DIR = path.join(DATA_DIR, 'Normal');
const FALL_DIR = path.join(DATA_DIR, 'Fall');
const SESSION_PREFIX = 'session';
let currentSessionId = null;
let currentSessionLabel = null;
let sessionStartTime = null;
let fallMarkers = []; // Lưu timestamps đánh dấu fall

// Tạo thư mục data nếu chưa có
if (!fs.existsSync(DATA_DIR)) {
  fs.mkdirSync(DATA_DIR, { recursive: true });
}
if (!fs.existsSync(NORMAL_DIR)) {
  fs.mkdirSync(NORMAL_DIR, { recursive: true });
}
if (!fs.existsSync(FALL_DIR)) {
  fs.mkdirSync(FALL_DIR, { recursive: true });
}

function inferLabelFromSessionId(sessionId) {
  if (typeof sessionId !== 'string') return null;
  const m = sessionId.match(/^label([01])_/);
  return m ? m[1] : null;
}

function getSessionDir(sessionId = currentSessionId, label = currentSessionLabel) {
  const effectiveLabel = label ?? inferLabelFromSessionId(sessionId) ?? CURRENT_LABEL;
  const base = effectiveLabel === '1' ? FALL_DIR : NORMAL_DIR;
  return path.join(base, sessionId);
}

// WebSocket connection
io.on('connection', (socket) => {
  connectedClients++;
  console.log(`✓ Client connected. Total: ${connectedClients}`);
  
  socket.on('disconnect', () => {
    connectedClients--;
    console.log(`✗ Client disconnected. Total: ${connectedClients}`);
  });
});

// REST API endpoint nhận dữ liệu từ cảm biến (single sample - legacy)
app.post('/api/sensor', (req, res) => {
  const sensorData = req.body;
  
  // Thêm timestamp nếu chưa có
  if (!sensorData.timestamp) {
    sensorData.timestamp = new Date().toISOString();
  }
  
  console.log('Received sensor data:', sensorData);
  
  // Broadcast dữ liệu đến tất cả client qua WebSocket
  io.emit('sensorData', sensorData);
  
  res.json({ 
    success: true, 
    message: 'Data received and broadcasted',
    clients: connectedClients 
  });
});

// REST API endpoint nhận batch data (sliding window)
app.post('/api/sensor-batch', (req, res) => {
  const batchData = req.body;
  
  if (!batchData.accel_data || !batchData.gyro_data) {
    return res.status(400).json({ 
      success: false, 
      error: 'Missing accel_data or gyro_data' 
    });
  }

  console.log(`✓ Received batch: ${batchData.accel_data.length} samples | BPM: ${batchData.bpm} | Mag: ${batchData.features?.magnitude_avg?.toFixed(2)}`);
  
  // Tạo session mới nếu chưa có
  if (!currentSessionId) {
    startNewSession();
  }

  // Lưu vào CSV (WEDA-FALL format)
  try {
    saveToCSV(batchData);
  } catch (error) {
    console.error('Error saving CSV:', error);
  }

  // Broadcast data đến clients
  io.emit('sensorBatch', batchData);
  
  res.json({ 
    success: true, 
    message: 'Batch data received and saved',
    session_id: currentSessionId,
    clients: connectedClients 
  });
});

// API để start/stop session
app.post('/api/session/start', (req, res) => {
  startNewSession(CURRENT_LABEL);  // Dùng label từ config
  res.json({ 
    success: true, 
    session_id: currentSessionId,
    start_time: sessionStartTime,
    label: CURRENT_LABEL
  });
});

app.post('/api/session/stop', (req, res) => {
  if (currentSessionId) {
    // Lưu fall markers vào file
    const sessionDir = getSessionDir();
    const markersPath = path.join(sessionDir, 'fall_markers.json');
    fs.writeFileSync(markersPath, JSON.stringify(fallMarkers, null, 2));
    console.log(`✓ Session ${currentSessionId} stopped - ${fallMarkers.length} fall markers saved`);
  }
  currentSessionId = null;
  currentSessionLabel = null;
  sessionStartTime = null;
  fallMarkers = [];
  res.json({ success: true });
});

// API để stop session hiện tại và start session mới ngay (tiện cho nút phần cứng)
app.post('/api/session/new', (req, res) => {
  const previousSessionId = currentSessionId;

  // Stop (giống /api/session/stop)
  if (currentSessionId) {
    const sessionDir = getSessionDir();
    const markersPath = path.join(sessionDir, 'fall_markers.json');
    fs.writeFileSync(markersPath, JSON.stringify(fallMarkers, null, 2));
    console.log(`✓ Session ${currentSessionId} stopped - ${fallMarkers.length} fall markers saved`);
  }

  currentSessionId = null;
  currentSessionLabel = null;
  sessionStartTime = null;
  fallMarkers = [];

  // Start new session
  const requestedLabel = (req.body && typeof req.body.label !== 'undefined') ? String(req.body.label) : null;
  const labelToUse = (requestedLabel === '0' || requestedLabel === '1') ? requestedLabel : CURRENT_LABEL;
  startNewSession(labelToUse);

  res.json({
    success: true,
    previous_session_id: previousSessionId,
    session_id: currentSessionId,
    start_time: sessionStartTime,
    label: labelToUse
  });
});

// API để mark fall event
app.post('/api/mark-fall', (req, res) => {
  if (!currentSessionId) {
    return res.status(400).json({ success: false, error: 'No active session' });
  }
  
  const currentTime = new Date().toISOString();
  const elapsedSeconds = (Date.now() - new Date(sessionStartTime).getTime()) / 1000;
  
  fallMarkers.push({
    timestamp: currentTime,
    elapsed_seconds: elapsedSeconds
  });
  
  console.log(`🔴 Fall marked at ${elapsedSeconds.toFixed(1)}s`);
  
  res.json({ 
    success: true, 
    elapsed_seconds: elapsedSeconds,
    marker_count: fallMarkers.length
  });
});

// Endpoint test
app.get('/api/status', (req, res) => {
  res.json({ 
    status: 'running',
    connectedClients: connectedClients,
    timestamp: new Date().toISOString()
  });
});

// Lấy IP từ .env hoặc tự động detect
function getLocalIP() {
  // Nếu có SERVER_IP trong .env, dùng luôn
  if (process.env.SERVER_IP) {
    return process.env.SERVER_IP;
  }

  const interfaces = os.networkInterfaces();
  const preferredPrefix = process.env.PREFERRED_IP_PREFIX || '192.168.1';
  let preferredIP = null;
  let fallbackIP = null;
  
  // Tìm IP theo prefix ưu tiên (192.168.1 hoặc 172.20.10)
  for (const name of Object.keys(interfaces)) {
    for (const iface of interfaces[name]) {
      if (iface.family === 'IPv4' && !iface.internal) {
        // Tìm IP khớp với prefix
        if (iface.address.startsWith(preferredPrefix)) {
          preferredIP = iface.address;
        }
        // Lưu IP đầu tiên làm fallback
        if (!fallbackIP) fallbackIP = iface.address;
      }
    }
  }
  
  return preferredIP || fallbackIP || 'localhost';
}

// ================================================================
// HELPER FUNCTIONS
// ================================================================
function startNewSession(label = '0') {
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);
  // Format: label0_timestamp hoặc label1_timestamp
  currentSessionId = `label${label}_${timestamp}`;
  currentSessionLabel = String(label);
  sessionStartTime = new Date().toISOString();
  
  const sessionDir = getSessionDir(currentSessionId, currentSessionLabel);
  if (!fs.existsSync(sessionDir)) {
    fs.mkdirSync(sessionDir, { recursive: true });
  }

  // Tạo file CSV headers
  const accelPath = path.join(sessionDir, 'accel.csv');
  const gyroPath = path.join(sessionDir, 'gyro.csv');
  const metadataPath = path.join(sessionDir, 'metadata.json');
  const labelPath = path.join(sessionDir, 'label.txt');

  fs.writeFileSync(accelPath, 'accel_time_list,accel_x_list,accel_y_list,accel_z_list\n');
  fs.writeFileSync(gyroPath, 'gyro_time_list,gyro_x_list,gyro_y_list,gyro_z_list\n');
  fs.writeFileSync(labelPath, label); // Chỉ ghi "0" hoặc "1"
  
  const metadata = {
    session_id: currentSessionId,
    start_time: sessionStartTime,
    device: 'ESP32-C3 + MPU6050 + MAX30102',
    sample_rate: 50,
    window_size: 100,
    label: parseInt(label)
  };
  fs.writeFileSync(metadataPath, JSON.stringify(metadata, null, 2));
  
  // Reset fall markers
  fallMarkers = [];

  const labelText = label === '1' ? 'FALL' : 'NORMAL';
  console.log(`\n✓ New session started: ${currentSessionId} [Label: ${label} - ${labelText}]`);
}

function saveToCSV(batchData) {
  if (!currentSessionId) return;

  const sessionDir = getSessionDir();
  const accelPath = path.join(sessionDir, 'accel.csv');
  const gyroPath = path.join(sessionDir, 'gyro.csv');

  // Append accel data
  let accelLines = '';
  batchData.accel_data.forEach(sample => {
    accelLines += `${sample.t},${sample.x},${sample.y},${sample.z}\n`;
  });
  fs.appendFileSync(accelPath, accelLines);

  // Append gyro data
  let gyroLines = '';
  batchData.gyro_data.forEach(sample => {
    gyroLines += `${sample.t},${sample.x},${sample.y},${sample.z}\n`;
  });
  fs.appendFileSync(gyroPath, gyroLines);
}

const PORT = process.env.PORT || 3000;
const localIP = getLocalIP();

server.listen(PORT, '0.0.0.0', () => {
  console.log('\n╔════════════════════════════════════════╗');
  console.log('║   SENSOR DATA SERVER RUNNING (V2)      ║');
  console.log('╠════════════════════════════════════════╣');
  console.log(`║ Local:    http://localhost:${PORT}       ║`);
  console.log(`║ Network:  http://${localIP}:${PORT}${' '.repeat(Math.max(0, 6 - localIP.length))} ║`);
  console.log('╠════════════════════════════════════════╣');
  console.log('║ POST endpoint: /api/sensor             ║');
  console.log('║ Status check:  /api/status             ║');
  console.log('╠════════════════════════════════════════╣');
  
  // Hiển thị config từ .env
  const prefix = process.env.PREFERRED_IP_PREFIX || 'auto-detect';
  console.log(`║ IP Prefix: ${prefix}${' '.repeat(Math.max(0, 27 - prefix.length))} ║`);
  
  if (process.env.SERVER_IP) {
    console.log('║ Mode: Fixed IP (from .env)             ║');
  } else {
    console.log('║ Mode: Auto-detect                      ║');
  }
  
  console.log('╚════════════════════════════════════════╝\n');
  console.log('💡 Đổi IP: Sửa file .env và restart server\n');
});

