# Fall Detection Smart Wearable Device — Architecture Plan

## Table of Contents

- [Current State Assessment](#current-state-assessment)
- [Phase 1: Short Term (Single Device)](#phase-1-short-term-single-device)
  - [1A. Server Redesign](#1a-server-redesign)
  - [1B. Mobile App (Flutter)](#1b-mobile-app-flutter)
- [Phase 2: Long Term (Multi-Device Platform)](#phase-2-long-term-multi-device-platform)
  - [2A. Server Redesign for Scale](#2a-server-redesign-for-scale)
  - [2B. Mobile App Expansion](#2b-mobile-app-expansion)
  - [2C. ESP32 Changes](#2c-esp32-changes)
- [Execution Order](#execution-order)
- [Key Decisions](#key-decisions)

---

## Current State Assessment

| Component | Status | Details |
|-----------|--------|---------|
| **ESP32 Hardware** | Functional | MPU6050 at 50Hz, Kalman filtering, 4-state FSM, sends 2s windows via HTTP POST |
| **Server** | Functional | Node.js/Express + Socket.IO, flat-file CSV/JSON storage, data collection endpoints |
| **Web Dashboard** | Functional | Browser-based Chart.js dashboard, real-time via Socket.IO |
| **ML Model** | NOT started | Only an EDA notebook and a prompt template exist |
| **Mobile App** | NOT started | No mobile code whatsoever |
| **Data Collected** | In progress | ~133 feature windows collected (target: 770+) |

### Current Data Flow

```
ESP32-C3 (Wearable)                    Node.js Server (Port 3000)              Browser Dashboard
┌──────────────────┐   HTTP POST       ┌──────────────────────┐   Socket.IO    ┌──────────────────┐
│ MPU6050 (I2C)    │   /api/sensor-    │ Express.js           │   sensorBatch  │ index_v2.html    │
│ 50Hz sampling    │──────batch───────►│ Save CSV/JSON        │──────event────►│ Chart.js graphs  │
│ Kalman filter    │   JSON payload    │ Broadcast Socket.IO  │                │ Stat cards       │
│ 4-state FSM      │                   │ Session management   │                │ Event log        │
│ 100-sample window│                   │ Flat file storage    │                │                  │
└──────────────────┘                   └──────────────────────┘                └──────────────────┘
```

### Current Technical Specs

| Parameter | Value |
|-----------|-------|
| IMU sample rate | 50 Hz (20ms interval) |
| Window size | 100 samples = 2 seconds |
| Data transmission rate | Every 2 seconds |
| Features computed on device | magnitude_avg, sma, max_accel, max_gyro, std_accel, jerk_peak |
| Communication | WiFi HTTP POST (JSON) |
| JSON payload size | ~16KB per window |
| Storage | Flat CSV + JSON files, no database |
| Heart rate sensor | MAX30102 present but unused (BPM simulated) |

### Current Server Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/sensor-batch` | POST | Primary — receives 100-sample windows from ESP32 |
| `/api/sensor` | POST | Legacy single-sample endpoint |
| `/api/session/start` | POST | Start a new data collection session |
| `/api/session/stop` | POST | Stop the current session |
| `/api/session/new` | POST | Combined stop + start |
| `/api/label` | GET/POST | Get or set current label (0=Normal, 1=Fall) |
| `/api/mark-fall` | POST | Record a fall timestamp marker |
| `/api/sessions/stats` | GET | Count session directories |
| `/api/status` | GET | Server status, client count, current session |

---

## Phase 1: Short Term (Single Device)

**Goal:** Get a working end-to-end fall detection system with one ESP32, one server, and a basic mobile app that receives fall notifications.

### 1A. Server Redesign

#### Architecture Diagram

```
┌─────────────┐     HTTP POST /api/sensor-batch      ┌──────────────────────────────────┐
│   ESP32-C3  │ ──────────────────────────────────>  │        Node.js Server            │
│  (Wearable) │  <─────────────────────────────────  │        (Express.js)              │
│             │    HTTP Response {fall, confidence}  │                                  │
└─────────────┘                                      │ ┌──────────────────────────────┐ │
                                                     │ │  Python ML Inference Service │ │
                                                     │ │  (Flask or FastAPI)          │ │
                                                     │ │  - Loads trained .h5 model   │ │
                                                     │ │  - POST /predict endpoint    │ │
                                                     │ │  - Runs on port 5000         │ │
                                                     │ └──────────────────────────────┘ │
┌─────────────┐     Firebase Cloud Messaging (FCM)   │                                  │
│  Mobile App │  ◄─────────────────────────────────  │ ┌──────────────────────────────┐ │
│  (Flutter)  │     + Socket.IO for real-time        │ │ Firebase Admin SDK           │ │
│             │ ──────────────────────────────────►  │ │ (firebase-admin npm package) │ │
└─────────────┘     HTTP (status, history, response) │ └──────────────────────────────┘ │
                                                     │                                  │
┌─────────────┐     Socket.IO (existing)             │  Storage: flat CSV/JSON files    │
│  Web Client │  ◄────────────────────────────────   │                                  │
│  (Browser)  │                                      └──────────────────────────────────┘
└─────────────┘
```

#### What to Keep (from existing server)

- Express.js + Socket.IO framework
- `/api/sensor-batch` endpoint
- CSV data collection & session management
- Socket.IO broadcast to web dashboard
- Flat file storage (sufficient for single device)

#### What to Add

##### 1. Python ML Microservice (sidecar process)

A lightweight Python service that runs alongside the Node.js server.

```
python_ml/
├── app.py              # Flask/FastAPI server
├── model/
│   └── fall_model.h5   # Trained model file
├── predict.py          # Inference logic
├── requirements.txt    # Python dependencies
└── start-ml.bat        # Startup script
```

**Single endpoint:**
```
POST http://localhost:5000/predict

Request body:
{
  "features": {
    "magnitude_avg": 9.81,
    "sma": 3.27,
    "max_accel": 12.5,
    "max_gyro": 1.2,
    "std_accel": 0.45,
    "jerk_peak": 150.3
  },
  "raw_window": {
    "accel_data": [...],
    "gyro_data": [...]
  }
}

Response body:
{
  "fall_detected": true,
  "confidence": 0.93,
  "model_version": "v1.0"
}
```

##### 2. Firebase Cloud Messaging (Push Notifications)

- Add `firebase-admin` npm package to Node.js server
- Initialize with Firebase service account credentials
- Store mobile app FCM tokens in a JSON file (`fcm_tokens.json` — supports multiple tokens for family members)
- When fall detected, send high-priority push notification to **ALL registered tokens**
- Phase 1 uses a simple model: any phone that registers gets all alerts (no roles/permissions yet)

##### 3. New Server Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/register-device` | POST | Mobile app registers its FCM token (supports multiple phones — wearer + family) |
| `/api/fall-history` | GET | Returns recent fall events (last 50, stored in JSON file) |
| `/api/device/status` | GET | Current device state: connected/disconnected, last seen timestamp |
| `/api/fall-response` | POST | Mobile app sends "dismiss" (false alarm) or "confirm" (real fall) |

##### 4. Fall Decision Flow

```
ESP32 sends batch
    │
    ▼
Node.js /api/sensor-batch receives data
    │
    ├──► Save to CSV (as before)
    ├──► Broadcast via Socket.IO to web dashboard (as before)
    │
    ├──► Call Python ML service POST http://localhost:5000/predict
    │       │
    │       ▼
    │    ML returns {fall_detected: true, confidence: 0.93}
    │       │
    │       ├── If fall_detected AND confidence > threshold (e.g., 0.80):
    │       │     ├──► Send FCM push notification to ALL registered phones (wearer + family)
    │       │     ├──► Emit Socket.IO 'fallDetected' event to web dashboard
    │       │     └──► Log fall event to fall_history.json
    │       │
    │       └── If not fall:
    │             └──► Normal operation continues
    │
    └──► Return HTTP response to ESP32:
         {ml_fall_detected: true/false, confidence: 0.93}
```

##### 5. Updated Project Structure (Phase 1)

```
Fall-Detection-Smart-Wearable-Device/
├── esp32_code/
│   └── fall_detection_arduino_v3.ino      # Existing (minor updates)
│
├── server/
│   ├── server.js                          # Updated Express.js server
│   ├── firebase-service-account.json      # Firebase credentials (gitignored)
│   ├── package.json                       # Updated with firebase-admin
│   ├── data/
│   │   ├── collected/                     # Existing CSV/JSON data
│   │   ├── fcm_tokens.json               # Array of FCM tokens (wearer + family members)
│   │   └── fall_history.json             # Recent fall events log
│   └── start-server.bat
│
├── python_ml/
│   ├── app.py                             # Flask/FastAPI ML service
│   ├── predict.py                         # Inference logic
│   ├── model/
│   │   └── fall_model.h5                  # Trained model
│   ├── requirements.txt
│   └── start-ml.bat
│
├── mobile_app/                            # Flutter project
│   └── (see 1B below)
│
├── client/
│   └── index_v2.html                      # Existing web dashboard
│
├── python/
│   └── compare_hieu_tien.ipynb            # Existing EDA notebook
│
├── start-all.bat                          # Updated: start server + ML service + open client
└── ARCHITECTURE_PLAN.md                   # This file
```

---

### 1B. Mobile App (Flutter)

#### Technology Choice

| Option | Pros | Cons | Verdict |
|--------|------|------|---------|
| **Flutter** | Cross-platform, single codebase, great Firebase/FCM support, strong UI toolkit | Dart learning curve | **Recommended** |
| React Native | JavaScript (familiar), large ecosystem | Worse performance, bridge overhead | Alternative |
| Native Kotlin/Swift | Best performance, full platform API access | Two separate codebases | Overkill for this project |

#### Screen Map

```
┌─────────────────────────────────────────────────────────┐
│                     App Navigation                      │
│                                                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐  │
│  │   Home   │  │   Fall   │  │  History │  │Settings │  │
│  │Dashboard │  │  Alert   │  │          │  │         │  │
│  │          │  │(Overlay) │  │          │  │         │  │
│  └──────────┘  └──────────┘  └──────────┘  └─────────┘  │
└─────────────────────────────────────────────────────────┘
```

| Screen | Purpose | Key Elements |
|--------|---------|--------------|
| **Home/Dashboard** | Shows device status | Connection indicator (green/red), last reading timestamp, current activity state, BPM (when real sensor works). For family members: shows monitored person's status |
| **Fall Alert** | Full-screen overlay on fall detection | Large warning icon, "I'm OK" dismiss button (wearer only), "Call Emergency" button, 60-second auto-escalation countdown |
| **Fall History** | List of past fall events | Timestamp, confidence score, status (confirmed/dismissed/escalated), pull-to-refresh |
| **Settings** | Configuration | Server IP/port, emergency contact phone number, notification sound toggle, alert timeout duration, role selection (wearer/family monitor) |

> **Note on Phase 1 family monitoring:** In Phase 1, any phone that registers with the server receives all fall alerts. The app has a simple role toggle in Settings — "I am the wearer" vs. "I am a family member." The wearer sees the "I'm OK" dismiss button; family members see "Call [wearer's name]" instead. No login or invitation system is needed — just connect to the same server IP.

#### Flutter Project Structure

```
mobile_app/
├── android/                          # Android platform files
├── ios/                              # iOS platform files
├── lib/
│   ├── main.dart                     # App entry point, Firebase init
│   │
│   ├── config/
│   │   └── app_config.dart           # Server URL, constants, thresholds
│   │
│   ├── models/
│   │   ├── fall_event.dart           # Fall event data model
│   │   └── device_status.dart        # Device status data model
│   │
│   ├── services/
│   │   ├── api_service.dart          # HTTP calls to Node.js server
│   │   ├── socket_service.dart       # Socket.IO client for real-time updates
│   │   ├── notification_service.dart # FCM setup + local notifications
│   │   └── alert_service.dart        # Fall alert logic + auto-escalation timer
│   │
│   ├── screens/
│   │   ├── home_screen.dart          # Dashboard with device status
│   │   ├── fall_alert_screen.dart    # Full-screen fall alert overlay
│   │   ├── history_screen.dart       # Fall event history list
│   │   └── settings_screen.dart      # App configuration
│   │
│   └── widgets/
│       ├── status_card.dart          # Reusable status indicator card
│       └── fall_event_tile.dart      # Single fall event list item
│
├── pubspec.yaml                      # Flutter dependencies
├── firebase.json                     # Firebase config
└── README.md
```

#### Key Flutter Dependencies

```yaml
dependencies:
  flutter:
    sdk: flutter
  firebase_core: ^2.x
  firebase_messaging: ^14.x       # FCM push notifications
  flutter_local_notifications: ^16.x  # Local notification display
  socket_io_client: ^2.x          # Socket.IO for real-time data
  http: ^1.x                      # HTTP requests to server
  shared_preferences: ^2.x        # Local settings storage
  url_launcher: ^6.x              # Open phone dialer for emergency call
  provider: ^6.x                  # State management
```

#### Critical Flow — Fall Alert Sequence

```
1. FCM push notification arrives
   │
   ├── App is KILLED or in BACKGROUND:
   │     └── System notification with alarm sound + vibration
   │           └── User taps notification → App opens → Fall Alert Screen
   │
   └── App is in FOREGROUND:
         └── Immediate full-screen Fall Alert overlay
               │
               ▼
2. Fall Alert Screen shows:
   ┌─────────────────────────────────┐
   │          FALL DETECTED          │
   │                                 │
   │   [Person's name] may have      │
   │   fallen at [timestamp]         │
   │                                 │
   │   Confidence: 93%               │
   │                                 │
   │   Auto-calling in: 00:47        │
   │                                 │
   │   ┌──────────┐ ┌──────────────┐ │
   │   │  I'm OK  │ │Call Emergency│ │
   │   └──────────┘ └──────────────┘ │
   └─────────────────────────────────┘
               │
               ├── User taps "I'm OK":
               │     ├── POST /api/fall-response {action: "dismiss"}
               │     ├── Cancel countdown timer
               │     └── Return to Home screen
               │
               ├── User taps "Call Emergency":
               │     ├── POST /api/fall-response {action: "confirmed"}
               │     └── Open phone dialer with emergency contact number
               │
               └── 60 seconds pass with no response:
                     ├── POST /api/fall-response {action: "escalated"}
                     └── Auto-dial emergency contact number
```

---

## Phase 2: Long Term (Multi-Device Platform)

### 2A. Server Redesign for Scale

#### Architecture Diagram

```
┌──────────┐                    ┌───────────────────────────────────────────────────┐
│ ESP32 #1 │──┐                 │              Cloud Server                         │
├──────────┤  │   HTTPS POST    │                                                   │
│ ESP32 #2 │──┼────────────────►│  ┌──────────────────────────────────────────────┐ │
├──────────┤  │   + Auth Token  │  │          API Gateway (Express.js)            │ │
│ ESP32 #N │──┘                 │  │  - JWT authentication                        │ │
                                │  │  - Rate limiting                             │ │
                                │  │  - Request routing                           │ │
┌──────────┐                    │  └───────┬──────────┬──────────┬────────────────┘ │
│  App #1  │──┐                 │          │          │          │                  │
├──────────┤  │   HTTPS + WSS   │  ┌───────▼───┐  ┌───▼────┐  ┌──▼──────────────┐   │
│  App #2  │──┼────────────────►│  │   Auth    │  │ Device │  │  ML Inference   │   │
├──────────┤  │   + JWT Token   │  │  Service  │  │Manager │  │  Service        │   │
│  App #N  │──┘                 │  └───────────┘  └────────┘  └─────────────────┘   │
                                │                                                   │
                                │  ┌──────────────────────────────────────────────┐ │
                                │  │      PostgreSQL Database                     │ │
                                │  └──────────────────────────────────────────────┘ │
                                │                                                   │
                                │  ┌──────────────────────────────────────────────┐ │
                                │  │      Firebase Admin SDK (FCM)                │ │
                                │  └──────────────────────────────────────────────┘ │
                                └───────────────────────────────────────────────────┘
```

#### Database Schema (PostgreSQL)

```sql
-- =============================================
-- Users & Authentication
-- =============================================
CREATE TABLE users (
    id            SERIAL PRIMARY KEY,
    email         VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    name          VARCHAR(100) NOT NULL,
    phone         VARCHAR(20),
    created_at    TIMESTAMP DEFAULT NOW(),
    updated_at    TIMESTAMP DEFAULT NOW()
);

-- =============================================
-- Physical ESP32 Wearable Devices
-- =============================================
CREATE TABLE devices (
    id               SERIAL PRIMARY KEY,
    user_id          INTEGER REFERENCES users(id) ON DELETE CASCADE,
    mac_address      VARCHAR(17) UNIQUE NOT NULL,   -- e.g., "AA:BB:CC:DD:EE:FF"
    name             VARCHAR(100),                   -- e.g., "Dad's wrist sensor"
    wear_position    VARCHAR(20) DEFAULT 'wrist',    -- 'wrist', 'ankle', 'waist'
    wear_side        VARCHAR(20) DEFAULT 'left',     -- 'left', 'right'
    wear_orientation VARCHAR(20) DEFAULT 'outer',    -- 'inner', 'outer'
    is_active        BOOLEAN DEFAULT TRUE,
    last_seen        TIMESTAMP,
    firmware_version VARCHAR(20),
    created_at       TIMESTAMP DEFAULT NOW()
);

-- =============================================
-- Mobile App FCM Tokens (for push notifications)
-- =============================================
CREATE TABLE device_tokens (
    id         SERIAL PRIMARY KEY,
    user_id    INTEGER REFERENCES users(id) ON DELETE CASCADE,
    fcm_token  TEXT NOT NULL,
    platform   VARCHAR(10) NOT NULL,               -- 'android', 'ios'
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(user_id, fcm_token)
);

-- =============================================
-- Emergency Contacts
-- =============================================
CREATE TABLE emergency_contacts (
    id       SERIAL PRIMARY KEY,
    user_id  INTEGER REFERENCES users(id) ON DELETE CASCADE,
    name     VARCHAR(100) NOT NULL,
    phone    VARCHAR(20) NOT NULL,
    priority INTEGER DEFAULT 1                     -- 1 = first to call
);

-- =============================================
-- Fall Events Log
-- =============================================
CREATE TABLE fall_events (
    id            SERIAL PRIMARY KEY,
    device_id     INTEGER REFERENCES devices(id) ON DELETE CASCADE,
    user_id       INTEGER REFERENCES users(id) ON DELETE CASCADE,
    timestamp     TIMESTAMP NOT NULL,
    confidence    FLOAT NOT NULL,                  -- 0.0 to 1.0
    features_json JSONB,                           -- snapshot of the 6 features
    fsm_state     INTEGER,                         -- FSM state from ESP32
    status        VARCHAR(20) DEFAULT 'pending',   -- 'pending', 'confirmed', 'dismissed', 'escalated'
    responded_at  TIMESTAMP,
    created_at    TIMESTAMP DEFAULT NOW()
);

-- =============================================
-- Family / Caregiver Monitoring Links
-- =============================================
CREATE TABLE monitoring_links (
    id                SERIAL PRIMARY KEY,
    wearer_id         INTEGER REFERENCES users(id) ON DELETE CASCADE,  -- person with device
    monitor_id        INTEGER REFERENCES users(id) ON DELETE CASCADE,  -- family member / caregiver
    role              VARCHAR(20) DEFAULT 'family',    -- 'family', 'caregiver', 'medical'
    can_view_history  BOOLEAN DEFAULT TRUE,
    can_dismiss_alert BOOLEAN DEFAULT FALSE,           -- only wearer by default
    status            VARCHAR(20) DEFAULT 'pending',   -- 'pending', 'active', 'revoked'
    created_at        TIMESTAMP DEFAULT NOW()
);

-- =============================================
-- Sensor Data Sessions (for model retraining)
-- =============================================
CREATE TABLE sessions (
    id           SERIAL PRIMARY KEY,
    device_id    INTEGER REFERENCES devices(id) ON DELETE CASCADE,
    label        INTEGER NOT NULL,                 -- 0=Normal, 1=Fall
    start_time   TIMESTAMP NOT NULL,
    end_time     TIMESTAMP,
    sample_count INTEGER DEFAULT 0,
    created_at   TIMESTAMP DEFAULT NOW()
);

-- =============================================
-- Indexes for Performance
-- =============================================
CREATE INDEX idx_devices_user ON devices(user_id);
CREATE INDEX idx_fall_events_user ON fall_events(user_id);
CREATE INDEX idx_fall_events_device ON fall_events(device_id);
CREATE INDEX idx_fall_events_timestamp ON fall_events(timestamp DESC);
CREATE INDEX idx_sessions_device ON sessions(device_id);
CREATE INDEX idx_monitoring_wearer ON monitoring_links(wearer_id);
CREATE INDEX idx_monitoring_monitor ON monitoring_links(monitor_id);
```

#### New/Modified API Endpoints (Phase 2)

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/api/auth/register` | POST | Public | User registration |
| `/api/auth/login` | POST | Public | User login, returns JWT |
| `/api/auth/refresh` | POST | JWT | Refresh access token |
| `/api/devices` | GET | JWT | List user's paired devices |
| `/api/devices` | POST | JWT | Register a new device |
| `/api/devices/:id` | PUT | JWT | Update device config (wear position, side, orientation) |
| `/api/devices/:id` | DELETE | JWT | Remove a device |
| `/api/devices/:id/status` | GET | JWT | Get device real-time status |
| `/api/sensor-batch` | POST | Device Token | Receive sensor data (existing, add auth) |
| `/api/falls` | GET | JWT | Get paginated fall history for user |
| `/api/falls/:id/respond` | POST | JWT | Respond to a fall (dismiss/confirm) |
| `/api/contacts` | GET/POST/PUT/DELETE | JWT | CRUD emergency contacts |
| `/api/notifications/register` | POST | JWT | Register FCM token |
| `/api/monitoring/invite` | POST | JWT | Wearer generates an invite code for a family member |
| `/api/monitoring/accept` | POST | JWT | Family member accepts invite, creates monitoring link |
| `/api/monitoring/links` | GET | JWT | List people I monitor + people who monitor me |
| `/api/monitoring/links/:id` | DELETE | JWT | Revoke a monitoring link |

---

### 2B. Mobile App Expansion

#### New Screens & Features

| Screen/Feature | Description |
|----------------|-------------|
| **Login / Register** | Email + password authentication with JWT |
| **Device Pairing** | Scan for ESP32 via BLE → register with server → provision WiFi credentials |
| **Device List** | Dashboard showing all paired devices with status indicators |
| **Device Config** | Set wear position (wrist/ankle/waist), side (left/right), orientation (inner/outer) |
| **Emergency Contacts** | Add/edit/reorder emergency contacts with call priority |
| **Family Monitoring** | Wearer invites family via share link/code; family members receive all fall alerts and can view history |
| **Enhanced Fall Response** | Countdown → Call contact #1 → Call contact #2 → Call emergency services (112/911) |
| **Activity Log** | Daily/weekly summary of activity patterns and fall risk indicators |

#### Family Monitoring & Invitation Flow (Phase 2)

```
Wearer opens app
    │
    ├── Goes to "Family & Caregivers" screen
    ├── Taps "Invite family member"
    ├── App calls POST /api/monitoring/invite → receives invite code
    ├── Shares invite code/link via SMS, Zalo, WhatsApp, etc.
    │
    ▼
Family member opens app
    ├── Creates account (or logs in)
    ├── Enters invite code → POST /api/monitoring/accept
    ├── monitoring_links row created: monitor_id ←→ wearer_id
    └── Now receives FCM fall alerts for the wearer
```

**Notification routing in Phase 2:**

```
Fall detected for Device X
    │
    ▼
Server looks up: Device X belongs to User (wearer)
    │
    ▼
Server queries: SELECT monitor_id FROM monitoring_links WHERE wearer_id = User AND status = 'active'
    │
    ├── Wearer's own phone     → FCM push
    ├── Son (monitor)          → FCM push
    ├── Daughter (monitor)     → FCM push
    └── Caretaker (monitor)    → FCM push
```

**Role-based permissions:**

| Role | Receive alerts | View history | Dismiss alert | Call emergency |
|------|---------------|-------------|---------------|----------------|
| **Wearer** | Yes | Yes (own) | Yes | Yes |
| **Family** | Yes | Yes (wearer's) | No (by default) | Yes |
| **Caregiver** | Yes | Yes (wearer's) | Configurable | Yes |

#### Updated Screen Map (Phase 2)

```
┌─────────────────────────────────────────────────────────────────────┐
│                         App Navigation                              │
│                                                                     │
│  Auth Flow:                                                         │
│  ┌──────────┐  ┌──────────┐                                         │
│  │  Login   │  │ Register │                                         │
│  └──────────┘  └──────────┘                                         │
│                                                                     │
│  Main Flow (after login):                                           │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐             │
│  │   Home   │  │ Devices  │  │ History  │  │ Settings │             │
│  │Dashboard │  │   List   │  │          │  │          │             │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘             │
│       │             │                           │                   │
│       │        ┌────▼─────┐              ┌──────▼──────┐            │
│       │        │  Device  │              │  Emergency  │            │
│       │        │  Config  │              │  Contacts   │            │
│       │        └──────────┘              └─────────────┘            │
│       │                                                             │
│  ┌────▼──────────┐   ┌──────────────────────┐                       │
│  │  Fall Alert   │   │  Family & Caregivers │                       │
│  │  (fullscreen) │   │  - Invite members    │                       │
│  └───────────────┘   │  - People I monitor  │                       │
│                      │  - Who monitors me   │                       │
│  Home adapts by      └──────────────────────┘                       │
│  perspective:                                                       │
│  - Wearer: own device status                                        │
│  - Monitor: list of people I monitor                                │
└─────────────────────────────────────────────────────────────────────┘
```

#### BLE Device Pairing Flow

```
1. User taps "Add Device" in mobile app
        │
        ▼
2. App starts BLE scan for ESP32 devices
   (ESP32 advertises a known service UUID)
        │
        ▼
3. User selects their ESP32 from the list
        │
        ▼
4. BLE connection established
        │
        ├── App sends WiFi credentials (SSID + password) via BLE characteristic
        ├── App sends server URL via BLE characteristic
        └── App sends device auth token via BLE characteristic
        │
        ▼
5. ESP32 stores credentials in NVS (non-volatile storage)
        │
        ▼
6. ESP32 connects to WiFi and registers with server
        │
        ▼
7. App registers device with server via REST API
        │
        ▼
8. Pairing complete — user configures wear position
```

---

### 2C. ESP32 Changes

| Change | Description | Priority |
|--------|-------------|----------|
| **BLE Service** | Add BLE GATT server for initial pairing with mobile app | HIGH |
| **WiFi Provisioning** | Receive WiFi credentials via BLE instead of hardcoding | HIGH |
| **NVS Storage** | Store WiFi creds, server URL, device token in non-volatile storage | HIGH |
| **Device ID** | Generate unique device ID (based on MAC), send with every HTTP request | HIGH |
| **Auth Header** | Include `Authorization: Bearer <token>` in all HTTP requests | HIGH |
| **Wear Position** | Store wear position config received from mobile app via BLE/server | MEDIUM |
| **OTA Updates** | Support over-the-air firmware updates from server | LOW |
| **Battery Monitoring** | Report battery level to server (if battery powered) | LOW |

#### Updated ESP32 Boot Sequence (Phase 2)

```
Power On
    │
    ▼
Check NVS for stored WiFi credentials
    │
    ├── Credentials found:
    │     ├── Connect to WiFi
    │     ├── If connected: register with server, enter normal operation
    │     └── If failed: enter BLE pairing mode
    │
    └── No credentials:
          └── Enter BLE pairing mode
                │
                ▼
          Advertise BLE service
          Wait for mobile app to connect and provide:
            - WiFi SSID + password
            - Server URL
            - Auth token
                │
                ▼
          Store in NVS → connect to WiFi → normal operation
```

---

## Execution Order

| Step | Task | Phase | Priority | Dependencies |
|------|------|-------|----------|--------------|
| **1** | Collect enough training data (target: 600+ windows, balanced Fall/Normal) | Prereq | CRITICAL | Current hardware works |
| **2** | Train ML model (CNN-LSTM or similar) | Prereq | CRITICAL | Step 1 |
| **3** | Build Python ML inference microservice (Flask/FastAPI, `/predict` endpoint) | Phase 1 | HIGH | Step 2 |
| **4** | Integrate ML service into Node.js server (call `/predict` on each batch) | Phase 1 | HIGH | Step 3 |
| **5** | Add Firebase FCM push notifications to Node.js server | Phase 1 | HIGH | Firebase project setup |
| **6** | Build minimal Flutter app (Home + Fall Alert + History + Settings) | Phase 1 | HIGH | Flutter SDK setup |
| **7** | Integrate end-to-end: ESP32 → Server → ML → FCM → Mobile alert | Phase 1 | HIGH | Steps 3-6 |
| **8** | Test with real falls, tune confidence threshold, fix false positives/negatives | Phase 1 | HIGH | Step 7 |
| **9** | Set up PostgreSQL database, migrate from flat files | Phase 2 | MEDIUM | — |
| **10** | Add user authentication (JWT) to server | Phase 2 | MEDIUM | Step 9 |
| **11** | Update Flutter app: login, register, multi-device dashboard | Phase 2 | MEDIUM | Step 10 |
| **12** | Add BLE pairing to ESP32 firmware + Flutter app | Phase 2 | MEDIUM | Step 11 |
| **13** | Multi-device support + device configuration (wear position) | Phase 2 | LOW | Steps 9-12 |
| **14** | Family monitoring: invitation system, monitoring_links, role-based notifications | Phase 2 | LOW | Steps 10-11 |
| **15** | Deploy server to cloud (DigitalOcean / AWS / GCP) | Phase 2 | LOW | Step 10 |
| **16** | Activity logs, advanced analytics features | Phase 2 | LOW | Steps 11-15 |

---

## Key Decisions

### 1. Mobile Framework

| Option | Recommendation |
|--------|---------------|
| **Flutter** | **Recommended** — cross-platform, single codebase, excellent Firebase support, strong UI toolkit |
| React Native | Alternative — JavaScript-based, but bridge overhead and less polished |
| Native (Kotlin/Swift) | Two codebases to maintain, overkill for this project |

### 2. ML Model Serving

| Option | Pros | Cons | Recommendation |
|--------|------|------|---------------|
| **Python sidecar (Flask/FastAPI)** | Simplest to implement, full Python ML ecosystem | Extra process to manage | **Recommended for Phase 1** |
| TFLite on ESP32 | No server dependency, offline inference | Very limited model size, hard to update | Consider for Phase 2 |
| ONNX Runtime in Node.js | Single process, no Python needed | ONNX conversion complexity | Alternative for Phase 2 |

### 3. Push Notification Service

| Option | Recommendation |
|--------|---------------|
| **Firebase Cloud Messaging (FCM)** | **Recommended** — free, reliable, works on both Android and iOS, excellent Flutter support |
| Self-hosted (e.g., ntfy, Gotify) | More control but complex to set up and maintain |

### 4. Database (Phase 2)

| Option | Pros | Cons | Recommendation |
|--------|------|------|---------------|
| **PostgreSQL** | Relational integrity, mature, great for structured device/user data | Schema migrations needed | **Recommended** |
| MongoDB | Flexible schema, natural for sensor JSON data | Weaker relational queries | Alternative |
| SQLite | Zero setup, file-based | Poor for concurrent multi-device writes | Not recommended |

### 5. Deployment (Phase 2)

| Option | Cost | Recommendation |
|--------|------|---------------|
| **DigitalOcean Droplet** | ~$6/month | **Recommended** — simple, affordable, full control |
| AWS EC2 / Lightsail | ~$5-10/month | Alternative — more services available |
| Railway / Render | Free tier available | Good for prototyping, limited free tier |

---

*Last updated: 2026-03-16*
*Project: Fall Detection Smart Wearable Device (PBL5)*
