# 🌾 SkyView AI — FPGA-Accelerated Edge AI for Smart Agriculture

An end-to-end intelligent agriculture system for Indian farmers, combining real-time IoT sensing, on-device agentic AI running on the Arduino Q (Qualcomm MPU), FPGA hardware-accelerated sensor fusion, a multilingual React dashboard, and a native Flutter mobile application with voice-first field controls.

Built for **Arduino Physical AI Challenge India 2026**.

---

## What It Does

- **Smart weather station** — ESP32 sensors (temperature, humidity, soil moisture, UV, PM2.5) transmit via LoRa to the Arduino UNO Q receiver
- **Arduino Q — Agentic AI Gateway** — the central hub running on the Qualcomm MPU. Hosts the FastAPI backend, database, frontend, and the entire multi-agent AI workflow on-device using a local LLM. Receives sensor data via LoRa, FPGA results via UART, and orchestrates all decision-making locally
- **FPGA hardware acceleration** — AMD ZYNQ-7000 FPGA connected to Arduino Q via UART runs dedicated IP cores for sensor fusion and rain prediction with parallel ML inference at hardware speed
- **On-device agentic AI** — multi-agent system (Weather Agent, Farm Advisor Agent, Alert Agent, Decision Engine) runs entirely on the Arduino Q's Qualcomm MPU, powered by an on-device LLM. Cloud Groq API is fallback only
- **Mandi rates** — live commodity prices from data.gov.in with 30-minute caching, MSP history, and buyer matching
- **Multilingual Mobile App** — a cross-platform Flutter application providing field-ready telemetry charts, mandi price lookups, AI advisors, scheme recommendations, and hands-free voice control
- **Voice interaction** — farmers can call in via Vapi.ai or WhatsApp (Twilio) and get AI responses in Hindi/regional languages via Sarvam AI TTS
- **Offline resilience** — Arduino Q buffers sensor data locally during connectivity loss and batch-uploads when back online
- **Government schemes** — AI-curated scheme recommendations based on farmer profile (land size, location, crops)

---

## Quick Start

### Docker (recommended)

```bash
cp .env.example .env
# Fill in your API keys in .env (at minimum: GROQ_API_KEYS)

docker compose up --build
```

- API: http://localhost:8000
- Docs: http://localhost:8000/docs
- Admin: http://localhost:8000/admin/tables
- DB Explorer UI: http://localhost:5173/db

### Local (Linux / macOS)

```bash
cp .env.example .env
# Edit .env

chmod +x run.sh
./run.sh
```

### Frontend (separate terminal)

```bash
cp .env.example .env.local   # frontend env
npm install
npm run dev
```

Frontend: http://localhost:5173

> **Note:** In dev mode, Vite proxies all `/api` requests to `localhost:8000` automatically. No `VITE_API_URL` needed locally.

---

## Project Structure

```
.
├── skyview/                    # Python backend package
│   ├── main.py                 # FastAPI app + router registration
│   ├── api/                    # Route handlers (one file per domain)
│   │   ├── core_routes.py      # GET /, /health, /api/llm-pool
│   │   ├── sensor_routes.py    # Sensor ingest + history
│   │   ├── auth_routes.py      # OTP login / signup
│   │   ├── chat_routes.py      # AI chat + weather insight + farm Q&A
│   │   ├── advisor_routes.py   # Category-based AI insights
│   │   ├── mandi_routes.py     # Live commodity prices
│   │   ├── profile_routes.py   # Farmer profile + govt schemes
│   │   ├── marketplace_routes.py # Buyer/seller matching
│   │   ├── voice_routes.py     # Vapi + Sarvam TTS + translation
│   │   ├── fpga_routes.py      # Hardware accelerator endpoints
│   │   └── webhook_routes.py   # Twilio WhatsApp webhooks
│   ├── agents/                 # AI agents
│   │   ├── supervisor.py       # Multi-agent orchestrator
│   │   ├── mandi_agent.py      # Live price fetcher + cache
│   │   ├── fpga_agent.py       # FPGA software simulation
│   │   └── agricultural_qa.py  # Crop Q&A engine
│   ├── admin/
│   │   └── admin_routes.py     # DB browser + analytics
│   ├── data/                   # Database layer
│   │   ├── db.py               # Engine + session factory
│   │   ├── schema.py           # Table creation (IF NOT EXISTS)
│   │   ├── queries.py          # Reusable query helpers
│   │   ├── seed.py             # Full seed data
│   │   └── seed_fast.py        # Quick seed for dev/CI
│   └── utils/
│       ├── config.py           # All env vars (Settings class)
│       ├── llm_pool.py         # Round-robin Groq key balancer
│       └── logger.py           # Structured logging
│
├── skyview_flutter_app/        # Flutter Mobile Application
│   └── skyview_flutter/        # Flutter source, assets, translations
│
├── src/                        # React frontend
│   ├── pages/                  # Route-level page components
│   ├── components/             # Shared UI components
│   ├── context/
│   │   ├── AuthContext.tsx      # OTP auth state
│   │   └── LanguageContext.tsx  # i18n (en/hi/te/...)
│   └── lib/
│       └── weatherData.ts      # Backend data fetching + mapping
│
├── hardware/                   # FPGA hardware design
│   ├── sensor_fusion.cpp       # HLS sensor fusion (C++)
│   ├── rain_predictor.cpp      # HLS rain prediction (C++)
│   ├── sensor_fusion.v         # RTL (Verilog)
│   └── sensor_fusion_axi.v     # AXI4-Lite interface
│
├── tests/                      # pytest test suite
│   ├── conftest.py             # Fixtures (TestClient + SQLite DB)
│   ├── test_api.py             # All API endpoint tests
│   └── test_ingestion.py       # Sensor ingestion pipeline tests
│
├── infra/                      # Infrastructure
│   ├── Dockerfile.backend      # Python 3.11 + uvicorn
│   ├── Dockerfile.frontend     # Node 20 build + nginx serve
│   ├── requirements.txt        # Python dependencies (pinned)
│   ├── mosquitto.conf          # MQTT broker config
│   └── nginx.conf              # Reverse proxy + SPA fallback
│
├── docker-compose.yml          # db + mqtt + backend services
├── run.sh                      # Local dev runner
├── .env.example                # All environment variables
└── test_all_routes.py          # End-to-end smoke tests (live server)
```

---

## API Reference

All routes are documented interactively at `/docs` (Swagger UI) when the server is running.

| Domain | Method | Path | Description |
|--------|--------|------|-------------|
| System | GET | `/` | Service info |
| System | GET | `/health` | DB + LLM pool health |
| Sensors | POST | `/api/sensors/data` | Ingest sensor reading (hardware or flat JSON) |
| Sensors | GET | `/api/sensors/latest/{id}` | Latest reading for a station |
| Sensors | GET | `/api/sensors/history/{id}` | Historical readings (`?hours=24&limit=200`) |
| Sensors | GET | `/api/sensors/stations` | List all registered stations |
| Sensors | POST | `/api/sensors/trends/store` | Store trends snapshot |
| Auth | POST | `/api/auth/send-otp` | Send OTP (returns OTP in dev mode) |
| Auth | POST | `/api/auth/verify-otp` | Verify OTP → token |
| Auth | POST | `/api/auth/signup` | Register new farmer |
| Chat | POST | `/api/chat` | Multi-agent conversational AI |
| Chat | POST | `/api/weather-insight` | Weather-based farm advice |
| Chat | POST | `/api/agriculture/qa` | Structured crop Q&A |
| Advisor | POST | `/api/advisor/insights` | AI insights by category |
| Mandi | GET | `/api/mandi/rates` | Live commodity prices (`?state=&commodity=`) |
| Mandi | GET | `/api/mandi/commodities` | Known commodity list |
| Mandi | GET | `/api/mandi/history` | Historical mandi data |
| Mandi | GET | `/api/mandi/msp` | Minimum support prices |
| Profile | GET | `/api/profile` | Fetch farmer profile (`?phone=`) |
| Profile | POST | `/api/profile/save` | Create / update profile |
| Schemes | GET | `/api/schemes` | Full government scheme catalog |
| Schemes | GET | `/api/schemes/recommendations` | AI govt scheme recommendations |
| Marketplace | POST | `/api/marketplace/match` | Find buyers for produce |
| Voice | POST | `/api/vapi/call` | Trigger outbound voice call |
| Voice | POST | `/api/speech/synthesize` | Text → speech (Sarvam AI) |
| Voice | POST | `/api/translate` | Translate texts to target language |
| Voice | POST | `/api/profile/voice-update` | Update profile from voice transcript |
| FPGA | GET | `/api/fpga/status` | Hardware accelerator status |
| FPGA | POST | `/api/fpga/fusion` | Sensor fusion computation |
| FPGA | POST | `/api/fpga/rain-predict` | Rain probability prediction |
| FPGA | POST | `/api/fpga/combined-analysis` | Full combined analysis |
| Edge AI | GET | `/api/edge/status` | Arduino Q gateway connectivity |
| Edge AI | GET | `/api/edge/health` | Detailed gateway health |
| Edge AI | GET | `/api/edge/data-quality` | Data quality analytics |
| Edge AI | POST | `/api/edge/alert-thresholds` | Push alert config to Arduino Q MCU |
| Sensors | POST | `/api/sensors/data/batch` | Batch ingest (store-and-forward) |
| Webhooks | GET | `/webhook/whatsapp` | Twilio verification handshake |
| Webhooks | POST | `/webhook/whatsapp` | Inbound WhatsApp message handler |
| Admin | GET | `/admin/tables` | List all DB tables |
| Admin | GET | `/admin/tables/{table}` | Browse table data (paginated) |
| Admin | GET | `/admin/analytics/overview` | System-wide stats |

---

## Environment Variables

Copy `.env.example` to `.env` and fill in:

| Variable | Required | Description |
|----------|----------|-------------|
| `DATABASE_URL` | Recommended | PostgreSQL connection string. Falls back to SQLite if unset. |
| `GROQ_API_KEYS` | Fallback only | Comma-separated Groq API keys — used only when Arduino Q on-device LLM is unavailable |
| `DATAGOV_API_KEY` | Yes (live mandi) | data.gov.in API key (free registration) |
| `VAPI_AI_API_KEY` | Optional | Vapi.ai key for outbound voice calls |
| `VAPI_ASSISTANT_ID` | Optional | Vapi assistant ID |
| `SARVAM_AI_API_KEY` | Optional | Sarvam AI key for Indian language TTS + translation |
| `TWILIO_ACCOUNT_SID` | Optional | Twilio SID for WhatsApp integration |
| `TWILIO_AUTH_TOKEN` | Optional | Twilio auth token |
| `TWILIO_WHATSAPP_NUMBER` | Optional | Twilio WhatsApp sender number |
| `ENABLE_FPGA` | No | `True` when AMD ZYNQ-7000 FPGA is connected via UART |
| `FPGA_PORT` | No | Serial port for FPGA bridge (`/dev/ttyUSB0` or `COM4`) |
| `FPGA_UART_PORT` | No | UART port for FPGA on Arduino Q (default: `/dev/ttyPS1`) |
| `FPGA_UART_BAUD` | No | FPGA UART baud rate (default: `115200`) |
| `ENABLE_EDGE_AI` | Yes (production) | `True` when running on Arduino Q with on-device LLM |
| `ARDUINO_Q_HOST` | No | Arduino Q hostname (default: `arduinoq.local`) |
| `ARDUINO_Q_PORT` | No | Arduino Q server port (default: `8080`) |
| `ARDUINO_Q_LLM_ENDPOINT` | No | On-device LLM API endpoint (default: `http://arduinoq.local:8080/v1/chat/completions`) |
| `ARDUINO_Q_LLM_MODEL` | No | Model running on Qualcomm MPU (default: `llama-3.2-1b`) |
| `ARDUINO_Q_LLM_TIMEOUT` | No | On-device inference timeout in seconds (default: `60`) |
| `EDGE_AI_FALLBACK_TO_CLOUD` | No | Fall back to cloud Groq when edge LLM fails (default: `True`) |
| `EDGE_ALERT_THRESHOLDS` | No | JSON thresholds for MCU real-time alerts |
| `MQTT_BROKER` | No | MQTT broker host (default: `localhost`) |
| `STATION_ID` | No | Default station ID (default: `WS01`) |

---

## Running Tests

```bash
# Unit tests (no live server needed — uses TestClient + SQLite)
pip install -r infra/requirements.txt
pytest tests/ -v

# End-to-end smoke tests (requires live server on :8000)
uvicorn skyview.main:app --port 8000 &
python test_all_routes.py
```

---

## Sending Sensor Data

### From hardware (ESP32 → Arduino Q → HTTP)

```json
POST /api/sensors/data
{
  "id": "WS01",
  "ts": 1700000000,
  "env": {"t": 28.5, "h": 65.0, "p": 1012.0},
  "wind": {"s": 3.2, "d": "NE"},
  "rain": 0.5,
  "soil": {"t": 27.0, "m": 52.0},
  "air": {"pm25": 55.0, "pm10": 110.0},
  "rad": {"uv": 4.0, "lux": 35000.0},
  "pwr": {"bat": 3.8, "sol": 5.1},
  "edge_ai": {
    "fusion_score": 72,
    "stress_index": 35,
    "rain_probability": 0.42,
    "model_version": "v1.2",
    "inference_ms": 12
  },
  "edge_flags": {
    "anomaly_score": 0.08,
    "data_quality": "good"
  }
}
```

> **Note:** The `edge_ai` and `edge_flags` fields are optional. Legacy payloads without them still work. The Arduino Q gateway adds these when edge AI models are deployed.

### Flat format (scripts / testing)

```json
POST /api/sensors/data
{
  "station_id": "WS01",
  "temperature": 28.5,
  "humidity": 65.0,
  "pressure": 1012.0,
  "wind_speed": 3.2,
  "wind_direction": "NE",
  "rainfall": 0.5,
  "soil_temperature": 27.0,
  "soil_moisture": 52.0,
  "pm25": 55.0,
  "pm10": 110.0,
  "uv_index": 4.0,
  "lux": 35000.0,
  "battery_voltage": 3.8,
  "solar_voltage": 5.1
}
```

---

## Seeding the Database

```bash
# Full seed (mandi history, MSP data, sample stations + readings)
python -m skyview.data.seed

# Fast seed (minimal data, good for CI)
python -m skyview.data.seed_fast
```

---

## Adding a New Route

1. Create `skyview/api/my_feature_routes.py`
2. Define a `router = APIRouter(prefix="/api/my-feature", tags=["My Feature"])`
3. Add your endpoints
4. Register in `skyview/main.py`:

```python
_register("skyview.api.my_feature_routes")
```

5. Add tests in `tests/test_api.py`

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.11, FastAPI, SQLAlchemy 2.0 (runs on Arduino Q) |
| On-Device AI | Arduino Q (Qualcomm MPU), on-device LLM (Llama 3.2 via llama.cpp) |
| Cloud AI (Fallback) | Groq API (Llama 3.1-8b-instant), multi-key load balancing |
| Database | PostgreSQL 16 (production), SQLite (dev fallback) |
| IoT | ESP32, LoRa, MQTT (Eclipse Mosquitto) |
| Agentic AI Gateway | Arduino UNO Q (Qualcomm MPU + STM32 MCU) — central hub |
| FPGA Accelerator | AMD ZYNQ-7000, sensor fusion & rain prediction IP cores, UART bridge to Arduino Q |
| Voice | Vapi.ai (calls), Sarvam AI (TTS + translation), Twilio (WhatsApp) |
| Frontend | React 18, Vite, TypeScript, Tailwind CSS, Recharts |
| Mobile App | Flutter, Riverpod, Multilingual Chat, Telemetry charts, Voice-first controls |
| Infrastructure | Docker, docker-compose, nginx, uvicorn |
| Testing | pytest, FastAPI TestClient, httpx |

---

## 📱 Mobile Application

A cross-platform Flutter application tailored for farmers, serving as a multilingual, voice-interactive client for the SkyView platform:
* **Multilingual Crop Advisory**: Field-ready access to specialized AI advisors.
* **Telemetry Dashboards**: Visualizes real-time ESP32 weather station readings directly on the phone.
* **Mandi Rate Tracker**: Easy mobile access to data.gov.in commodity rates and MSP history.

For setup, architecture, and developer best practices, refer to the [Mobile App Developer Guide](file:///c:/Users/vinay/OneDrive/Desktop/troubleshooters/Google_Solution_challenge/APP_DEVELOPMENT.md).

---

## License

Arduino Physical AI Challenge India 2026. MIT License.
