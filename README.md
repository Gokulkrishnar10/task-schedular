# Configurable Task Scheduler
### AI × ML Pipeline × Google Calendar × Real-Time Monitoring

> Production-grade SaaS platform for intelligent task scheduling, built to demonstrate:
> AI APIs · Custom ML Pipelines · Scalable Backend · Real-Time Data Pipelines · Live Deployment

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env and set:
#   GROQ_API_KEY=your_key_here
#   GOOGLE_CALENDAR_ID=your_calendar_id  (optional)

# 3. Google Calendar setup (optional)
# Download service account JSON from Google Cloud Console
# Save as: backend/service_account.json

# 4. Run
cd backend
python run.py

# 5. Open browser
http://localhost:8000
# API docs
http://localhost:8000/docs
```

---

## Project Structure

```
itoc/
├── backend/
│   ├── main.py                  # FastAPI — all 15 API routes
│   ├── scheduler.py             # APScheduler — configurable background jobs
│   ├── system_monitor.py        # psutil — live CPU/RAM/Disk/Network
│   ├── groq_ai.py               # Groq LLaMA 3.3 70B — 4 AI features
│   ├── ml_pipeline.py           # 3 scikit-learn models — full ML pipeline
│   ├── calendar_integration.py  # Google Calendar API v3 integration
│   └── run.py                   # Entry point
├── frontend/
│   └── index.html               # Production SaaS dashboard (969 lines)
├── requirements.txt
├── railway.toml
└── .env
```

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Backend | FastAPI (Python) | Async REST API, 15 endpoints |
| Scheduler | APScheduler | User-configurable task intervals |
| Monitoring | psutil | Live CPU/RAM/Disk/Network metrics |
| AI / LLM | Groq API — LLaMA 3.3 70B | Priority, prediction, reports, scheduling |
| ML Pipeline | scikit-learn + numpy | 3 custom models on live data |
| Calendar | Google Calendar API v3 | Event creation and sync |
| Database | SQLite (sqlite3) | Tasks, logs, calendar references |
| Frontend | HTML5 + Chart.js | Real-time SaaS dashboard |
| Deployment | Railway.app | Cloud deploy with live URL |

---

## AI Features (Groq LLaMA 3.3 70B)

| Function | Route | What it does |
|---|---|---|
| `suggest_priority()` | POST /api/tasks | Reads task name+desc → returns High/Medium/Low |
| `predict_failure()` | GET /api/tasks/{id}/run | Analyses live CPU/RAM → risk level |
| `generate_health_report()` | GET /api/health-report | Full system health in plain English |
| `suggest_schedule_time()` | POST /api/calendar/sync/{id} | Picks best hour to schedule a task |

---

## ML Pipeline — 3 Models (scikit-learn)

### Model 1 — Isolation Forest (Anomaly Detection)
- **Input**: Rolling buffer of 100 (cpu, ram, disk) readings
- **Algorithm**: IsolationForest with adaptive contamination (1–10%)
- **Output**: "Normal" / "Anomaly Detected" + raw anomaly score
- **Minimum data**: 10 readings (warms up in ~50 seconds at 5s refresh)

### Model 2 — Linear Regression (Resource Forecasting)
- **Input**: Same rolling buffer, time-indexed
- **Algorithm**: Two separate LinearRegression models (CPU and RAM)
- **Output**: Predicted CPU% and RAM% for the next 30-second tick
- **Minimum data**: 5 readings (warms up in ~25 seconds)

### Model 3 — Random Forest Classifier (Task Failure Prediction)
- **Input**: Historical task runs — (cpu, ram, disk, interval_s) → label (0=ok, 1=failure)
- **Algorithm**: RandomForestClassifier with StandardScaler + balanced class weights
- **Output**: "Low / Medium / High ML Risk" + failure probability confidence
- **Minimum data**: 5 runs with both outcomes seen
- **Self-improving**: Learns from every manual and auto-scheduled task run

---

## Google Calendar Integration

### Setup (One-Time)
1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Enable **Google Calendar API**
3. Create a **Service Account** → Download JSON key
4. Save key as `backend/service_account.json`
5. Share your Google Calendar with the service account email  
   (give **"Make changes to events"** permission)
6. Set `GOOGLE_CALENDAR_ID` in `.env` (use `primary` for your main calendar)

### What it does
- **Auto-create events**: Tick "Sync to Calendar" when creating a task → AI picks the best time of day → colour-coded Google Calendar event is created automatically
- **Per-task sync**: Click 📅 Sync on any existing task → same AI scheduling flow
- **Auto-delete**: Deleting a task from the scheduler also removes its calendar event
- **Live view**: The dashboard shows your upcoming task events pulled from Google Calendar
- **Colour coding**: High priority = red, Medium = yellow, Low = green

### Graceful degradation
If `service_account.json` is missing, the entire app still works — calendar routes return a clear error message, all other features are unaffected.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | /api/metrics | Live CPU, RAM, Disk, Network |
| GET | /api/metrics/history | Last 20 readings for chart |
| GET | /api/ml/pipeline | Fresh 3-model ML run |
| GET | /api/ml/status | Cached ML output (fast) |
| GET | /api/tasks | List all scheduled tasks |
| POST | /api/tasks | Create task (AI priority + optional calendar) |
| DELETE | /api/tasks/{id} | Delete task + cancel APScheduler job |
| GET | /api/tasks/{id}/run | Manual trigger + dual AI+ML risk check |
| GET | /api/health-report | AI health report (Groq + ML) |
| GET | /api/logs | Last 50 execution logs |
| GET | /api/stats | Dashboard summary stats |
| GET | /api/calendar/status | Is Google Calendar configured? |
| GET | /api/calendar/events | Upcoming task events |
| POST | /api/calendar/sync/{id} | Sync task → Calendar event |
| DELETE | /api/calendar/delete/{eid} | Remove calendar event |

---

## Deploy to Railway

```bash
# 1. Push to GitHub
git init && git add . && git commit -m "Configurable Task Scheduler v3"
git remote add origin https://github.com/YOUR_USERNAME/task-scheduler
git push origin main

# 2. Go to railway.app → Login with GitHub
# 3. New Project → Deploy from GitHub
# 4. Add environment variables:
#    GROQ_API_KEY=your_key
#    GOOGLE_CALENDAR_ID=your_calendar_id (optional)
# 5. Get live URL → add to submission
```

---

## Requirements Coverage Map

| Requirement | How This Project Meets It |
|---|---|
| **Configurable Task Scheduler** | APScheduler with user-defined intervals (any seconds value), priority levels, manual triggers |
| **AI dashboards** | Real-time Chart.js dashboard — CPU/RAM/Disk auto-refresh every 5s |
| **Industrial monitoring tools** | Live system metrics via psutil — CPU, RAM, Disk, Network, Boot time |
| **AI API integration** | Groq LLaMA 3.3 70B — 4 AI features (priority, risk, report, scheduling) |
| **Custom ML pipelines** | 3 scikit-learn models running on live data — Isolation Forest, Linear Regression, Random Forest |
| **Scalable backend systems** | FastAPI async microservice architecture — 15 endpoints, modular files |
| **Real-time data pipelines** | psutil → push_metrics() → ML buffer → Chart.js → 5s auto-refresh |
| **Production-ready applications** | SQLite persistence, error handling, graceful degradation, Railway deploy |
| **Live deployments** | Railway.app config included — deploy in 5 minutes |
| **Scalable architecture** | Modular: monitor → scheduler → ML → AI → calendar — each replaceable |
| **EV/Industrial monitoring** | Same architecture applies directly to sensor data, fleet telemetry, factory metrics |
| **1000+ user scalability** | FastAPI + async routes — handles concurrent requests; swap SQLite → PostgreSQL for scale |

---

*Built for: Intellifer Systems — Track 2: AI Full Stack Product*
