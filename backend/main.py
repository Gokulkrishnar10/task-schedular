"""
main.py — FastAPI Application Entry Point
=========================================
Configurable Task Scheduler Application v3.0

Route map:
  /                          → Dashboard HTML
  /api/metrics               → Live psutil system metrics
  /api/metrics/history       → Rolling 20-point history for chart
  /api/ml/pipeline           → Fresh ML pipeline run
  /api/ml/status             → Cached ML output (no recompute)
  /api/tasks                 → GET list / POST create
  /api/tasks/{id}            → DELETE
  /api/tasks/{id}/run        → Manually trigger + dual AI check
  /api/health-report         → Groq AI health report
  /api/logs                  → Last 50 execution logs
  /api/stats                 → Dashboard summary numbers
  /api/calendar/status       → Is Google Calendar configured?
  /api/calendar/events       → Upcoming calendar events
  /api/calendar/sync/{id}    → Sync one task → Google Calendar event
  /api/calendar/delete/{eid} → Delete a calendar event
  /api/chat                  → Conversational AI agent for task creation
"""

import sqlite3
import datetime
import os
from fastapi            import FastAPI, HTTPException
from fastapi.middleware.cors   import CORSMiddleware
from fastapi.responses  import FileResponse
from pydantic           import BaseModel
from typing             import Optional

from system_monitor       import get_system_metrics, get_metrics_history
from groq_ai              import suggest_priority, predict_failure, generate_health_report, suggest_schedule_time, chat_agent_turn
from scheduler            import scheduler, add_scheduled_task, remove_scheduled_task, get_active_job_count
from ml_pipeline          import run_full_pipeline, get_pipeline_output, add_task_run_to_history, get_task_history_count, push_metrics
from calendar_integration import (
    create_task_event, delete_task_event,
    list_upcoming_events, is_calendar_configured,
)

app = FastAPI(
    title       = "Configurable Task Scheduler",
    description = "AI-powered task scheduler with ML pipeline and Google Calendar sync",
    version     = "3.0.0",
)
scheduler.start()
print("Scheduler started successfully")

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

DB_PATH = "itoc.db"


# ── Database Initialisation ───────────────────────────────────────────────────

def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            name             TEXT    NOT NULL,
            description      TEXT    DEFAULT '',
            interval_seconds INTEGER DEFAULT 30,
            priority         TEXT    DEFAULT 'Medium',
            status           TEXT    DEFAULT 'Pending',
            last_run         TEXT,
            next_run         TEXT,
            run_count        INTEGER DEFAULT 0,
            created_at       TEXT    DEFAULT CURRENT_TIMESTAMP,
            ai_risk          TEXT    DEFAULT 'Unknown',
            ml_risk          TEXT    DEFAULT 'Insufficient Data',
            calendar_event_id TEXT   DEFAULT NULL,
            calendar_link     TEXT   DEFAULT NULL
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS task_logs (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id       INTEGER,
            task_name     TEXT,
            status        TEXT,
            message       TEXT,
            cpu_at_run    REAL,
            ram_at_run    REAL,
            anomaly_label TEXT,
            ml_risk       TEXT,
            timestamp     TEXT    DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Add calendar columns to existing DBs without them (safe migration)
    for col, typedef in [("calendar_event_id", "TEXT DEFAULT NULL"), ("calendar_link", "TEXT DEFAULT NULL")]:
        try:
            c.execute(f"ALTER TABLE tasks ADD COLUMN {col} {typedef}")
        except sqlite3.OperationalError:
            pass  # column already exists

    conn.commit()
    conn.close()

init_db()


# ── Pydantic Models ───────────────────────────────────────────────────────────

class TaskCreate(BaseModel):
    name:             str
    description:      Optional[str] = ""
    interval_seconds: Optional[int] = 30
    sync_to_calendar: Optional[bool] = False   # ← user can opt-in per task


class ChatMessage(BaseModel):
    role:    str   # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]  # full conversation history from client


# ── Static ────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def root():
    return FileResponse("../frontend/index.html")


# ── System Metrics ────────────────────────────────────────────────────────────

@app.get("/api/metrics", tags=["Monitoring"])
def get_metrics():
    """Live CPU / RAM / Disk / Network snapshot."""
    m = get_system_metrics()
    push_metrics(m["cpu"], m["ram"], m["disk"])
    return m


@app.get("/api/metrics/history", tags=["Monitoring"])
def metrics_history():
    """Last 20 readings for the real-time chart."""
    return get_metrics_history()


# ── ML Pipeline ───────────────────────────────────────────────────────────────

@app.get("/api/ml/pipeline", tags=["ML"])
def ml_pipeline():
    """Runs a fresh pass of all 3 ML models against current live metrics."""
    m      = get_system_metrics()
    result = run_full_pipeline(m["cpu"], m["ram"], m["disk"])
    result["current_metrics"] = m
    return result


@app.get("/api/ml/status", tags=["ML"])
def ml_status():
    """Returns the last cached ML output — fast, no recompute."""
    output = get_pipeline_output()
    output["task_history_count"] = get_task_history_count()
    return output


# ── Tasks ─────────────────────────────────────────────────────────────────────

@app.get("/api/tasks", tags=["Tasks"])
def list_tasks():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c    = conn.cursor()
    c.execute("SELECT * FROM tasks ORDER BY created_at DESC")
    tasks = [dict(row) for row in c.fetchall()]
    conn.close()
    return tasks


@app.post("/api/tasks", tags=["Tasks"])
async def create_task(task: TaskCreate):
    """
    Creates a task:
      1. Groq AI suggests priority
      2. Task is saved to DB + registered with APScheduler
      3. If sync_to_calendar=true:
           - Groq AI suggests best run time
           - Google Calendar event is created
    """
    priority = await suggest_priority(task.name, task.description)
    next_run = (
        datetime.datetime.now() + datetime.timedelta(seconds=task.interval_seconds)
    ).isoformat()

    calendar_event_id = None
    calendar_link     = None
    calendar_info     = {}

    if task.sync_to_calendar:
        ai_time   = await suggest_schedule_time(task.name, task.description, priority)
        cal_result = create_task_event(
            task_name        = task.name,
            description      = task.description,
            priority         = priority,
            interval_seconds = task.interval_seconds,
            suggested_hour   = ai_time["suggested_hour"],
            suggested_minute = ai_time["suggested_minute"],
            ai_reason        = ai_time["reason"],
        )
        if not cal_result.get("error"):
            calendar_event_id = cal_result["event_id"]
            calendar_link     = cal_result["event_link"]
        calendar_info = cal_result

    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute(
        """INSERT INTO tasks
           (name, description, interval_seconds, priority, status,
            next_run, calendar_event_id, calendar_link)
           VALUES (?,?,?,?,'Scheduled',?,?,?)""",
        (task.name, task.description, task.interval_seconds, priority,
         next_run, calendar_event_id, calendar_link),
    )
    task_id = c.lastrowid
    conn.commit()
    conn.close()

    add_scheduled_task(task_id, task.name, task.interval_seconds)

    return {
        "id":              task_id,
        "name":            task.name,
        "priority":        priority,
        "status":          "Scheduled",
        "calendar":        calendar_info,
    }


@app.delete("/api/tasks/{task_id}", tags=["Tasks"])
def delete_task(task_id: int):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c    = conn.cursor()
    c.execute("SELECT * FROM tasks WHERE id=?", (task_id,))
    row  = c.fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Task not found")

    # Delete from Google Calendar if synced
    if row["calendar_event_id"]:
        delete_task_event(row["calendar_event_id"])

    c.execute("DELETE FROM tasks WHERE id=?", (task_id,))
    conn.commit()
    conn.close()
    remove_scheduled_task(task_id)
    return {"message": "Task deleted"}


@app.get("/api/tasks/{task_id}/run", tags=["Tasks"])
async def manual_run(task_id: int):
    """
    Manually trigger a task:
      1. Capture live metrics
      2. Groq AI risk prediction (async)
      3. ML pipeline (3 models)
      4. Determine final status (dual AI vote)
      5. Persist update + log
      6. Feed result into ML training buffer
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c    = conn.cursor()
    c.execute("SELECT * FROM tasks WHERE id=?", (task_id,))
    task = c.fetchone()
    if not task:
        conn.close()
        raise HTTPException(status_code=404, detail="Task not found")

    m         = get_system_metrics()
    groq_risk = await predict_failure(m)
    ml_result = run_full_pipeline(m["cpu"], m["ram"], m["disk"], task["interval_seconds"])

    anomaly_label = ml_result["anomaly_label"]
    ml_risk       = ml_result["classifier_risk"]

    warn = (
        "High" in groq_risk
        or "High" in ml_risk
        or anomaly_label == "Anomaly Detected"
    )
    status   = "Warning" if warn else "Completed"
    now      = datetime.datetime.now().isoformat()
    next_run = (
        datetime.datetime.now() + datetime.timedelta(seconds=task["interval_seconds"])
    ).isoformat()

    c.execute(
        """UPDATE tasks
           SET status=?, last_run=?, next_run=?,
               run_count=run_count+1, ai_risk=?, ml_risk=?
           WHERE id=?""",
        (status, now, next_run, groq_risk, ml_risk, task_id),
    )
    c.execute(
        """INSERT INTO task_logs
           (task_id, task_name, status, message,
            cpu_at_run, ram_at_run, anomaly_label, ml_risk, timestamp)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        (
            task_id, task["name"], status,
            f"Manual | Groq: {groq_risk} | ML: {ml_risk} | Anomaly: {anomaly_label}",
            m["cpu"], m["ram"], anomaly_label, ml_risk, now,
        ),
    )
    conn.commit()
    add_task_run_to_history(m["cpu"], m["ram"], m["disk"], task["interval_seconds"], status)
    conn.close()

    return {
        "message":     "Task executed",
        "status":      status,
        "groq_risk":   groq_risk,
        "ml_pipeline": ml_result,
        "metrics":     m,
    }


# ── Health Report ─────────────────────────────────────────────────────────────

@app.get("/api/health-report", tags=["AI"])
async def health_report():
    m      = get_system_metrics()
    ml     = get_pipeline_output()
    report = await generate_health_report(m, ml)
    return {"report": report, "metrics": m, "ml_summary": ml}


# ── Logs ──────────────────────────────────────────────────────────────────────

@app.get("/api/logs", tags=["Monitoring"])
def get_logs():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c    = conn.cursor()
    c.execute("SELECT * FROM task_logs ORDER BY timestamp DESC LIMIT 50")
    logs = [dict(row) for row in c.fetchall()]
    conn.close()
    return logs


# ── Stats ─────────────────────────────────────────────────────────────────────

@app.get("/api/stats", tags=["Monitoring"])
def get_stats():
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute("SELECT COUNT(*) FROM tasks");                          total         = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM tasks WHERE status='Completed'"); completed     = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM tasks WHERE priority='High'");    high_priority = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM task_logs");                      total_runs    = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM tasks WHERE calendar_event_id IS NOT NULL"); cal_synced = c.fetchone()[0]
    conn.close()

    ml = get_pipeline_output()
    return {
        "total_tasks":       total,
        "completed":         completed,
        "high_priority":     high_priority,
        "total_runs":        total_runs,
        "calendar_synced":   cal_synced,
        "active_jobs":       get_active_job_count(),
        # ML fields
        "anomaly_label":     ml.get("anomaly_label",    "Warming Up"),
        "ml_data_points":    ml.get("data_points",      0),
        "forecast_cpu":      ml.get("forecast_cpu"),
        "forecast_ram":      ml.get("forecast_ram"),
        "classifier_risk":   ml.get("classifier_risk",  "Insufficient Data"),
        "anomaly_score":     ml.get("anomaly_score",    0.0),
        "model_status":      ml.get("model_status",     "initializing"),
    }


# ── Google Calendar Routes ────────────────────────────────────────────────────

@app.get("/api/calendar/status", tags=["Calendar"])
def calendar_status():
    """Check whether Google Calendar is set up correctly."""
    return is_calendar_configured()


@app.get("/api/calendar/events", tags=["Calendar"])
def calendar_events():
    """Return the next 10 upcoming task-scheduler events from Google Calendar."""
    return list_upcoming_events(max_results=10)


@app.post("/api/calendar/sync/{task_id}", tags=["Calendar"])
async def sync_task_to_calendar(task_id: int):
    """
    Sync an existing task to Google Calendar.
    AI picks the best time; a colour-coded event is created.
    The calendar_event_id is stored in the tasks table.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c    = conn.cursor()
    c.execute("SELECT * FROM tasks WHERE id=?", (task_id,))
    task = c.fetchone()
    if not task:
        conn.close()
        raise HTTPException(status_code=404, detail="Task not found")

    ai_time    = await suggest_schedule_time(task["name"], task["description"], task["priority"])
    cal_result = create_task_event(
        task_name        = task["name"],
        description      = task["description"],
        priority         = task["priority"],
        interval_seconds = task["interval_seconds"],
        suggested_hour   = ai_time["suggested_hour"],
        suggested_minute = ai_time["suggested_minute"],
        ai_reason        = ai_time["reason"],
    )

    if not cal_result.get("error"):
        c.execute(
            "UPDATE tasks SET calendar_event_id=?, calendar_link=? WHERE id=?",
            (cal_result["event_id"], cal_result["event_link"], task_id),
        )
        conn.commit()

    conn.close()
    return {"task_id": task_id, "ai_schedule": ai_time, "calendar": cal_result}


@app.delete("/api/calendar/delete/{event_id}", tags=["Calendar"])
def delete_calendar_event(event_id: str):
    """Delete a specific Google Calendar event by event ID."""
    result = delete_task_event(event_id)
    if result.get("deleted"):
        # Clear the reference in DB
        conn = sqlite3.connect(DB_PATH)
        c    = conn.cursor()
        c.execute("UPDATE tasks SET calendar_event_id=NULL, calendar_link=NULL WHERE calendar_event_id=?", (event_id,))
        conn.commit()
        conn.close()
    return result


# ── AI Chat Agent ────────────────────────────────────────────────────────────

@app.post("/api/chat", tags=["AI"])
async def chat_endpoint(req: ChatRequest):
    """
    Conversational AI agent that:
      - Asks clarifying questions to gather task details
      - Returns a CONFIRM payload when it has enough info
      - Frontend then auto-creates the task via POST /api/tasks
    """
    history = [{"role": m.role, "content": m.content} for m in req.messages]
    result  = await chat_agent_turn(history)
    return result


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    scheduler.start()
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
