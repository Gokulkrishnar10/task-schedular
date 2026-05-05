"""
scheduler.py — APScheduler Background Job Runner
=================================================
Configurable Task Scheduler Application

Every task registered here runs at user-defined intervals.
On each tick it:
  1. Reads live CPU / RAM via psutil
  2. Runs the ML pipeline (all 3 models)
  3. Determines status (Completed / Warning)
  4. Updates the tasks table and appends to task_logs
  5. Feeds the run outcome back into the ML classifier buffer
"""

import sqlite3
import datetime
import psutil
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval    import IntervalTrigger

from ml_pipeline import run_full_pipeline, add_task_run_to_history

scheduler = BackgroundScheduler()
DB_PATH   = "itoc.db"


def _execute_task(task_id: int, task_name: str, interval_seconds: int) -> None:
    """
    APScheduler callback — runs in a background thread on every interval tick.
    Kept synchronous (no async) because APScheduler's BackgroundScheduler
    runs in a plain thread pool, not an asyncio event loop.
    """
    cpu  = psutil.cpu_percent(interval=0.3)
    ram  = psutil.virtual_memory().percent
    disk = psutil.disk_usage("/").percent
    now  = datetime.datetime.now().isoformat()
    next_run = (
        datetime.datetime.now() + datetime.timedelta(seconds=interval_seconds)
    ).isoformat()

    # ── ML pipeline (all 3 models, synchronous) ──────────────────────────────
    ml  = run_full_pipeline(cpu, ram, disk, interval_seconds)
    anomaly_label = ml["anomaly_label"]
    ml_risk       = ml["classifier_risk"]

    # ── Threshold-based AI risk (no async Groq here) ─────────────────────────
    if cpu > 85 or ram > 90:
        ai_risk = "High Risk"
    elif cpu > 60 or ram > 70:
        ai_risk = "Medium Risk"
    else:
        ai_risk = "Low Risk"

    # ── Combine signals to decide final status ───────────────────────────────
    warn = (
        "High" in ai_risk
        or "High" in ml_risk
        or anomaly_label == "Anomaly Detected"
    )
    status = "Warning" if warn else "Completed"

    # ── Persist ──────────────────────────────────────────────────────────────
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()

    c.execute(
        """UPDATE tasks
           SET status=?, last_run=?, next_run=?,
               run_count=run_count+1, ai_risk=?, ml_risk=?
           WHERE id=?""",
        (status, now, next_run, ai_risk, ml_risk, task_id),
    )
    c.execute(
        """INSERT INTO task_logs
           (task_id, task_name, status, message,
            cpu_at_run, ram_at_run, anomaly_label, ml_risk, timestamp)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        (
            task_id, task_name, status,
            f"Auto-run | AI: {ai_risk} | ML: {ml_risk} | Anomaly: {anomaly_label}",
            cpu, ram, anomaly_label, ml_risk, now,
        ),
    )
    conn.commit()
    conn.close()

    # Feed result back into ML classifier buffer
    add_task_run_to_history(cpu, ram, disk, interval_seconds, status)


def add_scheduled_task(task_id: int, task_name: str, interval_seconds: int) -> None:
    job_id = f"task_{task_id}"
    if scheduler.get_job(job_id):
        scheduler.remove_job(job_id)
    scheduler.add_job(
        _execute_task,
        trigger=IntervalTrigger(seconds=interval_seconds),
        args=[task_id, task_name, interval_seconds],
        id=job_id,
        replace_existing=True,
        misfire_grace_time=30,
    )


def remove_scheduled_task(task_id: int) -> None:
    job_id = f"task_{task_id}"
    if scheduler.get_job(job_id):
        scheduler.remove_job(job_id)


def get_active_job_count() -> int:
    return len(scheduler.get_jobs())
