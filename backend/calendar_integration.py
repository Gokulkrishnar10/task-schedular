"""
calendar_integration.py — Google Calendar Integration
======================================================
Configurable Task Scheduler Application

Uses Google Calendar API v3 via a Service Account (no browser OAuth needed).
Each task run can be optionally synced to a Google Calendar event so that
operations managers see the schedule in their existing calendar tool.

SETUP (one-time, done by the user):
  1. Go to Google Cloud Console → APIs & Services → Enable "Google Calendar API"
  2. Create a Service Account → download JSON key → save as:
       backend/service_account.json
  3. Share your Google Calendar with the service account email
     (give it "Make changes to events" permission)
  4. Set GOOGLE_CALENDAR_ID in .env (default: primary)

If service_account.json is missing the module degrades gracefully —
all routes still work, calendar endpoints return a clear error message
instead of crashing the server.
"""

import os
import datetime
from typing import Optional

# ── Optional Google API imports ───────────────────────────────────────────────
try:
    from google.oauth2              import service_account
    from googleapiclient.discovery  import build
    _GOOGLE_OK = True
except ImportError:
    _GOOGLE_OK = False

SCOPES              = ["https://www.googleapis.com/auth/calendar"]
SERVICE_ACCOUNT_FILE = os.path.join(os.path.dirname(__file__), "service_account.json")
CALENDAR_ID         = os.getenv("GOOGLE_CALENDAR_ID", "primary")


def _get_service():
    """Build and return an authenticated Google Calendar service object."""
    if not _GOOGLE_OK:
        raise RuntimeError(
            "google-api-python-client not installed. "
            "Run: pip install google-api-python-client google-auth"
        )
    if not os.path.exists(SERVICE_ACCOUNT_FILE):
        raise RuntimeError(
            "service_account.json not found. "
            "Download your Google service account key and save it as "
            "backend/service_account.json"
        )
    creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES
    )
    return build("calendar", "v3", credentials=creds, cache_discovery=False)


def create_task_event(
    task_name:        str,
    description:      str,
    priority:         str,
    interval_seconds: int,
    suggested_hour:   int  = 9,
    suggested_minute: int  = 0,
    ai_reason:        str  = "",
) -> dict:
    """
    Creates a Google Calendar event for a scheduled task.

    The event:
      - Starts at AI-suggested time today (or tomorrow if that time has passed)
      - Duration = max(interval_seconds, 15 min) so it's visible on the calendar
      - Description includes priority, interval, AI reason, and ML status
      - Color-coded by priority: High=red(11), Medium=yellow(5), Low=green(10)

    Returns a dict with:
      event_id  : str  — Google Calendar event ID
      event_link: str  — URL to open the event in Google Calendar
      start_time: str  — ISO datetime of the scheduled start
    """
    try:
        service = _get_service()
    except RuntimeError as e:
        return {"error": str(e), "event_id": None, "event_link": None, "start_time": None}

    # ── Compute start time ────────────────────────────────────────────────────
    now   = datetime.datetime.now()
    start = now.replace(hour=suggested_hour, minute=suggested_minute, second=0, microsecond=0)
    if start <= now:                           # time already passed today → schedule tomorrow
        start += datetime.timedelta(days=1)

    duration_minutes = max(15, interval_seconds // 60)
    end = start + datetime.timedelta(minutes=duration_minutes)

    # ── Color map ─────────────────────────────────────────────────────────────
    color_map = {"High": "11", "Medium": "5", "Low": "10"}   # Tomato / Banana / Sage
    color_id  = color_map.get(priority, "1")

    # ── Build event body ──────────────────────────────────────────────────────
    event_body = {
        "summary": f"[{priority}] {task_name}",
        "description": (
            f"Scheduled by: Configurable Task Scheduler\n"
            f"Priority      : {priority}\n"
            f"Interval      : every {interval_seconds}s\n"
            f"Task Info     : {description or 'N/A'}\n"
            f"AI Schedule   : {ai_reason or 'Auto-scheduled'}\n"
            f"\nThis event was created automatically. "
            f"Deleting it here does NOT cancel the backend task."
        ),
        "start": {
            "dateTime": start.isoformat(),
            "timeZone": "Asia/Kolkata",        # IST — change to your tz if needed
        },
        "end": {
            "dateTime": end.isoformat(),
            "timeZone": "Asia/Kolkata",
        },
        "colorId": color_id,
        "reminders": {
            "useDefault": False,
            "overrides": [
                {"method": "popup",  "minutes": 10},
                {"method": "email",  "minutes": 30},
            ],
        },
    }

    try:
        created = service.events().insert(
            calendarId=CALENDAR_ID, body=event_body
        ).execute()
        return {
            "event_id":   created.get("id"),
            "event_link": created.get("htmlLink"),
            "start_time": start.isoformat(),
            "error":      None,
        }
    except Exception as e:
        return {"error": str(e), "event_id": None, "event_link": None, "start_time": None}


def delete_task_event(event_id: str) -> dict:
    """
    Deletes a calendar event by its Google Calendar event ID.
    Called when a task is deleted from the scheduler.
    """
    try:
        service = _get_service()
        service.events().delete(calendarId=CALENDAR_ID, eventId=event_id).execute()
        return {"deleted": True, "event_id": event_id}
    except Exception as e:
        return {"deleted": False, "error": str(e)}


def list_upcoming_events(max_results: int = 10) -> list:
    """
    Returns upcoming task-scheduler events from the calendar.
    Used by GET /api/calendar/events to show the schedule in the dashboard.
    """
    try:
        service = _get_service()
        now_utc = datetime.datetime.utcnow().isoformat() + "Z"
        result  = service.events().list(
            calendarId=CALENDAR_ID,
            timeMin=now_utc,
            maxResults=max_results,
            singleEvents=True,
            orderBy="startTime",
            q="Configurable Task Scheduler",   # only fetch our events
        ).execute()
        events = result.get("items", [])
        return [
            {
                "id":         e.get("id"),
                "summary":    e.get("summary"),
                "start":      e.get("start", {}).get("dateTime"),
                "end":        e.get("end",   {}).get("dateTime"),
                "link":       e.get("htmlLink"),
                "color_id":   e.get("colorId"),
            }
            for e in events
        ]
    except Exception as e:
        return [{"error": str(e)}]


def is_calendar_configured() -> dict:
    """
    Health-check endpoint: tells the frontend whether Google Calendar
    is ready to use. Returns a status dict shown in the dashboard.
    """
    if not _GOOGLE_OK:
        return {
            "configured": False,
            "reason": "Missing packages. Run: pip install google-api-python-client google-auth",
        }
    if not os.path.exists(SERVICE_ACCOUNT_FILE):
        return {
            "configured": False,
            "reason": "service_account.json not found in backend/ folder.",
        }
    return {"configured": True, "reason": "Google Calendar ready."}
