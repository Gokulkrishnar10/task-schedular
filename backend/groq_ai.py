"""
groq_ai.py — Groq LLaMA 3.3 70B AI Features
=============================================
Configurable Task Scheduler Application

Three AI capabilities powered by Groq's free-tier API:
  1. suggest_priority()      — Auto-assigns High/Medium/Low to new tasks
  2. predict_failure()       — Predicts system risk before each task runs
  3. generate_health_report()— Full plain-English system health summary
  4. suggest_schedule_time() — AI picks the best time to schedule a task
                               (used for Google Calendar event creation)
"""

import os
import httpx
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_URL     = "https://api.groq.com/openai/v1/chat/completions"
MODEL        = "llama-3.3-70b-versatile"

HEADERS = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type":  "application/json",
}


async def _call_groq(system_prompt: str, user_message: str, max_tokens: int = 200) -> str:
    payload = {
        "model":      MODEL,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
    }
    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            r    = await client.post(GROQ_URL, headers=HEADERS, json=payload)
            data = r.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return f"AI unavailable: {str(e)}"


# ── 1. Auto Priority ──────────────────────────────────────────────────────────

async def suggest_priority(task_name: str, description: str) -> str:
    """Returns 'High', 'Medium', or 'Low'."""
    system_prompt = (
        "You are an intelligent task prioritization engine for a configurable task scheduler. "
        "Given a task name and description, respond with ONLY one word: High, Medium, or Low. "
        "High = critical system/production tasks. Medium = regular operational tasks. Low = routine/maintenance."
    )
    result = await _call_groq(system_prompt, f"Task: {task_name}\nDescription: {description}", max_tokens=10)
    for p in ["High", "Medium", "Low"]:
        if p.lower() in result.lower():
            return p
    return "Medium"


# ── 2. Pre-run Failure Prediction ────────────────────────────────────────────

async def predict_failure(metrics: dict) -> str:
    """Returns 'Low Risk', 'Medium Risk', or 'High Risk'."""
    system_prompt = (
        "You are a predictive system health AI for a configurable task scheduler platform. "
        "Analyze system metrics and respond with ONLY one of: Low Risk, Medium Risk, High Risk. "
        "High Risk = CPU > 85% or RAM > 90%. Medium Risk = CPU 60-85% or RAM 70-90%. Low Risk = below those."
    )
    user_message = f"CPU: {metrics['cpu']}% | RAM: {metrics['ram']}% | Disk: {metrics['disk']}%"
    result = await _call_groq(system_prompt, user_message, max_tokens=10)
    for r in ["High Risk", "Medium Risk", "Low Risk"]:
        if r.lower() in result.lower():
            return r
    return "Low Risk"


# ── 3. System Health Report ───────────────────────────────────────────────────

async def generate_health_report(metrics: dict, ml_summary: dict = None) -> str:
    """Generates a 4-5 sentence professional health report, including ML findings."""
    system_prompt = (
        "You are an AI system health analyst for a Configurable Task Scheduler SaaS platform. "
        "Generate a concise, professional 4-5 sentence system health report. "
        "Cover: current performance status, potential risks, ML pipeline findings if provided, "
        "and one actionable recommendation. "
        "Use clear language suitable for operations managers."
    )
    ml_lines = ""
    if ml_summary:
        ml_lines = (
            f"\nML Pipeline:\n"
            f"  Anomaly Detection : {ml_summary.get('anomaly_label', 'N/A')}\n"
            f"  CPU Forecast 30s  : {ml_summary.get('forecast_cpu', 'N/A')}%\n"
            f"  RAM Forecast 30s  : {ml_summary.get('forecast_ram', 'N/A')}%\n"
            f"  Classifier Risk   : {ml_summary.get('classifier_risk', 'N/A')}\n"
            f"  Training Points   : {ml_summary.get('data_points', 0)}"
        )
    user_message = (
        f"CPU: {metrics['cpu']}% | RAM: {metrics['ram']}% ({metrics['ram_total_gb']} GB total) | "
        f"Disk: {metrics['disk']}% ({metrics['disk_total_gb']} GB total) | "
        f"Net Sent: {metrics['net_sent_mb']} MB | Net Recv: {metrics['net_recv_mb']} MB | "
        f"Cores: {metrics['cpu_cores']} | Boot: {metrics['boot_time']}"
        f"{ml_lines}"
    )
    return await _call_groq(system_prompt, user_message, max_tokens=350)



# ── 5. Conversational Task-Creation Agent ────────────────────────────────────

_CHAT_SYSTEM = """You are ITOC Assistant, an AI agent embedded in the Configurable Task Scheduler dashboard.
Your ONLY job is to help users create scheduled tasks through friendly conversation.

WORKFLOW:
1. Greet the user and ask what task they want to schedule (if they haven't described one).
2. Gather these 4 pieces of information — ask for missing ones naturally, ONE question at a time:
   - task_name       : short name for the task (required)
   - description     : what the task does (required)
   - interval_seconds: how often to run it — accept natural language like "every 5 minutes", "hourly", "every 30 seconds"
   - sync_to_calendar: should this be added to Google Calendar? (yes/no)
3. Once you have ALL 4, summarise the task and ask the user to confirm with YES.
4. When the user confirms, respond ONLY with this exact JSON block (no other text):
   {"action":"CREATE_TASK","task_name":"...","description":"...","interval_seconds":N,"sync_to_calendar":true/false}

RULES:
- Convert natural time to seconds: "every minute"=60, "every 5 minutes"=300, "hourly"=3600, "every 30 seconds"=30, "daily"=86400.
- Default interval = 60 seconds if the user doesn't care.
- Default sync_to_calendar = false unless user says yes/calendar/Google.
- Keep replies SHORT (2–3 sentences max except for the confirmation summary).
- Never discuss topics outside task scheduling.
- Never invent task details — only use what the user tells you.
- After the JSON block, output nothing else.
"""


async def chat_agent_turn(history: list[dict]) -> dict:
    """
    Takes the full conversation history (list of {role, content} dicts)
    and returns:
      {
        "reply":   str,            # assistant message to show
        "action":  None | "CREATE_TASK",
        "payload": None | {task_name, description, interval_seconds, sync_to_calendar}
      }
    """
    messages = [{"role": "system", "content": _CHAT_SYSTEM}] + history

    payload = {
        "model":      MODEL,
        "max_tokens": 300,
        "messages":   messages,
        "temperature": 0.4,
    }

    async with httpx.AsyncClient(timeout=20.0) as client:
        try:
            r    = await client.post(GROQ_URL, headers=HEADERS, json=payload)
            data = r.json()
            raw  = data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return {"reply": f"Sorry, I'm having trouble reaching the AI right now. ({e})", "action": None, "payload": None}

    # Check if the model returned the CREATE_TASK JSON trigger
    import json, re
    json_match = re.search(r'\{[^{}]*"action"\s*:\s*"CREATE_TASK"[^{}]*\}', raw, re.DOTALL)
    if json_match:
        try:
            task_data = json.loads(json_match.group())
            return {
                "reply":   raw,          # frontend will replace this with summary card
                "action":  "CREATE_TASK",
                "payload": {
                    "name":             task_data.get("task_name", "Unnamed Task"),
                    "description":      task_data.get("description", ""),
                    "interval_seconds": int(task_data.get("interval_seconds", 60)),
                    "sync_to_calendar": bool(task_data.get("sync_to_calendar", False)),
                }
            }
        except (json.JSONDecodeError, ValueError):
            pass  # Fall through to plain reply

    return {"reply": raw, "action": None, "payload": None}


async def suggest_schedule_time(task_name: str, description: str, priority: str) -> dict:
    """
    Returns a JSON-style dict with:
      suggested_hour   : int  (0-23, best hour to run this task today)
      suggested_minute : int  (0 or 30)
      reason           : str  (one-sentence explanation)
    Used by the Google Calendar route to create events intelligently.
    """
    system_prompt = (
        "You are a task scheduling optimizer. "
        "Given a task name, description, and priority, suggest the best time of day to run it. "
        "Respond in exactly this format (no extra text):\n"
        "HOUR: <0-23>\n"
        "MINUTE: <0 or 30>\n"
        "REASON: <one sentence>\n"
        "Rules: High priority = business hours (8-17). "
        "Low priority = off-peak (0-7 or 18-23). Medium = flexible."
    )
    user_message = f"Task: {task_name}\nDescription: {description}\nPriority: {priority}"
    result = await _call_groq(system_prompt, user_message, max_tokens=80)

    # Parse the structured response
    hour, minute, reason = 9, 0, "Scheduled during business hours"
    for line in result.splitlines():
        line = line.strip()
        if line.startswith("HOUR:"):
            try:
                hour = int(line.split(":", 1)[1].strip())
                hour = max(0, min(23, hour))
            except ValueError:
                pass
        elif line.startswith("MINUTE:"):
            try:
                minute = int(line.split(":", 1)[1].strip())
                minute = 0 if minute not in (0, 30) else minute
            except ValueError:
                pass
        elif line.startswith("REASON:"):
            reason = line.split(":", 1)[1].strip()

    return {"suggested_hour": hour, "suggested_minute": minute, "reason": reason}
