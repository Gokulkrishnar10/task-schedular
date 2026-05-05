"""
run.py — Application Entry Point
=================================
Starts APScheduler, then launches FastAPI via uvicorn.
Run: python run.py
"""
import uvicorn
import os
from scheduler import scheduler

if __name__ == "__main__":
    scheduler.start()
    print("╔══════════════════════════════════════════════════╗")
    print("║   Configurable Task Scheduler  v3.0             ║")
    print("║   AI × ML × Google Calendar × Real-time         ║")
    print("╠══════════════════════════════════════════════════╣")
    print("║  Backend  → http://localhost:8000               ║")
    print("║  Dashboard→ http://localhost:8000               ║")
    print("║  API Docs → http://localhost:8000/docs          ║")
    print("╚══════════════════════════════════════════════════╝")
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
