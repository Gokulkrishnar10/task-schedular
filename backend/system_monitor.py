"""
system_monitor.py — Live System Metrics via psutil
===================================================
Configurable Task Scheduler Application

Collects real CPU / RAM / Disk / Network data every time it's called.
Maintains a rolling 20-point deque used by the real-time Chart.js graph.
"""

import psutil
import datetime
from collections import deque

_history: deque = deque(maxlen=20)


def get_system_metrics() -> dict:
    cpu   = psutil.cpu_percent(interval=0.5)
    ram   = psutil.virtual_memory()
    disk  = psutil.disk_usage("/")
    net   = psutil.net_io_counters()
    boot  = datetime.datetime.fromtimestamp(psutil.boot_time()).strftime("%Y-%m-%d %H:%M")

    metrics = {
        "cpu":            cpu,
        "ram":            ram.percent,
        "disk":           disk.percent,
        "net_sent_mb":    round(net.bytes_sent  / (1024 ** 2), 2),
        "net_recv_mb":    round(net.bytes_recv  / (1024 ** 2), 2),
        "cpu_cores":      psutil.cpu_count(logical=True),
        "ram_total_gb":   round(ram.total        / (1024 ** 3), 2),
        "ram_used_gb":    round(ram.used         / (1024 ** 3), 2),
        "disk_total_gb":  round(disk.total       / (1024 ** 3), 2),
        "disk_used_gb":   round(disk.used        / (1024 ** 3), 2),
        "boot_time":      boot,
        "timestamp":      datetime.datetime.now().isoformat(),
    }

    _history.append({
        "time":  datetime.datetime.now().strftime("%H:%M:%S"),
        "cpu":   cpu,
        "ram":   ram.percent,
        "disk":  disk.percent,
    })

    return metrics


def get_metrics_history() -> list:
    return list(_history)
