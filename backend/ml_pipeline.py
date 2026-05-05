"""
ml_pipeline.py — Tier 1 Custom ML Pipeline
Integrates three models that run on live psutil data:

  1. Isolation Forest    → Anomaly Detection
  2. Linear Regression   → CPU/RAM Forecasting (next 30s)
  3. Random Forest       → Task Failure Classification
"""

import threading
import numpy as np
from collections import deque
from datetime import datetime

# ── scikit-learn imports (graceful fallback if not installed yet) ────────────
try:
    from sklearn.ensemble import IsolationForest, RandomForestClassifier
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    _SKLEARN_OK = True
except ImportError:
    _SKLEARN_OK = False

# ── Thread-safe rolling buffers ──────────────────────────────────────────────
_lock = threading.Lock()

# Stores (cpu, ram, disk) readings — used by anomaly + forecasting models
_metrics_buffer: deque = deque(maxlen=100)

# Stores (cpu, ram, disk, interval_seconds, label) — used by classifier
# label: 1 = Warning/Failure, 0 = Completed
_task_history: deque = deque(maxlen=200)

# Last computed pipeline output — served to API without recomputing every call
_last_output: dict = {
    "anomaly_label": "Warming Up",
    "anomaly_score": 0.0,
    "forecast_cpu": None,
    "forecast_ram": None,
    "classifier_risk": "Insufficient Data",
    "classifier_confidence": 0.0,
    "data_points": 0,
    "model_status": "initializing",
    "updated_at": None,
}

# ── Public helpers called from main.py ──────────────────────────────────────

def push_metrics(cpu: float, ram: float, disk: float):
    """Append a live reading to the metrics buffer."""
    with _lock:
        _metrics_buffer.append((cpu, ram, disk))


def add_task_run_to_history(
    cpu: float, ram: float, disk: float,
    interval_seconds: int, status: str
):
    """Record a completed task run for the classifier training set."""
    label = 1 if status in ("Warning", "Failed", "Error") else 0
    with _lock:
        _task_history.append((cpu, ram, disk, interval_seconds, label))


def get_task_history_count() -> int:
    with _lock:
        return len(_task_history)


def get_pipeline_output() -> dict:
    """Return the last computed ML output dict (no recompute)."""
    with _lock:
        return dict(_last_output)


# ── Core pipeline ────────────────────────────────────────────────────────────

def run_full_pipeline(
    cpu: float, ram: float, disk: float,
    interval_seconds: int = 30
) -> dict:
    """
    Push new metrics, then run all three models.
    Returns a fresh output dict AND updates _last_output in-place.
    """
    push_metrics(cpu, ram, disk)

    if not _SKLEARN_OK:
        result = {
            "anomaly_label": "scikit-learn not installed",
            "anomaly_score": 0.0,
            "forecast_cpu": None,
            "forecast_ram": None,
            "classifier_risk": "scikit-learn not installed",
            "classifier_confidence": 0.0,
            "data_points": 0,
            "model_status": "error",
            "updated_at": datetime.now().isoformat(),
        }
        _update_last(result)
        return result

    with _lock:
        buffer_snap = list(_metrics_buffer)
        history_snap = list(_task_history)

    n = len(buffer_snap)

    anomaly_label, anomaly_score = _run_anomaly(buffer_snap, cpu, ram, disk)
    forecast_cpu, forecast_ram   = _run_forecast(buffer_snap)
    clf_risk, clf_conf           = _run_classifier(history_snap, cpu, ram, disk, interval_seconds)

    result = {
        "anomaly_label": anomaly_label,
        "anomaly_score": round(anomaly_score, 4),
        "forecast_cpu": forecast_cpu,
        "forecast_ram": forecast_ram,
        "classifier_risk": clf_risk,
        "classifier_confidence": round(clf_conf, 3),
        "data_points": n,
        "model_status": "running" if n >= 10 else "warming_up",
        "updated_at": datetime.now().isoformat(),
    }
    _update_last(result)
    return result


# ── Model 1 — Isolation Forest Anomaly Detection ────────────────────────────

def _run_anomaly(
    buffer: list, cpu: float, ram: float, disk: float
) -> tuple[str, float]:
    """
    Train Isolation Forest on all buffered readings.
    Returns (label, raw_score) where score > 0 = normal, < 0 = anomaly.
    Needs at least 10 points.
    """
    n = len(buffer)
    if n < 10:
        return "Warming Up", 0.0

    X = np.array(buffer, dtype=float)

    # contamination: assume ~10 % of readings are anomalous
    contamination = min(0.1, max(0.01, 3 / n))
    model = IsolationForest(
        n_estimators=50,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X)

    sample = np.array([[cpu, ram, disk]])
    pred   = model.predict(sample)[0]        # 1 = normal, -1 = anomaly
    score  = model.score_samples(sample)[0]  # raw anomaly score

    label = "Normal" if pred == 1 else "Anomaly Detected"
    return label, float(score)


# ── Model 2 — Linear Regression Forecasting ─────────────────────────────────

def _run_forecast(buffer: list) -> tuple[float | None, float | None]:
    """
    Fit a simple Linear Regression on the last N readings to forecast
    the NEXT reading's CPU and RAM.
    Needs at least 5 points.
    """
    n = len(buffer)
    if n < 5:
        return None, None

    arr = np.array(buffer, dtype=float)
    X   = np.arange(n).reshape(-1, 1)

    # CPU forecast
    lr_cpu = LinearRegression()
    lr_cpu.fit(X, arr[:, 0])
    pred_cpu = float(np.clip(lr_cpu.predict([[n]])[0], 0, 100))

    # RAM forecast
    lr_ram = LinearRegression()
    lr_ram.fit(X, arr[:, 1])
    pred_ram = float(np.clip(lr_ram.predict([[n]])[0], 0, 100))

    return round(pred_cpu, 1), round(pred_ram, 1)


# ── Model 3 — Random Forest Task Failure Classifier ─────────────────────────

def _run_classifier(
    history: list, cpu: float, ram: float,
    disk: float, interval_seconds: int
) -> tuple[str, float]:
    """
    Train a Random Forest on past task runs (cpu, ram, disk, interval → label).
    Needs at least 5 runs with at least 1 of each class; otherwise falls back
    to a deterministic threshold rule.
    Returns (risk_label, confidence_0_to_1).
    """
    n = len(history)

    # Threshold fallback (used while warming up)
    if n < 5:
        return _threshold_risk(cpu, ram), 0.0

    X = np.array([[h[0], h[1], h[2], h[3]] for h in history], dtype=float)
    y = np.array([h[4] for h in history], dtype=int)

    # Need both classes present to train
    if len(np.unique(y)) < 2:
        return _threshold_risk(cpu, ram), 0.0

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = RandomForestClassifier(
        n_estimators=50,
        max_depth=5,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    clf.fit(X_scaled, y)

    sample = scaler.transform([[cpu, ram, disk, interval_seconds]])
    pred   = clf.predict(sample)[0]
    proba  = clf.predict_proba(sample)[0]

    # proba[1] = probability of failure/warning
    fail_prob = float(proba[1]) if len(proba) > 1 else 0.0

    if pred == 1 or fail_prob >= 0.6:
        risk = "High ML Risk"
    elif fail_prob >= 0.35:
        risk = "Medium ML Risk"
    else:
        risk = "Low ML Risk"

    return risk, fail_prob


def _threshold_risk(cpu: float, ram: float) -> str:
    """Simple threshold rule used before enough data exists."""
    if cpu > 85 or ram > 90:
        return "High ML Risk"
    if cpu > 60 or ram > 70:
        return "Medium ML Risk"
    return "Low ML Risk"


# ── Internal helpers ─────────────────────────────────────────────────────────

def _update_last(data: dict):
    global _last_output
    with _lock:
        _last_output = dict(data)
