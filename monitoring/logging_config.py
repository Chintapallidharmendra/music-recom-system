"""Central logging configuration for the whole project.

Every entrypoint (service/main.py, mlops/drift_report.py, mlops/drift_stream_monitor.py,
mlops/dags/*, data/feature_store.py, ...) calls `setup_logging(name)` once at import/startup
time instead of hand-rolling its own `logging.basicConfig`. That gives us:

- one place to control level/format for the whole app (env vars, no code changes)
- console output for local/dev runs and `docker compose logs`
- a rotating file per process under LOG_DIR, so history survives container restarts
  as long as the directory is volume-mounted (docker-compose already mounts ./mlruns
  for other state; LOG_DIR defaults next to it)
- optional JSON lines output (LOG_FORMAT=json) so logs can be shipped to something like
  Loki/ELK later without re-instrumenting call sites

Env vars:
    LOG_LEVEL   DEBUG|INFO|WARNING|ERROR   (default: INFO)
    LOG_FORMAT  text|json                  (default: text)
    LOG_DIR     directory for rotating log files (default: logs)
"""
import json
import logging
import logging.handlers
import os
from pathlib import Path

_CONFIGURED = False

TEXT_FORMAT = "%(asctime)s %(levelname)-8s %(name)s [%(threadName)s] %(message)s"
DATE_FORMAT = "%Y-%m-%dT%H:%M:%S%z"


class JsonFormatter(logging.Formatter):
    """Minimal JSON-lines formatter -- no extra dependency required."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, DATE_FORMAT),
            "level": record.levelname,
            "logger": record.name,
            "thread": record.threadName,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        # Allow call sites to attach structured fields via `extra={"fields": {...}}`.
        extra_fields = getattr(record, "fields", None)
        if extra_fields:
            payload.update(extra_fields)
        return json.dumps(payload)


def setup_logging(service_name: str = "app") -> logging.Logger:
    """Idempotent: safe to call from every module's import (only the first call wins),
    each entrypoint should still call it explicitly with its own service_name so the
    log file is named sensibly."""
    global _CONFIGURED
    root = logging.getLogger()

    if _CONFIGURED:
        return logging.getLogger(service_name)

    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    fmt = os.environ.get("LOG_FORMAT", "text").lower()
    log_dir = Path(os.environ.get("LOG_DIR", "logs"))

    formatter: logging.Formatter
    if fmt == "json":
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(TEXT_FORMAT, datefmt=DATE_FORMAT)

    root.setLevel(level)
    root.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / f"{service_name}.log", maxBytes=10_000_000, backupCount=5
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    except OSError:
        # Read-only filesystem, permissions issue, etc. -- console logging still works,
        # never let logging setup itself take the app down.
        root.warning("could not create log file under %s, console logging only", log_dir)

    # Noisy third-party loggers, quieted down but not silenced.
    logging.getLogger("kafka").setLevel(max(level, logging.WARNING))
    logging.getLogger("urllib3").setLevel(max(level, logging.WARNING))

    _CONFIGURED = True
    return logging.getLogger(service_name)
