"""Application-wide logging configuration."""

from __future__ import annotations

import logging
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = _PROJECT_ROOT / "logs"
LOG_FILE = LOG_DIR / "ai_credit_risk_analyst.log"

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logging to the project log file and the console."""
    root = logging.getLogger()
    if root.handlers:
        return

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    root.setLevel(level)
    root.addHandler(file_handler)
    root.addHandler(console_handler)

    logging.getLogger(__name__).info("Logging initialized; writing to %s", LOG_FILE)
