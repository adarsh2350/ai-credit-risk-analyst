"""Gemini LLM client for natural-language credit risk questions."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_ENV_FILE = _PROJECT_ROOT / ".env"


def ask_llm(question: str) -> str:
    """Send a question to Gemini and return the model's text reply.

    Configuration is read from the project ``.env`` file and process environment:
    ``GEMINI_API_KEY`` and ``GEMINI_MODEL``.

    Args:
        question: Non-empty natural-language prompt.

    Returns:
        ``response.text`` from the Gemini ``generate_content`` call.

    Raises:
        ValueError: If ``question`` is empty or not a non-blank string, or if
            ``GEMINI_API_KEY`` / ``GEMINI_MODEL`` are missing or blank.
    """
    if not isinstance(question, str):
        logger.warning("Rejected LLM request: question is not a string")
        raise ValueError("question must be a string")

    prompt = question.strip()
    if not prompt:
        logger.warning("Rejected LLM request: empty question after strip")
        raise ValueError("question must not be empty")

    if not _ENV_FILE.is_file():
        logger.warning("Rejected LLM request: .env file not found at %s", _ENV_FILE)
        raise ValueError(
            f"Missing {_ENV_FILE}. Copy .env.example to .env and set GEMINI_API_KEY."
        )

    load_dotenv(_ENV_FILE, override=True)

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        logger.warning(
            "Rejected LLM request: GEMINI_API_KEY missing or blank in %s", _ENV_FILE
        )
        hint = " (.env file is empty)" if _ENV_FILE.stat().st_size == 0 else ""
        raise ValueError(
            "GEMINI_API_KEY must be set in .env or the environment"
            f"{hint}. Expected line: GEMINI_API_KEY=your_key "
            f"(not gemini_key). File: {_ENV_FILE}"
        )

    model = os.getenv("GEMINI_MODEL", "").strip()
    if not model:
        logger.warning("Rejected LLM request: GEMINI_MODEL is missing or blank")
        raise ValueError(
            f"GEMINI_MODEL must be set in .env or the environment. File: {_ENV_FILE}"
        )

    logger.info("Sending question to Gemini model=%s (prompt length=%d)", model, len(prompt))
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(model=model, contents=prompt)
    text = response.text
    logger.info(
        "Received Gemini response (text length=%s)",
        len(text) if text is not None else "None",
    )
    return text
