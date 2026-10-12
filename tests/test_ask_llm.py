"""Tests for the Gemini ask_llm helper."""

from __future__ import annotations

import logging
import os
import sys

logging.disable(logging.CRITICAL)
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_client import ask_llm


class AskLlmTests(unittest.TestCase):
    def test_returns_response_text(self) -> None:
        mock_response = SimpleNamespace(text="Portfolio default rate is 3%.")
        env = {
            "GEMINI_API_KEY": "test-api-key",
            "GEMINI_MODEL": "gemini-2.0-flash",
        }
        with patch.dict(os.environ, env, clear=False):
            with patch("llm_client.load_dotenv"):
                with patch("llm_client.genai.Client") as client_cls:
                    client_cls.return_value.models.generate_content.return_value = (
                        mock_response
                    )
                    result = ask_llm("Summarize portfolio risk.")

        self.assertEqual(result, "Portfolio default rate is 3%.")
        client_cls.assert_called_once_with(api_key="test-api-key")
        client_cls.return_value.models.generate_content.assert_called_once_with(
            model="gemini-2.0-flash",
            contents="Summarize portfolio risk.",
        )

    def test_strips_leading_and_trailing_whitespace_from_question(self) -> None:
        mock_response = SimpleNamespace(text="ok")
        env = {
            "GEMINI_API_KEY": "test-api-key",
            "GEMINI_MODEL": "gemini-2.0-flash",
        }
        with patch.dict(os.environ, env, clear=False):
            with patch("llm_client.load_dotenv"):
                with patch("llm_client.genai.Client") as client_cls:
                    client_cls.return_value.models.generate_content.return_value = (
                        mock_response
                    )
                    ask_llm("  What is LGD?  ")

        client_cls.return_value.models.generate_content.assert_called_once_with(
            model="gemini-2.0-flash",
            contents="What is LGD?",
        )

    def test_empty_question_raises(self) -> None:
        with self.assertRaises(ValueError):
            ask_llm("")

    def test_whitespace_only_question_raises(self) -> None:
        with self.assertRaises(ValueError):
            ask_llm("   \n\t  ")

    def test_non_string_question_raises(self) -> None:
        with self.assertRaises(ValueError):
            ask_llm(42)  # type: ignore[arg-type]

    def test_missing_api_key_raises(self) -> None:
        env = {"GEMINI_MODEL": "gemini-2.0-flash"}
        with patch.dict(os.environ, env, clear=True):
            with patch("llm_client.load_dotenv"):
                with self.assertRaises(ValueError):
                    ask_llm("Valid question?")

    def test_missing_model_raises(self) -> None:
        env = {"GEMINI_API_KEY": "test-api-key"}
        with patch.dict(os.environ, env, clear=True):
            with patch("llm_client.load_dotenv"):
                with self.assertRaises(ValueError):
                    ask_llm("Valid question?")

    def test_blank_api_key_raises(self) -> None:
        env = {"GEMINI_API_KEY": "   ", "GEMINI_MODEL": "gemini-2.0-flash"}
        with patch.dict(os.environ, env, clear=True):
            with patch("llm_client.load_dotenv"):
                with self.assertRaises(ValueError):
                    ask_llm("Valid question?")


if __name__ == "__main__":
    unittest.main()
