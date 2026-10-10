"""Tests for portfolio risk metrics."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from risk_metrics import calculate_portfolio_metrics


class CalculatePortfolioMetricsTests(unittest.TestCase):
    def test_mixed_portfolio(self) -> None:
        result = calculate_portfolio_metrics([0, 1, 0, 1, 1])
        self.assertEqual(result["total_loans"], 5)
        self.assertEqual(result["total_defaults"], 3)
        self.assertAlmostEqual(result["default_rate"], 0.6)

    def test_zero_defaults(self) -> None:
        result = calculate_portfolio_metrics([0, 0, 0])
        self.assertEqual(result["total_loans"], 3)
        self.assertEqual(result["total_defaults"], 0)
        self.assertEqual(result["default_rate"], 0.0)

    def test_all_defaults(self) -> None:
        result = calculate_portfolio_metrics([1, 1])
        self.assertEqual(result["total_loans"], 2)
        self.assertEqual(result["total_defaults"], 2)
        self.assertEqual(result["default_rate"], 1.0)

    def test_empty_input_raises(self) -> None:
        with self.assertRaises(ValueError):
            calculate_portfolio_metrics([])

    def test_invalid_values_raise(self) -> None:
        with self.assertRaises(ValueError):
            calculate_portfolio_metrics([0, 2])
        with self.assertRaises(ValueError):
            calculate_portfolio_metrics([0, -1])
        with self.assertRaises(ValueError):
            calculate_portfolio_metrics([0, 1, 3])


if __name__ == "__main__":
    unittest.main()
