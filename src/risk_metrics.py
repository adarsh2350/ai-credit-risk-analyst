"""Portfolio-level credit risk metrics from binary default indicators."""

from __future__ import annotations


def calculate_portfolio_metrics(default_flags: list[int]) -> dict[str, int | float]:
    """Compute aggregate default statistics for a loan portfolio.

    Each entry in ``default_flags`` is a binary outcome: ``0`` for non-default
    and ``1`` for default.

    Args:
        default_flags: Non-empty list of ``0`` and ``1`` values, one per loan.

    Returns:
        Dictionary with:
            - ``total_loans``: number of loans in the portfolio
            - ``total_defaults``: count of loans with default flag ``1``
            - ``default_rate``: ``total_defaults / total_loans`` (proportion in
              ``[0, 1]``)

    Raises:
        ValueError: If ``default_flags`` is empty or any element is not ``0``
            or ``1``.
    """
    if not default_flags:
        raise ValueError("default_flags must not be empty")

    for index, flag in enumerate(default_flags):
        if flag not in (0, 1):
            raise ValueError(
                f"default_flags must contain only 0 or 1; got {flag!r} at index {index}"
            )

    total_loans = len(default_flags)
    total_defaults = sum(default_flags)
    default_rate = total_defaults / total_loans

    return {
        "total_loans": total_loans,
        "total_defaults": total_defaults,
        "default_rate": default_rate,
    }
