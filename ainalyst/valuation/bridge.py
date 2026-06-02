"""EV ↔ Equity Value bridge — the core corporate-finance identity.

    Equity Value = Enterprise Value − Total Debt + Cash & Cash Equivalents
    Intrinsic Value Per Share = Equity Value / Diluted Shares Outstanding
"""

from __future__ import annotations


def ev_to_equity(
    enterprise_value: float,
    total_debt: float,
    cash_and_equivalents: float,
) -> float:
    """Convert Enterprise Value to Equity Value.

    Parameters
    ----------
    enterprise_value : float
        Firm-level value (PV of FCFFs + terminal value).
    total_debt : float
        Total interest-bearing debt (short + long term).
    cash_and_equivalents : float
        Cash & near-cash items on the balance sheet.

    Returns
    -------
    float
        Equity Value attributable to common shareholders.
    """
    return enterprise_value - total_debt + cash_and_equivalents


def equity_per_share(
    equity_value: float,
    diluted_shares: float,
) -> float:
    """Compute intrinsic value per share.

    Raises
    ------
    ValueError
        If *diluted_shares* ≤ 0.
    """
    if diluted_shares <= 0:
        raise ValueError(f"diluted_shares must be > 0, got {diluted_shares}")
    return equity_value / diluted_shares


def margin_of_safety(
    intrinsic_per_share: float,
    current_price: float,
) -> float:
    """Return the Margin of Safety as a signed percentage.

    Positive → undervalued, negative → overvalued.
    """
    if current_price <= 0:
        return 0.0
    return (intrinsic_per_share - current_price) / current_price
