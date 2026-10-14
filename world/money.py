"""
WHAT: Pure currency conversion functions between rupees and integer paise.
It provides consistent arithmetic to translate decimal currency into exact integer subunits and back.
All financial modules throughout the simulation rely on these helpers to prevent fractional drift.

WHY: Financial ledgers and banking protocols must store exact whole integers so multi-hop transfers never lose fractions.
Floating-point arithmetic introduces IEEE-754 precision errors that break balance assertions over time.
Using integer paise ensures exact conservation of money across every simulated transaction.

TERM: rupee: The primary unit of Indian currency, symbolized as INR or Rs. For example, a bottle of milk might cost 30 rupees.
TERM: paise: The subunit of Indian currency where 1 rupee equals exactly 100 paise. For example, 50 paise is half a rupee.
TERM: floating-point drift: Subtle rounding discrepancies caused by computer hardware approximating decimals. For example, 0.1 + 0.2 computing as 0.30000000000000004.
"""


def rupees_to_paise(rupees: float | int) -> int:
    """
    WHAT: Converts a rupee amount into integer paise by multiplying by 100 and rounding to the nearest whole unit.
    This operation handles both floating-point numbers and integer rupee values cleanly.
    The resulting integer represents the exact number of paise without decimal precision loss.

    WHY: Financial accounting requires integer precision to avoid accumulating fractional currency errors across payments.
    Multi-hop transactions and ledger balance audits fail if fractional paise leak into balances.
    Converting to integer paise at the boundary guarantees exact arithmetic in all accounts.

    Args:
        rupees: Amount of currency in Indian rupees, supplied as an int or float.

    Returns:
        Integer number of paise equivalent to the given rupee amount.

    Example:
        # Converting 150.50 rupees yields 15050 paise:
        # paise = rupees_to_paise(150.50)
        # assert paise == 15050
    """
    # Scale decimal rupees by 100 to get paise, then round to the nearest whole number to strip decimal noise.
    return int(round(float(rupees) * 100))


def paise_to_rupees(paise: int) -> float:
    """
    WHAT: Converts an integer paise count back into standard decimal rupees.
    It divides the integer quantity by 100.0 to produce a floating-point rupee representation.
    This is used when printing reports or interacting with external systems expecting rupees.

    WHY: While internal calculations demand integer precision, human readability demands rupees.
    External analytics, logs, and user interfaces conventionally present monetary amounts in rupees.
    Providing a centralized conversion helper prevents ad-hoc division logic throughout the codebase.

    Args:
        paise: Amount of currency in integer paise subunits.

    Returns:
        Float value representing the same currency value in standard rupees.

    Example:
        # Converting 15050 paise back yields 150.5 rupees:
        # inr = paise_to_rupees(15050)
        # assert inr == 150.5
    """
    # Divide integer paise by 100.0 to convert subunits back to rupee units with decimals.
    return float(paise) / 100.0

