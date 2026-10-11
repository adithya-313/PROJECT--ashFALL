"""
WHAT: Pure currency conversion functions between rupees and integer paise.
WHY: Ensures all ledgers, payments, and account balances store exact whole integers to avoid IEEE-754 float drift.
TERM: paise: The subunit of Indian currency, where 1 rupee equals exactly 100 paise.
"""


def rupees_to_paise(rupees: float | int) -> int:
    """
    WHAT: Converts a rupee amount into integer paise, rounding to the nearest single paise.
    WHY: Financial ledgers must store integer quantities so multi-hop payments cannot lose or gain fractions of money.
    """
    # Multiply by 100 and round to nearest integer to eliminate decimal inaccuracy.
    return int(round(float(rupees) * 100))


def paise_to_rupees(paise: int) -> float:
    """
    WHAT: Converts integer paise back into standard rupees as a float.
    WHY: External reporting and human display require standard rupee representation.
    """
    # Divide integer paise by 100 to get rupee units.
    return float(paise) / 100.0
