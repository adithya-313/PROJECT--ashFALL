"""
WHAT: Placeholder numerical constants for the simulated world.
WHY: Stores numbers that have no formal empirical research finding yet, keeping them isolated in one place.
     These will move into background_realism.yaml in sub-phase 11B.
ISSUE IDs: E2, E3, E4 (supporting parameters).
TERM: placeholder: A temporary assumed constant clearly marked to be replaced by configuration later.
"""

# PLACEHOLDER assumption - not from research; moves to config in 11B.
# Market share of Android mobile operating systems in India (Statcounter, Sep 2026).
ANDROID_SHARE: float = 0.94

# PLACEHOLDER assumption - not from research; moves to config in 11B.
# Number of distinct operating system versions in circulation (1 = oldest, 4 = newest).
OS_GENERATIONS: int = 4

# PLACEHOLDER assumption - not from research; moves to config in 11B.
# Proportion of people belonging to each demographic lifestyle profile.
PERSONA_MIX: dict[str, float] = {
    "salaried_employee": 0.45,
    "student": 0.15,
    "retiree": 0.10,
    "small_shopkeeper": 0.15,
    "gig_worker": 0.15,
}

# PLACEHOLDER assumption - not from research; moves to config in 11B.
# Probability weights for shop categories generated across the town.
SHOP_KIND_WEIGHTS: dict[str, float] = {
    "grocery": 0.35,
    "restaurant": 0.20,
    "utility": 0.10,
    "landlord": 0.10,
    "employer": 0.25,
}

# PLACEHOLDER assumption - not from research; moves to config in 11B.
# Starting balance factor multiplied against a person's typical transaction size.
OPENING_BALANCE_MULTIPLE: int = 100

# PLACEHOLDER assumption - not from research; moves to config in 11B.
# Fixed opening balance in rupees given to non-employer merchants.
SHOP_OPENING_BALANCE_RUPEES: int = 50000

# PLACEHOLDER assumption - not from research; moves to config in 11B.
# Multiple of median payment used to compute monthly salary income.
SALARY_MULTIPLE: int = 120

# PLACEHOLDER assumption - not from research; moves to config in 11B.
# Multiple of median payment used to compute monthly home rent.
RENT_MULTIPLE: int = 30

# PLACEHOLDER assumption - not from research; moves to config in 11B.
# Multiple of median payment used to compute each recurring utility bill.
BILL_MULTIPLE: int = 5

# PLACEHOLDER assumption - not from research; moves to config in 11B.
# How many utility bills a salaried person pays per month.
BILLS_PER_MONTH: int = 2

# PLACEHOLDER assumption - not from research; moves to config in 11B.
# How many casual store transactions a person conducts each month.
EVERYDAY_PAYMENTS_PER_MONTH: int = 15

# PLACEHOLDER assumption - not from research; moves to config in 11B.
# How many small inbound credits a mule look-alike receives each month.
LOOKALIKE_SMALL_INCOMING_PER_MONTH: int = 25

# PLACEHOLDER assumption - not from research; moves to config in 11B.
# Calendar days within which employer salaries are disbursed.
SALARY_DAY_WINDOW: tuple[int, int] = (1, 5)

# PLACEHOLDER assumption - not from research; moves to config in 11B.
# Calendar days within which residential rent is collected.
RENT_DAY_WINDOW: tuple[int, int] = (1, 7)

# PLACEHOLDER assumption - not from research; moves to config in 11B.
# Calendar days within which utility bills are settled.
BILL_DAY_WINDOW: tuple[int, int] = (5, 25)

# PLACEHOLDER assumption - not from research; moves to config in 11B.
# Minute of the day (10:00 AM = 600 minutes past midnight) when scheduled transfers fire.
ROUTINE_MINUTE_OF_DAY: int = 600

# PLACEHOLDER assumption - not from research; moves to config in 11B.
# Number of payroll months an employer shop must be able to fund upfront.
EMPLOYER_SALARY_MONTHS_COVERED: int = 24

# PLACEHOLDER assumption - not from research; moves to config in 11B.
# Fraction of median payment transferred in each look-alike credit (10%).
LOOKALIKE_SMALL_AMOUNT_FRACTION: float = 0.1
