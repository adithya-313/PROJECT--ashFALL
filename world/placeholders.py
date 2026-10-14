"""
WHAT: Placeholder numerical constants and default assumptions for the simulated world.
These values govern device market shares, persona distributions, merchant categories, and transaction multipliers.
They serve as interim stand-ins until formal empirical research calibration is completed in sub-phase 11B.

WHY: Isolating uncalibrated assumptions in a single module prevents arbitrary magic numbers from scattering across code.
Explicitly marking these assumptions with issue identifiers keeps technical debt transparent and trackable.
This centralized location makes migrating constants into configuration files straightforward in future releases.

ISSUE IDs: E2, E3, E4 (supporting parameters).
TERM: placeholder: A temporary assumption or constant explicitly designated for future empirical calibration.
      For example, assuming Android devices comprise 94% of smartphones.
TERM: market share: The percentage of total users or devices belonging to a particular operating system or vendor.
      For example, 94 out of 100 mobile devices running Android.
TERM: persona mix: The demographic proportions defining how simulated citizens are distributed among
      lifestyle categories. For example, setting salaried employees at 45% of the population.
TERM: opening balance: The initial quantity of funds seeded into an account when the world is created.
      For example, granting a shop 50,000 rupees to support inventory purchases.
TERM: recurring transfer: A scheduled periodic payment such as monthly salary, apartment rent, or utility
      services. For example, paying an electricity bill once every month.
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
