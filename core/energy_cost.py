"""
File: core/energy_cost.py
Purpose: One small, reusable formula for "how much electricity does one inference cost?".
Connects: Used by benchmarks/measure_gpu_power_cost.py (Level 6 reporting). Level 10.5 reuses this same function for
          its energy cost, so the two levels are always computed the same way.

The formula:
    cost_per_inference_usd = avg_power_kW * seconds_per_inference / 3600 * price_per_kWh

Why it works: power (kW) multiplied by time (hours) is energy (kWh), and energy multiplied by the price per kWh is
money. The division by 3600 turns seconds into hours.

What is measured and what is assumed:
  - avg_power_kW and seconds_per_inference must be MEASURED (average power draw during a sustained run, and the
    time that run took divided by the number of inferences). Do not guess them.
  - price_per_kwh is an ASSUMPTION. The default below is only a placeholder-style estimate so the function works out
    of the box. Pass your own real electricity rate to get your real cost.
"""

# ESTIMATE, not a measured fact: a typical electricity price in US dollars per kilowatt-hour. Edit it, or pass your
# own value to cost_per_inference_usd(), to match your real tariff.
DEFAULT_PRICE_PER_KWH_USD = 0.12

SECONDS_PER_HOUR = 3600.0


def watts_to_kilowatts(power_watts: float) -> float:
    """Convert watts (what nvidia-smi reports) to kilowatts (what the formula needs)."""
    return power_watts / 1000.0


def cost_per_inference_usd(
    avg_power_kw: float,
    seconds_per_inference: float,
    price_per_kwh: float = DEFAULT_PRICE_PER_KWH_USD,
) -> float:
    """
    Electricity cost, in US dollars, of running one inference.

    Args:
        avg_power_kw: average power the hardware drew while doing the work, in kilowatts (measured).
        seconds_per_inference: how many seconds one inference takes on average, in seconds (measured).
        price_per_kwh: price of one kilowatt-hour in dollars (an assumption; see DEFAULT_PRICE_PER_KWH_USD).

    Returns:
        avg_power_kW * seconds_per_inference / 3600 * price_per_kWh

    Raises:
        ValueError: if any input is negative, because a negative power, time or price has no meaning here.
    """
    if avg_power_kw < 0 or seconds_per_inference < 0 or price_per_kwh < 0:
        raise ValueError("power, time per inference and price must not be negative")
    return avg_power_kw * seconds_per_inference / SECONDS_PER_HOUR * price_per_kwh
