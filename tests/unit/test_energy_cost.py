# File: tests/unit/test_energy_cost.py
# Purpose: Checks the reusable cost-per-inference formula in core/energy_cost.py with numbers worked out by hand.
import pytest

from core.energy_cost import DEFAULT_PRICE_PER_KWH_USD, cost_per_inference_usd, watts_to_kilowatts


def test_one_kilowatt_for_one_hour_costs_exactly_the_price():
    # 1 kW for 3600 seconds is 1 kWh, so the cost is exactly the price of one kWh.
    assert cost_per_inference_usd(1.0, 3600.0, price_per_kwh=0.25) == pytest.approx(0.25)


def test_worked_example():
    # 100 W = 0.1 kW, one inference takes 0.036 s: 0.1 * 0.036 / 3600 = 1e-6 kWh, at $0.10/kWh = $1e-7.
    assert cost_per_inference_usd(0.1, 0.036, price_per_kwh=0.10) == pytest.approx(1e-7)


def test_default_price_is_used_when_none_is_given():
    assert cost_per_inference_usd(1.0, 3600.0) == pytest.approx(DEFAULT_PRICE_PER_KWH_USD)


def test_zero_time_costs_nothing():
    assert cost_per_inference_usd(0.05, 0.0) == 0.0


def test_watts_to_kilowatts():
    assert watts_to_kilowatts(45.0) == pytest.approx(0.045)


def test_negative_inputs_are_rejected():
    for bad in [(-1.0, 1.0, 0.1), (1.0, -1.0, 0.1), (1.0, 1.0, -0.1)]:
        with pytest.raises(ValueError):
            cost_per_inference_usd(*bad)
