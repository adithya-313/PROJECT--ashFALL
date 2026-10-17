"""
WHAT: Tests for the interactive runner with the fraud switch on. With no fraud and a bank that allows everything it
must equal Level 11's run in every output. With liens, debit holds, held and declined payments, fraud payments and
accounts opened mid-run, genuine payments must still read exactly their Level 11 random numbers, money must be
conserved, and every bad input must be refused with a plain message.

WHY: The runner is the base every later Level 12 part builds on. If an empty fraud plan changed one byte, or one
genuine draw moved, every later fraud world would quietly differ from the genuine world it claims to sit on.

TERM: allow-all hook: a bank that answers "allow" to every payment (world.interactive_records.allow_all).
TERM: run stream: Level 11's child-1 random stream used for planning payments and deciding their failures.
TERM: call-recording proxy: a stand-in for a random stream that passes every call through to the real stream and
      writes down which method was called with which arguments, so two runs' draws can be compared one by one.
TERM: lien: keeps part of an account's money from being spent, without moving it.
TERM: debit hold: blocks every payment out of an account until released; payments in still arrive.
TERM: held payment: one the bank hook answered "hold" for; it waits until a resume_payment outside event.
TERM: planned payment id: the name of one payment, shared by all its pieces, e.g. "<run id>-p-000017".
TERM: money conservation: the total money in all accounts equals the starting total plus money from accounts opened
      during the run; payments only move money, never create or destroy it.
TERM: paise: Rs 1 = 100 paise.
"""

import functools
import hashlib
import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pytest

import world.calendar_runner as calendar_runner
import world.interactive_loop as interactive_loop
from core.config_validator import load_yaml
from world.calendar_runner import PlannedPayment, run_world
from world.checks import check_balances_equal_entry_sums, check_money_conserved, check_no_negative_balances
from world.fast_mode import FastRunSettings, event_lines, run_fast_world
from world.interactive_loop import REASON_DECLINED_BY_BANK, REASON_FUNDS_ON_HOLD
from world.interactive_records import FraudInputs, InteractiveWorldOutput, PaymentView, RunnerSettings, allow_all
from world.interactive_runner import (
    PreparedWorld,
    interactive_fast_output,
    prepare_world,
    run_interactive,
    run_interactive_world,
)
from world.interactive_state import RunnerState, start_runner_state
from world.outside_events import OutsideEvent
from world.run_hashes import hash_device_stamps, hash_ledger_entries, hash_people_devices
from world.run_identity import REPO_ROOT, load_runner_config

# The tiny world comes from the frozen-anchor config; the second world is one Level 11's own tests use.
FROZEN = load_yaml(REPO_ROOT / "config" / "frozen_anchors.yaml")["frozen_anchors"]["fraud_off_events"]
TINY = FastRunSettings(FROZEN["seed"], FROZEN["n_people"], FROZEN["n_shops"], FROZEN["n_days"], FROZEN["range_name"])
SECOND = FastRunSettings(seed=42, n_people=80, n_shops=15, n_days=61, range_name="training")
TEST_CODE_VERSION = "test-version"
FRAUD_ON = RunnerSettings(fraud_enabled=True)
MINUTES_PER_DAY = 1440
# The lien test looks for its debit from this event position on, so about a third of the run comes before it and the
# "every earlier event is unchanged" check covers real events.
LIEN_SEARCH_START = 1000


class CallRecorder:
    """
    WHAT: A call-recording proxy around a random generator. Every method call is passed to the real generator, and its
    name and arguments are appended to `calls`, so two runs' sequences of draws can be compared one by one.

    WHY: The run-stream guarantee is about the exact order of draws, not only their count. Recording names and
    arguments catches a draw that moved even when the totals happen to match.

    Args:
        generator: The real random generator to pass calls to.

    Returns:
        A CallRecorder with an empty `calls` list.

    Example:
        recorder = CallRecorder(numpy.random.default_rng(1)); recorder.uniform(0.0, 1.0) leaves
        recorder.calls == [("uniform", "(0.0, 1.0)", "[]")].
    """

    def __init__(self, generator: Any) -> None:
        """
        WHAT: Stores the real generator and starts an empty list of recorded calls.

        WHY: The proxy must forward to exactly the generator the run would have used, so its results stay identical
        and only the recording is added.

        Args:
            generator: The real random generator.

        Returns:
            None.

        Example:
            CallRecorder(numpy.random.default_rng(1)).calls == [].
        """
        # Keep the real generator and an empty record.
        self._generator = generator
        self.calls: list[tuple[str, str, str]] = []

    def __getattr__(self, name: str) -> Any:
        """
        WHAT: Returns a stand-in for the generator's method `name` that records the call (name, arguments, keyword
        arguments as text) and then returns the real method's result.

        WHY: Python calls this only for names the proxy does not have itself, so every generator method (uniform,
        integers, choice, poisson, lognormal and others) is recorded without listing them one by one.

        Args:
            name: The method name being looked up, e.g. "uniform".

        Returns:
            A function that records and forwards the call.

        Example:
            recorder.integers(0, 3) records ("integers", "(0, 3)", "[]") and returns the real integer.
        """
        # Look the real method up once.
        real_method = getattr(self._generator, name)

        def recorded(*args: Any, **kwargs: Any) -> Any:
            """
            WHAT: Writes the call's method name and arguments into the record, then calls the real method.

            WHY: Text forms of the arguments (repr) are exact and comparable, including the probability arrays that
            planning passes to choice.

            Args:
                args: The call's positional arguments.
                kwargs: The call's keyword arguments.

            Returns:
                Whatever the real method returns.

            Example:
                recorded(0.0, 1.0) for "uniform" appends ("uniform", "(0.0, 1.0)", "[]") and returns a float.
            """
            # Record first, then forward.
            self.calls.append((name, repr(args), repr(sorted(kwargs.items()))))
            return real_method(*args, **kwargs)

        return recorded


@dataclass
class ScriptedHook:
    """
    WHAT: A bank hook that answers from a script: "hold" for payment ids in `hold_ids` (only the first time each is
    asked when `hold_first_time_only` is True), "decline" for ids in `decline_ids` or for any payment of exactly
    `decline_amount` paise, and "allow" for everything else. It also keeps every PaymentView it was shown.

    WHY: Tests need a bank that intervenes on chosen payments only, so the effect of one hold or one decline can be
    seen against an otherwise unchanged world.

    Args:
        hold_ids: Planned payment ids to hold.
        decline_ids: Planned payment ids to decline.
        decline_amount: Decline any payment of this many paise, or None.
        hold_first_time_only: True to allow a held payment when it is asked again after a resume.
        seen: Every PaymentView shown, in order.

    Returns:
        A ScriptedHook.

    Example:
        ScriptedHook(hold_ids={"run-p-000005"})(PaymentView("run-p-000005", 9, "a", "b", 100)) returns "hold".
    """

    hold_ids: set[str] = field(default_factory=set)
    decline_ids: set[str] = field(default_factory=set)
    decline_amount: int | None = None
    hold_first_time_only: bool = False
    seen: list[PaymentView] = field(default_factory=list)

    def __call__(self, payment: PaymentView) -> str:
        """
        WHAT: Records the view and answers from the script: hold, decline, or allow.

        WHY: Keeping the views lets a test check what the bank was shown, for example that a resumed payment is asked
        again.

        Args:
            payment: The PaymentView.

        Returns:
            "hold", "decline" or "allow".

        Example:
            With decline_amount=15306555, a view with amount_paise 15306555 returns "decline".
        """
        # Count earlier questions about the same payment, keep what the bank saw, then follow the script.
        asked_before = sum(view.planned_payment_id == payment.planned_payment_id for view in self.seen)
        self.seen.append(payment)
        if payment.planned_payment_id in self.hold_ids and not (self.hold_first_time_only and asked_before):
            return "hold"
        if payment.planned_payment_id in self.decline_ids or payment.amount_paise == self.decline_amount:
            return "decline"
        return "allow"


def run_fraud_on_world(
    settings: FastRunSettings, inputs: FraudInputs = FraudInputs(), hook: Callable[[PaymentView], Any] = allow_all,
    state_factory: Callable[[PreparedWorld], RunnerState] | None = None,
) -> InteractiveWorldOutput:
    """
    WHAT: Runs a whole world with the fraud switch on and a fixed code version, so the run id (and with it every
    planned payment id) is the same in every test run.

    WHY: Tests pick payments by their planned payment id from one run and target them in another. A fixed code version
    keeps those ids stable even when files are being edited.

    Args:
        settings: The world settings.
        inputs: Fraud payments and outside events.
        hook: The bank hook.
        state_factory: Builds the runner state from the prepared world, or None.

    Returns:
        The InteractiveWorldOutput.

    Example:
        run_fraud_on_world(TINY).result.run_result.events has 2,915 events.
    """
    # Every test uses the same version string, so ids line up across runs.
    return run_interactive_world(
        settings, FRAUD_ON, inputs, hook, code_version=TEST_CODE_VERSION, state_factory=state_factory
    )


@functools.lru_cache(maxsize=None)
def baseline(settings: FastRunSettings) -> InteractiveWorldOutput:
    """
    WHAT: The allow-all, no-fraud run of a world, computed once per world and reused by the tests that need to find
    real payments to act on.

    WHY: Several tests read the baseline only (never change it); computing it once keeps the file quick.

    Args:
        settings: The world settings.

    Returns:
        The baseline InteractiveWorldOutput.

    Example:
        baseline(TINY).result.run_result.counts["attempts"] is 2915.
    """
    # Nothing is injected: no fraud, no outside events, the allow-all bank.
    return run_fraud_on_world(settings)


def opening_total(settings: FastRunSettings) -> int:
    """
    WHAT: Returns the total money in every account of a freshly prepared world before any payment runs.

    WHY: Money conservation compares the end total with this starting total (plus money created by accounts opened
    during the run). Payments may only move money.

    Args:
        settings: The world settings.

    Returns:
        The starting total in paise.

    Example:
        opening_total(TINY) is the sum of 72 opening balances (60 people and 12 shops).
    """
    # A fresh world has only its opening balances.
    world = prepare_world(settings, None).world
    return sum(ledger.balance_of(account) for ledger in world.ledgers for account in ledger.accounts)


def find_lien_target(settings: FastRunSettings) -> tuple[int, Any, int]:
    """
    WHAT: Finds a real successful one-piece payment D from a person's account in the baseline run, at or after event
    position LIEN_SEARCH_START, where no earlier event from the same account shares D's minute, and computes that
    account's balance just before D from the opening balance and every earlier successful event.

    WHY: A lien test needs a debit the account could pay, so the only reason it fails is the lien. Requiring no earlier
    same-minute debit from the account keeps every event before D untouched by the lien.

    Args:
        settings: The world settings.

    Returns:
        (D's position in the event list, D itself, the account's balance just before D in paise).

    Example:
        For the seed-101 world: (1000, the event of acc_person_000006 paying 18,008 paise at minute 30,231, 2220327).
    """
    # Start from opening balances and walk the baseline events in order.
    world = prepare_world(settings, None).world
    balances = {account: ledger.balance_of(account) for ledger in world.ledgers for account in ledger.accounts}
    events = baseline(settings).result.run_result.events
    for position, event in enumerate(events):
        earlier_same_minute = any(
            other.from_account == event.from_account and other.minute == event.minute for other in events[:position]
        )
        suitable = event.status == "success" and event.from_account.startswith("acc_person")
        if position >= LIEN_SEARCH_START and suitable and not earlier_same_minute:
            return position, event, balances[event.from_account]
        # Keep balances current: a successful event moves its amount from sender to receiver.
        if event.status == "success":
            balances[event.from_account] -= event.amount_paise
            balances[event.to_account] += event.amount_paise
    raise AssertionError("no suitable debit found in the baseline run")


def assert_money_rules_hold(output: InteractiveWorldOutput, settings: FastRunSettings) -> None:
    """
    WHAT: Checks Level 11's three money rules on a run's ledgers: no negative balance, every balance equals the sum of
    its entries, and the total equals the opening total plus money created by registered accounts.

    WHY: Liens, holds and declines must block payments without moving or losing money. These are Level 11's own checks,
    so a pass means the same thing it means for Level 11.

    Args:
        output: The run to check.
        settings: Its world settings, for the opening total.

    Returns:
        None. The test fails with Level 11's own message if a rule breaks.

    Example:
        For a run with one registered account opened with 500,000 paise, the expected total is opening total + 500,000.
    """
    # Level 11's three money checks, with the run's created money added to the expected total.
    ledgers = output.world.ledgers
    expected_total = opening_total(settings) + output.result.money_created_paise
    assert check_no_negative_balances(ledgers) == []
    assert check_balances_equal_entry_sums(ledgers) == []
    assert check_money_conserved(ledgers, expected_total) == []


def test_level_11_private_helpers_keep_their_parameters() -> None:
    """
    WHAT: Checks the parameter names of every private Level 11 helper the runner calls, so any change to them fails
    here with a clear message instead of somewhere deep in a run.

    WHY: The runner calls Level 11's private helpers directly rather than copying them (module notes in
    world/interactive_runner.py). This test is one of the guards on that choice.

    Args:
        None.

    Returns:
        None. The test fails if a helper's parameters changed.

    Example:
        _handle_live_payment's parameters are (world, payment, leg, f_bank, t_bank).
    """
    # The parameters each helper had when the runner was written.
    expected = {
        "_collect_all_planned_payments": ["world", "plan", "rng"],
        "_merge_planned_payments": ["normal", "episode_payments"],
        "_handle_live_payment": ["world", "payment", "leg", "f_bank", "t_bank"],
        "_record_event_counts": ["ev", "counts"],
        "_build_account_bank_map": ["ledgers"],
        "_apply_due_device_events": ["device_events", "dev_idx", "up_to_minute", "p_by_id", "dev_state"],
    }
    for name, parameters in expected.items():
        actual = list(inspect.signature(getattr(calendar_runner, name)).parameters)
        assert actual == parameters, f"Level 11 helper {name} changed its parameters to {actual}; update the runner."


@pytest.mark.parametrize("settings", [TINY, SECOND], ids=["seed101_60x12", "seed42_80x15"])
def test_empty_fraud_plan_equals_level_11_everywhere(settings: FastRunSettings) -> None:
    """
    WHAT: Test T2. Runs each world with the switch on, no fraud and the allow-all bank, and compares it with Level 11's
    run_fast_world on five outputs: the RunResult, the event-lines hash, the ledger hash, the people's device hash and
    the sender device stamp hash.

    WHY: The fraud-on loop re-implements run_world's order with a queue. If it adds or moves anything on its own, one of
    these five outputs differs. Two different worlds guard against a match that holds only by chance.

    Args:
        settings: The world settings for this case.

    Returns:
        None. The test fails if any of the five outputs differs.

    Example:
        For the seed-101 world both runs give 2,915 events and the event-lines hash 24fc0e25009d7946...
    """
    # Level 11's run and the runner's empty fraud-on run.
    level_11 = run_fast_world(settings)
    runner = baseline(settings)
    runner_fast = interactive_fast_output(runner)
    # All five outputs must match.
    assert runner.result.run_result == level_11.result
    assert hashlib.sha256("".join(event_lines(runner_fast)).encode()).hexdigest() == hashlib.sha256(
        "".join(event_lines(level_11)).encode()).hexdigest()
    assert hash_ledger_entries(runner.world.ledgers) == hash_ledger_entries(level_11.world.ledgers)
    assert hash_people_devices(runner.world.people) == hash_people_devices(level_11.world.people)
    assert hash_device_stamps(runner.result.run_result.event_sender_devices) == hash_device_stamps(
        level_11.result.event_sender_devices)


def busy_fraud_inputs(settings: FastRunSettings) -> tuple[FraudInputs, ScriptedHook]:
    """
    WHAT: Builds a run where everything happens at once: a lien that blocks a real debit, a debit hold and its release
    one day later on another account, one payment held (never resumed), one declined, and one fraud payment.

    WHY: Test T3 must show genuine draws stay in place with all features active together, not one at a time.

    Args:
        settings: The world settings.

    Returns:
        (the fraud inputs, the scripted hook).

    Example:
        For the seed-101 world: the lien targets the first suitable debit, the held payment is planned payment 100 and
        the declined one is planned payment 200.
    """
    # A lien that blocks a real debit (see find_lien_target).
    _, debit, balance_before = find_lien_target(settings)
    lien = OutsideEvent(debit.minute, "lien", debit.from_account, balance_before - debit.amount_paise + 1)
    # A debit hold on a different account, released one day later.
    planned_ids = baseline(settings).result.planned_payment_ids
    events = baseline(settings).result.run_result.events
    other = next(event for event in events[500:] if event.from_account not in (debit.from_account,)
                 and event.from_account.startswith("acc_person"))
    hold = OutsideEvent(other.minute, "debit_hold", other.from_account, encumbrance_id="enc-hold")
    release = OutsideEvent(other.minute + MINUTES_PER_DAY, "release", encumbrance_id="enc-hold")
    # One fraud payment between two real accounts, using a Level 11 purpose.
    fraud = PlannedPayment(other.minute, other.from_account, events[0].to_account, 1000, 0, "everyday")
    hook = ScriptedHook(hold_ids={planned_ids[100]}, decline_ids={planned_ids[200]})
    return FraudInputs(payments=(fraud,), outside_events=(lien, hold, release)), hook


def record_streams(monkeypatch: pytest.MonkeyPatch) -> dict[str, CallRecorder]:
    """
    WHAT: Replaces the loop's build_component_streams for one test with a version that wraps every runner stream
    (children 4 and 5) in a CallRecorder, and returns those recorders by component name once a run has built them.

    WHY: Test T3 must show the fraud payment's draw landed on child 4, and test T5 that a resumed payment drew from
    child 5. Recording the streams proves which stream was used instead of assuming it.

    Args:
        monkeypatch: pytest's monkeypatch helper, which restores the real function after the test.

    Returns:
        A dictionary that is filled with name -> CallRecorder when a run builds its streams.

    Example:
        After a run with one fraud payment, recorders["fraud_payment_failures"].calls holds at least one "uniform".
    """
    # Build the real streams, then wrap each one.
    recorders: dict[str, CallRecorder] = {}
    real_builder = interactive_loop.build_component_streams

    def recording_builder(seed: int, children: dict[str, int]) -> dict[str, Any]:
        """
        WHAT: Calls the real builder, wraps each stream in a CallRecorder, and keeps the recorders for the test.

        WHY: The run then draws through the recorders, so the test can read which stream took which draws.

        Args:
            seed: The world's seed.
            children: Component name -> child number.

        Returns:
            Component name -> CallRecorder.

        Example:
            recording_builder(101, {"fraud_payment_failures": 4}) wraps child 4 of seed 101.
        """
        # Wrap the real streams so every draw is written down.
        for name, stream in real_builder(seed, children).items():
            recorders[name] = CallRecorder(stream)
        return dict(recorders)

    monkeypatch.setattr(interactive_loop, "build_component_streams", recording_builder)
    return recorders


def test_genuine_draws_never_move(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    WHAT: Test T3. Records every call made on the run stream, first by Level 11's run_world and then by the fraud-on
    runner with a lien, a debit hold and its release, one held payment, one declined payment and one fraud payment.
    The two records must be identical, call for call, and the fraud payment's draw must appear on child 4.

    WHY: This is the run-stream guarantee: whatever fraud and banks do, every genuine payment reads exactly the random
    numbers it read in Level 11.

    Args:
        monkeypatch: pytest's monkeypatch helper, used to record the runner's own streams.

    Returns:
        None. The test fails if any recorded call differs or the fraud draw is not on child 4.

    Example:
        For the seed-101 world both records hold 14,439 calls (11,491 planning calls plus 2,948 failure draws).
    """
    # Level 11's own run, through a recorder.
    level_11_world = prepare_world(TINY, None)
    level_11_recorder = CallRecorder(level_11_world.run_rng)
    run_world(level_11_world.world, level_11_world.plan, level_11_recorder,  # type: ignore[arg-type]
              device_events=level_11_world.device_events, episode_payments=level_11_world.episode_payments)
    # The busy fraud-on run, through a recorder, with the runner's own streams recorded too.
    stream_recorders = record_streams(monkeypatch)
    inputs, hook = busy_fraud_inputs(TINY)
    runner_world = prepare_world(TINY, None)
    runner_recorder = CallRecorder(runner_world.run_rng)
    result = run_interactive(
        runner_world.world, runner_world.plan, runner_recorder,  # type: ignore[arg-type]
        runner_world.device_events, runner_world.episode_payments, settings=FRAUD_ON, fraud_inputs=inputs,
        hook=hook, run_id=baseline(TINY).result.run_id, runner_config=load_runner_config(),
    )
    # The busy run really did hold, decline, block and add a fraud payment.
    reasons = {event.reason for event in result.run_result.events}
    assert {REASON_FUNDS_ON_HOLD, REASON_DECLINED_BY_BANK} <= reasons and len(result.held_payments) == 1
    assert "fraud" in result.event_sources
    # Same calls, in the same order, with the same arguments.
    print(f"MEASURED T3: level11_calls={len(level_11_recorder.calls)} runner_calls={len(runner_recorder.calls)} "
          f"child4_calls={len(stream_recorders['fraud_payment_failures'].calls)}")
    assert runner_recorder.calls == level_11_recorder.calls
    assert stream_recorders["fraud_payment_failures"].calls[0][0] == "uniform"


def test_lien_blocks_exactly_the_chosen_debit() -> None:
    """
    WHAT: Test T4. Takes a real debit D (account X, minute m, amount a) from the baseline run, recomputes X's balance b
    just before D, and places a lien of b - a + 1 on X at minute m. Then the available balance is b - (b - a + 1) =
    a - 1, one paise short. D must fail as sender_bank / funds_on_hold, every event before D must equal the baseline,
    and Level 11's three money rules must hold.

    WHY: The lien must block exactly what the arithmetic says, at exactly the right moment, and must not move money.

    Args:
        None.

    Returns:
        None. The test fails if D does not fail with funds_on_hold, an earlier event differs, or a money rule breaks.

    Example:
        Seed-101 world: D is event 1,000, acc_person_000006 paying 18,008 paise at minute 30,231 with b = 2,220,327;
        the lien is 2,220,327 - 18,008 + 1 = 2,202,320, leaving 18,007 paise available, one short.
    """
    # The debit to block and the account's balance just before it.
    position, debit, balance_before = find_lien_target(TINY)
    lien_amount = balance_before - debit.amount_paise + 1
    assert balance_before - lien_amount == debit.amount_paise - 1
    lien = OutsideEvent(debit.minute, "lien", debit.from_account, lien_amount)
    output = run_fraud_on_world(TINY, FraudInputs(outside_events=(lien,)))
    events = output.result.run_result.events
    # D is blocked by the lien, and nothing before it changed.
    blocked = events[position]
    assert (blocked.minute, blocked.from_account, blocked.amount_paise) == (
        debit.minute, debit.from_account, debit.amount_paise)
    assert (blocked.status, blocked.failure_domain, blocked.reason) == ("failed", "sender_bank", REASON_FUNDS_ON_HOLD)
    assert events[:position] == baseline(TINY).result.run_result.events[:position]
    assert_money_rules_hold(output, TINY)


def test_held_payment_moves_nothing_until_resumed(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    WHAT: Test T5. Holds planned payment number 100 of the baseline run. It must leave no event and no ledger entry and
    be listed in held_payments. In a second run, a resume_payment one day later lets it go ahead: it is recorded once,
    at the resume minute, as "resumed", with its original planned payment id and its original sender stamp, and its
    draw comes from child 5.

    WHY: A hold must freeze a payment completely, and a resume must replay it without stamping the phone again (which
    could wrongly mark a known phone as new) and without touching the run stream.

    Args:
        monkeypatch: pytest's monkeypatch helper, used to record the runner's own streams.

    Returns:
        None. The test fails if the held payment leaves a trace, or the resumed one differs from what is described.

    Example:
        Planned payment 100 at minute m is held; resumed at m + 1,440 it appears once with source "resumed".
    """
    # The payment to hold, as it looked in the baseline run.
    base = baseline(TINY).result
    target_id = base.planned_payment_ids[100]
    original = base.run_result.events[base.planned_payment_ids.index(target_id)]
    original_stamp = base.run_result.event_sender_devices[base.planned_payment_ids.index(target_id)]
    held_run = run_fraud_on_world(TINY, hook=ScriptedHook(hold_ids={target_id}))
    # Held: no event, no ledger entry for it, and listed as held.
    assert target_id not in held_run.result.planned_payment_ids
    transfer_id = f"tx_{original.minute}_{original.from_account}_{original.to_account}_{original.amount_paise}"
    assert all(entry.transfer_id != transfer_id for ledger in held_run.world.ledgers for entry in ledger.entries)
    assert [view.planned_payment_id for view in held_run.result.held_payments] == [target_id]
    # Resumed one day later: one event, resumed, same id and stamp, drawn from child 5.
    recorders = record_streams(monkeypatch)
    resume = OutsideEvent(original.minute + MINUTES_PER_DAY, "resume_payment", planned_payment_id=target_id)
    resume_hook = ScriptedHook(hold_ids={target_id}, hold_first_time_only=True)
    resumed_run = run_fraud_on_world(TINY, FraudInputs(outside_events=(resume,)), resume_hook)
    resumed_result = resumed_run.result
    # The bank was asked twice about this payment: at its planned minute and again when it was resumed.
    assert [view.minute for view in resume_hook.seen if view.planned_payment_id == target_id] == [
        original.minute, original.minute + MINUTES_PER_DAY]
    position = resumed_result.planned_payment_ids.index(target_id)
    assert resumed_result.planned_payment_ids.count(target_id) == 1
    assert resumed_result.event_sources[position] == "planned"
    assert resumed_result.events_resumed[position] is True
    assert len(resumed_result.events_resumed) == len(resumed_result.run_result.events)
    assert resumed_result.run_result.events[position].minute == original.minute + MINUTES_PER_DAY
    assert resumed_result.run_result.event_sender_devices[position] == original_stamp
    assert len(recorders["resumed_payment_failures"].calls) >= 1
    assert_money_rules_hold(resumed_run, TINY)


def test_held_fraud_payment_keeps_origin_when_resumed(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    WHAT: Injects one fraud payment, holds it with a ScriptedHook the first time it is processed, then resumes it one
    day later with a resume_payment outside event. Asserts that the recorded event carries source "fraud", that
    events_resumed is True, that failure draws come from child 5, that run-stream calls match the baseline call for
    call, and that money rules hold.

    WHY: Proves that first_look and resume preserve the true origin ("fraud") even when the stream source becomes
    "resumed", that child 5 is used for draws, and that genuine run-stream draws never move.

    Args:
        monkeypatch: pytest helper used to record calls across runner streams.

    Returns:
        None. Fails if the resumed event drops origin "fraud" or alters genuine draws.

    Example:
        A held fraud payment resumed at minute m + 1440 has event_sources[pos] == "fraud" and events_resumed == True.
    """
    # 1. Prepare one fraud payment between real accounts and find its planned payment id.
    events = baseline(TINY).result.run_result.events
    other = next(ev for ev in events[500:] if ev.from_account.startswith("acc_person"))
    fraud = PlannedPayment(other.minute, other.from_account, events[0].to_account, 1000, 0, "everyday")
    prev = run_fraud_on_world(TINY, FraudInputs(payments=(fraud,)))
    match_pos = next(i for i, ev in enumerate(prev.result.run_result.events)
                     if ev.minute == fraud.minute and ev.amount_paise == 1000)
    target_id = prev.result.planned_payment_ids[match_pos]

    # 2. Record genuine Level 11 run stream calls as the baseline.
    lvl11 = prepare_world(TINY, None)
    base_rec = CallRecorder(lvl11.run_rng)
    run_world(lvl11.world, lvl11.plan, base_rec,
              device_events=lvl11.device_events, episode_payments=lvl11.episode_payments)

    # 3. Execute the fraud run with the fraud payment held then resumed, recording streams.
    streams = record_streams(monkeypatch)
    r_world = prepare_world(TINY, None)
    r_rec = CallRecorder(r_world.run_rng)
    hook = ScriptedHook(hold_ids={target_id}, hold_first_time_only=True)
    resume = OutsideEvent(fraud.minute + MINUTES_PER_DAY, "resume_payment", planned_payment_id=target_id)
    result = run_interactive(
        r_world.world, r_world.plan, r_rec, r_world.device_events, r_world.episode_payments,
        settings=FRAUD_ON, fraud_inputs=FraudInputs(payments=(fraud,), outside_events=(resume,)),
        hook=hook, run_id=baseline(TINY).result.run_id, runner_config=load_runner_config(),
    )

    # 4. Assert origin is preserved, resumed flag is True, draw is from child 5, and money rules hold.
    pos = result.planned_payment_ids.index(target_id)
    assert result.event_sources[pos] == "fraud" and result.events_resumed[pos] is True
    assert len(result.events_resumed) == len(result.run_result.events)
    assert len(streams["resumed_payment_failures"].calls) >= 1 and r_rec.calls == base_rec.calls
    out = InteractiveWorldOutput(
        r_world.world, r_world.plan, [], {}, tuple(r_world.device_events), result, {}, TEST_CODE_VERSION)
    assert_money_rules_hold(out, TINY)


def test_declined_payment_records_every_piece_and_moves_nothing() -> None:
    """
    WHAT: Test T6. Adds a fraud payment of 15,306,555 paise (three pieces at the 6,122,622-paise limit) and a bank that
    declines it, and also declines planned payment 200. Every piece of each must be recorded as sender_bank /
    declined_by_bank, one event per piece, and no ledger entry may exist for them.

    WHY: A decline is a refusal the bank makes visible: each piece is recorded so the feed shows what was tried, but no
    money moves.

    Args:
        None.

    Returns:
        None. The test fails if a piece is missing, recorded differently, or moved money.

    Example:
        15,306,555 = 6,122,622 + 6,122,622 + 3,061,311, so the fraud payment gives three declined events.
    """
    # A large fraud payment from a person to a shop, and the planned payment to decline.
    base = baseline(TINY).result
    sender = base.run_result.events[0].from_account
    fraud = PlannedPayment(3000, sender, "acc_shop_0001", 15306555, 0, "everyday")
    hook = ScriptedHook(decline_ids={base.planned_payment_ids[200]}, decline_amount=15306555)
    output = run_fraud_on_world(TINY, FraudInputs(payments=(fraud,)), hook)
    result = output.result
    # Collect the events of the two declined payments.
    fraud_positions = [i for i, source in enumerate(result.event_sources) if source == "fraud"]
    planned_positions = [i for i, pid in enumerate(result.planned_payment_ids) if pid == base.planned_payment_ids[200]]
    assert [result.run_result.events[i].amount_paise for i in fraud_positions] == [6122622, 6122622, 3061311]
    for position in fraud_positions + planned_positions:
        event = result.run_result.events[position]
        assert (event.status, event.failure_domain, event.reason) == ("failed", "sender_bank", REASON_DECLINED_BY_BANK)
    # No ledger entry belongs to the declined fraud payment, and money is conserved.
    assert all(not entry.transfer_id.startswith(f"tx_3000_{sender}_acc_shop_0001_")
               for ledger in output.world.ledgers for entry in ledger.entries)
    assert_money_rules_hold(output, TINY)


def test_debit_hold_blocks_debits_until_release_and_never_credits() -> None:
    """
    WHAT: Test T7. Places a debit hold on the person account with the most activity from day 10 to day 20, then
    releases it. Every debit from that account inside the window must fail (as funds_on_hold unless a drawn failure or
    a too-low balance came first), at least one as funds_on_hold; credits into it must never fail as funds_on_hold and
    at least one must succeed; after the release at least one debit must succeed again.

    WHY: A debit hold freezes spending, not receiving, and ends exactly at its release.

    Args:
        None.

    Returns:
        None. The test fails if a debit escapes the hold, a credit is blocked, or the account stays blocked.

    Example:
        Window: minute 14,400 (day 10) to minute 28,800 (day 20), 10 days of 1,440 minutes each.
    """
    # The window and the busiest person account in it.
    start, end = 10 * MINUTES_PER_DAY, 20 * MINUTES_PER_DAY
    events = baseline(TINY).result.run_result.events
    in_window = [event for event in events if start <= event.minute < end]
    accounts = {event.from_account for event in in_window if event.from_account.startswith("acc_person")}
    account = max(sorted(accounts), key=lambda a: min(sum(e.from_account == a for e in in_window),
                                                       sum(e.to_account == a for e in in_window)))
    hold = OutsideEvent(start, "debit_hold", account, encumbrance_id="enc-window")
    release = OutsideEvent(end, "release", encumbrance_id="enc-window")
    result = run_fraud_on_world(TINY, FraudInputs(outside_events=(hold, release))).result.run_result.events
    # Inside the window: every debit fails, credits are never blocked.
    debits = [e for e in result if e.from_account == account and start <= e.minute < end]
    credits = [e for e in result if e.to_account == account and start <= e.minute < end]
    assert debits and all(e.status == "failed" for e in debits)
    assert any(e.reason == REASON_FUNDS_ON_HOLD for e in debits)
    assert credits and all(e.reason != REASON_FUNDS_ON_HOLD for e in credits) and any(
        e.status == "success" for e in credits)
    # After the release the account can pay again.
    assert any(e.status == "success" for e in result if e.from_account == account and e.minute >= end)


def test_bad_inputs_during_a_run_are_refused() -> None:
    """
    WHAT: Part of test T8 that needs a running world: a hook answering "maybe", a release of an unknown encumbrance, a
    resume of a payment that is not held, and a fraud payment from an account that was never registered. Each must
    raise ValueError with its plain message.

    WHY: These mistakes can only be seen once the run is going. Each must stop the run with a sentence that names the
    cause instead of a bare KeyError or a silently wrong world.

    Args:
        None.

    Returns:
        None. The test fails if any case runs on or raises a different message.

    Example:
        A fraud payment from "acc_never_opened" raises "account acc_never_opened is not registered with the runner".
    """
    # A bank that gives an answer outside the three allowed ones.
    with pytest.raises(ValueError, match="it must answer allow, hold or decline"):
        run_fraud_on_world(TINY, hook=lambda view: "maybe")
    # Ending a block that never existed.
    with pytest.raises(ValueError, match="was never placed"):
        run_fraud_on_world(TINY, FraudInputs(outside_events=(OutsideEvent(60, "release", encumbrance_id="enc-x"),)))
    # Resuming a payment nobody held.
    with pytest.raises(ValueError, match="is not held, so it cannot be resumed"):
        run_fraud_on_world(TINY, FraudInputs(
            outside_events=(OutsideEvent(60, "resume_payment", planned_payment_id="nobody-p-000000"),)))
    # Paying from an account the runner was never told about.
    stranger = PlannedPayment(60, "acc_never_opened", "acc_shop_0001", 1000, 0, "everyday")
    with pytest.raises(ValueError, match="account acc_never_opened is not registered with the runner"):
        run_fraud_on_world(TINY, FraudInputs(payments=(stranger,)))


def register_new_account(prepared: PreparedWorld) -> RunnerState:
    """
    WHAT: Builds the runner state for a prepared world and registers one new account, "acc_new_000001", in bank 1 with
    an opening balance of 500,000 paise (Rs 5,000) and no owner phone.

    WHY: Test T9 needs an account that did not exist when Level 11 built the world, opened the supported way.

    Args:
        prepared: The prepared world.

    Returns:
        The runner state with the new account registered.

    Example:
        register_new_account(prepare_world(TINY, None)).money_created_paise == 500000.
    """
    # Same tables as run_world, plus the new account.
    state = start_runner_state(prepared.world)
    state.register_account("acc_new_000001", 1, None, 500000, 0)
    return state


def test_registered_account_can_pay_and_money_is_counted() -> None:
    """
    WHAT: Test T9. Registers a new account with 500,000 paise and adds a fraud payment of 1,000 paise from it to a shop
    at minute 600. The payment must succeed, money_created_paise must be 500,000, and the money rules must hold with
    that 500,000 added to the expected total.

    WHY: Later parts open mule accounts mid-run. They must be able to pay at once, and the money they start with must be
    counted as created, or conservation would report a false leak.

    Args:
        None.

    Returns:
        None. The test fails if the payment fails or the money does not add up.

    Example:
        Expected total = opening total + 500,000 paise.
    """
    # The new account pays a shop 1,000 paise at minute 600.
    payment = PlannedPayment(600, "acc_new_000001", "acc_shop_0001", 1000, 0, "everyday")
    output = run_fraud_on_world(TINY, FraudInputs(payments=(payment,)), state_factory=register_new_account)
    position = output.result.event_sources.index("fraud")
    event = output.result.run_result.events[position]
    assert (event.from_account, event.status) == ("acc_new_000001", "success")
    assert output.result.money_created_paise == 500000
    assert_money_rules_hold(output, TINY)


def test_same_inputs_give_the_same_result() -> None:
    """
    WHAT: Test T10. Runs the busy fraud-on world of test T3 twice with identical inputs and compares the two complete
    InteractiveRunResults with ==.

    WHY: Every fraud world must be repeatable from its inputs alone; a hidden source of randomness or order would make
    results impossible to audit.

    Args:
        None.

    Returns:
        None. The test fails if any field of the two results differs.

    Example:
        Both runs give the same events, ids, sources, held payments and lien log.
    """
    # Two runs from the same inputs.
    inputs, hook = busy_fraud_inputs(TINY)
    first = run_fraud_on_world(TINY, inputs, hook).result
    second = run_fraud_on_world(TINY, inputs, ScriptedHook(hook.hold_ids, hook.decline_ids)).result
    assert first == second
