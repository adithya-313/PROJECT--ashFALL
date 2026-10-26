"""
world/hold_replay.py

Two-pass administrative hold replay pipeline with lien injection through EncumbranceBook.
Executes Pass 1 to generate complaints and compute composite hold latencies, injects
resulting holds as outside lien events, and runs Pass 2 with exact balance reconstruction,
outcome classification, and naive single-hop inter-bank forwarding under Amendment A-029.

Terms defined:
    lien: a legal claim or hold placed by a bank on an account balance, preventing the
        funds from being withdrawn or transferred while remaining within the account.
    hold: an administrative freeze directive instructing a receiving bank to encumber
        funds associated with a suspected fraudulent transaction under the 1930 flow.
    one-hop: tracing funds strictly from the first recipient account to the immediate
        second recipient account without extending further downstream.
    too_late: an outcome occurring when an account has already been depleted before
        the administrative freeze directive lands at the receiving bank.
    two-pass replay: running the simulation twice with identical seeds, first to observe
        scam complaints and compute hold latencies, and second with the resulting holds
        injected as lien events into the encumbrance book.
"""

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from core.config_validator import load_yaml
from core.event_clock import world_minute_to_ist
from nationalsim.hold_flow import (
    HoldFlowTracker,
    HoldOutcome,
    HoldRequest,
    OutgoingTransfer,
    classify_hold_outcome,
    compute_hold_latency,
    decide_one_hop_forwarding,
    is_night_hour,
)
from world.config_reader import draw_in_range, get_parameter_entry
from world.fast_mode import FastRunSettings
from world.gang_config import GangConfig
from world.interactive_records import FraudInputs, InteractiveWorldOutput, RunnerSettings
from world.interactive_runner import run_interactive_world
from world.label_builder import build_label_events
from world.ledger import Ledger
from world.outside_events import OutsideEvent
from world.run_identity import REPO_ROOT

# Named integer salt isolating hold replay random draws from world RNG children 0-3.
HOLD_REPLAY_SALT: int = 13003


@dataclass(frozen=True)
class HoldReplayResult:
    """Immutable audit record encapsulating the two-pass hold replay outputs and classifications.

    What it is for:
        Provides structured inspection of Pass 1 and Pass 2 world outputs, generated
        primary hold directives, four-way outcome classifications, and forwarded holds.

    Inputs:
        pass1_output: Completed InteractiveWorldOutput from the initial baseline run.
        pass2_output: Completed InteractiveWorldOutput from the lien-injected replay run.
        primary_holds: List of primary HoldRequest directives issued from victim complaints.
        primary_outcomes: List of HoldOutcome records evaluated against pass-2 reconstructed balances.
        forwarded_holds: List of naive single-hop forwarded HoldRequest directives.
        forwarded_outcomes: List of HoldOutcome records evaluated for forwarded requests.

    What it returns:
        Frozen HoldReplayResult instance.

    How it works:
        Stores immutable references to pass outputs and evaluated hold directives.

    Errors and side effects:
        None. Dataclass is frozen.

    Limits:
        Encapsulates results across the two simulation passes only.
    """

    pass1_output: InteractiveWorldOutput
    pass2_output: InteractiveWorldOutput
    primary_holds: list[HoldRequest]
    primary_outcomes: list[HoldOutcome]
    forwarded_holds: list[HoldRequest]
    forwarded_outcomes: list[HoldOutcome]


def load_hold_configuration(nationalsim_config_path: Path | None = None) -> dict[str, Any]:
    """Load diurnal latency multipliers and operational windows from configuration.

    What it is for:
        Reads config/nationalsim.yaml parameters for night-time scaling and weekend
        multipliers without silent defaults or broad exceptions.

    Inputs:
        nationalsim_config_path: Optional filesystem Path to configuration file.
            Example: Path('config/nationalsim.yaml').

    What it returns:
        Dictionary holding 'night_hours', 'night_multiplier', and 'weekend_multiplier'.

    How it works:
        1. Resolve path to config/nationalsim.yaml.
        2. Verify file presence and load YAML mapping.
        3. Extract 'night_hours', 'night_multiplier', and 'weekend_multiplier' blocks.
        4. Validate structure and convert to typed primitives.

    Errors and side effects:
        Raises ValueError if configuration file or required parameters are missing.

    Limits:
        Reads static YAML configuration parameters.
    """
    path = nationalsim_config_path or (REPO_ROOT / "config" / "nationalsim.yaml")
    if not path.exists():
        raise ValueError(f"Configuration file does not exist: {path}")

    raw = load_yaml(path)
    if "nationalsim" not in raw or not isinstance(raw["nationalsim"], dict):
        raise ValueError(f"Missing 'nationalsim' block in configuration file: {path}")

    ns = raw["nationalsim"]
    for required_key in ("night_hours", "night_multiplier", "weekend_multiplier"):
        if required_key not in ns:
            raise ValueError(f"Missing '{required_key}' in {path}")
        entry = ns[required_key]
        if not isinstance(entry, dict) or "value" not in entry or entry["value"] is None:
            raise ValueError(f"Missing value dictionary for '{required_key}' in {path}")

    night_hours_val = ns["night_hours"]["value"]
    if not isinstance(night_hours_val, (list, tuple)) or len(night_hours_val) != 2:
        raise ValueError(f"Malformed 'night_hours' in {path}: expected 2-element sequence, got {night_hours_val}")

    return {
        "night_hours": (int(night_hours_val[0]), int(night_hours_val[1])),
        "night_multiplier": float(ns["night_multiplier"]["value"]),
        "weekend_multiplier": float(ns["weekend_multiplier"]["value"]),
    }


def reconstruct_account_balance(ledger: Ledger, account_id: str, minute: int) -> int:
    """Reconstruct an account's point-in-time balance by replaying ledger entry history.

    What it is for:
        Rebuilds exact historical balances at landing minute directly from double-entry
        ledger records without relying on ephemeral runtime state.

    Inputs:
        ledger: The authoritative bank Ledger containing accounts and entries.
        account_id: Account identifier whose balance is being queried (e.g., 'acc_mule_01').
        minute: Simulation minute timestamp representing the point-in-time boundary.

    What it returns:
        Integer balance in paise present in the account at the specified minute.

    How it works:
        1. Accumulate initial funding from opening entries with entry.minute <= minute.
        2. Accumulate signed transaction amounts (debits negative, credits positive)
           for all non-opening entries with entry.minute < minute.
        3. Return total accumulated integer paise.

    Errors and side effects:
        None. Reads immutable ledger entries.

    Limits:
        Does not deduct active liens; returns physical ledger balance.
    """
    accumulated_balance_paise = 0
    for entry in ledger.entries:
        if entry.account_id != account_id:
            continue
        if entry.kind == "opening":
            if entry.minute <= minute:
                accumulated_balance_paise += entry.amount_paise
        else:
            if entry.minute < minute:
                accumulated_balance_paise += entry.amount_paise
    return accumulated_balance_paise


def run_two_pass_replay(
    fast_settings: FastRunSettings,
    runner_settings: RunnerSettings | None = None,
    fraud_inputs: FraudInputs | None = None,
    gang_configs: Sequence[GangConfig] | None = None,
    code_version: str | None = None,
    nationalsim_config_path: Path | None = None,
    configs: Any = None,
) -> HoldReplayResult:
    """Execute the full two-pass administrative hold replay pipeline.

    What it is for:
        Implements Amendment A-029: runs Pass 1, builds victim complaints, computes
        composite hold latencies with bank and temporal factors, injects non-missed
        holds as outside lien events, executes Pass 2, reconstructs point-in-time
        balances, classifies hold outcomes, and performs naive single-hop forwarding.

    Inputs:
        fast_settings: Sizing and calendar configuration for the world run.
            Example: FastRunSettings(101, 60, 12, 61, 'training').
        runner_settings: Optional simulation feature flags (default fraud_enabled=True).
        fraud_inputs: Optional pre-configured fraud inputs.
        gang_configs: Optional sequence of gang configurations to execute.
        code_version: Optional version identifier string.
        nationalsim_config_path: Optional override path for config/nationalsim.yaml.

    What it returns:
        HoldReplayResult containing Pass 1 and Pass 2 outputs and all hold audit records.

    How it works:
        1. Execute Pass 1 simulation with fraud enabled to observe baseline transactions.
        2. If fraud is disabled, return identical pass outputs with empty hold lists.
        3. Build victim complaints using world/label_builder.
        4. Initialize isolated random generator using seed and HOLD_REPLAY_SALT.
        5. Read parameter ranges from world_ranges and config/nationalsim.yaml.
        6. For each complaint, compute composite hold latency and determine missed status.
        7. Enforce idempotency via HoldFlowTracker so each complaint produces at most one hold.
        8. Construct OutsideEvent liens for non-missed holds and inject into Pass 2 inputs.
        9. Re-run identical simulation seed with injected liens as Pass 2.
        10. Reconstruct pass-2 balances at landing minute and classify each primary hold.
        11. For unencumbered funds that departed before landing minute, inspect ledger
            and generate naive one-hop forwarded hold directives.
        12. Evaluate forwarded hold outcomes on pass-2 state without placing additional liens.

    Errors and side effects:
        Raises ValueError if required configurations or split parameters are missing.

    Limits:
        Forwarded holds are evaluated on pass-2 state and are not placed as liens (no third pass).
    """
    active_runner_settings = (
        runner_settings if runner_settings is not None else RunnerSettings(fraud_enabled=True)
    )
    active_fraud_inputs = fraud_inputs if fraud_inputs is not None else FraudInputs()
    active_gangs = tuple(gang_configs) if gang_configs is not None else ()

    # 1. Execute Pass 1 baseline world
    pass1_output = run_interactive_world(
        fast_settings,
        runner_settings=active_runner_settings,
        fraud_inputs=active_fraud_inputs,
        gang_configs=active_gangs,
        configs=configs,
        code_version=code_version,
    )

    # 2. If fraud is off, return identical runs with no holds
    if not active_runner_settings.fraud_enabled:
        return HoldReplayResult(
            pass1_output=pass1_output,
            pass2_output=pass1_output,
            primary_holds=[],
            primary_outcomes=[],
            forwarded_holds=[],
            forwarded_outcomes=[],
        )

    # 3. Extract victim complaints from Pass 1 output
    complaints = build_label_events(pass1_output)
    if not complaints:
        # No complaints generated in Pass 1: Pass 2 is identical
        return HoldReplayResult(
            pass1_output=pass1_output,
            pass2_output=pass1_output,
            primary_holds=[],
            primary_outcomes=[],
            forwarded_holds=[],
            forwarded_outcomes=[],
        )

    # 4. Resolve temporal hold configuration parameters
    hold_cfg = load_hold_configuration(nationalsim_config_path)
    effective = pass1_output.effective_config
    world_ranges = effective["world_configs"]["world_ranges"]
    split = effective["settings"]["range_name"]

    entry_latency = get_parameter_entry(world_ranges, "hold_latency_minutes")
    entry_missed = get_parameter_entry(world_ranges, "hold_missed_share")
    entry_unreach_share = get_parameter_entry(world_ranges, "share_of_victims_unable_to_reach_1930_first_try")
    entry_unreach_delay = get_parameter_entry(world_ranges, "unreachable_1930_extra_delay_minutes")

    # 5. Initialize isolated random generator via SeedSequence
    seed_sequence = np.random.SeedSequence([int(fast_settings.seed), HOLD_REPLAY_SALT])
    rng = np.random.default_rng(seed_sequence)

    # Map payment IDs to Pass 1 events to resolve disputed monetary amounts
    pmt_map = {
        pass1_output.result.payment_ids[i]: event
        for i, event in enumerate(pass1_output.result.run_result.events)
    }

    tracker = HoldFlowTracker()
    primary_hold_entries: list[tuple[HoldRequest, bool]] = []
    pass2_liens: list[OutsideEvent] = []

    # 6. Transform complaints into hold requests and compute landing moments
    for complaint in complaints:
        if not complaint.payment_ids:
            continue

        # Calculate total disputed funds across all payments named in the complaint
        payment_amounts = [
            pmt_map[pid].amount_paise for pid in complaint.payment_ids if pid in pmt_map
        ]
        if not payment_amounts:
            continue
        total_amount_paise = sum(payment_amounts)
        primary_payment_id = complaint.payment_ids[0]

        # Draw parameter bounds and stochastic uniforms
        base_latency = draw_in_range(entry_latency, rng, split, "hold_latency_minutes")
        missed_share = draw_in_range(entry_missed, rng, split, "hold_missed_share")
        unreachable_share = draw_in_range(
            entry_unreach_share, rng, split, "share_of_victims_unable_to_reach_1930_first_try"
        )
        unreachable_extra_delay = draw_in_range(entry_unreach_delay, rng, split, "unreachable_1930_extra_delay_minutes")

        u_unreachable = float(rng.uniform(0.0, 1.0))
        u_missed = float(rng.uniform(0.0, 1.0))

        # Query bank institutional delay factor
        receiving_bank = pass1_output.world.banks[complaint.reported_bank_index]
        bank_factor = receiving_bank.hold_delay_factor

        # Convert complaint arrival minute into timezone-aware IST datetime
        arrival_dt = world_minute_to_ist(complaint.arrival_minute, pass1_output.plan.clock.start_date)

        # Compute composite latency in minutes
        latency_minutes = compute_hold_latency(
            base_latency_minutes=base_latency,
            bank_hold_delay_factor=bank_factor,
            arrival_dt=arrival_dt,
            night_hours=hold_cfg["night_hours"],
            night_multiplier=hold_cfg["night_multiplier"],
            weekend_multiplier=hold_cfg["weekend_multiplier"],
            u_unreachable=u_unreachable,
            unreachable_share=unreachable_share,
            unreachable_extra_delay_minutes=unreachable_extra_delay,
        )
        landing_minute = complaint.arrival_minute + latency_minutes

        # Create idempotent hold directive
        hold_req = tracker.create_hold_request(
            complaint_id=complaint.label_id,
            payment_id=primary_payment_id,
            receiving_bank_index=complaint.reported_bank_index,
            account_id=complaint.reported_account,
            amount_paise=total_amount_paise,
            landing_minute=landing_minute,
            is_forwarded=False,
        )
        if hold_req is None:
            # Duplicate complaint identifier: dropped by exact-once idempotency rule
            continue

        # Inaction condition: bank ignores the directive if uniform draw falls under missed_share
        is_missed = u_missed < missed_share
        primary_hold_entries.append((hold_req, is_missed))

        # Only non-missed holds are enacted as physical liens in Pass 2
        if not is_missed:
            lien = OutsideEvent(
                minute=hold_req.landing_minute,
                kind="lien",
                account_id=hold_req.account_id,
                amount_paise=hold_req.amount_paise,
                encumbrance_id=f"lien_{hold_req.complaint_id}",
            )
            pass2_liens.append(lien)

    # 7. Merge outside lien events into Pass 2 inputs
    pass2_outside_events = tuple(list(active_fraud_inputs.outside_events) + pass2_liens)
    fraud_inputs_pass2 = replace(active_fraud_inputs, outside_events=pass2_outside_events)

    # 8. Execute Pass 2 simulation with injected liens
    pass2_output = run_interactive_world(
        fast_settings,
        runner_settings=active_runner_settings,
        fraud_inputs=fraud_inputs_pass2,
        gang_configs=active_gangs,
        configs=configs,
        code_version=code_version,
    )

    # Index all transfer credit legs across ledgers to trace departed funds for forwarding
    credit_lookup: dict[str, tuple[str, int]] = {}
    for b_idx, ledger in enumerate(pass2_output.world.ledgers):
        for entry in ledger.entries:
            if entry.kind == "transfer_credit":
                credit_lookup[entry.transfer_id] = (entry.account_id, b_idx)

    primary_holds: list[HoldRequest] = []
    primary_outcomes: list[HoldOutcome] = []
    encumbered_paise_by_account: dict[str, int] = {}

    forwarded_holds: list[HoldRequest] = []
    forwarded_outcomes: list[HoldOutcome] = []

    # 9. Classify primary holds against Pass 2 state at landing minute
    for hold_req, is_missed in primary_hold_entries:
        primary_holds.append(hold_req)

        if is_missed:
            outcome = classify_hold_outcome(
                available_balance_paise=0,
                requested_amount_paise=hold_req.amount_paise,
                account_id=hold_req.account_id,
                landing_minute=hold_req.landing_minute,
                missed=True,
            )
            primary_outcomes.append(outcome)
            continue

        target_ledger = pass2_output.world.ledgers[hold_req.receiving_bank_index]
        reconstructed_bal = reconstruct_account_balance(
            target_ledger, hold_req.account_id, hold_req.landing_minute
        )

        prior_encumbered = encumbered_paise_by_account.get(hold_req.account_id, 0)
        available_paise = max(0, reconstructed_bal - prior_encumbered)

        outcome = classify_hold_outcome(
            available_balance_paise=available_paise,
            requested_amount_paise=hold_req.amount_paise,
            account_id=hold_req.account_id,
            landing_minute=hold_req.landing_minute,
            missed=False,
        )
        primary_outcomes.append(outcome)

        if outcome.held_amount_paise > 0:
            encumbered_paise_by_account[hold_req.account_id] = prior_encumbered + outcome.held_amount_paise

        # 10. Evaluate naive single-hop forwarding for departed money
        if outcome.status in ("too_late", "partial"):
            # Inspect outgoing transfer debits from this account that posted before landing minute
            outgoing_transfers: list[OutgoingTransfer] = []
            for entry in target_ledger.entries:
                if (
                    entry.account_id == hold_req.account_id
                    and entry.kind == "transfer_debit"
                    and entry.minute <= hold_req.landing_minute
                ):
                    if entry.transfer_id in credit_lookup:
                        dest_account, dest_bank_idx = credit_lookup[entry.transfer_id]
                        outgoing_transfers.append(
                            OutgoingTransfer(
                                transfer_id=entry.transfer_id,
                                from_account=hold_req.account_id,
                                to_account=dest_account,
                                to_bank_index=dest_bank_idx,
                                amount_paise=abs(entry.amount_paise),
                                minute=entry.minute,
                            )
                        )

            fwd_directive = decide_one_hop_forwarding(hold_req, outcome, outgoing_transfers)
            if fwd_directive is not None:
                # Compute second-hop latency targeting the recipient institution
                fwd_dest_bank = pass2_output.world.banks[fwd_directive.receiving_bank_index]
                fwd_arrival_dt = world_minute_to_ist(
                    fwd_directive.landing_minute, pass2_output.plan.clock.start_date
                )
                fwd_base_lat = draw_in_range(entry_latency, rng, split, "hold_latency_minutes")
                fwd_u_unreach = float(rng.uniform(0.0, 1.0))
                fwd_lat = compute_hold_latency(
                    base_latency_minutes=fwd_base_lat,
                    bank_hold_delay_factor=fwd_dest_bank.hold_delay_factor,
                    arrival_dt=fwd_arrival_dt,
                    night_hours=hold_cfg["night_hours"],
                    night_multiplier=hold_cfg["night_multiplier"],
                    weekend_multiplier=hold_cfg["weekend_multiplier"],
                    u_unreachable=fwd_u_unreach,
                    unreachable_share=unreachable_share,
                    unreachable_extra_delay_minutes=unreachable_extra_delay,
                )
                fwd_landing = fwd_directive.landing_minute + fwd_lat
                final_fwd_hold = replace(fwd_directive, landing_minute=fwd_landing)
                forwarded_holds.append(final_fwd_hold)

                # Evaluate forwarded outcome on pass-2 state without placing a lien
                fwd_u_missed = float(rng.uniform(0.0, 1.0))
                is_fwd_missed = fwd_u_missed < missed_share
                if is_fwd_missed:
                    fwd_outcome = classify_hold_outcome(
                        available_balance_paise=0,
                        requested_amount_paise=final_fwd_hold.amount_paise,
                        account_id=final_fwd_hold.account_id,
                        landing_minute=fwd_landing,
                        missed=True,
                    )
                else:
                    fwd_ledger = pass2_output.world.ledgers[final_fwd_hold.receiving_bank_index]
                    fwd_bal = reconstruct_account_balance(
                        fwd_ledger, final_fwd_hold.account_id, fwd_landing
                    )
                    fwd_outcome = classify_hold_outcome(
                        available_balance_paise=fwd_bal,
                        requested_amount_paise=final_fwd_hold.amount_paise,
                        account_id=final_fwd_hold.account_id,
                        landing_minute=fwd_landing,
                        missed=False,
                    )
                forwarded_outcomes.append(fwd_outcome)

    return HoldReplayResult(
        pass1_output=pass1_output,
        pass2_output=pass2_output,
        primary_holds=primary_holds,
        primary_outcomes=primary_outcomes,
        forwarded_holds=forwarded_holds,
        forwarded_outcomes=forwarded_outcomes,
    )


def measure_hold_latencies_by_window(replay_result: HoldReplayResult) -> dict[str, list[int]]:
    """Group measured hold operational latencies by diurnal window and weekday vs weekend.

    What it is for:
        Partitions the empirical hold response times into day vs night and weekday vs weekend
        buckets to support statistical evaluation and showcase chart rendering in Round 3.

    Inputs:
        replay_result: Completed HoldReplayResult containing Pass 1 output and primary holds.
            Example: replay = run_two_pass_replay(settings, ...); measure_hold_latencies_by_window(replay)

    What it returns:
        Dictionary mapping partition names ('day', 'night', 'weekday', 'weekend') to lists of integer
        latencies in minutes.

    How it works:
        1. Extract victim complaints from Pass 1 output using build_label_events.
        2. Index complaints by label_id to resolve arrival minute for each primary hold.
        3. Load hold configuration to obtain night hours window boundaries.
        4. Resolve simulation starting calendar date from Pass 1 plan clock.
        5. For each primary hold, compute measured latency (landing_minute - arrival_minute).
        6. Convert complaint arrival minute into timezone-aware IST datetime.
        7. Evaluate diurnal status using is_night_hour: route to 'night' if true, else 'day'.
        8. Evaluate weekly status using weekday(): route to 'weekend' if >= 5, else 'weekday'.
        9. Return populated bucket dictionary.

    Errors and side effects:
        None. Inspects immutable output records.

    Limits:
        Evaluates primary holds; does not partition unplaced forwarded directives.
    """
    groups: dict[str, list[int]] = {
        "day": [],
        "night": [],
        "weekday": [],
        "weekend": [],
    }

    if not replay_result.primary_holds:
        return groups

    complaints = build_label_events(replay_result.pass1_output)
    complaints_by_id = {c.label_id: c for c in complaints}
    hold_cfg = load_hold_configuration()
    start_date = replay_result.pass1_output.plan.clock.start_date

    for hold in replay_result.primary_holds:
        complaint = complaints_by_id.get(hold.complaint_id)
        if complaint is None:
            continue

        latency_minutes = hold.landing_minute - complaint.arrival_minute
        arrival_dt = world_minute_to_ist(complaint.arrival_minute, start_date)

        if is_night_hour(arrival_dt.hour, hold_cfg["night_hours"]):
            groups["night"].append(latency_minutes)
        else:
            groups["day"].append(latency_minutes)

        if arrival_dt.weekday() >= 5:
            groups["weekend"].append(latency_minutes)
        else:
            groups["weekday"].append(latency_minutes)

    return groups

