# File: benchmarks/tune_dynamic_batching.py
# Purpose: Level 6 - choose the Triton dynamic_batching values from a real concurrency test instead of guessing.
#
# Why this exists: dynamic batching lets Triton hold single-row requests for a very short time and run them together
# as one batch. Its two settings (preferred_batch_size and max_queue_delay_microseconds) only matter when many
# requests arrive at once, so a one-client sweep cannot tune them. This script tries a small grid of settings, each
# under the same concurrent load, and picks a winner with a written rule.
#
# How a run works, for each candidate:
#   1. write the candidate dynamic_batching block into serving/model_repository/fraud_xgboost/config.pbtxt
#   2. restart ONLY the triton container (it reads the config at startup) and wait until the model is READY
#   3. run benchmarks/benchmark_dynamic_batching.py inside the client container (compose network) and read the result
# The original config.pbtxt is ALWAYS put back at the end, so candidates are never left behind in the repo.
#
# The selection rule (applied at one decision concurrency level):
#   - keep only candidates whose p99 <= 1.5 x the p99 with batching OFF
#   - among those, take the highest throughput that beats OFF
#   - near-tie (within 3 percent of the best throughput): prefer the smaller delay, then the smaller preferred size
#
# Example (client container "l6-client" must exist and hold the benchmark files and rows.npy):
#     PYTHONPATH=. python benchmarks/tune_dynamic_batching.py --run off 500:8,32 --raw-json raw.json
#     PYTHONPATH=. python benchmarks/tune_dynamic_batching.py --finalize --raw-json raw.json
import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import tritonclient.grpc as grpc_client

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "serving" / "model_repository" / "fraud_xgboost" / "config.pbtxt"
REPORT_PATH = REPO_ROOT / "reports" / "benchmarks" / "level6_batch.json"
MODEL_NAME = "fraud_xgboost"
HOST_GRPC_URL = "localhost:8001"
CLIENT_CONTAINER = "l6-client"
LEVELS = [10, 50]  # concurrency levels measured
DECISION_LEVEL = 50  # the level the rule is applied at: busy enough that requests really queue
P99_LIMIT = 1.5  # an ON config may be at most this many times slower (p99) than OFF
TIE_BAND = 0.03  # throughputs within 3 percent of the best count as a tie
SECTION_NAME = "dynamic_batching_tuning"


def label_for(delay_us: int, sizes: list[int]) -> str:
    """A short readable name such as 'delay500us_pref8-32'."""
    return f"delay{delay_us}us_pref{'-'.join(str(s) for s in sizes)}"


def parse_candidate(text: str) -> dict:
    """Turn 'off' or '500:8,32' (delay in microseconds : preferred sizes) into a candidate description."""
    if text == "off":
        return {"label": "off", "delay_us": None, "sizes": None}
    delay, sizes = text.split(":")
    sizes_list = [int(s) for s in sizes.split(",")]
    return {"label": label_for(int(delay), sizes_list), "delay_us": int(delay), "sizes": sizes_list}


def block_text(candidate: dict) -> str:
    """The dynamic_batching block to append to the config (nothing for the OFF baseline)."""
    if candidate["delay_us"] is None:
        return ""
    sizes = ", ".join(str(s) for s in candidate["sizes"])
    return (f"\ndynamic_batching {{\n  preferred_batch_size: [ {sizes} ]\n"
            f"  max_queue_delay_microseconds: {candidate['delay_us']}\n}}\n")


def restart_triton_and_wait() -> None:
    """Restart only the triton container and wait (up to 2 minutes) until the model reports READY."""
    subprocess.run(["docker", "compose", "--profile", "serving", "restart", "triton"], cwd=REPO_ROOT, check=True,
                   capture_output=True)
    deadline = time.time() + 120
    while time.time() < deadline:
        try:
            if grpc_client.InferenceServerClient(HOST_GRPC_URL).is_model_ready(MODEL_NAME):
                return
        except Exception:  # noqa: BLE001 - the server is still starting; keep waiting
            pass
        time.sleep(2)
    raise RuntimeError("Triton model did not become READY in time")


def active_batching_settings() -> dict | None:
    """Ask the running Triton what dynamic batching it loaded (None when it is off)."""
    config = grpc_client.InferenceServerClient(HOST_GRPC_URL).get_model_config(MODEL_NAME).config
    if not config.HasField("dynamic_batching"):
        return None
    batching = config.dynamic_batching
    return {"preferred_batch_size": list(batching.preferred_batch_size),
            "max_queue_delay_microseconds": batching.max_queue_delay_microseconds}


def measure(label: str, rows_in_container: str) -> dict:
    """Run the load driver inside the client container and bring its JSON result back."""
    out_in_container = f"/work/out_{label}.json"
    command = ["docker", "exec", CLIENT_CONTAINER, "python", "/work/benchmark_dynamic_batching.py",
               "--rows-npy", rows_in_container, "--levels", *[str(level) for level in LEVELS],
               "--output-json", out_in_container]
    subprocess.run(command, check=True)
    text = subprocess.run(["docker", "exec", CLIENT_CONTAINER, "cat", out_in_container], check=True,
                          capture_output=True, text=True).stdout
    return json.loads(text)["levels"]


def run_candidates(texts: list[str], raw_path: Path, rows_in_container: str) -> None:
    """Measure each candidate, saving results after every one. The original config is restored at the end."""
    raw = json.loads(raw_path.read_text()) if raw_path.exists() else {}
    original_config = CONFIG_PATH.read_text()
    try:
        for text in texts:
            candidate = parse_candidate(text)
            print(f"\n=== {candidate['label']} ===", flush=True)
            CONFIG_PATH.write_text(original_config.rstrip("\n") + "\n" + block_text(candidate))
            restart_triton_and_wait()
            loaded = active_batching_settings()
            print("Triton loaded dynamic_batching:", loaded, flush=True)
            raw[candidate["label"]] = {"candidate": candidate, "triton_reports": loaded,
                                       "levels": measure(candidate["label"], rows_in_container)}
            raw_path.write_text(json.dumps(raw, indent=2))
    finally:
        CONFIG_PATH.write_text(original_config)
        restart_triton_and_wait()
        print("\nOriginal config.pbtxt restored and Triton restarted.", flush=True)


def choose(raw: dict) -> dict:
    """Apply the selection rule at DECISION_LEVEL and explain what happened to every candidate."""
    level = str(DECISION_LEVEL)
    off = raw["off"]["levels"][level]
    p99_limit = P99_LIMIT * off["p99_ms"]
    verdicts, kept = [], []
    for label, entry in raw.items():
        if label == "off":
            continue
        stats = entry["levels"][level]
        reasons = []
        if stats["p99_ms"] > p99_limit:
            reasons.append("p99 above the 1.5x guardrail")
        if stats["throughput_rps"] <= off["throughput_rps"]:
            reasons.append("does not beat OFF on throughput")
        verdicts.append({"label": label, "kept": not reasons, "rejected_because": reasons})
        if not reasons:
            kept.append((label, entry["candidate"], stats))
    if not kept:
        return {"winner": None, "verdicts": verdicts, "p99_limit_ms": p99_limit}
    best_rps = max(stats["throughput_rps"] for _, _, stats in kept)
    near_best = [item for item in kept if item[2]["throughput_rps"] >= best_rps * (1 - TIE_BAND)]
    label, candidate, _ = min(near_best, key=lambda item: (item[1]["delay_us"], max(item[1]["sizes"])))
    return {"winner": label, "winner_candidate": candidate, "verdicts": verdicts, "p99_limit_ms": p99_limit}


def print_table(raw: dict) -> None:
    """One line per candidate and concurrency level."""
    print(f"{'concurrency':>11} | {'config':<22} | {'throughput_rps':>14} | {'p50_ms':>7} | {'p99_ms':>7}")
    for level in [str(level) for level in LEVELS]:
        for label, entry in raw.items():
            stats = entry["levels"][level]
            print(f"{level:>11} | {label:<22} | {stats['throughput_rps']:>14,.0f} | {stats['p50_ms']:>7.2f} | "
                  f"{stats['p99_ms']:>7.2f}")


def finalize(raw_path: Path) -> None:
    """Apply the rule to the saved results, print the table and write the report section."""
    raw = json.loads(raw_path.read_text())
    print_table(raw)
    decision = choose(raw)
    print("\nDecision:", json.dumps({k: v for k, v in decision.items() if k != "verdicts"}))
    for verdict in decision["verdicts"]:
        print(" ", verdict)
    section = {
        "what": "Choosing Triton dynamic_batching values from concurrent single-row gRPC load (fraud_xgboost, ONNX "
                "Runtime on CPU, one model instance)",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "load": {"pattern": "many separate single-row requests in flight at once (closed loop, several client "
                            "processes), from a container on the compose network",
                 "concurrency_levels_measured": LEVELS, "decision_level": DECISION_LEVEL,
                 "repeats_per_level": 3, "numbers": "median of 3 repeats; 20,000 timed requests per repeat",
                 "inputs": "1,000 real test-split rows (50 fraud), seed 42, one row per request"},
        "decision_rule": f"keep configs with p99 <= {P99_LIMIT}x the OFF p99 at concurrency {DECISION_LEVEL}; among "
                         f"those pick the highest throughput that beats OFF; throughputs within {TIE_BAND:.0%} are a "
                         "tie, then prefer the smaller delay, then the smaller preferred size",
        "off_baseline": raw["off"]["levels"],
        "candidates": {label: {"config": entry["candidate"], "levels": entry["levels"]}
                       for label, entry in raw.items() if label != "off"},
        "decision": decision,
    }
    report = json.loads(REPORT_PATH.read_text()) if REPORT_PATH.exists() else {}
    report[SECTION_NAME] = section
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nWrote the '{SECTION_NAME}' section to {REPORT_PATH}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Tune Triton dynamic batching from a concurrency test")
    parser.add_argument("--run", nargs="+", default=[], help="candidates: 'off' or 'DELAY_US:SIZE,SIZE'")
    parser.add_argument("--raw-json", type=Path, required=True, help="where raw results accumulate")
    parser.add_argument("--rows-in-container", default="/work/rows.npy")
    parser.add_argument("--finalize", action="store_true", help="apply the rule and write the report section")
    args = parser.parse_args()
    if args.run:
        run_candidates(args.run, args.raw_json, args.rows_in_container)
    if args.finalize:
        finalize(args.raw_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
