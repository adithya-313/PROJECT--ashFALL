# File: scripts/data_split.py
# Purpose: Splits the enriched PaySim parquet into stratified train, calibration and test sets (60/20/20).
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# Same seed convention as the rest of the repo, so the split is identical on every run.
RANDOM_SEED = 42

REPO_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = REPO_ROOT / "data" / "enriched" / "paysim_enriched.parquet"
OUTPUT_DIR = REPO_ROOT / "data" / "processed"

# Fraction of the data that goes to each split. The calibration set is used later to fit the probability
# calibrator; the test set is only touched for the final evaluation.
TRAIN_FRACTION = 0.60

# How close (in percentage points) each split's fraud rate must be to the overall fraud rate.
FRAUD_RATE_TOLERANCE_PP = 0.01

# Columns that together identify one transaction. The PaySim data has no unique ID column.
KEY_COLUMNS = ["nameOrig", "nameDest", "step", "amount"]

ADVERSARIAL_PATTERNS = ["mule_account", "account_takeover", "merchant_collusion"]


def load_data() -> pd.DataFrame:
    """Read the enriched parquet into memory (the file itself is never modified)."""
    return pd.read_parquet(INPUT_PATH)


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split into 60% train, 20% calibration, 20% test, keeping the fraud ratio the same in each."""
    # First pass: 60% train, 40% held back. stratify=isFraud makes every split contain the same share of fraud,
    # which matters because fraud is only about 0.13% of rows and a plain random split could skew it.
    train_df, temp_df = train_test_split(
        df, train_size=TRAIN_FRACTION, random_state=RANDOM_SEED, stratify=df["isFraud"]
    )

    # Second pass: cut the held-back 40% in half, giving 20% calibration and 20% test.
    calib_df, test_df = train_test_split(
        temp_df, train_size=0.5, random_state=RANDOM_SEED, stratify=temp_df["isFraud"]
    )
    return train_df, calib_df, test_df


def verify_splits(
    original_df: pd.DataFrame, train_df: pd.DataFrame, calib_df: pd.DataFrame, test_df: pd.DataFrame
) -> None:
    """Check the splits are complete, do not overlap, keep the fraud ratio, and contain every attack pattern."""
    splits = {"train": train_df, "calibration": calib_df, "test": test_df}

    # Check a: nothing was lost or duplicated, so the three row counts add up to the original.
    total_rows = sum(len(split) for split in splits.values())
    assert total_rows == len(original_df), f"Row counts {total_rows} do not add up to original {len(original_df)}"

    # Check b: no row appears in more than one split. The original index is unique per row, so overlapping
    # indexes would mean a row leaked between splits (which would make test results look better than they are).
    index_sets = {name: set(split.index) for name, split in splits.items()}
    assert not (index_sets["train"] & index_sets["calibration"]), "Rows overlap between train and calibration"
    assert not (index_sets["train"] & index_sets["test"]), "Rows overlap between train and test"
    assert not (index_sets["calibration"] & index_sets["test"]), "Rows overlap between calibration and test"

    # The same check using the composite transaction key, as a second line of defence. Each key is turned into one
    # 64-bit hash number so comparing millions of rows stays fast and light on memory. We compare keys ACROSS
    # splits only: PaySim can genuinely repeat a key inside one split.
    key_hashes = {
        name: pd.util.hash_pandas_object(split[KEY_COLUMNS], index=False) for name, split in splits.items()
    }
    assert not key_hashes["train"].isin(key_hashes["test"]).any(), "Transaction keys overlap train/test"
    assert not key_hashes["calibration"].isin(key_hashes["test"]).any(), "Transaction keys overlap calib/test"

    # Check c: the fraud percentage in each split is within 0.01 percentage points of the original.
    original_rate = original_df["isFraud"].mean() * 100
    for name, split in splits.items():
        split_rate = split["isFraud"].mean() * 100
        diff = abs(split_rate - original_rate)
        assert diff <= FRAUD_RATE_TOLERANCE_PP, (
            f"{name} fraud rate {split_rate:.4f}% differs from original {original_rate:.4f}% by {diff:.4f} pp"
        )

    # Check d: every adversarial pattern shows up at least once in every split, so each split can be used to
    # evaluate every attack type.
    for name, split in splits.items():
        for pattern in ADVERSARIAL_PATTERNS:
            count = int((split["adversarial_pattern_type"] == pattern).sum())
            assert count >= 1, f"Pattern {pattern} has no rows in the {name} split"


def print_summary(splits: dict[str, pd.DataFrame]) -> None:
    """Check e: print one summary row per split (rows, fraud, adversarial)."""
    print(f"{'split':<12}{'rows':>12}{'fraud':>10}{'fraud %':>10}{'adversarial':>14}{'adv %':>9}")
    for name, split in splits.items():
        rows = len(split)
        fraud = int(split["isFraud"].sum())
        adversarial = int(split["is_adversarial"].sum())
        print(
            f"{name:<12}{rows:>12,}{fraud:>10,}{fraud / rows * 100:>10.4f}"
            f"{adversarial:>14,}{adversarial / rows * 100:>9.4f}"
        )
    print()
    print(f"{'split':<12}" + "".join(f"{pattern:>22}" for pattern in ADVERSARIAL_PATTERNS))
    for name, split in splits.items():
        counts = [int((split["adversarial_pattern_type"] == p).sum()) for p in ADVERSARIAL_PATTERNS]
        print(f"{name:<12}" + "".join(f"{count:>22,}" for count in counts))


def save_splits(train_df: pd.DataFrame, calib_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Write the three splits to data/processed/ as parquet."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(OUTPUT_DIR / "train.parquet")
    calib_df.to_parquet(OUTPUT_DIR / "calibration.parquet")
    test_df.to_parquet(OUTPUT_DIR / "test.parquet")


def main() -> int:
    """Load, split, verify, summarise and save."""
    print("Loading data...")
    df = load_data()
    print(f"Loaded {len(df):,} rows")

    print("Splitting...")
    train_df, calib_df, test_df = split_data(df)

    print("Saving...")
    save_splits(train_df, calib_df, test_df)
    print(f"Saved to {OUTPUT_DIR}")

    print("Verifying...")
    verify_splits(df, train_df, calib_df, test_df)
    print("All verification checks passed.")
    print_summary({"train": train_df, "calibration": calib_df, "test": test_df})
    return 0


if __name__ == "__main__":
    sys.exit(main())
