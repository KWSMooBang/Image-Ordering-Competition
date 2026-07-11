"""Create reproducible train/validation splits at the original sample-ID level.

Important:
    Split the original rows first, then create pairwise rows independently for
    train and validation. Splitting after pair generation leaks images and
    captions from one original sample across both datasets.

Example:
    python -m src.data.make_split \
        --input data/raw/train.csv \
        --output-dir data/splits \
        --val-size 0.2 \
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd


ValSize = Union[int, float]


def _validate_source_dataframe(df: pd.DataFrame, id_column: str) -> None:
    if id_column not in df.columns:
        raise ValueError(
            f"Missing ID column {id_column!r}. Available columns: {list(df.columns)}"
        )

    if df[id_column].isna().any():
        missing_count = int(df[id_column].isna().sum())
        raise ValueError(f"{id_column!r} contains {missing_count} missing values.")

    duplicated = df[id_column].astype(str).duplicated(keep=False)
    if duplicated.any():
        duplicate_ids = (
            df.loc[duplicated, id_column].astype(str).drop_duplicates().head(10).tolist()
        )
        raise ValueError(
            f"{id_column!r} must be unique before splitting. "
            f"Duplicate examples: {duplicate_ids}"
        )


def _normalise_val_size(n_samples: int, val_size: ValSize) -> tuple[int, float]:
    if n_samples < 2:
        raise ValueError("At least two original samples are required for a split.")

    if isinstance(val_size, float):
        if not 0.0 < val_size < 1.0:
            raise ValueError("Float val_size must be between 0 and 1.")
        n_val = int(round(n_samples * val_size))
        n_val = min(max(n_val, 1), n_samples - 1)
        fraction = n_val / n_samples
        return n_val, fraction

    n_val = int(val_size)
    if not 1 <= n_val < n_samples:
        raise ValueError(
            f"Integer val_size must satisfy 1 <= val_size < {n_samples}; got {n_val}."
        )
    return n_val, n_val / n_samples


def _random_split_indices(
    n_samples: int,
    n_val: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(n_samples)
    val_indices = shuffled[:n_val]
    train_indices = shuffled[n_val:]
    return train_indices, val_indices


def make_split(
    df: pd.DataFrame,
    *,
    val_size: ValSize = 0.2,
    seed: int = 42,
    id_column: str = "Id",
    stratify_column: str | None = "No_ordering",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split original sample IDs into train and validation sets.

    Args:
        df: Original competition dataframe. One row must represent one sample.
        val_size: Validation fraction or exact number of validation samples.
        seed: Random seed.
        id_column: Unique original sample-ID column.
        stratify_column: Optional column used for stratification. If the column
            is absent or stratification is impossible, a deterministic random
            split is used instead.

    Returns:
        Two one-column dataframes: ``train_ids`` and ``val_ids``.
    """
    _validate_source_dataframe(df, id_column)
    n_val, val_fraction = _normalise_val_size(len(df), val_size)

    train_indices: np.ndarray
    val_indices: np.ndarray
    used_stratification = False

    if stratify_column and stratify_column in df.columns:
        labels = df[stratify_column].fillna("__MISSING__").astype(str)
        class_counts = labels.value_counts()

        can_stratify = (
            len(class_counts) > 1
            and int(class_counts.min()) >= 2
            and n_val >= len(class_counts)
            and (len(df) - n_val) >= len(class_counts)
        )

        if can_stratify:
            try:
                from sklearn.model_selection import train_test_split

                all_indices = np.arange(len(df))
                train_indices, val_indices = train_test_split(
                    all_indices,
                    test_size=n_val,
                    random_state=seed,
                    shuffle=True,
                    stratify=labels,
                )
                train_indices = np.asarray(train_indices)
                val_indices = np.asarray(val_indices)
                used_stratification = True
            except ImportError:
                warnings.warn(
                    "scikit-learn is unavailable; using a deterministic random split.",
                    stacklevel=2,
                )
            except ValueError as exc:
                warnings.warn(
                    f"Could not stratify by {stratify_column!r}: {exc}. "
                    "Using a deterministic random split.",
                    stacklevel=2,
                )

    if not used_stratification:
        train_indices, val_indices = _random_split_indices(
            len(df), n_val=n_val, seed=seed
        )

    train_ids = (
        df.iloc[train_indices][[id_column]]
        .copy()
        .rename(columns={id_column: "Id"})
        .reset_index(drop=True)
    )
    val_ids = (
        df.iloc[val_indices][[id_column]]
        .copy()
        .rename(columns={id_column: "Id"})
        .reset_index(drop=True)
    )

    train_set = set(train_ids["Id"].astype(str))
    val_set = set(val_ids["Id"].astype(str))
    overlap = train_set & val_set
    if overlap:
        raise RuntimeError(f"Split leakage detected. Overlapping IDs: {sorted(overlap)[:10]}")

    if len(train_ids) + len(val_ids) != len(df):
        raise RuntimeError("Split size mismatch.")

    train_ids.attrs["split_metadata"] = {
        "seed": seed,
        "val_size_requested": val_size,
        "val_fraction_actual": val_fraction,
        "stratify_column": stratify_column,
        "used_stratification": used_stratification,
    }
    return train_ids, val_ids


def save_split(
    train_ids: pd.DataFrame,
    val_ids: pd.DataFrame,
    output_dir: str | Path,
    metadata: dict | None = None,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_ids.to_csv(output_path / "train_ids.csv", index=False)
    val_ids.to_csv(output_path / "val_ids.csv", index=False)

    split_metadata = {
        "train_samples": len(train_ids),
        "val_samples": len(val_ids),
    }
    if metadata:
        split_metadata.update(metadata)

    with (output_path / "split_manifest.json").open("w", encoding="utf-8") as file:
        json.dump(split_metadata, file, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split the original image-ordering dataset by sample ID."
    )
    parser.add_argument("--input", required=True, help="Path to train.csv")
    parser.add_argument(
        "--output-dir",
        default="data/splits",
        help="Directory for train_ids.csv and val_ids.csv",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.2,
        help="Validation ratio. Example: 0.2",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--id-column", default="Id")
    parser.add_argument(
        "--stratify-column",
        default="No_ordering",
        help="Column used for stratification when possible.",
    )
    parser.add_argument(
        "--no-stratify",
        action="store_true",
        help="Disable stratification.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path)
    stratify_column = None if args.no_stratify else args.stratify_column

    train_ids, val_ids = make_split(
        df,
        val_size=args.val_size,
        seed=args.seed,
        id_column=args.id_column,
        stratify_column=stratify_column,
    )

    metadata = train_ids.attrs.get("split_metadata", {})
    save_split(train_ids, val_ids, args.output_dir, metadata)

    print(f"[split] source samples : {len(df):,}")
    print(f"[split] train samples  : {len(train_ids):,}")
    print(f"[split] val samples    : {len(val_ids):,}")
    print(f"[split] output dir     : {Path(args.output_dir).resolve()}")
    print(f"[split] stratified     : {metadata.get('used_stratification', False)}")


if __name__ == "__main__":
    main()