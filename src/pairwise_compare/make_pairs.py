"""Build caption-aware pairwise comparison rows."""

from __future__ import annotations

import argparse
import ast
from itertools import combinations
from pathlib import Path, PurePosixPath
from typing import Iterable, Sequence

import pandas as pd

from .captions import CaptionCache, captions_for_image_files, load_caption_cache


IMAGE_COLUMNS = ("Input_1", "Input_2", "Input_3", "Input_4")
REQUIRED_COLUMNS = ("Id", "Sentence", "Answer", *IMAGE_COLUMNS)
REQUIRED_TEST_COLUMNS = ("Id", "Sentence", *IMAGE_COLUMNS)


def parse_permutation(
    value: str | Sequence[int],
    *,
    expected_size: int = 4,
    field_name: str = "permutation",
) -> list[int]:
    parsed = value
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"Invalid {field_name}: {value!r}") from exc

    if not isinstance(parsed, (list, tuple)):
        raise ValueError(f"{field_name} must be a list or tuple; got {type(parsed).__name__}.")

    try:
        permutation = [int(item) for item in parsed]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must contain integers: {parsed!r}") from exc

    expected = list(range(1, expected_size + 1))
    if len(permutation) != expected_size or sorted(permutation) != expected:
        raise ValueError(f"{field_name} must be a permutation of {expected}; got {permutation}.")
    return permutation


def answer_to_order(answer: str | Sequence[int]) -> list[int]:
    positions = parse_permutation(answer, field_name="Answer")
    return sorted(range(1, len(positions) + 1), key=lambda image_index: positions[image_index - 1])


def order_to_answer(order: str | Sequence[int]) -> list[int]:
    chronological_order = parse_permutation(order, field_name="order")
    answer = [0] * len(chronological_order)
    for position, image_index in enumerate(chronological_order, start=1):
        answer[image_index - 1] = position
    return answer


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    if isinstance(value, (int, float)):
        return bool(value)

    normalised = str(value).strip().lower()
    if normalised in {"true", "1", "yes", "y", "t"}:
        return True
    if normalised in {"false", "0", "no", "n", "f", ""}:
        return False
    raise ValueError(f"Cannot interpret boolean value: {value!r}")


def validate_source_dataframe(df: pd.DataFrame, *, require_answer: bool = True) -> None:
    required = REQUIRED_COLUMNS if require_answer else REQUIRED_TEST_COLUMNS
    missing_columns = [column for column in required if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}. Available columns: {list(df.columns)}")

    if df["Id"].isna().any():
        raise ValueError("Id contains missing values.")

    duplicated = df["Id"].astype(str).duplicated(keep=False)
    if duplicated.any():
        examples = df.loc[duplicated, "Id"].astype(str).drop_duplicates().head(10).tolist()
        raise ValueError(f"Id must be unique. Duplicate examples: {examples}")


def _iter_pair_directions(pair_mode: str) -> Iterable[tuple[int, int]]:
    canonical_pairs = list(combinations(range(1, 5), 2))
    if pair_mode == "canonical":
        yield from canonical_pairs
        return
    if pair_mode == "bidirectional":
        for image_a, image_b in canonical_pairs:
            yield image_a, image_b
            yield image_b, image_a
        return
    raise ValueError("pair_mode must be 'canonical' or 'bidirectional'.")


def _row_image_files(row: pd.Series) -> dict[int, str]:
    return {index: str(row[column]) for index, column in enumerate(IMAGE_COLUMNS, start=1)}


def _check_pair_images(
    root: Path | None,
    sample_id: str,
    image_a_relative: PurePosixPath,
    image_b_relative: PurePosixPath,
) -> None:
    if root is None:
        raise ValueError("image_root is required when check_images=True.")

    missing_paths = [
        path
        for path in (
            root / Path(image_a_relative),
            root / Path(image_b_relative),
        )
        if not path.is_file()
    ]
    if missing_paths:
        raise FileNotFoundError(f"Missing image file(s) for sample {sample_id}: {missing_paths}")


def _build_pair_records_for_row(
    *,
    row: pd.Series,
    pair_mode: str,
    caption_cache: CaptionCache,
    caption_missing_policy: str,
    image_root: Path | None,
    check_images: bool,
    answer: list[int] | None,
) -> list[dict]:
    sample_id = str(row["Id"])
    no_ordering = _as_bool(row["No_ordering"]) if "No_ordering" in row.index else False
    image_files = _row_image_files(row)
    captions = captions_for_image_files(
        row_id=sample_id,
        image_files=image_files,
        caption_cache=caption_cache,
        missing_policy=caption_missing_policy,
    )

    records: list[dict] = []
    for image_a_index, image_b_index in _iter_pair_directions(pair_mode):
        image_a_file = image_files[image_a_index]
        image_b_file = image_files[image_b_index]
        image_a_relative = PurePosixPath(sample_id) / image_a_file
        image_b_relative = PurePosixPath(sample_id) / image_b_file

        if check_images:
            _check_pair_images(image_root, sample_id, image_a_relative, image_b_relative)

        label = int(answer[image_a_index - 1] < answer[image_b_index - 1]) if answer is not None else 0
        records.append(
            {
                "sample_id": sample_id,
                "pair_id": f"{sample_id}__{image_a_index}_{image_b_index}",
                "sentence": str(row["Sentence"]),
                "image_a_index": image_a_index,
                "image_b_index": image_b_index,
                "image_a_file": image_a_file,
                "image_b_file": image_b_file,
                "image_a_path": image_a_relative.as_posix(),
                "image_b_path": image_b_relative.as_posix(),
                "image_a_caption": captions[image_a_index],
                "image_b_caption": captions[image_b_index],
                "label": label,
                "no_ordering": no_ordering,
            }
        )
    return records


def build_pairwise_dataframe(
    df: pd.DataFrame,
    *,
    pair_mode: str = "canonical",
    exclude_no_ordering: bool = False,
    image_root: str | Path | None = None,
    check_images: bool = False,
    caption_cache_path: str | Path | None = None,
    caption_cache: CaptionCache | None = None,
    caption_missing_policy: str = "empty",
) -> pd.DataFrame:
    validate_source_dataframe(df, require_answer=True)
    root = Path(image_root) if image_root is not None else None
    cache = caption_cache if caption_cache is not None else load_caption_cache(caption_cache_path)

    records: list[dict] = []
    for row_number, (_, row) in enumerate(df.iterrows(), start=1):
        answer = parse_permutation(row["Answer"], field_name=f"Answer at row {row_number}")
        no_ordering = _as_bool(row["No_ordering"]) if "No_ordering" in df.columns else False
        if exclude_no_ordering and no_ordering:
            continue
        records.extend(
            _build_pair_records_for_row(
                row=row,
                pair_mode=pair_mode,
                caption_cache=cache,
                caption_missing_policy=caption_missing_policy,
                image_root=root,
                check_images=check_images,
                answer=answer,
            )
        )

    pair_df = pd.DataFrame.from_records(records)
    _validate_pair_count(pair_df, pair_mode)
    return pair_df


def build_test_pairwise_dataframe(
    df: pd.DataFrame,
    *,
    pair_mode: str = "canonical",
    image_root: str | Path | None = None,
    check_images: bool = False,
    caption_cache_path: str | Path | None = None,
    caption_cache: CaptionCache | None = None,
    caption_missing_policy: str = "empty",
) -> pd.DataFrame:
    validate_source_dataframe(df, require_answer=False)
    root = Path(image_root) if image_root is not None else None
    cache = caption_cache if caption_cache is not None else load_caption_cache(caption_cache_path)

    records: list[dict] = []
    for _, row in df.iterrows():
        records.extend(
            _build_pair_records_for_row(
                row=row,
                pair_mode=pair_mode,
                caption_cache=cache,
                caption_missing_policy=caption_missing_policy,
                image_root=root,
                check_images=check_images,
                answer=None,
            )
        )

    pair_df = pd.DataFrame.from_records(records)
    _validate_pair_count(pair_df, pair_mode)
    return pair_df


def _validate_pair_count(pair_df: pd.DataFrame, pair_mode: str) -> None:
    expected_pairs = 12 if pair_mode == "bidirectional" else 6
    if pair_df.empty:
        return
    counts = pair_df.groupby("sample_id").size()
    invalid_counts = counts[counts != expected_pairs]
    if not invalid_counts.empty:
        raise RuntimeError(f"Unexpected pair count per sample: {invalid_counts.head(10).to_dict()}")
    if not set(pair_df["label"].unique()).issubset({0, 1}):
        raise RuntimeError("Pair labels must be binary.")


def _filter_by_ids(source_df: pd.DataFrame, ids_path: str | Path) -> pd.DataFrame:
    ids_df = pd.read_csv(ids_path)
    if "Id" not in ids_df.columns:
        raise ValueError(f"{ids_path} must contain an 'Id' column.")

    source_ids = source_df["Id"].astype(str)
    wanted_ids = ids_df["Id"].astype(str)
    missing_ids = sorted(set(wanted_ids) - set(source_ids))
    if missing_ids:
        raise ValueError(f"{len(missing_ids)} split IDs do not exist in the source CSV. Examples: {missing_ids[:10]}")

    order_map = {sample_id: order for order, sample_id in enumerate(wanted_ids)}
    filtered = source_df[source_ids.isin(order_map)].copy()
    filtered["_split_order"] = filtered["Id"].astype(str).map(order_map)
    filtered = filtered.sort_values("_split_order").drop(columns="_split_order")
    return filtered.reset_index(drop=True)


def save_pairwise_splits(
    source_df: pd.DataFrame,
    *,
    output_dir: str | Path,
    split_dir: str | Path | None = None,
    pair_mode: str = "canonical",
    exclude_no_ordering: bool = False,
    image_root: str | Path | None = None,
    check_images: bool = False,
    caption_cache_path: str | Path | None = None,
    caption_missing_policy: str = "empty",
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    caption_cache = load_caption_cache(caption_cache_path)
    outputs: dict[str, Path] = {}

    if split_dir is None:
        pair_df = build_pairwise_dataframe(
            source_df,
            pair_mode=pair_mode,
            exclude_no_ordering=exclude_no_ordering,
            image_root=image_root,
            check_images=check_images,
            caption_cache=caption_cache,
            caption_missing_policy=caption_missing_policy,
        )
        destination = output_path / "all_pairs.csv"
        pair_df.to_csv(destination, index=False)
        outputs["all"] = destination
        return outputs

    split_path = Path(split_dir)
    split_files = {"train": split_path / "train_ids.csv", "val": split_path / "val_ids.csv"}
    for split_name, ids_path in split_files.items():
        if not ids_path.exists():
            raise FileNotFoundError(f"Split file not found: {ids_path}")
        split_df = _filter_by_ids(source_df, ids_path)
        pair_df = build_pairwise_dataframe(
            split_df,
            pair_mode=pair_mode,
            exclude_no_ordering=exclude_no_ordering,
            image_root=image_root,
            check_images=check_images,
            caption_cache=caption_cache,
            caption_missing_policy=caption_missing_policy,
        )
        destination = output_path / f"{split_name}_pairs.csv"
        pair_df.to_csv(destination, index=False)
        outputs[split_name] = destination
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert image-ordering samples into caption-aware pairwise rows.")
    parser.add_argument("--input", required=True, help="Path to train.csv")
    parser.add_argument("--output-dir", default="outputs/pairwise_compare/pairs")
    parser.add_argument("--split-dir", default=None, help="Directory containing train_ids.csv and val_ids.csv.")
    parser.add_argument("--pair-mode", choices=("canonical", "bidirectional"), default="canonical")
    parser.add_argument("--exclude-no-ordering", action="store_true")
    parser.add_argument("--image-root", default=None, help="Image root such as data/train.")
    parser.add_argument("--check-images", action="store_true")
    parser.add_argument("--caption-cache", default=None)
    parser.add_argument("--caption-missing-policy", choices=("empty", "fail"), default="empty")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    source_df = pd.read_csv(input_path)
    outputs = save_pairwise_splits(
        source_df,
        output_dir=args.output_dir,
        split_dir=args.split_dir,
        pair_mode=args.pair_mode,
        exclude_no_ordering=args.exclude_no_ordering,
        image_root=args.image_root,
        check_images=args.check_images,
        caption_cache_path=args.caption_cache,
        caption_missing_policy=args.caption_missing_policy,
    )

    for split_name, output_path in outputs.items():
        pair_df = pd.read_csv(output_path)
        sample_count = pair_df["sample_id"].nunique() if not pair_df.empty else 0
        caption_rate = (
            ((pair_df["image_a_caption"].astype(str).str.len() > 0) & (pair_df["image_b_caption"].astype(str).str.len() > 0)).mean()
            if len(pair_df)
            else float("nan")
        )
        print(
            f"[pairs] {split_name:<5} samples={sample_count:,} pairs={len(pair_df):,} "
            f"positive_rate={pair_df['label'].mean() if len(pair_df) else float('nan'):.4f} "
            f"caption_pair_rate={caption_rate:.4f}"
        )
        print(f"        saved={output_path.resolve()}")


if __name__ == "__main__":
    main()
