from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
import json
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, ImageFilter, ImageStat

from src.data_utils import INPUT_COLUMNS, image_paths_for_row
from src.submission import PERMUTATION, parse_answer_cell

CaptionCache = dict[tuple[str, int, str], str]


@dataclass(frozen=True)
class FrameRelevanceScores:
    relevance_scores: list[float | None]
    image_relevance_scores: list[float | None]
    caption_embedding_scores: list[float | None]


FrameRelevanceScorer = Callable[
    [Mapping[str, Any], Sequence[Path], Sequence[str | None]],
    FrameRelevanceScores | Sequence[float | None],
]


@dataclass(frozen=True)
class DataFilteringConfig:
    dark_mean_max: float = 8.0
    bright_mean_min: float = 247.0
    blank_std_max: float = 5.0
    flat_std_max: float = 3.0
    flat_entropy_max: float = 0.25
    duplicate_hash_distance: int = 4
    duplicate_mean_delta: float = 20.0
    drop_no_ordering: bool = True
    downweight_blank_frame_count: int = 1
    drop_blank_frame_count: int = 2
    low_relevance_threshold: float = 0.05
    low_relevance_frame_count: int = 1
    clean_weight: float = 1.0
    downweight_weight: float = 0.5

    def __post_init__(self) -> None:
        if self.downweight_blank_frame_count < 1:
            raise ValueError("downweight_blank_frame_count must be at least 1")
        if self.drop_blank_frame_count < self.downweight_blank_frame_count:
            raise ValueError("drop_blank_frame_count must be >= downweight_blank_frame_count")
        if not 0.0 <= self.downweight_weight <= self.clean_weight:
            raise ValueError("downweight_weight must be between 0 and clean_weight")


@dataclass(frozen=True)
class FrameAudit:
    image_index: int
    image: str
    exists: bool
    width: int | None
    height: int | None
    mean: float | None
    std: float | None
    entropy: float | None
    edge_density: float | None
    average_hash: str | None
    blank_kind: str
    error: str

    @property
    def is_blank(self) -> bool:
        return bool(self.blank_kind)


@dataclass(frozen=True)
class SampleAudit:
    row_id: str
    action: str
    sample_weight: float
    manual_review: bool
    reasons: list[str]
    no_ordering: bool
    answer_is_identity: bool
    blank_frames: list[int]
    duplicate_pairs: list[str]
    low_relevance_frames: list[int]
    missing_caption_count: int
    missing_relevance_count: int
    relevance_backend: str
    relevance_scores: list[float | None]
    image_relevance_scores: list[float | None]
    caption_embedding_scores: list[float | None]
    frame_audits: list[FrameAudit]

    def to_record(self) -> dict[str, Any]:
        return {
            "Id": self.row_id,
            "action": self.action,
            "sample_weight": self.sample_weight,
            "manual_review": self.manual_review,
            "reasons": ";".join(self.reasons),
            "No_ordering": self.no_ordering,
            "answer_is_identity": self.answer_is_identity,
            "blank_frame_count": len(self.blank_frames),
            "blank_frames": _json_dumps(self.blank_frames),
            "duplicate_pair_count": len(self.duplicate_pairs),
            "duplicate_pairs": _json_dumps(self.duplicate_pairs),
            "low_relevance_frame_count": len(self.low_relevance_frames),
            "low_relevance_frames": _json_dumps(self.low_relevance_frames),
            "missing_caption_count": self.missing_caption_count,
            "missing_relevance_count": self.missing_relevance_count,
            "relevance_backend": self.relevance_backend,
            "relevance_scores": _json_dumps([_round_optional(value) for value in self.relevance_scores]),
            "image_relevance_scores": _json_dumps(
                [_round_optional(value) for value in self.image_relevance_scores]
            ),
            "caption_embedding_scores": _json_dumps(
                [_round_optional(value) for value in self.caption_embedding_scores]
            ),
            "frame_blank_kinds": _json_dumps([frame.blank_kind for frame in self.frame_audits]),
            "frame_means": _json_dumps([_round_optional(frame.mean) for frame in self.frame_audits]),
            "frame_stds": _json_dumps([_round_optional(frame.std) for frame in self.frame_audits]),
            "frame_entropies": _json_dumps([_round_optional(frame.entropy) for frame in self.frame_audits]),
            "frame_edge_densities": _json_dumps(
                [_round_optional(frame.edge_density) for frame in self.frame_audits]
            ),
            "frame_hashes": _json_dumps([frame.average_hash for frame in self.frame_audits]),
            "frame_errors": _json_dumps([frame.error for frame in self.frame_audits]),
        }


def is_truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def load_caption_cache(path: str | Path) -> CaptionCache:
    cache_path = Path(path)
    if not cache_path.exists():
        return {}
    if cache_path.suffix.lower() == ".csv":
        return _load_caption_cache_csv(cache_path)
    return _load_caption_cache_jsonl(cache_path)


def analyze_frame(image_path: Path, image_index: int, *, config: DataFilteringConfig | None = None) -> FrameAudit:
    cfg = config or DataFilteringConfig()
    image_name = image_path.name
    if not image_path.exists():
        return FrameAudit(
            image_index=image_index,
            image=image_name,
            exists=False,
            width=None,
            height=None,
            mean=None,
            std=None,
            entropy=None,
            edge_density=None,
            average_hash=None,
            blank_kind="",
            error="missing",
        )

    try:
        with Image.open(image_path) as image:
            width, height = image.size
            gray = image.convert("L")
            analysis_image = gray.resize((64, 64))
            stat = ImageStat.Stat(analysis_image)
            mean = float(stat.mean[0])
            std = float(stat.stddev[0])
            entropy = float(analysis_image.entropy())
            edge_density = _edge_density(analysis_image)
            avg_hash = _average_hash(gray)
    except Exception as exc:  # Pillow can raise several image-specific errors here.
        return FrameAudit(
            image_index=image_index,
            image=image_name,
            exists=True,
            width=None,
            height=None,
            mean=None,
            std=None,
            entropy=None,
            edge_density=None,
            average_hash=None,
            blank_kind="",
            error=type(exc).__name__,
        )

    blank_kind = _classify_blank(mean=mean, std=std, entropy=entropy, config=cfg)
    return FrameAudit(
        image_index=image_index,
        image=image_name,
        exists=True,
        width=width,
        height=height,
        mean=mean,
        std=std,
        entropy=entropy,
        edge_density=edge_density,
        average_hash=avg_hash,
        blank_kind=blank_kind,
        error="",
    )


def analyze_sample(
    row: Mapping[str, Any] | pd.Series,
    image_root: Path,
    *,
    config: DataFilteringConfig | None = None,
    caption_cache: CaptionCache | None = None,
    relevance_scorer: FrameRelevanceScorer | None = None,
    relevance_backend: str = "external",
) -> SampleAudit:
    cfg = config or DataFilteringConfig()
    row_values = _row_to_dict(row)
    row_id = str(row_values["Id"])
    image_paths = image_paths_for_row(pd.Series(row_values), image_root)
    frame_audits = [
        analyze_frame(path, image_index=index, config=cfg)
        for index, path in enumerate(image_paths, start=1)
    ]
    no_ordering = is_truthy(row_values.get("No_ordering"))
    answer_is_identity = _answer_is_identity(row_values.get("Answer"))
    blank_frames = [frame.image_index for frame in frame_audits if frame.is_blank]
    duplicate_pairs = _find_duplicate_pairs(frame_audits, cfg)
    (
        effective_relevance_backend,
        relevance_scores,
        image_relevance_scores,
        caption_embedding_scores,
        low_relevance_frames,
        missing_caption_count,
        missing_relevance_count,
    ) = _frame_relevance(
        row_values=row_values,
        image_paths=image_paths,
        caption_cache=caption_cache,
        relevance_scorer=relevance_scorer,
        relevance_backend=relevance_backend,
        config=cfg,
    )
    action, sample_weight, manual_review, reasons = _decide_action(
        frame_audits=frame_audits,
        no_ordering=no_ordering,
        blank_frame_count=len(blank_frames),
        duplicate_pair_count=len(duplicate_pairs),
        low_relevance_frame_count=len(low_relevance_frames),
        missing_caption_count=missing_caption_count,
        missing_relevance_count=missing_relevance_count,
        config=cfg,
    )
    if no_ordering and answer_is_identity:
        reasons.append("identity_answer_for_no_ordering")

    return SampleAudit(
        row_id=row_id,
        action=action,
        sample_weight=sample_weight,
        manual_review=manual_review,
        reasons=reasons,
        no_ordering=no_ordering,
        answer_is_identity=answer_is_identity,
        blank_frames=blank_frames,
        duplicate_pairs=duplicate_pairs,
        low_relevance_frames=low_relevance_frames,
        missing_caption_count=missing_caption_count,
        missing_relevance_count=missing_relevance_count,
        relevance_backend=effective_relevance_backend,
        relevance_scores=relevance_scores,
        image_relevance_scores=image_relevance_scores,
        caption_embedding_scores=caption_embedding_scores,
        frame_audits=frame_audits,
    )


def build_audit_frame(
    train_df: pd.DataFrame,
    image_root: Path,
    *,
    config: DataFilteringConfig | None = None,
    caption_cache: CaptionCache | None = None,
    relevance_scorer: FrameRelevanceScorer | None = None,
    relevance_backend: str = "external",
) -> pd.DataFrame:
    records = [
        analyze_sample(
            row,
            image_root,
            config=config,
            caption_cache=caption_cache,
            relevance_scorer=relevance_scorer,
            relevance_backend=relevance_backend,
        ).to_record()
        for _, row in train_df.iterrows()
    ]
    return pd.DataFrame(records)


def filter_train_frame(
    train_df: pd.DataFrame,
    audit_df: pd.DataFrame,
    *,
    drop_actions: Sequence[str] = ("drop_from_supervised",),
) -> pd.DataFrame:
    if len(train_df) != len(audit_df):
        raise ValueError("train_df and audit_df must have the same number of rows")
    if train_df["Id"].astype(str).tolist() != audit_df["Id"].astype(str).tolist():
        raise ValueError("train_df and audit_df Id order must match")
    keep_mask = ~audit_df["action"].isin(set(drop_actions))
    return train_df.loc[keep_mask].copy()


def _load_caption_cache_jsonl(path: Path) -> CaptionCache:
    cache: CaptionCache = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            key = (str(record["Id"]), int(record["image_index"]), str(record["image"]))
            cache[key] = str(record["caption"])
    return cache


def _load_caption_cache_csv(path: Path) -> CaptionCache:
    cache: CaptionCache = {}
    frame = pd.read_csv(path, encoding="utf-8-sig")
    for _, record in frame.iterrows():
        key = (str(record["Id"]), int(record["image_index"]), str(record["image"]))
        cache[key] = str(record["caption"])
    return cache


def _row_to_dict(row: Mapping[str, Any] | pd.Series) -> dict[str, Any]:
    if isinstance(row, pd.Series):
        return row.to_dict()
    return dict(row)


def _average_hash(image: Image.Image, hash_size: int = 8) -> str:
    resized = image.convert("L").resize((hash_size, hash_size))
    pixels = list(resized.getdata())
    mean = sum(pixels) / len(pixels)
    value = 0
    for pixel in pixels:
        value = (value << 1) | int(pixel > mean)
    return f"{value:0{hash_size * hash_size // 4}x}"


def _edge_density(gray_image: Image.Image) -> float:
    edges = gray_image.filter(ImageFilter.FIND_EDGES)
    return float(ImageStat.Stat(edges).mean[0]) / 255.0


def _classify_blank(*, mean: float, std: float, entropy: float, config: DataFilteringConfig) -> str:
    if std <= config.blank_std_max and mean <= config.dark_mean_max:
        return "dark"
    if std <= config.blank_std_max and mean >= config.bright_mean_min:
        return "bright"
    if std <= config.flat_std_max and entropy <= config.flat_entropy_max:
        return "flat"
    return ""


def _find_duplicate_pairs(frame_audits: Sequence[FrameAudit], config: DataFilteringConfig) -> list[str]:
    pairs: list[str] = []
    for left, right in combinations(frame_audits, 2):
        if left.average_hash is None or right.average_hash is None:
            continue
        if left.mean is None or right.mean is None:
            continue
        distance = _hamming_distance(left.average_hash, right.average_hash)
        if distance <= config.duplicate_hash_distance and abs(left.mean - right.mean) <= config.duplicate_mean_delta:
            pairs.append(f"{left.image_index}-{right.image_index}:{distance}")
    return pairs


def _hamming_distance(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def _frame_relevance(
    *,
    row_values: Mapping[str, Any],
    image_paths: Sequence[Path],
    caption_cache: CaptionCache | None,
    relevance_scorer: FrameRelevanceScorer | None,
    relevance_backend: str,
    config: DataFilteringConfig,
) -> tuple[str, list[float | None], list[float | None], list[float | None], list[int], int, int]:
    if relevance_scorer is not None:
        captions, missing_caption_count = _captions_for_row(row_values, caption_cache)
        scores = _normalize_relevance_scores(relevance_scorer(row_values, image_paths, captions))
        _check_score_length(scores.relevance_scores, "relevance_scores")
        _check_score_length(scores.image_relevance_scores, "image_relevance_scores")
        _check_score_length(scores.caption_embedding_scores, "caption_embedding_scores")
        low_relevance_frames = _low_relevance_frames(scores.relevance_scores, config)
        missing_relevance_count = sum(score is None for score in scores.relevance_scores)
        return (
            relevance_backend,
            scores.relevance_scores,
            scores.image_relevance_scores,
            scores.caption_embedding_scores,
            low_relevance_frames,
            missing_caption_count,
            missing_relevance_count,
        )

    return "none", [], [], [], [], 0, 0

def _captions_for_row(
    row_values: Mapping[str, Any],
    caption_cache: CaptionCache | None,
) -> tuple[list[str | None], int]:
    if caption_cache is None:
        return [None] * len(INPUT_COLUMNS), 0

    captions: list[str | None] = []
    missing_caption_count = 0
    for image_index, column in enumerate(INPUT_COLUMNS, start=1):
        caption = None
        if caption_cache is not None:
            image_name = str(row_values[column])
            caption = caption_cache.get((str(row_values["Id"]), image_index, image_name))
        if caption is None:
            missing_caption_count += 1
        captions.append(caption)
    return captions, missing_caption_count


def _normalize_relevance_scores(result: FrameRelevanceScores | Sequence[float | None]) -> FrameRelevanceScores:
    if isinstance(result, FrameRelevanceScores):
        return result
    scores = list(result)
    return FrameRelevanceScores(
        relevance_scores=scores,
        image_relevance_scores=scores,
        caption_embedding_scores=[],
    )


def _check_score_length(scores: Sequence[float | None], name: str) -> None:
    if scores and len(scores) != len(INPUT_COLUMNS):
        raise ValueError(f"Expected {len(INPUT_COLUMNS)} {name}, got {len(scores)}")


def _low_relevance_frames(scores: Sequence[float | None], config: DataFilteringConfig) -> list[int]:
    return [
        image_index
        for image_index, score in enumerate(scores, start=1)
        if score is not None and score <= config.low_relevance_threshold
    ]


def _decide_action(
    *,
    frame_audits: Sequence[FrameAudit],
    no_ordering: bool,
    blank_frame_count: int,
    duplicate_pair_count: int,
    low_relevance_frame_count: int,
    missing_caption_count: int,
    missing_relevance_count: int,
    config: DataFilteringConfig,
) -> tuple[str, float, bool, list[str]]:
    reasons: list[str] = []
    action = "keep"
    manual_review = False

    if any(frame.error for frame in frame_audits):
        reasons.append("missing_or_unreadable_frame")
        return "drop_from_supervised", 0.0, True, reasons

    if no_ordering:
        reasons.append("no_ordering")
        if config.drop_no_ordering:
            return "drop_from_supervised", 0.0, False, reasons
        action = "downweight"
        manual_review = True

    if blank_frame_count >= config.drop_blank_frame_count:
        reasons.append("multiple_blank_frames")
        return "drop_from_supervised", 0.0, True, reasons
    if blank_frame_count >= config.downweight_blank_frame_count:
        reasons.append("blank_frame")
        action = "downweight"
        manual_review = True

    if duplicate_pair_count:
        reasons.append("duplicate_frame_candidate")
        action = "downweight"
        manual_review = True

    if low_relevance_frame_count >= config.low_relevance_frame_count:
        reasons.append("low_text_frame_relevance")
        action = "downweight"
        manual_review = True

    if missing_caption_count:
        reasons.append("missing_caption")
    if missing_relevance_count and not missing_caption_count:
        reasons.append("missing_relevance_score")

    if action == "downweight":
        return action, config.downweight_weight, manual_review, reasons
    return action, config.clean_weight, manual_review, reasons


def _answer_is_identity(value: object) -> bool:
    try:
        return parse_answer_cell(value) == PERMUTATION
    except (SyntaxError, ValueError):
        return False


def _round_optional(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 4)


def _json_dumps(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
