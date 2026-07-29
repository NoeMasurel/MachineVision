"""Typed configuration objects for the tracking / evaluation / optimization API."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DatasetConfig:
    """A video + its ground-truth annotations, and where predictions live."""

    video_path: Path
    gt_path: Path
    pred_path: Path | None = None


@dataclass
class TrackingConfig:
    """Parameters for a single tracking run (see tracking.pipeline.run_tracking)."""

    video_path: Path
    model: str
    output_path: Path
    confidence: float | None = None
    tracker: str | Path | None = None
    start: float | None = None
    end: float | None = None
    duration: float | None = None
    gt_path: Path | None = None
    display: bool = False
    save_video: Path | None = None


@dataclass
class EvaluationConfig:
    """Parameters for scoring prediction file(s) against ground truth."""

    gt_path: Path
    pred_path: Path | None = None
    pred_files: list[Path] | None = None
    occluded: bool = False
    verbose: bool = True


@dataclass
class OptimizationConfig:
    """Parameters for a NOMAD search over confidence + tracker parameters."""

    video_path: Path
    gt_path: Path
    models: list[str]
    tracker_name: str
    metric: str
    max_bb_eval: int
    stagnation_rel_threshold: float
    stagnation_window: int
    n_best: int
    results_dir: Path
    search_space: list[dict]
    x0_raw: list[float]
    tracker_override_defaults: dict = field(default_factory=dict)
    occluded: bool = False
