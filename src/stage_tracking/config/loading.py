"""YAML -> config-dataclass loading, with keyword overrides taking precedence
over the file's values (mirrors the old CLI's argparse-over-YAML behavior,
but returns a config object instead of mutating module globals)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from stage_tracking.config.models import OptimizationConfig


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {path}")
    return data


def resolve_path(path_value: str | Path, base_dir: Path) -> Path:
    path = Path(str(path_value)).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _require(config: dict, key: str, config_path: Path):
    if key not in config:
        raise ValueError(
            f"Missing required key '{key}' in config file: {config_path}. "
            f"Either add it to the YAML or pass the equivalent override."
        )
    return config[key]


def load_optimization_config(
    config_path: str | Path,
    base_dir: Path | None = None,
    **overrides: Any,
) -> OptimizationConfig:
    """
    Load an OptimizationConfig from a YAML file.

    `config_path` is resolved relative to `base_dir` (default: current working
    directory) if not already absolute. Every *other* relative path inside the
    YAML (video_path, gt_path, results_dir) is then resolved relative to the
    config file's own directory, so a config + its data can be moved together
    as a self-contained project folder regardless of where the command is run
    from. Any override whose value is not None takes precedence over the YAML.
    """
    config_path = Path(config_path).expanduser()
    if not config_path.is_absolute():
        config_path = (base_dir or Path.cwd()) / config_path
    config_path = config_path.resolve()
    file_dir = config_path.parent

    raw = load_yaml(config_path)

    def get(key: str, required: bool = True, default: Any = None):
        if overrides.get(key) is not None:
            return overrides[key]
        if required:
            return _require(raw, key, config_path)
        return raw.get(key, default)

    video_path = resolve_path(get("video_path"), file_dir)
    gt_path = resolve_path(get("gt_path"), file_dir)
    results_dir = resolve_path(get("results_dir"), file_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground-truth file not found: {gt_path}")

    return OptimizationConfig(
        video_path=video_path,
        gt_path=gt_path,
        models=get("models"),
        occluded=bool(get("occluded", required=False, default=False)),
        tracker_name=get("tracker_name"),
        metric=get("metric"),
        max_bb_eval=get("max_bb_eval"),
        stagnation_rel_threshold=get("stagnation_rel_threshold"),
        stagnation_window=get("stagnation_window"),
        n_best=get("n_best"),
        results_dir=results_dir,
        search_space=get("search_space"),
        x0_raw=get("x0_raw"),
        tracker_override_defaults=get("tracker_override_defaults", required=False, default={}),
    )
