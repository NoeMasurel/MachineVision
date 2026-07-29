from pathlib import Path

import pytest

from stage_tracking.config.loading import load_optimization_config
from stage_tracking.config.models import OptimizationConfig
from stage_tracking.optimization import nomad_runner as opti


def _write_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "opti.yaml"
    config_path.write_text(
        "video_path: sample.mp4\n"
        "gt_path: Data/GroundTruth/Vid1/gt.txt\n"
        "models:\n  - m\n"
        "tracker_name: tracktrack\n"
        "metric: hota\n"
        "max_bb_eval: 10\n"
        "stagnation_rel_threshold: 0.0001\n"
        "stagnation_window: 5\n"
        "n_best: 5\n"
        "results_dir: Results\n"
        "search_space:\n  - name: confidence\n    target: confidence\n    lower: 0.1\n    upper: 0.9\n    granularity: 0.05\n"
        "x0_raw:\n  - 0.5\n"
        "tracker_override_defaults: {}\n",
        encoding="utf-8",
    )

    (tmp_path / "sample.mp4").write_text("video", encoding="utf-8")

    gt_file = tmp_path / "Data" / "GroundTruth" / "Vid1" / "gt.txt"
    gt_file.parent.mkdir(parents=True, exist_ok=True)
    gt_file.write_text("dummy", encoding="utf-8")

    return config_path


def test_load_optimization_config_reads_yaml(tmp_path):
    config_path = _write_config(tmp_path)

    config = load_optimization_config(config_path)

    assert config.video_path == (tmp_path / "sample.mp4").resolve()
    assert config.metric == "hota"
    assert config.search_space[0]["name"] == "confidence"
    assert config.x0_raw == [0.5]
    assert config.results_dir == (tmp_path / "Results").resolve()


def test_load_optimization_config_overrides_take_precedence(tmp_path):
    config_path = _write_config(tmp_path)

    config = load_optimization_config(config_path, metric="idf1", n_best=3)

    assert config.metric == "idf1"
    assert config.n_best == 3


def _base_config(**overrides) -> OptimizationConfig:
    defaults = dict(
        video_path=Path("sample.mp4"),
        gt_path=Path("Data/GroundTruth/Vid1/gt.txt"),
        models=["m"],
        tracker_name="tracktrack",
        metric="idf1",
        max_bb_eval=10,
        stagnation_rel_threshold=0.0001,
        stagnation_window=5,
        n_best=5,
        results_dir=Path("Results"),
        search_space=[{"name": "track_buffer", "target": "tracker", "lower": 1, "upper": 10, "granularity": 1}],
        x0_raw=[5],
        tracker_override_defaults={},
    )
    defaults.update(overrides)
    return OptimizationConfig(**defaults)  # type: ignore[arg-type]


def test_validate_config_rejects_missing_confidence_target(monkeypatch):
    monkeypatch.setattr(opti, "list_valid_params", lambda _name: ["track_buffer"])
    config = _base_config()

    with pytest.raises(ValueError, match="confidence"):
        opti.validate_config(config)


def test_point_to_args_respects_integer_dimensions():
    search_space = [
        {"name": "confidence", "target": "confidence", "lower": 0.1, "upper": 0.9, "granularity": 0.05},
        {"name": "track_buffer", "target": "tracker", "lower": 10, "upper": 100, "granularity": 1, "is_int": True},
    ]

    confidence, overrides = opti.point_to_args(search_space, [0.5, 42.3])
    assert confidence == 0.5
    assert overrides["track_buffer"] == 42


def test_optimization_state_stagnates_after_window(tmp_path):
    config = _base_config(
        gt_path=tmp_path / "Data" / "GroundTruth" / "Vid1" / "gt.txt",
        results_dir=tmp_path / "Results",
    )
    state = opti.OptimizationState(config)
    state.recent_scores.extend([0.5, 0.5, 0.5, 0.5, 0.5])
    assert state._is_stagnated() is True
