import yaml

from stage_tracking.tracking import tracker_config as tracker


def test_build_tracker_config_writes_override_yaml(tmp_path):
    base_path = tmp_path / "base.yaml"
    base_path.write_text("tracker_type: bytetrack\ntrack_high_thresh: 0.5\n", encoding="utf-8")

    output_path = tmp_path / "custom.yaml"
    result = tracker.build_tracker_config(
        "bytetrack",
        output_path=output_path,
        base_config_path=base_path,
        track_high_thresh=0.8,
    )

    assert result == output_path
    assert output_path.exists()

    config = yaml.safe_load(output_path.read_text(encoding="utf-8"))
    assert config["tracker_type"] == "bytetrack"
    assert config["track_high_thresh"] == 0.8


def test_list_valid_params_uses_fetcher(monkeypatch):
    monkeypatch.setattr(tracker, "_fetch_base_config", lambda name: {"a": 1, "b": 2})
    assert tracker.list_valid_params("bytetrack") == ["a", "b"]