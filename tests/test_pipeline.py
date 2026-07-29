from pathlib import Path

from tracking.tracking import pipeline as pipeline_mod


def test_parse_tracker_spec_handles_none_and_path():
    assert pipeline_mod.parse_tracker_spec("none") is None
    assert pipeline_mod.parse_tracker_spec('{"name": "botsort", "with_reid": true}') == {
        "name": "botsort",
        "with_reid": True,
    }

    path = Path("tests/sample.yaml")
    path.write_text("tracker_type: test\n", encoding="utf-8")
    try:
        parsed = pipeline_mod.parse_tracker_spec(str(path))
        assert parsed == path
    finally:
        path.unlink(missing_ok=True)


def test_align_sweep_lists_broadcasts_singleton_values():
    result = pipeline_mod.align_sweep_lists([0.5], ["m"], [None, None])
    assert result == [("m", 0.5, None), ("m", 0.5, None)]


def test_build_pred_filename_includes_parameters():
    filename = pipeline_mod.build_pred_filename("models/yolo26m.pt", 0.25, Path("tracktrack.yaml"))
    assert filename == "yolo26m_conf0.25_tracktrack.txt"
