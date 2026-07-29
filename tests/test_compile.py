import pandas as pd

from stage_tracking.evaluation import compile as compile_mod


def test_pred_from_gt_resolves_prediction_folder(tmp_path):
    gt_path = tmp_path / "Data" / "GroundTruth" / "Vid1" / "gt.txt"
    gt_path.parent.mkdir(parents=True, exist_ok=True)
    gt_path.write_text("dummy", encoding="utf-8")

    result = compile_mod.pred_from_gt(gt_path)
    assert result == tmp_path / "Data" / "Prediction" / "Vid1"


def test_save_csv_merges_existing_results(tmp_path):
    csv_path = tmp_path / "metrics.csv"

    compile_mod.save_csv([{"model": "a.txt", "HOTA.HOTA": 0.1}], csv_path)
    compile_mod.save_csv([{"model": "b.txt", "HOTA.HOTA": 0.2}], csv_path)

    df = pd.read_csv(csv_path)
    assert set(df["model"]) == {"a.txt", "b.txt"}
    assert df.loc[df["model"] == "b.txt", "HOTA.HOTA"].iloc[0] == 0.2
