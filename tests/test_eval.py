import pandas as pd

from stage_tracking.evaluation import metrics as eval_mod


def test_load_gt_filters_ignored_and_occluded_rows(tmp_path):
    gt_path = tmp_path / "gt.txt"
    gt_path.write_text(
        "1,1,0,0,10,10,1,1,1\n"
        "1,2,0,0,10,10,0,1,1\n"
        "1,3,0,0,10,10,1,1,0\n",
        encoding="utf-8",
    )

    df = eval_mod.load_gt(str(gt_path), occluded=False)
    assert len(df) == 1
    assert df.iloc[0]["id"] == 1


def test_load_predictions_reads_expected_columns(tmp_path):
    pred_path = tmp_path / "pred.txt"
    pred_path.write_text("1,1,0,0,10,10,0.9,0,0,0\n", encoding="utf-8")

    df = eval_mod.load_predictions(str(pred_path))
    assert df.iloc[0]["frame"] == 1
    assert df.iloc[0]["id"] == 1
    assert df.iloc[0]["conf"] == 0.9


def test_get_id_counts_uses_distinct_ids():
    gt = pd.DataFrame({"id": [1, 1, 2]})
    pred = pd.DataFrame({"id": [1, 3]})
    assert eval_mod.get_id_counts(gt, pred) == (2, 2)