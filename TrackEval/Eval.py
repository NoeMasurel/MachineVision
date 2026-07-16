"""
compute_mot_score.py
--------------------
Computes HOTA, CLEAR (MOTA/MOTP), and Identity (IDF1) metrics using the
official TrackEval library, from:

  - A ground-truth file in the CVAT MOT format:
        frame, id, bb_left, bb_top, bb_width, bb_height, conf, class_id, visibility
  - A prediction file in standard MOTChallenge format:
        frame, id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z

Creates a temporary folder in the format expected by TrackEval.

Filter occluded object 

Usage:
    python compute_mot_score.py \
        --gt   data/gt/ground_truth.txt \
        --pred data/pred/predictions.txt

    # Unfilter occluded GT objects 
    python compute_mot_score.py \
        --gt   data/gt/ground_truth.txt \
        --pred data/pred/predictions.txt \
        --occluded False

    # Change the IoU matching threshold used by CLEAR/Identity (default 0.5)
    # Note: HOTA does not take a single threshold (Sum over all possible thresholds)
    # official way it is computed.
    python compute_mot_score.py \
        --gt   data/gt/ground_truth.txt \
        --pred data/pred/predictions.txt \
        --max-iou 0.5

    # Pick which metric families to compute
    python compute_mot_score.py \
        --gt data/gt/ground_truth.txt \
        --pred data/pred/predictions.txt \
        --metrics HOTA CLEAR Identity

Install:
    pip install trackeval
"""

import argparse
import os
import sys
import tempfile

import numpy as np
import pandas as pd

try:
    import trackeval
except ImportError:
    sys.exit(
        "[ERROR] TrackEval is not installed.\n"
        "        Run: pip install git+https://github.com/JonathonLuiten/TrackEval.git"
    )


# Load files

def load_gt(filepath: str, occluded: bool = True) -> pd.DataFrame:
    """
    Load a ground-truth file in the CVAT MOT format:
        frame, id, bb_left, bb_top, bb_width, bb_height, conf, class_id, visibility

    Where:
        conf         = "not ignored" flag  (1 = evaluate, 0 = ignore)
        class_id     = object class
        visibility   = 0 (occulded) 1 (visible) Configure in CVAT
    """
    columns = ["frame", "id", "bb_left", "bb_top",
               "bb_width", "bb_height", "conf", "class_id", "visibility"]
    try:
        df = pd.read_csv(
            filepath,
            header=None,
            comment="#",
            names=columns,
            usecols=range(9),        # ignore any extra trailing columns
        )
    except FileNotFoundError:
        sys.exit(f"[ERROR] GT file not found: {filepath}")
    except Exception as exc:
        sys.exit(f"[ERROR] Could not read GT file {filepath}: {exc}")

    df["frame"] = df["frame"].astype(int)
    df["id"] = df["id"].astype(int)

    # Drop rows explicitly marked as ignored (conf == 0)
    ignored = (df["conf"] == 0).sum()
    if ignored > 0:
        print(f"Dropping {ignored} ignored GT rows (conf == 0)")
        df = df[df["conf"] != 0]

    # Drop heavily occluded objects the tracker cannot reasonably detect
    if not occluded:
        before = len(df)
        df = df[df["visibility"] != 0]
        dropped = before - len(df)
        print(f"Dropping {dropped} occluded GT rows")

    return df.reset_index(drop=True)

def load_predictions(filepath: str) -> pd.DataFrame:
    """
    Load a prediction file in standard MOTChallenge format:
        frame, id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z
    """
    columns = ["frame", "id", "bb_left", "bb_top",
               "bb_width", "bb_height", "conf", "x", "y", "z"]

    try:
        df = pd.read_csv(
            filepath,
            header=None,
            comment="#",
            names=columns,
        )
    except FileNotFoundError:
        sys.exit(f"[ERROR] Prediction file not found: {filepath}")
    except Exception as exc:
        sys.exit(f"[ERROR] Could not read prediction file {filepath}: {exc}")

    df["frame"] = df["frame"].astype(int)
    df["id"] = df["id"].astype(int)

    return df.reset_index(drop=True)

# Bridge: write our dataframes out in the folder layout TrackEval expects

def write_trackeval_files(gt: pd.DataFrame,
                           pred: pd.DataFrame,
                           tmp_dir: str,
                           seq_name: str = "seq01",
                           tracker_name: str = "my_tracker") -> int:
    """
    Lay gt/pred out as:
        tmp_dir/gt/<seq_name>/gt/gt.txt
        tmp_dir/trackers/<tracker_name>/data/<seq_name>.txt

    Returns the number of frames in the sequence (needed by TrackEval's
    SEQ_INFO instead of a seqinfo.ini file).
    """
    gt_seq_dir = os.path.join(tmp_dir, "gt", seq_name, "gt")
    os.makedirs(gt_seq_dir, exist_ok=True)
    tracker_dir = os.path.join(tmp_dir, "trackers", tracker_name, "data")
    os.makedirs(tracker_dir, exist_ok=True)

    # GT rows here have already had ignored/occluded rows filtered out by
    # load_gt(), so every remaining row is "not ignored" (conf=1) and we
    # treat this as single-class tracking (class_id=1, "pedestrian" in
    # MOTChallenge's schema -- the label itself doesn't matter to TrackEval
    # since DO_PREPROC is disabled below, it's just required by the format).
    gt_out = gt.copy()
    gt_out["conf"] = 1
    gt_out["class_id"] = 1
    gt_out = gt_out[["frame", "id", "bb_left", "bb_top", "bb_width", "bb_height",
                      "conf", "class_id", "visibility"]]
    gt_out.to_csv(os.path.join(gt_seq_dir, "gt.txt"), header=False, index=False)

    pred_out = pred.copy()
    pred_out["x"] = -1
    pred_out["y"] = -1
    pred_out["z"] = -1
    pred_out = pred_out[["frame", "id", "bb_left", "bb_top", "bb_width", "bb_height",
                          "conf", "x", "y", "z"]]
    pred_out.to_csv(os.path.join(tracker_dir, f"{seq_name}.txt"), header=False, index=False)

    max_gt_frame = int(gt["frame"].max()) if len(gt) else 0
    max_pred_frame = int(pred["frame"].max()) if len(pred) else 0
    return max(max_gt_frame, max_pred_frame)

def run_trackeval(tmp_dir: str,
                   seq_name: str,
                   tracker_name: str,
                   num_frames: int,
                   metric_names: list,
                   iou_threshold: float = 0.5) -> dict:
    """
    Configure and run the TrackEval evaluator against the files written by
    write_trackeval_files(), using the MOTChallenge2DBox dataset class.
    """
    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_config.update({
        "GT_FOLDER": os.path.join(tmp_dir, "gt"),
        "TRACKERS_FOLDER": os.path.join(tmp_dir, "trackers"),
        "SEQ_INFO": {seq_name: num_frames},   # bypasses needing a seqmap/seqinfo.ini
        "SKIP_SPLIT_FOL": True,               # don't expect a <benchmark>-<split> subfolder
        "TRACKERS_TO_EVAL": [tracker_name],
        "CLASSES_TO_EVAL": ["pedestrian"],
        "DO_PREPROC": False,                  # we already did our own ignore/visibility filtering
        "PRINT_CONFIG": False,
    })

    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config.update({
        "USE_PARALLEL": False,
        "PRINT_RESULTS": False,
        "PRINT_CONFIG": False,
        "PRINT_ONLY_COMBINED": True,
        "OUTPUT_SUMMARY": False,
        "OUTPUT_EMPTY_CLASSES": False,
        "OUTPUT_DETAILED": False,
        "PLOT_CURVES": False,
        "TIME_PROGRESS": False,
        "DISPLAY_LESS_PROGRESS": True,
    })

    threshold_cfg = {"THRESHOLD": iou_threshold, "PRINT_CONFIG": False}
    metric_factory = {
        "HOTA": lambda: trackeval.metrics.HOTA(),          # sweeps thresholds itself
        "CLEAR": lambda: trackeval.metrics.CLEAR(threshold_cfg),
        "Identity": lambda: trackeval.metrics.Identity(threshold_cfg),
    }
    metrics_list = [metric_factory[name]() for name in metric_names if name in metric_factory]
    if not metrics_list:
        sys.exit(f"[ERROR] No valid metrics in {metric_names}")

    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]

    results, _messages = evaluator.evaluate(dataset_list, metrics_list)
    return results

def extract_summary(results: dict, tracker_name: str, seq_key: str = "COMBINED_SEQ") -> dict:
    """
    Flatten TrackEval's nested results dict into {"HOTA.HOTA": 0.71, "CLEAR.MOTA": 0.65, ...}.
    Metrics that return per-alpha-threshold arrays (HOTA family) are averaged,
    which is exactly how the official HOTA score itself is defined.
    """
    out = {}
    seq_results = results["MotChallenge2DBox"][tracker_name][seq_key]["pedestrian"]
    for metric_family, values in seq_results.items():
        for key, val in values.items():
            if hasattr(val, "__len__") and not isinstance(val, str):
                try:
                    val = float(np.mean(val))
                except (TypeError, ValueError):
                    continue
            out[f"{metric_family}.{key}"] = val
    return out

def get_id_counts(gt: pd.DataFrame, pred: pd.DataFrame) -> tuple[int, int]:
    """Count distinct object IDs in an already-loaded GT / prediction dataframe pair."""
    return int(gt["id"].nunique()), int(pred["id"].nunique())

# Entry Point

def hota_score(gt_file, occluded, pred_file, max_iou=0.5, return_summary=False,
              metrics=("HOTA", "CLEAR", "Identity")):
    """
    Convenience wrapper for calling this from other code, analogous to the
    original motmetrics-based mot_score(). Returns Hota by default;
    pass return_summary=True for the full dict of every computed metric, 
    including HOTA/DetA/AssA/IDF1/etc.
    """
    gt = load_gt(gt_file, occluded)
    pred = load_predictions(pred_file)

    with tempfile.TemporaryDirectory() as tmp_dir:
        num_frames = write_trackeval_files(gt, pred, tmp_dir)
        results = run_trackeval(tmp_dir, "seq01", "my_tracker", num_frames,
                                 list(metrics), iou_threshold=max_iou)

    summary = extract_summary(results, "my_tracker")
    gt_id_count, pred_id_count = get_id_counts(gt, pred)
    summary["gt_id_count"] = gt_id_count
    summary["pred_id_count"] = pred_id_count
    hota = summary.get("HOTA.HOTA")

    if return_summary:
        return hota, summary
    return hota

def IDF1_score(gt_file, occluded, pred_file, max_iou=0.5, return_summary=False,
              metrics=("HOTA", "CLEAR", "Identity")):
    """
    Convenience wrapper for calling this from other code, analogous to the
    original motmetrics-based mot_score(). Returns Hota by default;
    pass return_summary=True for the full dict of every computed metric, 
    including HOTA/DetA/AssA/IDF1/etc.
    """
    gt = load_gt(gt_file, occluded)
    pred = load_predictions(pred_file)

    with tempfile.TemporaryDirectory() as tmp_dir:
        num_frames = write_trackeval_files(gt, pred, tmp_dir)
        results = run_trackeval(tmp_dir, "seq01", "my_tracker", num_frames,
                                 list(metrics), iou_threshold=max_iou)

    summary = extract_summary(results, "my_tracker")
    gt_id_count, pred_id_count = get_id_counts(gt, pred)
    summary["gt_id_count"] = gt_id_count
    summary["pred_id_count"] = pred_id_count
    IDF1_score = summary.get("Identity.IDF1")

    if return_summary:
        return IDF1_score, summary
    return IDF1_score


# CLI
def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute HOTA / CLEAR (MOTA,MOTP) / Identity (IDF1) metrics "
                    "via TrackEval, from a GT and prediction file."
    )
    parser.add_argument("--gt", required=True,
                        help="Path to ground-truth file "
                             "(CVAT MOT format: frame,id,x,y,w,h,conf,class,visibility)")
    parser.add_argument("--pred", required=True,
                        help="Path to predictions file "
                             "(standard MOT format: frame,id,x,y,w,h,conf,x,y,z)")
    parser.add_argument("--occluded", type=bool, default=False,
                        help="Remove or not the occluded detections in the ground truth")
    parser.add_argument("--max-iou", type=float, default=0.5,
                        help="IoU threshold used by CLEAR/Identity matching (default: 0.5). "
                             "HOTA ignores this and sweeps its own 19 thresholds internally.")
    parser.add_argument("--metrics", nargs="+", default=["HOTA", "CLEAR", "Identity"],
                        choices=["HOTA", "CLEAR", "Identity"],
                        help="Which metric families to compute (default: all three)")
    return parser.parse_args()

def main():
    args = parse_args()

    print(f"\nLoading ground truth : {args.gt}")
    gt = load_gt(args.gt, occluded=args.occluded)
    print(f"  -> {len(gt)} detections | "
          f"{gt['frame'].nunique()} frames | "
          f"{gt['id'].nunique()} unique IDs")

    print(f"\nLoading predictions  : {args.pred}")
    pred = load_predictions(args.pred)
    print(f"  -> {len(pred)} detections | "
          f"{pred['frame'].nunique()} frames | "
          f"{pred['id'].nunique()} unique IDs")

    print("\nWriting TrackEval-format files and running evaluation...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        num_frames = write_trackeval_files(gt, pred, tmp_dir)
        results = run_trackeval(tmp_dir, "seq01", "my_tracker", num_frames,
                                 args.metrics, iou_threshold=args.max_iou)

    summary = extract_summary(results, "my_tracker")

    print("\n-- TrackEval Metrics -----------------------------------")
    if "HOTA" in args.metrics:
        print(f"  *  HOTA  = {summary.get('HOTA.HOTA', float('nan')) * 100:.2f} %")
        print(f"     DetA  = {summary.get('HOTA.DetA', float('nan')) * 100:.2f} %  (detection accuracy)")
        print(f"     AssA  = {summary.get('HOTA.AssA', float('nan')) * 100:.2f} %  (association accuracy)")
    if "CLEAR" in args.metrics:
        print(f"  *  MOTA  = {summary.get('CLEAR.MOTA', float('nan')) * 100:.2f} %")
        print(f"     MOTP  = {1 - summary.get('CLEAR.MOTP', float('nan')):.4f}  (lower raw distance = tighter boxes)")
    if "Identity" in args.metrics:
        print(f"  *  IDF1  = {summary.get('Identity.IDF1', float('nan')) * 100:.2f} %")



if __name__ == "__main__":
    main()