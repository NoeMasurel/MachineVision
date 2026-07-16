"""
compute_mot_score.py
--------------------
Computes MOTA and other MOTChallenge metrics from:
  - A ground-truth file in the CVAT MOT format:
        frame, id, bb_left, bb_top, bb_width, bb_height, conf, class_id, visibility
  - A prediction file in standard MOTChallenge format:
        frame, id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z

Usage:
    python compute_mot_score.py \
        --gt   data/gt/ground_truth.txt \
        --pred data/pred/predictions.txt

    # Filter occluded GT objects (recommended)
    python compute_mot_score.py \
        --gt   data/gt/ground_truth.txt \
        --pred data/pred/predictions.txt \
        --visibility 0.25

    # Change the IoU matching threshold (default 0.5)
    python compute_mot_score.py \
        --gt   data/gt/ground_truth.txt \
        --pred data/pred/predictions.txt \
        --visibility 0.25 \
        --max-iou 0.5
"""

import argparse
import sys

import numpy as np
import pandas as pd
import motmetrics as mm


def load_gt(filepath: str, visibility_threshold: float = 0.0) -> pd.DataFrame:
    """
    Load a ground-truth file in the CVAT MOT format:
        frame, id, bb_left, bb_top, bb_width, bb_height, conf, class_id, visibility

    Where:
        conf         = "not ignored" flag  (1 = evaluate, 0 = ignore)
        class_id     = object class
        visibility   = 0.0 (fully occluded) → 1.0 (fully visible)
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
    df["id"]    = df["id"].astype(int)

    # Drop rows explicitly marked as ignored (conf == 0)
    ignored = (df["conf"] == 0).sum()
    if ignored > 0:
        print(f"  → Dropping {ignored} ignored GT rows (conf == 0)")
        df = df[df["conf"] != 0]

    # Drop heavily occluded objects the tracker cannot reasonably detect
    if visibility_threshold > 0.0:
        before = len(df)
        df = df[df["visibility"] >= visibility_threshold]
        dropped = before - len(df)
        print(f"  → Dropping {dropped} occluded GT rows "
              f"(visibility < {visibility_threshold})")

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
    df["id"]    = df["id"].astype(int)

    return df.reset_index(drop=True)



def build_accumulator(gt: pd.DataFrame,
                      pred: pd.DataFrame,
                      max_iou: float = 0.5) -> mm.MOTAccumulator:
    """
    Walk through every frame in gt or pred and feed IoU distances
    into a MOTAccumulator.
    """
    acc = mm.MOTAccumulator(auto_id=True)

    all_frames = sorted(set(gt["frame"]) | set(pred["frame"]))

    for frame_id in all_frames:
        gt_frame   = gt[gt["frame"]     == frame_id]
        pred_frame = pred[pred["frame"] == frame_id]

        gt_ids   = gt_frame["id"].tolist()
        pred_ids = pred_frame["id"].tolist()

        gt_boxes   = gt_frame[["bb_left", "bb_top", "bb_width", "bb_height"]].values
        pred_boxes = pred_frame[["bb_left", "bb_top", "bb_width", "bb_height"]].values

        if len(gt_ids) > 0 and len(pred_ids) > 0:
            dist_matrix = mm.distances.iou_matrix(gt_boxes, pred_boxes,
                                                   max_iou=max_iou)
        else:
            dist_matrix = np.empty((len(gt_ids), len(pred_ids)))
            dist_matrix[:] = np.nan

        acc.update(gt_ids, pred_ids, dist_matrix)

    return acc

def mot_score(gt_file, visibility, pred_file, max_iou=0.5, return_summary = False):
    # Load
    gt = load_gt(gt_file, visibility)
    pred = load_predictions(pred_file)

    # Accumulate
    acc = build_accumulator(gt, pred, max_iou)

    # Metrics
    mh      = mm.metrics.create()
    summary = mh.compute(
        acc,
        metrics=mm.metrics.motchallenge_metrics,
        name="result",
    )

    mota = summary.loc["result", "mota"] # type: ignore

    if return_summary:
        return mota, summary
    
    return mota


def main():
    parser = argparse.ArgumentParser(
        description="Compute MOTA / MOTChallenge metrics from a GT and prediction file."
    )
    parser.add_argument("--gt",         required=True,
                        help="Path to ground-truth file "
                             "(CVAT MOT format: frame,id,x,y,w,h,conf,class,visibility)")
    parser.add_argument("--pred",       required=True,
                        help="Path to predictions file "
                             "(standard MOT format: frame,id,x,y,w,h,conf,x,y,z)")
    parser.add_argument("--visibility", type=float, default=0.0,
                        help="Minimum GT visibility to include (0.0 = keep all, "
                             "0.25 = recommended, 1.0 = fully visible only)")
    parser.add_argument("--max-iou",   type=float, default=0.5,
                        help="IoU threshold for matching (default: 0.5)")
    args = parser.parse_args()

    # Load
    print(f"\nLoading ground truth : {args.gt}")
    gt = load_gt(args.gt, visibility_threshold=args.visibility)
    print(f"  → {len(gt)} detections | "
          f"{gt['frame'].nunique()} frames | "
          f"{gt['id'].nunique()} unique IDs")

    print(f"\nLoading predictions  : {args.pred}")
    pred = load_predictions(args.pred)
    print(f"  → {len(pred)} detections | "
          f"{pred['frame'].nunique()} frames | "
          f"{pred['id'].nunique()} unique IDs")

    # Accumulate 
    print("\nBuilding accumulator…")
    acc = build_accumulator(gt, pred, max_iou=args.max_iou)

    # Compute metrics 
    mh      = mm.metrics.create()
    summary = mh.compute(
        acc,
        metrics=mm.metrics.motchallenge_metrics,
        name="result",
    )

    # Display
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names,
    )
    print("\n── MOTChallenge Metrics ──────────────────────────────")
    print(strsummary)

    mota = summary.loc["result", "mota"] # type: ignore
    motp = summary.loc["result", "motp"] # type: ignore
    idf1 = summary.loc["result", "idf1"] # type: ignore
    print(f"\n  ★  MOTA  = {mota * 100:.2f} %") # type: ignore
    print(f"  ★  MOTP  = {motp:.4f}  (lower = tighter boxes)")
    print(f"  ★  IDF1  = {idf1 * 100:.2f} %") # type: ignore

    if args.visibility == 0.0:
        print("\n  ⚠  Tip: re-run with --visibility 0.25 to exclude occluded GT objects.")



if __name__ == "__main__":
    main()