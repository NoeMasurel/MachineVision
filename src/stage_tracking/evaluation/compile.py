"""
Compile.py
----------
Sweeps object-tracking runs across combinations of model, detection
confidence, and tracker algorithm/config (via stage_tracking.tracking.pipeline),
then scores every resulting prediction against ground truth with TrackEval
(HOTA / CLEAR / Identity) and compiles the results into a CSV.

Pipeline:
    video --[tracking.pipeline]--> prediction .txt (MOT format)
          --[evaluation.metrics]--> HOTA / CLEAR / Identity metrics --> CSV

Directory layout expected under the GT path's grandparent:
    Data/
    ├── GroundTruth/<vid>/gt.txt
    └── Prediction/<vid>/
            metrics_results.csv           (all GT detections)
            metrics_results_occluded.csv  (occluded GT detections included)

USAGE
-----
Evaluate existing prediction files in Prediction/<vid>/ (no tracking run):
    stage-evaluate --gt Data/GroundTruth/vid1/gt.txt

Run tracker(s) on a video, then evaluate only the files just produced:
    stage-evaluate --gt Data/GroundTruth/vid1/gt.txt --video vid1.mp4

Sweep model sizes, confidence thresholds, and trackers *in lockstep* (paired,
NOT a cartesian product): the i-th run uses models[i], confidence[i] and
trackers[i]. All three lists must be the same length:
    stage-evaluate --gt ... --video vid1.mp4 \
        --models n s m --confidence 0.25 0.4 0.6 \
        --trackers bytetrack botsort none

Each --trackers token is one of:
    none                                        no tracker override
    bytetrack                                   tracker name, default params
    '{"name":"botsort","with_reid":true}'        tracker name + param overrides
    configs/my_tracker.yaml                      path to an existing tracker YAML file
        stage-evaluate --gt ... --video vid1.mp4 \
            --models n s --confidence 0.25 0.4 \
            --trackers none "{\"name\":\"botsort\",\"with_reid\":true}"
    (On Windows cmd.exe, escape inner quotes as \" as shown above;
    PowerShell and bash accept single-quoted JSON as-is.)

Include occluded ground-truth detections in scoring:
    stage-evaluate --gt ... --occluded

ARGUMENTS
---------
--gt            (required) Path to gt.txt.
--video         Optional. If given, runs tracker(s) on this video before
                evaluating; if omitted, evaluates existing prediction files
                in Prediction/<vid>/ instead.
--occluded      Include GT detections marked occluded (visibility == 0).
--confidence    One or more detection confidence thresholds to sweep,
                paired index-for-index with --models and --trackers.
--models        One or more model sizes (e.g. n, s, m), mapped to
                yolo26<size>.pt. Paired index-for-index with --confidence
                and --trackers.
--trackers      One or more tracker specs: 'none', a tracker name, a JSON
                dict with 'name' plus parameter overrides, or a path to a
                .yaml/.yml tracker config file. Paired index-for-index with
                --models and --confidence. See USAGE.
"""

from stage_tracking.evaluation.metrics import hota_score, IDF1_score, AssA_score
from stage_tracking.tracking.pipeline import parse_tracker_spec, run_tracking_sweep
from stage_tracking.config.models import EvaluationConfig
import argparse

from pathlib import Path
import pandas as pd

def pred_from_gt(gt_path: Path) -> Path:
    """
    Derive the prediction folder from the gt path.

    Fixed layout:
        Data/
        ├── GroundTruth/<vid>/gt.txt
        └── Prediction/<vid>/
    """
    return gt_path.parent.parent.parent / "Prediction" / gt_path.parent.name

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute MOTA / MOTChallenge metrics from a GT and prediction file.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--gt", required=True,
        help=("FolderName/Data/GroundTruth/<vid>/gt.txt\n"
            "Prediction folder can be  automatically resolved to FolderName/Data/Prediction"))
    parser.add_argument("--pred", default=None,
        help="Overwrite pred path")
    parser.add_argument("--video", type=str, default=None,
        help=("Video path. When provided the tracker(s) run and write\n"
            "output(s) into the resolved prediction folder before evaluation.\n"
            "Omit to evaluate existing TXT files in the prediction folder directly."),)
    parser.add_argument("--occluded", type=bool, default=False,
        help="Include the occluded detections from the ground truth")
    parser.add_argument("--confidence", nargs="+", type=float, default=[None],
        help="One or more detection confidence thresholds, paired index-for-index "
             "with --models and --trackers (NOT a cartesian product).")
    parser.add_argument("--models", nargs="+", default=["m"],
        help="One or more model sizes, e.g. n s m l x (mapped to yolo26<x>.pt). "
             "Paired index-for-index with --confidence and --trackers.")
    parser.add_argument("--trackers", nargs="+", type=parse_tracker_spec, default=[None],
                     help="Tracker specs: a name (e.g. bytetrack), a JSON dict with overrides "
                          "(e.g. '{\"name\":\"botsort\",\"with_reid\":true}'), a path to a "
                          ".yaml/.yml tracker config file, or 'none' for no tracker. "
                          "Paired index-for-index with --models and --confidence.")
    return parser.parse_args(argv)

# Evaluation

def evaluate_file_hota(gt_path: str, pred_file: Path, occluded: bool, verbose: bool = True) -> dict:
    """Evaluate a single prediction TXT file and return a metrics row."""
    hota, summary = hota_score(gt_path, occluded, str(pred_file), return_summary=True) # type: ignore

    if verbose:
        print(f"\n HOTA  = {hota * 100:.2f} % — {pred_file.name}") # type: ignore
        print(f"   IDs GT : {summary['gt_id_count']}  |  IDs Prediction : {summary['pred_id_count']}")

    row = {"model": pred_file.name}
    row.update(summary)
    return row

def evaluate_file_IDF1(gt_path: str, pred_file: Path, occluded: bool, verbose: bool = True) -> dict:
    """Evaluate a single prediction TXT file and return a metrics row."""
    Idf1, summary = IDF1_score(gt_path, occluded, str(pred_file), return_summary=True) # type: ignore

    if verbose:
        print(f"\n Idf1  = {Idf1 * 100:.2f} % — {pred_file.name}") # type: ignore
        print(f"   IDs GT : {summary['gt_id_count']}  |  IDs Prediction : {summary['pred_id_count']}")

    row = {"model": pred_file.name}
    row.update(summary)
    return row

def evaluate_file_AssA(gt_path: str, pred_file: Path, occluded: bool, verbose: bool = True) -> dict:
    """Evaluate a single prediction TXT file and return a metrics row."""
    AssA, summary = AssA_score(gt_path, occluded, str(pred_file), return_summary=True) # type: ignore

    if verbose:
        print(f"\n AssA  = {AssA * 100:.2f} % — {pred_file.name}") # type: ignore
        print(f"   IDs GT : {summary['gt_id_count']}  |  IDs Prediction : {summary['pred_id_count']}")

    row = {"model": pred_file.name}
    row.update(summary)
    return row

def evaluate_folder(gt_path: str, pred_folder: Path, occluded: bool, verbose: bool = True) -> list[dict]:
    """Evaluate every TXT file found in pred_folder."""
    txt_files = sorted(pred_folder.glob("*.txt"))
    if not txt_files:
        if verbose:
            print(f"\n No .txt file found in: {pred_folder}")
        return []

    return evaluate_files(gt_path, txt_files, occluded, verbose=verbose)

def evaluate_files(gt_path: str, pred_files: list[Path], occluded: bool, verbose: bool = True) -> list[dict]:
    """Evaluate exactly the given list of prediction TXT files."""
    if not pred_files:
        if verbose:
            print("\n No prediction file to evaluate.")
        return []

    return [evaluate_file_hota(gt_path, pred_file, occluded, verbose=verbose) for pred_file in pred_files]


def evaluate_predictions(config: EvaluationConfig) -> list[dict]:
    """
    Evaluate prediction file(s) against ground truth and return the metric
    rows (does not write a CSV -- pair with save_csv() for that side effect).
    """
    if config.pred_files is not None:
        return evaluate_files(str(config.gt_path), config.pred_files, config.occluded, verbose=config.verbose)

    pred_path = config.pred_path or pred_from_gt(config.gt_path)
    return evaluate_folder(str(config.gt_path), pred_path, config.occluded, verbose=config.verbose)

# CSV export

def save_csv(rows: list[dict], csv_path: Path) -> None:
    """
    Persist metric rows to a CSV file, merging with any existing results
    instead of overwriting them (rows are keyed on 'model' / filename, so
    re-evaluating a file updates its row rather than duplicating it).
    """
    if not rows:
        print("\n No result to save.")
        return

    df = pd.DataFrame(rows)

    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        existing = existing[~existing["model"].isin(df["model"])]
        df = pd.concat([existing, df], ignore_index=True)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"\n  Results saved to: {csv_path}")

# Display helpers

def print_summary(summary: dict, label: str) -> None:
    """Pretty-print a TrackEval summary dict."""
    print(f"\n── TrackEval Metrics — {label} " + "─" * max(0, 50 - len(label)))
    for key, value in summary.items():
        print(f"  {key:20s} : {value}")

# Entry point

def main(argv: list[str] | None = None) -> None:
    args      = parse_args(argv)
    gt_path   = Path(args.gt)
    if args.pred:
        pred_path = Path(args.pred)
    else:
        pred_path = pred_from_gt(gt_path)

    csv_name = "metrics_results_occluded.csv" if args.occluded else "metrics_results.csv"
    csv_path = pred_path / csv_name

    # Tracking
    if args.video is not None:
        new_files = run_tracking_sweep(args.video, str(gt_path), str(pred_path),
                                args.confidence, args.models, args.trackers)
        all_rows = evaluate_files(str(gt_path), new_files, args.occluded)
    else:
        all_rows = evaluate_folder(str(gt_path), pred_path, args.occluded)

    save_csv(all_rows, csv_path)

if __name__ == "__main__":
    main()
