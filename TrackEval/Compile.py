"""
Compile.py

Sweeps object-tracking runs across combinations of model, detection
confidence, and tracker algorithm/config, then scores every resulting
prediction against ground truth with TrackEval (HOTA / CLEAR / Identity)
and compiles the results into a CSV.

Pipeline:
    video --[ObjectTracking]--> prediction .txt (MOT format)
          --[TrackEval.Eval]--> HOTA / CLEAR / Identity metrics --> CSV

Directory layout expected under the GT path's grandparent:
    MotMetrics/
    ├── GroundTruth/<vid>/gt.txt
    └── Prediction/<vid>/
            metrics_results.csv           (all GT detections)
            metrics_results_occluded.csv  (occluded GT detections included)

USAGE
-----
Evaluate existing prediction files in Prediction/<vid>/ (no tracking run):
    python Compile.py --gt MotMetrics/GroundTruth/vid1/gt.txt

Run tracker(s) on a video, then evaluate only the files just produced:
    python Compile.py --gt MotMetrics/GroundTruth/vid1/gt.txt --video vid1.mp4

Sweep multiple model sizes and confidence thresholds (cartesian product):
    python Compile.py --gt ... --video vid1.mp4 \
        --models n s m --confidence 0.25 0.4 0.6

Sweep trackers too. Each --trackers token is one of:
    none                                        no tracker override
    bytetrack                                   tracker name, default params
    '{"name":"botsort","with_reid":true}'        tracker name + param overrides
        python Compile.py --gt ... --video vid1.mp4 \
            --trackers none bytetrack "{\"name\":\"botsort\",\"with_reid\":true}"
    (On Windows cmd.exe, escape inner quotes as \" as shown above;
    PowerShell and bash accept single-quoted JSON as-is.)

Include occluded ground-truth detections in scoring:
    python Compile.py --gt ... --occluded

ARGUMENTS
---------
--gt            (required) Path to gt.txt.
--video         Optional. If given, runs tracker(s) on this video before
                evaluating; if omitted, evaluates existing prediction files
                in Prediction/<vid>/ instead.
--occluded      Include GT detections marked occluded (visibility == 0).
--confidence    One or more detection confidence thresholds to sweep.
--models        One or more model sizes (e.g. n, s, m), mapped to
                yolo26<size>.pt.
--trackers      One or more tracker specs: 'none', a tracker name, or a
                JSON dict with 'name' plus parameter overrides. See USAGE.
"""

from TrackEval.Eval import hota_score, IDF1_score
from Ultralytics.saving_bboxes import ObjectTracking 
from Ultralytics.tracker import build_tracker_config
import argparse
import json
import itertools
from pathlib import Path
import pandas as pd

def pred_from_gt(gt_path: Path) -> Path:
    """
    Derive the prediction folder from the gt path.

    Fixed layout:
        MotMetrics/
        ├── GroundTruth/<vid>/gt.txt
        └── Prediction/<vid>/
    """
    return gt_path.parent.parent.parent / "Prediction" / gt_path.parent.name

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute MOTA / MOTChallenge metrics from a GT and prediction file.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--gt", required=True,
        help=("FolderName/Data/GroundTruth/<vid>/gt.txt\n"
            "Prediction folder is automatically resolved to FolderName/Data/Prediction"),)
    parser.add_argument("--video", type=str, default=None,
        help=("Video path. When provided the tracker(s) run and write\n"
            "output(s) into the resolved prediction folder before evaluation.\n"
            "Omit to evaluate existing TXT files in the prediction folder directly."),)
    parser.add_argument("--occluded", type=bool, default=False,
        help="Include the occluded detections from the ground truth")
    parser.add_argument("--confidence", nargs="+", type=float, default=[None],
        help="One or more detection confidence thresholds to sweep over.")
    parser.add_argument("--models", nargs="+", default=["m"],
        help="One or more model sizes, e.g. n s m l x (mapped to yolo26<x>.pt)")
    parser.add_argument("--trackers", nargs="+", type=parse_tracker_spec, default=[None],
                     help="Tracker specs: a name (e.g. bytetrack), a JSON dict with overrides "
                          "(e.g. '{\"name\":\"botsort\",\"with_reid\":true}'), or 'none' for no tracker.")
    return parser.parse_args()

# Tracker

def cli_to_model(models: list) -> list:
    return [f"models/yolo26{model}.pt" for model in models]

def parse_tracker_spec(value: str):
    """
    argparse `type=` for --trackers tokens. Accepts:
    - "none" / "None"                          -> None (no tracker override)
    - "bytetrack"                              -> tracker name string
    - '{"name":"botsort","with_reid":true}'    -> dict of name + overrides
    - "configs/my_tracker.yaml"                -> Path to an existing tracker
                                                   YAML file, used as-is
    """
    if value.lower() == "none":
        return None

    stripped = value.strip()
    if stripped.startswith("{"):
        try:
            spec = json.loads(stripped)
        except json.JSONDecodeError as e:
            raise argparse.ArgumentTypeError(f"Invalid JSON tracker spec {value!r}: {e}")
        if "name" not in spec:
            raise argparse.ArgumentTypeError(f"Tracker spec dict must include 'name': {value!r}")
        return spec

    path_candidate = Path(stripped)
    if path_candidate.suffix.lower() in (".yaml", ".yml") or path_candidate.is_file():
        if not path_candidate.exists():
            raise argparse.ArgumentTypeError(f"Tracker config file not found: {stripped!r}")
        return path_candidate

    return stripped

def build_pred_filename(model_file: str, confidence, tracker_file) -> str:
    """
    Build a prediction filename that encodes every swept parameter, plus the
    video id (taken from the GT folder name, same convention as pred_from_gt).
    """
    parts = [Path(model_file).stem]
    if confidence is not None:
        parts.append(f"conf{confidence}")
    if tracker_file is not None:
        parts.append(Path(tracker_file).stem)
    return "_".join(parts) + ".txt"

def run_trackers(video_path: str, gt_path: str, pred_path: str,
                confidences: list, models: list, trackers: list) -> list[Path]:
    """
    Run ObjectTracking for every combination (cartesian product) of
    model x confidence x tracker, writing one MOT output file per combo.

    `trackers` is a list of tracker *specs*, not raw file paths. Each spec is
    one of:
    - None                          -> no tracker override, ObjectTracking
                                        falls back to its own default.
    - "bytetrack"                   -> build that tracker's config with
                                        default params.
    - {"name": "botsort",
        "with_reid": True,
        "appearance_thresh": 0.85}   -> build that tracker's config with
                                        the given overrides.

    The actual YAML each spec resolves to is built via build_tracker_config()
    and cached within this call, so the same spec used across multiple
    model/confidence combos is only written to disk once.

    Returns the list of prediction files that were just created, so callers
    can evaluate only those instead of the whole prediction folder.
    """
    # vid = Path(gt_path).parent.name
    model_files = cli_to_model(models)
    new_files: list[Path] = []

    # spec -> built config Path, so identical tracker specs aren't rebuilt
    # on every model/confidence iteration.
    built_tracker_cache: dict[str, Path | None] = {}

    def resolve_tracker(spec) -> Path | None:
        if spec is None:
            return None

        if isinstance(spec, Path):
            # Explicit file path — use it directly, no build_tracker_config().
            return spec

        if isinstance(spec, str):
            tracker_name, overrides = spec, {}
        elif isinstance(spec, dict):
            spec = dict(spec)  # don't mutate caller's dict
            tracker_name = spec.pop("name")
            overrides = spec
        else:
            raise TypeError(
                f"Tracker spec must be None, a name string, or a dict with "
                f"'name', got {type(spec).__name__}: {spec!r}"
            )

        cache_key = json.dumps({"name": tracker_name, **overrides}, sort_keys=True, default=str)
        if cache_key not in built_tracker_cache:
            built_tracker_cache[cache_key] = build_tracker_config(tracker_name, **overrides)
        return built_tracker_cache[cache_key]

    for model_file, confidence, tracker_spec in itertools.product(model_files, confidences, trackers):
        tracker_path = resolve_tracker(tracker_spec)

        print(f"\n Running tracker | model={model_file}  confidence={confidence}  tracker={tracker_path}")

        kwargs = dict(source=video_path, model=model_file, display=False,
                    gt=gt_path, confidence=confidence)
        if tracker_path is not None:
            kwargs["tracker"] = str(tracker_path)

        tracker = ObjectTracking(**kwargs)  # type: ignore
        filename = build_pred_filename(model_file, confidence, tracker_path)
        output_path = Path(pred_path) / filename
        tracker.run(output_mot=str(output_path))
        new_files.append(output_path)

    return new_files

# Evaluation

def evaluate_file_hota(gt_path: str, pred_file: Path, occluded: bool) -> dict:
    """Evaluate a single prediction TXT file and return a metrics row."""
    hota, summary = hota_score(gt_path, occluded, str(pred_file), return_summary=True) # type: ignore

    print(f"\n HOTA  = {hota * 100:.2f} % — {pred_file.name}") # type: ignore
    print(f"   IDs GT : {summary['gt_id_count']}  |  IDs Prediction : {summary['pred_id_count']}")

    row = {"model": pred_file.name}
    row.update(summary)
    return row

def evaluate_file_IDF1(gt_path: str, pred_file: Path, occluded: bool) -> dict:
    """Evaluate a single prediction TXT file and return a metrics row."""
    Idf1, summary = IDF1_score(gt_path, occluded, str(pred_file), return_summary=True) # type: ignore

    print(f"\n Idf1  = {Idf1 * 100:.2f} % — {pred_file.name}") # type: ignore
    print(f"   IDs GT : {summary['gt_id_count']}  |  IDs Prediction : {summary['pred_id_count']}")

    row = {"model": pred_file.name}
    row.update(summary)
    return row

def evaluate_folder(gt_path: str, pred_folder: Path, occluded: bool) -> list[dict]:
    """Evaluate every TXT file found in pred_folder."""
    txt_files = sorted(pred_folder.glob("*.txt"))
    if not txt_files:
        print(f"\n Aucun fichier .txt trouvé dans : {pred_folder}")
        return []

    return evaluate_files(gt_path, txt_files, occluded)

def evaluate_files(gt_path: str, pred_files: list[Path], occluded: bool) -> list[dict]:
    """Evaluate exactly the given list of prediction TXT files."""
    if not pred_files:
        print("\n Aucun fichier de prédiction à évaluer.")
        return []

    return [evaluate_file_IDF1(gt_path, pred_file, occluded) for pred_file in pred_files]

# CSV export

def save_csv(rows: list[dict], csv_path: Path) -> None:
    """
    Persist metric rows to a CSV file, merging with any existing results
    instead of overwriting them (rows are keyed on 'model' / filename, so
    re-evaluating a file updates its row rather than duplicating it).
    """
    if not rows:
        print("\n Aucun résultat à sauvegarder.")
        return

    df = pd.DataFrame(rows)

    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        existing = existing[~existing["model"].isin(df["model"])]
        df = pd.concat([existing, df], ignore_index=True)

    df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"\n  Résultats sauvegardés dans : {csv_path}")

# Display helpers

def print_summary(summary: dict, label: str) -> None:
    """Pretty-print a TrackEval summary dict."""
    print(f"\n── TrackEval Metrics — {label} " + "─" * max(0, 50 - len(label)))
    for key, value in summary.items():
        print(f"  {key:20s} : {value}")

# Entry point

def main() -> None:
    args      = parse_args()
    gt_path   = Path(args.gt)
    pred_path = pred_from_gt(gt_path)

    csv_name = "metrics_results_occluded.csv" if args.occluded else "metrics_results.csv"
    csv_path = pred_path / csv_name

    # Tracking
    if args.video is not None:
        new_files = run_trackers(args.video, str(gt_path), str(pred_path),
                                args.confidence, args.models, args.trackers)
        all_rows = evaluate_files(str(gt_path), new_files, args.occluded)
    else:
        all_rows = evaluate_folder(str(gt_path), pred_path, args.occluded)

    save_csv(all_rows, csv_path)

if __name__ == "__main__":
    main()