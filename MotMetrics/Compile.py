from MotMetrics.MotScore import mot_score
from Ultralytics.saving_bboxes import ObjectTracking
import argparse
from pathlib import Path
import pandas as pd
import motmetrics as mm


MODELS = ["yolo26n.pt", "yolo26s.pt", "yolo26m.pt", "yolo26l.pt", "yolo26x.pt"]

def gt_from_pred(pred_path: Path) -> Path:
    """
    Derive the GT path from the prediction folder.
 
    Fixed layout:
        MotMetrics/
        ├── GroundTruth/<vid>/gt/gt.txt
        └── Prediction/<vid>/            ← pred_path
    """
    return pred_path.parent.parent / "GroundTruth" / pred_path.name / "gt" / "gt.txt"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute MOTA / MOTChallenge metrics from a GT and prediction file.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--pred", required=True,
        help=("MotMetrics/Prediction/<vid>/\n"
              "GT is automatically resolved to MotMetrics/GroundTruth/<vid>/gt/gt.txt"),)
    parser.add_argument("--video", type=str, default=None,
        help=("Video path. When provided the tracker runs and writes\n"
            "its output to --pred before evaluation.\n"
            "Omit to evaluate existing TXT files in --pred directly."),)
    return parser.parse_args()

# Tracker

def run_trackers(video_path: str, gt_path: str, pred_path: str) -> None:
    """Run ObjectTracking for every model and write MOT output files."""
    for model in MODELS:
        print(f"\n Running tracker with model : {model}")
        tracker = ObjectTracking(source=video_path, model=model, display=False, gt=gt_path)
        tracker.run(output_mot=pred_path)


# Evaluation

def evaluate_file(gt_path: str, pred_file: Path) -> dict:
    """Evaluate a single prediction TXT file and return a metrics row."""
    mota, summary = mot_score(gt_path, 0.25, pred_file, return_summary=True) # type: ignore

    print_summary(summary, label=pred_file.name) # type: ignore
    print(f"\n MOTA  = {mota * 100:.2f} % — {pred_file.name}")  # type: ignore

    return summary_to_row(summary, label=pred_file.name) # type: ignore


def evaluate_folder(gt_path: str, pred_folder: Path) -> list[dict]:
    """Evaluate every TXT file found in pred_folder."""
    txt_files = sorted(pred_folder.glob("*.txt"))
    if not txt_files:
        print(f"\n⚠  Aucun fichier .txt trouvé dans : {pred_folder}")
        return []

    rows: list[dict] = []
    for pred_file in txt_files:
        print(f"\n  Évaluation de : {pred_file.name}")
        rows.append(evaluate_file(gt_path, pred_file))
    return rows


# CSV export

def save_csv(rows: list[dict], csv_path: Path) -> None:
    """Persist all metric rows to a CSV file."""
    if not rows:
        print("\n Aucun résultat à sauvegarder.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"\n✔  Résultats sauvegardés dans : {csv_path}")


# Display helpers

def print_summary(summary: pd.DataFrame, label: str) -> None:
    """Pretty-print a motmetrics summary DataFrame."""
    mh = mm.metrics.create()
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names,
    )
    print(f"\n── MOTChallenge Metrics — {label} " + "─" * max(0, 50 - len(label)))
    print(strsummary)


def summary_to_row(summary: pd.DataFrame, label: str) -> dict:
    """Flatten a motmetrics summary DataFrame into a plain dict row."""
    flat: dict = {"model": label}
    for col in summary.columns:
        value = summary[col].iloc[0]
        friendly = mm.io.motchallenge_metric_names.get(col, col)
        flat[friendly] = value
    return flat


# Entry point

def main() -> None:
    args      = parse_args()
    pred_path = Path(args.pred)
    gt_path   = str(gt_from_pred(pred_path))
    csv_path  = pred_path / "metrics_results.csv"
 
    # Tracking
    if args.video is not None:
        run_trackers(args.video, gt_path, str(pred_path))

    all_rows = evaluate_folder(gt_path, pred_path)

    save_csv(all_rows, csv_path)


if __name__ == "__main__":
    main()