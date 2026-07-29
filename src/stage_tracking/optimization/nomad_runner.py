# type: ignore
"""
Optimizes a tracker's confidence threshold and tracker-specific parameters
using NOMAD (PyNomad), driven by an `OptimizationConfig` object instead of
module-level globals -- see `optimize_parameters()` for the public API.
"""
from __future__ import annotations

import json
import signal
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import PyNomad

from stage_tracking.config.loading import load_optimization_config
from stage_tracking.config.models import OptimizationConfig
from stage_tracking.evaluation.compile import pred_from_gt, save_csv
from stage_tracking.evaluation.metrics import AssA_score, IDF1_score, hota_score
from stage_tracking.tracking.pipeline import run_tracking_sweep
from stage_tracking.tracking.tracker_config import build_tracker_config, list_valid_params

DEFAULT_CONFIG_PATH = Path("opti_config.yaml")

METRIC_CONFIG = {
    "idf1": {"score_fn": IDF1_score, "summary_key": "Identity.IDF1", "label": "IDF1"},
    "hota": {"score_fn": hota_score, "summary_key": "HOTA.HOTA", "label": "HOTA"},
    "assa": {"score_fn": AssA_score, "summary_key": "HOTA.AssA", "label": "AssA"},
}


def parse_args(argv: list[str] | None = None):
    import argparse

    parser = argparse.ArgumentParser(
        description="Optimize tracker parameters with NOMAD.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to the YAML config file",
    )
    parser.add_argument("--video", dest="video_path", help="Override video path from config")
    parser.add_argument("--gt", dest="gt_path", help="Override GT path from config")
    parser.add_argument("--metric", choices=sorted(METRIC_CONFIG), help="Override metric")
    parser.add_argument("--results-dir", help="Override results directory")
    parser.add_argument("--models", nargs="+", help="Override the model list")
    parser.add_argument("--occluded", action="store_true", default=None, help="Include occluded GT rows")
    parser.add_argument("--max-bb-eval", type=int, help="Override max_bb_eval")
    parser.add_argument("--stagnation-window", type=int, help="Override stagnation_window")
    parser.add_argument("--stagnation-rel-threshold", type=float, help="Override stagnation_rel_threshold")
    parser.add_argument("--n-best", type=int, help="Override n_best")
    parser.add_argument(
        "--search-space",
        type=json.loads,
        help="Override search_space from config (JSON list of dimension dicts)",
    )
    parser.add_argument(
        "--x0-raw",
        type=json.loads,
        help="Override x0_raw from config (JSON list of starting-point values)",
    )
    parser.add_argument(
        "--tracker-override-defaults",
        type=json.loads,
        help="Override tracker_override_defaults from config (JSON object)",
    )
    return parser.parse_args(argv)


def validate_config(config: OptimizationConfig) -> None:
    """Catch config mistakes before spending any NOMAD evals."""
    if len(config.models) != 1:
        raise ValueError(
            "models must contain exactly one model - this optimizer evaluates one model's prediction at a time."
        )
    if len(config.x0_raw) != len(config.search_space):
        raise ValueError(
            f"x0_raw has {len(config.x0_raw)} values but search_space has {len(config.search_space)} dimensions."
        )
    if not any(d.get("target") == "confidence" for d in config.search_space):
        raise ValueError("search_space must include one dimension with target='confidence'.")
    if config.metric not in METRIC_CONFIG:
        raise ValueError(f"metric must be one of {sorted(METRIC_CONFIG)}, got {config.metric!r}.")

    valid = set(list_valid_params(config.tracker_name))
    for dim in config.search_space:
        if dim.get("target") == "tracker" and dim.get("name") not in valid:
            raise ValueError(
                f"'{dim['name']}' is not a valid parameter for tracker '{config.tracker_name}'. "
                f"Valid parameters: {sorted(valid)}"
            )


def point_to_args(search_space: list[dict], x_coords: list[float]) -> tuple[float, dict]:
    """Split a NOMAD coordinate vector into (confidence, tracker_overrides)."""
    confidence = None
    overrides = {}
    for dim, coord in zip(search_space, x_coords):
        value = int(round(coord)) if dim.get("is_int") else coord
        if dim.get("target") == "confidence":
            confidence = value
        else:
            overrides[dim["name"]] = value
    return confidence, overrides


def metrics_for_point(config: OptimizationConfig, confidence: float, tracker_overrides: dict):
    """Run the tracker at a given confidence + tracker-param combo and score it."""
    pred_path = pred_from_gt(config.gt_path)
    pred_path.mkdir(parents=True, exist_ok=True)

    tracker_overrides_full = {**config.tracker_override_defaults, **tracker_overrides}
    tracker_spec = {"name": config.tracker_name, **tracker_overrides_full}

    new_files = run_tracking_sweep(
        str(config.video_path),
        str(config.gt_path),
        str(pred_path),
        confidences=[confidence],
        models=config.models,
        trackers=[tracker_spec],
    )

    if len(new_files) != 1:
        raise RuntimeError(f"Expected exactly 1 prediction file, got {len(new_files)}: {new_files}")

    pred_file = new_files[0]

    metric_cfg = METRIC_CONFIG[config.metric]
    _score, summary = metric_cfg["score_fn"](str(config.gt_path), config.occluded, str(pred_file), return_summary=True)

    if metric_cfg["summary_key"] not in summary:
        raise KeyError(
            f"No '{metric_cfg['summary_key']}' key in summary. Available keys: {list(summary.keys())}"
        )

    id_diff = abs(summary["gt_id_count"] - summary["pred_id_count"])
    config_path = str(build_tracker_config(config.tracker_name, **tracker_overrides_full))
    csv_row = {"model": pred_file.name, **summary}
    return summary[metric_cfg["summary_key"]], id_diff, config_path, csv_row


def snap_to_granularity(value: float, lower: float, granularity: float) -> float:
    """Round value to the nearest point on the granularity grid anchored at lower."""
    if not granularity:
        return value
    steps = round((value - lower) / granularity)
    return round(lower + steps * granularity, 10)


def _update_top_n(top_list: list, eval_record: dict, key: str, reverse: bool, n: int):
    """Insert eval_record into top_list, keep it sorted, and truncate to the best n entries."""
    top_list.append(eval_record)
    top_list.sort(key=lambda e: e[key], reverse=reverse)
    del top_list[n:]


class OptimizationState:
    """Holds mutable optimizer state for results bookkeeping and soft-stop behavior."""

    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.results_path = config.results_dir / f"nomad_{config.tracker_name}_{datetime.now():%Y%m%d_%H%M%S}.json"
        self.partial_results_path = self.results_path.with_suffix(".partial.json")
        self.csv_path = pred_from_gt(config.gt_path) / (
            "metrics_results_occluded.csv" if config.occluded else "metrics_results.csv"
        )

        self.all_evals = []
        self.best_evals_by_metric = []
        self.best_evals_by_id_diff = []
        self.recent_scores = deque(maxlen=config.stagnation_window)
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.stop_reason = None

    def request_stop(self, reason: str):
        if not self.stop_event.is_set():
            self.stop_reason = reason
            print(f"\n[STOPPING] {reason} -- finishing in-flight evals, no new work will start.")
            self.stop_event.set()

    def record_failure(self, eval_record: dict):
        with self.lock:
            self.all_evals.append(eval_record)
            self.recent_scores.clear()
            self._save_partial()

    def record_success(self, eval_record: dict, csv_row: dict) -> bool:
        """Records a successful eval and returns True if it triggers the stagnation stop."""
        with self.lock:
            self.all_evals.append(eval_record)

            _update_top_n(self.best_evals_by_metric, eval_record, self.config.metric, reverse=True, n=self.config.n_best)
            _update_top_n(self.best_evals_by_id_diff, eval_record, "id_diff", reverse=False, n=self.config.n_best)

            self.recent_scores.append(eval_record[self.config.metric])
            stagnated = self._is_stagnated()

            self._save_partial()
            save_csv([csv_row], self.csv_path)
        return stagnated

    def _is_stagnated(self) -> bool:
        if len(self.recent_scores) < self.config.stagnation_window:
            return False
        window_max, window_min = max(self.recent_scores), min(self.recent_scores)
        rel_spread = (window_max - window_min) / abs(window_max) if window_max else 0.0
        return rel_spread <= self.config.stagnation_rel_threshold

    def _save_partial(self):
        """Flush progress so far to disk. Must be called while holding self.lock."""
        self.partial_results_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "tracker": self.config.tracker_name,
            "metric": self.config.metric,
            "search_space": self.config.search_space,
            "status": "in_progress" if not self.stop_event.is_set() else f"stopping ({self.stop_reason})",
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "all_evals": self.all_evals,
            f"best_evals_by_{self.config.metric}": self.best_evals_by_metric,
            "best_evals_by_id_diff": self.best_evals_by_id_diff,
        }
        tmp_path = self.partial_results_path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        tmp_path.replace(self.partial_results_path)

    def save_final(self, raw_result, elapsed_seconds: float, status: str) -> dict:
        self.config.results_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "tracker": self.config.tracker_name,
            "metric": self.config.metric,
            "video_path": str(self.config.video_path),
            "gt_path": str(self.config.gt_path),
            "models": self.config.models,
            "occluded": self.config.occluded,
            "search_space": self.config.search_space,
            "status": status,
            "stop_reason": self.stop_reason,
            "elapsed_seconds": elapsed_seconds,
            "completed_at": datetime.now().isoformat(timespec="seconds"),
            "num_evals": len(self.all_evals),
            "num_successful": sum(1 for e in self.all_evals if e.get("success")),
            "raw_nomad_result": raw_result,
            f"best_evals_by_{self.config.metric}": self.best_evals_by_metric,
            "best_evals_by_id_diff": self.best_evals_by_id_diff,
        }
        with open(self.results_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nResults saved to: {self.results_path.resolve()}")

        if self.partial_results_path.exists():
            self.partial_results_path.unlink()
        return payload


def make_blackbox(config: OptimizationConfig, state: OptimizationState):
    """Builds the NOMAD blackbox callback, closing over config + state."""
    label_metric = METRIC_CONFIG[config.metric]["label"]

    def bb(x):
        if state.stop_event.is_set():
            return 0

        x_coords = [x.get_coord(i) for i in range(len(config.search_space))]
        confidence, tracker_overrides = point_to_args(config.search_space, x_coords)
        label = f"confidence={confidence:.4f}  " + "  ".join(
            f"{k}={v}" for k, v in tracker_overrides.items()
        )
        eval_record = {
            "confidence": confidence,
            **tracker_overrides,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }

        try:
            score, id_diff, config_path, csv_row = metrics_for_point(config, confidence, tracker_overrides)
        except Exception as exc:
            print(f"[FAILED] {label} -> {exc}")
            eval_record.update({"success": False, "error": str(exc), "tracker_config_yaml": None})
            state.record_failure(eval_record)
            return 0

        print(f"[OK] {label}  {label_metric}={score:.4f}  ID_diff={id_diff}  config={config_path}")
        eval_record.update({
            "success": True,
            config.metric: score,
            "id_diff": id_diff,
            "tracker_config_yaml": config_path,
        })

        stagnated = state.record_success(eval_record, csv_row)
        if stagnated:
            best_so_far = state.best_evals_by_metric[0][config.metric] if state.best_evals_by_metric else float("nan")
            state.request_stop(
                f"Last {config.stagnation_window} consecutive successful evals all landed within "
                f"{config.stagnation_rel_threshold * 100:.3f}% (relative) of each other "
                f"({label_metric} range [{min(state.recent_scores):.4f}, {max(state.recent_scores):.4f}]) -- "
                f"search has converged (best {label_metric}={best_so_far:.4f})."
            )

        x.setBBO(f"{-score}".encode("UTF-8"))  # single output: minimize -score
        return 1

    return bb


def build_nomad_params(config: OptimizationConfig, x0: list[float]) -> list[str]:
    search_space = config.search_space
    lower_bounds = " ".join(str(d["lower"]) for d in search_space)
    upper_bounds = " ".join(str(d["upper"]) for d in search_space)
    granularity = " ".join(str(d["granularity"]) for d in search_space)
    input_types = " ".join("I" if d.get("is_int") else "R" for d in search_space)
    return [
        f"DIMENSION {len(search_space)}",
        "BB_OUTPUT_TYPE OBJ",
        "DIRECTION_TYPE ORTHO N+1 QUAD",
        f"MAX_BB_EVAL {config.max_bb_eval}",
        f"LOWER_BOUND ( {lower_bounds} )",
        f"UPPER_BOUND ( {upper_bounds} )",
        f"BB_INPUT_TYPE ( {input_types} )",
        "DISPLAY_DEGREE 2",
        "DISPLAY_ALL_EVAL true",
        f"GRANULARITY ( {granularity} )",
        f"MIN_MESH_SIZE ( {granularity} )",
        "NB_THREADS_PARALLEL_EVAL 6",
    ]


def _format_eval_line(config: OptimizationConfig, eval_record: dict) -> str:
    label_metric = METRIC_CONFIG[config.metric]["label"]
    param_names = [d["name"] for d in config.search_space]
    values = "  ".join(f"{name}={eval_record[name]}" for name in param_names)
    return (
        f"{values}  {label_metric}={eval_record[config.metric]:.4f}  "
        f"ID_diff={eval_record['id_diff']}  "
        f"config={eval_record.get('tracker_config_yaml')}"
    )


def print_best(config: OptimizationConfig, state: OptimizationState):
    label_metric = METRIC_CONFIG[config.metric]["label"]

    print(f"\n--- Top {config.n_best} by {label_metric} (highest first) ---")
    if not state.best_evals_by_metric:
        print("No successful evaluation found -- every trial failed.")
    else:
        for i, ev in enumerate(state.best_evals_by_metric, start=1):
            print(f"{i}. {_format_eval_line(config, ev)}")

    print(f"\n--- Top {config.n_best} by ID-count error (lowest first) ---")
    if not state.best_evals_by_id_diff:
        print("No successful evaluation found -- every trial failed.")
    else:
        for i, ev in enumerate(state.best_evals_by_id_diff, start=1):
            print(f"{i}. {_format_eval_line(config, ev)}")


def optimize_parameters(config: OptimizationConfig) -> dict:
    """
    Run the NOMAD search described by `config` and return the final results
    payload (the same structure written to <results_dir>/nomad_<tracker>_<timestamp>.json).
    """
    validate_config(config)

    state = OptimizationState(config)
    bb = make_blackbox(config, state)

    def handle_shutdown_signal(signum, frame):
        state.request_stop(f"Received signal {signal.Signals(signum).name}.")

    previous_sigint = signal.signal(signal.SIGINT, handle_shutdown_signal)
    previous_sigterm = signal.signal(signal.SIGTERM, handle_shutdown_signal)

    x0 = [snap_to_granularity(v, d["lower"], d["granularity"]) for v, d in zip(config.x0_raw, config.search_space)]
    params = build_nomad_params(config, x0)

    print(f"Optimizing {METRIC_CONFIG[config.metric]['label']} over:", ", ".join(d["name"] for d in config.search_space))
    print(f"Results will be saved to: {state.results_path.resolve()}")
    print(f"CSV rows will be merged into: {state.csv_path.resolve()}")

    start = time.monotonic()
    status = "completed"
    result = None
    try:
        result = PyNomad.optimize(bb, x0, [], [], params)
        if state.stop_event.is_set():
            status = "stagnated" if "converged" in (state.stop_reason or "") else "interrupted"
    except KeyboardInterrupt:
        state.request_stop("KeyboardInterrupt in main thread.")
        status = "interrupted"
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)
        elapsed = time.monotonic() - start
        print("\n--- Raw NOMAD result ---")
        print(result)
        print_best(config, state)
        payload = state.save_final(result, elapsed, status)

    return payload


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    overrides = dict(
        video_path=args.video_path,
        gt_path=args.gt_path,
        metric=args.metric,
        results_dir=args.results_dir,
        models=args.models,
        occluded=args.occluded,
        max_bb_eval=args.max_bb_eval,
        stagnation_window=args.stagnation_window,
        stagnation_rel_threshold=args.stagnation_rel_threshold,
        n_best=args.n_best,
        search_space=args.search_space,
        x0_raw=args.x0_raw,
        tracker_override_defaults=args.tracker_override_defaults,
    )
    config = load_optimization_config(args.config, **overrides)
    print(f"Using config: {args.config}")
    optimize_parameters(config)


if __name__ == "__main__":
    main()
