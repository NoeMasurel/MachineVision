# type: ignore
"""
Optimizes a tracker's confidence threshold and tracker-specific parameters
using NOMAD (PyNomad).

Runtime configuration is now loaded from a YAML file, and all file paths are
resolved relative to the repository root instead of relying on hard-coded
absolute paths.
"""
import argparse
import json
import signal
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
import PyNomad
import yaml
import TrackEval.Compile as compile_mod

from TrackEval.Eval import AssA_score, IDF1_score, hota_score, id_diff_score
from Ultralytics.tracker import build_tracker_config, list_valid_params

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = Path(__file__).with_name("opti_config.yaml")

METRIC_CONFIG = {
    "idf1": {
        "score_fn": IDF1_score,
        "summary_key": "Identity.IDF1",
        "label": "IDF1",
        "maximize": True,
    },
    "hota": {
        "score_fn": hota_score,
        "summary_key": "HOTA.HOTA",
        "label": "HOTA",
        "maximize": True,
    },
    "assa": {
        "score_fn": AssA_score,
        "summary_key": "HOTA.AssA",
        "label": "AssA",
        "maximize": True,
    },
    "id_diff": {
        "score_fn": id_diff_score,
        "summary_key": "id_diff",
        "label": "ID Diff",
        "maximize": False,
    },
}

# Runtime globals -- all populated from the YAML config (optionally overridden
# by argparse) in load_runtime_config(). No hard-coded defaults live here;
# anything not supplied on the command line must be present in the config file.
CONFIG_PATH = DEFAULT_CONFIG_PATH
VIDEO_PATH = None
GT_PATH = None
MODELS = None
OCCLUDED = False
TRACKER_NAME = None
METRIC = None

MAX_BB_EVAL = None
STAGNATION_REL_THRESHOLD = None
STAGNATION_WINDOW = None
N_BEST = None
RESULTS_DIR = None
RESULTS_PATH = None
PARTIAL_RESULTS_PATH = None
CSV_PATH = None
SEARCH_SPACE = None
X0_RAW = None
TRACKER_OVERRIDE_DEFAULTS = None


def parse_args():
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
    parser.add_argument("--occluded", action="store_true", help="Include occluded GT rows")
    parser.add_argument("--max-bb-eval", type=int, help="Override MAX_BB_EVAL")
    parser.add_argument("--stagnation-window", type=int, help="Override STAGNATION_WINDOW")
    parser.add_argument("--stagnation-rel-threshold", type=float, help="Override STAGNATION_REL_THRESHOLD")
    parser.add_argument("--n-best", type=int, help="Override N_BEST")
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
    return parser.parse_args()


def resolve_path(path_value, base_dir: Path = REPO_ROOT) -> Path:
    path = Path(str(path_value)).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _require(config: dict, key: str):
    """Fetch a required key from the config file, raising a clear error if absent."""
    if key not in config:
        raise ValueError(
            f"Missing required key '{key}' in config file: {CONFIG_PATH}. "
            f"Either add it to the YAML or pass the equivalent --{key.replace('_', '-')} argument."
        )
    return config[key]


def load_runtime_config(args):
    global CONFIG_PATH, VIDEO_PATH, GT_PATH, MODELS, OCCLUDED, TRACKER_NAME, METRIC
    global MAX_BB_EVAL, STAGNATION_REL_THRESHOLD, STAGNATION_WINDOW, N_BEST
    global RESULTS_DIR, RESULTS_PATH, PARTIAL_RESULTS_PATH, CSV_PATH, SEARCH_SPACE, X0_RAW
    global TRACKER_OVERRIDE_DEFAULTS

    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / config_path).resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}

    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {config_path}")

    CONFIG_PATH = config_path
    VIDEO_PATH = resolve_path(args.video_path or _require(config, "video_path"))
    GT_PATH = resolve_path(args.gt_path or _require(config, "gt_path"))

    MODELS = args.models or _require(config, "models")
    OCCLUDED = bool(args.occluded or config.get("occluded", False))

    TRACKER_NAME = _require(config, "tracker_name")
    METRIC = args.metric or _require(config, "metric")

    MAX_BB_EVAL = args.max_bb_eval if args.max_bb_eval is not None else _require(config, "max_bb_eval")
    STAGNATION_REL_THRESHOLD = (
        args.stagnation_rel_threshold
        if args.stagnation_rel_threshold is not None
        else _require(config, "stagnation_rel_threshold")
    )
    STAGNATION_WINDOW = (
        args.stagnation_window if args.stagnation_window is not None else _require(config, "stagnation_window")
    )
    N_BEST = args.n_best if args.n_best is not None else _require(config, "n_best")

    RESULTS_DIR = resolve_path(args.results_dir or _require(config, "results_dir"))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    RESULTS_PATH = RESULTS_DIR / f"nomad_{TRACKER_NAME}_{datetime.now():%Y%m%d_%H%M%S}.json"
    PARTIAL_RESULTS_PATH = RESULTS_PATH.with_suffix(".partial.json")

    SEARCH_SPACE = args.search_space if args.search_space is not None else _require(config, "search_space")
    X0_RAW = args.x0_raw if args.x0_raw is not None else _require(config, "x0_raw")
    TRACKER_OVERRIDE_DEFAULTS = (
        args.tracker_override_defaults
        if args.tracker_override_defaults is not None
        else _require(config, "tracker_override_defaults")
    )

    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"Video file not found: {VIDEO_PATH}")
    if not GT_PATH.exists():
        raise FileNotFoundError(f"Ground-truth file not found: {GT_PATH}")

    CSV_PATH = compile_mod.pred_from_gt(GT_PATH) / (
        "metrics_results_occluded.csv" if OCCLUDED else "metrics_results.csv"
    )


def validate_config():
    """Catch config mistakes before spending any NOMAD evals."""
    if len(MODELS) != 1:
        raise ValueError(
            "MODELS must contain exactly one model - this optimizer evaluates one model's prediction at a time."
        )
    if len(X0_RAW) != len(SEARCH_SPACE):
        raise ValueError(
            f"X0_RAW has {len(X0_RAW)} values but SEARCH_SPACE has {len(SEARCH_SPACE)} dimensions."
        )
    if not any(d.get("target") == "confidence" for d in SEARCH_SPACE):
        raise ValueError("SEARCH_SPACE must include one dimension with target='confidence'.")
    if METRIC not in METRIC_CONFIG:
        raise ValueError(f"METRIC must be one of {sorted(METRIC_CONFIG)}, got {METRIC!r}.")

    valid = set(list_valid_params(TRACKER_NAME))
    for dim in SEARCH_SPACE:
        if dim.get("target") == "tracker" and dim.get("name") not in valid:
            raise ValueError(
                f"'{dim['name']}' is not a valid parameter for tracker '{TRACKER_NAME}'. "
                f"Valid parameters: {sorted(valid)}"
            )


def point_to_args(x_coords: list[float]) -> tuple[float, dict]:
    """Split a NOMAD coordinate vector into (confidence, tracker_overrides)."""
    confidence = None
    overrides = {}
    for dim, coord in zip(SEARCH_SPACE, x_coords):
        value = int(round(coord)) if dim.get("is_int") else coord
        if dim.get("target") == "confidence":
            confidence = value
        else:
            overrides[dim["name"]] = value
    return confidence, overrides  


def metrics_for_point(confidence: float, tracker_overrides: dict):
    """Run the tracker at a given confidence + tracker-param combo and score it."""
    gt_path = GT_PATH
    pred_path = compile_mod.pred_from_gt(gt_path)
    pred_path.mkdir(parents=True, exist_ok=True)

    tracker_overrides_full = {**TRACKER_OVERRIDE_DEFAULTS, **tracker_overrides}
    tracker_spec = {"name": TRACKER_NAME, **tracker_overrides_full}

    new_files = compile_mod.run_trackers(
        str(VIDEO_PATH),
        str(gt_path),
        str(pred_path),
        confidences=[confidence],
        models=MODELS,
        trackers=[tracker_spec],
    )

    if len(new_files) != 1:
        raise RuntimeError(f"Expected exactly 1 prediction file, got {len(new_files)}: {new_files}")

    pred_file = new_files[0]

    metric_cfg = METRIC_CONFIG[METRIC]  
    _score, summary = metric_cfg["score_fn"](str(gt_path), OCCLUDED, str(pred_file), return_summary=True)

    if metric_cfg["summary_key"] not in summary:
        raise KeyError(
            f"No '{metric_cfg['summary_key']}' key in summary. Available keys: {list(summary.keys())}"
        )

    id_diff = abs(summary["gt_id_count"] - summary["pred_id_count"])
    config_path = str(build_tracker_config(TRACKER_NAME, **tracker_overrides_full))
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

    def __init__(self):
        self.all_evals = []
        self.best_evals_by_metric = []
        self.best_evals_by_id_diff = []
        self.recent_scores = deque(maxlen=STAGNATION_WINDOW)
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

            maximize = METRIC_CONFIG[METRIC]["maximize"]

            _update_top_n(self.best_evals_by_metric, eval_record, METRIC, reverse=maximize, n=N_BEST,)
            _update_top_n(self.best_evals_by_id_diff, eval_record, "id_diff", reverse=False, n=N_BEST)

            self.recent_scores.append(eval_record[METRIC])
            stagnated = self._is_stagnated()

            self._save_partial()
            compile_mod.save_csv([csv_row], CSV_PATH)
        return stagnated

    def _is_stagnated(self) -> bool:
        if len(self.recent_scores) < STAGNATION_WINDOW:
            return False
        window_max, window_min = max(self.recent_scores), min(self.recent_scores)
        rel_spread = (window_max - window_min) / abs(window_max) if window_max else 0.0
        return rel_spread <= STAGNATION_REL_THRESHOLD

    def _save_partial(self):
        """Flush progress so far to disk. Must be called while holding self.lock."""
        PARTIAL_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "tracker": TRACKER_NAME,
            "metric": METRIC,
            "search_space": SEARCH_SPACE,
            "status": "in_progress" if not self.stop_event.is_set() else f"stopping ({self.stop_reason})",
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "all_evals": self.all_evals,
            f"best_evals_by_{METRIC}": self.best_evals_by_metric,
            "best_evals_by_id_diff": self.best_evals_by_id_diff,
        }
        tmp_path = PARTIAL_RESULTS_PATH.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        tmp_path.replace(PARTIAL_RESULTS_PATH)

    def save_final(self, raw_result, elapsed_seconds: float, status: str):
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "tracker": TRACKER_NAME,
            "metric": METRIC,
            "video_path": str(VIDEO_PATH),
            "gt_path": str(GT_PATH),
            "models": MODELS,
            "occluded": OCCLUDED,
            "search_space": SEARCH_SPACE,
            "status": status,
            "stop_reason": self.stop_reason,
            "elapsed_seconds": elapsed_seconds,
            "completed_at": datetime.now().isoformat(timespec="seconds"),
            "num_evals": len(self.all_evals),
            "num_successful": sum(1 for e in self.all_evals if e.get("success")),
            "raw_nomad_result": raw_result,
            f"best_evals_by_{METRIC}": self.best_evals_by_metric,
            "best_evals_by_id_diff": self.best_evals_by_id_diff,
        }
        with open(RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nResults saved to: {RESULTS_PATH.resolve()}")

        if PARTIAL_RESULTS_PATH.exists():
            PARTIAL_RESULTS_PATH.unlink()


def make_blackbox(state: OptimizationState):
    """Builds the NOMAD blackbox callback, closing over state."""
    label_metric = METRIC_CONFIG[METRIC]["label"]

    def bb(x):
        if state.stop_event.is_set():
            return 0

        x_coords = [x.get_coord(i) for i in range(len(SEARCH_SPACE))]
        confidence, tracker_overrides = point_to_args(x_coords)
        label = f"confidence={confidence:.4f}  " + "  ".join(
            f"{k}={v}" for k, v in tracker_overrides.items()
        )
        eval_record = {
            "confidence": confidence,
            **tracker_overrides,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }

        try:
            score, id_diff, config_path, csv_row = metrics_for_point(confidence, tracker_overrides)
        except Exception as exc:
            print(f"[FAILED] {label} -> {exc}")
            eval_record.update({"success": False, "error": str(exc), "tracker_config_yaml": None})
            state.record_failure(eval_record)
            return 0

        print(f"[OK] {label}  {label_metric}={score:.4f}  ID_diff={id_diff}  config={config_path}")
        eval_record.update({
            "success": True,
            METRIC: score,
            "id_diff": id_diff,
            "tracker_config_yaml": config_path,
        })

        stagnated = state.record_success(eval_record, csv_row)
        if stagnated:
            best_so_far = state.best_evals_by_metric[0][METRIC] if state.best_evals_by_metric else float("nan")
            state.request_stop(
                f"Last {STAGNATION_WINDOW} consecutive successful evals all landed within "
                f"{STAGNATION_REL_THRESHOLD * 100:.3f}% (relative) of each other "
                f"({label_metric} range [{min(state.recent_scores):.4f}, {max(state.recent_scores):.4f}]) -- "
                f"search has converged (best {label_metric}={best_so_far:.4f})."
            )
        maximize = METRIC_CONFIG[METRIC]["maximize"]
        objective = -score if maximize else score
        x.setBBO(f"{objective}".encode("UTF-8"))

        return 1

    return bb


def build_nomad_params(x0: list[float]) -> list[str]:
    lower_bounds = " ".join(str(d["lower"]) for d in SEARCH_SPACE)
    upper_bounds = " ".join(str(d["upper"]) for d in SEARCH_SPACE)
    granularity = " ".join(str(d["granularity"]) for d in SEARCH_SPACE)
    input_types = " ".join("I" if d.get("is_int") else "R" for d in SEARCH_SPACE)
    return [
        f"DIMENSION {len(SEARCH_SPACE)}",
        "BB_OUTPUT_TYPE OBJ",
        "DIRECTION_TYPE ORTHO N+1 QUAD",
        f"MAX_BB_EVAL {MAX_BB_EVAL}",
        f"LOWER_BOUND ( {lower_bounds} )",
        f"UPPER_BOUND ( {upper_bounds} )",
        f"BB_INPUT_TYPE ( {input_types} )",
        "DISPLAY_DEGREE 2",
        "DISPLAY_ALL_EVAL true",
        f"GRANULARITY ( {granularity} )",
        f"MIN_MESH_SIZE ( {granularity} )",
        "NB_THREADS_PARALLEL_EVAL 6",
    ]


def _format_eval_line(eval_record: dict) -> str:
    label_metric = METRIC_CONFIG[METRIC]["label"]
    param_names = [d["name"] for d in SEARCH_SPACE]
    values = "  ".join(f"{name}={eval_record[name]}" for name in param_names)
    return (
        f"{values}  {label_metric}={eval_record[METRIC]:.4f}  "
        f"ID_diff={eval_record['id_diff']}  "
        f"config={eval_record.get('tracker_config_yaml')}"
    )


def print_best(state: OptimizationState):
    label_metric = METRIC_CONFIG[METRIC]["label"]
    direction = "highest" if METRIC_CONFIG[METRIC]["maximize"] else "lowest"

    print(f"\n--- Top {N_BEST} by {label_metric} ({direction} first) ---")
    if not state.best_evals_by_metric:
        print("No successful evaluation found -- every trial failed.")
    else:
        for i, ev in enumerate(state.best_evals_by_metric, start=1):
            print(f"{i}. {_format_eval_line(ev)}")

    print(f"\n--- Top {N_BEST} by ID Diff (lowest first) ---")

    if not state.best_evals_by_id_diff:
        print("No successful evaluation found -- every trial failed.")
    else:
        for i, ev in enumerate(state.best_evals_by_id_diff, start=1):
            print(f"{i}. {_format_eval_line(ev)}")


def main():
    args = parse_args()
    load_runtime_config(args)
    validate_config()

    state = OptimizationState()
    bb = make_blackbox(state)

    def handle_shutdown_signal(signum, frame):
        state.request_stop(f"Received signal {signal.Signals(signum).name}.")

    signal.signal(signal.SIGINT, handle_shutdown_signal)
    signal.signal(signal.SIGTERM, handle_shutdown_signal)

    x0 = [snap_to_granularity(v, d["lower"], d["granularity"]) for v, d in zip(X0_RAW, SEARCH_SPACE)]
    params = build_nomad_params(x0)

    print(f"Using config: {CONFIG_PATH}")
    print(f"Optimizing {METRIC_CONFIG[METRIC]['label']} over:", ", ".join(d["name"] for d in SEARCH_SPACE))
    print(f"Results will be saved to: {RESULTS_PATH.resolve()}")
    print(f"CSV rows will be merged into: {CSV_PATH.resolve()}")

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
        elapsed = time.monotonic() - start
        print("\n--- Raw NOMAD result ---")
        print(result)
        print_best(state)
        state.save_final(result, elapsed, status)


if __name__ == "__main__":
    main()