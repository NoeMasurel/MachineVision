"""
Optimizes a tracker's `confidence` threshold and tracker-specific parameters
to maximize IDF1, using NOMAD (PyNomad). ID-count error is recorded per eval
but does not affect the search.

USAGE
-----
1. Edit the CONFIG section below (paths, tracker name, MAX_BB_EVAL, x0).
2. Edit SEARCH_SPACE to list the dimensions to optimize. Each entry is one
   NOMAD dimension:

       {"name": "confidence", "target": "confidence", "lower": 0.1,
        "upper": 0.9, "granularity": 0.05}

       {"name": "track_high_thresh", "target": "tracker", "lower": 0.1,
        "upper": 0.9, "granularity": 0.05}

       {"name": "track_buffer", "target": "tracker", "lower": 10,
        "upper": 100, "granularity": 1, "is_int": True}

   `target: "confidence"` feeds ObjectTracking's confidence arg directly.
   `target: "tracker"` becomes an override passed to build_tracker_config()
   for TRACKER_NAME. Use `is_int: True` for parameters that must stay whole
   numbers. Run `list_valid_params(TRACKER_NAME)` to see valid names.
3. Run the script. It stops when either:
     - NOMAD's own mesh-size convergence is reached (MIN_MESH_SIZE /
       MIN_POLL_SIZE are set to each dimension's GRANULARITY, so NOMAD stops
       refining once it can't move by less than a grid step), or
     - the application-level stagnation check trips: the last
       STAGNATION_WINDOW *consecutive* successful evals all land within
       STAGNATION_REL_THRESHOLD (relative) of each other.
   Both stops are "soft": a flag is set and every subsequent blackbox call
   returns an instant failure, so NOMAD burns through its remaining
   MAX_BB_EVAL budget quickly and `main()` always reaches save_final_results().
   Ctrl+C / SIGTERM trigger the same soft-stop path, so results are always
   saved on exit.
4. Results (every eval + the best point) are written to RESULTS_PATH as
   JSON when the run finishes, and incrementally to a sibling
   "*.partial.json" after every successful eval so a crash loses nothing.
   Each eval record includes "tracker_config_yaml", the path to the YAML
   config used for that trial (see resolve behavior in build_tracker_config).

PARALLEL EVALUATIONS
---------------------
Set NB_THREADS_PARALLEL_EVAL (in NOMAD_PARAMS below) to let NOMAD call the
blackbox from multiple worker threads. All shared state lives in
OptimizationState and is guarded by a single lock, so concurrent evals can't
corrupt the results file or race on updating "best". Only the tracker run +
scoring happens outside the lock.
"""

import json
import signal
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import PyNomad

import TrackEval.Compile as compile_mod
from TrackEval.Eval import IDF1_score
from Ultralytics.tracker import list_valid_params, build_tracker_config

# CONFIG
VIDEO_PATH = "GX011504.MP4" #/home/noe/Research/Data/
GT_PATH = "Data/GroundTruth/Vid1/gt.txt"
MODELS = ["m"]
OCCLUDED = False
TRACKER_NAME = "tracktrack"

MAX_BB_EVAL = 1
STAGNATION_REL_THRESHOLD = 0.0001
STAGNATION_WINDOW = 5
N_BEST = 5


RESULTS_DIR = Path("Results")
RESULTS_PATH = RESULTS_DIR / f"nomad_{TRACKER_NAME}_{datetime.now():%Y%m%d_%H%M%S}.json"
PARTIAL_RESULTS_PATH = RESULTS_PATH.with_suffix(".partial.json")

SEARCH_SPACE = [
    {"name": "confidence",        "target": "confidence", "lower": 0.1,  "upper": 0.9,  "granularity": 0.05},
    {"name": "track_high_thresh", "target": "tracker",    "lower": 0.1,  "upper": 0.9,  "granularity": 0.05},
    {"name": "track_low_thresh",  "target": "tracker",    "lower": 0.05, "upper": 0.5,  "granularity": 0.05},
    {"name": "new_track_thresh",  "target": "tracker",    "lower": 0.1,  "upper": 0.9,  "granularity": 0.05},
    {"name": "match_thresh",      "target": "tracker",    "lower": 0.5,  "upper": 0.95, "granularity": 0.05},
    {"name": "track_buffer",      "target": "tracker",    "lower": 10,   "upper": 100,  "granularity": 1, "is_int": True},
    {"name": "lost_match_thr",    "target": "tracker",    "lower": 0.0,  "upper": 0.5,  "granularity": 0.05},
    {"name": "iou_weight",        "target": "tracker",    "lower": 0.2,  "upper": 0.8,  "granularity": 0.05},
    {"name": "reid_weight",       "target": "tracker",    "lower": 0.2,  "upper": 0.8,  "granularity": 0.05},
    {"name": "conf_weight",       "target": "tracker",    "lower": 0.05, "upper": 0.8,  "granularity": 0.05},
]


X0_RAW = [0.5, 0.6, 0.25, 0.5, 0.7, 55.0, 0.25, 0.5, 0.5, 0.25]

TRACKER_OVERRIDE_DEFAULTS = {"with_reid": True, "gmc_method": "sparseOptFlow"}

# VALIDATION
def validate_config():
    """Catch config mistakes before spending any NOMAD evals."""
    if len(MODELS) != 1:
        raise ValueError("MODELS must contain exactly one model -- this optimizer "
                          "evaluates one model's prediction at a time.")
    if len(X0_RAW) != len(SEARCH_SPACE):
        raise ValueError(f"X0_RAW has {len(X0_RAW)} values but SEARCH_SPACE has "
                          f"{len(SEARCH_SPACE)} dimensions.")
    if not any(d["target"] == "confidence" for d in SEARCH_SPACE):
        raise ValueError("SEARCH_SPACE must include one dimension with target='confidence'.")

    valid = set(list_valid_params(TRACKER_NAME))
    for dim in SEARCH_SPACE:
        if dim["target"] == "tracker" and dim["name"] not in valid:
            raise ValueError(
                f"'{dim['name']}' is not a valid parameter for tracker "
                f"'{TRACKER_NAME}'. Valid parameters: {sorted(valid)}"
            )

# CORE EVAL LOGIC
def point_to_args(x_coords: list[float]) -> tuple[float, dict]:
    """Split a NOMAD coordinate vector into (confidence, tracker_overrides)."""
    confidence = None
    overrides = {}
    for dim, coord in zip(SEARCH_SPACE, x_coords):
        value = int(round(coord)) if dim.get("is_int") else coord
        if dim["target"] == "confidence":
            confidence = value
        else:
            overrides[dim["name"]] = value
    return confidence, overrides # type: ignore

def metrics_for_point(confidence: float, tracker_overrides: dict):
    """Run the tracker at a given confidence + tracker-param combo and score it."""
    gt_path = Path(GT_PATH)
    pred_path = compile_mod.pred_from_gt(gt_path)
    pred_path.mkdir(parents=True, exist_ok=True)

    tracker_overrides_full = {**TRACKER_OVERRIDE_DEFAULTS, **tracker_overrides}
    tracker_spec = {"name": TRACKER_NAME, **tracker_overrides_full}

    new_files = compile_mod.run_trackers(
        VIDEO_PATH, str(gt_path), str(pred_path),
        confidences=[confidence], models=MODELS, trackers=[tracker_spec],
    )
    if len(new_files) != 1:
        raise RuntimeError(f"Expected exactly 1 prediction file, got {len(new_files)}: {new_files}")
    pred_file = new_files[0]

    _idf1, summary = IDF1_score_score(str(gt_path), OCCLUDED, str(pred_file), return_summary=True)  # type: ignore
    if "Identity.IDF1" not in summary:
        raise KeyError(f"No 'Identity.IDF1' key in summary. Available keys: {list(summary.keys())}")

    id_diff = abs(summary["gt_id_count"] - summary["pred_id_count"])
    config_path = str(build_tracker_config(TRACKER_NAME, **tracker_overrides_full))
    return summary["Identity.IDF1"], id_diff, config_path

def snap_to_granularity(value: float, lower: float, granularity: float) -> float:
    """Round `value` to the nearest point on the granularity grid anchored at `lower`."""
    if not granularity:
        return value
    steps = round((value - lower) / granularity)
    return round(lower + steps * granularity, 10)

def _update_top_n(top_list: list, eval_record: dict, key: str, reverse: bool, n: int):
    """
    Insert `eval_record` into `top_list` (mutated in place), keep it sorted
    by `eval_record[key]` (descending if reverse=True, ascending otherwise),
    and truncate to the best `n` entries.
    """
    top_list.append(eval_record)
    top_list.sort(key=lambda e: e[key], reverse=reverse)
    del top_list[n:]
# SHARED STATE (results bookkeeping + soft-stop handling)
class OptimizationState:
    """
    Holds everything that's mutated from inside NOMAD's blackbox callback.
    Bundled into one object (instead of module-level globals) so the
    locking/stagnation/save logic lives in one place and stays testable.
    """

    def __init__(self):
        self.all_evals = []
        self.best_evals_by_idf1 = []
        self.best_evals_by_id_diff = []
        self.recent_idf1s = deque(maxlen=STAGNATION_WINDOW)
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
            self.recent_idf1s.clear()
            self._save_partial()

    def record_success(self, eval_record: dict) -> bool:
        """Records a successful eval and returns True if it triggers the stagnation stop."""
        with self.lock:
            self.all_evals.append(eval_record)

            _update_top_n(self.best_evals_by_idf1, eval_record, "idf1", reverse=True, n=N_BEST)
            _update_top_n(self.best_evals_by_id_diff, eval_record, "id_diff", reverse=False, n=N_BEST)

            self.recent_idf1s.append(eval_record["idf1"])
            stagnated = self._is_stagnated()

            self._save_partial()
        return stagnated

    def _is_stagnated(self) -> bool:
        if len(self.recent_idf1s) < STAGNATION_WINDOW:
            return False
        window_max, window_min = max(self.recent_idf1s), min(self.recent_idf1s)
        rel_spread = (window_max - window_min) / abs(window_max) if window_max else 0.0
        return rel_spread <= STAGNATION_REL_THRESHOLD

    def _save_partial(self):
        """Flush progress so far to disk. Must be called while holding self.lock."""
        PARTIAL_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "tracker": TRACKER_NAME,
            "search_space": SEARCH_SPACE,
            "status": "in_progress" if not self.stop_event.is_set() else f"stopping ({self.stop_reason})",
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "all_evals": self.all_evals,
            "best_evals_by_idf1": self.best_evals_by_idf1,
            "best_evals_by_id_diff": self.best_evals_by_id_diff,
        }
        tmp_path = PARTIAL_RESULTS_PATH.with_suffix(".tmp")
        with open(tmp_path, "w") as f:
            json.dump(payload, f, indent=2)
        tmp_path.replace(PARTIAL_RESULTS_PATH)

    def save_final(self, raw_result, elapsed_seconds: float, status: str):
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "tracker": TRACKER_NAME,
            "video_path": VIDEO_PATH,
            "gt_path": GT_PATH,
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
            "best_evals_by_idf1": self.best_evals_by_idf1,
            "best_evals_by_id_diff": self.best_evals_by_id_diff,
        }
        with open(RESULTS_PATH, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nResults saved to: {RESULTS_PATH.resolve()}")

        if PARTIAL_RESULTS_PATH.exists():
            PARTIAL_RESULTS_PATH.unlink()


def make_blackbox(state: OptimizationState):
    """Builds the NOMAD blackbox callback, closing over `state`."""

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
            idf1, id_diff, config_path = metrics_for_point(confidence, tracker_overrides)
        except Exception as e:
            print(f"[FAILED] {label} -> {e}")
            eval_record.update({"success": False, "error": str(e), "tracker_config_yaml": None})
            state.record_failure(eval_record)
            return 0

        print(f"[OK] {label}  IDF1={idf1:.4f}  ID_diff={id_diff}  config={config_path}")
        eval_record.update({
            "success": True,
            "idf1": idf1,
            "id_diff": id_diff,
            "tracker_config_yaml": config_path,
        })

        stagnated = state.record_success(eval_record)
        if stagnated:
            state.request_stop(
                f"Last {STAGNATION_WINDOW} consecutive successful evals all landed within "
                f"{STAGNATION_REL_THRESHOLD * 100:.3f}% (relative) of each other "
                f"(IDF1 range [{min(state.recent_idf1s):.4f}, {max(state.recent_idf1s):.4f}]) -- "
                f"search has converged (best IDF1={state.best_evals_by_idf1[0]:.4f})." # type: ignore
            )

        x.setBBO(f"{-idf1}".encode("UTF-8"))  # single output: minimize -idf1
        return 1

    return bb

# MAIN
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
    param_names = [d["name"] for d in SEARCH_SPACE]
    values = "  ".join(f"{name}={eval_record[name]}" for name in param_names)
    return (f"{values}  IDF1={eval_record['idf1']:.4f}  ID_diff={eval_record['id_diff']}  "
            f"config={eval_record.get('tracker_config_yaml')}")


def print_best(state: OptimizationState):
    print(f"\n--- Top {N_BEST} by IDF1 (highest first) ---")
    if not state.best_evals_by_idf1:
        print("No successful evaluation found -- every trial failed.")
    else:
        for i, ev in enumerate(state.best_evals_by_idf1, start=1):
            print(f"{i}. {_format_eval_line(ev)}")

    print(f"\n--- Top {N_BEST} by ID-count error (lowest first) ---")
    if not state.best_evals_by_id_diff:
        print("No successful evaluation found -- every trial failed.")
    else:
        for i, ev in enumerate(state.best_evals_by_id_diff, start=1):
            print(f"{i}. {_format_eval_line(ev)}")

def main():
    validate_config()
    state = OptimizationState()
    bb = make_blackbox(state)

    def handle_shutdown_signal(signum, frame):
        state.request_stop(f"Received signal {signal.Signals(signum).name}.")

    signal.signal(signal.SIGINT, handle_shutdown_signal)
    signal.signal(signal.SIGTERM, handle_shutdown_signal)

    x0 = [snap_to_granularity(v, d["lower"], d["granularity"]) for v, d in zip(X0_RAW, SEARCH_SPACE)]
    params = build_nomad_params(x0)

    print("Optimizing over:", ", ".join(d["name"] for d in SEARCH_SPACE))
    print(f"Results will be saved to: {RESULTS_PATH.resolve()}")

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