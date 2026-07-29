"""
object_tracking.py — YOLO-based object tracking with optional display.

Output is written in the MOT (Multiple Object Tracking) challenge format:
    <frame>,<id>,<bb_left>,<bb_top>,<bb_width>,<bb_height>,<conf>,<x>,<y>,<z>

Usage examples:
  # Basic run (with GUI window)
  python -m tracking.tracking.pipeline videos/clip.mp4

  # SSH / headless server (no display)
  python -m tracking.tracking.pipeline videos/clip.mp4 --no-display

  # Trim to a time window, custom confidence, save output video
  python -m tracking.tracking.pipeline videos/clip.mp4 --start 10 --end 60 \
      --conf 0.4 --save-video output/annotated.mp4

  # Custom model and tracker, custom MOT output path
  python -m tracking.tracking.pipeline videos/clip.mp4 \
      --model yolo26n.pt --tracker deepocsort.yaml \
      --output-mot runtime/detections/my_results.txt

  # Full option reference
  python -m tracking.tracking.pipeline --help
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import cv2
import yaml
from ultralytics import YOLO
from ultralytics.utils.plotting import colors
import pandas as pd

from tracking.config.models import TrackingConfig
from tracking.tracking.tracker_config import build_tracker_config


def _load_tracker_params(tracker_path: str | None) -> dict:
    """
    Try to read the tracker .yaml file and return its contents as a dict.
    Returns an empty dict (with a note) if the file can't be found or parsed,
    so a missing YAML never crashes the run.
    """
    if tracker_path is None:
        return {"tracker": "ultralytics_default"}

    path = Path(tracker_path)
    if not path.exists():
        # Ultralytics ships built-in tracker configs by name (e.g. "deepocsort.yaml").
        # We can't read them from disk directly, so just record the name.
        return {"tracker": tracker_path, "note": "built-in config — file not on disk"}

    try:
        with open(path) as f:
            params = yaml.safe_load(f) or {}
        params["tracker"] = tracker_path
        return params
    except Exception as exc:
        return {"tracker": tracker_path, "parse_error": str(exc)}

def _resolve_class_indices(names: dict, words: list[str]) -> list[int]:
    """Return class indices for the given label strings; raise on unknown labels."""
    inv = {v.lower(): k for k, v in names.items()}
    indices = []
    unknown = []
    for w in words:
        idx = inv.get(w.lower())
        if idx is None:
            unknown.append(w)
        else:
            indices.append(idx)
    if unknown:
        available = sorted(names.values())
        raise ValueError(
            f"Unknown class(es): {unknown}\n"
            f"Available classes: {available}"
        )
    return indices

TRACKED_CLASS = "person"

class ObjectTracking:
    def __init__(
        self,
        source: str,
        model: str = "yolo26n.pt",
        start: float | None = None,
        end: float | None = None,
        duration: float | None = None,
        confidence: float | None = None,
        tracker: str | None = None,
        display: bool = True,
        save_video: str | None = None,
        gt: str | None = None,
    ):
        """
        Parameters
        ----------
        source      : path to the source video
        model       : YOLO .pt file (default: yolo26n.pt)
        start       : clip start in seconds (default: 0)
        end         : clip end in seconds — mutually exclusive with `duration`
        duration    : clip length in seconds — mutually exclusive with `end`
        confidence  : detection confidence threshold (0–1)
        tracker     : tracking config .yaml file
        display     : show a live OpenCV window (set False for SSH)
        save_video  : path to write an annotated output .mp4 (optional)
        gt          : path to a gt file to set the start and end frame (optional)
        """
        self.source        = source
        self.model_name    = model
        self.confidence    = confidence
        self.tracker       = tracker
        self.tracker_params = _load_tracker_params(tracker)
        self.display       = display
        self.save_video    = save_video

        # Load model
        self.model   = YOLO(model)
        self.names   = self.model.names          # {int: str}
        self.indices = _resolve_class_indices(self.names, [TRACKED_CLASS])

        # Open video
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video file: {source!r}")

        self.w   = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h   = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames   = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_duration = total_frames / self.fps

        print(
            f"Video: {self.w}×{self.h} @ {self.fps:.2f} FPS | "
            f"Duration: {total_duration:.2f}s ({total_frames} frames)"
        )
        # Resolve time window
        if gt is not None :
            if not Path(gt).exists():
                raise FileNotFoundError(f"File not found: {gt}")
            df = pd.read_csv(gt, header=None)
            self.start_frame = int(df.iloc[0, 0]) # type: ignore
            self.end_frame = int(df.iloc[-1, 0]) # type: ignore
            print(f"Analyzing frames {self.start_frame} - {self.end_frame}")
        else :
            if end is not None and duration is not None:
                raise ValueError("Provide either 'end' or 'duration', not both.")

            self.start_sec = float(start) if start is not None else 0.0

            if end is not None:
                self.end_sec = float(end)
            elif duration is not None:
                self.end_sec = self.start_sec + float(duration)
            else:
                self.end_sec = total_duration

            self.start_sec = max(0.0, self.start_sec)
            self.end_sec   = min(total_duration, self.end_sec)

            if self.start_sec >= self.end_sec:
                raise ValueError(
                    f"Invalid time window: [{self.start_sec:.2f}s, {self.end_sec:.2f}s]"
                )

            self.start_frame = int(self.start_sec * self.fps)
            self.end_frame   = int(self.end_sec   * self.fps)

            print(
                f"Processing: {self.start_sec:.2f}s → {self.end_sec:.2f}s "
                f"(frames {self.start_frame}–{self.end_frame})"
            )

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)

        # State
        self.track_history  = defaultdict(list)
        self.class_counts   = defaultdict(int)
        self.seen_ids: set  = set()

        # Video writer (optional)
        self._writer: cv2.VideoWriter | None = None
        if save_video:
            Path(save_video).parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v") # type: ignore
            self._writer = cv2.VideoWriter(save_video, fourcc, self.fps, (self.w, self.h))
            print(f"Output video: {save_video!r}")

        # Drawing config
        self.rect_width         = 2
        self.font_scale         = 1.0
        self.text_thickness     = 2
        self.label_padding      = 12
        self.label_margin       = 10
        self.polyline_thickness = 2

    # Drawing

    def _draw_bbox(self, im0, box, track_id: int, cls: int, conf: float) -> None:
        x1, y1, x2, y2 = map(int, box)
        color = colors(cls, True)

        cv2.rectangle(im0, (x1, y1), (x2, y2), color, self.rect_width)

        label = f"{self.names[cls]}:{track_id} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.text_thickness
        )

        bg_x1, bg_y2 = x1, y1
        bg_x2 = bg_x1 + tw + 2 * self.label_padding
        bg_y1 = bg_y2 - (th + 2 * self.label_margin)

        cv2.rectangle(im0, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)

        text_x = bg_x1 + ((bg_x2 - bg_x1) - tw) // 2
        text_y = bg_y1 + ((bg_y2 - bg_y1) + th) // 2 - 2
        cv2.putText(
            im0, label, (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX, self.font_scale,
            (255, 255, 255), self.text_thickness, cv2.LINE_AA,
        )

    def _draw_class_counts(self, frame) -> None:
        if not self.class_counts:
            return
        line_h  = 36
        padding = 10
        width   = 220
        height  = padding + len(self.class_counts) * line_h + padding

        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (10 + width, 10 + height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        for i, (class_name, count) in enumerate(sorted(self.class_counts.items())):
            y = 10 + padding + i * line_h + line_h // 2 + 8
            cv2.putText(
                frame, f"{class_name}: {count}", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                (0, 255, 200), 2, cv2.LINE_AA,
            )

    def _update_counts(self, track_id: int, cls: int) -> None:
        key = (track_id, cls)
        if key not in self.seen_ids:
            self.seen_ids.add(key)
            self.class_counts[self.names[cls]] += 1

    # Main loop

    def run(self, output_mot: str = "runtime/detections/detections.txt") -> list[tuple]:
        """
        Run tracking and return detections as a list of MOT rows.

        Each row is a tuple:
            (frame, id, bb_left, bb_top, bb_width, bb_height, conf, -1, -1, -1)

        The data is written to *output_mot* in CSV format (no header),
        """
        Path(output_mot).parent.mkdir(parents=True, exist_ok=True)

        # Print tracker config at start so it's visible in SSH output
        print("\nTracker parameters:")
        for k, v in self.tracker_params.items():
            print(f"  {k}: {v}")
        print()

        total_to_process = self.end_frame - self.start_frame
        current_frame    = self.start_frame
        processed        = 0

        # Accumulate all MOT rows; written to disk at the end.
        mot_rows: list[tuple] = []

        try:
            while self.cap.isOpened() and current_frame < self.end_frame:
                success, frame = self.cap.read()
                if not success:
                    print("End of video or read error at frame", current_frame)
                    break

                current_frame += 1
                processed     += 1

                # Inference
                kwargs: dict = {
                    "persist": True,
                    "verbose": False,
                    "classes": self.indices,
                }
                if self.confidence is not None:
                    kwargs["conf"] = self.confidence
                if self.tracker is not None:
                    kwargs["tracker"] = self.tracker

                results = self.model.track(source=frame, **kwargs)

                if not results:
                    continue

                result          = results[0]
                annotated_frame = result.plot()

                # Draw our own count overlay on top
                self._draw_class_counts(annotated_frame)

                # Optional display
                if self.display:
                    cv2.imshow("YOLO Tracking", annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("Quit key pressed — stopping early.")
                        break

                # Optional video save
                if self._writer is not None:
                    self._writer.write(annotated_frame)

                #Progress print (every 5 %)
                pct = processed / total_to_process * 100
                if processed == 1 or processed % max(1, total_to_process // 20) == 0:
                    print(f"  [{pct:5.1f}%] frame {current_frame}", end="\r", flush=True)

                # Record detections in MOT format
                if result.boxes is None or result.boxes.id is None:
                    continue

                boxes = result.boxes.xyxy.cpu().tolist() # type: ignore
                ids   = result.boxes.id.cpu().tolist() # type: ignore
                clss  = result.boxes.cls.cpu().tolist() # type: ignore
                confs = result.boxes.conf.cpu().tolist() # type: ignore

                for box, track_id, cls, conf in zip(boxes, ids, clss, confs):
                    tid = int(track_id)
                    cid = int(cls)
                    self._update_counts(tid, cid)

                    # Convert from xyxy to xywh (MOT uses top-left + width/height)
                    x1, y1, x2, y2 = box
                    bb_left   = round(x1, 1)
                    bb_top    = round(y1, 1)
                    bb_width  = round(x2 - x1, 1)
                    bb_height = round(y2 - y1, 1)

                    # MOT row: frame, id, left, top, width, height, conf, x, y, z
                    # frame is 1-based per the MOT convention
                    mot_rows.append((
                        current_frame,      # 1-based frame index
                        tid,                # track ID
                        bb_left,
                        bb_top,
                        bb_width,
                        bb_height,
                        round(conf, 4),
                        -1,                 # world x (unused for 2-D)
                        -1,                 # world y (unused for 2-D)
                        -1,                 # world z (unused for 2-D)
                    ))

        finally:
            # Always release resources, even on exception
            self.cap.release()
            if self._writer is not None:
                self._writer.release()
            if self.display:
                cv2.destroyAllWindows()

        print()  # newline after \r progress

        if len(self.class_counts):
            print("\nFinal class counts:")
            for class_name, count in sorted(self.class_counts.items()):
                print(f"  {class_name}: {count}")
        else :
            print("\nNo detections made")

        # Save MOT file
        mot_rows.sort(key=lambda r: (r[0], r[1]))

        with open(output_mot, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(mot_rows)

        print(f"MOT detections saved → {output_mot}")
        return mot_rows


def run_tracking(config: TrackingConfig) -> Path:
    """Run one tracking pass described by `config` and return the path of the
    written MOT prediction file (config.output_path)."""
    tracker = ObjectTracking(
        source=str(config.video_path),
        model=config.model,
        start=config.start,
        end=config.end,
        duration=config.duration,
        confidence=config.confidence,
        tracker=str(config.tracker) if config.tracker is not None else None,
        display=config.display,
        save_video=str(config.save_video) if config.save_video is not None else None,
        gt=str(config.gt_path) if config.gt_path is not None else None,
    )
    tracker.run(output_mot=str(config.output_path))
    return config.output_path


# Sweep helpers -- paired model/confidence/tracker combinations, as used by
# the evaluation and optimization CLIs (NOT a cartesian product; see
# align_sweep_lists).

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


def align_sweep_lists(confidences: list, models: list, trackers: list) -> list[tuple]:
    """
    Pair up models / confidences / trackers index-for-index instead of
    taking their cartesian product.

    Rules:
    - If a list has length 1 and another has length > 1, the length-1 list is
      broadcast (repeated) to match, so you can still fix e.g. a single model
      while sweeping confidence + tracker together.
    - Otherwise all lists that have length > 1 must share the same length,
      or this raises a ValueError.
    """
    lists = {"models": models, "confidence": confidences, "trackers": trackers}
    lengths = {name: len(lst) for name, lst in lists.items()}
    non_trivial = {name: n for name, n in lengths.items() if n > 1}

    if non_trivial:
        target_len = next(iter(non_trivial.values()))
        mismatched = {name: n for name, n in non_trivial.items() if n != target_len}
        if mismatched:
            raise ValueError(
                "When sweeping in lockstep, --models / --confidence / --trackers must "
                f"either have length 1 or all share the same length. Got lengths: {lengths}"
            )
    else:
        target_len = 1

    def broadcast(lst):
        return lst if len(lst) == target_len else lst * target_len

    return list(zip(broadcast(models), broadcast(confidences), broadcast(trackers)))


_tracker_spec_cache: dict[str, Path | None] = {}


def _resolve_tracker_spec(spec) -> Path | None:
    """
    Resolve a tracker *spec* (None / name / override-dict / existing yaml Path)
    into a concrete tracker YAML path via build_tracker_config(), caching by
    spec so an identical spec repeated across multiple sweep combos is only
    written to disk once.
    """
    if spec is None:
        return None

    if isinstance(spec, Path):
        return spec

    if isinstance(spec, str):
        tracker_name, overrides = spec, {}
    elif isinstance(spec, dict):
        spec = dict(spec)  # don't mutate caller's dict
        tracker_name = spec.pop("name")
        overrides = spec
    else:
        raise TypeError(
            f"Tracker spec must be None, a name string, a dict with 'name', "
            f"or a Path to a YAML file, got {type(spec).__name__}: {spec!r}"
        )

    cache_key = json.dumps({"name": tracker_name, **overrides}, sort_keys=True, default=str)
    if cache_key not in _tracker_spec_cache:
        _tracker_spec_cache[cache_key] = build_tracker_config(tracker_name, **overrides)
    return _tracker_spec_cache[cache_key]


def run_tracking_sweep(video_path: str, gt_path: str, pred_path: str,
                        confidences: list, models: list, trackers: list) -> list[Path]:
    """
    Run ObjectTracking for each paired (model, confidence, tracker) triple --
    i.e. the i-th run uses models[i], confidences[i], trackers[i] -- writing
    one MOT output file per combo. This is a lockstep zip, NOT a cartesian
    product: --models, --confidence and --trackers must be the same length
    (or length 1, in which case they're broadcast -- see align_sweep_lists).

    `trackers` is a list of tracker *specs*, not raw file paths (see
    _resolve_tracker_spec).

    Returns the list of prediction files that were just created, so callers
    can evaluate only those instead of the whole prediction folder.
    """
    model_files = cli_to_model(models)
    new_files: list[Path] = []
    combos = align_sweep_lists(confidences, model_files, trackers)

    for model_file, confidence, tracker_spec in combos:
        tracker_path = _resolve_tracker_spec(tracker_spec)

        print(f"\n Running tracker | model={model_file}  confidence={confidence}  tracker={tracker_path}")

        config = TrackingConfig(
            video_path=Path(video_path),
            model=model_file,
            output_path=Path(pred_path) / build_pred_filename(model_file, confidence, tracker_path),
            confidence=confidence,
            tracker=tracker_path,
            gt_path=Path(gt_path),
            display=False,
        )
        output_path = run_tracking(config)
        new_files.append(output_path)

    return new_files


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="stage-track",
        description=(
            "YOLO object tracking with optional headless mode. "
            "MOT format saving "
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Positional
    p.add_argument("source", help="Path to input video file.")

    # Model / tracker
    p.add_argument("--model", default="models/yolo26n.pt", help="YOLO model weights (.pt).")
    p.add_argument("--tracker", default=None, help="Tracker config .yaml (e.g. deepocsort.yaml).")
    p.add_argument("--conf", type=float, default=None, help="Confidence threshold (0–1). Uses model default if omitted.")

    # Time window (mutually exclusive)
    time_group = p.add_mutually_exclusive_group()
    time_group.add_argument("--end", type=float, default=None, help="End time in seconds (exclusive with --duration).")
    time_group.add_argument("--duration", type=float, default=None, help="Clip length in seconds from --start (exclusive with --end).")
    p.add_argument("--start", type=float, default=None, help="Start time in seconds.")

    p.add_argument("--gt", type=str, default=None, help="Filepath to csv ground truth in MOT format")

    # Output
    p.add_argument("--output", default="runtime/detections/detections.txt", help="Path for the MOT output file.",)
    p.add_argument("--save-video", default=None, help="Save annotated video to this path (e.g. output/annotated.mp4).")

    # Display
    p.add_argument("--no-display", action="store_true", help="Disable the OpenCV preview window (use this over SSH/headless).")

    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    config = TrackingConfig(
        video_path=Path(args.source),
        model=args.model,
        output_path=Path(args.output),
        confidence=args.conf,
        tracker=args.tracker,
        start=args.start,
        end=args.end,
        duration=args.duration,
        gt_path=Path(args.gt) if args.gt else None,
        display=not args.no_display,
        save_video=Path(args.save_video) if args.save_video else None,
    )
    run_tracking(config)


if __name__ == "__main__":
    main()
