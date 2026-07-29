"""
playback.py — Replay a video with bounding boxes from detections CSV.

Usage
    python -m stage_tracking.video.playback -v path/to/video.mp4
    python -m stage_tracking.video.playback -v path/to/video.mp4 -c detections.txt
    python -m stage_tracking.video.playback -v path/to/video.mp4 -c detections.txt --speed 2.0

Controls
    q / ESC : quit
    SPACE   : pause / resume
    d       : step forward one frame (while paused)
    s       : step backward one frame (while paused)
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

PALETTE = [
    (0, 255, 200), (255, 100,   0), (  0, 100, 255), (200, 255,   0),
    (255,   0, 200), (  0, 200, 255), (255, 200,   0), (100,   0, 255),
    (  0, 255, 100), (255,  50,  50),
]

HUD_X           = 8
HUD_Y           = 8
HUD_LINE_HEIGHT = 28
HUD_PADDING     = 8
HUD_ALPHA       = 0.55

LABEL_FONT       = cv2.FONT_HERSHEY_SIMPLEX
LABEL_FONT_SCALE = 0.65
LABEL_THICKNESS  = 2
LABEL_PADDING    = 6

@dataclass
class Detection:
    frame: int
    id: int
    # Stored as (x, y, w, h) — top-left origin, pixel coordinates
    bbox: list[float]
    conf: float
    cls: str = "person"
    # Index of the detections file this came from (0 for the first file, etc.)
    source: int = 0
    # Display name of the detections file this came from (e.g. its filename)
    source_name: str = ""

    @property
    def xyxy(self) -> tuple[int, int, int, int]:
        """Convert (x, y, w, h) → (x1, y1, x2, y2)."""
        x, y, w, h = self.bbox
        return int(x), int(y), int(x + w), int(y + h)

def load_detections(csv_path: Path, source: int = 0, source_name: str = "") -> list[Detection]:
    """
    Read a MOT-style CSV: <frame>,<id>,<bb_left>,<bb_top>,<bb_width>,<bb_height>,<conf>,…
    Returns detections sorted by frame number.

    `source` / `source_name` tag every detection with which input file it came
    from, so multiple detection files can be told apart after merging.
    """
    detections: list[Detection] = []
    with open(csv_path, newline="") as fh:
        for row in csv.reader(fh):
            if not row or row[0].startswith("#"):
                continue
            try:
                det = Detection(
                    frame=int(row[0]),
                    id=int(row[1]),
                    bbox=[float(row[i]) for i in range(2, 6)],
                    conf=float(row[6]),
                    cls="person",
                    source=source,
                    source_name=source_name or csv_path.stem,
                )
                detections.append(det)
            except (IndexError, ValueError) as exc:
                print(f"[WARN] Skipping malformed row {row}: {exc}", file=sys.stderr)

    detections.sort(key=lambda d: d.frame)
    return detections


def load_all_detections(csv_paths: list[Path]) -> list[Detection]:
    """Load and merge detections from one or more CSV files, tagging each
    with its originating file index/name."""
    all_detections: list[Detection] = []
    for idx, path in enumerate(csv_paths):
        all_detections.extend(load_detections(path, source=idx, source_name=path.stem))
    all_detections.sort(key=lambda d: d.frame)
    return all_detections

def build_frame_index(detections: list[Detection]) -> dict[int, list[Detection]]:
    """Group detections by frame into a plain dict for safe .get() access."""
    index: dict[int, list[Detection]] = defaultdict(list)
    for det in detections:
        index[det.frame].append(det)
    return dict(index)

def _track_color(track_id: int) -> tuple[int, int, int]:
    return PALETTE[track_id % len(PALETTE)]

def _source_color(source_idx: int) -> tuple[int, int, int]:
    return PALETTE[source_idx % len(PALETTE)]

def draw_bbox(frame: np.ndarray, det: Detection, color: tuple[int, int, int] | None = None) -> None:
    x1, y1, x2, y2 = det.xyxy
    if color is None:
        color = _track_color(det.id)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    label = f"{det.cls}:{det.id} {det.conf:.2f}"
    (tw, th), baseline = cv2.getTextSize(label, LABEL_FONT, LABEL_FONT_SCALE, LABEL_THICKNESS)
    bg_y1 = max(y1 - th - 2 * LABEL_PADDING, 0)
    bg_y2 = y1
    bg_x2 = x1 + tw + 2 * LABEL_PADDING

    cv2.rectangle(frame, (x1, bg_y1), (bg_x2, bg_y2), color, -1)
    cv2.putText(
        frame, label,
        (x1 + LABEL_PADDING, bg_y2 - LABEL_PADDING // 2 - baseline // 2 + th // 2),
        LABEL_FONT, LABEL_FONT_SCALE, (0, 0, 0), LABEL_THICKNESS, cv2.LINE_AA,
    )

def compute_hud_width(
    total_frames: int,
    class_names: list[str],
    source_legend: list[tuple[str, tuple[int, int, int]]] | None,
) -> int:
    font_main = (LABEL_FONT, 0.65, 1)
    font_small = (LABEL_FONT, 0.55, 1)

    def text_width(text, font):
        (w, _), _ = cv2.getTextSize(text, font[0], font[1], font[2])
        return w

    # Header (worst case with [PAUSED])
    header = f"Frame {total_frames}/{total_frames}  [PAUSED]"
    max_w = text_width(header, font_main)

    # Class counts
    for name in class_names:
        line = f"{name}: 9999"
        max_w = max(max_w, text_width(line, font_main))

    # Legend names
    if source_legend:
        for name, _ in source_legend:
            max_w = max(max_w, text_width(name, font_small) + 20)  # + swatch space

    return max_w + 20  # padding

def draw_hud(
    frame: np.ndarray,
    frame_idx: int,
    total_frames: int,
    paused: bool,
    class_counts: dict[str, int],
    source_legend: list[tuple[str, tuple[int, int, int]]] | None = None,
) -> None:
    HUD_WIDTH = compute_hud_width(
        total_frames,
        list(class_counts.keys()),
        source_legend
    )
    """Top-left semi-transparent overlay with counts and playback info.

    `source_legend`, if given, is a list of (file_name, color) pairs drawn as
    a small colored swatch + label — one line per detections file.
    """
    lines = [f"Frame {frame_idx}/{total_frames}  {'[PAUSED]' if paused else ''}"]
    lines += [f"  {name}: {cnt}" for name, cnt in sorted(class_counts.items())]

    legend_lines = len(source_legend) if source_legend else 0
    total_lines = len(lines) + legend_lines
    height = HUD_PADDING + total_lines * HUD_LINE_HEIGHT + HUD_PADDING
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (HUD_X, HUD_Y),
        (HUD_X + HUD_WIDTH, HUD_Y + height),
        (0, 0, 0), -1,
    )
    cv2.addWeighted(overlay, HUD_ALPHA, frame, 1 - HUD_ALPHA, 0, frame)

    for i, line in enumerate(lines):
        y = HUD_Y + HUD_PADDING + i * HUD_LINE_HEIGHT + HUD_LINE_HEIGHT // 2 + 6
        cv2.putText(
            frame, line, (HUD_X + 8, y),
            LABEL_FONT, 0.65, (0, 255, 200), 1, cv2.LINE_AA,
        )

    if source_legend:
        swatch_size = 14
        for j, (name, color) in enumerate(source_legend):
            y = HUD_Y + HUD_PADDING + (len(lines) + j) * HUD_LINE_HEIGHT + HUD_LINE_HEIGHT // 2 + 6
            sw_y1 = y - swatch_size + 4
            cv2.rectangle(
                frame,
                (HUD_X + 8, sw_y1),
                (HUD_X + 8 + swatch_size, sw_y1 + swatch_size),
                color, -1,
            )
            cv2.putText(
                frame, name, (HUD_X + 8 + swatch_size + 6, y),
                LABEL_FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
            )

class VideoPlayer:
    """Encapsulates video playback, seeking, rendering, and key-event handling."""

    def __init__(
        self,
        cap: cv2.VideoCapture,
        frame_index: dict[int, list[Detection]],
        start_frame: int,
        end_frame: int,
        speed: float = 1.0,
        source_names: list[str] | None = None,
    ):
        self.cap         = cap
        self.frame_index = frame_index
        self.start_frame = start_frame
        self.end_frame   = end_frame
        self.fps         = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.speed       = speed

        # When more than one detections file is loaded, color boxes by which
        # file they came from and show a legend; otherwise fall back to the
        # original per-track-ID coloring.
        self.source_names = source_names or []
        self.multi_source = len(self.source_names) > 1
        self.legend = (
            [(name, _source_color(i)) for i, name in enumerate(self.source_names)]
            if self.multi_source else None
        )

        self.paused        = False
        self.current_frame = start_frame
        self.seen_ids: set[tuple[int, int]] = set()
        self.class_counts: dict[str, int] = {}

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    def _render(self, frame: np.ndarray) -> None:
        """Draw detections and HUD onto `frame` in-place."""
        current_detections = self.frame_index.get(self.current_frame, [])
        for det in current_detections:
            color = _source_color(det.source) if self.multi_source else None
            draw_bbox(frame, det, color=color)

            # Key seen-IDs by (source, id) so the same numeric ID from two
            # different files is counted separately.
            key = (det.source, det.id)
            if key not in self.seen_ids:
                self.seen_ids.add(key)
                count_label = det.source_name if self.multi_source else det.cls
                self.class_counts[count_label] = self.class_counts.get(count_label, 0) + 1

        draw_hud(
            frame,
            self.current_frame - self.start_frame,
            self.end_frame - self.start_frame,
            self.paused,
            self.class_counts,
            source_legend=self.legend,
        )

    def _seek(self, target: int) -> np.ndarray | None:
        """Seek to `target` frame index; returns the decoded frame or None."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = target
        return frame if ret else None

    def run(self) -> None:
        print(f"Frames {self.start_frame}–{self.end_frame}  |  "
              f"{self.fps:.1f} FPS  |  speed ×{self.speed}")
        print("Controls: SPACE=pause  d/s=step  q/ESC=quit")

        frame: np.ndarray|None = None

        while self.cap.isOpened() and self.current_frame <= self.end_frame:
            delay_ms = max(1, int(1000 / (self.fps * self.speed)))

            if not self.paused:
                ret, frame = self.cap.read()
                if not ret:
                    break
                self.current_frame += 1

            if frame is not None:
                self._render(frame)
                cv2.imshow("Playback", frame)

            key = cv2.waitKey(1 if self.paused else delay_ms) & 0xFF

            if key in (ord("q"), 27):
                break

            elif key == ord(" "):
                self.paused = not self.paused

            elif key == ord("d") and self.paused:
                target = self.current_frame + 1
                if target <= self.end_frame:
                    frame = self._seek(target)

            elif key == ord("s") and self.paused:
                target = self.current_frame - 1
                if target >= self.start_frame:
                    frame = self._seek(target)

        self.cap.release()
        cv2.destroyAllWindows()
        print("Playback finished.")



def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Replay video with saved MOT bounding boxes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("-c", "--csv",   nargs="+", default=["runtime/detections/detections.txt"],
                    help="Path(s) to detections CSV (MOT format). Pass more than "
                         "one to overlay several detection sets, each drawn in "
                         "its own color.")
    ap.add_argument("-v", "--video", required=True,
                    help="Source video path")
    ap.add_argument("-s", "--speed", type=float, default=1.0,
                    help="Playback speed multiplier")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    csv_paths = [Path(p) for p in args.csv]
    for p in csv_paths:
        if not p.exists():
            sys.exit(f"[ERROR] CSV not found: {p}")

    vid_path = Path(args.video)
    if not vid_path.exists():
        sys.exit(f"[ERROR] Video not found: {vid_path}")

    detections = load_all_detections(csv_paths)
    if not detections:
        sys.exit("[ERROR] No valid detections found in CSV file(s).")

    if len(csv_paths) > 1:
        print("Loaded detection sources:")
        for i, p in enumerate(csv_paths):
            print(f"  [{i}] {p.name}  ->  color {_source_color(i)}")

    frame_index = build_frame_index(detections)
    start_frame = min(d.frame for d in detections)
    end_frame   = max(d.frame for d in detections)

    cap = cv2.VideoCapture(str(vid_path))
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open video: {vid_path}")

    player = VideoPlayer(
        cap=cap,
        frame_index=frame_index,
        start_frame=start_frame,
        end_frame=end_frame,
        speed=args.speed,
        source_names=[p.stem for p in csv_paths],
    )
    player.run()


if __name__ == "__main__":
    main()
