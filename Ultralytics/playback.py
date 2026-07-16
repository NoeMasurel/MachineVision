"""
playback.py — Replay a video with bounding boxes from detections CSV.

Usage
    python playback.py -v path/to/video.mp4
    python playback.py -v path/to/video.mp4 -c detections.txt
    python playback.py -v path/to/video.mp4 -c detections.txt --speed 2.0

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
HUD_WIDTH       = 230
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

    @property
    def xyxy(self) -> tuple[int, int, int, int]:
        """Convert (x, y, w, h) → (x1, y1, x2, y2)."""
        x, y, w, h = self.bbox
        return int(x), int(y), int(x + w), int(y + h)

def load_detections(csv_path: Path) -> list[Detection]:
    """
    Read a MOT-style CSV: <frame>,<id>,<bb_left>,<bb_top>,<bb_width>,<bb_height>,<conf>,…
    Returns detections sorted by frame number.
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
                )
                detections.append(det)
            except (IndexError, ValueError) as exc:
                print(f"[WARN] Skipping malformed row {row}: {exc}", file=sys.stderr)

    detections.sort(key=lambda d: d.frame)
    return detections

def build_frame_index(detections: list[Detection]) -> dict[int, list[Detection]]:
    """Group detections by frame into a plain dict for safe .get() access."""
    index: dict[int, list[Detection]] = defaultdict(list)
    for det in detections:
        index[det.frame].append(det)
    return dict(index)

def _track_color(track_id: int) -> tuple[int, int, int]:
    return PALETTE[track_id % len(PALETTE)]

def draw_bbox(frame: np.ndarray, det: Detection) -> None:
    x1, y1, x2, y2 = det.xyxy
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

def draw_hud(
    frame: np.ndarray,
    frame_idx: int,
    total_frames: int,
    paused: bool,
    class_counts: dict[str, int],
) -> None:
    """Top-left semi-transparent overlay with counts and playback info."""
    lines = [f"Frame {frame_idx}/{total_frames}  {'[PAUSED]' if paused else ''}"]
    lines += [f"  {name}: {cnt}" for name, cnt in sorted(class_counts.items())]

    height = HUD_PADDING + len(lines) * HUD_LINE_HEIGHT + HUD_PADDING
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

class VideoPlayer:
    """Encapsulates video playback, seeking, rendering, and key-event handling."""

    def __init__(
        self,
        cap: cv2.VideoCapture,
        frame_index: dict[int, list[Detection]],
        start_frame: int,
        end_frame: int,
        speed: float = 1.0,
    ):
        self.cap         = cap
        self.frame_index = frame_index
        self.start_frame = start_frame
        self.end_frame   = end_frame
        self.fps         = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.speed       = speed

        self.paused        = False
        self.current_frame = start_frame
        self.seen_ids: set[int]        = set()
        self.class_counts: dict[str, int] = {}

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    def _render(self, frame: np.ndarray) -> None:
        """Draw detections and HUD onto `frame` in-place."""
        current_detections = self.frame_index.get(self.current_frame, [])
        for det in current_detections:
            draw_bbox(frame, det)
            if det.id not in self.seen_ids:
                self.seen_ids.add(det.id)
                self.class_counts[det.cls] = self.class_counts.get(det.cls, 0) + 1

        draw_hud(
            frame,
            self.current_frame - self.start_frame,
            self.end_frame - self.start_frame,
            self.paused,
            self.class_counts,
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



def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Replay video with saved MOT bounding boxes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("-c", "--csv",   default="data/detections.txt",
                    help="Path to detections CSV (MOT format)")
    ap.add_argument("-v", "--video", required=True,
                    help="Source video path")
    ap.add_argument("-s", "--speed", type=float, default=1.0,
                    help="Playback speed multiplier")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        sys.exit(f"[ERROR] CSV not found: {csv_path}")

    vid_path = Path(args.video)
    if not vid_path.exists():
        sys.exit(f"[ERROR] Video not found: {vid_path}")

    detections = load_detections(csv_path)
    if not detections:
        sys.exit("[ERROR] No valid detections found in CSV.")

    frame_index = build_frame_index(detections)
    start_frame = detections[0].frame
    end_frame   = detections[-1].frame

    cap = cv2.VideoCapture(str(vid_path))
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open video: {vid_path}")

    player = VideoPlayer(
        cap=cap,
        frame_index=frame_index,
        start_frame=start_frame,
        end_frame=end_frame,
        speed=args.speed,
    )
    player.run()


if __name__ == "__main__":
    main()