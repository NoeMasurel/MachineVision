#!/usr/bin/env python3
"""
Draw COCO-format bounding boxes and play the video live in an OpenCV window.

Usage:
    python play_bboxes.py --json annotations.json --video input.mp4

    # Or from a folder of frames:
    python play_bboxes.py --json annotations.json --frames ./frames/

Controls:
    SPACE  - pause / resume
    Q      - quit
    LEFT   - step back 10 frames (video mode only)
    RIGHT  - step forward 10 frames (video mode only)
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import cv2
except ImportError:
    sys.exit("opencv-python is required. Install it with:  pip install opencv-python")

# --- Visual config ---
BOX_COLOR      = (0, 255, 0)
TEXT_COLOR     = (0, 0, 0)
BOX_THICKNESS  = 2
FONT           = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE     = 0.5
FONT_THICKNESS = 1


def load_coco(json_path: str) -> tuple[dict, dict, dict]:
    with open(json_path) as f:
        data = json.load(f)
    images      = {img["id"]: img for img in data.get("images", [])}
    ann_by_image: dict[int, list] = {}
    for ann in data.get("annotations", []):
        ann_by_image.setdefault(ann["image_id"], []).append(ann)
    categories  = {cat["id"]: cat["name"] for cat in data.get("categories", [])}
    return images, ann_by_image, categories


def draw_frame(frame, annotations: list, categories: dict) -> None:
    for ann in annotations:
        x, y, w, h = ann["bbox"]
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)

        cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)

        cat_name = categories.get(ann.get("category_id", -1), "obj")
        track_id = ann.get("attributes", {}).get("track_id")
        label    = f"{cat_name}" + (f" #{track_id}" if track_id is not None else "")

        (tw, th), baseline = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS)
        cv2.rectangle(frame, (x1, y1 - th - baseline - 4), (x1 + tw + 2, y1), BOX_COLOR, -1)
        cv2.putText(frame, label, (x1 + 1, y1 - baseline - 2),
                    FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)


def overlay_hud(frame, frame_num: int, paused: bool) -> None:
    """Draw frame counter and pause indicator."""
    status = "PAUSED" if paused else "PLAYING"
    text   = f"Frame {frame_num}  [{status}]  Q=quit  SPACE=pause  <>/>=seek"
    cv2.putText(frame, text, (10, 24), FONT, 0.55, (0, 0, 0),     2, cv2.LINE_AA)
    cv2.putText(frame, text, (10, 24), FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Video mode
# ---------------------------------------------------------------------------

def play_video(json_path: str, video_path: str) -> None:
    images, ann_by_image, categories = load_coco(json_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    delay = max(1, int(1000 / fps))   # ms per frame

    def frame_index(img_meta: dict):
        stem  = Path(img_meta["file_name"]).stem
        parts = stem.split("_")
        for p in reversed(parts):
            if p.isdigit():
                return int(p)
        return None

    index_to_image_id = {}
    for img_id, img_meta in images.items():
        idx = frame_index(img_meta)
        if idx is not None:
            index_to_image_id[idx] = img_id

    cv2.namedWindow("BBox Viewer", cv2.WINDOW_NORMAL)
    paused      = False
    frame_number = 0

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break

            image_id = index_to_image_id.get(frame_number)
            if image_id is not None:
                draw_frame(frame, ann_by_image.get(image_id, []), categories)

            overlay_hud(frame, frame_number, paused)
            cv2.imshow("BBox Viewer", frame)
            frame_number += 1

        key = cv2.waitKey(1 if paused else delay) & 0xFF

        if key == ord("q"):
            break
        elif key == ord(" "):
            paused = not paused
        elif key == 83:   # RIGHT arrow — seek forward 10 frames
            target = min(frame_number + 10, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            frame_number = target
        elif key == 81:   # LEFT arrow — seek back 10 frames
            target = max(frame_number - 10, 0)
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            frame_number = target

    cap.release()
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Frames folder mode
# ---------------------------------------------------------------------------

def play_frames(json_path: str, frames_dir: str) -> None:
    images, ann_by_image, categories = load_coco(json_path)

    frames_path = Path(frames_dir)
    exts        = {".png", ".jpg", ".jpeg", ".bmp"}
    frame_files = sorted(p for p in frames_path.iterdir() if p.suffix.lower() in exts)

    if not frame_files:
        sys.exit(f"No image files found in: {frames_dir}")

    name_to_image_id = {img["file_name"]: img_id for img_id, img in images.items()}

    cv2.namedWindow("BBox Viewer", cv2.WINDOW_NORMAL)
    paused = False
    i      = 0
    delay  = 33   # ~30 fps

    while i < len(frame_files):
        if not paused:
            fp    = frame_files[i]
            frame = cv2.imread(str(fp))
            if frame is None:
                i += 1
                continue

            image_id = name_to_image_id.get(fp.name)
            if image_id is not None:
                draw_frame(frame, ann_by_image.get(image_id, []), categories)

            overlay_hud(frame, i, paused)
            cv2.imshow("BBox Viewer", frame)
            i += 1

        key = cv2.waitKey(1 if paused else delay) & 0xFF

        if key == ord("q"):
            break
        elif key == ord(" "):
            paused = not paused
        elif key == 83:   # RIGHT — skip forward 10
            i = min(i + 10, len(frame_files) - 1)
        elif key == 81:   # LEFT — skip back 10
            i = max(i - 10, 0)

    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Play video with COCO bounding boxes drawn live."
    )
    parser.add_argument("-j", "--json",   required=True, help="Path to COCO JSON annotation file")
    parser.add_argument("-v", "--video",  default=None,  help="Input video file")
    parser.add_argument("-f", "--frames", default=None,  help="Folder of frame images")
    args = parser.parse_args()

    if not args.video and not args.frames:
        parser.error("Provide either --video or --frames.")

    if args.video:
        play_video(args.json, args.video)
    else:
        play_frames(args.json, args.frames)


if __name__ == "__main__":
    main()