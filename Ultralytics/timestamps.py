"""
timestamps.py — Interactive video timestamp tool.

Plays back a video and lets the user mark clip start/end points,
annotate each clip with a street code (letter + digit, e.g. A1),
and persist the results to a JSON file.

Controls
--------
S : start a new segment (or close current one immediately)
E : end the current segment
P : pause / resume
B : back 5 seconds
N : forward 5 seconds
Q / ESC : quit (saves any completed segments)
"""

import cv2
import argparse
import pandas as pd
import json
from dataclasses import dataclass, field
from pathlib import Path

KEY_NONE = 255
KEY_ENTER = 13
KEY_SPACE = 32
KEY_ESC = 27
KEY_C = 99
KEY_Q = 113
KEY_P = 112
KEY_N = 110
KEY_B = 98
KEY_S = 115
KEY_E = 101
KEY_BACKSPACE = 8 

ANNOTATION_FORMAT_HELP = "one letter followed by one digit (e.g. A1, B3)"

def is_valid_annotation(text):
    """Return True if *text* matches the required annotation format (e.g. 'A1')."""
    return len(text) == 2 and text[0].isalpha() and text[1].isdigit()

@dataclass
class Clip:
    start: int
    end: int

@dataclass
class PlaybackState:
    timestamps: list[int] = field(default_factory=list)
    annotations: list[str] = field(default_factory=list)
    fps: float = 0
    current_annotation: str = ""
    waiting_for_annotation: bool = False
    is_paused: bool = False
    is_recording: bool = False
    current_frame: int = 0
    last_display: object = None   # last frame (numpy array)

def format_seconds(total_seconds: int) -> str:
    """Convert a second count to a mm:ss string."""
    m = total_seconds // 60
    s = total_seconds % 60
    return f"{m:02d}:{s:02d}"

def format_timestamps(timestamps: list[int], fps : float) -> str:
    """Return a space-separated string of 'mm:ss-mm:ss' segments from a list of frames.

    An unpaired trailing timestamp is rendered as 'mm:ss-??:??' to signal
    an still-open segment.
    """
    segments: list[str] = []
    for i in range(0, len(timestamps), 2):
        start = format_seconds(int(timestamps[i] / fps))
        end = format_seconds(int(timestamps[i + 1] / fps)) if i + 1 < len(timestamps) else "??:??"
        segments.append(f"{start}-{end}")
    return " ".join(segments)

def build_clips(timestamps: list[int]) -> list[Clip]:
    """Return a list of Clip objects from a flat list of start/end seconds.

    Ignores any unclosed timestamps.
    """
    clips: list[Clip] = []
    for i in range(0, len(timestamps) - 1, 2):
        clips.append(Clip(
            start=timestamps[i],
            end=timestamps[i + 1]
        ))
    return clips

def update_csv(
    csv_path: Path,
    annotations: list[str],
    timestamps: list[int],
    source_path: Path,
) -> None:

    filename = source_path.name
    clips = build_clips(timestamps)

    if len(annotations) != len(clips):
        raise ValueError("annotations and clips must match")

    rows = [] 
    for annotation, clip in zip(annotations, clips): 
        rows.append({ 
            "video": filename, 
            "segment": annotation, 
            "start": clip.start, 
            "end": clip.end, }) 
        
    new_df = pd.DataFrame(rows)

    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            df = pd.DataFrame(columns=new_df.columns)
    else:
        df = pd.DataFrame(columns=new_df.columns)

    df = df.reindex(columns=new_df.columns)

    combined = pd.concat([df, new_df], ignore_index=True)

    combined.to_csv(csv_path, index=False)

def update_json(
    json_path: Path,
    annotations: list[str],
    timestamps: list[int],
    source_path: Path,
) -> None:
    """Append or update clip entries in the JSON data file.

    Repeated calls for the same street + filename *append* new clips rather
    than overwriting previous ones.
    """
    filename = source_path.name
    clips = build_clips(timestamps)

    # Load existing data
    if json_path.exists():
        with json_path.open("r") as f:
            try:
                entries: list[dict] = json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: {json_path} is not valid JSON — overwriting.")
                entries = []
    else:
        entries = []

    # Build a lookup by street name for O(1) access
    street_map: dict[str, dict] = {entry["street"]: entry for entry in entries}

    for i, annotation in enumerate(annotations):
        if i >= len(clips):
            break  # guard: more annotations than clips (shouldn't happen)

        clip_dict = {"start": clips[i].start, "end": clips[i].end}

        if annotation not in street_map:
            street_map[annotation] = {"street": annotation, "clips": {}}

        street_clips = street_map[annotation]["clips"]
        if filename not in street_clips:
            street_clips[filename] = []
        street_clips[filename].append(clip_dict)

    entries = list(street_map.values())
    with json_path.open("w") as f:
        json.dump(entries, f, indent=1)

def draw_overlay(
    frame,
    state: PlaybackState,
) -> object:
    """Return a copy of *frame* with the current playback state on top."""
    out = frame.copy()

    # --- Segment timeline strip ---
    if state.timestamps:
        parts = format_timestamps(state.timestamps, state.fps).split()
        labeled = []
        for i, part in enumerate(parts):
            ann = state.annotations[i] if i < len(state.annotations) else None
            labeled.append(f"{part}[{ann}]" if ann else part)
        seg_str = " ".join(labeled)

        (tw, th), _ = cv2.getTextSize(seg_str, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (6, 4), (14 + tw, 14 + th), (20, 20, 20), -1)
        color = (0, 100, 230) if state.is_recording else (0, 230, 100)
        cv2.putText(out, seg_str, (10, 10 + th),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    # --- Paused indicator ---
    if state.is_paused and not state.waiting_for_annotation:
        h, w = out.shape[:2]
        pause_str = "|| PAUSED"
        (pw, ph), _ = cv2.getTextSize(pause_str, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        x = w - pw - 14
        cv2.rectangle(out, (x - 6, 4), (x + pw + 6, 14 + ph), (20, 20, 20), -1)
        cv2.putText(out, pause_str, (x, 10 + ph),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2, cv2.LINE_AA)

    # --- Annotation prompt ---
    if state.waiting_for_annotation:
        prompt = f"Street name : {state.current_annotation}_"
        (pw, ph), _ = cv2.getTextSize(prompt, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        h, w = out.shape[:2]
        x, y = (w - pw) // 2, h // 2
        cv2.rectangle(out, (x - 10, y - ph - 10), (x + pw + 10, y + 10), (20, 20, 20), -1)
        cv2.rectangle(out, (x - 10, y - ph - 10), (x + pw + 10, y + 10), (0, 200, 255), 1)
        cv2.putText(out, prompt, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2, cv2.LINE_AA)

    return out

def handle_annotation_key(key: int, state: PlaybackState) -> None:
    """Process a keypress while the annotation prompt is active."""
    if key == KEY_NONE:
        return
    if key in (KEY_ENTER, 10):  # Enter
        if is_valid_annotation(state.current_annotation):
            state.annotations.append(state.current_annotation.upper())
            state.current_annotation = ""
            state.waiting_for_annotation = False
            state.is_paused = False
    elif key == KEY_BACKSPACE:
        state.current_annotation = state.current_annotation[:-1]
    elif len(state.current_annotation) < 2 and KEY_SPACE <= key <= 126:
        ch = chr(key)
        if len(state.current_annotation) == 0 and ch.isalpha():
            state.current_annotation += ch
        elif len(state.current_annotation) == 1 and ch.isdigit():
            state.current_annotation += ch

def handle_paused_key(key: int, state: PlaybackState, cap, start_frame: int, end_frame: int) -> bool:
    """Process a keypress while paused (but not waiting for annotation).

    Returns True if the caller should quit.
    """
    if key == KEY_NONE:
        return False
    if key in (KEY_Q, KEY_ESC):
        return True
    if key in (KEY_P, KEY_SPACE):
        state.is_paused = False
    elif key == KEY_B:
        new_pos = max(start_frame, state.current_frame - int(5 * state.fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
        state.current_frame = new_pos
        ret, frame = cap.read()
        if ret:
            state.last_display = frame
            state.current_frame += 1
    elif key == KEY_N:
        new_pos = min(end_frame, state.current_frame + int(5 * state.fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
        state.current_frame = new_pos
        ret, frame = cap.read()
        if ret:
            state.last_display = frame
            state.current_frame += 1
    elif key == KEY_S :
        _handle_start_key(state)
    elif key == KEY_E:
        _handle_end_key(state)
    return False

def handle_playback_key(key: int, state: PlaybackState, cap, start_frame: int, end_frame: int) -> bool:
    """Process a keypress during normal playback.

    Returns True if the caller should quit.
    """
    if key == KEY_NONE:
        return False
    if key in (KEY_Q, KEY_ESC):
        return True
    if key in (KEY_P, KEY_SPACE):
        state.is_paused = True
    elif key == KEY_B:
        new_pos = max(start_frame, state.current_frame - int(5 * state.fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
        state.current_frame = new_pos
    elif key == KEY_N:
        new_pos = min(end_frame, state.current_frame + int(5 * state.fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
        state.current_frame = new_pos
    elif key == KEY_S :
        _handle_start_key(state)
    elif key == KEY_E:
        _handle_end_key(state)
    elif key == KEY_C:
        _handle_delete_key(state)
    return False

def _handle_start_key(state: PlaybackState) -> None:
    """S : start a new segment, or close-and-reopen the current one."""
    t = state.current_frame
    if state.is_recording:
        # Close current segment immediately and open a new annotation prompt
        state.timestamps.append(t)
        state.timestamps.append(t)
        state.is_recording = False
        state.waiting_for_annotation = True
        state.is_paused = True
    else:
        state.is_recording = True
        state.timestamps.append(t)

def _handle_end_key(state: PlaybackState) -> None:
    """E: end the current segment."""
    t = state.current_frame
    if state.is_recording:
        state.is_recording = False
        state.timestamps.append(t)
        state.waiting_for_annotation = True
        state.is_paused = True
    else:
        if state.timestamps:
            state.timestamps.append(state.timestamps[-1])
            state.timestamps.append(t)
            state.waiting_for_annotation = True
            state.is_paused = True

def _handle_delete_key(state: PlaybackState) -> None :
    if state.timestamps:
        state.timestamps.pop()
        if len(state.timestamps) % 2 == 1:
            state.annotations.pop()
            state.is_recording = True
        else :
            state.is_recording = False
        
def collect_missing_annotations(state: PlaybackState) -> None:
    """Prompt the user in the terminal for any clips that still lack annotations."""
    while len(state.annotations) < len(state.timestamps) // 2:
        clip_num = len(state.annotations) + 1
        raw = input(f"Annotation for clip {clip_num} ({ANNOTATION_FORMAT_HELP}): ").strip()
        if is_valid_annotation(raw):
            state.annotations.append(raw.upper())
        else:
            print(f"Invalid : must be {ANNOTATION_FORMAT_HELP}.")

def finalize_timestamps(
    state: PlaybackState,
    source_path: Path,
    output_path: Path,
    format : str
) -> list | str:
    """Close any open segment, collect missing annotations, save, and return.

    *fps* is required to convert the stored `current_frame` value into seconds
    when closing a trailing open segment.
    """
    if not state.timestamps:
        print("\nNo timestamps registered.")
        return []

    if len(state.timestamps) % 2 == 1:
        state.timestamps.append(state.current_frame)

    collect_missing_annotations(state)

    result = format_timestamps(state.timestamps, state.fps)
    print("\nTimestamps :", result)

    if format == "json" :
        update_json(output_path, state.annotations, state.timestamps, source_path)
    else : 
        update_csv(output_path, state.annotations, state.timestamps, source_path)

    print(f"Saved at {output_path} in {format}.")
    return result

def get_timestamps(source_path: Path, output_path: Path, format: str, speed: float) -> list[str] | str:
    """Open *source_path*, let the user mark clips, and save results to *output_path*.

    Returns the formatted timestamp string on success, or an empty list if
    no timestamps were registered.
    """
    state = PlaybackState()
    start_frame = 0

    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {source_path}")

    state.fps = cap.get(cv2.CAP_PROP_FPS) or 30
    end_frame  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    state.current_frame = start_frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    print(f"Playing back {source_path}")
    print("Controls: S/Space=start  E=end  P=pause/play  B=back 5s  N=forward 5s  Q/ESC=quit")

    quit_requested = False

    delay_ms = max(1, int(1000 / (state.fps * speed)))

    while cap.isOpened() and state.current_frame < end_frame:

        # Paused or waiting for annotation
        if state.is_paused or state.waiting_for_annotation:
            if state.last_display is not None:
                display = draw_overlay(state.last_display, state)
                cv2.imshow("Playback", display) # type: ignore

            key = cv2.waitKey(30) & 0xFF

            if state.waiting_for_annotation:
                handle_annotation_key(key, state)
            else:
                quit_requested = handle_paused_key(
                    key, state, cap, start_frame, end_frame
                )
                if quit_requested:
                    break
            continue

        # Normal playback
        ret, frame = cap.read()
        if not ret:
            break
        state.current_frame += 1
        state.last_display = frame

        display = draw_overlay(frame, state)
        cv2.imshow("Playback", display) # type: ignore


        key = cv2.waitKey(delay_ms) & 0xFF
        quit_requested = handle_playback_key(
            key, state, cap, start_frame, end_frame
        )
        if quit_requested:
            break

    cap.release()
    cv2.destroyAllWindows()


    return finalize_timestamps(state, source_path, output_path, format)

def main() -> None:
    ap = argparse.ArgumentParser(description="Interactive video timestamp marker.")
    ap.add_argument("-v", "--video", default=None, required=True, help="Source video path")
    ap.add_argument("-o", "--output", default="data/timestamps.csv", help="Output file path")
    ap.add_argument("-f", "--format", default="csv", choices=["csv", "json"], help="Output format")
    ap.add_argument("-s", "--speed", default=1.5, type=float, help="Playback speed")

    args = ap.parse_args()
    source_path = Path(args.video)
    ext = f".{args.format}"
    output_path = Path(args.output).with_suffix(ext)
    speed = args.speed

    if speed <= 0 : 
        raise ValueError(f"Cannot have non positive speed")

    get_timestamps(source_path, output_path, args.format, speed)

if __name__ == "__main__":
    main()
    