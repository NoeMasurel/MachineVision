import ffmpeg
import argparse
from pathlib import Path
import pandas as pd

"""
USAGE :

    --video : specify the input path
    --timestamps : specify the timestamps file (csv with columns: video,segment,start,end)

"""

def get_fps(input_file):
    probe = ffmpeg.probe(str(input_file))
    video_stream = next(
        s for s in probe["streams"] if s["codec_type"] == "video"
    )
    num, den = video_stream["r_frame_rate"].split("/")
    return int(num) / int(den)

def frames_to_seconds(frame, fps):
    return frame / fps

def format_seconds(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02}:{m:02}:{s:02}" if h else f"{m:02}:{s:02}"

def prompt_user(segment_name, start_frame, end_frame, start_sec, end_sec):
    duration_sec = end_sec - start_sec
    print(
        f"\n  Segment : {segment_name}\n"
        f"  Start   : frame {start_frame}  ({format_seconds(start_sec)})\n"
        f"  End     : frame {end_frame}  ({format_seconds(end_sec)})\n"
        f"  Duration: {end_frame - start_frame} frames  ({format_seconds(duration_sec)})\n"
    )
    while True:
        answer = input("  Extract this segment? [y/n/q to quit] : ").strip().lower()
        if answer in ("y", "n", "q"):
            return answer
        print("  Please enter y, n, or q.")

def extract_clips(input_file, segments, fps):
    p = Path(input_file)
    extracted = 0
    skipped = 0

    for i, (segment_name, start_frame, end_frame) in enumerate(segments):
        start_sec = frames_to_seconds(start_frame, fps)
        end_sec   = frames_to_seconds(end_frame, fps)

        choice = prompt_user(segment_name, start_frame, end_frame, start_sec, end_sec)

        if choice == "q":
            print("\nQuitting early.")
            break

        if choice == "n":
            print("  → Skipped.")
            skipped += 1
            continue

        # choice == "y"
        try:
            duration_sec = end_sec - start_sec
            if duration_sec <= 0:
                raise ValueError(f"Bad segment: start={start_frame}, end={end_frame}")

            safe_name = str(segment_name).replace(" ", "_")
            output = str(p.with_name(f"{p.stem}_{safe_name}_{i}.mp4"))

            (
                ffmpeg
                .input(str(input_file), ss=start_sec, t=duration_sec)
                .output(output, vcodec="copy", acodec="copy")
                .run(overwrite_output=True)
            )

            print(f"  → Created: {output}")
            extracted += 1

        except Exception as e:
            print(f"  → Error extracting segment {i}: {e}")
            skipped += 1

    print(f"\nDone. {extracted} extracted, {skipped} skipped.")

def validate_file(path_str, label):
    path = Path(path_str)

    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")

    if not path.is_file():
        raise ValueError(f"{label} is not a file: {path}")

    return path

def main():
    ap = argparse.ArgumentParser(description="Cut video into clips using timestamps")

    ap.add_argument("-v", "--video", required=True, help="Source video path")
    ap.add_argument("-ts", "--timestamps", required=True, help="Timestamps CSV file")

    args = ap.parse_args()

    try:
        input_file = validate_file(args.video, "Video file")
    except Exception as e:
        print(f"Error: {e}")
        return

    try:
        tt_file = validate_file(args.timestamps, "Timestamps file")
        if tt_file.suffix != ".csv":
            raise TypeError("Timestamps file must be a .csv")
    except Exception as e:
        print(f"Error: {e}")
        return

    df = pd.read_csv(tt_file)
    df = df[df["video"] == input_file.name]

    if df.empty:
        print(f"No segments found for '{input_file.name}' in {tt_file}.")
        return

    try:
        fps = get_fps(input_file)
        print(f"\nDetected FPS: {fps:.4f}")
    except Exception as e:
        print(f"Error reading FPS from video: {e}")
        return

    segments = list(zip(df["segment"], df["start"], df["end"]))
    print(f"Found {len(segments)} segment(s) for '{input_file.name}'.")

    extract_clips(input_file, segments, fps)

if __name__ == "__main__":
    main()