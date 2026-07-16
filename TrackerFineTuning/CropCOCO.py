import cv2
import os
import json
from collections import defaultdict

def extract_reid_crops_coco(video_path, coco_json_path, output_dir, min_crops_per_id=4, pad=10, max_ids=None):
    """
    Extracts ReID crops from a video using CVAT COCO 1.0 export.
    Organizes crops by category name + track_id.
    """
    with open(coco_json_path, 'r') as f:
        coco = json.load(f)

    # Map image_id -> video frame number
    image_id_to_frame = {}
    for img in coco['images']:
        fname = img['file_name']  # e.g. "frame_000160.png"
        frame_num = int(fname.replace('frame_', '').split('.')[0])
        image_id_to_frame[img['id']] = frame_num

    # Map category_id -> category name
    category_id_to_name = {c['id']: c['name'] for c in coco['categories']}

    # Group annotations by (category_name, track_id)
    tracks = defaultdict(list)
    for ann in coco['annotations']:
        track_id = ann['attributes']['track_id']
        category_name = category_id_to_name[ann['category_id']]
        frame_num = image_id_to_frame[ann['image_id']]
        tracks[(category_name, track_id)].append({
            'frame': frame_num,
            'bbox': ann['bbox'],
            'occluded' : ann['attributes']['occluded']
        })

    # Filter by min crops and optionally limit number of IDs
    track_keys = [(cat, tid) for (cat, tid), anns in tracks.items() if len(anns) >= min_crops_per_id]
    track_keys = sorted(track_keys)
    if max_ids is not None:
        track_keys = track_keys[:max_ids]

    print(f"Found {len(tracks)} total tracks, processing {len(track_keys)} after filters")
    for cat in category_id_to_name.values():
        count = sum(1 for (c, _) in track_keys if c == cat)
        print(f"  {cat}: {count} tracks")

    # Cache only the frames we actually need (frames without annotation)
    needed_frames = set()
    for key in track_keys:
        for ann in tracks[key]:
            needed_frames.add(ann['frame'])

    print(f"Loading {len(needed_frames)} unique frames from video...")
    cap = cv2.VideoCapture(video_path)
    frames = {}
    for frame_num in sorted(needed_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if ret:
            frames[frame_num] = frame
        else:
            print(f"  Warning: could not read frame {frame_num}")
    cap.release()

    # Extract crops — organized as output_dir/CategoryName/person_XXXX/
    os.makedirs(output_dir, exist_ok=True)
    for (category_name, tid) in track_keys:
        person_dir = os.path.join(output_dir, category_name, f'person_{tid:04d}')
        os.makedirs(person_dir, exist_ok=True)

        for ann in tracks[(category_name, tid)]:
            frame = frames.get(ann['frame'])
            if frame is None:
                continue

            x, y, w, h = ann['bbox']
            x1 = max(0, int(x) - pad)
            y1 = max(0, int(y) - pad)
            x2 = min(frame.shape[1], int(x + w) + pad)
            y2 = min(frame.shape[0], int(y + h) + pad)

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            occluded_tag = "_occ" if ann['occluded'] else ""
            name = f'frame_{ann["frame"]:06d}{occluded_tag}.jpg'

            crop_path = os.path.join(person_dir, name)
            cv2.imwrite(crop_path, crop)

    print(f"Done. Extracted crops to {output_dir}/")


import argparse
from pathlib import Path

# assuming your function is already imported or defined above
# from your_module import extract_reid_crops_coco

def main():
    parser = argparse.ArgumentParser(description="Extract ReID crops from COCO + video")

    parser.add_argument('-v', "--video", type=str, required=True, help="Path to input video (e.g. GX011504.MP4)")
    parser.add_argument('-j', "--json",type=str, required=True, help="Path to COCO annotations JSON (e.g. instances_default.json)")
    parser.add_argument('-o',"--output",type=str, default="reid_dataset/raw", help="Output directory for crops")
    parser.add_argument("-m", "--max_ids", type=int, default=None, help="Max number of IDs to process")
    args = parser.parse_args()

    extract_reid_crops_coco(
        video_path=args.video,
        coco_json_path=args.json,
        output_dir=args.output,
        min_crops_per_id=5,
        pad=10,
        max_ids=args.max_ids
    )

if __name__ == "__main__":
    main()