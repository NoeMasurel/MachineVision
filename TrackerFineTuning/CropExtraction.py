import cv2
import os
import pandas as pd

def extract_reid_crops(video_path, mot_annotation_path, output_dir, min_crops_per_id=4, frame_offset=0):
    """
    MOT format: frame, id, x, y, w, h, conf, -1, -1, -1
    
    frame_offset: if your annotation starts at video frame 165, set frame_offset=164
                  (offset is 0-indexed: MOT frame 1 → video frame 1+164=165)
    """
    df = pd.read_csv(mot_annotation_path, header=None,
                     names=['frame','id','x','y','w','h','conf','_1','_2','_3'])

    # Shift MOT frame numbers to actual video frame positions
    df['video_frame'] = df['frame'] + frame_offset

    df_test = df.head(100)
    cap = cv2.VideoCapture(video_path)
    os.makedirs(output_dir, exist_ok=True)

    # Cache frames using the real video frame indices
    frames = {}
    for video_frame_id in df_test['video_frame'].unique():
        cap.set(cv2.CAP_PROP_POS_FRAMES, video_frame_id - 1)  # cv2 is 0-indexed
        ret, frame = cap.read()
        if ret:
            frames[video_frame_id] = frame
    cap.release()

    # Extract crops per person ID
    for person_id, group in df.groupby('id'):
        if len(group) < min_crops_per_id:
            continue

        person_dir = os.path.join(output_dir, f'person_{person_id:04d}')
        os.makedirs(person_dir, exist_ok=True)

        for _, row in group.iterrows():
            frame = frames.get(int(row['video_frame']))
            if frame is None:
                continue

            x, y, w, h = int(row['x']), int(row['y']), int(row['w']), int(row['h'])

            pad = 10
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(frame.shape[1], x + w + pad)
            y2 = min(frame.shape[0], y + h + pad)

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # Filename uses the MOT frame number (annotation-relative) for clarity
            crop_path = os.path.join(person_dir, f'frame_{int(row["frame"]):06d}.jpg')
            cv2.imwrite(crop_path, crop)
            print("")

    print(f"Done. Extracted crops to {output_dir}/")

def main():
    extract_reid_crops(
        video_path='GX011504.mp4',
        mot_annotation_path='504_annotations/gt.txt',
        output_dir='reid_dataset/raw',
        frame_offset=164   # MOT frame 1 = video frame 165, so offset = 165 - 1 = 164
    )

if __name__ == "__main__":
    main()