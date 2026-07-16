from MotMetrics.MotScore import mot_score
from Ultralytics.saving_bboxes import ObjectTracking
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="Compute MOTA / MOTChallenge metrics from a GT and prediction file."
    )
    parser.add_argument("--gt",         required=True,
                        help="Path to ground-truth file "
                             "(CVAT MOT format: frame,id,x,y,w,h,conf,class,visibility)")
    parser.add_argument("--pred",       required=True,
                        help="Path to predictions file "
                             "(standard MOT format: frame,id,x,y,w,h,conf,x,y,z)")
    parser.add_argument("--video", type=str, default=None,
                        help="Video path")
    args = parser.parse_args()

    VIDEO_PATH = args.video
    GT_PATH = args.gt
    PRED_PATH = args.pred


    models = ["yolo26n.pt", "yolo26s.pt", "yolo26m.pt", "yolo26l.pt", "yolo26x.pt", ]

    for model in models:
        tracker = ObjectTracking(source=VIDEO_PATH, model=model, display=False, gt = GT_PATH )
        tracker.run(output_mot=PRED_PATH)
        mota = mot_score(GT_PATH, 0.25, PRED_PATH)
        print(f"\n MOTA  = {mota * 100:.2f} % with model : {model}") # type: ignore

if __name__ == "__main__":
    main()