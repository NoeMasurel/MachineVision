import subprocess
import sys

for confidence in range(2, 10, 1):
    conf=str(confidence/10)
    result = subprocess.run(
        [sys.executable, "-m", "tracking.evaluation.compile","--video", "/home/noe/Research/Data/GX011504.MP4", "--gt", "Data/GroundTruth/Vid5/gt.txt", "--confidence",conf],
        capture_output=True,
        text=True
    )
    print(result.stdout)
    print(result.stderr)  
