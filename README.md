# stage-tracking

An installable Python package for building, evaluating, and tuning multi-object
tracking (MOT) pipelines with Ultralytics models and MOT evaluation metrics.

The project is organized around three workflows, each with a public API
function and a console script:

| Workflow                        | API                                             | Console script    |
|----------------------------------|--------------------------------------------------|-------------------|
| Run a tracker over a video       | `stage_tracking.tracking.run_tracking`            | `stage-track`     |
| Evaluate predictions vs. ground truth | `stage_tracking.evaluation.evaluate_predictions` | `stage-evaluate`  |
| Search tracker/detection params with NOMAD | `stage_tracking.optimization.optimize_parameters` | `stage-optimize`  |

## Repository layout

```text
src/stage_tracking/
├── config/        # DatasetConfig / TrackingConfig / EvaluationConfig / OptimizationConfig
│                   dataclasses, and YAML loading (config/loading.py)
├── tracking/       # tracker_config.py (YAML config builder) + pipeline.py
│                   (ObjectTracking, run_tracking, run_tracking_sweep)
├── evaluation/     # metrics.py (TrackEval wrappers: HOTA/CLEAR/Identity)
│                   + compile.py (evaluate_predictions, save_csv, sweep-and-score CLI)
├── optimization/   # nomad_runner.py (optimize_parameters, NOMAD search loop)
├── video/          # standalone video utilities: playback.py, cut.py, timestamps.py
└── cli/            # console-script entry points (thin delegates to each module's main())

examples/           # standalone scripts, not part of the installable package
TrackerFineTuning/   # standalone COCO-crop/annotation utilities (unrelated to the core pipeline)
Outdated/            # legacy/superseded implementations, kept for reference only
```

Data and generated artifacts live outside the package and are not installed
with it:

```text
Data/                       # GroundTruth/<vid>/gt.txt, Prediction/<vid>/*.txt + metrics CSVs
Results/                    # optimizer JSON results (nomad_<tracker>_<timestamp>.json)
models/                     # YOLO .pt weights
videos/                     # source video clips
runtime/                    # generated caches/configs: tracker_base_cache/, tracker_configs/, detections/
```

## Installation

```bash
pip install -e .
```

This installs the `stage_tracking` package (from `src/`) in editable mode
plus its dependencies (numpy, pandas, opencv-python, PyYAML, ultralytics,
motmetrics, trackeval, PyNomadBBO, ffmpeg-python), and registers the
`stage-track` / `stage-evaluate` / `stage-optimize` console scripts.

If you use the video cutting/preprocessing utilities in `stage_tracking.video`,
ensure FFmpeg is installed and available on your system PATH.

## Usage

### 1. Run tracking and evaluate results

The typical workflow is:

1. Provide a ground-truth file for a video sequence.
2. Run a tracker over the video to generate a MOT-style prediction file.
3. Evaluate the generated prediction file against the ground truth using
   TrackEval metrics.

```bash
stage-evaluate --gt Data/GroundTruth/Vid1/gt.txt --video path/to/video.mp4 --models m --confidence 0.25
```

This command will:

* resolve the prediction output folder from the ground-truth path
* run the tracker for the requested model/confidence pair
* save one or more prediction files in the resolved prediction folder
* write metrics outputs to CSV files such as `metrics_results.csv`

### 2. Evaluate existing predictions

If predictions already exist, you can skip the tracking step and evaluate
them directly:

```bash
stage-evaluate --gt Data/GroundTruth/Vid1/gt.txt
```

### 3. Generate tracker YAML configurations

`stage_tracking.tracking.build_tracker_config` builds tracker configuration
files programmatically:

```bash
python -m stage_tracking.tracking.tracker_config tracktrack --track_high_thresh 0.6 --track_buffer 45
```

This is useful when you want to create custom tracker parameter sets without
hand-editing YAML files.

### 4. Optimize tracker parameters with NOMAD

`stage-optimize` uses a YAML configuration and NOMAD to search over tracker
and detection parameters:

```bash
stage-optimize --config opti_config.yaml
```

The optimizer will:

* load the configuration from the YAML file into an `OptimizationConfig`
* run the tracker repeatedly at different parameter points
* score each output with the selected metric
* save JSON results and merge evaluation rows into CSV outputs

### Using the public API directly

```python
from pathlib import Path
from stage_tracking.config import TrackingConfig, EvaluationConfig
from stage_tracking.tracking import run_tracking
from stage_tracking.evaluation import evaluate_predictions

tracking_config = TrackingConfig(
    video_path=Path("videos/clip.mp4"),
    model="models/yolo26n.pt",
    output_path=Path("runtime/detections/clip.txt"),
    confidence=0.25,
)
run_tracking(tracking_config)

rows = evaluate_predictions(EvaluationConfig(
    gt_path=Path("Data/GroundTruth/Vid1/gt.txt"),
))
```

## Project configuration

### Optimizer configuration

The optimizer is driven by a YAML file (see `opti_config.yaml` at the repo
root for a working example) loaded into an `OptimizationConfig` via
`stage_tracking.config.load_optimization_config`. It defines:

* the input video and ground-truth path (`video_path`, `gt_path`)
* model list for the search (`models`)
* the metric to optimize (`metric`)
* the search-space dimensions (`search_space`)
* the starting point for NOMAD (`x0_raw`)
* tracker override defaults (`tracker_override_defaults`)
* where results are written (`results_dir`)

Relative paths inside the YAML are resolved relative to the config file's
own directory, so a config + its data can be moved together as a
self-contained project folder. You can override individual values from the
command line (e.g. `--metric idf1 --n-best 10`); the YAML file remains the
primary source of configuration.

## Data layout expectations

The evaluation and optimization scripts assume a structure similar to the
folders already present in this repository:

```bash
Data/
  GroundTruth/
    Vid1/
      gt.txt
  Prediction/
    Vid1/
      metrics_results.csv
```

The compile pipeline resolves the prediction folder from the path of the
ground-truth file when possible. This makes it easier to keep data organized
by video ID.

## Outputs

Depending on the workflow, the repository can produce:

* MOT-format prediction text files in prediction folders
* CSV metric tables such as `metrics_results.csv` and `metrics_results_occluded.csv`
* JSON result files in `Results/` for optimizer runs
* tracker YAML files under `runtime/tracker_configs/`

## Tests

```bash
pytest
```

Tests stub out the heavy runtime dependencies (`PyNomad`, `trackeval`,
`ultralytics`, `cv2`) in `tests/conftest.py`, so they run without a GPU,
network access, or the real model weights.

## Notes and limitations

* This repository is primarily intended for research and experimentation.
* `examples/` holds standalone scripts (parameter sweeps, correlation
  analysis) that call the package but aren't part of its installable API.
* `TrackerFineTuning/` and `Outdated/` are standalone/legacy code, unrelated
  to the tracking → evaluation → optimization pipeline, and are not packaged.
