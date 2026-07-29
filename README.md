# MachineVision Tracking Toolkit

This repository contains a collection of scripts and utilities for building, evaluating, and tuning multi-object tracking (MOT) pipelines with Ultralytics models and MOT evaluation metrics.

The project is organized around three main workflows:

- Track and evaluate predictions against ground truth using MOT metrics.
- Generate tracker configuration YAML files for Ultralytics trackers.
- Optimize tracker and detection parameters with NOMAD to improve tracking performance.

## Repository layout

- `TrackEval/` – wrappers around the TrackEval library for HOTA / CLEAR / Identity scoring.
- `MotMetrics/` – alternatives based on `motmetrics` for MOTChallenge-style evaluation.
- `Ultralytics/` – tracker config generation, video slicing, playback, and object tracking helpers.
- `TrackerFineTuning/` – utilities for extracting crops and preparing REID-style training data.
- `NOMAD/` – parameter search and optimization scripts.

## Features

- Run tracking inference with Ultralytics models and save MOT-format outputs.
- Evaluate predictions with HOTA, MOTA, MOTP, and IDF1 metrics.
- Build tracker YAML configurations programmatically.
- Sweep model / confidence / tracker parameters and compare results.
- Optimize tracker hyperparameters with NOMAD.

## Main workflows

### 1. Run tracking and evaluate results

The typical workflow is:

1. Provide a ground-truth file for a video sequence.
2. Run a tracker over the video to generate a MOT-style prediction file.
3. Evaluate the generated prediction file against the ground truth using TrackEval metrics.

This is handled primarily by the compile/evaluation pipeline in [TrackEval/Compile.py](TrackEval/Compile.py).

Example:

```bash
python -m TrackEval.Compile --gt Data/GroundTruth/Vid1/gt.txt --video path/to/video.mp4 --models m --confidence 0.25
```
This command will:

* resolve the prediction output folder from the ground-truth path
* run the tracker for the requested model/confidence pair
* save one or more prediction files in the resolved prediction folder
* write metrics outputs to CSV files such as metrics_results.csv

### 2. Evaluate existing predictions
If predictions already exist, you can skip the tracking step and evaluate them directly:

```bash 
python -m TrackEval.Compile --gt Data/GroundTruth/Vid1/gt.txt
```
### 3. Generate tracker YAML configurations
The tracker helper in tracker.py can build tracker configuration files programmatically.

Example:
```bash 
python Ultralytics/tracker.py tracktrack --track_high_thresh 0.6 --track_buffer 45
```

This is useful when you want to create custom tracker parameter sets without hand-editing YAML files.

### 4. Optimize tracker parameters with NOMAD
The optimizer in opti.py uses YAML configuration and NOMAD to search over tracker and detection parameters.

Example:
```bash
python NOMAD/opti.py --config NOMAD/opti_config.yaml
```

The optimizer will:

* load the configuration from the YAML file
* run the tracker repeatedly at different parameter points
* score each output with the selected metric
* save JSON results and merge evaluation rows into CSV outputs

## Installation
This repository is not currently packaged as an installable Python package. Install the dependencies directly with:

```bash
pip install -r requirements.txt
```

### Required dependencies
The main Python dependencies are:

* numpy
* pandas
* opencv-python
* PyYAML
* ultralytics
* motmetrics
* trackeval
* PyNomadBBO
* ffmpeg-python

If you use the video cutting or preprocessing utilities in Ultralytics, ensure FFmpeg is installed and available on your system PATH.

## Project configuration
### ptimizer configuration
The optimizer is driven by opti_config.yaml. This file defines:

* the input video and ground-truth path
* model list for the search
* the metric to optimize
* the search-space dimensions
* the starting point for NOMAD
* tracker override defaults

A typical configuration includes fields such as:

* video_path
* gt_path
* models
* tracker_name
* metric
* search_space
* x0_raw
* tracker_override_defaults
* results_dir

You can override individual values from the command line when needed, but the YAML file remains the primary source of configuration.

## Data layout expectations
The evaluation and optimization scripts assume a structure similar to the folders already present in this repository.

A typical layout is:
```bash 
Data/
  GroundTruth/
    Vid1/
      gt.txt
  Prediction/
    Vid1/
      metrics_results.csv
```

The compile pipeline will resolve the prediction folder from the path of the ground-truth file when possible. This makes it easier to keep data organized by video ID.

## Outputs
Depending on the workflow, the repository can produce:

* MOT-format prediction text files in prediction folders
* CSV metric tables such as metrics_results.csv and metrics_results_occluded.csv
* JSON result files in Results for optimizer runs
* tracker YAML files under the tracker configuration directories

## Notes and limitations

* This repository is primarily intended for research and experimentation.
* Several scripts are standalone entry points and rely on the repository layout and data conventions present here.
* The optimizer and evaluation flows are configurable but still benefit from careful path and dataset preparation.
* For more reusable or production-grade integrations, the logic would likely be packaged into a more formal Python module structure.