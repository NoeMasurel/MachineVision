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

## Installation

This repository is not yet packaged as an installable Python package. Install the dependencies with:

```bash
pip install -r requirements.txt
```

If you plan to use the video cutting utilities (`Ultralytics/cut.py`), make sure FFmpeg is installed on your system and available in your `PATH`.

## Quick start

### 1) Evaluate an existing prediction file

```bash
python TrackEval/Eval.py --gt path/to/gt.txt --pred path/to/predictions.txt
```

### 2) Run tracking and evaluate the produced output

```bash
python TrackEval/Compile.py --gt path/to/gt.txt --video path/to/video.mp4 --models m --confidence 0.25
```

### 3) Build a custom tracker config

```bash
python Ultralytics/tracker.py bytetrack --track_high_thresh 0.6 --track_buffer 45
```

### 4) Optimize tracker parameters with NOMAD

```bash
python NOMAD/opti.py
```

Note: the optimizer script contains hard-coded defaults and assumes your data layout is already configured. Edit the constants in `NOMAD/opti.py` before running it.

## Dependencies

The main dependencies are:

- `numpy`
- `pandas`
- `opencv-python`
- `PyYAML`
- `ultralytics`
- `motmetrics`
- `trackeval`
- `PyNomad`
- `ffmpeg-python`

## Notes

- This repository is best suited for research and experimentation workflows.
- Many scripts are written as standalone entry points and assume specific file layouts.
- For production or reusable integrations, consider packaging the logic into a proper Python package and adding automated tests.
