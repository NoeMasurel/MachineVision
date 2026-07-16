"""
build_tracker.py

Builds a custom Ultralytics tracker YAML config from a tracker name plus
keyword overrides, and returns the path to the written file. The path can
be passed straight to `tracker=` in `ObjectTracking(...)` / `model.track(...)`.

Base configs are pulled from the official ultralytics repo so this stays in
sync with new trackers/params without hardcoding defaults. Bases are cached
locally under CACHE_DIR after the first fetch.

Usage:
    from build_tracker import build_tracker_config

    path = build_tracker_config(
        "bytetrack",
        track_high_thresh=0.6,
        track_buffer=45,
    )
    # -> Path("trackers/bytetrack_<hash>.yaml")

    tracker = ObjectTracking(..., tracker=str(path))

CLI:
    python build_tracker.py botsort --with_reid True --appearance_thresh 0.85
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen
from urllib.error import URLError

import yaml

# Config

RAW_BASE_URL = "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/trackers/{name}.yaml"

CACHE_DIR = Path(__file__).parent / ".tracker_base_cache"
OUTPUT_DIR = Path(__file__).parent / "trackers"

VALID_TRACKERS = {
    "botsort",
    "bytetrack",
    "ocsort",
    "deepocsort",
    "fasttrack",
    "tracktrack",
}

# Parameters shared across (most) tracker configs, used only for a friendly
# warning if you pass something that looks like a typo. This is NOT an
# exhaustive allowlist -- each tracker's own base YAML is the real source of
# truth, since tracker-specific params vary (e.g. `delta_t` only exists for
# ocsort/deepocsort, `iou_weight` only for tracktrack).
SHARED_PARAMS = {
    "tracker_type",
    "track_high_thresh",
    "track_low_thresh",
    "new_track_thresh",
    "track_buffer",
    "match_thresh",
    "fuse_score",
    "gmc_method",
    "proximity_thresh",
    "appearance_thresh",
    "with_reid",
    "model",
}

PROTECTED_KEYS = {"tracker_type"}  # changing this breaks which algorithm runs


class TrackerConfigError(ValueError):
    pass

# Fetching / caching the base config

def _normalize_name(name: str) -> str:
    stem = Path(name).stem  # tolerate "bytetrack.yaml" or "bytetrack"
    stem = stem.lower()
    if stem not in VALID_TRACKERS:
        raise TrackerConfigError(
            f"Unknown tracker '{name}'. Valid options: {sorted(VALID_TRACKERS)}"
        )
    return stem


def _fetch_base_config(name: str) -> dict:
    """Return the official base YAML for `name` as a dict, using a local
    cache after the first successful fetch."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{name}.yaml"

    if cache_path.exists():
        return yaml.safe_load(cache_path.read_text(encoding="utf-8")) or {}

    url = RAW_BASE_URL.format(name=name)
    try:
        with urlopen(url, timeout=10) as resp:
            text = resp.read().decode("utf-8")
    except URLError as e:
        raise TrackerConfigError(
            f"Could not fetch base config for '{name}' from {url} ({e}). "
            "Check network access, or supply a local base via `base_config_path=`."
        ) from e

    cache_path.write_text(text, encoding="utf-8")
    return yaml.safe_load(text) or {}

# Public API


def build_tracker_config(
    name: str,
    output_path: str | Path | None = None,
    output_dir: str | Path = OUTPUT_DIR,
    base_config_path: str | Path | None = None,
    allow_new_keys: bool = False,
    **overrides,
) -> Path:
    """
    Build a tracker YAML config for Ultralytics and return its file path.

    Args:
        name: Tracker name, e.g. "botsort", "bytetrack", "ocsort",
            "deepocsort", "fasttrack", "tracktrack" (with or without
            ".yaml").
        output_path: Exact path to write to. If omitted, a path is derived
            from `output_dir` + tracker name + a short hash of the overrides,
            so identical calls reuse the same file and different overrides
            get distinct files.
        output_dir: Directory for auto-named output files (ignored if
            `output_path` is given).
        base_config_path: Optional local YAML file to use as the base
            instead of fetching from GitHub (e.g. if you're offline or want
            to start from your own template).
        allow_new_keys: If False (default), raises when an override key
            isn't present in the fetched base config (catches typos like
            `track_hi_thresh`). Set True to allow adding new keys.
        **overrides: Any tracker parameter to set/override, e.g.
            track_high_thresh=0.6, with_reid=True, appearance_thresh=0.85.

    Returns:
        Path to the written YAML file.
    """
    tracker_name = _normalize_name(name)

    for key in overrides:
        if key in PROTECTED_KEYS:
            raise TrackerConfigError(
                f"'{key}' is derived from the tracker name and can't be overridden "
                f"(changing it would silently switch algorithms)."
            )

    if base_config_path is not None:
        base = yaml.safe_load(Path(base_config_path).read_text(encoding="utf-8")) or {}
    else:
        base = _fetch_base_config(tracker_name)

    if not allow_new_keys:
        unknown = set(overrides) - set(base)
        if unknown:
            raise TrackerConfigError(
                f"Unknown parameter(s) for '{tracker_name}': {sorted(unknown)}. "
                f"Valid parameters for this tracker: {sorted(base)}. "
                "Pass allow_new_keys=True to bypass this check."
            )

    config = {**base, **overrides}

    if output_path is not None:
        out_path = Path(output_path)
    else:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if overrides:
            digest = hashlib.sha1(
                json.dumps(overrides, sort_keys=True, default=str).encode()
            ).hexdigest()[:8]
            out_path = out_dir / f"{tracker_name}_{digest}.yaml"
        else:
            out_path = out_dir / f"{tracker_name}_default.yaml"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    return out_path


def list_valid_params(name: str) -> list[str]:
    """Return the parameter names available for a given tracker, fetched
    from its official base config."""
    tracker_name = _normalize_name(name)
    return sorted(_fetch_base_config(tracker_name))


# CLI


def _parse_value(raw: str):
    """Best-effort type coercion for CLI overrides (bool/int/float/str)."""
    low = raw.lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def main():
    parser = argparse.ArgumentParser(
        description="Build an Ultralytics tracker YAML config.",
        epilog="Example: python build_tracker.py bytetrack --track_high_thresh 0.6 --track_buffer 45",
    )
    parser.add_argument("tracker", help="Tracker name, e.g. bytetrack, botsort, ocsort, deepocsort, fasttrack, tracktrack")
    parser.add_argument("--output", help="Explicit output file path")
    parser.add_argument("--allow-new-keys", action="store_true", help="Allow parameters not in the base config")
    args, unknown = parser.parse_known_args()

    overrides = {}
    i = 0
    while i < len(unknown):
        tok = unknown[i]
        if not tok.startswith("--"):
            raise SystemExit(f"Unexpected argument: {tok}")
        key = tok[2:]
        if i + 1 >= len(unknown):
            raise SystemExit(f"Missing value for --{key}")
        overrides[key] = _parse_value(unknown[i + 1])
        i += 2

    path = build_tracker_config(
        args.tracker,
        output_path=args.output,
        allow_new_keys=args.allow_new_keys,
        **overrides,
    )
    print(path)


if __name__ == "__main__":
    main()