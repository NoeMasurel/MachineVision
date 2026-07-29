"""Console-script entry points: stage-track / stage-evaluate / stage-optimize.

Each function is a thin delegate to the corresponding module's own
argparse-based `main()`, which parses arguments, builds a config object, and
calls the public API (run_tracking / evaluate_predictions / optimize_parameters).
"""

from stage_tracking.evaluation import compile as evaluation_compile
from stage_tracking.optimization import nomad_runner
from stage_tracking.tracking import pipeline


def track(argv: list[str] | None = None) -> None:
    pipeline.main(argv)


def evaluate(argv: list[str] | None = None) -> None:
    evaluation_compile.main(argv)


def optimize(argv: list[str] | None = None) -> None:
    nomad_runner.main(argv)
