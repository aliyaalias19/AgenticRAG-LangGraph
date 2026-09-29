"""MLflow experiment tracking.

Everything that produces a number logs it here with the configuration that
produced it: corpus hash, retrieval policy, model version, quantisation
settings. A benchmark result without its configuration is an anecdote, and
six weeks later nobody remembers which run the number in the README came
from.

MLflow is optional at import time. A missing MLflow degrades to structured
logs rather than breaking a run, because losing an experiment to a tracking
dependency is worse than losing the tracking.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentic_rag.config.settings import Settings, get_settings
from agentic_rag.obs.logging import get_logger

logger = get_logger(__name__)

EXPERIMENT_NAME = "agentic-rag"


def _mlflow() -> Any | None:
    try:
        import mlflow
    except ImportError:
        logger.warning("mlflow_unavailable", detail="metrics logged to stdout only")
        return None
    return mlflow


def _flatten(payload: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten a nested dict into dotted keys MLflow can store."""
    flat: dict[str, Any] = {}
    for key, value in payload.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten(value, f"{name}."))
        elif isinstance(value, (list, tuple)):
            flat[name] = ",".join(str(v) for v in value)
        else:
            flat[name] = value
    return flat


@dataclass
class RunTracker:
    """Records parameters, metrics and artefacts for one experiment run."""

    settings: Settings = field(default_factory=get_settings)
    params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)

    def log_params(self, **params: Any) -> None:
        """Record configuration values that define this run."""
        self.params.update(_flatten(params))

    def log_metrics(self, **metrics: Any) -> None:
        """Record numeric results."""
        for key, value in _flatten(metrics).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                self.metrics[key] = float(value)

    def log_corpus_provenance(self, manifest_path: Path) -> None:
        """Record which corpus build the run was measured against."""
        if not manifest_path.is_file():
            logger.warning("manifest_missing", path=str(manifest_path))
            return

        import json

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.log_params(
            corpus_hash=manifest.get("corpus_hash", "")[:16],
            document_count=manifest.get("document_count", 0),
            chunk_count=manifest.get("chunk_count", 0),
            chunk_max_chars=manifest.get("chunk_max_chars", 0),
            sources=",".join(s.get("name", "") for s in manifest.get("sources", [])),
        )


@contextmanager
def track_run(
    run_name: str,
    settings: Settings | None = None,
    tags: dict[str, str] | None = None,
) -> Iterator[RunTracker]:
    """Open a tracked run, flushing parameters and metrics on exit."""
    settings = settings or get_settings()
    tracker = RunTracker(settings=settings)
    mlflow = _mlflow()

    if mlflow is None:
        yield tracker
        logger.info("run_completed", run=run_name, **tracker.metrics)
        return

    mlflow.set_tracking_uri(f"file://{settings.paths.data_dir / 'mlruns'}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=run_name):
        if tags:
            mlflow.set_tags(tags)
        try:
            yield tracker
        finally:
            if tracker.params:
                mlflow.log_params(tracker.params)
            if tracker.metrics:
                mlflow.log_metrics(tracker.metrics)
            logger.info("run_completed", run=run_name, **tracker.metrics)
