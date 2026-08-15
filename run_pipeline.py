#!/usr/bin/env python
"""Run each local pipeline stage in a fixed, fail-fast order.

Each stage runs in a fresh Python process so configuration and failures remain
easy to understand. Mock mode is deterministic and does not require the network.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import time
from pathlib import Path

from src.config import DATA_SOURCE_MODE, ensure_directories

logger = logging.getLogger(__name__)

PIPELINE_STEPS = (
    ("src/setup_db.py", "Check optional local database"),
    ("src/ingest_311_service_requests.py", "Minimize and H3-index 311 events"),
    ("src/generate_h3_grid.py", "Generate H3 grid"),
    ("src/fetch_weather_features.py", "Calculate weather features"),
    ("src/extract_infrastructure_features.py", "Aggregate infrastructure features"),
    ("src/train_model.py", "Train synthetic demonstration model"),
)


def run_step(script: str, description: str, project_root: Path) -> float:
    """Run one checked-in script and raise when its process exits unsuccessfully."""
    logger.info("Starting: %s", description)
    start_time = time.monotonic()
    result = subprocess.run(  # noqa: S603 -- executable and scripts are fixed project inputs
        [sys.executable, script],
        cwd=project_root,
        check=False,
    )
    duration = time.monotonic() - start_time
    if result.returncode != 0:
        raise RuntimeError(f"Pipeline stage failed: {description}")
    logger.info("Completed in %.2f seconds: %s", duration, description)
    return duration


def main() -> int:
    """Run the complete local pipeline and return a process exit code."""
    ensure_directories()
    project_root = Path(__file__).resolve().parent
    logger.info("Starting pipeline in %s data mode", DATA_SOURCE_MODE)
    timings: list[tuple[str, float]] = []
    try:
        for script, description in PIPELINE_STEPS:
            timings.append((description, run_step(script, description, project_root)))
    except RuntimeError as exc:
        logger.error("%s", exc)
        return 1

    total_seconds = sum(duration for _, duration in timings)
    logger.info("Pipeline completed successfully in %.2f seconds", total_seconds)
    for description, duration in timings:
        logger.info("Stage duration %.2f seconds: %s", duration, description)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    sys.exit(main())
