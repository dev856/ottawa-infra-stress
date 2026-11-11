#!/usr/bin/env python
"""Train the model from the committed open-data snapshot.

    python run_pipeline.py            # train only, works offline
    python run_pipeline.py --refresh  # download fresh City and ECCC data first
"""

from __future__ import annotations

import argparse
import logging
import sys

from src import ottawa_data, train_model
from src.errors import ArtifactError, DataValidationError, ExternalServiceError

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    """Run the pipeline and return a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--refresh", action="store_true", help="download fresh open data first")
    arguments = parser.parse_args(argv)
    try:
        if arguments.refresh:
            ottawa_data.main()
        train_model.main()
    except (ArtifactError, DataValidationError, ExternalServiceError) as exc:
        logger.error("Pipeline failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    sys.exit(main())
