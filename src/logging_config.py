"""Single place where logging is configured (key=value lines, level from LOG_LEVEL)."""

import logging
import os

LOG_FORMAT = "ts=%(asctime)s level=%(levelname)s logger=%(name)s msg=%(message)r"


def configure_logging() -> None:
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=getattr(logging, level_name, logging.INFO), format=LOG_FORMAT)
