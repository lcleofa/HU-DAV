# utilities/logger_setup.py

import sys
from pathlib import Path
from loguru import logger


class LoggerSetup:
    """Reusable Loguru logger configuration class."""

    def __init__(self, config: dict, log_filename: str = "application.log"):
        """
        Args:
            config (dict): The loaded TOML configuration.
            log_filename (str): Name of the log file (e.g. "wk2_comparing_categories.log").
        """
        self.log_dir = Path(config["logging"]["logdir"]).resolve()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_filename = log_filename

    def setup(self):
        """Configure and return the logger instance."""
        logger.remove()  # Remove default handler
        logger.add(sys.stderr, level="INFO")  # Console output

        logfile = self.log_dir / self.log_filename
        logger.add(
            logfile,
            rotation="10 MB",
            retention="30 days",
            compression="zip",
            level="DEBUG",
            enqueue=True,
            encoding="utf-8"
        )

        return logger
