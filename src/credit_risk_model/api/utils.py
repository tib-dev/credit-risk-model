import logging
import sys
from pathlib import Path


def get_logger(name: str, log_level: int = logging.INFO) -> logging.Logger:
    """
    Configures and returns a logger instance that outputs to both 
    the console and a log file.
    """
    logger = logging.getLogger(name)

    # Prevent duplicate logs if the logger is already configured
    if logger.hasHandlers():
        return logger

    logger.setLevel(log_level)

    # Define format
    formatter = logging.Formatter(
        "%(asctime)s — %(name)s — %(levelname)s — %(message)s"
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Optional: File Handler
    # This will create a 'logs' directory in your root if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "api.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
