import logging
import sys


def setup_logger(name="vdpo") -> logging.Logger:
    """
    Set up and configure the logger.

    Args:
        name (str): Logger name

    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logger
    _logger = logging.getLogger(name)
    _logger.setLevel(logging.INFO)

    # Prevent duplicate handlers
    if _logger.handlers:
        return _logger

    # Create formatters
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    _logger.addHandler(console_handler)

    return _logger


# Create default logger instance
logger = setup_logger()
