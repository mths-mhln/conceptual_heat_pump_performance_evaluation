import logging

class ColoredFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[94m",  # Blue
        "INFO": "\033[92m",  # Green
        "WARNING": "\033[93m",  # Yellow
        "ERROR": "\033[91m",  # Red
        "CRITICAL": "\033[95m",  # Magenta
    }

    RESET = "\033[0m"  # Reset color

    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.RESET)
        if record.levelno >= logging.WARNING:
            fmt = "%(levelname)-8s %(filename)s:%(lineno)d - %(message)s"
        else:
            fmt = "%(levelname)-8s %(message)s"
        log_message = logging.Formatter(fmt).format(record)
        return f"{log_color}{log_message}{self.RESET}"


def setup_logger():
    # Get the logger instance
    logger = logging.getLogger(__name__)

    # Check if handlers are already added, to prevent duplicates
    if not logger.hasHandlers():
        # Set up the console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Create a formatter and attach it to the handler
        formatter = ColoredFormatter()
        ch.setFormatter(formatter)

        # Add the handler to the logger
        logger.addHandler(ch)

    logger.setLevel(logging.INFO)
    return logger