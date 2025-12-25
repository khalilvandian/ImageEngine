import logging
import os
import datetime


def setup_logger(log_dir="logs", console_level=logging.INFO, file_level=logging.DEBUG):
    """Return a logger that keeps verbose details in a file and a quieter view in the console."""
    logger = logging.getLogger(__name__)
    logger.setLevel(min(console_level, file_level))
    logger.propagate = False

    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_level)

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        log_filename = os.path.join(log_dir, f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(file_level)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger
