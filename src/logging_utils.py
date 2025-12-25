import logging
import os
import datetime

def setup_logger(log_dir="logs", level=logging.INFO):
    """
    Sets up a logger that writes to both a file and the console.

    Args:
        log_dir (str): The directory to save log files in.
        level (int): The logging level.

    Returns:
        logging.Logger: A configured logger instance.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(level)

    # Create handlers
    console_handler = logging.StreamHandler()
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    log_filename = os.path.join(log_dir, f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    file_handler = logging.FileHandler(log_filename)

    # Create formatters and add it to handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add handlers to the logger
    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger
