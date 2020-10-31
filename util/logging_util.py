import logging
import sys


def create_logger(log_file, message="LOGGER", stdout=False):
    logger = logging.getLogger(message)
    logger.setLevel(logging.DEBUG)
    # Create handlers
    if stdout:
        c_handler = logging.StreamHandler(sys.stdout)
        c_handler.setLevel(logging.DEBUG)
        c_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        c_handler.setFormatter(c_format)
        logger.addHandler(c_handler)

    f_handler = logging.FileHandler(log_file)
    f_handler.setLevel(logging.INFO)
    f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    f_handler.setFormatter(f_format)
    logger.addHandler(f_handler)
    return logger