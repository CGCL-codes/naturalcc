import logging
import sys


def setup_logger(log_level=logging.DEBUG):
    logger = logging.getLogger(sys._getframe(1).f_globals["__name__"])
    logger.setLevel(log_level)
    # file_handler = logging.FileHandler(f'{name}.log')
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    # file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    # logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger
