# -*- coding: utf-8 -*-

import logging

"""
CRITICAL
ERROR
WARNING
INFO
DEBUG
NOTSET
"""
LOGGING_LEVEL = logging.INFO

try:
    LOGGER = logging.getLogger(__name__)

    import colorlog

    stream = logging.StreamHandler()
    stream.setFormatter(
        colorlog.ColoredFormatter(
            fmt='%(log_color)s[%(asctime)-15s] %(levelname)7s >> %(message)s (%(filename)s:%(lineno)d, %(funcName)s())',
            datefmt='%Y-%m-%d %H:%M:%S',
            log_colors={
                "DEBUG": "white",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red",
            },
        )
    )
    LOGGER.setLevel(LOGGING_LEVEL)
    LOGGER.addHandler(stream)



except ImportError:
    logging.basicConfig(
        level=LOGGING_LEVEL,
        format='[%(asctime)-15s] %(levelname)7s >> %(message)s (%(filename)s:%(lineno)d, %(funcName)s())',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    LOGGER = logging.getLogger(__name__)

    LOGGER.warning(
        "colorlog has been not installed. If you want to get colorful loggings,"
        " please install colorlog with: pip install colorlog"
    )

__all__ = [
    "LOGGER",
]
