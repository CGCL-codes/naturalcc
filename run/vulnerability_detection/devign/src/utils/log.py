import logging.config

FORMAT = '%(asctime)s %(levelname)s %(name)s: %(message)s'

logging.basicConfig(filename="logs.log",
                    filemode='a',
                    format=FORMAT,
                    datefmt='%d/%m/%Y %I:%M:%S')


def log_info(logger_name, msg):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.info(msg)

def log_warning(logger_name, msg):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.WARNING)
    logger.warning(msg)
