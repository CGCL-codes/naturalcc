import logging
from pathlib import Path
import datetime

# main
def init_logger(logger_obj:logging.Logger, level = logging.DEBUG, level_file = logging.DEBUG, consol_level = logging.DEBUG, logfile:str = None):
    # logger，DEBUG，log
    logger_obj.setLevel(level)
    # handler，
    # 
    now = datetime.datetime.now()
    time_string = now.strftime("%Y%m%d%H%M%S")
    if not logfile:
        logfile = Path(f'./logs/{time_string}.log')  # 
        logfile.parent.mkdir(exist_ok=True, parents=True)
    fh = logging.FileHandler(logfile, mode='a')  # 
    fh.setLevel(level_file)  # filelog，DEBUG，DEBUGlog

    # handler，
    ch = logging.StreamHandler()
    ch.setLevel(consol_level)  # consolelog，WARNING，WARNINGlog

    # handler
    formatter = logging.Formatter('[%(asctime)s - %(filename)s, line:%(lineno)d] - %(levelname)s: %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # loggerhandler
    logger_obj.addHandler(fh)
    logger_obj.addHandler(ch)
    
# logger
logger = logging.getLogger('CC-HARD')


