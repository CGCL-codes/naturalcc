import logging
from pathlib import Path
import datetime


def init_logger(logger_obj:logging.Logger, level = logging.DEBUG, level_file = logging.DEBUG, consol_level = logging.INFO):
    # 设置logger的总等级，这里设置为最低等级DEBUG，只有等级大于等于该等级的log才会输出
    logger_obj.setLevel(level)
    # 创建一个handler，用于写入日志文件
    # 获取当前时间
    now = datetime.datetime.now()
    time_string = now.strftime("%Y%m%d%H%M%S")
    logfile = Path(f'./logs/{time_string}.log')  # 指定日志文件的位置
    logfile.parent.mkdir(exist_ok=True, parents=True)
    fh = logging.FileHandler(logfile, mode='a')  # 以追加模式打开日志文件
    fh.setLevel(level_file)  # 输出到file的log等级的开关，设置为DEBUG，表示只有DEBUG及以上级别的log才会输出到文件

    # 再创建一个handler，用于输出到控制台
    ch = logging.StreamHandler()
    ch.setLevel(consol_level)  # 输出到console的log等级的开关，设置为WARNING，表示只有WARNING及以上级别的log才会输出到控制台

    # 定义handler的输出格式
    formatter = logging.Formatter('[%(asctime)s - %(filename)s, line:%(lineno)d] - %(levelname)s: %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # 将logger添加到handler里面
    logger_obj.addHandler(fh)
    logger_obj.addHandler(ch)
    
# 全局的logger
logger = logging.getLogger('UICoder')
init_logger(logger)


