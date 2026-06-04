import os
import re
import datetime
import logging
import sys
from config import LOGS_NUM


try:
    import codecs
except ImportError:
    codecs = None


class MultiprocessHandler(logging.FileHandler):
    def __init__(self, filename, when='D', backupCount=0, encoding=None, delay=False):
        # 初始化参数
        self.prefix = filename
        self.backupCount = backupCount
        self.when = when.upper()
        self.extMath = r"^\d{4}-\d{2}-\d{2}"

        # 设置时间格式映射
        self.when_dict = {
            'S': "%Y-%m-%d-%H-%M-%S",
            'M': "%Y-%m-%d-%H-%M",
            'H': "%Y-%m-%d-%H",
            'D': "%Y-%m-%d"
        }

        self.suffix = self.when_dict.get(when)
        if not self.suffix:
            print('The specified date interval unit is invalid: ', self.when)
            sys.exit(1)

        # 日志文件路径
        self.filefmt = os.path.join('.', "logs", f'{self.prefix}-{self.suffix}.log')
        self.filePath = datetime.datetime.now().strftime(self.filefmt)

        # 创建日志目录
        _dir = os.path.dirname(self.filefmt)
        try:
            if not os.path.exists(_dir):
                os.makedirs(_dir)
        except Exception as e:
            print('Failed to create log file: ', e)
            print("log_path：" + self.filePath)
            sys.exit(1)

        if codecs is None:
            encoding = None

        logging.FileHandler.__init__(self, self.filePath, 'a+', encoding, delay)


    def shouldChangeFileToWrite(self):
        _filePath = datetime.datetime.now().strftime(self.filefmt)
        if _filePath != self.filePath:
            self.filePath = _filePath
            return True
        return False


    def doChangeFile(self):
        self.baseFilename = os.path.abspath(self.filePath)
        if self.stream:
            self.stream.close()
            self.stream = None

        if not self.delay:
            self.stream = self._open()
        
        if self.backupCount > 0:
            for s in self.getFilesToDelete():
                os.remove(s)


    def getFilesToDelete(self):
        dir_name, _ = os.path.split(self.baseFilename)
        file_names = os.listdir(dir_name)
        result = []
        prefix = self.prefix + '-'
        
        for file_name in file_names:
            if file_name[:len(prefix)] == prefix:
                suffix = file_name[len(prefix):-4]
                if re.compile(self.extMath).match(suffix):
                    result.append(os.path.join(dir_name, file_name))
        result.sort()

        if len(result) < self.backupCount:
            result = []
        else:
            result = result[:len(result) - self.backupCount]
        return result


    def emit(self, record):
        try:
            if self.shouldChangeFileToWrite():
                self.doChangeFile()
            logging.FileHandler.emit(self, record)
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)


def write_log(log_to_console=True):
    '''
    日志记录器
    
    Args:
        log_to_console: 是否同时输出到控制台，tmux后台运行时建议设为False
    '''
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    # 清除已存在的处理器，避免重复
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # 日志输出格式
    fmt = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(filename)s:%(lineno)d | %(funcName)s | %(message)s'
    )

    # 文件输出（主要日志）
    log_name = "srchvt"
    file_handler = MultiprocessHandler(log_name, when='D', backupCount=LOGS_NUM)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    file_handler.doChangeFile()
    logger.addHandler(file_handler)
    
    # 错误日志单独输出到error文件
    error_log_name = "srchvt-error"
    error_file_handler = MultiprocessHandler(error_log_name, when='D', backupCount=LOGS_NUM)
    error_file_handler.setLevel(logging.ERROR)
    error_file_handler.setFormatter(fmt)
    error_file_handler.doChangeFile()
    logger.addHandler(error_file_handler)

    # 输出到控制台（tmux中可能不需要）
    if log_to_console:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(fmt)
        logger.addHandler(stream_handler)

    return logger


# 在主程序入口判断是否在tmux中运行
def is_running_in_tmux():
    """检测是否在tmux中运行"""
    return os.environ.get('TMUX') is not None


# 初始化日志
LOGGER = write_log(log_to_console=not is_running_in_tmux())