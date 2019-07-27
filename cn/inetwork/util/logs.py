# 1.控制台输出日志
import logging

logging.basicConfig(level=logging.NOTSET,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

logging.debug('debug')
logging.info('info')
logging.warning('warning')
logging.error('error')
logging.critical('critical')

# 2.日志写入文件
import logging  # 引入logging模块
import os.path
import time

# 第一步，创建一个logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)  # Log等级总开关
# 第二步，创建一个handler，用于写入日志文件
rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
log_path = os.path.dirname(os.getcwd()) + '/logs/'
log_name = log_path + rq + '.log'
logpath = os.path.dirname(log_name)
if not os.path.exists(logpath):
    os.makedirs(logpath)
logfile = log_name
fh = logging.FileHandler(logfile, mode='w')
fh.setLevel(logging.DEBUG)  # 输出到file的log等级的开关
# 第三步，定义handler的输出格式
formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
fh.setFormatter(formatter)
# 第四步，将logger添加到handler里面
logger.addHandler(fh)
# 日志
logger.debug('this is a logger debug message')
logger.info('this is a logger info message')
logger.warning('this is a logger warning message')
logger.error('this is a logger error message')
logger.critical('this is a logger critical message')

# 3.通过装饰器调用
import traceback
import logging
from logging.handlers import TimedRotatingFileHandler


def logger(func):
    def inner(*args, **kwargs):  # 1
        try:
            # print "Arguments were: %s, %s" % (args, kwargs)
            func(*args, **kwargs)  # 2
        except:
            # print 'error',traceback.format_exc()
            print('error')

    return inner


def loggerInFile(filename):  # 带参数的装饰器需要2层装饰器实现,第一层传参数，第二层传函数，每层函数在上一层返回
    def decorator(func):
        def inner(*args, **kwargs):  # 1
            logFilePath = filename  # 日志按日期滚动，保留5天
            logger = logging.getLogger()
            logger.setLevel(logging.INFO)
            handler = TimedRotatingFileHandler(logFilePath,
                                               when="d",
                                               interval=1,
                                               backupCount=5)
            formatter = logging.Formatter('%(asctime)s  - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            try:
                # print "Arguments were: %s, %s" % (args, kwargs)
                result = func(*args, **kwargs)  # 2
                logger.info(result)
            except:
                logger.error(traceback.format_exc())

        return inner

    return decorator


@logger
def test():
    print(2 / 0)


test()


@loggerInFile('newloglog')
def test2(n):
    print(100 / n)
