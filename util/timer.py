import time
from apscheduler.schedulers.blocking import BlockingScheduler


def func1():
    ts = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    print('do func1 time:', ts)


def func2():
    ts = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    print('do func2 time:', ts)
    time.sleep(2)


def dojob():
    scheduler = BlockingScheduler()
    scheduler.add_job(func1, 'interval', seconds=2, id='test_job1')
    scheduler.add_job(func2, 'interval', seconds=3, id='test_job2')
    scheduler.start()


if __name__ == '__main__':
    dojob()
