from tools import send
from settings import *
from tools import DBObject


def check():
    log.info("begin to ...")
    o = DBObject()
    _, result = o.execsql("select * from test.user")
    print(result)
    log.info(result)


if __name__ == '__main__':
    # send(sender, receivers, subject, body)
    check()
