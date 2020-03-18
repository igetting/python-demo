from tools import send
from settings import *
from tools import DBObject
import logging

logging.basicConfig(level=logging.DEBUG,
                    filename='out.log',
                    filemode='a',
                    format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                    )


def check():
    o = DBObject()
    _, result = o.execsql("select * from test.user")
    print(result)
    logging.info(result)

if __name__ == '__main__':
    # send(sender, receivers, subject, body)
    check()
