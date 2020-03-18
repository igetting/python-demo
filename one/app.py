from tools import send
from settings import *
from tools import DBObject

def check():
    o = DBObject()
    _, result = o.execsql("select * from test.user")
    print(result)

if __name__ == '__main__':
    # send(sender, receivers, subject, body)
    check()
