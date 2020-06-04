import hashlib
import time


def Md5(key, s):
    sh = hashlib.md5(str(key).encode())
    sh.update(str(s).encode())
    return sh.hexdigest()


if __name__ == '__main__':
    t = time.strftime("%Y%m%d", time.localtime())
    print(Md5(t, "hello world"))
