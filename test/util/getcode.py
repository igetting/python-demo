import requests
import random
import time

url1 = r'http://zxgk.court.gov.cn/zhzxgk/captcha.do'
url2 = r'http://zxgk.court.gov.cn/waf_captcha/'
path1 = r'd:/temp/temp1/'
path2 = r'd:/temp/temp2/'

code = 200


def getimg(name):
    time.sleep(0.3)
    global code
    if code == 200:
        r1 = requests.request('get', url1)
        code = r1.status_code
        with open(path1 + name, 'wb') as f:
            f.write(r1.content)
            print(name)
        f.close()
    elif code == 302:
        r2 = requests.request('get', url2)
        with open(path2 + name, 'wb') as f:
            f.write(r2.content)
            print(name)
        f.close()

def getimg2(name):
    time.sleep(0.3)
    r = requests.request('get', url2)
    with open(path2 + name, 'wb') as f:
        f.write(r.content)
        print(name)
    f.close()


for i in range(10000):
    # getimg('img_' + str(round(time.time() * 1000))[-7:-1] + '.jpg')
    getimg2('img_' + str(round(time.time() * 1000))[-7:-1] + '.jpg')
