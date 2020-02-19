import time

import requests
from lxml import etree

headers = {
    'User-Agent': 'self-defind-user-agent'
}


def address3():
    f1 = open('/Users/c/Desktop/address2.txt', 'w', encoding='utf8')
    url = 'https://hotels.ctrip.com/hotel/taipei617/p'
    index = 1
    while True:
        res = requests.get(url + str(index), headers=headers)
        html = etree.HTML(res.text)
        data = html.xpath('//p[@class="hotel_item_htladdress"]//text()')

        print("".join(data))
        # for i in range(int(len(data) / 6)):
        #     line = data[6*i + 1] + data[6*i + 3] + data[6*i + 4]
        #     print(line)
        index = index + 1


if __name__ == "__main__":
    # getattr(__import__(__name__), input('func name:').strip())()
    address3()
