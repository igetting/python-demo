import time

import requests
from lxml import etree

headers = {
    'User-Agent': 'self-defind-user-agent'
}


def id1():
    f1 = open('/Users/c/Desktop/id1.txt', 'w', encoding='utf8')
    url = 'https://tw.51240.com/'
    while True:
        res = requests.get(url, headers=headers)
        # print(res)
        html = etree.HTML(res.text)
        # print(html)
        ids = html.xpath('//div[@id="shuaxinshenfenzheng"]/table//table//tr[position()>1]/td[2]//text()')
        print(ids)
        for content in ids:
            print(content)
            f1.write(content + '\n')
        f1.flush()
        time.sleep(1)
    f1.close()


def name1():
    f1 = open('/Users/c/Desktop/name1.txt', 'w', encoding='utf8')
    url = 'https://tw.51240.com/'
    while True:
        res = requests.get(url, headers=headers)
        # print(res)
        html = etree.HTML(res.text)
        # print(html)
        ids = html.xpath('//div[@id="shuaxinshenfenzheng"]/table//table//tr[position()>1]/td[1]//text()')
        print(ids)
        for content in ids:
            print(content)
            f1.write(content + '\n')
        f1.flush()
        time.sleep(1)
    f1.close()


def address1():
    f1 = open('/Users/c/Desktop/address1.txt', 'w', encoding='utf8')
    url = 'https://www.p2peye.com/yhwd/p16/page'
    index = 1
    while True:
        res = requests.get(url + str(index), headers=headers)
        html = etree.HTML(res.text)
        data = html.xpath(
            '//div[@class="ui-branchs-warp"]//li//dd[@class="ui-branchs-address"]/span[@class="ui-branchs-info-sp"]//text()')
        print(data)
        for content in data:
            f1.write(content + "\n")
        f1.flush()
        index = index + 1
    f1.close()


def company1():
    f1 = open('/Users/c/Desktop/company1.txt', 'w', encoding='utf8')
    url = 'https://waizi.mingluji.com/taxonomy/term/825?page='
    index = 0
    while True:
        res = requests.get(url + str(index), headers=headers)
        time.sleep(1)
        html = etree.HTML(res.text)
        data = html.xpath('//div[@class="view-content"]/div/div/h2/a/text()')
        print(data)
        if len(data) == 0:
            break
        for content in data:
            print(content)
            f1.write(content + "\n")
        f1.flush()
        index = index + 1
    f1.close()


def company2():
    f1 = open('/Users/c/Desktop/company2.txt', 'w', encoding='utf8')
    url = 'https://www.71ab.com/province_32'
    index = 1
    while True:
        if index == 1:
            uri = url + '.html'
        else:
            uri = url + '_p' + str(index) + '.html'
        print(uri)
        res = requests.get(uri, headers=headers)
        html = etree.HTML(res.text)
        data = html.xpath('//strong[@class="px14"]/text()')
        print(data)
        if len(data) == 0:
            break
        for content in data:
            print(content)
            f1.write(content + "\n")
        f1.flush()
        index = index + 1
    f1.close()


def address2():
    f1 = open('/Users/c/Desktop/address2.txt', 'w', encoding='utf8')
    url = 'https://yyk.99.com.cn/taiwan/'
    res = requests.get(url, headers=headers)
    # print(res.text)
    html = etree.HTML(res.text)
    # el = html.xpath('//div[@class="m-box"]//table//td//text()')
    el = html.xpath('//div[@class="m-box"]//table//td//a/@href')
    # print(el)
    for u in el:
        try:
            uri = 'https://yyk.99.com.cn' + u
            # print(uri)
            res1 = requests.get(uri, headers=headers)
            el1 = etree.HTML(res1.text).xpath('//dl[@class="wrap-info"]//p[4]/em/text()')[0]
            print(el1)
            f1.write(el1 + '\n')
            f1.flush()
        except Exception as e:
            print(e)
            continue
    f1.close()


if __name__ == "__main__":
    getattr(__import__(__name__), input('func name:').strip())()
